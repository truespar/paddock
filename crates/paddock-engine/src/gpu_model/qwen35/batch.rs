//! Qwen3.5/3.6 continuous batching: slots, batched prefill/decode, pipe, unified/mixed.

use super::*;
use crate::gpu::{GpuError, GpuExecutor, KvDtype, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use crate::kv_plan;
use crate::kv_pool::{BlockTable, KvPool};
use crate::paged_radix::PagedRadix;
use cudarc::driver::DevicePtr;
use cudarc::driver::sys::CUstreamCaptureMode;

/// Max rows a tick may overdraw to absorb a prompt tail that would otherwise
/// ride a whole extra tick (chunk-tail finding).
const TAIL_SLOP: usize = 64;

/// f8t unified arm master switch (see the bs_f8t_attn_p note in
/// `unified_launch_core`). Kill: PADDOCK_NO_F8T_UNIFIED.
fn f8t_unified_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_F8T_UNIFIED").is_none())
}

/// f8t CHUNK arm row ceiling: `prefill_batch_pass` rides the tile plane
/// while r <= this (65+ takes the launcher's tc5r 2-SM whole-prefill route,
/// which pads batch to 256 and TMA-reads the activation buffer to that pad -
/// d_f8t_q/d_f8t_rs are sized to next_multiple_of(256) of this value, see
/// the scratch alloc). Motivation: the split path's admission chunk passes
/// (r ~= 128 at the c32 board) ran q8_0/f8bs at ~18.6 ms/admission where the
/// tile plane's roofline says ~4 - the same mixed-tick mechanism the unified
/// arm already fixed at r <= 64. Default 256 = one full tc5r batch band.
/// PADDOCK_F8T_CHUNK_RMAX overrides; PADDOCK_NO_F8T_CHUNK kills (0).
/// tc5r@128 probe instrument: decode-side f8t arm batch bound.
/// Default 64 = shipped behavior exactly. PADDOCK_F8T_DEC_BMAX=128 lets the
/// b<=128 decode tick keep the f8t class (f8t_gemm routes 65..256 via tc5r,
/// one weight pass) instead of splitting into 2x64 halves - pair with
/// PADDOCK_NO_DEC_SPLIT=1 to probe.
pub(super) fn f8t_dec_bmax() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_F8T_DEC_BMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    })
}

pub(super) fn f8t_chunk_rmax() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        if paddock_models::dev_var_os!("PADDOCK_NO_F8T_CHUNK").is_some() {
            return 0;
        }
        paddock_models::dev_var!("PADDOCK_F8T_CHUNK_RMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

/// P65: fill TruncCat rows' ids from the device top-64 head plane
/// (`head` = the dtoh'd [rows, 64, 2] u32 (id, raw-logit bits) plane;
/// `plans[i]` indexes row i). Sentinel id u32::MAX = n < 64 padding.
pub(super) fn trunc_fill_ids(head: &[u32], plans: &[crate::generator::RowSample], ids: &mut [u32]) {
    use crate::generator::RowSample;
    use crate::sampler::DevicePlan;
    // engagement witness (bisect-trap law): once per process
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let n = plans
            .iter()
            .filter(|p| matches!(p, RowSample::Device(DevicePlan::TruncCat { .. })))
            .count();
        eprintln!("[trunc-head] engaged: rows={n} (device top-64 + host head sampling)");
    });
    for (i, p) in plans.iter().enumerate() {
        if let RowSample::Device(DevicePlan::TruncCat {
            inv_t,
            u,
            k,
            top_p,
            min_p,
        }) = *p
        {
            let base = i * 128;
            let pairs: Vec<(u32, f32)> = (0..64usize)
                .filter_map(|j| {
                    let id = head[base + j * 2];
                    (id != u32::MAX).then(|| (id, f32::from_bits(head[base + j * 2 + 1])))
                })
                .collect();
            if !pairs.is_empty() {
                ids[i] = crate::sampler::sample_trunc_head(&pairs, inv_t, u, k, top_p, min_p);
            }
        }
    }
}

/// P70 fused decode recurrence kill (PADDOCK_NO_DN_V2F restores the
/// split_gqa_norm + recurrent_v2 chain for A/B).
fn dn_v2f_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_DN_V2F").is_none())
}

/// Row ceiling for the P70/P71-R2 fused GDN recurrence (`v2f` and the strided
/// `v2f_g` twin). 24 came from a probe curve of the RECURRENCE KERNEL alone
/// (2.01x at b=1, 1.5x at 8, 1.22x at 12, 1.02x at 24, 0.94x at 32) - the
/// fused form loses 6% at b=32 because the recurrence is state-bandwidth-bound
/// there and the in-block norm recompute is redundant.
///
/// But the kernel is not the band. At b=32 the fused form ELIMINATES
/// `deltanet_split_gqa_norm` (159.1 ms in a long-prompt census) and, in
/// the strided form, `row_slice2_gate` (204.8 ms) - against a 6% premium on a
/// 1465.6 ms recurrence (+93.5 ms). Band arithmetic says -270 ms of an
/// 11 287 ms die, ~2.4%. That is worth a measurement the kernel probe could
/// never see. `PADDOCK_DN_V2F_BMAX` overrides.
fn dn_v2f_bmax() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_DN_V2F_BMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24)
    })
}

/// P65 small-b twin of the device prefilter: top-64 by f32 total order from
/// a full host row (identical head to pd_topk_rows modulo boundary-tie
/// choice, which was never contractual). Used at b <= 2 where the one-block
/// binary-search kernel is DRAM-latency-bound (c1 measured -13% before this
/// fallback existed).
pub(super) fn host_top64(row: &[f32]) -> Vec<(u32, f32)> {
    let k = 64.min(row.len());
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, |&a, &b| row[b as usize].total_cmp(&row[a as usize]));
        idx.truncate(k);
    }
    idx.into_iter().map(|i| (i, row[i as usize])).collect()
}

/// Varlen chunked-GDN route gate - the same env chain the unified tick's
/// `vl_route` static checks (GDN formulation band); kept in
/// sync by hand because that one is fn-local. Kill: PADDOCK_NO_DNC_VL.
/// DeltaNet state checkpoints the prefix cache holds PER SEATED SLOT.
///
/// Sized by demand, not by the card. Each checkpoint is a whole recurrent-state
/// snapshot (~150 MiB f32 on the 27B: 48 GDN layers of state + conv window),
/// and the working set a slot generates is its own conversation at the two
/// `ckpt_cuts` per prompt that multi-turn needs - so two per slot covers every
/// seated session's latest turn with one to spare. Until 2026-09-06 the pool
/// was a fifth of the grant regardless of width: 11.5 GiB of snapshots on a
/// 96 GB card for a single-slot server that could use a handful, measured
/// against vLLM's on-demand state pages and SGLang's int8 idle store as the
/// largest single policy term in paddock's one-slot 262k floor.
pub(super) const STATE_CKPTS_PER_SLOT: u64 = 2;
/// Floor: keeps prefix reuse alive at width 1 and on small cards - a 16-turn
/// restore window for one conversation. Mandatory: below it the plan refuses
/// rather than serve without a prefix cache.
pub(super) const STATE_CKPTS_MIN: u64 = 16;
/// Cap: a 128-distinct-prefix working set at two cuts per prompt.
pub(super) const STATE_CKPTS_MAX: u64 = 256;

/// How many DeltaNet state checkpoints the prefix cache may hold.
///
/// One definition, deliberately, called by both the `kv_plan` reserve and the
/// device allocation. They used to derive this separately - the reserve from
/// `grant`, the allocation from `vram_headroom()` - and only the allocation
/// honoured the override, so `PADDOCK_KV_STATE_CKPTS=256` asked the allocator
/// for ~42.5 GiB against a 14.61 GiB reservation. That is not a bigger cache,
/// it is an unplanned allocation, and it is why the 256-checkpoint arm read
/// the widest spread of its ladder (30.8%, one leg at 1743) rather than the
/// win its slot count promised.
///
/// `leftover` is what the grant has left once every OTHER term is charged: the
/// other reserves, the seated slots' state, and a full-context KV pool. The
/// pool takes its demand out of that and never out of the KV a slot was
/// promised, so a card that can seat the configuration but not the whole
/// checkpoint want gets a smaller cache, not a refusal. The floor stays
/// mandatory either way.
fn state_ckpt_count(slots: usize, per_ckpt: u64, leftover: u64) -> u32 {
    if per_ckpt == 0 {
        return 0; // pure full-attn: no recurrent state to checkpoint
    }
    if let Some(n) = paddock_models::dev_var!("PADDOCK_KV_STATE_CKPTS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    let want = (slots as u64 * STATE_CKPTS_PER_SLOT).clamp(STATE_CKPTS_MIN, STATE_CKPTS_MAX);
    want.min(leftover / per_ckpt).max(STATE_CKPTS_MIN) as u32
}

/// The `graph pools + headroom` reserve: what the plan cannot meet in the
/// ledger before it runs and what no other term names - CUDA graph
/// instantiation for every captured width, the decode arena, per-width sampler
/// planes, per-image vision transients, and allocator retained-not-live slack.
///
/// This is the residual of the old 3 GiB "graph/prefill scratch" guess once
/// its two large tenants got honest accounting: the prefill scratch is now
/// profiled at load (`enable_batch_sized` allocates it before the grant is
/// read) and the wave-pass buffers are computed (`pf_bufs_bytes`). The
/// operator's `graph_scratch_mib` override still applies, to this term.
///
/// Measured 2026-09-06 on the RTX PRO 6000, Qwen3.8-27B Q8_0, max_batch 32,
/// spec on, vision, after the smoke sweep: see the audit log
/// (`qwen35 VRAM plan audit`) - the constant is the measured residual with
/// headroom, and the audit line is how to re-check it after a change.
const GRAPH_POOLS_HEADROOM_MIB: u64 = 768;

pub(super) fn graph_pools_headroom_bytes() -> u64 {
    crate::kv_plan::graph_scratch_override_mib().unwrap_or(GRAPH_POOLS_HEADROOM_MIB) << 20
}

fn dnc_vl_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_NO_DNC_VL").is_none()
            && std::env::var("PADDOCK_DNC_RS").map(|v| v != "0").unwrap_or(false)
            && std::env::var("PADDOCK_DNC_S1MMA").map(|v| v != "0").unwrap_or(false)
            && paddock_models::dev_var_os!("PADDOCK_DNC_DWB16").is_none()
            // bf16 state stays falsified and excluded; f16 state rides the
            // ST walk_rs (PPL-gated +0.09%) so it does not disqualify VL.
            && paddock_models::dev_var_os!("PADDOCK_DN_STATE_BF16").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_DNC_MMA_V2").is_none()
            && paddock_models::dev_var_os!("PADDOCK_DNC_SCAN").is_none()
            && paddock_models::dev_var_os!("PADDOCK_DNC_FLA").is_none()
            && paddock_models::dev_var_os!("PADDOCK_DNC_SPLIT").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_DN").is_none()
    })
}

/// The multimodal extras one batched-prefill share carries through
/// `prefill_batch_pass` (parallel to `items`): the request-relative
/// [4, take] axis-major mrope rows, the per-row attention bound (image
/// equal-t visibility), the (row offset, rows) embedding splices with their
/// encoded outputs, and the request's final llama-position for the decode
/// mrope delta.
pub(super) struct MmShareCtx {
    pub(super) mrope: Vec<u32>,
    pub(super) bound: Vec<u32>,
    pub(super) splices: Vec<(usize, usize)>,
    pub(super) images: Vec<super::vision::VisionOutput>,
    pub(super) final_mrope_pos: usize,
}

impl GpuQwen35 {
    /// Allocate the continuous-batching state for up to `max_batch` concurrent
    /// sequences (one KV/recurrent slot each). Row i of every batched call drives
    /// slot i. Returns the enabled capacity.
    pub fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        assert!(max_batch >= 1);
        let r = self.enable_batch_sized(max_batch);
        if r.is_err() {
            // A failed attempt leaves what it already seated (BatchState with
            // its pool + tier, per-layer MoE caches, overlap stream, dflash
            // device state, scratch) holding VRAM. The width ladder then reads
            // "0.0 GB free after weights" and every halved retry refuses
            // regardless of width - measured on an 8 GB card (RTX 3070):
            // enable_batch(32) OOM mid-way, 32->16->1 all dead, startup fatal.
            // Release the attempt before the error reaches the ladder.
            self.release_batch_attempt();
        }
        r
    }

    fn enable_batch_sized(&mut self, requested: usize) -> Result<usize, GpuModelError> {
        // Profile the serving scratch at load. The widest prefill pass the
        // scheduler can issue is allocated HERE, before the grant is read, so
        // the plan meets its true cost in the ledger instead of a constant: it
        // was charged 3 GiB and measured 6.17 GB on the 27B at 8192 rows, the
        // difference riding on the 40% hedge this replaced. When the plan
        // cannot seat the configuration beside that scratch, the prefill chunk
        // steps down (8192 -> 4096 -> ...) and the scratch shrinks with it - a
        // smaller chunk is slower prefill, a refused server is no server.
        let mut chunk = super::chunk_tick_rows();
        loop {
            let rows = self.serving_scratch_rows(chunk, requested);
            self.scratch = None;
            self.exec.trim_mem_pool();
            self.ensure_scratch(rows)?;
            match self.enable_batch_at(requested, chunk)? {
                Ok(width) => return Ok(width),
                Err(refusal) if chunk > super::PREFILL_CHUNK_ROWS_MIN => {
                    tracing::warn!(
                        chunk_rows = chunk,
                        next = chunk / 2,
                        "qwen35: {refusal} - stepping the prefill chunk down so its scratch \
                         stops eating the KV the configuration needs"
                    );
                    self.release_batch_attempt();
                    chunk /= 2;
                }
                Err(refusal) => return Err(GpuModelError::Config(refusal)),
            }
        }
    }

    /// Rows the serving scratch must hold for a `chunk`-row prefill share at
    /// `max_batch` decode rows: what the span tick asks of `ensure_scratch`,
    /// the f8t chunk arm's doubled rows, and the unified tick's own floor.
    fn serving_scratch_rows(&self, chunk: usize, max_batch: usize) -> usize {
        let tick = chunk.max(unified_prefill_rows()).min(self.max_ctx) + max_batch;
        let f8t = if f8t_unified_on() && self.bs_f8t_attn.iter().any(Option::is_some) {
            2 * tick.min(f8t_chunk_rmax())
        } else {
            0
        };
        tick.max(f8t).max(64)
    }

    /// One plan-and-allocate attempt with the serving scratch already resident
    /// for a `chunk`-row prefill share. `Ok(Err(refusal))` is the plan's own
    /// refusal, which the caller may retry at a smaller chunk; `Err` is a real
    /// failure and propagates.
    fn enable_batch_at(
        &mut self,
        max_batch: usize,
        chunk: usize,
    ) -> Result<Result<usize, String>, GpuModelError> {
        let (max_batch, spec_live_cap) = self.width_by_vram(max_batch);
        // spec live degraded to buy width (see width_by_vram) - ensure_serve_spec
        // allocates at this cap instead of the env default
        self.spec_live_vram_cap = spec_live_cap;
        // k-quant models batch on the stage-2 W4A8 ladders: the dp4a GEMM at
        // decode widths, the 128x128 int8-MMA tile at prefill widths - same
        // activation class as the Q8_0 ladders. Spec/MTP and the decode pipe
        // stay off (next W4A8 rungs).
        // spec-batch graphs bake this batch state's buffer addresses (KV, conv
        // windows, recurrent states, d_slots) - a re-enable must rebuild them
        self.spec_batch = None;
        let e = &self.exec;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let state_elems = self.n_v_heads * self.state_size * self.state_size;
        let win_elems = (self.conv_k - 1) * self.conv_dim;
        // P3/P4/P5 paged KV mode selection (computed before the KV alloc so the
        // full-attn store is a PLANNED block budget - see crate::kv_plan - and
        // never a blind reservation).
        let blocks_per_slot = self.max_ctx.div_ceil(BLOCK_TOKENS);
        let n_full = self.n_layers - self.n_linear_layers();
        let paged_capable = e.has_paged_kv() && self.max_ctx.is_multiple_of(BLOCK_TOKENS);
        let per_block_bytes = BLOCK_TOKENS * kv_dim * kv_bytes * 2 * n_full;
        // One arbiter sizes the KV store: crate::kv_plan. Until recently there
        // were two arms here - an auto-sized pool that consulted vram_headroom(),
        // and a dense `max_batch × max_ctx` reservation beside it that did not.
        // `max_batch <= 1` took the second, which is how a Qwen3.8-27B server
        // configured `vram_budget = 30720` logged a 30.0 GiB budget and then put
        // ~41 GB on the card. Dense is no longer a separate concept:
        // it is a plan whose block count reached the addressable ceiling, and it
        // is refusable like any other plan.
        //
        // Pool mode is the DEFAULT even when dense KV would fit (was:
        // "dense + 25% headroom fits -> dense"). The zero-copy radix prefix cache
        // + DeltaNet checkpoints only exist in pool mode - the chunked serving
        // prefill maintains no dense-mode cache - so the old gate silently
        // disabled all prefix reuse on the hybrids whose full-attn KV is small
        // (27B/35B: every admission re-prefilled the whole prompt; 35B-Q8 c32
        // TTFT p50 43 s, zero resumes). Prefix-heavy agentic serving is the
        // tier-1 workload, and the plan is capped at the addressable ceiling so
        // small contexts still behave densely. PADDOCK_DENSE_KV=1 restores the
        // identity path.
        let explicit_pool_pin = paddock_models::dev_var!("PADDOCK_KV_POOL_BLOCKS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n > 0);
        let want_paged = paddock_models::dev_var_os!("PADDOCK_NO_PAGED_KV").is_none()
            && paddock_models::dev_var_os!("PADDOCK_DENSE_KV").is_none();
        // Honest refusal (dense-lane nuke): an operator who did not
        // opt out of paging must never silently land in dense mode - that flip
        // used to swallow prefix reuse + the paged serving lanes with no signal
        // (same trap class as the max_batch=1 serial fallback). paged_capable
        // false here means the PACK lacks the paged-KV kernels (max_ctx is
        // rounded to the page grid at serve entry).
        if want_paged && !paged_capable {
            return Err(GpuModelError::Config(format!(
                "kernel pack has no paged-KV support for qwen35 (paged kernels \
                 missing, or max_ctx {} not a multiple of {BLOCK_TOKENS}) - \
                 update the pack; PADDOCK_DENSE_KV=1 is the explicit dense A/B \
                 escape hatch (no prefix cache)",
                self.max_ctx
            )));
        }
        let paged = (want_paged || explicit_pool_pin.is_some()) && paged_capable;
        self.exec.trim_mem_pool(); // pool-held frees must not read as used
        // Budget-aware headroom, not raw device free: under a configured
        // vram_budget the KV must size inside our slice of the card, not against
        // bytes granted to other runners. A MISSING reading is an error, not
        // permission - treating "the driver did not answer" as "take what you
        // like" is the other half of the same bug.
        let grant = self.exec.vram_headroom().ok_or_else(|| {
            GpuModelError::Config(
                "qwen35: the driver gave no free-VRAM reading, so the KV cache \
                 cannot be sized inside this server's budget - refusing rather \
                 than allocating blind"
                    .into(),
            )
        })?;
        // What the process holds as the plan is made (weights, the profiled
        // serving scratch): the audit below adds the plan's own charges to it
        // and compares against the ledger once everything is allocated.
        let ledger_at_plan = self.exec.process_mem_used().unwrap_or(0);
        let state_win = (state_elems + win_elems) as u64;
        let n_lin = self.n_linear_layers() as u64;
        let per_ckpt = n_lin * state_win * 4;
        // The serving-spec draft state allocates LAZILY (first spec round), so the
        // plan must leave room for it or the card over-commits once spec engages.
        // 27B-Q4 measured: honest reserves alone budgeted an 11.8 GB pool, the
        // real lazy state pushed past free, and c1 collapsed 74 -> 31 t/s.
        let spec_live = self
            .spec_live_vram_cap
            .unwrap_or_else(|| self.serve_spec_live_max())
            .clamp(1, max_batch) as u64;
        let k1s = (self.serve_spec_k() + 1) as u64;
        // Drafter KV: on the paged lane it rides the pool as one more
        // full-attn-shaped STRIPE (see enable_spec_batch), priced into
        // block_bytes so the plan's block count already pays for it; the
        // per-slot dense term below applies only to the dense lane.
        let spec_possible = self.serve_spec_on() && self.mtp.is_some();
        let mtp_stripe_bytes: u64 = if spec_possible && paged {
            2 * BLOCK_TOKENS as u64 * kv_dim as u64 * kv_bytes as u64
        } else {
            0
        };
        // DFlash feature-KV stripe: n_layers × K,V × 16 tokens × kv_dim, f16
        // (dflash_ensure_state sizes the planes off the pool this plan buys)
        let dflash_stripe_bytes: u64 = match self.dflash.as_ref().filter(|_| paged) {
            Some(df) => {
                df.layers.len() as u64 * 2 * BLOCK_TOKENS as u64 * (df.n_kv * df.hd) as u64 * 2
            }
            None => 0,
        };
        // Verify-state term: the legacy path snapshots the full state per
        // draft position (n_lin x k1 x state_elems - ~87% of the draft
        // state, the 14-spec-row cap on 96 GB); the snapshot-free path
        // (dflash) stashes only the round's split/gate planes
        // (k_hat + v + g + beta), ~state_size times smaller. Same condition
        // as the enable_spec_batch alloc so plan and alloc cannot drift.
        let verify_state_row: u64 = if self.spec_snapshot_verify() {
            n_lin * k1s * state_elems as u64 * 4
        } else {
            n_lin * k1s * 2 * ((state_elems / self.state_size) as u64 + self.n_v_heads as u64) * 4
        };
        let spec_est: u64 = if self.serve_spec_on() {
            spec_live
                * (if paged {
                    0
                } else {
                    2 * self.max_ctx as u64 * kv_dim as u64 * kv_bytes as u64
                } + k1s * self.vocab as u64 * 4
                    + verify_state_row
                    + n_lin * (self.conv_k as u64 - 1 + k1s) * self.conv_dim as u64 * 4)
        } else {
            0
        };
        // Every term the plan charges, named. conv_ext + conv_out are
        // span-sized; the wave buffers are what `ensure_pf_bufs` will allocate
        // for a chunk-row wave (computed by the same arithmetic, checked
        // against the allocator on first use); the residual is measured.
        let conv_staging = 2
            * (self.conv_k as u64 - 1 + unified_prefill_rows().max(8192) as u64)
            * self.conv_dim as u64
            * 4;
        let ckpt_staging = 2 * n_lin * state_win * 4;
        let wave_bufs = self.pf_bufs_bytes(chunk, chunk, max_batch);
        let graph_headroom = graph_pools_headroom_bytes();
        let tier_staging = if crate::kv_tier::pool_tier::tier_ram_bytes().is_some() {
            crate::kv_tier::ram_transport::device_staging_bytes()
        } else {
            0
        };
        let block_bytes = per_block_bytes as u64 + mtp_stripe_bytes + dflash_stripe_bytes;
        // recurrent + conv state, this slot's logits row, its block table
        let per_slot_bytes =
            n_lin * state_win * 4 + self.vocab as u64 * 4 + blocks_per_slot as u64 * 4;
        // The checkpoint pool is sized by demand out of what the grant has left
        // once everything else - including a full-context pool for every slot -
        // is charged. Same count for the reserve and the P5c allocation below.
        let charged_without_pool = conv_staging
            + ckpt_staging
            + spec_est
            + wave_bufs
            + graph_headroom
            + tier_staging
            + max_batch as u64 * per_slot_bytes
            + max_batch as u64 * blocks_per_slot as u64 * block_bytes;
        let n_ckpt_planned = state_ckpt_count(
            max_batch,
            per_ckpt,
            grant.saturating_sub(charged_without_pool),
        );
        let n_ckpt_est = n_ckpt_planned as u64;
        let demand = kv_plan::Demand {
            family: "qwen35",
            max_ctx: self.max_ctx,
            slots: max_batch,
            blocks_per_slot,
            block_bytes,
            per_slot_bytes,
            floor_blocks_per_slot: 128,
            reserves: vec![
                kv_plan::Reserve::new("conv staging", conv_staging),
                kv_plan::Reserve::new("prefix state pool", n_ckpt_est * per_ckpt),
                kv_plan::Reserve::new("checkpoint staging", ckpt_staging),
                kv_plan::Reserve::new("draft state (spec)", spec_est),
                kv_plan::Reserve::new("prefill wave buffers", wave_bufs),
                kv_plan::Reserve::new("graph pools + headroom", graph_headroom),
                kv_plan::Reserve::new("kv-tier staging", tier_staging),
            ],
            // Issue 2: when the budget cannot back --max-ctx ×
            // --max-batch, refuse loudly instead of silently under-sizing -
            // measured, 16384×32 sized 23 of 32 slots and the 9 un-admittable
            // requests queued into 152 s TTFT tails with no warning. Dense
            // identity addressing also needs exactly the full ceiling (block b of
            // slot s lives at s*bps+j), so oversubscription is only offered on
            // the paged lane.
            when_short: if paged
                && paddock_models::dev_var_os!("PADDOCK_KV_OVERSUBSCRIBE").is_some()
            {
                kv_plan::WhenShort::Narrow
            } else {
                kv_plan::WhenShort::Refuse
            },
            // No hedge (was 40% of the grant, 2026-08 .. 2026-09-06): every
            // post-plan allocation is now either in the ledger before the grant
            // is read (the profiled serving scratch) or a named reserve above,
            // and the plan audit at the end of this function is what keeps it
            // that way. The hedge alone refused a 64.5 GiB budget for one 262k
            // slot that needed 16 GiB of KV - "only 14.96 fits" - with 26 GiB
            // of grant unspent.
            hedge_fraction: None,
            ..Default::default()
        };
        let plan = match demand.plan(grant) {
            Ok(p) => p,
            Err(e) => return Ok(Err(e.message)),
        };
        plan.report(&demand, grant);
        self.prefill_chunk_rows = chunk;
        if chunk < super::chunk_tick_rows() {
            tracing::warn!(
                elected = chunk,
                requested = super::chunk_tick_rows(),
                "qwen35 prefill chunk elected below the request: the serving scratch for \
                 the full chunk would not fit beside the KV this configuration needs"
            );
        }
        // What the plan leaves after its pools and every listed reserve is
        // the expert-offload slot cache's budget (sized at the end of this
        // function, once the pools are allocated).
        let plan_reserved: u64 = demand.reserves.iter().map(|r| r.bytes).sum::<u64>()
            + plan.pool_bytes
            + plan.slot_bytes;
        let moe_cache_budget = grant.saturating_sub(plan_reserved);
        // An explicit pin is a development instrument: it overrides the plan
        // outright, including past the budget, and says so.
        let pool_block_count = match explicit_pool_pin {
            Some(n) => {
                tracing::warn!(
                    pinned = n,
                    planned = plan.pool_blocks,
                    "PADDOCK_KV_POOL_BLOCKS overrides the KV plan - not budget-checked"
                );
                n as usize
            }
            None => plan.pool_blocks,
        };
        let pool_active = paged;
        // full-attn K (and V) store per layer. One expression for every mode:
        // pool mode gets the planned shared budget; dense identity gets a plan
        // that reached the addressable ceiling, which is max_batch × max_ctx.
        let full_kv_bytes = pool_block_count * BLOCK_TOKENS * kv_dim * kv_bytes;
        let (mut kv_k, mut kv_v, mut recur, mut conv_win) = (
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
        );
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Full(_) => {
                    kv_k.push(Some(e.alloc_u8(full_kv_bytes)?));
                    kv_v.push(Some(e.alloc_u8(full_kv_bytes)?));
                    recur.push(None);
                    conv_win.push(None);
                }
                Mixer::Linear(_) => {
                    kv_k.push(None);
                    kv_v.push(None);
                    recur.push(Some(e.alloc(max_batch * state_elems)?));
                    conv_win.push(Some(e.alloc(max_batch * win_elems)?));
                }
            }
        }
        let slots_host: Vec<u32> = (0..max_batch as u32).collect();
        // Paged KV device state. `d_block_tables` is [max_batch*bps] u32:
        //   P4 identity mode: filled bt[s*bps+j]=s*bps+j; the pool overlays the
        //     dense [max_batch,max_ctx,kv_dim] so tokens match dense bit-for-bit.
        //   P5 pool mode: starts zeroed; per-slot BlockTables grow it from the
        //     shared free-list on demand (ensure_slot_blocks), so KV follows the
        //     block budget not max_batch×max_ctx. On this A6000 (84 SMs < 128)
        //     attn_splits is always 1 -> the single-pass paged kernels cover
        //     decode; the split/GQA-fused paged path (≥128 SM) is P3b.
        let (d_block_tables, pool, tables, block_table_host) = if pool_active {
            let n = pool_block_count as u32;
            let gb = (n as f64
                * BLOCK_TOKENS as f64
                * kv_dim as f64
                * kv_bytes as f64
                * 2.0
                * n_full as f64)
                / 1e9;
            tracing::info!(
                "qwen35 paged KV BUDGET POOL active ({n} blocks, {gb:.2} GB full-attn KV, blocks_per_slot={blocks_per_slot})"
            );
            (
                Some(e.alloc_u32(max_batch * blocks_per_slot)?),
                Some(KvPool::with_blocks(n)),
                (0..max_batch).map(|_| BlockTable::new()).collect(),
                vec![0u32; max_batch * blocks_per_slot],
            )
        } else if paged {
            let bt: Vec<u32> = (0..(max_batch * blocks_per_slot) as u32).collect();
            tracing::info!(
                "qwen35 paged KV decode active (identity block table, blocks_per_slot={blocks_per_slot})"
            );
            (Some(e.to_device_u32(&bt)?), None, Vec::new(), Vec::new())
        } else {
            if want_paged || explicit_pool_pin.is_some() {
                tracing::info!(
                    "paged KV wanted but inactive - using dense (has_paged_kv={}, max_ctx={} not a {}-multiple)",
                    e.has_paged_kv(),
                    self.max_ctx,
                    BLOCK_TOKENS
                );
            }
            (None, None, Vec::new(), Vec::new())
        };
        // P5c zero-copy radix prefix cache (pool mode, unless PADDOCK_NO_PREFIX_CACHE).
        // Caches full-attn KV blocks (shared by refcount) + DeltaNet recurrent-state
        // checkpoints in d_state_pool (state_ckpt_f32 per checkpoint = the linear
        // layers' state + conv window).
        let state_ckpt_f32 = self.n_linear_layers() * (state_elems + win_elems);
        let (paged_prefix, d_state_pool) = if pool_active
            && paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_none()
        {
            // Exactly what the kv_plan reserve above charged for: same fn,
            // same budget. Deriving it a second time here (and from
            // `vram_headroom()` rather than the grant) is what let the
            // reservation and the allocation disagree.
            let n_ckpt = n_ckpt_planned;
            let mut pr = PagedRadix::new();
            // State-pool admission control. The board's c32 leg cycles 128
            // distinct prefixes through this pool, so plain LRU steal thrashes
            // it to a ~0% checkpoint hit rate; protecting proven prefixes also
            // skips the ~170 MiB snapshot on every refused admission.
            pr.set_protect_proven(paddock_models::dev_var_os!("PADDOCK_CKPT_PROTECT").is_some());
            let pool_f32 = if state_ckpt_f32 > 0 {
                pr.set_state_capacity(n_ckpt);
                Some(e.alloc(n_ckpt as usize * state_ckpt_f32)?)
            } else {
                None // pure full-attn model: no DeltaNet state to checkpoint
            };
            tracing::info!(
                "qwen35 paged zero-copy prefix cache active ({n_ckpt} state checkpoints)"
            );
            (Some(pr), pool_f32)
        } else {
            (None, None)
        };
        // KV tier (kv-offload 1b.3): full-attn pool planes; DeltaNet
        // checkpoint blobs ride as aux components. Loud decline on any
        // failure - serving continues untiered.
        let tier = match (
            paged_prefix.as_ref(),
            crate::kv_tier::pool_tier::tier_ram_bytes(),
        ) {
            (Some(_), Some(ram)) => {
                use crate::kv_tier::digest::{IdentityDigest, IdentityFields, PrivacyScope};
                use crate::kv_tier::{CacheNamespace, PlaneDesc, PoolTier, RamTransport};
                use cudarc::driver::DevicePtr;
                let stride = (BLOCK_TOKENS * kv_dim * kv_bytes) as u64;
                let mut planes = Vec::new();
                for li in 0..self.layers.len() {
                    if let (Some(k), Some(v)) = (kv_k[li].as_ref(), kv_v[li].as_ref()) {
                        for plane in [k, v] {
                            let (pp, _g) = plane.device_ptr(&e.stream);
                            planes.push(PlaneDesc {
                                base: pp,
                                stride,
                                bytes: stride,
                            });
                        }
                    }
                }
                let content_id = self.content_id;
                let architecture = format!(
                    "qwen35 v1 full_layers={} kv_dim={kv_dim} kvb={kv_bytes} max_ctx={} state_ckpt_f32={state_ckpt_f32}",
                    planes.len() / 2,
                    self.max_ctx,
                );
                let ns = CacheNamespace {
                    identity: IdentityDigest::compute(&IdentityFields {
                        model_tensors: &content_id.0,
                        adapter: b"",
                        architecture: architecture.as_bytes(),
                        cache_schema: b"pool-planes k/v interleaved + dn-ckpt aux v1",
                        layout_abi: 1,
                        tokenizer: &content_id.1,
                    }),
                    scope: PrivacyScope::Shared,
                };
                let transport = match crate::kv_tier::pool_tier::nvme_dir_for(&ns) {
                    Some((dir, quota)) => RamTransport::with_t2(e, ram, &dir, quota),
                    None => RamTransport::new(e, ram),
                };
                match transport
                    .map_err(|x| x.to_string())
                    .and_then(|t| PoolTier::new(&ns, planes, ram, t).map_err(|x| x.to_string()))
                {
                    Ok(mut t) => {
                        t.preload_from_t2();
                        Some(t)
                    }
                    Err(err) => {
                        tracing::warn!(err = %err, "qwen35 KV tier declined");
                        None
                    }
                }
            }
            _ => None,
        };
        let mut paged_prefix = paged_prefix;
        if let (Some(pr), Some(t)) = (paged_prefix.as_mut(), tier.as_ref()) {
            pr.set_tier_root(t.tier_root());
        }
        self.batch = Some(BatchState {
            max_batch,
            tier,
            mtp_cover: std::collections::HashSet::new(),
            dflash_cover: std::collections::HashSet::new(),
            kv_k,
            kv_v,
            recur,
            conv_win,
            d_tokens: e.alloc_u32(max_batch)?,
            d_pos: e.alloc_u32(max_batch)?,
            d_slots: e.to_device_u32(&slots_host)?,
            d_mrope: e.alloc_u32(4 * max_batch)?,
            d_logits: e.alloc(max_batch * self.vocab)?,
            d_samp_par: e.alloc_u32(max_batch * 4)?,
            d_samp_out: e.alloc_u32(max_batch)?,
            d_samp_head: e.alloc_u32(max_batch * 64 * 2)?,
            d_fin_head: e.alloc_u32(max_batch * 64 * 2)?,
            d_samp_tpar: e.alloc_u32(max_batch * 4)?,
            d_fin_tpar: e.alloc_u32(max_batch * 4)?,
            d_pipe_tpar: e.alloc_u32(2 * max_batch * 4)?,
            d_pipe_par: e.alloc_u32(2 * max_batch * 4)?,
            d_pipe_out: e.alloc_u32(2 * max_batch)?,
            d_fin_logits: e.alloc(max_batch * self.vocab)?,
            d_fin_par: e.alloc_u32(max_batch * 4)?,
            d_fin_out: e.alloc_u32(max_batch)?,
            d_xq: e.alloc_i8(max_batch * self.ff.max(2 * self.embd))?,
            d_xs: e.alloc(max_batch * self.ff.max(2 * self.embd) / 32)?,
            // K-split mma partial planes at the UNIFORM 64-row envelope: every
            // holder sizes identically so the mmq capacity predicate is the
            // same on both sides of every spec exactness pair, and bmm!'s
            // unchecked serving rung can never outgrow it (the 35B's
            // 8192-wide in_qkv/wq outgrew the old ff-based size at B>=24).
            d_ks_part: e.alloc(8 * 64 * self.ks_out_max())?,
            // lin-GEMV ticket regions (non-KV-overhead R2.2): gate|up at
            // offset 0, down at `lin_tick_dn_off()`. Zeroed once - the
            // kernel's atomicInc wraps at nz, restoring 0 each launch.
            d_lin_tick: e.alloc_u32(self.lin_tick_len())?,
            // fused gate|up GEMM landing ([max_batch, 2*ff]) - only when any
            // fused plane loaded (4.5 MB at 32x2x17408 on the 27B)
            //
            // The W8 fused wq|wk|wv decode arm lands here too ("into the gu
            // landing", the b>=8 f8_qkv arm), so its plane must be on this
            // list: on a MoE fp8-native serve (qwen3.6-35b-a3b) the attention
            // W8 planes load with no dense gate|up/f8ffn plane, and the old
            // list left this None while the arm engaged - expect("qkv fused
            // landing") panicked the engine at c8. Dense models
            // never saw it because bs_gu/bs_f8ffn always allocated the buffer
            // first. Same shape as d_dn_fused's !bs_w8.is_empty() arm below,
            // but on the exact plane the arm keys on (l.wq).
            d_gu_fused: if self.bs_gu.iter().any(Option::is_some)
                || self.bs_f8ffn.iter().any(Option::is_some)
                || self.bs_f8t_attn.iter().any(Option::is_some)
                || self.bs_w8.iter().any(|l| l.wq.is_some())
            {
                // the f8t mixer lane lands its fused [2q|k|v] row here too,
                // which the 2*ff sizing does not cover by construction
                let w = (2 * self.ff)
                    .max(2 * self.n_heads * self.head_dim + 2 * self.n_kv_heads * self.head_dim);
                Some(e.alloc(max_batch * w)?)
            } else {
                None
            },
            // fused DN landing (conv_dim + value_dim = 16384 on the 27B; 2 MB at
            // b=32). +128 for the optional alpha||beta tile block the f8t
            // in-proj plane can carry (see the fuse_ab note in load.rs); the
            // slack is one tile row and costs 512 B/row, so it is unconditional
            // rather than another shape to keep in sync.
            d_dn_fused: if self.bs_dn.iter().any(Option::is_some)
                || (!self.bs_w8.is_empty() && self.n_linear_layers() > 0)
                || self.bs_f8t_attn.iter().any(Option::is_some)
            {
                Some(e.alloc(max_batch * (self.conv_dim + self.n_v_heads * self.state_size + 128))?)
            } else {
                None
            },
            d_zero_state: e.alloc(state_elems)?,
            d_zero_win: e.alloc(win_elems)?,
            // conv ext staging: sized by the LARGEST RESUMED SPAN, not
            // max_ctx - every user (unified resumed shares <= cap+TAIL_SLOP,
            // serial/prefix-resume 2048-chunks; the mm path builds no ext,
            // fresh sequences conv in place) is span-bounded, and max_ctx
            // sizing made this pair the long-context blocker (2 x 18 GB at
            // 262k ctx, and real agentic traces need 128-256k). 8192-row
            // floor = 4x any current span
            // budget; the asserts at the ext-build sites guard the
            // invariant if a caller ever outgrows it.
            d_conv_ext: e
                .alloc((self.conv_k - 1 + unified_prefill_rows().max(8192)) * self.conv_dim)?,
            d_conv_out: e
                .alloc((self.conv_k - 1 + unified_prefill_rows().max(8192)) * self.conv_dim)?,
            d_ckpt_stage: {
                let blob = self.n_linear_layers() * (state_elems + win_elems);
                vec![e.alloc(blob)?, e.alloc(blob)?]
            },
            mrope_delta: vec![0; max_batch],
            paged,
            blocks_per_slot,
            d_block_tables,
            pool,
            tables,
            block_table_host,
            paged_prefix,
            d_ckpt_desc: if d_state_pool.is_some() {
                Some(self.exec.alloc_u64(self.n_layers * 6)?)
            } else {
                None
            },
            d_state_pool,
            state_ckpt_f32,
            graphs: std::collections::HashMap::new(),
            pf_bufs: None,
            pf_pass_graphs: std::collections::HashMap::new(),
        });
        self.last_reused = vec![0; max_batch];
        // Dedicated decode arena (pipe-scratch separation step 1):
        // the captured decode graphs bake scratch ADDRESSES, and baking the
        // shared prefill scratch means (a) every prefill-scratch realloc kills
        // the graphs and (b) queued pipe ticks can never overlap a prefill
        // pass (the pass would clobber the rows the queued graph reads). A
        // max_batch-row arena is ~tens of MB and makes the decode graph
        // self-contained. Built via the ensure_scratch swap so the layout is
        // byte-identical to the shared arena's - the graph body is unchanged.
        {
            let saved = self.scratch.take();
            self.ensure_scratch(max_batch)?;
            self.decode_arena = self.scratch.take();
            self.scratch = saved;
        }
        // ...and pre-size the shared prefill scratch to its ceiling, for the
        // other half of the same hazard. The decode graphs got their own arena
        // above; the BATCH and SPEC graphs still bake this buffer, and
        // ensure_scratch drops every one of them on a grow.
        //
        // Growth is admission-dependent, so the same code on the same workload
        // lands at different speeds run to run. Measured at wide batch,
        // no-spec: cap climbed 532 -> 1989 -> 2791 -> 3556 -> 3945 -> 4128 rows
        // with the last two arriving TEN SECONDS into the run, i.e. six full
        // graph drops mid-serve. Wave throughput ramped 1563 -> 2188 -> 2273
        // and only then settled. The same cell on identical code therefore
        // reads ~15% apart between sittings, depending on whether the legs
        // cleared the ramp.
        //
        // The ceiling is knowable - but it is the SCHEDULER's per-tick row
        // budget, not max_ctx. A tick carries at most one prefill share plus
        // the decode rows, and prefill is CHUNKED: `advance_chunks` caps every
        // share at chunk_tick_rows() (8192) and the fused tick at
        // unified_prefill_rows() (64). A prompt longer than that does not
        // arrive in one pass; it chunks across ticks. So max_ctx overstates
        // the requirement by max_ctx/8192 - 8x at 65536.
        //
        // That is the same quantity every serving engine sizes activation
        // workspace from: vLLM's `max_num_batched_tokens` and SGLang's
        // `chunked_prefill_size` are per-STEP token budgets, deliberately
        // decoupled from `max_model_len` for exactly this reason.
        //
        // MEASURED on an A6000 (48 GiB), qwen3.8-27b UD-Q4_K_XL,
        // max_batch=1, vram_budget 40 GiB. Presizing to max_ctx cost 19.52 GiB
        // of scratch at max_ctx=16384 and OOM'd outright at 32768 and 65536 -
        // and the failure was not confined to the presize: enable_batch(1)
        // returned Err, service.rs fell back to the serial loop, and the
        // serial loop has no prefix cache and no tuned W4A8 GEMV. A 65536-ctx
        // agentic serve therefore re-prefilled its whole context every tool
        // round (57,365 prefill tokens for a ~20k conversation, hit rate 0.0)
        // and decoded at a crawl. One 489-second answer, from this line.
        //
        // The anti-ramp benefit is unchanged: the allocation still happens
        // once, at boot, and still covers every tick the scheduler can build.
        // ensure_scratch stays the on-demand backstop for any path that ever
        // asks for more (it grows and drops graphs, as before) - this is a
        // pre-size, not a cap. Kill: PADDOCK_NO_QWEN35_PRESIZE_SCRATCH.
        if paddock_models::dev_var_os!("PADDOCK_NO_QWEN35_PRESIZE_SCRATCH").is_none() {
            // ensure_scratch pads by max_batch itself, so this is the prefill
            // share alone.
            let tick_rows = self.prefill_chunk_rows.max(super::unified_prefill_rows());
            self.ensure_scratch(tick_rows.min(self.max_ctx))?;
        }
        // Route-B overlap lane: fork a second execution lane so the
        // decode graphs capture - and thus replay - on their own stream, off
        // the prefill lane. fork_stream drains the parent stream first, so
        // everything enable_batch just allocated/zeroed is visible to the new
        // lane. With the lane forked but no scheduler interleave, behavior is
        // identical to single-lane (every tick stays host-serialized); the
        // fork is the substrate the overlap scheduler builds on.
        // Default on: at syn_2048x128_c32 with 8192-row
        // chunk ticks the overlap route (2o) pumped 19-36 decode ticks per
        // 5 s window under running spans (decode rows advancing ~4x more
        // during the prefill phase) and cut TTFT p50 2618 -> 616 ms at
        // equal throughput. The scheduler's overlap_ok conditions still
        // guard per tick; PADDOCK_OVERLAP=0 is the A/B kill switch.
        // [moe_offload]: the slot cache takes what the plan left, less a
        // 256 MiB allocator margin - the operator's levers stay max_ctx and
        // max_batch, exactly as for the KV pool.
        if !self.moe_cache_active() {
            self.enable_moe_cache(moe_cache_budget.saturating_sub(256 << 20))?;
        }
        let overlap_on = paddock_models::dev_var!("PADDOCK_OVERLAP")
            .map(|v| v != "0")
            .unwrap_or(true);
        // The slot cache is per-layer device state updated in-graph; two
        // execution lanes replaying against it would race the LRU bookkeeping.
        let overlap_on = overlap_on && !self.moe_cache_active();
        if self.overlap_exec.is_none() && overlap_on {
            self.overlap_exec = Some(Arc::new(self.exec.fork_stream()?));
            tracing::info!(
                "qwen35 overlap: decode execution lane forked (decode graphs capture on it)"
            );
        }
        // DFlash serving state arms here, before any graph capture - the
        // taps are baked into the pf/decode/verify graphs, so an armed-after
        // capture would leave every span untapped and every slot cold.
        self.dflash_ensure_state().map_err(GpuModelError::from)?;
        // Plan audit: what the plan charged (on top of what the process held
        // when it was made) against what the process holds now that every
        // pool, slot, stage and cache is allocated. The gap is what the old
        // 40% hedge silently covered; it belongs in `graph pools + headroom`
        // (or a new named term) and this line is how it stays visible. Serving
        // adds only what that reserve names - graphs, the decode arena, the
        // lazily seated spec state (its own reserve) - so a later reading of
        // the ledger against `expected` is the same check under load.
        let expected = ledger_at_plan + plan_reserved;
        let actual = self.exec.process_mem_used().unwrap_or(0);
        tracing::info!(
            expected_gib = expected as f64 / (1u64 << 30) as f64,
            ledger_gib = actual as f64 / (1u64 << 30) as f64,
            unplanned_mib = actual.saturating_sub(expected) as f64 / (1u64 << 20) as f64,
            moe_cache_active = self.moe_cache_active(),
            "qwen35 VRAM plan audit: ledger vs plan after enable_batch"
        );
        Ok(Ok(max_batch))
    }

    /// P5 budget pool: grow slot `slot`'s block table to back logical position
    /// `upto_pos`, allocating physical blocks from the shared free-list and
    /// re-uploading the (small) block table to the device. No-op in identity
    /// mode (`pool` is `None`). The device upload is outside any captured graph
    /// (callers invoke this before the capture/replay), and the paged kernels
    /// read the table by pointer at replay - so growth is graph-safe.
    pub(super) fn ensure_slot_blocks(
        &mut self,
        slot: usize,
        upto_pos: usize,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        if self.batch.as_ref().expect("batch").pool.is_none() {
            return Ok(());
        }
        let before = self.batch.as_ref().expect("batch enabled").tables[slot]
            .blocks()
            .len();
        // Grow the slot's table from the pool. On exhaustion, evict an LRU cached
        // prefix - its pages return to the pool (blocks a live slot still holds
        // survive on refcount) - and retry; when the tree is dry, surface
        // PoolExhausted for the scheduler to preempt.
        loop {
            let grew = {
                let bs = self.batch.as_mut().expect("batch enabled");
                let pool = bs.pool.as_mut().expect("pool checked above");
                bs.tables[slot].ensure(upto_pos, pool).is_ok()
            };
            if grew {
                break;
            }
            let evicted = {
                let exec = self.exec.clone();
                let state_bytes;
                let ckpt_base;
                {
                    let bs = self.batch.as_ref().expect("batch enabled");
                    state_bytes = (bs.state_ckpt_f32 * 4) as u64;
                    ckpt_base = bs.d_state_pool.as_ref().map(|sp| {
                        use cudarc::driver::DevicePtr;
                        let (pp, _g) = sp.device_ptr(&exec.stream);
                        pp
                    });
                }
                let bs = self.batch.as_mut().expect("batch enabled");
                let pool = bs.pool.as_mut().expect("pool checked above");
                match (bs.tier.as_mut(), bs.paged_prefix.as_mut()) {
                    (Some(tier), Some(pr)) => {
                        // tier-aware shed: closing runs (and their DeltaNet
                        // checkpoint blobs) demote to T1 before eviction -
                        // the cliff-grade press (see `make_room_blocking`:
                        // one pressure pass + 50ms was not enough while a
                        // parked restore's loads held the lane)
                        let want = pool.free_blocks() + 1;
                        tier.make_room_blocking(
                            pr,
                            pool,
                            want,
                            ckpt_base.map(|b| (b, state_bytes)),
                            &mut || exec.record_event().ok(),
                        )
                        .then_some(0u32)
                    }
                    (None, Some(pr)) => pr.evict_lru(pool),
                    _ => None,
                }
            };
            if evicted.is_none() {
                return Err(GpuModelError::PoolExhausted);
            }
        }
        let bs = self.batch.as_mut().expect("batch enabled");
        let pool = bs.pool.as_mut().expect("pool checked above");
        let (free, cap) = (pool.free_blocks(), pool.capacity());
        let blocks = bs.tables[slot].blocks().to_vec();
        if blocks.len() > before {
            if paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some() {
                tracing::info!(
                    "pool: slot {slot} grew {before}->{} blocks  ({free}/{cap} free)",
                    blocks.len()
                );
            }
            let base = slot * bs.blocks_per_slot;
            for (j, &blk) in blocks.iter().enumerate() {
                bs.block_table_host[base + j] = blk;
            }
            let dst = bs
                .d_block_tables
                .as_mut()
                .expect("pool implies device table");
            exec.stream
                .memcpy_htod(&bs.block_table_host, dst)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        Ok(())
    }

    /// Record whether state checkpoint `idx` (attached at boundary `c` for
    /// `slot`, prompt prefix `tokens[..c]`) has the DRAFTER's pool-stripe KV
    /// rows behind it - i.e. the slot's warm chain covers [0..c) with exactly
    /// these tokens. Called at every `attach_state` site; see
    /// `BatchState::mtp_cover` for the resume contract.
    /// DFlash sibling of `record_mtp_cover`: checkpoint `idx` carries the
    /// drafter's feature-KV stripe rows for [0..c) iff the writer's ring
    /// covers that span from position 0 with exactly these tokens (paged
    /// stripe mode only - the dense ring is slot-local and never rides).
    pub(super) fn record_dflash_cover(&mut self, idx: u32, slot: usize, c: usize, tokens: &[u32]) {
        let covered = self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .is_some_and(|st| {
                st.paged
                    && st
                        .feat
                        .get(slot)
                        .is_some_and(|&(s, e)| s == 0 && e as usize >= c)
                    && st
                        .cov
                        .get(slot)
                        .is_some_and(|cov| cov.len() >= c && cov[..c] == tokens[..c])
            });
        if let Some(bs) = self.batch.as_mut() {
            if covered {
                bs.dflash_cover.insert(idx);
            } else {
                bs.dflash_cover.remove(&idx);
            }
        }
    }

    pub(super) fn record_mtp_cover(&mut self, idx: u32, slot: usize, c: usize, tokens: &[u32]) {
        let covered = self.spec_batch.as_ref().is_some_and(|sb| {
            slot < sb.alloc_batch
                && sb.mtp_warm[slot]
                && sb.pos[slot] >= c
                && sb.mtp_toks[slot].len() >= c
                && sb.mtp_toks[slot][..c] == tokens[..c]
        });
        if let Some(bs) = self.batch.as_mut() {
            if covered {
                bs.mtp_cover.insert(idx);
            } else {
                bs.mtp_cover.remove(&idx);
            }
        }
    }

    /// P5c: snapshot `slot`'s DeltaNet recurrent state (states + conv windows,
    /// per linear layer) into the paged state pool at checkpoint index `idx` -
    /// the persisted resume point a later shared-prefix request restores.
    pub(super) fn snapshot_paged_state(
        &mut self,
        slot: usize,
        idx: u32,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let state_elems = self.n_v_heads * self.state_size * self.state_size;
        let win_elems = (self.conv_k - 1) * self.conv_dim;
        let n_layers = self.n_layers;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        let Some(sp) = bs.d_state_pool.as_ref() else {
            return Ok(());
        };
        let (pp, _g) = sp.device_ptr(&exec.stream);
        let mut descs: Vec<u64> = Vec::new();
        // state elements may be bf16 (PADDOCK_DN_STATE_BF16): the recur-side
        // offsets/lengths scale by esz, while the pool layout stays f32-sized
        // (checkpoint blocks half-used under bf16 - correctness over compaction)
        let esz = GpuExecutor::dn_state_esz();
        let mut boff = (idx as usize * bs.state_ckpt_f32 * 4) as u64;
        for li in 0..n_layers {
            let Some(r) = bs.recur[li].as_ref() else {
                continue;
            };
            let (rp, _g1) = r.device_ptr(&exec.stream);
            descs.extend([
                rp + slot as u64 * state_elems as u64 * esz,
                pp + boff,
                state_elems as u64 * esz,
            ]);
            boff += (state_elems * 4) as u64;
            let w = bs.conv_win[li].as_ref().expect("linear layer has window");
            let (wp, _g2) = w.device_ptr(&exec.stream);
            descs.extend([
                wp + (slot * win_elems * 4) as u64,
                pp + boff,
                (win_elems * 4) as u64,
            ]);
            boff += (win_elems * 4) as u64;
        }
        // No allocation on this path: the descriptor buffer is owned by the
        // batch state and address-stable (see BatchState::d_ckpt_desc).
        let n = descs.len();
        if n == 0 {
            return Ok(());
        }
        let Some(db) = bs.d_ckpt_desc.as_mut() else {
            return Ok(());
        };
        debug_assert!(n <= db.len());
        {
            let mut v = db.slice_mut(0..n);
            exec.stream
                .memcpy_htod(&descs, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        exec.batched_copy(&*db, n / 3)?;
        Ok(())
    }

    /// P5c fused-tail variant: attach checkpoint `idx` from staging blob
    /// `stage` (d_ckpt_stage - filled per-layer at the boundary during a
    /// fused tick) instead of the slot's live state, which by finish has
    /// advanced past the boundary. The blob is laid out exactly like a pool
    /// checkpoint, so this is one flat copy.
    pub(super) fn snapshot_staged_pool(
        &mut self,
        stage: usize,
        idx: u32,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        let n = bs.state_ckpt_f32;
        let Some(sp) = bs.d_state_pool.as_mut() else {
            return Ok(());
        };
        exec.copy_region(&bs.d_ckpt_stage[stage], 0, sp, idx as usize * n, n)?;
        Ok(())
    }

    /// P5c: restore `slot`'s DeltaNet state from the paged state pool checkpoint
    /// `idx` (the reverse of `snapshot_paged_state`) - the hybrid half of a
    /// zero-copy resume (the KV half is `BlockTable::share_prefix`).
    pub(super) fn restore_paged_state(
        &mut self,
        slot: usize,
        idx: u32,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let state_elems = self.n_v_heads * self.state_size * self.state_size;
        let win_elems = (self.conv_k - 1) * self.conv_dim;
        let n_layers = self.n_layers;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        let Some(sp) = bs.d_state_pool.as_ref() else {
            return Ok(());
        };
        let (pp, _g) = sp.device_ptr(&exec.stream);
        let mut descs: Vec<u64> = Vec::new();
        let esz = GpuExecutor::dn_state_esz();
        let mut boff = (idx as usize * bs.state_ckpt_f32 * 4) as u64;
        for li in 0..n_layers {
            let Some(r) = bs.recur[li].as_ref() else {
                continue;
            };
            let (rp, _g1) = r.device_ptr(&exec.stream);
            descs.extend([
                pp + boff,
                rp + slot as u64 * state_elems as u64 * esz,
                state_elems as u64 * esz,
            ]);
            boff += (state_elems * 4) as u64;
            let w = bs.conv_win[li].as_ref().expect("linear layer has window");
            let (wp, _g2) = w.device_ptr(&exec.stream);
            descs.extend([
                pp + boff,
                wp + (slot * win_elems * 4) as u64,
                (win_elems * 4) as u64,
            ]);
            boff += (win_elems * 4) as u64;
        }
        // No allocation on this path: the descriptor buffer is owned by the
        // batch state and address-stable (see BatchState::d_ckpt_desc).
        let n = descs.len();
        if n == 0 {
            return Ok(());
        }
        let Some(db) = bs.d_ckpt_desc.as_mut() else {
            return Ok(());
        };
        debug_assert!(n <= db.len());
        {
            let mut v = db.slice_mut(0..n);
            exec.stream
                .memcpy_htod(&descs, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        exec.batched_copy(&*db, n / 3)?;
        Ok(())
    }

    /// The checkpoint alignment step: the tier's run span when armed (runs
    /// are the restore granularity), BLOCK_TOKENS otherwise.
    pub(super) fn tier_ckpt_step(&self) -> usize {
        self.batch
            .as_ref()
            .and_then(|b| b.tier.as_ref())
            .map(|t| t.run_blocks() * crate::kv_pool::BLOCK_TOKENS)
            .unwrap_or(crate::kv_pool::BLOCK_TOKENS)
    }

    /// The per-tick tier pump (see `Generator::tier_pump`).
    pub fn tier_pump_impl(&mut self) {
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let (Some(tier), Some(pr), Some(pool)) =
            (bs.tier.as_mut(), bs.paged_prefix.as_mut(), bs.pool.as_mut())
        else {
            return;
        };
        tier.pump_completions(pr, pool);
        tier.pump_flows(pr, &mut || exec.record_event().ok());
        // 2.3 write-through: retained chains AND live checkpoint blobs
        // pre-store in slack so eviction (and ckpt-slot recycling) is free
        let state = bs.d_state_pool.as_ref().map(|sp| {
            use cudarc::driver::DevicePtr;
            let (cp, _g) = sp.device_ptr(&exec.stream);
            (cp, (bs.state_ckpt_f32 * 4) as u64)
        });
        tier.mirror_slack(pr, pool, exec.record_event().ok(), 2, state);
    }

    pub fn tier_observe_prefill_impl(&mut self, tokens: u32, wall_us: f64) {
        if let Some(t) = self.batch.as_mut().and_then(|b| b.tier.as_mut()) {
            t.cost.observe_prefill(tokens, wall_us);
        }
    }

    pub fn tier_stats_impl(&self) -> Option<crate::kv_tier::TierStats> {
        self.batch.as_ref()?.tier.as_ref().map(|t| t.tier_stats())
    }

    pub fn tier_report_impl(&self) -> Option<crate::kv_tier::TierReport> {
        self.batch
            .as_ref()?
            .tier
            .as_ref()
            .map(crate::kv_tier::PoolTier::report)
    }

    /// P5b free-on-completion: return the KV blocks of every slot that no longer
    /// holds a live sequence (`!occupied[slot]`) to the shared pool, so the memory
    /// is available to new admissions the moment a sequence ends - not only when
    /// its slot is next reused. Idempotent; no-op unless the budget pool is active.
    /// A freed slot's device block-table entries go stale but are never read (an
    /// inactive slot is re-cleared and regrown at its next prefill).
    pub fn release_inactive_slots(&mut self, occupied: &[bool]) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let Some(pool) = bs.pool.as_mut() else { return };
        let mut freed = 0usize;
        for (slot, table) in bs.tables.iter_mut().enumerate() {
            let idle = occupied.get(slot).copied() != Some(true);
            if idle && !table.blocks().is_empty() {
                freed += table.blocks().len();
                table.clear(pool);
            }
        }
        // Only touch the env when a completion actually freed blocks (rare vs the
        // per-tick call rate), so the stats probe costs nothing on the hot path.
        if freed > 0 && paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some() {
            tracing::info!(
                "pool: freed {freed} blocks on completion  ({}/{} free)",
                pool.free_blocks(),
                pool.capacity()
            );
        }
    }

    /// Free blocks in the budget pool, or `None` when not in pool mode. Drives
    /// the scheduler's P5b watermark admission (stop admitting when the pool is
    /// nearly full; free-on-completion reopens it).
    pub fn pool_free_blocks(&self) -> Option<usize> {
        let bs = self.batch.as_ref()?;
        let pool = bs.pool.as_ref()?;
        // free + prefix-reclaimable: the radix is a CACHE (`ensure_slot_blocks`
        // evicts LRU pages on demand), so admission must not count retained
        // pages as spoken for. The raw count watermark-starves a salted
        // wide-batch run: every completion parks ~80 pages in the cache, free
        // sinks to the watermark, and admission serializes - average live
        // slots collapse to ~7 of 32 and TTFT tails run into minutes. Same
        // accounting gemma4 landed.
        let evictable = bs
            .paged_prefix
            .as_ref()
            .map_or(0, |pr| pr.evictable_blocks(pool));
        Some(pool.free_blocks() + evictable)
    }

    /// Number of DeltaNet (linear-attention) layers.
    pub(super) fn n_linear_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::Linear(_)))
            .count()
    }

    /// Tokens the last prefill of `slot` served from the prefix cache (taken:
    /// resets to 0). Usage-reporting hook for the engine.
    pub fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.last_reused.get_mut(slot).map_or(0, std::mem::take)
    }

    /// Lever 1: batched WHOLE-prompt prefill - fuse the admitted cohort into one
    /// weight-amortized forward per cap-sized group, instead of the serial
    /// per-prompt default that re-reads the 256-expert MoE once per prompt (the
    /// c32 TTFT stall). Every
    /// op is per-row/per-slot isolated (attention is per-slot causal; the DeltaNet
    /// conv+recurrence split per prompt into base-0 temps exactly as the serial
    /// path does), so a batched pass is numerically the same class as running the
    /// prompts one at a time - only the proj/MoE weight reads amortize.
    ///
    /// v1 scope: prompts are prefilled FRESH (no prefix-cache resume/insert inside
    /// this path - the salted concurrency workload it targets has no reuse). Opt-in
    /// via `PADDOCK_QWEN35_BATCH_PREFILL`; default falls back to the serial path so
    /// nothing changes until caching is wired (v2) and it's validated for default-on.
    /// Over-cap single prompts run serially (bounds scratch).
    /// Batched multimodal prefill (the vi8 second half): every
    /// pending image request's rows in one weight-amortized pass. The encode
    /// batching removed the tower serialization; this removes the per-slot
    /// prefill serialization (the residual TTFT plateau). Mirrors
    /// `forward_prefill_batch`, threading the mm extras through
    /// `prefill_batch_pass`. `PADDOCK_NO_MM_BATCH_PREFILL` pins the serial
    /// per-slot path for A/B.
    pub fn forward_prefill_batch_mm(
        &mut self,
        reqs: Vec<(
            usize,
            Vec<crate::service::MmChunk>,
            Vec<super::vision::VisionOutput>,
        )>,
    ) -> Result<Vec<(Vec<f32>, usize)>, GpuModelError> {
        if paddock_models::dev_var_os!("PADDOCK_NO_MM_BATCH_PREFILL").is_some() || reqs.len() < 2 {
            if paddock_models::dev_var_os!("PADDOCK_ROUTE_WITNESS").is_some() {
                eprintln!("pd route: mm prefill SERIAL n={}", reqs.len());
            }
            let mut out = Vec::with_capacity(reqs.len());
            for (slot, chunks, images) in reqs {
                out.push(self.forward_prefill_slot_mm_encoded(slot, &chunks, images)?);
            }
            return Ok(out);
        }
        if paddock_models::dev_var_os!("PADDOCK_ROUTE_WITNESS").is_some() {
            eprintln!("pd route: mm prefill BATCHED n={}", reqs.len());
        }
        let max_batch = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .max_batch;

        // Peel the RESUMABLE requests onto the serial path first.
        // The batched pass below is fresh-only - the same v1 scope its text
        // twin documents - so putting a request with a cached prefix through it
        // would re-prefill a picture the radix is already holding, which is
        // exactly the reuse the content-keyed cache exists to provide. A
        // document conversation is a cache hit by construction, so this is the
        // common case, not the corner. What is left is genuinely cold and
        // batches as before.
        let mut out: Vec<Option<(Vec<f32>, usize)>> = (0..reqs.len()).map(|_| None).collect();
        let mut cold: Vec<(
            usize,
            usize,
            Vec<crate::service::MmChunk>,
            Vec<super::vision::VisionOutput>,
        )> = Vec::with_capacity(reqs.len());
        for (i, (slot, chunks, images)) in reqs.into_iter().enumerate() {
            if slot >= max_batch {
                return Err(GpuModelError::BatchTooLarge {
                    got: slot + 1,
                    max: max_batch,
                });
            }
            if self.mm_prefix_would_resume(&chunks, &images)? {
                out[i] = Some(self.forward_prefill_slot_mm_encoded(slot, &chunks, images)?);
            } else {
                cold.push((i, slot, chunks, images));
            }
        }
        if cold.len() == 1 {
            // a cohort of one is the serial path, and this one also gets to
            // publish its pages for the next turn
            let (i, slot, chunks, images) = cold.pop().expect("len 1");
            out[i] = Some(self.forward_prefill_slot_mm_encoded(slot, &chunks, images)?);
        } else if !cold.is_empty() {
            let mut orig: Vec<usize> = Vec::with_capacity(cold.len());
            let mut items: Vec<(usize, Vec<u32>)> = Vec::with_capacity(cold.len());
            let mut ctxs: Vec<MmShareCtx> = Vec::with_capacity(cold.len());
            for (i, slot, chunks, images) in cold {
                let grids: Vec<(usize, usize)> = images.iter().map(|v| (v.nx, v.ny)).collect();
                let MmLayout {
                    ids,
                    mrope,
                    bound,
                    splices,
                    t_len,
                    final_mrope_pos,
                } = build_mm_layout(&chunks, &grids)?;
                if t_len == 0 || t_len > self.max_ctx {
                    return Err(GpuModelError::BatchTooLarge {
                        got: t_len,
                        max: self.max_ctx,
                    });
                }
                orig.push(i);
                items.push((slot, ids));
                ctxs.push(MmShareCtx {
                    mrope,
                    bound,
                    splices,
                    images,
                    final_mrope_pos,
                });
            }
            let cap = batch_prefill_cap();
            let mut lout: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
            let mut group: Vec<usize> = Vec::new();
            let mut rows = 0usize;
            for i in 0..items.len() {
                let tl = items[i].1.len();
                if rows + tl > cap && !group.is_empty() {
                    self.prefill_batch_pass(&items, &group, &mut lout, Some(&ctxs))?;
                    group.clear();
                    rows = 0;
                }
                // an oversized single request just runs as its own pass (same
                // kernels as the solo path; no serial special case needed)
                group.push(i);
                rows += tl;
            }
            if !group.is_empty() {
                self.prefill_batch_pass(&items, &group, &mut lout, Some(&ctxs))?;
            }
            for (j, l) in lout.into_iter().enumerate() {
                out[orig[j]] = Some((l, items[j].1.len()));
            }
        }
        Ok(out
            .into_iter()
            .map(|o| o.expect("every request prefilled"))
            .collect())
    }

    /// Would this multimodal prompt find a resumable checkpoint in the radix?
    ///
    /// Asked before admission so `forward_prefill_batch_mm` can route it, and
    /// deliberately answered with the same `match_full` the real resume uses
    /// rather than a cheaper approximation - a router that disagrees with the
    /// thing it routes to is worse than no router. `match_full` only reads and
    /// touches LRU, so asking twice costs a tree walk.
    fn mm_prefix_would_resume(
        &mut self,
        chunks: &[crate::service::MmChunk],
        images: &[super::vision::VisionOutput],
    ) -> Result<bool, GpuModelError> {
        if self.batch.as_ref().is_none_or(|b| b.paged_prefix.is_none()) {
            return Ok(false);
        }
        let grids: Vec<(usize, usize)> = images.iter().map(|v| (v.nx, v.ny)).collect();
        let lay = build_mm_layout(chunks, &grids)?;
        let keys = mm_radix_keys(&lay, &mm_image_hashes(chunks));
        let bs = self.batch.as_mut().expect("checked above");
        let m = bs
            .paged_prefix
            .as_mut()
            .expect("checked above")
            .match_full(&keys);
        Ok(m.ckpt
            .is_some_and(|(pos, _)| pos >= min_cache_prefix() && pos < keys.len()))
    }

    pub fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GpuModelError> {
        // Gate: without the env, behave exactly like the trait default (serial).
        if std::env::var_os("PADDOCK_QWEN35_BATCH_PREFILL").is_none() {
            return items
                .iter()
                .map(|(s, t)| self.forward_prefill_slot(*s, t))
                .collect();
        }
        // Batching only pays when several prompts fuse enough tokens to amortize
        // the 256-expert MoE weight read past its ~512-token saturation. A tiny
        // cohort (e.g. dc4: 4×256) adds the per-segment copy overhead for little
        // amortization and REGRESSES - keep those serial. Override the min total
        // rows with PADDOCK_QWEN35_BATCH_PREFILL_MIN (default 2048).
        let total_rows: usize = items.iter().map(|(_, t)| t.len()).sum();
        let min_rows: usize = paddock_models::dev_var!("PADDOCK_QWEN35_BATCH_PREFILL_MIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048);
        if items.len() < 2 || total_rows < min_rows {
            return items
                .iter()
                .map(|(s, t)| self.forward_prefill_slot(*s, t))
                .collect();
        }
        let max_batch = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .max_batch;
        for (slot, tokens) in items {
            if *slot >= max_batch {
                return Err(GpuModelError::BatchTooLarge {
                    got: slot + 1,
                    max: max_batch,
                });
            }
            if tokens.is_empty() || tokens.len() > self.max_ctx {
                return Err(GpuModelError::ContextExceeded {
                    got: tokens.len(),
                    max: self.max_ctx,
                });
            }
        }
        let cap = batch_prefill_cap();
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        let mut group: Vec<usize> = Vec::new();
        let mut rows = 0usize;
        for i in 0..items.len() {
            let tl = items[i].1.len();
            if tl > cap {
                // a single prompt wider than the cap: flush, then run it alone on
                // the serial path (its own weight read is already amortized over
                // its own long tail).
                if !group.is_empty() {
                    self.prefill_batch_pass(items, &group, &mut out, None)?;
                    group.clear();
                    rows = 0;
                }
                out[i] = self.forward_prefill_slot(items[i].0, &items[i].1)?;
            } else {
                if rows + tl > cap {
                    self.prefill_batch_pass(items, &group, &mut out, None)?;
                    group.clear();
                    rows = 0;
                }
                group.push(i);
                rows += tl;
            }
        }
        if !group.is_empty() {
            self.prefill_batch_pass(items, &group, &mut out, None)?;
        }
        Ok(out)
    }

    /// One batched prefill pass over `group` (indices into `items`): concatenate
    /// the prompts' fresh token spans, run the stack once (weight-amortized), and
    /// write each prompt's last-token logits into `out`. Mirrors the unified tick's
    /// layer body with no decode rows and every span whole (done==0, finishing).
    /// `mm` (parallel to `items`) threads the multimodal extras
    /// through the same body: per-request 4-axis mrope, image visibility bounds
    /// (bound-driven segment attention), embedding splices after embed, and the
    /// per-slot mrope delta - the batched twin of forward_prefill_slot_mm_encoded.
    /// persistent `prefill_batch_pass` buffers (see `PfPassBufs`).
    /// Grow-only with headroom; growth moves device addresses, so it drops
    /// every captured pass graph.
    /// Bytes `ensure_pf_bufs(max_take, r, n_sh)` allocates - the same
    /// arithmetic, so the plan's `prefill wave buffers` reserve is what the
    /// wave pass will take. Kept next to the allocation it prices;
    /// `ensure_pf_bufs` checks itself against this on every allocation.
    pub(super) fn pf_bufs_bytes(&self, max_take: usize, r: usize, n_sh: usize) -> u64 {
        let vl_need = |rc: usize, nc: usize| 2 * (rc / 64 + nc) + 4 * nc;
        let take_cap = (max_take + 63) & !63usize;
        let r_cap = r + 256;
        let n_cap = n_sh.max(32);
        let q_dim = self.n_heads * self.head_dim;
        // f32 planes: dq, dk, dv, dattn (value_dim); g, beta (n_v_heads); qn,
        // attn (q_dim)
        let f32s = take_cap * (4 * self.value_dim + 2 * self.n_v_heads + 2 * q_dim);
        // u32 planes: seg_slot, seg_bound, seg_pos (take); vl; items; win;
        // gidx; tokens, pos, slots (r); mrope (4 r)
        let u32s = 3 * take_cap
            + vl_need(r_cap, n_cap)
            + 8 * n_cap
            + 4 * n_cap
            + n_cap
            + 3 * r_cap
            + 4 * r_cap;
        4 * (f32s + u32s) as u64
    }

    fn ensure_pf_bufs(
        &mut self,
        max_take: usize,
        r: usize,
        n_sh: usize,
    ) -> Result<(), GpuModelError> {
        let vl_need = |rc: usize, nc: usize| 2 * (rc / 64 + nc) + 4 * nc;
        {
            let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
            if matches!(&bs.pf_bufs, Some(p) if p.take_cap >= max_take && p.r_cap >= r
                && p.vl_cap >= vl_need(r, n_sh)
                && p.items_cap >= 8 * n_sh
                && p.win_cap >= 4 * n_sh)
            {
                return Ok(());
            }
        }
        // headroom so steady serving settles after one growth
        let v_pf0 = cudarc::driver::result::mem_get_info()
            .map(|(f, _)| f)
            .unwrap_or(0);
        let take_cap = (max_take + 63) & !63usize;
        let r_cap = r + 256;
        let n_cap = n_sh.max(32);
        let (value_dim, n_v_heads) = (self.value_dim, self.n_v_heads);
        let q_dim = self.n_heads * self.head_dim;
        let e = self.exec.clone();
        let drv = |x: cudarc::driver::DriverError| crate::gpu::from_driver(x);
        // fresh-prompt positions are 0..take: prebuild the iota once per cap
        let d_seg_pos = {
            let h: Vec<u32> = (0..take_cap as u32).collect();
            let mut b = e.alloc_u32(take_cap)?;
            {
                let mut v = b.slice_mut(0..take_cap);
                e.stream.memcpy_htod(&h, &mut v).map_err(drv)?;
            }
            b
        };
        let bufs = PfPassBufs {
            d_pf_dq: e.alloc(take_cap * value_dim)?,
            d_pf_dk: e.alloc(take_cap * value_dim)?,
            d_pf_dv: e.alloc(take_cap * value_dim)?,
            d_pf_g: e.alloc(take_cap * n_v_heads)?,
            d_pf_beta: e.alloc(take_cap * n_v_heads)?,
            d_pf_dattn: e.alloc(take_cap * value_dim)?,
            d_pf_qn: e.alloc(take_cap * q_dim)?,
            d_pf_attn: e.alloc(take_cap * q_dim)?,
            d_seg_slot: e.alloc_u32(take_cap)?,
            d_seg_bound: e.alloc_u32(take_cap)?,
            d_seg_pos,
            d_vl: e.alloc_u32(vl_need(r_cap, n_cap))?,
            d_items: e.alloc_u32(8 * n_cap)?,
            d_win: e.alloc_u32(4 * n_cap)?,
            d_gidx: e.alloc_u32(n_cap)?,
            d_tokens: e.alloc_u32(r_cap)?,
            d_pos: e.alloc_u32(r_cap)?,
            d_slots: e.alloc_u32(r_cap)?,
            d_mrope: e.alloc_u32(4 * r_cap)?,
            take_cap,
            r_cap,
            vl_cap: vl_need(r_cap, n_cap),
            items_cap: 8 * n_cap,
            win_cap: 4 * n_cap,
        };
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        // The old buffers' last reader was the previous pass, which ended
        // host-synchronized (its logits readback); queued pipe ticks never
        // touch these, so dropping mid-pipe is safe.
        bs.pf_bufs = Some(bufs);
        bs.pf_pass_graphs.clear();
        // The reserve was computed from the same arithmetic; a buffer added
        // here and not there is exactly the drift the hedge used to hide.
        let priced = self.pf_bufs_bytes(max_take, r, n_sh);
        let measured = v_pf0.saturating_sub(
            cudarc::driver::result::mem_get_info()
                .map(|(f, _)| f)
                .unwrap_or(v_pf0),
        );
        if measured as u64 > priced + (16 << 20) {
            tracing::warn!(
                priced_mib = priced >> 20,
                measured_mib = measured >> 20,
                "qwen35 prefill wave buffers cost more than pf_bufs_bytes priced - \
                 a buffer was added to ensure_pf_bufs without its term"
            );
        }
        Ok(())
    }

    fn prefill_batch_pass(
        &mut self,
        items: &[(usize, Vec<u32>)],
        group: &[usize],
        out: &mut [Vec<f32>],
        mm: Option<&[MmShareCtx]>,
    ) -> Result<(), GpuModelError> {
        if group.is_empty() {
            return Ok(());
        }
        // shares: (out index, slot, take rows). Fresh whole prompts, KV pos 0..take.
        let shares: Vec<(usize, usize, usize)> = group
            .iter()
            .map(|&i| (i, items[i].0, items[i].1.len()))
            .collect();
        let r: usize = shares.iter().map(|s| s.2).sum();
        let max_take = shares.iter().map(|s| s.2).max().unwrap_or(0);
        assert!(r > 0 && max_take <= self.max_ctx);
        if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
            tracing::info!(
                "qwen35-wave-pass(NO prefix publish/resume): {} prompts, {r} rows",
                shares.len()
            );
        }
        // shape probe for the prefill-graph key census
        if paddock_models::dev_var_os!("PADDOCK_UNIFIED_SHAPE_LOG").is_some() {
            let meta: Vec<(usize, usize)> = shares.iter().map(|s| (s.1, s.2)).collect();
            eprintln!("[pshape] r={r} mm={} shares={meta:?}", mm.is_some());
        }

        // ---- captured pass graphs -------------------------------
        // An ELIGIBLE pass pads every share to one uniform `bucket` row count
        // and runs a captured CUDA graph keyed by (n_shares, bucket): grids
        // bake only the bucket shape, true span geometry rides device
        // CONTENTS (VL quads, conv-win quads, gather indices, token/pos/slot
        // planes) - so one graph serves any slot assignment and any
        // in-bucket length mix (census: exact shapes never repeat
        // at the c32 board; lens 135-145, waves 7-12). Pad rows are token-0
        // rows at positions take..bucket on the share's own slot: finite
        // garbage discarded everywhere except KV, where the pad append at
        // position p is overwritten by the decode append at p before p is
        // first attended (backed to bucket-1 below; radix inserts only full
        // true pages). Eligibility mirrors every launch-shape branch the
        // walk takes; anything else runs the exact eager pass unchanged.
        let graph_ok = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let on = *ON.get_or_init(|| {
                paddock_models::dev_var_os!("PADDOCK_NO_PF_GRAPH").is_none()
                    && paddock_models::dev_var_os!("PADDOCK_SPEC_TRACE").is_none()
                    && paddock_models::dev_var_os!("PADDOCK_W8_TRACE").is_none()
                    && paddock_models::dev_var_os!("PADDOCK_NO_DN_OFFSET").is_none()
                    && paddock_models::dev_var_os!("PADDOCK_NO_CONV_WIN_VL").is_none()
                    && paddock_models::dev_var_os!("PADDOCK_NO_BFL").is_none()
            });
            let cm: Vec<usize> = shares.iter().map(|s| s.2.div_ceil(64)).collect();
            // SMALL passes only (measured): the wave-batched
            // admission passes (7-12 shares, r ~1700) are GPU-WORK-bound -
            // wave batching already amortized the launch tax, so the bucket
            // padding cost (+10-16% rows) exceeded the replay's idle reclaim:
            // c32 2599.9/235 unrestricted-off vs 2553.3/256 unrestricted-on.
            // The issue-bound regime capture does help is the small pass
            // (single-prompt 21.2 ms at 53% busy / 1396 launches, ttft_cap):
            // cap the padded row count so only those capture.
            let rmax = {
                static RM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                *RM.get_or_init(|| {
                    paddock_models::dev_var!("PADDOCK_PF_GRAPH_RMAX")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(384)
                })
            };
            on && shares.len() * max_take.next_multiple_of(32) <= rmax
                && mm.is_none()
                && self.batch.as_ref().is_some_and(|b| b.paged)
                && self.head_dim == 256
                && self.max_ctx.is_multiple_of(64)
                && pf_attn_dtype_ok(self.kv_dtype, self.n_heads, self.n_kv_heads)
                && self.exec.has_attn_prefill_f16_paged()
                && self.exec.has_hrow_gather()
                && self.exec.has_conv_win_store_vl()
                && self.exec.has_gated_delta_chunked_rs_vl()
                && self.exec.has_conv_silu_qkv()
                && self.state_size == 128
                && dnc_vl_on()
                // every share rides the VL recurrence + the in-place paged
                // attention arm, and one 64-window keeps the baked per-span
                // chunk counts honest against the true-length contents
                && shares.len() <= 32
                && shares.iter().all(|s| s.2 >= 128)
                && cm.iter().all(|&c| c == cm[0])
                // epilogue class that splits device work from readback:
                // hrow_gather + (gemv at n==1 | the BFL Q8/f8 GEMM)
                && (shares.len() == 1 || matches!(&self.output, QuantW::Q8(_)))
        };
        let bucket = if graph_ok {
            max_take.next_multiple_of(32)
        } else {
            0
        };
        // padded per-share row count; identity on the eager path
        let wtake = move |take: usize| if graph_ok { bucket } else { take };
        let r = if graph_ok { shares.len() * bucket } else { r };
        let max_take = if graph_ok { bucket } else { max_take };

        // Flat host inputs: each prompt's tokens at positions 0..take on its slot.
        let mut tokens_h: Vec<u32> = Vec::with_capacity(r);
        let mut pos_h: Vec<u32> = Vec::with_capacity(r);
        let mut slots_h: Vec<u32> = Vec::with_capacity(r);
        for &(i, slot, take) in &shares {
            tokens_h.extend_from_slice(&items[i].1);
            pos_h.extend(0u32..take as u32);
            slots_h.extend(std::iter::repeat_n(slot as u32, take));
            // graph pad rows: token 0 at positions take..bucket on the same
            // slot (finite garbage; see the graph_ok note above)
            let w = wtake(take);
            if w > take {
                tokens_h.extend(std::iter::repeat_n(0u32, w - take));
                pos_h.extend(take as u32..w as u32);
                slots_h.extend(std::iter::repeat_n(slot as u32, w - take));
            }
        }
        // text mrope: all four axes = the row's position (delta 0 for fresh).
        // mm: each request's [4, take] axis-major block rearranged into the
        // pass-global axis-major layout the mrope kernel indexes ([4, r]).
        let mrope_h: Vec<u32> = if let Some(m) = mm {
            let mut v = vec![0u32; 4 * r];
            let mut rbase = 0usize;
            for &(oi, _, take) in &shares {
                let ctx = &m[oi];
                for ax in 0..4 {
                    v[ax * r + rbase..ax * r + rbase + take]
                        .copy_from_slice(&ctx.mrope[ax * take..(ax + 1) * take]);
                }
                rbase += take;
            }
            v
        } else {
            (0..4).flat_map(|_| pos_h.iter().copied()).collect()
        };

        // Whole-self setup before splitting into disjoint field borrows.
        // f8t chunk arm headroom: the DN/FFN arms below land [r, 2*ff] in
        // d_ffn_gate (a cap*ff f32 buffer), so a pass that will ride the tile
        // plane needs cap >= 2r before the layer loop. Bounded by
        // 2*f8t_chunk_rmax(); bigger (full-prefill) groups keep cap = r.
        let f8t_chunk = r <= f8t_chunk_rmax()
            && f8t_unified_on()
            && self.bs_f8t_attn.iter().any(Option::is_some);
        self.ensure_scratch(if f8t_chunk { 2 * r } else { r })?;
        for &(_, slot, _) in &shares {
            self.zero_slot_state(slot)?;
            self.last_reused[slot] = 0;
            if self.batch.as_ref().expect("batch enabled").pool.is_some() {
                let bs = self.batch.as_mut().expect("batch enabled");
                let pool = bs.pool.as_mut().expect("pool checked above");
                bs.tables[slot].clear(pool);
            }
        }
        if self.batch.as_ref().expect("batch enabled").pool.is_some() {
            for &(_, slot, take) in &shares {
                // graph pads append KV up to bucket-1 (overwritten by decode
                // before first attended) - back those positions too
                self.ensure_slot_blocks(slot, wtake(take) - 1)?;
            }
        }

        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (state_size, n_k_heads, n_v_heads, conv_k) =
            (self.state_size, self.n_k_heads, self.n_v_heads, self.conv_k);
        let (conv_dim, ff, max_ctx, value_dim) =
            (self.conv_dim, self.ff, self.max_ctx, self.value_dim);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let vocab = self.vocab;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let km1 = conv_k - 1;
        let state_elems = n_v_heads * state_size * state_size;
        let moe_dims = self.moe;
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let chunk_min = if exec.sm_count() >= 128 { 384 } else { 128 };
        let no_chunked_dn = paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_DN").is_some();
        // Offset-aware DeltaNet: run conv + recurrence directly at each span's row
        // offset in the packed batch buffers (state per slot), instead of copying
        // the span to base-0 temps and back. Bit-identical (same kernels, same
        // bytes) - every batched span here is a FRESH prompt (zeroed conv window),
        // so a plain offset conv == the zero-window ext build. Removes the ~10
        // copy_region launches per Linear layer per span (the wide-batch
        // base-0 copy storm the TTFT profile flagged). Default off for the
        // parity A/B.
        // Offset-aware DeltaNet/conv is default on: the unified tick's
        // profile showed the base-0 copy storm costing ~100-135us of host
        // serialization per layer, and the offset form is byte-identical.
        // PADDOCK_NO_DN_OFFSET pins the
        // classic copy path for rebuild-free A/B.
        let dn_offset = paddock_models::dev_var_os!("PADDOCK_NO_DN_OFFSET").is_none();

        // persistent pass buffers, taken as owned locals under their
        // original names - the walk below is unchanged; put back after the
        // epilogue. (Formerly ~14 per-pass cuMemAllocs; address stability is
        // what lets a captured pass graph replay.)
        self.ensure_pf_bufs(max_take, r, shares.len())?;
        let PfPassBufs {
            mut d_pf_dq,
            mut d_pf_dk,
            mut d_pf_dv,
            mut d_pf_g,
            mut d_pf_beta,
            mut d_pf_dattn,
            mut d_pf_qn,
            mut d_pf_attn,
            mut d_seg_slot,
            mut d_seg_bound,
            d_seg_pos,
            d_vl: mut d_vl_buf,
            d_items: mut d_items_buf,
            d_win: mut d_win_buf,
            mut d_gidx,
            mut d_tokens,
            mut d_pos,
            mut d_slots,
            mut d_mrope,
            take_cap,
            r_cap,
            vl_cap,
            items_cap,
            win_cap,
        } = self
            .batch
            .as_mut()
            .expect("batch enabled")
            .pf_bufs
            .take()
            .expect("pf bufs present");
        // Multi-span recurrence packing for the batched pass (profiled at
        // 12x128-row spans per wide mixed tick): the per-span v2_at loop
        // serialized spans x 48-layer launches of a ~100us serial state walk
        // - 57 ms of a 149 ms mixed tick, tied with the prefill GEMMs for the
        // top spot. Two rungs, both the unified tick's own machinery at a
        // second site:
        //
        //  1. VARLEN CHUNKED (spans >= 128 rows): one stage1+walk pair per
        //     Linear layer covers every such span. Fresh whole prompts are
        //     exactly the case where the chunked class is the reference
        //     class (see the recurrence comment below). The v2_packed walk
        //     alone was tried first and kept the serial class but paid its
        //     wave latency - 12 concurrent 128-step walks ran 866us/layer
        //     (41.6 ms/tick) vs the VL pair's ~41us in the unified tick.
        //     n_spans <= 32: the d_dnc_* scratch pads one chunk per span
        //     (its 32-span design bound); bigger cohorts fall back whole.
        //     Kill: PADDOCK_NO_DNC_VL (same chain as the unified site).
        //
        //  2. v2_packed (the remaining < 128-row spans): one launch per
        //     layer, bit-exact vs their per-span v2_at walks (absolute
        //     rows, slot-stride states, distinct slots here).
        //     Kill: PADDOCK_NO_DN_PACKED (same as the unified site).
        //
        // Descriptors are layer-invariant: built + uploaded once per pass.
        let mut vl_pf = vec![false; shares.len()];
        let d_pf_vl: Option<(usize, usize, usize)> = {
            let elig = shares.iter().filter(|s| s.2 >= 128).count();
            if dnc_vl_on()
                && dn_offset
                && exec.has_gated_delta_chunked_rs_vl()
                && state_size == 128
                && elig > 0
                && elig <= 32
            {
                let mut chunk_items: Vec<u32> = Vec::new();
                let mut quads: Vec<u32> = Vec::new();
                let mut rb = 0usize;
                for (si, &(_, slot, take)) in shares.iter().enumerate() {
                    if take >= 128 {
                        vl_pf[si] = true;
                        let first_chunk = (chunk_items.len() / 2) as u32;
                        let (mut row, mut left) = (rb, take);
                        while left > 0 {
                            let clen = left.min(64);
                            chunk_items.extend_from_slice(&[row as u32, clen as u32]);
                            row += clen;
                            left -= clen;
                        }
                        quads.extend_from_slice(&[
                            first_chunk,
                            take as u32,
                            (slot * state_elems) as u32,
                            rb as u32,
                        ]);
                    }
                    rb += wtake(take);
                }
                let n_chunks = chunk_items.len() / 2;
                let span_off = chunk_items.len();
                chunk_items.extend_from_slice(&quads);
                assert!(chunk_items.len() <= vl_cap, "vl items outgrew pf buf");
                {
                    let mut v = d_vl_buf.slice_mut(0..chunk_items.len());
                    exec.stream.memcpy_htod(&chunk_items, &mut v).map_err(drv)?;
                }
                Some((n_chunks, span_off, quads.len() / 4))
            } else {
                None
            }
        };
        let d_pf_items: Option<usize> = {
            let packed_on = {
                static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_DN_PACKED").is_none())
            };
            if packed_on && dn_offset && exec.has_dn_recurrent_packed() && state_size == 128 {
                let mut items: Vec<u32> = Vec::with_capacity(8 * shares.len());
                let mut rb = 0usize;
                for (si, &(_, slot, take)) in shares.iter().enumerate() {
                    let chunked = take >= chunk_min && !no_chunked_dn;
                    if !chunked && !vl_pf[si] && take > 0 {
                        items.extend_from_slice(&[
                            rb as u32,
                            take as u32,
                            slot as u32,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]);
                    }
                    rb += wtake(take);
                }
                if items.is_empty() {
                    None
                } else {
                    let n = items.len() / 8;
                    assert!(items.len() <= items_cap, "packed items outgrew pf buf");
                    {
                        let mut v = d_items_buf.slice_mut(0..items.len());
                        exec.stream.memcpy_htod(&items, &mut v).map_err(drv)?;
                    }
                    Some(n)
                }
            } else {
                None
            }
        };

        // conv-window VL store: one launch per Linear layer commits
        // every span's window - the per-share copy_region was 48 layers x n_sh
        // launches per admission pass (576 at a 12-prompt c32 wave). Span
        // geometry rides device CONTENTS, which is also what lets a captured
        // pass graph bake only a padded bucket shape later. Only the
        // dn_offset fresh arm stores this way; a sub-km1 share (partial
        // window) keeps the copy path for the whole pass - mixing the two
        // per layer isn't worth the bookkeeping. PADDOCK_NO_CONV_WIN_VL
        // reverts to the per-share copies.
        let d_win_spans: Option<usize> = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let on =
                *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_CONV_WIN_VL").is_none());
            if on && dn_offset && exec.has_conv_win_store_vl() && shares.iter().all(|s| s.2 >= km1)
            {
                let mut quads: Vec<u32> = Vec::with_capacity(4 * shares.len());
                let mut rb = 0usize;
                for &(_, slot, take) in &shares {
                    quads.extend_from_slice(&[rb as u32, take as u32, slot as u32, 0]);
                    rb += wtake(take);
                }
                assert!(quads.len() <= win_cap, "win quads outgrew pf buf");
                {
                    let mut v = d_win_buf.slice_mut(0..quads.len());
                    exec.stream.memcpy_htod(&quads, &mut v).map_err(drv)?;
                }
                Some(quads.len() / 4)
            } else {
                None
            }
        };
        // graph-epilogue gather indices: each share's true last row inside the
        // padded layout (contents only - the graph bakes just n_shares)
        if graph_ok {
            let gidx: Vec<u32> = shares
                .iter()
                .enumerate()
                .map(|(i, &(_, _, take))| (i * bucket + take - 1) as u32)
                .collect();
            let mut v = d_gidx.slice_mut(0..gidx.len());
            exec.stream.memcpy_htod(&gidx, &mut v).map_err(drv)?;
        }

        let out_f8 = &self.out_f8;
        let sinks = &self.sinks;
        let layers = &self.layers;
        let tok_embd = &self.tok_embd;
        let output = &self.output;
        let out_f8_h = self.out_f8.as_ref();
        let out_norm = &self.out_norm;
        let kv_dtype = self.kv_dtype;
        // b1: fp8 W8A8 dense-proj planes (empty unless PADDOCK_QWEN35_W8), same
        // gating as prefill_slot_chunk. The batched cohort r is the LARGEST
        // batch the projections ever see, so this is where the fp8 GEMM
        // (TMA warp-specialized, 1.56x mmq_pipe) pays most.
        let bs_w8_all = &self.bs_w8;
        let w8_min = w8_min_batch();
        let bs_f8ffn_p = &self.bs_f8ffn;
        let bs_f8row_p = &self.bs_f8row_ffn;
        // f8t chunk arm (the split path's admission cost): chunk passes ride
        // the same tile plane the decode/unified ticks use - the q8_0 ladder
        // (r <= 64) and f8bs (65..=rmax, tuned for M >= 512) both lose to it
        // at these widths. Same mixed-tick mechanism the unified arm fixed;
        // this is the second site. Per-GEMM parity gate below: the launcher's
        // 65+ route is tc5r, which serves only even 128-tile out_dims - odd
        // planes keep their old route for that GEMM alone.
        let bs_f8t_attn_p = &self.bs_f8t_attn;
        let bs_f8t_ffn_p = &self.bs_f8t_ffn;
        let sc = self.scratch.as_mut().expect("scratch");
        let bs = self.batch.as_mut().expect("batch");

        debug_assert!(r <= r_cap && max_take <= take_cap);
        {
            let mut v = d_tokens.slice_mut(0..r);
            exec.stream.memcpy_htod(&tokens_h, &mut v).map_err(drv)?;
        }
        {
            let mut v = d_pos.slice_mut(0..r);
            exec.stream.memcpy_htod(&pos_h, &mut v).map_err(drv)?;
        }
        {
            let mut v = d_slots.slice_mut(0..r);
            exec.stream.memcpy_htod(&slots_h, &mut v).map_err(drv)?;
        }
        {
            let mut v = d_mrope.slice_mut(0..4 * r);
            exec.stream.memcpy_htod(&mrope_h, &mut v).map_err(drv)?;
        }
        // P73: pf7 varlen packed prefill attention for the EAGER wave passes.
        // The c16 admission-wave census showed 16 serialized
        // per-span pf7 launches per full-attn layer at 68 CTAs each (under
        // half the die busy for each ~19.5us launch, ~312us/layer) where the
        // packed twin fills the machine in one grid. Items are the
        // unified_launch_core stride-4 quads (q_row0, span_rows,
        // tile_flat_row0, slot) - tiles never cross spans, so the packed
        // launch is BIT-IDENTICAL to the per-span _at calls it replaces.
        // Graph passes keep the per-span form (their items would need a
        // persistent buffer re-uploaded per replay); non-qualifying spans
        // keep the loop. Kill: PADDOCK_NO_PF7_VL (PADDOCK_NO_PF7 kills both
        // pf7 forms pack-side).
        let attn_g = n_heads.checked_div(n_kv_heads).unwrap_or(0);
        let mut pf_vl_share = vec![false; shares.len()];
        let d_pf_attn_items: Option<(cudarc::driver::CudaSlice<u32>, usize)> = if !graph_ok
            && mm.is_none()
            && {
                static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *ON.get_or_init(|| {
                    paddock_models::dev_var_os!("PADDOCK_NO_PF7_VL").is_none()
                        && paddock_models::dev_var_os!("PADDOCK_NO_PF7").is_none()
                })
            }
            && bs.paged
            && matches!(kv_dtype, KvDtype::Fp8E4m3)
            && head_dim == 256
            && matches!(attn_g, 4 | 6 | 8)
            && max_ctx % 64 == 0
            && pf_attn_dtype_ok(kv_dtype, n_heads, n_kv_heads)
            && exec.has_attn_prefill_f16_paged_vl()
        {
            let mut it: Vec<u32> = Vec::new();
            let mut rb = 0usize;
            for (si, &(_, slot, take)) in shares.iter().enumerate() {
                let w = wtake(take);
                if w > 24 {
                    pf_vl_share[si] = true;
                    let mut t0 = 0usize;
                    while t0 < w * attn_g {
                        it.extend_from_slice(&[rb as u32, w as u32, t0 as u32, slot as u32]);
                        t0 += 64;
                    }
                }
                rb += w;
            }
            if it.is_empty() {
                None
            } else {
                let n_tiles = it.len() / 4;
                let mut d = exec.alloc_u32(it.len())?;
                {
                    let mut s = d.slice_mut(0..it.len());
                    exec.stream.memcpy_htod(&it, &mut s).map_err(drv)?;
                }
                static W: std::sync::Once = std::sync::Once::new();
                W.call_once(|| eprintln!("[pfvl] prefill_batch_pass packed attention engaged"));
                Some((d, n_tiles))
            }
        } else {
            None
        };
        // P73: per-row span-start plane for the VL conv+silu+qkv (every
        // fresh span's conv in one launch per DN layer instead of the
        // per-span offset storm - 16 launches/layer on the c16 wave; each
        // row's math is bit-identical, the causal window just gates on the
        // row's own span start). Eager wave passes only, like the attn
        // pack. Kill: PADDOCK_NO_CONV_QKV_VL.
        let d_conv_row0s: Option<cudarc::driver::CudaSlice<u32>> = if !graph_ok
            && mm.is_none()
            && dn_offset
            && exec.has_conv_silu_qkv()
            && exec.has_conv_silu_qkv_vl()
            && {
                static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_CONV_QKV_VL").is_none())
            } {
            let mut v: Vec<u32> = Vec::with_capacity(r);
            let mut rb = 0usize;
            for &(_, _, take) in &shares {
                let w_ = wtake(take);
                v.extend(std::iter::repeat_n(rb as u32, w_));
                rb += w_;
            }
            let mut d = exec.alloc_u32(v.len())?;
            {
                let mut s = d.slice_mut(0..v.len());
                exec.stream.memcpy_htod(&v, &mut s).map_err(drv)?;
            }
            Some(d)
        } else {
            None
        };
        // ---- capture bracket -------------------------------------
        // Replay: the captured graph is the walk + device epilogue over these
        // exact persistent buffers - the uploads above already staged this
        // pass's contents. Record: bracket the unchanged eager walk in a
        // stream capture, instantiate, then launch (capture only records).
        // The closure exists so an error mid-record still ends the capture
        // before surfacing (the pf_graphs pattern).
        let pf_key = (shares.len(), bucket);
        let pf_replay = graph_ok && bs.pf_pass_graphs.contains_key(&pf_key);
        let pf_capture = graph_ok && !pf_replay && bs.pf_pass_graphs.len() < 16;
        if pf_capture {
            exec.stream
                .synchronize()
                .map_err(|e| GpuError::Driver(format!("pf pre-capture sync: {e}")))?;
            exec.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| GpuError::Driver(format!("pf begin_capture: {e}")))?;
        }
        let rec: Result<(), GpuModelError> = if pf_replay {
            Ok(())
        } else {
            (|| {
                embed_any(&exec, tok_embd, &d_tokens, &mut sc.d_x, embd, r)?;
                // mm: overwrite each request's placeholder rows with its image embeddings
                if let Some(m) = mm {
                    let mut rbase = 0usize;
                    for &(oi, _, take) in &shares {
                        let ctx = &m[oi];
                        for (k, &(off, nrows)) in ctx.splices.iter().enumerate() {
                            exec.copy_region(
                                &ctx.images[k].embd,
                                0,
                                &mut sc.d_x,
                                (rbase + off) * embd,
                                nrows * embd,
                            )?;
                        }
                        rbase += take;
                    }
                }

                // P73: FFN-residual hoist for the wave pass (the P71a decode pattern
                // brought to prefill): the standalone per-layer `exec.add` (64/wave,
                // 66us + gap each on the census) folds into the next layer's fused
                // prenorm via prefill_add_norm_quant's residual arm - its own doc:
                // "values are bit-identical either way". Gated to the mmq route's
                // own conditions so the small-batch else-branch (which asserts
                // !proj_b16) never sees a hoisted bf16 residual; the last layer
                // flushes before the final norm. Kill: PADDOCK_NO_PF_RES_HOIST.
                let pf_res_hoist = {
                    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    *ON.get_or_init(|| {
                        paddock_models::dev_var_os!("PADDOCK_NO_PF_RES_HOIST").is_none()
                            && paddock_models::dev_var_os!("PADDOCK_DBG_NORM").is_none()
                    })
                } && r > 64
                    && embd % 4 == 0
                    && embd <= 24576;
                let mut pend_ffn: Option<bool> = None;

                for (li, layer) in layers.iter().enumerate() {
                    // W8 planes for this layer (always engaged here when loaded - the
                    // batch gate requires total_rows >= 2048 > w8_min).
                    let lw8 = bs_w8_all.get(li).filter(|_| r > w8_min);
                    if li == 0 && paddock_models::dev_var_os!("PADDOCK_W8_TRACE").is_some() {
                        tracing::info!(
                            rows = r,
                            w8_min,
                            planes = bs_w8_all.len(),
                            lw8 = lw8.is_some(),
                            site = "prefill_batch_pass",
                            "qwen35 prefill W8 consult"
                        );
                    }
                    // 2*r <= cap re-checks the invariant ensure_scratch established
                    // for the [r, 2*ff] landings (an older, smaller scratch could
                    // otherwise overflow d_ffn_gate).
                    let f8t_c = f8t_chunk
                        && 2 * r <= sc.cap
                        && bs_f8t_attn_p.get(li).and_then(|o| o.as_ref()).is_some();
                    let tc5r_ok = |out: usize| r <= 64 || ((out >> 7) & 1) == 0;
                    let keep_xn =
                        matches!(&layer.mixer, Mixer::Linear(_)) || lw8.is_some() || f8t_c;
                    let pf_pend_b16 = pend_ffn.take();
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        pf_pend_b16.map(|_| &sc.d_proj),
                        pf_pend_b16.unwrap_or(false),
                        &layer.attn_norm.buf,
                        &mut sc.d_xn,
                        keep_xn,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    let mut mixer_b16 = false;
                    match &layer.mixer {
                        Mixer::Full(w) => {
                            // Phase A fused-plane consumer: one GEMM over
                            // the full wq|wk|wv plane + pd_q36_qkg_nra_rows replaces
                            // 3 GEMMs + split_qg + 2x rmsnorm + 2x mrope + 2x paged
                            // append (bench/q36_qnf_bench.cu: -69..-78% at mixed-tick
                            // shapes, planes + pools bit-identical). PADDOCK_NO_QNF
                            // kills to the chain.
                            let qnf_caps = bs.paged
                                && head_dim == 256
                                && n_rot == 64
                                && exec.has_q36_qkg_nra_rows()
                                && {
                                    static QNF: std::sync::OnceLock<bool> =
                                        std::sync::OnceLock::new();
                                    *QNF.get_or_init(|| {
                                        paddock_models::dev_var_os!("PADDOCK_NO_QNF").is_none()
                                    })
                                };
                            // f8t chunk arm rides only with the fused-landing consumer
                            // (no split is wired for the fused layout here) and with
                            // both planes on tc5r-servable tile counts.
                            let f8t_qkv =
                                bs_f8t_attn_p.get(li).and_then(|o| o.as_ref()).filter(|p| {
                                    f8t_c
                                        && qnf_caps
                                        && tc5r_ok(p[0].scale.len())
                                        && tc5r_ok(p[1].scale.len())
                                });
                            let mut f8t_wo: Option<&crate::gpu::F8TilePlane> = None;
                            let qnf = (lw8.is_some() || f8t_qkv.is_some()) && qnf_caps;
                            if let Some([qkv_t, wo_t]) = f8t_qkv {
                                exec.quantize_e4m3_row(
                                    &sc.d_xn,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    embd,
                                    r,
                                )?;
                                let nqkv = w.wq.dims()[1] + w.wk.dims()[1] + w.wv.dims()[1];
                                exec.f8t_gemm(
                                    qkv_t,
                                    &sc.d_f8t_q,
                                    &sc.d_f8t_rs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_qg,
                                    embd,
                                    nqkv,
                                    r,
                                )?;
                                let bt = bs.d_block_tables.as_ref().expect("paged block tables");
                                let bps = bs.blocks_per_slot;
                                exec.q36_qkg_nra_rows(
                                    &sc.d_qg,
                                    0,
                                    nqkv,
                                    w.wq.dims()[1],
                                    w.wq.dims()[1] + w.wk.dims()[1],
                                    &w.q_norm.buf,
                                    &w.k_norm.buf,
                                    &mut sc.d_qn,
                                    &mut sc.d_gate,
                                    bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                    bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                    &d_pos,
                                    Some(&d_slots),
                                    &d_mrope,
                                    bt,
                                    bps,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    n_rot,
                                    eps,
                                    yarn,
                                    sections,
                                    r,
                                    kv_dtype,
                                )?;
                                f8t_wo = Some(wo_t);
                            } else if let Some(l8) = lw8 {
                                // One e4m3 quant of the normed hidden feeds wq/wk/wv.
                                exec.quantize_e4m3(
                                    &sc.d_xn,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * embd,
                                )?;
                                if qnf {
                                    let nqkv = w.wq.dims()[1] + w.wk.dims()[1] + w.wv.dims()[1];
                                    exec.f8_gemm_w8(
                                        l8.wq.as_ref().expect("full-attn W8 qkv plane"),
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_qg,
                                        w.wq.dims()[0],
                                        nqkv,
                                        r,
                                    )?;
                                    let bt =
                                        bs.d_block_tables.as_ref().expect("paged block tables");
                                    let bps = bs.blocks_per_slot;
                                    exec.q36_qkg_nra_rows(
                                        &sc.d_qg,
                                        0,
                                        nqkv,
                                        w.wq.dims()[1],
                                        w.wq.dims()[1] + w.wk.dims()[1],
                                        &w.q_norm.buf,
                                        &w.k_norm.buf,
                                        &mut sc.d_qn,
                                        &mut sc.d_gate,
                                        bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                        bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                        &d_pos,
                                        Some(&d_slots),
                                        &d_mrope,
                                        bt,
                                        bps,
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        n_rot,
                                        eps,
                                        yarn,
                                        sections,
                                        r,
                                        kv_dtype,
                                    )?;
                                } else {
                                    exec.f8_gemm_w8(
                                        l8.wq.as_ref().expect("full-attn W8 qkv plane"),
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_qg,
                                        w.wq.dims()[0],
                                        w.wq.dims()[1],
                                        r,
                                    )?;
                                    exec.split_qg(
                                        &sc.d_qg,
                                        &mut sc.d_q,
                                        &mut sc.d_gate,
                                        r,
                                        n_heads,
                                        head_dim,
                                    )?;
                                    exec.f8_gemm_w8(
                                        l8.wq.as_ref().expect("full-attn W8 qkv plane"),
                                        w.wq.dims()[1],
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_k,
                                        w.wk.dims()[0],
                                        w.wk.dims()[1],
                                        r,
                                    )?;
                                    exec.f8_gemm_w8(
                                        l8.wq.as_ref().expect("full-attn W8 qkv plane"),
                                        w.wq.dims()[1] + w.wk.dims()[1],
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_v,
                                        w.wv.dims()[0],
                                        w.wv.dims()[1],
                                        r,
                                    )?;
                                }
                            } else {
                                prefill_mm_pre_any(
                                    &exec,
                                    &w.wq,
                                    &sc.d_pxq,
                                    &sc.d_pxs,
                                    &sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &mut sc.d_qg,
                                    r,
                                )?;
                                exec.split_qg(
                                    &sc.d_qg,
                                    &mut sc.d_q,
                                    &mut sc.d_gate,
                                    r,
                                    n_heads,
                                    head_dim,
                                )?;
                                prefill_mm_pre_any(
                                    &exec,
                                    &w.wk,
                                    &sc.d_pxq,
                                    &sc.d_pxs,
                                    &sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &mut sc.d_k,
                                    r,
                                )?;
                                prefill_mm_pre_any(
                                    &exec,
                                    &w.wv,
                                    &sc.d_pxq,
                                    &sc.d_pxs,
                                    &sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &mut sc.d_v,
                                    r,
                                )?;
                            }
                            if !qnf {
                                exec.rmsnorm_batch(
                                    &sc.d_q,
                                    &w.q_norm.buf,
                                    &mut sc.d_qn,
                                    head_dim,
                                    eps,
                                    r * n_heads,
                                )?;
                                exec.rmsnorm_batch(
                                    &sc.d_k,
                                    &w.k_norm.buf,
                                    &mut sc.d_kn,
                                    head_dim,
                                    eps,
                                    r * n_kv_heads,
                                )?;
                                exec.mrope(
                                    &mut sc.d_qn,
                                    &d_mrope,
                                    r,
                                    n_heads,
                                    head_dim,
                                    n_rot,
                                    yarn,
                                    sections,
                                )?;
                                exec.mrope(
                                    &mut sc.d_kn,
                                    &d_mrope,
                                    r,
                                    n_kv_heads,
                                    head_dim,
                                    n_rot,
                                    yarn,
                                    sections,
                                )?;
                                if bs.paged {
                                    let bt =
                                        bs.d_block_tables.as_ref().expect("paged block tables");
                                    let bps = bs.blocks_per_slot;
                                    exec.kv_append_batch_paged(
                                        &sc.d_kn,
                                        bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                        &d_pos,
                                        Some(&d_slots),
                                        bt,
                                        bps,
                                        kv_dim,
                                        r,
                                        kv_dtype,
                                    )?;
                                    exec.kv_append_batch_paged(
                                        &sc.d_v,
                                        bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                        &d_pos,
                                        Some(&d_slots),
                                        bt,
                                        bps,
                                        kv_dim,
                                        r,
                                        kv_dtype,
                                    )?;
                                } else {
                                    exec.kv_append_batch(
                                        &sc.d_kn,
                                        bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                        &d_pos,
                                        Some(&d_slots),
                                        kv_dim,
                                        max_ctx,
                                        r,
                                        kv_dtype,
                                    )?;
                                    exec.kv_append_batch(
                                        &sc.d_v,
                                        bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                        &d_pos,
                                        Some(&d_slots),
                                        kv_dim,
                                        max_ctx,
                                        r,
                                        kv_dtype,
                                    )?;
                                }
                            }
                            // per-prompt segment attention: each span is one slot's
                            // contiguous rows -> exactly the serial prefill_slot_chunk
                            // attention (fast WMMA / paged-WMMA flash, SWA tile-skip),
                            // bit-identical. attn kernels have no q-row offset, so copy
                            // the span to base-0 (O(rows), negligible vs the O(rows^2) attn).
                            // P73: packed pf7 form - every qualifying span in one
                            // launch (bit-identical CTAs to the per-span _at calls
                            // below); covered spans skip in the loop.
                            let mut pf_vl_fired = false;
                            if let Some((it, n_tiles)) = d_pf_attn_items.as_ref() {
                                exec.attn_prefill_f16_paged_vl(
                                    &sc.d_qn,
                                    bs.kv_k[li].as_ref().expect("full-attn layer KV"),
                                    bs.kv_v[li].as_ref().expect("full-attn layer KV"),
                                    sinks,
                                    &mut sc.d_attn,
                                    &d_pos,
                                    it,
                                    *n_tiles,
                                    bs.d_block_tables.as_ref().expect("paged block tables"),
                                    bs.blocks_per_slot,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    kv_dim,
                                    0,
                                    scale,
                                    kv_dtype,
                                )?;
                                pf_vl_fired = true;
                            }
                            let mut rb = 0;
                            for (si, &(oi, slot, take)) in shares.iter().enumerate() {
                                // graph pads: attention runs at the bucket width (pad
                                // rows' outputs are discarded; their KV is backed)
                                let take = wtake(take);
                                if pf_vl_fired && pf_vl_share[si] {
                                    rb += take;
                                    continue;
                                }
                                if let Some(m) = mm {
                                    // mm segment: bound-driven attention (image rows see
                                    // their whole equal-t block) - the exact solo-mm
                                    // prefill_attn call at base-0, bit-identical to
                                    // forward_prefill_slot_mm_encoded's. The fast
                                    // in-place paged arm assumes bound == row position,
                                    // so mm keeps the copy path (segments are short).
                                    let ctx = &m[oi];
                                    exec.copy_region(
                                        &sc.d_qn,
                                        rb * q_dim,
                                        &mut d_pf_qn,
                                        0,
                                        take * q_dim,
                                    )?;
                                    let sl = vec![slot as u32; take];
                                    {
                                        let mut v = d_seg_slot.slice_mut(0..take);
                                        exec.stream.memcpy_htod(&sl, &mut v).map_err(drv)?;
                                    }
                                    {
                                        let mut v = d_seg_bound.slice_mut(0..take);
                                        exec.stream.memcpy_htod(&ctx.bound, &mut v).map_err(drv)?;
                                    }
                                    prefill_attn(
                                        &exec,
                                        &d_pf_qn,
                                        bs.kv_k[li].as_ref().expect("full-attn layer KV"),
                                        bs.kv_v[li].as_ref().expect("full-attn layer KV"),
                                        sinks,
                                        &mut d_pf_attn,
                                        &d_seg_bound,
                                        &d_seg_slot,
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        max_ctx,
                                        kv_dim,
                                        take,
                                        scale,
                                        kv_dtype,
                                        bs.d_block_tables
                                            .as_ref()
                                            .filter(|_| bs.paged)
                                            .map(|bt| (bt, bs.blocks_per_slot)),
                                        Some((&mut sc.d_attn_o, &mut sc.d_attn_ml)),
                                    )?;
                                    exec.copy_region(
                                        &d_pf_attn,
                                        0,
                                        &mut sc.d_attn,
                                        rb * q_dim,
                                        take * q_dim,
                                    )?;
                                    rb += take;
                                    continue;
                                }
                                if bs.paged
                                    && take > 24
                                    && head_dim == 256
                                    && max_ctx % 64 == 0
                                    && pf_attn_dtype_ok(kv_dtype, n_heads, n_kv_heads)
                                    && exec.has_attn_prefill_f16_paged()
                                {
                                    // serving class runs in PLACE at row rb: fresh-span
                                    // rows of d_pos are 0..take and d_slots the span's
                                    // slot, so the per-span pageable slot upload (a
                                    // hidden full-stream sync per span per layer) and
                                    // both rows×q_dim staging copies drop out
                                    // (bit-identical - same kernel, offset pointers).
                                    exec.attn_prefill_f16_paged_at(
                                        &sc.d_qn,
                                        bs.kv_k[li].as_ref().expect("full-attn layer KV"),
                                        bs.kv_v[li].as_ref().expect("full-attn layer KV"),
                                        sinks,
                                        &mut sc.d_attn,
                                        &d_pos,
                                        &d_slots,
                                        rb,
                                        bs.d_block_tables.as_ref().expect("paged block tables"),
                                        bs.blocks_per_slot,
                                        n_heads,
                                        n_kv_heads,
                                        head_dim,
                                        kv_dim,
                                        0,
                                        take,
                                        scale,
                                        kv_dtype,
                                    )?;
                                    rb += take;
                                    continue;
                                }
                                exec.copy_region(
                                    &sc.d_qn,
                                    rb * q_dim,
                                    &mut d_pf_qn,
                                    0,
                                    take * q_dim,
                                )?;
                                let sl = vec![slot as u32; take];
                                {
                                    let mut v = d_seg_slot.slice_mut(0..take);
                                    exec.stream.memcpy_htod(&sl, &mut v).map_err(drv)?;
                                }
                                prefill_attn(
                                    &exec,
                                    &d_pf_qn,
                                    bs.kv_k[li].as_ref().expect("full-attn layer KV"),
                                    bs.kv_v[li].as_ref().expect("full-attn layer KV"),
                                    sinks,
                                    &mut d_pf_attn,
                                    &d_seg_pos,
                                    &d_seg_slot,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_dim,
                                    take,
                                    scale,
                                    kv_dtype,
                                    bs.d_block_tables
                                        .as_ref()
                                        .filter(|_| bs.paged)
                                        .map(|bt| (bt, bs.blocks_per_slot)),
                                    Some((&mut sc.d_attn_o, &mut sc.d_attn_ml)),
                                )?;
                                exec.copy_region(
                                    &d_pf_attn,
                                    0,
                                    &mut sc.d_attn,
                                    rb * q_dim,
                                    take * q_dim,
                                )?;
                                rb += take;
                            }
                            exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, r * q_dim)?;
                            if let Some(wo_t) = f8t_wo {
                                // chunk-arm wo: same row-quant seam as the unified
                                // tick's; d_proj stays f32 (mixer_b16 false)
                                exec.quantize_e4m3_row(
                                    &sc.d_attn,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    w.wo.dims()[0],
                                    r,
                                )?;
                                exec.f8t_gemm(
                                    wo_t,
                                    &sc.d_f8t_q,
                                    &sc.d_f8t_rs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_proj,
                                    w.wo.dims()[0],
                                    w.wo.dims()[1],
                                    r,
                                )?;
                            } else if let Some(l8) = lw8 {
                                exec.quantize_e4m3(
                                    &sc.d_attn,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * w.wo.dims()[0],
                                )?;
                                // mixer bf16 seam: wo writes bf16, the post_norm
                                // entry (ABI-247 consumer) reads it back
                                static MO16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                                let mo16 = *MO16.get_or_init(|| {
                                    paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                        && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                                });
                                // P74: PADDOCK_QWEN35_O16_TC5 opts the sm_100 tc5
                                // bf16-store route in (the TMA lane is process-killed
                                // there); r >= 256 = the tc5v NT2 floor. MEASURED A
                                // WASH on B200 (tc5s pays ~10% store-poison for bf16
                                // stores - the muse f16 ledger class - and the b16
                                // consumers hold no net win); kept as probe infra.
                                static MO16T: std::sync::OnceLock<bool> =
                                    std::sync::OnceLock::new();
                                let mo16t = *MO16T.get_or_init(|| {
                                    paddock_models::dev_var_os!("PADDOCK_QWEN35_O16_TC5").is_some()
                                        && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                                });
                                if (mo16 || (mo16t && r >= 256)) && exec.has_f8_o16() {
                                    exec.f8_gemm_w8_o16(
                                        l8.wo.as_ref().expect("full-attn W8 wo plane"),
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_proj,
                                        w.wo.dims()[0],
                                        w.wo.dims()[1],
                                        r,
                                    )?;
                                    mixer_b16 = true;
                                } else {
                                    exec.f8_gemm_w8(
                                        l8.wo.as_ref().expect("full-attn W8 wo plane"),
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_proj,
                                        w.wo.dims()[0],
                                        w.wo.dims()[1],
                                        r,
                                    )?;
                                }
                            } else {
                                prefill_mm_any(
                                    &exec,
                                    &w.wo,
                                    &mut sc.d_pxq,
                                    &mut sc.d_pxs,
                                    &mut sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &sc.d_attn,
                                    &mut sc.d_proj,
                                    r,
                                )?;
                            }
                        }
                        Mixer::Linear(w) => {
                            // the fused in_qkv|gate plane runs as one
                            // two-buffer GEMM when the split route covers it (the
                            // split gate launch alone is a ~1.0x fractional wave on
                            // this die - q36_dnf_bench: pair -> merged -20/-13/-7%
                            // at r=512/1024/2048). d_z fills here and stays
                            // untouched until gated_rmsnorm; the second launch
                            // below is skipped. PADDOCK_NO_DNF reverts.
                            let mut dn_fused = false;
                            let mut dn_ab_done = false;
                            let mut f8t_ow: Option<&crate::gpu::F8TilePlane> = None;
                            let f8t_dn =
                                bs_f8t_attn_p.get(li).and_then(|o| o.as_ref()).filter(|p| {
                                    f8t_c && tc5r_ok(p[0].scale.len()) && tc5r_ok(p[1].scale.len())
                                });
                            if let Some([in_t, ow_t]) = f8t_dn {
                                let (nin, nc) = (w.in_qkv.dims()[0], w.in_qkv.dims()[1]);
                                let nz_ = w.gate_w.dims()[1];
                                // the plane's scale length is its out_dim; +128 marks
                                // the alpha||beta fold (see the unified arm)
                                let tot = in_t.scale.len();
                                dn_ab_done = tot == nc + nz_ + 128;
                                exec.quantize_e4m3_row(
                                    &sc.d_xn,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    nin,
                                    r,
                                )?;
                                // landing: d_ffn_gate (cap >= 2r rows of ff - the
                                // ensure_scratch headroom above; free until the FFN)
                                exec.f8t_gemm(
                                    in_t,
                                    &sc.d_f8t_q,
                                    &sc.d_f8t_rs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_ffn_gate,
                                    nin,
                                    tot,
                                    r,
                                )?;
                                if dn_ab_done && exec.has_row_slice4() {
                                    exec.row_slice4(
                                        &sc.d_ffn_gate,
                                        tot,
                                        r,
                                        &mut [
                                            (&mut sc.d_mixed, 0, nc),
                                            (&mut sc.d_z, nc, nz_),
                                            (&mut sc.d_a, nc + nz_, n_v_heads),
                                            (&mut sc.d_b, nc + nz_ + n_v_heads, n_v_heads),
                                        ],
                                    )?;
                                } else {
                                    exec.row_slice(&sc.d_ffn_gate, &mut sc.d_mixed, tot, 0, nc, r)?;
                                    exec.row_slice(&sc.d_ffn_gate, &mut sc.d_z, tot, nc, nz_, r)?;
                                    if dn_ab_done {
                                        exec.row_slice(
                                            &sc.d_ffn_gate,
                                            &mut sc.d_a,
                                            tot,
                                            nc + nz_,
                                            n_v_heads,
                                            r,
                                        )?;
                                        exec.row_slice(
                                            &sc.d_ffn_gate,
                                            &mut sc.d_b,
                                            tot,
                                            nc + nz_ + n_v_heads,
                                            n_v_heads,
                                            r,
                                        )?;
                                    }
                                }
                                dn_fused = true;
                                f8t_ow = Some(ow_t);
                            } else if let Some(l8) = lw8 {
                                // e4m3-quant the normed hidden once; it feeds in_qkv AND
                                // (on the two-launch path, unchanged since) gate_w below.
                                exec.quantize_e4m3(
                                    &sc.d_xn,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * embd,
                                )?;
                                static DNF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                                if *DNF.get_or_init(|| {
                                    paddock_models::dev_var_os!("PADDOCK_NO_DNF").is_none()
                                }) {
                                    dn_fused = exec.f8_gemm_w8_split(
                                        l8.in_qkv.as_ref().expect("DeltaNet W8 in_qkv plane"),
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_mixed,
                                        &mut sc.d_z,
                                        w.in_qkv.dims()[0],
                                        w.in_qkv.dims()[1],
                                        w.gate_w.dims()[1],
                                        r,
                                    )?;
                                }
                                if !dn_fused {
                                    exec.f8_gemm_w8(
                                        l8.in_qkv.as_ref().expect("DeltaNet W8 in_qkv plane"),
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_mixed,
                                        w.in_qkv.dims()[0],
                                        w.in_qkv.dims()[1],
                                        r,
                                    )?;
                                }
                            } else {
                                prefill_mm_pre_any(
                                    &exec,
                                    &w.in_qkv,
                                    &sc.d_pxq,
                                    &sc.d_pxs,
                                    &sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &mut sc.d_mixed,
                                    r,
                                )?;
                            }
                            // per-prompt window-extended causal conv (each window zeroed
                            // by zero_slot_state above -> implicit zero-pad, matches serial).
                            // P73: with the row0s plane, every fresh span's conv+split
                            // runs in one packed launch (bit-identical rows); the loop
                            // below then only commits windows.
                            // QKC (slots 446/447): only on an ALL-VL tick - every
                            // other d_dq/d_dk consumer reads f32 expanded.
                            let qkc_tick = dn_qkc(&exec)
                                && d_pf_vl.is_some()
                                && d_pf_items.is_none()
                                && vl_pf.iter().all(|&x| x);
                            if let Some(r0) = d_conv_row0s.as_ref() {
                                if qkc_tick {
                                    exec.causal_conv1d_silu_qkv_vl_qkc(
                                        &sc.d_mixed,
                                        &w.conv_w.buf,
                                        r0,
                                        &mut sc.d_dq,
                                        &mut sc.d_dk,
                                        &mut sc.d_dv,
                                        r,
                                        n_k_heads,
                                        n_v_heads,
                                        state_size,
                                        conv_k,
                                    )?;
                                } else {
                                    exec.causal_conv1d_silu_qkv_vl(
                                        &sc.d_mixed,
                                        &w.conv_w.buf,
                                        r0,
                                        &mut sc.d_dq,
                                        &mut sc.d_dk,
                                        &mut sc.d_dv,
                                        r,
                                        n_k_heads,
                                        n_v_heads,
                                        state_size,
                                        conv_k,
                                    )?;
                                }
                            }
                            let mut rb = 0;
                            for &(_, slot, take) in &shares {
                                // graph pads: causal conv at bucket width is pad-safe
                                // (window commit reads true geometry via d_win quads)
                                let take = wtake(take);
                                let woff = slot * km1 * conv_dim;
                                if dn_offset {
                                    // fresh-prompt window is zero -> conv directly on the span
                                    // (zero left-pad) at its row offset, write back at the
                                    // same offset. Bit-identical to the ext build; no big copies.
                                    if d_conv_row0s.is_some() {
                                        // conv+split already ran packed above
                                    } else if exec.has_conv_silu_qkv() {
                                        // fused conv+split+norm: q/k/v written directly,
                                        // d_conv never materializes (bit-exact composition;
                                        // the split call below is skipped on this path)
                                        exec.causal_conv1d_silu_qkv_at(
                                            &sc.d_mixed,
                                            &w.conv_w.buf,
                                            &mut sc.d_dq,
                                            &mut sc.d_dk,
                                            &mut sc.d_dv,
                                            rb,
                                            rb,
                                            take,
                                            n_k_heads,
                                            n_v_heads,
                                            state_size,
                                            conv_k,
                                        )?;
                                    } else {
                                        exec.causal_conv1d_silu_at(
                                            &sc.d_mixed,
                                            &w.conv_w.buf,
                                            &mut sc.d_conv,
                                            rb,
                                            rb,
                                            take,
                                            conv_dim,
                                            conv_k,
                                        )?;
                                    }
                                    // commit the trailing k-1 rows into the slot's persistent
                                    // window for the decode phase (window pre-zeroed, so a
                                    // short span lands in the tail rows == zero-padded front).
                                    // With d_win_spans the whole pass commits below in one
                                    // conv_win_store_vl launch instead (bit-identical bytes).
                                    if d_win_spans.is_none() {
                                        let win = bs.conv_win[li]
                                            .as_mut()
                                            .expect("DeltaNet layer window");
                                        if take >= km1 {
                                            exec.copy_region(
                                                &sc.d_mixed,
                                                (rb + take - km1) * conv_dim,
                                                win,
                                                woff,
                                                km1 * conv_dim,
                                            )?;
                                        } else {
                                            exec.copy_region(
                                                &sc.d_mixed,
                                                rb * conv_dim,
                                                win,
                                                woff + (km1 - take) * conv_dim,
                                                take * conv_dim,
                                            )?;
                                        }
                                    }
                                } else {
                                    {
                                        let win = bs.conv_win[li]
                                            .as_ref()
                                            .expect("DeltaNet layer window");
                                        assert!(
                                            (km1 + take) * conv_dim <= bs.d_conv_ext.len(),
                                            "resumed span {take} rows outgrew the conv ext staging"
                                        );
                                        exec.copy_region(
                                            win,
                                            woff,
                                            &mut bs.d_conv_ext,
                                            0,
                                            km1 * conv_dim,
                                        )?;
                                    }
                                    exec.copy_region(
                                        &sc.d_mixed,
                                        rb * conv_dim,
                                        &mut bs.d_conv_ext,
                                        km1 * conv_dim,
                                        take * conv_dim,
                                    )?;
                                    exec.causal_conv1d_silu(
                                        &bs.d_conv_ext,
                                        &w.conv_w.buf,
                                        &mut bs.d_conv_out,
                                        km1 + take,
                                        conv_dim,
                                        conv_k,
                                    )?;
                                    exec.copy_region(
                                        &bs.d_conv_out,
                                        km1 * conv_dim,
                                        &mut sc.d_conv,
                                        rb * conv_dim,
                                        take * conv_dim,
                                    )?;
                                    {
                                        let win = bs.conv_win[li]
                                            .as_mut()
                                            .expect("DeltaNet layer window");
                                        exec.copy_region(
                                            &bs.d_conv_ext,
                                            take * conv_dim,
                                            win,
                                            woff,
                                            km1 * conv_dim,
                                        )?;
                                    }
                                }
                                rb += take;
                            }
                            if let Some(n_ws) = d_win_spans {
                                let win = bs.conv_win[li].as_mut().expect("DeltaNet layer window");
                                exec.conv_win_store_vl(
                                    &sc.d_mixed,
                                    &d_win_buf,
                                    win,
                                    n_ws,
                                    km1,
                                    conv_dim,
                                )?;
                            }
                            if !(dn_offset && exec.has_conv_silu_qkv()) {
                                exec.deltanet_split_gqa_norm(
                                    &sc.d_conv,
                                    &mut sc.d_dq,
                                    &mut sc.d_dk,
                                    &mut sc.d_dv,
                                    r,
                                    n_k_heads,
                                    n_v_heads,
                                    state_size,
                                )?;
                            }
                            if dn_ab_done {
                                // alpha/beta already landed by the fused f8t in-proj
                                // GEMM above (same fold the decode tick takes)
                                exec.delta_gate(
                                    &sc.d_a,
                                    &sc.d_b,
                                    &w.ssm_a.buf,
                                    &w.dt_bias.buf,
                                    &mut sc.d_g,
                                    &mut sc.d_beta,
                                    r,
                                    n_v_heads,
                                )?;
                            } else if let Some(ab) = w
                                .ab_f32
                                .as_ref()
                                .filter(|_| r >= ab_f32_min_rows() || w.alpha_w.is_none())
                            {
                                // x2-v3: one f32-plane decay GEMM (64-col tile, x read once) +
                                // fused-layout gate; same values, tiled order (PPL-gated opt-in)
                                ab_gate(
                                    &exec,
                                    ab,
                                    &sc.d_xn,
                                    &mut sc.d_ab,
                                    &w.ssm_a.buf,
                                    &w.dt_bias.buf,
                                    &mut sc.d_g,
                                    &mut sc.d_beta,
                                    r,
                                    n_v_heads,
                                )?;
                            } else {
                                if exec.has_q8_0_gemm_repacked_x2() {
                                    // fused pair: x staged once for both decay projections
                                    // (bit-exact per output vs the two separate calls)
                                    exec.q8_0_gemm_repacked_x2(
                                        w.alpha_w.as_ref().expect("Q8 alpha (x2 path)"),
                                        w.beta_w.as_ref().expect("Q8 beta (x2 path)"),
                                        &sc.d_xn,
                                        &mut sc.d_a,
                                        &mut sc.d_b,
                                        r,
                                    )?;
                                } else {
                                    exec.q8_0_gemm_repacked(
                                        w.alpha_w.as_ref().expect("Q8 alpha"),
                                        None,
                                        &sc.d_xn,
                                        &mut sc.d_a,
                                        r,
                                    )?;
                                    exec.q8_0_gemm_repacked(
                                        w.beta_w.as_ref().expect("Q8 beta"),
                                        None,
                                        &sc.d_xn,
                                        &mut sc.d_b,
                                        r,
                                    )?;
                                }
                                exec.delta_gate(
                                    &sc.d_a,
                                    &sc.d_b,
                                    &w.ssm_a.buf,
                                    &w.dt_bias.buf,
                                    &mut sc.d_g,
                                    &mut sc.d_beta,
                                    r,
                                    n_v_heads,
                                )?;
                            }
                            // per-prompt recurrence: copy the span to base-0 temps and run
                            // the exact serial dispatch (chunked scan for long spans, else
                            // v2) into the slot's state - bit-identical to prefill_slot_chunk.
                            let mut rb = 0;
                            for (si, &(_, slot, take)) in shares.iter().enumerate() {
                                // graph pads: every padded share rides VL (true takes
                                // in device quads), so this loop only advances rb
                                let take = wtake(take);
                                let off = slot * state_elems;
                                let chunked =
                                    take >= chunk_min && state_size == 128 && !no_chunked_dn;
                                if vl_pf[si] {
                                    // rides the tick-wide varlen launch below
                                    rb += take;
                                    continue;
                                }
                                if dn_offset {
                                    // read the span in place at row `rb`, write d_dattn in
                                    // place - no base-0 copies. Same kernels/bytes -> identical.
                                    if chunked {
                                        exec.gated_delta_chunked_at(
                                            &sc.d_dq,
                                            &sc.d_dk,
                                            &sc.d_dv,
                                            &sc.d_g,
                                            &sc.d_beta,
                                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                                            off,
                                            &mut sc.d_dattn,
                                            rb,
                                            &mut sc.d_dnc_dw,
                                            &mut sc.d_dnc_du,
                                            &mut sc.d_dnc_coef,
                                            &mut sc.d_dnc_cg,
                                            take,
                                            n_v_heads,
                                            state_size,
                                        )?;
                                    } else if d_pf_items.is_none() {
                                        // short spans ride the one packed launch below
                                        exec.gated_delta_recurrent_v2_at(
                                            &sc.d_dq,
                                            &sc.d_dk,
                                            &sc.d_dv,
                                            &sc.d_g,
                                            &sc.d_beta,
                                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                                            off,
                                            &mut sc.d_dattn,
                                            rb,
                                            take,
                                            n_v_heads,
                                            state_size,
                                        )?;
                                    }
                                } else {
                                    exec.copy_region(
                                        &sc.d_dq,
                                        rb * value_dim,
                                        &mut d_pf_dq,
                                        0,
                                        take * value_dim,
                                    )?;
                                    exec.copy_region(
                                        &sc.d_dk,
                                        rb * value_dim,
                                        &mut d_pf_dk,
                                        0,
                                        take * value_dim,
                                    )?;
                                    exec.copy_region(
                                        &sc.d_dv,
                                        rb * value_dim,
                                        &mut d_pf_dv,
                                        0,
                                        take * value_dim,
                                    )?;
                                    exec.copy_region(
                                        &sc.d_g,
                                        rb * n_v_heads,
                                        &mut d_pf_g,
                                        0,
                                        take * n_v_heads,
                                    )?;
                                    exec.copy_region(
                                        &sc.d_beta,
                                        rb * n_v_heads,
                                        &mut d_pf_beta,
                                        0,
                                        take * n_v_heads,
                                    )?;
                                    if chunked {
                                        exec.gated_delta_chunked(
                                            &d_pf_dq,
                                            &d_pf_dk,
                                            &d_pf_dv,
                                            &d_pf_g,
                                            &d_pf_beta,
                                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                                            off,
                                            &mut d_pf_dattn,
                                            &mut sc.d_dnc_dw,
                                            &mut sc.d_dnc_du,
                                            &mut sc.d_dnc_coef,
                                            &mut sc.d_dnc_cg,
                                            take,
                                            n_v_heads,
                                            state_size,
                                        )?;
                                    } else {
                                        exec.gated_delta_recurrent_v2(
                                            &d_pf_dq,
                                            &d_pf_dk,
                                            &d_pf_dv,
                                            &d_pf_g,
                                            &d_pf_beta,
                                            None,
                                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                                            off,
                                            None,
                                            &mut d_pf_dattn,
                                            1,
                                            take,
                                            n_v_heads,
                                            state_size,
                                        )?;
                                    }
                                    exec.copy_region(
                                        &d_pf_dattn,
                                        0,
                                        &mut sc.d_dattn,
                                        rb * value_dim,
                                        take * value_dim,
                                    )?;
                                }
                                rb += take;
                            }
                            if let Some((n_chunks, span_off, n_spans)) = d_pf_vl {
                                // One stage1+walk pair for every >= 128-row span
                                if qkc_tick {
                                    exec.gated_delta_chunked_rs_vl_qkc(
                                        &sc.d_dq,
                                        &sc.d_dk,
                                        &sc.d_dv,
                                        &sc.d_g,
                                        &sc.d_beta,
                                        bs.recur[li].as_mut().expect("DeltaNet layer state"),
                                        &mut sc.d_dattn,
                                        &mut sc.d_dnc_dw,
                                        &mut sc.d_dnc_du,
                                        &mut sc.d_dnc_coef,
                                        &mut sc.d_dnc_cg,
                                        &d_vl_buf,
                                        n_chunks,
                                        span_off,
                                        n_spans,
                                        r,
                                        n_v_heads,
                                        n_k_heads,
                                        state_size,
                                    )?;
                                } else {
                                    exec.gated_delta_chunked_rs_vl(
                                        &sc.d_dq,
                                        &sc.d_dk,
                                        &sc.d_dv,
                                        &sc.d_g,
                                        &sc.d_beta,
                                        bs.recur[li].as_mut().expect("DeltaNet layer state"),
                                        &mut sc.d_dattn,
                                        &mut sc.d_dnc_dw,
                                        &mut sc.d_dnc_du,
                                        &mut sc.d_dnc_coef,
                                        &mut sc.d_dnc_cg,
                                        &d_vl_buf,
                                        n_chunks,
                                        span_off,
                                        n_spans,
                                        r,
                                        n_v_heads,
                                        state_size,
                                    )?;
                                }
                            }
                            if let Some(n_items) = d_pf_items {
                                // One launch for every remaining short span
                                exec.gated_delta_recurrent_v2_packed(
                                    &sc.d_dq,
                                    &sc.d_dk,
                                    &sc.d_dv,
                                    &sc.d_g,
                                    &sc.d_beta,
                                    &d_items_buf,
                                    bs.recur[li].as_mut().expect("DeltaNet layer state"),
                                    &mut sc.d_dattn,
                                    None,
                                    None,
                                    n_items,
                                    n_v_heads,
                                    state_size,
                                )?;
                            }
                            if paddock_models::dev_var_os!("PADDOCK_SPEC_TRACE").is_some() {
                                // contamination-hunt bracket: the state each slot region
                                // holds right after this layer's prefill recurrence
                                let fp = |o: usize| -> f64 {
                                    bs.recur[li]
                                        .as_ref()
                                        .and_then(|b| b.try_slice(o..o + 256))
                                        .and_then(|v| exec.stream.clone_dtoh(&v).ok())
                                        .map(|h| h.iter().map(|&x| x as f64).sum())
                                        .unwrap_or(f64::NAN)
                                };
                                tracing::info!(
                                    "TRACE post-prefill-recur li={li} fp_slot0 {:.6} fp_slot1 {:.6}",
                                    fp(0),
                                    fp(state_elems),
                                );
                            }
                            // d_xn/d_pxq/d_exs (or d_yq on the Q8 path) untouched since
                            // in_qkv's quant: reuse the same e4m3 activations for gate_w.
                            // (dn_fused via the f8t arm already landed d_z - skip both.)
                            if let Some(l8) = lw8 {
                                if !dn_fused {
                                    exec.f8_gemm_w8(
                                        l8.in_qkv.as_ref().expect("DeltaNet W8 in_qkv plane"),
                                        w.in_qkv.dims()[1],
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_z,
                                        w.gate_w.dims()[0],
                                        w.gate_w.dims()[1],
                                        r,
                                    )?;
                                }
                            } else if !dn_fused {
                                prefill_mm_pre_any(
                                    &exec,
                                    &w.gate_w,
                                    &sc.d_pxq,
                                    &sc.d_pxs,
                                    &sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &mut sc.d_z,
                                    r,
                                )?;
                            }
                            // DN out_proj glue (GDN formulation band):
                            // fused gated-rmsnorm + e4m3 quant on the w8 arm, with
                            // the f32 d_core store skipped - the GEMM's q/scale
                            // planes are the only consumer on this path. Bytes are
                            // bit-identical to the norm + standalone-quantize pair.
                            // f8t out_w arm excluded: it needs the f32 d_core + the
                            // row-quant seam, not the linear e4m3 planes.
                            let gr_fused =
                                lw8.is_some() && f8t_ow.is_none() && exec.has_gated_rmsnorm_e4m3();
                            if gr_fused {
                                exec.gated_rmsnorm_e4m3(
                                    &sc.d_dattn,
                                    &sc.d_z,
                                    &w.ssm_norm.buf,
                                    None,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * n_v_heads,
                                    state_size,
                                    eps,
                                )?;
                            } else {
                                exec.gated_rmsnorm(
                                    &sc.d_dattn,
                                    &sc.d_z,
                                    &w.ssm_norm.buf,
                                    &mut sc.d_core,
                                    r * n_v_heads,
                                    state_size,
                                    eps,
                                )?;
                            }
                            if let Some(ow_t) = f8t_ow {
                                exec.quantize_e4m3_row(
                                    &sc.d_core,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    w.out_w.dims()[0],
                                    r,
                                )?;
                                exec.f8t_gemm(
                                    ow_t,
                                    &sc.d_f8t_q,
                                    &sc.d_f8t_rs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_proj,
                                    w.out_w.dims()[0],
                                    w.out_w.dims()[1],
                                    r,
                                )?;
                            } else if let Some(l8) = lw8 {
                                if !gr_fused {
                                    exec.quantize_e4m3(
                                        &sc.d_core,
                                        &mut sc.d_pxq,
                                        &mut sc.d_exs,
                                        r * w.out_w.dims()[0],
                                    )?;
                                }
                                static MO16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                                let mo16 = *MO16.get_or_init(|| {
                                    paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                        && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                                });
                                // P74: PADDOCK_QWEN35_O16_TC5 opts the sm_100 tc5
                                // bf16-store route in (the TMA lane is process-killed
                                // there); r >= 256 = the tc5v NT2 floor. MEASURED A
                                // WASH on B200 (tc5s pays ~10% store-poison for bf16
                                // stores - the muse f16 ledger class - and the b16
                                // consumers hold no net win); kept as probe infra.
                                static MO16T: std::sync::OnceLock<bool> =
                                    std::sync::OnceLock::new();
                                let mo16t = *MO16T.get_or_init(|| {
                                    paddock_models::dev_var_os!("PADDOCK_QWEN35_O16_TC5").is_some()
                                        && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                                });
                                if (mo16 || (mo16t && r >= 256)) && exec.has_f8_o16() {
                                    exec.f8_gemm_w8_o16(
                                        l8.out_w.as_ref().expect("DeltaNet W8 out_w plane"),
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_proj,
                                        w.out_w.dims()[0],
                                        w.out_w.dims()[1],
                                        r,
                                    )?;
                                    mixer_b16 = true;
                                } else {
                                    exec.f8_gemm_w8(
                                        l8.out_w.as_ref().expect("DeltaNet W8 out_w plane"),
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_proj,
                                        w.out_w.dims()[0],
                                        w.out_w.dims()[1],
                                        r,
                                    )?;
                                }
                            } else {
                                prefill_mm_any(
                                    &exec,
                                    &w.out_w,
                                    &mut sc.d_pxq,
                                    &mut sc.d_pxs,
                                    &mut sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &sc.d_core,
                                    &mut sc.d_proj,
                                    r,
                                )?;
                            }
                        }
                    }
                    let mut proj_is_b16 = false;
                    match &layer.ffn {
                        Ffn::Dense { gate, up, down } => {
                            // prefill-FFN f8 arm: the W8 prefill class extended to
                            // the FFN, because ~70% of prefill bytes were still
                            // running through int8-mmq. f8_gemm_w8 measures
                            // 1.27-1.85x best-q8 at M >= 512.
                            // Same e4m3 planes the decode lane built; same w8_min gate.
                            // f8t chunk arm for the FFN half: same gu|down tile
                            // planes the decode/unified ticks ride.
                            let f8t_ffn =
                                bs_f8t_ffn_p.get(li).and_then(|o| o.as_ref()).filter(|p| {
                                    f8t_c && tc5r_ok(p[0].scale.len()) && tc5r_ok(p[1].scale.len())
                                });
                            let f8f = bs_f8ffn_p.get(li).and_then(|o| o.as_ref()).filter(|_| {
                                r > super::f8_ffn_pf_min()
                                    && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
                            });
                            // write_xn must include the f8t consumer: at r > 64 the
                            // mmq norm route writes d_xn only when asked (the unified
                            // arm never noticed - its r <= 64 else-route always wrote
                            // xn unconditionally).
                            let f8r = bs_f8row_p.get(li).and_then(|o| o.as_ref());
                            prefill_add_norm_quant(
                                &exec,
                                &mut sc.d_x,
                                Some(&sc.d_proj),
                                mixer_b16,
                                &layer.post_norm.buf,
                                &mut sc.d_xn,
                                f8r.is_some() || f8f.is_some() || f8t_ffn.is_some(),
                                &mut sc.d_pxq,
                                &mut sc.d_pxs,
                                &mut sc.d_yq,
                                embd,
                                r,
                                eps,
                            )?;
                            if let Some(p) = f8r {
                                super::ops::ffn_f8row_rows(
                                    &exec,
                                    p,
                                    &sc.d_xn,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    &mut sc.d_ffn_gate,
                                    &mut sc.d_ffn_up,
                                    &mut sc.d_proj,
                                    r,
                                )?;
                            } else if let Some([gu_t, dn_t]) = f8t_ffn {
                                exec.quantize_e4m3_row(
                                    &sc.d_xn,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    embd,
                                    r,
                                )?;
                                // P62 gluq silu twin: fused swiglu+quantize in the
                                // cutlass epilogue (act=1) replaces GEMM+swiglu+
                                // quant; decline falls through to the classic chain.
                                exec.f8t_gemm(
                                    gu_t,
                                    &sc.d_f8t_q,
                                    &sc.d_f8t_rs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_ffn_gate,
                                    embd,
                                    2 * ff,
                                    r,
                                )?;
                                exec.swiglu_fused(&sc.d_ffn_gate, &mut sc.d_ffn_up, ff, r)?;
                                exec.quantize_e4m3_row(
                                    &sc.d_ffn_up,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    ff,
                                    r,
                                )?;
                                exec.f8t_gemm(
                                    dn_t,
                                    &sc.d_f8t_q,
                                    &sc.d_f8t_rs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_proj,
                                    ff,
                                    embd,
                                    r,
                                )?;
                            } else if let Some([gu8, d8]) = f8f {
                                // fused plane, row-sliced: gate = rows [0,ff), up =
                                // rows [ff,2ff) - byte-identical to the old separate
                                // planes (same repack stream, offset math only)
                                let ffh = gu8.2 / 2;
                                exec.quantize_e4m3(
                                    &sc.d_xn,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * gu8.1,
                                )?;
                                // bf16 epilogue pair when the pack ships it: halves
                                // the gate/up store traffic (the rival's cutlass
                                // writes bf16; ours wrote f32) and the fused quant
                                // reads bf16 - else the f32 chain below.
                                static O16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                                let o16 = *O16.get_or_init(|| {
                                    paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                        && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                                });
                                // P74: see the MO16T note - sm_100 tc5 route, wash
                                static O16T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                                let o16 = o16
                                    || (*O16T.get_or_init(|| {
                                        paddock_models::dev_var_os!("PADDOCK_QWEN35_O16_TC5")
                                            .is_some()
                                            && paddock_models::dev_var_os!("PADDOCK_NO_O16")
                                                .is_none()
                                    }) && r >= 256);
                                if o16 && exec.has_f8_o16() {
                                    if exec.has_swiglu_b16_gu() {
                                        // One fused gate|up GEMM - bit-exact vs the
                                        // sliced pair (ffh is 128-tile-aligned, so
                                        // every output element keeps its K chain);
                                        // d_ffn_gate's r*ffh f32 capacity holds the
                                        // r*2ffh bf16 fused output byte-exactly.
                                        exec.f8_gemm_w8_o16(
                                            &gu8.0,
                                            0,
                                            &sc.d_pxq,
                                            &sc.d_exs,
                                            &mut sc.d_ffn_gate,
                                            gu8.1,
                                            gu8.2,
                                            r,
                                        )?;
                                        exec.quantize_e4m3_swiglu_b16_gu(
                                            &sc.d_ffn_gate,
                                            &mut sc.d_pxq,
                                            &mut sc.d_exs,
                                            r * d8.1,
                                            ffh,
                                        )?;
                                    } else {
                                        exec.f8_gemm_w8_o16(
                                            &gu8.0,
                                            0,
                                            &sc.d_pxq,
                                            &sc.d_exs,
                                            &mut sc.d_ffn_gate,
                                            gu8.1,
                                            ffh,
                                            r,
                                        )?;
                                        exec.f8_gemm_w8_o16(
                                            &gu8.0,
                                            ffh,
                                            &sc.d_pxq,
                                            &sc.d_exs,
                                            &mut sc.d_ffn_up,
                                            gu8.1,
                                            ffh,
                                            r,
                                        )?;
                                        exec.quantize_e4m3_swiglu_b16(
                                            &sc.d_ffn_gate,
                                            &sc.d_ffn_up,
                                            &mut sc.d_pxq,
                                            &mut sc.d_exs,
                                            r * d8.1,
                                        )?;
                                    }
                                } else {
                                    exec.f8_gemm_w8(
                                        &gu8.0,
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_ffn_gate,
                                        gu8.1,
                                        ffh,
                                        r,
                                    )?;
                                    exec.f8_gemm_w8(
                                        &gu8.0,
                                        ffh,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_ffn_up,
                                        gu8.1,
                                        ffh,
                                        r,
                                    )?;
                                    // fused swiglu+e4m3-quant: one pass instead of
                                    // swiglu-write + quant-read (286 MB/layer-tick of f32
                                    // round-trip at r=2048 - the bf16-activations gap
                                    // vs the engines that write bf16 epilogues, closed at
                                    // the seam that matters)
                                    exec.quantize_e4m3_swiglu(
                                        &sc.d_ffn_gate,
                                        &sc.d_ffn_up,
                                        &mut sc.d_pxq,
                                        &mut sc.d_exs,
                                        r * d8.1,
                                    )?;
                                }
                                if o16 && exec.has_f8_o16() && exec.has_add_b16() {
                                    // bf16 down out (halves the last 42 MB/layer-tick
                                    // f32 store of the FFN) - the tail add reads bf16
                                    exec.f8_gemm_w8_o16(
                                        &d8.0,
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_proj,
                                        d8.1,
                                        d8.2,
                                        r,
                                    )?;
                                    proj_is_b16 = true;
                                } else {
                                    exec.f8_gemm_w8(
                                        &d8.0,
                                        0,
                                        &sc.d_pxq,
                                        &sc.d_exs,
                                        &mut sc.d_proj,
                                        d8.1,
                                        d8.2,
                                        r,
                                    )?;
                                }
                            } else {
                                prefill_mm_pre_any(
                                    &exec,
                                    gate,
                                    &sc.d_pxq,
                                    &sc.d_pxs,
                                    &sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &mut sc.d_ffn_gate,
                                    r,
                                )?;
                                prefill_mm_pre_any(
                                    &exec,
                                    up,
                                    &sc.d_pxq,
                                    &sc.d_pxs,
                                    &sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &mut sc.d_ffn_up,
                                    r,
                                )?;
                                prefill_ffn_down_any(
                                    &exec,
                                    down,
                                    &mut sc.d_pxq,
                                    &mut sc.d_pxs,
                                    &mut sc.d_yq,
                                    &mut sc.d_xsums,
                                    &mut sc.d_ssums,
                                    &mut sc.d_skfix,
                                    &mut sc.d_ffn_gate,
                                    &sc.d_ffn_up,
                                    &mut sc.d_proj,
                                    ff,
                                    r,
                                )?;
                            }
                        }
                        Ffn::Nvf4Dense { gate, up, down } => {
                            // f8t tile arm first, off the planes load.rs builds from
                            // the NVFP4 checkpoint's own values - same chain and same
                            // election as the Dense arm above. write_xn stays true:
                            // both arms consume the f32 xn (f8t quantizes it itself).
                            // f8t_c is not optional here: it carries `2 * r <= sc.cap`,
                            // the invariant ensure_scratch establishes for the
                            // [r, 2*ff] landing into d_ffn_gate. Dropping it overruns
                            // that buffer on chunked prefill (caught live: throughput
                            // halves after the first rep, with multi-second tick
                            // stalls at `chunking 16`).
                            let f8t_ffn =
                                bs_f8t_ffn_p.get(li).and_then(|o| o.as_ref()).filter(|p| {
                                    f8t_c && tc5r_ok(p[0].scale.len()) && tc5r_ok(p[1].scale.len())
                                });
                            // Above the tile band the chain takes the same f8w arm
                            // the Dense lane runs (planes built from the checkpoint's
                            // own values in load.rs) instead of nvf4_ffn's W4A16
                            // decode walk.
                            // the DECODE lane's arm, at wave widths. Above
                            // the tile arm's decode-band row bound the fp4 gate|up +
                            let f8f = bs_f8ffn_p.get(li).and_then(|o| o.as_ref()).filter(|_| {
                                f8t_ffn.is_none()
                                    && r > nvf4_f8w_min_rows(w8_min)
                                    && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
                            });
                            prefill_add_norm_quant(
                                &exec,
                                &mut sc.d_x,
                                Some(&sc.d_proj),
                                mixer_b16,
                                &layer.post_norm.buf,
                                &mut sc.d_xn,
                                true,
                                &mut sc.d_pxq,
                                &mut sc.d_pxs,
                                &mut sc.d_yq,
                                embd,
                                r,
                                eps,
                            )?;
                            if let Some([gu_t, dn_t]) = f8t_ffn {
                                exec.quantize_e4m3_row(
                                    &sc.d_xn,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    embd,
                                    r,
                                )?;
                                exec.f8t_gemm(
                                    gu_t,
                                    &sc.d_f8t_q,
                                    &sc.d_f8t_rs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_ffn_gate,
                                    embd,
                                    2 * ff,
                                    r,
                                )?;
                                exec.swiglu_fused(&sc.d_ffn_gate, &mut sc.d_ffn_up, ff, r)?;
                                exec.quantize_e4m3_row(
                                    &sc.d_ffn_up,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    ff,
                                    r,
                                )?;
                                exec.f8t_gemm(
                                    dn_t,
                                    &sc.d_f8t_q,
                                    &sc.d_f8t_rs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_proj,
                                    ff,
                                    embd,
                                    r,
                                )?;
                            } else if let Some([gu8, d8]) = f8f {
                                proj_is_b16 = prefill_ffn_f8w(
                                    &exec,
                                    gu8,
                                    d8,
                                    &sc.d_xn,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    &mut sc.d_ffn_gate,
                                    &mut sc.d_ffn_up,
                                    &mut sc.d_proj,
                                    r,
                                )?;
                            } else {
                                // no plane pair (small card / kill switch): the
                                // checkpoint-exact W4A16 walk
                                nvf4_ffn(
                                    &exec,
                                    gate,
                                    up,
                                    down,
                                    &sc.d_xn,
                                    &mut sc.d_pxq,
                                    &mut sc.d_nvs,
                                    &mut sc.d_nv4part,
                                    &mut sc.d_ffn_gate,
                                    &mut sc.d_ffn_up,
                                    &mut sc.d_proj,
                                    ff,
                                    r,
                                )?;
                            }
                        }
                        Ffn::Moe(w) => {
                            prefill_add_norm_quant(
                                &exec,
                                &mut sc.d_x,
                                Some(&sc.d_proj),
                                mixer_b16,
                                &layer.post_norm.buf,
                                &mut sc.d_xn,
                                true,
                                &mut sc.d_pxq,
                                &mut sc.d_pxs,
                                &mut sc.d_yq,
                                embd,
                                r,
                                eps,
                            )?;
                            moe_ffn(
                                &exec,
                                w,
                                moe_dims.expect("moe dims"),
                                embd,
                                r,
                                true,
                                &sc.d_xn,
                                &mut sc.d_moe_xq,
                                &mut sc.d_moe_xs,
                                &mut sc.d_ssums,
                                &mut sc.d_moe_xs8,
                                &mut sc.d_moe_fs8,
                                &mut sc.d_moe_logits,
                                &sc.d_zero_bias,
                                &mut sc.d_moe_idx,
                                &mut sc.d_moe_w,
                                &mut sc.d_moe_fused,
                                &mut sc.d_moe_fq,
                                &mut sc.d_moe_fs,
                                &mut sc.d_moe_srow,
                                &mut sc.d_moe_sslot,
                                &mut sc.d_moe_bexp,
                                &mut sc.d_moe_part,
                                &mut sc.d_pxq,
                                &mut sc.d_pxs,
                                &mut sc.d_yq,
                                &mut sc.d_skfix,
                                &mut sc.d_ffn_gate,
                                &mut sc.d_ffn_up,
                                &mut sc.d_mixed,
                                &mut sc.d_proj,
                            )?;
                        }
                    }
                    if pf_res_hoist {
                        // P73 hoist: the next layer's fused prenorm consumes d_proj
                        pend_ffn = Some(proj_is_b16);
                    } else if proj_is_b16 {
                        exec.add_b16(&mut sc.d_x, &sc.d_proj, r * embd)?;
                    } else {
                        exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
                    }
                }

                // P73 hoist flush: the last layer's FFN residual lands before the
                // final norm (which has no residual arm)
                match pend_ffn.take() {
                    Some(true) => exec.add_b16(&mut sc.d_x, &sc.d_proj, r * embd)?,
                    Some(false) => exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?,
                    None => {}
                }

                exec.rmsnorm_batch(&sc.d_x, &out_norm.buf, &mut sc.d_h, embd, eps, r)?;
                if graph_ok {
                    // device half of the logits epilogue, captured with the walk:
                    // indexed gather of each share's true last row (d_gidx contents)
                    // + the same lm-head class BFL/serial-n1 uses - identical bytes.
                    let n_sh = shares.len();
                    exec.hrow_gather(&sc.d_h, &d_gidx, &mut sc.d_xn, n_sh, embd)?;
                    // f8 arm first: at n_sh == 1 this used to fall to the Q8_0 head
                    // unconditionally, which alone kept that plane alive once every
                    // other band moved (non-KV-overhead, qwen head reclaim).
                    if let Some((p8, pi, po)) =
                        out_f8.as_ref().filter(|_| n_sh >= super::f8_head_min())
                    {
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, n_sh * embd)?;
                        exec.f8d_gemm_mma_ks(
                            p8,
                            *pi,
                            *po,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut bs.d_ks_part,
                            &mut bs.d_logits,
                            n_sh,
                        )?;
                    } else if n_sh == 1 {
                        // no f8 head (or floor not lowered): one row belongs on the
                        // GEMV, not the batched dp4a GEMM
                        gemv_any(&exec, output, &sc.d_xn, &mut sc.d_logits)?;
                    } else {
                        exec.quantize_q8(&sc.d_xn, &mut bs.d_xq, &mut bs.d_xs, n_sh * embd)?;
                        super::stub_guard(output, "batch.rs prefill batched head")?;
                        exec.q8_0_gemm_mt_dp4a(
                            output.q8(),
                            &bs.d_xq,
                            &bs.d_xs,
                            &mut bs.d_logits,
                            n_sh,
                        )?;
                    }
                }
                Ok(())
            })()
        };
        if pf_capture {
            let g = crate::gpu::end_capture_no_flags(&exec.stream)
                .map_err(|e| GpuError::Driver(format!("pf end_capture: {e}")));
            rec?;
            let g = g?.ok_or_else(|| GpuError::Driver("pf capture produced no graph".into()))?;
            g.launch()
                .map_err(|e| GpuError::Driver(format!("pf graph launch: {e}")))?;
            bs.pf_pass_graphs.insert(pf_key, SendGraph(g));
        } else {
            rec?;
        }
        if pf_replay {
            bs.pf_pass_graphs[&pf_key]
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("pf graph replay: {e}")))?;
        }
        // host half: the one readback per pass (graph path), placed exactly
        // where the eager epilogue's readback sits
        if graph_ok {
            if shares.len() == 1 {
                out[shares[0].0] = exec.to_host(&sc.d_logits)?;
            } else {
                let all = exec.to_host_len(&bs.d_logits, shares.len() * vocab)?;
                for (i, &(oi, _, _)) in shares.iter().enumerate() {
                    out[oi] = all[i * vocab..(i + 1) * vocab].to_vec();
                }
            }
        }
        // each prompt's last row -> its next-token logits (one gemv per prompt).
        // Batched final logits: the serial form ran one 675us
        // q8_0_gemv_repacked + one BLOCKING to_host per share - n round-trips
        // per admission pass, ~21.6ms/wave at c32 (ttft_cap).
        // Gather the n last rows and run one lm_head GEMM + one transfer;
        // n >= 8 rides the shipped f8 lm_head class (the same class every
        // decode token's logits get at those widths). Kq files
        // keep the serial exact path. PADDOCK_NO_BFL restores the gemvs.
        static BFL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let bfl = *BFL.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_BFL").is_none());
        let n_sh = shares.len();
        if graph_ok {
            // logits epilogue already ran inside the capture bracket above
        } else if bfl && n_sh >= 2 && matches!(output, QuantW::Q8(_)) {
            let mut rb = 0;
            for (i, &(_oi, _slot, take)) in shares.iter().enumerate() {
                exec.copy_region(
                    &sc.d_h,
                    (rb + take - 1) * embd,
                    &mut sc.d_xn,
                    i * embd,
                    embd,
                )?;
                rb += take;
            }
            if let Some((p8, pi, po)) = self
                .out_f8
                .as_ref()
                .filter(|_| n_sh >= super::f8_head_min())
            {
                exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, n_sh * embd)?;
                exec.f8d_gemm_mma_ks(
                    p8,
                    *pi,
                    *po,
                    &sc.d_pxq,
                    &sc.d_exs,
                    &mut bs.d_ks_part,
                    &mut bs.d_logits,
                    n_sh,
                )?;
            } else {
                exec.quantize_q8(&sc.d_xn, &mut bs.d_xq, &mut bs.d_xs, n_sh * embd)?;
                super::stub_guard(output, "batch.rs prefill batched head")?;
                exec.q8_0_gemm_mt_dp4a(output.q8(), &bs.d_xq, &bs.d_xs, &mut bs.d_logits, n_sh)?;
            }
            let all = exec.to_host_len(&bs.d_logits, n_sh * vocab)?;
            for (i, &(oi, _slot, _take)) in shares.iter().enumerate() {
                out[oi] = all[i * vocab..(i + 1) * vocab].to_vec();
            }
        } else {
            let mut rb = 0;
            for &(oi, _slot, take) in &shares {
                exec.copy_region(&sc.d_h, (rb + take - 1) * embd, &mut sc.d_xn, 0, embd)?;
                // Live at n_sh == 1 on the non-graph path -- the f8 arms above
                // are both n_sh-gated and this serial fallback had none.
                head_logits_1row(
                    &exec,
                    out_f8_h,
                    output,
                    &sc.d_xn,
                    &mut sc.d_pxq,
                    &mut sc.d_exs,
                    &mut sc.d_head_part,
                    &mut sc.d_logits,
                    "batch.rs prefill serial head",
                )?;
                out[oi] = exec.to_host(&sc.d_logits)?;
                let _ = vocab;
                rb += take;
            }
        }

        // return the persistent pass buffers. The pass is
        // host-synchronized by the logits readback above, so nothing queued
        // still reads them.
        self.batch.as_mut().expect("batch enabled").pf_bufs = Some(PfPassBufs {
            d_pf_dq,
            d_pf_dk,
            d_pf_dv,
            d_pf_g,
            d_pf_beta,
            d_pf_dattn,
            d_pf_qn,
            d_pf_attn,
            d_seg_slot,
            d_seg_bound,
            d_seg_pos,
            d_vl: d_vl_buf,
            d_items: d_items_buf,
            d_win: d_win_buf,
            d_gidx,
            d_tokens,
            d_pos,
            d_slots,
            d_mrope,
            take_cap,
            r_cap,
            vl_cap,
            items_cap,
            win_cap,
        });

        // per-slot mrope delta: mm slots carry the diverged llama-position into
        // decode; text slots RESET to 0 (a slot that previously served an image
        // request would otherwise decode with a stale delta - latent before the
        // batched-mm path, load-bearing now).
        {
            let bs2 = self.batch.as_mut().expect("batch");
            for &(oi, slot, take) in &shares {
                bs2.mrope_delta[slot] =
                    mm.map_or(0, |m| m[oi].final_mrope_pos as i64 - take as i64);
            }
        }
        if mm.is_some() {
            // no spec warm for image prompts (placeholder ids would poison the
            // drafter's token history) - the solo mm path never warms either
            return Ok(());
        }

        // Warm hook, batched-cohort edition: without this the chunk-batched
        // prefill leaves slots COLD and spec never engages there - measurably
        // flat at those widths. Shares are always FRESH
        // whole prompts here (v1 semantics) -> start=0, zeroed h carry; each
        // share's h rows sit at its row base inside this pass's d_h. Runs
        // after the logits loop - the warm pass clobbers d_x/d_xn but only
        // reads d_h, which nothing touches below. Long prompts skip (same
        // WARM_MAX rationale as the serial hook); spec_warm_wanted is the
        // scheduler's live-count hint (live > spec cap -> skip).
        if self.spec_warm_wanted && self.serve_spec_on() {
            self.ensure_serve_spec()?;
            let warm_max: usize = std::env::var("PADDOCK_QWEN35_SPEC_WARM_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048);
            let embd_w = self.embd;
            let drv2 = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
            let mut rb = 0usize;
            for &(oi, slot, take) in &shares {
                let in_range = slot < self.spec_batch.as_ref().expect("spec enabled").alloc_batch;
                if in_range && take <= warm_max {
                    let zeros = vec![0f32; embd_w];
                    {
                        let sb = self.spec_batch.as_mut().expect("spec enabled");
                        let mut v = sb.pending_h.slice_mut(slot * embd_w..(slot + 1) * embd_w);
                        self.exec.stream.memcpy_htod(&zeros, &mut v).map_err(drv2)?;
                    }
                    self.mtp_warm_slot(slot, &items[oi].1, 0, rb)?;
                    let sb = self.spec_batch.as_mut().expect("spec enabled");
                    sb.pos[slot] = take;
                    sb.mtp_warm[slot] = true;
                    sb.mtp_toks[slot].clear();
                    sb.mtp_toks[slot].extend_from_slice(&items[oi].1[..take]);
                } else if in_range {
                    self.spec_batch.as_mut().expect("spec enabled").mtp_warm[slot] = false;
                }
                // padded row stride under a captured pass (identity eager)
                rb += wtake(take);
            }
        }
        Ok(())
    }

    /// One continuous-batching decode step: `tokens[i]` at `positions[i]` drives
    /// slot i for i in 0..B. Every matmul runs once for the whole batch (int8 MMQ),
    /// so the ~weight read amortizes across the B concurrent sequences - the
    /// aggregate-throughput lever. Returns [B, vocab] logits.
    pub fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        // Split-decode companion (see batch_sampled_impl): this unsampled
        // tick was the last full-width path - at b > 64 it captured and
        // replayed a b=128 graph whose every layer recorded the mma
        // fallback (record-arm witness: 64x arm=mma b=128), while the
        // <=64-row graphs record the f8t/CUTLASS class. Run halves with an
        // explicit identity slot map and read each half's logits before
        // the next launch overwrites the rows.
        let bfull = tokens.len();
        let bmax = f8t_dec_bmax();
        if bfull > bmax && paddock_models::dev_var_os!("PADDOCK_NO_DEC_SPLIT").is_none() {
            let vocab = self.vocab;
            let exec = self.exec.clone();
            let mut out = Vec::with_capacity(bfull * vocab);
            let mut b0 = 0usize;
            while b0 < bfull {
                let bn = (bfull - b0).min(bmax);
                let slot_ids: Vec<u32> = (b0 as u32..(b0 + bn) as u32).collect();
                self.launch_batch_step_slots(
                    &tokens[b0..b0 + bn],
                    &positions[b0..b0 + bn],
                    Some(&slot_ids),
                )?;
                let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
                let view = bs.d_logits.try_slice(0..bn * vocab).ok_or_else(|| {
                    crate::gpu::GpuError::Driver("logits slice out of range".into())
                })?;
                let half = exec
                    .stream
                    .clone_dtoh(&view)
                    .map_err(|e| crate::gpu::GpuError::Driver(e.to_string()))?;
                out.extend(half);
                b0 += bn;
            }
            return Ok(out);
        }
        self.launch_batch_step(tokens, positions)?;
        // read back only the b live rows (the buffer holds max_batch*vocab -
        // copying it whole cost a fixed ~1.4 ms/step at any B)
        let (exec, vocab) = (self.exec.clone(), self.vocab);
        let b = tokens.len();
        let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
        let view = bs
            .d_logits
            .try_slice(0..b * vocab)
            .ok_or_else(|| GpuError::Driver("logits slice out of range".into()))?;
        let out = exec
            .stream
            .clone_dtoh(&view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        Ok(out)
    }

    /// True when the pack ships the fused row-sampling kernel - lets the
    /// scheduler take `forward_batch_sampled` and skip the [B, vocab] readback.
    /// K-quant models report false: the batched serving paths are Q8_0-class
    /// until the stage-2 W4A8 kernels, so they serve on the serial spine.
    pub fn supports_device_sampling(&self) -> bool {
        // sampling reads d_logits - the k-quant W4A8 head fills the same buffer
        self.exec.has_sample_rows()
    }

    /// `forward_batch` + fused on-device sampling (the gpt-oss path ported to
    /// qwen35): `Device` rows come back as bare token ids (b × 4 bytes) instead
    /// of the full `[b, vocab]` logits readback; `Host` rows (penalties,
    /// constraints, logprobs) still get their own logits row copied back. The
    /// sampling kernel is model-agnostic - it reads the same `d_logits` buffer
    /// `forward_batch` fills, so this is bit-identical to host argmax on greedy
    /// rows.
    pub fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        self.batch_sampled_impl(tokens, positions, None, plans)
    }

    /// Run `f` on the OVERLAP DECODE LANE when it is forked: `self.exec` is
    /// swapped to the lane for the duration, so every launch, htod, event and
    /// graph capture inside lands on the decode lane's streams (a cudarc
    /// graph replays on the stream that captured it - capture and replay must
    /// agree on the lane). Re-entrant: nested wrapped calls
    /// (batch_sampled_impl -> launch_batch_step_slots, pipe error paths ->
    /// pipe_abort) run in the already-swapped frame. No cross-lane fence is
    /// taken here: every wrapped path ends host-synchronized (blocking ids
    /// readback or an explicit stream sync) and main-lane passes end in a
    /// full sync, so each lane only ever reads state the host already
    /// observed complete. The overlap scheduler (B2) is what relaxes that,
    /// with explicit event fences at the span/pipe interleave points.
    pub(super) fn with_decode_lane<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        if self.lane_swapped || self.overlap_exec.is_none() {
            return f(self);
        }
        let lane = self.overlap_exec.take().expect("lane checked above");
        let main = std::mem::replace(&mut self.exec, lane);
        self.overlap_exec = Some(main);
        self.lane_swapped = true;
        let r = f(self);
        self.lane_swapped = false;
        let lane = std::mem::replace(
            &mut self.exec,
            self.overlap_exec.take().expect("main lane parked"),
        );
        self.overlap_exec = Some(lane);
        r
    }

    /// `forward_batch_sampled` with an optional explicit slot mapping (see
    /// `launch_batch_step_slots`). `None` = identity (pure decode); `Some` =
    /// the mixed tick's compacted decode rows at arbitrary slots. Runs on the
    /// overlap decode lane when forked.
    fn batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        // Split decode tick (the c128 lane): every fast decode
        // arm (mma_ks ladder, f8t plane, W8 tile class) is bounded b <= 64
        // (pack guards + tile shapes), so a 65..128-row tick fell to slow
        // eager lanes (~62ms vs 10.4ms at 64). At b > 64 any <=64 kernel
        // needs two weight passes anyway - run the tick as <=64-row halves
        // through the fast graphed path instead: rows are independent
        // (per-slot KV/state, per-row glue/sampling), so per-row results
        // are the 64-row tick's exact outputs. Requires explicit slots
        // (identity mapping cannot survive a split); serve always passes
        // them. Kill: PADDOCK_NO_DEC_SPLIT.
        let b = tokens.len();
        let bmax = f8t_dec_bmax();
        if b > bmax && paddock_models::dev_var_os!("PADDOCK_NO_DEC_SPLIT").is_none() {
            // identity mapping is materialized so the halves stay correct
            // (the None fast path cannot survive a split) - this None case
            // was the last b=128 hole: forward_batch_sampled passes None,
            // and the first post-warmup tick captured the mma graph.
            let ident: Vec<u32>;
            let sl: &[u32] = match slots {
                Some(s) => s,
                None => {
                    ident = (0..b as u32).collect();
                    &ident
                }
            };
            let mut ids = Vec::with_capacity(b);
            let mut host_rows = Vec::new();
            let mut b0 = 0usize;
            while b0 < b {
                let bn = (b - b0).min(bmax);
                let step = self.with_decode_lane(|m| {
                    m.batch_sampled_inner(
                        &tokens[b0..b0 + bn],
                        &positions[b0..b0 + bn],
                        Some(&sl[b0..b0 + bn]),
                        &plans[b0..b0 + bn],
                    )
                })?;
                ids.extend(step.ids);
                host_rows.extend(step.host_rows.into_iter().map(|(r, l)| (r + b0, l)));
                b0 += bn;
            }
            return Ok(crate::generator::SampledStep { ids, host_rows });
        }
        self.with_decode_lane(|m| m.batch_sampled_inner(tokens, positions, slots, plans))
    }

    fn batch_sampled_inner(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        use crate::sampler::DevicePlan;
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        // pack per-row sampler params; the kernel skips mode-0 rows (holes AND
        // host rows - the host reads the latter's logits itself)
        // small-b: the one-block-per-row prefilter is DRAM-latency-bound
        // (c1 -13% pre-fallback); the classic full-row readback + host
        // top-64 is cheaper there and distribution-identical
        let small_b = b <= 2;
        // P67: full-device sampling (mode 5) whenever the pack ships slot
        // 435 - the token lands in d_samp_out like modes 1/2, no head
        // readback, no host tail. Old packs keep mode 4 (host-head).
        let dev_full = !small_b && exec.has_sample_rows_t();
        let mut par = vec![0u32; b * 4];
        let mut tpar = vec![0u32; b * 4];
        let mut any_trunc = false;
        let mut any_base = false;
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => {
                    par[i * 4 + 2] = 1;
                    any_base = true;
                }
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                    any_base = true;
                }
                RowSample::Device(DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = if dev_full { 5 } else { 4 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    any_trunc = true;
                }
                // RS plans are gemma4-only (supports_spec_rs); skip-safe
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        {
            let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
            let mut v = bs.d_samp_par.slice_mut(0..b * 4);
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if any_trunc && dev_full {
                let mut t = bs.d_samp_tpar.slice_mut(0..b * 4);
                exec.stream.memcpy_htod(&tpar, &mut t).map_err(drv)?;
            }
        }
        self.launch_batch_step_slots(tokens, positions, slots)?;
        // DFlash: this tick's rows just walked (taps baked in the step
        // graph) - fuse and ring-append before the ids readback syncs.
        if self.dflash.as_ref().is_some_and(|d| d.state.is_some()) && !super::dflash::fuse_off() {
            let ident: Vec<u32>;
            let slots_v: &[u32] = match slots {
                Some(s) => s,
                None => {
                    ident = (0..b as u32).collect();
                    &ident
                }
            };
            self.dflash_append_features(tokens, positions, slots_v, None)?;
        }
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        // folded launches: this site writes mode 5 for every trunc row (no
        // mode 6 exists here), so the p chain was 11 launches of nothing
        let fold_off = Self::samp_fold_off();
        if any_base || fold_off {
            exec.sample_rows(&bs.d_logits, &bs.d_samp_par, &mut bs.d_samp_out, b, vocab)?;
        }
        if any_trunc && dev_full {
            // engagement witness (bisect-trap law): once per process
            static DEV5: std::sync::Once = std::sync::Once::new();
            DEV5.call_once(|| {
                eprintln!("[trunc-dev5] engaged: b={b} (full-device truncation sampling)");
            });
            Self::samp_fold_witness(any_base, true, false);
            exec.sample_rows_t(
                &bs.d_logits,
                &bs.d_samp_par,
                &bs.d_samp_tpar,
                &mut bs.d_samp_out,
                b,
                vocab,
            )?;
            if fold_off {
                exec.sample_rows_p(
                    &bs.d_logits,
                    &bs.d_samp_par,
                    &bs.d_samp_tpar,
                    &mut bs.d_samp_out,
                    b,
                    vocab,
                )?;
            }
        } else if any_trunc && !small_b {
            exec.topk_rows(
                &bs.d_logits,
                &bs.d_samp_par,
                &mut bs.d_samp_head,
                b,
                vocab,
                64,
            )?;
        }
        let ids_view = bs
            .d_samp_out
            .try_slice(0..b)
            .ok_or_else(|| GpuError::Driver("samp_out slice out of range".into()))?;
        let mut ids = exec
            .stream
            .clone_dtoh(&ids_view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        if any_trunc && dev_full {
            // mode 5: ids already carry the device-sampled tokens
        } else if any_trunc && !small_b {
            let hv = bs
                .d_samp_head
                .try_slice(0..b * 128)
                .ok_or_else(|| GpuError::Driver("samp_head slice out of range".into()))?;
            let head = exec
                .stream
                .clone_dtoh(&hv)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            trunc_fill_ids(&head, plans, &mut ids);
        } else if any_trunc {
            for (i, p) in plans.iter().enumerate() {
                if let RowSample::Device(DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                }) = *p
                {
                    let view = bs
                        .d_logits
                        .try_slice(i * vocab..(i + 1) * vocab)
                        .ok_or_else(|| GpuError::Driver("logits row slice out of range".into()))?;
                    let row = exec
                        .stream
                        .clone_dtoh(&view)
                        .map_err(|e| GpuError::Driver(e.to_string()))?;
                    let head = host_top64(&row);
                    ids[i] = crate::sampler::sample_trunc_head(&head, inv_t, u, k, top_p, min_p);
                }
            }
        }
        let mut host_rows = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                let view = bs
                    .d_logits
                    .try_slice(i * vocab..(i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("logits row slice out of range".into()))?;
                let row = exec
                    .stream
                    .clone_dtoh(&view)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                host_rows.push((i, row));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// P65: the host-head sampling finish is wired (TruncCat plans legal)
    /// whenever the pack ships the top-K prefilter. Old packs -> false ->
    /// truncation rows keep the classic full-row Host readback.
    pub fn host_head_supported(&self) -> bool {
        self.batch.is_some() && self.exec.has_topk_rows()
    }

    /// P67b: TruncCat rows are fully device-executed (slot 435 mode 5) -
    /// the pipe/overlap paths may admit them.
    pub fn device_trunc_supported(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows_t() && self.exec.has_sample_rows_p()
    }

    /// Chunked prefill is available once enable_batch has run. OPT-IN
    /// (PADDOCK_CHUNKED_PREFILL, DEFAULT-ON via apply_default_stack;
    /// kill: PADDOCK_NO_CHUNKED_PREFILL - the kill pins the classic
    /// blocking prefill),
    /// which an early sweep showed beats the chunked/unified path on the
    /// prefill-heavy concurrency configs (c8/c32/pf8). Enable it (with
    /// PADDOCK_UNIFIED=1 for the fused tick) for decode-dominated workloads where
    /// it wins (dc4/c1). Reverted from default-on after that measurement.
    pub fn supports_chunked_prefill(&self) -> bool {
        self.batch.is_some() && std::env::var_os("PADDOCK_CHUNKED_PREFILL").is_some()
    }

    /// Register a chunked prefill on `slot`. The whole prompt is queued; the
    /// unified tick advances it a budgeted SPAN per tick from `done` (or
    /// `advance_chunks` prefills the tail whole under PADDOCK_NO_UNIFIED). We
    /// match the prefix cache up front (`prefix_resume_begin`): a shared prefix
    /// ADOPTS the cached KV pages + restores the DeltaNet state into the slot, and
    /// `done` starts at the resume position so only the divergent tail re-prefills.
    pub fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GpuModelError> {
        self.prefill_begin_hinted(slot, tokens, &[]).map(|_| ())
    }

    /// `prefill_begin` with the scheduler's checkpoint hints (see
    /// `ChunkedPrefill::hints`): page-aligned here, kept strictly inside the
    /// prompt, deduped - the scheduler passes raw divergence points. Returns
    /// the resume point: how many leading tokens came out of the prefix
    /// cache, which is what tells the scheduler whether this prompt still
    /// has uncached work inside a span it shares with others.
    pub fn prefill_begin_hinted(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
        hints: &[usize],
    ) -> Result<usize, GpuModelError> {
        let mut hints: Vec<usize> = hints
            .iter()
            .map(|&c| c / BLOCK_TOKENS * BLOCK_TOKENS)
            .filter(|&c| c > 0 && c < tokens.len())
            .collect();
        hints.sort_unstable();
        hints.dedup();
        let max_batch = self
            .batch
            .as_ref()
            .map(|b| b.max_batch)
            .ok_or(GpuModelError::BatchDisabled)?;
        if slot >= max_batch {
            return Err(GpuModelError::BatchTooLarge {
                got: slot + 1,
                max: max_batch,
            });
        }
        if tokens.is_empty() || tokens.len() > self.max_ctx {
            return Err(GpuModelError::ContextExceeded {
                got: tokens.len(),
                max: self.max_ctx,
            });
        }
        if self.chunked.len() >= crate::service::max_chunks_inflight() {
            return Err(GpuModelError::Unsupported(
                "chunked prefill queue is full".into(),
            ));
        }
        if self.chunked.iter().any(|c| c.slot == slot) {
            return Err(GpuModelError::Unsupported(
                "slot already has a chunked prefill in flight".into(),
            ));
        }
        // match+restore the cached prefix (sets mrope_delta, last_reused, seeds the
        // slot's KV table + DeltaNet state). `done` = resume position: the fused
        // tick covers only tokens[done..], attending the adopted prefix in KV.
        let done = self.prefix_resume_begin(slot, &tokens)?;
        self.chunked.push(ChunkedPrefill {
            slot,
            tokens,
            done,
            hints,
        });
        Ok(done)
    }

    /// Drop slot `slot`'s queued chunked prefill (the client hung up). Refused
    /// (false) while a unified span is in flight - shares reference chunk
    /// indices, so mutating `chunked` under one would advance the wrong
    /// prompt; the scheduler simply retries next tick. Slot-side KV/state
    /// needs no teardown here: an inactive slot is re-cleared and regrown at
    /// its next prefill (same as normal completion).
    pub fn prefill_abort(&mut self, slot: usize) -> bool {
        if self.unified_inflight.is_some() {
            return false;
        }
        let before = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.len() != before
    }

    /// Legacy (PADDOCK_NO_UNIFIED) prefill: run in-flight prompts this tick, FIFO
    /// (earliest admission first -> its first token sooner), up to a ~`budget` row
    /// soft cap but always at least one (so a prompt longer than the cap still
    /// makes progress and never hangs). Each prompt's divergent tail is prefilled
    /// whole from `done` (the resume point `prefill_begin` restored) via
    /// `prefill_slot_tail` - bit-identical to a standalone prefill of the same
    /// tokens. This staggers the cohort's first tokens over several ticks and lets
    /// decode run between them (the TTFT win) without splitting a prompt mid-tick.
    /// (The unified tick does the SOTA thing - intra-prompt spans fused into the
    /// decode forward; this whole-tail path is only the opt-out fallback.)
    /// Returns `(slot, last-token logits, prompt rows)` per finished prompt.
    fn advance_chunks(
        &mut self,
        budget: usize,
    ) -> Result<Vec<(usize, Vec<f32>, usize)>, GpuModelError> {
        let cap = budget.min(self.prefill_chunk_rows);
        // With a DFlash drafter armed the soft cap is HARD past prompt 0. The
        // drafter's fusion accumulator is sized for chunk_tick_rows() prefill
        // rows (+ the verify/decode share), and the soft cap's overshoot -
        // admit the next tail whenever `used < cap` - put 8 x 1075-token
        // tails (8601 rows) in one tick against an 8544-row cap. tap_band
        // then marks the state stale and the append wipes every ring in the
        // tick, decoding slots included - which costs a wide leg roughly
        // three quarters of its throughput. Deferring that
        // one tail by a tick costs nothing the TTFT stagger doesn't already
        // pay; prompt 0 keeps its always-admitted guarantee (a single tail
        // longer than the cap still wipes, and now says so in the log).
        let hard = self.dflash_armed().then(chunk_tick_rows);
        // Lever 2: BATCH this tick's prefill spans into one weight-amortized pass
        // (PADDOCK_QWEN35_CHUNK_BATCH, DEFAULT-ON via apply_default_stack;
        // kill: PADDOCK_NO_CHUNK_BATCH - the kill pins the serial path, which
        // prefills the
        // chunked prompts one at a time -> re-reads the 256-expert MoE per prompt,
        // which tanks decode throughput on every mixed tick. Batching keeps
        // the chunk path's low TTFT while restoring decode.
        if std::env::var_os("PADDOCK_QWEN35_CHUNK_BATCH").is_some() {
            return self.advance_chunks_batched(cap, hard);
        }
        let mut finished = Vec::new();
        let mut used = 0usize;
        while !self.chunked.is_empty() {
            // budget the tick, but guarantee forward progress on prompt 0
            if used > 0 && used >= cap {
                break;
            }
            let (slot, toks, done) = {
                let c = &self.chunked[0];
                (c.slot, c.tokens.clone(), c.done)
            };
            let len = toks.len();
            if used > 0 && hard.is_some_and(|h| used + len > h) {
                break; // the drafter's fusion cap: this tail goes next tick
            }
            // Resume from `done` (prefill_begin already restored the prefix state
            // + KV); prefill only the tail, snapshot + insert. Bit-identical to an
            // un-chunked prefill - chunked prefill changes only when prompts run.
            let logits = self.prefill_slot_tail(slot, &toks, done)?;
            finished.push((slot, logits, len));
            self.chunked.remove(0);
            used += len;
        }
        Ok(finished)
    }

    /// Lever 2 batched tick prefill: pop fresh (done==0) chunked prompts up to the
    /// row `cap` and prefill them in one weight-amortized pass (`prefill_batch_pass`),
    /// instead of the serial per-prompt loop. Resumed (done>0, prefix-cache-hit)
    /// prompts still go serial (they need the resume + snapshot path; rare on the
    /// salted concurrency workload this targets). v1: batched prompts are not
    /// re-inserted into the prefix cache (matches `prefill_batch_pass`); caching in
    /// the batched path is the same v2 follow-up as Lever 1. NOTE: this
    /// lane is not reachable under the default unified
    /// stack (verified live - the kill-switch A/B changed nothing); if it is ever
    /// re-defaulted, the missing insert means every prompt through it loses
    /// multi-turn prefix reuse, so the v2 follow-up becomes mandatory then.
    fn advance_chunks_batched(
        &mut self,
        cap: usize,
        hard: Option<usize>,
    ) -> Result<Vec<(usize, Vec<f32>, usize)>, GpuModelError> {
        let mut finished = Vec::new();
        // collect this tick's spans front-to-back up to the row cap. Fresh
        // spans are PEEKED (removed only after their pass succeeds) so a
        // fallible pass can't lose queue entries - the gemma4 destructive
        // take zombied its chunking slots on PoolExhausted.
        // Resumed spans keep their commit-after-success serial removal, so
        // after the walk the peeked fresh spans sit at queue positions
        // [0, batch.len()).
        let mut batch: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut lens: Vec<usize> = Vec::new();
        let mut used = 0usize;
        let mut i = 0usize;
        while i < self.chunked.len() {
            if used > 0 && used >= cap {
                break;
            }
            let (slot, toks, done) = {
                let c = &self.chunked[i];
                (c.slot, c.tokens.clone(), c.done)
            };
            let len = toks.len();
            if used > 0 && hard.is_some_and(|h| used + len > h) {
                break; // the drafter's fusion cap: this tail goes next tick
            }
            if done != 0 {
                // resumed span: keep the serial resume+snapshot path (correctness
                // over amortization for the rare cache-hit-mid-prompt case).
                let logits = self.prefill_slot_tail(slot, &toks, done)?;
                finished.push((slot, logits, len));
                self.chunked.remove(i);
                used += len;
                continue;
            }
            batch.push((slot, toks));
            lens.push(len);
            i += 1;
            used += len;
        }
        if !batch.is_empty() {
            // A lone admission rides the group pass when the f8t chunk arm
            // would engage there: the serial tail ran the q8/f8bs routes at
            // ~18.6 ms per c32 admission where the group pass does the same
            // rows on the decode tile plane (tc5r at 65+). Costs the v1
            // prefix-cache insert, like every grouped span here.
            let f8t_solo = batch.len() == 1
                && lens[0] <= f8t_chunk_rmax()
                && f8t_unified_on()
                && self.bs_f8t_attn.iter().any(Option::is_some);
            if batch.len() == 1 && !f8t_solo {
                // one fresh span: serial (its own tail already amortizes its
                // read) - and that path publishes on its own.
                let (slot, toks) = &batch[0];
                let logits = self.prefill_slot_tail(*slot, toks, 0)?;
                finished.push((*slot, logits, lens[0]));
                self.chunked.remove(0);
            } else {
                // Make the wave PUBLISH. `prefill_batch_pass`
                // prefills "fresh whole prompts, KV pos 0..take" and never
                // touches the radix, so every prompt riding it loses prefix
                // reuse - the v1 gap documented above. That gap went live when
                // PADDOCK_QWEN35_CHUNK_BATCH became default-on, which makes
                // the older "not reachable" note stale: instrumented, 96 of a
                // 112-request c16 leg ride this pass and their
                // `prefix_resume_begin` reads `ckpt None` forever (only the 16
                // warmup requests, which take the unified span path, resume).
                //
                // Teaching the pass per-share START positions would pull in the
                // resumed-span machinery `unified_launch_core` already owns.
                // Instead stop the wave at each prompt's DeltaNet checkpoint
                // boundary: after a pass over [0, cut) the slot's state sits
                // exactly on the boundary, so it snapshots exactly the way
                // `prefill_slot_tail_paged` does. The <=16-row remainder keeps
                // `done = cut` in the queue and rides the next tick, where the
                // unified tick fuses it with decode rather than paying a pass.
                // Splitting is bit-exact by the same argument the serial tail
                // makes: chunk splits don't change the math.
                //
                // One cut, not `ckpt_cuts`' two - a second stop costs a second
                // wave pass, and one resumable boundary is what a repeated
                // prompt needs. `>= MIN_CACHE_PREFIX` because a shallower
                // checkpoint is one `prefix_resume_begin` would refuse anyway.
                //
                // OPT-IN, DEFAULT off - this arm is MEASURED A NET LOSS and is
                // kept only as scaffolding for the real fix (below). Over 3
                // legs at c16 it costs both TTFT and throughput badly.
                // It works - legs 2-3 resume all 112 requests at `ckpt Some(176)`
                // and that saves ~126 ms of prefill - but splitting leaves a
                // <=16-row tail per prompt, and those tails are picked up by the
                // `done != 0` branch above, i.e. SERIALLY: 16 prompts x a full
                // weight pass ~= +850 ms, which swamps the win.
                //
                // So the split is the wrong shape. The fix has to leave no tail:
                // stage the DeltaNet state mid-pass at the boundary (the
                // `d_ckpt_stage` pattern `unified_finish_core` already uses for
                // its fused share, via `snapshot_staged_pool`) so the wave still
                // runs each prompt whole. That fights the wave's packed varlen
                // walk (one stage1+walk pair per layer), which is why it is a
                // separate piece of work rather than a widening of this one.
                // The alternative - per-share START positions so resumed spans
                // can ride the wave in a batch - is the `v2 follow-up` named
                // above and is the same machinery `unified_launch_core` owns.
                // NOTE: with publishing live the checkpoint pool BINDS (96
                // publishes/leg vs a 94-slot pool -> `attach_state` returns None
                // and nothing resumes); pool 256 is what made legs 2-3 resume.
                // That is why the earlier pool A/B read NULL: nothing published.
                let publish = self
                    .batch
                    .as_ref()
                    .is_some_and(|b| b.paged_prefix.is_some())
                    && paddock_models::dev_var_os!("PADDOCK_WAVE_PUBLISH").is_some();
                let cuts: Vec<Option<usize>> = lens
                    .iter()
                    .map(|&len| {
                        let c = ckpt_pos(len);
                        (publish && c >= min_cache_prefix() && c < len).then_some(c)
                    })
                    .collect();
                for (e, cut) in batch.iter_mut().zip(&cuts) {
                    if let Some(c) = *cut {
                        e.1.truncate(c);
                    }
                }
                let mut out: Vec<Vec<f32>> = vec![Vec::new(); batch.len()];
                let group: Vec<usize> = (0..batch.len()).collect();
                self.prefill_batch_pass(&batch, &group, &mut out, None)?;
                for i in 0..batch.len() {
                    let slot = batch[i].0;
                    match cuts[i] {
                        // ran whole: its last-token logits finish the prompt
                        None => finished.push((slot, std::mem::take(&mut out[i]), lens[i])),
                        // ran to the boundary: publish it, tail rides next tick
                        Some(c) => {
                            let toks = std::mem::take(&mut batch[i].1);
                            self.wave_publish_ckpt(slot, &toks, c)?;
                        }
                    }
                }
                // Finished prompts leave the queue; the split ones keep their
                // tail. Descending so the removals don't shift later indices.
                for i in (0..batch.len()).rev() {
                    match cuts[i] {
                        Some(c) => self.chunked[i].done = c,
                        None => {
                            self.chunked.remove(i);
                        }
                    }
                }
            }
        }
        Ok(finished)
    }

    /// P80: publish a wave-prefilled prefix into the radix - insert its full KV
    /// pages and attach the DeltaNet checkpoint the slot's state is sitting on.
    ///
    /// `toks` is the prompt truncated to `cut`, which is both the key
    /// `attach_state` walks and the exact row the wave stopped at, so the state
    /// it snapshots belongs to that key. `cut` is a page multiple by
    /// construction (`ckpt_pos`), which `attach_state` requires.
    ///
    /// A `None` from `attach_state` (state pool full, or a node that already
    /// carries a checkpoint) is not an error: the pages stay cached and the
    /// prompt simply has no resume point, exactly as before this existed.
    fn wave_publish_ckpt(
        &mut self,
        slot: usize,
        toks: &[u32],
        cut: usize,
    ) -> Result<(), GpuModelError> {
        let nb = cut / BLOCK_TOKENS;
        if nb == 0 {
            return Ok(());
        }
        let blocks: Vec<u32> =
            self.batch.as_ref().expect("batch enabled").tables[slot].blocks()[..nb].to_vec();
        let idx = {
            let bs = self.batch.as_mut().expect("batch enabled");
            let pool = bs.pool.as_mut().expect("prefix cache implies pool");
            let radix = bs.paged_prefix.as_mut().expect("prefix cache built");
            radix.insert(toks, &blocks, pool);
            radix.attach_state(toks, cut)
        };
        if let Some(idx) = idx {
            self.snapshot_paged_state(slot, idx)?;
            self.record_mtp_cover(idx, slot, cut, toks);
            self.record_dflash_cover(idx, slot, cut, toks);
        }
        if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
            tracing::info!("qwen35-wave-publish: slot {slot} cut {cut} ckpt {idx:?}");
        }
        Ok(())
    }

    /// One MIXED continuous-batching tick (device-sampled): advance the in-flight
    /// chunked prefills by `budget` rows, then decode every live row. Two
    /// weight-amortized passes (chunk span(s) + a slot-explicit decode) rather
    /// than gpt-oss's single fused pass - it already unfreezes the streams and
    /// staggers first tokens out of the cohort (the whole c32/pf8 TTFT cost);
    /// fusing the two into one pass is the follow-up amortization step.
    /// `decodes[i] = (slot, token, pos)`; returns the decode rows' sampled step
    /// plus `(slot, last-logits, rows)` per chunk that finished this tick.
    pub fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        _fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        use crate::generator::{FinishSample, SampledStep};
        assert_eq!(plans.len(), decodes.len(), "one plan per decode row");
        // two-forward fallback tick: prefill keeps the classic logits readback
        let finished = self
            .advance_chunks(budget)?
            .into_iter()
            .map(|(k, l, r)| (k, FinishSample::Logits(l), r))
            .collect();
        let step = if decodes.is_empty() {
            SampledStep {
                ids: Vec::new(),
                host_rows: Vec::new(),
            }
        } else {
            let toks: Vec<u32> = decodes.iter().map(|&(_, t, _)| t).collect();
            let pos: Vec<u32> = decodes.iter().map(|&(_, _, p)| p).collect();
            let slots: Vec<u32> = decodes.iter().map(|&(k, _, _)| k as u32).collect();
            self.batch_sampled_impl(&toks, &pos, Some(&slots), plans)?
        };
        Ok((step, finished))
    }

    /// True unified prefill+decode tick. One weight-
    /// amortized forward over a flat `R = B + L` batch: `B` live decode rows
    /// (q_len 1, at their slots) followed by `L` rows of one fresh prompt popped
    /// from the chunked queue. The expensive shared path (norms, all projection
    /// GEMMs, MoE, alpha/beta) reads each weight once for both populations -
    /// vs the "mixed" tick's two separate forwards (`advance_chunks` whole-prompt
    /// prefill + a decode pass) that read the weights twice and freeze decode.
    ///
    /// Drop-in for `forward_mixed_sampled`: same `(step, finished)` contract.
    /// Shared ops go through the PREFILL helpers over `R` (so prefill rows keep
    /// their exact reference numeric class - protects the DeltaNet state seeding;
    /// decode rows take that class for this step, greedy-robust). Attention is a
    /// single decode-class call over `R` (each row attends its slot's KV[0..=pos]
    /// = full history for decode, causal-within-prompt for prefill). Only conv +
    /// the DeltaNet recurrence split by sub-range (ragged batch×seqlen). Runs
    /// EAGER (variable fused shape -> no captured graph). See
    pub fn forward_unified_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        use crate::generator::SampledStep;
        assert_eq!(plans.len(), decodes.len(), "one plan per decode row");
        // Nothing queued to prefill -> this is a plain decode tick.
        if self.chunked.is_empty() {
            let step = if decodes.is_empty() {
                SampledStep {
                    ids: Vec::new(),
                    host_rows: Vec::new(),
                }
            } else {
                let toks: Vec<u32> = decodes.iter().map(|&(_, t, _)| t).collect();
                let pos: Vec<u32> = decodes.iter().map(|&(_, _, p)| p).collect();
                let slots: Vec<u32> = decodes.iter().map(|&(k, _, _)| k as u32).collect();
                self.batch_sampled_impl(&toks, &pos, Some(&slots), plans)?
            };
            return Ok((step, Vec::new()));
        }
        // Launch + immediate finish: the fused mixed tick, bit-identical to
        // the pre-split single body (the split exists so the overlap
        // scheduler can pump decode-lane ticks between the two halves).
        self.unified_launch_core(decodes, budget, plans, fin_plans)?;
        self.dflash_flush_pending_append()?;
        self.unified_finish_core()
    }

    /// Launch a prefill-only unified span (no decode rows) without waiting
    /// for it: the layer walk and finisher sampling are enqueued on the main
    /// lane and an event marks completion. Returns false (launching nothing)
    /// when the chunk queue is empty. While the span is in flight the
    /// overlap scheduler pumps decode-pipe ticks on the decode lane -
    /// disjoint slots, and finishers sample in the span-side d_fin_*
    /// buffers, so nothing touches the decode graph's rows. Admission may
    /// APPEND to the chunk queue during the flight; `unified_span_finish`
    /// must run before any other unified/prefill/mixed call.
    pub fn unified_span_launch(
        &mut self,
        budget: usize,
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<bool, GpuModelError> {
        if self.chunked.is_empty() {
            return Ok(false);
        }
        self.unified_launch_core(&[], budget, &[], fin_plans)?;
        self.dflash_flush_pending_append()?;
        Ok(true)
    }

    /// Non-blocking: has the in-flight span's GPU work (incl. finisher
    /// sampling) completed? True when no span is in flight.
    pub fn unified_span_done(&self) -> bool {
        self.unified_inflight
            .as_ref()
            .is_none_or(|s| self.exec.event_done(&s.ev))
    }

    /// Complete the in-flight span: drain the main lane, read finisher ids,
    /// run the warm/prefix hooks, advance the chunk queue. Returns the
    /// finished prompts.
    pub fn unified_span_finish(
        &mut self,
    ) -> Result<Vec<(usize, crate::generator::FinishSample, usize)>, GpuModelError> {
        let (_step, finished) = self.unified_finish_core()?;
        Ok(finished)
    }

    /// Which shares are FUSED CKPT shares (their tail share of the same
    /// chunk follows immediately in the same tick), and which d_ckpt_stage
    /// blob each owns. Single source of truth: the launch-side per-layer
    /// stage copies and the finish-side staged snapshot both recompute this
    /// from the shares list (deterministic - adjacency + the 2-blob cap).
    fn fused_stage_map(
        shares: &[(usize, usize, usize, usize, bool, Vec<u32>)],
    ) -> Vec<Option<usize>> {
        let mut map = vec![None; shares.len()];
        let mut stg = 0usize;
        for i in 0..shares.len().saturating_sub(1) {
            if shares[i].0 == shares[i + 1].0 && stg < 2 {
                map[i] = Some(stg);
                stg += 1;
            }
        }
        map
    }

    fn unified_launch_core(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<(), GpuModelError> {
        use crate::generator::{FinishSample, RowSample};
        use crate::sampler::DevicePlan;
        assert!(
            self.unified_inflight.is_none(),
            "unified span already in flight"
        );

        // Build prefill SHARES from the queue (FIFO) up to a token budget. Each
        // share advances its chunk by a SPAN `tokens[done..done+take]`: several
        // whole prompts fuse into one tick (multi-prompt), and a prompt longer
        // than the budget chunks across ticks (intra-prompt) with its DeltaNet
        // state + conv window persisting in the slot's buffers between ticks.
        // Share tuple: (chunk index, slot, done, take, finishing, span tokens).
        let b = decodes.len();
        // light prefill share per tick (see unified_prefill_rows) so decode stays
        // saturated; the scheduler's larger `budget` is an upper bound only.
        // 128-alignment of (b + span) to the lin-GEMM M-tile: FALSIFIED as a
        // budget trim (kept opt-in PADDOCK_ALIGN_TICK for
        // re-probes). The M-staircase is real (midm_bench: 544 rows costs
        // what 640 costs), but trimming the cap MIS-FACTORIZES prompts -
        // at r512 a 2048-token prompt became 4x480+128 (five spans incl. a
        // near-empty tail tick): c32 327.6 -> 260.8. At the r2048 default
        // TAIL_SLOP absorption swallows the whole prompt into the trimmed
        // cap and rebuilds the misaligned 2080-row tick: measured +0.6%
        // (noise). The intrinsic pad at (prompt 2048 + riders 32) is only
        // reachable kernel-side (M-tail tile), not by budget arithmetic.
        let base = budget.min(unified_prefill_rows()).max(1);
        // f8t band clamp: with decode riders aboard AND a shallow chunk
        // queue (the steady-churn case: ~1 admission per 4 decode ticks on
        // the c32 board), cap the tick at the tc5p/tc5q 64-row band so the
        // whole mixed tick rides the decode tile plane (see the f8t arms in
        // the layer loop) instead of the sub-w8_min int8-MMA ladder. A deep
        // queue (cold burst, warmup: many prompts at once) keeps the fat
        // span - the W8 route is 1.27-1.85x best-q8 at M >= 512 and drains
        // a 32-prompt burst in 2 fat ticks where 64-row spans need ~25.
        // Pure-prefill ticks (b=0, the c1 TTFT path) are never clamped.
        // This is the per-tick queue-depth choice the process-wide
        // PADDOCK_UNIFIED election note asks for (qwen35/load.rs).
        // MEASURED off (default 0 = never clamp), kept as the re-probe knob:
        // splitting a prompt into 64-b-row spans pays the span machinery
        // (DeltaNet base-0 copies, extra tick overhead) once per span, and
        // no concurrency won -- clamping always loses at every width (three
        // spans per admission for 4 riders; and when ISL=OSL, prefill demand
        // equals decode demand, so every tick goes mixed at 18 ms against
        // 11.7 ms pipe ticks). The f8t ARM below is
        // the part that pays: ticks that are naturally r <= 64 ride the
        // decode tile plane instead of the int8-MMA ladder (42.6 -> 18 ms).
        let queued_rows: usize = self.chunked.iter().map(|c| c.tokens.len() - c.done).sum();
        let f8t_qmax = {
            static QM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *QM.get_or_init(|| {
                paddock_models::dev_var!("PADDOCK_F8T_UNIFIED_QMAX")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            })
        };
        let base = if b > 0
            && b < 64
            && queued_rows <= f8t_qmax
            && f8t_unified_on()
            && self.bs_f8t_attn.iter().any(Option::is_some)
        {
            base.min(64 - b)
        } else {
            base
        };
        static ALIGN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let align =
            *ALIGN.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_ALIGN_TICK").is_some());
        let cap = if align && base + b > 128 {
            (((base + b) & !127usize) - b).max(1)
        } else {
            base
        };
        let mut shares: Vec<(usize, usize, usize, usize, bool, Vec<u32>)> = Vec::new();
        let mut room = cap;
        let mut fuse_stages = 0usize;
        for (idx, ch) in self.chunked.iter().enumerate() {
            if room == 0 {
                break;
            }
            let left = ch.tokens.len() - ch.done;
            let mut take = left.min(room);
            // Land a span exactly on this prompt's DeltaNet checkpoint
            // boundaries (the last two full pages, see ckpt_cuts) so the state
            // can be snapshotted there for prefix reuse - otherwise a span
            // crossing one leaves state past the boundary and there's no
            // resumable checkpoint. See the advance loop below.
            // PADDOCK_CKPT_ABSORB=1 (measurement arm):
            // skip the checkpoint cut entirely. The cut lands the DN state
            // snapshot on a page boundary for prefix reuse, but it makes
            // every non-page-aligned prompt pay a SECOND full weight pass
            // for its <=127-row tail - at c1 the 11-row tail tick ran the
            // sub-w8_min Q8 class for ~45 ms of the 212 ms pass.
            // Absorbing trades the resumable state snapshot for
            // one pass; the real fix is fusing both shares into one tick
            // with a mid-tick staged snapshot (d_ckpt_stage pattern).
            static ABSORB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let absorb = *ABSORB
                .get_or_init(|| paddock_models::dev_var_os!("PADDOCK_CKPT_ABSORB").is_some());
            // Both checkpoint boundaries (see ckpt_cuts): land a span on each so
            // the state can be snapshotted there - the next-turn resume needs
            // the second-to-last one whenever the trailing partial page is
            // shorter than the re-render's divergent generation header.
            let step = self.tier_ckpt_step();
            let cuts: Vec<usize> = if absorb {
                Vec::new()
            } else {
                super::chunk_cuts(ch, step)
            };
            if let Some(&b) = cuts
                .iter()
                .find(|&&b| b > 0 && ch.done < b && ch.done + take > b)
            {
                take = b - ch.done;
            } else if take < left
                && left <= room + TAIL_SLOP
                && cuts
                    .iter()
                    .all(|&b| b == 0 || ch.done >= b || ch.done + left <= b)
            {
                // Absorb a small tail instead of leaving it to ride a whole
                // extra tick: a ~2056-token prompt chunked 2048+8 paid the full
                // layer-walk fixed cost for 8 rows (the contention profile's
                // chunk-tail finding). Overdraw is bounded: room
                // saturates to 0 right after, so a tick is at most cap+TAIL_SLOP
                // rows and only one share can extend. The extension must never
                // cross a DeltaNet checkpoint boundary the snap would have
                // landed on (third condition) - that would silently lose the
                // resumable state snapshot prefix reuse depends on.
                take = left;
            }
            let finishing = ch.done + take == ch.tokens.len();
            let cut_at_ckpt = !finishing && cuts.iter().any(|&b| b > 0 && ch.done + take == b);
            shares.push((
                idx,
                ch.slot,
                ch.done,
                take,
                finishing,
                ch.tokens[ch.done..ch.done + take].to_vec(),
            ));
            room = room.saturating_sub(take);
            // FUSED CKPT TAIL: a share cut at the DN
            // checkpoint boundary used to leave its <=127-row tail to its own
            // tick - a second full weight pass in the sub-w8_min class
            // (~45 ms at c1 for eleven rows). Emit the tail as a SECOND
            // share of the same chunk in this tick: the per-layer share
            // loops sequence same-slot shares correctly (the tail's layer-L
            // recurrence reads the state the ckpt share's layer-L scan just
            // wrote; KV is appended for all rows before per-share
            // attention), and the boundary state/window are staged to
            // d_ckpt_stage between the two shares per layer so the prefix
            // checkpoint survives (see the stage copies in the walk and
            // snapshot_staged_pool at finish). Capped at the two stage
            // blobs; overflow tails keep the old two-tick path.
            // PADDOCK_NO_CKPT_FUSE reverts.
            static NO_FUSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let fuse = !*NO_FUSE
                .get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_CKPT_FUSE").is_some());
            // With two boundaries a prompt can land on the first (b1-16) and
            // still have the second (b1) plus the partial-page tail ahead, so
            // the fusion walks: each fused share runs to the next boundary or
            // the prompt end, consuming one stage blob per seam. A prompt whose
            // walk is cut short by room / the stage cap simply lands its last
            // share on a boundary and the next tick continues from there (the
            // finish-side takes the live snapshot for an unfused landing).
            if fuse && cut_at_ckpt {
                let mut from = ch.done + take;
                while room > 0 && fuse_stages < 2 && from < ch.tokens.len() {
                    let next_stop = cuts
                        .iter()
                        .copied()
                        .find(|&b| b > from)
                        .unwrap_or(ch.tokens.len());
                    let t2 = (next_stop - from).min(room);
                    if t2 == 0 {
                        break;
                    }
                    let landed = from + t2;
                    let fin2 = landed == ch.tokens.len();
                    shares.push((
                        idx,
                        ch.slot,
                        from,
                        t2,
                        fin2,
                        ch.tokens[from..landed].to_vec(),
                    ));
                    room = room.saturating_sub(t2);
                    fuse_stages += 1;
                    from = landed;
                    // keep fusing only when this share landed exactly on another
                    // boundary (a checkpoint seam that needs staging)
                    if !cuts.contains(&landed) {
                        break;
                    }
                }
            }
        }
        let l_total: usize = shares.iter().map(|s| s.3).sum();
        let r = b + l_total;
        // shape probe for the unified-graph key census: one line
        // per mixed/span tick with everything a captured graph would bake.
        if paddock_models::dev_var_os!("PADDOCK_UNIFIED_SHAPE_LOG").is_some() {
            let meta: Vec<(usize, usize, usize, bool)> =
                shares.iter().map(|s| (s.1, s.2, s.3, s.4)).collect();
            eprintln!("[ushape] b={b} r={r} shares={meta:?}");
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        // Drain still-queued async work (notably the depth-2 decode pipe from the
        // preceding ticks) before ensure_scratch may reallocate the scratch
        // buffers: a realloc frees the old device buffers, and a not-yet-executed
        // pipe/decode kernel still referencing them would then read freed memory
        // (illegal access). Only the realloc needs the drain - everything runs
        // on one stream, so queued kernels and this tick's serialize on-device
        // regardless. The old unconditional sync idled the GPU for the whole
        // host-side tick assembly every tick: 6.3% of the short-prompt-churn
        // wall (gap profile: ~24 gaps/s of 1-2 ms at wide batch).
        if !matches!(&self.scratch, Some(sc) if sc.cap >= r) {
            self.exec.synchronize()?;
        }
        self.ensure_scratch(r)?;
        // Fresh spans (done==0) zero the slot's DeltaNet state / conv window;
        // resumed spans (done>0, intra-prompt) inherit what the previous tick
        // left in place.
        for &(_, slot, done, _, _, _) in &shares {
            if done == 0 {
                self.zero_slot_state(slot)?;
                self.last_reused[slot] = 0;
            }
        }
        // P5 budget pool: back this tick's writes. Fresh spans clear+grow their
        // slot; resumed spans just grow; each decode row grows to its position.
        if self.batch.as_ref().expect("batch enabled").pool.is_some() {
            for &(_, slot, done, take, _, _) in &shares {
                if done == 0 {
                    let bs = self.batch.as_mut().expect("batch enabled");
                    let pool = bs.pool.as_mut().expect("pool checked above");
                    bs.tables[slot].clear(pool);
                }
                self.ensure_slot_blocks(slot, done + take - 1)?;
            }
            for &(k, _, p) in decodes {
                self.ensure_slot_blocks(k, p as usize)?;
            }
        }

        // host-side flat batch inputs: decode rows, then each share's span rows
        // (at their KV positions `done..done+take`).
        let mut tokens_h = Vec::with_capacity(r);
        let mut pos_h = Vec::with_capacity(r);
        let mut slots_h = Vec::with_capacity(r);
        for &(k, t, p) in decodes {
            tokens_h.push(t);
            pos_h.push(p);
            slots_h.push(k as u32);
        }
        for &(_, slot, done, take, _, ref span) in &shares {
            tokens_h.extend_from_slice(span);
            pos_h.extend((done as u32)..(done + take) as u32);
            slots_h.extend(std::iter::repeat_n(slot as u32, take));
        }
        // mrope: axis-major [4 * r], text -> all four axes = the row's llama pos
        // (decode rows carry their slot's delta; text delta is 0 either way).
        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (state_size, n_k_heads, n_v_heads, conv_k) =
            (self.state_size, self.n_k_heads, self.n_v_heads, self.conv_k);
        let (conv_dim, ff, max_ctx) = (self.conv_dim, self.ff, self.max_ctx);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let vocab = self.vocab;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let km1 = conv_k - 1;
        let state_elems = n_v_heads * state_size * state_size;
        let moe_dims = self.moe;
        let exec = self.exec.clone();
        // fused-ckpt staging geometry: the stage blob mirrors a pool
        // checkpoint (f32-strided per linear layer: [state][window]; bf16
        // states half-fill their region, exactly like d_state_pool), so the
        // finish-side attach is one flat copy (snapshot_staged_pool).
        let win_elems = km1 * conv_dim;
        let st_copy = state_elems * GpuExecutor::dn_state_esz() as usize / 4;
        let lin_ord: Vec<usize> = {
            let mut v = Vec::with_capacity(self.layers.len());
            let mut n = 0usize;
            for l in &self.layers {
                v.push(n);
                if matches!(l.mixer, Mixer::Linear(_)) {
                    n += 1;
                }
            }
            v
        };
        let stage_of = Self::fused_stage_map(&shares);

        // packed multi-span recurrence: one
        // pd_gated_delta_recurrent_v2_packed launch per Linear layer covers
        // the decode rows (len-1 items) AND every serial prefill span - the
        // per-span b=1 v2_at launches this replaces serialized 40us walks
        // after the decode step (31.6K launches/leg, 1.28s on
        // syn_2048x128_c32). Nearly all of those spans are FUSED CKPT TAIL
        // chains (a chunked leader cut at the DN checkpoint boundary + 1-2
        // short same-slot tails with a state stage copy between them), so
        // packing is per CHAIN: the maximal trailing run of serial members
        // of each same-chunk share group rides as one item (their rows are
        // contiguous), and each internal staged seam becomes an in-kernel
        // snapshot into the layer's stage-blob state region - replacing the
        // per-layer copy_region and keeping the serial numeric class
        // bit-exact. Leaders (chunked class) stay in the span loop; the
        // packed launch goes after it so chain state is already advanced.
        // Descriptors (stride 8) are layer-invariant - built + uploaded once
        // per tick. PADDOCK_NO_DN_PACKED reverts to the per-span path.
        let dn_packed_on = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_DN_PACKED").is_none())
        };
        let chunk_min_dn = if exec.sm_count() >= 128 { 384 } else { 128 };
        let no_chunked_dn_pre = paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_DN").is_some();
        // varlen chunked-GDN route gate (GDN formulation band rung
        // 2): hoisted above the packed-item build because it changes what
        // counts as a serial run member - a >= 128-row span rides the
        // varlen chunked launch instead, so its short ckpt tails can still
        // pack. Kill: PADDOCK_NO_DNC_VL.
        static DNC_VL_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let vl_route = *DNC_VL_ON.get_or_init(|| {
            paddock_models::dev_var_os!("PADDOCK_NO_DNC_VL").is_none()
                && std::env::var("PADDOCK_DNC_RS").map(|v| v != "0").unwrap_or(false)
                && std::env::var("PADDOCK_DNC_S1MMA").map(|v| v != "0").unwrap_or(false)
                && paddock_models::dev_var_os!("PADDOCK_DNC_DWB16").is_none()
                // bf16 excluded (falsified); f16 rides the ST walk_rs.
                && paddock_models::dev_var_os!("PADDOCK_DN_STATE_BF16").is_none()
                && paddock_models::dev_var_os!("PADDOCK_NO_DNC_MMA_V2").is_none()
                && paddock_models::dev_var_os!("PADDOCK_DNC_SCAN").is_none()
                && paddock_models::dev_var_os!("PADDOCK_DNC_FLA").is_none()
                && paddock_models::dev_var_os!("PADDOCK_DNC_SPLIT").is_none()
                && paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_DN").is_none()
        });
        let vl_possible = vl_route && exec.has_gated_delta_chunked_rs_vl() && state_size == 128;
        let mut packed_share = vec![false; shares.len()];
        let d_dn_items: Option<(CudaSlice<u32>, usize, bool)> =
            if dn_packed_on && exec.has_dn_recurrent_packed() && state_size == 128 {
                let mut items: Vec<u32> = Vec::with_capacity(8 * (decodes.len() + shares.len()));
                for (i, &(k, _, _)) in decodes.iter().enumerate() {
                    items.extend_from_slice(&[i as u32, 1, k as u32, 0, 0, 0, 0, 0]);
                }
                // row offset of each share in the tick buffers
                let rb_of: Vec<usize> = shares
                    .iter()
                    .scan(decodes.len(), |acc, s| {
                        let r = *acc;
                        *acc += s.3;
                        Some(r)
                    })
                    .collect();
                let goes_chunked = |take: usize| take >= chunk_min_dn && !no_chunked_dn_pre;
                let mut any_snap = false;
                let mut si = 0usize;
                while si < shares.len() {
                    // same-chunk consecutive group [si..=sj]
                    let mut sj = si;
                    while sj + 1 < shares.len() && shares[sj + 1].0 == shares[si].0 {
                        sj += 1;
                    }
                    // maximal trailing run of serial members (>= 128-row spans
                    // stop the run when the varlen chunked route is on - they
                    // ride the VL launch, leaving their short tails to pack)
                    let mut run0 = sj + 1;
                    while run0 > si
                        && shares[run0 - 1].3 > 0
                        && !goes_chunked(shares[run0 - 1].3)
                        && !(vl_possible && shares[run0 - 1].3 >= 128)
                    {
                        run0 -= 1;
                    }
                    if run0 <= sj {
                        // internal staged seams -> in-kernel snapshots (state
                        // after `acc` rows of the item, into blob `stg`)
                        let mut snaps: Vec<(u32, u32)> = Vec::new();
                        let mut acc = 0usize;
                        for x in run0..sj {
                            acc += shares[x].3;
                            if let Some(stg) = stage_of[x] {
                                snaps.push((acc as u32, stg as u32));
                            }
                        }
                        let len: usize = (run0..=sj).map(|x| shares[x].3).sum();
                        // pack only SHORT runs (<=128 rows): a span's walk uses
                        // 48*16*4 = 6144 warps - ~70% of this die's warp slots -
                        // so long walks were already at max concurrency as their
                        // own launches, and folding them in just re-serializes
                        // them as waves inside the packed launch while contending
                        // with the decode items' state sweep (leg-measured: items
                        // ~15us/chain linear, steady tick 231us packed vs 208us
                        // split). Short chains (the 16+16 ckpt tails) hide fully
                        // under the decode band and their stage copies fold
                        // in-kernel - those pack; walk-bound spans stay per-span.
                        if snaps.len() <= 2 && len <= 128 {
                            packed_share[run0..=sj].fill(true);
                            any_snap |= !snaps.is_empty();
                            let (sat, sas) = snaps.first().copied().unwrap_or((0, 0));
                            let (sbt, sbs) = snaps.get(1).copied().unwrap_or((0, 0));
                            items.extend_from_slice(&[
                                rb_of[run0] as u32,
                                len as u32,
                                shares[run0].1 as u32,
                                sat,
                                sas,
                                sbt,
                                sbs,
                                0,
                            ]);
                        }
                    }
                    si = sj + 1;
                }
                if items.is_empty() {
                    None
                } else {
                    let n = items.len() / 8;
                    let mut d = exec.alloc_u32(items.len())?;
                    {
                        let mut s = d.slice_mut(0..items.len());
                        exec.stream.memcpy_htod(&items, &mut s).map_err(drv)?;
                    }
                    Some((d, n, any_snap))
                }
            } else {
                None
            };

        // varlen chunked-GDN items (GDN formulation band): every span
        // >= 128 rows whose slot has no other
        // non-packed share rides one stage1+walk launch pair per layer -
        // the rival's varlen class. This also moves 128..chunk_min-1
        // resumed spans from the serial walk into the chunked class (the
        // class >= chunk_min resumed spans already serve; nonzero-init is
        // parity-tested). Boundary ckpt stages on VL spans are deferred to
        // after the VL launch (post-leader state, before the packed tails
        // advance the slot). Items are layer-invariant: chunk (row0, len)
        // pairs then span (first chunk, rows, state off, out row0) quads,
        // uploaded once. Kill: PADDOCK_NO_DNC_VL -> per-span dispatch.
        let mut vl_share = vec![false; shares.len()];
        let d_dn_vl: Option<(CudaSlice<u32>, usize, usize, usize)> = if vl_possible {
            // a span may ride VL when every other share of its slot is
            // PACKED: the packed launch runs after the VL launch, so a
            // leader's state advance still precedes its ckpt tails. Spans
            // with a boundary stage keep it - the copy is DEFERRED to
            // after the VL launch (post-leader state, before the packed
            // tails), preserving the in-loop ordering contract.
            let mut nonpacked_per_slot: std::collections::HashMap<usize, u32> =
                std::collections::HashMap::new();
            for (si, &(_, slot, _, _, _, _)) in shares.iter().enumerate() {
                if !packed_share[si] {
                    *nonpacked_per_slot.entry(slot).or_insert(0) += 1;
                }
            }
            let mut chunk_items: Vec<u32> = Vec::new();
            let mut span_quads: Vec<u32> = Vec::new();
            let mut rb2 = b;
            for (si, &(_, slot, _done, take, _fin, _)) in shares.iter().enumerate() {
                if take >= 128 && !packed_share[si] && nonpacked_per_slot[&slot] == 1 {
                    vl_share[si] = true;
                    let first_chunk = (chunk_items.len() / 2) as u32;
                    let (mut row, mut left) = (rb2, take);
                    while left > 0 {
                        let clen = left.min(64);
                        chunk_items.extend_from_slice(&[row as u32, clen as u32]);
                        row += clen;
                        left -= clen;
                    }
                    span_quads.extend_from_slice(&[
                        first_chunk,
                        take as u32,
                        (slot * state_elems) as u32,
                        rb2 as u32,
                    ]);
                }
                rb2 += take;
            }
            if span_quads.is_empty() {
                None
            } else {
                let n_chunks = chunk_items.len() / 2;
                let span_off = chunk_items.len();
                chunk_items.extend_from_slice(&span_quads);
                let mut d = exec.alloc_u32(chunk_items.len())?;
                {
                    let mut s = d.slice_mut(0..chunk_items.len());
                    exec.stream.memcpy_htod(&chunk_items, &mut s).map_err(drv)?;
                }
                Some((d, n_chunks, span_off, span_quads.len() / 4))
            }
        } else {
            None
        };

        // per-call base-0 scratch for each prefill span's DeltaNet recurrence
        // (Linear layers only), sized to the widest single span (<= l_total).
        // Eager alloc is fine off the hot pure-decode path.
        // (the dq/dk/dv/g/beta/dattn base-0 staging that used to live here is
        // gone: the DeltaNet _at wrappers run each span in place at its row
        // offset - 6 allocs + 6 copies per span per layer removed)
        // per-span base-0 scratch for the Full-attn split: each prefill span runs
        // the same per-segment attention as prefill_batch_pass (fast WMMA/paged
        // flash for fresh spans, decode-class for resumed spans) - Not one
        // decode-class call over all R (the balloon that made the fused tick a
        // wash). attn kernels have no q-row offset, so copy the span to base 0.
        let mut d_pf_qn = exec.alloc(l_total * q_dim)?;
        let mut d_pf_attn = exec.alloc(l_total * q_dim)?;
        // Per-share attention segment metadata (slot ids + absolute row
        // positions), uploaded once per tick while the stream is still
        // shallow. The per-layer uploads this replaces were pageable htods -
        // each a hidden FULL-STREAM SYNC under the cudarc of the day.
        // STALE MECHANISM NOTE: cudarc 0.19.8's
        // HostSlice for `[T]`/`Vec<T>` returns SyncOnDrop::Sync(None) - a
        // no-op - so pageable htods no longer stream-sync at all (the
        // driver stages the source host-synchronously, ~us for KBs). Keep
        // the once-per-tick shape for launch-count hygiene, but do not
        // build "pinned staging" rungs to remove syncs that no longer
        // exist; verified against the vendored cudarc source.
        let seg_meta: Vec<(CudaSlice<u32>, CudaSlice<u32>)> = {
            let mut v = Vec::with_capacity(shares.len());
            for &(_, slot, done, take, _, _) in &shares {
                let mut dsl = exec.alloc_u32(take)?;
                let mut dps = exec.alloc_u32(take)?;
                let sl = vec![slot as u32; take];
                let posv: Vec<u32> = ((done as u32)..(done + take) as u32).collect();
                {
                    let mut s = dsl.slice_mut(0..take);
                    exec.stream.memcpy_htod(&sl, &mut s).map_err(drv)?;
                }
                {
                    let mut s = dps.slice_mut(0..take);
                    exec.stream.memcpy_htod(&posv, &mut s).map_err(drv)?;
                }
                v.push((dsl, dps));
            }
            v
        };

        let mrope_delta_of = {
            let bs = self.batch.as_ref().expect("batch enabled");
            let mut v = vec![0i64; r];
            for (i, &(k, _, _)) in decodes.iter().enumerate() {
                v[i] = bs.mrope_delta[k];
            }
            v
        };
        let mrope_h: Vec<u32> = (0..4)
            .flat_map(|_| {
                pos_h
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (p as i64 + mrope_delta_of[i]) as u32)
            })
            .collect();

        let sinks = &self.sinks;
        let layers = &self.layers;
        let tok_embd = &self.tok_embd;
        let output = &self.output;
        let out_f8_h = self.out_f8.as_ref();
        let out_norm = &self.out_norm;
        let kv_dtype = self.kv_dtype;
        // pf7 varlen packed prefill attention (AF3): one
        // attn_prefill_f16_paged_vl launch per full-attn layer covers every
        // serving-class prefill span (the same take>24 spans the per-span
        // _at path serves). Stride-4 tile items (q_row0, span_rows,
        // tile_flat_row0, slot) are layer-invariant - built + uploaded once
        // per tick. Tiles never cross spans, so the packed launch is
        // BIT-IDENTICAL to the per-span pf7 launches it replaces (same
        // CTAs, one grid); non-serving spans (the 16-row ckpt tails) keep
        // the staged prefill_attn fallback in the loop.
        // PADDOCK_NO_PF7_VL reverts to per-span launches (PADDOCK_NO_PF7
        // kills both pf7 forms - pack-side the launcher guards it too).
        let attn_vl_on = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                paddock_models::dev_var_os!("PADDOCK_NO_PF7_VL").is_none()
                    && paddock_models::dev_var_os!("PADDOCK_NO_PF7").is_none()
            })
        };
        let attn_g = n_heads.checked_div(n_kv_heads).unwrap_or(0);
        let mut attn_vl_share = vec![false; shares.len()];
        let d_attn_items: Option<(CudaSlice<u32>, usize)> = if attn_vl_on
            && matches!(kv_dtype, KvDtype::Fp8E4m3)
            && head_dim == 256
            && matches!(attn_g, 4 | 6 | 8)
            && max_ctx % 64 == 0
            && pf_attn_dtype_ok(kv_dtype, n_heads, n_kv_heads)
            && exec.has_attn_prefill_f16_paged_vl()
            && self.batch.as_ref().is_some_and(|s| s.paged)
        {
            let mut items: Vec<u32> = Vec::new();
            let mut rb = decodes.len();
            // Every span packs, the <=24-row ones included (AF4a):
            // the short resumed spans - 2 x 16-row BLOCK_TOKENS ckpt tails
            // per admitted prompt - used to fall through to the decode-class
            // split+combine, where each tail row re-scanned its whole ~2k
            // prefix (traced: 8.3K krs launches + paired combines, ~0.4s of
            // the leg's decode band - an engine that never creates this tail
            // population pays none of it).
            // As vl items they cost ~2 extra tiles inside a launch that
            // already fills the machine. Tail rows change numeric class
            // (decode-split -> the pf7 regroup class the rest of the tick's
            // prefill rows already use) - serve-gated like the pf7 landing.
            for (si, &(_, slot, _, take, _, _)) in shares.iter().enumerate() {
                if take > 0 {
                    attn_vl_share[si] = true;
                    let mut t0 = 0usize;
                    while t0 < take * attn_g {
                        items.extend_from_slice(&[rb as u32, take as u32, t0 as u32, slot as u32]);
                        t0 += 64;
                    }
                }
                rb += take;
            }
            if items.is_empty() {
                None
            } else {
                let n_tiles = items.len() / 4;
                let mut d = exec.alloc_u32(items.len())?;
                {
                    let mut s = d.slice_mut(0..items.len());
                    exec.stream.memcpy_htod(&items, &mut s).map_err(drv)?;
                }
                Some((d, n_tiles))
            }
        } else {
            None
        };
        let sm_count = exec.sm_count();
        // b1 fp8 W8A8 dense-proj planes (same gate as the serial/batched
        // prefill paths - the unified tick was Q8-only until recently,
        // one of the reasons it lost to the mixed tick)
        let bs_w8_all = &self.bs_w8;
        let w8_min = w8_min_batch();
        let bs_f8ffn_p = &self.bs_f8ffn;
        let bs_f8row_p = &self.bs_f8row_ffn;
        // f8t unified arm (B200 c32 mixed-tick fix): below w8_min
        // lw8 is None, so an r<=64 unified tick walked every projection on
        // prefill_mm_pre_any's int8-MMA rung -- the one class B200 de-rates
        // (1148 TOPS int8 vs ~7.5P e4m3). Measured on the syn_128x128_c32
        // board: mixed ticks 46.7 ms avg against an 11.0 ms decode tick,
        // 28.8% of leg wall time. These planes are the same tile lane the
        // decode tick rides (tc5p/tc5q band, PPL-gated when elected), so at
        // r <= 64 the whole mixed tick takes it instead.
        let bs_f8t_attn_p = &self.bs_f8t_attn;
        let bs_f8t_ffn_p = &self.bs_f8t_ffn;
        // DFlash: the walk below taps every row; the append itself needs
        // &mut self after the walk's field borrows end, so stash the row
        // mirrors here and let unified_span_launch fuse+append on return.
        if self.dflash.as_ref().is_some_and(|d| d.state.is_some()) && !super::dflash::fuse_off() {
            self.dflash_pending_append = Some((tokens_h.clone(), pos_h.clone(), slots_h.clone()));
        }
        let sc = self.scratch.as_mut().expect("scratch");
        let bs = self.batch.as_mut().expect("batch");

        // device inputs (eager buffers; the fused shape is variable -> no graph)
        let mut d_tokens = exec.alloc_u32(r)?;
        let mut d_pos = exec.alloc_u32(r)?;
        let mut d_slots = exec.alloc_u32(r)?;
        let mut d_mrope = exec.alloc_u32(4 * r)?;
        exec.stream
            .memcpy_htod(&tokens_h, &mut d_tokens)
            .map_err(drv)?;
        exec.stream.memcpy_htod(&pos_h, &mut d_pos).map_err(drv)?;
        exec.stream
            .memcpy_htod(&slots_h, &mut d_slots)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&mrope_h, &mut d_mrope)
            .map_err(drv)?;
        // decode slots occupy the first B entries of d_slots; every multi-slot
        // kernel reads `batch=b` entries from base 0, so passing the full
        // `&d_slots` with count `b` addresses exactly the decode sub-range.
        embed_any(&exec, tok_embd, &d_tokens, &mut sc.d_x, embd, r)?;

        // DFlash feature taps over the whole unified tick (decode rows first,
        // then the prefill shares - the append in unified_finish_core walks
        // the same order).
        let mut dtap = self
            .dflash
            .as_mut()
            .filter(|d| d.state.is_some() && !super::dflash::fuse_off());

        for (li, layer) in layers.iter().enumerate() {
            if let Some(df) = dtap.as_mut()
                && let Some(band) = df.target_layers.iter().position(|&t| t == li)
            {
                super::dflash::tap_band(&exec, df, &sc.d_x, band, embd, r)?;
            }
            let lw8 = bs_w8_all.get(li).filter(|_| r > w8_min);
            if li == 0 && paddock_models::dev_var_os!("PADDOCK_W8_TRACE").is_some() {
                tracing::info!(
                    rows = r,
                    w8_min,
                    planes = bs_w8_all.len(),
                    lw8 = lw8.is_some(),
                    site = "unified_launch_core",
                    "qwen35 prefill W8 consult"
                );
            }
            // f8t unified arm: whole-tick decode-plane ride at r <= 64 (see
            // the bs_f8t_attn_p note above). Disjoint from lw8 by the same
            // w8_min=64 boundary, so each fork below stays two-armed.
            // 2*r <= cap: the DN/FFN arms land [r, 2*ff] in d_ffn_gate
            // (cap*ff f32) - a server whose scratch never grew past ~2r
            // (tiny prompts only) would otherwise overflow it.
            let f8t_u = r <= 64
                && f8t_unified_on()
                && 2 * r <= sc.cap
                && bs_f8t_attn_p.get(li).and_then(|o| o.as_ref()).is_some();
            let keep_xn = matches!(&layer.mixer, Mixer::Linear(_)) || lw8.is_some() || f8t_u;
            prefill_add_norm_quant(
                &exec,
                &mut sc.d_x,
                None,
                false,
                &layer.attn_norm.buf,
                &mut sc.d_xn,
                keep_xn,
                &mut sc.d_pxq,
                &mut sc.d_pxs,
                &mut sc.d_yq,
                embd,
                r,
                eps,
            )?;
            let mut mixer_b16 = false;
            match &layer.mixer {
                Mixer::Full(w) => {
                    // Phase A fused-plane consumer - same gate as the mixed
                    // tick (see there); PADDOCK_NO_QNF kills to the chain.
                    let qnf_caps = bs.paged
                        && head_dim == 256
                        && n_rot == 64
                        && exec.has_q36_qkg_nra_rows()
                        && {
                            static QNF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                            *QNF.get_or_init(|| {
                                paddock_models::dev_var_os!("PADDOCK_NO_QNF").is_none()
                            })
                        };
                    // f8t arm rides only when the fused [q|g|k|v] landing has
                    // its nra_rows consumer - without it there is no split
                    // wired for the fused layout here, so it falls through.
                    let f8t_qkv = bs_f8t_attn_p
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| f8t_u && qnf_caps);
                    let mut f8t_wo_u: Option<&crate::gpu::F8TilePlane> = None;
                    let qnf = (lw8.is_some() || f8t_qkv.is_some()) && qnf_caps;
                    if let Some([qkv_t, wo_t]) = f8t_qkv {
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            embd,
                            r,
                        )?;
                        let nqkv = w.wq.dims()[1] + w.wk.dims()[1] + w.wv.dims()[1];
                        exec.f8t_gemm(
                            qkv_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_qg,
                            embd,
                            nqkv,
                            r,
                        )?;
                        let bt = bs.d_block_tables.as_ref().expect("paged block tables");
                        let bps = bs.blocks_per_slot;
                        exec.q36_qkg_nra_rows(
                            &sc.d_qg,
                            0,
                            nqkv,
                            w.wq.dims()[1],
                            w.wq.dims()[1] + w.wk.dims()[1],
                            &w.q_norm.buf,
                            &w.k_norm.buf,
                            &mut sc.d_qn,
                            &mut sc.d_gate,
                            bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                            bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                            &d_pos,
                            Some(&d_slots),
                            &d_mrope,
                            bt,
                            bps,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            n_rot,
                            eps,
                            yarn,
                            sections,
                            r,
                            kv_dtype,
                        )?;
                        f8t_wo_u = Some(wo_t);
                    } else if let Some(l8) = lw8 {
                        // One e4m3 quant of the normed hidden feeds wq/wk/wv
                        // (same W8 branch as the serial/batched prefill paths)
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        if qnf {
                            let nqkv = w.wq.dims()[1] + w.wk.dims()[1] + w.wv.dims()[1];
                            exec.f8_gemm_w8(
                                l8.wq.as_ref().expect("full-attn W8 qkv plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_qg,
                                w.wq.dims()[0],
                                nqkv,
                                r,
                            )?;
                            let bt = bs.d_block_tables.as_ref().expect("paged block tables");
                            let bps = bs.blocks_per_slot;
                            exec.q36_qkg_nra_rows(
                                &sc.d_qg,
                                0,
                                nqkv,
                                w.wq.dims()[1],
                                w.wq.dims()[1] + w.wk.dims()[1],
                                &w.q_norm.buf,
                                &w.k_norm.buf,
                                &mut sc.d_qn,
                                &mut sc.d_gate,
                                bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                &d_pos,
                                Some(&d_slots),
                                &d_mrope,
                                bt,
                                bps,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                n_rot,
                                eps,
                                yarn,
                                sections,
                                r,
                                kv_dtype,
                            )?;
                        } else {
                            exec.f8_gemm_w8(
                                l8.wq.as_ref().expect("full-attn W8 qkv plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_qg,
                                w.wq.dims()[0],
                                w.wq.dims()[1],
                                r,
                            )?;
                            exec.split_qg(
                                &sc.d_qg,
                                &mut sc.d_q,
                                &mut sc.d_gate,
                                r,
                                n_heads,
                                head_dim,
                            )?;
                            exec.f8_gemm_w8(
                                l8.wq.as_ref().expect("full-attn W8 qkv plane"),
                                w.wq.dims()[1],
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_k,
                                w.wk.dims()[0],
                                w.wk.dims()[1],
                                r,
                            )?;
                            exec.f8_gemm_w8(
                                l8.wq.as_ref().expect("full-attn W8 qkv plane"),
                                w.wq.dims()[1] + w.wk.dims()[1],
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_v,
                                w.wv.dims()[0],
                                w.wv.dims()[1],
                                r,
                            )?;
                        }
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            &w.wq,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_qg,
                            r,
                        )?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.wk,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_k,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.wv,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_v,
                            r,
                        )?;
                    }
                    if !qnf {
                        exec.rmsnorm_batch(
                            &sc.d_q,
                            &w.q_norm.buf,
                            &mut sc.d_qn,
                            head_dim,
                            eps,
                            r * n_heads,
                        )?;
                        exec.rmsnorm_batch(
                            &sc.d_k,
                            &w.k_norm.buf,
                            &mut sc.d_kn,
                            head_dim,
                            eps,
                            r * n_kv_heads,
                        )?;
                        exec.mrope(
                            &mut sc.d_qn,
                            &d_mrope,
                            r,
                            n_heads,
                            head_dim,
                            n_rot,
                            yarn,
                            sections,
                        )?;
                        exec.mrope(
                            &mut sc.d_kn,
                            &d_mrope,
                            r,
                            n_kv_heads,
                            head_dim,
                            n_rot,
                            yarn,
                            sections,
                        )?;
                        if bs.paged {
                            let bt = bs.d_block_tables.as_ref().expect("paged block tables");
                            let bps = bs.blocks_per_slot;
                            exec.kv_append_batch_paged(
                                &sc.d_kn,
                                bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                &d_pos,
                                Some(&d_slots),
                                bt,
                                bps,
                                kv_dim,
                                r,
                                kv_dtype,
                            )?;
                            exec.kv_append_batch_paged(
                                &sc.d_v,
                                bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                &d_pos,
                                Some(&d_slots),
                                bt,
                                bps,
                                kv_dim,
                                r,
                                kv_dtype,
                            )?;
                        } else {
                            exec.kv_append_batch(
                                &sc.d_kn,
                                bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                &d_pos,
                                Some(&d_slots),
                                kv_dim,
                                max_ctx,
                                r,
                                kv_dtype,
                            )?;
                            exec.kv_append_batch(
                                &sc.d_v,
                                bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                &d_pos,
                                Some(&d_slots),
                                kv_dim,
                                max_ctx,
                                r,
                                kv_dtype,
                            )?;
                        }
                    }
                    // ATTENTION SPLIT (the fused-tick correctness+perf fix): the
                    // decode rows [0..b) take the decode-class kernel (each attends
                    // its slot's full KV[0..=pos]); each prefill span runs its own
                    // per-segment attention - the fast WMMA/paged flash prefill_attn
                    // for a fresh span (positions 0..take; bit-identical to
                    // prefill_batch_pass), decode-class for a resumed span (done>0,
                    // arbitrary start positions). This replaces the single
                    // decode-class-over-R call whose per-row scan over each 1k-4k
                    // prefill row ballooned attention +1.8s and made fusion a wash.
                    let paged_arg = bs
                        .d_block_tables
                        .as_ref()
                        .filter(|_| bs.paged)
                        .map(|bt| (bt, bs.blocks_per_slot));
                    let kv_k = bs.kv_k[li].as_ref().expect("full-attn layer KV");
                    let kv_v = bs.kv_v[li].as_ref().expect("full-attn layer KV");
                    if b > 0 {
                        attn_decode_dispatch(
                            &exec,
                            &sc.d_qn,
                            kv_k,
                            kv_v,
                            sinks,
                            &mut sc.d_attn_o,
                            &mut sc.d_attn_ml,
                            &mut sc.d_attn,
                            &d_pos,
                            Some(&d_slots),
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_dim,
                            b,
                            scale,
                            kv_dtype,
                            paged_arg,
                        )?;
                    }
                    // pf7 varlen packed form: every serving-class span in one
                    // launch (bit-identical CTAs to the per-span _at calls
                    // below - see the d_attn_items build). Spans it covers
                    // are skipped in the loop.
                    let mut vl_fired = false;
                    if let (Some((items, n_tiles)), Some((bt, bps))) =
                        (d_attn_items.as_ref(), paged_arg)
                    {
                        exec.attn_prefill_f16_paged_vl(
                            &sc.d_qn,
                            kv_k,
                            kv_v,
                            sinks,
                            &mut sc.d_attn,
                            &d_pos,
                            items,
                            *n_tiles,
                            bt,
                            bps,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            kv_dim,
                            0,
                            scale,
                            kv_dtype,
                        )?;
                        vl_fired = true;
                    }
                    let mut rb = b;
                    for (si, &(_, _, _, take, _, _)) in shares.iter().enumerate() {
                        if vl_fired && attn_vl_share[si] {
                            rb += take;
                            continue;
                        }
                        // Flash prefill for every span, resumed included: the
                        // positions are per-row (done..done+take) and the same
                        // prefill_attn helper serves the serial path's
                        // mid-prompt chunks with arbitrary starts bit-exactly
                        // (prefill_slot_chunk - every 4k prompt today). The old
                        // done==0 gate here fell back to decode-class attention
                        // for resumed spans: 2048 rows scanning full history
                        // row-by-row = 32.7ms/call, 69% of all GPU time in the
                        // unified pf8 profile - the entire reason
                        // unified lost to the mixed tick.
                        if let (Some((bt, bps)), true) = (
                            paged_arg,
                            take > 24
                                && head_dim == 256
                                && max_ctx % 64 == 0
                                && pf_attn_dtype_ok(kv_dtype, n_heads, n_kv_heads)
                                && exec.has_attn_prefill_f16_paged(),
                        ) {
                            // serving class runs in PLACE at row rb: d_pos rows
                            // rb..rb+take are done..done+take and d_slots the
                            // span's slot, so the seg_meta copy and both
                            // rows×q_dim staging copies drop out (bit-identical
                            // - same kernel, offset base pointers).
                            exec.attn_prefill_f16_paged_at(
                                &sc.d_qn,
                                kv_k,
                                kv_v,
                                sinks,
                                &mut sc.d_attn,
                                &d_pos,
                                &d_slots,
                                rb,
                                bt,
                                bps,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                kv_dim,
                                0,
                                take,
                                scale,
                                kv_dtype,
                            )?;
                        } else {
                            // non-serving classes keep the base-0 staging; segment
                            // slot/pos metadata comes from the per-tick seg_meta
                            // upload.
                            exec.copy_region(&sc.d_qn, rb * q_dim, &mut d_pf_qn, 0, take * q_dim)?;
                            prefill_attn(
                                &exec,
                                &d_pf_qn,
                                kv_k,
                                kv_v,
                                sinks,
                                &mut d_pf_attn,
                                &seg_meta[si].1,
                                &seg_meta[si].0,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                max_ctx,
                                kv_dim,
                                take,
                                scale,
                                kv_dtype,
                                paged_arg,
                                Some((&mut sc.d_attn_o, &mut sc.d_attn_ml)),
                            )?;
                            exec.copy_region(
                                &d_pf_attn,
                                0,
                                &mut sc.d_attn,
                                rb * q_dim,
                                take * q_dim,
                            )?;
                        }
                        rb += take;
                    }
                    exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, r * q_dim)?;
                    if let Some(wo_t) = f8t_wo_u {
                        // same plane pair the qkv arm took (one precision
                        // class per layer, mirroring the decode tick)
                        exec.quantize_e4m3_row(
                            &sc.d_attn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            w.wo.dims()[0],
                            r,
                        )?;
                        exec.f8t_gemm(
                            wo_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            w.wo.dims()[0],
                            w.wo.dims()[1],
                            r,
                        )?;
                    } else if let Some(l8) = lw8 {
                        exec.quantize_e4m3(
                            &sc.d_attn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * w.wo.dims()[0],
                        )?;
                        // mixer bf16 seam: wo writes bf16, the post_norm
                        // entry (ABI-247 consumer) reads it back
                        static MO16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let mo16 = *MO16.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        // P74: PADDOCK_QWEN35_O16_TC5 opts the sm_100 tc5
                        // bf16-store route in (the TMA lane is process-killed
                        // there); r >= 256 = the tc5v NT2 floor. MEASURED A
                        // WASH on B200 (tc5s pays ~10% store-poison for bf16
                        // stores - the muse f16 ledger class - and the b16
                        // consumers hold no net win); kept as probe infra.
                        static MO16T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let mo16t = *MO16T.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_QWEN35_O16_TC5").is_some()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        if (mo16 || (mo16t && r >= 256)) && exec.has_f8_o16() {
                            exec.f8_gemm_w8_o16(
                                l8.wo.as_ref().expect("full-attn W8 wo plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                w.wo.dims()[0],
                                w.wo.dims()[1],
                                r,
                            )?;
                            mixer_b16 = true;
                        } else {
                            exec.f8_gemm_w8(
                                l8.wo.as_ref().expect("full-attn W8 wo plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                w.wo.dims()[0],
                                w.wo.dims()[1],
                                r,
                            )?;
                        }
                    } else {
                        prefill_mm_any(
                            &exec,
                            &w.wo,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &sc.d_attn,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
                Mixer::Linear(w) => {
                    // one two-buffer GEMM over the fused plane when
                    // the split route covers it (see the mixed-tick site);
                    // d_z stays untouched until gated_rmsnorm below.
                    let mut dn_fused = false;
                    let mut dn_ab_done = false;
                    let mut f8t_ow_u: Option<&crate::gpu::F8TilePlane> = None;
                    let f8t_dn = bs_f8t_attn_p
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| f8t_u);
                    if let Some([in_t, ow_t]) = f8t_dn {
                        let (nin, nc) = (w.in_qkv.dims()[0], w.in_qkv.dims()[1]);
                        let nz_ = w.gate_w.dims()[1];
                        // the plane's scale length is its out_dim (see the
                        // decode-tick site); +128 marks the alpha||beta fold
                        let tot = in_t.scale.len();
                        dn_ab_done = tot == nc + nz_ + 128;
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            nin,
                            r,
                        )?;
                        // landing: d_ffn_gate ([cap, 2*ff] holds [r, tot];
                        // free until the FFN half of this layer)
                        exec.f8t_gemm(
                            in_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_ffn_gate,
                            nin,
                            tot,
                            r,
                        )?;
                        if dn_ab_done && exec.has_row_slice4() {
                            exec.row_slice4(
                                &sc.d_ffn_gate,
                                tot,
                                r,
                                &mut [
                                    (&mut sc.d_mixed, 0, nc),
                                    (&mut sc.d_z, nc, nz_),
                                    (&mut sc.d_a, nc + nz_, n_v_heads),
                                    (&mut sc.d_b, nc + nz_ + n_v_heads, n_v_heads),
                                ],
                            )?;
                        } else {
                            exec.row_slice(&sc.d_ffn_gate, &mut sc.d_mixed, tot, 0, nc, r)?;
                            exec.row_slice(&sc.d_ffn_gate, &mut sc.d_z, tot, nc, nz_, r)?;
                            if dn_ab_done {
                                exec.row_slice(
                                    &sc.d_ffn_gate,
                                    &mut sc.d_a,
                                    tot,
                                    nc + nz_,
                                    n_v_heads,
                                    r,
                                )?;
                                exec.row_slice(
                                    &sc.d_ffn_gate,
                                    &mut sc.d_b,
                                    tot,
                                    nc + nz_ + n_v_heads,
                                    n_v_heads,
                                    r,
                                )?;
                            }
                        }
                        dn_fused = true;
                        f8t_ow_u = Some(ow_t);
                    } else if let Some(l8) = lw8 {
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        static DNF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        if *DNF
                            .get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_DNF").is_none())
                        {
                            dn_fused = exec.f8_gemm_w8_split(
                                l8.in_qkv.as_ref().expect("DeltaNet W8 in_qkv plane"),
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_mixed,
                                &mut sc.d_z,
                                w.in_qkv.dims()[0],
                                w.in_qkv.dims()[1],
                                w.gate_w.dims()[1],
                                r,
                            )?;
                        }
                        if !dn_fused {
                            exec.f8_gemm_w8(
                                l8.in_qkv.as_ref().expect("DeltaNet W8 in_qkv plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_mixed,
                                w.in_qkv.dims()[0],
                                w.in_qkv.dims()[1],
                                r,
                            )?;
                        }
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            &w.in_qkv,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_mixed,
                            r,
                        )?;
                    }
                    // conv SPLITS: decode rows advance their per-slot windows
                    // (1-step); each prefill span runs a window-extended causal
                    // conv on its slot's window (persisted across intra-prompt
                    // ticks), writing into its rows of d_conv.
                    if b > 0 {
                        exec.conv_step_slots(
                            bs.conv_win[li].as_mut().expect("DeltaNet layer window"),
                            &sc.d_mixed,
                            &w.conv_w.buf,
                            &mut sc.d_conv,
                            &d_slots,
                            b,
                            conv_dim,
                            conv_k,
                        )?;
                    }
                    let mut rb = b;
                    for (si, &(_, slot, done, take, _, _)) in shares.iter().enumerate() {
                        let woff = slot * km1 * conv_dim;
                        if done == 0 {
                            // fresh span: the window is zero (zero_slot_state
                            // above), so conv runs in PLACE at the span's rows
                            // with the kernel's own zero left-pad - parity-
                            // proven byte-identical to the [window ++ span]
                            // ext build. Then commit the trailing k-1
                            // rows as the slot's persistent window (short
                            // spans land in the pre-zeroed tail).
                            if exec.has_conv_silu_qkv() {
                                // fused conv+split+norm straight to q/k/v -
                                // this span's rows skip the split below
                                exec.causal_conv1d_silu_qkv_at(
                                    &sc.d_mixed,
                                    &w.conv_w.buf,
                                    &mut sc.d_dq,
                                    &mut sc.d_dk,
                                    &mut sc.d_dv,
                                    rb,
                                    rb,
                                    take,
                                    n_k_heads,
                                    n_v_heads,
                                    state_size,
                                    conv_k,
                                )?;
                            } else {
                                exec.causal_conv1d_silu_at(
                                    &sc.d_mixed,
                                    &w.conv_w.buf,
                                    &mut sc.d_conv,
                                    rb,
                                    rb,
                                    take,
                                    conv_dim,
                                    conv_k,
                                )?;
                            }
                            let win = bs.conv_win[li].as_mut().expect("DeltaNet layer window");
                            if take >= km1 {
                                exec.copy_region(
                                    &sc.d_mixed,
                                    (rb + take - km1) * conv_dim,
                                    win,
                                    woff,
                                    km1 * conv_dim,
                                )?;
                            } else {
                                exec.copy_region(
                                    &sc.d_mixed,
                                    rb * conv_dim,
                                    win,
                                    woff + (km1 - take) * conv_dim,
                                    take * conv_dim,
                                )?;
                            }
                        } else {
                            // resumed span: the window carries real history -
                            // the ext build stays (a window-aware conv kernel
                            // would be the next cut)
                            let win = bs.conv_win[li].as_ref().expect("DeltaNet layer window");
                            assert!(
                                (km1 + take) * conv_dim <= bs.d_conv_ext.len(),
                                "resumed span {take} rows outgrew the conv ext staging"
                            );
                            exec.copy_region(win, woff, &mut bs.d_conv_ext, 0, km1 * conv_dim)?;
                            exec.copy_region(
                                &sc.d_mixed,
                                rb * conv_dim,
                                &mut bs.d_conv_ext,
                                km1 * conv_dim,
                                take * conv_dim,
                            )?;
                            exec.causal_conv1d_silu(
                                &bs.d_conv_ext,
                                &w.conv_w.buf,
                                &mut bs.d_conv_out,
                                km1 + take,
                                conv_dim,
                                conv_k,
                            )?;
                            exec.copy_region(
                                &bs.d_conv_out,
                                km1 * conv_dim,
                                &mut sc.d_conv,
                                rb * conv_dim,
                                take * conv_dim,
                            )?;
                            let win = bs.conv_win[li].as_mut().expect("DeltaNet layer window");
                            exec.copy_region(
                                &bs.d_conv_ext,
                                take * conv_dim,
                                win,
                                woff,
                                km1 * conv_dim,
                            )?;
                            if exec.has_conv_silu_qkv() {
                                // resumed rows split per-span (the fused fresh
                                // spans above never touch d_conv)
                                exec.deltanet_split_gqa_norm_at(
                                    &sc.d_conv,
                                    &mut sc.d_dq,
                                    &mut sc.d_dk,
                                    &mut sc.d_dv,
                                    rb,
                                    rb,
                                    take,
                                    n_k_heads,
                                    n_v_heads,
                                    state_size,
                                )?;
                            }
                        }
                        if let Some(stg) = stage_of[si] {
                            // fused ckpt share: stage the BOUNDARY window now -
                            // the tail share (next iteration) overwrites it
                            let dst = lin_ord[li] * (state_elems + win_elems) + state_elems;
                            exec.copy_region(
                                bs.conv_win[li].as_ref().expect("DeltaNet layer window"),
                                woff,
                                &mut bs.d_ckpt_stage[stg],
                                dst,
                                win_elems,
                            )?;
                        }
                        rb += take;
                    }
                    if exec.has_conv_silu_qkv() {
                        // decode rows [0, b) still come through d_conv
                        if b > 0 {
                            exec.deltanet_split_gqa_norm_at(
                                &sc.d_conv,
                                &mut sc.d_dq,
                                &mut sc.d_dk,
                                &mut sc.d_dv,
                                0,
                                0,
                                b,
                                n_k_heads,
                                n_v_heads,
                                state_size,
                            )?;
                        }
                    } else {
                        exec.deltanet_split_gqa_norm(
                            &sc.d_conv,
                            &mut sc.d_dq,
                            &mut sc.d_dk,
                            &mut sc.d_dv,
                            r,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                        )?;
                    }
                    // alpha/beta on the exact f32 repacked path (P6b decay rule)
                    if dn_ab_done {
                        // alpha/beta already landed by the fused f8t in-proj
                        // GEMM above (same fold the decode tick takes)
                        exec.delta_gate(
                            &sc.d_a,
                            &sc.d_b,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            r,
                            n_v_heads,
                        )?;
                    } else if let Some(ab) = w
                        .ab_f32
                        .as_ref()
                        .filter(|_| r >= ab_f32_min_rows() || w.alpha_w.is_none())
                    {
                        // x2-v3: one f32-plane decay GEMM (64-col tile, x read once) +
                        // fused-layout gate; same values, tiled order (PPL-gated opt-in)
                        ab_gate(
                            &exec,
                            ab,
                            &sc.d_xn,
                            &mut sc.d_ab,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            r,
                            n_v_heads,
                        )?;
                    } else {
                        if exec.has_q8_0_gemm_repacked_x2() {
                            // fused pair: x staged once for both decay projections
                            // (bit-exact per output vs the two separate calls)
                            exec.q8_0_gemm_repacked_x2(
                                w.alpha_w.as_ref().expect("Q8 alpha (x2 path)"),
                                w.beta_w.as_ref().expect("Q8 beta (x2 path)"),
                                &sc.d_xn,
                                &mut sc.d_a,
                                &mut sc.d_b,
                                r,
                            )?;
                        } else {
                            exec.q8_0_gemm_repacked(
                                w.alpha_w.as_ref().expect("Q8 alpha"),
                                None,
                                &sc.d_xn,
                                &mut sc.d_a,
                                r,
                            )?;
                            exec.q8_0_gemm_repacked(
                                w.beta_w.as_ref().expect("Q8 beta"),
                                None,
                                &sc.d_xn,
                                &mut sc.d_b,
                                r,
                            )?;
                        }
                        exec.delta_gate(
                            &sc.d_a,
                            &sc.d_b,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            r,
                            n_v_heads,
                        )?;
                    }
                    // recurrence SPLITS: decode = multi-slot 1-step in place;
                    // each prefill span = a scan into its slot's state (base 0
                    // is required by the multi-slot decode kernel, so each span's
                    // inputs are copied to base-0 temps and its output back). A
                    // whole prompt in one tick (done==0 && finishing) can take the
                    // parallel chunked scan (zero initial state, matches the
                    // reference class); split/resumed spans use the sequential v2
                    // - bit-exact when a prompt's spans concatenate across ticks
                    // (v2(a) then v2(b) == v2(a++b)), which chunked is not.
                    if d_dn_items.is_none() && b > 0 {
                        exec.gated_delta_recurrent_v2(
                            &sc.d_dq,
                            &sc.d_dk,
                            &sc.d_dv,
                            &sc.d_g,
                            &sc.d_beta,
                            Some(&d_slots),
                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                            0,
                            None,
                            &mut sc.d_dattn,
                            b,
                            1,
                            n_v_heads,
                            state_size,
                        )?;
                    }
                    let chunk_min = if sm_count >= 128 { 384 } else { 128 };
                    let no_chunked_dn =
                        paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_DN").is_some();
                    let mut rb = b;
                    for (si, &(_, slot, _done, take, finishing, _)) in shares.iter().enumerate() {
                        // chunked for every long-enough span, resumed included:
                        // the chunked kernel continues from the slot's state
                        // (nonzero-init is parity-tested), and the serial path
                        // already runs chunked per 2048-chunk with state
                        // continuation - that is the production numeric class.
                        // The old `done==0 && finishing` gate (v2-for-splits,
                        // chasing bit-exactness with a serial class that itself
                        // uses chunked) ran sequential v2 over 2048-token spans:
                        // 371us/layer, the #2 bomb in the unified pf8 profile.
                        // The _at wrappers read/write the span in PLACE at row
                        // rb (bit-identical bytes) - the 6-copy-per-span
                        // staging storm this loop used to run was ~100-135us
                        // of host serialization per layer in the pf8 profile.
                        let _ = finishing;
                        let off = slot * state_elems;
                        if vl_share[si] {
                            // rides the tick-wide varlen chunked launch below
                            rb += take;
                            continue;
                        }
                        if take >= chunk_min && state_size == 128 && !no_chunked_dn {
                            exec.gated_delta_chunked_at(
                                &sc.d_dq,
                                &sc.d_dk,
                                &sc.d_dv,
                                &sc.d_g,
                                &sc.d_beta,
                                bs.recur[li].as_mut().expect("DeltaNet layer state"),
                                off,
                                &mut sc.d_dattn,
                                rb,
                                &mut sc.d_dnc_dw,
                                &mut sc.d_dnc_du,
                                &mut sc.d_dnc_coef,
                                &mut sc.d_dnc_cg,
                                take,
                                n_v_heads,
                                state_size,
                            )?;
                        } else if !packed_share[si] {
                            // packed shares already walked in the one launch above
                            exec.gated_delta_recurrent_v2_at(
                                &sc.d_dq,
                                &sc.d_dk,
                                &sc.d_dv,
                                &sc.d_g,
                                &sc.d_beta,
                                bs.recur[li].as_mut().expect("DeltaNet layer state"),
                                off,
                                &mut sc.d_dattn,
                                rb,
                                take,
                                n_v_heads,
                                state_size,
                            )?;
                        }
                        if let Some(stg) = stage_of[si]
                            && !packed_share[si]
                        {
                            // fused ckpt share: stage the BOUNDARY state
                            // before the tail share's recurrence advances
                            // it (esz-aware: bf16 states half-fill the f32
                            // region). Packed chain members snapshot their
                            // seam IN-KERNEL instead.
                            let dst = lin_ord[li] * (state_elems + win_elems);
                            exec.copy_region(
                                bs.recur[li].as_ref().expect("DeltaNet layer state"),
                                slot * st_copy,
                                &mut bs.d_ckpt_stage[stg],
                                dst,
                                st_copy,
                            )?;
                        }
                        rb += take;
                    }
                    if let Some((d_vl, n_chunks, span_off, n_spans)) = d_dn_vl.as_ref() {
                        // One stage1+walk pair for every varlen-eligible span
                        // (n_tokens is per-span from the items; the total
                        // argument is informational only on this path)
                        exec.gated_delta_chunked_rs_vl(
                            &sc.d_dq,
                            &sc.d_dk,
                            &sc.d_dv,
                            &sc.d_g,
                            &sc.d_beta,
                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                            &mut sc.d_dattn,
                            &mut sc.d_dnc_dw,
                            &mut sc.d_dnc_du,
                            &mut sc.d_dnc_coef,
                            &mut sc.d_dnc_cg,
                            d_vl,
                            *n_chunks,
                            *span_off,
                            *n_spans,
                            rb,
                            n_v_heads,
                            state_size,
                        )?;
                        // deferred ckpt boundary stages for VL leaders:
                        // post-leader state, captured before the packed
                        // launch below advances the slot with the tails -
                        // same ordering contract as the in-loop copy
                        for (si2, &(_, slot2, _, _, _, _)) in shares.iter().enumerate() {
                            if vl_share[si2]
                                && let Some(stg) = stage_of[si2]
                            {
                                let dst = lin_ord[li] * (state_elems + win_elems);
                                exec.copy_region(
                                    bs.recur[li].as_ref().expect("DeltaNet layer state"),
                                    slot2 * st_copy,
                                    &mut bs.d_ckpt_stage[stg],
                                    dst,
                                    st_copy,
                                )?;
                            }
                        }
                    }
                    if let Some((d_items, n_items, any_snap)) = d_dn_items.as_ref() {
                        // One launch after the chunked leaders: decode len-1
                        // items + serial spans/chains (in-kernel seam snaps)
                        let dst = lin_ord[li] * (state_elems + win_elems);
                        let (snap0, snap1) = if *any_snap {
                            let (s0, s1) = bs.d_ckpt_stage.split_at_mut(1);
                            (Some((&mut s0[0], dst)), Some((&mut s1[0], dst)))
                        } else {
                            (None, None)
                        };
                        exec.gated_delta_recurrent_v2_packed(
                            &sc.d_dq,
                            &sc.d_dk,
                            &sc.d_dv,
                            &sc.d_g,
                            &sc.d_beta,
                            d_items,
                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                            &mut sc.d_dattn,
                            snap0,
                            snap1,
                            *n_items,
                            n_v_heads,
                            state_size,
                        )?;
                    }
                    if let Some(l8) = lw8 {
                        // d_pxq still holds the e4m3 xn quant from in_qkv
                        // (alpha/beta read f32 xn directly, nothing clobbers it)
                        if !dn_fused {
                            exec.f8_gemm_w8(
                                l8.in_qkv.as_ref().expect("DeltaNet W8 in_qkv plane"),
                                w.in_qkv.dims()[1],
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_z,
                                w.gate_w.dims()[0],
                                w.gate_w.dims()[1],
                                r,
                            )?;
                        }
                    } else if !dn_fused {
                        prefill_mm_pre_any(
                            &exec,
                            &w.gate_w,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_z,
                            r,
                        )?;
                    }
                    // DN out_proj glue (GDN formulation band):
                    // fused gated-rmsnorm + e4m3 quant on the w8 arm, with
                    // the f32 d_core store skipped - the GEMM's q/scale
                    // planes are the only consumer on this path. Bytes are
                    // bit-identical to the norm + standalone-quantize pair.
                    // f8t out_w arm excluded: it needs the f32 d_core + the
                    // row-quant seam, not the linear e4m3 planes. Without this
                    // guard, an f8t in_qkv tick that ALSO has lw8 (the arms
                    // overlap once w8_min < r, e.g. the shipped w8_min=0) runs
                    // the fused norm, skips the d_core store, and the f8t out
                    // arm below then row-quants a stale d_core -> corrupt out
                    // proj (matches the mixed-tick site's gr_fused).
                    let gr_fused =
                        lw8.is_some() && f8t_ow_u.is_none() && exec.has_gated_rmsnorm_e4m3();
                    if gr_fused {
                        exec.gated_rmsnorm_e4m3(
                            &sc.d_dattn,
                            &sc.d_z,
                            &w.ssm_norm.buf,
                            None,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * n_v_heads,
                            state_size,
                            eps,
                        )?;
                    } else {
                        exec.gated_rmsnorm(
                            &sc.d_dattn,
                            &sc.d_z,
                            &w.ssm_norm.buf,
                            &mut sc.d_core,
                            r * n_v_heads,
                            state_size,
                            eps,
                        )?;
                    }
                    if let Some(ow_t) = f8t_ow_u {
                        // gr_fused is lw8-gated, so d_core holds the gated
                        // norm here; row-quant feeds the tile lane (same
                        // seam as the decode tick's out_w arm)
                        exec.quantize_e4m3_row(
                            &sc.d_core,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            w.out_w.dims()[0],
                            r,
                        )?;
                        exec.f8t_gemm(
                            ow_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            w.out_w.dims()[0],
                            w.out_w.dims()[1],
                            r,
                        )?;
                    } else if let Some(l8) = lw8 {
                        if !gr_fused {
                            exec.quantize_e4m3(
                                &sc.d_core,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                r * w.out_w.dims()[0],
                            )?;
                        }
                        static MO16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let mo16 = *MO16.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        // P74: PADDOCK_QWEN35_O16_TC5 opts the sm_100 tc5
                        // bf16-store route in (the TMA lane is process-killed
                        // there); r >= 256 = the tc5v NT2 floor. MEASURED A
                        // WASH on B200 (tc5s pays ~10% store-poison for bf16
                        // stores - the muse f16 ledger class - and the b16
                        // consumers hold no net win); kept as probe infra.
                        static MO16T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let mo16t = *MO16T.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_QWEN35_O16_TC5").is_some()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        if (mo16 || (mo16t && r >= 256)) && exec.has_f8_o16() {
                            exec.f8_gemm_w8_o16(
                                l8.out_w.as_ref().expect("DeltaNet W8 out_w plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                w.out_w.dims()[0],
                                w.out_w.dims()[1],
                                r,
                            )?;
                            mixer_b16 = true;
                        } else {
                            exec.f8_gemm_w8(
                                l8.out_w.as_ref().expect("DeltaNet W8 out_w plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                w.out_w.dims()[0],
                                w.out_w.dims()[1],
                                r,
                            )?;
                        }
                    } else {
                        prefill_mm_any(
                            &exec,
                            &w.out_w,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &sc.d_core,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
            }
            let mut proj_is_b16 = false;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    // prefill-FFN f8 arm: the W8 prefill class extended to
                    // the FFN, because ~70% of prefill bytes were still
                    // running through int8-mmq. f8_gemm_w8 measures
                    // 1.27-1.85x best-q8 at M >= 512.
                    // Same e4m3 planes the decode lane built; same w8_min gate.
                    // f8t unified arm for the FFN half (see bs_f8t_attn_p
                    // note): same gu|down tile planes the decode tick rides.
                    let f8t_ffn = bs_f8t_ffn_p
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| f8t_u);
                    let f8f = bs_f8ffn_p.get(li).and_then(|o| o.as_ref()).filter(|_| {
                        r > super::f8_ffn_pf_min()
                            && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
                    });
                    let f8r = bs_f8row_p.get(li).and_then(|o| o.as_ref());
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        mixer_b16,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        f8r.is_some() || (f8f.is_some() && f8t_ffn.is_none()),
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    if let Some(p) = f8r {
                        super::ops::ffn_f8row_rows(
                            &exec,
                            p,
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else if let Some([gu_t, dn_t]) = f8t_ffn {
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            embd,
                            r,
                        )?;
                        // P62 gluq silu twin - same election as the first
                        // prefill site.
                        exec.f8t_gemm(
                            gu_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_ffn_gate,
                            embd,
                            2 * ff,
                            r,
                        )?;
                        exec.swiglu_fused(&sc.d_ffn_gate, &mut sc.d_ffn_up, ff, r)?;
                        exec.quantize_e4m3_row(
                            &sc.d_ffn_up,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            ff,
                            r,
                        )?;
                        exec.f8t_gemm(
                            dn_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            ff,
                            embd,
                            r,
                        )?;
                    } else if let Some([gu8, d8]) = f8f {
                        // fused plane, row-sliced: gate = rows [0,ff), up =
                        // rows [ff,2ff) - byte-identical to the old separate
                        // planes (same repack stream, offset math only)
                        let ffh = gu8.2 / 2;
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * gu8.1)?;
                        // bf16 epilogue pair when the pack ships it: halves
                        // the gate/up store traffic (the rival's cutlass
                        // writes bf16; ours wrote f32) and the fused quant
                        // reads bf16 - else the f32 chain below.
                        static O16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let o16 = *O16.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        // P74: see the MO16T note - sm_100 tc5 route, wash
                        static O16T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let o16 = o16
                            || (*O16T.get_or_init(|| {
                                paddock_models::dev_var_os!("PADDOCK_QWEN35_O16_TC5").is_some()
                                    && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                            }) && r >= 256);
                        if o16 && exec.has_f8_o16() {
                            if exec.has_swiglu_b16_gu() {
                                // One fused gate|up GEMM - bit-exact vs the
                                // sliced pair (ffh is 128-tile-aligned, so
                                // every output element keeps its K chain);
                                // d_ffn_gate's r*ffh f32 capacity holds the
                                // r*2ffh bf16 fused output byte-exactly.
                                exec.f8_gemm_w8_o16(
                                    &gu8.0,
                                    0,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut sc.d_ffn_gate,
                                    gu8.1,
                                    gu8.2,
                                    r,
                                )?;
                                exec.quantize_e4m3_swiglu_b16_gu(
                                    &sc.d_ffn_gate,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * d8.1,
                                    ffh,
                                )?;
                            } else {
                                exec.f8_gemm_w8_o16(
                                    &gu8.0,
                                    0,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut sc.d_ffn_gate,
                                    gu8.1,
                                    ffh,
                                    r,
                                )?;
                                exec.f8_gemm_w8_o16(
                                    &gu8.0,
                                    ffh,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut sc.d_ffn_up,
                                    gu8.1,
                                    ffh,
                                    r,
                                )?;
                                exec.quantize_e4m3_swiglu_b16(
                                    &sc.d_ffn_gate,
                                    &sc.d_ffn_up,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * d8.1,
                                )?;
                            }
                        } else {
                            exec.f8_gemm_w8(
                                &gu8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_gate,
                                gu8.1,
                                ffh,
                                r,
                            )?;
                            exec.f8_gemm_w8(
                                &gu8.0,
                                ffh,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_up,
                                gu8.1,
                                ffh,
                                r,
                            )?;
                            // fused swiglu+e4m3-quant: one pass instead of
                            // swiglu-write + quant-read (286 MB/layer-tick of f32
                            // round-trip at r=2048 - the bf16-activations gap
                            // vs the engines that write bf16 epilogues, closed at
                            // the seam that matters)
                            exec.quantize_e4m3_swiglu(
                                &sc.d_ffn_gate,
                                &sc.d_ffn_up,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                r * d8.1,
                            )?;
                        }
                        if o16 && exec.has_f8_o16() && exec.has_add_b16() {
                            // bf16 down out (halves the last 42 MB/layer-tick
                            // f32 store of the FFN) - the tail add reads bf16
                            exec.f8_gemm_w8_o16(
                                &d8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                d8.1,
                                d8.2,
                                r,
                            )?;
                            proj_is_b16 = true;
                        } else {
                            exec.f8_gemm_w8(
                                &d8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                d8.1,
                                d8.2,
                                r,
                            )?;
                        }
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            gate,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_gate,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            up,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_up,
                            r,
                        )?;
                        prefill_ffn_down_any(
                            &exec,
                            down,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            r,
                        )?;
                    }
                }
                Ffn::Nvf4Dense { gate, up, down } => {
                    // f8t tile arm first, off the planes load.rs builds from
                    // the NVFP4 checkpoint's own values - same chain and same
                    // election as the Dense arm above. write_xn stays true:
                    // both arms consume the f32 xn (f8t quantizes it itself).
                    let f8t_ffn = bs_f8t_ffn_p
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| f8t_u);
                    // wide-prefill arm (see the chunked site): the f8w plane
                    // pair above the tile band, nvf4_ffn only when neither
                    // plane set built.
                    // the DECODE lane's arm, at wave widths. Above
                    // the tile arm's decode-band row bound the fp4 gate|up +
                    let f8f = bs_f8ffn_p.get(li).and_then(|o| o.as_ref()).filter(|_| {
                        f8t_ffn.is_none()
                            && r > nvf4_f8w_min_rows(w8_min)
                            && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
                    });
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        mixer_b16,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        true,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    if let Some([gu_t, dn_t]) = f8t_ffn {
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            embd,
                            r,
                        )?;
                        exec.f8t_gemm(
                            gu_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_ffn_gate,
                            embd,
                            2 * ff,
                            r,
                        )?;
                        exec.swiglu_fused(&sc.d_ffn_gate, &mut sc.d_ffn_up, ff, r)?;
                        exec.quantize_e4m3_row(
                            &sc.d_ffn_up,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            ff,
                            r,
                        )?;
                        exec.f8t_gemm(
                            dn_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            ff,
                            embd,
                            r,
                        )?;
                    } else if let Some([gu8, d8]) = f8f {
                        proj_is_b16 = prefill_ffn_f8w(
                            &exec,
                            gu8,
                            d8,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else {
                        // no plane pair (small card / kill switch): the
                        // checkpoint-exact W4A16 walk
                        nvf4_ffn(
                            &exec,
                            gate,
                            up,
                            down,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_nvs,
                            &mut sc.d_nv4part,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            r,
                        )?;
                    }
                }
                Ffn::Moe(w) => {
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        mixer_b16,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        true,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    moe_ffn(
                        &exec,
                        w,
                        moe_dims.expect("moe dims"),
                        embd,
                        r,
                        true,
                        &sc.d_xn,
                        &mut sc.d_moe_xq,
                        &mut sc.d_moe_xs,
                        &mut sc.d_ssums,
                        &mut sc.d_moe_xs8,
                        &mut sc.d_moe_fs8,
                        &mut sc.d_moe_logits,
                        &sc.d_zero_bias,
                        &mut sc.d_moe_idx,
                        &mut sc.d_moe_w,
                        &mut sc.d_moe_fused,
                        &mut sc.d_moe_fq,
                        &mut sc.d_moe_fs,
                        &mut sc.d_moe_srow,
                        &mut sc.d_moe_sslot,
                        &mut sc.d_moe_bexp,
                        &mut sc.d_moe_part,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        &mut sc.d_skfix,
                        &mut sc.d_ffn_gate,
                        &mut sc.d_ffn_up,
                        &mut sc.d_mixed,
                        &mut sc.d_proj,
                    )?;
                }
            }
            if proj_is_b16 {
                exec.add_b16(&mut sc.d_x, &sc.d_proj, r * embd)?;
            } else {
                exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
            }
        }
        exec.rmsnorm_batch(&sc.d_x, &out_norm.buf, &mut sc.d_h, embd, eps, r)?;

        // Finishing spans: with a device fin_plan the first sampled token
        // joins the decode sampling batch at rows b+j of bs.d_logits - no
        // [vocab] readback, no host softmax, no mid-tick stream drain (the
        // recorded ~6-7 ms stall between the lm-head and the next embed).
        // Without a plan (constraint/logprobs/non-plannable/recompute) the
        // classic synchronous last-row readback runs. Capacity: decode rows
        // + chunking slots occupy DISTINCT slots, so b + nf <= max_batch =
        // the d_logits/d_samp_* row budget.
        let mut finished: Vec<(usize, FinishSample, usize)> = Vec::new();
        let mut fin_dev: Vec<DevicePlan> = Vec::new();
        {
            let mut rb = b;
            for &(_, slot, done, take, finishing, _) in &shares {
                if finishing {
                    exec.copy_region(&sc.d_h, (rb + take - 1) * embd, &mut sc.d_xn, 0, embd)?;
                    let plan = fin_plans.iter().find_map(|&(s, p)| {
                        if s == slot {
                            if let RowSample::Device(dp) = p {
                                Some(dp)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    });
                    if let Some(dp) = plan {
                        head_logits_1row(
                            &exec,
                            out_f8_h,
                            output,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            &mut sc.d_head_part,
                            &mut sc.d_logits,
                            "batch.rs finishing span (device plan)",
                        )?;
                        if b == 0 {
                            // span-only tick: stage in the span-side fin
                            // buffers - the decode graph owns d_logits rows,
                            // and the overlap scheduler may be replaying it
                            // on the decode lane right now
                            let row = fin_dev.len();
                            exec.copy_region(
                                &sc.d_logits,
                                0,
                                &mut bs.d_fin_logits,
                                row * vocab,
                                vocab,
                            )?;
                        } else {
                            let row = b + fin_dev.len();
                            exec.copy_region(
                                &sc.d_logits,
                                0,
                                &mut bs.d_logits,
                                row * vocab,
                                vocab,
                            )?;
                        }
                        fin_dev.push(dp);
                        // id patched from the shared ids readback below
                        finished.push((slot, FinishSample::Sampled(0), done + take));
                    } else {
                        head_logits_1row(
                            &exec,
                            out_f8_h,
                            output,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            &mut sc.d_head_part,
                            &mut sc.d_logits,
                            "batch.rs finishing span (readback)",
                        )?;
                        finished.push((
                            slot,
                            FinishSample::Logits(exec.to_host(&sc.d_logits)?),
                            done + take,
                        ));
                    }
                }
                rb += take;
            }
        }

        // decode rows [0..B] -> bs.d_logits -> device sampling (batch_sampled_impl
        // epilogue). Same lm-head ladder the decode graph uses at width B.
        // Device-planned finishers ride the same sample_rows pass at rows
        // [b..b+nf] (their logits staged above), their ids read back in the
        // one ids copy every sampled tick already pays.
        let nf = fin_dev.len();
        let tot = b + nf;
        if b == 0 && nf > 0 {
            // finishers only (always the case for overlap spans): no decode
            // lm-head; sample on the span-side fin buffers - same logits,
            // same sampler kernel, bit-identical ids, disjoint staging.
            let dev_full = exec.has_sample_rows_t();
            let mut par = vec![0u32; nf * 4];
            let mut tpar = vec![0u32; nf * 4];
            let mut any_trunc = false;
            let mut any_base = false;
            for (i, &dp) in fin_dev.iter().enumerate() {
                match dp {
                    DevicePlan::Greedy => {
                        par[i * 4 + 2] = 1;
                        any_base = true;
                    }
                    DevicePlan::Categorical { inv_t, u } => {
                        par[i * 4] = inv_t.to_bits();
                        par[i * 4 + 1] = u.to_bits();
                        par[i * 4 + 2] = 2;
                        any_base = true;
                    }
                    // P67 mode 5 (full device) / P65 mode 4 (host-head)
                    DevicePlan::TruncCat {
                        inv_t,
                        u,
                        k,
                        top_p,
                        min_p,
                    } => {
                        par[i * 4] = inv_t.to_bits();
                        par[i * 4 + 1] = u.to_bits();
                        par[i * 4 + 2] = if dev_full { 5 } else { 4 };
                        tpar[i * 4] = k;
                        tpar[i * 4 + 1] = top_p.to_bits();
                        tpar[i * 4 + 2] = min_p.to_bits();
                        any_trunc = true;
                    }
                    // RS plans are gemma4-only (supports_spec_rs); skip-safe
                    DevicePlan::RsVerify { .. } | DevicePlan::RsTrunc { .. } => {}
                }
            }
            {
                let mut v = bs.d_fin_par.slice_mut(0..nf * 4);
                exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
                if any_trunc && dev_full {
                    let mut t = bs.d_fin_tpar.slice_mut(0..nf * 4);
                    exec.stream.memcpy_htod(&tpar, &mut t).map_err(drv)?;
                }
            }
            let fold_off = Self::samp_fold_off();
            if any_base || fold_off {
                exec.sample_rows(
                    &bs.d_fin_logits,
                    &bs.d_fin_par,
                    &mut bs.d_fin_out,
                    nf,
                    vocab,
                )?;
            }
            if any_trunc && dev_full {
                Self::samp_fold_witness(any_base, true, false);
                exec.sample_rows_t(
                    &bs.d_fin_logits,
                    &bs.d_fin_par,
                    &bs.d_fin_tpar,
                    &mut bs.d_fin_out,
                    nf,
                    vocab,
                )?;
                if fold_off {
                    exec.sample_rows_p(
                        &bs.d_fin_logits,
                        &bs.d_fin_par,
                        &bs.d_fin_tpar,
                        &mut bs.d_fin_out,
                        nf,
                        vocab,
                    )?;
                }
            } else if any_trunc {
                exec.topk_rows(
                    &bs.d_fin_logits,
                    &bs.d_fin_par,
                    &mut bs.d_fin_head,
                    nf,
                    vocab,
                    64,
                )?;
            }
        } else if b > 0 {
            // f8 head first at every width. unified_launch_core had no f8 arm
            // at all -- it is the twin of record_batch_step's head block below
            // and simply never got one, so under the REPLACE lane (Q8_0 head
            // dropped at load) every decode tick through this path would read
            // a freed plane.
            if let Some(p) = super::head_f8(out_f8_h, b) {
                super::head_f8_gemm(
                    &exec,
                    p,
                    &sc.d_h,
                    &mut sc.d_pxq,
                    &mut sc.d_exs,
                    &mut bs.d_ks_part,
                    &mut bs.d_logits,
                    b,
                )?;
            } else if b == 1 {
                super::stub_guard(output, "batch.rs unified_launch_core b=1 head")?;
                match output {
                    // W4A8 lm-head at b=1 - the same serving class as the tick
                    // (PADDOCK_KQ_EXACT_GEMV=1 pins the exact-f32 GEMV)
                    QuantW::Kq(k)
                        if exec.has_kquant_gemv_w4a8()
                            && paddock_models::dev_var_os!("PADDOCK_KQ_EXACT_GEMV").is_none() =>
                    {
                        exec.quantize_q8_sums(
                            &sc.d_h,
                            &mut bs.d_xq,
                            &mut bs.d_xs,
                            &mut sc.d_ssums,
                            embd,
                        )?;
                        let needs = crate::gpu::kq_needs_sums(k.ty);
                        exec.kquant_gemv_w4a8(
                            k,
                            &bs.d_xq,
                            &bs.d_xs,
                            needs.then_some(&sc.d_ssums),
                            &mut bs.d_logits,
                        )?;
                    }
                    _ => gemv_any(&exec, output, &sc.d_h, &mut bs.d_logits)?,
                }
            } else if let QuantW::Kq(k) = output {
                // k-quant lm_head: the W4A8 dp4a GEMM (record_batch_step twin)
                exec.quantize_q8(&sc.d_h, &mut bs.d_xq, &mut bs.d_xs, b * embd)?;
                let needs = crate::gpu::kq_needs_sums(k.ty);
                if needs {
                    exec.q8_sums_strided(&bs.d_xq, &mut sc.d_ssums, k.dims[0], b)?;
                }
                exec.kquant_gemm_dp4a(
                    k,
                    &bs.d_xq,
                    &bs.d_xs,
                    needs.then_some(&sc.d_ssums),
                    &mut bs.d_logits,
                    b,
                )?;
            } else if let Some((p8, pi, po)) = self.out_f8.as_ref().filter(|_| {
                // f8 lm_head (labeled class, b >= 8 like the FFN f8 gate):
                // vocab tiles >> die so nz stays 1 and the part plane is
                // never touched (vocab-sized partials would not fit).
                // LOWB (a B200 bring-up arm): the b <= 4 arm below is
                // q8_0_gemv_dp4a_nc, one CTA per vocab row = 248320 CTAs of
                // 5120 elements each. At b=1 that is 681 us/tick, 1.98
                // TB/s, 10.6% of the whole decode tick - the largest single
                // kernel instance in the step and the only projection still
                // on the legacy int8 path. Separate opt-in from the plane
                // build so the b>=8 precision class stays exactly as shipped.
                b >= super::f8_head_min()
                    || paddock_models::dev_var_os!("PADDOCK_QWEN_F8_LMHEAD_LOWB").is_some()
            }) {
                exec.quantize_e4m3(&sc.d_h, &mut sc.d_pxq, &mut sc.d_exs, b * embd)?;
                exec.f8d_gemm_mma_ks(
                    p8,
                    *pi,
                    *po,
                    &sc.d_pxq,
                    &sc.d_exs,
                    &mut bs.d_ks_part,
                    &mut bs.d_logits,
                    b,
                )?;
            } else if b <= 4 {
                exec.quantize_q8(&sc.d_h, &mut bs.d_xq, &mut bs.d_xs, b * embd)?;
                super::stub_guard(output, "batch.rs unified decode head")?;
                exec.q8_0_gemv_dp4a_nc(output.q8(), &bs.d_xq, &bs.d_xs, &mut bs.d_logits, b)?;
            } else {
                exec.quantize_q8(&sc.d_h, &mut bs.d_xq, &mut bs.d_xs, b * embd)?;
                if b >= ks_min_batch() {
                    exec.q8_0_gemm_mma(output.q8(), &bs.d_xq, &bs.d_xs, &mut bs.d_logits, b)?;
                } else {
                    exec.q8_0_gemm_mt_dp4a(output.q8(), &bs.d_xq, &bs.d_xs, &mut bs.d_logits, b)?;
                }
            }
            let dev_full = exec.has_sample_rows_t();
            let mut par = vec![0u32; tot * 4];
            let mut tpar = vec![0u32; tot * 4];
            let mut any_trunc = false;
            let mut any_base = false;
            for (i, p) in plans
                .iter()
                .copied()
                .chain(fin_dev.iter().map(|&dp| RowSample::Device(dp)))
                .enumerate()
            {
                match p {
                    RowSample::Hole | RowSample::Host => {}
                    RowSample::Device(DevicePlan::Greedy) => {
                        par[i * 4 + 2] = 1;
                        any_base = true;
                    }
                    RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                        par[i * 4] = inv_t.to_bits();
                        par[i * 4 + 1] = u.to_bits();
                        par[i * 4 + 2] = 2;
                        any_base = true;
                    }
                    // P67 mode 5 (full device) / P65 mode 4 (host-head)
                    RowSample::Device(DevicePlan::TruncCat {
                        inv_t,
                        u,
                        k,
                        top_p,
                        min_p,
                    }) => {
                        par[i * 4] = inv_t.to_bits();
                        par[i * 4 + 1] = u.to_bits();
                        par[i * 4 + 2] = if dev_full { 5 } else { 4 };
                        tpar[i * 4] = k;
                        tpar[i * 4 + 1] = top_p.to_bits();
                        tpar[i * 4 + 2] = min_p.to_bits();
                        any_trunc = true;
                    }
                    // RS plans are gemma4-only (supports_spec_rs)
                    RowSample::Device(DevicePlan::RsVerify { .. })
                    | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
                }
            }
            {
                let mut v = bs.d_samp_par.slice_mut(0..tot * 4);
                exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
                if any_trunc && dev_full {
                    let mut t = bs.d_samp_tpar.slice_mut(0..tot * 4);
                    exec.stream.memcpy_htod(&tpar, &mut t).map_err(drv)?;
                }
            }
            let fold_off = Self::samp_fold_off();
            if any_base || fold_off {
                exec.sample_rows(&bs.d_logits, &bs.d_samp_par, &mut bs.d_samp_out, tot, vocab)?;
            }
            if any_trunc && dev_full {
                Self::samp_fold_witness(any_base, true, false);
                exec.sample_rows_t(
                    &bs.d_logits,
                    &bs.d_samp_par,
                    &bs.d_samp_tpar,
                    &mut bs.d_samp_out,
                    tot,
                    vocab,
                )?;
                if fold_off {
                    exec.sample_rows_p(
                        &bs.d_logits,
                        &bs.d_samp_par,
                        &bs.d_samp_tpar,
                        &mut bs.d_samp_out,
                        tot,
                        vocab,
                    )?;
                }
            } else if any_trunc {
                exec.topk_rows(
                    &bs.d_logits,
                    &bs.d_samp_par,
                    &mut bs.d_samp_head,
                    tot,
                    vocab,
                    64,
                )?;
            }
        }
        // Span-completion marker + inflight stash: everything the finish
        // half needs, including the per-call eager device buffers, which
        // must outlive the enqueued GPU work (finish drains before they
        // drop - a cudarc pool realloc reusing freed memory under an async
        // reader is an illegal access, same rule the classic ticks follow).
        let ev = exec.record_event()?;
        self.unified_inflight = Some(UnifiedInflight {
            shares,
            b,
            tot,
            nf,
            plans: plans.to_vec(),
            fin_dev,
            finished,
            ev,
            hold_u32: vec![d_tokens, d_pos, d_slots, d_mrope],
            hold_f32: vec![d_pf_qn, d_pf_attn],
            hold_seg: seg_meta,
        });
        Ok(())
    }

    fn unified_finish_core(
        &mut self,
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        use crate::generator::{FinishSample, RowSample, SampledStep};
        let inf = self
            .unified_inflight
            .take()
            .expect("no unified span in flight");
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        // Drain the MAIN lane before any readback and before the span's
        // eager buffers (held in `inf`) drop. The single-body version synced
        // here too (its blocking ids readback implied it); the decode lane
        // is untouched - pipe ticks enqueued during the span keep running.
        exec.synchronize()?;
        let UnifiedInflight {
            shares,
            b,
            tot,
            nf,
            plans,
            fin_dev,
            mut finished,
            ..
        } = inf;
        let step = if tot == 0 {
            SampledStep {
                ids: Vec::new(),
                host_rows: Vec::new(),
            }
        } else {
            let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
            let mut ids = if b == 0 {
                let view = bs
                    .d_fin_out
                    .try_slice(0..nf)
                    .ok_or_else(|| GpuError::Driver("fin_out slice out of range".into()))?;
                exec.stream.clone_dtoh(&view).map_err(drv)?
            } else {
                let view = bs
                    .d_samp_out
                    .try_slice(0..tot)
                    .ok_or_else(|| GpuError::Driver("samp_out slice out of range".into()))?;
                exec.stream.clone_dtoh(&view).map_err(drv)?
            };
            // P65: TruncCat rows (decode AND finisher) head-sample on the
            // host from the compact top-64 plane - same composite row order
            // as the par packing (plans then fin_dev). P67: when the pack
            // ships slot 435, the launch used mode 5 and the ids already
            // carry the device-sampled tokens - nothing to fill.
            if !exec.has_sample_rows_t() {
                use crate::sampler::DevicePlan;
                let comp: Vec<RowSample> = plans
                    .iter()
                    .copied()
                    .chain(fin_dev.iter().map(|&dp| RowSample::Device(dp)))
                    .collect();
                if comp
                    .iter()
                    .any(|p| matches!(p, RowSample::Device(DevicePlan::TruncCat { .. })))
                {
                    let (buf, rows) = if b == 0 {
                        (&bs.d_fin_head, nf)
                    } else {
                        (&bs.d_samp_head, tot)
                    };
                    let hv = buf
                        .try_slice(0..rows * 128)
                        .ok_or_else(|| GpuError::Driver("head slice out of range".into()))?;
                    let head = exec.stream.clone_dtoh(&hv).map_err(drv)?;
                    trunc_fill_ids(&head, &comp, &mut ids);
                }
            }
            // patch the device-planned finishers with their sampled ids (the
            // j-th Sampled entry rode row b+j - row j on the fin buffers
            // when b == 0 - in share order)
            let mut fj = 0usize;
            for f in finished.iter_mut() {
                if let FinishSample::Sampled(id) = &mut f.1 {
                    *id = ids[b + fj];
                    fj += 1;
                }
            }
            let mut host_rows = Vec::new();
            for (i, p) in plans.iter().enumerate() {
                if matches!(p, RowSample::Host) {
                    let view = bs
                        .d_logits
                        .try_slice(i * vocab..(i + 1) * vocab)
                        .ok_or_else(|| GpuError::Driver("logits row slice out of range".into()))?;
                    let row = exec.stream.clone_dtoh(&view).map_err(drv)?;
                    host_rows.push((i, row));
                }
            }
            SampledStep { ids, host_rows }
        };
        // Warm hook, UNIFIED edition: without it unified slots stay
        // MTP-cold and the spec win configs (c1/dc4) collapse to dense. Spans
        // chain across intra-prompt ticks exactly like the serial hook
        // (start = done; this tick's h rows sit at the span's rb offset in
        // d_h, which nothing below overwrites). Same WARM_MAX + live-count
        // hint gates as the other two hooks.
        if self.spec_warm_wanted && self.serve_spec_on() && self.batch.is_some() {
            self.ensure_serve_spec()?;
            let warm_max: usize = std::env::var("PADDOCK_QWEN35_SPEC_WARM_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048);
            let embd_w = self.embd;
            let mut rb = b;
            for &(idx, slot, done, take, _, _) in &shares {
                let (in_range, chain_ok) = {
                    let sb = self.spec_batch.as_ref().expect("spec enabled");
                    let ir = slot < sb.alloc_batch;
                    let within = done + take <= warm_max;
                    (
                        ir,
                        ir && within && (done == 0 || (sb.mtp_warm[slot] && sb.pos[slot] == done)),
                    )
                };
                if chain_ok {
                    if done == 0 {
                        let zeros = vec![0f32; embd_w];
                        let sb = self.spec_batch.as_mut().expect("spec enabled");
                        let mut v = sb.pending_h.slice_mut(slot * embd_w..(slot + 1) * embd_w);
                        self.exec
                            .stream
                            .memcpy_htod(&zeros, &mut v)
                            .map_err(|e| GpuError::Driver(e.to_string()))?;
                    }
                    let toks: Vec<u32> = self.chunked[idx].tokens[done..done + take].to_vec();
                    self.mtp_warm_slot(slot, &toks, done, rb)?;
                    let sb = self.spec_batch.as_mut().expect("spec enabled");
                    sb.pos[slot] = done + take;
                    sb.mtp_warm[slot] = true;
                    sb.mtp_toks[slot].truncate(done);
                    sb.mtp_toks[slot].extend_from_slice(&toks);
                } else if in_range {
                    self.spec_batch.as_mut().expect("spec enabled").mtp_warm[slot] = false;
                }
                rb += take;
            }
        }
        // Prefix-cache maintenance (pool mode): snapshot the DeltaNet state at a
        // prompt's checkpoint boundary (a span was made to land exactly there) so a
        // future turn resumes at it, and on finish cache every full KV page so the
        // shared prefix is adoptable. Mirrors prefill_slot_tail_paged, but the
        // snapshot is taken from the slot at the span boundary (state now sits at
        // `done+take` after the exit sync above). Only full pages / block-aligned
        // checkpoints are cached - the exact contract PagedRadix expects.
        if self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .paged_prefix
            .is_some()
        {
            let stage_of = Self::fused_stage_map(&shares);
            for (sidx, &(idx, slot, done, take, finishing, _)) in shares.iter().enumerate() {
                let t_len = self.chunked[idx].tokens.len();
                let step = self.tier_ckpt_step();
                let cuts = super::chunk_cuts(&self.chunked[idx], step);
                let landed = done + take;
                if !finishing && landed > 0 && cuts.contains(&landed) {
                    let toks: Vec<u32> = self.chunked[idx].tokens[..landed].to_vec();
                    let blocks: Vec<u32> = self.batch.as_ref().expect("batch enabled").tables[slot]
                        .blocks()[..landed / BLOCK_TOKENS]
                        .to_vec();
                    let si = {
                        let bs = self.batch.as_mut().expect("batch enabled");
                        let pool = bs.pool.as_mut().expect("prefix cache implies pool");
                        bs.paged_prefix
                            .as_mut()
                            .expect("prefix cache checked above")
                            .insert(&toks, &blocks, pool);
                        bs.paged_prefix
                            .as_mut()
                            .expect("prefix cache checked above")
                            .attach_state(&toks, landed)
                    };
                    if let Some(si) = si {
                        // fused tail in the same tick: the slot's live state
                        // has advanced past the boundary - attach from the
                        // staged blob the walk filled at the boundary instead
                        if let Some(stg) = stage_of[sidx] {
                            self.snapshot_staged_pool(stg, si)?;
                        } else {
                            self.snapshot_paged_state(slot, si)?;
                        }
                        self.record_mtp_cover(si, slot, landed, &toks);
                        self.record_dflash_cover(si, slot, landed, &toks);
                    }
                    if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
                        // The OUTCOME, not the branch: `attach_state` returns
                        // None when the node already holds a checkpoint or the
                        // admission policy refused, and only a Some pays the
                        // ~170 MiB snapshot. Counting the branch instead is how
                        // an earlier census over-reported snapshot traffic.
                        let (w, s, r) = self
                            .batch
                            .as_ref()
                            .expect("batch enabled")
                            .paged_prefix
                            .as_ref()
                            .expect("prefix cache checked above")
                            .state_stats();
                        tracing::info!(
                            "qwen35-chunk-publish: t_len {t_len} landed {landed} -> {} \
                             (writes {w} steals {s} refused {r})",
                            if si.is_some() { "SNAPSHOT" } else { "no-op" }
                        );
                    }
                } else if finishing {
                    let full = t_len / BLOCK_TOKENS;
                    if full > 0 {
                        let toks: Vec<u32> =
                            self.chunked[idx].tokens[..full * BLOCK_TOKENS].to_vec();
                        let blocks: Vec<u32> = self.batch.as_ref().expect("batch enabled").tables
                            [slot]
                            .blocks()[..full]
                            .to_vec();
                        let bs = self.batch.as_mut().expect("batch enabled");
                        let pool = bs.pool.as_mut().expect("prefix cache implies pool");
                        bs.paged_prefix
                            .as_mut()
                            .expect("prefix cache checked above")
                            .insert(&toks, &blocks, pool);
                    }
                }
            }
        }
        // Advance non-finishing chunks; drop finished ones (descending index so
        // earlier removals don't shift the rest).
        let mut remove: Vec<usize> = Vec::new();
        for &(idx, _, _, take, finishing, _) in &shares {
            if finishing {
                remove.push(idx);
            } else {
                self.chunked[idx].done += take;
            }
        }
        remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in remove {
            self.chunked.remove(idx);
        }
        Ok((step, finished))
    }

    /// `forward_mixed_sampled` without device sampling: returns the decode rows'
    /// flat `[nd, vocab]` logits (input order) for host sampling.
    pub fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        let finished = self.advance_chunks(budget)?;
        let logits = if decodes.is_empty() {
            Vec::new()
        } else {
            let exec = self.exec.clone();
            let vocab = self.vocab;
            let nd = decodes.len();
            let toks: Vec<u32> = decodes.iter().map(|&(_, t, _)| t).collect();
            let pos: Vec<u32> = decodes.iter().map(|&(_, _, p)| p).collect();
            let slots: Vec<u32> = decodes.iter().map(|&(k, _, _)| k as u32).collect();
            // Split-decode companion (see batch_sampled_impl): this was the
            // last full-width path - one warmup mixed tick captured the
            // b=128 graph whose every layer records the mma fallback.
            // Halves ride the <=64-row f8t graphs; logits read per half.
            let mut logits_out: Vec<f32> = Vec::with_capacity(nd * vocab);
            let bmax = f8t_dec_bmax();
            let mut b0 = 0usize;
            while b0 < nd {
                let bn = (nd - b0).min(bmax);
                self.launch_batch_step_slots(
                    &toks[b0..b0 + bn],
                    &pos[b0..b0 + bn],
                    Some(&slots[b0..b0 + bn]),
                )?;
                let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
                let view = bs
                    .d_logits
                    .try_slice(0..bn * vocab)
                    .ok_or_else(|| GpuError::Driver("logits slice out of range".into()))?;
                let half = exec
                    .stream
                    .clone_dtoh(&view)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                logits_out.extend(half);
                b0 += bn;
            }
            logits_out
        };
        Ok((logits, finished))
    }

    fn launch_batch_step(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuModelError> {
        self.launch_batch_step_slots(tokens, positions, None)
    }

    /// Capture the fixed-B batched decode step into a replayable graph, cached
    /// per batch size. Shared by the classic single-tick path and the pipelined
    /// path - both replay the same graph, so the pipe is bit-identical to
    /// `forward_batch_sampled`. Grid-stable for a given B (loop bounds like KV
    /// length come from the device position buffer), so one capture replays at
    /// every position.
    fn ensure_batch_graph(&mut self, b: usize) -> Result<(), GpuModelError> {
        if self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .graphs
            .contains_key(&b)
        {
            return Ok(());
        }
        if b > f8t_dec_bmax() {
            tracing::warn!(
                "ensure_batch_graph TRIPWIRE b={b} caller:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
        }
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        // capture against the DEDICATED decode arena (see enable_batch): the
        // graph must not bake the shared prefill scratch's addresses
        let saved = self.scratch.take();
        self.scratch = self.decode_arena.take();
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        let rec = self.record_batch_step(b);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        self.decode_arena = self.scratch.take();
        self.scratch = saved;
        rec?; // surface a record failure only after capture is cleanly ended
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("batch capture produced no graph".into()))?;
        self.batch
            .as_mut()
            .expect("batch enabled")
            .graphs
            .insert(b, SendGraph(graph));
        Ok(())
    }

    /// True when the pipelined pure-decode path can run: device sampling + the
    /// advance kernel, with no pin forcing a different dispatch. Runs in
    /// identity-KV mode AND under a FULLY-BACKED budget pool (capacity >=
    /// max_batch x blocks_per_slot - the enable_batch hard-reject guarantees
    /// this unless the user opted into oversubscription): per-tick block growth
    /// then cannot PoolExhaust (worst case it evicts radix pages), so
    /// `pipe_launch_tick` grows tables infallibly before each replay, exactly
    /// like the classic path does per step. Oversubscribed pools keep the
    /// classic tick (growth can fail there and the pipe has no clean unwind -
    /// the tick's KV/DeltaNet state advances before its ids are read back).
    /// Measured on an RTX PRO 6000, the pool gate cost ~7 ms/step of host
    /// serialization at b=32 (83% GPU busy on the classic tick).
    pub fn supports_decode_pipe(&self) -> bool {
        // A DFlash drafter needs every decode tick's feature append (the
        // pipe is device-driven and cannot append - the ring stalls behind
        // the cursor and the drafter goes permanently cold; see
        // dflash_ensure_warm). Classic ticks carry decode while attached.
        self.dflash.is_none()
            && self.exec.has_sample_rows()
            && self.exec.has_pipe_advance()
            // k-quant rides the pipe since the kq batch step is the
            // same captured-graph body (bmmq!'s dp4a/ks rungs are stream-ordered
            // and capture-safe - the ks launcher's device-attr statics warm on
            // the pre-capture run, like Q8's). Greedy A/B pipe-vs-classic gated.
            //
            // The static full-backing pool requirement is gone - it
            // silently disabled the pipe for every oversubscribed serve (the
            // FP8 headline stack runs PADDOCK_KV_OVERSUBSCRIBE=1), which the
            // 128x128c32 investigation exposed after five flat scheduling
            // fixes landed into dead code. Oversubscription safety moved to
            // the scheduler's DYNAMIC headroom gate: the pipe begins/continues
            // only while pool_free_blocks() covers a worst-case tick of block
            // growth plus the admission watermark (service.rs pipe loop), so
            // ensure_slot_blocks inside pipe_launch_tick can never PoolExhaust
            // mid-flight.
            && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none()
    }

    /// True when the overlap decode lane is forked (PADDOCK_OVERLAP at
    /// enable_batch) and the pipe machinery is available - the scheduler's
    /// gate for the overlapped span+decode tick (route B).
    pub fn overlap_ready(&self) -> bool {
        self.overlap_exec.is_some() && self.supports_decode_pipe()
    }

    /// Pack per-row sampler params exactly like `batch_sampled_impl` (Host/Hole
    /// are skip - the pipe never runs with Host rows). Returns
    /// (par, tpar, any_trunc): TruncCat rows pack mode 5 - the pipe only
    /// ever receives them when the pack ships slot 435 (the service gates
    /// on supports_device_trunc), so full-device is the only pipe form.
    /// Launch fold for the sampler chains (nemotron's (any5,any6)
    /// fold ported): launch only the chains this tick's rows NEED. The
    /// elected qwen truncation (top_k 20) is pure mode 5, so without this
    /// every tick also paid the 11-launch mode-6 nucleus chain (and the
    /// base modes-1/2 kernel) for zero rows. PADDOCK_NO_SAMP_FOLD=1
    /// restores launch-always for A/B legs. Behavior-identical: a skipped
    /// chain had no rows of its mode, so it wrote nothing.
    fn samp_fold_off() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_SAMP_FOLD").is_some())
    }

    /// Engagement witness (bisect-trap law): once per process, the first
    /// folded trunc tick prints which chains it actually launched.
    fn samp_fold_witness(any_base: bool, any5: bool, any6: bool) {
        static W: std::sync::Once = std::sync::Once::new();
        W.call_once(|| {
            eprintln!("[samp-fold] engaged: base={any_base} t5={any5} p6={any6}");
        });
    }

    fn pack_samp_par(
        plans: &[crate::generator::RowSample],
    ) -> (Vec<u32>, Vec<u32>, bool, bool, bool) {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        let mut tpar = vec![0u32; plans.len() * 4];
        let (mut any_base, mut any5, mut any6) = (false, false, false);
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => {
                    par[i * 4 + 2] = 1;
                    any_base = true;
                }
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                    any_base = true;
                }
                // P67b: full-device truncation sampling in the pipe
                RowSample::Device(DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = if *k >= 1 && *k <= 64 { 5 } else { 6 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    if *k >= 1 && *k <= 64 {
                        any5 = true
                    } else {
                        any6 = true
                    }
                }
                // RS plans are gemma4-only (supports_spec_rs); skip-safe
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        (par, tpar, any_base, any5, any6)
    }

    /// Host-build the section-major mrope buffer for a pipe tick - positions are
    /// deterministic (`pos0[i] + tick`), so RoPE needs no device->host roundtrip.
    /// Byte-identical layout to `launch_batch_step_slots`' host build.
    fn pipe_mrope_host(pos0: &[u32], delta: &[i64], tick: u64) -> Vec<u32> {
        let b = pos0.len();
        let mut v = vec![0u32; 4 * b];
        for sec in 0..4 {
            for i in 0..b {
                v[sec * b + i] = (pos0[i] as i64 + tick as i64 + delta[i]) as u32;
            }
        }
        v
    }

    /// Enqueue one pipelined tick: advance the device-dependent inputs (token =
    /// previous tick's sampled id, position += 1) from the previous out ring,
    /// rebuild mrope host-side, replay the shared step graph (no sync), sample
    /// on device into this tick's out ring, and record its readiness event.
    fn pipe_launch_tick(
        &mut self,
        plans: &[crate::generator::RowSample],
        advance: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let (b, tick) = {
            let p = self.pipe.as_ref().expect("pipe active");
            (p.b, p.tick)
        };
        // P5 budget pool: back the position this tick writes (pos0[i] + tick -
        // deterministic, same arithmetic as pipe_mrope_host) before any pipe
        // state mutates, so a growth error leaves the rings/inputs untouched.
        // The pipe only runs on a fully-backed pool (supports_decode_pipe), so
        // this cannot PoolExhaust; the table htod inside ensure_slot_blocks is
        // stream-ordered ahead of the graph replay below. Row i drives its
        // mapped slot (identity unless the pipe carries an explicit set).
        // No-op in identity/dense mode.
        if self.batch.as_ref().expect("batch enabled").pool.is_some() {
            let (pos0, slot_map) = {
                let p = self.pipe.as_ref().expect("pipe active");
                (p.pos0.clone(), p.slots.clone())
            };
            for (i, &p0) in pos0.iter().enumerate().take(b) {
                let slot = slot_map.as_ref().map_or(i, |s| s[i] as usize);
                self.ensure_slot_blocks(slot, p0 as usize + tick as usize)?;
            }
        }
        let ring = (tick % 2) as usize;
        let (par, tpar, any_base, any5, any6) = Self::pack_samp_par(plans);
        let fold_off = Self::samp_fold_off();
        let any_trunc = any5 || any6;
        let mrope = {
            let p = self.pipe.as_ref().expect("pipe active");
            Self::pipe_mrope_host(&p.pos0, &p.delta, tick)
        };
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let max_batch = self.batch.as_ref().expect("batch enabled").max_batch;
        // sampler params into this tick's ring plane
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let off = ring * max_batch * 4;
            let mut v = bs.d_pipe_par.slice_mut(off..off + b * 4);
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if any_trunc {
                let mut t = bs.d_pipe_tpar.slice_mut(off..off + b * 4);
                exec.stream.memcpy_htod(&tpar, &mut t).map_err(drv)?;
            }
        }
        // advance the DEVICE-dependent inputs from the previous ring's out plane
        if advance {
            let prev = ((tick + 1) % 2) as usize;
            let bs = self.batch.as_mut().expect("batch enabled");
            let (out, tok, pos) = (&bs.d_pipe_out, &mut bs.d_tokens, &mut bs.d_pos);
            exec.pipe_advance(out, prev * max_batch, tok, pos, b)?;
        }
        // rebuild mrope (deterministic positions) - stream-ordered before the
        // replay, so the captured graph reads this tick's RoPE positions
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let mut v = bs.d_mrope.slice_mut(0..4 * b);
            exec.stream.memcpy_htod(&mrope, &mut v).map_err(drv)?;
        }
        // replay the same per-B step graph the classic path uses
        self.batch.as_ref().expect("batch enabled").graphs[&b]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("pipe step graph launch: {e}")))?;
        // fused device sampling into this tick's out ring plane (folded:
        // launch only the chains this tick's row modes need)
        if any_base || fold_off {
            let bs = self.batch.as_mut().expect("batch enabled");
            let (logits, par_buf, out) = (&bs.d_logits, &bs.d_pipe_par, &mut bs.d_pipe_out);
            exec.sample_rows_at(
                logits,
                par_buf,
                ring * max_batch * 4,
                out,
                ring * max_batch,
                b,
                vocab,
            )?;
        }
        if any_trunc {
            Self::samp_fold_witness(any_base, any5, any6);
            // P67b: mode-5 rows sample in the same out ring plane - the
            // pipe_advance of tick N+1 consumes them device-side unchanged.
            // Engagement witness (bisect-trap law; its absence cost a
            // diagnosis round on): once per process.
            static PIPE5: std::sync::Once = std::sync::Once::new();
            PIPE5.call_once(|| {
                eprintln!("[trunc-pipe] engaged: b={b} (mode-5 rows in the decode pipe)");
            });
            let bs = self.batch.as_mut().expect("batch enabled");
            let (logits, par_buf, tpar_buf) = (&bs.d_logits, &bs.d_pipe_par, &bs.d_pipe_tpar);
            if any5 || fold_off {
                exec.sample_rows_t_at(
                    logits,
                    par_buf,
                    ring * max_batch * 4,
                    tpar_buf,
                    ring * max_batch * 4,
                    &mut bs.d_pipe_out,
                    ring * max_batch,
                    b,
                    vocab,
                )?;
            }
            if any6 || fold_off {
                exec.sample_rows_p_at(
                    logits,
                    par_buf,
                    ring * max_batch * 4,
                    tpar_buf,
                    ring * max_batch * 4,
                    &mut bs.d_pipe_out,
                    ring * max_batch,
                    b,
                    vocab,
                )?;
            }
        }
        let ev = exec.record_event()?;
        self.pipe.as_mut().expect("pipe active").ev[ring] = Some(ev);
        Ok(())
    }

    /// Begin a pipelined pure-decode over the same rows `forward_batch_sampled`
    /// would take (all Device/Hole plans - no Host rows). No ids come back yet;
    /// the first `decode_pipe_next` returns tick 0's while tick 1 runs.
    pub fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        self.with_decode_lane(|m| m.decode_pipe_begin_inner(tokens, positions, None, plans))
    }

    /// `decode_pipe_begin` over an ARBITRARY slot set: row i drives
    /// `slots[i]` instead of slot i. The overlap scheduler uses this for the
    /// churn-phase decode set (the non-chunking slots - never contiguous).
    /// The mapping is written to `d_slots` once here; the captured graph
    /// reads `d_slots` per replay, so every tick sees it. Identity is
    /// restored at drain/abort.
    pub fn decode_pipe_begin_slots(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        self.with_decode_lane(|m| m.decode_pipe_begin_inner(tokens, positions, Some(slots), plans))
    }

    fn decode_pipe_begin_inner(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        assert_eq!(positions.len(), b, "one position per row");
        if !self.supports_decode_pipe() {
            return Err(GpuModelError::Unsupported("decode pipe".into()));
        }
        // Split-decode companion (see batch_sampled_impl): the pipe graph
        // cannot split, so above the fast arms' b <= 64 bound it would
        // replay slow eager lanes (~62ms ticks at 128 rows). Decline; the
        // service falls back to classic sampled ticks, which split.
        if b > f8t_dec_bmax() && paddock_models::dev_var_os!("PADDOCK_NO_DEC_SPLIT").is_none() {
            return Err(GpuModelError::Unsupported(
                "decode pipe above the fast-arm bound".into(),
            ));
        }
        match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) if b > bs.max_batch => {
                return Err(GpuModelError::BatchTooLarge {
                    got: b,
                    max: bs.max_batch,
                });
            }
            _ => {}
        }
        if let Some(s) = slots {
            assert_eq!(s.len(), b, "one slot per row");
        }
        assert!(self.pipe.is_none(), "decode pipe already active");
        self.ensure_scratch(b)?;
        self.ensure_batch_graph(b)?;
        let delta: Vec<i64> = {
            let bs = self.batch.as_ref().expect("batch enabled");
            (0..b)
                .map(|i| bs.mrope_delta[slots.map_or(i, |s| s[i] as usize)])
                .collect()
        };
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        // tick-0 inputs land in the fixed graph buffers (advance=false won't set
        // them); d_slots is identity (pure decode) unless an explicit mapping
        // rides the whole pipe - written once here, restored at drain/abort.
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let mut vt = bs.d_tokens.slice_mut(0..b);
            exec.stream.memcpy_htod(tokens, &mut vt).map_err(drv)?;
            let mut vp = bs.d_pos.slice_mut(0..b);
            exec.stream.memcpy_htod(positions, &mut vp).map_err(drv)?;
            if let Some(s) = slots {
                let mut vs = bs.d_slots.slice_mut(0..b);
                exec.stream.memcpy_htod(s, &mut vs).map_err(drv)?;
            }
        }
        self.pipe = Some(PipeStateQ {
            b,
            tick: 0,
            ev: [None, None],
            pos0: positions.to_vec(),
            delta,
            slots: slots.map(<[u32]>::to_vec),
        });
        if let Err(e) = self.pipe_launch_tick(plans, false) {
            self.pipe_abort();
            return Err(e);
        }
        Ok(())
    }

    /// Enqueue the next tick (its inputs advance on device from the previous
    /// tick's sampler output) and return the ids of the OLDEST in-flight tick,
    /// read via the side stream while the new tick executes. `plans[i]` must be
    /// Device or Hole, same rows as begin.
    pub fn decode_pipe_next(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuModelError> {
        self.with_decode_lane(|m| m.decode_pipe_next_inner(plans))
    }

    fn decode_pipe_next_inner(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let (b, j) = {
            let p = self.pipe.as_ref().ok_or_else(|| {
                GpuModelError::Unsupported("decode_pipe_next without begin".into())
            })?;
            (p.b, p.tick)
        };
        assert_eq!(plans.len(), b, "one plan per row");
        self.pipe.as_mut().expect("pipe active").tick = j + 1;
        if let Err(e) = self.pipe_launch_tick(plans, true) {
            self.pipe_abort();
            return Err(e);
        }
        let ring = (j % 2) as usize;
        let max_batch = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .max_batch;
        let r = {
            let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
            let ev = self.pipe.as_ref().expect("pipe active").ev[ring]
                .as_ref()
                .expect("in-flight tick event");
            exec.to_host_u32_after(ev, &bs.d_pipe_out, ring * max_batch, b)
        };
        match r {
            Ok(ids) => Ok(ids),
            Err(e) => {
                self.pipe_abort();
                Err(e.into())
            }
        }
    }

    /// End the pipe: return the last in-flight tick's ids without enqueueing
    /// more work. Must be called before any other forward call - the fixed
    /// input buffers are left stale and every other path re-uploads them.
    pub fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GpuModelError> {
        self.with_decode_lane(|m| m.decode_pipe_drain_inner())
    }

    fn decode_pipe_drain_inner(&mut self) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let st = self
            .pipe
            .take()
            .ok_or_else(|| GpuModelError::Unsupported("decode_pipe_drain without begin".into()))?;
        let ring = (st.tick % 2) as usize;
        let max_batch = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .max_batch;
        let ev = st.ev[ring].as_ref().expect("in-flight tick event");
        let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
        let r = exec.to_host_u32_after(ev, &bs.d_pipe_out, ring * max_batch, st.b);
        // a slot-mapped pipe leaves real slots in d_slots - restore identity
        // (stream-ordered after the last replay's reads on this same stream)
        // so following pure-decode ticks see row i -> slot i again.
        if st.slots.is_some() {
            let ident: Vec<u32> = (0..st.b as u32).collect();
            let bs = self.batch.as_mut().expect("batch enabled");
            let mut v = bs.d_slots.slice_mut(0..st.b);
            let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
            exec.stream.memcpy_htod(&ident, &mut v).map_err(drv)?;
        }
        match r {
            Ok(ids) => Ok(ids),
            Err(e) => {
                let _ = exec.synchronize(); // state gone - quiesce readers of the rings
                Err(e.into())
            }
        }
    }

    /// Kill an in-flight pipe after an error (or on reset): quiesce the stream so
    /// nothing still reads the pipe buffers, then drop the state.
    pub(super) fn pipe_abort(&mut self) {
        // quiesce the lane the pipe actually ran on
        self.with_decode_lane(|m| {
            if let Some(st) = m.pipe.take() {
                let _ = m.exec.synchronize();
                // best-effort identity restore after a slot-mapped pipe (the
                // stream is quiesced, so this can't race a replay)
                if st.slots.is_some()
                    && let Some(bs) = m.batch.as_mut()
                {
                    let ident: Vec<u32> = (0..st.b as u32).collect();
                    let mut v = bs.d_slots.slice_mut(0..st.b);
                    let _ = m.exec.stream.memcpy_htod(&ident, &mut v);
                }
            }
        })
    }

    /// `launch_batch_step` with an optional explicit slot mapping. `None` = the
    /// identity mapping (row i drives slot i - the pure batched-decode path).
    /// `Some(slots)` maps row i to `slots[i]`: the batched-step kernels already
    /// take `bs.d_slots` for per-slot KV/state/pool indexing, so a mixed tick's
    /// compacted decode rows (arbitrary slots) ride the same captured graph - we
    /// just write the real slots into `bs.d_slots` before the replay and restore
    /// identity after (so a following pure-decode tick is unaffected).
    fn launch_batch_step_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
    ) -> Result<(), GpuModelError> {
        self.with_decode_lane(|m| m.launch_batch_step_slots_inner(tokens, positions, slots))
    }

    fn launch_batch_step_slots_inner(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
    ) -> Result<(), GpuModelError> {
        let b = tokens.len();
        assert_eq!(b, positions.len());
        if let Some(s) = slots {
            assert_eq!(s.len(), b, "one slot per row");
        }
        assert!(self.batch.is_some(), "enable_batch first");
        assert!(b >= 1 && b <= self.batch.as_ref().expect("batch enabled").max_batch);
        self.ensure_scratch(b)?;
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        {
            let bs = self.batch.as_mut().expect("batch enabled");
            // mrope delta is per-SLOT: identity indexes by row i, slot-explicit
            // by the row's actual slot (text sequences carry delta 0 either way).
            let mrope_host: Vec<u32> = (0..4)
                .flat_map(|_| {
                    positions.iter().enumerate().map(|(i, &p)| {
                        let d = match slots {
                            Some(s) => bs.mrope_delta[s[i] as usize],
                            None => bs.mrope_delta[i],
                        };
                        (p as i64 + d) as u32
                    })
                })
                .collect();
            exec.stream
                .memcpy_htod(tokens, &mut bs.d_tokens)
                .map_err(drv)?;
            exec.stream
                .memcpy_htod(positions, &mut bs.d_pos)
                .map_err(drv)?;
            if let Some(s) = slots {
                let mut v = bs.d_slots.slice_mut(0..b);
                exec.stream.memcpy_htod(s, &mut v).map_err(drv)?;
            }
            // d_mrope holds 4*max_batch; write the leading 4*b axis-major view
            let mut staged = vec![0u32; 4 * b];
            staged.copy_from_slice(&mrope_host);
            exec.stream
                .memcpy_htod(&staged, &mut bs.d_mrope)
                .map_err(drv)?;
        }

        // P5 budget pool: grow each row's ACTUAL slot block table to back the
        // token it writes this step (positions[i]) before the (captured) compute
        // reads the device table. The upload is outside the graph, so growth is
        // replay-safe. No-op in identity/dense mode.
        if self.batch.as_ref().expect("batch enabled").pool.is_some() {
            for (i, &p) in positions.iter().enumerate() {
                let slot = slots.map_or(i, |s| s[i] as usize);
                self.ensure_slot_blocks(slot, p as usize)?;
            }
        }

        // Capture the fixed-B step once, then replay per token - the compute is
        // grid-stable for a given B (loop bounds like the KV length come from the
        // device position buffer, exactly as the single-stream graph relies on).
        self.ensure_batch_graph(b)?;
        self.batch.as_ref().expect("batch enabled").graphs[&b]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("batch graph launch: {e}")))?;
        // restore the identity d_slots (stream-ordered after the graph read it)
        // so a following pure-decode tick sees row i -> slot i again.
        if slots.is_some() {
            let ident: Vec<u32> = (0..b as u32).collect();
            let bs = self.batch.as_mut().expect("batch enabled");
            let mut v = bs.d_slots.slice_mut(0..b);
            exec.stream.memcpy_htod(&ident, &mut v).map_err(drv)?;
        }
        exec.synchronize()?;
        Ok(())
    }

    /// The batched per-step compute (device-resident inputs, capture-safe ops
    /// only) - everything `forward_batch` replays via its per-B graph.
    fn record_batch_step(&mut self, b: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (state_size, n_k_heads, n_v_heads, conv_k) =
            (self.state_size, self.n_k_heads, self.n_v_heads, self.conv_k);
        let (conv_dim, ff, max_ctx) = (self.conv_dim, self.ff, self.max_ctx);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let sinks = &self.sinks;
        let layers = &self.layers;
        let bs_gu = &self.bs_gu;
        let bs_dn = &self.bs_dn;
        let bs_f8ffn = &self.bs_f8ffn;
        // b=1 lin-GEMV arm (non-KV-overhead R2.2) - OPT-IN until it earns the
        // default. Correct in serving and it wins the GPU-time ledger (the FFN
        // band is ~4% cheaper: 1670 vs 1723 ms of kernel time, and the whole
        // run is 168 ms lighter), but c1 wall clock is 0.84-1.1% slower in
        // state-matched A/Bs. GPU work down + wall time up = ~370 us/token of
        // idle this arm introduces and profiling did not
        // localize (graph replay is identical in both arms: 132 cuGraphLaunch,
        // so it is not a capture fallback; neither kernel uses PDL). The
        // reclaim this rung exists to unlock (~19.5 GB of Q8 FFN+head twins)
        // stays blocked until that gap is understood - buying VRAM with an
        // unexplained decode regression is the trade nobody asked for.
        let lin_gemv_on =
            exec.has_f8lin_gemv() && paddock_models::dev_var_os!("PADDOCK_NO_LIN_GEMV").is_none();
        let lin_dn_off = self.lin_tick_dn_off();

        let bs_w8_dec = &self.bs_w8;
        // decode norm+e4m3 fuse: stage the group e4m3 inside the norms when
        // the f8 lane serves this model (arms skip their d_xn quantizes)
        let e4m3_norms = !self.bs_f8ffn.is_empty() && b >= 8 && exec.has_add_rmsnorm_e4m3_xn();
        let tok_embd = &self.tok_embd;
        // b=1 serving class for k-quant weights: the W4A8 dp4a GEMV (mmvq
        // design point - llama's own decode class; the exact-f32 GEMV measured
        // issue-bound ~25% under the Q8 ref's byte rate). The exact GEMV stays
        // the ORACLE path: PADDOCK_KQ_EXACT_GEMV=1 pins serving back to it,
        // and the serial spine keeps it unconditionally.
        let kq_w4a8_b1 = self.kq_resident
            && self.exec.has_kquant_gemv_w4a8()
            && paddock_models::dev_var_os!("PADDOCK_KQ_EXACT_GEMV").is_none();
        let sc = self.scratch.as_mut().expect("scratch");
        let bs = self.batch.as_mut().expect("batch");

        // Quantize DEDUPE (from a wide-batch glue profile): wq/wk/wv share
        // the normed d_xn, and in_qkv/alpha/beta/gate share it too - yet each
        // bmm quantized it again into the same d_xq/d_xs (3-4 identical launches
        // per layer = the 41k pd_quantize_q8 instances at c32). Quantize once
        // after the norm; the group's bmm calls skip theirs via the `pre` flag.
        // BIT-EXACT: same kernels, same values, same layout - pure elimination.
        // (The gpt-oss-era FUSED rmsnorm_quant_q8_batch kernel was tried here
        // and REGRESSED B=64 decode 9% - its grid is `batch` blocks, so at
        // decode widths it under-fills the 188-SM die with quant work the
        // standalone wide-grid quantize did better. Dedupe keeps the win
        // without the kernel.) PADDOCK_NO_QDEDUP=1 restores per-bmm quantize.
        // The kq W4A8 b=1 class consumes bs.d_xq too, so the dedupe extends
        // to b == 1 on k-quant models (Q8 b=1 keeps the f32 GEMV and never
        // reads the staging - the extra quantize is skipped there).
        let qdedup =
            (b > 1 || kq_w4a8_b1) && paddock_models::dev_var_os!("PADDOCK_NO_QDEDUP").is_none();

        // batched matmul: quantize activations once per input, dp4a GEMM. At B=1
        // the plain decode GEMV wins (no quantize launch, peak single-row BW) and
        // keeps the degenerate batch numerically identical to the single path.
        // The optional 4th arg marks the input as already quantized into
        // bs.d_xq/d_xs (the fused norm+quant above) - the quantize is skipped.
        //
        // QuantW dispatch wrapper: Q8_0 takes the ladder below verbatim; the
        // k-quant arm rides the W4A8 dp4a GEMM off the same strided int8
        // staging (same activation class), with the per-16 sums plane for the
        // Q4/Q5 min term. B=1 rides the W4A8 GEMV (kq_w4a8_b1 above).
        macro_rules! bmmq {
            ($w:expr, $x:expr, $y:expr) => {
                bmmq!($w, $x, $y, false)
            };
            ($w:expr, $x:expr, $y:expr, $pre:expr) => {{
                match $w {
                    QuantW::Q8(q) => bmm!(q, $x, $y, $pre),
                    QuantW::Kq(k) => {
                        if b == 1 {
                            if kq_w4a8_b1 {
                                // fused quantize+sums: one node; $pre means the
                                // norm-site fused node already staged xq/xs/ssums
                                if !$pre {
                                    exec.quantize_q8_sums(
                                        $x,
                                        &mut bs.d_xq,
                                        &mut bs.d_xs,
                                        &mut sc.d_ssums,
                                        k.dims[0],
                                    )?;
                                }
                                let needs = crate::gpu::kq_needs_sums(k.ty);
                                exec.kquant_gemv_w4a8(
                                    k,
                                    &bs.d_xq,
                                    &bs.d_xs,
                                    needs.then_some(&sc.d_ssums),
                                    $y,
                                )?;
                            } else {
                                exec.kquant_gemv(k, $x, $y)?;
                            }
                        } else {
                            if !$pre {
                                exec.quantize_q8($x, &mut bs.d_xq, &mut bs.d_xs, b * k.dims[0])?;
                            }
                            let needs = crate::gpu::kq_needs_sums(k.ty);
                            if needs {
                                exec.q8_sums_strided(&bs.d_xq, &mut sc.d_ssums, k.dims[0], b)?;
                            }
                            // 2..5: the multi-column W4A8 GEMV (weight read at
                            // the b=1 GEMV's 631-648 GB/s class vs dp4a's
                            // 283-557 on these widths). 17..64: the dp4a MT
                            // tile starts z-tiling (2 weight passes) exactly
                            // where the Q8 ladder's K-split mma proved out -
                            // one weight pass for the whole batch, grid.z
                            // K-ranges fill the die. Between them dp4a stays
                            // bandwidth-optimal (1 pass, no combine).
                            if !kq_nc_off()
                                && exec.has_kquant_gemv_w4a8_nc()
                                && GpuExecutor::kquant_gemv_w4a8_nc_fits(k, b)
                            {
                                exec.kquant_gemv_w4a8_nc(
                                    k,
                                    &bs.d_xq,
                                    &bs.d_xs,
                                    needs.then_some(&sc.d_ssums),
                                    $y,
                                    b,
                                )?;
                            } else if b > kq_ks_min_batch(exec.sm_count())
                                && b <= 64
                                && exec.has_kquant_mma_ks()
                            {
                                exec.kquant_gemm_mma_ks(
                                    k,
                                    &bs.d_xq,
                                    &bs.d_xs,
                                    needs.then_some(&sc.d_ssums),
                                    &mut bs.d_ks_part,
                                    $y,
                                    b,
                                )?;
                            } else {
                                exec.kquant_gemm_dp4a(
                                    k,
                                    &bs.d_xq,
                                    &bs.d_xs,
                                    needs.then_some(&sc.d_ssums),
                                    $y,
                                    b,
                                )?;
                            }
                        }
                    }
                }
            }};
        }
        macro_rules! bmm {
            ($w:expr, $x:expr, $y:expr) => {
                bmm!($w, $x, $y, false)
            };
            ($w:expr, $x:expr, $y:expr, $pre:expr) => {{
                if b == 1 {
                    exec.q8_0_gemv_repacked($w, None, $x, $y)?;
                } else if b <= 4 {
                    // mmvq shape: gemv-class weight bandwidth for few columns
                    if !$pre {
                        exec.quantize_q8($x, &mut bs.d_xq, &mut bs.d_xs, b * $w.dims[0])?;
                    }
                    exec.q8_0_gemv_dp4a_nc($w, &bs.d_xq, &bs.d_xs, $y, b)?;
                } else if b <= 64 && (b >= ks_min_batch() || (b >= 8 && $w.dims[1] <= 4096)) {
                    // K-split mma (G4's pd_q8_0_gemm_mma_ks): the plain mma grid
                    // is N-tiles only (wk: 16 blocks) and idles a 188-SM die; the
                    // z-split partial planes + fixed-order combine won every qwen
                    // shape at B>=24 on GB202 (wk B=32: 18.4 -> 10.3 us, ffn_down
                    // B=48: 110 -> 61, in_qkv B=64: 77.8 -> 42.6) and the narrow
                    // out_dims (<= 4096: wk/wv/gate/ssm_out/ffn_down) from B=8
                    // already. Deterministic fixed-order combine; kbench ladder
                    // in qwen35_kbench reproduces the crossover.
                    if !$pre {
                        exec.quantize_q8($x, &mut bs.d_xq, &mut bs.d_xs, b * $w.dims[0])?;
                    }
                    exec.q8_0_gemm_mma_ks($w, &bs.d_xq, &bs.d_xs, &mut bs.d_ks_part, $y, b)?;
                } else if b <= MMA_MIN_BATCH {
                    // dp4a MT: bandwidth-optimal while the batch fits 1-2 weight
                    // passes. (A 32-rows/pass 1-row-per-warp wide variant tried for
                    // B>16 REGRESSED 600->522 @ B=32.)
                    if !$pre {
                        exec.quantize_q8($x, &mut bs.d_xq, &mut bs.d_xs, b * $w.dims[0])?;
                    }
                    exec.q8_0_gemm_mt_dp4a($w, &bs.d_xq, &bs.d_xs, $y, b)?;
                } else {
                    // tensor-core MMA: at B>=64 the dp4a INT pipe saturates and its
                    // z-tiling re-reads the weight ceil(B/24)x; the 64-wide MMA tile
                    // does the dots on tensor cores and re-reads only ceil(B/64)x
                    // (measured 1.8x on lm_head @ B=64). Same per-block-scale class.
                    if !$pre {
                        exec.quantize_q8($x, &mut bs.d_xq, &mut bs.d_xs, b * $w.dims[0])?;
                    }
                    exec.q8_0_gemm_mma($w, &bs.d_xq, &bs.d_xs, $y, b)?;
                }
            }};
        }

        embed_any(&exec, tok_embd, &bs.d_tokens, &mut sc.d_x, embd, b)?;

        // P71: hoist the per-layer FFN residual (x += d_proj, one
        // pd_add_inplace per layer) into the next layer's pre-norm via the
        // add_ twins of all three pre-norm arms - d_proj is untouched
        // between the loop bottom and the next pre-norm, and d_x is
        // re-embedded every tick so the pending residual always resolves
        // in-tick (final norm eats the last layer's). Same f32 add + the
        // P63 width-stable f64 sums = the chain's numeric class.
        // Kill: PADDOCK_NO_RES_HOIST restores the separate add.
        static RES_HOIST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // A DFlash drafter taps the RAW residual at its target layers (loop
        // top) - a hoisted residual would hand the tap a stale d_x, the
        // silent-fuse trap the gemma4 template asserts against. Drafter
        // correctness outranks the hoist's launch saving.
        let res_hoist = self.dflash.is_none()
            && exec.has_add_rmsnorm_e4m3_row()
            && *RES_HOIST.get_or_init(|| {
                let on = paddock_models::dev_var_os!("PADDOCK_NO_RES_HOIST").is_none();
                if on {
                    eprintln!("[res-hoist] engaged (per-layer residual folded into next pre-norm)");
                }
                on
            });
        let mut pending_res = false;
        // DFlash feature taps: fold the residual ENTERING each target layer
        // into the drafter's fusion accumulator, mid-walk (captured with the
        // step graph - the drafter arms at enable_batch, before any capture).
        let mut dtap = self
            .dflash
            .as_mut()
            .filter(|d| d.state.is_some() && !super::dflash::fuse_off());

        for (li, layer) in layers.iter().enumerate() {
            if let Some(df) = dtap.as_mut()
                && let Some(band) = df.target_layers.iter().position(|&t| t == li)
            {
                debug_assert!(!pending_res, "dflash tap with a hoisted residual");
                super::dflash::tap_band(&exec, df, &sc.d_x, band, embd, b)?;
            }
            // Does the f8t stack own every consumer of this layer's d_xn?
            // mixer + FFN + (since the head moved to f8t) lm_head, and for a
            // DeltaNet layer also alpha/beta -- those fall back to ab_gate on
            // d_xn when the export did not ship them Q8_0, which is the one
            // case the fold cannot cover. When all of that holds, nothing
            // reads d_xn or bs.d_xq/d_xs, so the qdedup staging is dead work
            // AND the fused norms (which never materialise d_xn) are legal at
            // batched widths, not just at b == 1.
            // Kill: PADDOCK_NO_QDEDUP_SKIP.
            let dn_ab_ok = match &layer.mixer {
                Mixer::Full(_) => true,
                Mixer::Linear(w) => w.alpha_w.is_some() && w.beta_w.is_some(),
            };
            // The ATTN half of that question, split out. d_xn's
            // lifetime runs attn_norm -> post_norm, and nothing inside it is an
            // FFN consumer: the FFN reads the POST-norm's d_xn, which the xn
            // arm below still writes. So an Nvf4Dense layer - whose fp4 lane
            // genuinely needs f32 d_xn at the post-norm - can still take the
            // fused row-e4m3 pre-norm, and skip the qdedup q8 staging with it.
            // That was 56 of 64 layers paying xn(7.5) + q8(3.5) + row1pc(2.7)
            // where the 8 Dense layers pay one 5.7 us kernel: 8.0 us x 64
            // sites = ~540 us/tick at c16.
            // Kill: PADDOCK_NO_QWEN_ATTN_COVERS (reverts to the layer-wide gate).
            let f8t_covers_attn_side = b <= f8t_dec_bmax()
                && self.bs_f8t_attn.get(li).and_then(|o| o.as_ref()).is_some()
                && bs.d_dn_fused.is_some()
                && bs.d_gu_fused.is_some()
                && dn_ab_ok
                && self.out_f8t.is_some()
                && paddock_models::dev_var_os!("PADDOCK_NO_QDEDUP_SKIP").is_none();
            let f8t_layer_covers = f8t_covers_attn_side
                && matches!(&layer.ffn, Ffn::Dense { .. })
                && self.bs_f8t_ffn.get(li).and_then(|o| o.as_ref()).is_some();
            let f8t_attn_covers =
                if paddock_models::dev_var_os!("PADDOCK_NO_QWEN_ATTN_COVERS").is_none() {
                    f8t_covers_attn_side
                } else {
                    f8t_layer_covers
                };

            // Pre-norm -> row-e4m3 in one kernel (gemma4 has had this trio for
            // a while; qwen35 got the f8t GEMM lane before the epilogues).
            // Only legal when the f8t mixer arm is CERTAIN to run, because the
            // fused kernel never materialises d_xn in f32 and every fallback
            // arm below reads it. The arm's predicate is known here, so the
            // choice is exact rather than optimistic.
            // Without PDL: rmsnorm 4.24 us + quantize 2.66 us -> one launch.
            let f8t_prenorm = self
                .bs_f8t_attn
                .get(li)
                .and_then(|o| o.as_ref())
                .is_some()
                && b <= f8t_dec_bmax()
                // both landings, because this layer may be either mixer kind
                // (Full lands in d_gu_fused, Linear in d_dn_fused) and both
                // are allocated together whenever any f8t attn plane exists
                && bs.d_dn_fused.is_some()
                && bs.d_gu_fused.is_some()
                && exec.has_rmsnorm_e4m3_row()
                && (!qdedup || f8t_attn_covers)
                && paddock_models::dev_var_os!("PADDOCK_NO_QWEN_PRENORM_FUSE").is_none();
            if f8t_prenorm {
                if pending_res {
                    exec.add_rmsnorm_e4m3_row(
                        &mut sc.d_x,
                        &sc.d_proj,
                        &layer.attn_norm.buf,
                        &mut sc.d_f8t_q,
                        &mut sc.d_f8t_rs,
                        embd,
                        eps,
                        b,
                    )?;
                } else {
                    exec.rmsnorm_e4m3_row(
                        &sc.d_x,
                        &layer.attn_norm.buf,
                        &mut sc.d_f8t_q,
                        &mut sc.d_f8t_rs,
                        embd,
                        eps,
                        b,
                    )?;
                }
            } else if e4m3_norms {
                let proj = if pending_res { Some(&sc.d_proj) } else { None };
                exec.add_rmsnorm_e4m3_xn(
                    &mut sc.d_x,
                    proj,
                    &layer.attn_norm.buf,
                    &mut sc.d_xn,
                    &mut sc.d_pxq,
                    &mut sc.d_exs,
                    embd,
                    b,
                    eps,
                )?;
            } else if pending_res {
                exec.add_rmsnorm_batch(
                    &mut sc.d_x,
                    &sc.d_proj,
                    &layer.attn_norm.buf,
                    &mut sc.d_xn,
                    embd,
                    eps,
                    b,
                )?;
            } else {
                exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, b)?;
            }
            pending_res = false;
            if qdedup && !f8t_attn_covers {
                // one quantize serves every xn consumer in this layer's group;
                // the b=1 kq tick uses the fused variant so the group's per-16
                // sums land in the same node
                if b == 1 && kq_w4a8_b1 {
                    exec.quantize_q8_sums(
                        &sc.d_xn,
                        &mut bs.d_xq,
                        &mut bs.d_xs,
                        &mut sc.d_ssums,
                        embd,
                    )?;
                } else {
                    exec.quantize_q8(&sc.d_xn, &mut bs.d_xq, &mut bs.d_xs, b * embd)?;
                }
            }
            match &layer.mixer {
                Mixer::Full(w) => {
                    // decode projections on the f8 lane: the W8
                    // planes (resident by default, same f8w format the f8d
                    // rung eats) serve wq/wk/wv/wo at decode widths - the
                    // last GEMM band still paying Q8_0's 1.0625 B/param.
                    let l8d = bs_w8_dec
                        .get(li)
                        .filter(|_| b >= super::f8_dec_min() && !bs_f8ffn.is_empty());
                    // wq holds the fused wq|wk|wv plane since the merge
                    let f8_qkv = l8d.is_some_and(|l| l.wq.is_some());
                    // set when the qkv arm below took the tile lane; wo then
                    // follows it (one plane pair, one precision class per layer)
                    let mut f8t_wo: Option<&crate::gpu::F8TilePlane> = None;
                    // set when q36_qkg_nra_rows produced qn/gate/pools in one
                    // launch - the norm/rope/append chain below is then skipped
                    let mut attn_nra_done = false;
                    // tcgen05 mixer lane (PADDOCK_QWEN_F8T). First, and with no
                    // b>=8 floor: the f8 rungs below are all batch-gated, so at
                    // c1/c4 serving widths they never fire and the projections
                    // ride Q8_0 int8-MMA - the exact trap the FFN half hit.
                    let f8t_qkv = self
                        .bs_f8t_attn
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| b <= f8t_dec_bmax() && bs.d_gu_fused.is_some());
                    if let Some([qkv_t, wo_t]) = f8t_qkv {
                        let (nq, nk2, nv2) = (w.wq.dims()[1], w.wk.dims()[1], w.wv.dims()[1]);
                        let nt = nq + nk2 + nv2;
                        // already produced by the fused pre-norm above
                        if !f8t_prenorm {
                            exec.quantize_e4m3_row(
                                &sc.d_xn,
                                &mut sc.d_f8t_q,
                                &mut sc.d_f8t_rs,
                                embd,
                                b,
                            )?;
                        }
                        let fused = bs.d_gu_fused.as_mut().expect("f8t qkv landing");
                        exec.f8t_gemm(
                            qkv_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            fused,
                            embd,
                            nt,
                            b,
                        )?;
                        // decode twin of the prefill Phase-A consumer
                        // (norm band): one pd_q36_qkg_nra_rows
                        // over the fused plane replaces row_slice4 + split_qg
                        // + 2x rmsnorm_batch + 2x mrope + 2x paged append -
                        // 7 launches/layer, bit-identical planes + pools.
                        // The nra kernel is PDL-armed while the chain it
                        // replaces is plain-launched - judge it end-to-end,
                        // per the second PDL law. PADDOCK_NO_QNF_DEC
                        // kills to the chain.
                        static QNF_DEC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let qnf_dec = bs.paged
                            && head_dim == 256
                            && n_rot == 64
                            && exec.has_q36_qkg_nra_rows()
                            && *QNF_DEC.get_or_init(|| {
                                paddock_models::dev_var_os!("PADDOCK_NO_QNF_DEC").is_none()
                            });
                        if qnf_dec {
                            let bt = bs.d_block_tables.as_ref().expect("paged block tables");
                            let bps = bs.blocks_per_slot;
                            exec.q36_qkg_nra_rows(
                                fused,
                                0,
                                nt,
                                nq,
                                nq + nk2,
                                &w.q_norm.buf,
                                &w.k_norm.buf,
                                &mut sc.d_qn,
                                &mut sc.d_gate,
                                bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                &bs.d_pos,
                                Some(&bs.d_slots),
                                &bs.d_mrope,
                                bt,
                                bps,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                n_rot,
                                eps,
                                yarn,
                                sections,
                                b,
                                self.kv_dtype,
                            )?;
                            attn_nra_done = true;
                        } else if exec.has_row_slice4() {
                            exec.row_slice4(
                                fused,
                                nt,
                                b,
                                &mut [
                                    (&mut sc.d_qg, 0, nq),
                                    (&mut sc.d_k, nq, nk2),
                                    (&mut sc.d_v, nq + nk2, nv2),
                                ],
                            )?;
                            exec.split_qg(
                                &sc.d_qg,
                                &mut sc.d_q,
                                &mut sc.d_gate,
                                b,
                                n_heads,
                                head_dim,
                            )?;
                        } else {
                            exec.row_slice(fused, &mut sc.d_qg, nt, 0, nq, b)?;
                            exec.row_slice(fused, &mut sc.d_k, nt, nq, nk2, b)?;
                            exec.row_slice(fused, &mut sc.d_v, nt, nq + nk2, nv2, b)?;
                            exec.split_qg(
                                &sc.d_qg,
                                &mut sc.d_q,
                                &mut sc.d_gate,
                                b,
                                n_heads,
                                head_dim,
                            )?;
                        }
                        f8t_wo = Some(wo_t);
                    } else if f8_qkv {
                        let l8 = l8d.expect("f8_qkv checked above");
                        if !e4m3_norms {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, b * embd)?;
                        }
                        let (nq, nk2, nv2) = (w.wq.dims()[1], w.wk.dims()[1], w.wv.dims()[1]);
                        let nt = nq + nk2 + nv2;
                        // One fused qkv GEMM (vLLM's 14336-out merge; -32
                        // launches/tick) into the gu landing (free at mixer
                        // time) + row_slice x3
                        let fused = bs.d_gu_fused.as_mut().expect("qkv fused landing");
                        exec.f8d_gemm_mma_ks(
                            l8.wq.as_ref().expect("full-attn W8 qkv plane"),
                            w.wq.dims()[0],
                            nt,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut bs.d_ks_part,
                            fused,
                            b,
                        )?;
                        exec.row_slice(fused, &mut sc.d_qg, nt, 0, nq, b)?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, b, n_heads, head_dim)?;
                        exec.row_slice(fused, &mut sc.d_k, nt, nq, nk2, b)?;
                        exec.row_slice(fused, &mut sc.d_v, nt, nq + nk2, nv2, b)?;
                    } else {
                        // b=1 q|k|v launch merge (entry 317): three same-input
                        // GEMVs pay three fixed ramp/drain tolls where wk/wv's
                        // small grids can't cover the die; one merged grid streams
                        // all three planes at the big-launch byte rate.
                        // Bit-identical per row to the splits (the multi kernel
                        // runs the exact single-plane body). Same merge the
                        // granite/laguna decode bands ship; the fp8 W8 planes
                        // fused these projections at load while the Q8 b=1 path
                        // paid per-plane since bring-up.
                        // Kill: PADDOCK_NO_GEMV_MULTI.
                        match (&w.wq, &w.wk, &w.wv) {
                            (QuantW::Q8(wq), QuantW::Q8(wk), QuantW::Q8(wv))
                                if b == 1
                                    && exec.has_q8_0_gemv_repacked_multi()
                                    && !super::no_gemv_multi() =>
                            {
                                exec.q8_0_gemv_repacked_multi(
                                    &mut [(wq, &mut sc.d_qg), (wk, &mut sc.d_k), (wv, &mut sc.d_v)],
                                    &sc.d_xn,
                                )?;
                                exec.split_qg(
                                    &sc.d_qg,
                                    &mut sc.d_q,
                                    &mut sc.d_gate,
                                    b,
                                    n_heads,
                                    head_dim,
                                )?;
                            }
                            _ => {
                                bmmq!(&w.wq, &sc.d_xn, &mut sc.d_qg, qdedup);
                                exec.split_qg(
                                    &sc.d_qg,
                                    &mut sc.d_q,
                                    &mut sc.d_gate,
                                    b,
                                    n_heads,
                                    head_dim,
                                )?;
                                bmmq!(&w.wk, &sc.d_xn, &mut sc.d_k, qdedup);
                                bmmq!(&w.wv, &sc.d_xn, &mut sc.d_v, qdedup);
                            }
                        }
                    }
                    if !attn_nra_done {
                        exec.rmsnorm_batch(
                            &sc.d_q,
                            &w.q_norm.buf,
                            &mut sc.d_qn,
                            head_dim,
                            eps,
                            b * n_heads,
                        )?;
                        exec.rmsnorm_batch(
                            &sc.d_k,
                            &w.k_norm.buf,
                            &mut sc.d_kn,
                            head_dim,
                            eps,
                            b * n_kv_heads,
                        )?;
                        exec.mrope(
                            &mut sc.d_qn,
                            &bs.d_mrope,
                            b,
                            n_heads,
                            head_dim,
                            n_rot,
                            yarn,
                            sections,
                        )?;
                        exec.mrope(
                            &mut sc.d_kn,
                            &bs.d_mrope,
                            b,
                            n_kv_heads,
                            head_dim,
                            n_rot,
                            yarn,
                            sections,
                        )?;
                        if bs.paged {
                            let bt = bs.d_block_tables.as_ref().expect("paged block tables");
                            let bps = bs.blocks_per_slot;
                            exec.kv_append_batch_paged(
                                &sc.d_kn,
                                bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                &bs.d_pos,
                                Some(&bs.d_slots),
                                bt,
                                bps,
                                kv_dim,
                                b,
                                self.kv_dtype,
                            )?;
                            exec.kv_append_batch_paged(
                                &sc.d_v,
                                bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                &bs.d_pos,
                                Some(&bs.d_slots),
                                bt,
                                bps,
                                kv_dim,
                                b,
                                self.kv_dtype,
                            )?;
                        } else {
                            exec.kv_append_batch(
                                &sc.d_kn,
                                bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                                &bs.d_pos,
                                Some(&bs.d_slots),
                                kv_dim,
                                max_ctx,
                                b,
                                self.kv_dtype,
                            )?;
                            exec.kv_append_batch(
                                &sc.d_v,
                                bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                                &bs.d_pos,
                                Some(&bs.d_slots),
                                kv_dim,
                                max_ctx,
                                b,
                                self.kv_dtype,
                            )?;
                        }
                    }
                    attn_decode_dispatch(
                        &exec,
                        &sc.d_qn,
                        bs.kv_k[li].as_ref().expect("full-attn layer KV"),
                        bs.kv_v[li].as_ref().expect("full-attn layer KV"),
                        sinks,
                        &mut sc.d_attn_o,
                        &mut sc.d_attn_ml,
                        &mut sc.d_attn,
                        &bs.d_pos,
                        Some(&bs.d_slots),
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        b,
                        scale,
                        self.kv_dtype,
                        bs.d_block_tables
                            .as_ref()
                            .filter(|_| bs.paged)
                            .map(|bt| (bt, bs.blocks_per_slot)),
                    )?;
                    exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, b * q_dim)?;
                    if let Some(wo_t) = f8t_wo {
                        exec.quantize_e4m3_row(
                            &sc.d_attn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            w.wo.dims()[0],
                            b,
                        )?;
                        exec.f8t_gemm(
                            wo_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            w.wo.dims()[0],
                            w.wo.dims()[1],
                            b,
                        )?;
                    } else if let Some(w8o) = l8d.and_then(|l| l.wo.as_ref()).filter(|_| f8_qkv) {
                        exec.quantize_e4m3(
                            &sc.d_attn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            b * w.wo.dims()[0],
                        )?;
                        exec.f8d_gemm_mma_ks(
                            w8o,
                            w.wo.dims()[0],
                            w.wo.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            b,
                        )?;
                    } else {
                        bmmq!(&w.wo, &sc.d_attn, &mut sc.d_proj);
                    }
                }
                Mixer::Linear(w) => {
                    // DN in_proj fusion: merged in_qkv|gate_w plane at decode
                    // widths - One 256-tile ks GEMM (nz=1) + row-slice split
                    // into d_mixed/d_z, replacing two GEMMs + combines
                    // (vLLM's exact DN merge). d_z is filled EARLY (it only
                    // reads d_xn, untouched until post_norm) and the gate_w
                    // GEMM below is skipped. b<8 keeps the per-tensor rungs.
                    let dnf = bs_dn
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| b >= super::f8_dec_min());
                    // tcgen05 twin of the merge, ahead of it for the same
                    // reason as the qkv arm: every rung below floors at b>=8.
                    // Same row-slice split, so d_mixed/d_z land identically.
                    let f8t_dn = self
                        .bs_f8t_attn
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| b <= f8t_dec_bmax() && bs.d_dn_fused.is_some());
                    let mut f8t_out_w: Option<&crate::gpu::F8TilePlane> = None;
                    // set when the plane carries the alpha||beta tile block, so
                    // the two decay gemvs below are skipped
                    let mut dn_ab_done = false;
                    // set when row_slice2_gate produced g/beta at the split
                    // (fold) - the separate delta_gate is then skipped
                    let mut dn_gate_done = false;
                    // set when the b=1 entry-317 merge landed d_z alongside
                    // d_mixed (the same early-z hoist the f8 lanes do: d_z
                    // only reads d_xn, untouched until post_norm) - the late
                    // gate_w GEMV below is then skipped
                    let mut dn_z_early = false;
                    // P71-R2: no split at all - conv/z/gate read the fused
                    // plane strided (offsets carried below)
                    let mut dn_strided = false;
                    let (mut dn_tot, mut dn_z_off, mut dn_ab_off) = (0usize, 0usize, 0usize);
                    let dn_fused_done = if let Some([in_t, ow_t]) = f8t_dn {
                        let (nin, nc) = (w.in_qkv.dims()[0], w.in_qkv.dims()[1]);
                        let nz_ = w.gate_w.dims()[1];
                        // the plane's scale length is its out_dim, so the fold
                        // is self-describing -- no parallel per-layer state
                        let tot = in_t.scale.len();
                        dn_ab_done = tot == nc + nz_ + 128;
                        // already produced by the fused pre-norm above
                        if !f8t_prenorm {
                            exec.quantize_e4m3_row(
                                &sc.d_xn,
                                &mut sc.d_f8t_q,
                                &mut sc.d_f8t_rs,
                                nin,
                                b,
                            )?;
                        }
                        let fused = bs.d_dn_fused.as_mut().expect("f8t dn landing");
                        exec.f8t_gemm(
                            in_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            fused,
                            nin,
                            tot,
                            b,
                        )?;
                        // one launch for the whole split (4 parts with the ab
                        // fold, 2 without) instead of one per part; with
                        // row_slice2_gate the ab parts become g/beta directly
                        // and the delta_gate launch below is skipped
                        // (bit-identical; kill PADDOCK_NO_RS2G)
                        static RS2G: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let rs2g = *RS2G.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_NO_RS2G").is_none()
                        });
                        // P71-R2: skip the split entirely - conv reads the
                        // fused plane strided, z feeds gated_rmsnorm_s, and
                        // g/beta are computed inside the v2f_g recurrence
                        // (slice2_gate's expressions verbatim). Requires the
                        // v2f election (same predicate as the recurrence
                        // site) and the plain gated_rmsnorm chain (the grer
                        // opt-in has no strided form).
                        // Kill: PADDOCK_NO_DN_STRIDED.
                        static DN_STRIDED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let strided_on = *DN_STRIDED.get_or_init(|| {
                            let on = paddock_models::dev_var_os!("PADDOCK_NO_DN_STRIDED").is_none();
                            if on {
                                eprintln!("[dn-strided] armed (fused-plane strided conv/z/gate)");
                            }
                            on
                        });
                        let grer_env =
                            paddock_models::dev_var_os!("PADDOCK_QWEN_GATED_ROW_FUSE").is_some();
                        if dn_ab_done
                            && strided_on
                            && !grer_env
                            && b <= dn_v2f_bmax()
                            && exec.has_gated_delta_recurrent_v2f()
                            && dn_v2f_on()
                            && exec.has_dn_fused_strided()
                        {
                            dn_strided = true;
                            dn_tot = tot;
                            dn_z_off = nc;
                            dn_ab_off = nc + nz_;
                            dn_gate_done = true;
                        } else if dn_ab_done && rs2g && exec.has_row_slice2_gate() {
                            exec.row_slice2_gate(
                                fused,
                                tot,
                                b,
                                &mut sc.d_mixed,
                                0,
                                nc,
                                &mut sc.d_z,
                                nc,
                                nz_,
                                nc + nz_,
                                n_v_heads,
                                &w.ssm_a.buf,
                                &w.dt_bias.buf,
                                &mut sc.d_g,
                                &mut sc.d_beta,
                            )?;
                            dn_gate_done = true;
                        } else if dn_ab_done && exec.has_row_slice4() {
                            exec.row_slice4(
                                fused,
                                tot,
                                b,
                                &mut [
                                    (&mut sc.d_mixed, 0, nc),
                                    (&mut sc.d_z, nc, nz_),
                                    (&mut sc.d_a, nc + nz_, n_v_heads),
                                    (&mut sc.d_b, nc + nz_ + n_v_heads, n_v_heads),
                                ],
                            )?;
                        } else {
                            exec.row_slice(fused, &mut sc.d_mixed, tot, 0, nc, b)?;
                            exec.row_slice(fused, &mut sc.d_z, tot, nc, nz_, b)?;
                            if dn_ab_done {
                                exec.row_slice(fused, &mut sc.d_a, tot, nc + nz_, n_v_heads, b)?;
                                exec.row_slice(
                                    fused,
                                    &mut sc.d_b,
                                    tot,
                                    nc + nz_ + n_v_heads,
                                    n_v_heads,
                                    b,
                                )?;
                            }
                        }
                        f8t_out_w = Some(ow_t);
                        true
                    } else if let (Some(dn), Some(fused)) = (dnf, bs.d_dn_fused.as_mut()) {
                        if !qdedup {
                            exec.quantize_q8(&sc.d_xn, &mut bs.d_xq, &mut bs.d_xs, b * dn.dims[0])?;
                        }
                        exec.q8_0_gemm_mma_ks(dn, &bs.d_xq, &bs.d_xs, &mut bs.d_ks_part, fused, b)?;
                        let (nt, w_in) = (dn.dims[1], conv_dim);
                        exec.row_slice(fused, &mut sc.d_mixed, nt, 0, w_in, b)?;
                        exec.row_slice(fused, &mut sc.d_z, nt, w_in, nt - w_in, b)?;
                        true
                    } else {
                        false
                    };
                    // in_qkv/alpha/beta/gate all read the same normed rows and
                    // nothing between them touches d_xq - one quantize serves
                    // all four (was 4 identical launches)
                    // f8 lane for the per-tensor DN projections (the fused
                    // bs_dn plane, when opted in, takes precedence above)
                    let l8d = bs_w8_dec
                        .get(li)
                        .filter(|_| b >= super::f8_dec_min() && !bs_f8ffn.is_empty());
                    // in_qkv now holds the fused in_qkv|gate_w plane (gate_w
                    // slot intentionally None since the merge)
                    let f8_dn = !dn_fused_done && l8d.is_some_and(|l| l.in_qkv.is_some());
                    if f8_dn {
                        let l8 = l8d.expect("f8_dn checked above");
                        if !e4m3_norms {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, b * embd)?;
                        }
                        let (nin, nc) = (w.in_qkv.dims()[0], w.in_qkv.dims()[1]);
                        let nz_ = w.gate_w.dims()[1];
                        let fused = bs.d_dn_fused.as_mut().expect("dn fused landing");
                        // One fused 16384-out GEMM (vLLM's DN merge; -48
                        // launches/tick) + row_slice split into mixed/z
                        exec.f8d_gemm_mma_ks(
                            l8.in_qkv.as_ref().expect("DeltaNet W8 in_qkv plane"),
                            nin,
                            nc + nz_,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut bs.d_ks_part,
                            fused,
                            b,
                        )?;
                        exec.row_slice(fused, &mut sc.d_mixed, nc + nz_, 0, nc, b)?;
                        exec.row_slice(fused, &mut sc.d_z, nc + nz_, nc, nz_, b)?;
                    } else if !dn_fused_done {
                        // b=1 in_qkv|gate_w launch merge (entry 317) - the Q8
                        // twin of the fused W8/f8t DN in-proj plane above,
                        // which the b=1 path never reached (b >= 8 gates).
                        // Kill: PADDOCK_NO_GEMV_MULTI.
                        match (&w.in_qkv, &w.gate_w) {
                            (QuantW::Q8(iq), QuantW::Q8(gw))
                                if b == 1
                                    && exec.has_q8_0_gemv_repacked_multi()
                                    && !super::no_gemv_multi() =>
                            {
                                exec.q8_0_gemv_repacked_multi(
                                    &mut [(iq, &mut sc.d_mixed), (gw, &mut sc.d_z)],
                                    &sc.d_xn,
                                )?;
                                dn_z_early = true;
                            }
                            _ => bmmq!(&w.in_qkv, &sc.d_xn, &mut sc.d_mixed, qdedup),
                        }
                    }
                    if dn_strided {
                        let fused = bs.d_dn_fused.as_ref().expect("dn fused landing");
                        exec.conv_step_slots_s(
                            bs.conv_win[li].as_mut().expect("DeltaNet layer window"),
                            fused,
                            0,
                            dn_tot,
                            &w.conv_w.buf,
                            &mut sc.d_conv,
                            &bs.d_slots,
                            b,
                            conv_dim,
                            conv_k,
                        )?;
                    } else {
                        exec.conv_step_slots(
                            bs.conv_win[li].as_mut().expect("DeltaNet layer window"),
                            &sc.d_mixed,
                            &w.conv_w.buf,
                            &mut sc.d_conv,
                            &bs.d_slots,
                            b,
                            conv_dim,
                            conv_k,
                        )?;
                    }
                    // gate first (no dep on the split) so the P70 fused
                    // recurrence below can consume g/beta directly
                    if dn_gate_done {
                        // g/beta already produced by row_slice2_gate at the split
                    } else if dn_ab_done {
                        // alpha/beta already landed by the fused in-proj GEMM
                        exec.delta_gate(
                            &sc.d_a,
                            &sc.d_b,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            b,
                            n_v_heads,
                        )?;
                    } else if let (Some(aw), Some(bw)) = (w.alpha_w.as_ref(), w.beta_w.as_ref()) {
                        // b=1 alpha|beta launch merge: two [embd -> n_v_heads]
                        // GEMVs are pure latency floor (~3 us each for ~0.1 MB
                        // of bytes) - one merged launch absorbs the second toll
                        // outright. Kill: PADDOCK_NO_GEMV_MULTI.
                        if b == 1 && exec.has_q8_0_gemv_repacked_multi() && !super::no_gemv_multi()
                        {
                            exec.q8_0_gemv_repacked_multi(
                                &mut [(aw, &mut sc.d_a), (bw, &mut sc.d_b)],
                                &sc.d_xn,
                            )?;
                        } else {
                            bmm!(aw, &sc.d_xn, &mut sc.d_a, qdedup);
                            bmm!(bw, &sc.d_xn, &mut sc.d_b, qdedup);
                        }
                        exec.delta_gate(
                            &sc.d_a,
                            &sc.d_b,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            b,
                            n_v_heads,
                        )?;
                    } else {
                        // non-Q8 alpha/beta: the mandatory f32 decay plane
                        // (exactly the serial spine's path, batched)
                        let ab = w.ab_f32.as_ref().expect("ab plane (loader guarantees)");
                        ab_gate(
                            &exec,
                            ab,
                            &sc.d_xn,
                            &mut sc.d_ab,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            b,
                            n_v_heads,
                        )?;
                    }
                    // P70: split + qk-L2-norm fused into the recurrence -
                    // one kernel instead of two per GDN layer per round and
                    // no dq/dk/dv plane round trip. STATE evolution is
                    // byte-identical to the chain; the q·S readout carries
                    // 1-ulp FMA-contraction diffs (probed <=3e-8 abs - the
                    // mode-2 reassociation class; readout never feeds
                    // state). b <= 24 by the probe curve (2.01x at b=1,
                    // 1.5x at 8, 1.22x at 12, 1.02x at 24, 0.94x at 32 -
                    // the recurrence is state-bandwidth-bound at high b and
                    // the z-redundant in-block norm recompute crosses over
                    // ~b=28). Kill: PADDOCK_NO_DN_V2F.
                    if dn_strided {
                        // gate-inline twin: g/beta straight off the fused
                        // plane (dn_strided implies the v2f election)
                        let fused = bs.d_dn_fused.as_ref().expect("dn fused landing");
                        exec.gated_delta_recurrent_v2f_g(
                            &sc.d_conv,
                            fused,
                            dn_ab_off,
                            dn_tot,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            Some(&bs.d_slots),
                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                            &mut sc.d_dattn,
                            b,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                        )?;
                    } else if b <= dn_v2f_bmax()
                        && exec.has_gated_delta_recurrent_v2f()
                        && dn_v2f_on()
                    {
                        exec.gated_delta_recurrent_v2f(
                            &sc.d_conv,
                            &sc.d_g,
                            &sc.d_beta,
                            Some(&bs.d_slots),
                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                            &mut sc.d_dattn,
                            b,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                        )?;
                    } else {
                        exec.deltanet_split_gqa_norm(
                            &sc.d_conv,
                            &mut sc.d_dq,
                            &mut sc.d_dk,
                            &mut sc.d_dv,
                            b,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                        )?;
                        exec.gated_delta_recurrent_v2(
                            &sc.d_dq,
                            &sc.d_dk,
                            &sc.d_dv,
                            &sc.d_g,
                            &sc.d_beta,
                            Some(&bs.d_slots),
                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                            0,
                            None,
                            &mut sc.d_dattn,
                            b,
                            1,
                            n_v_heads,
                            state_size,
                        )?;
                    }
                    if !dn_fused_done && !f8_dn && !dn_z_early {
                        bmmq!(&w.gate_w, &sc.d_xn, &mut sc.d_z, qdedup);
                    }
                    // Gated norm -> row-e4m3 in one kernel when the f8t
                    // out_proj arm is certain to consume it: the
                    // fused kernel skips the f32 d_core landing entirely, so
                    // it is legal only on the f8t branch (all fallback arms
                    // read d_core). Bit-identical to the two-kernel chain.
                    // OPT-IN, not default: even with the wait-only arm + late
                    // release (the second PDL law) the
                    // fuse measures -0.3% at c32 (2355.18 vs 2362.29) - the
                    // lab's 2us/layer win does not survive PDL topology at
                    // grid=b CTAs, and b<=8 widths only starve harder.
                    let grer = f8t_out_w.is_some()
                        && state_size == 128
                        && n_v_heads % 16 == 0
                        && exec.has_gated_rmsnorm_e4m3_row()
                        && paddock_models::dev_var_os!("PADDOCK_QWEN_GATED_ROW_FUSE").is_some();
                    if !grer {
                        if dn_strided {
                            let fused = bs.d_dn_fused.as_ref().expect("dn fused landing");
                            exec.gated_rmsnorm_s(
                                &sc.d_dattn,
                                fused,
                                dn_z_off,
                                dn_tot,
                                n_v_heads,
                                &w.ssm_norm.buf,
                                &mut sc.d_core,
                                b * n_v_heads,
                                state_size,
                                eps,
                            )?;
                        } else {
                            exec.gated_rmsnorm(
                                &sc.d_dattn,
                                &sc.d_z,
                                &w.ssm_norm.buf,
                                &mut sc.d_core,
                                b * n_v_heads,
                                state_size,
                                eps,
                            )?;
                        }
                    }
                    if let Some(ow_t) = f8t_out_w {
                        if grer {
                            exec.gated_rmsnorm_e4m3_row(
                                &sc.d_dattn,
                                &sc.d_z,
                                &w.ssm_norm.buf,
                                None,
                                &mut sc.d_f8t_q,
                                &mut sc.d_f8t_rs,
                                b,
                                n_v_heads,
                                state_size,
                                eps,
                            )?;
                        } else {
                            exec.quantize_e4m3_row(
                                &sc.d_core,
                                &mut sc.d_f8t_q,
                                &mut sc.d_f8t_rs,
                                w.out_w.dims()[0],
                                b,
                            )?;
                        }
                        exec.f8t_gemm(
                            ow_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            w.out_w.dims()[0],
                            w.out_w.dims()[1],
                            b,
                        )?;
                    } else if let Some(w8o) = l8d.and_then(|l| l.out_w.as_ref()).filter(|_| f8_dn) {
                        exec.quantize_e4m3(
                            &sc.d_core,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            b * w.out_w.dims()[0],
                        )?;
                        exec.f8d_gemm_mma_ks(
                            w8o,
                            w.out_w.dims()[0],
                            w.out_w.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            b,
                        )?;
                    } else {
                        bmmq!(&w.out_w, &sc.d_core, &mut sc.d_proj);
                    }
                }
            }
            // Post-norm -> row-e4m3 in one kernel, same rule as the pre-norm
            // above: legal only when the f8t FFN arm is CERTAIN to take it,
            // since the fused kernel never materialises d_xn in f32 and the
            // fallback FFN arms all read it.
            let f8t_postnorm = matches!(&layer.ffn, Ffn::Dense { .. })
                && self.bs_f8t_ffn.get(li).and_then(|o| o.as_ref()).is_some()
                && b <= 64
                && exec.has_add_rmsnorm_e4m3_row()
                && (!qdedup || f8t_layer_covers)
                && paddock_models::dev_var_os!("PADDOCK_NO_QWEN_POSTNORM_FUSE").is_none();
            if f8t_postnorm {
                exec.add_rmsnorm_e4m3_row(
                    &mut sc.d_x,
                    &sc.d_proj,
                    &layer.post_norm.buf,
                    &mut sc.d_f8t_q,
                    &mut sc.d_f8t_rs,
                    embd,
                    eps,
                    b,
                )?;
            } else if e4m3_norms {
                exec.add_rmsnorm_e4m3_xn(
                    &mut sc.d_x,
                    Some(&sc.d_proj),
                    &layer.post_norm.buf,
                    &mut sc.d_xn,
                    &mut sc.d_pxq,
                    &mut sc.d_exs,
                    embd,
                    b,
                    eps,
                )?;
            } else {
                exec.add_rmsnorm_batch(
                    &mut sc.d_x,
                    &sc.d_proj,
                    &layer.post_norm.buf,
                    &mut sc.d_xn,
                    embd,
                    eps,
                    b,
                )?;
            }
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    // sm_100 tcgen05 arm (PADDOCK_QWEN_F8T). This is the
                    // serving decode path - the pooled/batched engine - and it
                    // must sit ahead of the `f8` lane below, which is gated
                    // `b >= 8` and therefore never fires at the smoke shapes
                    // (c1 b=1, c4 b=4): those fall through to the Q8_0 int8
                    // fusion program, and int8 is the one class B200 de-rates
                    // (1148 TOPS vs ~7.5P e4m3). f8t covers the whole b<=64
                    // band through tc5p/tc5q.
                    if let Some(p) = self.bs_f8row_ffn.get(li).and_then(|o| o.as_ref()) {
                        // checkpoint-exact fp8 layer (the f8row class): every
                        // width has an arm, so this layer built no lin/Q8
                        // twin - nothing below may run for it. b=1 rides the
                        // f32-in row GEMV; b>=2 stages the per-row pair.
                        if b == 1 {
                            super::ops::ffn_f8row_gemv(
                                &exec,
                                p,
                                &sc.d_xn,
                                &mut sc.d_ffn_gate,
                                &mut sc.d_ffn_up,
                                &mut sc.d_proj,
                            )?;
                        } else {
                            super::ops::ffn_f8row_rows(
                                &exec,
                                p,
                                &sc.d_xn,
                                &mut sc.d_f8t_q,
                                &mut sc.d_f8t_rs,
                                &mut sc.d_ffn_gate,
                                &mut sc.d_ffn_up,
                                &mut sc.d_proj,
                                b,
                            )?;
                        }
                    } else if let Some([gu_t, dn_t]) = self
                        .bs_f8t_ffn
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| b <= f8t_dec_bmax())
                    {
                        tracing::info!("rec-ffn arm=f8t li={li} b={b}");
                        // already produced by the fused post-norm above
                        if !f8t_postnorm {
                            exec.quantize_e4m3_row(
                                &sc.d_xn,
                                &mut sc.d_f8t_q,
                                &mut sc.d_f8t_rs,
                                embd,
                                b,
                            )?;
                        }
                        // P62 gluq silu twin: only the b in [16, 64] slice of
                        // this decode arm - tc5r keeps the low rungs.
                        exec.f8t_gemm(
                            gu_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_ffn_gate,
                            embd,
                            2 * ff,
                            b,
                        )?;
                        // swiglu straight to row-e4m3 removes a launch and the
                        // f32 round trip of the widest row (ff=17408) -- and
                        // MEASURED worse, 111.52 -> 106.01 c1. The pair it
                        // replaces is not launch-bound: pd_swiglu_fused runs
                        // grid 68x1 at 1.98 us, genuinely parallel, while the
                        // one-block-per-row fusion needed to see the whole row
                        // for the max serialises it onto one CTA (and 68 KB of
                        // smem pins that CTA to its SM). Fusing only pays when
                        // the victim launch was already single-block, as the
                        // two norms were. Kept opt-in as the measured
                        // counterexample: PADDOCK_QWEN_SWIGLU_FUSE=1.
                        let sg_fused = exec.has_swiglu_e4m3_row()
                            && ff * 4 <= 200 * 1024
                            && paddock_models::dev_var_os!("PADDOCK_QWEN_SWIGLU_FUSE").is_some()
                            && exec
                                .swiglu_e4m3_row(
                                    &sc.d_ffn_gate,
                                    &mut sc.d_f8t_q,
                                    &mut sc.d_f8t_rs,
                                    ff,
                                    b,
                                )
                                .is_ok();
                        if !sg_fused {
                            exec.swiglu_fused(&sc.d_ffn_gate, &mut sc.d_ffn_up, ff, b)?;
                            exec.quantize_e4m3_row(
                                &sc.d_ffn_up,
                                &mut sc.d_f8t_q,
                                &mut sc.d_f8t_rs,
                                ff,
                                b,
                            )?;
                        }
                        exec.f8t_gemm(
                            dn_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            ff,
                            embd,
                            b,
                        )?;
                    } else {
                        // native-fp8 decode lane (PADDOCK_F8_DECODE, precision
                        // class): all three FFN GEMMs on the e4m3 rung - 1.031
                        // B/param stream vs Q8_0's 1.0625, measured 1.02-1.05x
                        // per shape. Takes precedence over the Q8/fused chains.
                        // byte-passthrough planes take precedence when loaded
                        // (PADDOCK_FP8_BS marker-8): same call, dispatch by marker
                        let f8 = self
                            .bs_f8ffn_bs
                            .get(li)
                            .and_then(|o| o.as_ref())
                            .or_else(|| bs_f8ffn.get(li).and_then(|o| o.as_ref()))
                            .filter(|_| b >= super::f8_ffn_min());
                        static F8R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let f8r = *F8R.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_some()
                        });
                        if let (Some([gu8, d8]), Some(fused)) = (f8, bs.d_gu_fused.as_mut()) {
                            tracing::info!("rec-ffn arm=mma li={li} b={b}");
                            // One fused 2ff GEMM (the b<=16 launch-economics fix:
                            // 640 -> fewer kernels/tick) + packed swiglu_fused;
                            // d_proj lands exactly like the q8 chain (shared
                            // residual add after the match)
                            if !e4m3_norms {
                                exec.quantize_e4m3(
                                    &sc.d_xn,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    b * gu8.1,
                                )?;
                            }
                            if f8r {
                                exec.f8r_gemm_mma_ks(
                                    &gu8.0,
                                    gu8.1,
                                    gu8.2,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut bs.d_ks_part,
                                    fused,
                                    b,
                                )?;
                            } else {
                                exec.f8d_gemm_mma_ks(
                                    &gu8.0,
                                    gu8.1,
                                    gu8.2,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut bs.d_ks_part,
                                    fused,
                                    b,
                                )?;
                            }
                            if exec.has_swiglu_fused_e4m3() {
                                // one kernel: silu*up + e4m3 quant straight off
                                // the fused landing (no [b,ff] f32 round trip)
                                exec.swiglu_fused_e4m3(fused, &mut sc.d_pxq, &mut sc.d_exs, ff, b)?;
                            } else {
                                exec.swiglu_fused(fused, &mut sc.d_ffn_gate, ff, b)?;
                                exec.quantize_e4m3(
                                    &sc.d_ffn_gate,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    b * d8.1,
                                )?;
                            }
                            if f8r {
                                exec.f8r_gemm_mma_ks(
                                    &d8.0,
                                    d8.1,
                                    d8.2,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_proj,
                                    b,
                                )?;
                            } else {
                                exec.f8d_gemm_mma_ks(
                                    &d8.0,
                                    d8.1,
                                    d8.2,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut bs.d_ks_part,
                                    &mut sc.d_proj,
                                    b,
                                )?;
                            }
                        } else {
                            // fusion program: merged gate|up plane at
                            // decode widths - One 2ff-wide ks GEMM (544 tiles -> nz=1
                            // on <=272-SM dies: direct-y, no K-split partials or
                            // combine) + packed swiglu. b<8 keeps the per-tensor
                            // rungs (dp4a_nc wins there; see the b=1 GEMV-fusion
                            // regression note in forward.rs).
                            let gu = bs_gu.get(li).and_then(|o| o.as_ref()).filter(|_| b >= 8);
                            // b=1 lin-GEMV arm (non-KV-overhead R2.2): the same e4m3
                            // boxes the b>=8 lane above reads, now served at one row -
                            // which is what lets this class stop carrying the Q8_0
                            // twin. Bit-class identical to the b>=8 arm (same plane,
                            // same per-32 ue8m0 scales), ~-2% vs the Q8 GEMV chain on
                            // the FFN shapes (bench/f8lin_gemv_bench). The skinny
                            // projections are not converted: at 40 row-tiles the box
                            // granularity starves the die and Q8 wins by 12%.
                            let lin1 = (b == 1 && lin_gemv_on)
                                .then(|| bs_f8ffn.get(li).and_then(|o| o.as_ref()))
                                .flatten()
                                .filter(|p| p[0].0.is_lin() && p[1].0.is_lin());
                            if let (Some([gu8, d8]), Some(fused)) = (lin1, bs.d_gu_fused.as_mut()) {
                                // One launch for the fused gate|up plane. Splitting it
                                // into two independent row-window launches - to hand
                                // the scheduler the same independence the Q8_0 chain
                                // has - was MEASURED and lost (46.90 vs 47.13 fused),
                                // so the missing concurrency is not recoverable by
                                // handing back independence either.
                                // Four variants tried, all worse than this one; see
                                exec.f8lin_gemv_at(
                                    &gu8.0,
                                    &sc.d_xn,
                                    &mut bs.d_ks_part,
                                    0,
                                    fused,
                                    0,
                                    Some((&mut bs.d_lin_tick, 0)),
                                    gu8.1,
                                    gu8.2,
                                )?;
                                exec.swiglu_fused(fused, &mut sc.d_ffn_gate, ff, b)?;
                                exec.f8lin_gemv_at(
                                    &d8.0,
                                    &sc.d_ffn_gate,
                                    &mut bs.d_ks_part,
                                    0,
                                    &mut sc.d_proj,
                                    0,
                                    Some((&mut bs.d_lin_tick, lin_dn_off)),
                                    d8.1,
                                    d8.2,
                                )?;
                            } else {
                                if let (Some(gu), Some(fused)) = (gu, bs.d_gu_fused.as_mut()) {
                                    exec.quantize_q8(
                                        &sc.d_xn,
                                        &mut bs.d_xq,
                                        &mut bs.d_xs,
                                        b * gu.dims[0],
                                    )?;
                                    exec.q8_0_gemm_mma_ks(
                                        gu,
                                        &bs.d_xq,
                                        &bs.d_xs,
                                        &mut bs.d_ks_part,
                                        fused,
                                        b,
                                    )?;
                                    exec.swiglu_fused(fused, &mut sc.d_ffn_gate, ff, b)?;
                                } else {
                                    bmmq!(gate, &sc.d_xn, &mut sc.d_ffn_gate);
                                    bmmq!(up, &sc.d_xn, &mut sc.d_ffn_up);
                                    exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, b * ff)?;
                                }
                                bmmq!(down, &sc.d_ffn_gate, &mut sc.d_proj);
                            }
                        }
                    } // end tcgen05-vs-warp arm (PADDOCK_QWEN_F8T)
                }
                Ffn::Nvf4Dense { gate, up, down } => {
                    // f8t tile arm first, exactly as the Dense arm above and
                    // off the same planes - load.rs builds them from the
                    // NVFP4 checkpoint's own values when headroom allows.
                    // This is the serving decode path, and the W4A16 nvf4
                    // chain below is L1-bound on software dequant
                    // (L1/TEX 85.1% vs DRAM 6.7%, ~0.7 TB/s against a 7 TB/s
                    // roof). Same-checkpoint wide A/B: ~2.7x.
                    let f8t = self
                        .bs_f8t_ffn
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| b <= f8t_dec_bmax());
                    if let Some([gu_t, dn_t]) = f8t {
                        // d_xn is always f32 here - the e4m3-fused norm routes
                        // above are gated on Ffn::Dense, so quantize always.
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            embd,
                            b,
                        )?;
                        exec.f8t_gemm(
                            gu_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_ffn_gate,
                            embd,
                            2 * ff,
                            b,
                        )?;
                        exec.swiglu_fused(&sc.d_ffn_gate, &mut sc.d_ffn_up, ff, b)?;
                        exec.quantize_e4m3_row(
                            &sc.d_ffn_up,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            ff,
                            b,
                        )?;
                        exec.f8t_gemm(
                            dn_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            ff,
                            embd,
                            b,
                        )?;
                    } else {
                        // above the row band the chain takes the W4A4 arm
                        nvf4_ffn(
                            &exec,
                            gate,
                            up,
                            down,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_nvs,
                            &mut sc.d_nv4part,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            b,
                        )?;
                    }
                }
                Ffn::Moe(w) => {
                    // moe_ffn picks sorted vs token-batched by row count
                    // (B*n_active >= 32 -> sorted, i.e. B >= 4 on the 35B)
                    moe_ffn(
                        &exec,
                        w,
                        self.moe.expect("moe dims"),
                        embd,
                        b,
                        true,
                        &sc.d_xn,
                        &mut sc.d_moe_xq,
                        &mut sc.d_moe_xs,
                        &mut sc.d_ssums,
                        &mut sc.d_moe_xs8,
                        &mut sc.d_moe_fs8,
                        &mut sc.d_moe_logits,
                        &sc.d_zero_bias,
                        &mut sc.d_moe_idx,
                        &mut sc.d_moe_w,
                        &mut sc.d_moe_fused,
                        &mut sc.d_moe_fq,
                        &mut sc.d_moe_fs,
                        &mut sc.d_moe_srow,
                        &mut sc.d_moe_sslot,
                        &mut sc.d_moe_bexp,
                        &mut sc.d_moe_part,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        &mut sc.d_skfix,
                        &mut sc.d_ffn_gate,
                        &mut sc.d_ffn_up,
                        &mut sc.d_mixed,
                        &mut sc.d_proj,
                    )?;
                }
            }
            if res_hoist {
                pending_res = true;
            } else {
                exec.add(&mut sc.d_x, &sc.d_proj, b * embd)?;
            }
        }

        if pending_res {
            exec.add_rmsnorm_batch(
                &mut sc.d_x,
                &sc.d_proj,
                &self.out_norm.buf,
                &mut sc.d_h,
                embd,
                eps,
                b,
            )?;
        } else {
            exec.rmsnorm_batch(&sc.d_x, &self.out_norm.buf, &mut sc.d_h, embd, eps, b)?;
        }
        if b == 1 {
            // f8 lm_head in the captured b=1 tick (B200 bring-up opt-in:
            // PADDOCK_F8_LMHEAD builds the plane, PADDOCK_QWEN_F8_LMHEAD_LOWB
            // elects it below the shipped b>=8 class boundary). At b=1 the
            // gemv_any fallback below measures 681 us/tick,
            // 1.98 TB/s, grid 248320 = one CTA per vocab row - 10.6% of the
            // whole decode tick and the last projection on the legacy int8
            // path while every neighbour runs f8.
            // f8t tile plane first: vocab/128 = 1940 tiles takes the wmma
            // route, the same one that carries FFN gate_up. Falls back to the
            // f8d lin head, then to gemv_any.
            let mut return_early_head = false;
            let f8t_head = self
                .out_f8t
                .as_ref()
                .filter(|_| paddock_models::dev_var_os!("PADDOCK_NO_F8T_LMHEAD").is_none());
            if let Some((pt, pi, po)) = f8t_head {
                exec.quantize_e4m3_row(&sc.d_h, &mut sc.d_f8t_q, &mut sc.d_f8t_rs, *pi, 1)?;
                exec.f8t_gemm(
                    pt,
                    &sc.d_f8t_q,
                    &sc.d_f8t_rs,
                    &mut bs.d_ks_part,
                    &mut bs.d_logits,
                    *pi,
                    *po,
                    1,
                )?;
                return_early_head = true;
            }
            // Default-on since the PPL gate; the plane only exists where the
            // loader built it (sm_100, or an explicit PADDOCK_F8_LMHEAD).
            let f8_head = self.out_f8.as_ref().filter(|_| {
                !return_early_head
                    && paddock_models::dev_var_os!("PADDOCK_NO_QWEN_F8_LMHEAD_LOWB").is_none()
            });
            if return_early_head {
                // head already produced by the f8t tile plane above
            } else if let Some((p8, pi, po)) = f8_head {
                exec.quantize_e4m3(&sc.d_h, &mut sc.d_pxq, &mut sc.d_exs, embd)?;
                exec.f8d_gemm_mma_ks(
                    p8,
                    *pi,
                    *po,
                    &sc.d_pxq,
                    &sc.d_exs,
                    &mut bs.d_ks_part,
                    &mut bs.d_logits,
                    1,
                )?;
            } else {
                match &self.output {
                    // W4A8 lm head in the captured b=1 tick - profiling caught
                    // this site still on the exact-f32 GEMV (1.39 ms/token, the
                    // single biggest non-projection item)
                    QuantW::Kq(k) if kq_w4a8_b1 => {
                        exec.quantize_q8_sums(
                            &sc.d_h,
                            &mut bs.d_xq,
                            &mut bs.d_xs,
                            &mut sc.d_ssums,
                            embd,
                        )?;
                        let needs = crate::gpu::kq_needs_sums(k.ty);
                        exec.kquant_gemv_w4a8(
                            k,
                            &bs.d_xq,
                            &bs.d_xs,
                            needs.then_some(&sc.d_ssums),
                            &mut bs.d_logits,
                        )?;
                    }
                    _ => gemv_any(&exec, &self.output, &sc.d_h, &mut bs.d_logits)?,
                }
            }
        } else if let Some((pt, pi, po)) = self
            .out_f8t
            .as_ref()
            .filter(|_| b <= 64 && paddock_models::dev_var_os!("PADDOCK_NO_F8T_LMHEAD").is_none())
        {
            // Same f8t tile-plane head across the whole batched decode band, not
            // just b==1. The b==1-only form left c4 on the legacy Q8 GEMM and
            // measured -15.1% there while c1 was +2.5%; the head
            // is the same ~6% of the tick at every width, and f8t_gemm carries
            // b <= 64 through tc5p/tc5q/wmma exactly as the projections do.
            exec.quantize_e4m3_row(&sc.d_h, &mut sc.d_f8t_q, &mut sc.d_f8t_rs, *pi, b)?;
            exec.f8t_gemm(
                pt,
                &sc.d_f8t_q,
                &sc.d_f8t_rs,
                &mut bs.d_ks_part,
                &mut bs.d_logits,
                *pi,
                *po,
                b,
            )?;
        } else if let Some((p8, pi, po)) = self.out_f8.as_ref().filter(|_| {
            // f8d lin head at batched widths - the same shipped class boundary
            // as the sampled-pass head (b >= 8). This fn predates the f8d head
            // and left every b > 1 width on the Q8 vocab GEMM: a wide-batch
            // kernel ledger shows 929 steps x 1406 us on
            // q8_0_gemm_mma grid 3880 (~1 TB/s) while the f8d head runs the
            // same step at ~505 us - ~0.9 ms of every batched decode tick.
            // b == 1 above and the sampled pass already elect f8 by default,
            // so this closes an omission, not a class change.
            b >= super::f8_head_min()
                || paddock_models::dev_var_os!("PADDOCK_QWEN_F8_LMHEAD_LOWB").is_some()
        }) {
            exec.quantize_e4m3(&sc.d_h, &mut sc.d_pxq, &mut sc.d_exs, b * embd)?;
            exec.f8d_gemm_mma_ks(
                p8,
                *pi,
                *po,
                &sc.d_pxq,
                &sc.d_exs,
                &mut bs.d_ks_part,
                &mut bs.d_logits,
                b,
            )?;
        } else if let QuantW::Kq(k) = &self.output {
            // k-quant lm_head: the W4A8 dp4a GEMM at every batched width (its
            // 4-row blocks fill the die at vocab-scale out_dims)
            exec.quantize_q8(&sc.d_h, &mut bs.d_xq, &mut bs.d_xs, b * embd)?;
            let needs = crate::gpu::kq_needs_sums(k.ty);
            if needs {
                exec.q8_sums_strided(&bs.d_xq, &mut sc.d_ssums, k.dims[0], b)?;
            }
            exec.kquant_gemm_dp4a(
                k,
                &bs.d_xq,
                &bs.d_xs,
                needs.then_some(&sc.d_ssums),
                &mut bs.d_logits,
                b,
            )?;
        } else if b <= 4 {
            exec.quantize_q8(&sc.d_h, &mut bs.d_xq, &mut bs.d_xs, b * embd)?;
            super::stub_guard(&self.output, "batch.rs record_batch_step head")?;
            exec.q8_0_gemv_dp4a_nc(self.output.q8(), &bs.d_xq, &bs.d_xs, &mut bs.d_logits, b)?;
        } else {
            exec.quantize_q8(&sc.d_h, &mut bs.d_xq, &mut bs.d_xs, b * embd)?;
            // lm_head is a single big Q8 GEMM ([B,embd]->[B,vocab~152k]) landing
            // straight in d_logits (no KS part buffer - the vocab-wide partials
            // would be huge). It's batch-gated like the bmm! projections: the
            // 64x64 mma tile wastes 56/64 token cols over the ~3142-tile vocab
            // grid at small B (dp4a MT is bandwidth-optimal there), but at B>=24
            // the tile amortizes and the tensor-core dots win. Measured on a
            // 35B lm_head: dp4a wins at c8, mma wins at c32. Same
            // per-block-scale numeric class.
            if b >= ks_min_batch() {
                exec.q8_0_gemm_mma(self.output.q8(), &bs.d_xq, &bs.d_xs, &mut bs.d_logits, b)?;
            } else {
                exec.q8_0_gemm_mt_dp4a(self.output.q8(), &bs.d_xq, &bs.d_xs, &mut bs.d_logits, b)?;
            }
        }
        Ok(())
    }
}
