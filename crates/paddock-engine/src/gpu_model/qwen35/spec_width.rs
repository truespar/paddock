//! How wide this lane speculates: the engagement cap, and the GB10 MTP
//! width election (depth 2 past the narrow tiers).

use super::GpuQwen35;

impl GpuQwen35 {
    /// Draft-state live preference fed to the width sizer (which degrades
    /// live before width under VRAM pressure - the election here is a CAP,
    /// not a reservation). Drafter-conditional (rung B): 32 with a
    /// block drafter attached - the batched round + 64-row budget make
    /// width drafting pay (b32r64: c16 856 / c32 913 vs dense 838) and the
    /// snapshot-free verify prices 32 rows at ~1.45 GiB on 27B. MTP-only
    /// lanes (qwen3.5-9b, qwen3.6) keep 8 - their chain at width measured
    /// worse than dense (cap-16 A/B) - except where a width re-measurement
    /// exists: GB10, `mtp_width_on` below.
    ///
    /// Widths at which speculation is allowed to engage at all.
    ///
    /// A drafter attaching used to raise this to 32 unconditionally. On sm_100
    /// that was worth two board cells, because **every threshold in this
    /// chain had been elected spec-vs-spec** - `dflash_live_max` went
    /// 1 -> 8 -> 32 on arguments that only ever beat the other spec arm.
    /// Nothing in the policy ever compared speculating against not
    /// speculating, and on a die where speculation always won that omission
    /// was free.
    ///
    /// It is not free here. qwen3.8 nvfp4 on B200, our spec against our own
    /// no-spec, with the spec path as fast as it has ever been:
    ///
    ///   c1 2.84x  c4 2.17x  c8 1.57x  |  c16 0.45x  c32 0.44x  imax 0.39x
    ///
    /// The crossover sits between c8 and c16 and it is not close on either
    /// side. Capping there roughly DOUBLES wide-batch throughput - by
    /// declining to speculate, not by making speculation faster.
    ///
    /// Die-aware rather than a blanket flip: sm_120 is measured at 32
    /// and its target decode is ~3x slower, so the verify tick still amortises
    /// there. The honest version of this is an election that MEASURES spec
    /// against no-spec per width instead of carrying a constant per die; this
    /// default is the interim, and the fact that it had to be measured on each
    /// die separately is the whole lesson.
    ///
    /// The MTP-only chain on GB10 is the other measured exception
    /// (`mtp_width_on`): above 8 live it runs at depth 2 rather than not at
    /// all, so its engagement cap is 32 too.
    pub(super) fn serve_spec_live_max(&self) -> usize {
        paddock_models::dev_var!("PADDOCK_QWEN35_SPEC_LIVE_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                if self.dflash.is_none() {
                    return if self.mtp_width_on() { 32 } else { 8 };
                }
                if self.exec.compute_capability().0 == 12 {
                    32
                } else {
                    8
                }
            })
    }

    /// Wide-batch MTP on GB10 (sm_121): the in-file MTP chain with no block
    /// drafter speculates past 8 live slots - at depth 3 (`mtp_width_depth`).
    ///
    /// Four gates kept it dense there: this lane's engagement cap of 8, the
    /// scheduler's 32-row round budget, its device-plans gate of 8, and with
    /// that gate the prefill warm hint - so a slot admitted above 8 live
    /// never fed its draft head, and the few rounds that did start were
    /// declined into a 256-tick cooldown. The cap of 8 was measured before
    /// the snapshot-free verify (verify_hold + commit_walk) removed the
    /// per-token state writes that made wide spec lose. Re-measured on GB10,
    /// Qwen3.6-35B-A3B Q8_0, greedy, 512-token replies, tok/s per stream,
    /// no-spec -> depth 2: c12 23.0 -> 27.4, c16 22.2 -> 24.8, c24 19.7 ->
    /// 21.1, c32 17.2 -> 18.2. Other dies are unmeasured and keep the old
    /// cap: on B200 wide spec lost by half.
    ///
    /// Depth 3, every slot (`mtp_floor`): this verify pads each slot
    /// to the round's widest chunk, so a slot that drafts less than the
    /// round saves no rows - it only forgoes tokens. The per-slot ramp
    /// (halve on a miss, double on a full accept) is right for a ragged
    /// verify and wrong here. Measured with every slot at the round's depth
    /// (same model, tok/s per stream, depth 1/2/3/4): c16 20.9 / 24.2 /
    /// 26.2 / 26.2, c32 15.8 / 18.2 / 19.1 / 19.3. Rounds at depth 1..4 cost
    /// 64 / 73 / 79 / 83 ms at c16 and 82 / 98 / 108 / 115 at c32. Depth 3
    /// is the MoE chain's own depth (`serve_spec_k_mtp`); 4 adds a chain
    /// step for no measured gain at c16. End to end against the depth-2
    /// election this replaced (tok/s per stream, greedy / sampled): 35B-A3B
    /// c16 25.0 -> 27.9 / 23.5 -> 25.9, c32 18.5 -> 20.3 / 16.4 -> 18.1;
    /// 9B c16 27.4 -> 31.0, c32 22.2 -> 24.5 / 21.5 -> 23.7; 27B (MTP
    /// only) c16 12.2 -> 14.3, c32 9.8 -> 11.1 / 9.6 -> 10.7. c1 and c8
    /// unchanged.
    ///
    /// Confidence-ranked depth was measured against this and does not pay
    /// on a padded verify. The chain's top-1 probability is well calibrated
    /// (accepted ~p at every position: 0.99 at p >= 0.9, 0.55 at 0.5-0.6),
    /// but replaying 630 depth-4 rounds offline, choosing each round's depth
    /// from it gains 0.3-0.7% over the best uniform depth (an oracle 2-3%).
    /// Ranking drafts per slot - DeepSeek DSpark's verify allocation - needs
    /// a packed verify to save rows; its replayed ceiling here is +3-5% over
    /// the best uniform depth. The census that measured this stays in
    /// (`spec_conf.rs`, PADDOCK_SPEC_CONF).
    pub(super) fn mtp_width_on(&self) -> bool {
        self.mtp.is_some() && self.dflash.is_none() && self.exec.compute_capability() == (12, 1)
    }

    /// Depth for a round of `live` slots under `mtp_width_on`: 3 past the 8
    /// live slots the narrow tiers own (their elected depths stand).
    /// (PADDOCK_QWEN35_MTP_WIDTH_DEPTH pins another depth for the A/Bs.)
    pub(crate) fn mtp_width_depth(&self, live: usize) -> Option<usize> {
        let depth = paddock_models::dev_var!("PADDOCK_QWEN35_MTP_WIDTH_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        (self.mtp_width_on() && live > 8).then_some(depth)
    }

    /// The post-miss floor of an MTP round under `mtp_width_on` (the last
    /// round's live count is the spec batch's): the round's own depth - the
    /// width depth past 8 live, the chain's depth (`serve_spec_k_mtp`, 3 on
    /// MoE, 4 dense) below - so a slot that missed drafts the full depth
    /// again instead of ramping back up from its accepted run.
    ///
    /// The ramp saves nothing on this lane at any width: the chain graph
    /// unrolls its whole depth whatever a slot asks for, and the verify pads
    /// every slot to the widest chunk; at one live slot the shorter verify
    /// it buys is worth less than the drafts it drops. Measured on GB10
    /// (tok/s per stream, ramp -> floor at depth, c1 / c2 / c4 / c8):
    /// Qwen3.6-35B-A3B 80.3 -> 87.1 / 63.3 -> 70.3 / 46.0 -> 51.3 /
    /// 33.3 -> 37.0; Qwen3.5-9B 37.2 -> 42.1 / 37.2 -> 42.0 / 34.4 -> 40.3 /
    /// 31.3 -> 35.3; Qwen3.8-27B (MTP only) 15.8 -> 18.0 / 16.7 -> 18.8 /
    /// 15.2 -> 17.1 / 14.4 -> 16.3. PADDOCK_QWEN35_NO_MTP_FLOOR restores the
    /// ramp for the A/B.
    pub(super) fn mtp_floor(&self) -> Option<usize> {
        if !self.mtp_width_on()
            || paddock_models::dev_var_os!("PADDOCK_QWEN35_NO_MTP_FLOOR").is_some()
        {
            return None;
        }
        let live = self.spec_batch.as_ref()?.batch;
        Some(
            self.mtp_width_depth(live)
                .unwrap_or_else(|| self.serve_spec_k_mtp()),
        )
    }

    /// Per-round post-miss k floor for the service controller. DFlash2
    /// drafts its whole block in one forward - depth is nearly free, and the
    /// classic post-miss rule made it spend most rounds re-climbing (the
    /// measured cross-leg k death-spiral, c4 457 -> 159). Answered by which
    /// drafter actually ran (spec_round_dflash), replacing the attach-time
    /// PADDOCK_SPEC_K_MISS_FLOOR env default that pinned floor 7 on MTP
    /// rounds too (measured as the attached-serve residual: c4 290 vs 361).
    /// An MTP round on GB10 drafts every slot to the round's depth
    /// (`mtp_floor`); elsewhere it keeps the classic floor.
    pub fn spec_k_miss_floor_mtp(&self) -> Option<usize> {
        if self.spec_round_dflash {
            self.dflash.as_ref().map(|d| d.block - 1)
        } else {
            self.mtp_floor()
        }
    }

    /// Steps the next draft-chain launch records: the depth its round
    /// verifies (`chain_run`, set by each round's draft entry point), at most
    /// the chain's allocation. The chain used to unroll its full depth every
    /// round, so a dense model at width (chain 4, rounds of 3) and at c7/c8
    /// (the row budget's depth 3) paid a whole head pass - MTP block plus the
    /// vocab projection - for a draft nothing verified.
    /// PADDOCK_QWEN35_FULL_CHAIN restores the full unroll for the A/B.
    pub(super) fn chain_steps(&self) -> usize {
        let depth = self.spec_batch.as_ref().map_or(1, |sb| sb.chain_depth);
        if paddock_models::dev_var_os!("PADDOCK_QWEN35_FULL_CHAIN").is_some() {
            return depth;
        }
        self.chain_run.clamp(1, depth)
    }

    /// The scheduler-side half of `mtp_width_on`, elected at batch enable -
    /// after a block drafter would have attached, before the serve loop
    /// reads either value: a round budget that holds depth 3 at 32 live
    /// (4 rows a slot) and the device-plans gate at 32, which also lets the
    /// prefill warm hint reach every slot. Explicit env always wins.
    pub(crate) fn elect_mtp_width(&self) {
        if !self.mtp_width_on() {
            return;
        }
        if std::env::var_os("PADDOCK_SPEC_MAX_ROWS").is_none() {
            crate::envset::set_env("PADDOCK_SPEC_MAX_ROWS", "128");
        }
        if std::env::var_os("PADDOCK_QWEN35_SPEC_PLANS_LIVE_MAX").is_none() {
            crate::envset::set_env("PADDOCK_QWEN35_SPEC_PLANS_LIVE_MAX", "32");
        }
    }
}
