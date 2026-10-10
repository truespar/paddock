//! Restore-vs-recompute election.
//!
//! The rivals' no-regression failures came from reflexive tiering: SGLang
//! ships `load_back_threshold = 10` with a "dynamically adjust" todo, vLLM
//! and LMCache ship no cost model at all, and LMCache's own paper calls
//! adaptive load-vs-recompute necessary. This
//! module is the elected alternative: per off-tier hit, predict both paths
//! under the CURRENT queues and pick, with hysteresis so near-ties do not
//! flap and a predicted-vs-actual error export so a drifting model is a
//! visible defect rather than a silent regression.
//!
//! Seeds come from the host/device probes (bus ceiling ~26 GB/s, per-op
//! fixed latencies in the tens of microseconds); every observation then feeds
//! a conservative EWMA, so the model tracks the ACTUAL machine under its load
//! without
//! any per-boot calibration pass being a hard prerequisite. "Always restore"
//! survives only as a test mode (forced by the caller in gates/benches, not
//! by users - no knobs).

/// Conservative exponential moving average. Deliberately slow (alpha 1/8):
/// the model must not chase one contended transfer into refusing restores
/// for the next hundred hits.
#[derive(Debug, Clone, Copy)]
pub struct Ewma {
    value: f64,
    alpha: f64,
}

impl Ewma {
    pub fn new(seed: f64) -> Self {
        Self {
            value: seed,
            alpha: 0.125,
        }
    }

    pub fn observe(&mut self, sample: f64) {
        self.value += self.alpha * (sample - self.value);
    }

    pub fn get(&self) -> f64 {
        self.value
    }
}

/// What the election is asked about: one off-tier hit for one request.
#[derive(Debug, Clone, Copy)]
pub struct HitShape {
    /// Payload bytes to move if we restore (pack + DMA + unpack all priced
    /// by the measured effective bandwidth, which is end-to-end by
    /// construction - observations time decision-to-GPU-ready).
    pub restore_bytes: u64,
    /// Tokens the restorable span covers - what recompute would re-prefill.
    pub restore_tokens: u32,
    /// Ops already queued ahead on the tier lane, in bytes - the current-
    /// queues term of the model (a restore behind a deep queue is not the same
    /// restore).
    pub queued_bytes: u64,
    /// How many of `restore_bytes` come off T2 (disk) rather than T1 (host
    /// RAM). The two tiers are an order of magnitude apart - the probes
    /// measured 26.7 GB/s over PCIe against 1-15 GB/s off local NVMe and 0.25
    /// off a RAID HDD array - so folding them into one bandwidth EWMA makes
    /// both predictions wrong: the T1 rate sags toward disk while disk
    /// restores get quoted at wire speed and blow their park deadlines. A
    /// chain can also straddle the two (its cold tail evicted to disk while
    /// its head still sits in RAM), so the prediction prices each leg at its
    /// own rate rather than picking a winner.
    pub nvme_bytes: u64,
}

/// The election, with both predictions attached so the caller can log a
/// decision-accountable record and later feed the actual back in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Election {
    Restore { est_us: f64, recompute_us: f64 },
    Recompute { est_us: f64, restore_us: f64 },
}

impl Election {
    pub fn is_restore(&self) -> bool {
        matches!(self, Election::Restore { .. })
    }

    /// The predicted cost of the CHOSEN arm - what `observe_*` compares
    /// against the measured actual.
    pub fn chosen_us(&self) -> f64 {
        match self {
            Election::Restore { est_us, .. } => *est_us,
            Election::Recompute { est_us, .. } => *est_us,
        }
    }
}

/// Fraction of a disk's measured read bandwidth that survives to GPU-ready.
/// See `seed_nvme` for the measurement behind it.
const NVME_END_TO_END: f64 = 0.20;

/// Minimal online cost model: two measured rates, two fixed latencies, one
/// hysteresis margin, one error ledger.
#[derive(Debug)]
pub struct CostModel {
    /// End-to-end restore bandwidth from T1, bytes/us (decision to
    /// GPU-ready: queue + pack + DMA + scatter - observed, not the raw DMA
    /// rate).
    restore_bpus: Ewma,
    /// The same, for T2-sourced restores: disk read + pinned bounce + H2D +
    /// scatter. Seeded from the open-time device probe (`seed_nvme`) so the
    /// first disk restore is priced by this device rather than by a constant
    /// someone chose on different hardware.
    restore_nvme_bpus: Ewma,
    /// Fixed per-restore latency (descriptor upload, event round trip,
    /// scheduler wake), us.
    restore_fixed_us: Ewma,
    /// Prefill throughput under the current serving shape, tokens/us.
    /// Batch-dependent in reality; the EWMA tracks the recent mix, which is
    /// the deliberately minimal stance (a full batch-conditional model comes
    /// later).
    prefill_tpus: Ewma,
    /// True once a prefill was measured in this process: until then
    /// `prefill_tpus` is a seed (generic, or the last run's calibration),
    /// and the first measurement REPLACES it rather than nudging it 1/8 of
    /// the way. Nudging from the generic 50 tok/ms took ~30 prompts to reach
    /// a 27B's ~1.3 tok/ms on GB10 - and every disk restore in between was
    /// priced against a recompute ~40x too fast and refused.
    prefill_measured: bool,
    /// Restore wins only when it beats recompute by this factor - hysteresis
    /// against flapping on near-ties, and a thumb on the scale for the path
    /// that cannot be wrong (recompute is always exact).
    margin: f64,
    /// Predicted-vs-actual accounting: mean absolute percentage error over
    /// the last observations (a drifting model must be visible).
    err_sum_pct: f64,
    err_n: u64,
    /// Test mode: always elect restore (gates/benches force the tier path).
    force_restore: bool,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CostModel {
    /// Seeds: the 26.7 GB/s bus ceiling derated to 60% for the end-to-end path
    /// (pack + wake overhead - replaced by observation within a handful of
    /// restores), 100 us fixed, and a deliberately OPTIMISTIC 50 tok/ms
    /// prefill seed - mispredicting toward recompute costs a missed
    /// optimization, mispredicting toward restore can cost a regression, and
    /// the no-regression gate is the one we preregistered.
    pub fn new() -> Self {
        Self {
            restore_bpus: Ewma::new(26.7e9 * 0.6 / 1e6),
            // conservative until the device says otherwise: the slowest probed
            // viable rung (1 GB/s) through the end-to-end derate
            restore_nvme_bpus: Ewma::new(1.0e9 * NVME_END_TO_END / 1e6),
            restore_fixed_us: Ewma::new(100.0),
            prefill_tpus: Ewma::new(50.0 / 1000.0),
            prefill_measured: false,
            margin: 1.1,
            err_sum_pct: 0.0,
            err_n: 0,
            force_restore: false,
        }
    }

    /// Gates/benches only - never reachable from user config.
    pub fn set_force_restore(&mut self, on: bool) {
        self.force_restore = on;
    }

    /// Seed the T2 rate from the store's open-time device probe (GB/s of
    /// payload). Derated hard: the probe measures the device, while a disk
    /// restore then pays the pinned bounce, the H2D and the scatter in
    /// series. Measured on the A6000 box (granite, 10 sessions cycling off
    /// disk): device 11.34 GB/s, end-to-end 2.08 - 18%. The seed rounds that
    /// to 20% and lets observation take over within a handful of restores.
    /// Erring low is the safe direction by the same logic as the prefill
    /// seed: under-quoting the tier costs a missed optimization, over-quoting
    /// it costs a regression on the path the no-regression gate watches.
    pub fn seed_nvme(&mut self, device_gbs: f64) {
        if device_gbs > 0.0 {
            self.restore_nvme_bpus = Ewma::new(device_gbs * 1e9 * NVME_END_TO_END / 1e6);
        }
    }

    /// A backend's first actual span replaces the generic boot seed. Subsequent
    /// observations still use the conservative EWMA; a CUDA throughput prior
    /// must not take dozens of requests to converge on a smaller Apple GPU.
    pub fn seed_prefill(&mut self, tokens: u32, actual_us: f64) {
        if tokens > 0 && actual_us.is_finite() && actual_us > 0.0 {
            self.prefill_tpus = Ewma::new(tokens as f64 / actual_us);
            self.prefill_measured = true;
        }
    }

    /// Current per-tier restore rates in bytes/us (T1, T2) - an export.
    pub fn rates_bpus(&self) -> (f64, f64) {
        (self.restore_bpus.get(), self.restore_nvme_bpus.get())
    }

    /// Elect for one hit. Pure - no state changes; feedback arrives via the
    /// observe calls.
    pub fn elect(&self, hit: HitShape) -> Election {
        let restore_us = self.predict_restore_us(hit);
        let recompute_us = hit.restore_tokens as f64 / self.prefill_tpus.get().max(1e-9);
        if self.force_restore || restore_us * self.margin < recompute_us {
            Election::Restore {
                est_us: restore_us,
                recompute_us,
            }
        } else {
            Election::Recompute {
                est_us: recompute_us,
                restore_us,
            }
        }
    }

    pub fn predict_restore_us(&self, hit: HitShape) -> f64 {
        let nvme = hit.nvme_bytes.min(hit.restore_bytes);
        let ram = (hit.restore_bytes - nvme) + hit.queued_bytes;
        ram as f64 / self.restore_bpus.get().max(1.0)
            + nvme as f64 / self.restore_nvme_bpus.get().max(1.0)
            + self.restore_fixed_us.get()
    }

    /// Feed a completed restore back: `bytes` moved, `actual_us` measured
    /// decision-to-GPU-ready, `predicted_us` what `elect` said. Updates the
    /// bandwidth EWMA and the error ledger.
    pub fn observe_restore(&mut self, bytes: u64, actual_us: f64, predicted_us: f64) {
        self.observe_restore_from(bytes, actual_us, predicted_us, false)
    }

    /// As [`Self::observe_restore`], attributing the sample to the tier it
    /// actually came from.
    pub fn observe_restore_from(
        &mut self,
        bytes: u64,
        actual_us: f64,
        predicted_us: f64,
        from_nvme: bool,
    ) {
        if actual_us > 0.0 {
            let fixed = self.restore_fixed_us.get();
            if actual_us > fixed && bytes > 0 {
                let rate = bytes as f64 / (actual_us - fixed);
                if from_nvme {
                    self.restore_nvme_bpus.observe(rate);
                } else {
                    self.restore_bpus.observe(rate);
                }
            }
            self.record_error(predicted_us, actual_us);
        }
    }

    /// Feed a measured prefill span back (tokens computed, wall us). The
    /// first one in a process replaces the seed (see `prefill_measured`).
    pub fn observe_prefill(&mut self, tokens: u32, actual_us: f64) {
        if !self.prefill_measured {
            self.seed_prefill(tokens, actual_us);
        } else if actual_us > 0.0 && tokens > 0 {
            self.prefill_tpus.observe(tokens as f64 / actual_us);
        }
    }

    /// This process's measured rates, once a prefill was measured - what a
    /// restart should start from instead of the generic seeds.
    pub fn calibration(&self) -> Option<Calibration> {
        self.prefill_measured.then(|| Calibration {
            prefill_tpus: self.prefill_tpus.get(),
            restore_bpus: self.restore_bpus.get(),
            restore_nvme_bpus: self.restore_nvme_bpus.get(),
            restore_fixed_us: self.restore_fixed_us.get(),
        })
    }

    /// Start from a previous run's calibration. Seeds only: the first live
    /// prefill still replaces the prefill rate (a new build or a different
    /// load may run at another speed), and every rate keeps learning.
    pub fn adopt_calibration(&mut self, c: Calibration) {
        self.prefill_tpus = Ewma::new(c.prefill_tpus);
        self.restore_bpus = Ewma::new(c.restore_bpus);
        self.restore_nvme_bpus = Ewma::new(c.restore_nvme_bpus);
        self.restore_fixed_us = Ewma::new(c.restore_fixed_us);
    }

    fn record_error(&mut self, predicted: f64, actual: f64) {
        if actual > 0.0 {
            self.err_sum_pct += ((predicted - actual) / actual).abs() * 100.0;
            self.err_n += 1;
        }
    }

    /// Mean absolute percentage error of restore predictions so far.
    /// None until anything was observed.
    pub fn prediction_error_pct(&self) -> Option<f64> {
        (self.err_n > 0).then(|| self.err_sum_pct / self.err_n as f64)
    }
}

/// The cost model's measured rates, persisted beside the T2 store so a
/// restart prices its first disk hits by this machine instead of the generic
/// seeds. A restart is exactly when that matters: RAM starts empty, so every
/// hit before the first prefill completes comes off disk - and the generic
/// prefill seed refused those as slower than recompute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    pub prefill_tpus: f64,
    pub restore_bpus: f64,
    pub restore_nvme_bpus: f64,
    pub restore_fixed_us: f64,
}

const CALIBRATION_MAGIC: &str = "paddock-kv-cost v1";

impl Calibration {
    /// Text form, tagged with the device it was measured on.
    pub fn encode(&self, device: &str) -> String {
        format!(
            "{CALIBRATION_MAGIC}\ndevice {}\nprefill_tpus {}\nrestore_bpus {}\nrestore_nvme_bpus {}\nrestore_fixed_us {}\n",
            device.replace('\n', " "),
            self.prefill_tpus,
            self.restore_bpus,
            self.restore_nvme_bpus,
            self.restore_fixed_us
        )
    }

    /// Parse `text` if it was measured on `device` and every rate is sane.
    /// Anything else - another GPU, an older format, a torn write - is
    /// ignored: the generic seeds are the safe fallback.
    pub fn decode(text: &str, device: &str) -> Option<Self> {
        let mut lines = text.lines();
        if lines.next()? != CALIBRATION_MAGIC {
            return None;
        }
        let mut field = |name: &str| -> Option<&str> {
            let line = lines.next()?;
            line.strip_prefix(name)?.strip_prefix(' ')
        };
        if field("device")? != device.replace('\n', " ") {
            return None;
        }
        let mut num = |name: &str| -> Option<f64> {
            field(name)?
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && *v > 0.0)
        };
        Some(Self {
            prefill_tpus: num("prefill_tpus")?,
            restore_bpus: num("restore_bpus")?,
            restore_nvme_bpus: num("restore_nvme_bpus")?,
            restore_fixed_us: num("restore_fixed_us")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(bytes: u64, tokens: u32) -> HitShape {
        HitShape {
            restore_bytes: bytes,
            restore_tokens: tokens,
            queued_bytes: 0,
            nvme_bytes: 0,
        }
    }

    /// Recompute estimate for `tokens` - the prefill rate, read back.
    fn recompute_us(m: &CostModel, tokens: u32) -> f64 {
        match m.elect(HitShape {
            restore_bytes: u64::MAX >> 8,
            ..hit(0, tokens)
        }) {
            Election::Recompute { est_us, .. } => est_us,
            Election::Restore { .. } => unreachable!("restore cannot win at that size"),
        }
    }

    /// The generic seed is a guess; the first measured prefill replaces it
    /// instead of nudging it 1/8 of the way - from 50 tok/ms that nudge took
    /// ~30 prompts to reach a 27B's ~1.3 tok/ms, and every disk restore in
    /// between was refused against a recompute ~40x too fast.
    #[test]
    fn the_first_measured_prefill_replaces_the_seed() {
        let mut m = CostModel::new();
        assert!(m.calibration().is_none(), "nothing measured yet");
        m.observe_prefill(10_000, 8_000_000.0); // 1.25 tok/ms
        assert!((recompute_us(&m, 10_000) - 8_000_000.0).abs() < 1.0);
        // later samples smooth, as before
        m.observe_prefill(10_000, 4_000_000.0);
        let us = recompute_us(&m, 10_000);
        assert!(us < 8_000_000.0 && us > 6_000_000.0, "one 1/8 step: {us}");
        assert!(m.calibration().is_some());
    }

    /// A restart starts from the last run's rates on this device - its first
    /// hits all come off disk, before any prefill is measured - and the first
    /// live prefill still replaces the carried-over rate.
    #[test]
    fn a_calibration_round_trips_for_its_own_device_only() {
        let mut m = CostModel::new();
        m.observe_prefill(10_000, 8_000_000.0);
        let c = m.calibration().expect("measured");
        let text = c.encode("NVIDIA GB10 sm_121 48sm");
        assert_eq!(
            Calibration::decode(&text, "NVIDIA GB10 sm_121 48sm"),
            Some(c)
        );
        assert_eq!(
            Calibration::decode(&text, "NVIDIA RTX PRO 6000 sm_120 188sm"),
            None
        );
        assert_eq!(
            Calibration::decode(&text[..text.len() / 2], "NVIDIA GB10 sm_121 48sm"),
            None
        );
        assert_eq!(Calibration::decode("prefill_tpus 1\n", "x"), None);
        let mut fresh = CostModel::new();
        fresh.adopt_calibration(c);
        assert!(
            (recompute_us(&fresh, 10_000) - 8_000_000.0).abs() < 1.0,
            "adopted"
        );
        assert!(
            fresh.calibration().is_none(),
            "carried over, not measured here"
        );
        fresh.observe_prefill(10_000, 5_000_000.0);
        assert!(
            (recompute_us(&fresh, 10_000) - 5_000_000.0).abs() < 1.0,
            "replaced live"
        );
    }

    /// The two tiers must not share a rate: a disk hit priced at PCIe speed
    /// is how a restore blows its park deadline and the request stalls
    /// behind IO it should have recomputed.
    #[test]
    fn tiers_are_priced_separately_and_observed_separately() {
        let mut m = CostModel::new();
        let t1 = hit(64 << 20, 8192);
        let t2 = HitShape {
            nvme_bytes: 64 << 20,
            ..hit(64 << 20, 8192)
        };
        assert!(
            m.predict_restore_us(t2) > m.predict_restore_us(t1) * 5.0,
            "disk must be priced far above host RAM"
        );
        // a slow disk sample must not contaminate the T1 rate
        let (t1_before, _) = m.rates_bpus();
        m.observe_restore_from(64 << 20, 500_000.0, 100_000.0, true);
        let (t1_after, t2_after) = m.rates_bpus();
        assert_eq!(t1_before, t1_after, "T1 rate moved on a T2 sample");
        assert!(
            t2_after < t1_after,
            "T2 rate should have sagged toward the sample"
        );
        // and the probe seed lands where the device says
        m.seed_nvme(10.0);
        let (_, seeded) = m.rates_bpus();
        assert!(seeded > t2_after, "a fast device must re-seed upward");
        // a straddling chain is priced as the sum of its legs, between the two
        let mixed = HitShape {
            nvme_bytes: 32 << 20,
            ..hit(64 << 20, 8192)
        };
        let (lo, hi) = (m.predict_restore_us(t1), m.predict_restore_us(t2));
        let mid = m.predict_restore_us(mixed);
        assert!(
            mid > lo && mid < hi,
            "mixed hit {mid} not between {lo} and {hi}"
        );
    }

    #[test]
    fn big_prefix_restores_small_prefix_recomputes() {
        let m = CostModel::new();
        // 256 blocks x ~3 MiB record vs 4096 tokens of prefill: restore.
        assert!(m.elect(hit(768 << 20, 4096)).is_restore());
        // one block of payload vs 16 tokens: fixed latency dominates,
        // recompute.
        assert!(!m.elect(hit(3 << 20, 16)).is_restore());
    }

    #[test]
    fn queue_depth_moves_the_crossover() {
        let m = CostModel::new();
        let shape = HitShape {
            restore_bytes: 64 << 20,
            restore_tokens: 1024,
            queued_bytes: 0,
            nvme_bytes: 0,
        };
        let idle = m.elect(shape);
        let queued = m.elect(HitShape {
            queued_bytes: 2 << 30,
            ..shape
        });
        // idle: 64 MiB ~ 4 ms vs 1024 tok ~ 20 ms -> restore. Behind 2 GiB
        // of queue the same hit must flip to recompute.
        assert!(idle.is_restore());
        assert!(!queued.is_restore());
    }

    #[test]
    fn hysteresis_biases_near_ties_toward_recompute() {
        let m = CostModel::new();
        // construct a near-tie: predicted restore just under recompute but
        // inside the 10% margin -> recompute wins.
        let bpus = 26.7e9 * 0.6 / 1e6;
        let tokens = 2000u32;
        let recompute_us = tokens as f64 / (50.0 / 1000.0);
        let bytes = ((recompute_us * 0.95 - 100.0) * bpus) as u64;
        let e = m.elect(hit(bytes, tokens));
        assert!(
            !e.is_restore(),
            "5% under must lose to the 10% margin: {e:?}"
        );
    }

    #[test]
    fn observations_move_the_model_and_fill_the_error_ledger() {
        let mut m = CostModel::new();
        let shape = hit(256 << 20, 4096);
        let before = m.predict_restore_us(shape);
        // the box turns out 4x slower than the seed - say so repeatedly
        for _ in 0..64 {
            let p = m.predict_restore_us(shape);
            m.observe_restore(shape.restore_bytes, before * 4.0, p);
        }
        let after = m.predict_restore_us(shape);
        assert!(
            after > before * 2.0,
            "EWMA must converge toward observed: {before} -> {after}"
        );
        let err = m.prediction_error_pct().expect("observed");
        assert!(err > 0.0);
    }

    #[test]
    fn prefill_observations_shift_recompute() {
        let mut m = CostModel::new();
        let shape = hit(48 << 20, 512);
        // measured prefill 10x faster than the seed: recompute becomes the
        // winner for this mid-size hit
        for _ in 0..64 {
            m.observe_prefill(512, 512.0 / (500.0 / 1000.0));
        }
        assert!(!m.elect(shape).is_restore());
    }

    #[test]
    fn force_restore_is_absolute() {
        let mut m = CostModel::new();
        m.set_force_restore(true);
        assert!(m.elect(hit(16, 1)).is_restore());
    }
}
