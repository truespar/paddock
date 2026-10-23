//! The admission gates' dev knobs: overlapped admissions and the
//! bunching holds (all opt-in, default off).

pub(super) fn overlap_admit_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_OVERLAP_ADMIT_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            // default off (falsified): with the pipe live, an overlapped
            // prefill stalls decode past the 2 queued ticks and churn-shaped
            // load loses badly. The mixed flow keeps decode rows riding the
            // admission pass; arrivals drain the pipe instead.
            .unwrap_or(0)
    })
}

/// Min-free admission gate (PADDOCK_ADM_MIN_FREE=K, opt-in, default 0=off).
/// At high concurrency the steady state staggers completions, so admissions
/// sprinkle: ~every decode round carries a prefill chunk, and each mixed pass
/// re-streams the touched experts' weights no matter how few prompt rows ride
/// it. The observed fast serving mode BUNCHES refills instead.
/// This gate holds admissions until K slots are free
/// (or ADM_MAX_HOLD_MS passes), so a refill lands as one shared wave - the
/// mixed-tick budget (8192 rows, 32 chunks) swallows it in a single pass.
/// Engages only when the server has >= 2K slots; c1..c32 serving is
/// unaffected unless explicitly armed.
pub(super) fn adm_min_free() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_MIN_FREE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Predictive admission bunching (PADDOCK_ADM_PREDICT_K, default 0 = off):
/// under ignore_eos the per-slot remaining tokens are exact, so when fewer
/// than K slots are free but at least K will be free within H rounds
/// (PADDOCK_ADM_PREDICT_H), hold the refill and admit the bunch together -
/// one shared prefill wave instead of a sprinkle. The hold self-expires (the
/// counted slots complete within H rounds by construction), bounding the
/// added TTFT at ~H rounds worst-case and ~zero once the cohort phase-locks
/// (the wm3 fast-mode evidence: bunched boots ran -2.5ms/round ITL).
pub(super) fn adm_predict_k() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_PREDICT_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

pub(super) fn adm_predict_h() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_PREDICT_H")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}

pub(super) fn adm_max_hold_ms() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_MAX_HOLD_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(25)
    })
}

/// In-leg cohort resync (PADDOCK_ADM_RESYNC=R, default 0 = off): the K=16
/// pipe gate keeps an aligned cohort but cannot FORCE alignment from a
/// staggered start (pipe-gate A/B: 1 lock / 6 legs, the lock born from a
/// full-drain wave). This forces it once per leg: after the cohort first
/// FILLS, hold the pipe poll until R slots are free, admit them as one
/// giant wave (a manufactured drain), then fall back to K maintenance.
/// Re-arms only when the server drains (leg boundary), so steady serving
/// pays the hold exactly once per leg.
pub(super) fn adm_resync() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_RESYNC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}
