//! The speculative round's row budget: how many verify rows a round may
//! take per live width, the per-slot depth inside it, and the operator's
//! speculation policy - the elected tiers and their dev pins.

use crate::generator::Generator;
use crate::spec_policy::SpecPolicy;

/// Serving spec-decode row budget per round: total rows (1 pending + drafts
/// per slot) must fit the models' verify-pass cap. Default 32 matches
/// gpt-oss's SPEC_BATCH_MAX_ROWS and qwen35's alloc; backends whose verify
/// path holds a whole wide round raise it via PADDOCK_SPEC_MAX_ROWS at load
/// (gemma4+MTP sets 160 = 32 slots × (1+4 drafts) - the wide-batch spec
/// rung). Read once: model load runs before the serve loop starts.
pub(super) fn serve_spec_max_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32)
    })
}

/// Per-slot draft-length ceiling (the round's row budget shrinks it further
/// when many slots are active). Default 7 keeps the measured qwen35/gpt-oss
/// behavior; gemma4+MTP raises to 15 at attach (deep chains at low live
/// concurrency).
pub(super) fn serve_spec_max_k() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_MAX_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7)
    })
}

/// Per-round draft budget. For raised row budgets (>32: the gemma4 wide-spec
/// path) this is TWO-TIER: prefer the half-budget round - 32 rows ride the
/// GEMM class's single weight pass (mma_ks BN32 col-tile; 33..64 rows read
/// weights twice) - and take the full budget only when the half-budget
/// yields zero drafts (wide live counts, where 2x weights for k=1 still
/// pays off). Budget <=32 keeps the classic single-tier formula
/// (qwen35/gpt-oss behavior unchanged).
/// Post-miss draft-depth floor: the classic rule drops k_now to the
/// accepted length after any miss - right for weak drafters, but a
/// ~78%/token MTP drafter spends most rounds re-climbing the ladder.
/// Backends with strong drafters raise it (env; default 1 = classic).
pub(super) fn serve_spec_k_floor() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_K_MISS_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    })
}

/// draft-length cap for MIXED ticks only (chunk in flight). The wide-spec
/// loss lives entirely in the mixed regime (a per-tick drafter prologue plus
/// extra whole-weight-stream passes) while wide PURE rounds gain - so mixed
/// and pure rounds get independent width policies. Default MAX = one shared
/// policy (status quo).
pub(super) fn serve_spec_mixed_k_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_MIXED_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX)
    })
}

/// Mixed rounds book a k-independent prompt-chunk surcharge into the Auto
/// controller's lat cells (lives 17..32 share one bucket), which drags the
/// goodput argmax down and leaves a chunk of the PURE rounds running
/// under-depth. Default: mixed rounds skip observe(); =1 restores the old
/// booking.
pub(super) fn spec_book_mixed() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_BOOK_MIXED")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// Async spec round kill: PADDOCK_NO_ASYNC_SPEC=1 restores the
/// synchronous drafts-then-verify sequence (the A/B surface for the
/// chain->verify boundary elimination).
pub(super) fn async_spec_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_ASYNC_SPEC").is_none())
}

/// Narrow-tier 64-row boundary for live 9..=16.
/// PADDOCK_SPEC_NARROW64=0 restores the flat 32-row pin.
fn spec_narrow64_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_NARROW64")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Verify-row boundary of the narrow tier's 9..=16 band (the "64" in
/// NARROW64). Drafter-class dependent: the MTP chain measures k=3 best there
/// (128 rows overshoot the band), while the qwen35 block drafter measures
/// k=7 best once the >64-row GEMM wall fell, so its attach sets 128 here
/// (PADDOCK_SPEC_NARROW_ROWS, gemma4-style env election; explicit env wins).
/// Default 64 = every other backend unchanged.
fn spec_narrow_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_NARROW_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 32)
            .unwrap_or(64)
    })
}

/// Verify-row boundary of the live <= 8 tier. Chain drafters keep the
/// 32-row pin (k3 and k7 measure flat on the MTP chain at c8). A BLOCK
/// drafter's round is one forward over `block` positions per slot, and at
/// the rejection-sampling acceptance the deeper round pays all the way to
/// the block - so the tier is `8 * block` there. PADDOCK_SPEC_LOW_ROWS
/// pins either for the A/B.
fn spec_low_rows(block: Option<usize>) -> usize {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let pin = *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_LOW_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 16)
    });
    pin.unwrap_or_else(|| block.map_or(32, |b| 8 * b.max(4)))
}

pub(super) fn serve_spec_k_budget(live: usize, block: Option<usize>) -> usize {
    let b = serve_spec_max_rows();
    let deep_live_max: usize = std::env::var("PADDOCK_SPEC_DEEP_LIVE_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0); // deep tier off by default: k15 measured worse than k7
    let k = if b > 32 {
        // low live: take the full budget - deep chains repay the wider round
        // now that the 33..64-row GEMM band is pipelined (BN64-ST2), and the
        // drafter's per-token acceptance compounds (what deep chains buy is
        // depth, not drafter quality)
        if live <= deep_live_max {
            (b / live).saturating_sub(1)
        } else {
            // narrow tier, two-step verify-row boundary: 32 rows (the
            // single-BN32-pass GEMM class) for live <= 8, 64 rows (the
            // BN64-ST2 pipelined band) for 9..=16. The flat 32-row pin put
            // live 11..16 at k=1 - a depth cliff bottoming exactly at 16
            // slots while 17 got (b/live)-1 = 6. k=3 recovers that whole
            // valley; k=7 overshoots the band (128 verify rows), and at
            // c8 k3 and k7 are flat. Kill: PADDOCK_SPEC_NARROW64=0 restores
            // the flat 32-row pin.
            let boundary = if live <= 8 {
                spec_low_rows(block)
            } else if !spec_narrow64_on() {
                32
            } else {
                spec_narrow_rows()
            };
            let narrow = if live <= 16 {
                (boundary / live).saturating_sub(1)
            } else {
                0
            };
            if narrow >= 1 {
                narrow
            } else {
                (b / live).saturating_sub(1)
            }
        }
    } else {
        (b / live).saturating_sub(1)
    };
    k.min(serve_spec_max_k())
}

/// The row budget's depth, as the backend qualifies it.
///
/// Narrowed last by whatever depth the backend has measured its drafter to be
/// worth (`spec_depth_cap`). The row budget is a width argument and knows
/// nothing about what a verify row costs on this board, so a backend with a
/// measured depth curve gets the final say downward - never upward, since the
/// rows still have to fit.
pub(super) fn backend_spec_k_budget(generator: &dyn Generator, live: usize) -> usize {
    if live == 0 || live > generator.spec_live_cap() || generator.spec_deferred() {
        return 0;
    }
    let qualified = generator
        .spec_fixed_draft_depth()
        .unwrap_or_else(|| serve_spec_k_budget(live, generator.spec_block_width()));
    let qualified = generator
        .spec_depth_cap()
        .map_or(qualified, |cap| qualified.min(cap));
    generator
        .spec_batch_draft_budget(live)
        .map_or(qualified, |cap| cap.min(qualified))
}

pub(super) fn slot_spec_k(requested: usize, budget: usize, fixed_depth: Option<usize>) -> usize {
    fixed_depth.unwrap_or(requested).min(budget)
}

#[test]
fn fixed_block_depth_bypasses_chain_ramp_but_respects_round_budget() {
    for requested in [0usize, 1, 2, 7, 15, 31] {
        for budget in [0usize, 3, 7, 15, 31] {
            assert_eq!(slot_spec_k(requested, budget, None), requested.min(budget));
            assert_eq!(slot_spec_k(requested, budget, Some(15)), 15.min(budget));
        }
    }
}

/// The operator's speculation policy, from the runner's `spec` config key
/// (threaded in as PADDOCK_SPEC). Read once - it is a serving-envelope choice,
/// not something to re-evaluate per tick.
///
/// Unset = `Ladder`, i.e. exactly the behavior that predates the controller.
/// The legacy kill switches still win outright: an operator who wrote
/// PADDOCK_NO_SPEC meant it, and having a new key quietly outrank an existing
/// "off" would be the worst kind of surprise.
pub(super) fn serve_spec_policy() -> SpecPolicy {
    static V: std::sync::OnceLock<SpecPolicy> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        if std::env::var_os("PADDOCK_NO_SPEC").is_some()
            || paddock_models::dev_var_os!("PADDOCK_NO_SERVE_SPEC").is_some()
        {
            return SpecPolicy::Off;
        }
        match std::env::var("PADDOCK_SPEC") {
            Ok(v) => match v.parse::<SpecPolicy>() {
                Ok(p) => p,
                Err(e) => {
                    // A malformed policy must not silently pick a behavior -
                    // say so and keep the status quo (no silent failures).
                    tracing::warn!("{e}; keeping the default ladder");
                    SpecPolicy::Ladder
                }
            },
            Err(_) => SpecPolicy::Ladder,
        }
    })
}
