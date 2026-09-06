//! Validated-arch policy - "supported" is a TESTED claim, not a compile
//! flag.
//!
//! Paddock is a specialized engine: a compute generation is SUPPORTED only
//! after its bring-up is complete (kernels tuned on the real die,
//! parity-gated, throughput measured). Every other Ampere-or-newer die still
//! serves - under a startup warning that names it UNVALIDATED, so a number
//! measured on it can never masquerade as a supported result. The warning
//! replaced the refusal and its `PADDOCK_UNVALIDATED_ARCH=1` override
//! (2026-09-06): a person with an untested card gets to run and gets told
//! what that means, instead of being sent to find an environment variable
//! (which the Studio never honoured anyway - it hid the start button on any
//! non-ready card).
//!
//! What stays a hard stop is what cannot run at all: pre-Ampere silicon (no
//! int8 mma ladder to build on - `gpu/mod.rs` refuses it before this gate)
//! and a die the pack carries no SASS for (the trial launch after this gate
//! catches it - the fatbin has no PTX, so there is no limp mode to fall
//! into).
//!
//! The case the refusal covered better is the same-major minor (GB10 / DGX
//! Spark, sm_121): plain sm_120 SASS forward-loads onto it, the trial launch
//! passes, and the `sm_120a`-only tensor-core families - which the pack's
//! per-device table resolves by MAJOR (exports.cuh) - fail at their first
//! launch rather than at startup. The warning says so; a Spark campaign is
//! the fix, not a gate.
//!
//! Lifecycle per generation: unknown -> serves with the warning; in bring-up
//! -> the same, with the campaign named; validated -> listed below, with the
//! campaign that closed it.

// The lists themselves are DATA, not code: `gpu-support.toml` at the repo
// root, parsed once by paddock-models and read here and by the manager alike.
// They used to be consts in this file, which meant the same fact also lived
// in the manager and in the Studio's prose - and all three managed to
// disagree at once. One file, several readers.
use paddock_models::gpu_support::{self, Status};

/// Capabilities whose bring-up campaign has closed, as `(major, minor, why)`.
fn validated() -> Vec<(u32, u32, &'static str)> {
    rows(Status::Supported)
}

/// Capabilities with an open campaign - named in the warning so the state is
/// visible.
fn in_bring_up() -> Vec<(u32, u32, &'static str)> {
    rows(Status::Bringup)
}

fn rows(want: Status) -> Vec<(u32, u32, &'static str)> {
    gpu_support::ALL
        .iter()
        .filter(|a| a.status == want)
        .map(|a| (a.cc.0, a.cc.1, a.campaign.unwrap_or(a.name)))
        .collect()
}

pub(super) enum Gate {
    /// Campaign closed - serve normally.
    Validated,
    /// Unvalidated silicon - serve, WARN with the stamp.
    Unvalidated(String),
}

/// Pure decision, so it is testable.
pub(super) fn gate(cc: (u32, u32), device: &str) -> Gate {
    gate_in(cc, device, &validated(), &in_bring_up())
}

/// The decision against GIVEN lists, so the bring-up branch stays under test
/// while `in_bring_up()` is empty. A branch nobody exercises is a branch that
/// rots, and this one only wakes up when a new generation opens - exactly
/// when it is least convenient to discover it stopped working.
fn gate_in(
    cc: (u32, u32),
    device: &str,
    validated: &[(u32, u32, &str)],
    in_bring_up: &[(u32, u32, &str)],
) -> Gate {
    let (maj, min) = cc;
    if validated.iter().any(|&(a, b, _)| (a, b) == (maj, min)) {
        return Gate::Validated;
    }
    let validated_list = validated
        .iter()
        .map(|&(a, b, _)| format!("sm_{a}{b}"))
        .collect::<Vec<_>>()
        .join(", ");
    let bring_up = in_bring_up
        .iter()
        .find(|&&(a, b, _)| (a, b) == (maj, min))
        .map(|&(.., note)| format!(" This generation's bring-up is IN PROGRESS ({note})."))
        .unwrap_or_default();
    Gate::Unvalidated(format!(
        "SERVING ON UNVALIDATED ARCH sm_{maj}{min} ({device}) - this engine build has \
         validated {validated_list} only.{bring_up} Paddock has not tuned or measured \
         its kernels on this generation: performance is unmeasured, and a kernel \
         family with no image for this die fails at its first launch rather than \
         at startup. Numbers from this machine are bring-up data, NOT supported \
         results; label them so."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validated_dies_serve_silently() {
        assert!(matches!(gate((8, 6), "A6000"), Gate::Validated));
        assert!(matches!(gate((12, 0), "RTX PRO 6000"), Gate::Validated));
        // sm_100 joined when its campaign closed - before that it sat in
        // bring-up, serving under the stamp.
        assert!(matches!(gate((10, 0), "B200"), Gate::Validated));
    }

    /// The GB10 / DGX Spark case: same major as the validated consumer die,
    /// different minor - exact matching must NOT read it as validated (plain
    /// sm_120 SASS forward-loads onto it, which is precisely what made it
    /// half-serve unannounced before this gate). It serves, stamped.
    #[test]
    fn same_major_different_minor_serves_with_the_stamp() {
        let Gate::Unvalidated(warn) = gate((12, 1), "GB10") else {
            panic!("sm_121 must not pass as validated");
        };
        assert!(warn.contains("sm_121"), "{warn}");
        assert!(warn.contains("UNVALIDATED"), "{warn}");
        assert!(warn.contains("first launch"), "{warn}");
    }

    /// A generation with an open campaign: served with the campaign named.
    ///
    /// Driven through a synthetic list because `in_bring_up()` is empty today -
    /// sm_100 was its last occupant and its campaign has closed. The behaviour
    /// has to keep working for whatever opens next, and an untested branch
    /// would not.
    #[test]
    fn bring_up_arch_names_its_campaign_in_the_stamp() {
        const NEXT: &[(u32, u32, &str)] = &[(13, 0, "Rubin - campaign open")];
        let Gate::Unvalidated(warn) = gate_in((13, 0), "Rubin", &validated(), NEXT) else {
            panic!("an open campaign must serve under the stamp");
        };
        assert!(warn.contains("IN PROGRESS"), "{warn}");
        assert!(warn.contains("UNVALIDATED"), "{warn}");
    }

    /// A future major (Rubin-class) gets the same stamp - the trial launch
    /// after the gate is what refuses a die the pack has no image for.
    #[test]
    fn future_major_serves_with_the_stamp() {
        assert!(matches!(gate((13, 0), "Rubin"), Gate::Unvalidated(_)));
    }

    /// The old override is gone for good: nothing in the decision reads the
    /// environment, so a stale `PADDOCK_UNVALIDATED_ARCH` in a shell changes
    /// nothing and the stamp never advertises it.
    #[test]
    fn the_stamp_does_not_advertise_an_override() {
        let Gate::Unvalidated(warn) = gate((8, 9), "GeForce RTX 4090") else {
            panic!("Ada is Built, not Supported");
        };
        assert!(!warn.contains("PADDOCK_"), "{warn}");
    }
}
