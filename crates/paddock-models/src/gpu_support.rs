//! Which graphics cards paddock serves.
//!
//! HARDCODED, deliberately: this is not configuration and no user may see or
//! touch it. What we serve is a claim about what we have
//! TESTED, so it changes when a bring-up campaign closes and at no other time
//! - a file somebody could edit would let a machine claim support that nobody
//!   ever measured.
//!
//! It lives here because this is the crate the engine and the manager both
//! already depend on, and they need the same answer for different reasons: the
//! engine gates on it at preflight (serve or refuse), the manager tells the
//! user with it (what you have, what we run, what to do). A second copy in
//! either would be a second copy to drift, and drift is not hypothetical: the
//! same fact has been wrong in three places at once (a kernel pack
//! months stale, a Windows build list three GPU generations behind its Linux
//! twin, and a B200 refused for three days after its campaign closed).
//!
//! Card names are NVIDIA's own, from
//! (https://developer.nvidia.com/cuda/gpus). They are here so a person can
//! find their card without knowing what a compute capability is: every
//! user-facing surface says "NVIDIA RTX A6000", never "sm_86".

use serde::Serialize;

/// How far a compute generation has got. Only `Supported` is a promise, and
/// it is a promise about TESTING rather than compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    /// Bring-up campaign closed: kernels tuned on the real die, parity suites
    /// green, throughput measured. The engine serves it.
    Supported,
    /// Campaign OPEN. Serves, with the campaign named in the startup warning
    /// that stamps the run UNVALIDATED, so a mid-campaign number can never
    /// masquerade as a supported result.
    Bringup,
    /// The next campaigns, not a shortfall: the engine refuses, because
    /// loading is not serving.
    ///
    /// Whether the card can LOAD the kernels is a separate question - read
    /// `in_pack`, not this. The two used to be the same thing, back when the
    /// pack was the dev `.so` with every arch in it; a release now welds an
    /// elected subset into the binary, and the fatbin carries no PTX, so most
    /// `Built` dies cannot load anything from a shipped build.
    Built,
    /// Below Ampere. The Q8_0 serving path is built on the int8 dp4a/mma
    /// ladder that first ships at 8.0; there is nothing to tune here and there
    /// never will be.
    TooOld,
}

impl Status {
    /// Whether the engine serves on it without an override.
    pub fn serves(self) -> bool {
        matches!(self, Status::Supported)
    }
}

/// One compute generation, with the cards a person would recognise it by.
#[derive(Debug, Clone, Serialize)]
pub struct Arch {
    /// compute capability as `(major, minor)` - a pair rather than a string,
    /// so a typo is a compile error instead of a card silently dropping out
    /// of the supported list
    pub cc: (u32, u32),
    pub status: Status,
    /// the generation's marketing name, for surfaces that would rather say
    /// that than a number
    pub name: &'static str,
    /// What closed the campaign - evidence, not a slogan. Only on `Supported`.
    pub campaign: Option<&'static str>,
    /// Whether we claim this generation is OPTIMIZED: smoke-tested, and
    /// measured fast enough on it that we would stand behind the claim.
    ///
    /// A performance claim, so it is set by HAND - never derived, never
    /// implied by `status`. It has to be defensible on demand, and `results`
    /// is where the evidence lives.
    pub optimized: bool,
    /// Where the measurements behind `optimized` are recorded. Required
    /// reading before `optimized` goes true, and the thing to open when
    /// somebody asks "compared to what".
    pub board: Option<&'static str>,
    /// Whether the SHIPPED kernel pack carries SASS for this die - a BUILD
    /// fact, deliberately separate from `status`, which is a serving promise.
    ///
    /// Every `Supported` die must have it (a card we promise to serve and then
    /// cannot load kernels for is the worst of both, and the test below
    /// enforces it). It is also true for dies we have not validated, and that
    /// is the whole point: the fatbin carries no PTX, so without SASS on
    /// board an unvalidated card - which serves under the UNVALIDATED
    /// warning, and that is how a bring-up campaign STARTS - has nothing to
    /// run. Found when the release began welding a validated-only pack into
    /// the binary: the (then) escape hatch still opened, onto a drop.
    ///
    /// Not serialized: this describes how we build, not what a user's card is,
    /// and the manager's supported-GPU sheet has no business rendering it.
    #[serde(skip)]
    pub in_pack: bool,
    pub datacenter: &'static [&'static str],
    pub workstation: &'static [&'static str],
    pub jetson: &'static [&'static str],
}

impl Arch {
    /// Every card name, in the order a reader scans them.
    pub fn cards(&self) -> impl Iterator<Item = &'static str> {
        self.datacenter
            .iter()
            .chain(self.workstation)
            .chain(self.jetson)
            .copied()
    }
}

/// Shorthand so the table below reads as data rather than as struct literals.
const fn arch(
    cc: (u32, u32),
    status: Status,
    name: &'static str,
    datacenter: &'static [&'static str],
    workstation: &'static [&'static str],
    jetson: &'static [&'static str],
) -> Arch {
    Arch {
        cc,
        status,
        name,
        campaign: None,
        optimized: false,
        board: None,
        in_pack: false,
        datacenter,
        workstation,
        jetson,
    }
}

/// The table, newest generation first.
///
/// `campaign` is filled in only where the status claims something that needs
/// evidence. `optimized` is dark on every row until somebody who has read the
/// boards sets it - the right default for a claim we would have to defend.
pub static ALL: &[Arch] = &[
    Arch {
        ..arch(
            (12, 1),
            Status::Built,
            "Blackwell (Jetson)",
            &[],
            &[],
            &["NVIDIA GB10 (DGX Spark)"],
        )
    },
    Arch {
        campaign: Some("RTX PRO 6000 - kernels tuned and parity-gated on the die"),
        in_pack: true,
        ..arch(
            (12, 0),
            Status::Supported,
            "Blackwell",
            &[
                "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "NVIDIA RTX PRO 4500 Blackwell Server Edition",
            ],
            &[
                "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                "NVIDIA RTX PRO 5000 Blackwell",
                "NVIDIA RTX PRO 4500 Blackwell",
                "NVIDIA RTX PRO 4000 Blackwell",
                "NVIDIA RTX PRO 4000 Blackwell SFF Edition",
                "NVIDIA RTX PRO 2000 Blackwell",
                "GeForce RTX 5090",
                "GeForce RTX 5080",
                "GeForce RTX 5070 Ti",
                "GeForce RTX 5070",
                "GeForce RTX 5060 Ti",
                "GeForce RTX 5060",
                "GeForce RTX 5050",
            ],
            &[],
        )
    },
    Arch {
        ..arch(
            (11, 0),
            Status::Built,
            "Blackwell (Jetson)",
            &[],
            &[],
            &["Jetson T5000", "Jetson T4000"],
        )
    },
    Arch {
        ..arch(
            (10, 3),
            Status::Built,
            "Blackwell Ultra",
            &["NVIDIA GB300", "NVIDIA B300"],
            &["NVIDIA GB300 (DGX Station)"],
            &[],
        )
    },
    Arch {
        campaign: Some(
            "B200 - kernels tuned on the die: f8w8 through the software ue8m0 fold, the \
             f8t perplexity gate, the cold-prefill profile, and a batch-9 decode cliff \
             found and fixed",
        ),
        in_pack: true,
        ..arch(
            (10, 0),
            Status::Supported,
            "Blackwell (data center)",
            &["NVIDIA GB200", "NVIDIA B200"],
            &[],
            &[],
        )
    },
    Arch {
        ..arch(
            (9, 0),
            Status::Built,
            "Hopper",
            &["NVIDIA GH200", "NVIDIA H200", "NVIDIA H100"],
            &[],
            &[],
        )
    },
    Arch {
        // The one UNVALIDATED DIE we SHIP KERNELS FOR.
        // Ada is the cheap-rental sweet spot named,
        // so it is where a bring-up campaign realistically starts - and a
        // campaign starts by running a SHIPPED binary on it (it serves under
        // the UNVALIDATED warning). Without SASS on board that opens onto
        // nothing, because the fatbin carries no PTX. ~5 MB to keep it honest.
        //
        // It is still `Built`: the engine stamps it, no board exists, and
        // nothing here is a claim that Ada is fast. Carrying the kernels is a
        // build decision; supporting is a measurement.
        in_pack: true,
        ..arch(
            (8, 9),
            Status::Built,
            "Ada Lovelace",
            &["NVIDIA L4", "NVIDIA L40", "NVIDIA L40S"],
            &[
                "NVIDIA RTX 6000 Ada",
                "NVIDIA RTX 5000 Ada",
                "NVIDIA RTX 4500 Ada",
                "NVIDIA RTX 4000 Ada",
                "NVIDIA RTX 4000 SFF Ada",
                "NVIDIA RTX 2000 Ada",
                "GeForce RTX 4090",
                "GeForce RTX 4080",
                "GeForce RTX 4070 Ti",
                "GeForce RTX 4070",
                "GeForce RTX 4060 Ti",
                "GeForce RTX 4060",
                "GeForce RTX 4050",
            ],
            &[],
        )
    },
    Arch {
        ..arch(
            (8, 7),
            Status::TooOld,
            "Ampere (Jetson)",
            &[],
            &[],
            &["Jetson AGX Orin", "Jetson Orin NX", "Jetson Orin Nano"],
        )
    },
    Arch {
        campaign: Some("A6000 - original bring-up plus the heavy parity suites"),
        in_pack: true,
        ..arch(
            (8, 6),
            Status::Supported,
            "Ampere",
            &["NVIDIA A40", "NVIDIA A10", "NVIDIA A16", "NVIDIA A2"],
            &[
                "NVIDIA RTX A6000",
                "NVIDIA RTX A5000",
                "NVIDIA RTX A4000",
                "NVIDIA RTX A3000",
                "NVIDIA RTX A2000",
                "GeForce RTX 3090 Ti",
                "GeForce RTX 3090",
                "GeForce RTX 3080 Ti",
                "GeForce RTX 3080",
                "GeForce RTX 3070 Ti",
                "GeForce RTX 3070",
                "GeForce RTX 3060 Ti",
                "GeForce RTX 3060",
                "GeForce RTX 3050 Ti",
                "GeForce RTX 3050",
            ],
            &[],
        )
    },
    Arch {
        ..arch(
            (8, 0),
            Status::Built,
            "Ampere (data center)",
            &["NVIDIA A100", "NVIDIA A30"],
            &[],
            &[],
        )
    },
    Arch {
        ..arch(
            (7, 5),
            Status::TooOld,
            "Turing",
            &["NVIDIA T4"],
            &[
                "QUADRO RTX 8000",
                "QUADRO RTX 6000",
                "QUADRO RTX 5000",
                "QUADRO RTX 4000",
                "QUADRO RTX 3000",
                "QUADRO T2000",
                "NVIDIA T1200",
                "NVIDIA T1000",
                "NVIDIA T600",
                "NVIDIA T500",
                "NVIDIA T400",
                "GeForce GTX 1650 Ti",
                "NVIDIA TITAN RTX",
                "GeForce RTX 2080 Ti",
                "GeForce RTX 2080",
                "GeForce RTX 2070",
                "GeForce RTX 2060",
            ],
            &[],
        )
    },
];

/// What we say about this device's compute capability. Silicon newer than this
/// build knows about is not in the table at all - not loadable, and certainly
/// not servable.
pub fn find(cc: (u32, u32)) -> Option<&'static Arch> {
    ALL.iter().find(|a| a.cc == cc)
}

/// The generations the engine will actually serve.
pub fn supported() -> impl Iterator<Item = &'static Arch> {
    ALL.iter().filter(|a| a.status.serves())
}

/// Can this device hold its KV cache in fp8-e4m3? Yes - on every die we serve.
///
/// This used to RETURN `cc >= (8, 9)`, on the reasoning that e4m3 is a
/// tensor-core format so a die without fp8 tensor cores "cannot do the
/// conversion at all". That reasoning was wrong, and the observation behind it
/// was a real bug wearing a hardware costume.
///
/// Storing an fp8 KV cache needs fp8 STORAGE, not fp8 MATH. The conversion goes
/// through `__nv_fp8_e4m3` from cuda_fp8.h, which is software-emulated below
/// sm_89 and exact: `kv_append_batch` at Fp8E4m3 writes bytes identical to a
/// host e4m3 codec on sm_86, at every magnitude including the saturating ones
/// (gpu_attn_parity.rs, storage arm).
///
/// What actually produced the garbage that got this gate written was two host
/// elections reaching kernel instantiations whose
/// bodies do not exist below sm_89 - the QK8/P8 e4m3-mma arms are
/// `#if __CUDA_ARCH__ >= 890` with no `#else`, so they accumulated nothing and
/// stored zeros while their launch reported success. `pd_fp8_mma_ok()` now
/// gates both. With that fixed, fp8 KV serves correctly on sm_86:
/// perplexity within noise of f16 across 1024..8000 tokens, deltas alternating
/// sign and not growing with depth.
///
/// KNOWN GAP, accepted deliberately: granite's fused writer
/// `rope_norm_qk_append_paged` is the one KV writer with no fp8 evidence either
/// way. Every other family's writer is covered. The decision was to
/// take the capacity now rather than hold it for one kernel, so a granite user
/// who asks for fp8 KV is on an unmeasured path.
///
/// It lives beside the support table for the same reason the table does: three
/// places need this answer and they must not each keep their own copy. The
/// RUNNER enforces it (`serving.rs::apply_kv_dtype`), the ESTIMATOR has to
/// price the width that will actually be served or the will-it-fit panel
/// mis-counts the KV pool by 2x, and the STUDIO greys the control when it
/// cannot work. The seam is kept - answering `true`
/// everywhere today is not the same as deleting the question, and a future die
/// whose e4m3 conversion is broken has a place to say so.
pub fn fp8_kv(_cc: (u32, u32)) -> bool {
    // The arch allowlist has already refused hardware we do not serve by the
    // time anyone asks this, so there is no second floor to invent here.
    true
}

/// Why fp8 KV is unavailable here, in words a person can act on - or `None`
/// when it is available, which is currently every die we serve.
///
/// Kept as the seam rather than deleted: if a die ever does need refusing, this
/// is where the reason goes, and all three consumers already read it.
pub fn fp8_kv_blocked(cc: (u32, u32)) -> Option<&'static str> {
    (!fp8_kv(cc)).then_some("this GPU cannot store an fp8 KV cache")
}

/// The generations the SHIPPED kernel pack compiles SASS for - a superset of
/// `supported()`, and the list both release lanes build against
/// (the release script on Windows, the Linux build script in the container).
///
/// Read, never restated: a second copy of this in a build script is how the
/// Windows pack once sat three GPU generations behind its Linux twin.
pub fn in_pack() -> impl Iterator<Item = &'static Arch> {
    ALL.iter().filter(|a| a.in_pack)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three closed campaigns, by capability - if one of these stops being
    /// true it is a product decision, not an edit.
    #[test]
    fn the_supported_set_is_the_three_closed_campaigns() {
        let mut ccs: Vec<_> = supported().map(|a| a.cc).collect();
        ccs.sort_unstable();
        assert_eq!(ccs, [(8, 6), (10, 0), (12, 0)]);
        for a in supported() {
            assert!(
                a.campaign.is_some(),
                "{:?} claims supported with no campaign",
                a.cc
            );
        }
    }

    /// What the shipped kernel pack carries SASS for. Both release lanes parse
    /// this literal - the release script and the Linux build script - so
    /// editing it moves the actual builds, which is the point: the fatbin has
    /// no PTX, so this list is the set of dies that can load paddock's kernels
    /// at all.
    #[test]
    fn the_shipped_pack_covers_every_supported_die_plus_ada() {
        let mut shipped: Vec<_> = in_pack().map(|a| a.cc).collect();
        shipped.sort_unstable();
        assert_eq!(shipped, [(8, 6), (8, 9), (10, 0), (12, 0)]);
        // A card we promise to serve and then cannot load kernels for is the
        // worst of both, so this direction is not optional.
        for a in supported() {
            assert!(
                a.in_pack,
                "{:?} is served but the shipped pack has no SASS for it",
                a.cc
            );
        }
    }

    /// A card a person owns resolves to a verdict, which is the whole reason
    /// the names are carried at all.
    #[test]
    fn a_card_a_person_owns_resolves() {
        let a = find((8, 6)).expect("Ampere");
        assert!(a.cards().any(|c| c == "NVIDIA RTX A6000"));
        assert!(a.status.serves());
        // Ada LOADS and does not SERVE - the distinction this table exists to
        // keep straight.
        let ada = find((8, 9)).expect("Ada");
        assert_eq!(ada.status, Status::Built);
        assert!(!ada.status.serves());
        assert!(ada.cards().any(|c| c == "GeForce RTX 4090"));
    }

    /// The fp8-KV floor, stated against real dies rather than as a repeated
    /// inequality - the whole point of hoisting it here is that the runner,
    /// the estimator and the Studio now read one answer.
    #[test]
    fn fp8_kv_needs_tensor_cores_that_ampere_does_not_have() {
        // the two ends of the bug this closes: an A6000 that produced garbage
        // from an fp8 cache, and the Blackwell parts where it is the default
        // fp8 KV needs fp8 STORAGE, not fp8 MATH - the e4m3 conversion is
        // software-emulated below sm_89 and byte-exact there.
        // These used to assert the opposite on the tensor-core threshold; the
        // garbage that justified it was the QK8/P8 arms storing zeros, not the
        // format. See the doc comment above.
        assert!(
            fp8_kv((8, 6)),
            "sm_86 Ampere (A6000, 3090) - MEASURED correct"
        );
        assert!(fp8_kv((8, 0)), "sm_80 A100");
        assert!(fp8_kv((7, 5)), "sm_75 Turing");
        assert!(fp8_kv((8, 9)), "sm_89 Ada");
        assert!(fp8_kv((9, 0)), "sm_90 Hopper");
        assert!(fp8_kv((12, 0)), "sm_120 Blackwell");

        // the refusal explains the HARDWARE, never a generation to go buy -
        // Ada has the cores and we still do not serve it
        assert_eq!(fp8_kv_blocked((8, 6)), None, "no longer refused on Ampere");
        assert_eq!(fp8_kv_blocked((12, 0)), None);
        assert_eq!(find((8, 9)).unwrap().status, Status::Built);
    }

    /// Every generation names itself and at least one card, or it cannot be
    /// rendered to anybody.
    #[test]
    fn every_generation_is_presentable() {
        assert!(!ALL.is_empty());
        for a in ALL {
            assert!(!a.name.is_empty(), "{:?} has no generation name", a.cc);
            assert!(a.cards().next().is_some(), "{:?} names no cards", a.cc);
        }
    }

    /// An optimized claim without a board is one we cannot defend, and the
    /// fairness rules say we must be able to. Untested silicon cannot be
    /// optimized at all - that would mean we measured something we do not
    /// serve.
    #[test]
    fn every_optimized_claim_cites_its_board() {
        for a in ALL {
            if a.optimized {
                assert!(
                    a.status.serves(),
                    "{:?} claims optimized without being supported",
                    a.cc
                );
                assert!(
                    a.board.is_some(),
                    "{:?} claims optimized with no board to cite",
                    a.cc
                );
            }
        }
    }
}
