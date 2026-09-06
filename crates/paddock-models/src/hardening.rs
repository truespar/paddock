//! The wall between development switches and a shipped binary.
//!
//! Paddock ships tested, elected settings; a user must not be able to A/B a
//! kernel or scheduler route on their own hardware. Same rule as the benchmark
//! protocol - vanilla defaults only, a bad result gets an engine fix and never
//! a config knob - and the validated-arch allowlist, where supported means
//! MEASURED. So `hardened` is not "release minus
//! strings": it is the build where the election is the only path, and the
//! switch that would have chosen otherwise is not there to be found.
//!
//! **`hardened` is a THIRD flavour, not a synonym for `--release`.**
//! Benchmarking and profiling use `--release` and need the real names and every
//! switch - release-mode profiling matters less than developer profiling, which
//! matters a lot. Only `--release --features hardened` (what the packaging
//! scripts pass) strips.
//!
//! # Two mechanisms, and both are needed
//!
//! 1. [`dev_var_os!`], [`dev_var!`] and [`dev_on!`] replace the raw `std::env`
//!    reads. Under `hardened` the read AND its NAME leave the token stream -
//!    a `#[cfg]` inside the macro, not a runtime `if`, so nothing survives for
//!    `strings` to find and nothing rests on the optimiser having felt like
//!    folding a constant that day.
//!
//! 2. [`seal_environment`] deletes every `PADDOCK_*` variable the process did
//!    not put there itself, at startup, before anything reads one. This is the
//!    half that reaches the CUDA PACK. The pack has its own `pd_env` reader
//!    (abi.cuh) and the engine elects tuned defaults *into the environment*
//!    for it to find (engine `envset::set_env` - the channel exists because
//!    Windows gives the nvcc-built pack its own UCRT env snapshot). Compiling
//!    out the Rust reads alone would therefore leave every pack-side kill
//!    switch live on the shell. Sealing first and electing after keeps the
//!    engine->pack channel working exactly as it was measured, while making a
//!    user-supplied value unreachable.
//!
//! # Which names survive, and why that is not a judgement call
//!
//! Four populations, and only the first is compiled out:
//!
//! - **Internal election** (the bulk): read by the engine, never written by
//!   it, never part of an operator's surface. `PADDOCK_NO_*`, `TC5*`, `G4_*`,
//!   `F8*`, `DNC_*`, `SPEC_*`... -> gone.
//! - **Engine->pack transport**: any name the engine writes (`envset::set_env`,
//!   including the tuned-defaults tables). Its reader has to stay live or the
//!   election silently stops arriving - the loudest way to ship a binary
//!   nobody measured. These names remain visible in the binary; the
//!   seal is what makes them inert to a user.
//! - **Operator surface**: what `merge_env` in the runner's and manager's
//!   config reads - the documented env spelling of config/CLI. Stays.
//! - **Escape hatches**: `PADDOCK_ALLOW_VRAM_OVERCOMMIT`, `PADDOCK_DATA`,
//!   `PADDOCK_RUNNER_BIN`. (`PADDOCK_UNVALIDATED_ARCH` was one until
//!   2026-09-06; an unvalidated arch now serves under a startup warning.)
//!   Deliberately "do something unsupported", they name themselves in the
//!   refusal text they belong to, and they are not tuning. Stay.
//!
//! The seal's allowlist lives beside each binary's `merge_env` for that
//! reason: the two can only drift if someone edits one and not the other in
//! the same file.

/// True when this crate was built for a shipped binary.
///
/// Read it for behaviour that must differ (the seal); do not read it as
/// `if HARDENED { .. } else { std::env::var(..) }` to gate a switch - that
/// leaves the name in `.rodata` and hopes the optimiser removes it. Use the
/// macros, which never emit the literal at all.
pub const HARDENED: bool = cfg!(feature = "hardened");

/// Truthy read for election/kill envs: set, non-empty, and not `"0"`.
///
/// The house contract for engine-elected defaults is "the env always wins,
/// `FOO=0` reverts", which a bare presence check cannot honour once a default
/// has been filled in. Pack-side twin: `pd_env_on` in abi.cuh; engine-side
/// caller for *dynamic* keys: `envset::env_on`.
pub fn env_on(key: &str) -> bool {
    std::env::var_os(key).is_some_and(|v| !v.is_empty() && v != "0")
}

/// Remove every `PADDOCK_*` variable this process inherited that is not in
/// `keep`, and return the names removed (sorted, for the startup log).
///
/// No-op unless the build is `hardened` - a dev box A/Bs from the shell all
/// day and that is the point of a dev box.
///
/// Call this first, before config merge and before any engine code runs: the
/// contract is "nothing a user typed is in the environment by the time an
/// election is read", and the engine writes its own elections afterwards.
///
/// SAFETY: `remove_var` is only sound before other threads exist. That is why
/// this belongs at the top of `main` and nowhere else.
pub fn seal_environment(keep: &[&str]) -> Vec<String> {
    if !HARDENED {
        return Vec::new();
    }
    let mut removed: Vec<String> = std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .filter(|k| k.starts_with("PADDOCK_") && !keep.contains(&k.as_str()))
        .collect();
    removed.sort();
    for k in &removed {
        // SAFETY: documented single-threaded precondition above.
        unsafe { std::env::remove_var(k) };
    }
    removed
}

/// `std::env::var_os` for a development switch: `None` in a shipped build,
/// and the name is not in the binary to be found.
#[macro_export]
macro_rules! dev_var_os {
    ($name:literal) => {{
        $crate::hardening_chained!();
        #[cfg(feature = "hardened")]
        let __v: ::core::option::Option<::std::ffi::OsString> = ::core::option::Option::None;
        #[cfg(not(feature = "hardened"))]
        let __v = ::std::env::var_os($name);
        __v
    }};
}

/// `std::env::var` for a development switch: `Err(NotPresent)` in a shipped
/// build, so every `.ok()` / `.as_deref()` / match arm downstream falls
/// through to the same default it takes today with the variable unset.
#[macro_export]
macro_rules! dev_var {
    ($name:literal) => {{
        $crate::hardening_chained!();
        #[cfg(feature = "hardened")]
        let __v: ::core::result::Result<::std::string::String, ::std::env::VarError> =
            ::core::result::Result::Err(::std::env::VarError::NotPresent);
        #[cfg(not(feature = "hardened"))]
        let __v = ::std::env::var($name);
        __v
    }};
}

/// [`env_on`] for a development switch: `false` in a shipped build.
#[macro_export]
macro_rules! dev_on {
    ($name:literal) => {{
        $crate::hardening_chained!();
        #[cfg(feature = "hardened")]
        let __v = false;
        #[cfg(not(feature = "hardened"))]
        let __v = $crate::hardening::env_on($name);
        __v
    }};
}

/// Compile error if the calling crate uses a `dev_*` switch without declaring
/// and chaining its own `hardened` feature.
///
/// The `#[cfg]` inside a `macro_rules!` body is resolved against the CALLER's
/// features, so a crate that forgets the chain gets a silently live switch in
/// a shipped binary - the precise rot this task exists to stop. Here it is a
/// build failure instead, at the first offending call site.
#[macro_export]
macro_rules! hardening_chained {
    () => {
        const _: () = ::core::assert!(
            !($crate::hardening::HARDENED && !::core::cfg!(feature = "hardened")),
            "this crate uses a dev_* switch but does not declare/chain the \
             `hardened` feature, so the switch would stay live in a shipped \
             build - add `hardened = [\"paddock-models/hardened\", ..]` to its \
             Cargo.toml"
        );
    };
}

#[cfg(test)]
mod tests {
    /// The whole point, stated as a test: with the feature off the macros are
    /// ordinary env reads, and with it on they are the unset answer. Only the
    /// arm this build compiled can be asserted, which is why both are here.
    #[test]
    fn dev_switches_read_the_environment_only_in_a_dev_build() {
        // SAFETY: single-threaded test, no other reader of this name.
        unsafe { std::env::set_var("PADDOCK_HARDENING_SELFTEST", "1") };
        let on = crate::dev_var_os!("PADDOCK_HARDENING_SELFTEST").is_some();
        assert_eq!(on, !super::HARDENED);
        assert_eq!(
            crate::dev_on!("PADDOCK_HARDENING_SELFTEST"),
            !super::HARDENED
        );
        assert_eq!(
            crate::dev_var!("PADDOCK_HARDENING_SELFTEST").is_ok(),
            !super::HARDENED
        );
        unsafe { std::env::remove_var("PADDOCK_HARDENING_SELFTEST") };
    }

    #[test]
    fn env_on_honours_the_zero_reverts_contract() {
        unsafe { std::env::set_var("PADDOCK_HARDENING_ZERO", "0") };
        assert!(!super::env_on("PADDOCK_HARDENING_ZERO"));
        unsafe { std::env::set_var("PADDOCK_HARDENING_ZERO", "1") };
        assert!(super::env_on("PADDOCK_HARDENING_ZERO"));
        unsafe { std::env::remove_var("PADDOCK_HARDENING_ZERO") };
    }

    /// A dev build must not seal - the A/B loop on a dev box depends on it.
    #[test]
    fn the_seal_is_a_no_op_off_the_hardened_flavour() {
        unsafe { std::env::set_var("PADDOCK_HARDENING_SEALTEST", "1") };
        let removed = super::seal_environment(&["PADDOCK_DATA"]);
        if super::HARDENED {
            assert!(removed.iter().any(|k| k == "PADDOCK_HARDENING_SEALTEST"));
            assert!(std::env::var_os("PADDOCK_HARDENING_SEALTEST").is_none());
        } else {
            assert!(removed.is_empty());
            assert!(std::env::var_os("PADDOCK_HARDENING_SEALTEST").is_some());
            unsafe { std::env::remove_var("PADDOCK_HARDENING_SEALTEST") };
        }
    }
}
