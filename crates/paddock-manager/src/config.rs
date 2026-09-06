//! Manager configuration. Deliberately small: the manager's own bind address
//! plus the box-level paths and spawn defaults. Runner serving config is the
//! RUNNER's (per-instance flags/env/toml); the manager's *elections* (which
//! models on which ports) will live in the managed config file (`managed.toml`,
//! doc §7) once the supervisor lands.

use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    /// Loopback by default; the Studio is a local control plane. Binding wider
    /// is an explicit decision (auth turns on automatically, like runners).
    pub host: IpAddr,
    /// Manager/Studio port. 11500 - deliberately below the canonical runner
    /// port (11540) so runner allocation grows upward without collisions.
    pub port: u16,
    /// Directories scanned for installed GGUF files (catalog install state,
    /// estimator probes, and future spawn elections).
    pub model_dirs: Vec<PathBuf>,
    /// Default serving envelope used for estimator math (and, later, as spawn
    /// defaults): context window and continuous-batching width.
    pub max_ctx: usize,
    pub max_batch: usize,
    /// Explicit API key for the manager surface; None + loopback = no auth.
    pub api_key: Option<String>,
    /// A reverse proxy stands in front of the Studio. Then every caller
    /// arrives from 127.0.0.1 and the loopback exemption in `auth_mw` would
    /// be no exemption at all, so it is off and the key is required from
    /// everyone. Off by default: a direct bind, the box's own browser.
    pub trusted_proxy: bool,
    /// Runner binary override (PADDOCK_RUNNER_BIN). Default: the
    /// `paddock-runner` beside this executable; runners/<version>/ election
    /// lands with the artifact scheme (doc §11.1).
    pub runner_bin: Option<PathBuf>,
    /// Device passed to spawned runners. Only "cuda" exists; there is no
    /// CPU path and no auto-fallback.
    pub device: String,
    /// Kernel pack passed to spawned runners (PADDOCK_KERNEL_PACK).
    pub kernel_pack: Option<PathBuf>,
    /// First runner port; allocation grows upward from here (doc §11.3).
    pub runner_base_port: u16,
    /// Spawn health-gate timeout (big models take a while to load).
    pub spawn_timeout_s: u64,
    /// What the collector persists. `Full` = per-request activity
    /// rows AND the usage rollups; `Aggregates` = rollups only (counts
    /// without content - no session ids, no request records), which keeps
    /// the usage history and the batch forecaster alive under the privacy
    /// switch; `Off` = the collector never runs. Not recording is a
    /// first-class configuration, not a degraded one.
    pub activity: ActivityMode,
    /// Days of collected activity to keep (hourly purge). 0 = keep until an
    /// explicit purge.
    pub activity_retention_days: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityMode {
    Full,
    Aggregates,
    Off,
}

impl Config {
    /// The box data dir: models/, logs/, servers/, paddock.db. Resolved by
    /// the shared three-mode ladder  - portable `data\` beside the
    /// exe, installed machine root, or `~/paddock` - so a portable unzip
    /// never reads or writes another install's state.
    pub fn data_dir() -> PathBuf {
        paddock_admin::data_root()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 11500,
            model_dirs: default_model_dirs(),
            max_ctx: 4096,
            max_batch: 32,
            api_key: None,
            trusted_proxy: false,
            runner_bin: None,
            device: "cuda".to_owned(),
            kernel_pack: None,
            runner_base_port: 11540,
            spawn_timeout_s: 300,
            activity: ActivityMode::Full,
            activity_retention_days: 30,
        }
    }
}

/// Find a kernel pack without configuration (portable/installed first run):
/// probe `packs/cuda/` beside the exe, then `<data root>/packs/cuda/`. A dir
/// holding exactly one pack library is unambiguous and wins; empty or
/// ambiguous dirs resolve nothing (configure explicitly). Same
/// drop-a-file-in convention as the pdfium sidecar.
pub fn autodetect_kernel_pack() -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(PathBuf::from))
    {
        dirs.push(d.join("packs").join("cuda"));
    }
    dirs.push(paddock_admin::data_root().join("packs").join("cuda"));
    for dir in dirs {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let packs: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|x| x.eq_ignore_ascii_case("dll") || x.eq_ignore_ascii_case("so"))
            })
            .collect();
        if packs.len() == 1 {
            return Some(packs[0].clone());
        }
    }
    None
}

fn default_model_dirs() -> Vec<PathBuf> {
    // <data root>/models - user-visible, plain files, per the storage
    // principle; the root follows the distribution mode
    vec![paddock_admin::data_root().join("models")]
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("environment override {name} is invalid: {value:?}")]
    BadEnv { name: &'static str, value: String },
}

impl Config {
    /// Overlay `PADDOCK_MANAGER_*` environment variables. An unparseable value
    /// is a loud error, not a panic. (The runner's `PADDOCK_*` names are left
    /// alone - two processes, two prefixes, no ambiguity about who reads what.)
    pub fn merge_env(&mut self) -> Result<(), ConfigError> {
        let get = |name: &str| std::env::var(name).ok().filter(|s| !s.is_empty());
        if let Some(v) = get("PADDOCK_MANAGER_HOST") {
            self.host = v.parse().map_err(|_| ConfigError::BadEnv {
                name: "PADDOCK_MANAGER_HOST",
                value: v,
            })?;
        }
        if let Some(v) = get("PADDOCK_MANAGER_PORT") {
            self.port = v.parse().map_err(|_| ConfigError::BadEnv {
                name: "PADDOCK_MANAGER_PORT",
                value: v,
            })?;
        }
        if let Some(v) = get("PADDOCK_MODEL_DIRS") {
            self.model_dirs = v.split(',').map(|s| PathBuf::from(s.trim())).collect();
        }
        if let Some(v) = get("PADDOCK_MANAGER_API_KEY") {
            self.api_key = Some(v);
        }
        if let Some(v) = get("PADDOCK_MANAGER_TRUSTED_PROXY") {
            self.trusted_proxy = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }
        if let Some(v) = get("PADDOCK_RUNNER_BIN") {
            self.runner_bin = Some(PathBuf::from(v));
        }
        if let Some(v) = get("PADDOCK_DEVICE") {
            self.device = v;
        }
        if let Some(v) = get("PADDOCK_KERNEL_PACK") {
            self.kernel_pack = Some(PathBuf::from(v));
        }
        if let Some(v) = get("PADDOCK_MANAGER_SPAWN_TIMEOUT_S") {
            self.spawn_timeout_s = v.parse().map_err(|_| ConfigError::BadEnv {
                name: "PADDOCK_MANAGER_SPAWN_TIMEOUT_S",
                value: v,
            })?;
        }
        if let Some(v) = get("PADDOCK_MANAGER_ACTIVITY") {
            self.activity = match v.to_ascii_lowercase().as_str() {
                "full" => ActivityMode::Full,
                "aggregates" => ActivityMode::Aggregates,
                "off" => ActivityMode::Off,
                _ => {
                    return Err(ConfigError::BadEnv {
                        name: "PADDOCK_MANAGER_ACTIVITY",
                        value: v,
                    });
                }
            };
        }
        // The pre-§6 boolean, kept as a synonym for "off".
        if get("PADDOCK_MANAGER_NO_ACTIVITY").is_some() {
            self.activity = ActivityMode::Off;
        }
        if let Some(v) = get("PADDOCK_ACTIVITY_RETENTION_DAYS") {
            self.activity_retention_days = v.parse().map_err(|_| ConfigError::BadEnv {
                name: "PADDOCK_ACTIVITY_RETENTION_DAYS",
                value: v,
            })?;
        }
        Ok(())
    }
}

/// The manager's whole `PADDOCK_*` environment surface: what `merge_env` reads
/// above, plus what is read elsewhere in the crate. A hardened build seals
/// everything else away in `main` before the first read.
///
/// Sealing here is mostly about what the manager passes on: spawned runners
/// inherit this environment, so a kernel switch exported in the shell that
/// started `paddock` would otherwise reach the engine through a door the
/// runner's own seal never sees.
///
/// Add a name here in the same edit that adds it to `merge_env` - the test
/// below reads this file and fails if you do not.
pub const ENV_SURFACE: &[&str] = &[
    "PADDOCK_MANAGER_HOST",
    "PADDOCK_MANAGER_PORT",
    "PADDOCK_MANAGER_API_KEY",
    "PADDOCK_MANAGER_TRUSTED_PROXY",
    "PADDOCK_MANAGER_ACTIVITY",
    "PADDOCK_MANAGER_NO_ACTIVITY",
    "PADDOCK_MANAGER_SPAWN_TIMEOUT_S",
    "PADDOCK_ACTIVITY_RETENTION_DAYS",
    "PADDOCK_MODEL_DIRS",
    "PADDOCK_DEVICE",
    "PADDOCK_KERNEL_PACK",
    // operator override for which runner binary to supervise (dev workflows)
    "PADDOCK_RUNNER_BIN",
    // where the CLI verbs find the running manager
    "PADDOCK_MANAGER_URL",
    // update/catalog origin, for a mirror or a staging endpoint
    "PADDOCK_API_BASE",
    // the box data root (paddock_admin::data_root_resolved)
    "PADDOCK_DATA",
    // let a model be started that the estimator says will not fit
    "PADDOCK_ALLOW_VRAM_OVERCOMMIT",
];

#[cfg(test)]
mod hardening_tests {
    use super::ENV_SURFACE;

    /// A name added to `merge_env` and not to `ENV_SURFACE` is a setting that
    /// works on every dev box and is deleted before it is read in the shipped
    /// binary. Read the file rather than trust the convention.
    #[test]
    fn every_env_merge_env_reads_is_declared_in_the_surface() {
        let src = include_str!("config.rs");
        let mut missing = Vec::new();
        for (i, _) in src.match_indices("get(\"PADDOCK_") {
            let rest = &src[i + "get(\"".len()..];
            let name = &rest[..rest.find('"').expect("closing quote")];
            if !ENV_SURFACE.contains(&name) {
                missing.push(name);
            }
        }
        missing.sort_unstable();
        missing.dedup();
        assert!(
            missing.is_empty(),
            "merge_env reads {missing:?} but ENV_SURFACE does not list them - a \
             hardened build would seal them away before merge_env ever runs"
        );
    }
}
