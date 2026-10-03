//! Runner supervision (doc §3, §5, §6.1, §11.3): spawn, health-gate, stop,
//! same-port takeover, and startup reconciliation. The manager is the OS's
//! child-minder for runners - but runners deliberately do not die with it (no
//! kill-on-job-close): the data plane survives a control-plane crash, and a
//! restarted manager re-attaches over the admin pipe.
//!
//! Spawn mechanics (§11.3): everything a runner needs travels as explicit
//! flags from the manager's election - the runner has no registry and no
//! notion of "current version". stdout/stderr go to a file handle the manager
//! opens and passes in (an inherited handle survives parent death on every
//! OS; a pipe would EPIPE the runner if the manager crashed).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use paddock_admin::client::AdminClient;
use serde::{Deserialize, Serialize};

/// How a record entered the table (doc §6.1). Adopted runners get read-only
/// visibility + frozen core ops; the manager never force-kills them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    /// Ours: spawned by this manager (child handle held), or attached via its
    /// `servers/<port>.toml` - the file gives the full spec back, so the
    /// endpoint is fully editable/persistable. Force-kill escalation still
    /// requires the child handle.
    Own,
    /// Found via admin-endpoint enumeration with no config file of ours -
    /// its config is not ours to report, and it is never force-killed.
    Adopted,
}

struct Record {
    origin: Origin,
    /// Held only for Own runners - the handle we wait on after shutdown.
    child: Option<std::process::Child>,
    model: Option<String>,
    /// Speculation mechanism label from the spawn's resolution ("MTP",
    /// "DFlash1", "off"); None for adopted runners and non-speculative models.
    spec_desc: Option<String>,
    pid: u32,
    /// §10.1 policy: a pinned runner is never auto-stopped to make room - the
    /// resident embedder, the prod endpoint. Excluded from the estimator's
    /// reclaimable VRAM and from eviction candidacy; explicit `stop` still
    /// works (operator intent beats policy).
    pinned: bool,
    /// The full spec this runner was spawned with - what the edit page reads
    /// and what a persist-toggle records as an election. None for adopted
    /// runners (their config is theirs; the manager won't guess it).
    spec: Option<SpawnSpec>,
    /// The inference key issued at spawn (Own runners only) - what network
    /// callers must send (loopback is exempt runner-side) and what the
    /// manager's own API-client role (Studio chat relay, §10) authenticates
    /// with. Reported via RunnerConfig since the bind-all flip; adopted
    /// runners have None (their key, if any, is theirs).
    api_key: Option<String>,
}

/// What we know about the process behind a record. `NoHandle` is an adopted or
/// attached runner - someone else's process, or one discovered after our boot -
/// where there is nothing to wait on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChildState {
    Alive,
    Exited,
    NoHandle,
}

/// Should a record for a port that did not answer `identify` be forgotten?
///
/// Only when it can be PROVEN gone, never merely because it went quiet. Gone
/// means the admin socket is no longer enumerable AND no process of ours is
/// still alive on it. The two halves both matter:
///
/// - a socket that still enumerates is a HUNG runner, and hung is a state an
///   operator needs to see ("unreachable"), not one to quietly erase;
/// - a live child with no socket yet is a runner still BOOTING, and dropping
///   its record would strand the process handle - downgrading an Own runner to
///   Adopted and losing the ability to stop it.
///
/// Before this existed nothing ever removed a record, so a crashed or
/// kill -9'd runner left one for the life of the manager process: the endpoint
/// showed as "unreachable" with no Start button, `configured()` reported it
/// `running` so the fleet list hid it entirely, and its port stayed
/// unallocatable.
fn silent_record_is_gone(socket_enumerates: bool, child: ChildState) -> bool {
    !socket_enumerates && child != ChildState::Alive
}

/// The rule for "this port has something on it": a record of ours, or an admin
/// endpoint that enumerates.
///
/// **Invariant: `list()` emits a row for exactly the ports this returns true
/// for, and `configured()` marks exactly those `running`.** The two used to be
/// separate expressions that did not agree, and the disagreement was not a
/// cosmetic one: `/api/servers` called a port running if a record existed OR
/// its socket enumerated, while `/api/runners` only emitted a row when
/// `identify` answered or a record existed. A port that enumerated but did not
/// answer satisfied the first and produced nothing for the second, so the
/// endpoint appeared in neither list and vanished from the UI with its config
/// still on disk.
///
/// They agree by construction now: `identify` answering implies the socket
/// enumerates (identify travels over that socket), so the three emission paths
/// in `list()` - answered, recorded-but-silent, enumerated-but-silent - are
/// precisely `record || enumerates`. Anything added to one side belongs on the
/// other; that is what this function is for.
fn port_has_endpoint(records: &HashMap<u16, Record>, serving: &[u16], port: u16) -> bool {
    records.contains_key(&port) || serving.contains(&port)
}

/// A runner as the API/CLI sees it. `status` is live (queried per call):
/// "ok" | "draining" | "unreachable" - unreachable means the admin endpoint
/// exists but nothing answered (likely a corpse or a hung process).
#[derive(Debug, Clone, Serialize)]
pub struct RunnerView {
    pub port: u16,
    pub pid: u32,
    pub origin: Origin,
    pub status: String,
    pub model: Option<String>,
    /// Speculation mechanism in words ("MTP", "DFlash1", "off") - absent for
    /// non-speculative models and adopted runners. Every surface that names
    /// the model shows this beside it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec: Option<String>,
    pub embedder: Option<String>,
    /// Served speech-to-text model, when the runner is a whisper-family one
    /// (which loads no generative model at all). Reported separately rather
    /// than folded into `model` because the two are not interchangeable: a
    /// whisper runner refuses chat, so a picker that treated it as a chat
    /// model would offer a conversation it cannot have.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asr: Option<String>,
    /// Served forced-alignment model id - a fourth serving role for the same
    /// reason `asr` is: an aligner runner refuses chat, transcription and
    /// embeddings alike, so folding it into any of those keys would offer a
    /// surface it cannot answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aligner: Option<String>,
    /// Served image-generation model id - the fifth serving role, for the
    /// same reason the others are: an image runner answers `/v1/images/*`
    /// and refuses every text surface, so it must not be offered as any of
    /// them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// Served decision model id (Laya) - the sixth serving role: a reader
    /// answers `/v1/systemone` only and refuses chat, so it must not be
    /// offered as a chat model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reader: Option<String>,
    /// Served tabular predictor id (Kumo-Tabular) - the seventh serving
    /// role: `/v1/tabular/*` only, rows in and predictions out, so it must
    /// not be offered as any text surface.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tabular: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diarization: Option<String>,
    /// The catalog's human name for the served model ("Qwen 3.5 9B") + its
    /// maker - the labels every UI surface shows, with the technical id kept
    /// for tooltips. Absent when the catalog doesn't know the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
    pub version: Option<String>,
    pub uptime_s: Option<u64>,
    pub in_flight: Option<u64>,
    pub endpoint: String,
    /// §10.1: never auto-stopped to make room (eviction-exempt).
    pub pinned: bool,
    /// The as-deployed configuration (from the retained spawn spec) - what a
    /// detail/edit page shows. None for adopted runners: their config is not
    /// ours to report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<RunnerConfig>,
    /// The runner-local model lifecycle, when the runner manages one (Whisper
    /// on demand): its phase, the policy in force and the load/unload
    /// counters. A row that is "running" with no VRAM is otherwise a puzzle -
    /// this says the listener is up and the model is simply not resident.
    /// Absent for runners that keep their model loaded for life.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residency: Option<paddock_admin::residency::Snapshot>,
}

/// The editable slice of a runner's spawn spec, as the API reports it. Since
/// runners bind all interfaces the api_key is reported: network callers
/// need it, and the manager's own API is loopback-only - same trust domain as
/// the operator reading this.
#[derive(Debug, Clone, Serialize)]
pub struct RunnerConfig {
    pub residency: Option<paddock_admin::residency::Config>,
    pub model: String,
    /// Weights-artifact choice (schema 3), e.g. "q4". None = default.
    pub artifact: Option<String>,
    /// Drafter-artifact pin, e.g. "drafter2" (DFlash2) vs "drafter" (DFlash1).
    /// None = the catalog default. Only meaningful for models cataloguing more
    /// than one drafter.
    pub drafter: Option<String>,
    pub max_ctx: Option<usize>,
    pub max_batch: Option<usize>,
    /// GPU pin as the config FILE carries it: a device UUID ("GPU-...") or an
    /// ordinal string.
    pub gpu: Option<String>,
    pub kv_cache_dtype: Option<String>,
    /// Speculation policy as deployed ("off" | "auto" | "ladder" | "<k>");
    /// None = the key is absent and the runner's default applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec: Option<String>,
    pub keyed: bool,
    /// The inference key network callers must send (loopback callers are
    /// exempt runner-side). None for adopted runners.
    pub api_key: Option<String>,
    pub runner_version: Option<String>,
    /// SERVER TOOLS this endpoint supplies (per-model config; the edit page
    /// prefills from these).
    pub web_search_provider: Option<String>,
    pub web_search_api_key: Option<String>,
    pub mcp_servers: Vec<serde_json::Value>,
    /// The `[forensics]` block this endpoint serves (the edit page prefills the
    /// Intelligence toggle from it). None = absent/disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forensics: Option<ForensicsSpec>,
    /// The `[kv_offload]` block this endpoint serves (the edit page prefills
    /// the prefix-cache fields from it). None = absent/disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_offload: Option<KvOffloadSpec>,
}

impl RunnerConfig {
    fn from_spec(s: &SpawnSpec) -> Self {
        Self {
            residency: s.residency.clone(),
            model: s.model.clone(),
            artifact: s.artifact.clone(),
            drafter: s.drafter.clone(),
            max_ctx: s.max_ctx,
            max_batch: s.max_batch,
            gpu: s.gpu.clone(),
            kv_cache_dtype: s.kv_cache_dtype.clone(),
            spec: s.spec_policy.clone(),
            keyed: s.api_key.is_some(),
            api_key: s.api_key.clone(),
            runner_version: s.runner_version.clone(),
            web_search_provider: s.web_search_provider.clone(),
            web_search_api_key: s.web_search_api_key.clone(),
            mcp_servers: s.mcp_servers.clone(),
            forensics: s.forensics.clone(),
            kv_offload: s.kv_offload.clone(),
        }
    }
}

pub(crate) const CONFIG_HEADER: &str = "# Written by the paddock manager (the Start/Edit page). This file IS the\n\
     # server's configuration - edit it and restart, or run it with no manager:\n\
     #   paddock-runner --config <this file>\n\n";

/// Every root key `render_server_config` is capable of emitting - the manager's
/// half of the config file. Anything not in here reached the file some other way
/// (a hand edit, a future runner flag the editor has no control for) and must
/// survive a re-render untouched.
///
/// This is a set of OWNED keys, not "keys the render happened to emit", and the
/// distinction is the whole point: the renderer omits a key to mean *off*
/// (`mmproj` absent = serve text-only, `spec` absent = the tuned ladder). A
/// merge that only added what the render contained would leave the old `mmproj`
/// line in place and vision would silently stay on. Absent-in-render therefore
/// means DELETE, and only membership here can tell that apart from a key the
/// manager never writes.
///
/// `render_emits_only_owned_keys` holds the list to the renderer, so adding a
/// key there without adding it here fails the build rather than quietly turning
/// that key into hand-edited state the manager can no longer change.
pub const OWNED_CONFIG_KEYS: &[&str] = &[
    "residency",
    "host",
    "port",
    "model",
    "catalog",
    "mmproj",
    "vision",
    "text_encoder",
    "vae",
    "mtp",
    "fp8_native",
    "device",
    "gpu",
    "kernel_pack",
    "model_dirs",
    "max_ctx",
    "max_batch",
    "api_key",
    "kv_cache_dtype",
    "spec",
    "vram_budget",
    "web_search_provider",
    "web_search_api_key",
    "mcp_servers",
    "forensics",
    "kv_offload",
];

/// Lay a freshly rendered config over the file as it stands: owned keys come
/// from `rendered` (absent there = removed), everything else survives from
/// `current`, comments and layout included.
///
/// Why not just hand back `rendered`: the Studio's Simple tab renders through
/// this so its settings can appear in the Advanced/file tabs, and a file may
/// hold keys Simple has no control for. Replacing the text wholesale would fix
/// one silent loss by introducing another.
///
/// `toml_edit` keeps the user's comments and key order, but inserting a new
/// scalar into a document that already ends in an array-of-tables would place it
/// after that block, where TOML reads it as a member of the last table. Rather
/// than reason about item positions, the merge VERIFIES itself: if the edited
/// text does not parse back to the value the merge intended, it falls back to a
/// plain re-serialization, which loses comments but can never lose meaning. The
/// common case - every owned key already present, because the manager wrote the
/// file - is an in-place value swap that never trips the fallback.
pub fn merge_owned_keys(current: &str, rendered: &str) -> Result<String, String> {
    use toml::Value;
    let cur_val: Value =
        toml::from_str(current).map_err(|e| format!("current config is not valid TOML: {e}"))?;
    let new_val: Value =
        toml::from_str(rendered).map_err(|e| format!("rendered config is not valid TOML: {e}"))?;
    let (Some(cur_tbl), Some(new_tbl)) = (cur_val.as_table(), new_val.as_table()) else {
        return Err("a config file must be a TOML table".into());
    };
    // what the merge means, as a value - the yardstick for the check below
    let mut want = cur_tbl.clone();
    for k in OWNED_CONFIG_KEYS {
        match new_tbl.get(*k) {
            Some(v) => {
                want.insert((*k).to_string(), v.clone());
            }
            None => {
                want.remove(*k);
            }
        }
    }
    // A key the renderer emits but nobody declared: keep it (dropping it would
    // silently un-set a real setting) and let the test say so on the next build.
    for (k, v) in new_tbl {
        if !OWNED_CONFIG_KEYS.contains(&k.as_str()) {
            tracing::warn!(key = %k, "rendered config key is not in OWNED_CONFIG_KEYS - add it there");
            want.insert(k.clone(), v.clone());
        }
    }
    let mut doc: toml_edit::DocumentMut = current
        .parse()
        .map_err(|e| format!("current config is not valid TOML: {e}"))?;
    let new_doc: toml_edit::DocumentMut = rendered
        .parse()
        .map_err(|e| format!("rendered config is not valid TOML: {e}"))?;
    for k in OWNED_CONFIG_KEYS {
        match new_doc.get(k) {
            Some(item) => doc[*k] = item.clone(),
            None => {
                doc.remove(k);
            }
        }
    }
    for (k, _) in new_tbl {
        if !OWNED_CONFIG_KEYS.contains(&k.as_str())
            && let Some(item) = new_doc.get(k)
        {
            doc[k.as_str()] = item.clone();
        }
    }
    let merged = doc.to_string();
    match toml::from_str::<Value>(&merged) {
        Ok(Value::Table(t)) if t == want => Ok(merged),
        _ => {
            tracing::debug!(
                "merged config did not round-trip - re-serializing without the original comments"
            );
            let body = toml::to_string_pretty(&Value::Table(want))
                .map_err(|e| format!("could not serialize the merged config: {e}"))?;
            Ok(format!("{CONFIG_HEADER}{body}"))
        }
    }
}

/// One endpoint as its config FILE describes it - the filesystem is the
/// enumeration, so this covers stopped endpoints too, which is the whole point:
/// a stopped model still has to say what it is.
pub struct ConfiguredEndpoint {
    pub port: u16,
    /// Catalog model id when the file can name one (its `[catalog]` block, or
    /// a weights path the registry still recognizes), else the weights path.
    pub model: Option<String>,
    /// Weights-artifact id ("q8", "q4"), when known.
    pub artifact: Option<String>,
    /// The `model` key verbatim - the bytes the runner will actually load.
    /// Kept beside the identity so nothing that needs the FILE has to rebuild
    /// the path from the id.
    pub weights: Option<String>,
    /// Speculation mechanism this config would wire ("MTP", "DFlash1",
    /// "off") - a stopped endpoint cannot be asked, so config + catalog
    /// answer, exactly like `capability` does at the route.
    pub spec_desc: Option<String>,
    pub running: bool,
}

/// An image lane's resident companions as resolved for one spawn: the text
/// encoder and the VAE the DiT loads beside itself (`--text-encoder`,
/// `--vae`). Carried as a pair because they are set and cleared together -
/// a config with one and not the other cannot start - and empty for every
/// other kind of model.
#[derive(Debug, Clone, Default)]
pub struct LanePaths {
    pub text_encoder: Option<PathBuf>,
    pub vae: Option<PathBuf>,
}

/// What a model name resolves to on disk for one spawn. Was a 4-tuple; the
/// drafter identity made it worth naming, because "which drafter did On
/// actually wire" is a question the UI now has to answer.
pub struct Resolution {
    pub weights: PathBuf,
    pub mmproj: Option<PathBuf>,
    /// An image lane's text encoder and VAE, resolved beside its DiT.
    pub lane: LanePaths,
    pub mtp: Option<PathBuf>,
    pub fp8: Option<PathBuf>,
    /// `(artifact id, label)` of the drafter actually wired, when one was.
    pub drafter: Option<(String, String)>,
    /// The speculation mechanism in words, for every surface that shows the
    /// model's name: "MTP" (in-file heads), a drafter's label ("DFlash1"),
    /// "off" (a model that could speculate, switched off), or None for a
    /// model with nothing to say. "Spec: on" answering which mechanism is
    /// what stops the user guessing.
    pub spec_desc: Option<String>,
}

impl Resolution {
    /// A bare weights path (hand-typed file, non-catalog model): nothing else
    /// is known, and nothing else is implied.
    fn weights_only(weights: PathBuf) -> Self {
        Self {
            weights,
            mmproj: None,
            lane: LanePaths::default(),
            mtp: None,
            fp8: None,
            drafter: None,
            spec_desc: None,
        }
    }
}

/// A config buffer as the Start/Edit page's Simple tab reads it - the answer to
/// `POST /api/servers/project`. Serialized straight to the browser, which then
/// binds these to its controls and derives nothing.
#[derive(Serialize)]
pub struct ConfigProjection {
    pub residency: Option<paddock_admin::residency::Config>,
    pub residency_supported: bool,
    /// Catalog model id when the text can name one (its `[catalog]` block, or a
    /// weights path the registry still recognizes), else the weights path -
    /// same rule, same code, as every other reader.
    pub model: String,
    pub artifact: Option<String>,
    /// Drafter-artifact pin, when this endpoint has one. Absent = the catalog
    /// default, which is what should track the catalog if the default moves.
    pub drafter: Option<String>,
    /// The `model` key verbatim: the bytes this endpoint would load.
    pub weights: Option<String>,
    /// The text carries an `mmproj`, i.e. this endpoint serves images.
    pub vision: bool,
    /// It carries an `fp8_native` snapshot dir.
    pub fp8_native: bool,
    pub max_ctx: Option<usize>,
    pub max_batch: Option<usize>,
    pub gpu: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub spec: Option<String>,
    pub api_key: Option<String>,
    pub vram_budget: Option<u64>,
    pub web_search_provider: Option<String>,
    pub web_search_api_key: Option<String>,
    pub mcp_servers: Vec<serde_json::Value>,
    /// Forensics (`[forensics]` block): the endpoint's context-
    /// enrichment gate. None = the block is absent (disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forensics: Option<ForensicsSpec>,
    /// Prefix-cache offload (`[kv_offload]`). None = the block is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_offload: Option<KvOffloadSpec>,
}

/// The `[forensics]` config table, as the Simple tab reads and writes it. The
/// Studio's Intelligence section flips `enabled`; `auto`/`tool`/`device` carry
/// through untouched from whatever the file (or the Advanced tab) set, so a
/// first-class toggle never clobbers a hand-tuned scope. A single owned key -
/// see `OWNED_CONFIG_KEYS` and `render_server_config`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ForensicsSpec {
    /// Forensics available on this endpoint at all.
    #[serde(default)]
    pub enabled: bool,
    /// Always-on scope: "off" | "images" | "all". None -> the product default
    /// ("all") when rendered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto: Option<String>,
    /// Expose the on-demand forensics tool. None -> true when rendered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<bool>,
    /// Pin the forensic GPU context to a device other than the model's. None =
    /// share the model's GPU (0 extra VRAM - the "Shared between models" case).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<usize>,
}

/// The `[kv_offload]` config table: budgets for keeping prefixes outside GPU
/// memory. Budgets only, by the no-customer-side-tuning rule - everything
/// about how the cache behaves is elected and measured in the engine, and the
/// two numbers here are the only decisions that are the operator's (how much
/// of their RAM, how much of their disk).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct KvOffloadSpec {
    #[serde(default)]
    pub enabled: bool,
    /// Host RAM the cache may hold, GiB. The entry point: the disk tier
    /// stores through RAM, so this is what arms either tier.
    #[serde(default)]
    pub ram_gb: f64,
    /// Disk the cache may use, GiB. Needs `nvme_path` to mean anything.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub nvme_gb: f64,
    /// Where on disk. Needs `nvme_gb` to mean anything.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nvme_path: Option<String>,
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// Everything a spawn request may carry. Model is a catalog id, an installed
/// model name, or a GGUF path - resolved in that order.
///
/// `Serialize` so an existing endpoint's spec can be handed BACK to a caller
/// that wants to change one thing and resend the rest. The switch route reads
/// an absent owned key as "cleared", which is right for the Studio (its form
/// surfaces those fields) and wrong for any client that sends a partial spec -
/// see the CLI's `switch`, which used to strip an endpoint's connectors and KV
/// dtype for not mentioning them.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpawnSpec {
    #[serde(default)]
    pub residency: Option<paddock_admin::residency::Config>,
    pub model: String,
    /// Weights-artifact selection for a catalog model (schema 3), e.g. "q4".
    /// None = the default choice, preferring an installed one.
    pub artifact: Option<String>,
    /// Drafter-artifact pin for a model cataloguing more than one, e.g.
    /// "drafter2" (DFlash2) vs "drafter" (DFlash1). None = the catalog
    /// default. Pinning one is the same consent marking it default expresses,
    /// so an explicitly pinned drafter wires even when it is not the default.
    #[serde(default)]
    pub drafter: Option<String>,
    /// Download missing pieces before serving. Default false - the deploy
    /// contract: acquiring is the Models page's job; only the CLI's
    /// `paddock serve` passes true (a terminal download is visible).
    #[serde(default)]
    pub pull: bool,
    /// Opt into native-FP8 plane ingestion (needs an installed fp8-snapshot
    /// artifact; the resolved snapshot dir is written into the config file's
    /// `fp8_native` field like every other setting).
    #[serde(default)]
    pub fp8_native: bool,
    /// Attach the model's vision tower (mmproj). None/true = attach when
    /// installed (vision is a default companion); Some(false) = serve
    /// text-only deliberately - the Studio's Vision switch,
    /// saving the tower's VRAM.
    #[serde(default)]
    pub vision: Option<bool>,
    /// Native desktop endpoints are local-only by default. None preserves
    /// the web/CLI LAN-serving default; a saved bind survives round-trips.
    #[serde(default)]
    pub host: Option<std::net::IpAddr>,
    pub port: Option<u16>,
    pub max_ctx: Option<usize>,
    pub max_batch: Option<usize>,
    pub api_key: Option<String>,
    /// Pin a runner artifact version (`runners/<version>/`, doc §11.2) for
    /// this endpoint. None = newest installed. A pin that is not installed is
    /// a hard error - the manager never silently substitutes a version.
    pub runner_version: Option<String>,
    /// Pin this runner to one GPU. Callers send an NVML index (as shown by
    /// /api/gpu) or a device UUID string; the manager resolves an index to
    /// the UUID and writes it into the config file's `gpu` field - the
    /// runner resolves that natively against the CUDA driver. No
    /// CUDA_VISIBLE_DEVICES anywhere. None = device 0.
    #[serde(default, deserialize_with = "de_gpu")]
    pub gpu: Option<String>,
    /// Record this spawn as an election in managed.toml (respawned on manager
    /// boot). Default true - what you started stays started, the service
    /// posture (§11.4). Bench/ephemeral spawns pass false.
    #[serde(default = "default_persist")]
    pub persist: bool,
    /// §10.1 policy: pin this runner - never auto-stopped to make room, VRAM
    /// excluded from the estimator's reclaimable figure. Persists with the
    /// election; togglable later via the pin endpoint.
    #[serde(default)]
    pub pinned: bool,
    /// SERVER TOOLS (per-model config): the web-search
    /// integration this endpoint supplies for bare `{type:"web_search"}`
    /// calls - "exa" | "tavily" | "firecrawl" | "brave" | "perplexity".
    /// Hosted-API ergonomics: callers declare the
    /// tool, the endpoint owns the integration.
    #[serde(default)]
    pub web_search_provider: Option<String>,
    #[serde(default)]
    pub web_search_api_key: Option<String>,
    /// Endpoint-attached MCP servers callers may name by bare label:
    /// [{server_label, server_url, headers?, require_approval?}]. The short
    /// list of tools this model personally offers - callers' own inline
    /// servers always work regardless (per-request, the spec's norm).
    #[serde(default)]
    pub mcp_servers: Vec<serde_json::Value>,
    /// KV cache dtype ("f16" | "fp8_e4m3"); None = the runner's "auto"
    /// per-family default. A real config-FILE field - it used
    /// to ride env, which kept it out of the file and its preview.
    #[serde(default)]
    pub kv_cache_dtype: Option<String>,
    /// Speculation policy, written to the file's `spec` key: "off" | "auto" |
    /// "ladder" | a pinned draft length. None leaves the key out.
    ///
    /// The manager's job here is the MECHANISM, not the arithmetic: a model
    /// carries MTP in-file (qwen3.5/3.6 `nextn`) or needs a drafter sideloaded
    /// (`mtp`), and only the manager knows where that file is. Asking to
    /// speculate therefore also resolves the drafter - or fails saying it is
    /// not downloaded, rather than serving without it.
    #[serde(default, rename = "spec")]
    pub spec_policy: Option<String>,
    /// Hard VRAM budget in MiB, written into the config file's `vram_budget`.
    /// Admission computes it (device residual clamped to the estimator's need
    /// at this spec's envelope) when the caller leaves it None; an explicit
    /// value is the operator's own grant and passes through untouched. The
    /// runner's engine sizes every pool inside it - the invariant that makes
    /// fleet overcommit impossible.
    #[serde(default)]
    pub vram_budget: Option<u64>,
    /// Caller-approved eviction plan: drain-stop these UNPINNED endpoints
    /// before admission. This is the 507's `eviction.plan` round-tripped
    /// through the user's explicit yes ("Stop Qwen 3.5 9B and start?") -
    /// never inferred, never silent. Transport-only: not rendered into the
    /// config file.
    #[serde(default)]
    pub evict: Vec<u16>,
    /// Forensics (`[forensics]`): the endpoint's context-enrichment
    /// gate. None = leave the block out (disabled). A first-class owned key -
    /// the Studio's Intelligence section writes it; render/project round-trip it.
    #[serde(default)]
    pub forensics: Option<ForensicsSpec>,
    /// Prefix-cache offload (`[kv_offload]`): how much RAM and disk this
    /// endpoint may keep prefixes in. None = leave the block out. Budgets
    /// only - everything else about the cache is elected in the engine.
    #[serde(default)]
    pub kv_offload: Option<KvOffloadSpec>,
}

fn default_persist() -> bool {
    true
}

/// `gpu` arrives as a JSON number (the NVML index the CLI's `--gpu` and the
/// UI send) or a string (an ordinal or "GPU-<uuid>") - both normalize to the
/// string form the config file carries.
fn de_gpu<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(match v {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Number(n)) => Some(n.to_string()),
        Some(serde_json::Value::String(s)) => {
            let s = s.trim();
            (!s.is_empty()).then(|| s.to_string())
        }
        Some(other) => {
            return Err(serde::de::Error::custom(format!(
                "gpu: expected a device index or UUID string, got {other}"
            )));
        }
    })
}

impl Default for SpawnSpec {
    fn default() -> Self {
        Self {
            residency: None,
            model: String::new(),
            artifact: None,
            drafter: None,
            pull: false,
            fp8_native: false,
            vision: None,
            host: None,
            port: None,
            max_ctx: None,
            max_batch: None,
            api_key: None,
            runner_version: None,
            gpu: None,
            persist: true,
            pinned: false,
            web_search_provider: None,
            web_search_api_key: None,
            mcp_servers: Vec::new(),
            kv_cache_dtype: None,
            spec_policy: None,
            vram_budget: None,
            evict: Vec::new(),
            forensics: None,
            kv_offload: None,
        }
    }
}

impl SpawnSpec {
    pub(crate) fn on_demand(&self) -> bool {
        self.residency.as_ref().is_some_and(|p| {
            p.load == paddock_admin::residency::LoadPolicy::OnDemand && p.validate().is_ok()
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    #[error("model {0:?} is not a catalog id, an installed model, or a file on disk")]
    ModelNotFound(String),
    /// The requested COMBINATION is invalid whatever is on disk - asking a
    /// model to speculate when this engine has no speculative path for it.
    /// Distinct from ModelNotFound deliberately: preview falls back to planned
    /// paths for a model that merely is not downloaded yet, and a policy
    /// refusal must not ride that fallback into a config that cannot start.
    #[error("{0}")]
    Unsupported(String),
    #[error("model pull failed: {0}")]
    Pull(String),
    #[error("no free runner port from {0} upward")]
    NoPort(u16),
    /// The second field says what holds the port - a pid, or the admin endpoint
    /// path. Without it "already serving" is unactionable exactly when it is
    /// wrong: a stale socket refuses every operation and names nothing.
    #[error(
        "port {0} is already serving - {1} (stop it or pick another; takeover is the switch endpoint)"
    )]
    PortTaken(u16, String),
    #[error("port {0} has no config file at {1} - `paddock serve <model> --port {0}` creates one")]
    NotConfigured(u16, String),
    #[error(
        "port {0} already has a configured endpoint (servers/{0}.toml) - `paddock start {0}` starts it, Edit changes it, or pick another port"
    )]
    AlreadyConfigured(u16),
    #[error(
        "the config file changed on disk since this edit was opened - reload and re-apply your changes (nothing was overwritten)"
    )]
    ConfigDrift,
    #[error("runner binary not found: {0}")]
    NoBinary(String),
    #[error("spawn failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("{}", died_on_startup_text(code, log_tail))]
    DiedOnStartup { code: Option<i32>, log_tail: String },
    #[error("runner did not become healthy within {0:?} - log tail:\n{1}")]
    HealthTimeout(Duration, String),
    #[error(
        "no GPU kernel pack is installed, so this machine cannot serve on CUDA yet - \
         put a pack (pd-cuda-*.dll) in packs\\cuda beside the paddock binary or under \
         the data dir, or set PADDOCK_KERNEL_PACK"
    )]
    NoKernelPack,
}

impl SpawnError {
    /// Stable, credential-free category for native UI diagnostics. The complete
    /// privileged message/log tail stays in the manager response, never in Swift.
    pub fn reason(&self) -> &'static str {
        match self {
            Self::Unsupported(_) => "unsupported_configuration",
            Self::ModelNotFound(_) => "model_missing",
            Self::Pull(_) => "download_failed",
            Self::NoPort(_) | Self::PortTaken(..) => "port_unavailable",
            Self::AlreadyConfigured(_) => "already_configured",
            Self::NoBinary(_) => "runner_missing",
            Self::Io(_) => "launch_io",
            Self::DiedOnStartup { .. } => "startup_exit",
            Self::HealthTimeout(..) => "startup_timeout",
            _ => "configuration_unavailable",
        }
    }
}

/// The runner's last log line is the actual reason ("device \"cuda\" needs a
/// kernel_pack path in config") - lead with it; the full tail follows for the
/// detail page. Never show Debug formatting (`Some(1)`) or ANSI color codes
/// to a person: the old text put the reason after a newline, and the fleet
/// row's single-line cell showed "exit Some(1) - log tail:" with nothing else.
/// Strip a tracing line's machinery so the SENTENCE is what a person meets.
///
/// A runner log line arrives as
///   `2026-08-17T16:36:22.840396Z ERROR paddock_runner::startup: server error
///    error=engine startup: qwen35 cannot serve max_ctx 131072 ...`
/// and the first ~90 characters of that are a timestamp, a level, a module
/// path and two layers of `error=` wrapping. Handed to a toast - which clamps
/// - the reader sees the timestamp and none of the answer. Seen in practice:
///   the whole 600-character explanation was reaching the browser correctly and
///   still read as "no error message, no nothing", because the part that fits was
///   all preamble.
///
/// Deliberately conservative: every step is optional, so a line that does not
/// look like this (a panic, a linker message, a bare string) passes through
/// untouched rather than being mangled by a guess.
fn human_line(line: &str) -> String {
    let mut s = line.trim();
    // ISO-8601 stamp, then the level word.
    if let Some((first, rest)) = s.split_once(char::is_whitespace)
        && first.len() >= 20
        && first.starts_with(|c: char| c.is_ascii_digit())
        && first.ends_with('Z')
    {
        s = rest.trim_start();
    }
    for lvl in ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"] {
        if let Some(rest) = s.strip_prefix(lvl) {
            s = rest.trim_start();
            break;
        }
    }
    // `some::module::path: ` - a target, only when it really looks like one.
    if let Some((head, rest)) = s.split_once(": ")
        && head.contains("::")
        && !head.contains(' ')
    {
        s = rest.trim_start();
    }
    // The runner's own wrappers: `server error error=` then `engine startup:`.
    // Both name the layer that caught it, not what went wrong.
    for cut in ["server error error=", "error="] {
        if let Some(i) = s.find(cut) {
            s = s[i + cut.len()..].trim_start();
            break;
        }
    }
    if let Some(rest) = s.strip_prefix("engine startup:") {
        s = rest.trim_start();
    }
    s.to_owned()
}

fn died_on_startup_text(code: &Option<i32>, tail: &str) -> String {
    let clean = strip_ansi(tail);
    let last = clean
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(human_line)
        .unwrap_or_default();
    let code_s = code
        .map(|c| format!(" (exit code {c})"))
        .unwrap_or_default();
    if last.is_empty() {
        format!("the model server exited during startup{code_s} and left no log")
    } else {
        // The reason leads. The exit code is bookkeeping and goes after it; the
        // full tail stays for the detail view, below a blank line so a UI that
        // shows only the first paragraph still shows the whole answer.
        format!("{last}\n\n(the model server exited during startup{code_s})\n\nlog tail:\n{clean}")
    }
}

/// Tiny ESC-sequence skipper - the runner logs colored output, and a color
/// code inside an error message is noise squared.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for d in chars.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Spawn-time defaults from the manager's config.
pub struct SpawnDefaults {
    /// Operator override for the runner binary (PADDOCK_RUNNER_BIN). Beats
    /// the artifact scan; loses only to an explicit per-spawn version pin.
    pub runner_bin: Option<PathBuf>,
    /// Side-by-side runner artifacts: `<runners_dir>/<version>/paddock-runner`
    /// (doc §11.2). Deleting a version dir is the garbage collection.
    pub runners_dir: PathBuf,
    pub device: String,
    pub kernel_pack: Option<PathBuf>,
    pub models_dirs: Vec<PathBuf>,
    /// Where runner-<port>.log files land.
    pub logs_dir: PathBuf,
    /// Working directory for spawned runners - deliberately not the manager's
    /// cwd, so a repo-local paddock.toml can never leak into an election.
    pub work_dir: PathBuf,
    pub base_port: u16,
    pub health_timeout: Duration,
}

pub struct Supervisor {
    records: tokio::sync::Mutex<HashMap<u16, Record>>,
    port_allocation: tokio::sync::Mutex<()>,
    defaults: SpawnDefaults,
    registry: Arc<crate::registry::Registry>,
    /// Desired-state mirror (managed.toml). None only in unit tests.
    elections: Option<Arc<crate::elections::Elections>>,
    /// NVML snapshot source, for resolving a GPU pin's index -> UUID at spawn.
    /// None in unit tests / NVML-less boxes (pins then fall back to index).
    gpu: Option<crate::telemetry::Telemetry>,
    /// Ports whose spawn is in FLIGHT: the record lands only after the
    /// health gate and the admin pipe only after the model loads, so a
    /// runner mid-load (a minute for a 30B) is invisible to `list()` - and
    /// a second start admitted during that load would re-sell the same VRAM.
    /// Admission counts these ports' configured budgets (std Mutex: held
    /// only for map ops, never across await).
    spawning: std::sync::Mutex<std::collections::HashSet<u16>>,
}

/// RAII marker for an in-flight spawn - the port leaves the set on every
/// exit path (success, died-on-startup, health timeout, panic).
struct SpawningGuard<'a> {
    sup: &'a Supervisor,
    port: u16,
}

impl Drop for SpawningGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut s) = self.sup.spawning.lock() {
            s.remove(&self.port);
        }
    }
}

impl Supervisor {
    pub fn new(
        defaults: SpawnDefaults,
        registry: Arc<crate::registry::Registry>,
        elections: Option<Arc<crate::elections::Elections>>,
        gpu: Option<crate::telemetry::Telemetry>,
    ) -> Self {
        Self {
            records: tokio::sync::Mutex::new(HashMap::new()),
            port_allocation: tokio::sync::Mutex::new(()),
            defaults,
            registry,
            elections,
            gpu,
            spawning: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Ports whose spawn is in flight (config written, process loading,
    /// record not yet landed) - admission counts their configured budgets.
    pub fn spawning_ports(&self) -> Vec<u16> {
        self.spawning
            .lock()
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Is anything on this port right now - recorded by us or answering on
    /// the admin surface? The boot-respawn pass uses this to skip elections
    /// that reconcile() already adopted.
    pub async fn is_serving(&self, port: u16) -> bool {
        self.port_blocker(port).await.is_some()
    }

    /// What is holding this port, phrased for the person who has to clear it -
    /// or None if it is free. The one predicate for "taken": it was written out
    /// four times (both spawn paths, `is_serving`, `remove_config`), and every
    /// copy answered with a bare bool, so each refusal could say that the port
    /// was busy and none could say what was on it. That is the whole of the
    /// DGX Spark dead end: a leftover admin socket held a port, three different
    /// messages refused three different operations, and not one of them named
    /// the file - which was also the only thing that would have fixed it.
    ///
    /// Order matters. Our own record is checked first because it carries a pid,
    /// which is more actionable than a path; the socket answer is for anything
    /// this manager did not start, or did not survive.
    async fn port_blocker(&self, port: u16) -> Option<String> {
        if let Some(pid) = self.records.lock().await.get(&port).map(|r| r.pid) {
            return Some(format!("this manager has a runner on it, pid {pid}"));
        }
        paddock_admin::enumerate().contains(&port).then(|| {
            format!(
                "something is listening on {}",
                paddock_admin::endpoint_display(port)
            )
        })
    }

    pub fn log_path(&self, port: u16) -> PathBuf {
        self.defaults.logs_dir.join(format!("runner-{port}.log"))
    }

    /// The inference key the manager issued to this runner at spawn, for its
    /// own API-client calls (§10). None for adopted/keyless runners.
    pub async fn runner_key(&self, port: u16) -> Option<String> {
        self.records
            .lock()
            .await
            .get(&port)
            .and_then(|r| r.api_key.clone())
    }

    /// Explicit native credential export names a process, not just a reusable
    /// port. Read identity and credential under the same record lock.
    pub async fn runner_key_checked(&self, port: u16, pid: u32) -> Result<Option<String>, String> {
        let records = self.records.lock().await;
        let runner = records
            .get(&port)
            .filter(|runner| runner.pid == pid)
            .ok_or("The selected instance changed or stopped.")?;
        Ok(runner.api_key.clone())
    }

    /// Toggle §10.1 pinning on a recorded runner. Updates the live record and
    /// the election (when one exists), so the pin survives a manager restart.
    pub async fn set_pinned(&self, port: u16, pinned: bool) -> Result<(), String> {
        {
            let mut recs = self.records.lock().await;
            let Some(rec) = recs.get_mut(&port) else {
                return Err(format!("no runner recorded on port {port}"));
            };
            rec.pinned = pinned;
        }
        if let Some(el) = &self.elections {
            // No election (bench spawn / adopted runner) is fine - the pin
            // then lives only as long as the runner does.
            el.set_pinned(port, pinned);
        }
        tracing::info!(port, pinned, "runner pin updated");
        Ok(())
    }

    /// Toggle start-on-boot for a running runner: on records an election from
    /// the retained spawn spec, off removes it. An adopted runner has no spec
    /// to record - the honest answer is "redeploy it through the manager".
    pub async fn set_persist(&self, port: u16, persist: bool) -> Result<(), String> {
        let Some(el) = &self.elections else {
            return Err("election persistence is not available".into());
        };
        if !persist {
            el.remove(port);
            tracing::info!(port, "election removed - will not respawn on boot");
            return Ok(());
        }
        let recs = self.records.lock().await;
        let Some(rec) = recs.get(&port) else {
            return Err(format!("no runner recorded on port {port}"));
        };
        let Some(spec) = &rec.spec else {
            return Err(format!(
                "runner on port {port} was not spawned by this manager (adopted) - its full config is unknown; redeploy it through the manager to persist it"
            ));
        };
        el.record(crate::elections::Election {
            model: spec.model.clone(),
            artifact: spec.artifact.clone(),
            port,
            config: self.server_config_path(port),
            runner_version: spec.runner_version.clone(),
            pinned: rec.pinned,
        });
        tracing::info!(port, "election recorded - respawns on manager boot");
        Ok(())
    }

    /// The deterministic home of an endpoint's config file.
    pub fn server_config_path(&self, port: u16) -> PathBuf {
        self.defaults
            .work_dir
            .join("servers")
            .join(format!("{port}.toml"))
    }

    /// The directory holding every endpoint's config file (the connector
    /// materializer walks it).
    pub fn servers_dir(&self) -> PathBuf {
        self.defaults.work_dir.join("servers")
    }

    /// The `vram_budget` (MiB) an endpoint's config file grants, if any -
    /// what admission sums across the running fleet ("Σ configured budgets",
    /// deterministic, instead of racing live ledgers).
    pub fn config_vram_budget(&self, port: u16) -> Option<u64> {
        let raw = std::fs::read_to_string(self.server_config_path(port)).ok()?;
        let v: toml::Value = toml::from_str(&raw).ok()?;
        v.get("vram_budget")
            .and_then(toml::Value::as_integer)
            .map(|n| n.max(0) as u64)
    }

    /// Every configured endpoint's prefix-cache RAM ceiling, in GiB.
    ///
    /// Read from the FILES, not from what is running: a stopped endpoint's
    /// ceiling is still a promise the box has to be able to keep the moment
    /// someone starts it, and the fit question a form asks is "can I give
    /// this one 24 GB too".
    pub fn configured_offload_ram_gb(&self) -> Vec<f64> {
        let dir = self.defaults.work_dir.join("servers");
        let Ok(rd) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        rd.flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".toml"))
            .filter_map(|e| {
                let raw = std::fs::read_to_string(e.path()).ok()?;
                let v: toml::Value = toml::from_str(&raw).ok()?;
                let kv = v.get("kv_offload")?;
                let on = kv
                    .get("enabled")
                    .and_then(toml::Value::as_bool)
                    .unwrap_or(false);
                let gb = kv
                    .get("ram_gb")
                    .and_then(toml::Value::as_float)
                    .unwrap_or(0.0);
                (on && gb > 0.0).then_some(gb)
            })
            .collect()
    }

    /// SHA-256 (hex) of an endpoint's config file - the edit page's
    /// optimistic-concurrency token. None = no file.
    pub fn config_file_hash(&self, port: u16) -> Option<String> {
        self.read_config_file(port).ok().map(|(_, hash)| hash)
    }

    /// The raw config file + its hash - what the Advanced editor loads.
    pub fn read_config_file(&self, port: u16) -> Result<(String, String), String> {
        use sha2::Digest;
        use std::io::Read;
        let path = self.server_config_path(port);
        let mut content = String::new();
        std::fs::File::open(&path)
            .and_then(|f| f.take(2 * 1024 * 1024 + 1).read_to_string(&mut content))
            .map_err(|e| format!("cannot read config file {}: {e}", path.display()))?;
        if content.len() > 2 * 1024 * 1024 {
            return Err("Endpoint configuration exceeds 2 MiB.".into());
        }
        let hash = crate::registry::hex(&sha2::Sha256::digest(content.as_bytes()));
        Ok((content, hash))
    }

    /// The Advanced editor's Save: syntax-gate, hash-guard (a file that moved
    /// since the edit opened is never clobbered), write VERBATIM, restart the
    /// endpoint from the file. Deliberately not the spec renderer - that
    /// would drop fields the manager's editor doesn't know; this path honors
    /// every knob the runner's config surface has.
    /// The config keys the RUNNER re-reads live (its LiveConfig view): a save
    /// changing only these on a running port is a plain file write - the
    /// runner serves the change on its next request, no drain, no relaunch
    /// (tools and web search are control-plane, restart-free).
    const LIVE_KEYS: &'static [&'static str] = &[
        "mcp_servers",
        "web_search_provider",
        "web_search_api_key",
        "residency",
    ];

    /// Native tools-only save. Unlike the general editor, this never spawns,
    /// drains or restarts a model, including when the endpoint is stopped.
    pub fn write_live_tools_config(
        &self,
        port: u16,
        content: &str,
        expected: &str,
    ) -> Result<(), String> {
        let (old, hash) = self.read_config_file(port)?;
        if hash != expected || !Self::only_live_keys_changed(&old, content) {
            return Err("Endpoint changed or non-tool settings were modified.".into());
        }
        crate::integrations::atomic_config(&self.server_config_path(port), content)
    }

    fn only_live_keys_changed(old: &str, new: &str) -> bool {
        let (Ok(a), Ok(b)) = (
            toml::from_str::<toml::value::Table>(old),
            toml::from_str::<toml::value::Table>(new),
        ) else {
            return false; // unparseable before/after = assume engine-binding
        };
        a.keys()
            .chain(b.keys())
            .all(|k| a.get(k) == b.get(k) || Self::LIVE_KEYS.contains(&k.as_str()))
    }

    /// Admission must not allocate/pin a new budget for a control-plane edit.
    /// Older runners and pending engine changes retain the restart path.
    pub(crate) async fn can_apply_live(&self, port: u16, content: &str) -> bool {
        let Ok((old, _)) = self.read_config_file(port) else {
            return false;
        };
        if !Self::only_live_keys_changed(&old, content) {
            return false;
        }
        let Ok(Ok(status)) = tokio::time::timeout(
            Duration::from_secs(2),
            AdminClient::new(port).config_status(),
        )
        .await
        else {
            return false;
        };
        let residency_changed = toml::from_str::<toml::Value>(&old)
            .ok()
            .and_then(|v| v.get("residency").cloned())
            != toml::from_str::<toml::Value>(content)
                .ok()
                .and_then(|v| v.get("residency").cloned());
        status.restart_required == Some(false)
            && (!residency_changed || status.residency_live == Some(true))
    }

    /// Returns the view plus whether the save applied live (true) or via the
    /// usual drain + same-port takeover (false).
    pub async fn write_config_file(
        &self,
        port: u16,
        content: &str,
        expect_hash: Option<&str>,
        drain_timeout_ms: u64,
        expected_pid: Option<u32>,
    ) -> Result<(RunnerView, bool), String> {
        // Syntax gate here; SEMANTIC errors surface from the runner's own
        // parse at start (deny_unknown_fields), with the log tail attached.
        self.validate_backend_config(content)?;
        if let Some(expect) = expect_hash {
            let now = self.config_file_hash(port);
            if now.as_deref() != Some(expect) {
                return Err(SpawnError::ConfigDrift.to_string());
            }
        }
        let path = self.server_config_path(port);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let running = self.is_serving(port).await;
        // Admission and composition can take time. Revalidate the native
        // confirmation after those awaits, before publishing the new file.
        self.verify_reviewed_runner(port, expected_pid).await?;
        let old = std::fs::read_to_string(&path).unwrap_or_default();
        if let Some(expect) = expect_hash
            && self.config_file_hash(port).as_deref() != Some(expect)
        {
            return Err(SpawnError::ConfigDrift.to_string());
        }
        crate::integrations::atomic_config(&path, content)?;
        let pending_restart = if running {
            tokio::time::timeout(
                Duration::from_secs(2),
                AdminClient::new(port).config_status(),
            )
            .await
            .ok()
            .and_then(Result::ok)
            .and_then(|s| s.restart_required)
        } else {
            Some(false)
        };
        // Native explicitly confirmed applying the saved engine geometry too;
        // a simultaneous live-policy edit must not swallow that restart.
        let apply_saved = expected_pid.is_some() && pending_restart == Some(true);
        let residency_changed = toml::from_str::<toml::Value>(&old)
            .ok()
            .and_then(|v| v.get("residency").cloned())
            != toml::from_str::<toml::Value>(content)
                .ok()
                .and_then(|v| v.get("residency").cloned());
        let residency_live = !residency_changed
            || tokio::time::timeout(Duration::from_secs(2), AdminClient::new(port).identify())
                .await
                .ok()
                .and_then(Result::ok)
                .is_some_and(|id| id.capabilities.iter().any(|c| c == "residency"));
        if running && Self::only_live_keys_changed(&old, content) && !apply_saved && residency_live
        {
            if residency_changed {
                let doc = toml::from_str(content).map_err(|_| "Invalid saved configuration.")?;
                let wanted = crate::native_endpoints::validate_residency(&doc)?;
                let client = AdminClient::new(port);
                let acknowledged = tokio::time::timeout(Duration::from_secs(5), async {
                    loop {
                        self.verify_reviewed_runner(port, expected_pid).await?;
                        if let Some(snapshot) =
                            client.residency().await.map_err(|e| e.to_string())?
                            && snapshot.policy == wanted
                        {
                            return Ok::<_, String>(());
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                })
                .await;
                if !matches!(acknowledged, Ok(Ok(()))) {
                    return Err("Settings were saved without restarting, but the runner has not acknowledged the residency policy. Review the running instance before retrying.".into());
                }
            }
            // control-plane-only edit: the running runner re-reads these on
            // its next request - bouncing the model would be pure downtime
            if let Some(v) = self.list().await.into_iter().find(|v| v.port == port) {
                return Ok((v, true));
            }
        }
        if running {
            // stop() treats every stop as a desired-state change and drops the
            // election - right for a real stop, wrong for this bounce: the
            // election carries the catalog identity, the runner-version pin,
            // and the §10.1 pin, none of which a config save may eat. Snapshot
            // it and put it back so start_config layers it as designed. (This
            // exact hole once rewrote an election's model id into a weights
            // path in managed.toml.)
            let prior = self
                .elections
                .as_ref()
                .and_then(|el| el.list().into_iter().find(|e| e.port == port));
            self.verify_reviewed_runner(port, expected_pid).await?;
            let stopped = self.stop(port, drain_timeout_ms).await;
            if let (Some(el), Some(p)) = (&self.elections, prior) {
                el.record(p);
            }
            let stopped = stopped?;
            if matches!(stopped, StopOutcome::StillRunning) {
                return Err("The configuration was saved, but the runner did not exit. Restart was not attempted.".into());
            }
        }
        self.start_config(port)
            .await
            .map(|v| (v, false))
            .map_err(|e| e.to_string())
    }

    async fn verify_reviewed_runner(&self, port: u16, expected: Option<u32>) -> Result<(), String> {
        if let Some(expected) = expected {
            let client = AdminClient::new(port);
            if !matches!(
                tokio::time::timeout(Duration::from_secs(2), client.identify()).await,
                Ok(Ok(id)) if id.pid == expected
            ) {
                return Err(
                    "The reviewed runner changed or stopped. Reload before restarting.".into(),
                );
            }
        }
        Ok(())
    }

    /// Shared backend contract for previews, saved files and starts.
    pub fn validate_backend_config(&self, content: &str) -> Result<(), String> {
        let doc = toml::from_str(content).map_err(|e| format!("not valid TOML: {e}"))?;
        crate::native_endpoints::validate_residency(&doc)?;
        crate::backend_contract::validate(&self.registry, &self.defaults.device, &doc)
    }

    pub fn kv_offload_supported(&self, doc: &toml::Value) -> bool {
        let path = Path::new(doc.get("model").and_then(toml::Value::as_str).unwrap_or(""));
        self.registry
            .identify_weights(path)
            .and_then(|(m, a)| {
                let model = self.registry.catalog_of(&m)?;
                Some(crate::backend_contract::metal_kv_offload(
                    model,
                    model.artifact(&a)?,
                ))
            })
            .unwrap_or_else(|| crate::backend_contract::path_kv_offload(path))
    }

    /// Save without applying: syntax/backend checks and a hash guard, then
    /// the write and nothing else. A stopped instance stays stopped.
    pub fn write_config_file_deferred(
        &self,
        port: u16,
        content: &str,
        expect_hash: Option<&str>,
    ) -> Result<(), String> {
        self.validate_backend_config(content)?;
        if let Some(expect) = expect_hash
            && self.config_file_hash(port).as_deref() != Some(expect)
        {
            return Err(SpawnError::ConfigDrift.to_string());
        }
        let path = self.server_config_path(port);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        crate::integrations::atomic_config(&path, content)?;
        tracing::info!(port, config = %path.display(), "configuration saved without applying");
        Ok(())
    }

    /// Publish a fully reviewed new configuration without clobbering a saved
    /// endpoint. Allocation and publication share the spawn allocation gate;
    /// persist_noclobber also protects against other manager processes.
    pub async fn create_config_file(
        &self,
        requested: Option<u16>,
        content: &str,
    ) -> Result<u16, String> {
        use std::io::Write;
        self.validate_backend_config(content)?;
        let _allocation = self.port_allocation.lock().await;
        let port = match requested {
            Some(port) if port >= 1024 => {
                if self.server_config_path(port).exists()
                    || self.records.lock().await.contains_key(&port)
                    || self
                        .spawning
                        .lock()
                        .is_ok_and(|ports| ports.contains(&port))
                    || AdminClient::new(port).is_present().await
                    || std::net::TcpListener::bind(("127.0.0.1", port)).is_err()
                {
                    return Err(
                        "This port is already in use. Choose Automatic or another port.".into(),
                    );
                }
                port
            }
            Some(_) => return Err("Use a port between 1024 and 65535.".into()),
            None => self
                .allocate_port()
                .await
                .map_err(|_| "No free model port is available.")?,
        };
        let mut doc: toml::Value =
            toml::from_str(content).map_err(|_| "Invalid model settings.")?;
        let table = doc.as_table_mut().ok_or("Invalid model settings.")?;
        if table.get("port").and_then(toml::Value::as_integer) != Some(0) {
            return Err("Only an unsaved configuration can be created.".into());
        }
        table.insert("port".into(), toml::Value::Integer(port.into()));
        let text = toml::to_string(&doc).map_err(|_| "Cannot prepare model settings.")?;
        let path = self.server_config_path(port);
        let parent = path.parent().ok_or("Invalid model settings location.")?;
        std::fs::create_dir_all(parent).map_err(|_| "Cannot create the model settings folder.")?;
        let mut file = tempfile::NamedTempFile::new_in(parent)
            .map_err(|_| "Cannot prepare model settings.")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.as_file()
                .set_permissions(std::fs::Permissions::from_mode(0o600))
                .map_err(|_| "Cannot protect model settings.")?;
        }
        file.write_all(text.as_bytes())
            .and_then(|_| file.as_file().sync_all())
            .map_err(|_| "Cannot write model settings.")?;
        file.persist_noclobber(&path).map_err(
            |_| "This endpoint already exists or could not be saved. Nothing was overwritten.",
        )?;
        Ok(port)
    }

    /// Save without applying: render `spec` into the endpoint's config FILE and
    /// leave the process exactly as it is - a running model keeps serving what
    /// it loaded, a stopped one stays stopped.
    ///
    /// The edit page's third answer. Writing a configuration and
    /// interrupting service are two different acts, and only the second one
    /// needs consent; before this, every save path restarted (`switch` by
    /// definition, `write_config_file` by falling through to `start_config`),
    /// so "save this and let it take effect next time" could not be expressed
    /// at all - and a stopped endpoint got STARTED by being edited.
    ///
    /// No VRAM admission here, deliberately: nothing is being loaded, and
    /// refusing to SAVE because the new envelope would not fit right now (with
    /// the incumbent still resident, holding the very memory it would hand
    /// back) would refuse the common case. The start prices it when it starts.
    pub async fn write_spec_config(
        &self,
        port: u16,
        spec: SpawnSpec,
        expect_config_hash: Option<String>,
    ) -> Result<PathBuf, String> {
        // Same optimistic-concurrency guard as every other save path: the edit
        // opened against a specific file state and must never clobber a change
        // it never showed the user.
        if let Some(expect) = &expect_config_hash
            && self.config_file_hash(port).as_deref() != Some(expect.as_str())
        {
            return Err(SpawnError::ConfigDrift.to_string());
        }
        let text = self.render_spec_config(port, spec).await?;
        let path = self.server_config_path(port);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&path, &text).map_err(|e| format!("write {}: {e}", path.display()))?;
        tracing::info!(
            port,
            config = %path.display(),
            "configuration saved without applying - the endpoint keeps running what it loaded"
        );
        Ok(path)
    }

    /// The config-file TEXT for a spec, resolved but not written - the one
    /// serializer, reachable without side effects.
    ///
    /// `render_server_config`'s own comment always promised this ("Save writes
    /// exactly this, and the Start/Edit page's preview shows exactly this") and
    /// there was no way to ask for it. Without it the Studio's Simple tab could
    /// not show its own settings as TOML, so the three edit tabs were editing
    /// two different documents and whichever one was open at Save silently won.
    ///
    /// Everything here mirrors `spawn_overwrite`'s resolution minus the spawn:
    /// same resolver, same companion refusal, same GPU-pin normalization, same
    /// writer - so a preview, a deferred save and an applied one agree
    /// byte-for-byte.
    pub async fn render_spec_config(
        &self,
        port: u16,
        mut spec: SpawnSpec,
    ) -> Result<String, String> {
        // Identity the editor does not surface must survive a save, exactly as
        // it survives a takeover (see `switch`): the API key (clients keep
        // working), the GPU pin, the fp8 planes, the runner-version pin. The
        // live record knows it while the model runs; for a STOPPED endpoint the
        // file is the only witness, so read it back.
        let incumbent = {
            let recs = self.records.lock().await;
            recs.get(&port).and_then(|r| r.spec.as_ref()).cloned()
        };
        let prior = match incumbent {
            Some(s) => Some(s),
            None => self
                .spec_from_config_file(&self.server_config_path(port))
                .ok(),
        };
        if let Some(old) = prior {
            if spec.host.is_none() {
                spec.host = old.host;
            }
            if spec.api_key.is_none() {
                spec.api_key = old.api_key.clone();
            }
            if spec.gpu.is_none() {
                spec.gpu = old.gpu.clone();
            }
            if !spec.fp8_native {
                spec.fp8_native = old.fp8_native;
            }
            if spec.runner_version.is_none() {
                spec.runner_version = old.runner_version.clone();
            }
        }
        let (weights, mmproj, lane, mtp, fp8_dir, drafter_pick) = match self
            .resolve_model(
                &spec.model,
                spec.artifact.as_deref(),
                spec.pull,
                spec.spec_policy.as_deref(),
                spec.drafter.as_deref(),
            )
            .await
        {
            Ok(r) => (r.weights, r.mmproj, r.lane, r.mtp, r.fp8, r.drafter),
            // A model that is simply not downloaded yet still renders, with the
            // paths its files will land on - same rule as `preview_config`, and
            // the reason is the same: the edit page has to be able to show and
            // stage a configuration for a model whose download has not run. A
            // POLICY refusal still refuses, because that config could never
            // start. (`Unsupported` is the policy arm; everything else is
            // "missing files", which the registry can plan for.)
            Err(e @ SpawnError::Unsupported(_)) => return Err(e.to_string()),
            Err(e) => match self
                .registry
                .planned_paths(&spec.model, spec.artifact.as_deref())
            {
                Some((w, mm, mt)) => (w, mm, self.planned_lane(&spec), mt, None, None),
                None => return Err(e.to_string()),
            },
        };
        let _ = &drafter_pick;
        let mmproj = if spec.vision == Some(false) {
            None
        } else {
            mmproj
        };
        if mmproj.is_none()
            && let Some(needed) = self.registry.required_companion(&spec.model)
        {
            return Err(match needed {
                Ok(p) => format!(
                    "{}: its required speech/vision companion is installed at {} but did not reach the composition - this is a manager bug, please report it",
                    spec.model,
                    p.display()
                ),
                Err(label) => format!(
                    "{}: this model cannot serve without its {label}, which is not downloaded - get it on the Models page",
                    spec.model
                ),
            });
        }
        // same rule for an image lane's text encoder and VAE, which have no
        // switch at all
        if let Some(label) = self
            .registry
            .missing_required_lane_companion(&spec.model, spec.artifact.as_deref())
        {
            return Err(format!(
                "{}: this model cannot serve without its {label}, which is not downloaded - get it on the Models page",
                spec.model
            ));
        }
        let fp8 = if spec.fp8_native {
            match &fp8_dir {
                Some(dir) => Some(dir.clone()),
                None => {
                    return Err(format!(
                        "fp8_native requested but no FP8 snapshot artifact of {:?} is installed - get it on the Models page",
                        spec.model
                    ));
                }
            }
        } else {
            None
        };
        spec.port = Some(port);
        spec.gpu = self.resolve_gpu(spec.gpu.as_deref(), true).await;
        self.render_server_config(
            port,
            &weights,
            &mmproj,
            &lane,
            &mtp,
            spec.gpu.as_deref(),
            fp8.as_deref(),
            &spec,
        )
        .map_err(|e| e.to_string())
    }

    /// An image lane's planned companion paths for a spec - where the text
    /// encoder and VAE will land once pulled - for the save and preview paths
    /// that render a not-yet-downloaded model. Empty for every other kind of
    /// model.
    fn planned_lane(&self, spec: &SpawnSpec) -> LanePaths {
        let (text_encoder, vae) = self
            .registry
            .planned_lane_companions(&spec.model, spec.artifact.as_deref());
        LanePaths { text_encoder, vae }
    }

    /// Reconstruct a SpawnSpec from an endpoint's config FILE alone - the file
    /// is the truth for everything the endpoint serves with, catalog identity
    /// included: `[catalog]` names the model, `model` names the
    /// bytes, and `Registry::identity_for` reconciles the two. A file without
    /// the block falls back to its weights path (a filesystem path is a valid
    /// spawn model) and an election can still layer identity on top, which is
    /// what every config written before the `[catalog]` block existed does.
    pub fn spec_from_config_file(&self, path: &Path) -> Result<SpawnSpec, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|err| format!("config file {} unreadable: {err}", path.display()))?;
        self.spec_from_config_text(&raw)
            .map_err(|err| format!("config file {}: {err}", path.display()))
    }

    /// The same projection, over config TEXT that need not be on disk yet.
    ///
    /// This is what `/api/servers/project` serves, and it exists so the Studio's
    /// Simple tab has no rule of its own. The editor works on an unsaved buffer,
    /// so it cannot read the saved file - and the previous answer, re-deriving
    /// the model identity in the browser, is what produced two separate
    /// disagreements in one day (`/api/servers` vs `heal_spec_identity`, then
    /// the browser vs `identify_weights`). A mirror kept in sync by hand is a
    /// defect with a delay fuse; one implementation, reachable over HTTP, has no
    /// second copy to drift.
    pub fn spec_from_config_text(&self, raw: &str) -> Result<SpawnSpec, String> {
        // A leading BOM is tolerated by Rust's toml and refused by the Studio's
        // parser, which cost a session. Strip it here so the one parser
        // both halves now share cannot disagree about it either.
        let raw = raw.strip_prefix('\u{feff}').unwrap_or(raw);
        let v: toml::Value =
            toml::from_str(raw).map_err(|err| format!("does not parse as TOML: {err}"))?;
        crate::native_endpoints::validate_residency(&v)?;
        let get_usize = |k: &str| {
            v.get(k)
                .and_then(toml::Value::as_integer)
                .map(|n| n as usize)
        };
        let get_str = |k: &str| v.get(k).and_then(toml::Value::as_str).map(String::from);
        let mcp_servers = v
            .get("mcp_servers")
            .and_then(|x| serde_json::to_value(x).ok())
            .and_then(|x| match x {
                serde_json::Value::Array(a) => Some(a),
                _ => None,
            })
            .unwrap_or_default();
        // The `[forensics]` block round-trips as a whole: enabled + any hand-set
        // auto/tool/device. A block that does not deserialize (a stray key) is
        // simply dropped from the projection rather than failing the whole
        // parse - the same forgiving stance the rest of this reader takes.
        let forensics = v
            .get("forensics")
            .cloned()
            .and_then(|x| x.try_into::<ForensicsSpec>().ok());
        let kv_offload = v
            .get("kv_offload")
            .cloned()
            .and_then(|x| x.try_into::<KvOffloadSpec>().ok());
        // Catalog identity: what the file DECLARES, reconciled against the
        // weights it points at. `identity_for` handles a file with no block -
        // which is every config written before that block existed - by recognizing the
        // weights, so nothing on disk needs migrating.
        let weights = get_str("model").unwrap_or_default();
        let declared = v.get("catalog").and_then(toml::Value::as_table);
        let ident = self.registry.identity_for(
            declared.and_then(|c| {
                Some((
                    c.get("model").and_then(toml::Value::as_str)?,
                    c.get("artifact").and_then(toml::Value::as_str),
                ))
            }),
            Path::new(&weights),
        );
        Ok(SpawnSpec {
            residency: v
                .get("residency")
                .cloned()
                .map(|v| v.try_into())
                .transpose()
                .map_err(|_| "Invalid model residency policy.")?,
            kv_offload,
            // identity when we have one, the weights path otherwise - a
            // filesystem path is a valid spawn model
            model: ident.as_ref().map_or(weights, |(id, _)| id.clone()),
            artifact: ident.and_then(|(_, a)| a),
            drafter: declared
                .and_then(|c| c.get("drafter"))
                .and_then(toml::Value::as_str)
                .map(str::to_owned),
            // a start re-serves what is on disk; never re-downloads a
            // deleted model behind the operator's back
            pull: false,
            fp8_native: get_str("fp8_native").is_some(),
            // the config file speaks for itself: a file with mmproj serves
            // vision, and one that says `vision = false` stays off through any
            // re-render (None would let the render resolve the tower again)
            vision: v
                .get("vision")
                .and_then(toml::Value::as_bool)
                .filter(|on| !on),
            host: get_str("host")
                .map(|host| host.parse().map_err(|_| "invalid endpoint bind address"))
                .transpose()?,
            port: v
                .get("port")
                .and_then(toml::Value::as_integer)
                .map(|n| n as u16),
            max_ctx: get_usize("max_ctx"),
            max_batch: get_usize("max_batch"),
            api_key: get_str("api_key"),
            runner_version: None,
            gpu: get_str("gpu"),
            persist: true,
            pinned: false,
            web_search_provider: get_str("web_search_provider"),
            web_search_api_key: get_str("web_search_api_key"),
            mcp_servers,
            kv_cache_dtype: get_str("kv_cache_dtype"),
            spec_policy: get_str("spec"),
            vram_budget: v
                .get("vram_budget")
                .and_then(toml::Value::as_integer)
                .map(|n| n.max(0) as u64),
            evict: Vec::new(),
            forensics,
        })
    }

    /// The Start/Edit page's Simple tab, projected from config TEXT.
    ///
    /// Everything here is READ from the buffer; nothing is invented and nothing
    /// is written. It exists so the browser holds no reconciliation rule of its
    /// own - see `spec_from_config_text` for what that cost when it did.
    pub fn project_config_text(&self, raw: &str) -> Result<ConfigProjection, String> {
        let spec = self.spec_from_config_text(raw)?;
        // `mmproj` presence is the optional vision switch, and SpawnSpec deliberately
        // carries `vision: None` for a file ("the file speaks for itself" - a
        // spawn must not re-decide it), so read it here rather than bend that.
        let trimmed = raw.strip_prefix('\u{feff}').unwrap_or(raw);
        let v: toml::Value = toml::from_str(trimmed).map_err(|e| e.to_string())?;
        let embedded_vision = self
            .registry
            .catalog_of(&spec.model)
            .and_then(|m| spec.artifact.as_deref().and_then(|id| m.artifact(id)))
            .is_some_and(|a| a.runtime.embedded_vision);
        Ok(ConfigProjection {
            residency: spec.residency,
            residency_supported: crate::native_endpoints::residency_supported(&v),
            weights: v
                .get("model")
                .and_then(toml::Value::as_str)
                .map(String::from),
            vision: (v.get("mmproj").is_some() || embedded_vision)
                && v.get("vision").and_then(toml::Value::as_bool) != Some(false),
            fp8_native: spec.fp8_native,
            model: spec.model,
            artifact: spec.artifact,
            drafter: spec.drafter,
            max_ctx: spec.max_ctx,
            max_batch: spec.max_batch,
            gpu: spec.gpu,
            kv_cache_dtype: spec.kv_cache_dtype,
            spec: spec.spec_policy,
            api_key: spec.api_key,
            vram_budget: spec.vram_budget,
            web_search_provider: spec.web_search_provider,
            web_search_api_key: spec.web_search_api_key,
            mcp_servers: spec.mcp_servers,
            forensics: spec.forensics,
            kv_offload: spec.kv_offload,
        })
    }

    /// `spec_from_config_file` + the election's launch mechanics. Reconcile-
    /// reclaims come through here. Identity comes from the FILE when it can
    /// state it - see `start_config` for why the election is only the
    /// fallback.
    pub fn spec_from_election(&self, e: &crate::elections::Election) -> Result<SpawnSpec, String> {
        let mut s = self.spec_from_config_file(&e.config)?;
        if s.model.contains('/') || s.model.contains('\\') {
            s.model = e.model.clone();
            s.artifact = e.artifact.clone();
        }
        s.port = Some(e.port);
        s.runner_version = e.runner_version.clone();
        s.pinned = e.pinned;
        Ok(s)
    }

    /// The `(model id, artifact id)` to stamp into a config file for this
    /// spawn. A spec that already names a catalog id is the declaration; one
    /// that names a path has none, and the registry falls back to recognizing
    /// the file. Either way `Registry::identity_for` owns the reconciliation,
    /// so the writer and every reader agree by construction.
    fn catalog_identity(
        &self,
        spec: &SpawnSpec,
        weights: &Path,
    ) -> Option<(String, Option<String>)> {
        let m = spec.model.as_str();
        let declared =
            (!(m.contains('/') || m.contains('\\'))).then_some((m, spec.artifact.as_deref()));
        self.registry.identity_for(declared, weights)
    }

    /// A spec whose `model` is a weights PATH gets its catalog identity back
    /// when the registry recognizes the file - so the edit page and
    /// managed.toml speak model ids, not paths. No-op for a proper catalog id
    /// or an unrecognized file. Returns whether it healed.
    ///
    /// This is the REPAIR path, not the identity path: a config file
    /// written by any current manager declares `[catalog]`, `spec_from_config_file`
    /// reads it, and this then finds a non-path-shaped model and does nothing.
    /// It still runs for files written before that block existed, which is what
    /// makes them keep working without a migration.
    fn heal_spec_identity(&self, spec: &mut SpawnSpec) -> bool {
        let m = spec.model.as_str();
        if !(m.contains('/') || m.contains('\\')) {
            return false; // not path-shaped - already an id (or a bare name)
        }
        let Some((id, art)) = self.registry.identify_weights(Path::new(m)) else {
            return false;
        };
        tracing::info!(model = %id, artifact = %art, was = %m, "restored catalog identity from the weights path");
        spec.model = id;
        spec.artifact = Some(art);
        true
    }

    /// Write the `[catalog]` block into a config file that predates it,
    /// once the identity is known.
    ///
    /// ADDITIVE only, and that is what makes it safe under "a start launches
    /// the file verbatim": it never changes a value, never removes a key, and
    /// the key it adds is inert to the runner. What it buys is permanence - an
    /// endpoint whose identity we can still recover TODAY by recognizing its
    /// filename records that fact before a rename, a copy, or a catalog edit
    /// takes the last chance away.
    ///
    /// Silent on every failure: an unwritable or unparseable file is not this
    /// function's problem, and the start it rides on has its own error path.
    /// Verifies the edit round-trips before committing it - a table appended to
    /// a document is unambiguous where a bare key would not be, but the check
    /// costs nothing and the alternative is corrupting a config on a start.
    fn stamp_catalog_identity(cfg_path: &Path, spec: &SpawnSpec) {
        let m = spec.model.as_str();
        if m.is_empty() || m.contains('/') || m.contains('\\') {
            return; // no identity to record
        }
        let Ok(raw) = std::fs::read_to_string(cfg_path) else {
            return;
        };
        let Ok(mut doc) = raw.parse::<toml_edit::DocumentMut>() else {
            return;
        };
        if doc.get("catalog").is_some() {
            return; // already declared - the file speaks for itself
        }
        let mut t = toml_edit::Table::new();
        t.insert("model", toml_edit::value(m.to_owned()));
        if let Some(a) = &spec.artifact {
            t.insert("artifact", toml_edit::value(a.clone()));
        }
        if let Some(d) = &spec.drafter {
            t.insert("drafter", toml_edit::value(d.clone()));
        }
        doc.insert("catalog", toml_edit::Item::Table(t));
        let text = doc.to_string();
        let round_trips = toml::from_str::<toml::Value>(&text).is_ok_and(|v| {
            v.get("catalog")
                .and_then(|c| c.get("model"))
                .and_then(toml::Value::as_str)
                == Some(m)
                && v.get("model").and_then(toml::Value::as_str)
                    == toml::from_str::<toml::Value>(&raw)
                        .ok()
                        .as_ref()
                        .and_then(|o| o.get("model"))
                        .and_then(toml::Value::as_str)
        });
        if !round_trips {
            tracing::debug!(config = %cfg_path.display(), "not stamping catalog identity - the edit did not round-trip");
            return;
        }
        if std::fs::write(cfg_path, text).is_ok() {
            tracing::info!(config = %cfg_path.display(), model = %m, artifact = ?spec.artifact, "recorded the catalog identity in the config file");
        }
    }

    /// Put back an mmproj line the manager itself failed to write.
    ///
    /// A start launches the config file VERBATIM, which is the rule and stays
    /// the rule - the manager does not inject settings behind an operator's
    /// back. This is the one exception, and it is narrow by construction: it
    /// fires only for a companion the catalog marks required, i.e. one the
    /// engine refuses to serve the architecture without. Serving without it is
    /// not a configuration an operator could have meant; it is a file that
    /// cannot start. An optional companion (every vision tower) is never
    /// touched, so "I removed the mmproj to get its VRAM back" keeps working.
    ///
    /// Why this exists at all: a resolve bug wrote exactly such a file for
    /// Qwen3-ASR. Fixing the resolve repairs new endpoints; the
    /// ones already on disk would have stayed broken with no way out but
    /// deleting and re-creating the endpoint.
    fn repair_required_companion(
        &self,
        cfg_path: &Path,
        spec: &SpawnSpec,
    ) -> Result<(), SpawnError> {
        let raw = match std::fs::read_to_string(cfg_path) {
            Ok(s) => s,
            Err(_) => return Ok(()), // unreadable is the caller's problem, not ours
        };
        let parsed: toml::Value = match toml::from_str(&raw) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        if parsed.get("mmproj").is_some() {
            return Ok(());
        }
        match self.registry.required_companion(&spec.model) {
            Some(Ok(path)) => {
                // TOML values are quoted literals; a Windows path is full of
                // backslashes, so it must ride as a LITERAL string, exactly
                // as render_server_config writes it.
                let line = format!("mmproj = '{}'\n", path.display());
                let patched = match raw.find("model = ") {
                    // keep the file's alphabetical key order (mmproj < model)
                    Some(at) => format!("{}{line}{}", &raw[..at], &raw[at..]),
                    None => format!("{raw}{line}"),
                };
                std::fs::write(cfg_path, patched)?;
                tracing::info!(
                    config = %cfg_path.display(),
                    mmproj = %path.display(),
                    "config was missing a REQUIRED companion the model cannot start without - restored it"
                );
                Ok(())
            }
            Some(Err(label)) => Err(SpawnError::ModelNotFound(format!(
                "{}: this model cannot serve without its {label}, which is not downloaded - get it on the Models page",
                spec.model
            ))),
            None => Ok(()),
        }
    }

    /// Cheap fleet metadata for the estimator's policy layer (§10.1):
    /// (port, model, pinned) per recorded runner - a map read, no admin I/O
    /// (unlike `list()`, which round-trips every runner's admin pipe).
    pub async fn fleet_meta(&self) -> Vec<(u16, Option<String>, bool)> {
        self.records
            .lock()
            .await
            .iter()
            .map(|(&port, r)| (port, r.model.clone(), r.pinned))
            .collect()
    }

    /// The scan roots for installed models - the estimator resolves eviction
    /// costs (weights on disk) against the same set spawns resolve against.
    pub fn models_dirs(&self) -> &[PathBuf] {
        &self.defaults.models_dirs
    }

    /// The kernel pack the manager spawns runners with - the editor's picker
    /// offers it as the known-good candidate.
    pub fn kernel_pack(&self) -> Option<&PathBuf> {
        self.defaults.kernel_pack.as_ref()
    }

    /// The directory runner logs (and the manager's own tee) land in - the
    /// log-stream endpoint (§11.3) tails files here.
    pub fn logs_dir(&self) -> &Path {
        &self.defaults.logs_dir
    }

    /// §6.1 startup reconciliation: enumerate admin endpoints, identify each,
    /// adopt what answers. Never kills, never restarts - a healthy serving
    /// process is left alone because the control plane came up.
    pub async fn reconcile(&self) {
        for port in paddock_admin::enumerate() {
            let client = AdminClient::new(port);
            match tokio::time::timeout(Duration::from_secs(2), client.identify()).await {
                Ok(Ok(id)) => {
                    // Already ours: refresh only what identify can tell us and
                    // leave the rest of the record alone. This runs on a timer
                    // now, and the wholesale insert below would set `child` to
                    // None every pass - dropping the process handle `stop()`
                    // waits on and kills with, for every runner we launched.
                    // Adoption is for ports we do not already know.
                    {
                        let mut recs = self.records.lock().await;
                        if let Some(rec) = recs.get_mut(&port) {
                            rec.pid = id.pid;
                            rec.model = id.model.clone();
                            continue;
                        }
                    }
                    // Reclaim our own first: a port whose servers/<port>.toml
                    // exists is a runner a previous manager life configured -
                    // the config FILE gives the full spec back (API key
                    // included, so the Studio relay keeps working). Own again,
                    // never "adopted"; the election, when one exists, layers
                    // catalog identity + version pin + policy on top. A file
                    // with no election attaches the same way but records no
                    // desired state (a stopped-then-hand-started endpoint must
                    // not resurrect its boot election).
                    let election = self
                        .elections
                        .as_ref()
                        .and_then(|el| el.list().into_iter().find(|e| e.port == port));
                    let cfg_path = self.server_config_path(port);
                    let reclaimed = match &election {
                        Some(e) => self.spec_from_election(e).ok(),
                        None if cfg_path.exists() => self.spec_from_config_file(&cfg_path).ok(),
                        None => None,
                    };
                    if let Some(mut spec) = reclaimed {
                        // Heal a path-shaped identity (and persist the repair
                        // when an election carried it - self-corrects files
                        // written before the write_config_file fix).
                        let healed = self.heal_spec_identity(&mut spec);
                        if healed && let (Some(el), Some(e)) = (&self.elections, &election) {
                            el.record(crate::elections::Election {
                                model: spec.model.clone(),
                                artifact: spec.artifact.clone(),
                                port,
                                config: e.config.clone(),
                                runner_version: e.runner_version.clone(),
                                pinned: e.pinned,
                            });
                        }
                        let pinned = election.as_ref().is_some_and(|e| e.pinned);
                        tracing::info!(
                            port,
                            pid = id.pid,
                            model = id.model.as_deref().unwrap_or("-"),
                            elected = election.is_some(),
                            "reclaimed runner (config file restored its spec)"
                        );
                        self.records.lock().await.insert(
                            port,
                            Record {
                                origin: Origin::Own,
                                child: None,
                                model: id.model,
                                // attach keeps the runner alive across manager
                                // restarts - derive the label or every badge
                                // would blank until a full endpoint restart
                                spec_desc: self.describe_spec(&spec),
                                pid: id.pid,
                                api_key: spec.api_key.clone(),
                                pinned,
                                spec: Some(spec),
                            },
                        );
                        continue;
                    }
                    tracing::info!(
                        port,
                        pid = id.pid,
                        model = id.model.as_deref().unwrap_or("-"),
                        version = %id.version,
                        "adopted existing runner (same user; core ops via admin pipe)"
                    );
                    self.records.lock().await.insert(
                        port,
                        Record {
                            origin: Origin::Adopted,
                            child: None,
                            model: id.model,
                            spec_desc: None,
                            pid: id.pid,
                            api_key: None,
                            pinned: false,
                            spec: None,
                        },
                    );
                }
                Ok(Err(e)) => {
                    tracing::warn!(port, %e, "admin endpoint present but not answering identify (stale socket or hung runner) - left alone");
                }
                Err(_) => {
                    tracing::warn!(
                        port,
                        "admin endpoint present but identify timed out - left alone"
                    );
                }
            }
        }
        self.forget_departed().await;
    }

    /// The other direction of reconciliation: records whose runner is gone.
    ///
    /// The loop above only visits ports that still enumerate, so nothing in it
    /// can notice a record whose socket has disappeared - and until this
    /// existed, nothing anywhere did. A crashed or `kill -9`'d runner left its
    /// record for the life of the manager process, which hid its endpoint from
    /// the fleet list, pinned it as an "Unreachable" row with no Start button,
    /// and held its port against `allocate_port`.
    ///
    /// Gone means the same thing it means in `list()`: no admin socket AND no
    /// process of ours still alive. A live child with no socket yet is a runner
    /// still booting and is kept - see `silent_record_is_gone`.
    async fn forget_departed(&self) {
        let serving = paddock_admin::enumerate();
        let mut recs = self.records.lock().await;
        let gone: Vec<u16> = recs
            .iter_mut()
            .filter_map(|(port, rec)| {
                let child = match rec.child.as_mut() {
                    Some(c) => match c.try_wait() {
                        Ok(Some(_)) => ChildState::Exited,
                        _ => ChildState::Alive,
                    },
                    None => ChildState::NoHandle,
                };
                silent_record_is_gone(serving.contains(port), child).then_some(*port)
            })
            .collect();
        for port in gone {
            recs.remove(&port);
            tracing::info!(
                port,
                "runner gone (no admin socket, no live process) - record dropped"
            );
        }
    }

    /// Test the instance's own bookkeeping without enumerating the developer's
    /// live runners. A preparation/publication test must not require stopping them.
    #[cfg(test)]
    pub(crate) async fn has_records_for_test(&self) -> bool {
        !self.records.lock().await.is_empty()
    }

    /// Live view: own + adopted records, refreshed against enumeration, each
    /// queried for identify/health with a short timeout.
    pub async fn list(&self) -> Vec<RunnerView> {
        // Merge: recorded ports ∪ currently-enumerable endpoints (a runner
        // started after our boot shows up here - adoption on sight).
        let serving = paddock_admin::enumerate();
        let mut ports: Vec<u16> = {
            let recs = self.records.lock().await;
            recs.keys().copied().collect()
        };
        for p in serving.iter().copied() {
            if !ports.contains(&p) {
                ports.push(p);
            }
        }
        ports.sort_unstable();

        let mut out = Vec::new();
        for port in ports {
            let client = AdminClient::new(port);
            let id = tokio::time::timeout(Duration::from_secs(2), client.identify()).await;
            match id {
                Ok(Ok(id)) => {
                    let health =
                        tokio::time::timeout(Duration::from_secs(2), client.health()).await;
                    let (status, in_flight, uptime) = match health {
                        Ok(Ok(h)) => (h.status, Some(h.in_flight), Some(h.uptime_s)),
                        _ => ("unreachable".into(), None, None),
                    };
                    // Only a runner that advertises the capability is asked,
                    // so every other row costs nothing extra here.
                    let residency = if id.capabilities.iter().any(|c| c == "residency") {
                        tokio::time::timeout(Duration::from_secs(1), client.residency())
                            .await
                            .ok()
                            .and_then(Result::ok)
                            .flatten()
                    } else {
                        None
                    };
                    // Read only. This used to adopt on sight - insert a record
                    // for any port that answered - which made a *read* path an
                    // owner of authoritative state, and that was the shape
                    // behind the whole bug: `stop()` dropped a record, the SSE
                    // watcher called list() 2s later while the runner was still
                    // draining and answering, and the record came back with
                    // nothing left to remove it. The adoption it did was also
                    // lossier than the real one in `reconcile()`: it never
                    // consulted elections, so a PINNED endpoint came back
                    // pinned:false and silently lost its never-auto-stop
                    // policy. reconcile() owns the map now; list() renders it.
                    let cached = {
                        let recs = self.records.lock().await;
                        recs.get(&port).map(|r| {
                            (
                                r.origin,
                                r.pinned,
                                r.spec_desc.clone(),
                                r.spec.as_ref().map(RunnerConfig::from_spec),
                                r.spec.as_ref().map(|s| s.model.clone()),
                            )
                        })
                    };
                    let (origin, pinned, spec_desc, config, declared) = match cached {
                        Some(t) => t,
                        // Not adopted yet - the next reconcile pass will take
                        // it. Render what its config file says in the meantime
                        // so a just-started endpoint is not a blank row for a
                        // couple of seconds. Same rule reconcile uses: our file
                        // on the port = ours, no file = a foreign runner.
                        None => match self.spec_from_config_file(&self.server_config_path(port)) {
                            Ok(mut spec) => {
                                self.heal_spec_identity(&mut spec);
                                let desc = self.describe_spec(&spec);
                                (
                                    Origin::Own,
                                    false,
                                    desc,
                                    Some(RunnerConfig::from_spec(&spec)),
                                    Some(spec.model.clone()),
                                )
                            }
                            Err(_) => (Origin::Adopted, false, None, None, None),
                        },
                    };
                    let labels = id
                        .model
                        .as_deref()
                        .or(id.embedder.as_deref())
                        .or(id.asr.as_deref())
                        .or(id.aligner.as_deref())
                        .or(id.image.as_deref())
                        .or(id.reader.as_deref())
                        .or(id.tabular.as_deref())
                        .or(id.diarization.as_deref())
                        .and_then(|n| self.registry.display_of(n))
                        // The runner names its weights FILE. When the catalog
                        // does not know that file - a copy, a rename, a quant
                        // the catalog lacks - the endpoint's own declaration
                        // (its `[catalog]` block, reconciled at load into the
                        // spec's model id) still names the model, so the row
                        // keeps its name and its vendor mark. Same rule
                        // `identity_for` states for /api/servers.
                        .or_else(|| {
                            declared
                                .as_deref()
                                .and_then(|m| self.registry.display_of(m))
                        });
                    out.push(RunnerView {
                        port,
                        pid: id.pid,
                        origin,
                        status,
                        // live self-report first; election-time prediction
                        // covers runners that predate the identify field
                        spec: self.live_spec_label(&id).or(spec_desc),
                        model: id.model,
                        embedder: id.embedder,
                        asr: id.asr,
                        aligner: id.aligner,
                        image: id.image,
                        reader: id.reader,
                        tabular: id.tabular,
                        diarization: id.diarization,
                        display: labels.as_ref().map(|(d, _)| d.clone()),
                        vendor: labels.and_then(|(_, v)| v),
                        version: Some(id.version),
                        uptime_s: uptime,
                        in_flight,
                        endpoint: format!("http://{}:{port}", Self::lan_ip()),
                        pinned,
                        config,
                        residency,
                    });
                }
                _ => {
                    // Recorded or enumerated but silent: report honestly.
                    // Forgetting it is `forget_departed`'s job, on the
                    // reconcile pass - a read path does not get to decide a
                    // runner is dead. Until that pass runs (seconds) a departed
                    // runner shows as unreachable, which is what it is.
                    let recs = self.records.lock().await;
                    if let Some(rec) = recs.get(&port) {
                        let labels = rec
                            .model
                            .as_deref()
                            .and_then(|n| self.registry.display_of(n));
                        out.push(RunnerView {
                            port,
                            pid: rec.pid,
                            origin: rec.origin,
                            status: "unreachable".into(),
                            model: rec.model.clone(),
                            spec: rec.spec_desc.clone(),
                            embedder: None,
                            asr: None,
                            aligner: None,
                            image: None,
                            reader: None,
                            tabular: None,
                            diarization: None,
                            display: labels.as_ref().map(|(d, _)| d.clone()),
                            vendor: labels.and_then(|(_, v)| v),
                            version: None,
                            uptime_s: None,
                            in_flight: None,
                            endpoint: format!("http://{}:{port}", Self::lan_ip()),
                            pinned: rec.pinned,
                            config: rec.spec.as_ref().map(RunnerConfig::from_spec),
                            residency: None,
                        });
                    } else {
                        // Enumerated, silent, and not ours - a runner someone
                        // else started, or one that outlived our record, which
                        // has stopped answering. This branch used to emit
                        // nothing while `configured()` counted the very same
                        // port as running, and an endpoint that is "running"
                        // with no row renders in neither list: the vanishing.
                        // See `port_has_endpoint` for the invariant.
                        out.push(RunnerView {
                            port,
                            // identify is what would have told us, and it did
                            // not answer. 0 reads as unknown rather than
                            // inventing one.
                            pid: 0,
                            origin: Origin::Adopted,
                            status: "unreachable".into(),
                            model: None,
                            spec: None,
                            embedder: None,
                            asr: None,
                            aligner: None,
                            image: None,
                            reader: None,
                            tabular: None,
                            diarization: None,
                            display: None,
                            vendor: None,
                            version: None,
                            uptime_s: None,
                            in_flight: None,
                            endpoint: format!("http://{}:{port}", Self::lan_ip()),
                            pinned: false,
                            config: None,
                            residency: None,
                        });
                    }
                }
            }
        }
        out
    }

    /// Resolve a spawn's model to a serving composition: catalog id (with the
    /// selected weights artifact; downloads only when `pull` - the CLI path;
    /// the UI's deploy never downloads) -> installed model name -> filesystem
    /// path. Returns (weights, mmproj, mtp drafter, fp8 snapshot dir).
    ///
    /// `want_spec` is the endpoint's speculation policy, and it decides the
    /// DRAFTER, which is the manager-side half of "is spec on":
    ///
    /// - asked for (`auto`/`ladder`/pinned K): take any installed drafter, and
    ///   error if the catalog declares one that is not downloaded - serving
    ///   silently without the drafter someone just asked for is a silent
    ///   failure with a performance bug for a symptom;
    /// - explicit `off`: never wire one, which is what actually returns the
    ///   VRAM (the runtime policy alone cannot - the weights are resident);
    /// - unset: apply the selected artifact's default policy first; if it has
    ///   none, only a drafter the catalog marks `default`, so a non-default
    ///   companion (laguna's DFlash) stays opt-in.
    ///
    /// Models with in-file MTP declare no drafter artifact, so all three arms
    /// leave `mtp` empty and the loader finds `nextn` in the GGUF itself.
    async fn resolve_model(
        &self,
        name: &str,
        artifact: Option<&str>,
        pull: bool,
        want_spec: Option<&str>,
        drafter: Option<&str>,
    ) -> Result<Resolution, SpawnError> {
        match self.registry.resolve(name, artifact, pull, drafter).await {
            Ok(Some(r)) => {
                let default_policy = self
                    .registry
                    .identify_weights(&r.weights)
                    .and_then(|(model, artifact)| {
                        self.registry.catalog_of(&model)?.artifact(&artifact)
                    })
                    .and_then(|a| a.runtime.default_spec.as_deref());
                let effective_policy = want_spec.or(default_policy);
                // A catalog default is a policy, not an enable flag: Bonsai
                // explicitly defaults to "off" without offering speculation
                // in the UI. Interpret the inherited value exactly as an
                // explicit one before validating or electing any drafter.
                // Unknown policies still reach the runner's strict parser.
                let spec_off = effective_policy.is_some_and(|s| {
                    matches!(
                        s.trim().to_ascii_lowercase().as_str(),
                        "off" | "false" | "no" | "none" | "0"
                    )
                });
                let spec_on = effective_policy.is_some() && !spec_off;
                let mtp = if spec_off {
                    None
                } else if spec_on {
                    if !r.speculative {
                        // Plain "on" was every endpoint's form default before the
                        // capability gate existed, so a legacy granite-class toml
                        // carries it through no choice of the user's - and the
                        // current form cannot even write it for this model. Heal
                        // that one value to unset with a loud log instead of
                        // refusing the start (the stricter pre-flight would brick
                        // every legacy non-spec endpoint). Anything ELSE
                        // (adaptive, a pinned depth) is a deliberate setting and
                        // keeps the honest error.
                        let plain_on =
                            want_spec.is_some_and(|s| s.trim().eq_ignore_ascii_case("on"));
                        if !plain_on {
                            return Err(SpawnError::Unsupported(format!(
                                "{name}: this engine has no speculative decode for this model - it \
                                 ships no drafter and carries no in-file MTP. Serve it with \
                                 spec = \"off\" (or leave the key out)."
                            )));
                        }
                        tracing::warn!(
                            model = %name,
                            "legacy spec = \"on\" on a model with no drafter and no in-file \
                             MTP - serving without speculation (edit + save the endpoint to \
                             drop the stale key)"
                        );
                        r.mtp
                    } else {
                        if r.drafter_any.is_none() && r.drafter_declared {
                            return Err(SpawnError::ModelNotFound(format!(
                                "{name}: speculative decode needs this model's drafter, which is not \
                             downloaded - get it on the Models page (or serve with spec = \"off\")"
                            )));
                        }
                        r.drafter_any
                    }
                } else {
                    r.mtp
                };
                // The drafter identity only means something when one was
                // actually wired - spec "off" resolves a path and then drops it.
                let pick = mtp.is_some().then_some(r.drafter_pick).flatten();
                let adaptive =
                    effective_policy.is_some_and(|s| s.trim().eq_ignore_ascii_case("adaptive"));
                let spec_desc = if !r.speculative {
                    None // nothing to speculate with - no badge, not a choice
                } else if spec_off {
                    Some("off".to_owned())
                } else {
                    // CUDA can route between both mechanisms. Metal elects
                    // one companion, and an MLX export has no in-file heads.
                    let in_file = self
                        .registry
                        .catalog_of(name)
                        .is_some_and(|c| c.mtp_in_file)
                        && (self.defaults.device != "metal"
                            || (!r.weights.is_dir() && mtp.is_none()));
                    let mech = match &pick {
                        Some((_, label)) if in_file => {
                            format!("MTP + {}", Self::spec_token(label))
                        }
                        Some((_, label)) => Self::spec_token(label),
                        None if mtp.is_some() && in_file => "MTP + drafter".to_owned(),
                        None if mtp.is_some() => "drafter".to_owned(),
                        None => "MTP".to_owned(),
                    };
                    Some(if adaptive {
                        format!("{mech} · adaptive")
                    } else {
                        mech
                    })
                };
                return Ok(Resolution {
                    weights: r.weights,
                    mmproj: r.mmproj,
                    lane: LanePaths {
                        text_encoder: r.text_encoder,
                        vae: r.vae,
                    },
                    mtp,
                    fp8: r.fp8_snapshot,
                    drafter: pick,
                    spec_desc,
                });
            }
            Ok(None) => {}
            Err(e) => return Err(SpawnError::Pull(e.to_string())),
        }
        let p = PathBuf::from(name);
        if p.exists() {
            return Ok(Resolution::weights_only(p));
        }
        let store = paddock_models::ModelStore::new(self.defaults.models_dirs.clone());
        if let Ok(models) = store.list()
            && let Some(m) = models.into_iter().find(|m| m.id == name)
        {
            return Ok(Resolution::weights_only(m.path));
        }
        Err(SpawnError::ModelNotFound(name.to_string()))
    }

    /// Elect the runner binary for a spawn (doc §11.2/§11.5). Precedence:
    ///
    /// 1. An explicit per-spawn version pin - `runners/<pin>/paddock-runner`.
    ///    Missing pin = hard error; substituting a different version behind a
    ///    pin would be a silent failure.
    ///    (see `runner_kernels_builtin` below for the capability probe)
    ///
    /// 2. The operator's PADDOCK_RUNNER_BIN override (dev workflows).
    /// 3. The newest installed artifact under `runners/` (numeric dotted
    ///    version dirs; `delete a dir = garbage collection`).
    /// 4. The runner beside this manager binary (dev + simple install).
    fn runner_bin(&self, pin: Option<&str>) -> Result<PathBuf, SpawnError> {
        let exe_name = if cfg!(windows) {
            "paddock-runner.exe"
        } else {
            "paddock-runner"
        };
        if let Some(v) = pin {
            let b = self.defaults.runners_dir.join(v).join(exe_name);
            if b.exists() {
                return Ok(b);
            }
            return Err(SpawnError::NoBinary(format!(
                "{} (pinned runner version {v:?} is not installed)",
                b.display()
            )));
        }
        if let Some(b) = &self.defaults.runner_bin {
            if b.exists() {
                return Ok(b.clone());
            }
            return Err(SpawnError::NoBinary(b.display().to_string()));
        }
        if let Some((v, b)) = newest_runner_artifact(&self.defaults.runners_dir) {
            tracing::info!(version = %v, bin = %b.display(), "elected newest installed runner artifact");
            return Ok(b);
        }
        let exe = std::env::current_exe()?;
        let sibling = exe.with_file_name(exe_name);
        if sibling.exists() {
            return Ok(sibling);
        }
        Err(SpawnError::NoBinary(format!(
            "{} (set PADDOCK_RUNNER_BIN or install a runner artifact under {})",
            sibling.display(),
            self.defaults.runners_dir.display()
        )))
    }

    /// Whether a runner carries its own CUDA kernels, so no pack file is
    /// needed. Public so the readiness/Instrument surfaces can say which
    /// kernels a box is actually running.
    pub fn runner_has_builtin_kernels(&self, pin: Option<&str>) -> bool {
        self.runner_bin(pin)
            .map(|b| runner_kernels_builtin(&b))
            .unwrap_or(false)
    }

    /// Next free port from the base upward: not in our records, no admin
    /// endpoint, and TCP-bindable right now.
    async fn allocate_port(&self) -> Result<u16, SpawnError> {
        let taken = paddock_admin::enumerate();
        let recs = self.records.lock().await;
        for port in self.defaults.base_port.max(1024)..=self.defaults.base_port.saturating_add(63) {
            if recs.contains_key(&port)
                || self.server_config_path(port).exists()
                || self
                    .spawning
                    .lock()
                    .is_ok_and(|ports| ports.contains(&port))
                || (taken.contains(&port) && AdminClient::new(port).is_present().await)
            {
                continue;
            }
            if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
                return Ok(port);
            }
        }
        Err(SpawnError::NoPort(self.defaults.base_port))
    }

    /// The host's primary LAN address - endpoint strings must work from other
    /// machines now that runners bind all interfaces. The
    /// UDP-connect trick routes without sending a packet; loopback fallback
    /// when there is no route. Cached: the answer doesn't move.
    pub fn lan_ip() -> std::net::IpAddr {
        static IP: std::sync::OnceLock<std::net::IpAddr> = std::sync::OnceLock::new();
        *IP.get_or_init(|| {
            std::net::UdpSocket::bind("0.0.0.0:0")
                .and_then(|s| {
                    s.connect("8.8.8.8:80")?;
                    s.local_addr()
                })
                .map(|a| a.ip())
                .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        })
    }

    /// Serialize an endpoint's full launch config to `<data>/servers/<port>.toml`
    /// - the file is the truth, readable/diffable/copyable, and runnable
    ///   standalone (`paddock-runner --config <file>`). The runner's
    ///   deny_unknown_fields is the drift guard: a key this writer gets wrong
    ///   refuses at spawn with a clear parse error, never silently.
    #[allow(clippy::too_many_arguments)]
    fn write_server_config(
        &self,
        port: u16,
        weights: &Path,
        mmproj: &Option<PathBuf>,
        lane: &LanePaths,
        mtp: &Option<PathBuf>,
        gpu: Option<&str>,
        fp8: Option<&Path>,
        spec: &SpawnSpec,
    ) -> Result<PathBuf, SpawnError> {
        let dir = self.defaults.work_dir.join("servers");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{port}.toml"));
        let text = self.render_server_config(port, weights, mmproj, lane, mtp, gpu, fp8, spec)?;
        let mut options = std::fs::OpenOptions::new();
        options.create(true).write(true).truncate(true);
        // The file carries runner and connector credentials. New native
        // endpoints must not inherit a permissive shell umask. Existing
        // configs rewritten by an explicit edit are tightened as well.
        #[cfg(unix)]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            options.mode(0o600);
            if path.exists() {
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
            }
        }
        std::io::Write::write_all(&mut options.open(&path)?, text.as_bytes())?;
        Ok(path)
    }

    /// The config-file TEXT for a spec - Save writes exactly this, and the
    /// Start/Edit page's preview shows exactly this (one serializer; a
    /// preview that could drift from the file would be worse than none).
    #[allow(clippy::too_many_arguments)]
    fn render_server_config(
        &self,
        port: u16,
        weights: &Path,
        mmproj: &Option<PathBuf>,
        lane: &LanePaths,
        mtp: &Option<PathBuf>,
        gpu: Option<&str>,
        fp8: Option<&Path>,
        spec: &SpawnSpec,
    ) -> Result<String, SpawnError> {
        let identity = self.catalog_identity(spec, weights);
        let selected = identity.as_ref().and_then(|(id, art)| {
            let model = self.registry.catalog_of(id)?;
            model.artifact(art.as_deref()?).map(|a| (model, a))
        });
        let fixed_kv = selected.and_then(|(_, a)| a.runtime.kv_cache_dtype.as_deref());
        // An image lane has no token envelope - a picture is one pass, nothing
        // batched or windowed - so the keys stay out of its file instead of
        // carrying the chat defaults as if the runner read them.
        let image = selected.is_some_and(|(model, _)| {
            crate::estimate::kind_for(&model.capability) == paddock_estimator::ModelKind::Image
        });
        let max_ctx = if image {
            None
        } else {
            spec.max_ctx
                .or_else(|| selected.map(|(_, a)| a.runtime.default_envelope().0))
        };
        let max_batch = if image {
            None
        } else {
            spec.max_batch
                .or_else(|| selected.map(|(_, a)| a.runtime.default_envelope().1))
        };
        let no_spec = selected.is_some_and(|(model, a)| {
            (self.defaults.device == "metal" || a.runtime.capability.is_some())
                && !a.capabilities(model).iter().any(|c| c == "speculative")
        });
        // Apply the export contract to previews as well as real spawns. A
        // not-yet-downloaded checkpoint can otherwise take the planned-path
        // fallback and produce a config the runner must refuse.
        if let Some((model, a)) = selected {
            if let Some(memory) = &a.runtime.memory
                && max_ctx.is_some_and(|ctx| ctx == 0 || ctx as u64 > memory.max_ctx)
            {
                return Err(SpawnError::Unsupported(format!(
                    "{} on {} requires max_ctx <= {} (backend implementation limit)",
                    a.label, self.defaults.device, memory.max_ctx
                )));
            }
            if let Some(memory) = &a.runtime.memory
                && max_batch.is_some_and(|batch| batch == 0 || batch as u64 > memory.max_batch)
            {
                return Err(SpawnError::Unsupported(format!(
                    "{} on {} requires max_batch in 1..={} (backend implementation limit)",
                    a.label, self.defaults.device, memory.max_batch
                )));
            }
            if let Some(id) = spec.drafter.as_deref()
                && !model.artifacts.iter().any(|companion| {
                    companion.id == id
                        && companion.kind == crate::registry::ArtifactKind::Drafter
                        && companion.runtime.supports_backend(&self.defaults.device)
                        && a.runtime.allows_companion(id)
                })
            {
                return Err(SpawnError::Unsupported(format!(
                    "{} does not support companion {id} on {}",
                    a.label, self.defaults.device
                )));
            }
            if no_spec
                && spec.spec_policy.as_deref().is_some_and(|s| {
                    !matches!(
                        s.trim().to_ascii_lowercase().as_str(),
                        "off" | "false" | "no" | "none" | "0"
                    )
                })
            {
                return Err(SpawnError::Unsupported(format!(
                    "{} does not support speculative decoding; use spec = \"off\"",
                    a.label
                )));
            }
            if !a.runtime.supports_backend(&self.defaults.device) {
                return Err(SpawnError::Unsupported(format!(
                    "{} requires backend {}",
                    a.label,
                    a.runtime.backends.join(" or ")
                )));
            }
            if let (Some(required), Some(asked)) = (fixed_kv, spec.kv_cache_dtype.as_deref())
                && asked != required
                && !(required == "f16" && asked == "auto")
            {
                return Err(SpawnError::Unsupported(format!(
                    "{} requires kv_cache_dtype = {required:?} (checkpoint-native cache)",
                    a.label
                )));
            }
            if (spec.fp8_native || fp8.is_some())
                && !model.artifacts.iter().any(|companion| {
                    companion.kind == crate::registry::ArtifactKind::Fp8Snapshot
                        && companion.runtime.supports_backend(&self.defaults.device)
                        && a.runtime.allows_companion(&companion.id)
                })
            {
                return Err(SpawnError::Unsupported(format!(
                    "{} does not support FP8/native snapshot companions on {}",
                    a.label, self.defaults.device
                )));
            }
            if a.runtime.companions.as_ref().is_some_and(Vec::is_empty)
                && (mmproj.is_some()
                    || mtp.is_some()
                    || fp8.is_some()
                    || spec.fp8_native
                    || spec.drafter.is_some())
            {
                return Err(SpawnError::Unsupported(format!(
                    "{} does not support companion artifacts",
                    a.label
                )));
            }
            if a.runtime.embedded_vision && spec.vision == Some(false) {
                return Err(SpawnError::Unsupported(format!(
                    "{} includes a built-in vision tower; it cannot be disabled as an optional mmproj",
                    a.label
                )));
            }
            if a.runtime.capability.is_some()
                && spec.vision == Some(true)
                && !a.capabilities(model).iter().any(|c| c == "vision")
            {
                return Err(SpawnError::Unsupported(format!(
                    "{} currently supports text only",
                    a.label
                )));
            }
        }
        let mut t = toml::value::Table::new();
        // Existing web/CLI callers retain the LAN default. The native app
        // explicitly elects loopback; never widen that on a config edit.
        t.insert(
            "host".into(),
            spec.host
                .map_or_else(|| "0.0.0.0".into(), |host| host.to_string())
                .into(),
        );
        t.insert("port".into(), i64::from(port).into());
        t.insert("model".into(), weights.display().to_string().into());
        // PROVENANCE: which catalog entry those bytes are. The
        // manager needs this for a STOPPED endpoint - to name it, to preselect
        // it in the editor, to say what starting it would get you - and until
        // now the only record was the election, which is deleted on stop. What
        // was left was matching the filename against the catalog, which loses
        // the model on any rename, copy, or import-in-place.
        //
        // `model` above stays the path deliberately: one file, one meaning, and
        // still runnable by hand on a box that has no catalog.
        if let Some((id, art)) = identity {
            let mut c = toml::value::Table::new();
            c.insert("model".into(), id.into());
            if let Some(a) = art {
                c.insert("artifact".into(), a.into());
            }
            // Which drafter, when the endpoint pinned one. Only written when
            // pinned: an absent key means "the catalog default", which is what
            // should follow the catalog if the default later changes.
            if let Some(d) = &spec.drafter {
                c.insert("drafter".into(), d.clone().into());
            }
            t.insert("catalog".into(), toml::Value::Table(c));
        }
        if let Some(mm) = mmproj {
            t.insert("mmproj".into(), mm.display().to_string().into());
        } else if spec.vision == Some(false) {
            // Off has to be SAID. The runner loads a tower it finds beside the
            // weights when the file names none, so a file that only left the
            // mmproj line out went on serving images with the switch off - and
            // holding memory the admission never counted.
            t.insert("vision".into(), false.into());
        }
        // an image lane's two pieces beside its DiT (`model`); absent on every
        // other kind of model, and absent-in-render means removed
        if let Some(p) = &lane.text_encoder {
            t.insert("text_encoder".into(), p.display().to_string().into());
        }
        if let Some(p) = &lane.vae {
            t.insert("vae".into(), p.display().to_string().into());
        }
        let spec_off = spec.spec_policy.as_deref().is_some_and(|s| {
            matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "off" | "false" | "no" | "none" | "0"
            )
        });
        if let Some(m) = mtp.as_ref().filter(|_| !spec_off && !no_spec) {
            t.insert("mtp".into(), m.display().to_string().into());
        }
        // native-FP8 plane ingestion: the resolved snapshot dir, a plain
        // config field like everything else
        if let Some(d) = fp8 {
            t.insert("fp8_native".into(), d.display().to_string().into());
        }
        t.insert("device".into(), self.defaults.device.clone().into());
        // GPU pin: a device UUID (enumeration-order-proof) or an ordinal -
        // the runner resolves it natively; no CUDA_VISIBLE_DEVICES anywhere
        if let Some(g) = gpu {
            t.insert("gpu".into(), g.to_owned().into());
        }
        // Pin a pack only when this runner cannot supply its own. A path here
        // OVERRIDES built-in kernels (that is what makes it useful during
        // bring-up), so writing one by default would mean the
        // shipped, matched-vintage kernels were never once used - which is
        // exactly what happens to a static build, silently.
        //
        // Cannot tell -> write it, which is the old behaviour and the safe way
        // round: a redundant pin still serves, a missing one might not.
        let needs_pack = !self
            .runner_bin(spec.runner_version.as_deref())
            .map(|b| runner_kernels_builtin(&b))
            .unwrap_or(false);
        if let Some(p) = &self.defaults.kernel_pack
            && needs_pack
        {
            t.insert("kernel_pack".into(), p.display().to_string().into());
        }
        if !self.defaults.models_dirs.is_empty() {
            t.insert(
                "model_dirs".into(),
                toml::Value::Array(
                    self.defaults
                        .models_dirs
                        .iter()
                        .map(|d| d.display().to_string().into())
                        .collect(),
                ),
            );
        }
        if let Some(c) = max_ctx {
            t.insert("max_ctx".into(), (c as i64).into());
        }
        if let Some(b) = max_batch {
            t.insert("max_batch".into(), (b as i64).into());
        }
        if let Some(k) = &spec.api_key {
            t.insert("api_key".into(), k.clone().into());
        }
        if let Some(kv) = fixed_kv.or(spec.kv_cache_dtype.as_deref()) {
            t.insert("kv_cache_dtype".into(), kv.to_owned().into());
        }
        if no_spec {
            t.insert("spec".into(), "off".into());
        } else if let Some(sp) = spec
            .spec_policy
            .as_deref()
            .or_else(|| selected.and_then(|(_, a)| a.runtime.default_spec.as_deref()))
        {
            t.insert("spec".into(), sp.to_owned().into());
        }
        // the admission grant (MiB): the engine's hard cage - pools size
        // inside it, so co-resident endpoints can never sum past the card
        if let Some(b) = spec.vram_budget {
            t.insert("vram_budget".into(), (b as i64).into());
        }
        // SERVER TOOLS - the per-model integrations this endpoint supplies
        if let Some(p) = &spec.web_search_provider {
            t.insert("web_search_provider".into(), p.clone().into());
        }
        if let Some(k) = &spec.web_search_api_key {
            t.insert("web_search_api_key".into(), k.clone().into());
        }
        if !spec.mcp_servers.is_empty() {
            let arr = toml::Value::try_from(&spec.mcp_servers).map_err(|e| {
                SpawnError::Io(std::io::Error::other(format!(
                    "mcp_servers entry not TOML-expressible: {e}"
                )))
            })?;
            t.insert("mcp_servers".into(), arr);
        }
        // Forensics ([forensics]). Written only when ENABLED - an
        // absent block is disabled, and the overlay removes the key when it is
        // not rendered, so turning forensics off in the editor clears it. `auto`
        // and `tool` fall back to the product default (analyze everything,
        // expose the tool) when the caller sends bare `{enabled:true}`; a
        // hand-set scope rides through the projection and is preserved here.
        if let Some(f) = &spec.forensics
            && f.enabled
        {
            // Normalize the bare `{enabled:true}` the toggle sends up to the
            // full product default, then serialize the whole block in one shot
            // (no per-field `insert`, which would also trip the owned-key scan).
            let block = ForensicsSpec {
                enabled: true,
                auto: Some(f.auto.clone().unwrap_or_else(|| "all".into())),
                tool: Some(f.tool.unwrap_or(true)),
                device: f.device,
            };
            let fv = toml::Value::try_from(&block).map_err(|e| {
                SpawnError::Io(std::io::Error::other(format!(
                    "forensics block not TOML-expressible: {e}"
                )))
            })?;
            t.insert("forensics".into(), fv);
        }
        if let Some(policy) = &spec.residency {
            policy
                .validate()
                .map_err(|e| SpawnError::Io(std::io::Error::other(e)))?;
            t.insert(
                "residency".into(),
                toml::Value::try_from(policy)
                    .map_err(|e| SpawnError::Io(std::io::Error::other(e)))?,
            );
        }
        // `[kv_offload]` round-trips as a whole. Budgets only, and only when
        // ram_gb is real - the disk tier stores through RAM, so writing a
        // disk budget with no RAM budget would render a block that arms
        // nothing and warns at every start.
        if let Some(kv) = &spec.kv_offload
            && kv.enabled
            && kv.ram_gb > 0.0
        {
            let block = KvOffloadSpec {
                enabled: true,
                ram_gb: kv.ram_gb,
                // The BUDGET is what arms the disk tier; the runner defaults
                // the location under its own data root. So the budget rides
                // alone, and the path is written only when it overrides that
                // default - a path with no budget still arms nothing and is
                // dropped rather than written as a half-tier.
                nvme_gb: kv.nvme_gb,
                nvme_path: (kv.nvme_gb > 0.0).then(|| kv.nvme_path.clone()).flatten(),
            };
            let kvv = toml::Value::try_from(&block).map_err(|e| {
                SpawnError::Io(std::io::Error::other(format!(
                    "kv_offload block not TOML-expressible: {e}"
                )))
            })?;
            t.insert("kv_offload".into(), kvv);
        }
        let doc = toml::Value::Table(t);
        crate::backend_contract::validate(&self.registry, &self.defaults.device, &doc)
            .map_err(SpawnError::Unsupported)?;
        let body = toml::to_string_pretty(&doc)
            .map_err(|e| SpawnError::Io(std::io::Error::other(e.to_string())))?;
        Ok(format!("{CONFIG_HEADER}{body}"))
    }

    /// Render the config file a spec would produce - no write, no spawn, no
    /// key generation (the preview shows what the user typed; Save issues a
    /// key when none was given). A not-installed catalog model previews with
    /// its would-be install paths, so the page can show the file before the
    /// download.
    pub async fn preview_config(&self, spec: SpawnSpec) -> Result<String, SpawnError> {
        let port = spec.port.unwrap_or(self.defaults.base_port);
        let (weights, mmproj, lane, mtp, fp8_dir) = match self
            .resolve_model(
                &spec.model,
                spec.artifact.as_deref(),
                false,
                spec.spec_policy.as_deref(),
                spec.drafter.as_deref(),
            )
            .await
        {
            Ok(r) => (r.weights, r.mmproj, r.lane, r.mtp, r.fp8),
            // A model that simply is not downloaded yet still previews -
            // show where its files will land. A policy refusal does not:
            // previewing a config the spawn will reject would hand back a
            // file that looks fine and cannot start.
            Err(e @ SpawnError::Unsupported(_)) => return Err(e),
            Err(e) => match self
                .registry
                .planned_paths(&spec.model, spec.artifact.as_deref())
            {
                Some((w, mm, mt)) => (w, mm, self.planned_lane(&spec), mt, None),
                None => return Err(e),
            },
        };
        // Preview must apply the same explicit vision-off election as spawn,
        // including when resolve_model fell back to not-yet-downloaded paths.
        let mmproj = if spec.vision == Some(false) {
            None
        } else {
            mmproj
        };
        let fp8 = if spec.fp8_native { fp8_dir } else { None };
        // no NVML wait in a preview - an unsampled tracker just previews the
        // numeric index; Save resolves it properly
        let gpu = self.resolve_gpu(spec.gpu.as_deref(), false).await;
        self.render_server_config(
            port,
            &weights,
            &mmproj,
            &lane,
            &mtp,
            gpu.as_deref(),
            fp8.as_deref(),
            &spec,
        )
    }

    fn log_tail(&self, port: u16, lines: usize) -> String {
        let path = self.log_path(port);
        match std::fs::read_to_string(&path) {
            Ok(s) => {
                let all: Vec<&str> = s.lines().collect();
                let start = all.len().saturating_sub(lines);
                all[start..].join("\n")
            }
            Err(_) => format!("(no log at {})", path.display()),
        }
    }

    /// Spawn a runner per spec and health-gate it. Returns the live view.
    /// This is the CREATE verb: an explicit port whose config file already
    /// exists refuses - a configured endpoint is started (`start_config`) or
    /// edited (the switch path), never silently re-created over.
    pub async fn spawn(&self, spec: SpawnSpec) -> Result<RunnerView, SpawnError> {
        // Pre-flight (first-run honesty): a cuda spawn with no kernel pack
        // anywhere is doomed - the runner will refuse at load - so say it now,
        // before a config file is written or a download's start plan queues,
        // with what to do about it. The honest "can't serve on this yet",
        // never a late cryptic exit.
        if self.defaults.device == "cuda" && self.defaults.kernel_pack.is_none() {
            // ...unless the runner brings its own kernels, in which case there
            // is no pack to be missing. Ask the binary rather than assume:
            // runners ship on their own version line, so the one under
            // runners/<v>/ may answer differently from the one beside this
            // manager.
            let bin = self.runner_bin(spec.runner_version.as_deref())?;
            if !runner_kernels_builtin(&bin) {
                return Err(SpawnError::NoKernelPack);
            }
        }
        // (a RUNNING port falls through to the PortTaken error instead -
        // "start it" would be the wrong advice there)
        if let Some(p) = spec.port
            && self.server_config_path(p).exists()
            && !self.is_serving(p).await
        {
            return Err(SpawnError::AlreadyConfigured(p));
        }
        self.spawn_overwrite(spec).await
    }

    /// `spawn` without the existing-endpoint guard - the switch/takeover
    /// path, where rewriting the port's file is the point.
    async fn spawn_overwrite(&self, mut spec: SpawnSpec) -> Result<RunnerView, SpawnError> {
        // Runners bind all interfaces  - so every runner
        // gets a key unless the caller brought one. Network peers must send
        // it; loopback callers (this manager included) are exempt runner-side.
        if spec.api_key.is_none() {
            spec.api_key = Some(format!("pd-{}", uuid::Uuid::new_v4().simple()));
        }
        let r = self
            .resolve_model(
                &spec.model,
                spec.artifact.as_deref(),
                spec.pull,
                spec.spec_policy.as_deref(),
                spec.drafter.as_deref(),
            )
            .await?;
        let spec_desc = r.spec_desc.clone();
        let (weights, mmproj, lane, mtp, fp8_dir) = (r.weights, r.mmproj, r.lane, r.mtp, r.fp8);
        // Vision is a default companion; Some(false) is the deliberate
        // text-only serve (the tower's VRAM back).
        let mmproj = if spec.vision == Some(false) {
            None
        } else {
            mmproj
        };
        // Never WRITE a config that cannot start. A required companion is one
        // the engine refuses to serve the architecture without, so a file
        // missing it is a guaranteed exit-1 dressed up as a configured
        // endpoint - which is exactly what Qwen3-ASR once shipped as.
        // Refusing here names the problem while the operator is still looking
        // at the start dialog.
        if mmproj.is_none()
            && let Some(needed) = self.registry.required_companion(&spec.model)
        {
            return Err(match needed {
                Ok(p) => SpawnError::Unsupported(format!(
                    "{}: its {} is installed at {} but did not reach the composition - this is a manager bug, please report it",
                    spec.model,
                    "required speech/vision companion",
                    p.display()
                )),
                Err(label) => SpawnError::ModelNotFound(format!(
                    "{}: this model cannot serve without its {label}, which is not downloaded - get it on the Models page",
                    spec.model
                )),
            });
        }
        // The same rule for an image lane's text encoder and VAE - pieces
        // with no switch, so there is no "off" that would excuse their
        // absence.
        if let Some(label) = self
            .registry
            .missing_required_lane_companion(&spec.model, spec.artifact.as_deref())
        {
            return Err(SpawnError::ModelNotFound(format!(
                "{}: this model cannot serve without its {label}, which is not downloaded - get it on the Models page",
                spec.model
            )));
        }
        // FP8-native planes are opt-in (the official checkpoints' coarse block
        // scales measured worse than our bf16-derived planes - operator's
        // call, never auto-elected). The resolved snapshot dir is written
        // into the config file's `fp8_native` field like every other setting.
        let fp8 = if spec.fp8_native {
            match &fp8_dir {
                Some(dir) => Some(dir.clone()),
                None => {
                    return Err(SpawnError::Pull(format!(
                        "fp8_native requested but no FP8 snapshot artifact of {:?} is installed - get it on the Models page",
                        spec.model
                    )));
                }
            }
        } else {
            None
        };
        let port = {
            // Serialize selection through the in-flight reservation. This does
            // not hold the lock during model loading. Saved endpoints and
            // another spawn's pending port are never reused by automatic starts.
            let _allocation = self.port_allocation.lock().await;
            let port = match spec.port {
                Some(p) => {
                    if self.spawning.lock().is_ok_and(|ports| ports.contains(&p)) {
                        return Err(SpawnError::PortTaken(
                            p,
                            "a model is already starting".into(),
                        ));
                    }
                    if let Some(why) = self.port_blocker(p).await {
                        return Err(SpawnError::PortTaken(p, why));
                    }
                    p
                }
                None => self.allocate_port().await?,
            };
            if let Ok(mut s) = self.spawning.lock() {
                s.insert(port);
            }
            port
        };
        let _spawning = SpawningGuard { sup: self, port };
        let bin = self.runner_bin(spec.runner_version.as_deref())?;

        // GPU pin: resolve an NVML index to the device UUID now so the FILE
        // carries it (UUIDs are enumeration-order-proof; the runner resolves
        // them natively against the CUDA driver). The retained spec carries
        // the resolved form too, so a takeover-edit inherits it verbatim.
        spec.gpu = self.resolve_gpu(spec.gpu.as_deref(), true).await;

        // ── the per-endpoint config FILE - The model configuration ─────────
        // one TOML per endpoint in <data>/servers/,
        // holding everything the runner needs - resolved model paths, the
        // serving envelope, the API key, the GPU pin, and this model's SERVER
        // TOOLS (web search + named MCP servers). The manager stores none of
        // it; it is the editor and launcher of these files. A standalone user
        // writes the same file by hand and runs `paddock-runner --config
        // <file>` with no manager at all - identical capability, no
        // second-class path.
        let cfg_path = self.write_server_config(
            port,
            &weights,
            &mmproj,
            &lane,
            &mtp,
            spec.gpu.as_deref(),
            fp8.as_deref(),
            &spec,
        )?;
        tracing::info!(
            port,
            model = %weights.display(),
            bin = %bin.display(),
            "spawning runner"
        );
        self.launch_config(port, &bin, &cfg_path, spec.clone(), spec_desc.clone())
            .await?;

        // Desired state: a healthy spawn is an election (managed.toml) unless
        // the caller opted out. Keyed by port, so a takeover replaces. The
        // election is SMALL - the endpoint's real config is its file.
        if spec.persist
            && let Some(el) = &self.elections
        {
            el.record(crate::elections::Election {
                model: spec.model.clone(),
                artifact: spec.artifact.clone(),
                port,
                config: cfg_path.clone(),
                // Only an explicit pin persists; unpinned elections re-elect
                // the newest artifact at every respawn (§11.5 rollback = pin).
                runner_version: spec.runner_version.clone(),
                pinned: spec.pinned,
            });
        }

        // One authoritative view of what actually came up.
        let views = self.list().await;
        views.into_iter().find(|v| v.port == port).ok_or_else(|| {
            SpawnError::Io(std::io::Error::other("runner vanished after health-gate"))
        })
    }

    /// Start an EXISTING endpoint from its config file, verbatim - no
    /// re-render, no registry re-resolution: hand-edits (including fields the
    /// manager's own editor doesn't know) are honored exactly as written.
    /// `paddock start <port>` and boot respawns come through here. If the
    /// file's paths have moved, the runner fails loudly at startup and the
    /// error carries its log tail.
    /// The live speculation label from a runner's identify self-report - the
    /// process that did the attaching outranks every prediction. None when
    /// the runner predates the field (the record's election-time label then
    /// falls back) or has nothing to say.
    fn live_spec_label(&self, id: &paddock_admin::types::Identify) -> Option<String> {
        let s = id.spec.as_ref()?;
        if s.off {
            return Some("off".to_owned());
        }
        let drafter = s
            .drafter
            .as_deref()
            .map(|stem| self.drafter_token(id.model.as_deref(), stem));
        match (s.heads, drafter) {
            (true, Some(d)) => Some(format!("MTP + {d}")),
            (true, None) => Some("MTP".to_owned()),
            (false, Some(d)) => Some(d),
            (false, None) => None,
        }
    }

    /// Decorate a runner-reported drafter file stem with its catalog label's
    /// mechanism token ("dflash2-Q4_K_M" -> "DFlash2") - the runner reports
    /// SHAPE, the catalog owns NAMES. Unknown files keep the honest stem.
    fn drafter_token(&self, model: Option<&str>, stem: &str) -> String {
        model
            .and_then(|m| self.registry.catalog_of(m))
            .and_then(|cat| {
                cat.artifacts
                    .iter()
                    .filter(|a| matches!(a.kind, crate::registry::ArtifactKind::Drafter))
                    .find(|a| a.files.iter().any(|f| f.dest.contains(stem)))
            })
            .map(|a| Self::spec_token(&a.label))
            .unwrap_or_else(|| stem.to_owned())
    }

    /// Display-only speculation label for a VERBATIM start, where no
    /// resolution runs: same vocabulary as `Resolution::spec_desc` ("MTP",
    /// a drafter's label, "off", None), derived from the config + catalog.
    /// The drafter election is approximated (pin, else default, else first) -
    /// close enough for a badge; the spawn path stays the exact source.
    /// The badge token from a drafter's catalog label: the parenthesized
    /// mechanism name ("Faster single chats (DFlash2)" -> "DFlash2"), else
    /// the whole label. Pickers read the benefit words; badges wear the NAME
    /// - "Latency drafter" on a chip read as a defect.
    fn spec_token(label: &str) -> String {
        label
            .rsplit_once('(')
            .and_then(|(_, tail)| tail.strip_suffix(')'))
            .map(|t| t.trim().to_owned())
            .unwrap_or_else(|| label.to_owned())
    }

    fn describe_spec(&self, spec: &SpawnSpec) -> Option<String> {
        let cat = self.registry.catalog_of(&spec.model)?;
        let weights = spec
            .artifact
            .as_deref()
            .and_then(|id| cat.artifact(id))
            .or_else(|| {
                self.registry
                    .identify_weights(Path::new(&spec.model))
                    .and_then(|(_, id)| cat.artifact(&id))
            })
            .or_else(|| cat.default_weights_for_backend(&self.defaults.device, None))?;
        if !weights.capabilities(cat).iter().any(|c| c == "speculative") {
            return None;
        }
        let policy = spec
            .spec_policy
            .as_deref()
            .or(weights.runtime.default_spec.as_deref())
            .map(|s| s.trim().to_ascii_lowercase());
        if matches!(
            policy.as_deref(),
            Some("off" | "false" | "no" | "none" | "0")
        ) {
            return Some("off".to_owned());
        }
        let drafters: Vec<_> = cat
            .artifacts
            .iter()
            .filter(|a| matches!(a.kind, crate::registry::ArtifactKind::Drafter))
            .filter(|a| {
                a.runtime.supports_backend(&self.defaults.device)
                    && weights.runtime.allows_companion(&a.id)
            })
            // installed-only, mirroring the spawn election: a stopped row must
            // not claim a drafter a start would not actually wire
            .filter(|a| self.registry.is_artifact_installed(a))
            .collect();
        let picked = spec
            .drafter
            .as_deref()
            .and_then(|id| drafters.iter().find(|a| a.id == id))
            .or_else(|| drafters.iter().find(|a| a.default))
            .or_else(|| drafters.first())
            .map(|a| Self::spec_token(&a.label));
        // Mirror the backend-specific mechanism election in resolve_model.
        let in_file = cat.mtp_in_file
            && !weights.runtime.checkpoint_dir
            && (self.defaults.device != "metal" || picked.is_none());
        let mech = match picked {
            Some(t) if in_file => format!("MTP + {t}"),
            Some(t) => t,
            None if in_file => "MTP".to_owned(),
            None => return None,
        };
        Some(if policy.as_deref() == Some("adaptive") {
            format!("{mech} (adaptive)")
        } else {
            mech
        })
    }

    pub async fn start_config(&self, port: u16) -> Result<RunnerView, SpawnError> {
        let cfg_path = self.server_config_path(port);
        if !cfg_path.exists() {
            return Err(SpawnError::NotConfigured(
                port,
                cfg_path.display().to_string(),
            ));
        }
        self.validate_backend_config(&std::fs::read_to_string(&cfg_path)?)
            .map_err(SpawnError::Unsupported)?;
        if let Some(why) = self.port_blocker(port).await {
            return Err(SpawnError::PortTaken(port, why));
        }
        // visible to VRAM admission from here until the record lands
        if let Ok(mut s) = self.spawning.lock() {
            s.insert(port);
        }
        let _spawning = SpawningGuard { sup: self, port };
        let mut spec = self
            .spec_from_config_file(&cfg_path)
            .map_err(|e| SpawnError::Io(std::io::Error::other(e)))?;
        spec.port = Some(port);
        // An existing election contributes the runner-version pin and the pin
        // flag - launch mechanics the config file has no key for.
        //
        // It no longer contributes IDENTITY except as a fallback. The
        // file declares that now, and the file outranks the election on
        // everything it can state itself: an election is deleted on stop, so
        // letting it win would mean the same endpoint answered "which model is
        // this" differently depending on whether it had been started recently.
        let election = self
            .elections
            .as_ref()
            .and_then(|el| el.list().into_iter().find(|e| e.port == port));
        if let Some(e) = &election {
            spec.runner_version = e.runner_version.clone();
            spec.pinned = e.pinned;
            // Legacy file: no [catalog] block and an unrecognized weights
            // path, so `model` is still that path. The election is the only
            // record left.
            if spec.model.contains('/') || spec.model.contains('\\') {
                spec.model = e.model.clone();
                spec.artifact = e.artifact.clone();
            }
        }
        // Last resort for a file with no block, no election, and a weights path
        // the registry can still recognize - so the election recorded below
        // speaks ids.
        self.heal_spec_identity(&mut spec);
        // ...and, now that the identity is known, put back a required
        // companion the file is missing - see repair_required_companion for
        // why this is the one thing a start may rewrite.
        self.repair_required_companion(&cfg_path, &spec)?;
        // ...and record the identity in the file while we still know it, so a
        // legacy endpoint stops depending on its filename staying put.
        Self::stamp_catalog_identity(&cfg_path, &spec);
        let bin = self.runner_bin(spec.runner_version.as_deref())?;
        tracing::info!(port, config = %cfg_path.display(), "starting endpoint from its config file");
        let spec_desc = self.describe_spec(&spec);
        self.launch_config(port, &bin, &cfg_path, spec.clone(), spec_desc.clone())
            .await?;
        // a started endpoint is desired state again (stop removed it)
        if let Some(el) = &self.elections {
            el.record(crate::elections::Election {
                model: spec.model.clone(),
                artifact: spec.artifact.clone(),
                port,
                config: cfg_path.clone(),
                runner_version: spec.runner_version.clone(),
                pinned: spec.pinned,
            });
        }
        let views = self.list().await;
        views.into_iter().find(|v| v.port == port).ok_or_else(|| {
            SpawnError::Io(std::io::Error::other("runner vanished after health-gate"))
        })
    }

    /// Resolve a spawn's `gpu` selector to the form the config FILE carries.
    /// An NVML index (what /api/gpu shows and the CLI/UI send) becomes the
    /// device UUID; a "GPU-..." selector passes through untouched (respawns
    /// re-read it from the file). A bare index with no NVML answer is written
    /// as-is, loudly: the runner reads it as a CUDA ordinal, which can differ
    /// from the NVML order on multi-GPU boxes.
    async fn resolve_gpu(&self, sel: Option<&str>, wait: bool) -> Option<String> {
        let sel = sel?.trim().to_string();
        if sel.is_empty() {
            return None;
        }
        let Ok(idx) = sel.parse::<u32>() else {
            return Some(sel);
        };
        if let Some(t) = &self.gpu {
            // The NVML sampler may still be starting (a boot respawn races it
            // by milliseconds) - wait briefly for the first real sample
            // rather than degrading a UUID pin to ordinal guessing. Bounded:
            // 2 s, spawn-path only (previews don't wait).
            let mut s = t.latest();
            if wait {
                for _ in 0..20 {
                    if !s.gpus.is_empty() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    s = t.latest();
                }
            }
            if let Some(u) = s
                .gpus
                .iter()
                .find(|g| g.index == idx)
                .and_then(|g| g.uuid.clone())
            {
                tracing::info!(gpu = idx, uuid = %u, "GPU pin resolved to device UUID");
                return Some(u);
            }
        }
        tracing::warn!(
            gpu = idx,
            "no NVML UUID for the GPU pin - writing the numeric index (the runner reads it as a CUDA ordinal, which may differ from the NVML order on multi-GPU boxes)"
        );
        Some(sel)
    }

    /// The process-mechanics half of a launch: run `paddock-runner --config
    /// <file>` - nothing else on the command line and nothing injected into
    /// its environment; the file is the entire configuration, so a manager
    /// launch is byte-identical to a systemd/SCM/terminal launch (a hard
    /// invariant) - health-gate it, and record it as Own.
    async fn launch_config(
        &self,
        port: u16,
        bin: &Path,
        cfg_path: &Path,
        spec: SpawnSpec,
        spec_desc: Option<String>,
    ) -> Result<(), SpawnError> {
        std::fs::create_dir_all(&self.defaults.logs_dir)?;
        // One log per serving GENERATION: appending across generations mixed
        // three different models' lifetimes into one confusing file (a port
        // is reused; the maintainer hit qwen history above his gemma start). The prior
        // generation rotates to .prev.log for post-mortems - a crash story
        // is never lost, and the live tail shows only the current model.
        let path = self.log_path(port);
        if path.exists() {
            let prev = path.with_extension("prev.log");
            let _ = std::fs::remove_file(&prev);
            if let Err(e) = std::fs::rename(&path, &prev) {
                // e.g. a live tail holds the file open on Windows - append
                // rather than fail the spawn over a log file
                tracing::debug!(port, %e, "log rotation skipped (file busy) - appending");
            }
        }
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let log_err = log.try_clone()?;

        let mut cmd = std::process::Command::new(bin);
        cmd.arg("--config").arg(cfg_path);
        // Neutral working dir: never the manager's cwd (a repo-local
        // paddock.toml would silently join the election - §11.3).
        std::fs::create_dir_all(&self.defaults.work_dir)?;
        cmd.current_dir(&self.defaults.work_dir);
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::from(log));
        cmd.stderr(std::process::Stdio::from(log_err));
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // Own process group + no console window. Deliberately no job
            // object with kill-on-close - runners must survive the manager
            // (§11.4; per-runner Job Objects without that flag come with the
            // services work).
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // Own process group: a signal aimed at the manager's group can't
            // hit the data plane.
            cmd.process_group(0);
        }

        let mut child = cmd.spawn()?;
        let pid = child.id();
        #[cfg(windows)]
        assign_runner_job(&child, port);

        // Health-gate: admin identify + network healthz, watching for early
        // death so a load failure surfaces its actual log lines, not a timeout.
        let deadline = tokio::time::Instant::now() + self.defaults.health_timeout;
        let client = AdminClient::new(port);
        let healthz = format!("http://127.0.0.1:{port}/healthz");
        let http = reqwest::Client::new();
        loop {
            if let Ok(Some(status)) = child.try_wait() {
                let e = SpawnError::DiedOnStartup {
                    code: status.code(),
                    log_tail: self.log_tail(port, 20),
                };
                // RECORD it, don't only return it. Returning to one HTTP caller
                // is not the same as writing it down: a start that fails used to
                // leave manager.log ending mid-sentence - "starting endpoint",
                // "assigned to its own job object", then nothing at all - so the
                // Logs view and anyone reading afterwards found no trace of a
                // failure that definitely happened - three failed
                // starts of one port in a row, each invisible. A background
                // respawn or reconciliation start has no HTTP caller to tell.
                tracing::error!(port, code = ?status.code(), "start failed: {e}");
                return Err(e);
            }
            if tokio::time::Instant::now() >= deadline {
                // Startup hung: kill our own spawn rather than leak a
                // half-loaded process holding VRAM.
                let _ = child.kill();
                let e = SpawnError::HealthTimeout(
                    self.defaults.health_timeout,
                    self.log_tail(port, 20),
                );
                tracing::error!(port, "start failed: {e}");
                return Err(e);
            }
            let admin_ok = matches!(
                tokio::time::timeout(Duration::from_secs(1), client.identify()).await,
                Ok(Ok(_))
            );
            if admin_ok
                && let Ok(res) = http
                    .get(&healthz)
                    .timeout(Duration::from_secs(2))
                    .send()
                    .await
                && res.status().is_success()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }

        self.records.lock().await.insert(
            port,
            Record {
                origin: Origin::Own,
                child: Some(child),
                model: Some(spec.model.clone()),
                spec_desc,
                pid,
                api_key: spec.api_key.clone(),
                pinned: spec.pinned,
                spec: Some(spec),
            },
        );
        tracing::info!(port, pid, "runner healthy");
        Ok(())
    }

    /// Stop the runner on `port`: in-band drain+shutdown over the admin pipe,
    /// then wait for process exit. Own runners escalate to kill if the process
    /// outlives the timeout (safe: stateless on disk, driver reclaims VRAM);
    /// adopted runners are never force-killed (§6.1) - we report instead.
    ///
    /// The record is dropped twice deliberately. `stop_inner` drops it up front
    /// so nothing treats the port as live while it drains, but the drain window
    /// is seconds to minutes long and the runner keeps answering `identify`
    /// throughout it (shutdown acks immediately, then drains in a background
    /// task). Adoption runs on a timer - `reconcile()` now, `list()` before it -
    /// and takes any port that answers, so the record came back mid-drain and
    /// nothing ever removed it again. Moving adoption out of the read path did
    /// not retire this: a draining runner answers whoever asks, whenever they
    /// ask. The second drop is the guard, and it stays. That zombie made
    /// `configured()` report `running: true` forever, which hid the endpoint
    /// from the fleet list, pinned it as an "Unreachable" row with no Start
    /// button, and blocked its port in `allocate_port`. Dropping it again here,
    /// once the process is genuinely gone, closes that window.
    pub async fn stop(&self, port: u16, drain_timeout_ms: u64) -> Result<StopOutcome, String> {
        let out = self.stop_inner(port, drain_timeout_ms).await;
        // StillRunning means an adopted runner ignored the drain and is alive:
        // its record is the truth and must stay.
        if !matches!(out, Ok(StopOutcome::StillRunning))
            && self.records.lock().await.remove(&port).is_some()
        {
            tracing::debug!(port, "record re-adopted during the drain, dropped again");
        }
        out
    }

    async fn stop_inner(&self, port: u16, drain_timeout_ms: u64) -> Result<StopOutcome, String> {
        // The stop request is the desired-state change - drop the election
        // first, even if nothing answers (a crashed runner's stale election
        // must not respawn a model the operator explicitly stopped).
        if let Some(el) = &self.elections {
            el.remove(port);
        }
        let client = AdminClient::new(port);
        let ack = client.shutdown(Some(drain_timeout_ms)).await;
        let mut recs = self.records.lock().await;
        let rec = recs.remove(&port);
        drop(recs);
        let origin = rec.as_ref().map(|r| r.origin);
        match ack {
            Ok(_) => {}
            Err(e) => {
                // No pipe answer: nothing to stop, or it's hung.
                if rec.is_none() {
                    return Err(format!("no runner answering on port {port}: {e}"));
                }
            }
        }
        // Wait for the process to actually exit.
        if let Some(Record {
            child: Some(mut child),
            ..
        }) = rec
        {
            let exited = tokio::task::spawn_blocking(move || {
                // shutdown ack + drain timeout + margin
                let deadline =
                    std::time::Instant::now() + Duration::from_millis(drain_timeout_ms + 10_000);
                loop {
                    match child.try_wait() {
                        Ok(Some(_)) => return true,
                        Ok(None) if std::time::Instant::now() >= deadline => {
                            // Escalation ladder (§5): the admin channel had its
                            // chance; a hung process gets terminated. Safe by
                            // design - stateless on disk, VRAM reclaimed by
                            // the driver on teardown however it dies.
                            let _ = child.kill();
                            let _ = child.wait();
                            return false;
                        }
                        Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                        Err(_) => return true,
                    }
                }
            })
            .await
            .unwrap_or(false);
            return Ok(if exited {
                StopOutcome::Stopped
            } else {
                StopOutcome::Killed
            });
        }
        // Adopted (no handle): poll the pipe until it disappears.
        let deadline =
            tokio::time::Instant::now() + Duration::from_millis(drain_timeout_ms + 10_000);
        loop {
            if !AdminClient::new(port).is_present().await {
                return Ok(StopOutcome::Stopped);
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(match origin {
                    Some(Origin::Adopted) | None => StopOutcome::StillRunning,
                    Some(Origin::Own) => StopOutcome::StillRunning,
                });
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    /// Same-port takeover (§5): drain + stop the incumbent, then spawn the
    /// new election on the same port. The outage window is model-load time.
    pub async fn switch(
        &self,
        port: u16,
        mut spec: SpawnSpec,
        drain_timeout_ms: u64,
        expect_config_hash: Option<String>,
    ) -> Result<RunnerView, String> {
        // Optimistic concurrency: the edit page opened
        // against a specific file state; if the file moved since (hand-edit,
        // another session), refuse - Save must never clobber a change it
        // never showed the user.
        if let Some(expect) = &expect_config_hash
            && self.config_file_hash(port).as_deref() != Some(expect.as_str())
        {
            return Err(SpawnError::ConfigDrift.to_string());
        }
        // Takeover-as-edit: the new spec starts from the incumbent's, so the
        // endpoint keeps its identity - the same API key (clients keep
        // working) and the launch facts the editor doesn't surface (GPU pin,
        // fp8 planes). Passing a new value replaces the old one.
        {
            let recs = self.records.lock().await;
            if let Some(old) = recs.get(&port).and_then(|r| r.spec.as_ref()) {
                if spec.api_key.is_none() {
                    spec.api_key = old.api_key.clone();
                }
                if spec.gpu.is_none() {
                    spec.gpu = old.gpu.clone();
                }
                if !spec.fp8_native {
                    spec.fp8_native = old.fp8_native;
                }
                // the editor never surfaces the version pin - an edit must
                // not silently unpin (§11.5: unpinning is an explicit act)
                if spec.runner_version.is_none() {
                    spec.runner_version = old.runner_version.clone();
                }
            }
        }
        // Only ports we can see; a takeover of a foreign runner is an explicit
        // operator action and goes through the same path.
        match self.stop(port, drain_timeout_ms).await {
            Ok(StopOutcome::StillRunning) => {
                return Err(format!(
                    "runner on port {port} did not exit (adopted process; not force-killed) - takeover aborted"
                ));
            }
            Ok(_) => {}
            // Nothing answered and nothing serves the port: a STOPPED
            // configured endpoint being edited - the takeover degenerates
            // into a plain start from the new spec.
            Err(_) if !AdminClient::new(port).is_present().await => {}
            Err(e) => return Err(e),
        }
        spec.port = Some(port);
        self.spawn_overwrite(spec).await.map_err(|e| e.to_string())
    }

    /// Remove a STOPPED endpoint's configuration: delete servers/<port>.toml
    /// and any election. Refused while the port serves - stopping is how a
    /// running model ends; removal is for configuration you no longer want.
    pub async fn remove_config(&self, port: u16) -> Result<(), String> {
        self.remove_config_checked(port, None).await
    }

    /// Native removal is revision-guarded again after the async serving probe.
    pub async fn remove_config_checked(
        &self,
        port: u16,
        expected: Option<&str>,
    ) -> Result<(), String> {
        let path = self.server_config_path(port);
        if !path.exists() {
            return Err(format!(
                "port {port} has no config file at {}",
                path.display()
            ));
        }
        if let Some(why) = self.port_blocker(port).await {
            return Err(format!(
                "port {port} is serving - {why}. Stop it first, then remove"
            ));
        }
        if let Some(expected) = expected
            && self.config_file_hash(port).as_deref() != Some(expected)
        {
            return Err(SpawnError::ConfigDrift.to_string());
        }
        std::fs::remove_file(&path).map_err(|e| format!("delete {}: {e}", path.display()))?;
        if let Some(el) = &self.elections {
            el.remove(port);
        }
        tracing::info!(port, config = %path.display(), "endpoint configuration removed");
        Ok(())
    }

    /// Every configured endpoint on disk (servers/*.toml): port, the model
    /// path its file names, and whether it is serving right now. The
    /// filesystem is the enumeration - a stopped endpoint (no election, no
    /// record) is still configured.
    pub async fn configured(&self) -> Vec<ConfiguredEndpoint> {
        let dir = self.defaults.work_dir.join("servers");
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&dir) else {
            return out;
        };
        let recs = self.records.lock().await;
        let serving = paddock_admin::enumerate();
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(port) = name
                .strip_suffix(".toml")
                .and_then(|s| s.parse::<u16>().ok())
            else {
                continue;
            };
            // Identity through `spec_from_config_file` rather than a second
            // reader of the same file - the fleet list and the start path used
            // to answer "which model is this endpoint" separately, and the
            // answers differed. An unparseable file still lists, with
            // nothing claimed about it.
            let spec = self.spec_from_config_file(&e.path()).ok();
            let weights = std::fs::read_to_string(e.path())
                .ok()
                .and_then(|s| toml::from_str::<toml::Value>(&s).ok())
                .and_then(|v| {
                    v.get("model")
                        .and_then(toml::Value::as_str)
                        .map(String::from)
                });
            out.push(ConfiguredEndpoint {
                port,
                model: spec
                    .as_ref()
                    .map(|s| s.model.clone())
                    .filter(|m| !m.is_empty()),
                spec_desc: spec.as_ref().and_then(|s| self.describe_spec(s)),
                artifact: spec.and_then(|s| s.artifact),
                weights,
                // The same rule `list()` emits rows by - see
                // `port_has_endpoint`. These two must never drift apart again.
                running: port_has_endpoint(&recs, &serving, port),
            });
        }
        out.sort_by_key(|x| x.port);
        out
    }
}

/// Newest installed runner artifact: scan `runners_dir` for numeric dotted
/// version dirs (`1.7.2`) containing the runner executable, pick the highest.
/// Non-version dir names are skipped with a debug log - the install layout is
/// ours, so plain release versions are the contract (no pre-release tags in
/// artifact dirs; a build that needs pinning uses the explicit pin).
/// Does this runner carry its own CUDA kernels ?
///
/// ASKED of the BINARY, never assumed. Runners ship on their own version line
/// under `runners/<v>/`, so the manager routinely supervises a runner it did
/// not build; a compile-time guess here would be an assertion about somebody
/// else's file, which is the drift the gpu_support consolidation was written
/// to stop.
///
/// Cached per (path, mtime): the answer is a property of the file, a runner
/// update changes the mtime, and this is on the spawn path.
fn runner_kernels_builtin(bin: &Path) -> bool {
    type Cache = std::sync::Mutex<HashMap<(PathBuf, Option<std::time::SystemTime>), bool>>;
    static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    let key = (
        bin.to_path_buf(),
        std::fs::metadata(bin).and_then(|m| m.modified()).ok(),
    );
    let cache = CACHE.get_or_init(Default::default);
    if let Some(v) = cache.lock().expect("caps cache").get(&key) {
        return *v;
    }
    let v = probe_runner_kernels(bin);
    tracing::info!(bin = %bin.display(), kernels_builtin = v, "probed runner capabilities");
    cache.lock().expect("caps cache").insert(key, v);
    v
}

/// One `--capabilities` exec. Everything that is not a clear yes is a no:
/// a runner predating the flag exits non-zero on an unknown argument, which
/// is the right answer anyway - the flag and the built-in kernels shipped
/// together, so "does not understand the question" and "has no kernels of its
/// own" are the same runner.
fn probe_runner_kernels(bin: &Path) -> bool {
    let Ok(out) = std::process::Command::new(bin)
        .arg("--capabilities")
        .output()
    else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    serde_json::from_slice::<serde_json::Value>(&out.stdout)
        .ok()
        .and_then(|v| {
            v.get("kernels_builtin")
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(false)
}

fn newest_runner_artifact(runners_dir: &Path) -> Option<(String, PathBuf)> {
    let exe_name = if cfg!(windows) {
        "paddock-runner.exe"
    } else {
        "paddock-runner"
    };
    let mut best: Option<(Vec<u64>, String, PathBuf)> = None;
    for entry in std::fs::read_dir(runners_dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(key) = parse_version_dir(&name) else {
            tracing::debug!(dir = %name, "runners/ entry is not a version dir - skipped");
            continue;
        };
        let bin = entry.path().join(exe_name);
        if !bin.exists() {
            tracing::debug!(dir = %name, "version dir has no runner executable - skipped");
            continue;
        }
        if best.as_ref().is_none_or(|(k, _, _)| key > *k) {
            best = Some((key, name, bin));
        }
    }
    best.map(|(_, v, b)| (v, b))
}

/// `"1.7.2"` -> `[1,7,2]`; any non-numeric segment disqualifies. Numeric-tuple
/// ordering gives `1.10.0 > 1.9.0` (lexical sort would not).
fn parse_version_dir(name: &str) -> Option<Vec<u64>> {
    let parts: Vec<u64> = name
        .split('.')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    (!parts.is_empty()).then_some(parts)
}

/// §11.4: put a freshly spawned runner in its own Job Object - the Windows
/// analog of the systemd transient scope. The mechanics carry the semantics:
///
/// - The job is created with kill-on-close, the child is assigned, the job
///   handle is duplicated into the child, and the manager's own handle is
///   closed. The only live handle then sits inside the runner process, so
///   the runner SURVIVES the manager (we hold nothing whose closing could
///   fire the kill), and when the runner exits - cleanly or by crash - the
///   OS closes its handle table, the job's last handle goes away, and
///   kill-on-close terminates any children the runner may have spawned.
///   No orphaned grandchildren, no lifetime tie to the control plane.
/// - The job is named (`Local\paddock-runner-<port>-<pid>`; pid keeps a
///   takeover's fresh spawn out of any stale namesake) so accounting can
///   OpenJobObject on demand without keeping a persistent handle.
/// - Honest race: the child runs briefly before assignment; a child spawned
///   in that window would escape the job. Today's runner spawns no children,
///   and nested jobs (Win8+) keep the design extensible.
///
/// Failure is logged, never fatal - a runner without a job still serves.
#[cfg(windows)]
fn assign_runner_job(child: &std::process::Child, port: u16) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{
        CloseHandle, DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE,
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let name: Vec<u16> = format!("Local\\paddock-runner-{port}-{}\0", child.id())
        .encode_utf16()
        .collect();
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), name.as_ptr());
        if job.is_null() {
            tracing::warn!(port, err = %std::io::Error::last_os_error(), "CreateJobObject failed: runner not jobbed");
            return;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&raw const info).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        ) == 0
        {
            tracing::warn!(port, err = %std::io::Error::last_os_error(), "SetInformationJobObject failed: runner not jobbed");
            CloseHandle(job);
            return;
        }
        let proc_h = child.as_raw_handle() as HANDLE;
        if AssignProcessToJobObject(job, proc_h) == 0 {
            tracing::warn!(port, err = %std::io::Error::last_os_error(), "AssignProcessToJobObject failed: runner not jobbed");
            CloseHandle(job);
            return;
        }
        // Park the job's surviving handle inside the runner. From here on the
        // job's lifetime is the runner's lifetime.
        let mut dup: HANDLE = std::ptr::null_mut();
        if DuplicateHandle(
            GetCurrentProcess(),
            job,
            proc_h,
            &mut dup,
            0,
            0,
            DUPLICATE_SAME_ACCESS,
        ) == 0
        {
            // Can't park a handle in the child, and closing OURS now would
            // fire kill-on-close on the runner we just spawned. Strip the
            // flag, then close: the job stays as inert accounting, children
            // cleanup is honestly lost for this runner.
            tracing::warn!(port, err = %std::io::Error::last_os_error(), "DuplicateHandle into runner failed: job kept without kill-on-close (no child cleanup)");
            info.BasicLimitInformation.LimitFlags = 0;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            CloseHandle(job);
            return;
        }
        CloseHandle(job);
        tracing::info!(
            port,
            pid = child.id(),
            "runner assigned to its own job object (survives the manager; children die with the runner)"
        );
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopOutcome {
    /// Exited after in-band drain+shutdown.
    Stopped,
    /// Had to be terminated after the timeout (own runners only).
    Killed,
    /// Would not die and we don't own it - reported, never forced (§6.1).
    StillRunning,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise the installed-model resolver, not preview's missing-download
    /// fallback. These tiny files satisfy catalog presence checks only; no
    /// weights are downloaded or loaded by these control-plane tests.
    fn installed_model_supervisor(
        dir: &Path,
        model_id: &str,
        default_spec: Option<&str>,
    ) -> Supervisor {
        let models = dir.join("models");
        let source = crate::registry::Registry::new(models.clone()).with_backend("metal");
        let mut model = source.catalog_of(model_id).unwrap().clone();
        for artifact in &mut model.artifacts {
            if let Some(policy) = default_spec {
                artifact.runtime.default_spec = Some(policy.into());
            }
            for file in &mut artifact.files {
                file.size = 1;
                let path = models.join(&file.dest);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, b"x").unwrap();
            }
        }
        let registry = crate::registry::Registry::from_catalog(
            crate::registry::Catalog {
                schema: 3,
                models: vec![model],
            },
            models.clone(),
        )
        .with_backend("metal");
        Supervisor::new(
            SpawnDefaults {
                runner_bin: None,
                runners_dir: dir.join("runners"),
                device: "metal".into(),
                kernel_pack: None,
                models_dirs: vec![models],
                logs_dir: dir.join("logs"),
                work_dir: dir.into(),
                base_port: 18100,
                health_timeout: Duration::from_secs(1),
            },
            Arc::new(registry),
            None,
            None,
        )
    }

    #[tokio::test]
    async fn invalid_metal_edits_never_replace_saved_configuration() {
        let dir = tempfile::tempdir().unwrap();
        let sup = installed_model_supervisor(dir.path(), "gemma-4-31b", None);
        let valid = sup
            .preview_config(SpawnSpec {
                model: "gemma-4-31b".into(),
                artifact: Some("mlx-4bit".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let port = 18100;
        sup.write_config_file_deferred(port, &valid, None).unwrap();
        let bad = format!("{valid}\n[kv_offload]\nenabled = true\nram_gb = 8\nnvme_gb = 0\n");
        assert!(sup.write_config_file_deferred(port, &bad, None).is_err());
        assert!(
            sup.write_config_file(port, &bad, None, 100, None)
                .await
                .is_err()
        );
        assert_eq!(sup.read_config_file(port).unwrap().0, valid);
        assert!(!sup.is_serving(port).await);
    }

    /// Vision OFF has to reach the file as `vision = false`: the runner loads
    /// a tower it finds beside the weights whenever the file names none, so
    /// leaving the mmproj line out served images with the switch off. The
    /// projection and the file-derived spec must both read the off back, and
    /// a re-render from that spec must not resolve the tower again.
    #[tokio::test]
    async fn vision_off_is_written_read_back_and_survives_a_re_render() {
        let dir = tempfile::tempdir().unwrap();
        let sup = installed_model_supervisor(dir.path(), "qwen3.5-9b", None);
        let spec = |vision| SpawnSpec {
            model: "qwen3.5-9b".into(),
            artifact: Some("q8".into()),
            vision,
            ..Default::default()
        };
        let on = sup.preview_config(spec(None)).await.unwrap();
        let v: toml::Value = toml::from_str(&on).unwrap();
        assert!(v.get("mmproj").is_some(), "{on}");
        assert!(v.get("vision").is_none(), "{on}");

        let off = sup.preview_config(spec(Some(false))).await.unwrap();
        let v: toml::Value = toml::from_str(&off).unwrap();
        assert!(v.get("mmproj").is_none(), "{off}");
        assert_eq!(v["vision"].as_bool(), Some(false), "{off}");

        assert!(sup.project_config_text(&on).unwrap().vision);
        assert!(!sup.project_config_text(&off).unwrap().vision);
        let from_file = sup.spec_from_config_text(&off).unwrap();
        assert_eq!(from_file.vision, Some(false));
        assert_eq!(sup.spec_from_config_text(&on).unwrap().vision, None);
        let again = sup.preview_config(from_file).await.unwrap();
        let v: toml::Value = toml::from_str(&again).unwrap();
        assert!(v.get("mmproj").is_none(), "{again}");
        assert_eq!(v["vision"].as_bool(), Some(false), "{again}");
    }

    #[tokio::test]
    async fn installed_non_speculative_models_honor_catalog_off() {
        for (model, artifact) in [
            ("bonsai-2-27b", "mlx-2bit"),
            ("gemma-4-31b", "mlx-4bit"),
            ("muse-glimmer-30b", "mlx-4bit"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let sup = installed_model_supervisor(dir.path(), model, None);
            let resolved = sup
                .resolve_model(model, Some(artifact), false, None, None)
                .await
                .unwrap();
            assert!(resolved.mtp.is_none(), "{model}");
            assert!(resolved.drafter.is_none(), "{model}");
            assert!(resolved.spec_desc.is_none(), "{model}");
            let request = SpawnSpec {
                model: model.into(),
                artifact: Some(artifact.into()),
                ..Default::default()
            };
            let preview = sup.preview_config(request.clone()).await.unwrap();
            let config: toml::Value = toml::from_str(&preview).unwrap();
            assert_eq!(config["spec"].as_str(), Some("off"), "{model}");
            assert!(config.get("mtp").is_none(), "{model}");
            sup.render_spec_config(18100, request).await.unwrap();
            // Deliberately enabling an unsupported mechanism must still fail.
            for policy in ["adaptive", "4"] {
                assert!(matches!(
                    sup.resolve_model(model, Some(artifact), false, Some(policy), None)
                        .await,
                    Err(SpawnError::Unsupported(_))
                ));
            }
        }
    }

    #[tokio::test]
    async fn installed_speculative_model_default_off_does_not_load_a_drafter() {
        for policy in ["off", " OFF ", "false", "no", "none", "0"] {
            let dir = tempfile::tempdir().unwrap();
            let sup = installed_model_supervisor(dir.path(), "qwen3.8-27b", Some(policy));
            let resolve =
                |want| sup.resolve_model("qwen3.8-27b", Some("mlx-4bit"), false, want, None);
            let inherited = resolve(None).await.unwrap();
            assert!(inherited.mtp.is_none(), "{policy}");
            assert!(inherited.drafter.is_none(), "{policy}");
            assert_eq!(inherited.spec_desc.as_deref(), Some("off"), "{policy}");

            let enabled = resolve(Some("adaptive")).await.unwrap();
            assert!(enabled.mtp.is_some(), "explicit policy overrides {policy}");
            assert!(enabled.drafter.is_some());
            assert!(enabled.spec_desc.unwrap().contains("adaptive"));

            let disabled = resolve(Some("off")).await.unwrap();
            assert!(disabled.mtp.is_none());
            assert_eq!(disabled.spec_desc.as_deref(), Some("off"));
        }
    }

    #[tokio::test]
    async fn automatic_port_skips_saved_loading_and_unrelated_listeners() {
        let dir = tempfile::tempdir().unwrap();
        let socket = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let base = socket.local_addr().unwrap().port();
        let sup = Supervisor::new(
            SpawnDefaults {
                runner_bin: None,
                runners_dir: dir.path().join("runners"),
                device: "metal".into(),
                kernel_pack: None,
                models_dirs: vec![],
                logs_dir: dir.path().join("logs"),
                work_dir: dir.path().into(),
                base_port: base,
                health_timeout: Duration::from_secs(1),
            },
            Arc::new(crate::registry::Registry::new(dir.path().join("models"))),
            None,
            None,
        );
        let first = sup.allocate_port().await.unwrap();
        assert_ne!(first, base, "never take over another service's listener");
        let config = sup.server_config_path(first);
        std::fs::create_dir_all(config.parent().unwrap()).unwrap();
        std::fs::write(&config, "# saved endpoint fixture\n").unwrap();
        let second = sup.allocate_port().await.unwrap();
        assert_ne!(second, first, "a stopped model's address stays reserved");
        sup.spawning.lock().unwrap().insert(second);
        let third = sup.allocate_port().await.unwrap();
        assert!(![base, first, second].contains(&third));
        assert_eq!(
            std::fs::read_to_string(config).unwrap(),
            "# saved endpoint fixture\n"
        );
        assert!(socket.local_addr().is_ok(), "unrelated listener stays open");
    }

    /// A refusal that does not name what it is refusing over sent a user
    /// hunting for an hour. The blocker has to survive into the message.
    #[test]
    fn a_taken_port_says_what_is_holding_it() {
        let sock = "/run/user/1000/paddock/runner-11540.sock";
        let msg =
            SpawnError::PortTaken(11540, format!("something is listening on {sock}")).to_string();
        assert!(msg.contains("11540"), "{msg}");
        assert!(
            msg.contains(sock),
            "the operator cannot act on what is not named: {msg}"
        );

        let msg = SpawnError::PortTaken(11540, "this manager has a runner on it, pid 4242".into())
            .to_string();
        assert!(msg.contains("pid 4242"), "{msg}");
    }

    fn bare_record(pid: u32) -> Record {
        Record {
            origin: Origin::Adopted,
            child: None,
            model: None,
            spec_desc: None,
            pid,
            pinned: false,
            spec: None,
            api_key: None,
        }
    }

    /// `list()` and `configured()` answer "is anything on this port" for two
    /// different surfaces, and when they disagreed the endpoint rendered on
    /// neither. One rule, both callers.
    #[test]
    fn a_port_counts_as_occupied_from_either_side_alone() {
        let mut recs = HashMap::new();
        assert!(!port_has_endpoint(&recs, &[], 11540), "nothing anywhere");

        // Socket only: a runner we did not start, or one that outlived our
        // record. This is the case that used to satisfy `configured().running`
        // while producing no runner row at all.
        assert!(port_has_endpoint(&recs, &[11540], 11540));

        // Record only: ours, still booting, socket not up yet.
        recs.insert(11540, bare_record(4242));
        assert!(port_has_endpoint(&recs, &[], 11540));
        assert!(port_has_endpoint(&recs, &[11540], 11540));

        // Neighbouring ports are not implicated by either signal.
        assert!(!port_has_endpoint(&recs, &[11540], 11541));
    }

    /// The zombie-record rule, which had no rule before: quiet is not gone.
    #[test]
    fn a_silent_record_is_only_forgotten_when_it_is_provably_gone() {
        // Gone: nothing enumerates and the process we launched has exited, or
        // there was never a process of ours to begin with (adopted/attached).
        assert!(silent_record_is_gone(false, ChildState::Exited));
        assert!(silent_record_is_gone(false, ChildState::NoHandle));

        // Booting: our child is alive, its socket is not up yet. Dropping this
        // record strands the process handle and we could never stop it again.
        assert!(!silent_record_is_gone(false, ChildState::Alive));

        // Hung: the socket is still there but identify does not answer. That is
        // a real state an operator has to be able to see, so it keeps its row.
        assert!(!silent_record_is_gone(true, ChildState::Alive));
        assert!(!silent_record_is_gone(true, ChildState::Exited));
        assert!(!silent_record_is_gone(true, ChildState::NoHandle));
    }

    /// A line the browser actually met. What a toast can show
    /// is the first ~100 characters, so those characters have to be the answer.
    #[test]
    fn a_start_failure_leads_with_the_reason_not_a_timestamp() {
        let tail = "2026-08-17T16:36:20.802698Z  INFO paddock_runner::serving: kv cache: f16\n\
             2026-08-17T16:36:22.840396Z ERROR paddock_runner::startup: server error \
             error=engine startup: qwen35 cannot serve max_ctx 131072 x max_batch 1: needs \
             8.00 GiB of KV (8192 blocks), 3.75 GiB fits (3844 blocks, 61504 tokens shared). \
             Fixes: lower max_ctx to <=61504, or raise vram_budget";
        let text = died_on_startup_text(&Some(1), tail);
        let first = text.lines().next().unwrap();
        assert!(
            first.starts_with("qwen35 cannot serve max_ctx 131072"),
            "{first}"
        );
        // none of the machinery survives into the part a person reads
        for noise in [
            "2026-08-17T",
            "ERROR",
            "paddock_runner::startup",
            "error=",
            "engine startup:",
        ] {
            assert!(!first.contains(noise), "{noise:?} leaked into: {first}");
        }
        // and the actionable half is still in that same first line
        assert!(first.contains("Fixes: lower max_ctx"), "{first}");
        // the exit code and the whole tail remain, for the detail view
        assert!(text.contains("exit code 1"));
        assert!(text.contains("log tail:"));
        assert!(text.contains("kv cache: f16"));
    }

    /// A line that is not a tracing line must survive untouched - the stripper
    /// guesses, so it has to fail safe.
    #[test]
    fn human_line_leaves_unrecognised_shapes_alone() {
        for raw in [
            "thread 'main' panicked at src/main.rs:12:5: assertion failed",
            "LINK : fatal error LNK1181: cannot open input file 'pd-cuda.lib'",
            "CUDA error 2: out of memory",
            "",
        ] {
            assert_eq!(human_line(raw), raw.trim(), "mangled: {raw}");
        }
    }

    #[test]
    fn a_startup_death_with_no_log_still_says_something() {
        let text = died_on_startup_text(&Some(101), "");
        assert!(text.contains("exited during startup"), "{text}");
        assert!(text.contains("101"), "{text}");
    }

    #[test]
    fn version_dirs_order_numerically_not_lexically() {
        assert!(parse_version_dir("1.10.0").unwrap() > parse_version_dir("1.9.0").unwrap());
        assert!(parse_version_dir("2.0").unwrap() > parse_version_dir("1.99.99").unwrap());
        assert!(parse_version_dir("v1.2.3").is_none());
        assert!(parse_version_dir("1.2.3-rc1").is_none());
        assert!(parse_version_dir("").is_none());
    }

    #[test]
    fn newest_artifact_wins_and_binaryless_dirs_are_skipped() {
        let exe_name = if cfg!(windows) {
            "paddock-runner.exe"
        } else {
            "paddock-runner"
        };
        let dir = std::env::temp_dir().join(format!("paddock-runners-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for v in ["1.4.0", "1.10.2", "1.9.9"] {
            std::fs::create_dir_all(dir.join(v)).unwrap();
            std::fs::write(dir.join(v).join(exe_name), b"stub").unwrap();
        }
        // newer version dir but no executable inside - must not be elected
        std::fs::create_dir_all(dir.join("2.0.0")).unwrap();
        std::fs::create_dir_all(dir.join("not-a-version")).unwrap();

        let (v, bin) = newest_runner_artifact(&dir).expect("an artifact");
        assert_eq!(v, "1.10.2");
        assert!(bin.ends_with(std::path::Path::new("1.10.2").join(exe_name)));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every key the renderer writes must be DECLARED owned, or `merge_owned_keys`
    /// treats it as hand-edited state: the manager would then be unable to change
    /// or clear it through the Simple tab, silently and only for that one key.
    ///
    /// Read off the source rather than by rendering, so a key is caught the moment
    /// it is typed - no spec has to be able to reach it first.
    #[test]
    fn render_emits_only_owned_keys() {
        let src = include_str!("supervisor.rs");
        let start = src.find("fn render_server_config").expect("the renderer");
        // its body ends where the next item at impl indentation begins
        let rest = &src[start..];
        let end = rest[1..]
            .find("\n    /// ")
            .map(|i| i + 1)
            .unwrap_or(rest.len());
        let body = &rest[..end];

        let mut missing: Vec<&str> = Vec::new();
        for (i, _) in body.match_indices("t.insert(\"") {
            let after = &body[i + "t.insert(\"".len()..];
            let key = &after[..after.find('"').expect("a closed key literal")];
            if !OWNED_CONFIG_KEYS.contains(&key) {
                missing.push(key);
            }
        }
        assert!(
            missing.is_empty(),
            "render_server_config writes {missing:?}, which OWNED_CONFIG_KEYS does not declare - \
             add them there or the Simple tab can never clear them"
        );
        // and the scan actually found the renderer, not an empty slice
        assert!(
            body.contains("t.insert(\"model\""),
            "the key scan matched nothing - the body split moved"
        );
    }

    /// The `[forensics]` owned key round-trips exactly as render writes it and
    /// project reads it - the two halves of the Simple-tab contract. The
    /// renderer normalizes a bare `{enabled:true}` to the product default; the
    /// projector must read every field back. A hand-set scope survives.
    #[test]
    fn forensics_block_round_trips_render_shape_and_project_shape() {
        // What render_server_config serializes for a bare enable.
        let normalized = ForensicsSpec {
            enabled: true,
            auto: Some("all".into()),
            tool: Some(true),
            device: None,
        };
        let text = toml::to_string_pretty(&toml::Value::try_from(&normalized).unwrap()).unwrap();
        assert!(text.contains("enabled = true"), "{text}");
        assert!(text.contains("auto = \"all\""), "{text}");
        assert!(text.contains("tool = true"), "{text}");
        assert!(
            !text.contains("device"),
            "device omitted when sharing the model GPU: {text}"
        );

        // What project_config_text reads back out of a `[forensics]` block -
        // including a hand-set scope and a cross-GPU device pin.
        let file = "[forensics]\nenabled = true\nauto = \"images\"\ntool = false\ndevice = 1\n";
        let v: toml::Value = toml::from_str(file).unwrap();
        let parsed: ForensicsSpec = v.get("forensics").cloned().unwrap().try_into().unwrap();
        assert!(parsed.enabled);
        assert_eq!(parsed.auto.as_deref(), Some("images"));
        assert_eq!(parsed.tool, Some(false));
        assert_eq!(parsed.device, Some(1));
    }

    /// `[kv_offload]` must survive the render/project round trip, and the two
    /// halves of the disk tier must travel together: a path with no budget or
    /// a budget with no path arms nothing and warns at every start, so the
    /// renderer never writes half a pair.
    #[test]
    fn kv_offload_block_round_trips_and_the_disk_budget_stands_alone() {
        let full = KvOffloadSpec {
            enabled: true,
            ram_gb: 24.0,
            nvme_gb: 200.0,
            nvme_path: Some("D:/paddock-cache".into()),
        };
        let text = toml::to_string_pretty(&toml::Value::try_from(&full).unwrap()).unwrap();
        assert!(text.contains("enabled = true"), "{text}");
        assert!(text.contains("ram_gb = 24.0"), "{text}");
        assert!(text.contains("nvme_gb = 200.0"), "{text}");
        assert!(text.contains("nvme_path"), "{text}");

        // RAM only: the disk keys stay out of the file entirely rather than
        // appearing as zeroes a reader would have to interpret
        let ram_only = KvOffloadSpec {
            enabled: true,
            ram_gb: 8.0,
            ..Default::default()
        };
        let text = toml::to_string_pretty(&toml::Value::try_from(&ram_only).unwrap()).unwrap();
        assert!(text.contains("ram_gb = 8.0"), "{text}");
        assert!(
            !text.contains("nvme_gb"),
            "an unset disk budget is absent, not 0: {text}"
        );
        assert!(!text.contains("nvme_path"), "{text}");

        // a disk budget with no folder is complete on its own - the runner
        // defaults the location, so this is the commonest shape and must not
        // be mistaken for half a tier
        let no_path = KvOffloadSpec {
            enabled: true,
            ram_gb: 8.0,
            nvme_gb: 64.0,
            nvme_path: None,
        };
        let text = toml::to_string_pretty(&toml::Value::try_from(&no_path).unwrap()).unwrap();
        assert!(
            text.contains("nvme_gb = 64.0"),
            "the budget survives on its own: {text}"
        );
        assert!(
            !text.contains("nvme_path"),
            "no folder means no key: {text}"
        );

        // and it reads back out of a hand-written file
        let file = "[kv_offload]\nenabled = true\nram_gb = 12.5\nnvme_gb = 64.0\n\
                    nvme_path = \"/var/cache/paddock\"\n";
        let v: toml::Value = toml::from_str(file).unwrap();
        let parsed: KvOffloadSpec = v.get("kv_offload").cloned().unwrap().try_into().unwrap();
        assert!(parsed.enabled);
        assert_eq!(parsed.ram_gb, 12.5);
        assert_eq!(parsed.nvme_gb, 64.0);
        assert_eq!(parsed.nvme_path.as_deref(), Some("/var/cache/paddock"));
    }

    /// The backfill a legacy endpoint gets on its next start: additive only,
    /// appended after an array-of-tables (where a bare key could not go), and
    /// never written twice.
    #[test]
    fn stamping_adds_the_block_once_and_changes_nothing_else() {
        let dir = std::env::temp_dir().join(format!("pd-stamp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("11540.toml");
        let before = "\
# a note the operator wrote
model = 'E:\\models\\Tiny-Q8_0.gguf'
max_ctx = 4096

[[mcp_servers]]
server_label = \"tic\"
";
        std::fs::write(&path, before).unwrap();
        let spec = SpawnSpec {
            model: "tiny".into(),
            artifact: Some("q8".into()),
            ..Default::default()
        };
        Supervisor::stamp_catalog_identity(&path, &spec);

        let after = std::fs::read_to_string(&path).unwrap();
        let v: toml::Value = toml::from_str(&after).unwrap();
        assert_eq!(v["catalog"]["model"].as_str(), Some("tiny"));
        assert_eq!(v["catalog"]["artifact"].as_str(), Some("q8"));
        // everything the operator had is untouched, comment included
        assert_eq!(v["model"].as_str(), Some(r"E:\models\Tiny-Q8_0.gguf"));
        assert_eq!(v["max_ctx"].as_integer(), Some(4096));
        assert_eq!(v["mcp_servers"][0]["server_label"].as_str(), Some("tic"));
        assert!(after.contains("# a note the operator wrote"));

        // a second start must not touch it, and a path-shaped model has no
        // identity to record
        Supervisor::stamp_catalog_identity(&path, &spec);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), after);
        std::fs::write(&path, before).unwrap();
        Supervisor::stamp_catalog_identity(
            &path,
            &SpawnSpec {
                model: r"E:\models\Tiny-Q8_0.gguf".into(),
                ..Default::default()
            },
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The CLI's `switch` changes one field and resends the rest, which only
    /// works if a spec survives serde in both directions.
    ///
    /// It used to send just model/max_ctx/max_batch, and the switch route reads
    /// an absent OWNED key as "cleared" - so the verb silently stripped an
    /// endpoint's kv_cache_dtype, spec policy, MCP connectors and web-search
    /// settings for the crime of not mentioning them. The verb now rebuilds its
    /// request from `GET /api/servers/{port}/file`'s `spec`, so anything lost
    /// in this round trip is lost on every swap.
    #[test]
    fn a_spec_round_trips_so_a_swap_can_keep_what_it_did_not_mention() {
        let original = SpawnSpec {
            model: "tiny".into(),
            host: Some(std::net::Ipv4Addr::LOCALHOST.into()),
            max_ctx: Some(8192),
            max_batch: Some(4),
            kv_cache_dtype: Some("f16".into()),
            spec_policy: Some("off".into()),
            api_key: Some("pd-keepme".into()),
            web_search_provider: Some("brave".into()),
            mcp_servers: vec![serde_json::json!({ "server_label": "tic" })],
            ..Default::default()
        };
        let json = serde_json::to_value(&original).expect("a spec must serialize");
        // The wire name the switch route reads, not the Rust field name.
        assert_eq!(
            json["spec"].as_str(),
            Some("off"),
            "spec_policy must serialize as `spec`"
        );

        let back: SpawnSpec = serde_json::from_value(json).expect("and deserialize");
        assert_eq!(back.host, Some(std::net::Ipv4Addr::LOCALHOST.into()));
        // the fields the old verb dropped
        assert_eq!(back.kv_cache_dtype.as_deref(), Some("f16"));
        assert_eq!(back.spec_policy.as_deref(), Some("off"));
        assert_eq!(back.api_key.as_deref(), Some("pd-keepme"));
        assert_eq!(back.web_search_provider.as_deref(), Some("brave"));
        assert_eq!(back.mcp_servers.len(), 1, "connectors must survive a swap");
        // and the envelope, which the verb may legitimately override
        assert_eq!(back.max_ctx, Some(8192));
        assert_eq!(back.max_batch, Some(4));
    }

    /// The upgrade path every existing endpoint takes: a file written before
    /// the `[catalog]` block existed has none, and the first save has to add one.
    ///
    /// This is the exact shape `merge_owned_keys` warns about - inserting a new
    /// item into a document that already ends in an array-of-tables, where TOML
    /// would read a naively-appended key as a member of that last table. A
    /// config with MCP servers attached is not an edge case, so the merge is
    /// pinned here rather than trusted.
    #[test]
    fn merge_adds_a_catalog_block_to_a_file_that_ends_in_mcp_servers() {
        let current = "\
model = 'E:\\models\\Tiny-Q8_0.gguf'
max_ctx = 4096

[[mcp_servers]]
server_label = \"tic\"
server_url = \"https://mcp.tic.io\"
";
        // the render re-emits mcp_servers because the spec still has them -
        // absent in the render would mean the operator turned them off
        let rendered = "\
model = 'E:\\models\\Tiny-Q8_0.gguf'
max_ctx = 8192

[catalog]
model = \"tiny\"
artifact = \"q8\"

[[mcp_servers]]
server_label = \"tic\"
server_url = \"https://mcp.tic.io\"
";
        let out = merge_owned_keys(current, rendered).unwrap();
        let v: toml::Value = toml::from_str(&out).unwrap();
        assert_eq!(v["catalog"]["model"].as_str(), Some("tiny"));
        assert_eq!(v["catalog"]["artifact"].as_str(), Some("q8"));
        assert_eq!(v["max_ctx"].as_integer(), Some(8192));
        // the block landed as its own table, not swallowed into mcp_servers
        assert_eq!(v["mcp_servers"].as_array().map(Vec::len), Some(1));
        assert_eq!(v["mcp_servers"][0]["server_label"].as_str(), Some("tic"));
        assert!(
            v["mcp_servers"][0].get("model").is_none(),
            "catalog keys leaked into the MCP entry"
        );
    }

    /// Absent-in-render means DELETE for every owned key, and `catalog` is no
    /// exception: point an endpoint at a GGUF the catalog does not know and its
    /// stale identity has to go, or the editor keeps offering a model that is
    /// no longer being served.
    #[test]
    fn merge_clears_a_catalog_block_the_render_dropped() {
        let current =
            "model = \"/models/old.gguf\"\n\n[catalog]\nmodel = \"tiny\"\nartifact = \"q8\"\n";
        let rendered = "model = \"/models/imported.gguf\"\n";
        let out = merge_owned_keys(current, rendered).unwrap();
        let v: toml::Value = toml::from_str(&out).unwrap();
        assert_eq!(v["model"].as_str(), Some("/models/imported.gguf"));
        assert!(
            v.get("catalog").is_none(),
            "the catalog block should have been cleared"
        );
    }

    #[test]
    fn merge_keeps_foreign_keys_and_clears_owned_ones() {
        let current = "\
# my own note
model = \"/models/old.gguf\"
max_ctx = 4096
mmproj = \"/models/tower.gguf\"
log_file = \"/tmp/mine.log\"
";
        // vision turned off (no mmproj), context raised, a flag we do not own
        let rendered = "model = \"/models/new.gguf\"\nmax_ctx = 8192\n";
        let out = merge_owned_keys(current, rendered).unwrap();
        let v: toml::Value = toml::from_str(&out).unwrap();
        assert_eq!(v["model"].as_str(), Some("/models/new.gguf"));
        assert_eq!(v["max_ctx"].as_integer(), Some(8192));
        // absent in the render = off, not "leave the old one"
        assert!(v.get("mmproj").is_none(), "mmproj should have been cleared");
        // never ours, never touched
        assert_eq!(v["log_file"].as_str(), Some("/tmp/mine.log"));
        assert!(
            out.contains("# my own note"),
            "the user's comment should survive"
        );
    }

    /// The positional trap: a new scalar appended to a document that ends in an
    /// array-of-tables reads as a member of the last table. The merge must notice
    /// and fall back rather than hand back text whose MEANING changed.
    #[test]
    fn merge_falls_back_rather_than_mangle_a_trailing_table() {
        let current = "model = \"/m.gguf\"\n\n[[mcp_servers]]\nserver_label = \"github\"\n";
        let rendered =
            "model = \"/m.gguf\"\nmax_ctx = 8192\n\n[[mcp_servers]]\nserver_label = \"github\"\n";
        let out = merge_owned_keys(current, rendered).unwrap();
        let v: toml::Value = toml::from_str(&out).unwrap();
        // max_ctx is a ROOT key, not a field of the mcp_servers entry
        assert_eq!(v["max_ctx"].as_integer(), Some(8192));
        assert_eq!(v["mcp_servers"].as_array().map(Vec::len), Some(1));
        assert!(v["mcp_servers"][0].get("max_ctx").is_none());
    }
}
