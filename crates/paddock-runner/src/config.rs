//! Server configuration: `paddock.toml` + `PADDOCK_*` env + CLI flags.
//!
//! Precedence: CLI args > environment variables > config file > built-in
//! defaults. Deliberately a config FILE, not env-vars-only - Ollama's env-only
//! config is a documented complaint class (#11076). The layering and the CLI
//! surface live in `startup.rs`; this module owns the struct, the TOML load,
//! and the env overlay.

use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;

use serde::Deserialize;

/// KV offload tiering: budgets only - everything else
/// is elected (the no-tuning rule). `ram_gb` > 0 with `enabled` arms the RAM
/// tier; `nvme_gb`/`nvme_path` arm the NVMe tier.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvOffload {
    pub enabled: bool,
    pub ram_gb: f64,
    pub nvme_gb: f64,
    pub nvme_path: Option<PathBuf>,
}

/// `[moe_offload]` - MoE expert offload: the routed-expert planes of a MoE
/// model live in page-locked host RAM the GPU reads over PCIe, with a VRAM
/// slot cache of the hot experts sized from what the KV plan leaves. Lets a
/// model whose experts do not fit the card serve at cache-hit speed; only
/// k-quant expert seats offload today. `vram_gb` caps the slot cache (unset =
/// auto: everything the plan leaves). Lower `max_ctx` / `max_batch` to give
/// the cache more room.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MoeOffload {
    pub enabled: bool,
    pub vram_gb: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// All interfaces by default: the endpoint exists for other
    /// machines. The auth policy is what makes that safe - a non-loopback
    /// bind with no key auto-generates and requires one, and loopback peers
    /// are exempt (auth_mw) unless a proxy stands in front (`trusted_proxy`,
    /// or a forwarding header on the request). 175k publicly exposed KEYLESS
    /// Ollama servers is the cautionary tale we answer with keys, not with
    /// hiding.
    pub host: IpAddr,
    pub port: u16,
    /// Directories scanned for GGUF files.
    pub model_dirs: Vec<PathBuf>,
    /// GGUF to load and serve at startup (None = serve /v1/models only).
    pub model: Option<PathBuf>,
    /// Which catalog entry `model` is - provenance, written by the manager.
    ///
    /// The runner does not act on it, and that is the design: `model` stays a
    /// PATH so this file keeps meaning exactly one set of bytes and keeps
    /// running standalone on a box with no catalog at all. The block exists
    /// because the manager needs to know which model an endpoint serves while
    /// it is STOPPED, and its previous answer - match the filename against the
    /// catalog - lost the model the moment anyone renamed, copied, or imported
    /// a file in place.
    ///
    /// It is declared here because it has to be: `deny_unknown_fields` turns
    /// any key the runner doesn't know into a hard spawn failure, so a manager
    /// that writes this needs a runner that accepts it in the same release.
    pub catalog: Option<CatalogRef>,
    /// Serving device - "cuda" is the only backend (GPU-only law; the CPU
    /// reference arm is gone). Kept a string for the ROCm /
    /// Metal / Vulkan packs to come.
    pub device: String,
    /// Which GPU to serve on: a CUDA device ordinal ("1") or a device UUID
    /// ("GPU-d56cd6c9-...", the nvidia-smi spelling - order-proof; a unique
    /// prefix is enough). Resolved against the driver at startup - a real
    /// config field, not a CUDA_VISIBLE_DEVICES trick, so a manager-started,
    /// service-started, and hand-started server all pin identically. None =
    /// device 0.
    pub gpu: Option<String>,
    /// Kernel pack path (required for device = "cuda").
    pub kernel_pack: Option<PathBuf>,
    /// Max context length for the served model's KV cache.
    pub max_ctx: usize,
    /// Continuous-batching width - concurrent sequences the engine batches per
    /// step (1 = serial loop). Lower it on a tight card to trade concurrency for
    /// headroom: each extra slot costs KV + hybrid state, and batch>1 also brings
    /// up the paged pool + prefix cache.
    pub max_batch: usize,
    /// Vision tower GGUF (mmproj) for multimodal models; enables image input.
    pub mmproj: Option<PathBuf>,
    /// MTP drafter GGUF (separate-model speculative drafter, e.g. gemma4's
    /// mtp-*.gguf); enables serving spec rounds with model drafts.
    pub mtp: Option<PathBuf>,
    /// Official-FP8 (or bf16) safetensors snapshot dir to source the e4m3
    /// serving planes from, skipping the Q8_0 middle hop (opt-in - the
    /// measured default keeps the bf16-derived planes). Threaded to the
    /// engine as an explicit load option; the engine never reads env.
    pub fp8_native: Option<PathBuf>,
    /// Hard VRAM budget for this server, in MiB (nvidia-smi units). The
    /// engine sizes every pool/cache inside it and refuses a load that can't
    /// fit it - so co-resident servers each granted a slice can never sum
    /// past the card (overcommit on WDDM freezes the box). The manager computes and
    /// writes this at admission; hand-run servers may set or drop it freely.
    /// None = size against free VRAM at load (single-server behavior).
    pub vram_budget: Option<u64>,
    /// See [`KvOffload`].
    pub kv_offload: KvOffload,
    /// See [`MoeOffload`].
    pub moe_offload: MoeOffload,
    /// Override the fixed 3 GiB `graph/prefill scratch` KV-plan reserve, in
    /// MiB. The default is sized for 16-48 GB cards; on 8 GB cards it alone
    /// exhausts the grant and starves both the KV pool and the
    /// `[moe_offload]` slot cache (which is sized from the plan's leftovers).
    /// None = keep the 3 GiB default.
    pub graph_scratch_mib: Option<u64>,
    /// Default max output tokens per reply when a request doesn't specify.
    pub max_tokens: Option<usize>,
    /// API key for Bearer auth. Empty + loopback bind = no auth; empty +
    /// non-loopback = auto-generate and require (see startup).
    pub api_key: Option<String>,
    /// Explicit auth opt-out for network binds (--no-auth); see startup.rs.
    pub no_auth: bool,
    /// Web-search provider for the server-executed `web_search` tool: one of
    /// `paddock_websearch::Provider` ("exa", "tavily", "firecrawl", "brave",
    /// "perplexity"). The runner is stateless, so this is declared config (the
    /// manager passes it at spawn from its stored settings); None = requests
    /// declaring a web_search tool get a clear 400.
    pub web_search_provider: Option<String>,
    /// API key for the web-search provider.
    pub web_search_api_key: Option<String>,
    /// Soft cap on pages rendered per PDF. Pages past this are dropped and the
    /// truncation surfaced (never silent). Also bounded by the context budget
    /// at request time.
    pub pdf_max_pages: usize,
    /// Target long-edge (px) for each rendered PDF page; per-page DPI is derived
    /// from it (capped at 300). 1568 matches the Qwen vision sweet spot.
    pub pdf_page_long_edge: u32,

    // --- Abuse controls for intentionally-exposed (public/demo) instances.
    // All off by default; a private Paddock is unaffected. See `ratelimit.rs`.
    /// Hard ceiling on generated tokens per reply - clamps `max_output_tokens`
    /// regardless of what a request asks (`None` = no clamp). Set on an exposed
    /// instance so a hand-crafted request can't demand a huge, costly
    /// generation. Independent of `max_ctx` (the conversation window).
    pub max_output_ceiling: Option<usize>,
    /// Per-client generation-request rate limit, requests/minute (`None` = off).
    pub ratelimit_per_minute: Option<u32>,
    /// Per-client generation-request quota, requests/day (`None` = off).
    pub ratelimit_per_day: Option<u32>,
    /// A reverse proxy stands in front of this runner. Two things follow: the
    /// rate limiter keys clients on the proxy's `X-Real-IP` (it overwrites any
    /// client value; `X-Forwarded-For` is never trusted, a client can prepend
    /// to it), and the API key is required from loopback peers too - behind a
    /// proxy on the same host EVERY caller arrives from 127.0.0.1, so the
    /// loopback exemption would let the whole internet in. Off for a direct
    /// bind. nginx adds no forwarding headers unless told to, so set this
    /// rather than relying on the runner noticing the proxy.
    pub trusted_proxy: bool,

    // --- Server-side sampling OVERRIDES (llama.cpp-style flags, and
    // Option-typed deliberately). Used only when a request omits the
    // field - request always wins. `None` is not "the OpenAI value", it is
    // "the operator said nothing", which is what lets the served checkpoint's
    // own published profile (paddock_models::sampling) fill the slot instead.
    // Setting one of these pins that knob for every request that omits it and
    // takes the model's election out of the loop, which is why the resolved
    // set and its provenance are logged at startup and served on /api/server.
    /// Pin the default temperature (`--temp`).
    pub temp: Option<f32>,
    /// Pin the default top-k, 0 = off (`--top-k`).
    pub top_k: Option<usize>,
    /// Pin the default nucleus top-p (`--top-p`).
    pub top_p: Option<f32>,
    /// Pin the default min-p (`--min-p`).
    pub min_p: Option<f32>,
    /// Pin the default repetition penalty, 1.0 = off (`--repeat-penalty`).
    pub repeat_penalty: Option<f32>,
    /// Default repetition-penalty window (`--repeat-last-n`).
    pub repeat_last_n: usize,
    /// Default RNG seed when a request omits `seed` (`--seed`). None = a
    /// time-derived seed per request (the OpenAI semantics).
    pub seed: Option<u64>,
    /// KV cache element type: "auto" (per-family default: gemma4 fp8-e4m3
    /// when pooled, others f16), "f16", or "fp8_e4m3" (`--kv-cache-dtype`).
    pub kv_cache_dtype: String,
    /// Serve the model under this id instead of the file-derived one
    /// (`--served-model-name`, the vLLM flag).
    pub served_model_name: Option<String>,
    /// Disable speculative decode / MTP (`--no-spec`) - first-class switch for
    /// the engine's PADDOCK_NO_SPEC kill mechanism. Kept for configs (and
    /// muscle memory) that predate `spec`; it always wins when set.
    pub no_spec: bool,
    /// Speculation policy - the one key that answers "is spec on for this
    /// endpoint", whatever shape the model's drafter takes:
    ///
    /// Canonical spellings (what `SpecPolicy::Display` writes back, so these
    /// are what a round trip produces and what the Studio offers):
    ///
    ///   "off"      never speculate (no drafter loaded - the only setting that
    ///              gives the VRAM back). Also: false/no/none/0
    ///   "on"       the hand-tuned batch->K ladder (today's default, so adding
    ///              this key changed no existing endpoint). Also: true/yes/
    ///              ladder/legacy
    ///   "adaptive" closed-loop: the engine re-picks the draft length every
    ///              round to maximize goodput, including 0 when speculation is
    ///              not paying at the current load (crate::spec_policy). Also:
    ///              auto
    ///   <1..16>    pin the draft length - bench/A-B/parity only
    ///
    /// The aliases all parse; this comment used to lead with two of them
    /// ("auto"/"ladder"), which read as the canonical pair and disagreed with
    /// what the runner then hands the engine through PADDOCK_SPEC.
    ///
    /// Deliberately one key rather than two: models carry MTP in-file
    /// (qwen3.5/3.6 `nextn`) or as a sideloaded drafter (`mtp`), and asking a
    /// user which kind they have before they can turn speculation on is a
    /// question only the loader can answer.
    pub spec: Option<String>,
    /// Disable the per-request event ring (`--no-events`, doc §8.7). On by
    /// default: metadata only, fixed RAM, zero disk.
    pub no_events: bool,
    /// Disable the Prometheus `/metrics` surface (`--no-metrics`).
    /// On by default and gated INDEPENDENTLY of the event ring - turning one
    /// off must not silently kill the other.
    pub no_metrics: bool,
    /// `/metrics` auth override (`metrics_auth`). Unset = the API
    /// key is required for non-loopback callers only, loopback scrapes stay
    /// open (the vLLM/SGLang posture, and the only one a headerless
    /// scraper can work with). true = key required from everyone; false =
    /// open to everyone who can reach the port.
    pub metrics_auth: Option<bool>,
    /// `--vad-gate`: skip transcription windows with no speech in them before
    /// the encoder runs. Off by default, and deliberately so -
    /// every reference system ships this opt-in too (faster-whisper's
    /// `vad_filter`, whisper.cpp's `--vad`), because it changes what a
    /// transcript CONTAINS. A board leg measured with it on is a different
    /// measurement and has to say so.
    pub vad_gate: bool,
    /// Headers whose value is captured as the event record's session id
    /// (`--session-headers`, comma-separated; matched case-insensitively).
    pub session_headers: Vec<String>,

    // --- Request filters + admission (doc §13, the llama-swap lessons).
    /// Alternative model ids this endpoint answers to (llama-swap `aliases` -
    /// e.g. impersonate "gpt-4o-mini" for tools that hardcode cloud ids).
    pub aliases: Vec<String>,
    /// Parameter variants (llama-swap `setParamsByID`): `[variants.high]`
    /// tables of request fields, addressable as `<model>:high` in requests
    /// and listed in /v1/models. One resident model, several behaviours.
    pub variants: std::collections::HashMap<String, serde_json::Value>,
    /// Client-sent request fields to remove (llama-swap `stripParams`) -
    /// server-side enforcement, e.g. pin sampling by stripping overrides.
    pub strip_params: Vec<String>,
    /// Request fields forced to these values regardless of what clients send
    /// (llama-swap `setParams`): a `[force_params]` table.
    pub force_params: std::collections::HashMap<String, serde_json::Value>,
    /// Explicit admission cap: max in-flight inference requests before an
    /// Overloaded refusal (llama-swap `concurrencyLimit`). Queue depth -
    /// distinct from `max_batch`, the compute width. None = uncapped.
    pub concurrency_limit: Option<usize>,
    /// Registered MCP servers this runner resolves bare `server_label`s (and
    /// Anthropic name-only `mcp_servers` entries) against - expanded JSON:
    /// [{server_label, server_url, headers?, require_approval?}]. Set via
    /// PADDOCK_MCP_SERVERS; the manager injects its stored registry here at
    /// spawn so every model serves the same tools with no runtime manager
    /// dependency. Empty = only inline server_url tools work.
    pub mcp_servers: Vec<serde_json::Value>,

    /// `[forensics]` - the image/document forensic preprocessing gate. Off by
    /// default; when enabled, image attachments are run through the forensic
    /// analyzers (paddock-forensics) on their ORIGINAL bytes and the findings
    /// are injected into the model's context. GPU-first with a CPU fallback.
    #[serde(default)]
    pub forensics: ForensicsConfig,
    /// Append logs to this file instead of stdout. Service mode (no console)
    /// defaults it to `<config>.log` beside the config file.
    pub log_file: Option<PathBuf>,
}

/// The `[catalog]` block: the catalog id + weights-artifact id behind the
/// `model` path. Deliberately not fed to `served_model_name` - that would flip
/// /v1/models from the file-derived id ("Qwen3.5-9B-Q8_0") to the catalog id
/// ("qwen3.5-9b") for every manager-started endpoint, which is a wire change
/// that would break anything pinning a model name (every bench scenario, for a
/// start). Worth doing deliberately one day; not as a side effect of provenance.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogRef {
    /// Catalog model id, e.g. "qwen3.5-9b".
    pub model: String,
    /// Weights-artifact id within that model, e.g. "q8". Optional: an endpoint
    /// can be known to be a model without the artifact being pinned down.
    #[serde(default)]
    pub artifact: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // All interfaces by default: agents on other
            // machines are the tier-1 workload. The API key + the loopback
            // exemption in auth_mw are what make this safe - network callers
            // authenticate, local ones don't have to.
            host: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port: 11540,
            model_dirs: default_model_dirs(),
            model: None,
            catalog: None,
            // "cuda": GPU-only is law - the "cpu" default was a
            // relic of the abolished CPU reference models. A config without a
            // device now means the GPU path, and a missing kernel_pack fails
            // loudly instead of silently degrading to a CPU crawl.
            device: "cuda".to_owned(),
            gpu: None,
            kernel_pack: None,
            max_ctx: 4096,
            // 32: the old 16 silently CAPPED 32-way agentic serving - 32
            // concurrent streams degenerated to queueing and the aggregate
            // flatlined at the 8-way level. Matching the width to the offered
            // concurrency more than doubled it. Admission stays KV-pool-gated,
            // so idle slots cost only scratch - the paged watermark keeps
            // small-VRAM cards safe. vLLM's own default is far higher
            // (max_num_seqs=256).
            max_batch: 32,
            mmproj: None,
            mtp: None,
            fp8_native: None,
            vram_budget: None,
            kv_offload: KvOffload::default(),
            moe_offload: MoeOffload::default(),
            graph_scratch_mib: None,
            max_tokens: None,
            api_key: None,
            no_auth: false,
            web_search_provider: None,
            web_search_api_key: None,
            pdf_max_pages: 20,
            pdf_page_long_edge: 1568,
            max_output_ceiling: None,
            ratelimit_per_minute: None,
            ratelimit_per_day: None,
            trusted_proxy: false,
            // None = unpinned: the served model's own published sampling
            // profile decides, and only where it publishes nothing do the
            // OpenAI wire values stand.
            temp: None,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: 64,
            seed: None,
            kv_cache_dtype: "auto".to_owned(),
            served_model_name: None,
            no_spec: false,
            spec: None,
            no_events: false,
            no_metrics: false,
            metrics_auth: None,
            vad_gate: false,
            session_headers: crate::events::default_session_headers(),
            aliases: Vec::new(),
            variants: std::collections::HashMap::new(),
            strip_params: Vec::new(),
            force_params: std::collections::HashMap::new(),
            concurrency_limit: None,
            mcp_servers: Vec::new(),
            log_file: None,
            forensics: ForensicsConfig::default(),
        }
    }
}

/// When the forensic preprocessing lane runs automatically over attachments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ForensicsAuto {
    /// Never run automatically (the tool surface, when enabled, still exposes it).
    #[default]
    Off,
    /// Run over image attachments.
    Images,
    /// Run over images and (once the PDF lane lands) documents. Today == Images.
    All,
}

/// The `[forensics]` block. All-off by default - enabling it is a deliberate
/// per-deployment choice, like `web_search_provider`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForensicsConfig {
    /// Master switch. When false the runtime is never built and there is zero
    /// cost on the request path.
    pub enabled: bool,
    /// Automatic preprocessing mode (see [`ForensicsAuto`]).
    pub auto: ForensicsAuto,
    /// Expose the on-demand `analyze_document_forensics` tool to the model.
    /// (Tool surface lands in a later wave; the flag is accepted now so config
    /// does not have to change when it does.)
    pub tool: bool,
    /// CUDA device ordinal for the forensic GPU context. `None` -> device 0.
    /// Only consulted in a `forensics-cuda` build; a CPU build ignores it.
    pub device: Option<usize>,
}

impl Default for ForensicsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auto: ForensicsAuto::Off,
            tool: false,
            device: None,
        }
    }
}

fn default_model_dirs() -> Vec<PathBuf> {
    // <data root>/models - user-visible, plain files, per the storage
    // principle; the root follows the distribution mode
    vec![paddock_admin::data_root().join("models")]
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not read config file {0}: {1}")]
    Read(PathBuf, std::io::Error),
    #[error("config file {0} is invalid: {1}")]
    Parse(PathBuf, toml::de::Error),
    #[error("environment override {name} is invalid: {value:?}")]
    BadEnv { name: &'static str, value: String },
    #[error("flag {name} has an invalid value: {value:?}")]
    BadArg { name: &'static str, value: String },
    /// A flag for a feature paddock deliberately doesn't have - the message
    /// is the complete explanation (honest rejection, not "unknown flag").
    #[error("{0}")]
    Unsupported(&'static str),
}

impl Config {
    /// The resolved web-search provider config, if a usable provider + key are
    /// declared. Unknown provider names are a loud None with a warning - a
    /// typo'd provider must not silently disable a declared capability.
    pub fn web_search(&self) -> Option<crate::websearch::SearchConfig> {
        crate::websearch::SearchConfig::from_fields(
            self.web_search_provider.as_deref(),
            self.web_search_api_key.as_deref(),
        )
    }

    /// Load a TOML config file onto the built-in defaults. A malformed file is
    /// a hard error, never a silent fallback to defaults.
    pub fn from_toml(path: &PathBuf) -> Result<Self, ConfigError> {
        let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::Read(path.clone(), e))?;
        toml::from_str(&raw).map_err(|e| ConfigError::Parse(path.clone(), e))
    }

    /// Overlay `PADDOCK_*` environment variables onto this config. Every field
    /// is overridable; an unparseable value is a loud error, not a panic.
    ///
    /// Every name read here must also appear in [`ENV_SURFACE`] - in a
    /// hardened build the seal deletes anything that does not, so an addition
    /// made in one place and not the other is a setting that silently stops
    /// working in the shipped binary and nowhere else. Keeping the two in the
    /// same file is the whole defence.
    pub fn merge_env(&mut self) -> Result<(), ConfigError> {
        if let Some(v) = env_str("PADDOCK_HOST") {
            self.host = v.parse().map_err(|_| bad_env("PADDOCK_HOST", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_PORT") {
            self.port = v.parse().map_err(|_| bad_env("PADDOCK_PORT", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_MODEL") {
            self.model = Some(PathBuf::from(v));
        }
        if let Some(v) = env_str("PADDOCK_MODEL_DIRS") {
            self.model_dirs = v.split(',').map(|s| PathBuf::from(s.trim())).collect();
        }
        if let Some(v) = env_str("PADDOCK_DEVICE") {
            self.device = v;
        }
        if let Some(v) = env_str("PADDOCK_GRAPH_SCRATCH_MIB") {
            self.graph_scratch_mib = Some(
                v.parse()
                    .map_err(|_| bad_env("PADDOCK_GRAPH_SCRATCH_MIB", &v))?,
            );
        }
        if let Some(v) = env_str("PADDOCK_GPU") {
            self.gpu = Some(v);
        }
        if let Some(v) = env_str("PADDOCK_KERNEL_PACK") {
            self.kernel_pack = Some(PathBuf::from(v));
        }
        if let Some(v) = env_str("PADDOCK_MAX_CTX") {
            self.max_ctx = v.parse().map_err(|_| bad_env("PADDOCK_MAX_CTX", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_MAX_BATCH") {
            self.max_batch = v.parse().map_err(|_| bad_env("PADDOCK_MAX_BATCH", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_MMPROJ") {
            self.mmproj = Some(PathBuf::from(v));
        }
        if let Some(v) = env_str("PADDOCK_MTP") {
            self.mtp = Some(PathBuf::from(v));
        }
        if let Some(v) = env_str("PADDOCK_FP8_NATIVE") {
            self.fp8_native = Some(PathBuf::from(v));
        }
        // Alias for NVFP4 (llm-compressor) snapshots - same native-checkpoint
        // plane-source field, named for what the dir actually holds so serve
        // command lines read honestly. The loader detects the dialect from
        // the tensors, not the env name.
        if let Some(v) = env_str("PADDOCK_NVFP4_NATIVE") {
            self.fp8_native = Some(PathBuf::from(v));
        }
        if let Some(v) = env_str("PADDOCK_VRAM_BUDGET") {
            self.vram_budget = Some(v.parse().map_err(|_| bad_env("PADDOCK_VRAM_BUDGET", &v))?);
        }
        if let Some(v) = env_str("PADDOCK_MAX_OUTPUT_TOKENS") {
            self.max_tokens = Some(
                v.parse()
                    .map_err(|_| bad_env("PADDOCK_MAX_OUTPUT_TOKENS", &v))?,
            );
        }
        if let Some(v) = env_str("PADDOCK_MAX_OUTPUT_CEILING") {
            self.max_output_ceiling = Some(
                v.parse()
                    .map_err(|_| bad_env("PADDOCK_MAX_OUTPUT_CEILING", &v))?,
            );
        }
        if let Some(v) = env_str("PADDOCK_RATELIMIT_PER_MINUTE") {
            self.ratelimit_per_minute = Some(
                v.parse()
                    .map_err(|_| bad_env("PADDOCK_RATELIMIT_PER_MINUTE", &v))?,
            );
        }
        if let Some(v) = env_str("PADDOCK_RATELIMIT_PER_DAY") {
            self.ratelimit_per_day = Some(
                v.parse()
                    .map_err(|_| bad_env("PADDOCK_RATELIMIT_PER_DAY", &v))?,
            );
        }
        if let Some(v) = env_str("PADDOCK_TRUSTED_PROXY") {
            self.trusted_proxy = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }
        if let Some(v) = env_str("PADDOCK_TEMP") {
            self.temp = Some(v.parse().map_err(|_| bad_env("PADDOCK_TEMP", &v))?);
        }
        if let Some(v) = env_str("PADDOCK_TOP_K") {
            self.top_k = Some(v.parse().map_err(|_| bad_env("PADDOCK_TOP_K", &v))?);
        }
        if let Some(v) = env_str("PADDOCK_TOP_P") {
            self.top_p = Some(v.parse().map_err(|_| bad_env("PADDOCK_TOP_P", &v))?);
        }
        if let Some(v) = env_str("PADDOCK_MIN_P") {
            self.min_p = Some(v.parse().map_err(|_| bad_env("PADDOCK_MIN_P", &v))?);
        }
        if let Some(v) = env_str("PADDOCK_REPEAT_PENALTY") {
            self.repeat_penalty = Some(
                v.parse()
                    .map_err(|_| bad_env("PADDOCK_REPEAT_PENALTY", &v))?,
            );
        }
        if let Some(v) = env_str("PADDOCK_REPEAT_LAST_N") {
            self.repeat_last_n = v
                .parse()
                .map_err(|_| bad_env("PADDOCK_REPEAT_LAST_N", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_SEED") {
            self.seed = Some(v.parse().map_err(|_| bad_env("PADDOCK_SEED", &v))?);
        }
        if let Some(v) = env_str("PADDOCK_KV_CACHE_DTYPE") {
            self.kv_cache_dtype = v;
        }
        if let Some(v) = env_str("PADDOCK_SERVED_MODEL_NAME") {
            self.served_model_name = Some(v);
        }
        if env_str("PADDOCK_NO_SPEC").is_some() {
            self.no_spec = true;
        }
        if let Some(v) = env_str("PADDOCK_SPEC") {
            self.spec = Some(v);
        }
        if env_str("PADDOCK_NO_EVENTS").is_some() {
            self.no_events = true;
        }
        if env_str("PADDOCK_NO_METRICS").is_some() {
            self.no_metrics = true;
        }
        if let Some(v) = env_str("PADDOCK_METRICS_AUTH") {
            self.metrics_auth = Some(matches!(v.as_str(), "1" | "true" | "on"));
        }
        if env_str("PADDOCK_VAD_GATE").is_some() {
            self.vad_gate = true;
        }
        if let Some(v) = env_str("PADDOCK_SESSION_HEADERS") {
            self.session_headers = v
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Some(v) = env_str("PADDOCK_API_KEY") {
            self.api_key = Some(v);
        }
        if let Some(v) = env_str("PADDOCK_NO_AUTH") {
            self.no_auth = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }
        if let Some(v) = env_str("PADDOCK_WEB_SEARCH_PROVIDER") {
            self.web_search_provider = Some(v);
        }
        if let Some(v) = env_str("PADDOCK_MCP_SERVERS") {
            self.mcp_servers = serde_json::from_str::<Vec<serde_json::Value>>(&v)
                .map_err(|_| bad_env("PADDOCK_MCP_SERVERS", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_LOG_FILE") {
            self.log_file = Some(PathBuf::from(v));
        }
        if let Some(v) = env_str("PADDOCK_WEB_SEARCH_API_KEY") {
            self.web_search_api_key = Some(v);
        }
        if let Some(v) = env_str("PADDOCK_PDF_MAX_PAGES") {
            self.pdf_max_pages = v
                .parse()
                .map_err(|_| bad_env("PADDOCK_PDF_MAX_PAGES", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_PDF_PAGE_LONG_EDGE") {
            self.pdf_page_long_edge = v
                .parse()
                .map_err(|_| bad_env("PADDOCK_PDF_PAGE_LONG_EDGE", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_ALIASES") {
            self.aliases = v
                .split(',')
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Some(v) = env_str("PADDOCK_VARIANTS") {
            // one JSON object: {"high": {"reasoning_effort": "high"}, ...}
            self.variants =
                serde_json::from_str(&v).map_err(|_| bad_env("PADDOCK_VARIANTS", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_STRIP_PARAMS") {
            self.strip_params = v
                .split(',')
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect();
        }
        if let Some(v) = env_str("PADDOCK_FORCE_PARAMS") {
            self.force_params =
                serde_json::from_str(&v).map_err(|_| bad_env("PADDOCK_FORCE_PARAMS", &v))?;
        }
        if let Some(v) = env_str("PADDOCK_CONCURRENCY_LIMIT") {
            self.concurrency_limit = Some(
                v.parse()
                    .map_err(|_| bad_env("PADDOCK_CONCURRENCY_LIMIT", &v))?,
            );
        }
        Ok(())
    }
}

/// The runner's whole `PADDOCK_*` environment surface: what [`Config::merge_env`]
/// reads, plus the escape hatches that live outside config.
///
/// A hardened build seals everything else away before the first read (see
/// `paddock_models::hardening`). That is not only about hiding names -
/// the CUDA pack reads its own election switches straight out of the
/// environment with `pd_env`, so this list is what decides whether a value a
/// user exported can still change which kernel runs. It cannot.
///
/// The escape hatches are here deliberately and they are not tuning: each one
/// says "do something we do not support" and names itself in the refusal text
/// it belongs to.
pub const ENV_SURFACE: &[&str] = &[
    // --- config surface (mirrors merge_env, same file, edit together) ---
    "PADDOCK_ALIASES",
    "PADDOCK_API_KEY",
    "PADDOCK_CONCURRENCY_LIMIT",
    "PADDOCK_DEVICE",
    "PADDOCK_FORCE_PARAMS",
    "PADDOCK_FP8_NATIVE",
    "PADDOCK_GPU",
    "PADDOCK_GRAPH_SCRATCH_MIB",
    "PADDOCK_HOST",
    "PADDOCK_KERNEL_PACK",
    "PADDOCK_KV_CACHE_DTYPE",
    "PADDOCK_LOG_FILE",
    "PADDOCK_MAX_BATCH",
    "PADDOCK_MAX_CTX",
    "PADDOCK_MAX_OUTPUT_CEILING",
    "PADDOCK_MAX_OUTPUT_TOKENS",
    "PADDOCK_MCP_SERVERS",
    "PADDOCK_METRICS_AUTH",
    "PADDOCK_MIN_P",
    "PADDOCK_MMPROJ",
    "PADDOCK_MODEL",
    "PADDOCK_MODEL_DIRS",
    "PADDOCK_MTP",
    "PADDOCK_NO_AUTH",
    "PADDOCK_NO_EVENTS",
    "PADDOCK_NO_METRICS",
    "PADDOCK_NO_SPEC",
    "PADDOCK_NVFP4_NATIVE",
    "PADDOCK_PDF_MAX_PAGES",
    "PADDOCK_PDF_PAGE_LONG_EDGE",
    "PADDOCK_PORT",
    "PADDOCK_RATELIMIT_PER_DAY",
    "PADDOCK_RATELIMIT_PER_MINUTE",
    "PADDOCK_REPEAT_LAST_N",
    "PADDOCK_REPEAT_PENALTY",
    "PADDOCK_SEED",
    "PADDOCK_SERVED_MODEL_NAME",
    "PADDOCK_SESSION_HEADERS",
    "PADDOCK_SPEC",
    "PADDOCK_STRIP_PARAMS",
    "PADDOCK_TEMP",
    "PADDOCK_TOP_K",
    "PADDOCK_TOP_P",
    "PADDOCK_TRUSTED_PROXY",
    "PADDOCK_VAD_GATE",
    "PADDOCK_VARIANTS",
    "PADDOCK_VRAM_BUDGET",
    "PADDOCK_WEB_SEARCH_API_KEY",
    "PADDOCK_WEB_SEARCH_PROVIDER",
    // --- outside merge_env ---
    // where the box's data lives (paddock_admin::data_root_resolved)
    "PADDOCK_DATA",
];

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

fn bad_env(name: &'static str, value: &str) -> ConfigError {
    ConfigError::BadEnv {
        name,
        value: value.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forensics_block_parses_and_defaults_off() {
        // absent block -> all off
        let c: Config = toml::from_str("").expect("empty config");
        assert!(!c.forensics.enabled);
        assert_eq!(c.forensics.auto, ForensicsAuto::Off);

        // explicit block
        let c: Config = toml::from_str(
            "[forensics]\nenabled = true\nauto = \"images\"\ntool = true\ndevice = 1\n",
        )
        .expect("parse [forensics]");
        assert!(c.forensics.enabled);
        assert_eq!(c.forensics.auto, ForensicsAuto::Images);
        assert!(c.forensics.tool);
        assert_eq!(c.forensics.device, Some(1));

        // unknown key inside the block is rejected (deny_unknown_fields)
        assert!(toml::from_str::<Config>("[forensics]\nnope = true\n").is_err());
    }

    /// The drift guard for the seal. Adding a name to `merge_env` and not to
    /// `ENV_SURFACE` gives you a setting that works on every dev box and is
    /// deleted before it is read in the shipped binary - a bug that only ever
    /// shows up in a customer's hands. Read our own source rather than trust a
    /// convention: the file is the only thing that cannot go stale.
    #[test]
    fn every_env_merge_env_reads_is_declared_in_the_surface() {
        let src = include_str!("config.rs");
        let mut missing = Vec::new();
        for (i, _) in src.match_indices("env_str(\"PADDOCK_") {
            let rest = &src[i + "env_str(\"".len()..];
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

    #[test]
    fn defaults_bind_all_interfaces() {
        // Bind * so other machines can call the API; the
        // key requirement for non-loopback peers (auth_mw) is the guard.
        let cfg = Config::default();
        assert!(cfg.host.is_unspecified(), "default must serve the network");
    }
    #[test]
    fn graph_scratch_mib_parses_and_defaults_to_none() {
        // absent -> None (the 3 GiB engine default stands)
        let c: Config = toml::from_str("").expect("empty config");
        assert!(c.graph_scratch_mib.is_none());
        // explicit value
        let c: Config = toml::from_str("graph_scratch_mib = 512").expect("parse");
        assert_eq!(c.graph_scratch_mib, Some(512));
    }

    #[test]
    fn unknown_keys_are_rejected_not_ignored() {
        // a typo'd key silently ignored = config that lies; deny_unknown_fields
        let err = toml::from_str::<Config>("prot = 1234").expect_err("must fail");
        assert!(err.to_string().contains("prot"));
    }

    /// The manager writes `[catalog]` into every config it renders.
    /// `deny_unknown_fields` means a runner that does not declare it refuses the
    /// file OUTRIGHT - the endpoint would not start at all - so this is the
    /// guard on a cross-crate contract with no compile-time link.
    #[test]
    fn the_catalog_provenance_block_parses() {
        let cfg: Config = toml::from_str(
            "model = 'E:\\models\\Tiny-Q8_0.gguf'\n\n[catalog]\nmodel = \"tiny\"\nartifact = \"q8\"\n",
        )
        .expect("a config the manager writes must load");
        let c = cfg.catalog.expect("the block");
        assert_eq!(c.model, "tiny");
        assert_eq!(c.artifact.as_deref(), Some("q8"));
        // artifact is optional - an endpoint can be known to be a model without
        // the weights choice being pinned down
        let cfg: Config = toml::from_str("[catalog]\nmodel = \"tiny\"\n").expect("id alone");
        assert_eq!(cfg.catalog.expect("the block").artifact, None);
        // and the block is closed too: a typo inside it is not silently kept
        toml::from_str::<Config>("[catalog]\nmodel = \"tiny\"\nartefact = \"q8\"\n")
            .expect_err("an unknown key inside the block must fail like any other");
    }

    #[test]
    fn bad_env_is_an_error_not_a_panic() {
        // SAFETY: single-threaded test; no other thread reads the env here.
        unsafe { std::env::set_var("PADDOCK_PORT", "not-a-port") };
        let mut cfg = Config::default();
        let err = cfg.merge_env().expect_err("must reject a non-numeric port");
        assert!(matches!(
            err,
            ConfigError::BadEnv {
                name: "PADDOCK_PORT",
                ..
            }
        ));
        unsafe { std::env::remove_var("PADDOCK_PORT") };
    }
}
