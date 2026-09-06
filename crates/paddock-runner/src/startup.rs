//! CLI surface, config layering, and the startup banner.
//!
//! Precedence: CLI args > `PADDOCK_*` env > `paddock.toml` > built-in defaults.
//! `paddock.toml` is auto-discovered in the working directory, or pointed at
//! with `--config`. `resolve` also records where each visible value came from
//! (env/arg) so the banner can show it honestly.

use std::io::IsTerminal;
use std::net::IpAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::config::{Config, ConfigError};

const COPYRIGHT: &str = "Copyright (c) 2026 Truespar. MIT OR Apache-2.0.";

/// `paddock-runner service ...` - run a model server under the OS's own
/// supervisor instead of (or alongside) the paddock manager. The config FILE
/// is the whole configuration either way.
#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Install/uninstall this model server as an OS service
    #[command(subcommand)]
    Service(ServiceAction),
}

#[derive(Subcommand, Debug)]
pub enum ServiceAction {
    /// Register an auto-start OS service for a config file (Windows service /
    /// systemd unit). Windows needs an elevated terminal.
    Install {
        /// The server's config file (servers/<port>.toml)
        #[arg(long, value_name = "PATH")]
        config: PathBuf,
        /// Service name (default: paddock-runner-<port>)
        #[arg(long, value_name = "NAME")]
        name: Option<String>,
    },
    /// Remove a previously installed service
    Uninstall {
        /// The config file the service was installed for
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        /// Service name, when it was installed with a custom one
        #[arg(long, value_name = "NAME")]
        name: Option<String>,
    },
    /// (internal) the entry point the Windows service manager invokes
    #[command(hide = true)]
    Run {
        #[arg(long, value_name = "PATH")]
        config: PathBuf,
    },
}

/// `paddock-runner` - the data-plane binary. Everything is also settable via
/// `paddock.toml` and `PADDOCK_*` env; flags win over both.
#[derive(Parser, Debug)]
#[command(
    name = "paddock-runner",
    // See the manager's Cli: the stamp, so a runner that was updated
    // separately from its manager says so.
    version = paddock_admin::version::LONG,
    about = "Paddock runner - one model, one port, the full OpenAI/Anthropic API (headless; `paddock` is the manager + Studio)"
)]
pub struct Cli {
    /// OS-service management: install/uninstall this model server as a
    /// Windows service or systemd unit - the OS supervises it, no manager.
    #[command(subcommand)]
    pub cmd: Option<Cmd>,
    /// Config file to load (default: ./paddock.toml if present)
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
    /// Address to bind the HTTP API to (default: 0.0.0.0 - network callers
    /// need a key; loopback callers are exempt unless --trusted-proxy)
    #[arg(long, value_name = "IP")]
    pub host: Option<IpAddr>,
    /// Port to bind the HTTP API to (default: 11540)
    #[arg(long, value_name = "PORT")]
    pub port: Option<u16>,
    /// GGUF model to load and serve at startup
    #[arg(short, long, value_name = "PATH")]
    pub model: Option<PathBuf>,
    /// Directory to scan for GGUF models (repeatable)
    #[arg(long = "model-dir", value_name = "PATH")]
    pub model_dir: Vec<PathBuf>,
    /// Compute device; only "cuda" exists
    #[arg(long, value_name = "DEVICE")]
    pub device: Option<String>,
    /// Which GPU to serve on: a CUDA ordinal ("1") or a device UUID
    /// ("GPU-..." as nvidia-smi prints it; a unique prefix is enough)
    #[arg(long, value_name = "ID")]
    pub gpu: Option<String>,
    /// Kernel pack path. Only needed by a build that has no kernels of its
    /// own - see --capabilities - and it OVERRIDES built-in kernels when both
    /// exist, which is how a bring-up campaign runs an architecture the
    /// shipped binary deliberately omits.
    #[arg(long, value_name = "PATH")]
    pub kernel_pack: Option<PathBuf>,
    /// Print what this build can do as one JSON line, then exit.
    ///
    /// For the manager, which supervises runners it did not build: they are
    /// separately versioned, so `runners/<v>/paddock-runner` may answer
    /// differently from the one beside `paddock.exe`, and a compile-time
    /// guess by the manager would be an assertion about somebody else's
    /// binary. Asking is the only honest way.
    #[arg(long)]
    pub capabilities: bool,
    /// Max context length (KV cache) for the served model.
    /// Aliases: --ctx-size (llama.cpp), --max-model-len (vLLM).
    /// (-c is llama.cpp muscle memory for context size)
    #[arg(
        short = 'c',
        long,
        value_name = "N",
        visible_alias = "ctx-size",
        alias = "max-model-len"
    )]
    pub max_ctx: Option<usize>,
    /// Continuous-batching width (concurrent sequences; 1 = serial loop).
    /// Aliases: --parallel/--np (llama.cpp), --max-num-seqs (vLLM).
    #[arg(
        long = "max-batch",
        value_name = "N",
        visible_alias = "parallel",
        alias = "np",
        alias = "max-num-seqs"
    )]
    pub max_batch: Option<usize>,
    /// Vision tower GGUF (mmproj) enabling image input
    #[arg(long, value_name = "PATH")]
    pub mmproj: Option<PathBuf>,
    /// MTP drafter GGUF (speculative decoding; gemma4's mtp-*.gguf)
    #[arg(long, value_name = "PATH")]
    pub mtp: Option<PathBuf>,
    /// Official-FP8/bf16 safetensors snapshot dir for native-fp8 plane
    /// ingestion (opt-in; skips the Q8_0 middle hop on fp8 hardware)
    #[arg(long = "fp8-native", value_name = "DIR")]
    pub fp8_native: Option<PathBuf>,
    /// Hard VRAM budget in MiB: pools/caches size inside it and a load that
    /// can't fit it refuses (co-resident servers can't oversubscribe the card)
    #[arg(long = "vram-budget", value_name = "MIB")]
    pub vram_budget: Option<u64>,
    /// Override the KV plan's graph/scratch reserve, in MiB (gpt-oss: the
    /// fixed 3 GiB graph/prefill scratch, which starves KV and the moe_offload
    /// slot cache on 8 GB cards; qwen35: only the 768 MiB graph-pools residual,
    /// its prefill scratch is profiled at load)
    #[arg(long = "graph-scratch-mib", value_name = "MIB")]
    pub graph_scratch_mib: Option<u64>,
    /// Default max output tokens per reply (when a request doesn't specify)
    #[arg(long = "max-output-tokens", value_name = "N")]
    pub max_output_tokens: Option<usize>,
    /// API key required for Bearer auth (empty on loopback = no auth)
    #[arg(long = "api-key", value_name = "KEY")]
    pub api_key: Option<String>,
    /// Web-search provider for the server-executed web_search tool:
    /// exa | tavily | firecrawl | brave | perplexity
    #[arg(long = "web-search-provider", value_name = "NAME")]
    pub web_search_provider: Option<String>,
    /// API key for the web-search provider
    #[arg(long = "web-search-api-key", value_name = "KEY")]
    pub web_search_api_key: Option<String>,
    /// Explicitly disable Bearer auth on a network bind (default: a
    /// non-loopback bind auto-generates and requires a key). Only for
    /// deployments where an external boundary (firewall/reverse proxy)
    /// carries the auth - the server will log a loud warning.
    #[arg(long = "no-auth")]
    pub no_auth: bool,
    /// Hard ceiling on generated tokens per reply, clamping whatever a
    /// request asks (abuse control for exposed instances)
    #[arg(long = "max-output-ceiling", value_name = "N")]
    pub max_output_ceiling: Option<usize>,
    /// Per-client generation-request rate limit, requests/minute
    #[arg(long = "ratelimit-per-minute", value_name = "N")]
    pub ratelimit_per_minute: Option<u32>,
    /// Per-client generation-request quota, requests/day
    #[arg(long = "ratelimit-per-day", value_name = "N")]
    pub ratelimit_per_day: Option<u32>,
    /// Behind a reverse proxy: key rate limits on its X-Real-IP, and require
    /// the API key from loopback too (every caller arrives from 127.0.0.1)
    #[arg(long = "trusted-proxy")]
    pub trusted_proxy: bool,
    /// Max pages rendered per PDF (past this is dropped, surfaced, never silent)
    #[arg(long = "pdf-max-pages", value_name = "N")]
    pub pdf_max_pages: Option<usize>,
    /// Target long-edge (px) for rendered PDF pages
    #[arg(long = "pdf-page-long-edge", value_name = "PX")]
    pub pdf_page_long_edge: Option<u32>,
    /// Registered MCP servers as one JSON array, e.g.
    /// [{"server_label":"github","server_url":"https://.../mcp"}]
    #[arg(long = "mcp-servers", value_name = "JSON")]
    pub mcp_servers: Option<String>,
    /// Append logs to this file instead of stdout
    #[arg(long = "log-file", value_name = "PATH")]
    pub log_file: Option<PathBuf>,

    // --- Server-side sampling overrides (llama.cpp-style; request wins).
    // Unset means the served model's own published profile decides - see
    // paddock_models::sampling. Setting one pins that knob instead.
    /// Pin the default temperature (unset: the model's published value)
    #[arg(long = "temp", value_name = "T")]
    pub temp: Option<f32>,
    /// Pin the default top-k cutoff, 0 = off
    #[arg(long = "top-k", value_name = "K")]
    pub top_k: Option<usize>,
    /// Pin the default nucleus (top-p) cutoff
    #[arg(long = "top-p", value_name = "P")]
    pub top_p: Option<f32>,
    /// Pin the default min-p cutoff
    #[arg(long = "min-p", value_name = "P")]
    pub min_p: Option<f32>,
    /// Pin the default repetition penalty, 1.0 = off
    #[arg(long = "repeat-penalty", value_name = "X")]
    pub repeat_penalty: Option<f32>,
    /// Default repetition-penalty window (tokens)
    #[arg(long = "repeat-last-n", value_name = "N")]
    pub repeat_last_n: Option<usize>,
    /// Default RNG seed when a request omits `seed` (unset = per-request
    /// time-derived seed, the OpenAI semantics)
    #[arg(long, value_name = "N")]
    pub seed: Option<u64>,
    /// KV cache dtype: auto | f16 | fp8_e4m3 (auto = per-family default)
    #[arg(long = "kv-cache-dtype", value_name = "DTYPE")]
    pub kv_cache_dtype: Option<String>,
    /// Serve the model under this id in /v1/models (vLLM flag)
    #[arg(long = "served-model-name", value_name = "NAME")]
    pub served_model_name: Option<String>,
    /// Disable speculative decode / MTP for this run
    #[arg(long = "no-spec")]
    pub no_spec: bool,
    /// Speculation policy: off | auto | ladder | <draft length 1..16>.
    /// "auto" is the closed-loop goodput controller (it turns speculation off
    /// by itself when the load says it does not pay); a number pins the draft
    /// length for A/B work.
    #[arg(long = "spec", value_name = "POLICY")]
    pub spec: Option<String>,
    /// Disable the per-request event ring (metadata-only, RAM-only; on by default)
    #[arg(long = "no-events")]
    pub no_events: bool,
    /// Disable the Prometheus /metrics surface (on by default; independent of
    /// --no-events)
    #[arg(long = "no-metrics")]
    pub no_metrics: bool,
    /// /metrics auth: on = API key required from everyone, off = open.
    /// Default: key required for network callers only, loopback scrapes open
    #[arg(long = "metrics-auth", value_name = "on|off")]
    pub metrics_auth: Option<String>,
    /// Skip 30 s transcription windows the VAD finds no speech in, before the
    /// encoder sees them (whisper lane). Off by default: it changes what a
    /// transcript contains, so it is a deliberate choice, not a default
    #[arg(long = "vad-gate")]
    pub vad_gate: bool,
    /// Session headers captured into event records (comma-separated;
    /// default X-Session-ID,X-Litellm-Session-Id)
    #[arg(long = "session-headers", value_name = "H1,H2")]
    pub session_headers: Option<String>,

    // --- Request filters + admission (doc §13, the llama-swap lessons).
    /// Alternative model ids this endpoint answers to (comma-separated) -
    /// e.g. impersonate "gpt-4o-mini" for tools that hardcode cloud ids
    #[arg(long, value_name = "ID1,ID2")]
    pub aliases: Option<String>,
    /// Parameter variants as one JSON object, e.g.
    /// {"high":{"chat_template_kwargs":{"reasoning_effort":"high"}}} -
    /// each key becomes a selectable model id <model>:<key>
    #[arg(long, value_name = "JSON")]
    pub variants: Option<String>,
    /// Client request fields to strip server-side (comma-separated), e.g.
    /// "temperature,top_p" to pin sampling to the server defaults
    #[arg(long = "strip-params", value_name = "F1,F2")]
    pub strip_params: Option<String>,
    /// Request fields forced to these values as one JSON object, e.g.
    /// {"temperature":0.7} - applied regardless of what clients send
    #[arg(long = "force-params", value_name = "JSON")]
    pub force_params: Option<String>,
    /// Max in-flight inference requests before an Overloaded refusal
    /// (queue depth; distinct from --max-batch, the compute width)
    #[arg(long = "concurrency-limit", value_name = "N")]
    pub concurrency_limit: Option<usize>,

    // --- Flags from other engines paddock deliberately does not support:
    // accepted by the parser so the error is honest, not "unknown flag".
    #[arg(long = "tensor-parallel-size", value_name = "N", hide = true)]
    pub tensor_parallel_size: Option<String>,
    #[arg(long = "gpu-memory-utilization", value_name = "F", hide = true)]
    pub gpu_memory_utilization: Option<String>,
    #[arg(long, value_name = "DTYPE", hide = true)]
    pub dtype: Option<String>,

    // --- Full llama.cpp / vLLM compat matrix (a swapped
    // launch line must never die on "unknown flag"). Three dispositions:
    // MAP (we have it under another name), ACCEPT+log (paddock self-tunes
    // what the flag hand-tunes - logged, never silent), REJECT with one
    // sentence (capability we lack). All hidden from --help; the visible
    // surface stays paddock's own.
    /// llama.cpp -ngl: paddock is always fully on-GPU; partial offload is rejected
    #[arg(long = "n-gpu-layers", alias = "ngl", value_name = "N", hide = true)]
    pub n_gpu_layers: Option<i64>,
    #[arg(long = "flash-attn", alias = "fa", value_name = "MODE", hide = true)]
    pub flash_attn: Option<String>,
    #[arg(long = "cache-type-k", alias = "ctk", value_name = "T", hide = true)]
    pub cache_type_k: Option<String>,
    #[arg(long = "cache-type-v", alias = "ctv", value_name = "T", hide = true)]
    pub cache_type_v: Option<String>,
    #[arg(long, hide = true)]
    pub mlock: bool,
    #[arg(long = "no-mmap", hide = true)]
    pub no_mmap: bool,
    #[arg(long, value_name = "PATH", hide = true)]
    pub lora: Option<String>,
    #[arg(short = 't', long, value_name = "N", hide = true)]
    pub threads: Option<String>,
    /// llama.cpp -n/--n-predict -> --max-output-tokens
    #[arg(short = 'n', long = "n-predict", value_name = "N", hide = true)]
    pub n_predict: Option<usize>,
    #[arg(short = 'b', long = "batch-size", value_name = "N", hide = true)]
    pub batch_size: Option<String>,
    #[arg(long = "ubatch-size", alias = "ub", value_name = "N", hide = true)]
    pub ubatch_size: Option<String>,
    #[arg(long, hide = true)]
    pub jinja: bool,
    /// llama.cpp --no-webui -> --no-studio
    #[arg(long = "no-webui", hide = true)]
    pub no_webui: bool,
    /// llama.cpp -a/--alias -> --served-model-name
    #[arg(short = 'a', long = "alias", value_name = "NAME", hide = true)]
    pub model_alias: Option<String>,
    #[arg(long = "enable-prefix-caching", hide = true)]
    pub enable_prefix_caching: bool,
    #[arg(long, value_name = "Q", hide = true)]
    pub quantization: Option<String>,
    #[arg(long = "trust-remote-code", hide = true)]
    pub trust_remote_code: bool,
    #[arg(long = "enable-auto-tool-choice", hide = true)]
    pub enable_auto_tool_choice: bool,
    #[arg(long = "tool-call-parser", value_name = "P", hide = true)]
    pub tool_call_parser: Option<String>,
    #[arg(long = "reasoning-parser", value_name = "P", hide = true)]
    pub reasoning_parser: Option<String>,
    #[arg(long = "speculative-config", value_name = "JSON", hide = true)]
    pub speculative_config: Option<String>,
    #[arg(long = "draft-model", value_name = "PATH", hide = true)]
    pub draft_model: Option<String>,
}

/// Where a resolved value came from, for the banner. `None` = file or default.
#[derive(Debug, Clone, Copy)]
pub enum Override {
    Env,
    Arg,
}

impl Override {
    fn tag(self) -> &'static str {
        match self {
            Override::Env => "env",
            Override::Arg => "arg",
        }
    }
}

/// Per-field override provenance shown next to values in the banner.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConfigSources {
    pub host_port: Option<Override>,
    pub model: Option<Override>,
    pub device: Option<Override>,
    pub model_dirs: Option<Override>,
}

/// Everything the banner needs beyond the resolved `Config`.
pub struct Banner {
    pub config_path: Option<String>,
    pub sources: ConfigSources,
    /// The effective API key to display (auth on), set by `run` after auth is
    /// resolved. `None` = no auth (loopback).
    pub auth_key: Option<String>,
    /// Compat-flag notes, held until logging exists.
    ///
    /// `resolve` runs before the subscriber is installed (it is what produces
    /// `log_file`), so `apply_compat` cannot log at the moment it decides. It
    /// used to, and the reorder that fixed `--log-file` silently broke the
    /// compat matrix's own promise - "ACCEPT+log ... logged, never silent".
    /// Collect here, replay once logging is up.
    pub compat_notes: Vec<String>,
}

/// Layer the config: defaults <- paddock.toml <- PADDOCK_* env <- CLI flags.
pub fn resolve(cli: &Cli) -> Result<(Config, Banner), ConfigError> {
    // 1. base - explicit --config, else auto-detected ./paddock.toml, else defaults
    let (mut cfg, config_path) = match &cli.config {
        Some(p) => (Config::from_toml(p)?, Some(p.display().to_string())),
        None => {
            let auto = PathBuf::from("paddock.toml");
            if auto.exists() {
                (Config::from_toml(&auto)?, Some("paddock.toml".to_string()))
            } else {
                (Config::default(), None)
            }
        }
    };

    // 2. environment overlay (record provenance before merging) - but never
    // for an explicit --config run: the endpoint's file is the whole config.
    // The manager spawns runners purely `paddock-runner --config <file>` and
    // children inherit the MANAGER's environment, so an env overlay here was
    // a silent second config channel overriding every endpoint's file
    // (found via the banner's "Models dir ... (env)" line -).
    // Hand-typed CLI flags still win below; the engine's debug knobs
    // (PADDOCK_NO_MOE_BS & co) are read directly by the engine and stay live.
    let mut src = ConfigSources::default();
    if cli.config.is_none() {
        if std::env::var_os("PADDOCK_HOST").is_some() || std::env::var_os("PADDOCK_PORT").is_some()
        {
            src.host_port = Some(Override::Env);
        }
        if std::env::var_os("PADDOCK_MODEL").is_some() {
            src.model = Some(Override::Env);
        }
        if std::env::var_os("PADDOCK_DEVICE").is_some() {
            src.device = Some(Override::Env);
        }
        if std::env::var_os("PADDOCK_MODEL_DIRS").is_some() {
            src.model_dirs = Some(Override::Env);
        }
        cfg.merge_env()?;
    }

    // 3. CLI overlay (highest priority)
    if let Some(h) = cli.host {
        cfg.host = h;
        src.host_port = Some(Override::Arg);
    }
    if let Some(p) = cli.port {
        cfg.port = p;
        src.host_port = Some(Override::Arg);
    }
    if let Some(m) = &cli.model {
        cfg.model = Some(m.clone());
        src.model = Some(Override::Arg);
    }
    if !cli.model_dir.is_empty() {
        cfg.model_dirs = cli.model_dir.clone();
        src.model_dirs = Some(Override::Arg);
    }
    if let Some(d) = &cli.device {
        cfg.device = d.clone();
        src.device = Some(Override::Arg);
    }
    if let Some(g) = &cli.gpu {
        cfg.gpu = Some(g.clone());
    }
    if let Some(k) = &cli.kernel_pack {
        cfg.kernel_pack = Some(k.clone());
    }
    if let Some(c) = cli.max_ctx {
        cfg.max_ctx = c;
    }
    if let Some(b) = cli.max_batch {
        cfg.max_batch = b;
    }
    if let Some(mm) = &cli.mmproj {
        cfg.mmproj = Some(mm.clone());
    }
    if let Some(mt) = &cli.mtp {
        cfg.mtp = Some(mt.clone());
    }
    if let Some(d) = &cli.fp8_native {
        cfg.fp8_native = Some(d.clone());
    }
    if let Some(b) = cli.vram_budget {
        cfg.vram_budget = Some(b);
    }
    if let Some(m) = cli.graph_scratch_mib {
        cfg.graph_scratch_mib = Some(m);
    }
    if let Some(n) = cli.max_output_tokens {
        cfg.max_tokens = Some(n);
    }
    if let Some(k) = &cli.api_key {
        cfg.api_key = Some(k.clone());
    }
    if cli.no_auth {
        cfg.no_auth = true;
    }
    if let Some(n) = cli.max_output_ceiling {
        cfg.max_output_ceiling = Some(n);
    }
    if let Some(n) = cli.ratelimit_per_minute {
        cfg.ratelimit_per_minute = Some(n);
    }
    if let Some(n) = cli.ratelimit_per_day {
        cfg.ratelimit_per_day = Some(n);
    }
    if cli.trusted_proxy {
        cfg.trusted_proxy = true;
    }
    if let Some(n) = cli.pdf_max_pages {
        cfg.pdf_max_pages = n;
    }
    if let Some(n) = cli.pdf_page_long_edge {
        cfg.pdf_page_long_edge = n;
    }
    if let Some(v) = &cli.mcp_servers {
        cfg.mcp_servers = serde_json::from_str(v).map_err(|_| ConfigError::BadArg {
            name: "--mcp-servers",
            value: v.clone(),
        })?;
    }
    if let Some(p) = &cli.log_file {
        cfg.log_file = Some(p.clone());
    }
    if let Some(v) = &cli.web_search_provider {
        cfg.web_search_provider = Some(v.clone());
    }
    if let Some(v) = &cli.web_search_api_key {
        cfg.web_search_api_key = Some(v.clone());
    }
    // these stay Option all the way through: an unset flag must reach the
    // resolver as "unset" so the model's own published profile can fill it
    // not as an OpenAI value that overrides it
    if let Some(v) = cli.temp {
        cfg.temp = Some(v);
    }
    if let Some(v) = cli.top_k {
        cfg.top_k = Some(v);
    }
    if let Some(v) = cli.top_p {
        cfg.top_p = Some(v);
    }
    if let Some(v) = cli.min_p {
        cfg.min_p = Some(v);
    }
    if let Some(v) = cli.repeat_penalty {
        cfg.repeat_penalty = Some(v);
    }
    if let Some(v) = cli.repeat_last_n {
        cfg.repeat_last_n = v;
    }
    if let Some(v) = cli.seed {
        cfg.seed = Some(v);
    }
    if let Some(v) = &cli.kv_cache_dtype {
        cfg.kv_cache_dtype = v.clone();
    }
    if let Some(v) = &cli.served_model_name {
        cfg.served_model_name = Some(v.clone());
    }
    if cli.no_spec {
        cfg.no_spec = true;
    }
    if let Some(v) = &cli.spec {
        cfg.spec = Some(v.clone());
    }
    if cli.vad_gate {
        cfg.vad_gate = true;
    }
    if cli.no_events {
        cfg.no_events = true;
    }
    if cli.no_metrics {
        cfg.no_metrics = true;
    }
    if let Some(v) = &cli.metrics_auth {
        cfg.metrics_auth = Some(match v.as_str() {
            "on" | "true" | "1" => true,
            "off" | "false" | "0" => false,
            _ => {
                return Err(ConfigError::BadArg {
                    name: "--metrics-auth",
                    value: v.clone(),
                });
            }
        });
    }
    if let Some(v) = &cli.session_headers {
        cfg.session_headers = v
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
    }
    if let Some(v) = &cli.aliases {
        cfg.aliases = v
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
    }
    if let Some(v) = &cli.variants {
        cfg.variants = serde_json::from_str(v).map_err(|_| ConfigError::BadArg {
            name: "--variants",
            value: v.clone(),
        })?;
    }
    if let Some(v) = &cli.strip_params {
        cfg.strip_params = v
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
    }
    if let Some(v) = &cli.force_params {
        cfg.force_params = serde_json::from_str(v).map_err(|_| ConfigError::BadArg {
            name: "--force-params",
            value: v.clone(),
        })?;
    }
    if let Some(v) = cli.concurrency_limit {
        cfg.concurrency_limit = Some(v);
    }
    let mut compat_notes = Vec::new();
    apply_compat(cli, &mut cfg, &mut compat_notes)?;
    // Honest rejection of other engines' flags for features paddock lacks -
    // a swapped-in vLLM launch line should fail with an explanation, never
    // silently accept-and-ignore (no-silent-failures principle).
    if cli.tensor_parallel_size.is_some() {
        return Err(crate::config::ConfigError::Unsupported(
            "--tensor-parallel-size: paddock serves one GPU per process today (multi-GPU              tensor parallelism is not yet supported) - drop the flag",
        ));
    }
    if cli.gpu_memory_utilization.is_some() {
        return Err(crate::config::ConfigError::Unsupported(
            "--gpu-memory-utilization: paddock sizes pools from measured will-it-fit              estimation instead of a fraction - drop the flag (use --max-ctx/--max-batch              to bound the footprint)",
        ));
    }
    if cli.dtype.is_some() {
        return Err(crate::config::ConfigError::Unsupported(
            "--dtype: the GGUF file decides weight precision (labeled quants, honest              naming) - pick a different quant file instead; --kv-cache-dtype controls              the KV cache",
        ));
    }

    Ok((
        cfg,
        Banner {
            config_path,
            sources: src,
            auth_key: None,
            compat_notes,
        },
    ))
}

/// The llama.cpp / vLLM compat matrix. MAP flags fold into the
/// canonical config; ACCEPT flags log what paddock does instead (self-tuned -
/// loud, never silent); REJECT flags fail with the one-sentence reason.
fn apply_compat(
    cli: &Cli,
    cfg: &mut crate::config::Config,
    notes: &mut Vec<String>,
) -> Result<(), crate::config::ConfigError> {
    use crate::config::ConfigError::Unsupported;
    let mut accept = |flag: &str, what: &str| {
        notes.push(format!("{flag}: accepted for compatibility - {what}"));
    };
    // ---- MAP ----
    if let Some(n) = cli.n_predict {
        cfg.max_tokens = Some(n);
    }
    if cli.no_webui {
        accept(
            "--no-webui",
            "the runner is always headless - the Studio lives in the manager (`paddock`)",
        );
    }
    if let Some(a) = &cli.model_alias {
        cfg.served_model_name = Some(a.clone());
    }
    match (cli.cache_type_k.as_deref(), cli.cache_type_v.as_deref()) {
        (None, None) => {}
        (Some("f16"), Some("f16") | None) | (None, Some("f16")) => {
            cfg.kv_cache_dtype = "f16".into();
        }
        (k, v) => {
            let t = k.or(v).unwrap_or("?");
            return Err(Unsupported(match t {
                "q8_0" | "q4_0" | "q4_1" | "q5_0" | "q5_1" | "iq4_nl" => {
                    "--cache-type-k/v: paddock's quantized KV cache is fp8-e4m3, not the llama.cpp integer types - use --kv-cache-dtype fp8_e4m3 (or f16 for exact)"
                }
                _ => {
                    "--cache-type-k/v: expected f16 here; paddock's lossy KV mode is  --kv-cache-dtype fp8_e4m3"
                }
            }));
        }
    }
    // ---- ACCEPT (paddock self-tunes; logged, not silent) ----
    if let Some(n) = cli.n_gpu_layers {
        if !(0..99).contains(&n) {
            accept(
                "-ngl/--n-gpu-layers",
                "paddock always runs the whole model on the GPU",
            );
        } else {
            return Err(Unsupported(
                "--n-gpu-layers: partial CPU/GPU layer offload is not yet supported - paddock serves fully on-GPU (MoE expert offload is on the roadmap)",
            ));
        }
    }
    if let Some(m) = &cli.flash_attn {
        if m == "off" || m == "0" {
            return Err(Unsupported(
                "--flash-attn off: paddock's fused attention kernels are the only attention path - there is no scalar fallback to switch to",
            ));
        }
        accept(
            "--flash-attn",
            "paddock's fused attention kernels are always on",
        );
    }
    if cli.batch_size.is_some() || cli.ubatch_size.is_some() {
        accept(
            "-b/-ub batch sizing",
            "paddock tunes prefill chunking automatically (see --max-batch for concurrency width)",
        );
    }
    if cli.jinja {
        accept(
            "--jinja",
            "chat templates are always rendered from the GGUF's template",
        );
    }
    if cli.enable_prefix_caching {
        accept(
            "--enable-prefix-caching",
            "the radix prefix cache is always on when batching",
        );
    }
    if cli.trust_remote_code {
        accept(
            "--trust-remote-code",
            "GGUF models carry no code - nothing to trust",
        );
    }
    if cli.enable_auto_tool_choice {
        accept(
            "--enable-auto-tool-choice",
            "tool choice is auto by default on every endpoint",
        );
    }
    if let Some(p) = cli
        .tool_call_parser
        .as_deref()
        .or(cli.reasoning_parser.as_deref())
    {
        accept(
            "--tool-call-parser/--reasoning-parser",
            &format!(
                "paddock selects the parser from the model architecture (harmony/qwen-xml/gemma-channel); requested {p:?}"
            ),
        );
    }
    // ---- REJECT ----
    if cli.mlock || cli.no_mmap {
        return Err(Unsupported(
            "--mlock/--no-mmap: paddock memory-maps the GGUF for load and keeps serving state on the GPU - host-side pinning knobs are not supported",
        ));
    }
    if cli.lora.is_some() {
        return Err(Unsupported(
            "--lora: LoRA adapters are not yet supported - merge the adapter into the  GGUF for now",
        ));
    }
    if cli.threads.is_some() {
        return Err(Unsupported(
            "-t/--threads: paddock's serving path is GPU-resident; there is no CPU  thread pool to size",
        ));
    }
    if cli.speculative_config.is_some() || cli.draft_model.is_some() {
        return Err(Unsupported(
            "--speculative-config/--draft-model: speculative decode is auto-on when the  GGUF carries MTP tensors, or pass a drafter GGUF with --mtp; disable with  --no-spec",
        ));
    }
    if cli.quantization.is_some() {
        return Err(Unsupported(
            "--quantization: the GGUF file already decides weight precision - pick the quant file you want to serve",
        ));
    }
    Ok(())
}

/// Parse args, resolve config, and run the server until stopped.
/// What this build can do, as one JSON line.
///
/// The audience is a supervisor holding the binary, not a person: the manager
/// spawns runners it did not build and that ship on their own version line
/// (`runners/<v>/`), so the only honest way for it to know whether a runner
/// needs a kernel pack is to ask that runner.
///
/// Additive forever - a reader older than the binary must be able to ignore
/// keys it does not know, and one NEWER must treat an absent key as "no". Both
/// fall out of JSON objects, which is the whole reason this is not a bare
/// exit code.
pub fn capabilities() -> String {
    serde_json::json!({
        // Kernels are linked in, so `kernel_pack` is not required and, when
        // given anyway, is an OVERRIDE (bring-up campaigns on an arch the
        // shipped binary omits).
        "kernels_builtin": cfg!(feature = "static-pack"),
        "version": paddock_admin::version::LONG,
    })
    .to_string()
}

pub fn run() -> std::process::ExitCode {
    // llama.cpp writes multi-char SHORT options (-ngl, -fa, -np, ...) that a
    // clap parser reads as clustered single-char shorts (-n gl). Rewrite the
    // known ones to their long forms before parsing so a pasted llama.cpp
    // launch line parses as its author meant (see the compat matrix).
    let args = std::env::args_os().map(|a| match a.to_str() {
        Some("-ngl") => "--n-gpu-layers".into(),
        Some("-fa") => "--flash-attn".into(),
        Some("-np") => "--parallel".into(),
        Some("-ub") => "--ubatch-size".into(),
        Some("-ctk") => "--cache-type-k".into(),
        Some("-ctv") => "--cache-type-v".into(),
        _ => a,
    });
    let cli = Cli::parse_from(args);

    // Answered before anything else - no config, no logging, no device. The
    // caller is a supervisor holding this file and asking what it is; it must
    // work on a box with no GPU, no config and no models, and it must print
    // one line to stdout and nothing else.
    if cli.capabilities {
        println!("{}", capabilities());
        return std::process::ExitCode::SUCCESS;
    }

    // SEAL the ENVIRONMENT. Everything below - config merge, model
    // load, the engine's own elections - happens with only the documented
    // surface left standing. No-op unless this is a hardened build.
    //
    // This is the half that reaches the CUDA pack. The pack reads its election
    // and kill switches straight out of the environment (`pd_env`, abi.cuh),
    // and the engine writes its tuned defaults there for it to find, so
    // compiling out the Rust reads alone would leave every pack-side switch
    // live on the shell. Here, before any of it: a value a user exported is
    // gone, and the elections the engine writes afterwards land untouched.
    //
    // It has to precede the thread pool for the same reason `remove_var` is
    // unsafe, and it precedes logging because the point is that nothing has
    // read one yet - hence the deferred report a few lines down.
    let sealed = paddock_models::hardening::seal_environment(crate::config::ENV_SURFACE);

    // Service verbs own their process (incl. logging - a service has no
    // console, so tracing goes to a file there, not stdout).
    if let Some(Cmd::Service(action)) = cli.cmd {
        return crate::service::dispatch(action);
    }

    // Resolve before logging starts, because `log_file` is a resolved setting:
    // it can come from the config file as well as `--log-file`, and the old
    // order (subscriber first) is precisely why the flag was inert on a
    // hand-started runner for so long - by the time we knew the path, the
    // subscriber was already installed stdout-only. Nothing in
    // `resolve` logs, so no line is lost by waiting; a config error is a plain
    // stderr message because at that point there is no logging yet, by design.
    let (cfg, banner) = match resolve(&cli) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("config error: {e}");
            return std::process::ExitCode::from(2);
        }
    };
    paddock_admin::logging::init(cfg.log_file.as_deref());
    // Now that there is somewhere to say it: the compat decisions made during
    // resolve. Never silent is the whole point of accepting these flags.
    for note in &banner.compat_notes {
        tracing::info!("{note}");
    }
    // And what the seal took, held since before there was a subscriber. Saying
    // it matters more than doing it quietly would: somebody who exported a
    // switch and got the elected route anyway deserves to read why, rather
    // than conclude the variable is broken.
    if !sealed.is_empty() {
        tracing::info!(
            "ignored {} PADDOCK_* variable(s) from the environment: {}. This build \
             serves the settings it was tested with; the supported surface is the \
             config file and the CLI.",
            sealed.len(),
            sealed.join(", ")
        );
    }

    // Publish the output-token default to the Responses request deserializer.
    // SAFETY: single-threaded startup, before the tokio runtime spawns threads.
    if let Some(n) = cfg.max_tokens {
        unsafe { std::env::set_var("PADDOCK_MAX_OUTPUT_TOKENS", n.to_string()) };
    }

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!(error = %e, "failed to start async runtime");
            return std::process::ExitCode::FAILURE;
        }
    };
    match rt.block_on(crate::run(cfg, banner)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            // tracing, not eprintln: logging is up by this point, and the whole
            // reason somebody passes --log-file is to find out why a runner did
            // not come up. On eprintln the file was created and left empty,
            // which reads as "nothing happened" rather than "it failed here".
            // The terminal still sees it - the stdout layer is always present.
            tracing::error!(error = %e, "server error");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Show an unspecified bind address (0.0.0.0 / ::) as localhost for the URL.
fn display_addr(host: IpAddr, port: u16) -> String {
    match host {
        IpAddr::V4(v4) if v4.is_unspecified() => format!("localhost:{port}"),
        IpAddr::V6(v6) if v6.is_unspecified() => format!("localhost:{port}"),
        IpAddr::V6(v6) => format!("[{v6}]:{port}"),
        IpAddr::V4(v4) => format!("{v4}:{port}"),
    }
}

/// Print the startup banner. Called from `run()` after the socket binds, so the
/// address shown is one we actually hold.
pub fn print_startup_banner(cfg: &Config, banner: &Banner) {
    // Respect NO_COLOR and non-terminal stdout (piped logs stay clean).
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let (dim, cyan, reset) = if color {
        ("\x1b[2m", "\x1b[36m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let version = paddock_admin::version::LONG;
    let tag = |o: Option<Override>| match o {
        Some(o) => format!("  {dim}({}){reset}", o.tag()),
        None => String::new(),
    };

    let addr = display_addr(cfg.host, cfg.port);

    println!();
    println!("  {cyan}█▀▄ ▄▀▄ █▀▄ █▀▄ ▄▀▄ ▄▀▀ █▄▀{reset}   {dim}v{version}{reset}");
    println!("  {cyan}█▀▀ █▀█ █ █ █ █ █ █ █   █▀▄{reset}   {dim}{COPYRIGHT}{reset}");
    println!("  {cyan}▀   ▀ ▀ ▀▀▀ ▀▀▀ ▀▀▀ ▀▀▀ ▀ ▀{reset}");
    println!();
    println!(
        "  {dim}API{reset}         {cyan}http://{addr}{reset}  {dim}(OpenAI + Anthropic){reset}{}",
        tag(banner.sources.host_port)
    );
    // the URL above is what a person types; this is what the socket actually
    // holds, and the two differ exactly when it matters
    let bind = if cfg.host.is_unspecified() {
        format!(
            "{}:{}  {dim}(all interfaces - reachable from the network){reset}",
            cfg.host, cfg.port
        )
    } else if cfg.host.is_loopback() {
        format!("{}:{}  {dim}(this machine only){reset}", cfg.host, cfg.port)
    } else {
        format!("{}:{}", cfg.host, cfg.port)
    };
    println!("  {dim}Bind{reset}        {bind}");
    println!(
        "  {dim}Device{reset}      {}{}",
        cfg.device,
        tag(banner.sources.device)
    );
    match &cfg.model {
        Some(m) => println!(
            "  {dim}Model{reset}       {}{}",
            m.file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default(),
            tag(banner.sources.model)
        ),
        None => println!("  {dim}Model{reset}       {dim}none - GET /v1/models to browse{reset}"),
    }
    let dirs = &cfg.model_dirs;
    let first = dirs
        .first()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let extra = if dirs.len() > 1 {
        format!("  {dim}(+{} more){reset}", dirs.len() - 1)
    } else {
        String::new()
    };
    println!(
        "  {dim}Models dir{reset}  {first}{extra}{}",
        tag(banner.sources.model_dirs)
    );
    match &banner.config_path {
        Some(p) => println!("  {dim}Config{reset}      loaded {p}"),
        None => println!(
            "  {dim}Config{reset}      {dim}built-in defaults (see paddock.example.toml){reset}"
        ),
    }
    if let Some(key) = &banner.auth_key {
        let scope = if cfg.trusted_proxy {
            "required from everyone (trusted_proxy)"
        } else {
            "required from the network, loopback callers exempt"
        };
        println!("  {dim}API key{reset}     {cyan}{key}{reset}  {dim}{scope}{reset}");
    } else if cfg.no_auth {
        println!("  {dim}Auth{reset}        {dim}none (--no-auth){reset}");
    } else {
        println!("  {dim}Auth{reset}        {dim}none (loopback bind){reset}");
    }
    println!(
        "  {dim}Studio{reset}      {dim}runners are headless - run `paddock` (the manager) for the Studio{reset}"
    );
    #[cfg(windows)]
    println!(
        "  {dim}Admin{reset}       {dim}\\\\.\\pipe\\paddock-runner-{} (local, OS-identity auth){reset}",
        cfg.port
    );
    #[cfg(unix)]
    println!(
        "  {dim}Admin{reset}       {dim}{} (local, OS-identity auth){reset}",
        paddock_admin::socket_path(cfg.port).display()
    );
    println!();
    println!("  {dim}Docs{reset}        {cyan}https://truespar.com/paddock/docs{reset}");
    println!();
    println!("  {dim}Press Ctrl+C to stop the server{reset}");
    println!("  {dim}Run with --help for all options{reset}");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compat matrix promises three dispositions, one of which is
    /// "ACCEPT+log - logged, never silent". That promise lives across a
    /// boundary the compiler cannot see: `apply_compat` decides during
    /// `resolve`, which deliberately runs before the subscriber exists so it
    /// can produce `log_file` first. It logged directly once, and
    /// the reorder turned every acceptance into silence with nothing failing.
    ///
    /// So assert the note is RECORDED, not that something was printed - a test
    /// that watched stdout would pass against a subscriber-less binary too.
    #[test]
    fn accepted_compat_flags_leave_a_note_for_the_log() {
        let cli = Cli::parse_from([
            "paddock-runner",
            "--model",
            "x.gguf",
            "--jinja",
            "--enable-prefix-caching",
        ]);
        let (_cfg, banner) = resolve(&cli).expect("resolve");
        assert_eq!(
            banner.compat_notes.len(),
            2,
            "both accepted flags must be reported, got {:?}",
            banner.compat_notes
        );
        assert!(banner.compat_notes.iter().any(|n| n.contains("--jinja")));
        assert!(
            banner
                .compat_notes
                .iter()
                .any(|n| n.contains("--enable-prefix-caching")),
            "got {:?}",
            banner.compat_notes
        );
    }

    /// A plain launch says nothing - the notes exist for flags the user passed,
    /// not as startup chatter everyone pays for.
    #[test]
    fn a_launch_with_no_compat_flags_is_quiet() {
        let cli = Cli::parse_from(["paddock-runner", "--model", "x.gguf"]);
        let (_cfg, banner) = resolve(&cli).expect("resolve");
        assert!(
            banner.compat_notes.is_empty(),
            "got {:?}",
            banner.compat_notes
        );
    }

    /// REJECT stays a hard error, not a note: these name capabilities we do not
    /// have, and accepting them quietly would be the silent-failure the product
    /// principles forbid.
    #[test]
    fn rejected_compat_flags_still_fail_the_launch() {
        for flag in ["--mlock", "--trust-remote-code"] {
            let cli = Cli::parse_from(["paddock-runner", "--model", "x.gguf", flag]);
            let got = resolve(&cli);
            if flag == "--mlock" {
                assert!(got.is_err(), "{flag} must refuse, not accept");
            } else {
                // --trust-remote-code is an ACCEPT: GGUF carries no code.
                assert!(got.is_ok(), "{flag} should be accepted");
            }
        }
    }
}
