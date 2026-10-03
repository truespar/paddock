//! `paddock` - the manager + Studio. The consumer binary: supervises runner
//! processes, owns the catalog/downloads, the one SQLite, NVML telemetry, and
//! serves the Studio web UI on its own port. It contains zero inference code -
//! serving is `paddock-runner`, a separately versioned artifact (see
//!
//! CLI verbs (Docker/gh-class, no full-screen TUI - doc §7) land against the
//! manager API as the supervisor grows; today: run the manager, and `inspect`.

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "paddock",
    // The build stamp, not the bare SemVer: `--version` is what someone pastes
    // into a bug report, and `0.1.0` alone cannot tell two week-apart builds
    // apart during 0.x.
    version = paddock_admin::version::LONG,
    about = "Paddock - local AI serving. This runs the manager and the Studio; `paddock-runner` serves the models."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Address to bind the Studio/API to (default: 127.0.0.1)
    #[arg(long, value_name = "IP")]
    host: Option<std::net::IpAddr>,
    /// Port for the Studio/API (default: 11500 - runner ports allocate upward
    /// from 11540)
    #[arg(long, value_name = "PORT")]
    port: Option<u16>,
    /// Behind a reverse proxy: require the API key from loopback callers too
    /// (every caller arrives from 127.0.0.1 there)
    #[arg(long)]
    trusted_proxy: bool,
    /// Directory to scan for GGUF models (repeatable)
    #[arg(long = "model-dir", value_name = "PATH")]
    model_dir: Vec<std::path::PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect a GGUF file: architecture, geometry, quant mix
    Inspect {
        /// Path to a .gguf file
        path: std::path::PathBuf,
        /// Emit the full report as JSON instead of the human card
        #[arg(long)]
        json: bool,
        // Generator plumbing for the shapes generator, not a verb
        // a user has any reason to run. Hidden rather than absent: the block it
        // prints is checked into models.toml, so being able to regenerate one by
        // hand is how a reviewer confirms a published number.
        /// Emit the models.toml `[model.artifact.shape]` block instead of the card
        #[arg(long, hide = true)]
        shape: bool,
        /// Resident weight bytes measured off a real load; without it `--shape`
        /// falls back to the file size and stamps `source = "probed"`
        #[arg(long, hide = true, value_name = "BYTES")]
        resident: Option<u64>,
        /// Price the shape as an encoder (embeddings/rerank): no KV cache
        #[arg(long, hide = true)]
        encoder: bool,
    },
    /// Download a model (weights + companion files) from the built-in catalog
    /// into the models directory, checksum-verified. Downloading is the
    /// manager's job; serve the result with `paddock-runner -m <id>`.
    Pull {
        /// Catalog model id (see the Studio's models page for the list)
        id: String,
    },
    /// Start a new model endpoint: writes ~/paddock/servers/<port>.toml,
    /// downloads the model if it is missing, launches
    /// `paddock-runner --config <file>`, waits until it answers, and prints
    /// the endpoint. The file is yours to keep, edit by hand, or run with no
    /// manager at all. To start one that already exists, use `paddock start`.
    Serve {
        /// Catalog id, installed model name, or GGUF path
        model: String,
        /// Explicit runner port (default: allocated upward from 11540)
        #[arg(long)]
        port: Option<u16>,
        /// Context window for the new endpoint
        #[arg(long = "max-ctx")]
        max_ctx: Option<usize>,
        /// How many requests it batches together
        #[arg(long = "max-batch")]
        max_batch: Option<usize>,
        /// Pin it: never stopped automatically to make room for another
        /// model, and its VRAM is not counted as reclaimable
        #[arg(long)]
        pin: bool,
        // String, not u32, because the UUID is the form the rest of the stack
        // prefers and the CLI could not express it: SpawnSpec.gpu is already
        // Option<String>, its `de_gpu` already accepts a bare number, and the
        // runner's own `gpu` config takes either. This flag was the one place
        // in the chain that could only say "index". Kept as a plain comment,
        // not a doc comment: clap prints doc comments as `--help` text, and
        // this reasoning is ours, not the user's.
        /// Which GPU to use: an index (as the GPU page numbers them) or a
        /// device UUID ("GPU-d56cd6c9-...", the nvidia-smi spelling). Prefer the
        /// UUID when the machine has more than one card - indices are enumeration
        /// order and move when a card is added, removed, or the driver
        /// reorders them.
        #[arg(long, value_name = "INDEX|UUID")]
        gpu: Option<String>,
        /// Weights-artifact choice for a catalog model (e.g. q4); default =
        /// the model's default weights, preferring an installed one
        #[arg(long, value_name = "ID")]
        artifact: Option<String>,
        /// Serve with native-FP8 plane ingestion (needs the model's
        /// fp8-snapshot artifact installed)
        #[arg(long = "fp8-native")]
        fp8_native: bool,
    },
    /// List the models running on this machine, with live health
    Ps,
    /// Pin or unpin a model: a pinned one is never stopped automatically to
    /// make room for another, and the pin is remembered
    Pin {
        port: u16,
        /// Remove the pin instead of setting it
        #[arg(long)]
        off: bool,
    },
    /// Start an already-configured endpoint from its servers/<port>.toml,
    /// verbatim - hand-edits included. Takes a port, or a model name when
    /// exactly one configured endpoint matches it.
    Start {
        /// Port, or a model name to match against configured endpoints
        target: String,
    },
    /// Stop a running model: it finishes the requests it already has, then
    /// exits. Takes a port, or a model name when exactly one running endpoint
    /// matches. The endpoint stays configured - `paddock start` brings it back.
    Stop {
        /// Port, or a model name to match against running endpoints
        target: String,
        /// How long to wait for in-flight requests, in ms, before forcing it
        /// (default 30000)
        #[arg(long = "timeout-ms")]
        timeout_ms: Option<u64>,
    },
    /// Change which model a port serves, keeping the port
    Switch {
        port: u16,
        /// Catalog id, installed model name, or GGUF path
        model: String,
        #[arg(long = "max-ctx")]
        max_ctx: Option<usize>,
        #[arg(long = "max-batch")]
        max_batch: Option<usize>,
    },
    /// Stream logs: one model's, the manager's own, or all of them together
    /// (each line prefixed with where it came from). Default: all of them.
    Logs {
        /// Port of one running model (shorthand for --runner <PORT>)
        port: Option<u16>,
        /// The manager's own log
        #[arg(long, conflicts_with_all = ["port", "runner", "all"])]
        manager: bool,
        /// One model's log, by port
        #[arg(long, value_name = "PORT", conflicts_with_all = ["port", "all"])]
        runner: Option<u16>,
        /// The manager and every model together (the default)
        #[arg(long)]
        all: bool,
        /// Keep following, tail -f style
        #[arg(short = 'f', long)]
        follow: bool,
        /// History lines per source (default 200)
        #[arg(long)]
        tail: Option<usize>,
        /// Skip the history - only new lines (pair with --follow)
        #[arg(long = "no-history")]
        no_history: bool,
    },
}

/// Where the CLI verbs find the manager. The verbs are clients of the same
/// API the Studio calls (doc §7) - a running manager is required.
fn manager_url() -> String {
    std::env::var("PADDOCK_MANAGER_URL").unwrap_or_else(|_| "http://127.0.0.1:11500".into())
}

/// One shared client-verb runtime + error style: any failure prints and exits
/// nonzero; a connection failure names the fix (start the manager).
fn run_verb<F, T>(fut: F) -> std::process::ExitCode
where
    F: std::future::Future<Output = Result<T, String>>,
{
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to start async runtime: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match rt.block_on(fut) {
        Ok(_) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn connect_hint(e: reqwest::Error) -> String {
    if e.is_connect() {
        format!(
            "cannot reach the manager at {} - start it first (run `paddock`), or set PADDOCK_MANAGER_URL",
            manager_url()
        )
    } else {
        format!("manager request failed: {e}")
    }
}

/// Surface the manager's error message, not just a status code.
async fn expect_ok(res: reqwest::Response) -> Result<serde_json::Value, String> {
    let status = res.status();
    let body: serde_json::Value = res.json().await.unwrap_or(serde_json::Value::Null);
    if status.is_success() {
        Ok(body)
    } else {
        let msg = body
            .pointer("/error/message")
            .and_then(|m| m.as_str())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("HTTP {status}"));
        Err(msg)
    }
}

/// Resolve a start/stop TARGET - a port, or a model name matched against the
/// given rows (case-insensitive substring of the row's model). Exactly one
/// match or an honest error listing what exists; never a guess.
fn resolve_target(target: &str, rows: &[serde_json::Value], what: &str) -> Result<u16, String> {
    if let Ok(p) = target.parse::<u16>() {
        return Ok(p);
    }
    let needle = target.to_lowercase();
    let describe =
        |v: &serde_json::Value| format!("{} ({})", v["port"].as_u64().unwrap_or(0), served_id(v));
    let hits: Vec<&serde_json::Value> = rows
        .iter()
        .filter(|v| served_id(v).to_lowercase().contains(&needle))
        .collect();
    match hits.as_slice() {
        [one] => Ok(one["port"].as_u64().unwrap_or(0) as u16),
        [] => Err(if rows.is_empty() {
            format!("no {what} endpoints")
        } else {
            format!(
                "{target:?} matches no {what} endpoint - have: {}",
                rows.iter().map(describe).collect::<Vec<_>>().join(", ")
            )
        }),
        many => Err(format!(
            "{target:?} matches {} endpoints - pick a port: {}",
            many.len(),
            many.iter()
                .map(|v| describe(v))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// An encoder runner carries its id in `embedder` (it has no generative
/// model) - the MODEL column shows whichever the runner serves.
fn served_id(v: &serde_json::Value) -> String {
    // a decision model (Laya, Clef) serves no chat model: its id is `reader`
    v["model"]
        .as_str()
        .or_else(|| v["embedder"].as_str())
        .or_else(|| v["reader"].as_str())
        .unwrap_or("-")
        .to_owned()
}

fn print_runner_row(v: &serde_json::Value) {
    println!(
        "{:<7} {:<8} {:<28} {:<10} {:<9} {:<8} {:<5} {}",
        v["port"].as_u64().unwrap_or(0),
        v["pid"].as_u64().unwrap_or(0),
        served_id(v),
        v["status"].as_str().unwrap_or("-"),
        v["origin"].as_str().unwrap_or("-"),
        v["version"].as_str().unwrap_or("-"),
        if v["pinned"].as_bool().unwrap_or(false) {
            "pin"
        } else {
            "-"
        },
        v["endpoint"].as_str().unwrap_or("-"),
    );
}

/// Tracing to stdout - teed to `logs/manager.log` in manager mode, so the
/// §11.3 log stream (`--manager` / `--all` selectors) has the manager's own
/// lines on disk next to the runner logs. CLI verbs stay stdout-only: a `ps`
/// must never write the service's log.
///
/// The setup itself lives in `paddock_admin::logging` - it is shared with both
/// runner entry points, which is what stops the four copies from drifting
/// apart again (the systemd one already had).
fn init_tracing(tee: Option<std::path::PathBuf>) {
    paddock_admin::logging::init(tee.as_deref());
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    // Seal before anything reads  - including `data_dir()` on the
    // next line, which honours PADDOCK_DATA and must see the same environment
    // everything else does. No-op unless this is a hardened build.
    //
    // The manager's own share of this is small; the reason it seals at all is
    // that runners INHERIT its environment. A switch exported in the shell
    // that launched `paddock` would otherwise ride into every runner it
    // spawns, which is the one path where the runner's own seal is not the
    // first line of defence.
    //
    // SAFETY: single-threaded, first statement after argument parsing.
    let sealed = paddock_models::hardening::seal_environment(paddock_manager::config::ENV_SURFACE);
    init_tracing(cli.command.is_none().then(|| {
        paddock_manager::config::Config::data_dir()
            .join("logs")
            .join("manager.log")
    }));
    if !sealed.is_empty() {
        tracing::info!(
            "ignored {} PADDOCK_* variable(s) from the environment: {}. This build \
             serves the settings it was tested with.",
            sealed.len(),
            sealed.join(", ")
        );
    }
    match cli.command {
        Some(Command::Inspect {
            path,
            json,
            shape,
            resident,
            encoder,
        }) => {
            let want = shape.then_some(match resident {
                Some(b) => paddock_manager::inspect::Shape::Measured(b),
                None => paddock_manager::inspect::Shape::Probed,
            });
            paddock_manager::inspect::run(&path, json, want, encoder)
        }
        Some(Command::Serve {
            model,
            port,
            max_ctx,
            max_batch,
            pin,
            gpu,
            artifact,
            fp8_native,
        }) => run_verb(async move {
            // pull:true is the CLI convenience - a terminal download is
            // visible and expected; the UI's deploy never downloads.
            let body = serde_json::json!({
                "model": model, "port": port, "max_ctx": max_ctx, "max_batch": max_batch,
                "pinned": pin, "gpu": gpu,
                "artifact": artifact, "fp8_native": fp8_native, "pull": true,
            });
            println!("starting {model} (downloading it if needed; loading can take a while)...");
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(3600))
                .build()
                .map_err(|e| e.to_string())?;
            let res = client
                .post(format!("{}/api/runners", manager_url()))
                .json(&body)
                .send()
                .await
                .map_err(connect_hint)?;
            let v = expect_ok(res).await?;
            println!(
                "serving {} on {}  (pid {}, runner v{})",
                served_id(&v),
                v["endpoint"].as_str().unwrap_or("-"),
                v["pid"],
                v["version"].as_str().unwrap_or("-"),
            );
            Ok(())
        }),
        Some(Command::Ps) => run_verb(async {
            let res = reqwest::get(format!("{}/api/runners", manager_url()))
                .await
                .map_err(connect_hint)?;
            let v = expect_ok(res).await?;
            let rows = v.as_array().cloned().unwrap_or_default();
            if rows.is_empty() {
                println!("no models running (start one: paddock serve <model>)");
                return Ok(());
            }
            println!(
                "{:<7} {:<8} {:<28} {:<10} {:<9} {:<8} {:<5} ENDPOINT",
                "PORT", "PID", "MODEL", "STATUS", "ORIGIN", "VERSION", "PIN"
            );
            for row in &rows {
                print_runner_row(row);
            }
            Ok(())
        }),
        Some(Command::Pin { port, off }) => run_verb(async move {
            let client = reqwest::Client::new();
            let res = client
                .post(format!("{}/api/runners/{port}/pin", manager_url()))
                .json(&serde_json::json!({ "pinned": !off }))
                .send()
                .await
                .map_err(connect_hint)?;
            let v = expect_ok(res).await?;
            println!(
                "port {port}: {}",
                if v["pinned"].as_bool().unwrap_or(false) {
                    "pinned"
                } else {
                    "unpinned"
                }
            );
            Ok(())
        }),
        Some(Command::Start { target }) => run_verb(async move {
            let port = if let Ok(p) = target.parse::<u16>() {
                p
            } else {
                let res = reqwest::get(format!("{}/api/servers", manager_url()))
                    .await
                    .map_err(connect_hint)?;
                let v = expect_ok(res).await?;
                let rows: Vec<serde_json::Value> = v.as_array().cloned().unwrap_or_default();
                // start targets what is not already running
                let stopped: Vec<serde_json::Value> = rows
                    .into_iter()
                    .filter(|r| !r["running"].as_bool().unwrap_or(false))
                    .collect();
                resolve_target(&target, &stopped, "configured (stopped)")?
            };
            println!(
                "starting endpoint {port} from its config file (model load can take a while)..."
            );
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(3600))
                .build()
                .map_err(|e| e.to_string())?;
            let res = client
                .post(format!("{}/api/servers/{port}/start", manager_url()))
                .send()
                .await
                .map_err(connect_hint)?;
            let v = expect_ok(res).await?;
            println!(
                "serving {} on {}  (pid {})",
                served_id(&v),
                v["endpoint"].as_str().unwrap_or("-"),
                v["pid"],
            );
            Ok(())
        }),
        Some(Command::Stop { target, timeout_ms }) => run_verb(async move {
            let port = if let Ok(p) = target.parse::<u16>() {
                p
            } else {
                let res = reqwest::get(format!("{}/api/runners", manager_url()))
                    .await
                    .map_err(connect_hint)?;
                let v = expect_ok(res).await?;
                let rows: Vec<serde_json::Value> = v.as_array().cloned().unwrap_or_default();
                resolve_target(&target, &rows, "running")?
            };
            let mut url = format!("{}/api/runners/{port}", manager_url());
            if let Some(t) = timeout_ms {
                url.push_str(&format!("?timeout_ms={t}"));
            }
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .map_err(|e| e.to_string())?;
            let res = client.delete(url).send().await.map_err(connect_hint)?;
            let v = expect_ok(res).await?;
            println!(
                "port {port}: {} (still configured - `paddock start {port}` brings it back)",
                v["outcome"].as_str().unwrap_or("stopped")
            );
            Ok(())
        }),
        Some(Command::Switch {
            port,
            model,
            max_ctx,
            max_batch,
        }) => run_verb(async move {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(3600))
                .build()
                .map_err(|e| e.to_string())?;
            // START from the ENDPOINT'S own SPEC, not from these three flags.
            //
            // `switch` reads an absent OWNED key as "cleared" - right for the
            // Studio, whose edit form surfaces every one of them, and wrong
            // here: this verb changes which model a port serves and says
            // nothing about the rest, so sending only what was typed silently
            // removed the endpoint's kv_cache_dtype, spec policy, MCP
            // connectors, forensics and web-search settings. (api_key, gpu and
            // fp8_native survived only because supervisor::switch carries
            // those four forward by hand.)
            //
            // So fetch, overlay, resend whole. `hash` rides along as
            // expect_config_hash for the same optimistic-concurrency check the
            // edit page gets: a file that moved between the read and the write
            // refuses instead of clobbering.
            let mut body = serde_json::json!({});
            let mut expect_hash = None;
            if let Ok(res) = client
                .get(format!("{}/api/servers/{port}/file", manager_url()))
                .send()
                .await
                && res.status().is_success()
                && let Ok(v) = res.json::<serde_json::Value>().await
            {
                expect_hash = v["hash"].as_str().map(str::to_owned);
                if v["spec"].is_object() {
                    body = v["spec"].clone();
                }
            }
            let o = body.as_object_mut().expect("json! makes an object");
            o.insert("model".into(), serde_json::json!(model));
            // Only what was actually typed overrides the file; `None` here
            // means "not mentioned", which is the whole point of this block.
            if let Some(c) = max_ctx {
                o.insert("max_ctx".into(), serde_json::json!(c));
            }
            if let Some(b) = max_batch {
                o.insert("max_batch".into(), serde_json::json!(b));
            }
            // The budget is a per-MODEL quantity - it answers "how much does
            // this model need at this envelope". Carrying the outgoing model's
            // cage onto the incoming one is how a 9B inherits a 27B's grant, or
            // worse, a 27B inherits a 9B's and cannot start. Dropping it lets
            // admission re-grant, which is what runners_switch's own comment
            // says a takeover should get.
            o.remove("vram_budget");
            if let Some(h) = expect_hash {
                o.insert("expect_config_hash".into(), serde_json::json!(h));
            }
            println!("port {port}: stopping the model it serves now, then loading {model}...");
            let res = client
                .post(format!("{}/api/runners/{port}/switch", manager_url()))
                .json(&body)
                .send()
                .await
                .map_err(connect_hint)?;
            let v = expect_ok(res).await?;
            println!(
                "serving {} on {}  (pid {})",
                served_id(&v),
                v["endpoint"].as_str().unwrap_or("-"),
                v["pid"],
            );
            Ok(())
        }),
        Some(Command::Logs {
            port,
            manager,
            runner,
            all: _,
            follow,
            tail,
            no_history,
        }) => {
            run_verb(async move {
                let target = if manager {
                    "manager".to_owned()
                } else if let Some(p) = runner.or(port) {
                    p.to_string()
                } else {
                    "all".to_owned()
                };
                let url = format!(
                    "{}/api/logs?target={target}&follow={follow}&tail={}&history={}",
                    manager_url(),
                    tail.unwrap_or(200),
                    !no_history
                );
                let res = reqwest::get(url).await.map_err(connect_hint)?;
                if !res.status().is_success() {
                    let v: serde_json::Value = res.json().await.unwrap_or(serde_json::Value::Null);
                    return Err(v
                        .pointer("/error/message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("no logs")
                        .to_owned());
                }
                // Print chunks as they arrive - with --follow this runs until
                // Ctrl+C; without it the manager closes after the history.
                use futures::StreamExt;
                use std::io::Write;
                let mut stream = res.bytes_stream();
                let mut out = std::io::stdout();
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.map_err(|e| e.to_string())?;
                    out.write_all(&chunk).map_err(|e| e.to_string())?;
                    out.flush().ok();
                }
                Ok(())
            })
        }
        Some(Command::Pull { id }) => {
            let mut cfg = paddock_manager::config::Config::default();
            if let Err(e) = cfg.merge_env() {
                eprintln!("config error: {e}");
                return std::process::ExitCode::from(2);
            }
            if !cli.model_dir.is_empty() {
                cfg.model_dirs = cli.model_dir;
            }
            let models_dir = cfg
                .model_dirs
                .first()
                .cloned()
                .unwrap_or_else(|| std::path::PathBuf::from("./models"));
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("failed to start async runtime: {e}");
                    return std::process::ExitCode::FAILURE;
                }
            };
            let registry = paddock_manager::registry::Registry::new(models_dir.clone())
                .with_cc(paddock_manager::readiness::probe_for_backend(&cfg.device).cc)
                .with_backend(&cfg.device);
            println!("pulling {id} into {} ...", models_dir.display());
            match rt.block_on(registry.resolve(&id, None, true, None)) {
                Ok(Some(r)) => {
                    println!("done: {}", r.weights.display());
                    if let Some(m) = r.mmproj {
                        // "encoder", not "vision encoder": this is the mmproj
                        // companion whichever sense it serves, and on
                        // Qwen3-ASR it hears rather than sees.
                        println!("      {} (mmproj encoder)", m.display());
                    }
                    if let Some(m) = r.mtp {
                        println!("      {} (MTP drafter)", m.display());
                    }
                    // an image lane's pieces: the DiT above, and these two
                    // it cannot render without
                    if let Some(m) = r.text_encoder {
                        println!("      {} (text encoder)", m.display());
                    }
                    if let Some(m) = r.vae {
                        println!("      {} (VAE)", m.display());
                    }
                    println!("serve it: paddock-runner -m {id}");
                    std::process::ExitCode::SUCCESS
                }
                Ok(None) => {
                    eprintln!("unknown model id {id:?} - this build's catalog:");
                    for m in &registry.catalog().models {
                        eprintln!("  {}", m.id);
                    }
                    std::process::ExitCode::from(2)
                }
                Err(e) => {
                    eprintln!("pull failed: {e}");
                    std::process::ExitCode::FAILURE
                }
            }
        }
        None => {
            let mut cfg = paddock_manager::config::Config::default();
            if let Err(e) = cfg.merge_env() {
                eprintln!("config error: {e}");
                return std::process::ExitCode::from(2);
            }
            if let Some(h) = cli.host {
                cfg.host = h;
            }
            if let Some(p) = cli.port {
                cfg.port = p;
            }
            if cli.trusted_proxy {
                cfg.trusted_proxy = true;
            }
            if !cli.model_dir.is_empty() {
                cfg.model_dirs = cli.model_dir;
            }
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("failed to start async runtime: {e}");
                    return std::process::ExitCode::FAILURE;
                }
            };
            match rt.block_on(paddock_manager::run(cfg)) {
                Ok(()) => std::process::ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("manager error: {e}");
                    std::process::ExitCode::FAILURE
                }
            }
        }
    }
}
