//! The paddock **manager** - the control plane. Supervisor + catalog +
//! collector + the resident API client; the Studio is its UI. Owns the one
//! SQLite, model/pack/runner downloads, and NVML device telemetry. It never
//! terminates, proxies, or forwards an inference request - clients hit runner
//! endpoints directly (doc §3: manager-as-client is not a proxy).
//!
//! Contains zero inference code: upgrading the manager can't touch serving.

pub mod api;
pub mod artifacts;
pub mod cloud;
pub mod cloud_loop;
pub mod collector;
pub mod config;
pub mod connectors;
pub mod elections;
pub mod estimate;
pub mod feedback;
pub mod forensics;
pub mod graph;
pub mod hostmem;
pub mod inspect;
pub mod logs;
pub mod nvml;
pub mod oauth;
pub mod push;
pub mod readiness;
pub mod registry;
pub mod routes;
pub mod static_assets;
pub mod store;
pub mod supervisor;
pub mod telemetry;
pub mod updates;
pub mod usage;

use std::sync::Arc;

use config::Config;
use routes::AppState;

/// Bind and serve the manager (Studio + its API) until stopped.
pub async fn run(cfg: Config) -> Result<(), Box<dyn std::error::Error>> {
    let db = Arc::new(
        store::Store::open(&store::default_db_path())
            .map_err(|e| format!("failed to open store: {e}"))?,
    );

    // Can this machine serve at all, and on what silicon? Probed first because
    // everything below wants the answer  - including the registry,
    // whose default-weights resolution is compute-capability-aware (NVFP4 is
    // the default on Blackwell, Q8_0 elsewhere). Never fails, never blocks:
    // hardware does not change under a running process.
    let readiness = Arc::new(readiness::probe());

    // Model registry (compiled-in manifest): the catalog the Studio browses and
    // pulls from. The origin (Cloudflare R2) is a dumb file host.
    let registry = Arc::new(
        registry::Registry::new(
            cfg.model_dirs
                .first()
                .cloned()
                .unwrap_or_else(|| std::path::PathBuf::from("./models")),
        )
        .with_cc(readiness.cc),
    );
    tracing::info!(
        models = registry.catalog().models.len(),
        dir = %registry.models_dir().display(),
        "model registry ready (embedded manifest)"
    );

    // Auth policy mirrors the runner's: explicit key wins; a non-loopback bind
    // with no key auto-generates and requires one; loopback with no key = none.
    let auth_key = match cfg.api_key.as_deref().filter(|s| !s.is_empty()) {
        Some(k) => Some(k.to_owned()),
        None if !cfg.host.is_loopback() => match db.create_api_key("auto (network bind)") {
            Ok((_, key)) => {
                tracing::warn!("network bind with no key - generated one (shown below)");
                Some(key)
            }
            Err(e) => {
                // Fail closed. Serving a network bind keyless because the key
                // store hiccupped is the one outcome worse than not serving.
                return Err(format!(
                    "network bind ({}) with no api key, and generating one failed: {e}. Refusing to start without auth: set PADDOCK_MANAGER_API_KEY, or bind loopback (--host 127.0.0.1)",
                    cfg.host
                )
                .into());
            }
        },
        None => None,
    };

    // Log what that probe found (the probe itself ran above, before the
    // registry, so its cc could steer default-weights resolution).
    match readiness.state {
        readiness::State::Ready => tracing::info!(
            card = readiness.card.as_deref().unwrap_or("?"),
            "graphics card supported - models can run on this computer"
        ),
        readiness::State::Untested => tracing::warn!(
            card = readiness.card.as_deref().unwrap_or("?"),
            "graphics card found, but paddock has not finished testing it - \
             models will run, stamped UNVALIDATED in their logs; speed and \
             stability on this card are unmeasured"
        ),
        readiness::State::DriverTooOld => tracing::warn!(
            card = readiness.card.as_deref().unwrap_or("?"),
            driver = readiness.driver.as_deref().unwrap_or("?"),
            needs = %readiness.cuda_needed,
            "graphics driver is too old for this build - update it and models can run"
        ),
        readiness::State::NoCard => tracing::warn!(
            "no usable NVIDIA graphics card found - models cannot run on this \
             computer, but cloud models work normally"
        ),
    }

    // Device telemetry: NVML runs in exactly one process per box - this one
    // (doc §9). Runners self-report from their allocator ledger instead.
    //
    // Not even attempted without a card. The probe above has already asked
    // NVML the same question, so starting the sampler anyway meant a second
    // init, a thread that samples nothing forever, and - the part a user
    // actually sees - a `NVML unavailable ... LoadLibraryExW failed` line
    // logged as if something had gone wrong, one line above the line that
    // calmly explains there is no NVIDIA card. Nothing went
    // wrong. It is a computer without an NVIDIA GPU.
    let gpu = if readiness.state == readiness::State::NoCard {
        telemetry::Telemetry::disabled()
    } else {
        telemetry::start()
    };

    // Runner supervision (doc §3): spawn/stop/takeover + §6.1 reconciliation.
    let data_dir = Config::data_dir();
    // Desired state (§11.2): the manager-written managed.toml election set.
    let elections = Arc::new(elections::Elections::load(data_dir.join("managed.toml")));
    let supervisor = Arc::new(supervisor::Supervisor::new(
        supervisor::SpawnDefaults {
            runner_bin: cfg.runner_bin.clone(),
            device: cfg.device.clone(),
            // explicit config wins; else probe packs/cuda beside the exe and
            // under the data root - the portable "drop the pack in" path
            kernel_pack: cfg
                .kernel_pack
                .clone()
                .or_else(config::autodetect_kernel_pack),
            models_dirs: cfg.model_dirs.clone(),
            logs_dir: data_dir.join("logs"),
            runners_dir: data_dir.join("runners"),
            work_dir: data_dir.clone(),
            base_port: cfg.runner_base_port,
            health_timeout: std::time::Duration::from_secs(cfg.spawn_timeout_s),
        },
        registry.clone(),
        Some(elections.clone()),
        Some(gpu.clone()),
    ));
    // Meet existing runners: re-attach/adopt, never restart (§6.1).
    supervisor.reconcile().await;

    // §11.4 flow 5: respawn elected runners that are not already serving.
    // Background + sequential - model loads take minutes and must not block
    // the control plane's bind, and loading two models at once on one card
    // is an avoidable OOM. Reconcile ran first, so an adopted survivor on an
    // elected port is left alone. A failed respawn keeps its election (the
    // operator sees the error; next boot retries) - silently dropping desired
    // state would be a silent failure.
    {
        let sup = supervisor.clone();
        let el = elections.clone();
        let causes = db.clone();
        tokio::spawn(async move {
            for e in el.list() {
                if sup.is_serving(e.port).await {
                    tracing::info!(port = e.port, model = %e.model, "election already serving (reconcile reclaimed it) - left alone");
                    continue;
                }
                tracing::info!(port = e.port, model = %e.model, "respawning elected runner");
                // The collector will see the new generation and consume this
                // note into its lifecycle band's start_cause.
                let _ = causes.note_start_cause(e.port, "boot-election");
                // Launch the FILE verbatim (servers/<port>.toml - the truth):
                // a respawn never re-renders the config, so hand-edits -
                // including fields the manager's editor doesn't know - are
                // honored exactly as written.
                if let Err(err) = sup.start_config(e.port).await {
                    tracing::error!(port = e.port, model = %e.model, %err, "elected runner failed to respawn (election kept; will retry next boot)");
                }
            }
        });
    }

    // The collector: event-ring subscription (§8.1) + the usage-metrics
    // scrape, per the activity mode - `aggregates`
    // keeps the rollups while writing no per-request rows, and `off` runs
    // neither: not recording is first-class (§8.7).
    match cfg.activity {
        config::ActivityMode::Off => {
            tracing::info!("activity persistence disabled - nothing is written to SQLite");
        }
        mode => collector::start(db.clone(), cfg.activity_retention_days, mode),
    }

    // §9 reconciliation gauge: NVML outside view vs runner allocator ledgers.
    // Nothing to compare on a box with no card, so nothing wakes up to try.
    let recon = if readiness.state == readiness::State::NoCard {
        telemetry::no_reconciler()
    } else {
        telemetry::start_reconciler(gpu.clone(), supervisor.clone())
    };

    // The box's own TLS identity. Established before the bind so
    // the banner can print the real scheme.
    //
    // Not a setting, and deliberately so: browsers withhold the microphone,
    // the clipboard and `crypto.randomUUID` from any origin that is not a
    // secure context, so a Studio opened on a LAN address over plain http is
    // not merely unencrypted - it is missing features, and dictation cannot be
    // shimmed back the way the other two were. A security property nobody
    // switches on is one that is off.
    //
    // A failure here degrades to cleartext and says so. It never stops the
    // manager: a box that cannot write a key file is still a box that should
    // come up.
    let tls = match paddock_tls::Identity::load_or_create(&data_dir.join("tls")) {
        Ok(id) => {
            if id.issued {
                tracing::info!(names = ?id.names, "issued this computer's certificate");
            }
            Some(id)
        }
        Err(e) => {
            tracing::error!(%e, "could not establish a TLS identity - serving over plain http, \
                 which means no microphone and no clipboard for any browser that is not on this \
                 computer");
            None
        }
    };

    // What Host headers the MCP endpoints will answer to (rmcp 3.x checks
    // this to stop DNS rebinding). On a loopback bind the three loopback names
    // are the whole truth; on a network bind the box legitimately answers to
    // its hostname and interface addresses too, and omitting them would break
    // every off-box MCP client rather than protect anyone.
    let mcp_allowed_hosts: Vec<String> = if cfg.host.is_loopback() {
        vec!["localhost".into(), "127.0.0.1".into(), "::1".into()]
    } else {
        paddock_tls::box_names().into_iter().collect()
    };
    let state = Arc::new(AppState {
        db,
        auth_key: auth_key.clone(),
        trusted_proxy: cfg.trusted_proxy,
        mcp_allowed_hosts,
        graphs: Arc::new(graph::Bridge::new()),
        gpu,
        readiness,
        registry,
        probes: estimate::ProbeCache::default(),
        max_ctx: cfg.max_ctx,
        max_batch: cfg.max_batch,
        supervisor,
        elections: Some(elections),
        recon,
        push: crate::push::Hub::new(),
        admission: tokio::sync::Mutex::new(()),
        updates: crate::updates::Cache::default(),
        update_dl: std::sync::Mutex::new(None),
        tls: tls.as_ref().map(|id| {
            Arc::new(routes::TlsFacts {
                root_pem: id.root_pem.clone(),
                fingerprint: id.fingerprint.clone(),
                names: id.names.clone(),
            })
        }),
    });
    // server-push watcher: sweeps fleet state only while a Studio tab holds
    // the /api/events stream open, publishing on change (crate::push)
    crate::push::spawn_watcher(state.clone());

    // A one-time CUDA fetch used to start here, for a box that had
    // a supported card and no maths libraries. Paddock ships no NVIDIA
    // redistributable and fetches none, so there is no such box: a supported
    // card with a current driver is ready the moment the manager is.

    let addr = std::net::SocketAddr::new(cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, tls = tls.is_some(), "paddock manager listening");
    print_banner(&cfg, auth_key.as_deref(), tls.as_ref());

    // One port, both schemes, chosen per connection by the first byte
    // (paddock_tls::serve). `axum::serve` cannot do that, so the accept loop
    // is ours - which also means re-attaching ConnectInfo by hand, since
    // auth_mw exempts loopback peers and a request without it looks remote.
    paddock_tls::serve(
        listener,
        routes::router(state),
        tls.map(|t| t.server),
        cfg.port,
    )
    .await?;
    Ok(())
}

fn print_banner(cfg: &Config, auth_key: Option<&str>, tls: Option<&paddock_tls::Identity>) {
    use std::io::IsTerminal;
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let (dim, cyan, reset) = if color {
        ("\x1b[2m", "\x1b[36m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let version = paddock_admin::version::LONG;
    let host = if cfg.host.is_unspecified() {
        "localhost".to_string()
    } else {
        cfg.host.to_string()
    };
    let scheme = if tls.is_some() { "https" } else { "http" };
    println!();
    println!("  {cyan}█▀▄ ▄▀▄ █▀▄ █▀▄ ▄▀▄ ▄▀▀ █▄▀{reset}   {dim}v{version} (manager){reset}");
    println!(
        "  {cyan}█▀▀ █▀█ █ █ █ █ █ █ █   █▀▄{reset}   {dim}Copyright (c) 2026 Truespar. MIT OR Apache-2.0.{reset}"
    );
    println!("  {cyan}▀   ▀ ▀ ▀▀▀ ▀▀▀ ▀▀▀ ▀▀▀ ▀ ▀{reset}");
    println!();
    println!(
        "  {dim}Studio{reset}      {cyan}{scheme}://{host}:{}{reset}",
        cfg.port
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
    // where data lives must be STATED, with the mode that chose it:
    // a portable unzip and a dev checkout on the same box resolve
    // differently, and silence here is how the wrong root goes unnoticed
    let (data, how) = paddock_admin::data_root_resolved();
    println!(
        "  {dim}Data{reset}        {} {dim}({how}){reset}",
        data.display()
    );
    let dirs = &cfg.model_dirs;
    let first = dirs
        .first()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    println!("  {dim}Models dir{reset}  {first}");
    match auth_key {
        Some(k) => {
            let scope = if cfg.trusted_proxy {
                "required from everyone (trusted_proxy)"
            } else {
                "required from the network, loopback callers exempt"
            };
            println!("  {dim}API key{reset}     {cyan}{k}{reset}  {dim}{scope}{reset}");
        }
        None => println!("  {dim}Auth{reset}        {dim}none (loopback bind){reset}"),
    }
    // The fingerprint is the only way a person can tell that the root they are
    // about to install came from this computer and not from whatever answered
    // first, so it is printed where they can compare it against the dialog.
    match tls {
        Some(id) => {
            println!(
                "  {dim}Certificate{reset} {dim}sha256 {}{reset}",
                id.fingerprint
            );
            println!("  {dim}            covers {}{reset}", id.names.join(", "));
            // A fingerprint with nowhere to go is trivia. Print the address of
            // the page that hands out the root - but only when another device
            // could reach this machine at all: on a loopback bind nothing off-box
            // can connect, and localhost is already a secure context, so the
            // line would be advice about a problem the reader does not have.
            let reachable = if cfg.host.is_loopback() {
                None
            } else if cfg.host.is_unspecified() {
                // Bound to everything, so ASK which address actually carries
                // traffic rather than taking the first one enumerated - a
                // virtual switch address here would send the reader nowhere.
                paddock_tls::primary_address()
                    .map(|ip| ip.to_string())
                    .filter(|ip| id.names.contains(ip))
                    .or_else(|| {
                        id.names
                            .iter()
                            .find(|n| {
                                n.parse::<std::net::IpAddr>()
                                    .is_ok_and(|ip| !ip.is_loopback())
                            })
                            .cloned()
                    })
            } else {
                Some(cfg.host.to_string())
            };
            if let Some(h) = reachable {
                let h = if h.contains(':') { format!("[{h}]") } else { h };
                // Continuation of the Certificate block rather than a label of
                // its own: it is a fact about this certificate, and the banner
                // keeps one label column.
                println!(
                    "  {dim}            install it on another device:{reset} {cyan}https://{h}:{}/manage/trust{reset}",
                    cfg.port
                );
            }
        }
        None => println!(
            "  {dim}Certificate{reset} {dim}none - browsers away from this computer get no \
             microphone or clipboard{reset}"
        ),
    }
    println!();
    println!(
        "  {dim}Runners are separate processes (`paddock-runner`) - inference bytes never touch the manager{reset}"
    );
    println!("  {dim}Press Ctrl+C to stop the manager (runners keep serving){reset}");
    println!();
}
