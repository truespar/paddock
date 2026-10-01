//! The Swift application is the manager. This private C ABI embeds the same
//! product state as the web host. Native management uses explicit commands;
//! the full Studio uses an authenticated, app-lifetime loopback host. No
//! separate service or certificate. Runner keys remain in Rust.
use axum::{body::Body, http::Request};
use paddock_manager::{HostMode, ManagerCore, config::Config};
use std::sync::{Arc, Mutex};
use std::{
    ffi::{CString, c_char, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    time::Duration,
};
use tower::ServiceExt;

mod catalog;
mod chat;
mod commands;
mod connections;
mod downloads;
mod integrations;
mod logs;
mod maintenance;
mod studio;

const MAX_SNAPSHOT: usize = 8 * 1024 * 1024;

struct Desktop {
    runtime: Option<tokio::runtime::Runtime>,
    core: ManagerCore,
    jobs: Arc<Mutex<commands::Jobs>>,
    chats: paddock_manager::studio_chat::Sessions,
    studio: Mutex<Option<studio::Host>>,
    connections: connections::Sessions,
    integrations: integrations::Sessions,
    logs: logs::Sessions,
    maintenance: maintenance::Sessions,
}

impl Desktop {
    fn open() -> Result<Self, String> {
        // Deliberately no web-manager env merge: no public bind, API key,
        // trusted proxy or remote-manager settings in an embedded host.
        let config = Config {
            device: "metal".into(),
            runner_bin: bundled_runner(),
            ..Config::default()
        };
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("paddock-core")
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        let core = runtime
            .block_on(paddock_manager::initialize(&config, HostMode::Desktop))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            runtime: Some(runtime),
            core,
            jobs: Arc::new(Mutex::new(commands::Jobs::default())),
            chats: Default::default(),
            studio: Mutex::new(None),
            connections: Default::default(),
            integrations: Default::default(),
            logs: Default::default(),
            maintenance: Default::default(),
        })
    }

    fn snapshot(&self) -> Result<String, String> {
        let app = paddock_manager::routes::router(self.core.state.clone());
        self.runtime
            .as_ref()
            .ok_or("The management core is closed")?
            .block_on(async {
                let raw = tokio::time::timeout(Duration::from_secs(15), snapshot(app))
                    .await
                    .map_err(|_| "The management snapshot timed out".to_string())??;
                let mut value: serde_json::Value =
                    serde_json::from_str(&raw).map_err(|e| e.to_string())?;
                value["jobs"] = serde_json::to_value(
                    self.jobs
                        .lock()
                        .map_err(|_| "Model job state unavailable")?
                        .snapshot(),
                )
                .map_err(|e| e.to_string())?;
                if let Some(servers) = value["servers"].as_array_mut() {
                    for server in servers {
                        if let Some(port) =
                            server["port"].as_u64().and_then(|p| u16::try_from(p).ok())
                        {
                            if server["running"].as_bool() == Some(true) {
                                let client = paddock_admin::client::AdminClient::new(port);
                                if let Ok(Ok(status)) = tokio::time::timeout(
                                    Duration::from_secs(1),
                                    client.config_status(),
                                )
                                .await
                                {
                                    server["runtime_state"] =
                                        serde_json::to_value(status).unwrap_or_default();
                                }
                            }
                            match paddock_manager::native_endpoints::projection(
                                &self.core.state.supervisor,
                                port,
                            ) {
                                Ok(serde_json::Value::Object(fields)) => {
                                    if let Some(object) = server.as_object_mut() {
                                        object.extend(fields);
                                    }
                                }
                                Err(message) => {
                                    server["config_error"] = serde_json::json!(message);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                let json = value.to_string();
                if json.len() > MAX_SNAPSHOT {
                    return Err("Management snapshot exceeds its size limit".into());
                }
                Ok(json)
            })
    }

    fn submit(&self, bytes: &[u8]) -> Result<String, String> {
        let command = serde_json::from_slice(bytes).map_err(|_| "Invalid native model command")?;
        if let commands::Command::Prepare { model, artifact } = &command {
            let value = self
                .runtime
                .as_ref()
                .ok_or("The management core is closed")?
                .block_on(paddock_manager::native_endpoints::prepare(
                    &self.core.state,
                    model,
                    artifact,
                ))?;
            return serde_json::to_string(&value).map_err(|_| "Cannot read model defaults.".into());
        }
        if let commands::Command::Poll { id } = command {
            return serde_json::to_string(
                &self
                    .jobs
                    .lock()
                    .map_err(|_| "Model job state unavailable")?
                    .get(id)?,
            )
            .map_err(|_| "Cannot read the model operation receipt.".into());
        }
        let job = commands::submit(
            self.runtime
                .as_ref()
                .ok_or("The management core is closed")?,
            self.jobs.clone(),
            self.core.state.clone(),
            command,
        )?;
        serde_json::to_string(&job).map_err(|e| e.to_string())
    }

    fn chat(&self, bytes: &[u8]) -> Result<String, String> {
        let command = serde_json::from_slice(bytes).map_err(|_| "Invalid native chat command")?;
        let mut value = self
            .runtime
            .as_ref()
            .ok_or("The management core is closed")?
            .block_on(self.chats.command(self.core.state.clone(), command))?;
        chat::project(&mut value)?;
        let json = value.to_string();
        if json.len() > MAX_SNAPSHOT {
            return Err("Native chat response exceeds 8 MiB".into());
        }
        Ok(json)
    }

    fn studio(&self, bytes: &[u8]) -> Result<String, String> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Open {
            assets: Option<std::path::PathBuf>,
        }
        let request: Open =
            serde_json::from_slice(bytes).map_err(|_| "Invalid bundled Studio request")?;
        let mut host = self.studio.lock().map_err(|_| "Studio host unavailable")?;
        if let Some(host) = host.as_ref() {
            if let Some(assets) = request.assets {
                host.mount_assets(assets)?;
            }
        } else {
            *host = Some(
                self.runtime
                    .as_ref()
                    .ok_or("The management core is closed")?
                    .block_on(studio::Host::start(self.core.state.clone(), request.assets))?,
            );
        }
        Ok(host
            .as_ref()
            .ok_or("Studio host unavailable")?
            .descriptor()
            .to_string())
    }
}

fn bundled_runner() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let macos = exe.parent()?;
    let contents = macos.parent()?;
    (macos.file_name()? == "MacOS"
        && contents.file_name()? == "Contents"
        && contents.parent()?.extension()? == "app")
        .then(|| contents.join("Helpers/paddock-runner"))
}

impl Drop for Desktop {
    fn drop(&mut self) {
        // Cancel collectors/watchers before releasing directory ownership.
        // Existing independent inference endpoints are not killed on app quit.
        if let Some(runtime) = self.runtime.take() {
            if let Ok(host) = self.studio.get_mut() {
                *host = None;
            }
            self.chats.cancel_all();
            self.connections.cancel_checks();
            self.integrations.cancel_reads();
            self.logs.close_all();
            self.maintenance.close_all();
            runtime.block_on(async {
                let _ = tokio::time::timeout(Duration::from_secs(2), async {
                    while self.chats.active() {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                })
                .await;
            });
            // A lifecycle transaction must settle before dropping the runtime:
            // aborting a spawn future can strand a half-loaded child. The UI
            // warns before quit while active; this is its last-resort safety net.
            runtime.block_on(async {
                self.core.state.registry.pause_downloads().await;
                while self.connections.saving() || self.integrations.saving() {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                while self.jobs.lock().is_ok_and(|jobs| jobs.active()) {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            });
            runtime.shutdown_timeout(Duration::from_secs(2));
        }
    }
}

async fn snapshot(app: axum::Router) -> Result<String, String> {
    // Reuse the tested projections, in memory. This is not a HTTP request over
    // a socket, and no arbitrary path/method is exposed to the Swift/UI layer.
    let mut result = serde_json::Map::new();
    for (key, path) in [
        ("identity", "/api/server"),
        ("readiness", "/api/readiness"),
        ("catalog", "/api/models/catalog"),
        ("runners", "/api/runners"),
        ("servers", "/api/servers"),
        ("gpu", "/api/gpu"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::get(path)
                    .body(Body::empty())
                    .map_err(|e| e.to_string())?,
            )
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!(
                "Management projection {key}: {}",
                response.status()
            ));
        }
        let bytes = axum::body::to_bytes(response.into_body(), MAX_SNAPSHOT)
            .await
            .map_err(|e| e.to_string())?;
        let mut value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        project_for_ui(key, &mut value)?;
        result.insert(key.into(), value);
    }
    let json = serde_json::to_string(&result).map_err(|e| e.to_string())?;
    if json.len() > MAX_SNAPSHOT {
        return Err("Management snapshot exceeds its size limit".into());
    }
    Ok(json)
}

/// The browser's privileged management projections can contain runner config,
/// API keys, search keys and MCP headers. Never rely on Swift ignoring unknown
/// JSON fields: remove them before allocation/copy across the native boundary.
/// Allowlisting also prevents newly added management fields leaking by default.
fn project_for_ui(key: &str, value: &mut serde_json::Value) -> Result<(), String> {
    fn keep(value: &mut serde_json::Value, names: &[&str]) -> Result<(), String> {
        let object = value.as_object_mut().ok_or("Invalid management object")?;
        object.retain(|name, _| names.contains(&name.as_str()));
        Ok(())
    }
    match key {
        "gpu" => {
            keep(value, &["available", "ts", "gpus", "reconciliation"])?;
            if let Some(gpus) = value
                .get_mut("gpus")
                .and_then(serde_json::Value::as_array_mut)
            {
                for gpu in gpus {
                    keep(
                        gpu,
                        &[
                            "index",
                            "name",
                            "util_gpu",
                            "mem_used",
                            "mem_total",
                            "power_w",
                            "temp_c",
                            "metal",
                        ],
                    )?;
                    if let Some(metal) = gpu.get_mut("metal").filter(|m| !m.is_null()) {
                        keep(
                            metal,
                            &[
                                "unified_memory_total",
                                "recommended_working_set",
                                "thermal_pressure",
                                "memory_pressure",
                                "counter_sets",
                            ],
                        )?;
                    }
                }
            }
            if let Some(recon) = value.get_mut("reconciliation").filter(|r| !r.is_null()) {
                keep(recon, &["ts", "runners"])?;
                if let Some(runners) = recon
                    .get_mut("runners")
                    .and_then(serde_json::Value::as_array_mut)
                {
                    for runner in runners {
                        keep(runner, &["port", "pid", "self_mem", "metal", "engine"])?;
                        if let Some(metal) = runner.get_mut("metal").filter(|m| !m.is_null()) {
                            keep(
                                metal,
                                &[
                                    "allocated_bytes",
                                    "completed_commands",
                                    "gpu_seconds_total",
                                    "last_command_ms",
                                ],
                            )?;
                        }
                        if let Some(engine) = runner.get_mut("engine").filter(|e| !e.is_null()) {
                            keep(
                                engine,
                                &[
                                    "tok_s",
                                    "phase",
                                    "active_slots",
                                    "kv_used",
                                    "kv_total",
                                    "tokens_total",
                                ],
                            )?;
                        }
                    }
                }
            }
        }
        "servers" => {
            for server in value
                .as_array_mut()
                .ok_or("Invalid configured endpoint inventory")?
            {
                keep(
                    server,
                    &[
                        "port",
                        "model",
                        "artifact",
                        "running",
                        "display",
                        "vendor",
                        "capability",
                    ],
                )?;
            }
        }
        "runners" => {
            for runner in value.as_array_mut().ok_or("Invalid runner inventory")? {
                keep(
                    runner,
                    &[
                        "port",
                        "pid",
                        "status",
                        "model",
                        "embedder",
                        "asr",
                        "aligner",
                        "image",
                        "reader",
                        "tabular",
                        "display",
                        "endpoint",
                        "version",
                        "in_flight",
                        "vendor",
                        "spec",
                        "origin",
                        "uptime_s",
                        "pinned",
                        "vram",
                    ],
                )?;
            }
        }
        "identity" => {
            keep(value, &["role", "version", "build", "registry"])?;
            if let Some(registry) = value.get_mut("registry") {
                keep(
                    registry,
                    &["enabled", "models_dir", "disk_free", "disk_total"],
                )?;
            }
        }
        "readiness" => keep(value, &["backend", "state", "card", "generation", "os"])?,
        // Compiled-in public model/artifact metadata; no user credentials.
        "catalog" => {}
        _ => return Err("Unsupported native projection".into()),
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "C" fn paddock_desktop_abi_version() -> u32 {
    12
}

/// # Safety
/// Same serial lifetime as snapshot. Typed input 1..16 KiB. Background reads
/// return bounded, credential-free JSON; polling never waits for runner I/O.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_maintenance(
    core: *mut c_void,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        let core = unsafe { core.cast::<Desktop>().as_ref() }.ok_or("Core is not open")?;
        if bytes.is_null() || len == 0 || len > 16 * 1024 {
            return Err("Invalid management command length".into());
        }
        let command = serde_json::from_slice(unsafe { std::slice::from_raw_parts(bytes, len) })
            .map_err(|_| "Invalid native management command")?;
        let _entered = core.runtime.as_ref().ok_or("Core is closed")?.enter();
        let json = core
            .maintenance
            .execute(core.core.state.clone(), command)?
            .to_string();
        if json.len() > MAX_SNAPSHOT {
            return Err("Management response exceeds its size limit.".into());
        }
        CString::new(json).map_err(|_| "Invalid management response".into())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe {
                report(error, message);
            }
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// Same serial core lifetime as snapshot. Bounded typed subscriptions only;
/// no paths, caller-selected URLs or log deletion. File I/O stays off this ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_logs(
    core: *mut c_void,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        let core = unsafe { core.cast::<Desktop>().as_ref() }.ok_or("Core is not open")?;
        if bytes.is_null() || len == 0 || len > 1024 {
            return Err("Invalid log command length".into());
        }
        let command = serde_json::from_slice(unsafe { std::slice::from_raw_parts(bytes, len) })
            .map_err(|_| "Invalid native log command")?;
        let runtime = core.runtime.as_ref().ok_or("Core is closed")?;
        let _entered = runtime.enter();
        let json = core
            .logs
            .execute(core.core.state.clone(), command)?
            .to_string();
        if json.len() > 4 * 1024 * 1024 {
            return Err("Log batch exceeds its limit".into());
        }
        CString::new(json).map_err(|_| "Invalid log response".into())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe { report(error, message) };
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// Serial core lifetime; typed JSON <=128 KiB. Stored credentials never return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_integrations(
    core: *mut c_void,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        let core = unsafe { core.cast::<Desktop>().as_ref() }.ok_or("Core is not open")?;
        if bytes.is_null() || len == 0 || len > 128 * 1024 {
            return Err("Invalid tools command length".into());
        }
        let command = serde_json::from_slice(unsafe { std::slice::from_raw_parts(bytes, len) })
            .map_err(|_| "Invalid native tools command")?;
        let _entered = core.runtime.as_ref().ok_or("Core is closed")?.enter();
        let origin = core
            .studio
            .lock()
            .map_err(|_| "Studio host unavailable")?
            .as_ref()
            .map(|h| h.origin.clone());
        CString::new(
            core.integrations
                .execute(core.core.state.clone(), command, origin)?
                .to_string(),
        )
        .map_err(|_| "Invalid tools response".into())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe { report(error, message) };
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// Same serial core lifetime as snapshot. Typed JSON <=128 KiB. Incoming keys
/// go directly to Rust, never a web view; results contain metadata/receipts only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_connections(
    core: *mut c_void,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        let core = unsafe { core.cast::<Desktop>().as_ref() }.ok_or("Core is not open")?;
        if bytes.is_null() || len == 0 || len > 128 * 1024 {
            return Err("Invalid connection command length".into());
        }
        let command = serde_json::from_slice(unsafe { std::slice::from_raw_parts(bytes, len) })
            .map_err(|_| "Invalid native connection command")?;
        let runtime = core.runtime.as_ref().ok_or("Core is closed")?;
        let _entered = runtime.enter();
        let json = core
            .connections
            .execute(core.core.state.clone(), command)?
            .to_string();
        if json.len() > MAX_SNAPSHOT {
            return Err("Connection response exceeds its size limit".into());
        }
        CString::new(json).map_err(|_| "Invalid connection response".into())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe {
                report(error, message);
            }
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// Same serial lifetime as snapshot; typed input <=16 KiB. Calls admit work
/// without waiting for bytes. The light list projection never probes runners.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_downloads(
    core: *mut c_void,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        let core = unsafe { core.cast::<Desktop>().as_ref() }.ok_or("Core is not open")?;
        if bytes.is_null() || len == 0 || len > 16384 {
            return Err("Invalid download command length".into());
        }
        let command = serde_json::from_slice(unsafe { std::slice::from_raw_parts(bytes, len) })
            .map_err(|_| "Invalid native download command")?;
        let runtime = core.runtime.as_ref().ok_or("Core is closed")?;
        let _entered = runtime.enter();
        CString::new(downloads::execute(&core.core.state.registry, command)?)
            .map_err(|e| e.to_string())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe {
                report(error, message);
            }
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// Same serial core lifetime as snapshot. Input names the app's bundled asset
/// directory (1..8192 UTF-8 JSON bytes). The session in the result is for Swift
/// to install as an HttpOnly cookie, never expose to JavaScript or logs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_studio(
    core: *mut c_void,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        let core =
            unsafe { core.cast::<Desktop>().as_ref() }.ok_or("The management core is not open")?;
        if bytes.is_null() || len == 0 || len > 8192 {
            return Err("Invalid Studio request length".into());
        }
        CString::new(core.studio(unsafe { std::slice::from_raw_parts(bytes, len) })?)
            .map_err(|e| e.to_string())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe {
                report(error, message);
            }
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// Same serial core lifetime as snapshot. Input is typed UTF-8 JSON, 1..256 KiB;
/// output is at most 8 MiB and must be freed with paddock_desktop_string_free.
/// Poll never waits for tokens. Accepted sends run on Rust's runtime; cancellation
/// drops only that request, never the independently serving runner process.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_chat(
    core: *mut c_void,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        let core =
            unsafe { core.cast::<Desktop>().as_ref() }.ok_or("The management core is not open")?;
        if bytes.is_null() || len == 0 || len > 256 * 1024 {
            return Err("Invalid native chat request length".into());
        }
        CString::new(core.chat(unsafe { std::slice::from_raw_parts(bytes, len) })?)
            .map_err(|e| e.to_string())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe {
                report(error, message);
            }
            std::ptr::null_mut()
        }
    }
}

/// Anonymous catalog access, independent of any open management core.
/// # Safety
/// `bytes` must be readable for `len` bytes (1..1024); `error` must be NULL or
/// point to a writable NULL string pointer. Free returned strings with this ABI.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_browse(
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        if bytes.is_null() || len == 0 || len > 1024 {
            return Err("Invalid cloud catalog request length".into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(bytes, len) };
        CString::new(catalog::browse(bytes)?).map_err(|e| e.to_string())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe { report(error, message) };
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `core` and `error` follow snapshot's contract. `bytes` must point to `len`
/// readable bytes for this call; commands are bounded to 64 KiB. The returned
/// job receipt is Rust-owned. Submission does not wait for the operation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_submit(
    core: *mut c_void,
    bytes: *const u8,
    len: usize,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        let core =
            unsafe { core.cast::<Desktop>().as_ref() }.ok_or("The management core is not open")?;
        if bytes.is_null() || len == 0 || len > 64 * 1024 {
            return Err("Invalid native command length".into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(bytes, len) };
        CString::new(core.submit(bytes)?).map_err(|e| e.to_string())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe {
                report(error, message);
            }
            std::ptr::null_mut()
        }
    }
}

fn boundary<T>(operation: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or_else(|_| Err("The embedded management core failed unexpectedly".into()))
}

// Every exported pointer contract is also documented in include/paddock_desktop.h.
// The Swift bridge owns these values on one serial queue; Rust never calls Swift.
unsafe fn report(error: *mut *mut c_char, message: String) {
    if !error.is_null() {
        let text = CString::new(message.replace('\0', "�")).expect("NUL removed");
        unsafe {
            *error = text.into_raw();
        }
    }
}

/// # Safety
/// `error` must be NULL or a writable pointer to a NULL string pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_open(error: *mut *mut c_char) -> *mut c_void {
    match boundary(Desktop::open) {
        Ok(core) => Box::into_raw(Box::new(core)).cast(),
        Err(message) => {
            unsafe {
                report(error, message);
            }
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `core` must be a live open result, with no concurrent close. `error` as above.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_snapshot(
    core: *mut c_void,
    error: *mut *mut c_char,
) -> *mut c_char {
    match boundary(|| {
        let core =
            unsafe { core.cast::<Desktop>().as_ref() }.ok_or("The management core is not open")?;
        CString::new(core.snapshot()?).map_err(|e| e.to_string())
    }) {
        Ok(value) => value.into_raw(),
        Err(message) => {
            unsafe {
                report(error, message);
            }
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `value` must be NULL or an unfreed string returned by this library.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_string_free(value: *mut c_char) {
    if !value.is_null() {
        drop(unsafe { CString::from_raw(value) });
    }
}

/// # Safety
/// `core` must be NULL or a live open result. No calls may race or follow close.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn paddock_desktop_close(core: *mut c_void) {
    if !core.is_null() {
        let _ = boundary(|| {
            drop(unsafe { Box::from_raw(core.cast::<Desktop>()) });
            Ok(())
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_projection_keeps_measurements_but_never_unknown_fields() {
        let mut value = serde_json::json!({
            "available": true, "ts": 10, "secret": "private-top",
            "gpus": [{"index": 0, "name": "Apple M5 Max", "secret": "private-device",
                "metal": {"unified_memory_total": 128, "recommended_working_set": 96,
                    "thermal_pressure": "fair", "memory_pressure": null,
                    "counter_sets": ["timestamp"], "secret": "private-hardware"}}],
            "reconciliation": {"ts": 10, "secret": "private-recon", "runners": [{
                "port": 11540, "pid": 42, "secret": "private-runner",
                "engine": {"tok_s": 30.0, "secret": "private-engine"},
                "metal": {"allocated_bytes": 12, "completed_commands": 30,
                    "gpu_seconds_total": 0.8, "last_command_ms": 22.0, "secret": "private-metal"}
            }]}
        });
        project_for_ui("gpu", &mut value).unwrap();
        assert_eq!(value["gpus"][0]["metal"]["thermal_pressure"], "fair");
        assert!(value["gpus"][0]["metal"]["memory_pressure"].is_null());
        assert_eq!(value["gpus"][0]["metal"]["counter_sets"][0], "timestamp");
        assert_eq!(
            value["reconciliation"]["runners"][0]["metal"]["allocated_bytes"],
            12
        );
        assert!(!value.to_string().contains("private-"));
    }
    #[test]
    fn stopped_speech_capability_survives_without_exporting_config() {
        let mut servers = serde_json::json!([{
            "port": 11542, "model": "whisper", "running": false,
            "capability": ["transcription"], "vendor": "OpenAI",
            "weights": "private-path", "api_key": "private-key", "config": "private-config"
        }]);
        project_for_ui("servers", &mut servers).unwrap();
        assert_eq!(
            servers[0]["capability"],
            serde_json::json!(["transcription"])
        );
        assert_eq!(servers[0]["running"], false);
        assert!(!servers.to_string().contains("private-"));
    }
    #[test]
    fn runner_credentials_never_cross_the_native_boundary() {
        let mut runners = serde_json::json!([{
            "port": 12345, "pid": 42, "reader": "laya", "status": "running",
            "endpoint": "http://127.0.0.1:12345", "in_flight": 2,
            "api_key": "test-only-root-secret",
            "future_secret_field": "test-only-future-secret",
            "config": { "api_key": "test-only-runner-secret", "web_search_api_key": "test-only-search-secret",
                "mcp_servers": [{"headers": {"Authorization": "Bearer test-only-mcp-secret"}}] }
        }]);
        project_for_ui("runners", &mut runners).unwrap();
        assert_eq!(runners[0]["port"], 12345);
        assert_eq!(runners[0]["in_flight"], 2);
        assert_eq!(runners[0]["reader"], "laya");
        assert!(!runners.to_string().contains("secret"));
        assert!(runners[0].get("config").is_none());
        assert!(project_for_ui("arbitrary", &mut runners).is_err());
        assert!(project_for_ui("runners", &mut serde_json::json!({})).is_err());
    }
    #[test]
    fn abi_null_and_panic_paths_do_not_cross_boundary() {
        assert_eq!(paddock_desktop_abi_version(), 12);
        assert!(boundary::<()>(|| panic!("test-only")).is_err());
        unsafe {
            let mut error = std::ptr::null_mut();
            assert!(paddock_desktop_snapshot(std::ptr::null_mut(), &mut error).is_null());
            assert!(!error.is_null());
            paddock_desktop_string_free(error);
            paddock_desktop_string_free(std::ptr::null_mut());
            paddock_desktop_close(std::ptr::null_mut());
        }
    }
    #[tokio::test]
    async fn embedded_projection_is_the_real_catalog_without_a_listener() {
        let result = snapshot(paddock_manager::routes::router(std::sync::Arc::new(
            paddock_manager::routes::AppState::for_tests(),
        )))
        .await
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(value["identity"]["role"], "manager");
        assert_eq!(value["catalog"]["schema"], 3);
        assert!(value["catalog"]["models"].as_array().unwrap().len() > 10);
    }
}
