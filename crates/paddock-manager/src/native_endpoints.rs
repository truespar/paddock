//! Credential-free native endpoint views and narrowly typed edits. The file is
//! still the authority and the web manager's save/admission/restart path applies
//! the result. Swift never round-trips raw TOML or a saved runner key.
use crate::{routes::AppState, supervisor::Supervisor};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::Digest;
use std::{collections::HashSet, net::IpAddr, sync::Arc};

#[cfg(test)]
#[path = "native_endpoints_tests.rs"]
mod tests;

mod composition;
mod creation_tools;
mod offload;
mod options;
pub use creation_tools::CreationTools;

/// Diagnostic output is not a credential export. Withhold credential-bearing
/// lines wholesale (not a fixed-width mask), plus literal known saved/live
/// secrets. An unreadable existing config fails closed. No raw file crosses ABI.
pub fn safe_log_text(
    supervisor: &Supervisor,
    port: u16,
    text: &str,
    live_key: Option<&str>,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut secrets = Vec::new();
    if let Some(key) = live_key.filter(|v| !v.is_empty()) {
        secrets.push(key.to_owned());
    }
    if supervisor.server_config_path(port).exists() {
        let Ok((raw, _)) = supervisor.read_config_file(port) else {
            return "[Log batch withheld: credential protection unavailable]\n".into();
        };
        let Ok(doc) = toml::from_str::<toml::Value>(&raw) else {
            return "[Log batch withheld: invalid saved configuration]\n".into();
        };
        fn collect(value: &toml::Value, secret: bool, out: &mut Vec<String>) {
            match value {
                toml::Value::String(v) if secret && !v.is_empty() => out.push(v.clone()),
                toml::Value::Table(t) => {
                    for (key, value) in t {
                        let key = key.to_ascii_lowercase();
                        collect(
                            value,
                            secret
                                || [
                                    "key",
                                    "token",
                                    "secret",
                                    "password",
                                    "headers",
                                    "authorization",
                                ]
                                .iter()
                                .any(|s| key.contains(s)),
                            out,
                        );
                    }
                }
                toml::Value::Array(a) => {
                    for value in a {
                        collect(value, secret, out);
                    }
                }
                _ => {}
            }
        }
        collect(&doc, false, &mut secrets);
    }
    let mut out = String::new();
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        let sensitive = [
            "api_key",
            "api-key",
            "apikey",
            "authorization",
            "bearer ",
            "access_token",
            "refresh_token",
            "password",
            "client_secret",
            "cookie:",
        ]
        .iter()
        .any(|s| lower.contains(s));
        if sensitive || secrets.iter().any(|key| line.contains(key)) {
            out.push_str("[Credential-bearing log line withheld]");
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

#[derive(Debug, Deserialize)]
#[serde(
    tag = "field",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Change {
    MaxCtx(Option<usize>),
    MaxBatch(Option<usize>),
    Spec(Option<String>),
    Host(IpAddr),
    ApiKey(String),
    Forensics(bool),
    KvCacheDtype(Option<String>),
    VramBudget(Option<u64>),
    Composition(Composition),
    Runtime(serde_json::Map<String, Value>),
    KvOffload(offload::Settings),
    Residency(paddock_admin::residency::Config),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Composition {
    model: String,
    artifact: String,
    vision: bool,
    audio: Option<bool>,
    drafter: Option<String>,
}
impl Change {
    fn key(&self) -> &'static str {
        match self {
            Self::MaxCtx(_) => "max_ctx",
            Self::MaxBatch(_) => "max_batch",
            Self::Spec(_) => "spec",
            Self::Host(_) => "host",
            Self::ApiKey(_) => "api_key",
            Self::Forensics(_) => "forensics",
            Self::KvCacheDtype(_) => "kv_cache_dtype",
            Self::VramBudget(_) => "vram_budget",
            Self::Composition(_) => "composition",
            Self::Runtime(_) => "runtime",
            Self::KvOffload(_) => "kv_offload",
            Self::Residency(_) => "residency",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Apply {
    Defer,
    Restart,
}

/// Read revision and controls from one file read, never pair the digest of one
/// version with values from a later version. No config values are used as errors.
pub fn projection(supervisor: &Supervisor, port: u16) -> Result<Value, String> {
    let (raw, revision) = supervisor
        .read_config_file(port)
        .map_err(|_| "Saved endpoint settings are unavailable.")?;
    let doc = parse(&raw, port)?;
    let media = supervisor.project_config_text(&raw)?;
    Ok(settings_projection(
        &doc,
        Some(&revision),
        supervisor.kv_offload_supported(&doc),
        (media.vision, media.audio),
        media.vram_budget,
    ))
}

fn settings_projection(
    doc: &toml::Value,
    revision: Option<&str>,
    offload_supported: bool,
    media: (bool, bool),
    memory_limit: Option<u64>,
) -> Value {
    let host = doc
        .get("host")
        .and_then(toml::Value::as_str)
        .unwrap_or("0.0.0.0");
    let max_ctx = doc.get("max_ctx").and_then(toml::Value::as_integer);
    let max_batch = doc.get("max_batch").and_then(toml::Value::as_integer);
    json!({
        "revision": revision,
        "local_only":host.parse::<IpAddr>().is_ok_and(|v|v.is_loopback()),
        "max_ctx":max_ctx, "max_batch":max_batch,
        "settings": {
            "host":host, "max_ctx":max_ctx, "max_batch":max_batch,
            "spec":doc.get("spec").and_then(toml::Value::as_str),
            "no_spec":doc.get("no_spec").and_then(toml::Value::as_bool).unwrap_or(false),
            "runtime_options": options::projection(doc),
            "kv_offload": offload::projection(doc),
            "kv_offload_supported": offload_supported,
            "residency": doc.get("residency").cloned().and_then(|v| v.try_into::<paddock_admin::residency::Config>().ok()),
            "residency_supported": residency_supported(doc),
            "kv_cache_dtype":doc.get("kv_cache_dtype").and_then(toml::Value::as_str),
            "has_api_key":doc.get("api_key").and_then(toml::Value::as_str).is_some_and(|v|!v.is_empty()),
            "vision":media.0,
            "audio":media.1,
            "forensics":doc.get("forensics").and_then(|v|v.get("enabled")).and_then(toml::Value::as_bool).unwrap_or(false),
            "device":doc.get("device").and_then(toml::Value::as_str).unwrap_or("auto"),
            "drafter":doc.get("catalog").and_then(|v| v.get("drafter")).and_then(toml::Value::as_str),
            "vram_budget":memory_limit,
        }
    })
}

/// A read-only draft uses the very same serializer and control schema as Edit.
/// Port zero is an unsaved identity, never a listener or an endpoint file.
async fn initial_config(state: &AppState, model: &str, artifact: &str) -> Result<String, String> {
    let entry = state
        .registry
        .catalog_of(model)
        .ok_or("Select a catalog model.")?;
    let art = entry.artifact(artifact).ok_or("Select catalog weights.")?;
    if art.kind != crate::registry::ArtifactKind::Weights
        || !art.runtime.supports_backend("metal")
        || !state.registry.is_artifact_installed(art)
    {
        return Err("Select downloaded weights compatible with Metal.".into());
    }
    let runtime = art.runtime.for_backend("metal");
    state
        .supervisor
        .render_spec_config(
            0,
            crate::supervisor::SpawnSpec {
                model: model.into(),
                artifact: Some(artifact.into()),
                host: Some(std::net::Ipv4Addr::LOCALHOST.into()),
                max_ctx: Some(runtime.default_envelope().0),
                max_batch: Some(1),
                persist: false,
                pull: false,
                ..Default::default()
            },
        )
        .await
        .map_err(|_| {
            "Cannot prepare this model's settings. Check its compatible companion files.".into()
        })
}

pub async fn prepare(state: &AppState, model: &str, artifact: &str) -> Result<Value, String> {
    let text = initial_config(state, model, artifact).await?;
    let doc = parse(&text, 0)?;
    let media = state.supervisor.project_config_text(&text)?;
    let mut value = settings_projection(
        &doc,
        None,
        state.supervisor.kv_offload_supported(&doc),
        (media.vision, media.audio),
        media.vram_budget,
    );
    let object = value
        .as_object_mut()
        .expect("the settings projection is a JSON object");
    object.insert("port".into(), json!(0));
    object.insert("running".into(), json!(false));
    object.insert("model".into(), json!(model));
    object.insert("artifact".into(), json!(artifact));
    Ok(value)
}

/// Validate all fields through Edit's allowlist before publishing anything.
pub async fn create_config(
    state: &AppState,
    model: &str,
    artifact: &str,
    port: Option<u16>,
    changes: &[Change],
    allow_network: bool,
    tools: Option<CreationTools>,
) -> Result<u16, String> {
    let _guard = crate::connectors::MUTATIONS.lock().await;
    let raw = initial_config(state, model, artifact).await?;
    let content = if changes.is_empty() {
        raw
    } else {
        patch(&raw, 0, changes, allow_network)?
    };
    let content = resolve_changes(state, 0, content, changes).await?;
    let doc = parse(&content, 0)?;
    if !doc
        .get("host")
        .and_then(toml::Value::as_str)
        .and_then(|h| h.parse::<IpAddr>().ok())
        .is_some_and(|h| h.is_loopback())
        && !allow_network
    {
        return Err("Confirm network access before starting this model.".into());
    }
    for key in ["mmproj", "mtp"] {
        if doc
            .get(key)
            .and_then(toml::Value::as_str)
            .is_some_and(|p| !std::path::Path::new(p).is_file())
        {
            return Err(
                "Download the selected vision or speculative companion before starting.".into(),
            );
        }
    }
    let (content, connectors) =
        creation_tools::prepare(state, &content, &tools.unwrap_or_default())?;
    let mut doc: toml::Value = toml::from_str(&content).map_err(|_| "Invalid model settings.")?;
    if !doc
        .get("api_key")
        .and_then(toml::Value::as_str)
        .is_some_and(|k| !k.is_empty())
    {
        doc.as_table_mut()
            .expect("a parsed endpoint document is a table")
            .insert(
                "api_key".into(),
                toml::Value::String(format!("pd-{}", uuid::Uuid::new_v4().simple())),
            );
    }
    let content = toml::to_string(&doc).map_err(|_| "Cannot prepare model settings.")?;
    let port = state.supervisor.create_config_file(port, &content).await?;
    creation_tools::register(state, port, &connectors)?;
    Ok(port)
}

fn parse(raw: &str, port: u16) -> Result<toml::Value, String> {
    if raw.len() > 1024 * 1024 {
        return Err("Endpoint configuration exceeds 1 MiB.".into());
    }
    let doc: toml::Value =
        toml::from_str(raw).map_err(|_| "Saved endpoint settings are not valid TOML.")?;
    if !doc.is_table() || doc.get("port").and_then(toml::Value::as_integer) != Some(i64::from(port))
    {
        return Err("The endpoint file does not declare its expected port. Correct the file before editing.".into());
    }
    Ok(doc)
}

/// Patch only reviewed fields; preserve model identity, towers, drafter paths,
/// tool credentials, custom keys and nested table contents. Verify serialization
/// so TOML array-of-table ordering cannot silently move a new scalar into MCP.
fn patch(raw: &str, port: u16, changes: &[Change], allow_network: bool) -> Result<String, String> {
    let mut want = parse(raw, port)?;
    if changes.is_empty() || changes.len() > 12 {
        return Err("Select the settings to change.".into());
    }
    let mut seen = HashSet::new();
    let mut doc: toml_edit::DocumentMut = raw
        .parse()
        .map_err(|_| "Saved endpoint settings are invalid.")?;
    for change in changes {
        let key = change.key();
        if !seen.insert(key) {
            return Err("A setting was submitted more than once.".into());
        }
        // Registry composition is resolved asynchronously before publication.
        if matches!(change, Change::Composition(_)) {
            continue;
        }
        if let Change::Runtime(values) = change {
            options::patch(&mut doc, &mut want, values)?;
            continue;
        }
        // A reviewed policy replaces the legacy spelling too. Keeping
        // no_spec=true alongside spec=on would make the form lie at restart.
        if matches!(change, Change::Spec(_)) {
            want.as_table_mut()
                .ok_or("Invalid endpoint settings.")?
                .remove("no_spec");
            doc.remove("no_spec");
        }
        let value = match change {
            Change::MaxCtx(v) => {
                if v.is_some_and(|n| !(256..=1_048_576).contains(&n)) {
                    return Err("Context must be between 256 and 1048576 tokens.".into());
                }
                v.map(|v| toml::Value::Integer(v as i64))
            }
            Change::MaxBatch(v) => {
                if v.is_some_and(|n| !(1..=256).contains(&n)) {
                    return Err("Concurrency must be between 1 and 256.".into());
                }
                v.map(|v| toml::Value::Integer(v as i64))
            }
            Change::Spec(v) => {
                if v.as_deref()
                    .is_some_and(|v| !["on", "off", "adaptive", "auto", "ladder"].contains(&v))
                {
                    return Err("Choose On, Off or Adaptive speculation.".into());
                }
                v.clone().map(toml::Value::String)
            }
            Change::Host(v) => {
                if v.is_multicast() {
                    return Err("A multicast address cannot host a model endpoint.".into());
                }
                Some(toml::Value::String(v.to_string()))
            }
            Change::ApiKey(v) => {
                if !(16..=4096).contains(&v.len()) || !v.bytes().all(|v| v.is_ascii_graphic()) {
                    return Err(
                        "Use an API key of 16-4096 printable ASCII characters without spaces."
                            .into(),
                    );
                }
                Some(toml::Value::String(v.clone()))
            }
            Change::Forensics(enabled) => {
                let mut block = want
                    .get(key)
                    .and_then(toml::Value::as_table)
                    .cloned()
                    .unwrap_or_default();
                block.insert("enabled".into(), toml::Value::Boolean(*enabled));
                Some(toml::Value::Table(block))
            }
            Change::KvCacheDtype(v) => {
                if v.as_deref()
                    .is_some_and(|v| !["auto", "f16", "f32"].contains(&v))
                {
                    return Err(
                        "Choose backend-native conversation memory for this Metal export.".into(),
                    );
                }
                v.clone().map(toml::Value::String)
            }
            Change::VramBudget(v) => {
                if v.is_some_and(|v| !(256..=1_048_576).contains(&v)) {
                    return Err(
                        "Memory budget must be 256-1048576 MiB, or the shared default.".into(),
                    );
                }
                v.map(|v| toml::Value::Integer(v as i64))
            }
            Change::KvOffload(v) => Some(offload::value(&want, v)?),
            Change::Residency(v) => {
                v.validate()?;
                if v.enabled() && !residency_supported(&want) {
                    return Err(
                        "On-demand loading supports Whisper on CUDA/Metal and DiffusionGemma on Metal.".into(),
                    );
                }
                Some(toml::Value::try_from(v).map_err(|_| "Invalid model residency policy.")?)
            }
            Change::Composition(_) | Change::Runtime(_) => unreachable!(),
        };
        if let Some(value) = value {
            want.as_table_mut()
                .ok_or("Invalid endpoint settings.")?
                .insert(key.into(), value.clone());
            // Parse a one-field document to retain TOML's exact value types.
            let text = toml::to_string(&toml::Table::from_iter([(key.to_owned(), value)]))
                .map_err(|_| "Cannot prepare the endpoint setting.")?;
            let field: toml_edit::DocumentMut = text
                .parse()
                .map_err(|_| "Cannot prepare the endpoint setting.")?;
            doc[key] = field[key].clone();
        } else {
            want.as_table_mut()
                .ok_or("Invalid endpoint settings.")?
                .remove(key);
            doc.remove(key);
        }
    }
    let host: IpAddr = want
        .get("host")
        .and_then(toml::Value::as_str)
        .unwrap_or("0.0.0.0")
        .parse()
        .map_err(|_| "The saved bind address is invalid.")?;
    if !host.is_loopback()
        && (!allow_network
            || !want
                .get("api_key")
                .and_then(toml::Value::as_str)
                .is_some_and(|v| v.len() >= 16))
    {
        return Err("Network access requires an explicit confirmation and an API key of at least 16 characters.".into());
    }
    let text = doc.to_string();
    if toml::from_str::<toml::Value>(&text).ok().as_ref() == Some(&want) {
        Ok(text)
    } else {
        toml::to_string(&want).map_err(|_| "Cannot serialize endpoint settings.".into())
    }
}

pub(crate) fn residency_supported(doc: &toml::Value) -> bool {
    let Some(path) = doc
        .get("model")
        .and_then(toml::Value::as_str)
        .map(std::path::Path::new)
    else {
        return false;
    };
    let metal = doc.get("device").and_then(toml::Value::as_str) == Some("metal");
    if metal && path.is_dir() {
        return paddock_models::mlx::DiffusionConfig::read(path).is_ok();
    }
    paddock_models::mapped::MappedGguf::open(path)
        .ok()
        .is_some_and(|map| {
            map.gguf().architecture() == Some("whisper")
                || (metal && map.gguf().architecture() == Some("diffusion-gemma"))
        })
}

pub(crate) fn validate_residency(
    doc: &toml::Value,
) -> Result<paddock_admin::residency::Config, String> {
    let policy = doc
        .get("residency")
        .cloned()
        .map(|v| v.try_into::<paddock_admin::residency::Config>())
        .transpose()
        .map_err(|_| "Invalid model residency policy.")?
        .unwrap_or_default();
    policy.validate()?;
    if policy.enabled() && !residency_supported(doc) {
        return Err(
            "On-demand loading supports Whisper on CUDA/Metal and DiffusionGemma on Metal.".into(),
        );
    }
    Ok(policy)
}

pub(crate) fn on_demand_config(raw: &str) -> bool {
    toml::from_str::<toml::Value>(raw)
        .ok()
        .and_then(|doc| validate_residency(&doc).ok())
        .is_some_and(|policy| policy.load == paddock_admin::residency::LoadPolicy::OnDemand)
}

pub async fn edit(
    state: Arc<AppState>,
    port: u16,
    revision: String,
    pid: Option<u32>,
    changes: Vec<Change>,
    apply: Apply,
    allow_network: bool,
) -> Result<String, String> {
    // Native search/scope saves share this gate. A save must not erase a new
    // connector credential while waiting for admission or a drain.
    let _guard = crate::connectors::MUTATIONS.lock().await;
    let (raw, current) = state
        .supervisor
        .read_config_file(port)
        .map_err(|_| "The endpoint was removed or its settings are unavailable.")?;
    if current != revision {
        return Err(
            "This endpoint changed. Reload its settings before saving; your draft has been kept."
                .into(),
        );
    }
    let content = if changes.is_empty() && matches!(apply, Apply::Restart) {
        // Explicitly apply already-saved pending settings, still revision/PID guarded.
        raw.clone()
    } else {
        patch(&raw, port, &changes, allow_network)?
    };
    let content = resolve_changes(&state, port, content, &changes).await?;
    finish_edit(state, port, revision, pid, apply, content).await
}

async fn resolve_changes(
    state: &AppState,
    port: u16,
    mut content: String,
    changes: &[Change],
) -> Result<String, String> {
    if let Some(composition) = changes.iter().find_map(|c| match c {
        Change::Composition(v) => Some(v),
        _ => None,
    }) {
        content = composition::resolve(state, port, &content, composition).await?;
    }
    let updated = parse(&content, port)?;
    if updated
        .get("kv_offload")
        .and_then(|v| v.get("enabled"))
        .and_then(toml::Value::as_bool)
        == Some(true)
        && !state.supervisor.kv_offload_supported(&updated)
    {
        return Err("This model's Metal graph does not support KV offloading. Disable it before changing to this model family.".into());
    }
    let spec = state
        .supervisor
        .spec_from_config_text(&content)
        .map_err(|_| "The endpoint configuration cannot be applied.")?;
    // F32 is a fixed checkpoint ABI, not a new precision toggle for existing
    // GGUF/BF16 graphs. Permit the catalog contract or a validated local Bonsai
    // directory; unknown paths must not acquire an unsupported configuration.
    if spec.kv_cache_dtype.as_deref() == Some("f32") {
        let catalog_f32 = state
            .registry
            .catalog_of(&spec.model)
            .and_then(|entry| spec.artifact.as_deref().and_then(|id| entry.artifact(id)))
            .is_some_and(|art| {
                art.runtime.supports_backend("metal")
                    && art.runtime.for_backend("metal").kv_cache_dtype.as_deref() == Some("f32")
            });
        let local_f32 = updated
            .get("model")
            .and_then(toml::Value::as_str)
            .is_some_and(|path| {
                paddock_models::bonsai::BonsaiConfig::read(std::path::Path::new(path)).is_ok()
            });
        if !catalog_f32 && !local_f32 {
            return Err("F32 conversation memory requires a native Bonsai checkpoint.".into());
        }
    }
    if let Some(entry) = state.registry.catalog_of(&spec.model)
        && let Some(art) = spec.artifact.as_deref().and_then(|id| entry.artifact(id))
    {
        if !art.runtime.supports_backend("metal") {
            return Err("This endpoint is not a Metal model.".into());
        }
        let runtime = art.runtime.for_backend("metal");
        if let Some(memory) = &runtime.memory
            && (spec.max_ctx.is_some_and(|v| v as u64 > memory.max_ctx)
                || spec.max_batch.is_some_and(|v| v as u64 > memory.max_batch))
        {
            return Err(format!(
                "This Metal export supports at most {} context tokens and {} concurrent requests.",
                memory.max_ctx, memory.max_batch
            ));
        }
        if let Some(required) = &runtime.kv_cache_dtype
            && spec
                .kv_cache_dtype
                .as_ref()
                .is_some_and(|dtype| dtype != required)
        {
            return Err("Choose backend-native conversation memory for this export.".into());
        }
        if changes.iter().any(|c| matches!(c, Change::Forensics(true)))
            && !(runtime.embedded_vision || parse(&content, port)?.get("mmproj").is_some())
        {
            return Err(
                "Forensics requires vision input. Enable Vision or select a vision model.".into(),
            );
        }
    }
    // Enabling speculation may need a separate drafter. Resolve through the
    // same composition logic as web; do not quietly serve dense after promising
    // a speculative policy. An explicitly configured drafter is preserved.
    if changes
        .iter()
        .any(|c| matches!(c, Change::Spec(v) if v.as_deref() != Some("off")))
        && parse(&content, port)?.get("mtp").is_none()
    {
        let rendered = state.supervisor.render_spec_config(port, spec).await
            .map_err(|_| "The speculative composition is unavailable. Check that its compatible drafter is downloaded.")?;
        let rendered: toml::Value =
            toml::from_str(&rendered).map_err(|_| "Cannot resolve the speculative composition.")?;
        if let Some(path) = rendered.get("mtp").and_then(toml::Value::as_str) {
            if !std::path::Path::new(path).is_file() {
                return Err(
                    "Download the compatible drafter before enabling speculative decoding.".into(),
                );
            }
            let mut doc: toml_edit::DocumentMut = content
                .parse()
                .map_err(|_| "Cannot prepare the speculative composition.")?;
            doc["mtp"] = toml_edit::value(path);
            let mut expected = parse(&content, port)?;
            expected
                .as_table_mut()
                .ok_or("Invalid configuration.")?
                .insert("mtp".into(), toml::Value::String(path.into()));
            content = doc.to_string();
            if toml::from_str::<toml::Value>(&content).ok().as_ref() != Some(&expected) {
                content = toml::to_string(&expected)
                    .map_err(|_| "Cannot serialize the speculative composition.")?;
            }
        }
    }
    Ok(content)
}

async fn finish_edit(
    state: Arc<AppState>,
    port: u16,
    revision: String,
    pid: Option<u32>,
    apply: Apply,
    content: String,
) -> Result<String, String> {
    if matches!(apply, Apply::Restart) {
        let current = state
            .supervisor
            .list()
            .await
            .into_iter()
            .find(|r| r.port == port);
        if pid.is_none() || current.as_ref().map(|r| r.pid) != pid {
            return Err("The reviewed runner changed or stopped. Reload before restarting. Nothing was saved.".into());
        }
    }
    let deferred = matches!(apply, Apply::Defer);
    let written_hash = crate::registry::hex(&sha2::Sha256::digest(content.as_bytes()));
    let response = crate::routes::save_endpoint_file(
        state.clone(),
        port,
        content,
        revision,
        deferred,
        if deferred { None } else { pid },
    )
    .await;
    if response.status().is_success() {
        let live = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .is_some_and(|value| value["applied"] == "live");
        return Ok(if deferred {
            "Settings saved without restarting. Live settings apply automatically; other changes apply on the next start."
        } else if live {
            "Settings saved and applied live. The model did not need to restart."
        } else {
            "Settings saved and the model restarted on the same port."
        }
        .into());
    }
    let saved = state.supervisor.config_file_hash(port).as_deref() == Some(&written_hash);
    Err(if saved {
        "The settings were saved, but the model could not restart. Review its status and runner log before retrying."
    } else if response.status().as_u16() == 507 {
        "This configuration does not fit available model memory. Nothing was saved or automatically stopped."
    } else {
        "The endpoint changed or its settings could not be applied. Your draft is kept; reload the endpoint to review its current state."
    }.into())
}

pub async fn remove(state: Arc<AppState>, port: u16, revision: &str) -> Result<String, String> {
    let _guard = crate::connectors::MUTATIONS.lock().await;
    if state.supervisor.config_file_hash(port).as_deref() != Some(revision) {
        return Err("The endpoint changed or was removed. Refresh before removing it.".into());
    }
    state
        .supervisor
        .remove_config_checked(port, Some(revision))
        .await
        .map_err(|_| "Could not remove this endpoint. Stop it first and refresh its settings.")?;
    Ok("Endpoint configuration removed. Model weights and conversations were kept.".into())
}
