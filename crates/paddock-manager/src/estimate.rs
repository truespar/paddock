//! `/api/models/estimate` - honest will-it-fit for the whole catalog at once.
//!
//! The models page needs a VRAM figure per row, and the only figure worth
//! showing is one derived from the actual file. So this endpoint answers for
//! every catalog model in one request, at a caller-supplied (or
//! server-default) concurrency, and says plainly which rows it could not
//! measure rather than inventing a number for them.
//!
//! Context is an OUTPUT, not a parameter: each model's trained window and the
//! cache the card can afford decide it together, so the response carries both
//! the value and a whole `curve` of it against concurrency.
//!
//! Geometry comes from `paddock_models::probe` (bounded header read, never the
//! weights) and the arithmetic from `paddock_estimator`. Nothing here does math
//! of its own - a second copy of the KV formula is exactly how the old
//! `total_size * 1.2 + 1 GB` guess drifted 2.8× away from reality.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use paddock_estimator::{Device, Envelope, KvDtype, ModelKind, ModelShape};
use paddock_models::probe::{ModelReport, probe_path};
use serde::Deserialize;

/// Probing reads up to a 256 MB header prefix per file, so a naive per-request
/// sweep of a 14-model catalog would be gigabytes of I/O every time the user
/// nudges the context slider. Keyed by path + mtime + len so a re-pulled or
/// swapped file re-probes instead of serving a stale shape.
#[derive(Default)]
pub struct ProbeCache(Mutex<HashMap<(PathBuf, u64, u64), Arc<ModelReport>>>);

impl ProbeCache {
    pub fn get(&self, path: &Path) -> Option<Arc<ModelReport>> {
        let md = std::fs::metadata(path).ok()?;
        let stamp = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let key = (path.to_path_buf(), stamp, md.len());
        if let Some(hit) = self.0.lock().ok()?.get(&key) {
            return Some(hit.clone());
        }
        let report = Arc::new(probe_path(path).ok()?);
        self.0.lock().ok()?.insert(key, report.clone());
        Some(report)
    }
}

#[derive(Debug, Deserialize)]
pub struct EstimateQuery {
    /// Edit/switch preview: this instance releases its own allocation first.
    freeing_port: Option<u16>,
    /// Concurrent sequences to price. Defaults to the configured `max_batch`.
    /// There is deliberately no `ctx` parameter - context is derived from what
    /// the card can back at this concurrency, capped by the model's own
    /// trained window. Asking callers to pick it from a list was both
    /// arbitrary and wrong at the edges.
    batch: Option<u64>,
    /// `f16` or `fp8_e4m3`. Absent = each model's own default (its catalog
    /// `kv_default`, what an `auto` runner serves).
    kv: Option<String>,
    /// Price speculative decode: the drafter's resident bytes plus the wider
    /// verify logits plane. Off by default. Per-MODEL, since the drafter (and
    /// whether there is a separate one at all) differs per row.
    spec: Option<bool>,
    /// Which GPU to price against (NVML index, default 0). Reclaimable VRAM
    /// counts only the runners attributed to this device.
    gpu: Option<u32>,
    /// Price the vision/audio tower. Defaults to true, which is both the old
    /// behaviour and the safe direction: a caller that says nothing gets the
    /// heavier answer. `false` mirrors the start form's vision switch, which
    /// the supervisor really honours (`supervisor.rs`: `spec.vision ==
    /// Some(false)` drops the mmproj), so leaving it out of the estimate
    /// over-charged every vision model by its whole tower.
    vision: Option<bool>,
    /// Price a split-tower model's audio tower (EmbeddingGemma 2): the start
    /// form's Audio switch. Absent = the artifact's catalog default (off),
    /// which is what the supervisor serves for an absent `spec.audio`.
    audio: Option<bool>,
    /// This device's compute capability as `"major.minor"` (e.g. `"8.6"`), so
    /// the estimate can price the KV width the RUNNER will actually serve
    /// rather than the one that was asked for. Optional: an older Studio
    /// sends nothing and gets the request honoured verbatim, which is the
    /// earlier behaviour rather than a new guess.
    cc: Option<String>,
    /// Prefix-cache offload budget in GiB - the form's "In memory" field,
    /// which becomes `[kv_offload] ram_gb`. Absent or 0 = no tier.
    ///
    /// It has to reach the estimate for the same reason `spec` does: arming
    /// the tier reserves device staging out of the VRAM the pool is sized
    /// from, so an estimate that ignored it would draw a context the runner
    /// then seats smaller. The GiB figure itself is not VRAM and never enters
    /// a device total - it rides through to `host_ram` so the form can show
    /// what the feature actually costs the machine.
    offload_ram_gb: Option<f64>,
    /// Ceiling on what this endpoint may hold, in MiB - the form's "how much
    /// of the card" choice, which becomes the config file's `vram_budget`.
    ///
    /// Without it the estimate priced against all free VRAM while the spawn
    /// obeyed the ceiling, so a 20 GB limit drew a 37 GB endpoint. The budget
    /// is exactly "act as if the card had only this much free", so that is
    /// what it does here.
    budget: Option<u64>,
}

/// Capabilities served by one pass per input with nothing cached between
/// calls: embedding, rerank and alignment encoders, decision models (typed
/// questions in, one distribution per question out), tabular predictors
/// (labelled rows in, the missing column out), dense prediction (image chips
/// in, rasters out) and image generation (a prompt in, a render out).
/// None of them holds a KV cache, so none is priced with decode terms -
/// weights, companions, workspace, and that is all. Pricing one as generative
/// is how a 0.6B embedding model once came out "needing" 124 GB.
fn is_single_pass(capability: &str) -> bool {
    matches!(
        capability,
        "embeddings"
            | "rerank"
            | "alignment"
            | "diarization"
            | "decision"
            | "tabular"
            | "segmentation"
            | "masks"
            | "image-generation"
    )
}

/// The estimator's kind for a model with these capabilities. Image
/// generation is its own single-pass kind (its workspace is sized by a
/// picture, not a token window); the encoder capabilities share `Encoder`;
/// everything else decodes.
pub(crate) fn kind_for(capabilities: &[String]) -> ModelKind {
    if capabilities.iter().any(|c| c == "image-generation") {
        ModelKind::Image
    } else if capabilities.iter().any(|c| is_single_pass(c)) {
        ModelKind::Encoder
    } else {
        ModelKind::Generative
    }
}

/// Is this model (catalog id or a path the catalog recognises) an
/// image-generation lane? The one kind with no token envelope at all: a
/// picture is one pass, nothing batched or windowed, so `max_ctx` /
/// `max_batch` and the chat-side connectors mean nothing to it and the
/// spawn path keeps them out of its config.
pub(crate) fn is_image_lane(reg: &crate::registry::Registry, model: &str) -> bool {
    reg.catalog_of(model)
        .is_some_and(|m| kind_for(&m.capability) == ModelKind::Image)
}

/// `"8.6"` -> `(8, 6)`. Anything unparseable is None, and None never gates -
/// same fail-open stance as the runner's own device singleton: refusing to
/// price because a string was malformed helps nobody.
fn parse_cc(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once('.')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// Concurrency steps the curve is sampled at, so the UI can show the
/// context/concurrency trade-off without a request per point.
const CURVE: [u64; 6] = [1, 2, 4, 8, 16, 32];

/// Bytes of weights a runner's model would need reloaded after eviction -
/// the computed restore cost. Resolution mirrors spawn's: catalog id (the
/// installed - else default - weights artifact) -> installed model name ->
/// filesystem path. None when the model string can't be resolved (adopted
/// runner with a foreign path, deleted file).
fn resolve_weight_bytes(state: &crate::routes::AppState, model: &str) -> Option<u64> {
    resolve_weight_bytes_for(state, model, None)
}

/// Like `resolve_weight_bytes`, honoring an explicit weights-artifact choice
/// (the admission guard prices the artifact a spawn actually selects).
pub(crate) fn resolve_weight_bytes_for(
    state: &crate::routes::AppState,
    model: &str,
    artifact: Option<&str>,
) -> Option<u64> {
    resolve_weights_for(state, model, artifact).map(|(_, bytes, _, _)| bytes)
}

/// Full weights resolution for admission: the on-disk GGUF path (probeable),
/// the artifact's byte size, how paddock would serve it (encoders hold no
/// decode cache - pricing them with one is the 124-GB-embedding-model bug),
/// and the artifact's declared serving workspace (0 when it declares none, or
/// when the model isn't a catalog row at all - a raw path carries no such
/// manifest data, and under-charging there is the pre-existing behaviour).
/// Same resolution order as spawn: catalog id -> installed model name -> raw
/// path. The path may not exist yet (pre-download admission) - callers probe
/// and fall back to weights-only arithmetic when it doesn't.
pub(crate) fn resolve_weights_for(
    state: &crate::routes::AppState,
    model: &str,
    artifact: Option<&str>,
) -> Option<(PathBuf, u64, ModelKind, u64)> {
    if let Some(m) = state
        .registry
        .catalog()
        .models
        .iter()
        .find(|m| m.id == model)
    {
        let a = match artifact {
            Some(id) => m.artifact(id),
            None => None,
        }
        .or_else(|| {
            m.weights()
                .find(|a| state.registry.is_artifact_installed(a))
        })
        .or_else(|| m.default_weights())?;
        let kind = kind_for(&m.capability);
        let path = state.registry.models_dir().join(&a.files.first()?.dest);
        return Some((path, a.total_size(), kind, a.workspace.unwrap_or(0)));
    }
    let store = paddock_models::ModelStore::new(state.supervisor.models_dirs().to_vec());
    if let Ok(models) = store.list()
        && let Some(m) = models.into_iter().find(|m| m.id == model)
    {
        let bytes = checkpoint_file_bytes(&m.path)?;
        return Some((m.path, bytes, ModelKind::Generative, 0));
    }
    let p = Path::new(model);
    if p.exists() {
        let bytes = checkpoint_file_bytes(p)?;
        return Some((p.to_path_buf(), bytes, ModelKind::Generative, 0));
    }
    None
}

/// Imported checkpoints need a bounded grant too. For a local MLX folder,
/// charge every safetensors shard, never the directory inode's byte length.
fn checkpoint_file_bytes(path: &Path) -> Option<u64> {
    if path.is_file() {
        return Some(std::fs::metadata(path).ok()?.len());
    }
    let mut bytes = 0u64;
    for entry in std::fs::read_dir(path).ok()? {
        let path = entry.ok()?.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            bytes = bytes.checked_add(std::fs::metadata(path).ok()?.len())?;
        }
    }
    (bytes > 0).then_some(bytes)
}

/// Resident bytes of the vision tower this model would serve with, 0 for a
/// text-only one. Same precedence spawn uses (`Registry::resolve` wires the
/// first INSTALLED `Vision` artifact, and the row-level download bundle pulls
/// the DEFAULT one), so the estimate prices the mmproj that will actually
/// load rather than the largest one on offer.
///
/// Charged when the caller asks for vision, exactly like the drafter is
/// charged when it asks for spec - the start form has a vision switch and the
/// supervisor honours it (`supervisor.rs`: `spec.vision == Some(false)` drops
/// the mmproj before spawn), so a vision-off server genuinely does not pay
/// this. This used to be unconditional on the belief that no toggle existed,
/// which over-charged every vision-off estimate by the whole tower - 0.9 GB on
/// qwen3.8-27b. The CALLER decides; this function still answers
/// "what would the tower cost", which is a question with one answer.
///
/// Pricing a not-yet-downloaded tower over-states by exactly its file size,
/// which is the safe direction for a fit check and is reported as its own line
/// either way.
///
/// The file's bytes are the resident bytes, for every family: all three vision
/// loaders keep their weight planes at 16 bits and accumulate in f32 (granite
/// gemma4 + qwen3.5/3.6). This used to be a floor - those
/// two widened every plane to f32 and held about twice the file - and the fix
/// went into the loaders rather than a per-family multiplier here, which would
/// have been a magic constant in the wrong crate that went stale the moment
/// they were fixed.
pub(crate) fn tower_bytes_for(
    m: &crate::registry::CatalogModel,
    reg: &crate::registry::Registry,
    weights: Option<&crate::registry::CatalogArtifact>,
) -> u64 {
    use crate::registry::ArtifactKind;
    // Vision and Audio towers are the same KIND of cost - an mmproj companion
    // held from startup to shutdown - so they are charged by one rule. Only
    // the capability they imply differs, and that is the catalog's business,
    // not this function's.
    let tower = || {
        m.artifacts
            .iter()
            .filter(|a| weights.is_none_or(|w| w.runtime.allows_companion(&a.id)))
            .filter(|a| matches!(a.kind, ArtifactKind::Vision | ArtifactKind::Audio))
    };
    tower()
        .find(|a| reg.is_artifact_installed(a))
        .or_else(|| tower().find(|a| a.default))
        .or_else(|| tower().next())
        // file bytes + the persistent workspace the tower pins at attach
        // (catalog data, measured per release - deepseek-ocr's encode slabs
        // are ~950 MiB, more than its weight file; see CatalogArtifact::
        // workspace for why this is a manifest field and not a constant here)
        .map_or(0, |a| a.total_size() + a.workspace.unwrap_or(0))
}

/// Every tower an endpoint holds resident, by its switches. A model with one
/// tower charges it when `vision` is on (the old rule, [`tower_bytes_for`]); a
/// split-tower model (`CatalogModel::split_towers`, EmbeddingGemma 2) charges
/// its picture tower by `vision` and its audio tower by `audio`, absent =
/// that artifact's catalog default (off) - the supervisor's own reading of
/// `spec.audio`, so the estimate and the served endpoint agree.
pub(crate) fn towers_bytes_for(
    m: &crate::registry::CatalogModel,
    reg: &crate::registry::Registry,
    weights: Option<&crate::registry::CatalogArtifact>,
    vision: bool,
    audio: Option<bool>,
) -> u64 {
    use crate::registry::ArtifactKind;
    if let Some(towers) = weights.and_then(|w| w.runtime.optional_towers.as_ref()) {
        let price = |tower: &crate::registry::TowerMemory| {
            tower.weight_bytes.saturating_add(tower.workspace_bytes)
        };
        return towers
            .vision
            .as_ref()
            .filter(|_| vision)
            .map_or(0, price)
            .saturating_add(
                towers
                    .audio
                    .as_ref()
                    .filter(|tower| audio.unwrap_or(tower.default))
                    .map_or(0, price),
            );
    }
    if !m.split_towers() {
        return if vision {
            tower_bytes_for(m, reg, weights)
        } else {
            0
        };
    }
    let one = |kind: ArtifactKind| {
        let of = || {
            m.artifacts.iter().filter(move |a| {
                a.kind == kind && weights.is_none_or(|w| w.runtime.allows_companion(&a.id))
            })
        };
        of().find(|a| reg.is_artifact_installed(a))
            .or_else(|| of().next())
            .map_or(0, |a| a.total_size() + a.workspace.unwrap_or(0))
    };
    let audio = audio.unwrap_or_else(|| m.split_audio_tower().is_some_and(|a| a.default));
    (if vision { one(ArtifactKind::Vision) } else { 0 })
        + if audio { one(ArtifactKind::Audio) } else { 0 }
}

/// The companions a model holds resident beside its weights with no switch:
/// an image-generation lane's text encoder and VAE (the DiT conditions on
/// one and decodes through the other), and a document reader's layout model.
/// Each loads at startup and stays for the endpoint's life, like a tower.
/// Elected per kind the way the tower is (installed, else default, else
/// first), within what the weights artifact allows, since the compact DiT
/// pairs with the compact text encoder. Zero for every other kind of model.
pub(crate) fn lane_companion_bytes_for(
    m: &crate::registry::CatalogModel,
    reg: &crate::registry::Registry,
    weights: Option<&crate::registry::CatalogArtifact>,
) -> u64 {
    use crate::registry::ArtifactKind;
    [
        ArtifactKind::TextEncoder,
        ArtifactKind::Vae,
        ArtifactKind::Layout,
    ]
    .into_iter()
    .map(|kind| {
        let of = || {
            m.artifacts.iter().filter(|a| {
                a.kind == kind && weights.is_none_or(|w| w.runtime.allows_companion(&a.id))
            })
        };
        of().find(|a| reg.is_artifact_installed(a))
            .or_else(|| of().find(|a| a.default))
            .or_else(|| of().next())
            .map_or(0, |a| a.total_size() + a.workspace.unwrap_or(0))
    })
    .sum()
}

/// Shared geometry for fit and admission, including directory checkpoints.
pub(crate) fn artifact_shape(
    state: &crate::routes::AppState,
    model: &crate::registry::CatalogModel,
    artifact: &crate::registry::CatalogArtifact,
    vision: bool,
    audio: Option<bool>,
) -> Option<ModelShape> {
    // the towers follow their switches; an image lane's text encoder and
    // VAE, and a document reader's layout model, have no switch and are
    // always charged
    let tower = lane_companion_bytes_for(model, &state.registry, Some(artifact))
        + towers_bytes_for(model, &state.registry, Some(artifact), vision, audio);
    let path = artifact.entry_path(state.registry.models_dir())?;
    let published = artifact.shape.clone().or_else(|| {
        // Validate the dense Qwen architecture before borrowing its GGUF
        // geometry; NEVER borrow that format's resident weight-byte count.
        if model.id != "qwen3.8-27b"
            || !artifact.runtime.checkpoint_dir
            || paddock_models::mlx::QwenConfig::read(&path).is_err()
        {
            return None;
        }
        let mut shape = model.weights().find_map(|a| a.shape.clone())?;
        shape.weight_bytes = artifact.total_size();
        shape.nextn_bytes = 0;
        Some(shape)
    });
    let mut shape = published
        .map(|s| s.into_model_shape(tower, artifact.workspace.unwrap_or(0)))
        .or_else(|| {
            state.probes.get(&path).map(|p| {
                let kind = kind_for(artifact.capabilities(model));
                ModelShape {
                    tower_bytes: tower,
                    workspace_bytes: artifact.workspace.unwrap_or(0),
                    ..ModelShape::from_report(&p, artifact.total_size(), kind)
                }
            })
        })?;
    if let Some(memory) = &artifact.runtime.memory {
        memory.apply(&mut shape);
    }
    Some(shape)
}

/// Qwen's Metal pool has three context slots per live slot without a tier,
/// and live slots + one restore context + two staging blocks per slot with
/// a tier (paddock-metal/qwen35/load.rs). Scale the catalog's maximum-envelope
/// reservation for the concurrency actually requested.
pub(crate) fn metal_cache_shape(
    shape: &mut ModelShape,
    model: &crate::registry::CatalogModel,
    env: &Envelope,
) {
    if matches!(
        model.family.as_deref(),
        Some("qwen3.5" | "qwen3.6" | "qwen3.8" | "bonsai")
    ) && model.id != "qwen3.8-flash-next"
    {
        shape.kv_reserve_sequences = if env.offload.is_some() {
            1 + (64 * env.concurrency).div_ceil(shape.max_ctx.max(1))
        } else {
            2 * env.concurrency
        };
    }
}

/// Metal uses fixed context reservations, not CUDA's dynamically sized pool
/// or its 40%-of-VRAM cap. Keep the conservative resident/workspace allowance,
/// but invert the actual cache geometry against the remaining unified budget.
pub(crate) fn backend_estimate(
    backend: &str,
    shape: &ModelShape,
    env: &Envelope,
    device: &Device,
) -> paddock_estimator::Estimate {
    let mut result = paddock_estimator::estimate(shape, env, device);
    if backend != "metal" || shape.kind != ModelKind::Generative {
        return result;
    }
    let sequences = env
        .concurrency
        .max(1)
        .saturating_add(shape.kv_reserve_sequences);
    // Qwen retains three recurrent/conv slots per live slot. The shared
    // estimator already charged live state and checkpoint allowances; top
    // that up if a larger batch exceeds that allowance.
    if let Some(r) = &shape.recurrent {
        let held = result
            .state
            .saturating_add(result.overhead_parts.prefix_checkpoints);
        let required = 3
            * env.concurrency.max(1)
            * r.layers
            * (r.state_elems + 3 * r.conv_elems)
            * r.elem_bytes;
        let extra = required.saturating_sub(held);
        result.state += extra;
        result.resident += extra;
    }
    result.overhead_parts.prefix_pool_extra = 0;
    let headroom = device.free_bytes.saturating_sub(result.resident);
    let cost = |ctx: u64| {
        shape
            .kv_per_sequence(ctx.div_ceil(32) * 32, env.kv_dtype)
            .saturating_mul(sequences)
    };
    let (mut low, mut high) = (0, shape.max_ctx);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if cost(mid) <= headroom {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    result.max_ctx = low;
    result.kv_pool = cost(low);
    result.limited_by = if low == shape.max_ctx {
        paddock_estimator::LimitedBy::Model
    } else {
        paddock_estimator::LimitedBy::Vram
    };
    result.fit = if result.resident > device.free_bytes {
        paddock_estimator::Fit::DoesNotFit {
            short_by_bytes: result.resident - device.free_bytes,
        }
    } else if low < 4096.min(shape.max_ctx) {
        paddock_estimator::Fit::Tight {
            headroom_bytes: headroom,
        }
    } else {
        paddock_estimator::Fit::Fits {
            headroom_bytes: headroom,
        }
    };
    result
}

pub(crate) fn artifact_spec(
    model: &crate::registry::CatalogModel,
    artifact: &crate::registry::CatalogArtifact,
    requested: bool,
) -> Option<paddock_estimator::SpecCost> {
    (requested
        && artifact
            .capabilities(model)
            .iter()
            .any(|c| c == "speculative"))
    .then(|| {
        let drafter = model
            .artifacts
            .iter()
            .filter(|a| {
                a.kind == crate::registry::ArtifactKind::Drafter
                    && artifact.runtime.allows_companion(&a.id)
                    && a.runtime
                        .backends
                        .iter()
                        .any(|b| artifact.runtime.backends.contains(b))
            })
            .max_by_key(|a| a.total_size())
            .map_or(0, |a| a.total_size());
        paddock_estimator::SpecCost {
            drafter_bytes: drafter,
            ..Default::default()
        }
    })
}

pub async fn handle(
    State(state): State<Arc<crate::routes::AppState>>,
    Query(q): Query<EstimateQuery>,
) -> Response {
    let asked_kv = match q.kv.as_deref() {
        Some("f32") => KvDtype::F32,
        Some("fp8_e4m3" | "fp8") => KvDtype::Fp8E4m3,
        _ => KvDtype::F16,
    };
    // Price what the RUNNER will serve, not what was asked. On a card with no
    // FP8 tensor cores `serving.rs::apply_kv_dtype` downgrades fp8 to f16 and
    // says so in its log; an estimate that kept the fp8 rate would report half
    // the KV pool the server then allocates - the panel saying "fits" about a
    // configuration twice the size it drew. One shared predicate,
    // so this cannot drift from the runner's own gate.
    let cc = q.cc.as_deref().and_then(parse_cc);
    let kv_blocked = cc.and_then(paddock_models::gpu_support::fp8_kv_blocked);
    let kv_downgraded = asked_kv == KvDtype::Fp8E4m3 && kv_blocked.is_some();
    let env = Envelope {
        concurrency: q
            .batch
            .unwrap_or(if state.readiness.backend == "metal" {
                1
            } else {
                state.max_batch as u64
            })
            .max(1),
        kv_dtype: if kv_downgraded {
            KvDtype::F16
        } else {
            asked_kv
        },
        // filled per model below - the drafter is a per-model artifact
        spec: None,
        offload: q
            .offload_ram_gb
            .filter(|g| *g > 0.0)
            .map(|g| paddock_estimator::OffloadCost::armed((g * (1u64 << 30) as f64) as u64)),
    };
    let want_spec = q.spec.unwrap_or(false);
    // Absent = charge it, so a caller that never heard of this parameter keeps
    // the old (heavier, safer) answer.
    let want_vision = q.vision.unwrap_or(true);

    // Free VRAM, not total: the engine sizes against what is actually
    // available, and so must anything claiming to predict it. Multi-GPU:
    // the caller picks the device (`gpu` = NVML index, default 0) and both
    // the free figure and the reclaimable fleet VRAM are per-that-device.
    let snap = state.gpu.latest();
    let sel = q.gpu.unwrap_or(0);
    let gpu = snap.gpus.iter().find(|g| g.index == sel);
    // The running fleet's allocator self-reports (model_mem = weights +
    // KV/state pools), via the reconciler's join over the admin pipes. That
    // VRAM comes back when runners are stopped/switched, so the estimate adds
    // it to what a swapped-in model could have. Only runners attributed to
    // the selected device count; on a single-GPU box a runner NVML can't
    // attribute (WDDM blind spot) can only be here, so it counts too - on a
    // multi-GPU box an unattributable runner honestly counts for none.
    //
    // §10.1 policy rides on top of the fit math: PINNED runners (the resident
    // embedder, the prod endpoint) are never auto-stopped to make room, so
    // their VRAM is not reclaimable-by-swap and they never appear as eviction
    // candidates. Who yields is policy; the estimator only answers fit.
    let single_gpu = snap.gpus.len() <= 1;
    let fleet = state.supervisor.fleet_meta().await;
    let pinned_ports: std::collections::HashSet<u16> = fleet
        .iter()
        .filter(|(_, _, pinned)| *pinned)
        .map(|(p, _, _)| *p)
        .collect();
    let recon = state.recon.borrow().clone();
    let on_device: Vec<&crate::telemetry::RunnerVram> = match &*recon {
        Some(r) => r
            .runners
            .iter()
            .filter(|rv| rv.gpu == Some(sel) || (single_gpu && rv.gpu.is_none()))
            .collect(),
        None => Vec::new(),
    };
    let reclaimable: u64 = on_device
        .iter()
        .filter(|rv| !pinned_ports.contains(&rv.port))
        .filter_map(|rv| rv.self_mem)
        .sum();
    // Eviction order (llama-swap's `evict_costs` lesson, computed instead of
    // hand-declared): unpinned device runners, cheapest-to-restore first -
    // restore cost ≈ bytes of weights to reload, so a small fast-loading model
    // yields before a 30 GB one. Falls back to the runner's resident VRAM when
    // the weights can't be resolved (labeled, never silent).
    let mut eviction: Vec<serde_json::Value> = on_device
        .iter()
        .filter(|rv| !pinned_ports.contains(&rv.port))
        .map(|rv| {
            let model = fleet
                .iter()
                .find(|(p, _, _)| *p == rv.port)
                .and_then(|(_, m, _)| m.clone());
            let weights = model
                .as_deref()
                .and_then(|m| resolve_weight_bytes(&state, m));
            let vram = rv.self_mem.or(rv.nvml_mem).unwrap_or(0);
            let (cost, basis) = match weights {
                Some(w) => (w, "weights"),
                None => (vram, "vram"),
            };
            serde_json::json!({
                "port": rv.port,
                "model": model,
                "vram": vram,
                "evict_cost": cost,
                "cost_basis": basis,
            })
        })
        .collect();
    eviction.sort_by_key(|e| e["evict_cost"].as_u64().unwrap_or(u64::MAX));
    let pinned_vram: u64 = on_device
        .iter()
        .filter(|rv| pinned_ports.contains(&rv.port))
        .filter_map(|rv| rv.self_mem)
        .sum();
    // `model_mem` covers weights + KV/state pools but not the CUDA context,
    // cuBLAS workspaces and allocator slack the process also holds (~2.5 GB
    // here). Those are exactly what the estimate's own graph margin budgets
    // for, so leaving them in "used by others" charges them twice and
    // under-reports what a swapped-in model could have.
    let others_raw = gpu
        .and_then(|g| g.mem_used)
        .unwrap_or(0)
        .saturating_sub(reclaimable);
    let in_use_by_others = if reclaimable > 0 {
        others_raw.saturating_sub(paddock_estimator::GRAPH_MARGIN)
    } else {
        others_raw
    };
    let budget_bytes = q.budget.map(|mib| mib << 20);
    let metal = if state.readiness.backend == "metal" {
        crate::metal_memory::available(&state, q.freeing_port).await
    } else {
        None
    };
    let device = metal
        .as_ref()
        .map(|(s, free)| Device {
            total_bytes: s.limit,
            free_bytes: {
                let free = free.saturating_sub(
                    q.offload_ram_gb
                        .filter(|g| g.is_finite() && *g > 0.0)
                        .map_or(0, |g| (g * (1u64 << 30) as f64) as u64),
                );
                budget_bytes.map_or(free, |b| free.min(b))
            },
        })
        .or_else(|| {
            gpu.and_then(|g| g.mem_total).map(|t| Device {
                // The ceiling caps what is on offer; it never invents room that is not
                // there, so it is a min() against real free VRAM rather than a
                // replacement for it.
                free_bytes: {
                    let free = t.saturating_sub(in_use_by_others);
                    budget_bytes.map_or(free, |b| free.min(b))
                },
                total_bytes: t,
            })
        });

    let models_dir = state.registry.models_dir().to_path_buf();
    let mut rows = serde_json::Map::new();
    for m in &state.registry.catalog().models {
        // Embedding, rerank, alignment, dense-prediction and image generation
        // are single-pass - one pass per input, nothing cached between calls.
        // They must not be priced with a decode cache.
        let kind = kind_for(&m.capability);
        // Speculation, priced per model. A separate drafter artifact is
        // resident weights; in-file MTP (qwen3.5/3.6 `nextn`) contributes 0
        // because those tensors already sit inside the weights file we are
        // counting. Either way the verify plane widens, which is the term the
        // Default carries. Only offered for models the engine can speculate
        // for, so an unsupported row is never priced for a thing it cannot do.
        let spec = (want_spec && m.capability.iter().any(|c| c == "speculative")).then(|| {
            let drafter_bytes = m
                .artifacts
                .iter()
                .find(|a| a.kind == crate::registry::ArtifactKind::Drafter)
                .map_or(0, |a| a.total_size());
            paddock_estimator::SpecCost {
                drafter_bytes,
                ..Default::default()
            }
        });
        let env = Envelope { spec, ..env };
        // The vision tower is shared across the weights alternatives - one
        // mmproj serves the Q8 and the Q4 alike - so its bytes belong in every
        // artifact row, not in one of them.
        // One row per WEIGHTS ARTIFACT (schema 3): Q8 and Q4 are different
        // footprints of one model, and the picker's fit verdicts need both.
        let mut art_rows = serde_json::Map::new();
        // The checkpoint's own architecture, learned from whichever artifact we
        // could probe. Needed for the elected sampling profile below, which is
        // keyed on arch - and deliberately only known for a DOWNLOADED model,
        // because the arch is read from the file rather than declared.
        let mut arch: Option<String> = None;
        // No `kv` asked = the class the runner serves on auto: the row's
        // family default (`kv_default`, KV8 for every generative family but
        // three since 2026-10-04), still subject to the die's fp8 gate.
        let model_kv = if q.kv.is_none()
            && kv_blocked.is_none()
            && m.kv_default
                .as_deref()
                .is_some_and(|k| matches!(k, "fp8_e4m3" | "fp8"))
        {
            KvDtype::Fp8E4m3
        } else {
            env.kv_dtype
        };
        for a in m.weights() {
            let env = Envelope {
                kv_dtype: a.runtime.estimate_kv_dtype(model_kv),
                spec: artifact_spec(m, a, want_spec),
                ..env
            };
            // the towers follow their switches; an image lane's text
            // encoder and VAE, and a document reader's layout model, have no
            // switch and are always charged
            let tower = lane_companion_bytes_for(m, &state.registry, Some(a))
                + towers_bytes_for(m, &state.registry, Some(a), want_vision, q.audio);
            let weights = a.total_size();
            let published = a.shape.clone();
            // Only an installed file can be probed. Rather than guess geometry
            // for the rest, say so: "download to measure" is a true answer,
            // and the disk size we do know is still shown.
            //
            // LAZY, because a probe reads up to a 256 MB header prefix and the
            // cache is cold on the first request after a restart. An artifact
            // that publishes a shape needs nothing from the file (26 of 33
            // and the one thing the probe still supplies -
            // `arch`, for the elected-sampling row - is per MODEL, so the
            // first artifact that yields it ends the probing for this model.
            // Probing every installed artifact and discarding most of the
            // results was seconds of cold I/O on the first paint of the
            // Start/Edit page, all of it for numbers the published block
            // already carried.
            let probed = (published.is_none() || arch.is_none())
                .then(|| {
                    a.files
                        .first()
                        .map(|f| models_dir.join(&f.dest))
                        .filter(|p| p.exists())
                        .and_then(|p| state.probes.get(&p))
                })
                .flatten();

            if let Some(r) = probed.as_ref()
                && arch.is_none()
            {
                arch.clone_from(&r.architecture);
            }

            // One shape, whether or not the file is here. The
            // published block wins over a local probe rather than being its
            // fallback, and that ordering is the point: it carries RESIDENT
            // weight bytes, which the probe cannot produce - `total_size()` is
            // the file, and the loader repacks on the way to the GPU (a Q4_K
            // costs ~13.7% more in VRAM than on disk). Preferring the
            // probe when installed would keep exactly the two-answer split this
            // was built to remove, with the worse number winning after the
            // download.
            //
            // Probe geometry still fills in for an artifact published before
            // this existed, and for a format the generator cannot read.
            let shape_source = published.as_ref().map(|s| s.source);
            let mut shape = artifact_shape(&state, m, a, want_vision, q.audio);
            if state.readiness.backend == "metal"
                && let Some(shape) = &mut shape
            {
                metal_cache_shape(shape, m, &env);
            }
            // What the row SAYS the weights cost: resident where we know it,
            // the file size otherwise - never a scaled guess. And the same
            // weights term the estimate itself used, which means subtracting
            // in-file nextn when speculation is off, because the engine does
            // not load those blocks then. FitChart derives the scratch band as
            // `resident - weights - tower - workspace`, so a `weights` bigger
            // than the one inside `resident` eats the whole band and clamps it
            // to zero - on nemotron that is 1.42 GiB of chart.
            let shown = shape.as_ref().map_or(weights, |s| {
                if env.spec.is_some() {
                    s.weight_bytes
                } else {
                    s.weight_bytes.saturating_sub(s.nextn_bytes)
                }
            });
            let source = match shape_source {
                Some(paddock_estimator::ShapeSource::Measured) => "measured",
                Some(paddock_estimator::ShapeSource::Probed) => "probed",
                None => "file",
            };

            let mut row = match (shape, device) {
                (Some(shape), Some(dev)) => {
                    let est = backend_estimate(&state.readiness.backend, &shape, &env, &dev);
                    // the whole trade-off in one payload: how the window shrinks as
                    // sessions are added, so the UI never has to guess or re-ask
                    let curve: Vec<_> = CURVE
                        .iter()
                        .map(|&n| {
                            let env = Envelope {
                                concurrency: n,
                                ..env
                            };
                            let mut shape = shape.clone();
                            if state.readiness.backend == "metal" {
                                metal_cache_shape(&mut shape, m, &env);
                            }
                            let ctx =
                                backend_estimate(&state.readiness.backend, &shape, &env, &dev)
                                    .max_ctx;
                            serde_json::json!({ "at": n, "ctx": ctx })
                        })
                        .collect();
                    serde_json::json!({
                        "known": true, "kind": kind, "weights": shown,
                        "tower": tower, "weights_source": source,
                        "estimate": est, "curve": curve,
                    })
                }
                // A shape but no GPU telemetry: the footprint is still real, there
                // is just nothing to compare it against.
                (Some(shape), None) => serde_json::json!({
                    "known": true,
                    "kind": kind,
                    "weights": shown,
                    "tower": tower,
                    "weights_source": source,
                    "kv_bytes_per_token": shape.kv_bytes_per_token(env.kv_dtype),
                    "reason": "no GPU telemetry - cannot judge fit",
                }),
                // No published shape and nothing to probe. Today this is the
                // safetensors lane only (nemotron's NVFP4 arm): probe_path is
                // GGUF-only, so the shapes generator cannot generate a block for
                // it. Still the honest answer rather than a scaled file size.
                (None, _) => serde_json::json!({
                    "known": false,
                    "kind": kind,
                    "weights": weights,
                    "tower": tower,
                    "weights_source": source,
                    "reason": "VRAM for this format is measured from a load, not guessed",
                }),
            };
            row["kv_dtype"] = serde_json::json!(env.kv_dtype);
            row["backend"] = serde_json::json!(state.readiness.backend);
            art_rows.insert(a.id.clone(), row);
        }
        // The decoding parameters this checkpoint's own authors published
        // The runner resolves pin -> election -> wire, so a form
        // that shows a blank field has to be able to say what blank RESOLVES
        // to; without this it could only claim "the model's default" and hope.
        //
        // `instruct` rides along when the card publishes a second row for
        // thinking-off (the qwen family does), because "the default" is then
        // genuinely two values and picking one to display would be a guess.
        let knobs = |k: &paddock_models::sampling::Knobs| {
            serde_json::json!({
                "temperature": paddock_models::sampling::as_written(k.temperature),
                "top_k": k.top_k,
                "top_p": paddock_models::sampling::as_written(k.top_p),
                "min_p": paddock_models::sampling::as_written(k.min_p),
            })
        };
        // LightOnOCR-3 is a qwen35 file, so its own row is named by the catalog
        // family - the same identity key the runner reads off the file
        let lightonocr = m.family.as_deref() == Some(paddock_models::sampling::LIGHTONOCR3);
        let sampling = lightonocr
            .then_some(paddock_models::sampling::LIGHTONOCR3)
            .or(arch.as_deref())
            .and_then(paddock_models::sampling::elected)
            .map(|e| {
                let mut o = knobs(&e.thinking).as_object().cloned().unwrap_or_default();
                o.insert("source".into(), serde_json::json!(e.source));
                if let Some(i) = e.instruct {
                    o.insert("instruct".into(), knobs(&i));
                }
                serde_json::Value::Object(o)
            });
        rows.insert(
            m.id.clone(),
            serde_json::json!({ "kind": kind, "artifacts": art_rows, "sampling": sampling }),
        );
    }

    Json(serde_json::json!({
        "envelope": {
            "batch": env.concurrency,
            // The width these numbers were PRICED at, which is what the runner
            // will serve - not necessarily what was asked for. A UI labelling
            // its KV row from its own form control instead of from here can
            // say "8-bit" over bytes counted at 16.
            "kv_dtype": env.kv_dtype,
            "kv_asked": asked_kv,
            // Set when the two differ, with the hardware reason, so the panel
            // can say why rather than silently showing a bigger number than
            // the control implies.
            "kv_downgraded": kv_downgraded.then_some(kv_blocked).flatten(),
            // The ceiling these numbers were priced under, so a panel can say
            // the fit is against the LIMIT rather than against the card.
            "budget": budget_bytes,
            // Whether the vision/audio tower is in these numbers. Mirrors the
            // start form's switch and the supervisor's `spec.vision`.
            "vision": want_vision,
            // a split-tower model's audio tower, as asked (absent = default)
            "audio": q.audio,
            // The server's own --max-ctx caps what it will actually serve,
            // independently of what the card could back. A model may report a
            // 262144 window while this server is configured for 32768; the UI
            // has to be able to say so rather than promise the larger number.
            "server_ctx": state.max_ctx,
            "server_batch": state.max_batch,
            // What the prefix cache was priced with, so a panel can say the
            // context it draws is the one an ARMED tier leaves.
            "offload_ram_gb": q.offload_ram_gb.filter(|g| *g > 0.0),
        },
        // Host memory: the resource prefix-cache offload actually spends, and
        // the one this manager never used to price. `total` is None on a
        // platform we cannot ask (see hostmem) - the panel then shows the
        // commitment without a denominator rather than inventing one.
        "host": {
            "total": crate::hostmem::total_bytes(),
            "available": metal.as_ref().map(|(_, free)| *free),
            // ceilings other endpoints have already promised their caches;
            // every one is reachable at once, so it is what to subtract
            "committed": crate::hostmem::committed_bytes(
                state
                    .supervisor
                    .configured_offload_ram_gb()
                    .into_iter()
                    .map(Some),
            ),
            "requested": q
                .offload_ram_gb
                .filter(|g| *g > 0.0)
                .map(|g| (g * (1u64 << 30) as f64) as u64),
            // What the folder field defaults to. The store appends its own
            // `kv-cache` segment inside this, so showing the ROOT means the
            // placeholder is something a user could paste back verbatim
            // without landing in kv-cache/kv-cache.
            "cache_dir": paddock_admin::data_root().display().to_string(),
        },
        "device": device.map(|d| serde_json::json!({
            // which GPU this estimate priced (NVML index) - label it
            "index": sel,
            // Two different numbers, and conflating them reads as a bug: a card
            // showing "38 of 48 GB used" cannot also be "43 GB free". `free` is
            // what a model would GET (the loaded one is released first);
            // `free_now` is what is unallocated at this instant. The UI must
            // label which it is showing.
            "free": d.free_bytes,
            "free_now": metal.as_ref().map_or_else(|| d.total_bytes.saturating_sub(gpu.and_then(|g| g.mem_used).unwrap_or(0)), |(_, free)| *free),
            "total": d.total_bytes,
            "name": metal.as_ref().map(|(s, _)| s.name.clone()).or_else(|| gpu.map(|g| g.name.clone())),
            "unified": metal.is_some(),
            "physical": metal.as_ref().map(|(s, _)| s.physical),
            "planning_basis": if metal.is_some() {
                "conservative unified-memory budget; runtime rechecks allocations"
            } else if gpu.is_some_and(|g| g.host_memory) {
                // a GB10: NVML has no framebuffer, so the device is the
                // machine's RAM as the OS books it (the runner gate's reading)
                "NVML device, the host's RAM (MemAvailable)"
            } else {
                "NVML"
            },
            "held_by_loaded_model": if metal.is_some() { 0 } else { reclaimable },
            "used_by_others": if metal.is_some() { d.total_bytes.saturating_sub(d.free_bytes) } else { in_use_by_others },
            // §10.1 pinned runners: resident by policy, so their VRAM is part
            // of used_by_others for the fit math - this labels the subset so
            // the UI can say "of which pinned: X" instead of "other apps".
            "held_by_pinned": pinned_vram,
            // Who yields when room must be made (unpinned device runners,
            // cheapest-to-restore first). Policy input for serve-also/compare;
            // the estimator itself never stops anything.
            "eviction": eviction,
            // paddock's CUDA context / cuBLAS workspaces / allocator slack:
            // real, resident, and not inside `model_mem`. Reported as its own
            // line so total - others - runtime - model == free_now actually
            // balances; folding it into "other apps" left a ~2 GB hole in a
            // tooltip whose entire job is to reconcile against nvidia-smi.
            "paddock_runtime": if metal.is_none() && reclaimable > 0 { paddock_estimator::GRAPH_MARGIN } else { 0 },
        })),
        "models": rows,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use crate::registry::{ArtifactKind, Registry};

    #[test]
    fn imported_checkpoint_counts_weight_shards_not_directory_size() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(super::checkpoint_file_bytes(dir.path()), None);
        std::fs::write(dir.path().join("part-1.safetensors"), [0; 7]).unwrap();
        std::fs::write(dir.path().join("part-2.safetensors"), [0; 11]).unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        assert_eq!(super::checkpoint_file_bytes(dir.path()), Some(18));
        assert_eq!(
            super::checkpoint_file_bytes(&dir.path().join("part-1.safetensors")),
            Some(7)
        );
    }

    #[test]
    fn metal_fixed_pool_is_batch_scaled_and_does_not_inherit_cuda_pool_cap() {
        use paddock_estimator::{Device, Envelope, KvDtype};
        let reg = Registry::new(std::env::temp_dir()).with_backend("metal");
        let model = reg.catalog_of("bonsai-2-27b").unwrap();
        let artifact = model.artifact("mlx-2bit").unwrap();
        for batch in [1, 4] {
            let mut shape = artifact.shape.clone().unwrap().into_model_shape(0, 0);
            artifact.runtime.memory.as_ref().unwrap().apply(&mut shape);
            shape.max_ctx = 32768;
            let env = Envelope {
                concurrency: batch,
                kv_dtype: KvDtype::F32,
                spec: None,
                offload: None,
            };
            super::metal_cache_shape(&mut shape, model, &env);
            assert_eq!(shape.kv_reserve_sequences, batch * 2);
            let device = Device {
                free_bytes: 100 << 30,
                total_bytes: 100 << 30,
            };
            let ample = super::backend_estimate("metal", &shape, &env, &device);
            assert_eq!(ample.max_ctx, 32768);
            assert_eq!(
                ample.kv_pool,
                shape.kv_per_sequence(32768, KvDtype::F32) * batch * 3
            );
            let exact = Device {
                free_bytes: ample.resident + ample.kv_pool,
                ..device
            };
            assert_eq!(
                super::backend_estimate("metal", &shape, &env, &exact).max_ctx,
                32768
            );
            let too_small = Device {
                free_bytes: exact.free_bytes - 1,
                ..device
            };
            assert!(super::backend_estimate("metal", &shape, &env, &too_small).max_ctx < 32768);
            let cuda = super::backend_estimate("cuda", &shape, &env, &device);
            assert_eq!(
                serde_json::to_value(cuda).unwrap(),
                serde_json::to_value(paddock_estimator::estimate(&shape, &env, &device)).unwrap()
            );
        }
    }

    // NOTE: the float-noise invariant is tested where the helper lives
    // (`paddock_models::sampling::as_written`), against the whole elected
    // table. Both this endpoint and the runner's capability surface publish
    // the same numbers through it, so one test covers both.

    /// A model with no decode loop must never be priced with one. Checked over
    /// the whole catalog rather than by id, so the next single-pass capability
    /// someone adds either lands in `is_single_pass` or fails here: every such
    /// model has to publish an `encoder` shape, and nothing generative may.
    /// The dense-prediction model is the case that motivated this - priced as
    /// generative it would have been charged a logits plane, block tables and
    /// the whole graph margin for a graph it does not have.
    #[test]
    fn single_pass_models_are_priced_without_a_decode_cache() {
        let reg = Registry::new(std::env::temp_dir());
        let mut seen = 0;
        for m in &reg.catalog().models {
            let single = m.capability.iter().any(|c| super::is_single_pass(c));
            for a in m.weights() {
                let Some(shape) = &a.shape else { continue };
                let no_decode = shape.kind != paddock_estimator::ModelKind::Generative;
                assert_eq!(
                    no_decode, single,
                    "{}/{}: capability {:?} but shape kind {:?}",
                    m.id, a.id, m.capability, shape.kind
                );
                // and the image kind is reserved for image generation, which
                // must say so - a render priced as a token encoder would size
                // its workspace by a window it does not have
                assert_eq!(
                    shape.kind == paddock_estimator::ModelKind::Image,
                    m.capability.iter().any(|c| c == "image-generation"),
                    "{}/{}: shape kind {:?} vs capability {:?}",
                    m.id,
                    a.id,
                    shape.kind,
                    m.capability
                );
                seen += usize::from(single);
            }
        }
        assert!(
            seen > 0,
            "no single-pass model publishes a shape - the test checked nothing"
        );
        // Every dense-prediction row, however many there are. The published
        // catalog may carry none (the first such model is held in the optional
        // private catalog), so this is a rule about the rows that exist, not
        // a claim that one does.
        for seg in reg
            .catalog()
            .models
            .iter()
            .filter(|m| m.capability.iter().any(|c| c == "segmentation"))
        {
            let a = seg.default_weights().expect("it has weights");
            assert!(
                a.shape.is_some() && a.workspace.is_some_and(|w| w > 0),
                "{}: a dense-prediction model's workspace outweighs its weights at any useful \
                 pass width - a fit estimate that skips it says 'fits' about a start that does not",
                seg.id
            );
        }
    }

    /// Where the private catalog is compiled in, its rows really are in the
    /// catalog the manager serves - the merge is a build-time branch, and a
    /// branch nobody runs is one that rots.
    #[cfg(private_catalog)]
    #[test]
    fn private_rows_join_the_published_catalog() {
        let reg = crate::registry::Registry::new(std::env::temp_dir());
        let private: crate::registry::Catalog =
            toml::from_str(include_str!("../models.private.toml")).expect("it parses");
        assert!(
            !private.models.is_empty(),
            "an empty private catalog is a stray file"
        );
        for m in &private.models {
            assert!(
                reg.catalog().models.iter().any(|c| c.id == m.id),
                "{} is in models.private.toml but not in the served catalog",
                m.id
            );
        }
    }

    /// The query field is deserialized by NAME, so a rename or a typo on
    /// either side degrades to "absent" - which is silently the old, wrong
    /// answer rather than an error. This is the one plumbing failure the
    /// arithmetic tests in `paddock-estimator` cannot see, so it is checked
    /// where the name actually crosses the wire.
    #[test]
    fn the_offload_budget_survives_the_query_string() {
        // through the real extractor, so this is the path a request takes
        let parse = |qs: &str| -> super::EstimateQuery {
            let uri: axum::http::Uri = format!("/api/models/estimate?{qs}").parse().unwrap();
            axum::extract::Query::try_from_uri(&uri)
                .map(|axum::extract::Query(q)| q)
                .expect("parse")
        };
        assert_eq!(
            parse("batch=4&kv=f16&offload_ram_gb=24").offload_ram_gb,
            Some(24.0)
        );
        // absent stays absent - no tier, and no accidental default that would
        // charge staging to an endpoint that never armed one
        assert_eq!(parse("batch=4").offload_ram_gb, None);
        // and 0 is treated as "no tier" downstream, not as an armed one
        let zero = parse("offload_ram_gb=0");
        assert_eq!(zero.offload_ram_gb, Some(0.0));
        assert!(zero.offload_ram_gb.filter(|g| *g > 0.0).is_none());
    }

    /// The estimate has to price the width the RUNNER will serve. Before this,
    /// the manager had no compute-capability input at all, so on an A6000 it
    /// counted fp8 bytes for a cache the runner then allocated at f16 - half
    /// the pool it drew.
    #[test]
    fn the_kv_width_follows_the_card_not_the_request() {
        use paddock_models::gpu_support::fp8_kv;
        assert_eq!(super::parse_cc("8.6"), Some((8, 6)));
        assert_eq!(super::parse_cc("12.0"), Some((12, 0)));
        // a malformed string must not gate - absent means "honour the request",
        // the same fail-open stance the runner's own device singleton takes
        assert_eq!(super::parse_cc("ampere"), None);
        assert_eq!(super::parse_cc("8"), None);

        // the property, stated against the two cards this actually decides
        // the estimator must price the width the runner will actually serve -
        // and that is fp8 on Ampere too (gpu_support::fp8_kv)
        assert!(
            fp8_kv(super::parse_cc("8.6").unwrap()),
            "A6000 is no longer downgraded"
        );
        assert!(
            fp8_kv(super::parse_cc("12.0").unwrap()),
            "Blackwell keeps fp8"
        );
    }

    /// The bug this closes was invisible by construction: the estimate looped
    /// over `m.weights()`, so a companion could never appear in it no matter
    /// how large. Assert the property over the whole catalog rather than one
    /// model, so adding a vision model without pricing its tower fails here.
    #[test]
    fn every_tower_model_prices_its_tower_and_no_one_else_does() {
        // a models dir that cannot exist, so nothing reads as installed and
        // the precedence falls through to the artifact the bundle would pull
        let reg = Registry::new(std::path::PathBuf::from("./this-dir-does-not-exist"));
        let mut priced = 0;
        for m in &reg.catalog().models {
            // every tower switched on
            let tower = super::towers_bytes_for(m, &reg, None, true, Some(true));
            let has_vision = m.artifacts.iter().any(|a| a.kind == ArtifactKind::Vision);
            let has_audio = m.artifacts.iter().any(|a| a.kind == ArtifactKind::Audio);
            let has_tower = has_vision || has_audio;
            assert_eq!(
                has_tower,
                tower > 0,
                "{}: an mmproj artifact must be charged",
                m.id
            );
            // An AUDIO tower and the `transcription` capability are the same
            // claim said twice - the picker must not offer speech input for a
            // model priced as text-only, nor price a tower it never labels.
            // Whisper is the deliberate exception on one side: it transcribes
            // with no mmproj at all, because its audio encoder ships inside
            // the weights file rather than as a companion. So the implication
            // runs one way only: audio tower => transcription.
            //
            // An EMBEDDING row is the other: its towers are embedding inputs -
            // pictures and audio become vectors, not image chat or speech to
            // text - so it claims neither (`transcription` would route the
            // Studio's microphone to it). EmbeddingGemma 2 is the case.
            let embeds = m.capability.iter().any(|c| c == "embeddings");
            let claims_audio = m.capability.iter().any(|c| c == "transcription");
            assert!(
                !has_audio || claims_audio || embeds,
                "{}: an audio mmproj must come with the `transcription` capability",
                m.id
            );
            assert!(
                !embeds || !claims_audio,
                "{}: an embedding row's audio tower is not speech to text",
                m.id
            );
            // the catalog's own two claims about images have to agree, or the
            // picker offers image input for a model priced as text-only.
            //
            // "Takes images" is not one capability string: `vision` is general
            // image chat, `documents` is granite-vision's structured extraction
            // (IBM's card says it may not generalize past that, so it must not
            // wear the general chip - models.toml explains the split). What
            // must hold is that a tower comes with exactly one of them. A new
            // image capability has to be added here deliberately; forgetting
            // fails this assertion rather than quietly shipping a model the
            // picker prices but never labels.
            //
            // An image-GENERATION row is the one place a vision tower comes
            // without an image-input chip: qwen-image's mmproj is the editing
            // encoder (references go through it into the text encoder), and
            // the picker labels that as `edit` on the image lane, not as chat
            // image input. Such a row still gets priced (asserted above), it
            // just doesn't claim `vision`.
            const IMAGE_CAPS: [&str; 2] = ["vision", "documents"];
            let claims_images = m
                .capability
                .iter()
                .filter(|c| IMAGE_CAPS.contains(&c.as_str()))
                .count();
            let generates_images = m.capability.iter().any(|c| c == "image-generation");
            if generates_images || embeds {
                assert_eq!(
                    claims_images, 0,
                    "{}: an image-generation or embedding row must not also claim image chat input",
                    m.id
                );
            } else {
                assert_eq!(
                    claims_images,
                    usize::from(has_vision),
                    "{}: exactly one image capability ({IMAGE_CAPS:?}) goes with a vision artifact",
                    m.id
                );
            }
            // ...and an image capability must never ride on an audio tower.
            assert!(
                !has_audio || claims_images == 0,
                "{}: an audio mmproj must not claim image input",
                m.id
            );
            // A tower's `workspace` bytes ride the same charge as its file
            // bytes - deepseek-ocr's encode slabs are BIGGER than its mmproj
            // file, and a fit that skips them is off by a gigabyte.
            // (a split-tower model holds both towers, each with its own)
            if m.split_towers() {
                let both: u64 = m
                    .artifacts
                    .iter()
                    .filter(|a| a.kind.is_mmproj())
                    .map(|a| a.total_size() + a.workspace.unwrap_or(0))
                    .sum();
                assert_eq!(tower, both, "{}: both towers must be charged", m.id);
            } else if let Some(a) = m
                .artifacts
                .iter()
                .find(|a| a.kind.is_mmproj() && a.workspace.is_some())
            {
                assert_eq!(
                    tower,
                    a.total_size() + a.workspace.unwrap(),
                    "{}: the tower's workspace must be charged on top of its file",
                    m.id
                );
            }
            priced += u32::from(has_tower);
        }
        assert!(priced > 0, "the catalog has tower models to price");
        // the field is live in the shipped catalog: unlimited-ocr declares it
        let reg = Registry::new(std::path::PathBuf::from("./this-dir-does-not-exist"));
        let ocr = reg
            .catalog()
            .models
            .iter()
            .find(|m| m.id == "unlimited-ocr");
        let ocr = ocr.expect("unlimited-ocr row in models.toml");
        assert!(
            ocr.artifacts
                .iter()
                .any(|a| a.workspace.unwrap_or(0) > 900 << 20),
            "unlimited-ocr's mmproj must declare its ~950 MiB workspace"
        );
    }

    /// The image lane's two companions are the same claim as its capability,
    /// said three times: an `image-generation` row carries a text encoder AND
    /// a VAE, both required and default (the DiT cannot render without either,
    /// and "download it" must fetch them), every weights alternative admits
    /// one of each, and nothing else in the catalog carries such a piece. And
    /// they are CHARGED - a fit that priced the DiT alone would be short by
    /// the text encoder, which on the full lane outweighs the DiT itself.
    #[test]
    fn every_image_lane_carries_and_prices_its_text_encoder_and_vae() {
        let reg = Registry::new(std::path::PathBuf::from("./this-dir-does-not-exist"));
        let mut lanes = 0;
        for m in &reg.catalog().models {
            let claims = m.capability.iter().any(|c| c == "image-generation");
            let pieces: Vec<_> = m
                .artifacts
                .iter()
                .filter(|a| a.kind.is_lane_companion())
                .collect();
            assert_eq!(
                claims,
                !pieces.is_empty(),
                "{}: image-generation = {claims}, lane companions = {}",
                m.id,
                pieces.len()
            );
            if !claims {
                continue;
            }
            lanes += 1;
            for a in &pieces {
                assert!(
                    a.required && a.default,
                    "{}/{}: a lane companion is required and part of the download",
                    m.id,
                    a.id
                );
            }
            for w in m.weights() {
                if w.runtime.checkpoint_dir {
                    // Self-contained diffusion packs charge the text encoder
                    // and VAE in their own resident-weight ledger, not twice
                    // as separately downloaded GGUF companions.
                    assert_eq!(w.runtime.companions.as_deref(), Some([].as_slice()));
                    for component in [
                        "text_encoder/model.safetensors",
                        "transformer/model.safetensors",
                        "vae/model.safetensors",
                    ] {
                        assert!(
                            w.files.iter().any(|f| f.dest.ends_with(component)),
                            "{}/{}: missing {component}",
                            m.id,
                            w.id
                        );
                    }
                    assert!(
                        w.shape
                            .as_ref()
                            .is_some_and(|s| s.weight_bytes > 8_000_000_000)
                    );
                    assert_eq!(super::lane_companion_bytes_for(m, &reg, Some(w)), 0);
                    continue;
                }
                for kind in [ArtifactKind::TextEncoder, ArtifactKind::Vae] {
                    assert!(
                        pieces
                            .iter()
                            .any(|a| a.kind == kind && w.runtime.allows_companion(&a.id)),
                        "{}/{}: admits no {kind:?} companion",
                        m.id,
                        w.id
                    );
                }
                let charged = super::lane_companion_bytes_for(m, &reg, Some(w));
                let smallest = pieces
                    .iter()
                    .filter(|a| w.runtime.allows_companion(&a.id))
                    .map(|a| a.total_size())
                    .min()
                    .unwrap_or(0);
                assert!(
                    charged > smallest,
                    "{}/{}: lane companions charged {charged} bytes - both pieces must be priced",
                    m.id,
                    w.id
                );
            }
        }
        assert!(
            lanes > 0,
            "the catalog has an image-generation row to check"
        );
    }
}
