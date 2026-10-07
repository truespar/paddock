//! Artifact-level contracts: a quantized export need not support every route
//! that the model family supports. These are loader facts, not tuning knobs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

fn legacy_backends() -> Vec<String> {
    // The older catalog describes CUDA. Missing metadata must not silently
    // advertise a newly added backend (or one added in a future release).
    vec!["cuda".into()]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Qualification {
    // Read old manifests without resurrecting test-progress admission gates.
    // Qualification is product catalog metadata, not a benchmark assertion.
    #[serde(alias = "unqualified", alias = "experimental")]
    Qualified,
}

/// Resident layout is a property of a loader, not just a GGUF. For example,
/// GPT-OSS Metal retains full paged KV even on sliding layers so prefixes can
/// resume exactly; CUDA's ring-cache shape would underprice that allocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMemory {
    /// Directory-primary backends may have a native shape where the legacy
    /// artifact has none. Do not invent a CUDA footprint to publish Metal's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published_shape: Option<paddock_estimator::PublishedShape>,
    pub weight_bytes: u64,
    pub source: paddock_estimator::ShapeSource,
    /// Implementation ceiling, not a claim of qualification at this length.
    pub max_ctx: u64,
    pub max_batch: u64,
    pub full_paged_kv: bool,
    pub kv_reserve_sequences: u64,
    /// Some native loaders omit in-file draft heads entirely. Their resident
    /// weight count must not have nextn subtracted a second time by estimates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nextn_bytes: Option<u64>,
    /// Checkpoint metadata can count non-recurrent mixer blocks as recurrent.
    /// Override only when the native graph has measured its actual state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recurrent: Option<paddock_estimator::RecurrentShape>,
    /// CUDA's artifact-level workspace is not a Metal allocation. None
    /// preserves it; an explicit replacement is backend-scoped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_bytes: Option<u64>,
}

impl BackendMemory {
    pub fn apply(&self, shape: &mut paddock_estimator::ModelShape) {
        shape.weight_bytes = self.weight_bytes;
        shape.max_ctx = shape.max_ctx.min(self.max_ctx);
        shape.kv_reserve_sequences = self.kv_reserve_sequences;
        if let Some(n) = self.nextn_bytes {
            shape.nextn_bytes = n;
        }
        if let Some(r) = self.recurrent {
            shape.recurrent = Some(r);
        }
        if let Some(w) = self.workspace_bytes {
            shape.workspace_bytes = w;
        }
        if self.full_paged_kv {
            for layer in &mut shape.kv_layers {
                layer.window = None;
            }
        }
    }

    pub fn apply_published(&self, shape: &mut paddock_estimator::PublishedShape) {
        shape.weight_bytes = self.weight_bytes;
        shape.source = self.source;
        if let Some(n) = self.nextn_bytes {
            shape.nextn_bytes = n;
        }
        if let Some(r) = self.recurrent {
            shape.recurrent = Some(r);
        }
        shape.max_ctx = shape.max_ctx.min(self.max_ctx);
        if self.full_paged_kv {
            for layer in &mut shape.kv_layers {
                layer.window = None;
            }
        }
    }
}

/// Only execution contracts vary by backend; file identity, provenance and
/// checkpoint layout do not. Resolve this once when constructing a registry
/// so download, preview, spawn and the Studio all consume the same contract.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendRuntime {
    /// Backend-specific election, independent of model test-progress metadata.
    pub default: Option<bool>,
    pub qualification: Option<Qualification>,
    pub capability: Option<Vec<String>>,
    pub companions: Option<Vec<String>>,
    pub kv_cache_dtype: Option<String>,
    pub note: Option<String>,
    pub memory: Option<BackendMemory>,
    /// Backend-specific launch recommendation, not the concurrency ceiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_max_batch: Option<usize>,
    /// Backend-specific target window. A TARGET: a start that leaves max_ctx
    /// unset is shrunk onto what the card backs at admission (routes.rs
    /// fit_window), the same rule the Studio form follows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_max_ctx: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRuntime {
    /// Legacy entries mean CUDA; an explicit empty list allows no backend.
    #[serde(default = "legacy_backends")]
    pub backends: Vec<String>,
    /// None inherits the family; Some replaces it, never adds to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability: Option<Vec<String>>,
    /// Allowed companion artifact IDs. None inherits; [] allows none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub companions: Option<Vec<String>>,
    /// Directory-primary checkpoints must be passed as a directory, not as
    /// shard 1 (which the GGUF loader would otherwise try to parse).
    #[serde(default)]
    pub checkpoint_dir: bool,
    /// The weights bundle contains its tower; no optional mmproj is involved.
    #[serde(default)]
    pub embedded_vision: bool,
    /// Fixed loader contract, e.g. auto selects checkpoint-native BF16 KV.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_dtype: Option<String>,
    #[serde(default)]
    pub experimental: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qualification: Option<Qualification>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<BackendMemory>,
    /// Elected launch envelope when the caller omits dimensions. These do
    /// not select an artifact or relax its memory limits. Most entries keep
    /// runner defaults; a large resident MLX model must not inherit 32 slots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_max_ctx: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_max_batch: Option<usize>,
    /// Elected policy for new configurations only. Explicit Off and existing
    /// saved configurations remain authoritative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_spec: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub backend_overrides: BTreeMap<String, BackendRuntime>,
}

impl Default for ArtifactRuntime {
    fn default() -> Self {
        Self {
            backends: legacy_backends(),
            capability: None,
            companions: None,
            checkpoint_dir: false,
            embedded_vision: false,
            kv_cache_dtype: None,
            experimental: false,
            note: None,
            qualification: None,
            memory: None,
            default_max_ctx: None,
            default_max_batch: None,
            default_spec: None,
            backend_overrides: BTreeMap::new(),
        }
    }
}

impl ArtifactRuntime {
    /// Defaults must fit the selected backend, not the generic chat envelope.
    /// Explicit user settings are validated separately and are never clamped.
    pub fn default_envelope(&self) -> (usize, usize) {
        let ctx = self.default_max_ctx.unwrap_or(4096);
        let batch = self.default_max_batch.unwrap_or(32);
        match &self.memory {
            Some(memory) => (
                ctx.min(memory.max_ctx as usize),
                batch.min(memory.max_batch as usize),
            ),
            None => (ctx, batch),
        }
    }

    /// Fit previews must price the cache that the loader will actually
    /// allocate, even when a caller inherited the CUDA FP8 preference.
    pub fn estimate_kv_dtype(
        &self,
        requested: paddock_estimator::KvDtype,
    ) -> paddock_estimator::KvDtype {
        match self.kv_cache_dtype.as_deref() {
            Some("f32") => paddock_estimator::KvDtype::F32,
            Some("f16" | "bf16") => paddock_estimator::KvDtype::F16,
            // The runner requires `auto` for native Metal checkpoint dirs,
            // where it means BF16, not CUDA's requested FP8 storage.
            Some("auto") if self.checkpoint_dir && self.backends == ["metal"] => {
                paddock_estimator::KvDtype::F16
            }
            _ => requested,
        }
    }

    pub fn supports_backend(&self, backend: &str) -> bool {
        self.backends.iter().any(|b| b == backend)
    }

    pub fn for_backend(&self, backend: &str) -> Self {
        let mut effective = self.clone();
        if let Some(contract) = self.backend_overrides.get(backend) {
            if let Some(batch) = contract.default_max_batch {
                effective.default_max_batch = Some(batch);
            }
            if let Some(ctx) = contract.default_max_ctx {
                effective.default_max_ctx = Some(ctx);
            }
            if let Some(memory) = &contract.memory {
                effective.memory = Some(memory.clone());
            }
            if let Some(v) = &contract.capability {
                effective.capability = Some(v.clone());
            }
            if let Some(v) = &contract.companions {
                effective.companions = Some(v.clone());
            }
            if let Some(v) = &contract.kv_cache_dtype {
                effective.kv_cache_dtype = Some(v.clone());
            }
            if let Some(v) = &contract.note {
                effective.note = Some(v.clone());
            }
            if let Some(v) = contract.qualification {
                effective.qualification = Some(v);
                effective.experimental = v != Qualification::Qualified;
            }
        }
        effective
    }

    pub fn allows_companion(&self, id: &str) -> bool {
        self.companions
            .as_ref()
            .is_none_or(|ids| ids.iter().any(|x| x == id))
    }
}

#[cfg(test)]
mod availability_tests {
    use super::*;

    #[test]
    fn legacy_statuses_decode_without_reintroducing_gates() {
        for label in ["unqualified", "experimental", "qualified"] {
            let qualification: Qualification =
                serde_json::from_str(&format!("\"{label}\"")).unwrap();
            assert_eq!(qualification, Qualification::Qualified);
            assert_eq!(serde_json::to_value(qualification).unwrap(), "qualified");
        }
        let legacy = ArtifactRuntime::default();
        assert!(!legacy.supports_backend("metal"));
        assert!(legacy.supports_backend("cuda"));
    }
}

/// Provenance belongs to the export, not just the base model. A community
/// conversion must never be presented as an official Qwen quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSource {
    pub repo: String,
    pub revision: String,
    pub base_model: String,
    pub license: String,
    pub license_url: String,
    /// The files are downloaded from `repo` itself on Hugging Face, behind
    /// its licence gate - the user accepts the terms there with their own
    /// account and the download carries their token (`registry/gated.rs`).
    /// Set where the licence does not let us mirror the weights.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gated: bool,
}
