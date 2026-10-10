//! Model registry + download engine. The set of models a paddock build can pull
//! is a **compiled-in manifest** (`models.toml`, embedded via `include_str!`), so
//! artifact/backend contracts determine what this build can offer. Test coverage
//! and benchmark progress are tracked separately, never used as availability gates.
//! There is no remote catalog to fetch or keep in sync; the origin
//! (Cloudflare R2) is a dumb file host, and each manifest entry carries the file's
//! stable URL, sha256 and size.
//!
//! The puller itself is fast: parallel HTTP-Range segments, resumable (a sidecar
//! segment map), SHA-256-verified, atomic temp->rename. Origin-agnostic - any host
//! with Range support works; a non-Range origin falls back to a single stream. The
//! browser never downloads - the server pulls to disk, where models load from.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, mpsc};

mod gated;
#[cfg(test)]
mod multimodal_mlx_tests;
pub(crate) use gated::valid_token;
pub use gated::{TokenSource, huggingface_token};
mod recovery;
mod transfer;
use transfer::fetch_range;
#[cfg(test)]
mod recovery_tests;
mod runtime;
pub use runtime::{ArtifactRuntime, ArtifactSource, Qualification, TowerMemory};

// ─── the embedded manifest (models.toml, compiled into the binary) ──────────

/// The blessed-models manifest this release ships with. Parsed once from the
/// compiled-in `models.toml`; author-controlled, so a parse failure is a build
/// bug, never a runtime condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    pub schema: u32,
    /// One entry per model. TOML spells this `[[model]]`; JSON emits `models`.
    #[serde(rename(serialize = "models", deserialize = "model"), default)]
    pub models: Vec<CatalogModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub display: String,
    /// Maker (e.g. "OpenAI", "Alibaba") - shown as its own column in the Studio.
    #[serde(default)]
    pub vendor: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    pub capability: Vec<String>,
    /// The model's speculative heads live in the weights file (qwen nextn,
    /// nemotron MTP): speculation needs no companion, and an attached
    /// drafter forms a HYBRID with them rather than replacing them. False =
    /// a drafter is the only mechanism (muse, gemma4, laguna).
    #[serde(default)]
    pub mtp_in_file: bool,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    /// KV cache precision this family serves at when nothing overrides it:
    /// "f16" or "fp8_e4m3". Absent = f16.
    ///
    /// MIRRORS the ENGINE and must be kept in step with it - the family
    /// default an `auto` runner serves. Since 2026-10-04 that is fp8-e4m3
    /// (KV8) for every generative family except paddleocr-vl, deepseek-ocr
    /// and qwen3-asr (each loader states why). The Studio PRESELECTS it and
    /// admission and `/estimate` PRICE an auto endpoint by it, so a row
    /// that drifts from its engine misprices every start.
    #[serde(default)]
    pub kv_default: Option<String>,
    /// Vendor-sourced spec sheet, shown when a row is expanded in the Studio.
    /// Grouped in one struct (with Default) so adding a field never breaks the
    /// construction sites; TOML spells it `[model.specs]`.
    #[serde(default)]
    pub specs: ModelSpecs,
    /// The model's PIECES (schema 3): weight alternatives + optional
    /// companions, each independently downloadable. TOML spells this
    /// `[[model.artifact]]`; JSON emits `artifacts`.
    #[serde(rename(serialize = "artifacts", deserialize = "artifact"), default)]
    pub artifacts: Vec<CatalogArtifact>,
}

/// What role a piece plays in a serving composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactKind {
    /// Alternatives - a server picks exactly one (the quality/format choice).
    Weights,
    /// mmproj vision tower (optional companion to GGUF weights).
    Vision,
    /// mmproj AUDIO tower - the speech encoder a generative ASR model
    /// (Qwen3-ASR) needs to hear anything. Deliberately its own kind rather
    /// than a Vision reuse: the two imply different capabilities and the
    /// picker must not offer image input for a model that only takes audio.
    /// Both are charged identically, because both are resident from startup.
    Audio,
    /// MTP/speculative drafter sideload (in-file MTP exports need none).
    Drafter,
    /// Official FP8/bf16 safetensors checkpoint dir. Today a native-plane
    /// source over a GGUF base (PADDOCK_FP8_NATIVE); becomes a weights
    /// alternative when the engine serves it directly.
    Fp8Snapshot,
    /// The text encoder an image-generation lane conditions on (Qwen-Image's
    /// Qwen3-VL GGUF). Required, resident from startup, never switchable -
    /// the DiT has nothing to draw from without it. Its own kind rather than
    /// a weights alternative because a server picks ONE weights artifact and
    /// this loads beside it; not Vision, because it rides `--text-encoder`
    /// and implies no image input.
    TextEncoder,
    /// The VAE that turns an image lane's latents into pixels (the official
    /// safetensors). Required and resident like the text encoder; rides
    /// `--vae`.
    Vae,
    /// A layout model a document reader runs ahead of itself (PaddleOCR-VL's
    /// PP-DocLayoutV3): a checkpoint folder inside the weights' folder, which
    /// the runner discovers there and loads beside the engine. Resident and
    /// charged like the lane companions, but not required - without it the
    /// reader still serves, page by page instead of region by region - and
    /// no flag carries it, so it has no switch either.
    Layout,
}

impl ArtifactKind {
    /// An image lane's resident companions - the pieces a DiT cannot serve
    /// without, resolved and charged beside it. Neither has a switch.
    pub fn is_lane_companion(self) -> bool {
        matches!(self, ArtifactKind::TextEncoder | ArtifactKind::Vae)
    }

    /// Does this ride the runner's `--mmproj` flag? Vision and Audio are
    /// separate KINDS because they imply different input capabilities, but
    /// they are the same KIND of FILE and the runner takes them through one
    /// flag - it reads the tower out of the GGUF and reports `vision` or
    /// `audio` accordingly. Anywhere the composition is assembled must ask
    /// this rather than name Vision, or a speech model resolves with no
    /// companion and refuses to start (Qwen3-ASR is the case that showed it).
    pub fn is_mmproj(self) -> bool {
        matches!(self, ArtifactKind::Vision | ArtifactKind::Audio)
    }
}

/// One independently downloadable piece of a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogArtifact {
    /// Unique within the model, e.g. "q8", "q4", "vision", "fp8".
    pub id: String,
    pub kind: ArtifactKind,
    /// "gguf" | "safetensors" - what the bytes are, for honest labeling.
    pub format: String,
    /// Human label, e.g. "Full quality", "Vision (mmproj BF16)".
    pub label: String,
    #[serde(default)]
    pub runtime: ArtifactRuntime,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ArtifactSource>,
    /// The honest quant tag for weights ("Q8_0", "UD-Q4_K_XL", "MXFP4").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    /// Part of the row-level Download bundle (default weights + every
    /// default companion).
    #[serde(default)]
    pub default: bool,
    /// This companion is the model's point - a vision artifact marked
    /// required gets no on/off switch in the Studio (granite-vision without
    /// its tower is a plain text model with the purpose gone).
    #[serde(default)]
    pub required: bool,
    /// Where this weights artifact sits on the model's choice axes
    /// (`ModelSpecs::choices`), by axis name: Kumo Tabular's
    /// `{ Size = "Large", Task = "Classification" }`. Empty for every model
    /// whose weights are one list of quality levels.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub choice: BTreeMap<String, String>,
    /// Minimum compute capability this artifact can be SERVED on, as
    /// `[major, minor]` - absent means every GPU the engine supports.
    ///
    /// Not every weight format runs everywhere. NVFP4's W4A16 consumers are
    /// compiled for sm_120a only (consumer Blackwell), and off that target the
    /// engine falls back to the base build - correct, but a fallback nobody
    /// warned about is a download that changed nothing. The Studio greys the
    /// choice out with the requirement named, instead of letting someone pull
    /// 22 GB to get the answers they already had.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_cc: Option<[u32; 2]>,
    /// Device bytes the loaded artifact holds BEYOND its file bytes - the
    /// persistent workspace the engine pins for it from load to shutdown.
    /// Most artifacts hold nothing worth naming; the two kinds that do:
    /// towers (deepseek-ocr's DeepEncoder pins ~950 MiB of encode slabs,
    /// more than its weight file) and MoE weights (gemma-4-26B-A4B's expert
    /// serving scratch self-reports 5.79 GiB at the default 32-slot width -
    /// double the estimator's whole graph margin). A fit estimate that skips
    /// either says "fits" about a start that isn't. Measured per release,
    /// recorded here as data next to the equally build-measured sha256/size,
    /// not as a per-family constant in estimate.rs (which the tower-pricing
    /// comment there rightly bans).
    #[serde(default)]
    pub workspace: Option<u64>,
    /// The artifact's SHAPE - everything will-it-fit needs that is intrinsic to
    /// the file. Published so the estimate is the same estimate
    /// before and after download.
    ///
    /// Before this, the picker had two answers: a real one for installed files
    /// (probe the header, run the estimator) and, for everything else,
    /// `file_bytes * 1.05 + tower + 1.5 GiB` - a fudge in the same family as the
    /// `total_size * 1.2 + 1 GB` guess estimate.rs records drifting 2.8x. The
    /// split was never measured-vs-predicted; both are estimates, and only one
    /// was built out of the model's real geometry.
    ///
    /// Wanted on every weights artifact - that is what "always" buys: no row can
    /// be shown to a user unpriced, and no artifact can be published without
    /// someone having established its cost. `source` says whether the numbers
    /// were probed or measured, because the GGUF probe cannot read every format
    /// we ship (nemotron's NVFP4 arm is safetensors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<paddock_estimator::PublishedShape>,
    /// TOML spells this `[[model.artifact.file]]`; JSON emits `files`.
    #[serde(rename(serialize = "files", deserialize = "file"), default)]
    pub files: Vec<CatalogFile>,
}

impl CatalogArtifact {
    pub fn entry_path(&self, models_dir: &Path) -> Option<PathBuf> {
        let file = models_dir.join(&self.files.first()?.dest);
        if self.runtime.checkpoint_dir {
            file.parent().map(Path::to_path_buf)
        } else {
            Some(file)
        }
    }

    pub fn capabilities<'a>(&'a self, model: &'a CatalogModel) -> &'a [String] {
        self.runtime
            .capability
            .as_deref()
            .unwrap_or(&model.capability)
    }

    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// Whether this artifact can run on a GPU of compute-capability `cc`: no
    /// `min_cc` floor, or `cc` meets it. `cc = None` (no card / unknown) fails
    /// any floor - a Blackwell-only lane must never be handed to a card we
    /// cannot confirm is Blackwell, so the picker falls back to a floorless one.
    pub fn fits_cc(&self, cc: Option<[u32; 2]>) -> bool {
        match self.min_cc {
            None => true,
            Some(f) => cc.is_some_and(|c| c[0] > f[0] || (c[0] == f[0] && c[1] >= f[1])),
        }
    }
}

/// Does this artifact contain a file whose dest matches `name` (an already
/// lower-cased last path component)? Name, stem, or parent-DIR - see
/// `Registry::identify_weights` for why all three. Shared so the forward and
/// reverse lookups cannot drift apart on what "the same file" means.
fn artifact_holds(a: &CatalogArtifact, name: &str) -> bool {
    a.files.iter().any(|f| {
        let d = Path::new(&f.dest);
        d.file_name()
            .is_some_and(|x| x.to_string_lossy().to_lowercase() == name)
            || d.file_stem()
                .is_some_and(|x| x.to_string_lossy().to_lowercase() == name)
            || d.parent()
                .and_then(|p| p.file_name())
                .is_some_and(|x| x.to_string_lossy().to_lowercase() == name)
    })
}

impl CatalogModel {
    /// Backend gates are independent of CUDA compute capability. Never pick
    /// an MLX checkpoint for CUDA just because it has no `min_cc` floor.
    pub fn default_weights_for_backend(
        &self,
        backend: &str,
        cc: Option<[u32; 2]>,
    ) -> Option<&CatalogArtifact> {
        let compatible = || {
            self.weights()
                .filter(|a| a.runtime.supports_backend(backend))
        };
        compatible()
            .find(|a| a.default && a.fits_cc(cc))
            .or_else(|| compatible().find(|a| a.fits_cc(cc)))
            .or_else(|| compatible().next())
    }

    pub fn default_bundle_for_backend(
        &self,
        backend: &str,
        cc: Option<[u32; 2]>,
    ) -> Vec<&CatalogArtifact> {
        let Some(w) = self.default_weights_for_backend(backend, cc) else {
            return Vec::new();
        };
        let mut out = vec![w];
        out.extend(self.artifacts.iter().filter(|a| {
            a.kind != ArtifactKind::Weights
                && a.default
                && a.runtime.supports_backend(backend)
                && w.runtime.allows_companion(&a.id)
                // a Blackwell-only drafter (nemotron's DSpark) is not pulled
                // onto a card that cannot run it
                && a.fits_cc(cc)
        }));
        out
    }

    /// A picture tower AND an audio tower, each its own artifact: EmbeddingGemma
    /// 2, whose upstream projector the catalog cuts in two so each tower is
    /// its own download. The picture tower then rides `mmproj`, the audio
    /// tower `audio_mmproj` with its own switch; every other model has one
    /// tower (or none) and keeps the single-`mmproj` rule.
    pub fn split_towers(&self) -> bool {
        self.artifacts
            .iter()
            .any(|a| a.kind == ArtifactKind::Vision)
            && self.artifacts.iter().any(|a| a.kind == ArtifactKind::Audio)
    }

    /// The audio tower of a split-tower model (see [`Self::split_towers`]).
    pub fn split_audio_tower(&self) -> Option<&CatalogArtifact> {
        self.split_towers()
            .then(|| {
                self.artifacts
                    .iter()
                    .find(|a| a.kind == ArtifactKind::Audio)
            })
            .flatten()
    }

    pub fn weights(&self) -> impl Iterator<Item = &CatalogArtifact> {
        self.artifacts
            .iter()
            .filter(|a| a.kind == ArtifactKind::Weights)
    }

    /// The NOMINAL default weights: the one marked `default`, else the first
    /// listed. Compute-capability-agnostic - for display and tests. The live
    /// serve/download paths use `default_weights_for` so a Blackwell-gated
    /// default falls back on a card that cannot run it.
    pub fn default_weights(&self) -> Option<&CatalogArtifact> {
        self.weights()
            .find(|a| a.default)
            .or_else(|| self.weights().next())
    }

    /// Hardware-aware default: the marked `default` if it runs on a GPU of
    /// compute-capability `cc`, else the first weights that does, else the
    /// marked default (readiness then refuses with a card-specific message
    /// rather than this returning None). This is what makes NVFP4 the default
    /// on Blackwell while Q8_0 stays the default on everything else, with no
    /// per-model `default` juggling.
    pub fn default_weights_for(&self, cc: Option<[u32; 2]>) -> Option<&CatalogArtifact> {
        let marked = self.default_weights();
        match marked {
            Some(m) if m.fits_cc(cc) => Some(m),
            _ => self.weights().find(|a| a.fits_cc(cc)).or(marked),
        }
    }

    pub fn artifact(&self, id: &str) -> Option<&CatalogArtifact> {
        self.artifacts.iter().find(|a| a.id == id)
    }

    /// The row-level Download bundle for a GPU of compute-capability `cc`: the
    /// hardware-aware default weights + every default companion - what "just
    /// download it" means for this model on this machine.
    pub fn default_bundle_for(&self, cc: Option<[u32; 2]>) -> Vec<&CatalogArtifact> {
        let mut out: Vec<&CatalogArtifact> = Vec::new();
        if let Some(w) = self.default_weights_for(cc) {
            out.push(w);
        }
        // companions the chosen weights take, on a card that can run them
        let weights = out.first().copied();
        out.extend(self.artifacts.iter().filter(|a| {
            a.kind != ArtifactKind::Weights
                && a.default
                && a.fits_cc(cc)
                && weights.is_none_or(|w| w.runtime.allows_companion(&a.id))
        }));
        out
    }

    /// cc-agnostic bundle (display/tests) - the NOMINAL marked default weights
    /// + default companions.
    pub fn default_bundle(&self) -> Vec<&CatalogArtifact> {
        let mut out: Vec<&CatalogArtifact> = Vec::new();
        if let Some(w) = self.default_weights() {
            out.push(w);
        }
        out.extend(
            self.artifacts
                .iter()
                .filter(|a| a.kind != ArtifactKind::Weights && a.default),
        );
        out
    }
}

/// Exact specs taken from the vendor's model card. Every field is optional - the
/// Studio renders whichever are present. Add fields freely: `Default` keeps the
/// struct-literal test sites compiling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelSpecs {
    /// Upstream model publication date (YYYY-MM-DD), not the export revision,
    /// upload time or registry insertion date. Absent until sourced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published_at: Option<String>,
    /// Evidence for the publication date, independent of the current model card.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published_source: Option<String>,
    /// Parameter count, e.g. "20.9B total · 3.6B active (MoE)".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<String>,
    /// Native max context length, e.g. "128K" - the value shown in the column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Extended context beyond native (e.g. via YaRN), e.g. "up to 1M with YaRN".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_max: Option<String>,
    /// Embedding dimension (embedding models), e.g. "1024 (MRL)".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dims: Option<String>,
    /// Canonical vendor model-card URL - the authoritative spec source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,
    /// One-line description in the vendor's own words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub about: Option<String>,
    /// Short "good at" bullets for the deploy comparison card. Factual and
    /// sourced (vendor card / our measurements) - honest-naming applies to
    /// prose too, so no marketing adjectives.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strengths: Vec<String>,
    /// The other side of the same card - what picking this model costs you.
    /// A catalog that only lists strengths is an ad, not a comparison.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tradeoffs: Vec<String>,
    /// The axes this model's weights vary along when they are not quality
    /// levels. Kumo Tabular ships six checkpoints that are a Size times a
    /// Task; laid out as one list of six "Quality" cards, the one real
    /// question (how big, which task) was hidden in the labels. Declared here
    /// in display order, with each artifact naming its value on every axis
    /// (`CatalogArtifact::choice`), so the Studio asks one small question per
    /// axis. TOML spells it `[[model.specs.choice]]`.
    #[serde(
        rename(serialize = "choices", deserialize = "choice"),
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub choices: Vec<ChoiceAxis>,
}

/// One question a model's weights answer: its heading (also the key every
/// artifact's `choice` uses) and its options in display order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceAxis {
    pub name: String,
    pub options: Vec<ChoiceOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceOption {
    pub value: String,
    /// One line that tells this option from its neighbours - sourced, like
    /// the strengths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFile {
    /// Stable, absolute download URL - never changes once published.
    pub url: String,
    /// Where the file lands, relative to the models dir. Same layout a manually
    /// placed model uses, so a pull de-dups against an already-present file.
    pub dest: String,
    pub sha256: String,
    pub size: u64,
}

// ─── download engine ────────────────────────────────────────────────────────

const SEGMENT: u64 = 16 * 1024 * 1024; // 16 MiB range segments
const WORKERS: usize = 8; // concurrent range connections

#[derive(Debug, thiserror::Error)]
pub enum DlError {
    #[error("network transfer failed: {0}")]
    Transport(String),
    #[error("origin is busy (status {status}); retry later")]
    RetryableStatus {
        status: u16,
        retry_after: Option<u64>,
    },
    #[error("http error: {0}")]
    Http(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("checksum mismatch for {name}: expected {expected}, got {got}")]
    Checksum {
        name: String,
        expected: String,
        got: String,
    },
    #[error("origin returned status {0}")]
    Status(u16),
    #[error(
        "model file is no longer available at the origin (removed, moved, or access revoked): {url}"
    )]
    NotFound { url: String },
    #[error("size mismatch: manifest says {expected}, origin served {got}")]
    Size { expected: u64, got: u64 },
    #[error("not enough disk space at {dir}: need {need} bytes, {free} free")]
    Disk { need: u64, free: u64, dir: String },
    #[error("download cancelled")]
    Cancelled,
    /// The origin is Hugging Face and it refused: a gated model this account
    /// has not been granted, or no token at all.
    #[error(
        "Hugging Face refused the download: this model is gated. Sign in at huggingface.co, \
         accept its licence at {page}, then add your Hugging Face token under Manager > \
         Settings (or sign in with `hf auth login`)"
    )]
    Gated { page: String },
}

/// Map a non-success HTTP status to the right error: a *definitively gone* file
/// (404 deleted / 410 gone / 403 access revoked - e.g. someone force-deleted it
/// from R2) is a distinct `NotFound` so callers can report it clearly and never
/// confuse it with a transient origin hiccup (502/503) that's worth retrying.
fn classify_status(status: reqwest::StatusCode, url: &str) -> DlError {
    use reqwest::StatusCode;
    // Hugging Face answers a gated file 401 without a token and 403 with one
    // that has not been granted access - neither means the file is gone
    if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
        && let Some(page) = gated::repo_page(url)
    {
        return DlError::Gated { page };
    }
    match status {
        StatusCode::NOT_FOUND | StatusCode::GONE | StatusCode::FORBIDDEN => DlError::NotFound {
            url: url.to_owned(),
        },
        s => DlError::Status(s.as_u16()),
    }
}

/// Free (available-to-the-user) bytes on the volume holding `path`; `None` if
/// the platform query fails.
pub fn disk_free(path: &Path) -> Option<u64> {
    fs4::available_space(path).ok()
}
/// Total bytes on the volume holding `path`.
pub fn disk_total(path: &Path) -> Option<u64> {
    fs4::total_space(path).ok()
}

/// Lowercase hex of a digest. Shared because `sha2` 0.11 moved its output to
/// `hybrid_array::Array`, which - unlike the old `GenericArray` - does not
/// implement `LowerHex`, so `format!("{:x}", h.finalize())` no longer compiles.
/// Encoding the bytes ourselves is a few lines and keeps a hex crate out of the
/// graph.
pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Cross-platform positioned write. Each worker holds its own file handle and
/// writes a disjoint region, so there's no shared-cursor race.
fn write_at(f: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.write_all_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        // seek_write is a short-write API, so we loop - shadowed here rather
        // than `mut` on the args, which would warn on the unix arm.
        let (mut buf, mut offset) = (buf, offset);
        while !buf.is_empty() {
            let n = f.seek_write(buf, offset)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "seek_write wrote 0",
                ));
            }
            buf = &buf[n..];
            offset += n as u64;
        }
        Ok(())
    }
}

fn part_path(dest: &Path) -> PathBuf {
    sidecar_path(dest, ".part")
}
fn state_path(dest: &Path) -> PathBuf {
    sidecar_path(dest, ".part.state")
}
fn sidecar_path(dest: &Path, suffix: &str) -> PathBuf {
    let mut name = dest.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn bind_resume(dest: &Path, url: &str, size: u64, sha: &str) -> Result<(), DlError> {
    let identity = sidecar_path(dest, ".part.identity");
    let expected = hex(&Sha256::digest(format!("{url}\n{size}\n{sha}")));
    if std::fs::read_to_string(&identity).ok().as_deref() != Some(&expected)
        || !std::fs::metadata(part_path(dest)).is_ok_and(|m| m.is_file() && m.len() == size)
    {
        match std::fs::remove_file(state_path(dest)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    let tmp = sidecar_path(dest, ".part.identity.tmp");
    std::fs::write(&tmp, expected)?;
    // Opened for writing: Windows refuses to flush a read-only handle
    // ("Access is denied"), where Unix lets a read-only fd fsync.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&tmp)?
        .sync_all()?;
    std::fs::rename(tmp, identity)?;
    Ok(())
}

fn seg_len(idx: usize, size: u64) -> u64 {
    let start = idx as u64 * SEGMENT;
    (start + SEGMENT).min(size) - start
}

/// Load the resume bitmap (`1` byte per completed segment). Returns all-false
/// when absent or stale (wrong length).
fn load_state(path: &Path, n_seg: usize) -> Vec<bool> {
    match std::fs::read(path) {
        Ok(bytes) if bytes.len() == n_seg => bytes.iter().map(|&b| b == 1).collect(),
        _ => vec![false; n_seg],
    }
}

/// Probe the origin: `Ok(true)` = honours Range (206), `Ok(false)` = serves the
/// whole file (2xx, no Range). A definitively-gone file (404/410/403) errors
/// here as `NotFound` - fail fast, before any `.part` file is created on disk.
async fn supports_range(client: &reqwest::Client, url: &str) -> Result<bool, DlError> {
    let resp = client
        .get(url)
        .timeout(std::time::Duration::from_secs(20))
        .header(reqwest::header::ACCEPT_ENCODING, "identity")
        .header(reqwest::header::RANGE, "bytes=0-0")
        .send()
        .await
        .map_err(|e| DlError::Http(e.to_string()))?;
    let status = resp.status();
    if status == reqwest::StatusCode::PARTIAL_CONTENT {
        return Ok(true);
    }
    if status.is_success() {
        return Ok(false);
    }
    Err(classify_status(status, url))
}

async fn sha256_file_cancel(
    path: PathBuf,
    cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> Result<String, DlError> {
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut f = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 8 * 1024 * 1024];
        loop {
            if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(DlError::Cancelled);
            }
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok::<String, DlError>(hex(&hasher.finalize()))
    })
    .await
    .map_err(|e| DlError::Http(format!("hash task: {e}")))?
}

/// Dropping an in-flight HTTP future closes that request. Hashing is different:
/// its blocking worker checks the flag itself and is always joined before exit.
async fn cancellable<T>(
    cancel: Option<&Arc<std::sync::atomic::AtomicBool>>,
    future: impl std::future::Future<Output = Result<T, DlError>>,
) -> Result<T, DlError> {
    tokio::pin!(future);
    loop {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(DlError::Cancelled);
        }
        tokio::select! {
            result = &mut future => return result,
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
        }
    }
}

/// Download `url` -> `dest`, verifying `sha256` (+ `size`). Parallel range
/// segments (single-stream fallback for non-Range origins); resumes from a
/// `<dest>.part` + `.part.state` sidecar; atomic rename on success. `downloaded`
/// is the live byte counter the caller reads for progress. `cancel` (optional)
/// is checked between segments/chunks: on cancel the partial bytes and the
/// sidecar STAY on disk so a later call resumes. **CPU/IO + network heavy** -
/// call from a task, not a latency-sensitive path.
pub async fn download_file(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    sha256: &str,
    size: u64,
    downloaded: Arc<AtomicU64>,
    cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(), DlError> {
    download_file_staged(client, url, dest, sha256, size, downloaded, cancel, None).await
}

async fn download_file_staged(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    sha256: &str,
    size: u64,
    downloaded: Arc<AtomicU64>,
    cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
    stage: Option<&std::sync::Mutex<&'static str>>,
) -> Result<(), DlError> {
    if let Some(stage) = stage {
        *stage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = "downloading";
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = part_path(dest);
    let want_sha = sha256.to_lowercase();

    if cancellable(cancel.as_ref(), supports_range(client, url)).await? {
        bind_resume(dest, url, size, &want_sha)?;
        download_ranged(client, url, &part, dest, size, &downloaded, cancel.as_ref()).await?;
    } else {
        download_stream(client, url, &part, size, &downloaded, cancel.as_ref()).await?;
    }

    // verify then publish atomically
    if let Some(stage) = stage {
        *stage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = "verifying";
    }
    let length = std::fs::metadata(&part)?.len();
    if length != size {
        return Err(DlError::Size {
            expected: size,
            got: length,
        });
    }
    let got = sha256_file_cancel(part.clone(), cancel.clone()).await?;
    if got != want_sha {
        // Never trust the same bitmap on Retry after failed integrity. The
        // incomplete bytes remain recoverable, but every segment is fetched again.
        let _ = std::fs::remove_file(state_path(dest));
        return Err(DlError::Checksum {
            name: dest
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_owned(),
            expected: want_sha,
            got,
        });
    }
    if cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
        return Err(DlError::Cancelled);
    }
    std::fs::OpenOptions::new()
        .write(true)
        .open(&part)?
        .sync_all()?;
    std::fs::rename(&part, dest)?;
    let _ = std::fs::remove_file(state_path(dest));
    let _ = std::fs::remove_file(sidecar_path(dest, ".part.identity"));
    Ok(())
}

async fn download_ranged(
    client: &reqwest::Client,
    url: &str,
    part: &Path,
    dest: &Path,
    size: u64,
    downloaded: &Arc<AtomicU64>,
    cancel: Option<&Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(), DlError> {
    // preallocate the part file to the final size (idempotent on resume)
    {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(part)?;
        f.set_len(size)?;
    }
    let n_seg = size.div_ceil(SEGMENT) as usize;
    let statep = state_path(dest);
    let done = load_state(&statep, n_seg);
    // already-done bytes count toward progress immediately
    for (i, &d) in done.iter().enumerate() {
        if d {
            downloaded.fetch_add(seg_len(i, size), Ordering::Relaxed);
        }
    }
    let pending: Vec<usize> = (0..n_seg).filter(|&i| !done[i]).collect();
    if pending.is_empty() {
        return Ok(());
    }
    let queue = Arc::new(Mutex::new(pending.into_iter()));

    // one persister task: single-byte positioned writes to the state sidecar as
    // segments complete, so a crash resumes without re-downloading them.
    let (state_tx, mut state_rx) = mpsc::channel::<usize>(WORKERS * 2);
    let statep2 = statep.clone();
    let datap = part.to_path_buf();
    let n_seg_u = n_seg as u64;
    let persister = tokio::spawn(async move {
        let sf = Arc::new(
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(&statep2)?,
        );
        let data = Arc::new(std::fs::OpenOptions::new().write(true).open(datap)?);
        // full length up front so an out-of-order completion never leaves the
        // sidecar short of n_seg (which load_state requires to resume).
        sf.set_len(n_seg_u)?;
        while let Some(idx) = state_rx.recv().await {
            let mut batch = vec![idx];
            while let Ok(idx) = state_rx.try_recv() {
                batch.push(idx);
            }
            let sf = sf.clone();
            let data = data.clone();
            // Commit data before its resume bits, off the async executor.
            // Group completed segments so durable progress doesn't demand a
            // filesystem sync for every worker's individual write.
            tokio::task::spawn_blocking(move || {
                data.sync_data()?;
                for idx in batch {
                    write_at(&sf, &[1u8], idx as u64)?;
                }
                sf.sync_data()?;
                Ok::<(), DlError>(())
            })
            .await
            .map_err(|e| DlError::Http(format!("resume commit: {e}")))??;
        }
        sf.sync_all()?;
        Ok::<(), DlError>(())
    });

    let mut tasks = Vec::new();
    // A terminal range failure stops peers promptly, without detaching disk
    // writes or returning while workers can still mutate resumable files.
    let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    for _ in 0..WORKERS.min(n_seg) {
        let client = client.clone();
        let url = url.to_owned();
        let part = part.to_path_buf();
        let queue = queue.clone();
        let downloaded = downloaded.clone();
        let state_tx = state_tx.clone();
        let cancel = cancel.cloned();
        let failed = failed.clone();
        tasks.push(tokio::spawn(async move {
            let outcome = async {
                let fh = Arc::new(std::fs::OpenOptions::new().write(true).open(&part)?);
                loop {
                    // cooperative cancel between segments: completed segments are
                    // already persisted in the sidecar, so nothing is lost
                    if failed.load(Ordering::Relaxed)
                        || cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed))
                    {
                        return Err(DlError::Cancelled);
                    }
                    let idx = {
                        let mut q = queue.lock().await;
                        q.next()
                    };
                    let Some(idx) = idx else { break };
                    let start = idx as u64 * SEGMENT;
                    let end = (start + SEGMENT).min(size) - 1;
                    let (bytes, progress) = cancellable(
                        cancel.as_ref(),
                        cancellable(
                            Some(&failed),
                            fetch_range(&client, &url, start, end, size, &downloaded),
                        ),
                    )
                    .await?;
                    let file = fh.clone();
                    tokio::task::spawn_blocking(move || write_at(&file, &bytes, start))
                        .await
                        .map_err(|e| DlError::Http(format!("range write: {e}")))??;
                    state_tx.send(idx).await.map_err(|_| {
                        DlError::Http("Resume checkpoint could not be saved".into())
                    })?;
                    progress.commit();
                }
                Ok::<(), DlError>(())
            }
            .await;
            if outcome.is_err() {
                failed.store(true, Ordering::Relaxed);
            }
            outcome
        }));
    }
    drop(state_tx); // so the persister ends when all workers finish
    let mut outcome: Result<(), DlError> = Ok(());
    for t in tasks {
        let r = t
            .await
            .unwrap_or_else(|e| Err(DlError::Http(format!("worker: {e}"))));
        // join every worker before reporting (a cancel hits all of them);
        // keep the first real error, with Cancelled winning only over Ok
        match (&outcome, r) {
            (Ok(()), Err(e)) => outcome = Err(e),
            (Err(DlError::Cancelled), Err(e)) if !matches!(e, DlError::Cancelled) => {
                outcome = Err(e);
            }
            _ => {}
        }
    }
    persister
        .await
        .map_err(|e| DlError::Http(format!("resume-state worker: {e}")))??;
    outcome
}

/// Non-Range origin: one stream, written sequentially.
async fn download_stream(
    client: &reqwest::Client,
    url: &str,
    part: &Path,
    size: u64,
    downloaded: &Arc<AtomicU64>,
    cancel: Option<&Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(), DlError> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    let resp = cancellable(cancel, async {
        client
            .get(url)
            .header(reqwest::header::ACCEPT_ENCODING, "identity")
            .send()
            .await
            .map_err(|e| DlError::Http(e.to_string()))
    })
    .await?;
    if !resp.status().is_success() {
        return Err(classify_status(resp.status(), url));
    }
    let mut file = tokio::fs::File::create(part).await?;
    let mut stream = resp.bytes_stream();
    let mut received = 0u64;
    while let Some(chunk) = cancellable(cancel, async {
        tokio::time::timeout(std::time::Duration::from_secs(30), stream.next())
            .await
            .map_err(|_| DlError::Http("Origin stalled while downloading".into()))
    })
    .await?
    {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(DlError::Cancelled);
        }
        let chunk = chunk.map_err(|e| DlError::Http(e.to_string()))?;
        received += chunk.len() as u64;
        if received > size {
            return Err(DlError::Size {
                expected: size,
                got: received,
            });
        }
        file.write_all(&chunk).await?;
        downloaded.fetch_add(chunk.len() as u64, Ordering::Relaxed);
    }
    file.flush().await?;
    Ok(())
}

// ─── pull manager (stateful, on AppState) ───────────────────────────────────

/// Status of a pull job, JSON-tagged for the Studio to poll.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "lowercase")]
pub enum PullStatus {
    Running,
    Done,
    /// User-stopped. Partial bytes + the segment sidecar stay on disk, so a
    /// resume (a fresh job over the same selection) continues where this one
    /// stopped instead of starting over.
    Cancelled,
    Error {
        message: String,
    },
}

/// One in-flight (or finished) model pull. `downloaded`/`total` drive the bar.
pub struct PullJob {
    pub id: String,
    pub model_id: String,
    /// The catalog's human name at start time - so every progress surface can
    /// say "Qwen 3.5 9B" without a catalog lookup.
    pub display: String,
    /// The artifact selection this job was started with (None = default
    /// bundle) - retained so a resume re-pulls exactly the same pieces.
    pub artifacts: Option<Vec<String>>,
    pub downloaded: Arc<AtomicU64>,
    pub total: u64,
    /// Unix millis at creation - orders the jobs list.
    pub created_ms: u64,
    pub status: std::sync::Mutex<PullStatus>,
    /// Cooperative cancel: download workers check it between range segments.
    pub cancel: Arc<std::sync::atomic::AtomicBool>,
    /// A queued follow-up ("start this spawn spec when the bytes land") and
    /// its live state. The ROUTES layer owns the meaning and the orchestration;
    /// they ride the job so one snapshot tells the whole story and a resume
    /// keeps the plan.
    pub follow: std::sync::Mutex<Option<serde_json::Value>>,
    pub follow_state: std::sync::Mutex<Option<serde_json::Value>>,
    /// Frozen catalog identity and destinations prevent resume across an
    /// updated manifest and overlapping writers across different model IDs.
    files: Vec<CatalogFile>,
    selection_digest: String,
    phase: std::sync::Mutex<String>,
    stage: std::sync::Mutex<&'static str>,
}

impl PullJob {
    pub fn snapshot(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "id": self.id,
            "model": self.model_id,
            "display": self.display,
            "artifacts": self.artifacts,
            "downloaded": self.downloaded.load(Ordering::Relaxed),
            "total": self.total,
            "created_ms": self.created_ms,
            "status": &*self.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            "phase": &*self.phase.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            "stage": &*self.stage.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            "cancelling": self.cancel.load(Ordering::Relaxed),
        });
        if let Some(f) = &*self
            .follow
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            v["start"] = serde_json::json!({
                "port": f.get("spec").and_then(|s| s.get("port")),
                "action": f.get("action"),
                "state": &*self.follow_state.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            });
        }
        v
    }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Model registry: the compiled-in manifest of pullable models plus the pull
/// jobs into the models dir. One per server (on `AppState`).
pub struct Registry {
    client: reqwest::Client,
    /// Immutable declared contracts. Re-project from these when selecting a
    /// backend, never from an already narrowed Metal view of the catalog.
    source_catalog: Catalog,
    catalog: Catalog,
    models_dir: PathBuf,
    jobs: std::sync::Mutex<std::collections::HashMap<String, Arc<PullJob>>>,
    /// This box's GPU compute-capability, from `readiness::probe` (None = no
    /// card / not wired). The live serve + download paths resolve default
    /// weights against it so a Blackwell-only lane (NVFP4) is the default here
    /// and Q8_0 the default on a card that cannot run it - see
    /// `CatalogModel::default_weights_for`.
    cc: Option<[u32; 2]>,
    backend: String,
    store: Option<Arc<crate::store::Store>>,
}

/// The release's blessed-models manifest, compiled into the binary.
const MANIFEST_TOML: &str = include_str!("../models.toml");

/// Rows that are not published with the source: checkpoints we serve but do
/// not hand out. The file is optional - `build.rs` sets `private_catalog` only
/// where it exists - so a tree without it builds the published catalog and
/// nothing else.
#[cfg(private_catalog)]
const PRIVATE_MANIFEST_TOML: &str = include_str!("../models.private.toml");

/// A model resolved to a concrete serving composition on disk.
#[derive(Debug)]
pub struct Resolved {
    /// The weights to load (the selected weights artifact's main file).
    pub weights: PathBuf,
    /// The vision tower (mmproj), when a vision artifact is installed.
    pub mmproj: Option<PathBuf>,
    /// An image lane's text encoder and VAE, when installed - the DiT is
    /// `weights`, these ride `--text-encoder` / `--vae`. Restricted to what
    /// the chosen weights allow, so the compact DiT gets the compact encoder.
    pub text_encoder: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    /// The drafter to wire in without being asked: installed AND marked
    /// default in the catalog - e.g. gemma-4-31b's 470M assistant.
    pub mtp: Option<PathBuf>,
    /// Any installed drafter, default or not. This is what an explicit
    /// `spec = "auto"` reaches for: opting into speculation is exactly the
    /// permission a non-default drafter was waiting for.
    pub drafter_any: Option<PathBuf>,
    /// The catalog declares a drafter for this model (installed or not). Lets
    /// a spawn that was ASKED for speculation fail with "download the drafter"
    /// instead of quietly serving without it.
    pub drafter_declared: bool,
    /// Which drafter artifact `mtp`/`drafter_any` came from - `(id, label)`,
    /// e.g. `("drafter2", "Speed drafter (DFlash2)")`. None when the model
    /// speculates from an IN-FILE MTP (no drafter artifact) or declares none.
    /// Surfaced so "Speculation: On" can say which drafter it wired: with two
    /// DFlash versions catalogued, "on" alone is no longer self-explanatory.
    pub drafter_pick: Option<(String, String)>,
    /// The catalog claims the `speculative` capability: this engine implements
    /// speculative decode for this model, either from in-file MTP (`nextn`) or
    /// a drafter artifact. A model without it must REFUSE a spec request -
    /// accepting a setting that does nothing is the silent failure the
    /// principles forbid.
    pub speculative: bool,
    /// An installed FP8/bf16 snapshot DIRECTORY (native-plane source for the
    /// engine's PADDOCK_FP8_NATIVE ingestion). Present ≠ used: the spawn only
    /// wires it when explicitly asked.
    pub fp8_snapshot: Option<PathBuf>,
}

impl Registry {
    /// Where a catalog model's pieces live - or will live once pulled
    /// (models_dir + each file's dest). None for unknown ids. The config
    /// PREVIEW uses this so a not-yet-downloaded model still shows its real
    /// future paths instead of refusing to render.
    pub fn planned_paths(
        &self,
        id: &str,
        artifact: Option<&str>,
    ) -> Option<(PathBuf, Option<PathBuf>, Option<PathBuf>)> {
        let m = self.catalog.models.iter().find(|m| m.id == id)?;
        let w = match artifact {
            Some(a) => m
                .artifacts
                .iter()
                .find(|x| x.id == a && x.kind == ArtifactKind::Weights)?,
            None => m.default_weights_for_backend(&self.backend, self.cc)?,
        };
        if !w.runtime.supports_backend(&self.backend) {
            return None;
        }
        let dest = |a: &CatalogArtifact| a.files.first().map(|f| self.models_dir.join(&f.dest));
        let weights = w.entry_path(&self.models_dir)?;
        // Vision OR Audio: both are mmproj companions riding the same
        // `--mmproj` flag, and a model has one or the other (a speech tower
        // and an image tower are different files for different senses). Only
        // matching Vision here is what left Qwen3-ASR unservable - the
        // manager pulled its speech encoder and then never passed it, so the
        // runner refused with "pass its audio mmproj" for a file already on
        // disk.
        let split = m.split_towers();
        let mmproj = m
            .artifacts
            .iter()
            .find(|a| {
                a.kind.is_mmproj()
                    && (!split || a.kind == ArtifactKind::Vision)
                    && a.default
                    && w.runtime.allows_companion(&a.id)
                    && a.runtime.supports_backend(&self.backend)
            })
            .and_then(dest);
        let mtp = m
            .artifacts
            .iter()
            .find(|a| {
                a.kind == ArtifactKind::Drafter
                    && a.default
                    && w.runtime.allows_companion(&a.id)
                    && a.runtime.supports_backend(&self.backend)
                    && a.fits_cc(self.cc)
            })
            .and_then(dest);
        Some((weights, mmproj, mtp))
    }

    /// A split-tower model's audio tower for a spawn of `artifact`: where it
    /// lives (or will, once pulled), whether it is on disk, whether the
    /// catalog marks it default, and its label. None for every other model.
    pub fn audio_tower(
        &self,
        model_id: &str,
        artifact: Option<&str>,
    ) -> Option<(PathBuf, bool, bool, String)> {
        let m = self.catalog.models.iter().find(|m| m.id == model_id)?;
        let a = m.split_audio_tower()?;
        let w = self.weights_for(m, artifact)?;
        if !w.runtime.allows_companion(&a.id) || !a.runtime.supports_backend(&self.backend) {
            return None;
        }
        let f = a.files.first()?;
        Some((
            self.models_dir.join(&f.dest),
            self.is_artifact_installed(a),
            a.default,
            a.label.clone(),
        ))
    }

    /// The mmproj companion this model cannot be served without, if it
    /// declares one: `Some(Ok(path))` when it is on disk, `Some(Err(label))`
    /// when the catalog declares it but the bytes are missing, `None` when the
    /// model needs no companion (or is not ours).
    ///
    /// `required` is the discriminator, and it is what makes acting on this
    /// safe: a VISION tower is a default-but-optional companion an operator
    /// may deliberately drop to get its VRAM back, while a required one means
    /// "the engine refuses to serve this architecture without it" - serving
    /// without it is not a choice, it is a crash.
    pub fn required_companion(&self, model_id: &str) -> Option<Result<PathBuf, String>> {
        let m = self.catalog.models.iter().find(|m| m.id == model_id)?;
        let a = m
            .artifacts
            .iter()
            .find(|a| a.kind.is_mmproj() && a.required)?;
        let f = a.files.first()?;
        Some(if self.is_artifact_installed(a) {
            Ok(self.models_dir.join(&f.dest))
        } else {
            Err(a.label.clone())
        })
    }

    /// The weights artifact a bare model id means: the backend's default if
    /// installed, else any installed weights for the backend, else the
    /// default - so a not-yet-downloaded model still names its plan.
    fn elect_weights<'a>(&self, model: &'a CatalogModel) -> Option<&'a CatalogArtifact> {
        model
            .default_weights_for_backend(&self.backend, self.cc)
            .filter(|a| self.is_artifact_installed(a))
            .or_else(|| {
                model.weights().find(|a| {
                    a.runtime.supports_backend(&self.backend) && self.is_artifact_installed(a)
                })
            })
            .or_else(|| model.default_weights_for_backend(&self.backend, self.cc))
    }

    /// The weights artifact a spawn spec names - an explicit id, else the
    /// election above.
    fn weights_for<'a>(
        &self,
        model: &'a CatalogModel,
        artifact: Option<&str>,
    ) -> Option<&'a CatalogArtifact> {
        match artifact {
            Some(a) => model
                .artifacts
                .iter()
                .find(|x| x.id == a && x.kind == ArtifactKind::Weights),
            None => self.elect_weights(model),
        }
    }

    /// Where an image lane's text encoder and VAE live, or will: the default
    /// of each kind the chosen weights allow, installed or not - the preview's
    /// and the save's planned paths for a model whose download has not run.
    pub fn planned_lane_companions(
        &self,
        id: &str,
        artifact: Option<&str>,
    ) -> (Option<PathBuf>, Option<PathBuf>) {
        let Some(m) = self.catalog.models.iter().find(|m| m.id == id) else {
            return (None, None);
        };
        let Some(w) = self.weights_for(m, artifact) else {
            return (None, None);
        };
        let planned = |kind: ArtifactKind| {
            m.artifacts
                .iter()
                .find(|a| {
                    a.kind == kind
                        && a.default
                        && w.runtime.allows_companion(&a.id)
                        && a.runtime.supports_backend(&self.backend)
                })
                .and_then(|a| a.files.first().map(|f| self.models_dir.join(&f.dest)))
        };
        (
            planned(ArtifactKind::TextEncoder),
            planned(ArtifactKind::Vae),
        )
    }

    /// The label of a required text encoder or VAE the chosen weights need
    /// and the disk does not have; None when the model needs none, or has
    /// them all. `required_companion`'s contract for the lane pieces, which
    /// unlike a vision tower have no switch: a config missing one is a start
    /// that cannot succeed, so the save refuses and names the download.
    pub fn missing_required_lane_companion(
        &self,
        id: &str,
        artifact: Option<&str>,
    ) -> Option<String> {
        let m = self.catalog.models.iter().find(|m| m.id == id)?;
        let w = self.weights_for(m, artifact)?;
        m.artifacts
            .iter()
            .filter(|a| {
                a.kind.is_lane_companion()
                    && a.required
                    && w.runtime.allows_companion(&a.id)
                    && a.runtime.supports_backend(&self.backend)
            })
            .find(|a| !self.is_artifact_installed(a))
            .map(|a| a.label.clone())
    }

    /// Parse the embedded manifest. It ships with the binary and is author-
    /// controlled, so a parse failure is a build bug, not a runtime condition.
    pub fn new(models_dir: PathBuf) -> Self {
        #[cfg_attr(not(private_catalog), allow(unused_mut))]
        let mut catalog: Catalog =
            toml::from_str(MANIFEST_TOML).expect("embedded models.toml is malformed");
        // After the published rows, so nothing about their order or their
        // defaults depends on whether this file was there at build time.
        #[cfg(private_catalog)]
        {
            let private: Catalog = toml::from_str(PRIVATE_MANIFEST_TOML)
                .expect("embedded models.private.toml is malformed");
            assert_eq!(
                private.schema, catalog.schema,
                "models.private.toml is on a different schema than models.toml"
            );
            for m in private.models {
                assert!(
                    catalog.models.iter().all(|p| p.id != m.id),
                    "models.private.toml repeats the published id {:?}",
                    m.id
                );
                catalog.models.push(m);
            }
        }
        Self::from_catalog(catalog, models_dir)
    }

    /// Build a registry over an explicit catalog - used by tests to point the
    /// puller at a local origin instead of the baked-in R2 URLs.
    pub fn from_catalog(catalog: Catalog, models_dir: PathBuf) -> Self {
        Self {
            client: reqwest::Client::new(),
            source_catalog: catalog.clone(),
            catalog,
            models_dir,
            jobs: std::sync::Mutex::new(std::collections::HashMap::new()),
            cc: None,
            backend: "cuda".into(),
            store: None,
        }
        .with_backend("cuda")
    }

    /// Wire the local GPU's compute-capability (from `readiness::probe`) so the
    /// live default-weights resolution is hardware-aware. Builder form: the
    /// registry is built once at startup, then wrapped in an Arc.
    pub fn with_cc(mut self, cc: Option<[u32; 2]>) -> Self {
        self.cc = cc;
        self
    }

    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backend = backend.into();
        self.catalog = self.source_catalog.clone();
        for model in &mut self.catalog.models {
            for artifact in &mut model.artifacts {
                if let Some(default) = artifact
                    .runtime
                    .backend_overrides
                    .get(&self.backend)
                    .and_then(|contract| contract.default)
                {
                    artifact.default = default;
                }
                artifact.runtime = artifact.runtime.for_backend(&self.backend);
                if let Some(shape) = artifact
                    .runtime
                    .memory
                    .as_ref()
                    .and_then(|m| m.published_shape.clone())
                {
                    artifact.shape = Some(shape);
                }
                if let Some(workspace) = artifact
                    .runtime
                    .memory
                    .as_ref()
                    .and_then(|m| m.workspace_bytes)
                {
                    artifact.workspace = Some(workspace);
                }
                if let (Some(memory), Some(shape)) = (&artifact.runtime.memory, &mut artifact.shape)
                {
                    memory.apply_published(shape);
                }
                // A personal Metal chat starts with 32K, not the runner's
                // legacy 4K server default. Publish the same recommendation
                // consumed by native creation, previews and web admission.
                // Export-specific defaults and real loader/model ceilings win;
                // speech/embedding companions and saved files are untouched.
                if self.backend == "metal"
                    && artifact.kind == ArtifactKind::Weights
                    && artifact.runtime.supports_backend("metal")
                    && artifact
                        .runtime
                        .capability
                        .as_ref()
                        .unwrap_or(&model.capability)
                        .iter()
                        .any(|c| c == "chat")
                {
                    let ceiling = artifact
                        .shape
                        .as_ref()
                        .map(|s| s.max_ctx)
                        .into_iter()
                        .chain(artifact.runtime.memory.as_ref().map(|m| m.max_ctx))
                        .min()
                        .unwrap_or(1_048_576) as usize;
                    artifact.runtime.default_max_ctx = Some(
                        artifact
                            .runtime
                            .default_max_ctx
                            .unwrap_or(32_768)
                            .min(ceiling),
                    );
                }
            }
        }
        self
    }

    /// The manifest is always compiled in; `false` only if it were empty.
    pub fn enabled(&self) -> bool {
        !self.catalog.models.is_empty()
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// The manifest with install state annotated per ARTIFACT (and rolled up
    /// per model), so the Studio shows piece-level state without stat'ing from
    /// the browser. `installed` on the model = at least one weights artifact
    /// present (it is servable); `total_size` = the default bundle's bytes
    /// (what the row-level Download button means).
    pub fn catalog_annotated(&self) -> serde_json::Value {
        let models: Vec<serde_json::Value> = self
            .catalog
            .models
            .iter()
            .map(|m| {
                let mut v = serde_json::to_value(m).unwrap_or_default();
                v["installed"] = serde_json::json!(self.is_installed(m));
                v["total_size"] = serde_json::json!(
                    m.default_bundle_for_backend(&self.backend, self.cc)
                        .iter()
                        .map(|a| a.total_size())
                        .sum::<u64>()
                );
                if let Some(arts) = v.get_mut("artifacts").and_then(|a| a.as_array_mut()) {
                    for (av, a) in arts.iter_mut().zip(&m.artifacts) {
                        av["installed"] = serde_json::json!(self.is_artifact_installed(a));
                        av["total_size"] = serde_json::json!(a.total_size());
                        av["backend_supported"] =
                            serde_json::json!(a.runtime.supports_backend(&self.backend));
                        av["kv_offload_supported"] = serde_json::json!(
                            self.backend != "metal"
                                || crate::backend_contract::metal_kv_offload(m, a)
                        );
                    }
                }
                v
            })
            .collect();
        serde_json::json!({ "schema": self.catalog.schema, "models": models })
    }

    /// Is every file of this artifact present locally at the right size?
    pub fn is_artifact_installed(&self, a: &CatalogArtifact) -> bool {
        !a.files.is_empty()
            && a.files.iter().all(|f| {
                std::fs::metadata(self.models_dir.join(&f.dest))
                    .map(|md| md.len())
                    .ok()
                    == Some(f.size)
            })
    }

    /// Servable now: at least one weights artifact fully present.
    pub fn is_installed(&self, m: &CatalogModel) -> bool {
        m.weights()
            .any(|a| a.runtime.supports_backend(&self.backend) && self.is_artifact_installed(a))
    }

    /// Reverse resolution: which catalog `(model id, weights-artifact id)`
    /// does this weights file belong to? Matched by file NAME or STEM
    /// (case-insensitive) against every weights artifact's dest - the same
    /// rule the Studio's edit page uses. The stem match matters because a
    /// runner's file-derived model id drops the `.gguf` ("Qwen3.5-9B-Q8_0"),
    /// and the parent-DIR match because directory-shaped serving (safetensors
    /// checkpoints: the forced aligner, HF-dir lanes) reports the directory
    /// as its id ("Qwen3-ForcedAligner-0.6B-hf") - without it those runners
    /// showed no name and no vendor in the fleet.
    ///
    /// GUESSWORK, and known to be: it reads a NAME. A renamed file, a copy
    /// outside the models dir, or an imported-in-place GGUF returns None even
    /// though the endpoint is perfectly well configured. That is why config
    /// files carry their own `[catalog]` block now and why this is
    /// the FALLBACK inside `identity_for`, not the answer.
    pub fn identify_weights(&self, path: &Path) -> Option<(String, String)> {
        // Safetensors packages commonly share `model.safetensors` (and shard
        // basenames). The qualified checkpoint path outranks that basename,
        // independent of catalog ordering and the host's path separator.
        let normalized = path.to_string_lossy().replace('\\', "/").to_lowercase();
        let normalized = Path::new(&normalized);
        for m in &self.catalog.models {
            for a in m.weights() {
                if a.files.iter().any(|f| {
                    let dest = f.dest.replace('\\', "/").to_lowercase();
                    let dest = Path::new(&dest);
                    dest.parent().is_some_and(|p| !p.as_os_str().is_empty())
                        && normalized.ends_with(dest)
                }) {
                    return Some((m.id.clone(), a.id.clone()));
                }
            }
        }
        let name = normalized.file_name()?.to_string_lossy();
        let mut found: Option<(String, String)> = None;
        for m in &self.catalog.models {
            for a in m.weights() {
                if artifact_holds(a, &name) {
                    if found.as_ref().is_some_and(|(id, _)| id != &m.id) {
                        // A declaration may disambiguate this in identity_for;
                        // guessing would attach another model's settings.
                        return None;
                    }
                    found.get_or_insert_with(|| (m.id.clone(), a.id.clone()));
                }
            }
        }
        found
    }

    /// The catalog identity of an endpoint, from what its config file DECLARES
    /// (`[catalog]`) reconciled against the weights path it actually serves.
    /// One rule, one place - the two surfaces that used to answer this
    /// separately (`/api/servers` and `heal_spec_identity`) disagreed, which is
    /// how the edit page ended up showing a model's name in the row and
    /// "select" in the dropdown beside it.
    ///
    /// `declared` is `(model id, artifact id)` straight off the file. The
    /// reconciliation, in full:
    ///
    /// - **They agree** -> the declaration names the model, the FILE names the
    ///   artifact. Point `model` at the same model's other quant and the block
    ///   does not need editing to stay honest.
    /// - **They name different models** -> the file wins, loudly. Somebody
    ///   repointed `model` and left the block behind; serving gemma while
    ///   claiming qwen is the one outcome worth a warning.
    /// - **The catalog does not recognize the file** -> the declaration stands.
    ///   This is the case `identify_weights` can never serve - a renamed file,
    ///   a copy, a path outside the models dir - and the whole reason the block
    ///   exists. An endpoint does not lose its identity because someone
    ///   reorganized their disk.
    /// - **No block** -> `identify_weights` alone, i.e. exactly the earlier
    ///   behaviour, so every config file already on disk keeps working and
    ///   nothing needs migrating.
    ///
    /// A declared id the catalog has never heard of is discarded rather than
    /// passed through: every consumer here assumes `model` is a catalog id or a
    /// path, and a third kind ("an id nobody can look up") would show the user
    /// a selection they cannot select.
    pub fn identity_for(
        &self,
        declared: Option<(&str, Option<&str>)>,
        weights: &Path,
    ) -> Option<(String, Option<String>)> {
        let by_file = self.identify_weights(weights);
        let declared = declared.filter(|(id, _)| self.catalog.models.iter().any(|m| &m.id == id));
        match (declared, by_file) {
            (Some((id, _)), Some((fid, fart))) if id == fid => Some((fid, Some(fart))),
            (Some((id, art)), Some((fid, fart))) => {
                tracing::warn!(
                    declared = %id, declared_artifact = ?art, found = %fid,
                    weights = %weights.display(),
                    "config file's [catalog] block does not match the weights it points at - trusting the file"
                );
                Some((fid, Some(fart)))
            }
            (Some((id, art)), None) => Some((id.to_string(), art.map(str::to_string))),
            (None, Some((fid, fart))) => Some((fid, Some(fart))),
            (None, None) => None,
        }
    }

    /// The catalog entry behind a runner's advertised model - matched by
    /// catalog id first, then by weights file name for path-shaped names.
    /// None when this catalog doesn't know the model (a hand-typed GGUF, a
    /// foreign runner).
    pub fn catalog_of(&self, name: &str) -> Option<&CatalogModel> {
        match self.catalog.models.iter().find(|m| m.id == name) {
            Some(m) => Some(m),
            None => {
                let (id, _) = self.identify_weights(Path::new(name))?;
                self.catalog.models.iter().find(|m| m.id == id)
            }
        }
    }

    /// The same backend-aware defaults feed native creation, web admission and
    /// saved runner configuration. Missing values are choices, not chat defaults.
    pub fn default_envelope(&self, name: &str, artifact: Option<&str>) -> (usize, usize) {
        let selected = self.catalog_of(name).and_then(|model| {
            let identified = self.identify_weights(Path::new(name));
            let id = artifact.or_else(|| identified.as_ref().map(|(_, id)| id.as_str()));
            match id {
                Some(id) => model.artifact(id),
                None => model.default_weights_for_backend(&self.backend, self.cc),
            }
        });
        selected
            .map(|a| a.runtime.default_envelope())
            .unwrap_or((4096, 32))
    }

    /// Human labels for a runner's advertised model: `(display, vendor)`.
    /// None when the catalog doesn't know it: callers fall back to the raw
    /// name rather than invent a pretty one.
    pub fn display_of(&self, name: &str) -> Option<(String, Option<String>)> {
        let m = self.catalog_of(name)?;
        Some((m.display.clone(), m.vendor.clone()))
    }

    /// What the catalog says this model can do ("chat", "vision",
    /// "transcription", ...). The only capability answer available for an
    /// endpoint that is not running: a live runner advertises its own, but a
    /// stopped one has nothing to ask, and the Studio still has to know
    /// whether starting it would get you a speech model.
    pub fn capability_of(&self, name: &str) -> Option<Vec<String>> {
        let m = self.catalog_of(name)?;
        if let Some((_, id)) = self.identify_weights(Path::new(name)) {
            return Some(m.artifact(&id)?.capabilities(m).to_vec());
        }
        Some(m.capability.clone())
    }

    /// Resolve a model id to a serving composition. `weights` selects the
    /// weights artifact (None = the default choice, preferring an INSTALLED
    /// one). `pull = false` - the deploy contract - never downloads: a
    /// missing selection is an honest error naming the fix; only installed
    /// companions join the composition. `pull = true` (the CLI convenience)
    /// fetches the selected weights + default companions, blocking until
    /// done. `Ok(None)` when `name` is not a manifest id, so the caller
    /// treats it as a filesystem path instead.
    /// `drafter` names a drafter artifact id when the endpoint pinned one
    /// (muse catalogues DFlash1 and DFlash2); None takes the catalog default.
    pub async fn resolve(
        &self,
        name: &str,
        weights: Option<&str>,
        pull: bool,
        drafter: Option<&str>,
    ) -> Result<Option<Resolved>, DlError> {
        let Some(model) = self.catalog.models.iter().find(|m| m.id == name).cloned() else {
            return Ok(None); // not a known id -> the caller loads it as a path
        };

        // elect the weights artifact
        let chosen = match weights {
            Some(id) => {
                let a = model
                    .artifact(id)
                    .ok_or_else(|| DlError::Http(format!("model {name} has no artifact {id:?}")))?;
                if a.kind != ArtifactKind::Weights {
                    return Err(DlError::Http(format!(
                        "artifact {id:?} of {name} is not a weights artifact ({:?})",
                        a.kind
                    )));
                }
                a.clone()
            }
            None => self
                .elect_weights(&model)
                .ok_or_else(|| {
                    DlError::Http(format!(
                        "model {name} has no compatible weights for backend {}",
                        self.backend
                    ))
                })?
                .clone(),
        };

        if !chosen.runtime.supports_backend(&self.backend) {
            return Err(DlError::Http(format!(
                "artifact {} of {name} requires backend {}; configured backend is {}",
                chosen.id,
                chosen.runtime.backends.join(" or "),
                self.backend
            )));
        }
        if let Some(id) = drafter
            && !chosen.runtime.allows_companion(id)
        {
            return Err(DlError::Http(format!(
                "artifact {} of {name} does not support companion {id}",
                chosen.id
            )));
        }

        // the pieces this composition wants: the chosen weights + the default
        // companions (pull mode fetches them; no-pull mode uses what's there)
        let mut wanted: Vec<CatalogArtifact> = vec![chosen.clone()];
        wanted.extend(
            model
                .artifacts
                .iter()
                .filter(|a| {
                    a.kind != ArtifactKind::Weights
                        && a.default
                        && chosen.runtime.allows_companion(&a.id)
                        && a.runtime.supports_backend(&self.backend)
                })
                .cloned(),
        );

        if pull {
            let missing: Vec<&CatalogFile> = wanted
                .iter()
                .flat_map(|a| a.files.iter())
                .filter(|f| {
                    std::fs::metadata(self.models_dir.join(&f.dest))
                        .map(|m| m.len())
                        .ok()
                        != Some(f.size)
                })
                .collect();
            // disk guard for the not-yet-present bytes (keep ~1 GiB headroom)
            let need: u64 = missing.iter().map(|f| f.size).sum();
            if let Some(free) = disk_free(&self.models_dir)
                && need > free.saturating_sub(1 << 30)
            {
                return Err(DlError::Disk {
                    need,
                    free,
                    dir: self.models_dir.display().to_string(),
                });
            }
            for f in missing {
                tracing::info!(model = %name, file = %f.dest, size = f.size, "pulling missing model file");
                download_file(
                    &self.client_for(&f.url),
                    &f.url,
                    &self.models_dir.join(&f.dest),
                    &f.sha256,
                    f.size,
                    Arc::new(AtomicU64::new(0)),
                    None,
                )
                .await?;
            }
        } else if !self.is_artifact_installed(&chosen) {
            return Err(DlError::Http(format!(
                "model {name} ({}) is not downloaded - get it on the Models page (or `paddock pull {name}`)",
                chosen.label
            )));
        }

        // assemble the composition from what is actually on disk
        let weights_path = chosen.entry_path(&self.models_dir).ok_or_else(|| {
            DlError::Http(format!("artifact {} of {name} has no files", chosen.id))
        })?;
        let installed_path = |kind: ArtifactKind| -> Option<PathBuf> {
            model
                .artifacts
                .iter()
                .filter(|a| {
                    a.kind == kind
                        && chosen.runtime.allows_companion(&a.id)
                        && a.runtime.supports_backend(&self.backend)
                })
                .find(|a| self.is_artifact_installed(a))
                .and_then(|a| a.files.first())
                .map(|f| self.models_dir.join(&f.dest))
        };
        // The mmproj companion, whichever SENSE it serves - see planned_paths
        // for why this is not a Vision-only lookup.
        // a split-tower model's `mmproj` is its picture tower; the audio
        // tower is `audio_mmproj`, chosen at render by its own switch
        let split = model.split_towers();
        let installed_mmproj = || -> Option<PathBuf> {
            model
                .artifacts
                .iter()
                .filter(|a| {
                    a.kind.is_mmproj()
                        && (!split || a.kind == ArtifactKind::Vision)
                        && chosen.runtime.allows_companion(&a.id)
                        && a.runtime.supports_backend(&self.backend)
                })
                .find(|a| self.is_artifact_installed(a))
                .and_then(|a| a.files.first())
                .map(|f| self.models_dir.join(&f.dest))
        };
        // Which drafter, when a model catalogues more than one (muse ships
        // DFlash1 and DFlash2). Three rungs, and every rung requires the bytes
        // to be on DISK: the id the endpoint ASKED for, else the catalog
        // default, else any installed sibling. The election used to be able to
        // return an artifact that was not installed (the pin arm skipped the
        // check), and the three consumers below each patched around that with
        // their own installed-filter and their own fallback - so in the
        // pin-and-default-both-missing corner they DISAGREED: `drafter_any`
        // wired the installed sibling while `drafter_pick` said nothing was
        // wired, silencing "which drafter did On get me" exactly where the
        // answer is least guessable.
        let ds = || {
            model.artifacts.iter().filter(|a| {
                a.kind == ArtifactKind::Drafter
                    && chosen.runtime.allows_companion(&a.id)
                    && a.runtime.supports_backend(&self.backend)
                    && a.fits_cc(self.cc)
            })
        };
        // The pin's rung is consent for that artifact and nothing else: a pin
        // whose bytes are missing must not silently elect a non-default
        // sibling. Default companions enable speculation automatically;
        // merely having downloaded a non-default draft does not opt into it.
        let pin_hit =
            drafter.and_then(|w| ds().find(|a| a.id == w && self.is_artifact_installed(a)));
        let default_hit = ds()
            .filter(|a| a.default)
            .find(|a| self.is_artifact_installed(a));
        let picked = pin_hit
            .or(default_hit)
            .or_else(|| ds().find(|a| self.is_artifact_installed(a)));
        let picked_path = |a: &CatalogArtifact| -> Option<PathBuf> {
            a.files.first().map(|f| self.models_dir.join(&f.dest))
        };
        // the snapshot is a DIRECTORY (config.json + shards) - resolve to it
        let fp8_snapshot = installed_path(ArtifactKind::Fp8Snapshot)
            .and_then(|p| p.parent().map(Path::to_path_buf));
        Ok(Some(Resolved {
            weights: weights_path,
            mmproj: installed_mmproj(),
            text_encoder: installed_path(ArtifactKind::TextEncoder),
            vae: installed_path(ArtifactKind::Vae),
            // What wires without being asked: the pin (an explicit choice is
            // the same consent that marking it default expresses), else the
            // installed default - never a non-default sibling.
            mtp: pin_hit.or(default_hit).and_then(picked_path),
            // What an explicit "on" wires: the same election with the
            // installed-sibling rung included.
            drafter_any: picked.and_then(picked_path),
            // One elected artifact feeds all three fields, so what is WIRED
            // and what is NAMED cannot part ways again.
            drafter_pick: picked.map(|a| (a.id.clone(), a.label.clone())),
            drafter_declared: ds().next().is_some(),
            speculative: chosen
                .capabilities(&model)
                .iter()
                .any(|c| c == "speculative"),
            fp8_snapshot,
        }))
    }

    pub fn job(&self, id: &str) -> Option<Arc<PullJob>> {
        self.jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    /// Every pull job this manager has run since boot, oldest first - the
    /// downloads surface (the Studio's header indicator + list).
    pub fn jobs(&self) -> Vec<Arc<PullJob>> {
        let mut v: Vec<Arc<PullJob>> = self
            .jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect();
        v.sort_by_key(|j| j.created_ms);
        v
    }

    /// Ask a running job to stop. Cooperative: workers notice between range
    /// segments and the job settles to Cancelled shortly after. False when
    /// the job is unknown or already finished.
    pub fn cancel_pull(&self, id: &str) -> bool {
        let Some(job) = self.job(id) else {
            return false;
        };
        let running = matches!(
            &*job
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            PullStatus::Running
        );
        if running {
            job.cancel.store(true, Ordering::Relaxed);
        }
        running
    }

    /// Resume a cancelled/failed pull: a FRESH job over the same selection.
    /// Already-complete files are skipped and partial files continue from
    /// their segment sidecars, so only the missing bytes move. The old job's
    /// queued follow-up (start-after-download) carries over; the caller
    /// re-arms its watcher.
    pub fn resume_pull(&self, id: &str) -> Result<String, DlError> {
        let old = self
            .job(id)
            .ok_or_else(|| DlError::Http(format!("unknown pull job {id}")))?;
        if matches!(
            &*old
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            PullStatus::Running | PullStatus::Done
        ) {
            return Err(DlError::Http("job is still running or already done".into()));
        }
        if recovery::digest(&self.pull_files(&old.model_id, old.artifacts.as_deref())?)
            != old.selection_digest
        {
            return Err(DlError::Http("The catalog changed since this download. Review the model's current download options before continuing.".into()));
        }
        let new_id = self.start_pull(&old.model_id, old.artifacts.as_deref())?;
        if let Some(new) = self.job(&new_id) {
            *new.follow
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = old
                .follow
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
        }
        Ok(new_id)
    }

    /// Start pulling pieces of `model_id` into the models dir. `artifacts` =
    /// specific artifact ids, or None for the default bundle (default weights
    /// + default companions). Returns the job id; the downloads run on a
    ///   background task, tracked in `jobs` so the Studio can poll
    ///   `job(id).snapshot()`.
    pub fn start_pull(
        &self,
        model_id: &str,
        artifacts: Option<&[String]>,
    ) -> Result<String, DlError> {
        let model = self
            .catalog
            .models
            .iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| DlError::Http(format!("unknown model {model_id}")))?
            .clone();
        let selected: Vec<&CatalogArtifact> = match artifacts {
            Some(ids) => {
                let mut out = Vec::with_capacity(ids.len());
                for id in ids {
                    out.push(model.artifact(id).ok_or_else(|| {
                        DlError::Http(format!("model {model_id} has no artifact {id:?}"))
                    })?);
                }
                out
            }
            None => model.default_bundle_for_backend(&self.backend, self.cc),
        };
        if selected.is_empty() {
            return Err(DlError::Http(format!(
                "model {model_id} has no compatible artifacts for backend {}",
                self.backend
            )));
        }
        for a in &selected {
            if !a.runtime.supports_backend(&self.backend) {
                return Err(DlError::Http(format!(
                    "artifact {} of {model_id} requires backend {}; configured backend is {}",
                    a.id,
                    a.runtime.backends.join(" or "),
                    self.backend
                )));
            }
        }
        let files = self.pull_files(model_id, artifacts)?;
        let total: u64 = files.iter().map(|f| f.size).sum();

        // Admission and insertion share one lock. Never let two selections
        // write the same part file or spend the same disk-space reservation.
        let mut jobs = self
            .jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let active: Vec<_> = jobs
            .values()
            .filter(|j| {
                matches!(
                    *j.status
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                    PullStatus::Running
                )
            })
            .collect();
        if active.len() >= 2 {
            return Err(DlError::Http(
                "Two downloads are already active. Pause one or wait for it to finish.".into(),
            ));
        }
        if active.iter().any(|j| {
            j.files
                .iter()
                .any(|prior| files.iter().any(|f| f.dest == prior.dest))
        }) {
            return Err(DlError::Http(
                "These files are already downloading. Open Downloads to view their progress."
                    .into(),
            ));
        }
        if jobs.len() >= 128 {
            let oldest = jobs
                .values()
                .filter(|j| {
                    matches!(
                        *j.status
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner),
                        PullStatus::Done
                    )
                })
                .min_by_key(|j| j.created_ms)
                .map(|j| j.id.clone());
            if let Some(id) = oldest {
                jobs.remove(&id);
            } else {
                return Err(DlError::Http("Download history is full of unfinished work. Resume the existing downloads first.".into()));
            }
        }

        // disk guard: refuse the pull up front if the not-yet-present bytes
        // wouldn't fit (keep ~1 GiB headroom), so the UI can warn instead of
        // filling the drive mid-download.
        let need: u64 = files
            .iter()
            .map(|f| recovery::additional_bytes(&self.models_dir, f))
            .sum();
        std::fs::create_dir_all(&self.models_dir)?;
        let reserved: u64 = jobs
            .values()
            .filter(|j| {
                matches!(
                    *j.status
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                    PullStatus::Running
                )
            })
            .flat_map(|j| &j.files)
            .map(|f| recovery::additional_bytes(&self.models_dir, f))
            .sum();
        if let Some(free) = disk_free(&self.models_dir)
            && need.saturating_add(reserved) > free.saturating_sub(1 << 30)
        {
            return Err(DlError::Disk {
                need,
                free,
                dir: self.models_dir.display().to_string(),
            });
        }

        let job = Arc::new(PullJob {
            id: uuid::Uuid::new_v4().simple().to_string(),
            model_id: model_id.to_owned(),
            display: model.display.clone(),
            artifacts: Some(selected.iter().map(|a| a.id.clone()).collect()),
            downloaded: Arc::new(AtomicU64::new(0)),
            total,
            created_ms: unix_ms(),
            status: std::sync::Mutex::new(PullStatus::Running),
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            follow: std::sync::Mutex::new(None),
            follow_state: std::sync::Mutex::new(None),
            files: files.clone(),
            selection_digest: recovery::digest(&files),
            phase: std::sync::Mutex::new("Preparing download".into()),
            stage: std::sync::Mutex::new("preparing"),
        });
        if let Some(store) = &self.store {
            store
                .save_download(&job.record())
                .map_err(|e| DlError::Http(e.to_string()))?;
        }
        jobs.insert(job.id.clone(), job.clone());
        drop(jobs);

        // one client a file: a Hugging Face file's carries the user's token
        let clients: Vec<reqwest::Client> = files.iter().map(|f| self.client_for(&f.url)).collect();
        let models_dir = self.models_dir.clone();
        let job2 = job.clone();
        let store = self.store.clone();
        tokio::spawn(async move {
            let mut outcome: Result<(), DlError> = Ok(());
            for (f, client) in files.iter().zip(&clients) {
                if job2.cancel.load(Ordering::Relaxed) {
                    outcome = Err(DlError::Cancelled);
                    break;
                }
                let dest = models_dir.join(&f.dest);
                *job2
                    .phase
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = f.dest.clone();
                // Size alone is not an integrity check. Imported/pre-existing
                // bytes are read off-thread and must match this exact artifact.
                if std::fs::metadata(&dest).map(|m| m.len()).ok() == Some(f.size) {
                    *job2
                        .stage
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = "verifying";
                    match sha256_file_cancel(dest.clone(), Some(job2.cancel.clone())).await {
                        Ok(sha) if sha == f.sha256.to_lowercase() => {
                            job2.downloaded.fetch_add(f.size, Ordering::Relaxed);
                            continue;
                        }
                        Ok(sha) => {
                            outcome = Err(DlError::Checksum {
                                name: f.dest.clone(),
                                expected: f.sha256.clone(),
                                got: sha,
                            });
                            break;
                        }
                        Err(e) => {
                            outcome = Err(e);
                            break;
                        }
                    }
                }
                if let Err(e) = download_file_staged(
                    client,
                    &f.url,
                    &dest,
                    &f.sha256,
                    f.size,
                    job2.downloaded.clone(),
                    Some(job2.cancel.clone()),
                    Some(&job2.stage),
                )
                .await
                {
                    outcome = Err(e);
                    break;
                }
            }
            *job2
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = match outcome {
                Ok(()) => PullStatus::Done,
                Err(DlError::Cancelled) => PullStatus::Cancelled,
                Err(e) => PullStatus::Error {
                    message: e.to_string(),
                },
            };
            if let Some(store) = store
                && let Err(e) = store.save_download(&job2.record())
            {
                *job2
                    .status
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = PullStatus::Error {
                    message: format!(
                        "Files were processed, but saving download state failed: {e}. Resume to verify."
                    ),
                };
            }
        });
        Ok(job.id.clone())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod backend_tests;
#[cfg(test)]
mod contract_tests;
#[cfg(test)]
mod embeddinggemma2_tests;
#[cfg(test)]
mod flash_next_mlx_tests;
#[cfg(test)]
mod flash_next_nvfp4_tests;
#[cfg(test)]
mod kolibri_tests;
#[cfg(test)]
mod paddleocr_layout_tests;
