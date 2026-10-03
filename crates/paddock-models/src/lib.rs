//! The model store.
//!
//! Principles: models are plain GGUF
//! files in user-visible paths. We scan, we don't copy; import means "point at it".
//! Content-addressing is for downloads/transfer only and never the on-disk layout.
//!
//! The scan is recursive (bounded depth) so repo-shaped layouts
//! (`models/<repo>/<file>.gguf`, the unsloth *-MTP-GGUF convention) list
//! without flattening; mmproj companions and hidden dirs are skipped.

pub mod bonsai;
pub mod clef;
pub mod diarization;
pub mod dinov3;
pub mod ggml_type;
pub mod gguf;
pub mod gpu_support;
pub mod granite;
pub mod hadamard;
pub mod hardening;
pub mod kumo;
pub mod kv_tier_geom;
pub mod laya;
pub mod mapped;
pub mod meminfo;
pub mod mlx;
pub mod modelopt;
pub mod nemotron;
pub mod probe;
pub mod qwen4exp;
pub mod safetensors;
pub mod sampling;
pub mod splash;
pub mod split;
#[cfg(test)]
pub(crate) mod testutil;

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use split::parse_split_name;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("model directory {0} is not readable: {1}")]
    Unreadable(PathBuf, std::io::Error),
}

/// A model the store knows about. Grows real metadata in P1.
#[derive(Debug, Clone)]
pub struct StoredModel {
    /// Client-facing id: file stem, e.g. "gpt-oss-20b-MXFP4" for gpt-oss-20b-MXFP4.gguf.
    /// Split families drop the shard suffix: "m" for m-00001-of-00003.gguf.
    pub id: String,
    /// The path to open - the first shard for split families.
    pub path: PathBuf,
    /// File mtime in unix seconds; serves as the "created" field in /v1/models.
    pub modified_unix: u64,
    /// Total on-disk bytes - summed across all present shards for families.
    pub size_bytes: u64,
}

/// Scans one or more root directories for GGUF files.
#[derive(Debug, Clone)]
pub struct ModelStore {
    roots: Vec<PathBuf>,
}

impl ModelStore {
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self { roots }
    }

    /// How deep below each root the scan descends. Repo-shaped layouts
    /// (`models/Qwen3.5-9B-MTP-GGUF/...`) are one level; the bound gives slack
    /// for a nesting convention or two while making link cycles harmless.
    const MAX_DEPTH: usize = 3;

    /// List every model under the roots, recursively (bounded depth). A split
    /// family (`-00001-of-00003.gguf` siblings) is one model: listed once, id
    /// without the shard suffix, size summed over the shards present.
    ///
    /// Ids are file stems. When two files share a stem (the same quant name
    /// in two repo dirs), every colliding entry is re-identified by its
    /// root-relative path (`Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q8_0`) - an
    /// ambiguous id would make "which file did I just serve?" a guess, and
    /// honest naming is a product principle.
    ///
    /// `mmproj-*.gguf` vision companions are not listed: they are not
    /// servable models (loading one as a model is a guaranteed error) and
    /// travel with their model via the registry/`--mmproj` instead.
    pub fn list(&self) -> Result<Vec<StoredModel>, StoreError> {
        let mut out: Vec<(PathBuf, StoredModel)> = Vec::new();
        for root in &self.roots {
            Self::walk(root, root, 0, &mut out);
        }

        // Collision pass: any id claimed by more than one file gets the
        // root-relative path id instead (all claimants, not just later ones -
        // keeping one bare winner would still be a guess).
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (_, m) in &out {
            *counts.entry(m.id.as_str()).or_default() += 1;
        }
        let colliding: std::collections::HashSet<String> = counts
            .iter()
            .filter(|&(_, &n)| n > 1)
            .map(|(id, _)| (*id).to_owned())
            .collect();
        let mut models: Vec<StoredModel> = out
            .into_iter()
            .map(|(root, mut m)| {
                if colliding.contains(&m.id)
                    && let Ok(rel) = m.path.strip_prefix(&root)
                    && let Some(parent) = rel.parent()
                    && !parent.as_os_str().is_empty()
                {
                    let dir = parent.to_string_lossy().replace('\\', "/");
                    m.id = format!("{dir}/{}", m.id);
                }
                m
            })
            .collect();
        // Residual collisions (same stem at two roots' top level) can't be
        // path-qualified relative to their roots - say so instead of guessing.
        let mut seen = std::collections::HashSet::new();
        for m in &models {
            if !seen.insert(m.id.clone()) {
                tracing::warn!(
                    id = %m.id,
                    path = %m.path.display(),
                    "duplicate model id across roots - resolution by id picks the first match; rename one file to disambiguate"
                );
            }
        }
        models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(models)
    }

    fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(PathBuf, StoredModel)>) {
        // a missing root isn't an error at list time - the dir may simply not
        // be created yet; we log and keep going so /v1/models never 500s on it
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(dir = %dir.display(), %err, "model dir not readable, skipping");
                return;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir {
                let hidden = path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'));
                if !hidden && depth < Self::MAX_DEPTH {
                    Self::walk(root, &path, depth + 1, out);
                }
                continue;
            }
            if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                && !path.file_name().is_some_and(|n| {
                    n.to_string_lossy()
                        .to_ascii_lowercase()
                        .starts_with("mmproj")
                })
                && let Some(model) = Self::probe_entry(&path)
            {
                out.push((root.to_path_buf(), model));
            }
        }
    }

    fn probe_entry(path: &Path) -> Option<StoredModel> {
        let meta = std::fs::metadata(path).ok()?;
        let modified_unix = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // split families: shard 1 represents the family, later shards hide
        if let Some(name) = parse_split_name(path) {
            if name.no_1based != 1 {
                return None;
            }
            let id = Path::new(&name.prefix)
                .file_name()?
                .to_string_lossy()
                .into_owned();
            // sum what's on disk; an incomplete family still lists (the load
            // path is where a missing shard becomes a loud error, per the
            // no-silent-failures principle the listing itself stays honest
            // about bytes actually present)
            let size_bytes = (1..=name.count)
                .filter_map(|i| std::fs::metadata(name.sibling(i)).ok())
                .map(|m| m.len())
                .sum();
            return Some(StoredModel {
                id,
                path: path.to_path_buf(),
                modified_unix,
                size_bytes,
            });
        }

        Some(StoredModel {
            id: path.file_stem()?.to_string_lossy().into_owned(),
            path: path.to_path_buf(),
            modified_unix,
            size_bytes: meta.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_gguf_files_and_ignores_others() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("tiny.gguf"), b"not-really-gguf").expect("write");
        std::fs::write(dir.path().join("notes.txt"), b"ignore me").expect("write");

        let store = ModelStore::new(vec![dir.path().to_path_buf()]);
        let models = store.list().expect("list");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "tiny");
    }

    #[test]
    fn missing_root_is_not_an_error() {
        let store = ModelStore::new(vec![PathBuf::from("Z:/does/not/exist")]);
        assert!(store.list().expect("list").is_empty());
    }

    #[test]
    fn scans_repo_subdirs_and_skips_mmproj_and_hidden() {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir.path().join("Qwen3.5-9B-MTP-GGUF");
        std::fs::create_dir_all(&repo).expect("mkdir");
        std::fs::write(repo.join("Qwen3.5-9B-Q8_0.gguf"), b"g").expect("write");
        std::fs::write(repo.join("mmproj-BF16.gguf"), b"g").expect("write");
        let hidden = dir.path().join(".cache");
        std::fs::create_dir_all(&hidden).expect("mkdir");
        std::fs::write(hidden.join("stale.gguf"), b"g").expect("write");

        let store = ModelStore::new(vec![dir.path().to_path_buf()]);
        let models = store.list().expect("list");
        assert_eq!(models.len(), 1, "{models:?}");
        assert_eq!(models[0].id, "Qwen3.5-9B-Q8_0");
        assert!(
            models[0]
                .path
                .ends_with(Path::new("Qwen3.5-9B-MTP-GGUF").join("Qwen3.5-9B-Q8_0.gguf"))
        );
    }

    #[test]
    fn colliding_stems_get_path_qualified_ids() {
        let dir = tempfile::tempdir().expect("tempdir");
        for repo in ["repo-a", "repo-b"] {
            let d = dir.path().join(repo);
            std::fs::create_dir_all(&d).expect("mkdir");
            std::fs::write(d.join("model-Q8_0.gguf"), b"g").expect("write");
        }
        std::fs::write(dir.path().join("unique.gguf"), b"g").expect("write");

        let store = ModelStore::new(vec![dir.path().to_path_buf()]);
        let models = store.list().expect("list");
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["repo-a/model-Q8_0", "repo-b/model-Q8_0", "unique"]
        );
    }

    #[test]
    fn split_family_lists_once_with_summed_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("big-00001-of-00003.gguf"), vec![0u8; 10]).expect("write");
        std::fs::write(dir.path().join("big-00002-of-00003.gguf"), vec![0u8; 20]).expect("write");
        std::fs::write(dir.path().join("big-00003-of-00003.gguf"), vec![0u8; 30]).expect("write");
        std::fs::write(dir.path().join("small.gguf"), vec![0u8; 5]).expect("write");

        let store = ModelStore::new(vec![dir.path().to_path_buf()]);
        let models = store.list().expect("list");
        assert_eq!(models.len(), 2);
        let big = models
            .iter()
            .find(|m| m.id == "big")
            .expect("family listed");
        assert_eq!(big.size_bytes, 60);
        assert!(
            big.path
                .to_string_lossy()
                .ends_with("big-00001-of-00003.gguf")
        );
    }
}
