//! Companion files - the vision/audio projector (mmproj) and the MTP drafter
//! a model's weights come with - found beside the weights or named in the
//! config.
//!
//! A catalog pull lays a model out one per folder, so "whatever mmproj sits
//! next to it" used to be good enough, and discovery took the first name
//! match in directory order. A folder several models share breaks that: the
//! scan picked another model's projector and the load died on a tensor the
//! file does not have (`v.patch_embd.weight.1 not present`), naming nothing
//! the operator could act on (issue #33). So every candidate is now checked
//! against the weights' own header before it is taken, the same check
//! llama.cpp's mtmd makes when it refuses a projector whose output width is
//! not the text model's embedding width.
//!
//! What the headers say, measured across every model folder on the dev box
//! (19 pairings, all pass):
//! - a projector is `general.architecture = "clip"`, and its
//!   `clip.{vision,audio}.projection_dim` is the text model's
//!   `{arch}.embedding_length` - it has to be, the projector's output rows go
//!   straight into the text model's embedding stream;
//! - a drafter shares the text model's vocabulary (its `token_embd` rows),
//!   and when it has its own narrower width it names the target's in
//!   `{arch}.embedding_length_out` (gemma4's assistant: 1024 inside, 5376 out).
//!
//! A key that is absent proves nothing either way, so it never refuses: a
//! file the check cannot read is taken, as before, behind a verified one.

use std::path::{Path, PathBuf};

use paddock_models::gguf::GgufFile;

use crate::config::Config;

/// The header facts the checks compare.
#[derive(Debug, Default, Clone, PartialEq)]
struct Shape {
    arch: Option<String>,
    /// `{arch}.embedding_length`
    n_embd: Option<u64>,
    /// `{arch}.vocab_size`, else the rows of `token_embd.weight`
    vocab: Option<u64>,
    /// a drafter's `{arch}.embedding_length_out` - the target width it feeds
    width_out: Option<u64>,
    /// a projector's `clip.vision.projection_dim` / `clip.audio.projection_dim`
    projection: Vec<u64>,
    /// a projector's towers: (picture, audio) - which `*.projector_type` it names
    towers: (bool, bool),
}

impl Shape {
    fn of(f: &GgufFile) -> Self {
        let u = |k: &str| f.metadata.get(k).and_then(|v| v.as_u64());
        let arch = f.architecture().map(str::to_owned);
        let key = |suffix: &str| arch.as_deref().and_then(|a| u(&format!("{a}.{suffix}")));
        // GGUF lays a 2-D weight out [in, out]: token_embd is [n_embd, vocab]
        let embd_rows = f
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.dims.get(1).copied());
        Self {
            n_embd: key("embedding_length"),
            vocab: key("vocab_size").or(embd_rows),
            width_out: key("embedding_length_out"),
            projection: ["clip.vision.projection_dim", "clip.audio.projection_dim"]
                .iter()
                .filter_map(|k| u(k))
                .collect(),
            towers: (
                f.metadata.contains_key("clip.vision.projector_type"),
                f.metadata.contains_key("clip.audio.projector_type"),
            ),
            arch,
        }
    }

    fn read(path: &Path) -> Option<Self> {
        paddock_models::probe::read_header(path)
            .ok()
            .map(|f| Self::of(&f))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Verdict {
    /// every key the check needed was there and agreed
    Match,
    /// nothing to compare - taken, but behind a verified candidate
    Unknown,
    /// the file belongs to another model; the reason names the numbers
    Mismatch(String),
}

impl Verdict {
    fn rank(&self) -> Option<u8> {
        match self {
            Verdict::Match => Some(0),
            Verdict::Unknown => Some(1),
            Verdict::Mismatch(_) => None,
        }
    }
}

fn mmproj_verdict(model: &Shape, cand: &Shape) -> Verdict {
    if let Some(a) = cand.arch.as_deref()
        && a != "clip"
    {
        return Verdict::Mismatch(format!("it is a `{a}` model, not a projector"));
    }
    let Some(n_embd) = model.n_embd else {
        return Verdict::Unknown;
    };
    if cand.projection.is_empty() {
        return Verdict::Unknown;
    }
    match cand.projection.iter().find(|&&p| p != n_embd) {
        Some(p) => Verdict::Mismatch(format!(
            "it projects to width {p}, and the model's embedding width is {n_embd}"
        )),
        None => Verdict::Match,
    }
}

fn mtp_verdict(model: &Shape, cand: &Shape) -> Verdict {
    let mut checked = false;
    if let (Some(v), Some(want)) = (cand.vocab, model.vocab) {
        if v != want {
            return Verdict::Mismatch(format!(
                "its vocabulary has {v} tokens, and the model's has {want}"
            ));
        }
        checked = true;
    }
    // The drafter's own `embedding_length` is its INNER width when it has
    // one (gemma4's assistant runs at 1024), so it only says something about
    // the target when the drafter is the target's own architecture.
    let same_arch = cand.arch.is_some() && cand.arch == model.arch;
    let target = cand
        .width_out
        .or(if same_arch { cand.n_embd } else { None });
    if let (Some(w), Some(want)) = (target, model.n_embd) {
        if w != want {
            return Verdict::Mismatch(format!(
                "it drafts for embedding width {w}, and the model's is {want}"
            ));
        }
        checked = true;
    }
    if checked {
        Verdict::Match
    } else {
        Verdict::Unknown
    }
}

/// Characters the two lowercased file names share from the start - the
/// tie-break between two verified candidates (`gemma-4-31B-it-mmproj` beats
/// `mmproj-model` beside `gemma-4-31B-it-Q8_0`).
fn shared_prefix(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// The best candidate: verified before unknown, then the longest name shared
/// with the weights, then name order - never directory order, which is what
/// made a shared folder's pick depend on the filesystem.
fn pick(
    weights: &Path,
    cands: Vec<PathBuf>,
    verdict: impl Fn(&Path) -> Verdict,
    what: &str,
) -> Option<PathBuf> {
    let w = file_name(weights);
    let mut ok: Vec<(u8, usize, String, PathBuf)> = Vec::new();
    for c in cands {
        match verdict(&c) {
            Verdict::Mismatch(why) => tracing::warn!(
                file = %c.display(),
                "{what} beside the weights skipped: {why} - it belongs to another model"
            ),
            v => {
                let name = file_name(&c);
                let rank = v.rank().unwrap_or(u8::MAX);
                ok.push((rank, shared_prefix(&name, &w), name, c));
            }
        }
    }
    ok.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    ok.into_iter().next().map(|(.., p)| p)
}

/// An mmproj/MTP path the config names is the operator's choice, so a
/// mismatch refuses with the reason instead of being swapped for another
/// file - and instead of the loader's missing-tensor error further on.
fn check_named(path: &Path, what: &str, key: &str, verdict: Verdict) -> Result<(), String> {
    match verdict {
        Verdict::Mismatch(why) => Err(format!(
            "{key} = {} does not belong to this model: {why}. Point {key} at the {what} \
             that ships with the weights, or leave it unset to use the one beside them",
            path.display()
        )),
        _ => Ok(()),
    }
}

fn is_gguf(p: &Path) -> bool {
    p.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// Check the configured companions against the weights, then fill the unset
/// ones from the weights' own folder - the layout catalog pulls produce.
/// A PP-DocLayoutV3 checkpoint directory for PaddleOCR-VL weights: one named
/// `PP-DocLayoutV3*` inside the weights' folder, else beside it (the
/// catalog's one-folder-per-repo layout), its `config.json` naming the
/// architecture.
fn layout_dir(weights: &Path) -> Option<PathBuf> {
    let folder = weights.parent()?;
    let is_layout = |d: &Path| {
        d.join("model.safetensors").is_file()
            && std::fs::read_to_string(d.join("config.json"))
                .is_ok_and(|c| c.contains("\"pp_doclayout_v3\""))
    };
    [Some(folder), folder.parent()]
        .into_iter()
        .flatten()
        .find_map(|root| {
            let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name().is_some_and(|n| {
                        n.to_string_lossy()
                            .to_lowercase()
                            .starts_with("pp-doclayoutv3")
                    }) && is_layout(p)
                })
                .collect();
            dirs.sort();
            dirs.into_iter().next()
        })
}

pub(crate) fn resolve(cfg: &mut Config, weights: &Path) -> Result<(), String> {
    // `vision = false` / `--no-mmproj`: the tower is switched OFF. Leaving the
    // mmproj line out used to be the only off switch, and discovery below
    // loaded the tower beside the weights anyway - the manager's Vision switch
    // wrote exactly such a file, and the endpoint kept serving images, holding
    // memory its admission never counted. Naming a tower as well is a
    // contradiction; refusing it beats guessing which one was meant.
    if cfg.vision == Some(false)
        && let Some(p) = &cfg.mmproj
    {
        return Err(format!(
            "`vision = false` switches the vision tower off, but `mmproj` names one ({}) - \
             remove one of the two",
            p.display()
        ));
    }
    // A checkpoint DIRECTORY (safetensors-primary lane) carries everything
    // inside itself; scanning its PARENT would treat unrelated sibling
    // models' companions as this model's.
    if cfg.audio == Some(false)
        && let Some(p) = &cfg.audio_mmproj
    {
        return Err(format!(
            "`audio = false` switches the audio tower off, but `audio_mmproj` names one ({}) - \
             remove one of the two",
            p.display()
        ));
    }
    if weights.is_dir() {
        return Ok(());
    }
    let model = Shape::read(weights).unwrap_or_default();
    // EmbeddingGemma 2's catalog layout cuts its projector in two, a picture
    // tower and an audio tower, each its own download: both are projectors
    // of the right width, so the generic pick could seat the audio file in
    // `mmproj`. Its discovery sorts the candidates by the tower they carry.
    let split_towers = model.arch.as_deref() == Some("gemma-embedding2");
    // the layout lane is CUDA's; elsewhere the page is read whole
    if model.arch.as_deref() == Some("paddleocr")
        && cfg.device == "cuda"
        && cfg.layout.is_none()
        && let Some(d) = layout_dir(weights)
    {
        tracing::info!(layout = %d.display(), "PaddleOCR-VL layout companion discovered");
        cfg.layout = Some(d);
        cfg.layout_discovered = true;
    }
    for (slot, what, key, check) in [
        (
            &cfg.mmproj,
            "projector",
            "mmproj",
            mmproj_verdict as fn(&Shape, &Shape) -> Verdict,
        ),
        (
            &cfg.audio_mmproj,
            "audio projector",
            "audio_mmproj",
            mmproj_verdict,
        ),
        (&cfg.mtp, "drafter", "mtp", mtp_verdict),
    ] {
        if let Some(p) = slot
            && is_gguf(p)
            && let Some(c) = Shape::read(p)
        {
            check_named(p, what, key, check(&model, &c))?;
        }
    }
    let audio_settled = !split_towers || cfg.audio_mmproj.is_some() || cfg.audio == Some(false);
    if cfg.mmproj.is_some() && cfg.mtp.is_some() && audio_settled {
        return Ok(());
    }
    let Some(dir) = weights.parent() else {
        return Ok(());
    };
    let weights_name = weights.file_name().map(|n| n.to_os_string());
    let (mut mmproj, mut mtp) = (Vec::new(), Vec::new());
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            // never resolve the weights file itself as its own companion
            // (e.g. a main GGUF whose NAME contains "mmproj")
            if Some(e.file_name()) == weights_name {
                continue;
            }
            let fname = e.file_name().to_string_lossy().to_lowercase();
            if !fname.ends_with(".gguf") {
                continue;
            }
            if fname.contains("mmproj") {
                mmproj.push(e.path());
            } else if fname.starts_with("mtp")
                || fname.contains("-mtp.")
                || fname.starts_with("dflash")
                || fname.contains("-dflash")
            {
                // a block drafter (DFlash/DFlash2 GGUF) rides the same `mtp`
                // seat: qwen3.8's elected default is DFlash2 beside the
                // in-file MTP head (the catalog's `drafter2`, default on),
                // and a bare runner used to miss it - only `mtp*` names were
                // looked for, so it served the MTP chain alone
                mtp.push(e.path());
            }
        }
    }
    let read_or_unknown = |p: &Path, f: fn(&Shape, &Shape) -> Verdict| match Shape::read(p) {
        Some(c) => f(&model, &c),
        None => Verdict::Unknown,
    };
    let towers = |p: &Path| Shape::read(p).map(|s| s.towers);
    // split layout: pictures seat in `mmproj`, an audio-only file in
    // `audio_mmproj`
    let audio_only: Vec<PathBuf> = if split_towers {
        let (audio, rest): (Vec<_>, Vec<_>) = mmproj
            .into_iter()
            .partition(|p| towers(p) == Some((false, true)));
        mmproj = rest;
        audio
    } else {
        Vec::new()
    };
    if cfg.mmproj.is_none()
        && let Some(p) = pick(
            weights,
            mmproj,
            |p| read_or_unknown(p, mmproj_verdict),
            "mmproj",
        )
    {
        if cfg.vision == Some(false) {
            tracing::info!(
                mmproj = %p.display(),
                "vision is off (vision = false): the tower beside the weights is not loaded"
            );
        } else {
            tracing::info!(mmproj = %p.display(), "companion mmproj discovered beside the weights");
            cfg.mmproj = Some(p);
        }
    }
    // an audio tower of its own, unless the picture file already carries one
    // (the upstream projector holds both)
    let audio_inside = cfg.mmproj.as_deref().and_then(towers).is_some_and(|t| t.1);
    if !audio_settled
        && !audio_inside
        && let Some(p) = pick(
            weights,
            audio_only,
            |p| read_or_unknown(p, mmproj_verdict),
            "audio mmproj",
        )
    {
        tracing::info!(audio_mmproj = %p.display(), "companion audio tower discovered beside the weights");
        cfg.audio_mmproj = Some(p);
    }
    if cfg.mtp.is_none()
        && let Some(p) = pick(
            weights,
            mtp,
            |p| read_or_unknown(p, mtp_verdict),
            "MTP drafter",
        )
    {
        tracing::info!(mtp = %p.display(), "companion drafter discovered beside the weights");
        cfg.mtp = Some(p);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lm(arch: &str, n_embd: u64, vocab: u64) -> Shape {
        Shape {
            arch: Some(arch.into()),
            n_embd: Some(n_embd),
            vocab: Some(vocab),
            ..Shape::default()
        }
    }

    fn projector(dims: &[u64]) -> Shape {
        Shape {
            arch: Some("clip".into()),
            projection: dims.to_vec(),
            ..Shape::default()
        }
    }

    /// Issue #33's case: a Qwen3.8-27B (5120 wide) sharing a folder with the
    /// 9B's projector (4096).
    #[test]
    fn another_models_projector_is_a_mismatch() {
        let m = lm("qwen35", 5120, 248320);
        assert_eq!(mmproj_verdict(&m, &projector(&[5120])), Verdict::Match);
        let Verdict::Mismatch(why) = mmproj_verdict(&m, &projector(&[4096])) else {
            panic!("a 4096 projector must not pair with a 5120 model");
        };
        assert!(why.contains("4096") && why.contains("5120"), "{why}");
        // an omni projector carries both encoders; both have to agree
        assert!(matches!(
            mmproj_verdict(&m, &projector(&[5120, 2048])),
            Verdict::Mismatch(_)
        ));
        // a text model named *mmproj* is not a projector at all
        assert!(matches!(
            mmproj_verdict(&m, &lm("qwen35", 5120, 248320)),
            Verdict::Mismatch(_)
        ));
    }

    #[test]
    fn missing_keys_prove_nothing() {
        let m = lm("qwen35", 5120, 248320);
        assert_eq!(mmproj_verdict(&m, &projector(&[])), Verdict::Unknown);
        assert_eq!(
            mmproj_verdict(&Shape::default(), &projector(&[4096])),
            Verdict::Unknown
        );
        assert_eq!(mtp_verdict(&m, &Shape::default()), Verdict::Unknown);
    }

    /// gemma4's assistant is 1024 wide inside and names its target's 5376 in
    /// `embedding_length_out`; its inner width must not read as a mismatch.
    #[test]
    fn a_drafter_is_checked_on_its_target_width_and_vocab() {
        let m = lm("gemma4", 5376, 262144);
        let assistant = Shape {
            arch: Some("gemma4-assistant".into()),
            n_embd: Some(1024),
            vocab: Some(262144),
            width_out: Some(5376),
            ..Shape::default()
        };
        assert_eq!(mtp_verdict(&m, &assistant), Verdict::Match);
        // the 26B's assistant beside the 31B: same vocab, other target width
        let other = Shape {
            width_out: Some(2816),
            ..assistant.clone()
        };
        assert!(matches!(mtp_verdict(&m, &other), Verdict::Mismatch(_)));
        // another family's drafter: the vocab gives it away
        let foreign = Shape {
            vocab: Some(248320),
            ..assistant
        };
        assert!(matches!(mtp_verdict(&m, &foreign), Verdict::Mismatch(_)));
        // a sideloaded nextn file of the model's own arch has no _out key
        let nextn = lm("qwen35", 5120, 248320);
        assert_eq!(
            mtp_verdict(&lm("qwen35", 5120, 248320), &nextn),
            Verdict::Match
        );
        assert!(matches!(
            mtp_verdict(&lm("qwen35", 4096, 248320), &nextn),
            Verdict::Mismatch(_)
        ));
    }

    #[test]
    fn the_pick_prefers_verified_then_the_closest_name() {
        let w = Path::new("models/gemma-4-31B-it-Q8_0.gguf");
        let verdicts = |p: &Path| match file_name(p).as_str() {
            "mmproj-9b.gguf" => Verdict::Mismatch("x".into()),
            "mmproj-unknown.gguf" => Verdict::Unknown,
            _ => Verdict::Match,
        };
        let c = |n: &str| PathBuf::from(format!("models/{n}"));
        // a mismatch is never taken, even alone
        assert_eq!(pick(w, vec![c("mmproj-9b.gguf")], verdicts, "mmproj"), None);
        // verified beats unknown whatever the order
        assert_eq!(
            pick(
                w,
                vec![c("mmproj-unknown.gguf"), c("mmproj-model.gguf")],
                verdicts,
                "mmproj"
            ),
            Some(c("mmproj-model.gguf"))
        );
        // two verified: the one named like the weights
        assert_eq!(
            pick(
                w,
                vec![c("mmproj-model.gguf"), c("gemma-4-31B-it-mmproj-BF16.gguf")],
                verdicts,
                "mmproj"
            ),
            Some(c("gemma-4-31B-it-mmproj-BF16.gguf"))
        );
    }

    #[test]
    fn a_named_companion_that_does_not_fit_refuses_with_the_reason() {
        let m = lm("qwen35", 5120, 248320);
        let p = Path::new("other/mmproj-BF16.gguf");
        let e = check_named(
            p,
            "projector",
            "mmproj",
            mmproj_verdict(&m, &projector(&[4096])),
        )
        .unwrap_err();
        assert!(e.contains("mmproj = other/mmproj-BF16.gguf"), "{e}");
        assert!(e.contains("4096") && e.contains("5120"), "{e}");
        assert!(check_named(p, "projector", "mmproj", Verdict::Unknown).is_ok());
    }

    /// A model folder with its tower beside the weights - the layout every
    /// catalog pull produces. Empty files: an unreadable header counts as
    /// "unknown", which discovery still takes, so this drives the real scan.
    fn folder_with_tower(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("pd-vision-off-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let weights = dir.join("Model-Q8_0.gguf");
        let tower = dir.join("mmproj-BF16.gguf");
        std::fs::write(&weights, b"").expect("weights");
        std::fs::write(&tower, b"").expect("tower");
        (dir, weights, tower)
    }

    /// The Vision switch's contract: off means the tower beside the weights is
    /// NOT loaded. Before the key existed, "off" was a missing mmproj line and
    /// this scan put the tower straight back.
    #[test]
    fn vision_off_leaves_the_tower_beside_the_weights_unloaded() {
        let (dir, weights, tower) = folder_with_tower("off");
        let mut on = crate::config::Config::default();
        resolve(&mut on, &weights).expect("resolve");
        assert_eq!(
            on.mmproj.as_deref(),
            Some(tower.as_path()),
            "discovery as before"
        );

        let mut off = crate::config::Config {
            vision: Some(false),
            ..Default::default()
        };
        resolve(&mut off, &weights).expect("resolve");
        assert_eq!(
            off.mmproj, None,
            "vision = false must keep the tower unloaded"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// qwen3.8's catalog layout: the elected DFlash2 drafter beside the
    /// weights (the MTP head rides in-file). A bare runner must wire it the
    /// way a manager-written config does - only `mtp*` names were looked for.
    #[test]
    fn a_dflash_drafter_beside_the_weights_is_discovered() {
        let dir = std::env::temp_dir().join(format!("pd-dflash-disc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let weights = dir.join("Qwen3.8-27B-Q8_0.gguf");
        let drafter = dir.join("dflash2-Q4_K_M.gguf");
        std::fs::write(&weights, b"").expect("weights");
        std::fs::write(&drafter, b"").expect("drafter");
        let mut cfg = crate::config::Config::default();
        resolve(&mut cfg, &weights).expect("resolve");
        assert_eq!(cfg.mtp.as_deref(), Some(drafter.as_path()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Off plus a named tower is a contradiction, refused with both named.
    #[test]
    fn vision_off_with_a_named_tower_is_refused() {
        let (dir, weights, tower) = folder_with_tower("both");
        let mut cfg = crate::config::Config {
            vision: Some(false),
            mmproj: Some(tower.clone()),
            ..Default::default()
        };
        let e = resolve(&mut cfg, &weights).unwrap_err();
        assert!(e.contains("vision = false") && e.contains("mmproj"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
