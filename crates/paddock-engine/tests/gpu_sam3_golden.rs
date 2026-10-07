//! SAM 3's image encoder against Meta's own outputs.
//!
//! The oracle is Meta's implementation (facebookresearch/sam3), not
//! transformers: it was run outside Paddock on the same checkpoint over the
//! pictures Meta ships in its repo, twice - fp32 (TF32 off, the ViT MLP's
//! hardcoded-bf16 fused fc1 swapped for the same math in f32: the arithmetic
//! truth) and Meta's shipped bf16 runtime (autocast everywhere, TF32 on) - and
//! every stage boundary was saved. The goldens are SAM Materials-derived and
//! stay off every repository; point `SAM3_GOLDENS` at the directory.
//!
//! What "matching" means. The engine runs the vision towers' class - f16
//! operands with 11 significant bits, f32 residual and norms - so it is
//! neither run and cannot be bit-equal to either. The bar is the one every
//! tower here meets: the engine's distance from the fp32 truth must be no
//! larger than Meta's OWN bf16 run's distance from it, plane by plane. A wrong
//! rope, a dropped bias, a mis-laid conv or the wrong GELU flavour lands an
//! order of magnitude past that bar, not near it.
//!
//! Needs `facebook/sam3` under a model root (or `SAM3_DIR`) and the goldens.
//! Heavy only in the sense of a 1.7 GB upload and ~1 GB of workspace.

mod common;

use std::path::{Path, PathBuf};

use paddock_engine::gpu_model::sam3::{GpuSam3Vision, Sam3Plane};
use paddock_models::safetensors::{SafetensorsFile, StDtype};

fn say(msg: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{msg}");
}

fn goldens() -> Option<PathBuf> {
    let p = std::env::var_os("SAM3_GOLDENS").map(PathBuf::from)?;
    p.join("manifest.json").exists().then_some(p)
}

/// Meta's presence logit for a golden's probabilities: SAM 3's processor
/// scores sigmoid(class) x sigmoid(presence); SAM 3.1's detector has folded
/// presence into `pred_logits` already (`supervise_joint_box_scores`; the
/// goldens' manifest says `joint_scores`), so its probability is
/// sigmoid(pred_logits) - an infinite presence logit here.
fn meta_presence(gold: &Path, logit: f32) -> f32 {
    let man: serde_json::Value = serde_json::from_slice(
        &std::fs::read(gold.join("manifest.json")).expect("goldens manifest"),
    )
    .expect("parse goldens manifest");
    if man["joint_scores"] == true {
        f32::INFINITY
    } else {
        logit
    }
}

/// A golden tensor as f32, whatever it was saved as (Meta's bf16 run saves
/// its neck levels in bf16, its trunk in f32).
fn tensor(path: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let f = SafetensorsFile::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (t, b) = f
        .bytes(name)
        .unwrap_or_else(|| panic!("{}: no tensor {name}", path.display()));
    let v = match t.dtype {
        StDtype::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        StDtype::Bf16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        other => panic!("{name}: {other:?}"),
    };
    (t.shape.clone(), v)
}

/// The u8 picture Meta's processor fed the model, recovered from its saved
/// pixel tensor - exactly: torchvision's (u * f32(1/255) - 0.5) / 0.5 is
/// invertible on u8, and the round trip is checked value by value.
fn picture(input: &Path) -> Vec<u8> {
    let (shape, px) = tensor(input, "pixel");
    let (c, h, w) = (shape[1], shape[2], shape[3]);
    assert_eq!(c, 3);
    let mut out = vec![0u8; h * w * 3];
    for ch in 0..3 {
        for i in 0..h * w {
            let p = px[ch * h * w + i];
            let u = ((p * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            assert_eq!(
                (u as f32 * (1.0f32 / 255.0) - 0.5) / 0.5,
                p,
                "pixel {i} band {ch} is not a u8 picture through Meta's normalize"
            );
            out[i * 3 + ch] = u;
        }
    }
    out
}

/// `ours` in its raster-or-window-major layout `[rows][c]`, the golden NCHW
/// `[c][side][side]`; `cell(r)` maps our row to its (y, x).
fn rel_rms(
    ours: &[f32],
    gold: &[f32],
    c: usize,
    side: usize,
    cell: impl Fn(usize) -> (usize, usize),
) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for r in 0..side * side {
        let (y, x) = cell(r);
        for ch in 0..c {
            let g = gold[(ch * side + y) * side + x] as f64;
            let o = ours[r * c + ch] as f64;
            num += (o - g) * (o - g);
            den += g * g;
        }
    }
    (num / den.max(1e-300)).sqrt()
}

/// Window-major row -> grid cell (the engine's trunk order).
fn wm_cell(r: usize, grid: usize, win: usize) -> (usize, usize) {
    let (wi, l) = (r / (win * win), r % (win * win));
    let nwx = grid / win;
    ((wi / nwx) * win + l / win, (wi % nwx) * win + l % win)
}

#[test]
fn image_encoder_meets_metas_own_bf16_bar() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some(gold) = goldens() else {
        common::missing("SAM3_GOLDENS (Meta's reference outputs) not set or empty");
        return;
    };
    let mut model = GpuSam3Vision::load_dir(exec, &dir, 1).expect("load sam3 image encoder");
    let cfg = model.config().clone();
    let (g, win, d, f) = (cfg.grid(), cfg.window, cfg.hidden, cfg.fpn_dim);
    say(&format!(
        "sam3: weights {} MiB, workspace {} MiB",
        model.weight_bytes() >> 20,
        model.workspace_bytes() >> 20
    ));

    let mut failures = Vec::new();
    for stem in ["groceries", "test_image", "truck"] {
        let idir = gold.join("image").join(stem);
        let px = picture(&idir.join("input.safetensors"));
        // the tracker's planes are gated on truck, where the interactive goldens live
        let tracker = stem == "truck";
        model.encode(&px, 1, tracker).expect("encode");

        let fp32 = idir.join("backbone.fp32.safetensors");
        let bf16 = idir.join("backbone.bf16.safetensors");
        let ours = model.read_plane(Sam3Plane::Trunk, 1).expect("sam3 gate");
        let (_, g32) = tensor(&fp32, "trunk");
        let (_, g16) = tensor(&bf16, "trunk");
        let e_ours = rel_rms(&ours, &g32, d, g, |r| wm_cell(r, g, win));
        // Meta's bf16 trunk against its fp32 trunk, both NCHW: read the bf16
        // one as if it were "ours" in raster rows of an NHWC copy
        let meta = nchw_rel(&g16, &g32);
        report(&mut failures, stem, "trunk", e_ours, meta);

        for l in 0..3 {
            let side = model.level_side(l);
            let ours = model.read_plane(Sam3Plane::Det(l), 1).expect("sam3 gate");
            let (_, g32) = tensor(&fp32, &format!("fpn{l}"));
            let (_, g16) = tensor(&bf16, &format!("fpn{l}"));
            let e = rel_rms(&ours, &g32, f, side, raster_of(side));
            report(
                &mut failures,
                stem,
                &format!("det fpn{l}"),
                e,
                nchw_rel(&g16, &g32),
            );
        }

        if tracker {
            let pdir = gold.join("pvs").join(stem);
            let n32 = pdir.join("sam2_neck.fp32.safetensors");
            let n16 = pdir.join("sam2_neck.bf16.safetensors");
            for (plane, name, c) in [
                (Sam3Plane::TrkS0, "fpn0", f / 8),
                (Sam3Plane::TrkS1, "fpn1", f / 4),
                (Sam3Plane::Trk(2), "fpn2", f),
            ] {
                let side = match plane {
                    Sam3Plane::TrkS0 => model.level_side(0),
                    Sam3Plane::TrkS1 => model.level_side(1),
                    _ => model.level_side(2),
                };
                let ours = model.read_plane(plane, 1).expect("sam3 gate");
                let (_, g32) = tensor(&n32, name);
                let (_, g16) = tensor(&n16, name);
                let e = rel_rms(&ours, &g32, c, side, raster_of(side));
                report(
                    &mut failures,
                    stem,
                    &format!("trk {name}"),
                    e,
                    nchw_rel(&g16, &g32),
                );
            }
        }
    }
    assert!(
        failures.is_empty(),
        "past Meta's own bf16 distance: {failures:#?}"
    );
}

fn raster_of(side: usize) -> impl Fn(usize) -> (usize, usize) {
    move |r| (r / side, r % side)
}

/// rel-RMS of two same-layout planes.
fn nchw_rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        num += (x - y) * (x - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

fn report(failures: &mut Vec<String>, stem: &str, what: &str, ours: f64, meta: f64) {
    let ok = ours <= meta;
    say(&format!(
        "  {stem:10} {what:9}: ours {ours:.3e}  Meta bf16 {meta:.3e}  ({:.1}x under){}",
        meta / ours.max(1e-300),
        if ok { "" } else { "  <-- FAIL" }
    ));
    if !ok {
        failures.push(format!("{stem} {what}: {ours:.3e} > {meta:.3e}"));
    }
}

/// The engine builds its rope tables on the host; Meta saved its own in the
/// checkpoint. Same f32 recipe, so they agree to a few ulp of cos/sin.
#[test]
fn rope_tables_are_metas() {
    let Some(gold) = goldens() else {
        common::missing("SAM3_GOLDENS (Meta's reference outputs) not set or empty");
        return;
    };
    let rope = gold.join("rope.safetensors");
    let (wt, gt) = paddock_engine::gpu_model::sam3::rope_tables_for_gate(72, 24, 64, 10000.0);
    let (_, wc) = tensor(&rope, "window.cos");
    let (_, ws) = tensor(&rope, "window.sin");
    let (_, gc) = tensor(&rope, "global.cos");
    let (_, gs) = tensor(&rope, "global.sin");
    let mut worst = 0.0f32;
    for i in 0..wc.len() {
        worst = worst
            .max((wt.0[i] - wc[i]).abs())
            .max((wt.1[i] - ws[i]).abs());
    }
    // ours is window-major, Meta's raster
    for r in 0..72 * 72 {
        let (y, x) = wm_cell(r, 72, 24);
        let m = y * 72 + x;
        for j in 0..32 {
            worst = worst
                .max((gt.0[r * 32 + j] - gc[m * 32 + j]).abs())
                .max((gt.1[r * 32 + j] - gs[m * 32 + j]).abs());
        }
    }
    say(&format!("rope: worst |ours - Meta's| {worst:.3e}"));
    assert!(worst <= 2e-6, "rope tables {worst:.3e} from Meta's");
}

/// The prompt tokenizer against Meta's own on 46 prompts: every id, the
/// 32-slot layout, and a refusal - not Meta's silent truncation - past 30.
#[test]
fn tokenizer_is_metas() {
    use paddock_tokenizer::sam3::{Sam3TokenizeError, Sam3Tokenizer};
    let Some(gold) = goldens() else {
        common::missing("SAM3_GOLDENS (Meta's reference outputs) not set or empty");
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let tok = Sam3Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer.json");
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(gold.join("tokens.json")).expect("tokens.json"))
            .expect("parse tokens.json");
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .expect("sam3 gate")
            .iter()
            .map(|x| x.as_u64().expect("sam3 gate") as u32)
            .collect()
    };
    let mut bad = Vec::new();
    for p in man["prompts"].as_array().expect("sam3 gate") {
        let text = p["prompt"].as_str().expect("sam3 gate");
        let bpe = tok.bpe(text).expect("sam3 gate");
        if bpe != ids(&p["bpe"]) {
            bad.push(format!(
                "{text:?}: ours {bpe:?} vs Meta {:?}",
                ids(&p["bpe"])
            ));
            continue;
        }
        match tok.encode(text) {
            Ok(t) => {
                assert!(
                    !p["too_long"].as_bool().expect("sam3 gate"),
                    "{text:?} should be refused"
                );
                if t.ids.to_vec() != ids(&p["ids"]) {
                    bad.push(format!("{text:?}: layout {:?}", t.ids));
                }
            }
            Err(Sam3TokenizeError::TooLong { .. }) => {
                assert!(
                    p["too_long"].as_bool().expect("sam3 gate"),
                    "{text:?} refused but fits"
                );
            }
            Err(e) => panic!("{text:?}: {e}"),
        }
    }
    say(&format!(
        "tokenizer: {} prompts, {} differ",
        man["prompts"].as_array().expect("sam3 gate").len(),
        bad.len()
    ));
    assert!(bad.is_empty(), "{bad:#?}");
}

// ---------------------------------------------------------------- phase 2

/// A golden bool tensor (safetensors BOOL: one byte each).
fn bools(path: &Path, name: &str) -> Vec<bool> {
    let f = SafetensorsFile::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (_, b) = f
        .bytes(name)
        .unwrap_or_else(|| panic!("{}: no tensor {name}", path.display()));
    b.iter().map(|&x| x != 0).collect()
}

/// NCHW [1][c][s][s] -> raster rows [s*s][c].
fn nchw_to_rows(v: &[f32], c: usize, s: usize) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for ch in 0..c {
        for p in 0..s * s {
            out[p * c + ch] = v[ch * s * s + p];
        }
    }
    out
}

/// rel-RMS of `ours` against `gold` over the rows `rows` of width `w`.
fn rel_rows(
    ours: &[f32],
    gold: &[f32],
    w: usize,
    rows: impl Iterator<Item = (usize, usize)>,
) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (ro, rg) in rows {
        for i in 0..w {
            let (o, g) = (ours[ro * w + i] as f64, gold[rg * w + i] as f64);
            num += (o - g) * (o - g);
            den += g * g;
        }
    }
    (num / den.max(1e-300)).sqrt()
}

/// The prompt's valid rows (mask false = valid), compacted in order.
fn compact(prompt: &[f32], mask: &[bool], d: usize) -> (Vec<f32>, Vec<usize>) {
    let idx: Vec<usize> = (0..mask.len()).filter(|&i| !mask[i]).collect();
    let mut out = Vec::with_capacity(idx.len() * d);
    for &i in &idx {
        out.extend_from_slice(&prompt[i * d..(i + 1) * d]);
    }
    (out, idx)
}

struct Case {
    stem: String,
    file: String,
    boxes: Vec<paddock_engine::gpu_model::sam3::Sam3Box>,
    label: String,
}

fn cases(gold: &Path) -> Vec<Case> {
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(gold.join("manifest.json")).expect("sam3 gate"))
            .expect("sam3 gate");
    let mut out = Vec::new();
    for c in man["cases"].as_array().expect("sam3 gate") {
        if c["mode"].as_str() != Some("fp32") {
            continue;
        }
        let file = c["file"]
            .as_str()
            .expect("sam3 gate")
            .trim_end_matches(".fp32.safetensors")
            .to_owned();
        let boxes = if c["kind"].as_str() == Some("box") {
            let b: Vec<f32> = c["prompt"]
                .as_array()
                .expect("sam3 gate")
                .iter()
                .map(|x| x.as_f64().expect("sam3 gate") as f32)
                .collect();
            vec![paddock_engine::gpu_model::sam3::Sam3Box {
                cx: b[0],
                cy: b[1],
                w: b[2],
                h: b[3],
                positive: true,
            }]
        } else {
            Vec::new()
        };
        out.push(Case {
            stem: c["image"].as_str().expect("sam3 gate").to_owned(),
            label: file.rsplit('/').next().expect("sam3 gate").to_owned(),
            file,
            boxes,
        });
    }
    out
}

/// Kept-query yardsticks against fp32: |dprob| and box L-inf (in 1008-px
/// units), over the queries fp32 keeps.
fn kept_deltas(
    probs: &[f32],
    boxes: &[[f32; 4]],
    gold_logits: &[f32],
    gold_presence: f32,
    gold_boxes: &[f32],
) -> (usize, usize, f64, f64) {
    let pres = 1.0 / (1.0 + (-gold_presence as f64).exp());
    let (mut kept_gold, mut kept_both, mut dp, mut db) = (0usize, 0usize, 0.0f64, 0.0f64);
    for q in 0..probs.len() {
        let pg = (1.0 / (1.0 + (-gold_logits[q] as f64).exp())) * pres;
        if pg > 0.5 {
            kept_gold += 1;
            if probs[q] > 0.5 {
                kept_both += 1;
            }
            dp = dp.max((probs[q] as f64 - pg).abs());
            for k in 0..4 {
                db = db.max(((boxes[q][k] - gold_boxes[q * 4 + k]) as f64).abs() * 1008.0);
            }
        }
    }
    (kept_gold, kept_both, dp, db)
}

/// Each detector stage on Meta's OWN fp32 inputs, against Meta's fp32
/// outputs - the stage's error in isolation - with Meta's bf16 run (which
/// carries its whole upstream drift) as the bar.
#[test]
fn detector_stages_meet_metas_own_bf16_bar() {
    use paddock_engine::gpu_model::sam3::GpuSam3Detector;
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some(gold) = goldens() else {
        common::missing("SAM3_GOLDENS (Meta's reference outputs) not set or empty");
        return;
    };
    let mut det = GpuSam3Detector::load_dir(exec.clone(), &dir, 4).expect("load sam3 detector");
    let g = det.geom();
    let (d, t, s) = (g.d, g.tokens(), g.grid);
    say(&format!(
        "sam3 detector: weights {} MiB, workspace {} MiB",
        det.weight_bytes() >> 20,
        det.workspace_bytes() >> 20
    ));
    let mut failures = Vec::new();

    // the position table against Meta's (computed on its GPU at build)
    {
        let bb = gold
            .join("image")
            .join("truck")
            .join("backbone.fp32.safetensors");
        let (_, p2) = tensor(&bb, "pos2");
        let gold_pos = nchw_to_rows(&p2, d, s);
        let ours = exec.to_host_len(det.pos_table(), t * d).expect("sam3 gate");
        let worst = ours
            .iter()
            .zip(&gold_pos)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        say(&format!("  pos table: worst |ours - Meta's| {worst:.3e}"));
        if worst > 1e-4 {
            failures.push(format!("pos table {worst:.3e}"));
        }
    }

    for c in cases(&gold) {
        let idir = gold.join("image").join(&c.stem);
        let bb32 = idir.join("backbone.fp32.safetensors");
        let f32p = gold.join(format!("{}.fp32.safetensors", c.file));
        let f16p = gold.join(format!("{}.bf16.safetensors", c.file));
        let up = |v: &[f32]| exec.to_device(v).expect("sam3 gate");
        let level = |l: usize| {
            let (_, v) = tensor(&bb32, &format!("fpn{l}"));
            nchw_to_rows(&v, d, (4 * s) >> l)
        };
        let fpn2 = up(&level(2));

        // ---- geometry (and prompt assembly) on Meta's text tokens + level ----
        let (_, lf) = tensor(&f32p, "language_features");
        let lmask = bools(&f32p, "language_mask");
        let n_valid = lmask.iter().filter(|m| !**m).count();
        let text = up(&lf);
        det.encode_prompt(&text, 0, n_valid, &fpn2, &c.boxes)
            .expect("encode_prompt");
        let ours_prompt = det.read_prompt().expect("sam3 gate");
        let (_, pb32) = tensor(&f32p, "prompt_before_enc");
        let (_, pb16) = tensor(&f16p, "prompt_before_enc");
        let n_geo = c.boxes.len() + 1;
        let geo_rows = || (0..n_geo).map(|i| (n_valid + i, 32 + i));
        let e = rel_rows(&ours_prompt, &pb32, d, geo_rows());
        let m = rel_rows(&pb16, &pb32, d, (0..n_geo).map(|i| (32 + i, 32 + i)));
        report(&mut failures, &c.label, "geometry", e, m);

        // ---- fusion encoder on Meta's prompt ----
        let pmask = bools(&f32p, "prompt_mask");
        let (cp, _) = compact(&pb32, &pmask, d);
        let prompt_dev = up(&cp);
        det.set_prompt(&prompt_dev, cp.len() / d)
            .expect("sam3 gate");
        det.fuse(&fpn2).expect("fuse");
        let ours_e = det.read_memory().expect("sam3 gate");
        let (_, e32) = tensor(&f32p, "encoder_hidden_states");
        let (_, e16) = tensor(&f16p, "encoder_hidden_states");
        report(
            &mut failures,
            &c.label,
            "fusion",
            nchw_rel(&ours_e, &e32),
            nchw_rel(&e16, &e32),
        );

        // ---- decoder + scorer on Meta's memory and prompt ----
        det.set_memory(&up(&e32)).expect("sam3 gate");
        det.decode().expect("decode");
        let ours_q = det.read_queries().expect("sam3 gate");
        let (_, q32) = tensor(&f32p, "queries");
        let (_, q16) = tensor(&f16p, "queries");
        report(
            &mut failures,
            &c.label,
            "queries",
            nchw_rel(&ours_q, &q32),
            nchw_rel(&q16, &q32),
        );
        let dets = det.read_detections().expect("sam3 gate");
        let (_, lg32) = tensor(&f32p, "pred_logits");
        let (_, lg16) = tensor(&f16p, "pred_logits");
        let (_, pr32) = tensor(&f32p, "presence_logit_dec");
        let (_, pr16) = tensor(&f16p, "presence_logit_dec");
        let (_, bx32) = tensor(&f32p, "pred_boxes");
        let (_, bx16) = tensor(&f16p, "pred_boxes");
        let ours_p = det.picture_scores(&dets);
        let (g32p, g16p) = (meta_presence(&gold, pr32[0]), meta_presence(&gold, pr16[0]));
        let (kg, kb, dp, db) = kept_deltas(&ours_p, &dets.boxes, &lg32, g32p, &bx32);
        let p16: Vec<f32> = {
            let pres = 1.0 / (1.0 + (-g16p).exp());
            lg16.iter()
                .map(|l| (1.0 / (1.0 + (-l).exp())) * pres)
                .collect()
        };
        let b16: Vec<[f32; 4]> = bx16.as_chunks::<4>().0.to_vec();
        let (_, kb16, dp16, db16) = kept_deltas(&p16, &b16, &lg32, g32p, &bx32);
        say(&format!(
            "  {:28} kept {kb}/{kg} (Meta bf16 {kb16}/{kg}); |dprob| {dp:.4} (bf16 {dp16:.4}); \
             box {db:.2} px (bf16 {db16:.2}); presence {:.4} vs {:.4} (bf16 {:.4})",
            c.label, dets.presence_logit, pr32[0], pr16[0]
        ));
        if kb != kg {
            failures.push(format!("{}: kept {kb} of Meta fp32's {kg}", c.label));
        }
        if dp > dp16.max(0.01) || db > db16.max(1.0) {
            failures.push(format!(
                "{}: kept-query drift |dprob| {dp:.4} / box {db:.2} px past Meta bf16's \
                 {dp16:.4} / {db16:.2}",
                c.label
            ));
        }

        // ---- segmentation head on Meta's memory + levels, our queries ----
        let (fpn0, fpn1) = (up(&level(0)), up(&level(1)));
        det.segment(&fpn0, &fpn1).expect("segment");
        let ours_m = det.read_masks().expect("sam3 gate");
        let (_, m32) = tensor(&f32p, "pred_masks");
        let (_, m16) = tensor(&f16p, "pred_masks");
        let px = 16 * t;
        let mut worst_iou = 1.0f64;
        let mut worst_iou16 = 1.0f64;
        let pres = 1.0 / (1.0 + (-g32p as f64).exp());
        for q in 0..g.queries {
            let pg = (1.0 / (1.0 + (-lg32[q] as f64).exp())) * pres;
            if pg <= 0.5 {
                continue;
            }
            let (mut i_o, mut u_o, mut i_m, mut u_m) = (0usize, 0usize, 0usize, 0usize);
            for p in 0..px {
                let gm = m32[q * px + p] > 0.0;
                let om = ours_m[p * g.queries + q] > 0.0;
                let bm = m16[q * px + p] > 0.0;
                i_o += usize::from(gm && om);
                u_o += usize::from(gm || om);
                i_m += usize::from(gm && bm);
                u_m += usize::from(gm || bm);
            }
            worst_iou = worst_iou.min(i_o as f64 / u_o.max(1) as f64);
            worst_iou16 = worst_iou16.min(i_m as f64 / u_m.max(1) as f64);
        }
        say(&format!(
            "  {:28} mask IoU at 288 (kept): ours {worst_iou:.4}, Meta bf16 {worst_iou16:.4}",
            c.label
        ));
        if worst_iou < worst_iou16.min(0.995) {
            failures.push(format!(
                "{}: mask IoU {worst_iou:.4} under Meta bf16's {worst_iou16:.4}",
                c.label
            ));
        }
        let ours_sem = det.read_semantic().expect("sam3 gate");
        let (_, s32) = tensor(&f32p, "semantic_seg");
        let (_, s16) = tensor(&f16p, "semantic_seg");
        report(
            &mut failures,
            &c.label,
            "semantic",
            nchw_rel(&ours_sem, &s32),
            nchw_rel(&s16, &s32),
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The whole concept path from the picture's pixels and the prompt's text:
/// our image encoder, text tower and detector chained, held to Meta's fp32
/// DECISIONS - the same kept instances, scores and boxes within Meta's own
/// bf16 drift, masks at 288^2 by IoU.
#[test]
fn concept_prompts_end_to_end() {
    use paddock_engine::gpu_model::sam3::{GpuSam3Detector, GpuSam3Text};
    use paddock_tokenizer::sam3::Sam3Tokenizer;
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some(gold) = goldens() else {
        common::missing("SAM3_GOLDENS (Meta's reference outputs) not set or empty");
        return;
    };
    let tok = Sam3Tokenizer::from_file(&dir.join("tokenizer.json")).expect("sam3 gate");
    let mut vis = GpuSam3Vision::load_dir(exec.clone(), &dir, 1).expect("vision");
    let mut text = GpuSam3Text::load_dir(exec.clone(), &dir, 1).expect("text");
    let mut det = GpuSam3Detector::load_dir(exec.clone(), &dir, 4).expect("detector");
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(gold.join("manifest.json")).expect("sam3 gate"))
            .expect("sam3 gate");
    let mut failures = Vec::new();
    let mut last_stem = String::new();
    for c in cases(&gold) {
        if c.stem != last_stem {
            let px = picture(&gold.join("image").join(&c.stem).join("input.safetensors"));
            vis.encode(&px, 1, false).expect("encode");
            last_stem = c.stem.clone();
        }
        let entry = man["cases"]
            .as_array()
            .expect("sam3 gate")
            .iter()
            .find(|e| {
                e["file"].as_str().expect("sam3 gate").starts_with(&c.file) && e["mode"] == "fp32"
            })
            .expect("sam3 gate");
        let words = entry["encoded_text"].as_str().expect("sam3 gate");
        let tk = tok.encode(words).expect("sam3 gate");
        text.encode(&tk.ids, 1).expect("sam3 gate");

        // the text tower against Meta's, valid rows
        let f32p = gold.join(format!("{}.fp32.safetensors", c.file));
        let f16p = gold.join(format!("{}.bf16.safetensors", c.file));
        let ours_t = text.read_features(1).expect("sam3 gate");
        let (_, l32) = tensor(&f32p, "language_features");
        let (_, l16) = tensor(&f16p, "language_features");
        let rows = || (0..tk.valid).map(|i| (i, i));
        report(
            &mut failures,
            &c.label,
            "text",
            rel_rows(&ours_t, &l32, 256, rows()),
            rel_rows(&l16, &l32, 256, rows()),
        );

        det.encode_prompt(text.features(), 0, tk.valid, vis.det_level(2), &c.boxes)
            .expect("sam3 gate");
        det.fuse(vis.det_level(2)).expect("sam3 gate");
        det.decode().expect("sam3 gate");
        let dets = det.read_detections().expect("sam3 gate");
        let (_, lg32) = tensor(&f32p, "pred_logits");
        let (_, pr32) = tensor(&f32p, "presence_logit_dec");
        let (_, bx32) = tensor(&f32p, "pred_boxes");
        let ours_p = det.picture_scores(&dets);
        let g32p = meta_presence(&gold, pr32[0]);
        let (kg, kb, dp, db) = kept_deltas(&ours_p, &dets.boxes, &lg32, g32p, &bx32);
        let ours_kept = ours_p.iter().filter(|p| **p > 0.5).count();
        say(&format!(
            "  {:28} end to end: kept {ours_kept} (both {kb}, Meta fp32 {kg}); |dprob| {dp:.4}; \
             box {db:.2} px",
            c.label
        ));
        // the end-to-end bar: the same decisions, and scores within the worst
        // kept-instance drift Meta's own bf16 run showed (0.0097); boxes
        // within 2 px of the 1008-px input (Meta bf16's worst was 0.7 px of
        // an 800-px picture, ~0.9 px here)
        if ours_kept != kg || kb != kg || dp > 0.0097 || db > 2.0 {
            failures.push(format!(
                "{}: kept {ours_kept}/{kb} vs {kg}, |dprob| {dp:.4}, box {db:.2} px",
                c.label
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The request path at the picture's own size: our JPEG decode
/// (libjpeg-turbo's bytes), torchvision's resize, the whole model, and each
/// kept mask brought back to the picture and run-length encoded - against
/// Meta's processor outputs (kept scores, xyxy boxes, masks at the picture).
#[test]
fn request_path_matches_metas_processor() {
    use paddock_engine::gpu_model::sam3::{GpuSam3, Sam3Request};
    use paddock_tokenizer::sam3::Sam3Tokenizer;
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some(gold) = goldens() else {
        common::missing("SAM3_GOLDENS (Meta's reference outputs) not set or empty");
        return;
    };
    let assets = std::env::var_os("SAM3_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| gold.join("..").join("sam3").join("assets").join("images"));
    let tok = Sam3Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let mut sam = GpuSam3::load_dir(exec, &dir, 16_000_000, 4).expect("load sam3");
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(gold.join("manifest.json")).expect("manifest"))
            .expect("parse manifest");
    let mut failures = Vec::new();
    for c in cases(&gold) {
        let file = man["images"][&c.stem]["file"].as_str().expect("image file");
        let bytes = std::fs::read(assets.join(file)).expect("asset picture");
        let img = paddock_jpeg::decode_rgb(&bytes, 64_000_000).expect("decode");
        let entry = man["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|e| {
                e["file"].as_str().is_some_and(|f| f.starts_with(&c.file)) && e["mode"] == "fp32"
            })
            .expect("case entry");
        let tk = tok
            .encode(entry["encoded_text"].as_str().expect("text"))
            .expect("tokenize");
        let req = Sam3Request {
            ids: tk.ids,
            valid: tk.valid,
            boxes: c.boxes.clone(),
            threshold: 0.5,
        };
        let out = sam
            .segment(&img.rgb, img.width, img.height, &req)
            .expect("segment");

        // the resize, against the pixels Meta's processor fed its model
        let ours_in = sam.read_input().expect("input");
        let theirs = picture(&gold.join("image").join(&c.stem).join("input.safetensors"));
        let off = ours_in.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        if off != 0 {
            failures.push(format!("{}: resize differs on {off} bytes", c.label));
        }

        let f32p = gold.join(format!("{}.fp32.safetensors", c.file));
        let (_, ks) = tensor(&f32p, "kept_scores");
        let (_, kb) = tensor(&f32p, "kept_boxes");
        let km = {
            let f = SafetensorsFile::open(&f32p).expect("golden");
            f.bytes("kept_masks")
                .map(|(_, b)| b.to_vec())
                .unwrap_or_default()
        };
        let (w, h) = (img.width, img.height);
        let mut worst = (0.0f64, 0.0f64, 1.0f64);
        if out.instances.len() != ks.len() {
            failures.push(format!(
                "{}: {} instances against Meta fp32's {}",
                c.label,
                out.instances.len(),
                ks.len()
            ));
        } else {
            for (gi, gs) in ks.iter().enumerate() {
                let gbox = &kb[gi * 4..gi * 4 + 4];
                // ours ordered by score, theirs by query: pair by box
                let (oi, _) = out
                    .instances
                    .iter()
                    .enumerate()
                    .map(|(i, o)| {
                        let d = (0..4)
                            .map(|k| (o.bbox[k] - gbox[k]).abs())
                            .fold(0.0f32, f32::max);
                        (i, d)
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("an instance");
                let o = &out.instances[oi];
                let db = (0..4)
                    .map(|k| (o.bbox[k] - gbox[k]).abs() as f64)
                    .fold(0.0, f64::max);
                // our RLE back to a row-major picture
                let mut mine = vec![0u8; w * h];
                let (mut k, mut val) = (0usize, 0u8);
                for &run in &o.rle {
                    for kk in k..k + run as usize {
                        let (x, y) = (kk / h, kk % h);
                        mine[y * w + x] = val;
                    }
                    k += run as usize;
                    val ^= 1;
                }
                let theirs = &km[gi * w * h..(gi + 1) * w * h];
                let (mut i, mut u) = (0usize, 0usize);
                for (a, b) in mine.iter().zip(theirs) {
                    i += usize::from(*a != 0 && *b != 0);
                    u += usize::from(*a != 0 || *b != 0);
                }
                let iou = i as f64 / u.max(1) as f64;
                worst.0 = worst.0.max((o.score - gs).abs() as f64);
                worst.1 = worst.1.max(db);
                worst.2 = worst.2.min(iou);
            }
        }
        say(&format!(
            "  {:28} {}x{}: {} kept, |dscore| {:.4}, box {:.2} px, mask IoU {:.4}; \
             resize {:.1} ms encode {:.1} prompt {:.1} detect {:.1} masks {:.1}",
            c.label,
            w,
            h,
            out.instances.len(),
            worst.0,
            worst.1,
            worst.2,
            out.timings.resize_ms,
            out.timings.encode_ms,
            out.timings.prompt_ms,
            out.timings.detect_ms,
            out.timings.masks_ms
        ));
        // Meta's own bf16 yardsticks on the same outputs: 0.0097, 0.70 px, 0.9963
        if worst.0 > 0.0097 || worst.1 > 0.70 || worst.2 < 0.9963 {
            failures.push(format!(
                "{}: |dscore| {:.4}, box {:.2} px, mask IoU {:.4}",
                c.label, worst.0, worst.1, worst.2
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// An int64 golden tensor (the click labels Meta saved).
fn ints(path: &Path, name: &str) -> Vec<i64> {
    let f = SafetensorsFile::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (t, b) = f
        .bytes(name)
        .unwrap_or_else(|| panic!("{}: no tensor {name}", path.display()));
    assert_eq!(t.dtype, StDtype::I64, "{name}");
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

/// Meta's interactive predictor (SAM3InteractiveImagePredictor) on truck.jpg
/// with its SAM 1 notebook prompts: one click (three candidates), two clicks
/// +/+ and +/- refining the first answer's best low-res logits, and a box.
/// Gated per case on Meta's own bf16 distance from its fp32 run: the 288^2
/// low-res logits (rel-RMS), the predicted IoU, and the mask at the
/// picture's 1800 x 1200 (IoU against fp32's).
#[test]
fn click_prompts_meet_metas_own_bf16_bar() {
    use paddock_engine::gpu_model::sam3::{GpuSam3Pvs, PvsMask, PvsPoint, PvsPrompt};
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some(gold) = goldens() else {
        common::missing("SAM3_GOLDENS (Meta's reference outputs) not set or empty");
        return;
    };
    let mut vision = GpuSam3Vision::load_dir(exec.clone(), &dir, 1).expect("load sam3 encoder");
    let mut pvs = GpuSam3Pvs::load_dir(exec.clone(), &dir).expect("load sam3 click heads");
    say(&format!(
        "sam3 click heads: weights {} MiB, workspace {} MiB",
        pvs.weight_bytes() >> 20,
        pvs.workspace_bytes() >> 20
    ));
    let px = picture(&gold.join("image").join("truck").join("input.safetensors"));
    vision.encode(&px, 1, true).expect("encode");

    let pdir = gold.join("pvs");
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(pdir.join("manifest.json")).expect("manifest"))
            .expect("parse manifest");
    let wh = &man["image"]["size_wh"];
    let (w, h) = (
        wh[0].as_u64().expect("width") as usize,
        wh[1].as_u64().expect("height") as usize,
    );
    // Meta's transform_coords: x / W * 1008, then the encoder's +0.5 and / 1008
    let nx = |x: f32| (x / w as f32 * 1008.0 + 0.5) / 1008.0;
    let ny = |y: f32| (y / h as f32 * 1008.0 + 0.5) / 1008.0;
    let side = 288usize;
    let mut maskbuf = exec.alloc_u8(w * h).expect("mask plane");
    let mut failures = Vec::new();
    for case in ["point1", "point2-pos-pos", "point2-pos-neg", "box"] {
        let f32p = pdir.join("truck").join(format!("{case}.fp32.safetensors"));
        let bf16p = pdir.join("truck").join(format!("{case}.bf16.safetensors"));
        let entry = man["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|e| e["case"] == case && e["mode"] == "fp32")
            .expect("case entry");
        let multimask = entry["multimask"].as_bool().expect("multimask");
        let file = SafetensorsFile::open(&f32p).expect("golden");
        let points: Vec<PvsPoint> = if file.bytes("point_coords").is_some() {
            let (_, xy) = tensor(&f32p, "point_coords");
            let labels = ints(&f32p, "point_labels");
            labels
                .iter()
                .enumerate()
                .map(|(i, &l)| PvsPoint {
                    x: nx(xy[2 * i]),
                    y: ny(xy[2 * i + 1]),
                    positive: l == 1,
                })
                .collect()
        } else {
            Vec::new()
        };
        let bbox = file.bytes("box_xyxy").map(|_| {
            let (_, b) = tensor(&f32p, "box_xyxy");
            [nx(b[0]), ny(b[1]), nx(b[2]), ny(b[3])]
        });
        let mask_in = file
            .bytes("mask_input")
            .map(|_| tensor(&f32p, "mask_input").1);
        let prompt = PvsPrompt {
            points,
            bbox,
            mask: mask_in.as_deref().map(PvsMask::Host),
            multimask,
        };
        let res = pvs.predict(&vision, &prompt).expect("predict");
        let logits = pvs.read_logits().expect("logits");

        let (gshape, g32) = tensor(&f32p, "low_res_logits");
        let (_, g16) = tensor(&bf16p, "low_res_logits");
        let (_, s32) = tensor(&f32p, "scores");
        let (_, s16) = tensor(&bf16p, "scores");
        let m32 = SafetensorsFile::open(&f32p)
            .expect("golden")
            .bytes("masks")
            .map(|(_, b)| b.to_vec())
            .expect("masks");
        let m16 = SafetensorsFile::open(&bf16p)
            .expect("golden")
            .bytes("masks")
            .map(|(_, b)| b.to_vec())
            .expect("masks");
        let n_out = gshape[0];
        // Meta returns masks 1..3 in order for multimask, its one choice else
        let ours_k: Vec<usize> = if multimask {
            (1..=n_out).collect()
        } else {
            vec![res.candidates[0].0]
        };
        assert_eq!(ours_k.len(), n_out, "{case}: candidate count");
        let plane = side * side;
        for (i, &k) in ours_k.iter().enumerate() {
            let ours: Vec<f32> = (0..plane)
                .map(|p| logits[p * 4 + k].clamp(-32.0, 32.0))
                .collect();
            let gold_i = &g32[i * plane..(i + 1) * plane];
            let e = nchw_rel(&ours, gold_i);
            let meta = nchw_rel(&g16[i * plane..(i + 1) * plane], gold_i);
            report(&mut failures, case, &format!("logits{i}"), e, meta);

            let ds = (res.iou[k] - s32[i]).abs();
            let bar = (s16[i] - s32[i]).abs().max(0.002);
            say(&format!(
                "  {case:10} score{i}   : ours {:.5} Meta fp32 {:.5} bf16 {:.5} (|d| {ds:.5}, bar {bar:.5}){}",
                res.iou[k],
                s32[i],
                s16[i],
                if ds <= bar { "" } else { "  <-- FAIL" }
            ));
            if ds > bar {
                failures.push(format!("{case} score{i}: |d| {ds:.5} > {bar:.5}"));
            }

            // the mask at the picture's size: the hole-filled logits, bilinear, > 0
            exec.sam3_mask_up(pvs.masks(), &mut maskbuf, side, 4, k, h, w)
                .expect("mask upsample");
            let cm = exec.to_host_u8_len(&maskbuf, w * h).expect("mask");
            let (mut inter, mut uni, mut i16, mut u16) = (0u64, 0u64, 0u64, 0u64);
            for y in 0..h {
                for x in 0..w {
                    let o = cm[x * h + y] != 0;
                    let g = m32[(i * h + y) * w + x] != 0;
                    let b = m16[(i * h + y) * w + x] != 0;
                    inter += u64::from(o && g);
                    uni += u64::from(o || g);
                    i16 += u64::from(b && g);
                    u16 += u64::from(b || g);
                }
            }
            let iou = inter as f64 / uni.max(1) as f64;
            let iou16 = i16 as f64 / u16.max(1) as f64;
            let bar = iou16.min(0.995);
            say(&format!(
                "  {case:10} mask{i}    : IoU vs Meta fp32 {iou:.4} (Meta bf16 {iou16:.4}){}",
                if iou >= bar { "" } else { "  <-- FAIL" }
            ));
            if iou < bar {
                failures.push(format!("{case} mask{i}: IoU {iou:.4} < {bar:.4}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "past Meta's own bf16 distance: {failures:#?}"
    );
}

/// A COCO RLE over a column-major `h`-tall mask back to a row-major plane.
fn rle_rows(rle: &[u32], w: usize, h: usize) -> Vec<u8> {
    let mut rows = vec![0u8; w * h];
    let (mut k, mut val) = (0usize, 0u8);
    for &run in rle {
        for kk in k..k + run as usize {
            rows[(kk % h) * w + kk / h] = val;
        }
        k += run as usize;
        val ^= 1;
    }
    rows
}

/// Clicks through the request path, the way the runner calls it: our JPEG
/// decode and resize, `segment_points`, each candidate brought back to the
/// picture and run-length encoded. Meta's notebook chain on truck.jpg - one
/// click (three candidates), two clicks refining that answer by the
/// `refine_id` it handed out, and a box - against Meta's fp32 masks at the
/// picture's 1800 x 1200, Meta bf16's own IoU and score drift as the bars.
/// A refine id that is no longer the last answer is refused.
#[test]
fn click_requests_refine_like_metas_notebook() {
    use paddock_engine::gpu_model::sam3::{GpuSam3, Sam3Click, Sam3Fail, Sam3PointRequest};
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some(gold) = goldens() else {
        common::missing("SAM3_GOLDENS (Meta's reference outputs) not set or empty");
        return;
    };
    let assets = std::env::var_os("SAM3_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| gold.join("..").join("sam3").join("assets").join("images"));
    let mut sam = GpuSam3::load_dir(exec, &dir, 16_000_000, 4).expect("load sam3");
    let bytes = std::fs::read(assets.join("truck.jpg")).expect("truck.jpg");
    let img = paddock_jpeg::decode_rgb(&bytes, 64_000_000).expect("decode");
    let (w, h) = (img.width, img.height);
    let pdir = gold.join("pvs").join("truck");

    let clicks_of = |case: &str| -> Vec<Sam3Click> {
        let p = pdir.join(format!("{case}.fp32.safetensors"));
        if SafetensorsFile::open(&p)
            .expect("golden")
            .bytes("point_coords")
            .is_none()
        {
            return Vec::new();
        }
        let (_, xy) = tensor(&p, "point_coords");
        ints(&p, "point_labels")
            .iter()
            .enumerate()
            .map(|(i, &l)| Sam3Click {
                x: xy[2 * i],
                y: xy[2 * i + 1],
                positive: l == 1,
            })
            .collect()
    };
    let mut failures = Vec::new();
    let check = |failures: &mut Vec<String>,
                 case: &str,
                 out: &paddock_engine::gpu_model::sam3::Sam3Output| {
        let f32p = pdir.join(format!("{case}.fp32.safetensors"));
        let bf16p = pdir.join(format!("{case}.bf16.safetensors"));
        let (_, s32) = tensor(&f32p, "scores");
        let (_, s16) = tensor(&bf16p, "scores");
        let plane = |p: &Path| {
            SafetensorsFile::open(p)
                .expect("golden")
                .bytes("masks")
                .map(|(_, b)| b.to_vec())
                .expect("masks")
        };
        let (m32, m16) = (plane(&f32p), plane(&bf16p));
        if out.instances.len() != s32.len() {
            failures.push(format!(
                "{case}: {} candidates against Meta's {}",
                out.instances.len(),
                s32.len()
            ));
            return;
        }
        // ours best first, Meta's in decoder order: pair by score
        for (i, (&g, &b)) in s32.iter().zip(&s16).enumerate() {
            let o = out
                .instances
                .iter()
                .min_by(|x, y| (x.score - g).abs().total_cmp(&(y.score - g).abs()))
                .expect("a candidate");
            let ds = (o.score - g).abs();
            let sbar = (b - g).abs().max(0.002);
            let mine = rle_rows(&o.rle, w, h);
            let at = |m: &[u8], k: usize| m[i * w * h + k] != 0;
            let (mut inter, mut uni, mut i16, mut u16) = (0u64, 0u64, 0u64, 0u64);
            for (k, &v) in mine.iter().enumerate() {
                let (o, g, b) = (v != 0, at(&m32, k), at(&m16, k));
                inter += u64::from(o && g);
                uni += u64::from(o || g);
                i16 += u64::from(b && g);
                u16 += u64::from(b || g);
            }
            let iou = inter as f64 / uni.max(1) as f64;
            let ibar = (i16 as f64 / u16.max(1) as f64).min(0.995);
            let area_ok = o.area == mine.iter().map(|&v| u64::from(v)).sum::<u64>();
            say(&format!(
                "  {case:14} mask{i}: score {:.5} vs {g:.5} (bar {sbar:.5}), IoU {iou:.4} \
                 (bar {ibar:.4}), box {:?}; detect {:.1} ms masks {:.1}{}",
                o.score,
                o.bbox,
                out.timings.detect_ms,
                out.timings.masks_ms,
                if out.timings.image_reused {
                    ", picture reused"
                } else {
                    ""
                }
            ));
            if ds > sbar || iou < ibar || !area_ok {
                failures.push(format!(
                    "{case} mask{i}: |dscore| {ds:.5} (bar {sbar:.5}), IoU {iou:.4} (bar \
                     {ibar:.4}), area agrees {area_ok}"
                ));
            }
        }
    };

    let ask = |sam: &mut GpuSam3, clicks: Vec<Sam3Click>, bbox, refine| {
        sam.segment_points(
            &img.rgb,
            w,
            h,
            &Sam3PointRequest {
                clicks,
                bbox,
                multimask: None,
                refine,
            },
        )
    };
    // one click: three candidates, best first, and a refine id
    let first = ask(&mut sam, clicks_of("point1"), None, None).expect("point1");
    check(&mut failures, "point1", &first);
    let id = first.refine_id.expect("a refine id");
    // two clicks refining it: one answer
    let out = ask(&mut sam, clicks_of("point2-pos-pos"), None, Some(id)).expect("pos-pos");
    check(&mut failures, "point2-pos-pos", &out);
    // that id is spent now
    match ask(&mut sam, clicks_of("point2-pos-neg"), None, Some(id)) {
        Err(Sam3Fail::Request(_)) => {}
        Err(e) => failures.push(format!("stale refine: {e}")),
        Ok(_) => failures.push("a stale refine id was served".into()),
    }
    let again = ask(&mut sam, clicks_of("point1"), None, None).expect("point1 again");
    let id = again.refine_id.expect("a refine id");
    let out = ask(&mut sam, clicks_of("point2-pos-neg"), None, Some(id)).expect("pos-neg");
    check(&mut failures, "point2-pos-neg", &out);
    // a box alone
    let b = {
        let p = pdir.join("box.fp32.safetensors");
        let (_, b) = tensor(&p, "box_xyxy");
        [b[0], b[1], b[2], b[3]]
    };
    let out = ask(&mut sam, Vec::new(), Some(b), None).expect("box");
    check(&mut failures, "box", &out);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Meta's own checkpoint read the engine's way - the torch zip reader (no
/// pickle code runs) and the Meta -> transformers renaming - gives SAM 3's
/// `model.safetensors` back tensor for tensor: same dtype, shape and bytes,
/// including the cut q|k|v thirds, the position table without its class row
/// and the stacked point embeddings. The one tensor it does not give is
/// `text_projection` (transposed by transformers, never loaded). This is the
/// gate for loading SAM 3.1, which ships only in Meta's form.
#[test]
fn metas_checkpoint_reads_as_the_safetensors() {
    use paddock_engine::gpu_model::sam3::MetaCheckpoint;
    use paddock_models::safetensors::{ShardedSafetensors, TensorSource};
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let pt = dir.join("sam3.pt");
    if !pt.exists() {
        common::missing("sam3.pt beside model.safetensors");
        return;
    }
    let t0 = std::time::Instant::now();
    let meta = MetaCheckpoint::open(&pt).expect("read sam3.pt");
    let opened = t0.elapsed();
    let st = ShardedSafetensors::open_dir(&dir).expect("model.safetensors");
    let (mut same, mut missing, mut differ) = (0usize, Vec::new(), Vec::new());
    let mut names = st.tensor_names();
    names.sort();
    for name in &names {
        let (want, wb) = st.tensor(name).expect("listed");
        match meta.tensor(name) {
            None => missing.push(name.clone()),
            Some((got, gb)) => {
                if got.dtype == want.dtype && got.shape == want.shape && gb == wb {
                    same += 1;
                } else {
                    differ.push(format!(
                        "{name}: {:?} {:?} vs {:?} {:?}",
                        got.dtype, got.shape, want.dtype, want.shape
                    ));
                }
            }
        }
    }
    say(&format!(
        "  sam3.pt read in {:.0} ms: {same} of {} safetensors tensors byte for byte; missing {missing:?}",
        opened.as_secs_f64() * 1e3,
        names.len()
    ));
    for d in differ.iter().take(20) {
        say(&format!("    differs: {d}"));
    }
    assert!(differ.is_empty(), "{} tensors differ", differ.len());
    assert_eq!(
        missing,
        vec!["detector_model.text_encoder.text_projection.weight".to_owned()]
    );
}
