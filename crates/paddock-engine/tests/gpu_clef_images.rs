//! Clef's image lane against the reference's own evaluation (the oracle's
//! image fixtures, `<CLEF_DIR>/golden/{f32,bf16}/img_*`: Cloudflare's
//! `joint_schema_model.py` under Transformers 5.10.2, F32 with TF32 off, and
//! the vendor's BF16 beside it):
//!
//!   - the processor: given the processor's decoded pixels, every fixture's
//!     pixel rows are the processor's own, bit for bit (resize, normalize,
//!     patch order);
//!   - the tower: its last block's rows and the merger's are closer to the
//!     F32 reference than the vendor's BF16 is (relative RMS);
//!   - the model: final hidden rows, logits and probabilities closer than
//!     the vendor's, no answer flipped;
//!   - a pass of every image fixture and a text request packed back to back
//!     gives each request the logits it gets alone, bit for bit.
//!
//! `PADDOCK_HEAVY_TESTS=1` and `CLEF_DIR` run it. Each fixture prints one
//! JSON line.
//!
//! With `CLEF_GGUF` and `CLEF_COMPANION` it gates the GGUF lane - the GGUF's
//! backbone and head, the companion's tower - against the same-weights
//! reference in `<CLEF_DIR>/golden/gguf-f32` (the oracle's `--gguf`). No
//! vendor evaluation of those weights exists, so the bars are the BF16
//! lane's class: the processor bit for bit; the tower (the release's own
//! weights either way) and merger within 5e-4 relative RMS; logits 1e-3,
//! probabilities 1e-4, no flips.

mod common;

use std::path::{Path, PathBuf};

use paddock_engine::gpu_model::clef::{ClefImage, ClefQuestion, ClefRequest, GpuClef};

const IMAGE_PAD: u32 = 248056;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// `||a - b|| / ||b||` over the whole plane, F64 sums.
fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "shape");
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += f64::from(x - y).powi(2);
        den += f64::from(*y).powi(2);
    }
    (num / den).sqrt()
}

fn line(v: serde_json::Value) {
    use std::io::Write;
    let _ = writeln!(std::io::stdout(), "{v}");
}

fn softmax(z: &[f32]) -> Vec<f64> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = z.iter().map(|&x| (x as f64 - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .map_or(0, |(i, _)| i)
}

struct Fixture {
    id: String,
    ids: Vec<u32>,
    questions: Vec<ClefQuestion>,
    images: Vec<ClefImage>,
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(
        &std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .expect("golden parses")
}

/// The reference the lane is held to: the release's F32 evaluation, or the
/// GGUF's weights in it.
fn reference() -> &'static str {
    if std::env::var_os("CLEF_GGUF").is_some() {
        "gguf-f32"
    } else {
        "f32"
    }
}

/// Every `img_*` fixture of the F32 golden: ids and question spans as the
/// reference encoded them, images as the processor decoded them, each placed
/// on its run of `<|image_pad|>` rows.
fn fixtures(dir: &Path) -> Vec<Fixture> {
    let rf = dir.join("golden").join(reference());
    let mut names: Vec<String> = std::fs::read_dir(&rf)
        .expect("golden reference")
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with("img_") && n.ends_with(".json"))
        .collect();
    names.sort();
    let mut out = Vec::new();
    for name in names {
        let gold = json(&rf.join(&name));
        let id = gold["id"].as_str().expect("id").to_owned();
        let ids: Vec<u32> = gold["input_ids"]
            .as_array()
            .expect("input_ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let pair = |v: &serde_json::Value| {
            (
                v[0].as_u64().expect("span") as usize,
                v[1].as_u64().expect("span") as usize,
            )
        };
        let questions = gold["questions"]
            .as_array()
            .expect("questions")
            .iter()
            .map(|q| ClefQuestion {
                qtype: q["type"].as_u64().expect("type") as u32,
                span: pair(&q["span"]),
                options: q["option_spans"]
                    .as_array()
                    .expect("option_spans")
                    .iter()
                    .map(pair)
                    .collect(),
            })
            .collect();
        // each image's first pad row: the runs of pad tokens, in order
        let mut rows = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            if t == IMAGE_PAD && (i == 0 || ids[i - 1] != IMAGE_PAD) {
                rows.push(i);
            }
        }
        let media = &gold["media"];
        let grids = media["image_grid_thw"].as_array().expect("grid");
        let sizes = media["images"].as_array().expect("images");
        assert_eq!(grids.len(), rows.len(), "{id}: an image per pad run");
        let images = grids
            .iter()
            .zip(sizes)
            .zip(&rows)
            .enumerate()
            .map(|(k, ((g, sz), &row))| {
                let (height, width) = (
                    sz["height"].as_u64().expect("height") as usize,
                    sz["width"].as_u64().expect("width") as usize,
                );
                // CLEF_RGB_DIR: another decode of the same files (the
                // endpoint's, kept by the runner's parity test) - then only
                // the answers are held to the reference, the pixels cannot be
                let rgb_dir =
                    std::env::var_os("CLEF_RGB_DIR").map_or_else(|| rf.clone(), PathBuf::from);
                let rgb = std::fs::read(rgb_dir.join(format!("{id}.rgb{k}.u8"))).expect("rgb");
                assert_eq!(rgb.len(), height * width * 3, "{id}: decoded size");
                ClefImage {
                    rgb,
                    width,
                    height,
                    resized: (
                        g[1].as_u64().expect("grid") as usize * 16,
                        g[2].as_u64().expect("grid") as usize * 16,
                    ),
                    row,
                }
            })
            .collect();
        out.push(Fixture {
            id,
            ids,
            questions,
            images,
        });
    }
    out
}

fn scores(dir: &Path, dtype: &str, id: &str) -> Vec<(Vec<f64>, Vec<f64>)> {
    let gold = json(&dir.join(format!("golden/{dtype}/{id}.json")));
    let floats = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .expect("scores")
            .iter()
            .map(|x| x.as_f64().expect("score"))
            .collect()
    };
    gold["questions"]
        .as_array()
        .expect("questions")
        .iter()
        .map(|q| (floats(&q["logits"]), floats(&q["probabilities"])))
        .collect()
}

#[test]
fn clef_images_match_the_f32_reference() {
    if !common::heavy() {
        return;
    }
    let Some(dir) = std::env::var_os("CLEF_DIR").map(PathBuf::from) else {
        common::missing("CLEF_DIR is not set");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let t0 = std::time::Instant::now();
    let gguf = std::env::var_os("CLEF_GGUF").map(PathBuf::from);
    let companion = std::env::var_os("CLEF_COMPANION").map(PathBuf::from);
    let mut clef = match &gguf {
        Some(g) => GpuClef::load_gguf(exec, g, companion.as_deref()),
        None => GpuClef::load(exec, &dir),
    }
    .expect("load");
    let rf = reference();
    assert!(
        clef.has_vision(),
        "the pack and checkpoint carry the image lane"
    );
    line(serde_json::json!({
        "load_s": t0.elapsed().as_secs_f64(),
        "weight_bytes": clef.weight_bytes(),
        "workspace_bytes": clef.workspace_bytes(),
    }));
    let all = fixtures(&dir);
    assert!(!all.is_empty(), "no img_* fixtures under golden/{rf}");
    let only = std::env::var("CLEF_ONLY").ok();
    let mut failures = Vec::new();
    let mut solo: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    for fx in all
        .iter()
        .filter(|f| only.as_ref().is_none_or(|o| &f.id == o))
    {
        let imgs: Vec<&ClefImage> = fx.images.iter().collect();
        // the processor: bit for bit
        let px = clef.image_pixels(&imgs).expect("pixels");
        let px_ref = read_f32(&dir.join(format!("golden/{rf}/{}.pixels.f32", fx.id)));
        assert_eq!(px.len(), px_ref.len(), "{}: pixel rows", fx.id);
        let differ = px
            .iter()
            .zip(&px_ref)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        // the tower
        let t = std::time::Instant::now();
        let (tower, merged) = clef.encode_images(&imgs).expect("tower");
        let tower_ms = t.elapsed().as_secs_f64() * 1e3;
        // the bar: the vendor's BF16 evaluation, or for a GGUF the BF16
        // lane's own class
        let rel = |name: &str, ours: &[f32]| {
            let f = read_f32(&dir.join(format!("golden/{rf}/{}.{name}.f32", fx.id)));
            if gguf.is_some() {
                return (rel_rms(ours, &f), 5e-4);
            }
            let b = read_f32(&dir.join(format!("golden/bf16/{}.{name}.f32", fx.id)));
            (rel_rms(ours, &f), rel_rms(&b, &f))
        };
        let (tower_err, tower_vendor) = rel("vtower", &tower);
        let (merged_err, merged_vendor) = rel("vision", &merged);
        // the model
        let req = ClefRequest {
            ids: &fx.ids,
            questions: &fx.questions,
            images: &fx.images,
        };
        let t = std::time::Instant::now();
        let out = clef.forward(std::slice::from_ref(&req)).expect("forward");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let hidden = clef.hidden_to_host(fx.ids.len()).expect("hidden");
        let hidden_err = rel_rms(
            &hidden,
            &read_f32(&dir.join(format!("golden/{rf}/{}.hidden.f32", fx.id))),
        );
        let f32_ref = scores(&dir, rf, &fx.id);
        let bf16_ref = if gguf.is_some() {
            f32_ref.clone()
        } else {
            scores(&dir, "bf16", &fx.id)
        };
        let (mut ol, mut op, mut vl, mut vp, mut flips, mut vflips) =
            (0f64, 0f64, 0f64, 0f64, 0, 0);
        for ((z, (rl, rp)), (bl, bp)) in out[0].iter().zip(&f32_ref).zip(&bf16_ref) {
            let p = softmax(z);
            for i in 0..z.len() {
                ol = ol.max((z[i] as f64 - rl[i]).abs());
                op = op.max((p[i] - rp[i]).abs());
                vl = vl.max((bl[i] - rl[i]).abs());
                vp = vp.max((bp[i] - rp[i]).abs());
            }
            flips += usize::from(argmax(&p) != argmax(rp));
            vflips += usize::from(argmax(bp) != argmax(rp));
        }
        line(serde_json::json!({
            "fixture": fx.id, "tokens": fx.ids.len(),
            "patches": px.len() / 1536, "pixels_differ": differ,
            "tower_ms": tower_ms, "forward_ms": ms,
            "tower_rel_rms": tower_err, "vendor_tower_rel_rms": tower_vendor,
            "merged_rel_rms": merged_err, "vendor_merged_rel_rms": merged_vendor,
            "hidden_rel_rms": hidden_err,
            "logit_max": ol, "prob_max": op, "flips": flips,
            "vendor_logit_max": vl, "vendor_prob_max": vp, "vendor_flips": vflips,
        }));
        if differ > 0 && std::env::var_os("CLEF_RGB_DIR").is_none() {
            failures.push(format!(
                "{}: {differ} pixel values differ from the processor's",
                fx.id
            ));
        }
        if std::env::var_os("CLEF_RGB_DIR").is_none()
            && (tower_err >= tower_vendor || merged_err >= merged_vendor)
        {
            failures.push(format!(
                "{}: tower {tower_err:.2e} / merged {merged_err:.2e} not under the vendor's \
                 {tower_vendor:.2e} / {merged_vendor:.2e}",
                fx.id
            ));
        }
        let (lbar, pbar) = if gguf.is_some() {
            (1e-3, 1e-4)
        } else {
            (vl.max(1e-6), vp.max(1e-7))
        };
        if flips > 0 || ol > lbar || op > pbar {
            failures.push(format!(
                "{}: logits {ol:.5} / probabilities {op:.6} / flips {flips} against the bars \
                 {lbar:.5} / {pbar:.6} (vendor BF16 {vl:.5} / {vp:.6} / {vflips})",
                fx.id
            ));
        }
        solo.push((fx.id.clone(), out[0].clone()));
    }

    // every image fixture packed into one pass: each request's logits must
    // be the bits it got alone
    if only.is_none() && solo.len() > 1 {
        let reqs: Vec<ClefRequest<'_>> = all
            .iter()
            .map(|fx| ClefRequest {
                ids: &fx.ids,
                questions: &fx.questions,
                images: &fx.images,
            })
            .collect();
        let rows: usize = all.iter().map(|f| f.ids.len()).sum();
        if rows <= clef.max_rows() {
            let packed = clef.forward(&reqs).expect("packed forward");
            for ((id, alone), got) in solo.iter().zip(&packed) {
                let same = alone
                    .iter()
                    .flatten()
                    .zip(got.iter().flatten())
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                line(serde_json::json!({"packed_logits": id, "bit_identical": same}));
                if !same {
                    failures.push(format!("{id}: packed logits differ from the request alone"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
