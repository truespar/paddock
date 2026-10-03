//! Clef's engine gate against the reference's own evaluation
//! (the oracle: Cloudflare's `joint_schema_model.py`
//! under Transformers 5.10.2, F32 with TF32 off, written to
//! `<CLEF_DIR>/golden/f32`; the vendor's served BF16 class beside it in
//! `golden/bf16`):
//!
//!   - the backbone's final hidden state of every fixture is closer to the
//!     F32 reference than the vendor's BF16 is (relative RMS);
//!   - so are the head's logits and probabilities (largest difference), with
//!     no answer flipped;
//!   - a pass of several requests packed back to back gives every request
//!     the bits it gets alone, hidden rows and logits.
//!
//! `PADDOCK_HEAVY_TESTS=1` and `CLEF_DIR` run it. Each fixture prints one
//! JSON line.
//!
//! With `CLEF_GGUF` (and `CLEF_COMPANION` for the vision tower) it gates the
//! GGUF lane instead, against the same-weights reference in
//! `<CLEF_DIR>/golden/gguf-f32` (the oracle's `--gguf`: the release with
//! that GGUF's Q8_0 weights). There is no vendor evaluation of those
//! weights, so the bars are the F32 class's budget, one for every Clef -
//! hidden rows within 5e-4 relative RMS, logits 5e-4, probabilities 1e-4, no
//! flips (the 27B's 64 layers spend up to half of it; the vendor's BF16 is
//! ~100x past it) - and, where `golden/llamacpp/<id>.json` holds the newest
//! llama.cpp's answers on the same file, probabilities no further from the
//! reference than llama.cpp's.

mod common;

use std::path::{Path, PathBuf};

use paddock_engine::gpu_model::clef::{ClefPass, ClefQuestion, ClefRequest, GpuClef};

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

struct Fixture {
    id: String,
    ids: Vec<u32>,
    questions: Vec<ClefQuestion>,
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

/// Per question: the reference's logits and probabilities.
fn golden_scores(dir: &Path, dtype: &str, id: &str) -> Vec<(Vec<f64>, Vec<f64>)> {
    try_golden_scores(dir, dtype, id).expect("golden")
}

fn try_golden_scores(dir: &Path, dtype: &str, id: &str) -> Option<Vec<(Vec<f64>, Vec<f64>)>> {
    let gold: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(format!("golden/{dtype}/{id}.json"))).ok()?)
            .expect("golden parses");
    let floats = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .expect("golden json shape")
            .iter()
            .map(|x| x.as_f64().expect("golden json shape"))
            .collect()
    };
    Some(
        gold["questions"]
            .as_array()
            .expect("golden json shape")
            .iter()
            .map(|q| (floats(&q["logits"]), floats(&q["probabilities"])))
            .collect(),
    )
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

fn fixtures(dir: &Path) -> Vec<Fixture> {
    let mut out = Vec::new();
    let list = std::fs::read_to_string(dir.join("golden/fixtures.jsonl")).expect("fixtures.jsonl");
    for l in list.lines().filter(|l| !l.trim().is_empty()) {
        let req: serde_json::Value = serde_json::from_str(l).expect("fixture");
        let id = req["id"].as_str().expect("id").to_owned();
        let gold: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join(format!("golden/{}/{id}.json", reference()))).expect("golden"),
        )
        .expect("golden parses");
        let ids = gold["input_ids"]
            .as_array()
            .expect("golden json shape")
            .iter()
            .map(|v| v.as_u64().expect("golden json shape") as u32)
            .collect();
        let pair = |v: &serde_json::Value| {
            (
                v[0].as_u64().expect("golden json shape") as usize,
                v[1].as_u64().expect("golden json shape") as usize,
            )
        };
        let questions = gold["questions"]
            .as_array()
            .expect("golden json shape")
            .iter()
            .map(|q| ClefQuestion {
                qtype: q["type"].as_u64().expect("golden json shape") as u32,
                span: pair(&q["span"]),
                options: q["option_spans"]
                    .as_array()
                    .expect("golden json shape")
                    .iter()
                    .map(pair)
                    .collect(),
            })
            .collect();
        out.push(Fixture { id, ids, questions });
    }
    out
}

#[test]
fn clef_backbone_matches_the_f32_reference() {
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
    line(serde_json::json!({
        "load_s": t0.elapsed().as_secs_f64(),
        "weight_bytes": clef.weight_bytes(),
        "workspace_bytes": clef.workspace_bytes(),
    }));
    let h = clef.cfg.hidden;
    let all = fixtures(&dir);
    let only = std::env::var("CLEF_ONLY").ok();
    let mut failures = Vec::new();
    let mut alone: Vec<(String, Vec<f32>)> = Vec::new();
    for fx in all
        .iter()
        .filter(|f| only.as_ref().is_none_or(|o| &f.id == o))
    {
        let n = fx.ids.len();
        let t = std::time::Instant::now();
        clef.backbone(&ClefPass {
            ids: &fx.ids,
            runs: &[(0, n)],
            images: &[],
        })
        .expect("backbone");
        let ours = clef.hidden_to_host(n).expect("hidden");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let f32_ref = read_f32(&dir.join(format!("golden/{rf}/{}.hidden.f32", fx.id)));
        assert_eq!(f32_ref.len(), n * h, "{}: reference shape", fx.id);
        // the bar: the vendor's BF16 evaluation, or for a GGUF the BF16
        // lane's own class
        let ours_err = rel_rms(&ours, &f32_ref);
        let vendor_err = if gguf.is_some() {
            5e-4
        } else {
            rel_rms(
                &read_f32(&dir.join(format!("golden/bf16/{}.hidden.f32", fx.id))),
                &f32_ref,
            )
        };
        let finite = ours.iter().all(|v| v.is_finite());
        line(serde_json::json!({
            "fixture": fx.id, "tokens": n, "ms": ms,
            "hidden_rel_rms": ours_err, "vendor_bf16_rel_rms": vendor_err,
        }));
        if !finite || ours_err >= vendor_err {
            failures.push(format!(
                "{}: hidden rel-RMS {ours_err:.4} not under the vendor BF16's {vendor_err:.4}",
                fx.id
            ));
        }
        if n <= 2048 {
            alone.push((fx.id.clone(), ours));
        }
    }

    // the short fixtures packed into one pass: every request's rows must be
    // the bits it got alone
    if alone.len() > 1 {
        let mut ids = Vec::new();
        let mut runs = Vec::new();
        for (id, _) in &alone {
            let fx = all.iter().find(|f| &f.id == id).expect("golden json shape");
            runs.push((ids.len(), fx.ids.len()));
            ids.extend(&fx.ids);
        }
        clef.backbone(&ClefPass {
            ids: &ids,
            runs: &runs,
            images: &[],
        })
        .expect("packed backbone");
        let packed = clef.hidden_to_host(ids.len()).expect("hidden");
        for ((id, solo), &(s, n)) in alone.iter().zip(&runs) {
            let same = packed[s * h..(s + n) * h]
                .iter()
                .zip(solo)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            line(serde_json::json!({"packed": id, "bit_identical": same}));
            if !same {
                failures.push(format!("{id}: packed rows differ from the request alone"));
            }
        }
    }

    // the whole model: logits and probabilities against the reference
    let mut solo_logits: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    for fx in all
        .iter()
        .filter(|f| only.as_ref().is_none_or(|o| &f.id == o))
    {
        let t = std::time::Instant::now();
        let out = clef
            .forward(&[ClefRequest {
                ids: &fx.ids,
                questions: &fx.questions,
                images: &[],
            }])
            .expect("forward");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let ours = &out[0];
        let f32_ref = golden_scores(&dir, rf, &fx.id);
        // the head alone, on the reference's own F32 hidden rows
        let ref_hidden = read_f32(&dir.join(format!("golden/{rf}/{}.hidden.f32", fx.id)));
        let head_only = clef
            .head_on_hidden(
                &[ClefRequest {
                    ids: &fx.ids,
                    questions: &fx.questions,
                    images: &[],
                }],
                &ref_hidden,
            )
            .expect("head on the reference hidden");
        let mut head_l = 0f64;
        for (z, (rl, _)) in head_only[0].iter().zip(&f32_ref) {
            for i in 0..z.len() {
                head_l = head_l.max((z[i] as f64 - rl[i]).abs());
            }
        }
        line(serde_json::json!({"fixture": fx.id, "head_only_logit_max": head_l}));
        if head_l > 1e-3 {
            failures.push(format!(
                "{}: the head alone is {head_l:.6} off the reference",
                fx.id
            ));
        }
        // the rival on the same weights: the vendor's BF16 evaluation, or
        // for a GGUF the newest llama.cpp's answers when they are there
        let rival = if gguf.is_some() { "llamacpp" } else { "bf16" };
        let bf16_ref = match try_golden_scores(&dir, rival, &fx.id) {
            Some(r) => r,
            None if gguf.is_some() => f32_ref.clone(),
            None => panic!("{}: golden/{rival} missing", fx.id),
        };
        let (mut ol, mut op, mut vl, mut vp, mut flips, mut vflips) =
            (0f64, 0f64, 0f64, 0f64, 0, 0);
        for ((z, (rl, rp)), (bl, bp)) in ours.iter().zip(&f32_ref).zip(&bf16_ref) {
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
        if gguf.is_some() {
            // llama.cpp's "logits" are log-probabilities: not comparable
            vl = f64::NAN;
        }
        line(serde_json::json!({
            "fixture": fx.id, "forward_ms": ms,
            "logit_max": ol, "prob_max": op, "flips": flips,
            "rival": rival, "rival_logit_max": vl, "rival_prob_max": vp, "rival_flips": vflips,
        }));
        // a GGUF's bars: the BF16 lane's class, and probabilities no worse
        // than llama.cpp's where its answers are there (it reports no
        // logits; a missing rival compares as zero)
        let (lbar, pbar) = if gguf.is_some() {
            (5e-4, if vp > 0.0 { vp.min(1e-4) } else { 1e-4 })
        } else {
            (vl.max(1e-6), vp.max(1e-7))
        };
        if flips > 0 || ol > lbar || op > pbar {
            failures.push(format!(
                "{}: logits {ol:.5} / probabilities {op:.6} / flips {flips} against the \
                 vendor BF16's {vl:.5} / {vp:.6} / {vflips}",
                fx.id
            ));
        }
        if fx.ids.len() <= 2048 {
            solo_logits.push((fx.id.clone(), ours.clone()));
        }
    }
    if solo_logits.len() > 1 {
        let reqs: Vec<ClefRequest<'_>> = solo_logits
            .iter()
            .map(|(id, _)| {
                let fx = all.iter().find(|f| &f.id == id).expect("golden json shape");
                ClefRequest {
                    ids: &fx.ids,
                    questions: &fx.questions,
                    images: &[],
                }
            })
            .collect();
        let packed = clef.forward(&reqs).expect("packed forward");
        for ((id, solo), got) in solo_logits.iter().zip(&packed) {
            let same = solo
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
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
