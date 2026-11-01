//! Nemotron 3.5 Super VL's picture path vs its reference dump (the picture
//! oracle script beside the text one): per picture the processor's tile and
//! patch rows (the checkpoint's own float antialiased bicubic, eager torch)
//! and the projector's output rows (an f32 re-run of the tower + projector,
//! and vLLM's bf16 serving rows beside it); per picture prompt the expanded
//! row count and teacher-forced top-1 over vLLM's greedy answer; and the
//! batch slot's picture prefill against the serial lane's.
//!
//! One load for all three (75 GB of weights): `PADDOCK_HEAVY_TESTS=1`. The
//! dump sits beside the checkpoint as `super-vl-oracle/pictures/` or at
//! `SUPER_VL_PICTURES`. Every check is collected before the asserts, so one
//! failing stage still prints the others.

mod common;

use std::path::{Path, PathBuf};

use half::f16;
use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;
use paddock_engine::service::MmChunk;

const CKPT_ENV: &str = "SUPER_VL_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4";
const IMG_START: u32 = 19;
const IMG_END: u32 = 20;

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .expect("id array")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect()
}

fn argmax(l: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

fn f32s(path: &Path) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (words, _) = raw.as_chunks::<4>();
    words.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "row counts differ");
    let (mut num, mut den) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

/// The case's pictures as the runner hands them over: decoded RGB + size.
fn pictures(dir: &Path, case: &serde_json::Value) -> Vec<(String, Vec<u8>, usize, usize)> {
    case["pictures"]
        .as_array()
        .expect("pictures")
        .iter()
        .map(|p| {
            let file = p["file"].as_str().expect("file");
            let w = p["w"].as_u64().expect("w") as usize;
            let h = p["h"].as_u64().expect("h") as usize;
            let rgb = std::fs::read(dir.join(file)).expect("rgb");
            assert_eq!(rgb.len(), w * h * 3, "{file}");
            (file.trim_end_matches(".rgb").to_owned(), rgb, w, h)
        })
        .collect()
}

/// vLLM's expanded prompt back into the runner's chunks: each
/// `<img><image>xN</img>` run becomes the next picture.
fn chunks_of(prompt: &[u32], pics: &[(String, Vec<u8>, usize, usize)]) -> Vec<MmChunk> {
    let mut out = Vec::new();
    let mut text = Vec::new();
    let mut next = 0usize;
    let mut i = 0usize;
    while i < prompt.len() {
        if prompt[i] == IMG_START {
            let end = prompt[i..]
                .iter()
                .position(|&t| t == IMG_END)
                .expect("an </img> after every <img>")
                + i;
            if !text.is_empty() {
                out.push(MmChunk::Text(std::mem::take(&mut text)));
            }
            let (_, rgb, w, h) = &pics[next];
            out.push(MmChunk::Image {
                rgb: rgb.clone(),
                w: *w,
                h: *h,
            });
            next += 1;
            i = end + 1;
        } else {
            text.push(prompt[i]);
            i += 1;
        }
    }
    if !text.is_empty() {
        out.push(MmChunk::Text(text));
    }
    assert_eq!(next, pics.len(), "picture runs vs pictures");
    out
}

#[test]
fn super_vl_pictures_match_the_reference() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_mamba2() || !exec.has_nvf4_moe() || !exec.has_nemotron_vl() {
        common::missing("pack lacks the Super kernels (mamba2, nvf4 moe, slots 904-907)");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let pdir = std::env::var("SUPER_VL_PICTURES").map_or_else(
        |_| {
            dir.parent()
                .expect("models root")
                .join("super-vl-oracle/pictures")
        },
        PathBuf::from,
    );
    let Ok(raw) = std::fs::read(pdir.join("pictures.json")) else {
        common::missing(&format!("no picture dump in {}", pdir.display()));
        return;
    };
    let oracle: serde_json::Value = serde_json::from_slice(&raw).expect("pictures json");
    let t0 = std::time::Instant::now();
    let mut model = GpuNemotron::load_dir(exec, &dir, 8192).expect("load");
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    assert!(
        model.has_pictures(),
        "the omni checkpoint loaded without its tower"
    );
    let cases = oracle["cases"].as_array().expect("cases");
    let mut failures: Vec<String> = Vec::new();

    // ---- 1. per picture: tile, patch rows, projector rows -------------------
    for case in cases {
        let name = case["case"].as_str().expect("case");
        let pics = pictures(&pdir, case);
        let mut ours_rows = Vec::new();
        for (k, (file, rgb, w, h)) in pics.iter().enumerate() {
            let t = std::time::Instant::now();
            let probe = model.picture_probe(rgb, *w, *h, pics.len()).expect("probe");
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let want = &case["hf_imgs_sizes"][k];
            let want_tile = (
                want[1].as_u64().expect("w") as usize / 16,
                want[0].as_u64().expect("h") as usize / 16,
            );
            if probe.tile != want_tile {
                failures.push(format!(
                    "{name}/{file}: tile {:?} vs {want_tile:?}",
                    probe.tile
                ));
            }
            // the processor's f32 pixels rounded the way the patchify rounds.
            // torch's compiled CPU resize may contract its multiply-adds (FMA
            // on aarch64), so its own values move by an f32 ulp with the
            // platform: a pixel on an f16 rounding edge may land one f16 ulp
            // over - never further
            let px = f32s(&pdir.join(format!("{name}.{file}.px.f32")));
            let (mut off, mut far, mut worst) = (0usize, 0usize, 0f32);
            if px.len() == probe.patches.len() {
                for (&a, &b) in probe.patches.iter().zip(&px) {
                    let want = f16::from_f32(b);
                    if a != want {
                        off += 1;
                        let d = (a.to_f32() - want.to_f32()).abs();
                        worst = worst.max(d);
                        // one f16 ulp at the value's magnitude - or, near zero
                        // where that ulp shrinks below it, the f32 deviation an
                        // ulp of the 0..255 scale becomes after normalizing
                        let mag = f16::from_bits(want.to_bits() & 0x7fff);
                        let ulp = f16::from_bits(mag.to_bits() + 1).to_f32() - mag.to_f32();
                        far += usize::from(d > ulp.max(8e-6) * 1.01);
                    }
                }
            } else {
                failures.push(format!(
                    "{name}/{file}: {} patch values vs the processor's {}",
                    probe.patches.len(),
                    px.len()
                ));
            }
            println!(
                "{name}/{file}: {w}x{h} -> tile {:?}, {} rows, encode {ms:.0} ms; patch f16 \
                 values off the processor's {off}/{} ({far} past one ulp, worst {worst:.2e})",
                probe.tile,
                probe.rows.len() / 4096,
                px.len()
            );
            // a resize bug moves most values, and by far more than an ulp
            if far > 0 || off * 200 > px.len().max(1) {
                failures.push(format!(
                    "{name}/{file}: {off} patch values off the processor, {far} past one ulp"
                ));
            }
            ours_rows.extend_from_slice(&probe.rows);
        }
        let r32 = pdir.join(format!("{name}.rows32.f32"));
        let vb = f32s(&pdir.join(format!("{name}.rows.f32")));
        let vs_vllm = rel_l2(&ours_rows, &vb);
        if r32.exists() {
            let f = f32s(&r32);
            let ours_err = rel_l2(&ours_rows, &f);
            let vllm_err = rel_l2(&vb, &f);
            println!(
                "{name}: projector rows rel L2 vs the f32 reference {ours_err:.4} (vLLM bf16 \
                 {vllm_err:.4}); vs vLLM {vs_vllm:.4}"
            );
            // our tower is f16 with f32 residuals, vLLM's bf16 throughout -
            // ours should land no further from the math than vLLM's does
            if ours_err > vllm_err.max(0.01) {
                failures.push(format!(
                    "{name}: rows {ours_err:.4} from the f32 reference, vLLM {vllm_err:.4}"
                ));
            }
        } else {
            println!("{name}: projector rows rel L2 vs vLLM {vs_vllm:.4} (no f32 reference)");
            if vs_vllm > 0.05 {
                failures.push(format!("{name}: rows {vs_vllm:.4} from vLLM's"));
            }
        }
    }

    // ---- 2. picture prompts: rows, then teacher-forced over the answer ------
    let (mut agree_all, mut total_all) = (0usize, 0usize);
    let mut first_ok = 0usize;
    for case in cases {
        let name = case["case"].as_str().expect("case");
        let pics = pictures(&pdir, case);
        let prompt = ids(&case["prompt_ids"]);
        let greedy = ids(&case["greedy_ids"]);
        let top5 = case["top5"].as_array().expect("top5");
        let chunks = chunks_of(&prompt, &pics);
        model.reset();
        let t = std::time::Instant::now();
        let (mut logits, n) = model
            .forward_multimodal(&chunks)
            .expect("picture prefill")
            .expect("the omni lane takes pictures");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if n != prompt.len() {
            failures.push(format!("{name}: {n} rows vs vLLM's {}", prompt.len()));
        }
        let (mut agree, mut gaps) = (0usize, Vec::new());
        for (i, &want) in greedy.iter().enumerate() {
            let got = argmax(&logits);
            let ref5 = top5[i].as_array().expect("top5 row");
            if got == want {
                agree += 1;
                first_ok += usize::from(i == 0);
            } else {
                let best = ref5[0]["logprob"].as_f64().expect("logprob");
                let gap = ref5
                    .iter()
                    .find(|t| t["id"].as_u64() == Some(got as u64))
                    .and_then(|t| t["logprob"].as_f64())
                    .map(|lp| best - lp);
                // the reference's logits are bf16 (its logprobs come in 0.25 -
                // 0.5 nat steps at this magnitude), so it cannot order two
                // picks closer than that: a tie
                if i == 0 && gap.is_some_and(|g| g <= 0.5 + 1e-3) {
                    first_ok += 1;
                }
                gaps.push(gap);
            }
            if i + 1 < greedy.len() {
                logits = model.forward(want).expect("teacher-forced step");
            }
        }
        let near = gaps.iter().filter(|g| g.is_some_and(|g| g <= 0.5)).count();
        println!(
            "{name}: {n} rows, prefill {ms:.0} ms; teacher-forced {agree}/{} top-1, {near} of {} \
             misses within 0.5 nats (reference {:?})",
            greedy.len(),
            gaps.len(),
            case["text"].as_str().unwrap_or("")
        );
        agree_all += agree;
        total_all += greedy.len();
    }
    let rate = agree_all as f64 / total_all.max(1) as f64;
    println!(
        "picture prompts: teacher-forced {agree_all}/{total_all} ({:.2}%), first token {first_ok}/{}",
        100.0 * rate,
        cases.len()
    );
    if rate < 0.90 {
        failures.push(format!(
            "teacher-forced top-1 {:.2}% under 90%",
            100.0 * rate
        ));
    }
    if first_ok != cases.len() {
        failures.push(format!(
            "first token {first_ok}/{} (tie-aware)",
            cases.len()
        ));
    }

    // ---- 3. the batch slot's picture prefill vs the serial lane's -----------
    if model.enable_batch(2).is_ok() && model.supports_mm_slots() {
        let case = &cases[0];
        let name = case["case"].as_str().expect("case");
        let pics = pictures(&pdir, case);
        let chunks = chunks_of(&ids(&case["prompt_ids"]), &pics);
        model.reset();
        let (serial, n_s) = model
            .forward_multimodal(&chunks)
            .expect("serial")
            .expect("pictures");
        let (slot, n_b) = model
            .forward_prefill_multimodal(1, &chunks)
            .expect("slot picture prefill");
        // the batch lane's own kernel class (paged attention tile, W8A8
        // dynamic activations) changes summation order - the family's batch
        // gates hold it to the same pick and a mean |delta| / rms band
        let dmax = serial
            .iter()
            .zip(&slot)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rms = (serial.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
            / serial.len().max(1) as f64)
            .sqrt();
        let mean = serial
            .iter()
            .zip(&slot)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / serial.len().max(1) as f64;
        println!(
            "{name}: slot prefill {n_b} rows vs serial {n_s}; argmax {} vs {}, mean |d| {mean:.4} \
             vs rms {rms:.3} ({:.2}%), max {dmax:.3}",
            argmax(&slot),
            argmax(&serial),
            100.0 * mean / rms.max(1e-3)
        );
        if n_b != n_s || argmax(&slot) != argmax(&serial) || mean / rms.max(1e-3) >= 0.10 {
            failures.push(format!(
                "{name}: slot prefill parts from serial (mean |d| {mean:.4}, rms {rms:.3})"
            ));
        }
    } else {
        failures.push("batch slots unavailable for the picture prefill".into());
    }

    assert!(failures.is_empty(), "picture gates: {failures:#?}");
}
