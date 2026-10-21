//! Nemotron 3.5 ASR on CUDA against the model's F64 evaluation.
//!
//! Fixtures (one directory each under `NASR_REFERENCE_DIR`, default
//! `target/nemotron-asr-reference/`) come from the Transformers oracle in
//! F64 on the GGUF the fixture's `meta.json` names (its Q8_0 dequantized
//! exactly, its own relative-position table and filterbank): what is left
//! between the two is arithmetic. A fixture holds the PCM, the planes of the
//! offline pass (the subsampler's output, the saved blocks', the prompt
//! projector's and the joint's encoder half) and the greedy tokens with
//! their frames.
//!
//! Gates, per fixture:
//!   - every plane of the offline pass holds F32-class error against F64
//!     (`PLANE_REL`, relative to the plane's largest magnitude);
//!   - the tokens and their frames are the oracle's;
//!   - three transport partitions of the same audio as a live stream (16001
//!     samples a push, 1103, and one chunk's audio) give the offline pass's
//!     tokens and projected frames to the bit;
//!   - every fixture of a checkpoint decoded at once - all the streams in
//!     the same passes, fed a chunk's audio at a time, interleaved - gives
//!     each fixture's own tokens and frames.
//!
//! `PADDOCK_HEAVY_TESTS=1` runs it.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use paddock_engine::gpu_model::nemotron_asr::{GpuNemotronAsr, Trace};
use paddock_models::nemotron_asr::{self as arch, HOP};

/// F32-class bound against the F64 oracle: the largest difference over the
/// plane's largest magnitude.
const PLANE_REL: f64 = 1e-4;

fn read(dir: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(name))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.join(name).display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// (largest difference, over the reference's largest magnitude)
fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "plane lengths");
    let mut max = 0f64;
    let mut scale = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        max = max.max((x as f64 - y as f64).abs());
        scale = scale.max((y as f64).abs());
    }
    (max, max / scale.max(1e-30))
}

fn stream(
    m: &mut GpuNemotronAsr,
    pcm: &[f32],
    lookahead: usize,
    prompt: u32,
    piece: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    m.trace = Some(Trace::default());
    let mut s = m.session(lookahead, prompt).expect("open a session");
    for part in pcm.chunks(piece) {
        m.push(&mut s, part).expect("push audio");
    }
    m.finish(&mut s).expect("finish the stream");
    let t = m.trace.take().expect("the trace");
    let proj = t.plane("proj").expect("the proj plane").to_vec();
    (s.tokens, s.frames, proj)
}

#[test]
fn nemotron_asr_matches_the_f64_model() {
    if !common::heavy() {
        return;
    }
    let root = std::env::var_os("NASR_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/nemotron-asr-reference")
        });
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .map(|r| r.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    dirs.retain(|d| d.join("meta.json").exists());
    dirs.retain(|d| std::env::var("NASR_ONLY").map_or(true, |o| d.ends_with(o)));
    dirs.sort();
    if dirs.is_empty() {
        common::missing(&format!(
            "no Nemotron ASR fixtures under {}",
            root.display()
        ));
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_nemotron_asr() {
        common::missing("this pack predates the Nemotron 3.5 ASR lane (slots 853-861)");
        return;
    }
    let mut models: HashMap<String, GpuNemotronAsr> = HashMap::new();
    let mut failures = Vec::new();
    // (name, pcm, lookahead, prompt, tokens, frames, weights) of each
    // fixture's own offline pass, for the batched check
    let mut singles = Vec::new();
    for dir in &dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        let weights = meta["weights"].as_str().expect("meta.weights").to_owned();
        let m = models.entry(weights.clone()).or_insert_with(|| {
            GpuNemotronAsr::load(exec.clone(), Path::new(&weights)).expect("load the checkpoint")
        });
        let mut fail = |what: String| {
            eprintln!("FAIL {name}: {what}");
            failures.push(format!("{name}: {what}"));
        };
        let lookahead = meta["lookahead"].as_u64().unwrap() as usize;
        let prompt = meta["prompt"].as_u64().unwrap() as u32;
        let layers: Vec<usize> = meta["layers_saved"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let want_tokens: Vec<u32> = meta["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want_frames: Vec<u32> = meta["token_frames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let pcm = read(dir, "pcm.f32");
        let frames = arch::encoder_frames(arch::mel_frames(pcm.len()));
        assert_eq!(frames as u64, meta["encoder_frames"].as_u64().unwrap());

        // the offline pass, every plane against F64
        m.trace = Some(Trace {
            layers: layers.clone(),
            ..Default::default()
        });
        let t0 = Instant::now();
        let s = m.transcribe(&pcm, lookahead, prompt).unwrap();
        let seconds = t0.elapsed().as_secs_f64();
        let trace = m.trace.take().unwrap();
        let mut planes = vec!["sub".to_owned()];
        planes.extend(layers.iter().map(|l| format!("layer_{l}")));
        planes.extend(["fused".to_owned(), "proj".to_owned()]);
        let mut report = serde_json::Map::new();
        for p in &planes {
            let (max, rel) = diff(trace.plane(p).unwrap(), &read(dir, &format!("{p}.f32")));
            report.insert(p.clone(), serde_json::json!({"max": max, "rel": rel}));
            if rel > PLANE_REL {
                fail(format!(
                    "{p}: max {max:.3e}, {rel:.3e} of the plane's scale"
                ));
            }
        }
        let first_diff = s.tokens.iter().zip(&want_tokens).position(|(a, b)| a != b);
        if s.tokens != want_tokens || s.frames != want_frames {
            fail(format!(
                "tokens differ: {} vs {} (first at {first_diff:?}, oracle's smallest margin {})",
                s.tokens.len(),
                want_tokens.len(),
                meta["min_margin"]
            ));
        }
        let offline_proj = trace.plane("proj").unwrap().to_vec();

        // live: three transport partitions, the offline pass's bits
        let chunk_audio = arch::chunk_mel_frames(lookahead, false) * HOP;
        for piece in [16_001, 1_103, chunk_audio] {
            let (tokens, tframes, proj) = stream(m, &pcm, lookahead, prompt, piece);
            if tokens != s.tokens || tframes != s.frames {
                fail(format!(
                    "stream {piece}: tokens differ from the offline pass"
                ));
            }
            let same = proj.len() == offline_proj.len()
                && proj
                    .iter()
                    .zip(&offline_proj)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            if !same {
                let (max, _) = if proj.len() == offline_proj.len() {
                    diff(&proj, &offline_proj)
                } else {
                    (f64::NAN, f64::NAN)
                };
                fail(format!(
                    "stream {piece}: projected frames not bit-identical (max {max:.3e}, {} vs {} values)",
                    proj.len(),
                    offline_proj.len()
                ));
            }
        }
        let text = arch::detokenize(m.vocab(), &s.tokens, false);
        singles.push((
            name.clone(),
            pcm.clone(),
            lookahead,
            prompt,
            s.tokens.clone(),
            s.frames.clone(),
            weights.clone(),
        ));
        println!(
            "{}",
            serde_json::json!({
                "fixture": name,
                "frames": frames,
                "tokens": s.tokens.len(),
                "seconds": seconds,
                "planes": report,
                "text": text,
            })
        );
    }
    // all fixtures of a checkpoint as concurrent streams: shared passes,
    // interleaved arrivals, each stream's own tokens
    let mut by_weights: HashMap<String, Vec<(String, Vec<f32>, usize, u32, Vec<u32>, Vec<u32>)>> =
        HashMap::new();
    for (name, pcm, lookahead, prompt, tokens, frames, weights) in singles {
        by_weights
            .entry(weights)
            .or_default()
            .push((name, pcm, lookahead, prompt, tokens, frames));
    }
    for (weights, set) in by_weights {
        let m = models.get_mut(&weights).expect("loaded above");
        let mut sessions: Vec<_> = set
            .iter()
            .map(|(_, _, lookahead, prompt, _, _)| {
                m.session(*lookahead, *prompt).expect("open a session")
            })
            .collect();
        let mut at = vec![0usize; set.len()];
        loop {
            let mut fed = false;
            for (i, (_, pcm, lookahead, _, _, _)) in set.iter().enumerate() {
                let step = arch::chunk_mel_frames(*lookahead, false) * HOP;
                if at[i] < pcm.len() {
                    let end = (at[i] + step).min(pcm.len());
                    m.feed(&mut sessions[i], &pcm[at[i]..end]).expect("feed");
                    at[i] = end;
                    if end == pcm.len() {
                        m.close(&mut sessions[i]);
                    }
                    fed = true;
                }
            }
            let mut refs: Vec<_> = sessions.iter_mut().collect();
            while m.run(&mut refs).expect("a pass") > 0 {}
            if !fed {
                break;
            }
        }
        for (s, (name, _, _, _, tokens, frames)) in sessions.iter().zip(&set) {
            if &s.tokens != tokens || &s.frames != frames {
                eprintln!("FAIL {name}: batched passes differ from the stream alone");
                failures.push(format!(
                    "{name}: batched passes differ from the stream alone"
                ));
            }
        }
        println!(
            "{}",
            serde_json::json!({"batched": set.len(), "weights": weights, "identical": failures.is_empty()})
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
