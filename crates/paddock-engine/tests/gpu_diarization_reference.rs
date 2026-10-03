//! Nemotron 3 Diarization on CUDA against the model's F64 evaluation.
//!
//! Fixtures (one directory each under `DIAR_REFERENCE_DIR`, default
//! `target/diarization-reference/`) come from the Transformers oracle in
//! F64 on the same weights the fixture's `meta.json` names (MLX BF16
//! directory or GGUF Q8_0 file), driven by a port of the shared streaming
//! windows: what is left between the two is arithmetic. A fixture holds the
//! recording's PCM, one forward window (features, stacked-frame embeddings,
//! probabilities - the Metal oracle's layout) and, with `--stream`, every
//! preset's probabilities over the whole recording.
//!
//! Gates, per fixture:
//!   - the host frontend (F64) reproduces the oracle's log-mels;
//!   - the GPU frontend + projection, the projection alone and the encoder
//!     window each hold F32-class error against F64 (`FORWARD_MAX`);
//!   - every preset's stream holds the same probability error and decision
//!     agreement over the whole recording, and three transport partitions
//!     (16001, 1103 and the competing-file quantum) give bit-identical outputs.
//!
//! `PADDOCK_HEAVY_TESTS=1` runs it. Each preset prints one JSON line a
//! partition with its segments and times - the labelled DER/JER scorer's
//! input.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use paddock_engine::diarization::{AudioWindow, Backend, Stream};
use paddock_engine::gpu_model::diarization::GpuDiarization;
use paddock_models::diarization::{HOP, Preset, SAMPLE_RATE, STACK, segments};

/// F32-class bounds against the F64 oracle: largest probability difference
/// of a forward window and of a stream, and the share of speaker/frame
/// decisions (threshold 0.5) a stream may flip. A stream feeds its own
/// outputs back through the speaker cache, so it is allowed the decisions a
/// near-threshold frame can flip, never a drift.
const FORWARD_MAX: f32 = 2e-4;
const STREAM_RMSE: f64 = 1e-3;
const STREAM_FLIPS: f64 = 1e-4;

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

struct Diff {
    max: f32,
    rmse: f64,
    flips: usize,
}

fn diff(a: &[f32], b: &[f32]) -> Diff {
    assert_eq!(a.len(), b.len(), "shape");
    assert!(a.iter().chain(b).all(|v| v.is_finite()), "nonfinite");
    Diff {
        max: a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0., f32::max),
        rmse: (a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(x - y).powi(2))
            .sum::<f64>()
            / a.len().max(1) as f64)
            .sqrt(),
        flips: a
            .iter()
            .zip(b)
            .filter(|(x, y)| (**x > 0.5) != (**y > 0.5))
            .count(),
    }
}

fn line(v: serde_json::Value) {
    use std::io::Write;
    let _ = writeln!(std::io::stdout(), "{v}");
}

#[test]
fn diarization_matches_the_f64_model() {
    if !common::heavy() {
        return;
    }
    let root = std::env::var_os("DIAR_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/diarization-reference")
        });
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .map(|r| r.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    dirs.retain(|d| d.join("meta.json").exists());
    dirs.retain(|d| std::env::var("DIAR_ONLY").map_or(true, |o| d.ends_with(o)));
    dirs.sort();
    if dirs.is_empty() {
        common::missing(&format!("no diarization fixtures under {}", root.display()));
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_diarization() {
        common::missing("this pack predates the diarization lane (slots 718-723)");
        return;
    }
    let mut models: HashMap<String, GpuDiarization> = HashMap::new();
    let mut failures = Vec::new();
    for dir in &dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
        let weights = meta["weights"].as_str().expect("meta.weights").to_owned();
        let m = models.entry(weights.clone()).or_insert_with(|| {
            GpuDiarization::load(exec.clone(), Path::new(&weights)).expect("load the checkpoint")
        });
        let mut fail = |what: String| {
            eprintln!("FAIL {name}: {what}");
            failures.push(format!("{name}: {what}"));
        };
        let frames = meta["frames"].as_u64().unwrap() as usize;
        let padded = meta["padded"].as_u64().unwrap() as usize;
        let valid = frames.div_ceil(8);
        let pcm = read(dir, "pcm.f32");
        let fe = m.frontend().unwrap();

        // frontends: the host F64 one, then the GPU one through the projection
        let mel = diff(
            &fe.features(&pcm, 0, pcm.len(), 0, padded),
            &read(dir, "features.f32"),
        );
        let embedded = read(dir, "embedded.f32");
        // the window reads samples up to count * 160 + 256; a window is at
        // most 64 seconds of PCM
        let count = padded.min(3040);
        let audio = &pcm[..pcm.len().min(count * 160 + 512)];
        let gpu_emb = m
            .encode_audio(AudioWindow {
                audio,
                offset: 0,
                total: audio.len(),
                start: 0,
                count,
            })
            .unwrap();
        let frontend = diff(&gpu_emb, &embedded[..gpu_emb.len()]);
        let pre = diff(
            &m.pre_encode(&read(dir, "features.f32")).unwrap(),
            &embedded,
        );
        let start = Instant::now();
        let (p, gpu) = m.predict(&embedded, valid).unwrap();
        let wall = start.elapsed().as_secs_f64();
        let flat: Vec<f32> = p.iter().flatten().copied().collect();
        let forward = diff(&flat, &read(dir, "probabilities.f32"));
        line(
            serde_json::json!({"fixture":name,"rows":embedded.len()/512,"valid":valid,
            "mel_max":mel.max,"frontend_max":frontend.max,"frontend_rmse":frontend.rmse,
            "pre_encode_max":pre.max,"forward_max":forward.max,"forward_rmse":forward.rmse,
            "forward_flips":forward.flips,"wall_seconds":wall,"gpu_seconds":gpu,
            "weight_bytes":m.weight_bytes(),"workspace_bytes":Backend::workspace_bytes(m)}),
        );
        if mel.max >= 1e-3 {
            fail(format!("host frontend drift {}", mel.max));
        }
        if forward.max >= FORWARD_MAX {
            fail(format!("forward max {}", forward.max));
        }

        for (preset_name, preset) in [
            ("offline", Preset::Offline),
            ("low", Preset::Low),
            ("very_low", Preset::VeryLow),
            ("ultra_low", Preset::UltraLow),
        ] {
            let path = dir.join(format!("{preset_name}.f32"));
            if !path.exists()
                || std::env::var("DIAR_PRESETS")
                    .is_ok_and(|p| !p.split(',').any(|p| p == preset_name))
            {
                continue;
            }
            let expected = read(dir, &format!("{preset_name}.f32"));
            let mut previous: Option<Vec<u32>> = None;
            // The shared service shortens file turns when a second session
            // is reserved. Qualify that arrival pattern on the real CUDA
            // backend too, not only a fake backend or the Metal equivalent.
            let file_quantum = (preset.geometry().chunk * STACK * HOP).min(SAMPLE_RATE);
            for chunk in [16001, 1103, file_quantum] {
                let start = Instant::now();
                let mut s = Stream::new(preset);
                let mut probs = Vec::new();
                for a in pcm.chunks(chunk) {
                    probs.extend(s.feed(m, &fe, a, false).unwrap());
                }
                probs.extend(s.feed(m, &fe, &[], true).unwrap());
                let wall = start.elapsed().as_secs_f64();
                let flat: Vec<f32> = probs.iter().flatten().copied().collect();
                let d = diff(&flat, &expected);
                line(
                    serde_json::json!({"fixture":name,"preset":preset_name,"chunk":chunk,
                    "max_abs":d.max,"rmse":d.rmse,"flips":d.flips,"decisions":flat.len(),
                    "wall_seconds":wall,"gpu_seconds":s.gpu_seconds,"frames":s.frames,
                    "segments":segments(&probs,0,0.5)}),
                );
                if d.rmse >= STREAM_RMSE || d.flips as f64 > STREAM_FLIPS * flat.len() as f64 {
                    fail(format!(
                        "{preset_name}/{chunk}: rmse {} flips {}",
                        d.rmse, d.flips
                    ));
                }
                let bits: Vec<_> = flat.iter().map(|v| v.to_bits()).collect();
                if let Some(p) = &previous
                    && *p != bits
                {
                    fail(format!(
                        "{preset_name}: transport partition changed the output"
                    ));
                }
                previous = Some(bits);
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
