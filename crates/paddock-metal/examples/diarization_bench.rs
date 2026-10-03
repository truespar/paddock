//! Counter-free, complete-stream benchmark. Compare digests between binaries
//! as well as same-checkpoint oracle error; never time diagnostic snapshots.
#![allow(clippy::unwrap_used)]
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use paddock_engine::diarization::{Backend, Stream};
    use paddock_models::diarization::Preset;
    use std::{path::Path, time::Instant};

    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: diarization_bench MODEL FIXTURE_DIR [REPEATS=5]".into());
    }
    let repeats: usize = args.get(3).map_or(Ok(5), |s| s.parse())?;
    if !(1..=100).contains(&repeats) {
        return Err("repeats must be in 1..=100".into());
    }
    let root = Path::new(&args[2]);
    let read = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(root.join(name))?;
        if !bytes.len().is_multiple_of(4) {
            return Err("invalid F32 fixture length".into());
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    };
    let pcm = read("pcm.f32")?;
    let mut model = paddock_metal::Diarization::load(Path::new(&args[1]), None)?;
    let frontend = model.frontend()?;
    println!(
        "{}",
        serde_json::json!({"model": args[1], "fixture": args[2],
        "audio_seconds": pcm.len() as f64 / 16000., "weight_bytes": model.weight_bytes(),
        "workspace_bytes": model.workspace_bytes(), "repeats": repeats})
    );
    for (name, preset) in [
        ("offline", Preset::Offline),
        ("low", Preset::Low),
        ("very_low", Preset::VeryLow),
        ("ultra_low", Preset::UltraLow),
    ] {
        let expected = read(&format!("{name}.f32"))?;
        let mut digest = None;
        for repeat in 0..=repeats {
            // Warm each geometry independently. Alternate transport partitions;
            // every repetition must return exactly the same frame probabilities.
            let chunk = if repeat % 2 == 0 { 16001 } else { 1103 };
            let mut stream = Stream::new(preset);
            let mut output = Vec::new();
            let start = Instant::now();
            for audio in pcm.chunks(chunk) {
                output.extend(stream.feed(&mut model, &frontend, audio, false)?);
            }
            output.extend(stream.feed(&mut model, &frontend, &[], true)?);
            let wall = start.elapsed().as_secs_f64();
            let actual: Vec<_> = output.iter().flatten().copied().collect();
            assert_eq!(actual.len(), expected.len());
            assert!(!actual.is_empty() && actual.iter().chain(&expected).all(|v| v.is_finite()));
            let rmse = (actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / actual.len() as f64)
                .sqrt();
            let different = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
                .count();
            assert!(
                rmse < 0.005 && different as f64 / actual.len() as f64 <= 0.001,
                "{name}: numerical qualification failed: RMSE {rmse}, decisions {different}"
            );
            let bytes: Vec<_> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
            let hash = blake3::hash(&bytes).to_hex().to_string();
            if let Some(previous) = &digest {
                assert_eq!(
                    &hash, previous,
                    "repeat/partition must preserve exact output"
                );
            }
            digest = Some(hash.clone());
            println!(
                "{}",
                serde_json::json!({"preset": name, "repeat": repeat, "warmup": repeat == 0,
                "chunk": chunk, "wall_seconds": wall, "gpu_seconds": stream.gpu_seconds,
                "rmse": rmse, "decision_differences": different, "probabilities_blake3": hash})
            );
        }
    }
    Ok(())
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS Metal");
}
