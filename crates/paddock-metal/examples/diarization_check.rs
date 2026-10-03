//! Same-checkpoint qualification, not a serving fallback.
#![allow(clippy::unwrap_used)]
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use paddock_engine::diarization::{Backend, Stream};
    use paddock_models::diarization::{Preset, segments};
    use std::{path::Path, time::Instant};
    let args: Vec<_> = std::env::args().collect();
    if ![3, 4].contains(&args.len()) {
        return Err(
            "usage: diarization_check MODEL FIXTURE_DIR [NEW_TRACE_OUTPUT_DIR | --forward-only]"
                .into(),
        );
    }
    let forward_only = args.get(3).is_some_and(|s| s == "--forward-only");
    let trace_dir = args.get(3).filter(|_| !forward_only);
    if let Some(path) = trace_dir {
        std::fs::create_dir(path)?;
    }
    let root = Path::new(&args[2]);
    let read = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(std::fs::read(root.join(name))?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    };
    let failed = std::cell::Cell::new(false);
    let compare = |name: &str, a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        let max = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rmse = (a
            .iter()
            .zip(b)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt();
        assert!(a.iter().chain(b).all(|v| v.is_finite()), "nonfinite {name}");
        if name == "mel" {
            assert!(max < 0.002, "mel drift {max}");
        }
        if !["mel", "pre_encode"].contains(&name) {
            let disagreements = a
                .iter()
                .zip(b)
                .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
                .count();
            if rmse >= 0.005 || disagreements as f64 / a.len() as f64 > 0.001 {
                failed.set(true);
                eprintln!(
                    "qualification failed: {name} RMSE={rmse}, decisions={disagreements}/{}",
                    a.len()
                );
            }
        }
        println!(
            "{}",
            serde_json::json!({"stage":name,"values":a.len(),"max_abs":max,"rmse":rmse,"threshold_disagreements":a.iter().zip(b).filter(|(a,b)|(**a>0.5)!=(**b>0.5)).count()})
        );
    };
    let mut model = paddock_metal::Diarization::load(Path::new(&args[1]), None)?;
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("meta.json"))?)?;
    let valid = meta["frames"].as_u64().unwrap() as usize;
    let pcm = read("pcm.f32")?;
    let fe = model.frontend()?;
    let own = fe.features(
        &pcm,
        0,
        pcm.len(),
        0,
        meta["padded"].as_u64().unwrap() as usize,
    );
    compare("mel", &own, &read("features.f32")?);
    let start = Instant::now();
    let emb = model.pre_encode(&read("features.f32")?)?;
    compare("pre_encode", &emb, &read("embedded.f32")?);
    if !forward_only && root.join("trace_embed_norm.f32").exists() {
        for (name, actual) in model.trace(&read("embedded.f32")?, valid.div_ceil(8))? {
            if let Some(path) = trace_dir {
                std::fs::write(
                    Path::new(path).join(format!("{name}.f32")),
                    actual
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
            let expected = read(&format!("trace_{name}.f32"))?;
            let rmse = (actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / actual.len() as f64)
                .sqrt();
            let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            println!(
                "{}",
                serde_json::json!({"trace":name,"rmse":rmse,"different":mismatches,"values":actual.len()})
            );
        }
    }
    let (p, gpu) = model.predict(&emb, valid.div_ceil(8))?;
    let flat: Vec<_> = p.iter().flatten().copied().collect();
    compare("forward", &flat, &read("probabilities.f32")?);
    println!(
        "{}",
        serde_json::json!({"wall_seconds":start.elapsed().as_secs_f64(),"gpu_seconds":gpu,"weight_bytes":model.weight_bytes(),"workspace_bytes":model.workspace_bytes()})
    );
    for (name, preset) in [
        ("offline", Preset::Offline),
        ("low", Preset::Low),
        ("very_low", Preset::VeryLow),
        ("ultra_low", Preset::UltraLow),
    ] {
        if forward_only {
            break;
        }
        let expected = root.join(format!("{name}.f32"));
        if !expected.exists() {
            continue;
        }
        let mut previous = None;
        let file_quantum = (preset.geometry().chunk
            * paddock_models::diarization::STACK
            * paddock_models::diarization::HOP)
            .min(paddock_models::diarization::SAMPLE_RATE);
        for chunk in [16001, 1103, file_quantum] {
            let start = Instant::now();
            let mut s = Stream::new(preset);
            let mut probs = Vec::new();
            for a in pcm.chunks(chunk) {
                probs.extend(s.feed(&mut model, &fe, a, false)?);
            }
            probs.extend(s.feed(&mut model, &fe, &[], true)?);
            let flat: Vec<_> = probs.iter().flatten().copied().collect();
            compare(name, &flat, &read(&format!("{name}.f32"))?);
            let bits: Vec<_> = flat.iter().map(|v| v.to_bits()).collect();
            if let Some(p) = previous {
                assert_eq!(bits, p, "transport partition must not change inference");
            }
            previous = Some(bits);
            println!(
                "{}",
                serde_json::json!({"preset":name,"chunk":chunk,"wall_seconds":start.elapsed().as_secs_f64(),"gpu_seconds":s.gpu_seconds,"retained_rows":s.retained_rows(),"frames":s.frames,"segments":segments(&probs,0,0.5)})
            );
        }
    }
    if failed.get() {
        return Err("numerical qualification failed (all presets reported)".into());
    }
    Ok(())
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS Metal");
}
