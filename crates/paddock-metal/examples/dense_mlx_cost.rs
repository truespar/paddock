//! Fixed-token full-logit diagnostic for Bonsai/Gemma. Not an HTTP benchmark.
//! Freeze this executable before changing kernels, then compare every digest.
#[cfg(not(target_os = "macos"))]
fn main() {
    panic!("requires macOS Metal");
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use paddock_engine::generator::Generator;
    use std::{path::Path, time::Instant};
    trait TimedGenerator: Generator {
        fn gpu_seconds(&self) -> f64;
    }
    impl TimedGenerator for paddock_metal::Qwen35 {
        fn gpu_seconds(&self) -> f64 {
            self.last_gpu_seconds
        }
    }
    impl TimedGenerator for paddock_metal::Gemma4 {
        fn gpu_seconds(&self) -> f64 {
            self.last_gpu_seconds
        }
    }
    let args: Vec<_> = std::env::args().collect();
    let family = args
        .get(1)
        .ok_or("dense_mlx_cost bonsai|gemma MODEL C PROMPT STEPS ROUNDS")?;
    let path = Path::new(args.get(2).ok_or("model directory")?);
    let c: usize = args.get(3).ok_or("concurrency")?.parse()?;
    let p: usize = args.get(4).ok_or("prompt tokens")?.parse()?;
    let steps: usize = args.get(5).ok_or("decode steps")?.parse()?;
    let rounds: usize = args.get(6).ok_or("rounds (including warm-up)")?.parse()?;
    if !(1..=4).contains(&c) || p == 0 || steps == 0 || rounds == 0 || !path.is_dir() {
        return Err("requires MLX directory and positive sizes, concurrency 1..4".into());
    }
    let context = (p + steps + 16).max(4096);
    let mut model: Box<dyn TimedGenerator> = match family.as_str() {
        "bonsai" => Box::new(paddock_metal::Qwen35::load(path, context, c, None)?),
        "gemma" => Box::new(paddock_metal::Gemma4::load(path, context, c, None)?),
        _ => return Err("expected bonsai or gemma".into()),
    };
    let vocab = model.vocab();
    for round in 0..rounds {
        model.reset();
        // Distinct first tokens prevent warm-prefix reuse; the same round
        // always has the same tokens in old/new binaries, independent of timing.
        for slot in 0..c {
            model.prefill_begin(
                slot,
                (0..p)
                    .map(|i| 1000 + ((i * 17 + slot * 41 + round * 997) % 16000) as u32)
                    .collect(),
            )?;
        }
        let mut hash = blake3::Hasher::new();
        let mut digest = |row: &[f32]| {
            assert!(row.iter().all(|v| v.is_finite()));
            for value in row {
                hash.update(&value.to_bits().to_le_bytes());
            }
        };
        let mut logits = vec![Vec::new(); c];
        let start = Instant::now();
        let mut complete = 0;
        let mut prefill_gpu = 0.0;
        while complete < c {
            let (_, done) = model.forward_mixed(&[], 512)?;
            prefill_gpu += model.gpu_seconds();
            for (slot, row, _) in done {
                logits[slot] = row;
                complete += 1;
            }
        }
        let prefill = start.elapsed().as_secs_f64();
        for row in &logits {
            digest(row);
        }
        let mut times = Vec::new();
        let mut gpu_times = Vec::new();
        for step in 0..steps {
            let tokens: Vec<_> = (0..c)
                .map(|slot| 7000 + step as u32 + slot as u32)
                .collect();
            let start = Instant::now();
            let row = model.forward_batch(&tokens, &vec![(p + step) as u32; c])?;
            times.push(start.elapsed().as_secs_f64());
            gpu_times.push(model.gpu_seconds());
            assert_eq!(row.len(), c * vocab);
            digest(&row);
        }
        println!(
            "DENSE_MLX_COST {}",
            serde_json::json!({
                "family":family,"concurrency":c,"prompt":p,"steps":steps,"round":round,
                "prefill_s":prefill,"prefill_gpu_s":prefill_gpu,
                "decode_step_s":times,"decode_gpu_s":gpu_times,
                "logits_blake3":hash.finalize().to_hex().to_string(),
                "allocated_bytes":model.device_mem_used(),"instrumented":std::env::var_os("PADDOCK_METAL_PROFILE").is_some()
            })
        );
    }
    Ok(())
}
