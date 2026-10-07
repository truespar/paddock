//! SAM 3 request latency, stage by stage: one picture, one prompt, warm
//! repeats. Build with `--features static-pack` (or set PADDOCK_PACK); the
//! model is `SAM3_DIR` (default E:/paddock/models/sam3).
//!
//!   cargo run --release -p paddock-engine --features static-pack --example sam3_bench -- \
//!       [picture.jpg] [prompt] [runs]
//!
//! Prints each stage's median and best over the runs, twice: cold (both
//! encodings forgotten before every run - a new picture and a new prompt) and
//! warm (the same picture again, so its encoding is reused). Under nsys it is
//! the profile target: `nsys profile -t cuda,nvtx -o sam3 <exe> ...`.

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::sam3::{GpuSam3, Sam3Request};
use paddock_tokenizer::sam3::Sam3Tokenizer;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pic = args
        .first()
        .cloned()
        .unwrap_or_else(|| "E:/dev/sam3-ref/sam3/assets/images/groceries.jpg".into());
    let prompt = args.get(1).cloned().unwrap_or_else(|| "paper bag".into());
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let dir = std::env::var_os("SAM3_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "E:/paddock/models/sam3".into());
    let pack = std::env::var_os("PADDOCK_PACK").map(std::path::PathBuf::from);

    let exec = Arc::new(GpuExecutor::with_pack(0, pack.as_deref()).expect("executor"));
    let tok = Sam3Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let mut sam = GpuSam3::load_dir(exec, &dir, 24_000_000, 16).expect("load sam3");
    let bytes = std::fs::read(&pic).expect("picture");
    let img = paddock_jpeg::decode_rgb(&bytes, 64_000_000).expect("decode");
    let tk = tok.encode(&prompt).expect("tokenize");
    let req = Sam3Request {
        ids: tk.ids,
        valid: tk.valid,
        boxes: Vec::new(),
        threshold: 0.5,
    };
    println!(
        "{pic} {}x{}, prompt {prompt:?} ({} tokens), {runs} runs after 3 warm",
        img.width, img.height, tk.valid
    );
    for _ in 0..3 {
        sam.forget();
        sam.segment(&img.rgb, img.width, img.height, &req)
            .expect("warm");
    }
    for cold in [true, false] {
        let mut stages: [Vec<f64>; 6] = Default::default();
        let mut kept = 0;
        for _ in 0..runs {
            if cold {
                sam.forget();
            }
            let t0 = std::time::Instant::now();
            let out = sam
                .segment(&img.rgb, img.width, img.height, &req)
                .expect("segment");
            let total = t0.elapsed().as_secs_f64() * 1e3;
            let t = &out.timings;
            assert_eq!(t.image_reused, !cold, "image reuse");
            for (i, v) in [
                t.resize_ms,
                t.encode_ms,
                t.prompt_ms,
                t.detect_ms,
                t.masks_ms,
                total,
            ]
            .into_iter()
            .enumerate()
            {
                stages[i].push(v);
            }
            kept = out.instances.len();
        }
        println!(
            "{}",
            if cold {
                "cold (new picture, new prompt):"
            } else {
                "warm (the same picture and prompt again):"
            }
        );
        for (name, v) in ["resize", "encode", "prompt", "detect", "masks", "total"]
            .iter()
            .zip(stages.iter_mut())
        {
            let best = v.iter().copied().fold(f64::INFINITY, f64::min);
            println!(
                "  {name:7} median {:8.2} ms   best {best:8.2} ms",
                median(v)
            );
        }
        println!("  kept {kept} instances");
    }
}
