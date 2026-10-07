//! Batched-prefill timing probe for the laguna body (Laguna or Kolibri):
//! enable the batched lane, prefill a PF_TOKENS prompt into slot 0 REPS
//! times - a fresh prompt each pass, so none is a prefix-cache hit (the first
//! pass warms) - and print each pass's wall and tok/s; the shape to run
//! under nsys for a per-kernel split.
//!
//! Usage: LAGUNA_GGUF=... (or LAGUNA_DIR=<Kolibri NVFP4 checkpoint dir>)
//!        PADDOCK_PACK=... [PF_TOKENS=10000] [REPS=3]
//!        [MAX_CTX=32768] [PF_TEXT=file] laguna_prefill
//! PF_TEXT tokenizes a real text with the file's tokenizer (its first
//! PF_TOKENS ids): routing over natural text spreads across the experts the
//! way a served prompt does, where the synthetic id walk concentrates it.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let (n, reps, max_ctx) = (
        env("PF_TOKENS", 10000),
        env("REPS", 3),
        env("MAX_CTX", 32768),
    );
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let (mut m, tok) = match std::env::var("LAGUNA_DIR") {
        Ok(dir) => (
            GpuLaguna::load_kolibri_nvfp4(exec.clone(), dir.as_ref(), max_ctx).expect("load"),
            GgufTokenizer::from_hf_dir(dir.as_ref()).expect("tokenizer"),
        ),
        Err(_) => {
            let model = std::env::var("LAGUNA_GGUF").expect("set LAGUNA_GGUF or LAGUNA_DIR");
            let map = MappedGguf::open(model.as_ref()).expect("open gguf");
            (
                GpuLaguna::load(exec.clone(), &map, max_ctx).expect("load"),
                GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer"),
            )
        }
    };
    m.enable_batch(2).expect("enable_batch");
    // one prompt a pass: the same ids twice would be a prefix-cache hit
    let prompts: Vec<Vec<u32>> = match std::env::var("PF_TEXT") {
        Ok(f) => {
            let text = std::fs::read_to_string(f).expect("PF_TEXT");
            let ids = tok.encode(&text).expect("encode");
            assert!(
                ids.len() >= n * reps,
                "PF_TEXT holds {} tokens, want {n} x {reps}",
                ids.len()
            );
            ids.chunks(n).take(reps).map(<[u32]>::to_vec).collect()
        }
        // a synthetic id walk (routes to far fewer experts than real text)
        Err(_) => (0..reps as u32)
            .map(|r| {
                (0..n as u32)
                    .map(|i| 1000 + (i * 7919 + r * 104_729) % 120000)
                    .collect()
            })
            .collect(),
    };
    for (r, prompt) in prompts.iter().enumerate() {
        let t0 = std::time::Instant::now();
        m.forward_prefill(0, prompt).expect("prefill");
        exec.synchronize().unwrap();
        let s = t0.elapsed().as_secs_f64();
        println!(
            "pass {r}: {n} tokens in {s:.3}s = {:.0} tok/s",
            n as f64 / s
        );
    }
}
