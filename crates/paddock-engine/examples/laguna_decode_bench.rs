//! Laguna-body decode-tick bench (Laguna or Kolibri): the serving hot path
//! (`forward_batch_sampled`, graph-replayed) timed over N steps at R rows,
//! each slot prefilled from its own window of a real text (PF_TEXT) - routing
//! over text spreads across the experts the way a served stream does, and at
//! R > 1 the rows' distinct experts are what the tick streams. Under nsys pass
//! --cuda-graph-trace=node, or the graph ticks vanish from the trace.
//!
//! Usage: LAGUNA_GGUF=... (or LAGUNA_DIR=<Kolibri NVFP4 checkpoint dir>)
//!        PADDOCK_PACK=... PF_TEXT=file [R=4] [N=128] [PROMPT_LEN=512]
//!        [MAX_CTX=4096] laguna_decode_bench
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::generator::{Generator, RowSample};
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_engine::sampler::DevicePlan;
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
    let (r, n, plen, max_ctx) = (
        env("R", 4),
        env("N", 128),
        env("PROMPT_LEN", 512),
        env("MAX_CTX", 4096),
    );
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let (mut m, tok) = match std::env::var("LAGUNA_DIR") {
        Ok(dir) => (
            GpuLaguna::load_kolibri_nvfp4(exec, dir.as_ref(), max_ctx).expect("load"),
            GgufTokenizer::from_hf_dir(dir.as_ref()).expect("tokenizer"),
        ),
        Err(_) => {
            let model = std::env::var("LAGUNA_GGUF").expect("set LAGUNA_GGUF or LAGUNA_DIR");
            let map = MappedGguf::open(model.as_ref()).expect("open gguf");
            (
                GpuLaguna::load(exec, &map, max_ctx).expect("load"),
                GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer"),
            )
        }
    };
    let cap = m.enable_batch(r).expect("enable_batch");
    assert!(cap >= r, "enable_batch gave {cap} slots, want {r}");

    let text =
        std::fs::read_to_string(std::env::var("PF_TEXT").expect("set PF_TEXT")).expect("PF_TEXT");
    let ids = tok.encode(&text).expect("encode");
    assert!(ids.len() >= r * plen, "PF_TEXT too short for {r} x {plen}");
    let mut tokens = Vec::with_capacity(r);
    for (slot, p) in ids.chunks(plen).take(r).enumerate() {
        let logits = m.forward_prefill(slot, p).expect("prefill");
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i as u32);
        tokens.push(best);
    }
    let mut positions = vec![plen as u32; r];
    let plans: Vec<RowSample> = (0..r)
        .map(|_| RowSample::Device(DevicePlan::Greedy))
        .collect();
    let mut step = |tokens: &mut Vec<u32>, positions: &mut Vec<u32>| {
        let s = m
            .forward_batch_sampled(tokens, positions, &plans)
            .expect("step");
        tokens.copy_from_slice(&s.ids[..r]);
        for p in positions.iter_mut() {
            *p += 1;
        }
    };
    // warmup: the first ticks capture the graph
    for _ in 0..4 {
        step(&mut tokens, &mut positions);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        step(&mut tokens, &mut positions);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "R={r} N={n} ctx {plen}: {:.2} ms/step, {:.1} tok/s aggregate",
        dt * 1e3 / n as f64,
        (r * n) as f64 / dt
    );
}
