//! Teacher-forced logprob dump for the laguna body (Laguna or Kolibri GGUF):
//! every text in TEACHER_IN is tokenized with the file's own tokenizer and
//! walked through the serial exact-f32 lane one token at a time; each
//! position's top-k log-probabilities land in TEACHER_OUT next to the ids, so
//! another engine fed the SAME ids (e.g. vLLM's `prompt_logprobs`) can be
//! compared position by position - top-1 agreement and the logprob gap -
//! where no same-weights llama.cpp release exists.
//!
//! Usage: LAGUNA_GGUF=... (or LAGUNA_DIR=<Kolibri NVFP4 checkpoint dir>)
//!        PADDOCK_PACK=... TEACHER_IN=texts.json
//!        TEACHER_OUT=out.json [TOPK=5] [MAX_CTX=4096] laguna_teacher
//! TEACHER_IN: {"cases": [{"name": "...", "text": "..."}, ...]}
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;
use serde_json::{Value, json};

fn main() {
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let input = std::env::var("TEACHER_IN").expect("set TEACHER_IN");
    let output = std::env::var("TEACHER_OUT").expect("set TEACHER_OUT");
    let topk: usize = std::env::var("TOPK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let max_ctx: usize = std::env::var("MAX_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let cases: Value = serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();

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

    let mut out = Vec::new();
    for case in cases["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap_or("case");
        let mut ids = Vec::new();
        if tok.add_bos {
            ids.extend(tok.bos_id);
        }
        ids.extend(tok.encode(case["text"].as_str().expect("text")).unwrap());
        m.reset();
        let t0 = std::time::Instant::now();
        let mut top = Vec::with_capacity(ids.len());
        for &t in &ids {
            let logits = m.forward(t).expect("forward");
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx
                + logits
                    .iter()
                    .map(|&v| ((v - mx) as f64).exp())
                    .sum::<f64>()
                    .ln() as f32;
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.select_nth_unstable_by(topk, |&a, &b| logits[b].total_cmp(&logits[a]));
            let mut best: Vec<usize> = order[..topk].to_vec();
            best.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            top.push(
                best.iter()
                    .map(|&i| json!([i, logits[i] - lse]))
                    .collect::<Vec<_>>(),
            );
        }
        eprintln!(
            "{name}: {} tokens in {:.1}s",
            ids.len(),
            t0.elapsed().as_secs_f32()
        );
        out.push(json!({"name": name, "ids": ids, "top": top}));
    }
    std::fs::write(&output, serde_json::to_vec(&json!({"cases": out})).unwrap()).unwrap();
}
