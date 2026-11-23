//! Teacher-forced logprob dump for the laguna body (Laguna or Kolibri GGUF):
//! every text in TEACHER_IN is tokenized with the file's own tokenizer and
//! walked through the serial exact-f32 lane one token at a time; each
//! position's top-k log-probabilities land in TEACHER_OUT next to the ids, so
//! another engine fed the SAME ids (e.g. vLLM's `prompt_logprobs`) can be
//! compared position by position - top-1 agreement and the logprob gap -
//! where no same-weights llama.cpp release exists.
//!
//! Usage: LAGUNA_GGUF=... (or LAGUNA_DIR=<Kolibri NVFP4 checkpoint dir>, or
//!        RIPPLE_DIR=<Ripple-65B NVFP4 checkpoint dir>)
//!        PADDOCK_PACK=... TEACHER_IN=texts.json
//!        TEACHER_OUT=out.json [TOPK=5] [MAX_CTX=4096] [GREEDY=n] laguna_teacher
//! GREEDY=n continues each case n tokens greedily after its text and prints
//! the continuation (a coherence check beside the logprob dump). BATCHED=1
//! walks each case through the batched lane's prefill instead (one walk a
//! case: run with PADDOCK_NO_PREFIX_CACHE=1 and PADDOCK_PF_ROWS_WIDE past
//! the longest case) - the prefill classes' own teacher-forced gate.
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
    let greedy: usize = std::env::var("GREEDY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let ripple = std::env::var("RIPPLE_DIR").ok();
    let (mut m, tok) = match std::env::var("LAGUNA_DIR") {
        _ if ripple.is_some() => {
            let dir = ripple.clone().unwrap();
            (
                GpuLaguna::load_ripple_nvfp4(exec, dir.as_ref(), max_ctx).expect("load"),
                GgufTokenizer::from_hf_dir(dir.as_ref()).expect("tokenizer"),
            )
        }
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

    let batched = std::env::var_os("BATCHED").is_some();
    if batched {
        m.enable_batch(1).expect("enable_batch");
    }
    // one position's top-k log-probabilities
    let top_of = |logits: &[f32]| {
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
        best.iter()
            .map(|&i| json!([i, logits[i] - lse]))
            .collect::<Vec<_>>()
    };
    let mut out = Vec::new();
    for case in cases["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap_or("case");
        // a case may carry its ids (tokenized by the reference's own
        // tokenizer) - then both engines walk exactly the same sequence
        let ids: Vec<u32> = match case["ids"].as_array() {
            Some(given) => given
                .iter()
                .map(|v| v.as_u64().expect("id") as u32)
                .collect(),
            None => {
                let mut ids = Vec::new();
                if tok.add_bos {
                    ids.extend(tok.bos_id);
                }
                ids.extend(tok.encode(case["text"].as_str().expect("text")).unwrap());
                ids
            }
        };
        let t0 = std::time::Instant::now();
        let mut top = Vec::with_capacity(ids.len());
        if batched {
            m.prefill_rows_logits(0, &ids, |_, logits| top.push(top_of(logits)))
                .expect("batched prefill");
        } else {
            m.reset();
            for &t in &ids {
                top.push(top_of(&m.forward(t).expect("forward")));
            }
        }
        eprintln!(
            "{name}: {} tokens in {:.1}s",
            ids.len(),
            t0.elapsed().as_secs_f32()
        );
        if greedy > 0 && !batched {
            // continue from the last position's logits, one argmax at a time
            let mut next = top
                .last()
                .and_then(|t| t.first())
                .and_then(|e| e[0].as_u64());
            let mut cont = Vec::new();
            for _ in 0..greedy {
                let Some(t) = next else { break };
                cont.push(t as u32);
                let logits = m.forward(t as u32).expect("forward");
                next = (0..logits.len())
                    .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                    .map(|i| i as u64);
            }
            eprintln!(
                "{name} greedy: {:?}",
                tok.decode(&cont, false).unwrap_or_default()
            );
        }
        out.push(json!({"name": name, "ids": ids, "top": top}));
    }
    std::fs::write(&output, serde_json::to_vec(&json!({"cases": out})).unwrap()).unwrap();
}
