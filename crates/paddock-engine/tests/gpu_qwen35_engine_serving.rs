//! End-to-end continuous-batching serving through the ENGINE scheduler (the same
//! service the HTTP server runs): spawn the engine on qwen35, submit N concurrent
//! greedy requests with the same prompt, and require identical streams that match
//! the direct single-sequence path. This proves the Generator batch seam
//! (enable_batch / forward_prefill_batch / forward_batch) end-to-end.
//!
//! Heavy GPU test: --test-threads=1.

mod common;

use std::sync::Arc;
use std::time::Instant;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_engine::sampler::SamplingParams;
use paddock_engine::service::{Engine, GenRequest, TokenEvent};
use paddock_models::mapped::MappedGguf;

#[test]
fn engine_scheduler_batches_qwen35() {
    let Some(pack) = common::pack() else {
        return;
    };
    let Some(path) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };

    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369];
    let n_new = 48usize;

    // direct single-sequence reference
    let reference = {
        let exec = match GpuExecutor::new(0, &pack) {
            Ok(e) => Arc::new(e),
            Err(e) => {
                eprintln!("no CUDA ({e}) - skipping");
                return;
            }
        };
        let map = MappedGguf::open(&path).expect("open gguf");
        let mut m = GpuQwen35::load(exec, &map, 4096).expect("load 9B");
        // the exact class this test is stated in (the KV8 default is lossy)
        m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
        m.generate_greedy(&prompt, n_new, None).expect("reference")
    };

    // engine with the continuous-batching scheduler (drops the direct model first
    // so both fit trivially; the engine builds its own on its thread)
    let pack2 = pack.clone();
    let path2 = path.clone();
    let engine = Engine::spawn(8, move || {
        let exec = Arc::new(GpuExecutor::new(0, &pack2).map_err(|e| e.to_string())?);
        let map = MappedGguf::open(&path2).map_err(|e| e.to_string())?;
        let mut m = GpuQwen35::load(exec, &map, 4096).map_err(|e| e.to_string())?;
        // the exact class this test is stated in (the KV8 default is lossy)
        m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
        Ok(Box::new(m) as Box<dyn paddock_engine::generator::Generator>)
    })
    .expect("spawn engine");

    // submit 6 concurrent greedy requests
    let n_req = 6usize;
    let mut rxs = Vec::new();
    let t0 = Instant::now();
    for _ in 0..n_req {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        engine
            .submit(GenRequest {
                prompt: prompt.clone(),
                max_tokens: n_new,
                sampler: SamplingParams::default(),
                stop_tokens: vec![],
                events: tx,
                mm_chunks: None,
                constraint: None,
                logprobs: None,
                submitted: None,
                canvas_read: None,
                user_turn: false,
            })
            .expect("submit");
        rxs.push(rx);
    }

    let mut streams = Vec::new();
    for mut rx in rxs {
        let mut toks = Vec::new();
        loop {
            match rx.blocking_recv().expect("event") {
                TokenEvent::Prefilled { .. } => {}
                TokenEvent::Token { id: t, .. } => toks.push(t),
                TokenEvent::Done(..) => break,
                TokenEvent::Error(e) => panic!("engine error: {e}"),
            }
        }
        streams.push(toks);
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "{n_req} concurrent requests × {n_new} tokens in {dt:.2}s = {:.1} tok/s aggregate",
        (n_req * n_new) as f64 / dt
    );

    for (i, s) in streams.iter().enumerate() {
        assert_eq!(
            s, &reference,
            "request {i} diverged from the reference stream"
        );
    }
    eprintln!("ENGINE SERVING OK: {n_req} concurrent streams identical to the reference");
}
