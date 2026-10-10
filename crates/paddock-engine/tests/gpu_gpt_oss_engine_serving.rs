//! End-to-end continuous-batching serving for gpt-oss through the ENGINE
//! scheduler, exercising the SPECULATIVE round (per-slot n-gram drafts ->
//! forward_spec_batch) against the dense fallback: the same requests run on
//! a spec-enabled engine and a PADDOCK_NO_SERVE_SPEC one, and every greedy
//! stream must match - templated-JSON prompts, the established clear-margin
//! bar. A seeded temperature request rides along on the spec engine to prove
//! the greedy gate (a sampling slot forces the dense path while it lives,
//! spec rounds resume after it finishes) without deadlock or corruption.
//!
//! Heavy GPU test: --test-threads=1, PADDOCK_HEAVY_TESTS=1.

mod common;

use std::sync::Arc;
use std::time::Instant;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::gpt_oss::GpuGptOss;
use paddock_engine::sampler::SamplingParams;
use paddock_engine::service::{Engine, GenRequest, TokenEvent};
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn collect(mut rx: tokio::sync::mpsc::UnboundedReceiver<TokenEvent>) -> Vec<u32> {
    let mut toks = Vec::new();
    loop {
        match rx.blocking_recv().expect("event") {
            TokenEvent::Prefilled { .. } => {}
            TokenEvent::Token { id: t, .. } => toks.push(t),
            TokenEvent::Done(..) => break,
            TokenEvent::Error(e) => panic!("engine error: {e}"),
        }
    }
    toks
}

#[test]
fn engine_scheduler_specs_gpt_oss() {
    if !common::heavy() {
        return;
    }
    // keeps the pack PATH: each engine is spawned on its own thread and builds
    // its own executor from it
    let Some(pack) = common::pack() else {
        return;
    };
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    // spec-vs-dense scheduler exactness is an int8-class property (like the
    // spec parity gates): the block-scale fp8 prefill shifts hidden states
    // enough to flip near-tie tokens between the dp4a dense steps and the
    // mmq verify rows. Pin one class end to end for this gate.
    paddock_engine::gpu_model::gpt_oss::set_moe_bs(false);
    let items = [
        "apple 1 3.50\nbanana 2 1.25\ncherry 3 8.00\ndamson 4 2.75\n",
        "kiwi 11 2.10\nlemon 12 0.80\nmango 13 5.40\nnectarine 14 3.30\n",
        "pear 21 1.90\nquince 22 6.60\nraisin 23 0.40\nsloe 24 9.90\n",
    ];
    let prompts: Vec<Vec<u32>> = {
        let map = MappedGguf::open(&model_path).expect("open gguf");
        let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
        items
            .iter()
            .map(|it| {
                let text = format!(
                    "Convert each item to a JSON object with fields name, id and \
                     price, one per line:\n{it}\n{{\"name\": \"x\", \"id\": 0, \
                     \"price\": 0.0}}\n"
                );
                tok.encode(&text).expect("encode")
            })
            .collect()
    };
    let n_new = 40usize;

    let spawn_engine = |pack: std::path::PathBuf, path: std::path::PathBuf| {
        Engine::spawn(4, move || {
            let exec = Arc::new(GpuExecutor::new(0, &pack).map_err(|e| e.to_string())?);
            let map = MappedGguf::open(&path).map_err(|e| e.to_string())?;
            let mut m = GpuGptOss::load(exec, &map, 2048).map_err(|e| e.to_string())?;
            // the exact class this test is stated in (the KV8 default is lossy)
            m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
            Ok(Box::new(m) as Box<dyn paddock_engine::generator::Generator>)
        })
        .expect("spawn engine")
    };
    let submit_all = |engine: &Engine, with_temp: bool| {
        let mut rxs = Vec::new();
        for p in &prompts {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            engine
                .submit(GenRequest {
                    prompt: p.clone(),
                    max_tokens: n_new,
                    sampler: SamplingParams::default(), // pure greedy
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
        let mut temp_rx = None;
        if with_temp {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            engine
                .submit(GenRequest {
                    prompt: prompts[0].clone(),
                    max_tokens: 8,
                    sampler: SamplingParams {
                        temperature: 0.8,
                        ..Default::default()
                    },
                    stop_tokens: vec![],
                    events: tx,
                    mm_chunks: None,
                    constraint: None,
                    logprobs: None,
                    submitted: None,
                    canvas_read: None,
                    user_turn: false,
                })
                .expect("submit temp");
            temp_rx = Some(rx);
        }
        (rxs, temp_rx)
    };

    // spec-enabled engine, with a seeded temperature request riding along
    let t0 = Instant::now();
    let engine = spawn_engine(pack.clone(), model_path.clone());
    let (rxs, temp_rx) = submit_all(&engine, true);
    let spec_streams: Vec<Vec<u32>> = rxs.into_iter().map(collect).collect();
    let temp_stream = collect(temp_rx.expect("temp rx"));
    drop(engine);
    eprintln!(
        "spec engine: {} greedy x {n_new} + temp x {} in {:.1}s",
        prompts.len(),
        temp_stream.len(),
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(
        temp_stream.len(),
        8,
        "temperature request must run to length"
    );

    // dense-only engine (spec pinned off) - the same request mix, temp
    // included: while the temp slot lives both engines run identical b=4
    // dense ticks, so the only difference under test is spec rounds vs
    // dense ticks after it retires (b=3 vs b=4 dense alone already crosses
    // the dp4a/mmq MoE class boundary and can flip near-ties).
    // SAFETY: heavy GPU tests run --test-threads=1 (serial), per repo policy
    unsafe { std::env::set_var("PADDOCK_NO_SERVE_SPEC", "1") };
    let engine = spawn_engine(pack, model_path);
    let (rxs, temp_rx) = submit_all(&engine, true);
    let dense_streams: Vec<Vec<u32>> = rxs.into_iter().map(collect).collect();
    let _ = collect(temp_rx.expect("temp rx"));
    drop(engine);
    // SAFETY: see above
    unsafe { std::env::remove_var("PADDOCK_NO_SERVE_SPEC") };

    for (i, (s, d)) in spec_streams.iter().zip(&dense_streams).enumerate() {
        assert_eq!(s.len(), n_new, "greedy stream {i} short");
        assert_eq!(
            s, d,
            "request {i}: spec round diverged from dense scheduler"
        );
    }
    eprintln!(
        "ENGINE SPEC SERVING OK: {} streams identical spec vs dense",
        prompts.len()
    );
    paddock_engine::gpu_model::gpt_oss::set_moe_bs(true);
}
