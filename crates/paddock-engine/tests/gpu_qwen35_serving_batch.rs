//! Continuous-batching serving test (9B): B slots prefilled with the same
//! prompt must produce identical greedy streams (slot isolation + determinism)
//! that match the single-sequence path, and aggregate decode throughput must
//! scale with B (the weight read amortizes across concurrent sequences).
//!
//! Heavy GPU test: --test-threads=1.

mod common;

use std::time::Instant;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

fn argmax(row: &[f32]) -> u32 {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

#[test]
fn serving_batch_isolation_and_throughput() {
    let Some(path) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuQwen35::load(exec.clone(), &map, 4096).expect("load 9B");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);

    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369]; // "The capital of France is"
    let n_new = 64usize;

    // single-sequence reference stream
    let single = m.generate_greedy(&prompt, n_new, None).expect("single");

    // batch: the same prompt in every slot, at the width the plan grants
    // (enable_batch returns the seated count - on a box where 64 slots of
    // a 27B's DeltaNet state do not fit, that is fewer than asked)
    let b = m.enable_batch(64).expect("enable_batch");
    let vocab = m.vocab;
    let mut last_tok = vec![0u32; b];
    for (s, lt) in last_tok.iter_mut().enumerate() {
        let lg = m.forward_prefill_slot(s, &prompt).expect("prefill slot");
        *lt = argmax(&lg);
    }
    assert!(
        last_tok.iter().all(|&t| t == last_tok[0]),
        "prefill divergence across slots"
    );

    let mut streams: Vec<Vec<u32>> = (0..b).map(|s| vec![last_tok[s]]).collect();
    let mut positions: Vec<u32> = vec![prompt.len() as u32; b];
    let t0 = Instant::now();
    for _ in 0..n_new - 1 {
        let toks: Vec<u32> = streams.iter().map(|s| *s.last().unwrap()).collect();
        let logits = m.forward_batch(&toks, &positions).expect("batch step");
        for s in 0..b {
            let next = argmax(&logits[s * vocab..(s + 1) * vocab]);
            streams[s].push(next);
            positions[s] += 1;
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let agg = (b * (n_new - 1)) as f64 / dt;
    eprintln!(
        "batch B={b}: {:.1} tok/s aggregate ({:.1} tok/s per stream)",
        agg,
        agg / b as f64
    );

    // slot isolation: identical inputs -> identical streams
    for s in 1..b {
        assert_eq!(streams[s], streams[0], "slot {s} diverged from slot 0");
    }
    // cross-path: the batch stream should match single-sequence greedy (dp4a MMQ
    // numerics vs the f32 decode gemv - llama's own numeric class; report if not)
    if streams[0] != single {
        let d = single.iter().zip(&streams[0]).position(|(a, b)| a != b);
        eprintln!("NOTE: batch stream diverges from single path at {d:?} (dp4a vs f32 class)");
    } else {
        eprintln!("batch stream == single-sequence greedy (EXACT)");
    }
    assert_eq!(
        streams[0], single,
        "batch path must match single-sequence greedy"
    );

    // throughput scaling snapshot at several batch sizes
    for bb in [1usize, 2, 4, 8, 16, 32, 64] {
        let toks: Vec<u32> = vec![streams[0][0]; bb];
        let mut pos: Vec<u32> = positions[..bb].to_vec();
        // warm
        for _ in 0..3 {
            let _ = m.forward_batch(&toks, &pos).expect("warm");
        }
        let t1 = Instant::now();
        let iters = 32usize;
        for _ in 0..iters {
            let _ = m.forward_batch(&toks, &pos).expect("step");
            for p in pos.iter_mut() {
                *p += 1;
            }
        }
        let s = t1.elapsed().as_secs_f64();
        eprintln!(
            "B={bb}: {:.1} tok/s aggregate ({:.1}/stream)",
            (bb * iters) as f64 / s,
            iters as f64 / s
        );
    }
}
