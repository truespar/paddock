//! Per-slot (batched) MTP speculative decoding on Qwen3.6-27B: B concurrent
//! prompts, each drafting K tokens per round, verified in one backbone pass
//! over B×(K+1) rows.
//!
//! GATE: every per-slot stream must be BIT-IDENTICAL to `generate_greedy_spec`
//! (single-slot spec) on the same prompt. By spec-decode losslessness the
//! emitted stream depends only on the verify-pass numerics (drafts only move
//! round boundaries), and both paths verify through the same row-invariant
//! MT-MMQ kernels - so this exactly gates all the per-slot machinery (slot
//! KV/state routing, v2 slots+snapshot recurrence, ragged commit, per-slot
//! MTP KV + pending_h). Single-slot spec is itself bit-gated against the
//! externally-verified base path, closing the chain.
//!
//! The f32-exact single-stream path is reported but not asserted against the
//! mmq paths: the 27B has genuinely marginal logits where the dp4a activation
//! -quant class (llama's own) picks a different token - measured here at a
//! 0.7% logit margin, deterministic across slots (see the probe in git
//! history). Same-class comparisons stay bit-exact; cross-class may drift.
//!
//! Heavy (~27 GB load): gated on PADDOCK_HEAVY_TESTS, --test-threads=1.

mod common;

use std::time::Instant;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[test]
fn spec_batch_matches_single_and_compounds() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN36_27B_GGUF", common::QWEN36_27B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuQwen35::load(exec, &map, 4096).expect("load 27B");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);

    // four different prompts with different lengths (per-slot positions differ)
    let prompts: Vec<Vec<u32>> = vec![
        vec![760, 6511, 314, 9338, 369],
        vec![760, 6511, 314],
        vec![314, 9338, 369, 760],
        vec![9338, 369],
    ];
    let b = prompts.len();
    let n_new = 48usize;
    let n_draft = 4usize;

    // single-slot spec references - the same verify numeric class (MT MMQ),
    // so the per-slot streams must match bit-for-bit
    let mut refs = Vec::with_capacity(b);
    for p in &prompts {
        refs.push(
            m.generate_greedy_spec(p, n_new, None, n_draft)
                .expect("spec reference"),
        );
    }
    // f32-exact single-stream, for the drift report only
    let mut refs_f32 = Vec::with_capacity(b);
    for p in &prompts {
        refs_f32.push(m.generate_greedy(p, n_new, None).expect("f32 reference"));
    }

    m.enable_batch(b).expect("enable_batch");
    // allocate the spec-batch buffers (~3 GB of snapshots at B=4, K=4) outside
    // the timed region - generate_greedy_spec_batch reuses a matching state
    m.enable_spec_batch(b, n_draft).expect("enable_spec_batch");

    // --- plain batch baseline (timing + numeric-drift report) -----------------
    {
        let mut outs: Vec<Vec<u32>> = Vec::with_capacity(b);
        let mut vocab = 0usize;
        for (slot, p) in prompts.iter().enumerate() {
            let logits = m.forward_prefill_slot(slot, p).expect("prefill slot");
            vocab = logits.len();
            outs.push(vec![argmax(&logits)]);
        }
        let t0 = Instant::now();
        let mut positions: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
        while outs[0].len() < n_new {
            let toks: Vec<u32> = outs.iter().map(|o| *o.last().unwrap()).collect();
            let logits = m.forward_batch(&toks, &positions).expect("batch step");
            for slot in 0..b {
                outs[slot].push(argmax(&logits[slot * vocab..(slot + 1) * vocab]));
                positions[slot] += 1;
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let drift: Vec<usize> = outs
            .iter()
            .zip(&refs_f32)
            .map(|(o, r)| o.iter().zip(r).take_while(|(a, c)| a == c).count())
            .collect();
        eprintln!(
            "base batch B={b}: {:.1} tok/s aggregate; f32-single match prefix per slot {drift:?} of {n_new} (mmq-vs-f32 class drift is expected)",
            (b * (n_new - 1)) as f64 / dt
        );
    }

    // --- per-slot spec decode ---------------------------------------------------
    let t1 = Instant::now();
    let spec = m
        .generate_greedy_spec_batch(&prompts, n_new, n_draft)
        .expect("spec batch");
    let spec_dt = t1.elapsed().as_secs_f64();
    for (slot, o) in spec.iter().enumerate() {
        assert_eq!(
            o, &refs[slot],
            "spec batch slot {slot} diverged from single-slot spec (same numeric class)"
        );
    }
    eprintln!(
        "spec batch B={b} k={n_draft}: {:.1} tok/s aggregate (incl prefill+warm)",
        (b * n_new) as f64 / spec_dt
    );
    eprintln!("EXACT MATCH: per-slot spec == single-slot spec on all {b} prompts");

    // --- B=8 leg: 8-slot routing, per-slot MTP KV and ragged commit at scale.
    // K must be >= 4 here: the single-slot reference verifies k1 = K+1 rows,
    // and only r > 4 shares the MT MMQ class with the per-slot pass (r <= 4
    // takes the nc dp4a kernel - a different accumulation order, so K = 2
    // refs are a different numeric class and can legitimately flip a marginal
    // token). The B=4 leg above already cross-proves the 24-row MT tile
    // (refs at r=5 ride the 16-row tile, the per-slot pass at r=20 the
    // 24-row one), which is what the recommended B=8 K=2 config verifies on.
    let prompts8: Vec<Vec<u32>> = (0..8)
        .map(|s| {
            let pool = [760u32, 6511, 314, 9338, 369, 1207, 4552, 88];
            (0..3 + s).map(|i| pool[(s + i) % pool.len()]).collect()
        })
        .collect();
    let (b8, k8) = (prompts8.len(), 4usize);
    let mut refs8 = Vec::with_capacity(b8);
    for p in &prompts8 {
        refs8.push(
            m.generate_greedy_spec(p, n_new, None, k8)
                .expect("spec reference"),
        );
    }
    m.enable_batch(b8).expect("enable_batch B=8");
    m.enable_spec_batch(b8, k8).expect("enable_spec_batch B=8");
    let spec8 = m
        .generate_greedy_spec_batch(&prompts8, n_new, k8)
        .expect("spec batch B=8");
    for (slot, o) in spec8.iter().enumerate() {
        assert_eq!(
            o, &refs8[slot],
            "B=8 K={k8} spec batch slot {slot} diverged from single-slot spec"
        );
    }
    eprintln!("EXACT MATCH: B=8 K={k8} per-slot spec == single-slot spec on all {b8} prompts");
}
