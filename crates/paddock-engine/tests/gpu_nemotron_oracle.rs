//! nemotron_h_moe engine vs a vLLM reference oracle dump:
//! per-layer stage sums over the oracle's 15-token prefill, then 32 greedy
//! decode steps against the oracle's ids.
//!
//! Numeric expectations: the arbiter runs a bf16 residual stream (rounded
//! at every layer boundary), Marlin W4A16 GEMMs and fp8 KV; we run an f32
//! residual stream, our own kernels and f16 KV. Stage sums are therefore a
//! BAND gate (structure errors are orders of magnitude, numerics are
//! percent-level); the sharp gates are top-1 at the prompt boundary and the
//! greedy id sequence.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;

const CKPT_ENV: &str = "NEMOTRON_NVFP4_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
const ORACLE: &str = "/models/nemotron-battery/oracle/decoder-oracle.json";

#[test]
fn nemotron_engine_matches_decoder_oracle() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_mamba2() || !exec.has_nvf4_moe() || !exec.has_nvf4_ckpt() {
        common::missing("pack lacks nemotron kernels (cc != 12.0?)");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let oracle_path = std::env::var("NEMOTRON_ORACLE").unwrap_or_else(|_| ORACLE.into());
    let Ok(raw) = std::fs::read(&oracle_path) else {
        common::missing(&format!("no oracle dump at {oracle_path}"));
        return;
    };
    let oracle: serde_json::Value = serde_json::from_slice(&raw).expect("oracle json");
    let prompt_ids: Vec<u32> = oracle["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let greedy_ids: Vec<u32> = oracle["greedy_ids"]
        .as_array()
        .expect("greedy_ids")
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let t0 = std::time::Instant::now();
    let mut model = GpuNemotron::load_dir(exec, &dir, 4096).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ---- prefill walk, accumulating per-layer stage sums -------------------
    let n_layer = 52usize;
    let mut sums = vec![0f64; n_layer];
    let mut abs_sums = vec![0f64; n_layer];
    let mut last_logits: Vec<f32> = Vec::new();
    for (i, &tok) in prompt_ids.iter().enumerate() {
        let mut stages: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
        let logits = model.forward_probed(tok, &mut stages).expect("forward");
        assert_eq!(stages.len(), n_layer);
        for (li, hs) in stages.iter().enumerate() {
            for &v in hs {
                sums[li] += v as f64;
                abs_sums[li] += (v as f64).abs();
            }
        }
        if i + 1 == prompt_ids.len() {
            last_logits = logits;
        }
    }

    // Report the whole profile first, then gate: cross-engine numerics
    // (bf16 residual + Marlin + fp8 KV on the arbiter side vs f32 residual
    // + our kernels + f16 KV here) drift percent-level and compound with
    // depth; structure errors are ORDERS of magnitude. The band is wide on
    // purpose - the sharp gates below are top-1 and the greedy sequence.
    let mut worst_rel = 0f64;
    let mut worst_layer = 0usize;
    let mut failed = 0usize;
    for li in 0..n_layer {
        let o = &oracle["stage_sums"][li.to_string()];
        let o_abs = o["abs_sum"].as_f64().expect("abs_sum");
        let o_sum = o["sum"].as_f64().expect("sum");
        let rel_abs = (abs_sums[li] - o_abs).abs() / o_abs.max(1e-3);
        let rel_sum = (sums[li] - o_sum).abs() / o_abs.max(1e-3);
        if rel_abs > worst_rel {
            worst_rel = rel_abs;
            worst_layer = li;
        }
        let flag = if rel_abs >= 0.15 || rel_sum >= 0.15 {
            " <-- OUT OF BAND"
        } else {
            ""
        };
        if !flag.is_empty() {
            failed += 1;
        }
        println!(
            "layer {li:2}: abs_sum {:>12.2} vs {o_abs:>12.2} (rel {rel_abs:.4})  sum {:>10.2} vs {o_sum:>10.2}{flag}",
            abs_sums[li], sums[li]
        );
    }
    println!("stage sums: worst abs_sum rel {worst_rel:.4} at layer {worst_layer}");
    assert_eq!(failed, 0, "{failed} layers out of the 15% stage band");

    // ---- greedy decode vs the oracle ids -----------------------------------
    let argmax = |l: &[f32]| -> u32 {
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in l.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        bi as u32
    };
    let mut ours: Vec<u32> = Vec::with_capacity(greedy_ids.len());
    let mut logits = last_logits;
    for step in 0..greedy_ids.len() {
        let tok = argmax(&logits);
        ours.push(tok);
        if step + 1 < greedy_ids.len() {
            logits = model.forward(tok).expect("decode step");
        }
    }
    let matches = ours
        .iter()
        .zip(greedy_ids.iter())
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "greedy: {matches}/{} match\n  ours:   {ours:?}\n  oracle: {greedy_ids:?}",
        greedy_ids.len()
    );
    assert_eq!(
        ours,
        greedy_ids,
        "greedy sequence diverged ({matches}/{} matched)",
        greedy_ids.len()
    );
}
