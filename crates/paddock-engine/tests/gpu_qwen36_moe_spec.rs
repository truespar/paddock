//! MTP spec gates for the MoE variant (Qwen3.6-35B-A3B). The dense 27B gate
//! asserts spec == base greedy BIT-IDENTICAL and that holds there
//! empirically; on a 256-expert MoE it structurally cannot: the emitted
//! stream is greedy under the VERIFY pass's numeric class (batched mmq/dp4a),
//! base is greedy under the b=1 gemv class, and any hidden-state drift
//! between the two can flip one of the router's top-8 picks - an expert-set
//! change, not an ulp - at a knife-edge token (measured: identical for 43
//! tokens on the France prompt, then one flip, both streams coherent; llama's
//! MTP spec has the same property vs its own b=1 path). What MoE spec can
//! promise, and what this gate pins:
//!   1. determinism - two spec runs are bit-identical;
//!   2. batched spec == single-slot spec (the class-pinned invariant; the
//!      B legs live in gpu_qwen36_spec_batch pointed at this model);
//!   3. a long exact common prefix with base greedy (structural breakage
//!      diverges at token 0-2, class drift takes tens of tokens);
//!   4. completion + valid ids, with the speedup reported.

mod common;

use std::time::Instant;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

#[test]
fn moe_spec_deterministic_prefix_and_faster() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN36_MOE_GGUF", common::QWEN36_35B_A3B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuQwen35::load(exec, &map, 4096).expect("load moe");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    let vocab = m.vocab;

    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369];
    let n = 96usize;

    let _ = m.generate_greedy(&prompt, 8, None).expect("warm");
    let t0 = Instant::now();
    let base = m.generate_greedy(&prompt, n, None).expect("base");
    let base_s = t0.elapsed().as_secs_f64();

    for k in [4usize, 8] {
        let t1 = Instant::now();
        let spec = m.generate_greedy_spec(&prompt, n, None, k).expect("spec");
        let spec_s = t1.elapsed().as_secs_f64();
        let spec2 = m
            .generate_greedy_spec(&prompt, n, None, k)
            .expect("spec rerun");
        assert_eq!(spec, spec2, "spec (k={k}) must be deterministic run-to-run");
        assert_eq!(spec.len(), n, "spec (k={k}) did not run to completion");
        assert!(
            spec.iter().all(|&t| (t as usize) < vocab),
            "invalid token ids"
        );
        let prefix = base.iter().zip(&spec).take_while(|(a, b)| a == b).count();
        eprintln!(
            "k={k}: base {:.2} tok/s | spec {:.2} tok/s ({:.2}x) | exact base prefix {prefix}/{n}",
            n as f64 / base_s,
            n as f64 / spec_s,
            base_s / spec_s
        );
        assert!(
            prefix >= 32,
            "spec (k={k}) diverged from base at token {prefix} - structural breakage, \
             not router-knife-edge class drift"
        );
    }
}
