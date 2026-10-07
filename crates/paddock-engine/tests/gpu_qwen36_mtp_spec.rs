//! MTP speculative decoding on Qwen3.6-27B: greedy spec output must be
//! BIT-IDENTICAL to the plain greedy path (the target verifies every draft, so
//! acceptance rate only affects speed) - and materially faster, since each
//! verify pass amortizes the ~27 GB weight read over the accepted tokens.
//!
//! The base path itself is gated vs b9895 (qwen36_27b_vs_llamacpp), so
//! spec == base == llama transitively.
//!
//! Heavy (~27 GB load): gated on PADDOCK_HEAVY_TESTS, --test-threads=1.

mod common;

use std::time::Instant;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

#[test]
fn spec_decode_matches_base_and_is_faster() {
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

    // "The capital of France is" in the qwen35 vocab (same ids as the 27B gate).
    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369];
    let n = 96usize;

    // base greedy (warm run first so CUDA init doesn't skew timing)
    let _ = m.generate_greedy(&prompt, 8, None).expect("warm");
    let t0 = Instant::now();
    let base = m.generate_greedy(&prompt, n, None).expect("base");
    let base_s = t0.elapsed().as_secs_f64();

    // spec greedy with the MTP head
    for k in [4usize, 8] {
        let t1 = Instant::now();
        let spec = m.generate_greedy_spec(&prompt, n, None, k).expect("spec");
        let spec_s = t1.elapsed().as_secs_f64();
        assert_eq!(
            spec, base,
            "spec (k={k}) output must be bit-identical to base greedy"
        );
        eprintln!(
            "n={n}: base {:.2} tok/s | spec k={k} {:.2} tok/s ({:.2}x)",
            n as f64 / base_s,
            n as f64 / spec_s,
            base_s / spec_s
        );
    }
    eprintln!("EXACT MATCH: spec == base greedy for all k");
}
