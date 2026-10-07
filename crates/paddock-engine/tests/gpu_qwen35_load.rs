//! Loader milestone for the Qwen3.5 hybrid model: open the real 9B GGUF, upload
//! every tensor, and assert the decoded geometry + hybrid layer split. Proves the
//! `qwen35.*` metadata reads, the DeltaNet/full-attn per-layer tensor names, and
//! the 3:1 classification are all correct before the forward graph is wired.
//!
//! Heavy (uploads ~9 GB to the device) - gated on the model file, a built pack,
//! and a CUDA device; run with `--test-threads=1`.

mod common;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

#[test]
fn loads_qwen35_9b_and_geometry_matches() {
    let Some(path) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let map = MappedGguf::open(&path).expect("open gguf");
    assert_eq!(map.gguf().architecture(), Some("qwen35"), "arch string");

    let model = GpuQwen35::load(exec, &map, 4096).expect("load qwen35");
    eprintln!("{}", model.geometry());

    // Qwen3.5-9B ground truth (32 blocks, no MTP block).
    let (full, linear) = model.layer_counts();
    assert_eq!(full, 8, "full-attn layers"); // every 4th of 32
    assert_eq!(linear, 24, "DeltaNet layers");
    assert!(model.vocab > 100_000, "vocab {}", model.vocab);
}

#[test]
fn decode_throughput() {
    let Some(path) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut model = GpuQwen35::load(exec, &map, 4096).expect("load qwen35");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);

    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369];
    let _ = model.generate_greedy(&prompt, 8, None).expect("warmup"); // warm CUDA + alloc

    let n = 200usize;
    let t0 = std::time::Instant::now();
    let out = model.generate_greedy(&prompt, n, None).expect("decode");
    let dt = t0.elapsed();
    assert_eq!(out.len(), n);
    let tok_s = n as f64 / dt.as_secs_f64();
    eprintln!(
        "paddock decode: {n} tokens in {dt:?} = {tok_s:.1} tok/s (incl {} prefill steps)",
        prompt.len()
    );
    eprintln!("(compare against llama.cpp's llama-bench tg128 on the identical GGUF)");

    // prefill throughput (pp) - the int8 MMA prefill path.
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_some() {
        for pp in [128usize, 512] {
            let prompt: Vec<u32> = (0..pp).map(|i| 314 + (i as u32 % 4096)).collect();
            model.reset();
            model.prefill(&prompt).expect("warm prefill");
            let mut best = f64::MAX;
            for _ in 0..5 {
                model.reset();
                let t = std::time::Instant::now();
                model.prefill(&prompt).expect("prefill"); // returns host logits => synced
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "paddock pp{pp}: {:.2} ms = {:.0} tok/s",
                best * 1e3,
                pp as f64 / best
            );
        }
    }
}

/// Profiling harness: pp512 prefill in a loop, nothing else - run under
/// `nsys profile` / ncu for the kernel-time breakdown of a prefill pass.
/// Gated on PADDOCK_PP_PROF so it never runs in a normal test sweep.
#[test]
fn pp512_profile() {
    if std::env::var_os("PADDOCK_PP_PROF").is_none() {
        eprintln!("PADDOCK_PP_PROF not set - skipping");
        return;
    }
    let Some(path) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut model = GpuQwen35::load(exec, &map, 4096).expect("load qwen35");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    let prompt: Vec<u32> = (0..512).map(|i| 314 + (i as u32 % 4096)).collect();
    model.reset();
    model.prefill(&prompt).expect("warm prefill");
    for _ in 0..8 {
        model.reset();
        model.prefill(&prompt).expect("prefill");
    }
}

/// fp8 KV cache drift gate - the qwen sibling of the gpt-oss fp8 gate: decode
/// the same prompt greedily with fp16 then fp8 KV and report the divergence
/// point. fp8 is a lossy opt-in class, so the assertion is only completion +
/// valid ids (drift is EXPECTED); the printed match count is the honest signal.
/// Full attention is 8 of 32 layers here, so drift should be mild.
#[test]
fn kv_fp8_decodes_and_reports_drift() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuQwen35::load(exec, &map, 2048).expect("load qwen35");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    let vocab = m.vocab;

    let prompt: Vec<u32> = vec![760, 6511, 314, 9338, 369]; // robust counting-style
    let steps = 64usize;
    let fp16 = m
        .generate_greedy(&prompt, steps, None)
        .expect("fp16 decode");
    m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp8E4m3);
    let fp8 = m.generate_greedy(&prompt, steps, None).expect("fp8 decode");

    let matched = fp16.iter().zip(&fp8).take_while(|(a, b)| a == b).count();
    let same = fp16.iter().zip(&fp8).filter(|(a, b)| a == b).count();
    eprintln!(
        "fp8 KV drift: {same}/{steps} greedy tokens match fp16 ({matched} before first divergence)"
    );
    assert_eq!(fp8.len(), steps, "fp8 decode did not run to completion");
    assert!(
        fp8.iter().all(|&t| (t as usize) < vocab),
        "fp8 produced invalid token ids"
    );
}
