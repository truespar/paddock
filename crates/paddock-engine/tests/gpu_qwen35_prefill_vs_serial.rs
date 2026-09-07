//! Batched prefill vs the single-stream step, same file, same prompt: the
//! in-engine oracle for a dense lane. The serial `forward_one` walks the
//! GEMV lanes one token at a time; `prefill` runs the same tokens through
//! the batched GEMM lanes (dp4a at <= 64 rows, the W4A8 pipes above). Any
//! gap between the two is OURS - no reference engine, no tokenizer, no
//! sampling in the loop - which is what makes it decisive when the
//! llama.cpp greedy parity shows a spread that a same-weights check
//! should not (2026-09-06: Qwen3.5-9B UD-IQ2_XXS sat at a median 0.13 nats
//! on the winner's logprob where the same model's UD-Q4_K_XL sits at
//! 0.0001). Heavy: `QWEN35_ORACLE_GGUF` names the file; defaults to the
//! dense i-quant demo file, and the elected 9B UD-Q4_K_XL is the control.
//!
//! Two prompt lengths on purpose: 23 tokens stays on the <= 64-row GEMM
//! class, ~110 tokens crosses into the > 64-row class (the pipe GEMMs for
//! k-quant, the dp4a lane for i-quant), so a lane that is only wrong on
//! one side of that boundary still shows.

mod common;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const SHORT: &str = "The quick brown fox jumps over the lazy dog near the riverbank at dawn while";
const LONG_UNIT: &str = "In the beginning the universe was created. This has made a lot of people very angry and been widely regarded as a bad move. ";

fn log_softmax(l: &[f32]) -> Vec<f32> {
    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = l.iter().map(|&v| ((v - m) as f64).exp()).sum();
    let lz = z.ln() as f32;
    l.iter().map(|&v| v - m - lz).collect()
}

fn argmax(l: &[f32]) -> usize {
    l.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum();
    num.sqrt() / den.sqrt().max(1e-12)
}

#[test]
fn batched_prefill_matches_serial_step() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN35_ORACLE_GGUF", common::QWEN35_9B_UD_IQ2) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen35::load(exec, &map, 2048).expect("load");

    let short = tok.encode(SHORT).expect("encode");
    let long: Vec<u32> = {
        let unit = tok.encode(LONG_UNIT).expect("encode");
        // QWEN35_ORACLE_LONG moves the long prompt's length for bisecting a
        // width-dependent lane (64/65/96/...)
        let n = std::env::var("QWEN35_ORACLE_LONG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(110usize);
        unit.iter().cycle().take(n).copied().collect()
    };
    let mut worst_lp = 0f32;
    for (name, prompt) in [("short", &short), ("long", &long)] {
        // serial: one token at a time through the GEMV lanes
        m.reset();
        let mut serial = Vec::new();
        for &t in prompt {
            serial = m.forward_one(t).expect("forward_one");
        }
        // batched: the same tokens as one prefill through the GEMM lanes
        m.reset();
        let batched = m.prefill(prompt).expect("prefill");
        assert_eq!(serial.len(), batched.len());

        let (ls, lb) = (log_softmax(&serial), log_softmax(&batched));
        let (ts, tb) = (argmax(&serial), argmax(&batched));
        let d_top = (ls[ts] - lb[ts]).abs();
        let margin_s = {
            let mut v = ls.clone();
            v.sort_by(|a, b| b.total_cmp(a));
            v[0] - v[1]
        };
        let re = rel_err(&batched, &serial);
        eprintln!(
            "{name:<5} {:>4} tokens: top-1 serial={ts} batched={tb} | winner logprob delta {d_top:.5} nats \
             (serial top-1/top-2 margin {margin_s:.3}) | logits rel_err {re:.2e}",
            prompt.len()
        );
        worst_lp = worst_lp.max(d_top);
        assert_eq!(
            ts, tb,
            "{name}: batched prefill picks a different token than the serial step"
        );
    }
    // the bar is the k-quant lanes' own agreement (9B UD-Q4_K_XL: ~1e-4);
    // a lane that misses it by orders of magnitude is not "in the class",
    // it is wrong somewhere between the window unpack and the reduction
    assert!(
        worst_lp < 2e-2,
        "batched prefill and serial step disagree by {worst_lp:.4} nats on the winner's logprob"
    );
}

/// Batched prefill rate on the oracle file: `prefill` over a 2048-token
/// prompt (the chunk width the serving path feeds the GEMM lanes), best of
/// three after a warm-up. Heavy + ignored:
/// `cargo test ... prefill_rate_bench -- --ignored --nocapture`.
/// `QWEN35_ORACLE_LONG` sets the token count.
#[test]
#[ignore]
fn prefill_rate_bench() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN35_ORACLE_GGUF", common::QWEN35_9B_UD_IQ2) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let n = std::env::var("QWEN35_ORACLE_LONG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048usize);
    let mut m = GpuQwen35::load(exec, &map, n.max(2048)).expect("load");
    let unit = tok.encode(LONG_UNIT).expect("encode");
    let prompt: Vec<u32> = unit.iter().cycle().take(n).copied().collect();
    let mut best = f64::MAX;
    for i in 0..4 {
        m.reset();
        let t = std::time::Instant::now();
        let logits = m.prefill(&prompt).expect("prefill");
        let dt = t.elapsed().as_secs_f64();
        if i > 0 {
            best = best.min(dt);
        }
        eprintln!(
            "prefill {n} tokens: {:.1} ms ({:.0} tok/s), top-1 {}",
            dt * 1e3,
            n as f64 / dt,
            argmax(&logits)
        );
    }
    eprintln!(
        "best: {:.1} ms = {:.0} tok/s over {n} tokens",
        best * 1e3,
        n as f64 / best
    );
}
