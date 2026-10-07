//! A dense i-quant file end to end: Qwen3.5-9B UD-IQ2_XXS (every projection,
//! the head and the embedding IQ2_XXS / IQ2_S / IQ3_S) through the qwen35
//! walk on the dense i-quant lanes (slot 578). Heavy: `QWEN35_IQ2_GGUF`
//! names the file.
//!
//! Gate: llama.cpp's greedy path on the same file, teacher forced
//! (`QWEN35_IQ2_TOP10` = one line per position, `<chosen> <id>:<logprob>...`
//! from llama-server's `n_probs`): our top-1 equals llama's at most
//! positions and llama's token is never outside our top-5. Without the
//! reference it prints the greedy continuation and the decode rate.

mod common;

use std::time::Instant;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const PROMPT: &str = "The capital of France is";

#[test]
fn iq2_dense_file_serves_at_llama_parity() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN35_IQ2_GGUF", common::QWEN35_9B_UD_IQ2) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_kquant_iq() || !exec.has_kquant_iq_dense() {
        eprintln!("pack lacks the dense i-quant lanes (slots 577/578) - skipping");
        return;
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let t0 = Instant::now();
    let mut m = GpuQwen35::load(exec.clone(), &map, 2048).expect("load dense i-quant file");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    eprintln!("load {:.1}s", t0.elapsed().as_secs_f64());
    let prompt = tok.encode(PROMPT).expect("encode");
    let n = 24usize;
    let t1 = Instant::now();
    let out = m.generate_greedy(&prompt, n, None).expect("generate");
    let gen_s = t1.elapsed().as_secs_f64();
    let out2 = m.generate_greedy(&prompt, n, None).expect("generate 2");
    let dec_s = t1.elapsed().as_secs_f64() - gen_s;
    eprintln!("greedy {:?}", tok.decode(&out, false).unwrap_or_default());
    eprintln!(
        "{n} tokens: {:.1} tok/s first run (prefill + capture), {:.1} tok/s rerun, deterministic={}",
        n as f64 / gen_s,
        n as f64 / dec_s,
        out == out2
    );
    assert_eq!(out, out2, "greedy must be deterministic run-to-run");
    let Ok(top10) = std::env::var("QWEN35_IQ2_TOP10") else {
        eprintln!("QWEN35_IQ2_TOP10 not set - no llama.cpp gate this run");
        return;
    };
    let steps: Vec<(u32, Vec<(u32, f32)>)> = std::fs::read_to_string(&top10)
        .expect("top10 file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace();
            let chosen: u32 = it.next().unwrap().parse().unwrap();
            let tops = it
                .map(|kv| {
                    let (i, lp) = kv.split_once(':').unwrap();
                    (i.parse::<u32>().unwrap(), lp.parse::<f32>().unwrap())
                })
                .collect();
            (chosen, tops)
        })
        .collect();
    let top1 = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b }) as u32
    };
    let rank_of = |v: &[f32], id: u32| v.iter().filter(|&&x| x > v[id as usize]).count();
    // the same walk generate_greedy takes: prefill, then one token at a time
    m.reset();
    let mut logits = m.prefill(&prompt).expect("prefill");
    let (mut agree, mut worst) = (0usize, 0usize);
    for (pos, (chosen, tops)) in steps.iter().enumerate() {
        let ours = top1(&logits);
        let r = rank_of(&logits, *chosen);
        worst = worst.max(r);
        agree += usize::from(ours == *chosen);
        let margin =
            tops.first().map(|t| t.1).unwrap_or(0.0) - tops.get(1).map(|t| t.1).unwrap_or(-99.0);
        eprintln!(
            "pos {pos:2}: llama {chosen:6} {:?} (margin {margin:.2}) | ours {ours:6} {:?}, llama's at rank {r}",
            tok.decode(&[*chosen], false).unwrap_or_default(),
            tok.decode(&[ours], false).unwrap_or_default()
        );
        logits = m.forward_one(*chosen).expect("decode");
    }
    eprintln!(
        "top-1 agreement {agree}/{}, worst rank {worst}",
        steps.len()
    );
    assert!(
        agree * 4 >= steps.len() * 3,
        "top-1 agreement {agree}/{} below 3/4",
        steps.len()
    );
    assert!(worst < 5, "llama's token fell to rank {worst}");
}
