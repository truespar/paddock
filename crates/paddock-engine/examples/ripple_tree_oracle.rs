//! What would a draft TREE buy Ripple's DFlash2 rounds over the chain? Each
//! prompt runs the greedy chain spec stream (the target's own picks decide
//! every token), and every round keeps the selector's lattice. Afterwards
//! each kept lattice is grown into trees of several shapes and walked against
//! the stream the target actually produced from that round on: the depth it
//! reaches is the tree round's acceptance at that position. A one-child tree
//! must reproduce the chain's drafts (the lattice kernel against the walk).
//!
//! Usage: RIPPLE_DIR=<NVFP4 dir> RIPPLE_DRAFT=<DFlash2 dir> [N_GEN=256]
//!        ripple_tree_oracle   (build with --features static-pack)
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_engine::gpu_model::laguna::dflash2_tree::build_tree;
use paddock_tokenizer::GgufTokenizer;

const PROMPTS: [&str; 4] = [
    "Write a Python class implementing an LRU cache with get/put and O(1) operations, with type hints and docstrings.",
    "Write a Rust function that parses a CSV line with quoted fields, plus unit tests.",
    "Explain how a hash map handles collisions, then implement open addressing in C.",
    "Write a bash script that finds the 10 largest files under a directory and prints their sizes in human units.",
];

/// (nodes with the anchor, children a node, proposal temperature)
const SHAPES: [(usize, usize, f32); 6] = [
    (16, 1, 1.0),
    (32, 16, 1.5),
    (32, 4, 1.5),
    (32, 2, 1.0),
    (24, 16, 1.5),
    (32, 16, 1.0),
];

fn main() {
    let dir = std::env::var("RIPPLE_DIR").expect("set RIPPLE_DIR");
    let ddir = std::env::var("RIPPLE_DRAFT").expect("set RIPPLE_DRAFT");
    let n_gen: usize = std::env::var("N_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let exec = Arc::new(GpuExecutor::with_pack(0, None).expect("executor"));
    let tok = GgufTokenizer::from_hf_dir(dir.as_ref()).expect("tokenizer");
    let mut m = GpuLaguna::load_ripple_nvfp4(exec, dir.as_ref(), 8192).expect("load");
    m.attach_dflash2(ddir.as_ref()).expect("attach");
    m.enable_batch(2).expect("enable_batch");
    let stops = tok.stop_ids();
    let (k, slot) = (32usize, 1usize);
    let mut totals = vec![(0usize, 0usize); SHAPES.len() + 1];
    for (pi, p) in PROMPTS.iter().enumerate() {
        let text = format!(
            "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let prompt = tok.encode(&text).expect("encode");
        let logits = m.forward_prefill(slot, &prompt).expect("prefill");
        let mut pending = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        let mut pos = prompt.len();
        // the stream (the pending token first), and each round's
        // (index of its pending token in the stream, chain accepts, lattice)
        let mut stream = vec![pending];
        let mut rounds: Vec<(usize, usize, Vec<u32>, Vec<f32>, Vec<u32>)> = Vec::new();
        while stream.len() < n_gen {
            assert!(
                m.spec_ensure_warm(slot, &[], (pos - 1) as u32)
                    .expect("warm")
            );
            let drafts = m
                .spec_draft_batch(&[(slot, pending)], 15)
                .expect("draft")
                .and_then(|v| v.into_iter().next())
                .expect("a warm slot drafts");
            let (ids, scores) = m.spec_last_lattice(1).expect("lattice").remove(0);
            let mut chunk = vec![pending];
            chunk.extend(&drafts);
            let picks = m
                .forward_spec_batch(&[(slot, pos, chunk.clone())])
                .expect("verify")
                .expect("the round runs");
            let mut a = 0;
            while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
                a += 1;
            }
            rounds.push((stream.len() - 1, a, ids, scores, drafts));
            stream.extend(&chunk[1..=a]);
            stream.push(picks[a]);
            pending = picks[a];
            pos += a + 1;
            if chunk[..=a].iter().any(|t| stops.contains(t)) || stops.contains(&pending) {
                break;
            }
        }
        println!(
            "prompt {pi}: {} tokens in {} chain rounds ({:.2} a round)",
            stream.len(),
            rounds.len(),
            stream.len() as f64 / rounds.len() as f64
        );
        // rounds whose whole continuation the stream holds (the tail rounds
        // would be scored against a cut-off stream)
        let scored: Vec<_> = rounds.iter().filter(|r| r.0 + 16 < stream.len()).collect();
        let chain: usize = scored.iter().map(|r| r.1 + 1).sum();
        totals[0].0 += chain;
        totals[0].1 += scored.len();
        print!("  chain {:.2}", chain as f64 / scored.len() as f64);
        for (si, &(nodes, children, temp)) in SHAPES.iter().enumerate() {
            let mut sum = 0usize;
            for r in &scored {
                let (at, _, ids, scores, drafts) = r;
                let t = build_tree(ids, scores, stream[*at], k, 15, nodes, children, temp);
                if children == 1 && nodes == 16 {
                    assert_eq!(&t.tokens[1..], &drafts[..], "the 1-child tree is the chain");
                }
                let next = |n: usize| {
                    stream
                        .get(at + t.depth[n] as usize + 1)
                        .copied()
                        .unwrap_or(u32::MAX)
                };
                sum += t.accept(next).len() + 1;
            }
            totals[si + 1].0 += sum;
            totals[si + 1].1 += scored.len();
            print!(
                " | {nodes}n x{children} T{temp}: {:.2}",
                sum as f64 / scored.len() as f64
            );
        }
        println!();
    }
    print!("all: chain {:.2}", totals[0].0 as f64 / totals[0].1 as f64);
    for (si, &(nodes, children, temp)) in SHAPES.iter().enumerate() {
        print!(
            " | {nodes}n x{children} T{temp}: {:.2}",
            totals[si + 1].0 as f64 / totals[si + 1].1 as f64
        );
    }
    println!(" (tokens a round)");
}
