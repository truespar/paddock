//! Recursive's Flint on the laguna body's Flint flavor, from the official
//! NVFP4 GGUF (PADDOCK_HEAVY_TESTS + the file):
//!
//! - `flint_planes_repack_exactly`: a dense plane and an expert stack, read
//!   from the file, repacked to the row layout and uploaded, dequantize on
//!   the device to exactly ggml's block decode times the tensor's factor.
//! - `flint_answers_and_lanes_agree`: a known-answer prompt answers and
//!   stops, and the batched lane's greedy stream tracks the serial lane's (a
//!   divergence must sit on a near-tie of the serial logits - the batched
//!   prefill's tensor-core GEMMs read bf16 rows, the serial GEMVs f32).
//!
//! The same-weights black-box reference is llama.cpp (`flint` since PR
//! #30928) on the identical file, driven over HTTP by the greedy-parity
//! harness rather than from here.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu::PfQ;
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_models::mapped::MappedGguf;
use paddock_models::modelopt::{e2m1_to_f32, e4m3_to_f32};
use paddock_tokenizer::GgufTokenizer;

/// ggml's NVFP4 decode, from the block definition (ggml-common.h): 64
/// elements, four UE4M3 scales, sub-block `s` byte `j` = element `j` low,
/// `j + 8` high; a 0x7F scale reads as zero.
fn ggml_nvfp4(bytes: &[u8], e: usize) -> f32 {
    let blk = &bytes[e / 64 * 36..][..36];
    let (s, m) = ((e % 64) / 16, e % 16);
    let byte = blk[4 + s * 8 + m % 8];
    let code = if m < 8 { byte & 0x0f } else { byte >> 4 };
    let d = if blk[s] == 0x7f {
        0.0
    } else {
        e4m3_to_f32(blk[s])
    };
    e2m1_to_f32(code) * d
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn flint_planes_repack_exactly() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("FLINT_GGUF", common::FLINT_NVFP4) else {
        return;
    };
    let Some(exec) = common::gpu() else { return };
    let map = MappedGguf::open(&path).expect("open gguf");
    // (tensor, its factor tensor, rows checked): q is a dense [4096, 2048]
    // plane; the gate stack's first two experts ride their own factors
    for (name, rows_checked) in [
        ("blk.5.attn_q.weight", 4096usize),
        ("blk.7.ffn_gate_exps.weight", 1024),
    ] {
        let (info, bytes) = map.tensor_bytes(name).expect("tensor");
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        let in_dim = dims[0];
        let rows: usize = dims[1..].iter().product();
        let per_factor = dims[1];
        let base = name.strip_suffix(".weight").expect("weight");
        let (_, sb) = map.tensor_bytes(&format!("{base}.scale")).expect("factor");
        let factors = f32s(sb);
        let (codes, scales, stats) =
            paddock_models::gguf_nvfp4::repack_rows(bytes, rows, in_dim).expect("repack");
        println!(
            "{name}: {rows} x {in_dim}, {} factors, {} NaN scales",
            factors.len(),
            stats.nan_scales
        );
        // a plane per factor block: the dense one whole, the stack's
        // experts one at a time
        let mut checked = 0usize;
        for (fi, &f) in factors.iter().enumerate() {
            if checked >= rows_checked {
                break;
            }
            let r0 = fi * per_factor;
            let n = per_factor.min(rows_checked - checked);
            let c = &codes[r0 * in_dim / 2..(r0 + n) * in_dim / 2];
            let s = &scales[r0 * in_dim / 16..(r0 + n) * in_dim / 16];
            let plane = exec.nvf4_upload(c, s, f, n, in_dim).expect("upload");
            let mut d = exec.alloc(n * in_dim).expect("alloc");
            exec.nvf4_dequant_f32(&plane, &mut d).expect("dequant");
            let got = exec.stream.clone_dtoh(&d).expect("back");
            let src = &bytes[r0 * in_dim / 64 * 36..];
            for (i, g) in got.iter().enumerate() {
                let want = ggml_nvfp4(src, i) * f;
                assert_eq!(
                    g.to_bits(),
                    want.to_bits(),
                    "{name} row {} element {}",
                    r0 + i / in_dim,
                    i % in_dim
                );
            }
            checked += n;
        }
    }
    println!("repacked planes dequantize bit-exactly to ggml's decode");
}

/// The GGUF's tokenizer (pre "flint": digit runs, CJK runs, then the main
/// split) against the checkpoint's own tokenizer.json, on text that
/// exercises each stage. FLINT_HF_DIR names a directory holding Recursive's
/// tokenizer.json (RecursiveSI/flint).
#[test]
fn flint_gguf_tokenizer_matches_hf() {
    let Some(path) = common::model("FLINT_GGUF", common::FLINT_NVFP4) else {
        return;
    };
    let Some(hf) = std::env::var_os("FLINT_HF_DIR").map(std::path::PathBuf::from) else {
        common::missing("FLINT_HF_DIR (official tokenizer.json) unset");
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let g = GgufTokenizer::from_gguf(map.gguf()).expect("gguf tokenizer");
    let h = GgufTokenizer::from_hf_dir(&hf).expect("hf tokenizer");
    let texts = [
        "What is the capital of France?",
        "1234567 apples, 3.14159 and 2026-11-04; v12.0.1 at 08:30",
        "The quick brown fox jumps over the lazy dog.\n\n  Indented   spaces\ttab",
        "fn main() {\n    let v: Vec<u32> = (0..10).map(|i| i * 2).collect();\n}",
        "It's they're we've I'm you'll he'd DON'T",
        "<|user|>Hi<|assistant|><think>plan</think>Hello<tool_call>read<arg_key>path</arg_key><arg_value>a.rs</arg_value></tool_call>",
        "\u{4e2d}\u{6587}\u{6587}\u{672c} \u{3072}\u{3089}\u{304c}\u{306a} \u{30ab}\u{30bf}\u{30ab}\u{30ca} emoji \u{1f426} na\u{ef}ve caf\u{e9} \u{2014}",
    ];
    for t in texts {
        let a = g.encode(t).expect("gguf encode");
        let b = h.encode(t).expect("hf encode");
        assert_eq!(a, b, "tokenizers part on {t:?}");
    }
    println!("gguf tokenizer == tokenizer.json on {} texts", texts.len());
}

fn top2(l: &[f32]) -> (u32, f32) {
    let (mut b, mut bv, mut sv) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            sv = bv;
            bv = v;
            b = i;
        } else if v > sv {
            sv = v;
        }
    }
    (b as u32, bv - sv)
}

#[test]
fn flint_answers_and_lanes_agree() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("FLINT_GGUF", common::FLINT_NVFP4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    assert!(!tok.add_bos, "flint leads with no BOS");
    let stops = tok.stop_ids();
    for id in [150001u32, 150003, 150005, 150018] {
        assert!(stops.contains(&id), "{id} must stop the turn: {stops:?}");
    }
    let mut m = GpuLaguna::load(exec, &map, 4096).expect("load");
    // the template's rendering of one user turn and the generation prompt
    let prompt = tok
        .encode("<|user|>What is the capital of France? Answer in one short sentence.<|assistant|>")
        .expect("encode");
    let n_gen = 400usize;

    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward(t).expect("serial prefill");
    }
    let first_serial = logits.clone();
    let (mut serial, mut margins) = (Vec::new(), Vec::new());
    while serial.len() < n_gen {
        let (t, gap) = top2(&logits);
        serial.push(t);
        margins.push(gap);
        if stops.contains(&t) {
            break;
        }
        logits = m.forward(t).expect("serial decode");
    }
    let text = tok.decode(&serial, false).expect("decode");
    println!("serial ({} tokens): {text:?}", serial.len());
    assert!(
        text.contains("Paris"),
        "serial lane does not answer: {text:?}"
    );
    assert!(
        serial.last().is_some_and(|t| stops.contains(t)),
        "serial lane never stopped within {n_gen} tokens"
    );

    m.enable_batch(2).expect("enable_batch");
    let mut logits = m.forward_prefill(0, &prompt).expect("batched prefill");
    let dmax = first_serial
        .iter()
        .zip(&logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("first-token logits: serial vs batched max |diff| {dmax:.4}");
    let mut batched = Vec::new();
    while batched.len() < serial.len() {
        let (t, _) = top2(&logits);
        batched.push(t);
        if stops.contains(&t) {
            break;
        }
        let pos = (prompt.len() + batched.len() - 1) as u32;
        logits = m.forward_batch(&[t], &[pos]).expect("batched decode");
    }
    let btext = tok.decode(&batched, false).expect("decode");
    println!("batched ({} tokens): {btext:?}", batched.len());
    assert!(
        btext.contains("Paris"),
        "batched lane does not answer: {btext:?}"
    );
    match (0..serial.len().min(batched.len())).find(|&i| serial[i] != batched[i]) {
        Some(i) => {
            println!("lanes part at token {i} (serial margin {:.4})", margins[i]);
            assert!(
                margins[i] < 1.0,
                "lanes part at token {i} on a {:.3} margin - not a near-tie",
                margins[i]
            );
        }
        None => {
            assert_eq!(serial.len(), batched.len(), "one lane stopped early");
            println!(
                "lanes agree on all {} tokens through the stop",
                serial.len()
            );
        }
    }
}

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

/// Random NVFP4 rows: every code, unsigned e4m3 block scales in [2^-3, 2).
fn nv_bytes(rows: usize, in_dim: usize, seed: &mut u64) -> (Vec<u8>, Vec<u8>) {
    let codes = (0..rows * in_dim / 2).map(|_| lcg(seed) as u8).collect();
    let scales = (0..rows * in_dim / 16)
        .map(|_| 0x20 + (lcg(seed) % 0x20) as u8)
        .collect();
    (codes, scales)
}

/// The W16 expert pair on row-major planes (slots 918-919) against the
/// W4A16 GEMVs on the same planes: the same function in another activation
/// class (bf16 rows on tensor cores vs f32 rows on FFMA), so close, and
/// BATCH-INVARIANT - a token's output is the same bits alone or among 40
/// rows. Synthetic planes at Flint's geometry, 16 experts.
#[test]
fn flint_w16r_pair_tracks_the_gemvs_and_is_batch_invariant() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_nvf4_moe_w16r() || !exec.has_q4x_moe_gu_swiglu() {
        common::missing("pack has no row-plane W16 pair (slots 918-919)");
        return;
    }
    let (embd, ff, ffs, ne, k, rows) = (2048usize, 512usize, 1024usize, 16usize, 6usize, 40usize);
    let mut seed = 0x5eed_u64;
    let s2 = |n: usize, seed: &mut u64| -> Vec<f32> {
        (0..n)
            .map(|_| 0.002 + (lcg(seed) % 1000) as f32 * 1e-5)
            .collect()
    };
    let moe = |rows_per: usize, in_dim: usize, seed: &mut u64| {
        let (c, s) = nv_bytes(ne * rows_per, in_dim, seed);
        exec.nvf4_moe_upload(&c, &s, &s2(ne, seed), ne, rows_per, in_dim)
            .expect("moe plane")
    };
    let dense = |out: usize, in_dim: usize, seed: &mut u64| {
        let (c, s) = nv_bytes(out, in_dim, seed);
        exec.nvf4_upload(&c, &s, s2(1, seed)[0], out, in_dim)
            .expect("plane")
    };
    let (gate, up, down) = (
        moe(ff, embd, &mut seed),
        moe(ff, embd, &mut seed),
        moe(embd, ff, &mut seed),
    );
    let (sg, su, sd) = (
        dense(ffs, embd, &mut seed),
        dense(ffs, embd, &mut seed),
        dense(embd, ffs, &mut seed),
    );
    let x: Vec<f32> = (0..rows * embd)
        .map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    // k distinct experts per row, weights summing to one
    let mut idx = Vec::with_capacity(rows * k);
    let mut wts = Vec::with_capacity(rows * k);
    for _ in 0..rows {
        let mut picks: Vec<u32> = Vec::new();
        while picks.len() < k {
            let e = lcg(&mut seed) % ne as u32;
            if !picks.contains(&e) {
                picks.push(e);
            }
        }
        let raw: Vec<f32> = (0..k)
            .map(|_| 0.1 + (lcg(&mut seed) % 100) as f32 / 100.0)
            .collect();
        let sum: f32 = raw.iter().sum();
        idx.extend(picks);
        wts.extend(raw.iter().map(|v| v / sum));
    }

    // the W16 pair over `n` rows starting at row `r0`
    let w16 = |r0: usize, n: usize| -> Vec<f32> {
        let d_x = exec.to_device(&x[r0 * embd..(r0 + n) * embd]).expect("x");
        let d_idx = exec.to_device_u32(&idx[r0 * k..(r0 + n) * k]).expect("idx");
        let d_w = exec.to_device(&wts[r0 * k..(r0 + n) * k]).expect("w");
        let nb = (n * k + ne * 31).div_ceil(32).min(n * k);
        let mut srow = exec.alloc_u32(nb * 32).expect("srow");
        let mut sslot = exec.alloc_u32(nb * 32).expect("sslot");
        let mut bexp = exec.alloc_u32(nb).expect("bexp");
        exec.moe_align(&d_idx, &mut srow, &mut sslot, &mut bexp, n, k, ne, nb)
            .expect("align");
        let mut x16 = exec.stream_alloc_bf16(n * embd).expect("x16");
        exec.convert_f32_bf16(&d_x, &mut x16, n * embd)
            .expect("cast");
        let mut act = exec.stream_alloc_bf16(n * (k * ff + ffs)).expect("act");
        exec.nvf4_moe_gu_swiglu_w16r(
            &gate, &up, &sg, &su, &srow, &sslot, &bexp, &x16, &mut act, k, nb, n,
        )
        .expect("gu");
        let mut part = exec.alloc(n * (k + 1) * embd).expect("part");
        exec.nvf4_moe_down_part_w16r(
            &down, &sd, &srow, &sslot, &bexp, &d_w, &act, &mut part, k, nb, n,
        )
        .expect("down");
        let mut y = exec.alloc(n * embd).expect("y");
        exec.moe_slot_combine_init_at(&part, &mut y, 0, embd, k + 1, n)
            .expect("fold");
        exec.stream.clone_dtoh(&y).expect("back")
    };
    let all = w16(0, rows);

    // the W4A16 GEMV path on the same planes (f32 rows)
    let d_x = exec.to_device(&x).expect("x");
    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_w = exec.to_device(&wts).expect("w");
    let mut fused = exec.alloc(rows * k * ff).expect("fused");
    exec.q4x_moe_gu_swiglu(&gate, &up, &d_idx, &d_x, &mut fused, k, rows)
        .expect("gemv gu");
    let mut y = exec.alloc(rows * embd).expect("y");
    exec.nvf4_moe_down_acc(&down, &d_idx, &d_w, &fused, &mut y, None, k, rows, false)
        .expect("gemv down");
    let (mut g, mut u) = (
        exec.alloc(rows * ffs).expect("g"),
        exec.alloc(rows * ffs).expect("u"),
    );
    let mut sh = exec.alloc(rows * embd).expect("sh");
    exec.nvf4_gemv_batch(&sg, &d_x, &mut g, None, rows)
        .expect("sh gate");
    exec.nvf4_gemv_batch(&su, &d_x, &mut u, None, rows)
        .expect("sh up");
    exec.swiglu(&mut g, &u, rows * ffs).expect("swiglu");
    exec.nvf4_gemv_batch(&sd, &g, &mut sh, None, rows)
        .expect("sh down");
    exec.add(&mut y, &sh, rows * embd).expect("add");
    let gemv = exec.stream.clone_dtoh(&y).expect("back");

    let rms = (gemv.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / gemv.len() as f64).sqrt();
    let max_err = all
        .iter()
        .zip(&gemv)
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0f64, f64::max);
    println!(
        "W16 pair vs GEMVs: max |diff| {max_err:.3e} at rms {rms:.3e} ({:.2e} rel)",
        max_err / rms
    );
    assert!(
        max_err / rms < 0.05,
        "W16 pair strays from the GEMV class: {max_err:.3e} vs rms {rms:.3e}"
    );

    // batch invariance: every token alone, and a ragged 7-row window
    for t in 0..rows {
        let one = w16(t, 1);
        let want = &all[t * embd..(t + 1) * embd];
        assert!(
            one.iter()
                .zip(want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "row {t} alone differs from row {t} among {rows}"
        );
    }
    let win = w16(11, 7);
    assert!(
        win.iter()
            .zip(&all[11 * embd..18 * embd])
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "a 7-row window differs from the same rows among {rows}"
    );
    println!("W16 pair: every row bit-identical alone, in 7 and in {rows}");
}

/// A prompt that crosses the 513-key window four times over and asks for a
/// fact from its start: the batched prefill (paged SWA rings, the prefill
/// attention arms with Flint's sinks, the W16 expert pair) against the serial
/// lane's decode-kernel walk. Both must pick the same next token and then
/// answer the question.
#[test]
fn flint_long_prefill_tracks_serial() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("FLINT_GGUF", common::FLINT_NVFP4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuLaguna::load(exec, &map, 4096).expect("load");
    let mut text = String::from("<|user|>Here is a shipping log:\n");
    for i in 0..120 {
        text.push_str(&format!(
            "Entry {i}: parcel P-{} from Ulm to Kiel, weight {} kg.\n",
            1000 + 37 * i,
            1 + i % 17
        ));
    }
    text.push_str(
        "What was the weight of entry 7? Answer with the number and unit only.<|assistant|>",
    );
    let prompt = tok.encode(&text).expect("encode");
    assert!(prompt.len() > 4 * 513, "prompt {} tokens", prompt.len());
    let stops = tok.stop_ids();
    let n_gen = 200usize;

    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward(t).expect("serial prefill");
    }
    let first = logits.clone();
    let mut serial = Vec::new();
    while serial.len() < n_gen {
        let (t, _) = top2(&logits);
        serial.push(t);
        if stops.contains(&t) {
            break;
        }
        logits = m.forward(t).expect("serial decode");
    }
    m.enable_batch(2).expect("enable_batch");
    let mut logits = m.forward_prefill(0, &prompt).expect("batched prefill");
    let (ts, gap) = top2(&first);
    let (tb, _) = top2(&logits);
    let dmax = first
        .iter()
        .zip(&logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "{} tokens: serial top1 {ts} (margin {gap:.3}) batched top1 {tb}, max |diff| {dmax:.4}",
        prompt.len()
    );
    assert_eq!(ts, tb, "the lanes pick different first tokens");
    let mut batched = Vec::new();
    while batched.len() < n_gen {
        let (t, _) = top2(&logits);
        batched.push(t);
        if stops.contains(&t) {
            break;
        }
        let pos = (prompt.len() + batched.len() - 1) as u32;
        logits = m.forward_batch(&[t], &[pos]).expect("batched decode");
    }
    let (st, bt) = (
        tok.decode(&serial, false).expect("decode"),
        tok.decode(&batched, false).expect("decode"),
    );
    println!("serial: {st:?}\nbatched: {bt:?}");
    let answer = |s: &str| s.rsplit("</think>").next().unwrap_or(s).contains("8 kg");
    assert!(answer(&st), "serial lane misses entry 7's weight: {st:?}");
    assert!(answer(&bt), "batched lane misses entry 7's weight: {bt:?}");
}

/// ||a - b|| / ||b||.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den).sqrt()
}

/// max |a - b| over the rms of `b`.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let rms = (b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / b.len() as f64).sqrt();
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() as f64)
        .fold(0.0f64, f64::max)
        / rms
}

/// The W4A16 prefill tile (slots 920-923) on synthetic planes at Flint's
/// geometry: each task within the bf16 class of the f32 GEMVs on the same
/// planes, and row-invariant - a token's bits are the same alone, in a
/// ragged window and in a full chunk.
#[test]
fn flint_prefill_tile_tracks_the_gemvs_and_is_row_invariant() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_nvf4_pf() || !exec.has_q4x_moe_gu_swiglu() {
        common::missing("pack has no W4A16 prefill tile (slots 920-923)");
        return;
    }
    let (embd, qd, ff, ffs, ne, k) = (2048usize, 4096usize, 512usize, 1024usize, 16usize, 6usize);
    let mut seed = 0x9e37_u64;
    let s2 = |seed: &mut u64| 0.002 + (lcg(seed) % 1000) as f32 * 1e-5;
    let dense = |out: usize, in_dim: usize, seed: &mut u64| {
        let (c, s) = nv_bytes(out, in_dim, seed);
        exec.nvf4_upload(&c, &s, s2(seed), out, in_dim)
            .expect("plane")
    };
    let moe = |rows_per: usize, in_dim: usize, seed: &mut u64| {
        let (c, s) = nv_bytes(ne * rows_per, in_dim, seed);
        let f: Vec<f32> = (0..ne).map(|_| s2(seed)).collect();
        exec.nvf4_moe_upload(&c, &s, &f, ne, rows_per, in_dim)
            .expect("moe plane")
    };
    let rows = 300usize;
    let x: Vec<f32> = (0..rows * qd)
        .map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    let bf = |v: &[f32]| {
        let d = exec.to_device(v).expect("x");
        let mut h = exec.stream_alloc_bf16(v.len()).expect("x16");
        exec.convert_f32_bf16(&d, &mut h, v.len()).expect("cast");
        (d, h)
    };

    // 920 on the q plane (in 2048 -> out 4096) and the o plane (4096 -> 2048)
    for (out, in_dim) in [(qd, embd), (embd, qd)] {
        let w = dense(out, in_dim, &mut seed);
        let xs = &x[..rows * in_dim];
        let (d_x, x16) = bf(xs);
        let mut y = exec.alloc(rows * out).expect("y");
        exec.nvf4_pf_gemm(&w, &x16, in_dim, 0, &mut y, out, 0, rows)
            .expect("pf gemm");
        let got = exec.stream.clone_dtoh(&y).expect("back");
        let mut yr = exec.alloc(rows * out).expect("yr");
        exec.nvf4_gemv_batch(&w, &d_x, &mut yr, None, rows)
            .expect("gemv");
        let want = exec.stream.clone_dtoh(&yr).expect("back");
        let e = rel(&got, &want);
        println!(
            "920 [{out}, {in_dim}] x {rows}: max|diff|/rms {e:.2e}, l2 {:.2e}",
            rel_l2(&got, &want)
        );
        assert!(
            rel_l2(&got, &want) < 1e-2 && e < 0.03,
            "920 strays from the GEMV class: {e:.2e}"
        );
        // a 13-row window alone
        let (_, x16w) = bf(&xs[7 * in_dim..20 * in_dim]);
        let mut yw = exec.alloc(13 * out).expect("yw");
        exec.nvf4_pf_gemm(&w, &x16w, in_dim, 0, &mut yw, out, 0, 13)
            .expect("window");
        let gw = exec.stream.clone_dtoh(&yw).expect("back");
        assert!(
            gw.iter()
                .zip(&got[7 * out..20 * out])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "920 rows differ between a 13-row window and {rows} rows"
        );
    }

    // 921 on a shared-expert pair (2048 -> 1024), strided into a wider plane
    let (sg, su) = (dense(ffs, embd, &mut seed), dense(ffs, embd, &mut seed));
    let xs = &x[..rows * embd];
    let (d_x, x16) = bf(xs);
    let (lda, aoff) = (ffs + 64, 32);
    let mut act = exec.stream_alloc_bf16(rows * lda).expect("act");
    exec.nvf4_pf_gu(&sg, &su, &x16, embd, &mut act, lda, aoff, rows)
        .expect("pf gu");
    let mut act_f = exec.alloc(rows * lda).expect("act f32");
    exec.convert_bf16_f32(&act, &mut act_f, rows * lda)
        .expect("widen");
    let a = exec.stream.clone_dtoh(&act_f).expect("back");
    let got: Vec<f32> = (0..rows)
        .flat_map(|t| a[t * lda + aoff..t * lda + aoff + ffs].to_vec())
        .collect();
    let (mut g, mut u) = (
        exec.alloc(rows * ffs).expect("g"),
        exec.alloc(rows * ffs).expect("u"),
    );
    exec.nvf4_gemv_batch(&sg, &d_x, &mut g, None, rows)
        .expect("g");
    exec.nvf4_gemv_batch(&su, &d_x, &mut u, None, rows)
        .expect("u");
    exec.swiglu(&mut g, &u, rows * ffs).expect("swiglu");
    let want = exec.stream.clone_dtoh(&g).expect("back");
    let (e, l2) = (rel(&got, &want), rel_l2(&got, &want));
    println!("921 [{ffs}, {embd}] x {rows}: max|diff|/rms {e:.2e}, l2 {l2:.2e}");
    // a SwiGLU product's tail sets the max; the bf16 class shows in the norm
    assert!(l2 < 1e-2, "921 strays from the GEMV class: l2 {l2:.2e}");

    // 922 / 923 + the shared expert: the whole prefill MoE against the GEMVs
    let (gate, up, down) = (
        moe(ff, embd, &mut seed),
        moe(ff, embd, &mut seed),
        moe(embd, ff, &mut seed),
    );
    let sd = dense(embd, ffs, &mut seed);
    let mut idx = Vec::with_capacity(rows * k);
    let mut wts = Vec::with_capacity(rows * k);
    for _ in 0..rows {
        let mut picks: Vec<u32> = Vec::new();
        while picks.len() < k {
            let e = lcg(&mut seed) % ne as u32;
            if !picks.contains(&e) {
                picks.push(e);
            }
        }
        let raw: Vec<f32> = (0..k)
            .map(|_| 0.1 + (lcg(&mut seed) % 100) as f32 / 100.0)
            .collect();
        let sum: f32 = raw.iter().sum();
        idx.extend(picks);
        wts.extend(raw.iter().map(|v| v / sum));
    }
    let aw = k * ff + ffs;
    let moe_pf = |r0: usize, n: usize| -> Vec<f32> {
        let (_, x16) = bf(&x[r0 * embd..(r0 + n) * embd]);
        let d_idx = exec.to_device_u32(&idx[r0 * k..(r0 + n) * k]).expect("idx");
        let d_w = exec.to_device(&wts[r0 * k..(r0 + n) * k]).expect("w");
        let nb = (n * k + ne * 63).div_ceil(64).min(n * k);
        let mut srow = exec.alloc_u32(nb * 64).expect("srow");
        let mut sslot = exec.alloc_u32(nb * 64).expect("sslot");
        let mut bexp = exec.alloc_u32(nb).expect("bexp");
        exec.moe_align_bm(&d_idx, &mut srow, &mut sslot, &mut bexp, n, k, ne, 64, nb)
            .expect("align");
        let mut act = exec.stream_alloc_bf16(n * aw).expect("act");
        exec.nvf4_pf_moe_gu(&gate, &up, &srow, &sslot, &bexp, &x16, &mut act, aw, nb)
            .expect("moe gu");
        exec.nvf4_pf_gu(&sg, &su, &x16, embd, &mut act, aw, k * ff, n)
            .expect("shared gu");
        let mut y = exec.alloc(n * embd).expect("y");
        exec.nvf4_pf_gemm(&sd, &act, aw, k * ff, &mut y, embd, 0, n)
            .expect("shared dn");
        // bf16 partials in an f32-typed buffer, the fold's storage
        let mut part = exec.alloc((n * k * embd).div_ceil(2)).expect("part");
        exec.nvf4_pf_moe_dn(
            &down, &srow, &sslot, &bexp, &d_w, &act, aw, &mut part, k, nb,
        )
        .expect("moe dn");
        exec.moe_slot_combine_bf16(&part, &mut y, embd, k, n)
            .expect("fold");
        exec.stream.clone_dtoh(&y).expect("back")
    };
    let all = moe_pf(0, rows);
    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_w = exec.to_device(&wts).expect("w");
    let mut fused = exec.alloc(rows * k * ff).expect("fused");
    exec.q4x_moe_gu_swiglu(&gate, &up, &d_idx, &d_x, &mut fused, k, rows)
        .expect("gemv gu");
    let mut y = exec.alloc(rows * embd).expect("y");
    exec.nvf4_moe_down_acc(&down, &d_idx, &d_w, &fused, &mut y, None, k, rows, false)
        .expect("gemv down");
    let mut sh = exec.alloc(rows * embd).expect("sh");
    exec.nvf4_gemv_batch(&sg, &d_x, &mut g, None, rows)
        .expect("g");
    exec.nvf4_gemv_batch(&su, &d_x, &mut u, None, rows)
        .expect("u");
    exec.swiglu(&mut g, &u, rows * ffs).expect("swiglu");
    exec.nvf4_gemv_batch(&sd, &g, &mut sh, None, rows)
        .expect("sh down");
    exec.add(&mut y, &sh, rows * embd).expect("add");
    let want = exec.stream.clone_dtoh(&y).expect("back");
    let e = rel(&all, &want);
    println!(
        "922/923 MoE x {rows}: max|diff|/rms {e:.2e}, l2 {:.2e}",
        rel_l2(&all, &want)
    );
    assert!(
        rel_l2(&all, &want) < 1e-2,
        "the prefill MoE strays from the GEMV class: {e:.2e}"
    );
    for (r0, n) in [(0usize, 1usize), (41, 1), (17, 9), (250, 50)] {
        let w = moe_pf(r0, n);
        assert!(
            w.iter()
                .zip(&all[r0 * embd..(r0 + n) * embd])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "MoE rows {r0}..{} differ alone vs among {rows}",
            r0 + n
        );
    }
    println!("prefill tile: every task row-invariant");
}

/// Bit-equality of two device planes' host copies.
fn same_bits<T: Copy + PartialEq>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y)
}

#[test]
fn flint_prefill_tails_and_bf16_attention_keep_the_chains_bits() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_add_norm16() || !exec.has_attn_prefill_f16_paged_o16() {
        common::missing("pack has no prefill tails / bf16 attention (slots 924-925)");
        return;
    }
    let (n, k, eps) = (2048usize, 6usize, 1e-6f32);
    let mut seed = 0x51ed_u64;
    let mut rnd = |len: usize, amp: f32| -> Vec<f32> {
        (0..len)
            .map(|_| ((lcg(&mut seed) % 20001) as f32 / 10000.0 - 1.0) * amp)
            .collect()
    };
    let w = exec.to_device(&rnd(n, 1.5)).expect("w");
    let bits = |d: &cudarc::driver::CudaSlice<f32>| -> Vec<u32> {
        exec.stream
            .clone_dtoh(d)
            .expect("back")
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let bits16 = |d: &cudarc::driver::CudaSlice<half::bf16>| -> Vec<u16> {
        exec.stream
            .clone_dtoh(d)
            .expect("back")
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    // 3 rows ride the decode width, 64 / 300 / 2048 the wide one
    for rows in [3usize, 64, 300, 2048] {
        let x0 = rnd(rows * n, 4.0);
        let proj = exec.to_device(&rnd(rows * n, 2.0)).expect("proj");
        // routed partials as bf16 pairs in an f32-typed plane
        let part16: Vec<f32> = rnd(rows * k * n / 2, 1.0)
            .iter()
            .map(|v| {
                let h = half::bf16::from_f32(*v).to_bits() as u32;
                let l = half::bf16::from_f32(-v * 0.37).to_bits() as u32;
                f32::from_bits(l | (h << 16))
            })
            .collect();
        let part = exec.to_device(&part16).expect("part");
        // mode 0 against add_rmsnorm_batch + the cast
        let mut xa = exec.to_device(&x0).expect("x");
        let mut xna = exec.alloc(rows * n).expect("xn");
        let mut ha = exec.stream_alloc_bf16(rows * n).expect("h");
        exec.add_rmsnorm_batch(&mut xa, &proj, &w, &mut xna, n, eps, rows)
            .expect("chain");
        exec.convert_f32_bf16(&xna, &mut ha, rows * n)
            .expect("cast");
        let mut xb = exec.to_device(&x0).expect("x");
        let mut xnb = exec.alloc(rows * n).expect("xn");
        let mut hb = exec.stream_alloc_bf16(rows * n).expect("h");
        let ran = exec
            .add_norm16(
                &mut xb,
                &proj,
                None,
                &w,
                Some(&mut xnb),
                Some(&mut hb),
                n,
                eps,
                rows,
                false,
            )
            .expect("924 mode 0");
        assert!(ran, "924 declined mode 0 at {rows} rows");
        assert!(
            same_bits(&bits(&xa), &bits(&xb)),
            "mode 0 x differs at {rows} rows"
        );
        assert!(
            same_bits(&bits(&xna), &bits(&xnb)),
            "mode 0 xn differs at {rows} rows"
        );
        assert!(
            same_bits(&bits16(&ha), &bits16(&hb)),
            "mode 0 bf16 differs at {rows} rows"
        );
        // mode 1 against the bf16 fold + add + rmsnorm_batch + the cast, with
        // the partials and without
        for kk in [k, 0] {
            let mut xa = exec.to_device(&x0).expect("x");
            let mut pa = exec.alloc(rows * n).expect("p");
            exec.stream.memcpy_dtod(&proj, &mut pa).expect("copy");
            if kk > 0 {
                exec.moe_slot_combine_bf16(&part, &mut pa, n, kk, rows)
                    .expect("fold");
            }
            exec.add(&mut xa, &pa, rows * n).expect("add");
            let mut xna = exec.alloc(rows * n).expect("xn");
            exec.rmsnorm_batch(&xa, &w, &mut xna, n, eps, rows)
                .expect("norm");
            let mut ha = exec.stream_alloc_bf16(rows * n).expect("h");
            exec.convert_f32_bf16(&xna, &mut ha, rows * n)
                .expect("cast");
            let mut xb = exec.to_device(&x0).expect("x");
            let mut xnb = exec.alloc(rows * n).expect("xn");
            let mut hb = exec.stream_alloc_bf16(rows * n).expect("h");
            let p = (kk > 0).then_some((&part, kk));
            let ran = exec
                .add_norm16(
                    &mut xb,
                    &proj,
                    p,
                    &w,
                    Some(&mut xnb),
                    Some(&mut hb),
                    n,
                    eps,
                    rows,
                    true,
                )
                .expect("924 mode 1");
            assert!(ran, "924 declined mode 1 at {rows} rows");
            assert!(
                same_bits(&bits(&xa), &bits(&xb)),
                "mode 1 (k {kk}) x differs at {rows} rows"
            );
            assert!(
                same_bits(&bits(&xna), &bits(&xnb)),
                "mode 1 (k {kk}) xn differs at {rows} rows"
            );
            assert!(
                same_bits(&bits16(&ha), &bits16(&hb)),
                "mode 1 (k {kk}) bf16 differs at {rows} rows"
            );
        }
    }
    println!("924: both tails bit-identical to their chains at 3 / 64 / 300 / 2048 rows");

    // 925 against the f32 entry + the cast: Flint's G = 8, hd 128, an f16
    // pool, learned sinks; a full layer over a whole chunk and an SWA span
    // at a row offset inside it
    let (nh, n_kv, hd) = (32usize, 4usize, 128usize);
    let (kv_dim, rows) = (n_kv * hd, 900usize);
    let nblk = rows.div_ceil(16);
    let mut pool = |amp: f32| -> cudarc::driver::CudaSlice<u8> {
        let v: Vec<u8> = rnd(nblk * 16 * kv_dim, amp)
            .iter()
            .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
            .collect();
        exec.to_device_u8(&v).expect("pool")
    };
    let (pk, pv) = (pool(1.0), pool(1.0));
    let q = exec.to_device(&rnd(rows * nh * hd, 2.0)).expect("q");
    let sinks = exec.to_device(&rnd(nh, 1.0)).expect("sinks");
    let pos: Vec<u32> = (0..rows as u32).collect();
    let pos = exec.to_device_u32(&pos).expect("pos");
    let slots = exec.to_device_u32(&vec![0u32; rows]).expect("slots");
    let bt: Vec<u32> = (0..nblk as u32).rev().collect();
    let bt = exec.to_device_u32(&bt).expect("bt");
    let scale = 1.0 / (hd as f32).sqrt();
    let kv16 = paddock_engine::gpu::KvDtype::Fp16;
    for (window, off, len) in [(0usize, 0usize, rows), (513, 200, 300)] {
        let mut out = exec.alloc(rows * nh * hd).expect("out");
        exec.attn_prefill_f16_paged_at(
            &q, &pk, &pv, &sinks, &mut out, &pos, &slots, off, &bt, nblk, nh, n_kv, hd, kv_dim,
            window, len, scale, kv16,
        )
        .expect("f32 attention");
        let mut ha = exec.stream_alloc_bf16(rows * nh * hd).expect("h");
        exec.convert_f32_bf16(&out, &mut ha, rows * nh * hd)
            .expect("cast");
        let zeros = vec![half::bf16::ZERO; rows * nh * hd];
        let mut hb = exec.to_device_bf16(&zeros).expect("h");
        let ran = exec
            .attn_prefill_f16_paged_o16_at(
                PfQ::F32(&q),
                &pk,
                &pv,
                &sinks,
                &mut hb,
                &pos,
                &slots,
                off,
                &bt,
                nblk,
                nh,
                n_kv,
                hd,
                kv_dim,
                window,
                len,
                scale,
                kv16,
            )
            .expect("925");
        assert!(ran, "925 declined Flint's shape");
        let (a, b) = (bits16(&ha), bits16(&hb));
        let span = off * nh * hd..(off + len) * nh * hd;
        assert!(
            same_bits(&a[span.clone()], &b[span.clone()]),
            "925 differs from the f32 output's cast (window {window}, rows {off}..{})",
            off + len
        );
        assert!(
            b[..span.start]
                .iter()
                .chain(&b[span.end..])
                .all(|&v| v == 0),
            "925 wrote outside rows {off}..{}",
            off + len
        );
    }
    println!("925: bf16 attention rows are the f32 output's cast, full and SWA");

    // 929 + 930 against the f32 norm + rope and 925: q handed over in f16
    // pre-scaled must land the same bf16 rows, and k the same f32 bits - a
    // roped (SWA) layer and a NoPE one (freq_scale 0)
    if !exec.has_qkv_norm_rope_batch_q16() || !exec.has_attn_prefill_f16_paged_q16() {
        common::missing("pack has no f16-q norm + rope / attention (slots 929-930)");
        return;
    }
    assert!(
        exec.attn_prefill_q16_ready(nh, n_kv, hd, kv16),
        "930's probe refused Flint's shape"
    );
    let qraw = exec.to_device(&rnd(rows * nh * hd, 3.0)).expect("q");
    let kraw = exec.to_device(&rnd(rows * kv_dim, 3.0)).expect("k");
    let (qw, kw) = (
        exec.to_device(&rnd(hd, 1.5)).expect("qw"),
        exec.to_device(&rnd(hd, 1.5)).expect("kw"),
    );
    let ts = 10000f32.powf(-2.0 / hd as f32);
    for (window, rp) in [
        (513usize, (ts, 1.0, 0.0, 0.0, 0.0, 1.0)),
        (0, (ts, 0.0, 0.0, 0.0, 0.0, 1.0)),
    ] {
        let mut v = exec.to_device(&rnd(rows * kv_dim, 1.0)).expect("v");
        let mut qn = exec.alloc(rows * nh * hd).expect("qn");
        let mut kn = exec.alloc(rows * kv_dim).expect("kn");
        exec.qk_norm_rope_batch(
            &qraw, &kraw, &mut v, &qw, &kw, &mut qn, &mut kn, &pos, nh, n_kv, hd, eps, rp, rows,
        )
        .expect("f32 norm + rope");
        let mut ha = exec.stream_alloc_bf16(rows * nh * hd).expect("h");
        let ran = exec
            .attn_prefill_f16_paged_o16_at(
                PfQ::F32(&qn),
                &pk,
                &pv,
                &sinks,
                &mut ha,
                &pos,
                &slots,
                0,
                &bt,
                nblk,
                nh,
                n_kv,
                hd,
                kv_dim,
                window,
                rows,
                scale,
                kv16,
            )
            .expect("925");
        assert!(ran, "925 declined Flint's shape");
        let mut q16 = exec.alloc_f16(rows * nh * hd).expect("q16");
        let mut kn2 = exec.alloc(rows * kv_dim).expect("kn");
        let ran = exec
            .qk_norm_rope_batch_q16(
                &qraw,
                &kraw,
                &mut v,
                &qw,
                &kw,
                &mut q16,
                &mut kn2,
                &pos,
                (nh, n_kv, hd),
                eps,
                rp,
                (0, rows),
                scale * std::f32::consts::LOG2_E,
            )
            .expect("929");
        assert!(ran, "929 declined Flint's shape");
        assert!(
            same_bits(&bits(&kn), &bits(&kn2)),
            "929's k differs from the f32 norm + rope (window {window})"
        );
        let mut hb = exec.stream_alloc_bf16(rows * nh * hd).expect("h");
        let ran = exec
            .attn_prefill_f16_paged_o16_at(
                PfQ::F16(&q16),
                &pk,
                &pv,
                &sinks,
                &mut hb,
                &pos,
                &slots,
                0,
                &bt,
                nblk,
                nh,
                n_kv,
                hd,
                kv_dim,
                window,
                rows,
                scale,
                kv16,
            )
            .expect("930");
        assert!(ran, "930 declined Flint's shape");
        assert!(
            same_bits(&bits16(&ha), &bits16(&hb)),
            "930 on 929's q differs from 925 on the f32 q (window {window})"
        );
    }
    println!("929 + 930: the f16 q hand-over lands 925's bf16 rows and the same k");
}

#[test]
fn flint_decode_planes_fuse_bit_exactly() {
    let Some(exec) = common::gpu() else { return };
    let mut seed = 0x3c4d_u64;
    let s2 = |seed: &mut u64| 0.002 + (lcg(seed) % 1000) as f32 * 1e-5;
    let plane = |out: usize, seed: &mut u64| {
        let (c, s) = nv_bytes(out, 2048, seed);
        exec.nvf4_upload(&c, &s, s2(seed), out, 2048)
            .expect("plane")
    };
    // Flint's q / k / v and shared gate / up shapes
    let (q, k, v) = (
        plane(4096, &mut seed),
        plane(512, &mut seed),
        plane(512, &mut seed),
    );
    let (g, u) = (plane(1024, &mut seed), plane(1024, &mut seed));
    for rows in [2usize, 4, 8] {
        let x: Vec<f32> = (0..rows * 2048)
            .map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let dx = exec.to_device(&x).expect("x");
        let one = |w: &paddock_engine::gpu::Nvf4Plane| {
            let mut y = exec.alloc(rows * w.out_dim).expect("y");
            exec.nvf4_gemv_batch(w, &dx, &mut y, None, rows)
                .expect("gemv");
            exec.stream.clone_dtoh(&y).expect("back")
        };
        let (wq, wk, wv) = (one(&q), one(&k), one(&v));
        let (mut yq, mut yk, mut yv) = (
            exec.alloc(rows * 4096).expect("y"),
            exec.alloc(rows * 512).expect("y"),
            exec.alloc(rows * 512).expect("y"),
        );
        let ran = exec
            .nvf4_gemv_batch3(
                &mut [(&q, &mut yq), (&k, &mut yk), (&v, &mut yv)],
                &dx,
                rows,
            )
            .expect("927");
        if !ran {
            common::missing("pack has no fused decode planes (slot 927)");
            return;
        }
        for (want, got, n) in [(&wq, &yq, "q"), (&wk, &yk, "k"), (&wv, &yv, "v")] {
            let got = exec.stream.clone_dtoh(got).expect("back");
            assert!(
                same_bits(want, &got),
                "927 {n} differs from its plane's GEMV at {rows} rows"
            );
        }
        let (wg, wu) = (one(&g), one(&u));
        let (mut yg, mut yu) = (
            exec.alloc(rows * 1024).expect("y"),
            exec.alloc(rows * 1024).expect("y"),
        );
        exec.nvf4_gemv_batch3(&mut [(&g, &mut yg), (&u, &mut yu)], &dx, rows)
            .expect("927 pair");
        assert!(
            same_bits(&wg, &exec.stream.clone_dtoh(&yg).expect("back")),
            "927 gate differs at {rows}"
        );
        assert!(
            same_bits(&wu, &exec.stream.clone_dtoh(&yu).expect("back")),
            "927 up differs at {rows}"
        );
    }
    println!("927: q|k|v and gate|up bit-identical to their GEMVs at 2 / 4 / 8 rows");
}
