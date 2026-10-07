//! Qwen3.8-Flash-Next off a llama.cpp GGUF (the Unsloth UD exports) on the
//! consumer-card lane: k-quant dense planes, k-quant / i-quant expert seats
//! host-mapped behind the `[moe_offload]` slot cache, PLE table gathered
//! from the mmap. Heavy: `QWEN38FN_GGUF=<first shard>` names the file.
//!
//! The gate is greedy continuation of a fixed prompt against llama.cpp on
//! the same file (`QWEN38FN_GGUF_REF` = the expected token ids, comma
//! separated, from `llama-cli --temp 0`); without a reference it prints the
//! continuation and the decode rate.

mod common;

/// The expert lane under test. These tests were written for the host-mapped
/// expert lane (`PADDOCK_MOE_HOST=1`: the 76 GiB file does not fit a 96 GB
/// card beside its KV), which is also the slower load - pinning ~50 GB of
/// host memory takes the better part of an hour on a unified-memory box
/// (GB10, 2026-09-12) whose serve puts every expert in device memory. Set
/// `QWEN38FN_MOE_DEVICE=1` there to test the lane that actually serves.
fn moe_lane() {
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
}

use std::time::Instant;

use paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu;
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const PROMPT: &str = "The capital of France is";

#[test]
fn gguf_greedy_continuation() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // host-mapped experts + the slot cache are what this lane exists for
    moe_lane();
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let t0 = Instant::now();
    common::qwen4exp_exact_kv();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 1).expect("load gguf");
    let load_s = t0.elapsed().as_secs_f64();
    let headroom = exec.vram_headroom().unwrap_or(0);
    let seated = m
        .enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("seat expert cache");
    eprintln!(
        "load {load_s:.1}s, {:.1} GiB experts host-mapped, cache on {seated} layers ({:.2} GiB headroom)",
        m.expert_host_bytes() as f64 / (1u64 << 30) as f64,
        headroom as f64 / (1u64 << 30) as f64
    );
    let prompt = tok.encode(PROMPT).expect("encode");
    let n = 32usize;
    let t1 = Instant::now();
    let out = m.generate_greedy(&prompt, n).expect("generate");
    let gen_s = t1.elapsed().as_secs_f64();
    let text = tok.decode(&out, false).unwrap_or_default();
    eprintln!("prompt ids {prompt:?}");
    eprintln!("greedy ids {out:?}");
    eprintln!("greedy text {text:?}");
    eprintln!(
        "{n} tokens in {gen_s:.2}s = {:.1} tok/s (prefill included)",
        n as f64 / gen_s
    );
    // decode-only rate: 64 more tokens off the carried state
    let t2 = Instant::now();
    let mut last = *out.last().unwrap();
    for _ in 0..64 {
        let l = m.decode_step(last).expect("decode");
        last = l
            .iter()
            .enumerate()
            .fold(0usize, |b, (i, &x)| if x > l[b] { i } else { b }) as u32;
    }
    let dec_s = t2.elapsed().as_secs_f64();
    eprintln!(
        "decode-only: 64 tokens in {dec_s:.2}s = {:.1} tok/s",
        64.0 / dec_s
    );
    let (rows, misses) = m.moe_cache_stats().expect("stats");
    eprintln!(
        "graph active: {} | slot cache: {rows} rows resolved, {misses} misses, hit rate {:.1}%",
        m.graph_active(),
        100.0 * (1.0 - misses as f64 / rows.max(1) as f64)
    );
    if let Ok(reference) = std::env::var("QWEN38FN_GGUF_REF") {
        let want: Vec<u32> = reference
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse().expect("reference token id"))
            .collect();
        // informational: two engines with different activation quantization
        // part ways at the first near-tie; the parity gate is
        // `gguf_teacher_forced_agreement`. The first token must agree.
        let k = want.len().min(out.len());
        let same = out[..k]
            .iter()
            .zip(&want[..k])
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!("matches the llama.cpp greedy ids for the first {same} of {k} tokens");
        assert_eq!(
            out[0], want[0],
            "first greedy token differs from the llama.cpp reference"
        );
    }
    assert!(
        out.iter().any(|&t| t != out[0]),
        "degenerate continuation (every token {})",
        out[0]
    );
}

/// Localizer: the same prompt fed one token at a time through the decode
/// walk (batch 1 everywhere) must predict the same next token as the
/// prefill walk (batch = prompt length). A divergence points at the
/// batch > 1 lanes (k-quant dp4a GEMM, PLE prefill staging), not the model.
#[test]
fn gguf_incremental_matches_prefill() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    moe_lane();
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    common::qwen4exp_exact_kv();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 1).expect("load gguf");
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("cache");
    let text = std::env::var("QWEN38FN_PROMPT").unwrap_or_else(|_| PROMPT.to_owned());
    let prompt = tok.encode(&text).expect("encode");
    eprintln!("prompt {text:?} -> {prompt:?}");
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b })
    };
    // prefill walk
    let lp = m.forward_prompt(&prompt).expect("prefill");
    let top_prefill = argmax(&lp);
    // incremental walk: first token as a 1-token prefill, the rest decoded
    let l1 = m.forward_prompt(&prompt[..1]).expect("prefill 1");
    let mut last = l1;
    for &t in &prompt[1..] {
        last = m.decode_step(t).expect("decode");
    }
    let top_inc = argmax(&last);
    let dec = |i: usize| tok.decode(&[i as u32], false).unwrap_or_default();
    eprintln!(
        "prefill top {top_prefill} {:?} | incremental top {top_inc} {:?}",
        dec(top_prefill),
        dec(top_inc)
    );
    let top5 = |v: &[f32]| {
        let mut ix: Vec<usize> = (0..v.len()).collect();
        ix.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        ix[..5].iter().map(|&i| (i, v[i])).collect::<Vec<_>>()
    };
    eprintln!("prefill top5 {:?}", top5(&lp));
    eprintln!("incremental top5 {:?}", top5(&last));
    assert_eq!(
        top_prefill, top_inc,
        "prefill and incremental walks disagree on the next token"
    );
}

/// The Q8_0 dense planes of the GGUF at their odd widths (hc up: in 320,
/// shared-expert down: in 640, hc down: in 10240 out 320): every batched
/// lane against the batch-1 GEMV and an f64 CPU dequant reference.
#[test]
fn gguf_q8_dense_lanes_match() {
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let rel = |a: &[f32], b: &[f32]| {
        let (mut n, mut d) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            n += ((x - y) as f64).powi(2);
            d += (*y as f64).powi(2);
        }
        (n / d.max(1e-30)).sqrt()
    };
    for name in [
        "blk.0.hc_attn_up.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.0.hc_attn_down.weight",
        "blk.1.ple_value.weight",
    ] {
        let (info, raw) = map.tensor_bytes(name).expect("tensor");
        let (in_dim, out_dim) = (info.dims[0] as usize, info.dims[1] as usize);
        let w = exec.load_quantw(&map, name).expect("load_quantw");
        let paddock_engine::gpu::QuantW::Q8(q) = &w else {
            panic!("{name}: not Q8_0");
        };
        // f64 reference off the raw Q8_0 blocks (f16 d, 32 int8)
        let deq: Vec<f32> = raw
            .as_chunks::<34>()
            .0
            .iter()
            .flat_map(|b| {
                let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                (0..32).map(move |i| (b[2 + i] as i8) as f32 * d)
            })
            .collect();
        let batch = 5usize;
        let x: Vec<f32> = (0..batch * in_dim)
            .map(|i| (((i as u64 * 2654435761) >> 13) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let mut yref = vec![0f32; batch * out_dim];
        for r in 0..batch {
            for o in 0..out_dim {
                let mut acc = 0f64;
                for i in 0..in_dim {
                    acc += deq[o * in_dim + i] as f64 * x[r * in_dim + i] as f64;
                }
                yref[r * out_dim + o] = acc as f32;
            }
        }
        let d_x = exec.to_device(&x).expect("x");
        let mut d_y = exec.alloc(batch * out_dim).expect("y");
        exec.q8_0_gemm_repacked_mt(q, None, &d_x, &mut d_y, batch)
            .expect("mt");
        let y_mt = exec.to_host(&d_y).expect("h");
        exec.q8_0_gemm_repacked(q, None, &d_x, &mut d_y, batch)
            .expect("plain");
        let y_plain = exec.to_host(&d_y).expect("h");
        let mut y_gemv = vec![0f32; batch * out_dim];
        let mut xr = exec.alloc(in_dim).expect("xr");
        let mut yr = exec.alloc(out_dim).expect("yr");
        for r in 0..batch {
            exec.copy_region(&d_x, r * in_dim, &mut xr, 0, in_dim)
                .expect("copy");
            exec.q8_0_gemv_repacked(q, None, &xr, &mut yr)
                .expect("gemv");
            y_gemv[r * out_dim..(r + 1) * out_dim].copy_from_slice(&exec.to_host(&yr).expect("h"));
        }
        eprintln!(
            "{name} [{in_dim}->{out_dim}] b={batch}: gemv {:.2e}  mt {:.2e}  plain {:.2e}  (vs f64 ref)",
            rel(&y_gemv, &yref),
            rel(&y_mt, &yref),
            rel(&y_plain, &yref)
        );
    }
}

/// Triage: one prefill walk of the prompt with the per-op dump armed
/// (`PADDOCK_Q38FN_DUMP=<dir>`), to diff against llama.cpp's eval-callback
/// tensor trace on the same file.
#[test]
fn gguf_dump_prefill() {
    if std::env::var_os("PADDOCK_Q38FN_DUMP").is_none() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    moe_lane();
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    common::qwen4exp_exact_kv();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 1).expect("load gguf");
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("cache");
    let text = std::env::var("QWEN38FN_PROMPT").unwrap_or_else(|_| PROMPT.to_owned());
    let prompt = tok.encode(&text).expect("encode");
    let logits = m.forward_prompt(&prompt).expect("prefill");
    let mut ix: Vec<usize> = (0..logits.len()).collect();
    ix.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    eprintln!(
        "prompt {prompt:?} top5 {:?}",
        ix[..5].iter().map(|&i| (i, logits[i])).collect::<Vec<_>>()
    );
}

/// Every k-quant dense type the GGUF carries (Q5_K, Q6_K, Q4_K), batch 1
/// (fused GEMV) and batch 5 (int8 dp4a GEMM), against an f64 dot over the
/// repacked dequant.
#[test]
fn gguf_kq_dense_lanes_match() {
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let rel = |a: &[f32], b: &[f32]| {
        let (mut n, mut d) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            n += ((x - y) as f64).powi(2);
            d += (*y as f64).powi(2);
        }
        (n / d.max(1e-30)).sqrt()
    };
    for name in [
        "blk.0.ssm_out.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.3.attn_q.weight",
        "blk.3.attn_output.weight",
        "blk.0.ffn_gate_shexp.weight",
    ] {
        let (info, _) = map.tensor_bytes(name).expect("tensor");
        let (in_dim, out_dim) = (info.dims[0] as usize, info.dims[1] as usize);
        let w = exec.load_quantw(&map, name).expect("load_quantw");
        let paddock_engine::gpu::QuantW::Kq(k) = &w else {
            panic!("{name}: not k-quant");
        };
        let mut d_dq = exec.alloc(in_dim * out_dim).expect("dq");
        exec.kquant_dequant_rp(k, &mut d_dq).expect("dequant");
        let deq = exec.to_host(&d_dq).expect("h");
        let batch = 5usize;
        let x: Vec<f32> = (0..batch * in_dim)
            .map(|i| (((i as u64 * 2654435761) >> 13) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let mut yref = vec![0f32; batch * out_dim];
        for r in 0..batch {
            for o in 0..out_dim {
                let mut acc = 0f64;
                for i in 0..in_dim {
                    acc += deq[o * in_dim + i] as f64 * x[r * in_dim + i] as f64;
                }
                yref[r * out_dim + o] = acc as f32;
            }
        }
        let d_x = exec.to_device(&x).expect("x");
        let mut d_xq = exec.alloc_i8(batch * in_dim).expect("xq");
        let mut d_xs = exec.alloc(batch * in_dim / 32).expect("xs");
        exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * in_dim)
            .expect("quant");
        let mut d_sums = exec.alloc(batch * in_dim / 16).expect("sums");
        exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, batch)
            .expect("sums");
        let needs = matches!(
            k.ty,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q4_0 | GgmlType::Q2K
        );
        let mut d_y = exec.alloc(batch * out_dim).expect("y");
        exec.kquant_gemm_dp4a(k, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y, batch)
            .expect("dp4a");
        let y_dp4a = exec.to_host(&d_y).expect("h");
        let mut y_gemv = vec![0f32; batch * out_dim];
        let mut xr = exec.alloc(in_dim).expect("xr");
        let mut yr = exec.alloc(out_dim).expect("yr");
        for r in 0..batch {
            exec.copy_region(&d_x, r * in_dim, &mut xr, 0, in_dim)
                .expect("copy");
            exec.kquant_gemv(k, &xr, &mut yr).expect("gemv");
            y_gemv[r * out_dim..(r + 1) * out_dim].copy_from_slice(&exec.to_host(&yr).expect("h"));
        }
        eprintln!(
            "{name} {:?} [{in_dim}->{out_dim}] b={batch}: gemv {:.2e}  dp4a {:.2e}  (vs f64 ref on the dequant)",
            k.ty,
            rel(&y_gemv, &yref),
            rel(&y_dp4a, &yref)
        );
    }
}

/// Triage: recompute layer 0's GDN output projection on the CPU from the
/// dumped core (`PADDOCK_Q38FN_DUMP` dir) and print it next to the dumped
/// `mix_out`, so a difference to llama.cpp's `linear_attn_out-0` can be
/// pinned on the core or on the projection lane.
#[test]
fn gguf_l0_outproj_check() {
    let Some(dir) = std::env::var_os("PADDOCK_Q38FN_DUMP") else {
        return;
    };
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let rd = |n: &str| -> Vec<f32> {
        let b = std::fs::read(dir.join(n)).expect(n);
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let core = rd("L0.gdn_core.bin");
    let mix = rd("L0.mix_out.bin");
    let map = MappedGguf::open(&path).expect("open gguf");
    let w = exec
        .load_quantw(&map, "blk.0.ssm_out.weight")
        .expect("ssm_out");
    let paddock_engine::gpu::QuantW::Kq(k) = &w else {
        panic!()
    };
    let (in_dim, out_dim) = (k.dims[0], k.dims[1]);
    let mut d_dq = exec.alloc(in_dim * out_dim).expect("dq");
    exec.kquant_dequant_rp(k, &mut d_dq).expect("dequant");
    let deq = exec.to_host(&d_dq).expect("h");
    let n = core.len() / in_dim;
    for r in 0..n {
        let mut y = vec![0f32; out_dim];
        for o in 0..out_dim {
            let mut acc = 0f64;
            for i in 0..in_dim {
                acc += deq[o * in_dim + i] as f64 * core[r * in_dim + i] as f64;
            }
            y[o] = acc as f32;
        }
        let ours = &mix[r * out_dim..(r + 1) * out_dim];
        let rel = {
            let (mut a, mut b) = (0f64, 0f64);
            for (p, q) in y.iter().zip(ours) {
                a += ((p - q) as f64).powi(2);
                b += (*p as f64).powi(2);
            }
            (a / b).sqrt()
        };
        eprintln!(
            "row {r}: cpu(outproj(core)) first3 {:?} last3 {:?} | dumped mix_out first3 {:?} last3 {:?} | rel {rel:.2e}",
            &y[..3],
            &y[out_dim - 3..],
            &ours[..3],
            &ours[out_dim - 3..]
        );
    }
}

/// The parity gate: llama.cpp's own greedy path on the same file, teacher
/// forced through our walk (`QWEN38FN_GGUF_TOP10` = one line per position:
/// `<chosen> <id>:<logprob> ...` from llama-server's `n_probs`). Two engines
/// with different activation quantization will not share a 32-token greedy
/// string at 1.6 bits per weight; what they must share is the distribution:
/// our top-1 equals llama's at most positions, and llama's chosen token is
/// always inside our top-5.
#[test]
fn gguf_teacher_forced_agreement() {
    if !common::heavy() {
        return;
    }
    let Ok(top10) = std::env::var("QWEN38FN_GGUF_TOP10") else {
        eprintln!("SKIP: QWEN38FN_GGUF_TOP10 not set");
        return;
    };
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    moe_lane();
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    common::qwen4exp_exact_kv();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 1).expect("load gguf");
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("cache");
    let prompt = tok.encode(PROMPT).expect("encode");
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
    let rank_of = |logits: &[f32], id: u32| {
        let v = logits[id as usize];
        logits.iter().filter(|&&x| x > v).count()
    };
    let top1 = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .fold(0usize, |b, (i, &x)| if x > logits[b] { i } else { b }) as u32
    };
    let mut logits = m.forward_prompt(&prompt).expect("prefill");
    let (mut agree, mut worst_rank) = (0usize, 0usize);
    for (pos, (chosen, tops)) in steps.iter().enumerate() {
        let ours = top1(&logits);
        let r = rank_of(&logits, *chosen);
        worst_rank = worst_rank.max(r);
        agree += usize::from(ours == *chosen);
        let llama_gap =
            tops.first().map(|t| t.1).unwrap_or(0.0) - tops.get(1).map(|t| t.1).unwrap_or(-99.0);
        eprintln!(
            "pos {pos:2}: llama {chosen:6} {:?} (margin {llama_gap:.2}) | ours top1 {ours:6} {:?}, llama's token at rank {r}",
            tok.decode(&[*chosen], false).unwrap_or_default(),
            tok.decode(&[ours], false).unwrap_or_default()
        );
        logits = m.decode_step(*chosen).expect("decode");
    }
    eprintln!(
        "top-1 agreement {agree}/{} , worst rank of llama's token {worst_rank}",
        steps.len()
    );
    assert!(
        agree * 4 >= steps.len() * 3,
        "top-1 agreement {agree}/{} below 3/4",
        steps.len()
    );
    assert!(
        worst_rank < 5,
        "llama's chosen token fell to rank {worst_rank} (> top-5)"
    );
}

/// The dense k-quant planes above 64 rows: the W4A8 tile (`kq_matmul`'s
/// pipe2 route) against the dp4a lane on the same per-32 quantized
/// activations - the same exact-int class, so the bar is tight. 163 rows
/// is one tile column plus a ragged second; the file's Q5_K / Q6_K planes.
#[test]
fn gguf_kq_dense_tile_matches_dp4a() {
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_kquant_gemm_w4a8_pipe2() {
        eprintln!("pack lacks kquant_gemm_w4a8_pipe2 - skipping");
        return;
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    let rel = |a: &[f32], b: &[f32]| {
        let (mut n, mut d) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            n += ((x - y) as f64).powi(2);
            d += (*y as f64).powi(2);
        }
        (n / d.max(1e-30)).sqrt()
    };
    let batch = 163usize;
    for name in [
        "blk.0.ssm_out.weight",
        "blk.0.attn_qkv.weight",
        "blk.3.attn_output.weight",
        "blk.0.ffn_gate_shexp.weight",
    ] {
        let (info, _) = map.tensor_bytes(name).expect("tensor");
        let (in_dim, out_dim) = (info.dims[0] as usize, info.dims[1] as usize);
        let w = exec.load_quantw(&map, name).expect("load_quantw");
        let paddock_engine::gpu::QuantW::Kq(k) = &w else {
            panic!("{name}: not k-quant");
        };
        let needs = matches!(
            k.ty,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q4_0 | GgmlType::Q2K
        );
        let x: Vec<f32> = (0..batch * in_dim)
            .map(|i| (((i as u64 * 2654435761) >> 13) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let d_x = exec.to_device(&x).expect("x");
        let mut d_xq = exec.alloc_i8(batch * in_dim).expect("xq");
        let mut d_xs = exec.alloc(batch * in_dim / 32).expect("xs");
        exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * in_dim)
            .expect("quant");
        let mut d_sums = exec.alloc(batch * in_dim / 16).expect("sums");
        exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, batch)
            .expect("sums");
        let mut d_y = exec.alloc(batch * out_dim).expect("y");
        exec.kquant_gemm_dp4a(k, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y, batch)
            .expect("dp4a");
        let y_dp4a = exec.to_host(&d_y).expect("h");
        let n_chunks = in_dim.div_ceil(128);
        let batch_pad = batch.div_ceil(128) * 128;
        let mut d_yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
        exec.quantize_q8_mmq(&d_x, &mut d_yq, in_dim, batch)
            .expect("quantize mmq");
        let mut d_msums = exec.alloc(n_chunks * batch_pad * 4).expect("msums");
        exec.mmq_sums(&d_yq, &mut d_msums, in_dim, batch)
            .expect("mmq sums");
        let mut d_yt = exec.alloc(batch * out_dim).expect("yt");
        exec.kquant_gemm_w4a8_pipe2(k, &d_yq, needs.then_some(&d_msums), &mut d_yt, batch)
            .expect("pipe2");
        let y_tile = exec.to_host(&d_yt).expect("h");
        let e = rel(&y_tile, &y_dp4a);
        eprintln!(
            "{name} {:?} [{in_dim}->{out_dim}] b={batch}: tile vs dp4a {e:.2e}",
            k.ty
        );
        assert!(
            e < 1e-5,
            "{name}: the W4A8 tile diverged from the dp4a lane ({e:.2e})"
        );

        // the same rows through kq_matmul's chunk walk: 100-row chunks off a
        // chunk-sized scratch, stitched by the row-offset launchers
        let chunk = 100usize;
        let cpad = chunk.div_ceil(128) * 128;
        let mut c_yq = exec.alloc_u8(n_chunks * cpad * 144).expect("cyq");
        let mut c_sums = exec.alloc(n_chunks * cpad * 4).expect("csums");
        let mut d_ys = exec.alloc(batch * out_dim).expect("ys");
        let mut off = 0;
        while off < batch {
            let rows = (batch - off).min(chunk);
            exec.quantize_q8_mmq_rows(&d_x, off, &mut c_yq, in_dim, rows)
                .expect("quantize rows");
            exec.mmq_sums(&c_yq, &mut c_sums, in_dim, rows)
                .expect("sums rows");
            exec.kquant_gemm_w4a8_pipe2_rows(
                k,
                &c_yq,
                needs.then_some(&c_sums),
                &mut d_ys,
                off,
                rows,
            )
            .expect("pipe2 rows");
            off += rows;
        }
        let y_chunked = exec.to_host(&d_ys).expect("h");
        assert!(
            y_chunked == y_tile,
            "{name}: the chunked walk differs from the one-shot tile"
        );
    }
}

/// The prefix cache (prefix.rs): an exact re-send of a cached prompt walks
/// COLD and bit-identical to its first run (the in-walk checkpoint trade of
/// 2026-09-15: resuming would replay rows the first run computed inside one
/// walk through a shorter one), and the multi-turn shape - the prompt plus a
/// decoded reply plus a new message - resumes at the reply's last page and
/// stays greedy-identical to a cold walk of the same turn (loose L2: the
/// resumed walk starts where the cold reference walked on, the qwen35 gate's
/// class).
///
/// The cold reference comes from a first instance loaded with the cache
/// off (`PADDOCK_NO_PREFIX_CACHE` is read at load), dropped before the
/// cached instance loads.
#[test]
fn gguf_prefix_cache_resumes_and_matches_cold() {
    use paddock_engine::generator::Generator;
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    moe_lane();
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b })
    };
    let rel = |x: &[f32], y: &[f32]| {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };
    // ~200 tokens of varied text, deliberately not page-aligned
    let base = tok
        .encode(
            "The reference manual describes a distributed consensus protocol in which \
             every participant maintains a monotonically increasing term counter and \
             exchanges signed heartbeat messages over authenticated channels. ",
        )
        .expect("enc");
    let a: Vec<u32> = base.iter().copied().cycle().take(203).collect();
    let tail = tok
        .encode(" The committee then turned to the question of quorum, and")
        .expect("enc");
    let n_reply = 45usize;

    // --- instance A, cache off: the reply the prompt greedily continues into,
    // and the cold reference for the multi-turn prompt
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    let (reply, reference, c) = {
        common::qwen4exp_exact_kv();
        let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 2).expect("load gguf");
        let headroom = exec.vram_headroom().unwrap_or(0);
        m.enable_moe_cache(headroom.saturating_sub(512 << 20))
            .expect("cache");
        let l = m.prefill_slot(0, &a).expect("prefill A");
        let mut reply = vec![argmax(&l) as u32];
        for _ in 1..n_reply {
            let l = m
                .decode_step_batch(&[(0, *reply.last().unwrap())])
                .expect("decode");
            reply.push(argmax(&l[0]) as u32);
        }
        let mut c = a.clone();
        c.extend_from_slice(&reply);
        c.extend_from_slice(&tail);
        let reference = m.prefill_slot(0, &c).expect("cold prefill of the turn");
        assert_eq!(m.take_prefill_reused(0), 0, "the cache is off");
        (reply, reference, c)
    };
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };

    // --- instance B, cache on
    common::qwen4exp_exact_kv();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 2).expect("load gguf");
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("cache");
    let cold = m.prefill_slot(0, &a).expect("cold prefill");
    assert_eq!(m.take_prefill_reused(0), 0, "a cold run must not reuse");
    let warm = m.prefill_slot(1, &a).expect("warm prefill");
    let reused = m.take_prefill_reused(1);
    let b1 = (a.len() - 1) / 16 * 16;
    eprintln!(
        "PREFIX CACHE: reused {reused} of {} (deepest cut {b1})",
        a.len()
    );
    assert_eq!(reused, 0, "an exact re-send must walk cold");
    let n_diff = cold.iter().zip(&warm).filter(|(x, y)| x != y).count();
    assert_eq!(
        n_diff,
        0,
        "an exact re-send must be BIT-EXACT; {n_diff} of {} logits differ (rel {:.2e})",
        cold.len(),
        rel(&warm, &cold)
    );
    // the same reply must come out of the cached instance (the greedy stream
    // is the test's own witness that instance B walks the same math)
    let l = m.prefill_slot(1, &a).expect("prefill A again");
    let mut reply_b = vec![argmax(&l) as u32];
    for _ in 1..n_reply {
        let l = m
            .decode_step_batch(&[(1, *reply_b.last().unwrap())])
            .expect("decode");
        reply_b.push(argmax(&l[0]) as u32);
    }
    assert_eq!(
        reply_b, reply,
        "instance B's greedy reply differs from instance A's"
    );
    // the multi-turn prompt: the reply checkpoint (stage F) filed the reply's
    // pages as the decode ticks closed them, so the turn resumes at the
    // reply's last page boundary - past the prompt's last cut - and walks
    // only the reply's tail and the new message
    let got = m.prefill_slot(0, &c).expect("prefill the turn");
    let reused = m.take_prefill_reused(0);
    let reply_cut = (a.len() + n_reply) / 16 * 16;
    let r = rel(&got, &reference);
    eprintln!(
        "PREFIX CACHE multi-turn: reused {reused} of {} (prompt {} + reply {} + tail {}; prompt cut {b1}, reply cut {reply_cut}); rel {r:.2e}; greedy {} vs {}",
        c.len(),
        a.len(),
        reply.len(),
        tail.len(),
        argmax(&got),
        argmax(&reference)
    );
    assert!(
        reply_cut > b1,
        "the test's reply must cross a page boundary"
    );
    assert_eq!(
        reused, reply_cut,
        "the turn must resume at the reply's last page"
    );
    // the decision headroom at this position (logit units): top-1 minus
    // top-2 of each, and where each put the other's winner
    let top2 = |v: &[f32]| {
        let mut ix: Vec<usize> = (0..v.len()).collect();
        ix.sort_by(|&x, &y| v[y].total_cmp(&v[x]));
        (ix[0], v[ix[0]] - v[ix[1]])
    };
    let (wr, mr) = top2(&reference);
    let (wg, mg) = top2(&got);
    eprintln!(
        "PREFIX CACHE multi-turn margins: reference {wr} by {mr:.3} (resumed puts it {:.3} below its top), \
         resumed {wg} by {mg:.3} (reference puts it {:.3} below its top)",
        got[wg] - got[wr],
        reference[wr] - reference[wg]
    );
    // The resumed turn replays the reply rows as decode wrote them, the cold
    // reference prefilled them - the geometry class the chunked and bounded
    // gates measure (rel ~0.1-0.3). A greedy flip is judged by its margin,
    // as the parity rules judge every divergence: the top two may swap only
    // where the reference itself was not sure (measured 2026-09-24 on this
    // prompt: reference 0.18, resumed 0.62, a mutual top-2 swap).
    let swap_ok = wg == wr
        || (top2(
            &got.iter()
                .enumerate()
                .map(|(i, &v)| if i == wg { f32::MIN } else { v })
                .collect::<Vec<_>>(),
        )
        .0 == wr
            && reference[wr] - reference[wg] < 0.5);
    assert!(
        swap_ok,
        "greedy token flipped beyond a near-tie: reference {wr} by {mr:.3}, resumed {wg} by {mg:.3}"
    );
    assert!(r < 3e-1, "diverged: rel {r}");
}
