//! Bit gate for the sorted NVFP4 ring pair (slots 830 / 831) against the
//! 32-row pair it replaces (slots 631 / 758): the same routed rows through
//! `moe_align` + gate|up + bf16 down at 32-row blocks and through
//! `moe_align_bm(64)` + the ring kernels must land identical nvfp4 bytes for
//! every (token, slot) and identical bf16 partials. The sorted positions
//! differ between the two layouts (and moe_align's fill order is not fixed),
//! so the fq/fs planes are compared keyed by (token, slot), not by position.
//!
//! Synthetic planes at Kolibri's expert shape (2560 -> 512 -> 2560, top-6),
//! trimmed in expert count; routing skewed so hot experts span several
//! (partial) 64-row blocks and cold ones none.

mod common;

use paddock_engine::gpu::GpuExecutor;

const PAD: u32 = u32::MAX;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed;
    (0..len).map(|_| lcg(&mut s) as u8).collect()
}

/// e4m3 scales of modest magnitude so the products stay in range
fn scale_bytes(seed: u64, len: usize) -> Vec<u8> {
    bytes(seed, len).iter().map(|b| 0x30 | (b & 0x07)).collect()
}

/// `k` distinct experts a token, popularity falling with the expert id
/// (u^2 skew) - hot experts take several blocks, cold ones few or none
fn route(tokens: usize, k: usize, n_expert: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    let mut idx = Vec::with_capacity(tokens * k);
    for _ in 0..tokens {
        let mut picked: Vec<u32> = Vec::with_capacity(k);
        while picked.len() < k {
            let u = (lcg(&mut s) as f64) / (1u64 << 31) as f64;
            let e = ((u * u) * n_expert as f64) as u32 % n_expert as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        idx.extend(picked);
    }
    idx
}

/// fq / fs rows keyed by (token, slot): `row_bytes` bytes of `plane` a
/// sorted position, gathered through the layout's sorted_row / sorted_slot.
/// Blocks past the used ones carry no sorted rows at all (moe_align PADs
/// only the blocks it fills) - their expert is PAD, as the kernels read it.
fn by_pair(
    l: &Layout,
    plane: &[u8],
    row_bytes: usize,
    k: usize,
    pairs: usize,
) -> Vec<Option<Vec<u8>>> {
    let bm = l.srow.len() / l.nb;
    let mut out = vec![None; pairs];
    for (p, (&t, &s)) in l.srow.iter().zip(&l.sslot).enumerate() {
        if l.bexp[p / bm] == PAD || t == PAD {
            continue;
        }
        let key = t as usize * k + s as usize;
        assert!(out[key].is_none(), "pair ({t}, {s}) sorted twice");
        out[key] = Some(plane[p * row_bytes..(p + 1) * row_bytes].to_vec());
    }
    out
}

struct Layout {
    nb: usize,
    bexp: Vec<u32>,
    srow: Vec<u32>,
    sslot: Vec<u32>,
    fq: Vec<u8>,
    fs: Vec<u8>,
    part: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run(
    exec: &GpuExecutor,
    ring: bool,
    gate: &paddock_engine::gpu::Nvf4MoePlane,
    up: &paddock_engine::gpu::Nvf4MoePlane,
    down: &paddock_engine::gpu::Nvf4MoePlane,
    idx: &[u32],
    topk_w: &[f32],
    x: &[f32],
    tokens: usize,
    k: usize,
    n_expert: usize,
) -> Layout {
    let (in_dim, ff, embd) = (gate.in_dim, gate.ff, down.ff);
    let bm = if ring { 64 } else { 32 };
    let pairs = tokens * k;
    let nb = (pairs + n_expert * (bm - 1))
        .div_ceil(bm)
        .min(pairs.div_ceil(bm) + n_expert);
    let d_idx = exec.to_device_u32(idx).expect("idx");
    let d_w = exec.to_device(topk_w).expect("topk_w");
    let d_x = exec.to_device(x).expect("x");
    let mut xq = exec.alloc_i8(tokens * in_dim / 2).expect("xq");
    let mut xs = exec.alloc_u8(tokens * in_dim / 16).expect("xs");
    exec.quantize_nvf4(&d_x, &mut xq, &mut xs, tokens * in_dim)
        .expect("quantize_nvf4");
    let mut srow = exec.alloc_u32(nb * bm).expect("srow");
    let mut sslot = exec.alloc_u32(nb * bm).expect("sslot");
    let mut bexp = exec.alloc_u32(nb).expect("bexp");
    // poisoned: a byte the gate|up pair should write and does not shows up
    let mut fq = exec.alloc_u8_filled(nb * bm * ff / 2, 0xA5).expect("fq");
    let mut fs = exec.alloc_u8_filled(nb * bm * ff / 16, 0xA5).expect("fs");
    let mut part = exec.to_device(&vec![f32::NAN; pairs * embd]).expect("part");
    if ring {
        exec.moe_align_bm(
            &d_idx, &mut srow, &mut sslot, &mut bexp, tokens, k, n_expert, 64, nb,
        )
        .expect("moe_align_bm");
        exec.nvf4_moe_gu_swiglu_ms(gate, up, &srow, &bexp, &xq, &xs, &mut fq, &mut fs, nb, 0)
            .expect("gu ms");
        exec.nvf4_moe_down_ms_b16_at(
            down,
            &srow,
            &sslot,
            &bexp,
            Some(&d_w),
            &fq,
            &fs,
            &mut part,
            k,
            k,
            0,
            nb,
            0,
        )
        .expect("down ms");
    } else {
        exec.moe_align_at(
            &d_idx, 0, &mut srow, &mut sslot, &mut bexp, tokens, k, n_expert, nb,
        )
        .expect("moe_align");
        exec.nvf4_moe_gu_swiglu_bs(gate, up, &srow, &bexp, &xq, &xs, &mut fq, &mut fs, nb, 0)
            .expect("gu bs");
        exec.nvf4_moe_down_bs_b16_at(
            down,
            &srow,
            &sslot,
            &bexp,
            Some(&d_w),
            &fq,
            &fs,
            &mut part,
            k,
            k,
            0,
            nb,
            0,
        )
        .expect("down bs b16");
    }
    Layout {
        nb,
        bexp: exec.to_host_u32_len(&bexp, nb).expect("bexp"),
        srow: exec.to_host_u32_len(&srow, nb * bm).expect("srow"),
        sslot: exec.to_host_u32_len(&sslot, nb * bm).expect("sslot"),
        fq: exec.to_host_u8_len(&fq, nb * bm * ff / 2).expect("fq"),
        fs: exec.to_host_u8_len(&fs, nb * bm * ff / 16).expect("fs"),
        // bf16 partials: pairs * embd of them in the first half of the plane
        part: exec.to_host_len(&part, pairs * embd / 2).expect("part"),
    }
}

#[test]
fn ring_pair_lands_the_32_row_pairs_bytes() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_nvf4_moe_ms()
        || !exec.has_nvf4_moe_gu_swiglu_bs()
        || !exec.has_nvf4_moe_down_bs_b16()
    {
        common::missing("pack has no sorted NVFP4 ring pair (slots 830 / 831)");
        return;
    }
    let (n_expert, in_dim, ff, embd, k) = (24usize, 2560usize, 512usize, 2560usize, 6usize);
    let gate = exec
        .nvf4_moe_upload(
            &bytes(11, n_expert * ff * in_dim / 2),
            &scale_bytes(12, n_expert * ff * in_dim / 16),
            &(0..n_expert)
                .map(|e| 0.5 + 0.02 * e as f32)
                .collect::<Vec<_>>(),
            n_expert,
            ff,
            in_dim,
        )
        .expect("gate");
    let up = exec
        .nvf4_moe_upload(
            &bytes(13, n_expert * ff * in_dim / 2),
            &scale_bytes(14, n_expert * ff * in_dim / 16),
            &(0..n_expert)
                .map(|e| 0.7 - 0.01 * e as f32)
                .collect::<Vec<_>>(),
            n_expert,
            ff,
            in_dim,
        )
        .expect("up");
    let down = exec
        .nvf4_moe_upload(
            &bytes(15, n_expert * embd * ff / 2),
            &scale_bytes(16, n_expert * embd * ff / 16),
            &(0..n_expert)
                .map(|e| 0.3 + 0.01 * e as f32)
                .collect::<Vec<_>>(),
            n_expert,
            embd,
            ff,
        )
        .expect("down");
    // 240 tokens: hot experts take several partial 64-row blocks; 5 tokens:
    // one block an expert at most, nearly all pad
    for (tokens, seed) in [(240usize, 21u64), (5, 22)] {
        let idx = route(tokens, k, n_expert, seed);
        let mut s = seed ^ 0x5eed;
        let topk_w: Vec<f32> = (0..tokens * k)
            .map(|_| 0.05 + (lcg(&mut s) % 1000) as f32 / 2000.0)
            .collect();
        let x: Vec<f32> = (0..tokens * in_dim)
            .map(|_| (lcg(&mut s) as f32 / (1u64 << 31) as f32) - 0.5)
            .collect();
        let a = run(
            &exec, false, &gate, &up, &down, &idx, &topk_w, &x, tokens, k, n_expert,
        );
        let b = run(
            &exec, true, &gate, &up, &down, &idx, &topk_w, &x, tokens, k, n_expert,
        );
        let pairs = tokens * k;
        let (qa, qb) = (
            by_pair(&a, &a.fq, ff / 2, k, pairs),
            by_pair(&b, &b.fq, ff / 2, k, pairs),
        );
        let (sa, sb) = (
            by_pair(&a, &a.fs, ff / 16, k, pairs),
            by_pair(&b, &b.fs, ff / 16, k, pairs),
        );
        for p in 0..pairs {
            assert!(qa[p].is_some() && qb[p].is_some(), "pair {p} never sorted");
            assert_eq!(qa[p], qb[p], "{tokens} tokens: fq differs at pair {p}");
            assert_eq!(sa[p], sb[p], "{tokens} tokens: fs differs at pair {p}");
        }
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        let (pa, pb) = (bits(&a.part), bits(&b.part));
        let first = pa.iter().zip(&pb).position(|(x, y)| x != y);
        assert!(
            first.is_none(),
            "{tokens} tokens: bf16 partials differ at word {first:?}"
        );
        // the partials were all written (the plane started NaN)
        assert!(
            a.part.iter().all(|v| v.is_finite()),
            "{tokens} tokens: an unwritten partial"
        );
        eprintln!(
            "{tokens} tokens: {} pairs, {} / {} blocks (32 / 64 rows), fq + fs + bf16 partials \
             identical",
            pairs, a.nb, b.nb
        );
    }
}
