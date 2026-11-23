//! Ripple-65B Preview (Ricursive) on the laguna body's Ripple flavor.
//!
//! - `ripple_glue_*`: slots 968 / 969, the FFN's clamped SwiGLU and the
//!   value-residual blend, against their definitions (the reference's
//!   `silu(gate.clamp(max=L)) * up.clamp(-L, L)` and `torch.lerp`). Light.
//! - `ripple_clamped_gate_up_tile`: slot 970, the W4A16 prefill tile's
//!   gate|up with that clamp in its epilogue. Light.
//! - `ripple_static_scale_nvf4`: slot 971, the activation quantize against
//!   the checkpoint's static input scales, and the scaled W4A4 GEMM. Light.
//! - `ripple_ffn_prefill_classes`: the FFN at Ripple's shapes, W4A16 tile vs
//!   the W4A4 serving class, agreement and time. Light-ish.
//! - `ripple_verify_band_rows`: slots 972 / 973, the verify widths' W4A4
//!   tile (64 columns) and BF16 TMA arm - every row of a 48- or 64-row call
//!   is the 16-row call's bit for bit (a slot verifies alike whoever shares
//!   its round). Light.
//! - `ripple_rowq_after_pdl_wait`: the e4m3 row quantizer at every elector
//!   arm straight behind an armed rmsnorm, against a stream sync between
//!   the two (a read ahead of the PDL wait shows as stale rows). Light.
//! - `ripple_nvfp4_lanes` (PADDOCK_HEAVY_TESTS + the official NVFP4 dir): one
//!   load, then the serial lane's greedy answer and two prompt tails against
//!   the batched lane - the decode stream tracks the serial one (parting only
//!   on a near-tie), a prefill past the 1024-key window picks what the serial
//!   lane picks, and a rewritten tail resumes behind its divergence off the
//!   SWA prefix cache's checkpoints. The serial lane runs ~4 tokens a second,
//!   so the prompts stay near 1.3K tokens (~12 minutes of serial prefill).
//!
//! - `ripple_dflash2_tree_greedy` (heavy): the draft-tree round through the
//!   service's armed protocol (begin, a verify of placeholder chunks, the
//!   fetched replay) against the plain greedy stream - parting only on a
//!   near-tie, and more tokens a round than the chain's gate.
//!
//! The architecture cross-check is Ricursive's own Spark bundle (SGLang) on
//! the same checkpoint, teacher-forced, outside the test suite.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_tokenizer::GgufTokenizer;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).max(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
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

/// log-softmax of one logit row at `i`.
fn logprob(l: &[f32], i: usize) -> f32 {
    let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = mx + l.iter().map(|&v| ((v - mx) as f64).exp()).sum::<f64>().ln() as f32;
    l[i] - lse
}

/// The batched lane's prompt rows ride the W4A4 FFN class (the checkpoint's
/// static input scales) where the serial lane runs the W4A16 GEMVs, so a
/// prefill's last row is held to the serial pick and that pick's
/// log-probability, plus a loose bound over the whole vocabulary - its tail
/// moves most (2.26 measured at 1357 tokens; 0.43 on the W4A16 tile).
fn same_pick(serial: &[f32], batched: &[f32], what: &str) {
    let (ts, gap) = top2(serial);
    let (tb, _) = top2(batched);
    let d = maxd(serial, batched);
    let dl = (logprob(serial, ts as usize) - logprob(batched, ts as usize)).abs();
    println!(
        "{what}: serial top1 {ts} (margin {gap:.3}) batched top1 {tb}, its |dlogprob| {dl:.4}, \
         max |dlogit| {d:.4}"
    );
    assert_eq!(ts, tb, "{what} picks differently (serial margin {gap:.3})");
    assert!(dl < 1.0, "{what}: the pick's log-probability parts by {dl}");
    assert!(d < 4.0, "{what}: logits part by {d}");
}

fn maxd(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn ripple_glue_matches_its_definitions() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_ripple_ops() {
        common::missing("pack has no Ripple glue (slots 968-969)");
        return;
    }
    // past the clamp on both sides, plus a ragged tail
    let n = 4099usize;
    let limit = 20.0f32;
    let g: Vec<f32> = det(n, 3).into_iter().map(|v| 40.0 * v).collect();
    let u: Vec<f32> = det(n, 5).into_iter().map(|v| 40.0 * v).collect();
    let mut d_g = exec.to_device(&g).expect("gate");
    let d_u = exec.to_device(&u).expect("up");
    exec.ripple_swiglu_clamp(&mut d_g, &d_u, n, limit)
        .expect("swiglu_clamp");
    let got = exec.to_host(&d_g).expect("gate back");
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&g, &u)| {
            let g = g.min(limit);
            g / (1.0 + (-g).exp()) * u.clamp(-limit, limit)
        })
        .collect();
    // relative: the products reach ~400
    let rel = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0f32, f32::max);
    println!("swiglu_clamp: max rel diff {rel:.2e}");
    assert!(
        rel < 1e-5,
        "swiglu_clamp parts from its definition by {rel}"
    );
    assert!(
        got.iter().all(|v| v.abs() <= limit * limit + 1e-3),
        "a clamped product escaped L^2"
    );

    let e = det(n, 7);
    let x: Vec<f32> = det(n, 11).into_iter().map(|v| 3.0 * v).collect();
    let t = 0.3127f32;
    let d_e = exec.to_device(&e).expect("e");
    let d_x = exec.to_device(&x).expect("x");
    let mut d_o = exec.alloc(n).expect("out");
    exec.ripple_valres_mix(&mut d_o, &d_e, &d_x, n, t)
        .expect("valres_mix");
    let got = exec.to_host(&d_o).expect("out back");
    let want: Vec<f32> = e.iter().zip(&x).map(|(&e, &x)| e + t * (x - e)).collect();
    let d = maxd(&got, &want);
    println!("valres_mix: max |diff| {d:.2e}");
    // one fma's rounding at most
    assert!(d < 1e-6, "valres_mix parts from lerp by {d}");
}

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

/// Slot 970, the W4A16 prefill tile's gate|up with the FFN clamp: at limit 0
/// the bits of slot 921, and clamped it is the definition applied to the
/// plain tile's own gate and up rows (slot 920, the same k walk) - with a
/// factor that drives a good share of both past the clamp.
#[test]
fn ripple_clamped_gate_up_tile() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_nvf4_pf() || !exec.has_nvf4_pf_gu_clamp() {
        common::missing("pack has no clamped gate|up tile (slot 970)");
        return;
    }
    let (k, ff, rows, limit) = (512usize, 256usize, 70usize, 20.0f32);
    let mut seed = 0x51ce_u64;
    let plane = |seed: &mut u64| {
        let codes: Vec<u8> = (0..ff * k / 2).map(|_| lcg(seed) as u8).collect();
        let scales: Vec<u8> = (0..ff * k / 16)
            .map(|_| 0x20 + (lcg(seed) % 0x20) as u8)
            .collect();
        exec.nvf4_upload(&codes, &scales, 1.0, ff, k)
            .expect("plane")
    };
    let (gate, up) = (plane(&mut seed), plane(&mut seed));
    let x: Vec<f32> = det(rows * k, 13);
    let d_x = exec.to_device(&x).expect("x");
    let mut x16 = exec.stream_alloc_bf16(rows * k).expect("x16");
    exec.convert_f32_bf16(&d_x, &mut x16, rows * k)
        .expect("cast");
    let act = |clamp: Option<f32>| {
        let mut a = exec.stream_alloc_bf16(rows * ff).expect("act");
        match clamp {
            None => exec.nvf4_pf_gu(&gate, &up, &x16, k, &mut a, ff, 0, rows),
            Some(l) => exec.nvf4_pf_gu_clamp(&gate, &up, &x16, k, &mut a, ff, 0, rows, l),
        }
        .expect("gu");
        let mut f = exec.alloc(rows * ff).expect("widen");
        exec.convert_bf16_f32(&a, &mut f, rows * ff).expect("widen");
        exec.to_host(&f).expect("back")
    };
    let plain = act(None);
    assert!(
        act(Some(0.0))
            .iter()
            .zip(&plain)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "limit 0 is not slot 921's bits"
    );
    let got = act(Some(limit));
    let rows_of = |w| {
        let mut y = exec.alloc(rows * ff).expect("y");
        exec.nvf4_pf_gemm(w, &x16, k, 0, &mut y, ff, 0, rows)
            .expect("pf gemm");
        exec.to_host(&y).expect("back")
    };
    let (g, u) = (rows_of(&gate), rows_of(&up));
    let past = g.iter().chain(&u).filter(|v| v.abs() > limit).count();
    println!("{past} of {} gate/up values past the clamp", 2 * g.len());
    assert!(
        past > g.len() / 4,
        "the planes barely reach the clamp ({past})"
    );
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&g, &u)| {
            let g = g.min(limit);
            let v = g / (1.0 + (-g).exp()) * u.clamp(-limit, limit);
            half::bf16::from_f32(v).to_f32()
        })
        .collect();
    // one bf16 ulp at most (device and host expf part in the last bits)
    let bad = got
        .iter()
        .zip(&want)
        .filter(|(a, b)| (*a - *b).abs() > b.abs() * 8e-3 + 1e-6)
        .count();
    println!(
        "slot 970 vs the definition: {bad} of {} beyond one bf16 ulp",
        want.len()
    );
    assert_eq!(bad, 0, "slot 970 parts from the clamped SwiGLU");
}

/// Slot 971, the nvfp4 activation quantize against a static per-tensor
/// scale, and the GEMM's matching factor: at a power-of-two global the block
/// scales are the dynamic quantizer's shifted by its exponent and the codes
/// are the same, so the scaled W4A4 GEMM lands the same bits.
#[test]
fn ripple_static_scale_nvf4() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_quantize_nvf4_gs() || !exec.has_nvf4_gemm_f4() {
        common::missing("pack has no static-scale nvf4 quantize (slot 971)");
        return;
    }
    let (k, out, rows) = (512usize, 256usize, 96usize);
    let x = det(rows * k, 41);
    let d_x = exec.to_device(&x).expect("x");
    let quant = |gs: Option<f32>| {
        let (mut q, mut s) = (
            exec.alloc_i8(rows * k / 2).expect("q"),
            exec.alloc_u8(rows * k / 16).expect("s"),
        );
        match gs {
            None => exec.quantize_nvf4(&d_x, &mut q, &mut s, rows * k),
            Some(g) => exec.quantize_nvf4_gs(&d_x, &mut q, &mut s, rows * k, g),
        }
        .expect("quantize");
        (q, s)
    };
    let (q0, s0) = quant(None);
    let (q4, s4) = quant(Some(4.0));
    let (hq0, hq4) = (
        exec.stream.clone_dtoh(&q0).expect("q0"),
        exec.stream.clone_dtoh(&q4).expect("q4"),
    );
    let (hs0, hs4) = (
        exec.stream.clone_dtoh(&s0).expect("s0"),
        exec.stream.clone_dtoh(&s4).expect("s4"),
    );
    assert_eq!(hq0, hq4, "codes move under a power-of-two global");
    // e4m3: four is two exponent steps (bits 3..6)
    assert!(
        hs0.iter()
            .zip(&hs4)
            .all(|(a, b)| *a == 0 || *b == a + (2 << 3)),
        "block scales are not the dynamic ones times four"
    );
    let mut seed = 0x971_u64;
    let codes: Vec<u8> = (0..out * k / 2).map(|_| lcg(&mut seed) as u8).collect();
    let scales: Vec<u8> = (0..out * k / 16)
        .map(|_| 0x20 + (lcg(&mut seed) % 0x20) as u8)
        .collect();
    let w = exec
        .nvf4_upload(&codes, &scales, 0.01, out, k)
        .expect("plane");
    let mut y0 = exec.alloc(rows * out).expect("y0");
    let mut y4 = exec.alloc(rows * out).expect("y4");
    exec.nvf4_gemm_f4(&w, &q0, &s0, &mut y0, None, rows, None)
        .expect("dynamic");
    exec.nvf4_gemm_f4_scaled(&w, 0.25, &q4, &s4, &mut y4, None, rows, None)
        .expect("static");
    let (h0, h4) = (
        exec.to_host(&y0).expect("y0"),
        exec.to_host(&y4).expect("y4"),
    );
    assert!(
        h0.iter().zip(&h4).all(|(a, b)| a.to_bits() == b.to_bits()),
        "the scaled GEMM parts from the dynamic class at a power-of-two global"
    );
}

/// The verify band's two arms are row-invariant: a spec round's slot gets
/// the same bits from a 48- or 64-row call (other slots in the round) as from
/// its own 16-row block. W4A4: the 64-column tile (slot 972, past 32 rows)
/// against the 32-column one (slot 597). BF16: the TMA arm (slot 973) both
/// ways of owning its tiles (row tiles on x at 48 rows of a 4096-row plane,
/// column tiles on x at 64 rows of a 16384-row one) against its 16-row call,
/// and within bf16-product noise of the ladder.
#[test]
fn ripple_verify_band_rows() {
    let Some(exec) = common::gpu() else { return };
    let (k, slot) = (1024usize, 16usize);
    // W4A4: the narrow tiles' election (dies under 128 SMs; slots 597 / 972
    // are NULL off cc 12)
    if exec.has_nvf4_gemm_f4() && exec.has_quantize_nvf4_gs() && exec.sm_count() < 128 {
        let out = 4096usize;
        let mut seed = 0x972_u64;
        let codes: Vec<u8> = (0..out * k / 2).map(|_| lcg(&mut seed) as u8).collect();
        let scales: Vec<u8> = (0..out * k / 16)
            .map(|_| 0x20 + (lcg(&mut seed) % 0x20) as u8)
            .collect();
        let w = exec
            .nvf4_upload(&codes, &scales, 0.01, out, k)
            .expect("plane");
        for rows in [48usize, 64] {
            let x = det(rows * k, 0x97 + rows as u64);
            let mut q = exec.alloc_i8(rows * k / 2).expect("q");
            let mut s = exec.alloc_u8(rows * k / 16).expect("s");
            exec.quantize_nvf4_gs(
                &exec.to_device(&x).expect("x"),
                &mut q,
                &mut s,
                rows * k,
                2.0,
            )
            .expect("quantize");
            let mut y = exec.alloc(rows * out).expect("y");
            exec.nvf4_gemm_f4_scaled(&w, 0.5, &q, &s, &mut y, None, rows, None)
                .expect("wide");
            let wide = exec.to_host(&y).expect("y");
            for b in 0..rows / slot {
                // the static-scale quantize is row-local: a block's codes are
                // the wide quantize's rows
                let xb = exec
                    .to_device(&x[b * slot * k..(b + 1) * slot * k])
                    .expect("xb");
                let mut qb = exec.alloc_i8(slot * k / 2).expect("qb");
                let mut sb = exec.alloc_u8(slot * k / 16).expect("sb");
                exec.quantize_nvf4_gs(&xb, &mut qb, &mut sb, slot * k, 2.0)
                    .expect("quantize block");
                let mut yb = exec.alloc(slot * out).expect("yb");
                exec.nvf4_gemm_f4_scaled(&w, 0.5, &qb, &sb, &mut yb, None, slot, None)
                    .expect("block");
                let hb = exec.to_host(&yb).expect("yb");
                let wb = &wide[b * slot * out..(b + 1) * slot * out];
                assert!(
                    wb.iter().zip(&hb).all(|(a, c)| a.to_bits() == c.to_bits()),
                    "W4A4 at {rows} rows parts from the 16-row block {b}"
                );
            }
        }
    } else {
        common::missing("no W4A4 tiles on this die (slots 597 / 972)");
    }
    // BF16 TMA arm
    for (out, rows) in [(4096usize, 48usize), (16384, 64)] {
        let w = paddock_engine::gpu::QuantTensor {
            bytes: exec
                .to_device_u8(&bf16_bytes(&det(out * k, 0x973 + out as u64)))
                .expect("w"),
            ty: paddock_models::ggml_type::GgmlType::Bf16,
            dims: vec![k, out],
        };
        let x = det(rows * k, 0x31 + rows as u64);
        let d_x = exec.to_device(&x).expect("x");
        let mut y = exec.alloc(rows * out).expect("y");
        if !exec
            .bf16_gemm_vb_rows(&w, 0, out, &d_x, &mut y, rows)
            .expect("wide")
        {
            common::missing("pack declines the bf16 verify band (slot 973)");
            return;
        }
        let wide = exec.to_host(&y).expect("y");
        let mut yl = exec.alloc(rows * out).expect("yl");
        exec.bf16_gemm_rows(&w, 0, out, &d_x, &mut yl, rows)
            .expect("ladder");
        let ladder = exec.to_host(&yl).expect("yl");
        let scale = ladder.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(
            maxd(&wide, &ladder) <= 1e-4 * scale.max(1.0),
            "the TMA arm parts from the ladder at {out} x {k}, {rows} rows: {} of {scale}",
            maxd(&wide, &ladder)
        );
        for b in 0..rows / slot {
            let xb = exec
                .to_device(&x[b * slot * k..(b + 1) * slot * k])
                .expect("xb");
            let mut yb = exec.alloc(slot * out).expect("yb");
            assert!(
                exec.bf16_gemm_vb_rows(&w, 0, out, &xb, &mut yb, slot)
                    .expect("block")
            );
            let hb = exec.to_host(&yb).expect("yb");
            let wb = &wide[b * slot * out..(b + 1) * slot * out];
            assert!(
                wb.iter().zip(&hb).all(|(a, c)| a.to_bits() == c.to_bits()),
                "the bf16 TMA arm at {rows} rows of {out} parts from the 16-row block {b}"
            );
        }
    }
}

/// The drafter's e4m3 GEMM (f8row) at its feature-append widths: every row
/// of a 96- or 128-row call is the 16-row call's bit for bit, on the
/// drafter's own shapes (fc 5120 -> 5120, k / v 5120 -> 1024).
#[test]
fn ripple_f8row_rows() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_bf16_to_f8row() || !exec.has_f8row_gemm() {
        common::missing("no f8row lane in this pack");
        return;
    }
    // (out, in, rows): the feature append's fc / k-v, then the draft pass's
    // q, gate|up and down at a full pass of blocks
    let shapes = [
        (5120usize, 5120usize, 96usize),
        (1024, 5120, 128),
        (5120, 5120, 128),
        (10240, 5120, 128),
        (40960, 5120, 128),
        (5120, 40960, 128),
        (5120, 40960, 96),
    ];
    for (out, k, rows) in shapes {
        let w = exec
            .bf16_to_f8row(&bf16_bytes(&det(out * k, 0x8f + out as u64)), k, out)
            .expect("plane");
        let x = det(rows * k, 0x18 + rows as u64);
        // ONE activation plane and output plane at every width, as the
        // drafter's scratch is
        let mut q = exec.alloc_i8(rows * k).expect("q");
        let mut rs = exec.alloc(rows).expect("rs");
        let mut y = exec.alloc(rows * out).expect("y");
        let run = |x: &[f32], r: usize, q: &mut _, rs: &mut _, y: &mut _| {
            let d_x = exec.to_device(x).expect("x");
            exec.quantize_e4m3_row(&d_x, q, rs, k, r).expect("quantize");
            exec.f8row_gemm(&w, q, rs, y, k, out, r).expect("gemm");
            exec.to_host(y).expect("y")
        };
        let wide = run(&x, rows, &mut q, &mut rs, &mut y);
        for b in 0..rows / 16 {
            let hb = run(
                &x[b * 16 * k..(b + 1) * 16 * k],
                16,
                &mut q,
                &mut rs,
                &mut y,
            );
            let wb = &wide[b * 16 * out..(b + 1) * 16 * out];
            let worst = wb
                .iter()
                .zip(&hb[..16 * out])
                .map(|(a, c)| (a - c).abs() / c.abs().max(1e-3))
                .fold(0f32, f32::max);
            assert!(
                worst < 1e-3,
                "f8row {k} -> {out} at {rows} rows: block {b} parts from its 16-row call (rel {worst})"
            );
        }
        // and the wide call again, after the narrow ones on the same planes
        let again = run(&x, rows, &mut q, &mut rs, &mut y);
        assert!(
            again
                .iter()
                .zip(&wide)
                .all(|(a, c)| a.to_bits() == c.to_bits()),
            "f8row {k} -> {out} at {rows} rows: a repeat call parts from the first"
        );
    }
}

/// The e4m3 row quantizer straight behind an armed predecessor, at every arm
/// of its elector: rmsnorm writes the plane (one CTA a row, a PDL primary)
/// and the quantize reads it next, as the drafter's feature append does. Its
/// bytes must be the ones a stream sync between the two gives. The plane
/// holds the other input's rows beforehand, so a quantize reading ahead of
/// its wait shows as stale rows - ptxas once scheduled row1p<V=8>'s loads
/// above griddepcontrol.wait, the drafter's 96+-row defects.
#[test]
fn ripple_rowq_after_pdl_wait() {
    let Some(exec) = common::gpu() else { return };
    // (width, rows) on a 48-SM die: the C-chunk arm, the column-sliced band,
    // the narrow 256-thread V=8 form, V=4, the 1024-thread V=8 form (16K-32K
    // wide) and the plain row kernel past 32K
    let shapes = [
        (5120usize, 8usize),
        (5120, 64),
        (5120, 96),
        (5120, 128),
        (10240, 128),
        (17408, 128),
        (40960, 96),
    ];
    for (k, rows) in shapes {
        let n = rows * k;
        let w = exec.to_device(&det(k, 0x77 + k as u64)).expect("w");
        let xs = [
            exec.to_device(&det(n, 0x31 + k as u64)).expect("xa"),
            exec.to_device(&det(n, 0x91 + rows as u64)).expect("xb"),
        ];
        let mut h = exec.alloc(n).expect("h");
        let mut q = exec.alloc_i8(n).expect("q");
        let mut rs = exec.alloc(rows).expect("rs");
        let mut run = |i: usize, sync: bool| {
            exec.rmsnorm_batch(&xs[i], &w, &mut h, k, 1e-6, rows)
                .expect("norm");
            if sync {
                exec.synchronize().expect("sync");
            }
            exec.quantize_e4m3_row(&h, &mut q, &mut rs, k, rows)
                .expect("quantize");
            (
                exec.to_host_i8(&q).expect("q"),
                exec.to_host(&rs).expect("rs"),
            )
        };
        let want = [run(0, true), run(1, true)];
        for it in 0..8 {
            let (gq, gr) = run(it % 2, false);
            let (wq, wr) = &want[it % 2];
            let bad = (0..rows)
                .filter(|&r| {
                    gq[r * k..(r + 1) * k] != wq[r * k..(r + 1) * k]
                        || gr[r].to_bits() != wr[r].to_bits()
                })
                .count();
            assert!(
                bad == 0,
                "e4m3 row quantize at {rows} x {k}: {bad} rows read ahead of the PDL wait (run {it})"
            );
        }
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x * 0.05).to_le_bytes())
        .collect()
}

/// Ripple's FFN at its own shapes (5120 -> 40960 -> 5120) over a prefill
/// chunk, synthetic planes: the W4A16 prefill tile (slots 970 + 920) against
/// the W4A4 class (nvfp4 rows on the block-scaled fp4 GEMM, the checkpoint's
/// own deployment class) - both within the activation-quant noise of each
/// other, and the time each takes. Light-ish (420 MB of planes).
#[test]
fn ripple_ffn_prefill_classes() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_nvf4_pf() || !exec.has_nvf4_pf_gu_clamp() || !exec.has_nvf4_gemm_f4() {
        common::missing("pack has no W4A16 prefill tile / W4A4 GEMM");
        return;
    }
    let (embd, ff, limit) = (5120usize, 40960usize, 20.0f32);
    let mut seed = 0x7e57_u64;
    // codes and block scales as a calibrated checkpoint has them: the
    // factor puts gate / up in the units the clamp bites at
    let plane = |out: usize, k: usize, s2: f32, seed: &mut u64| {
        let codes: Vec<u8> = (0..out * k / 2).map(|_| lcg(seed) as u8).collect();
        let scales: Vec<u8> = (0..out * k / 16)
            .map(|_| 0x20 + (lcg(seed) % 0x20) as u8)
            .collect();
        exec.nvf4_upload(&codes, &scales, s2, out, k)
            .expect("plane")
    };
    let gate = plane(ff, embd, 0.25, &mut seed);
    let up = plane(ff, embd, 0.25, &mut seed);
    let down = plane(embd, ff, 0.01, &mut seed);
    for rows in [1218usize, 2048] {
        let x = det(rows * embd, 31);
        let d_x = exec.to_device(&x).expect("x");
        let mut x16 = exec.stream_alloc_bf16(rows * embd).expect("x16");
        let mut act = exec.stream_alloc_bf16(rows * ff).expect("act");
        let mut y16 = exec.alloc(rows * embd).expect("y");
        let a16 = |x16: &mut _, act: &mut _, y: &mut _| {
            exec.convert_f32_bf16(&d_x, x16, rows * embd).expect("cast");
            exec.nvf4_pf_gu_clamp(&gate, &up, x16, embd, act, ff, 0, rows, limit)
                .expect("gu");
            exec.nvf4_pf_gemm(&down, act, ff, 0, y, embd, 0, rows)
                .expect("down");
        };
        let (mut xq, mut xs) = (
            exec.alloc_i8(rows * ff / 2).expect("xq"),
            exec.alloc_u8(rows * ff / 16).expect("xs"),
        );
        let (mut g, mut u) = (
            exec.alloc(rows * ff).expect("g"),
            exec.alloc(rows * ff).expect("u"),
        );
        let mut y4 = exec.alloc(rows * embd).expect("y4");
        // the serving recipe: static scales at a calibration-like amax
        let (s_in, s_dn) = (1.0 / (6.0 * 448.0), 400.0 / (6.0 * 448.0));
        let a4 = |xq: &mut _, xs: &mut _, g: &mut _, u: &mut _, y: &mut _| {
            exec.quantize_nvf4_gs(&d_x, xq, xs, rows * embd, 1.0 / s_in)
                .expect("q x");
            exec.nvf4_gemm_f4_scaled(&gate, s_in, xq, xs, g, None, rows, None)
                .expect("gate");
            exec.nvf4_gemm_f4_scaled(&up, s_in, xq, xs, u, None, rows, None)
                .expect("up");
            exec.ripple_swiglu_clamp(g, u, rows * ff, limit)
                .expect("clamp");
            exec.quantize_nvf4_gs(g, xq, xs, rows * ff, 1.0 / s_dn)
                .expect("q act");
            exec.nvf4_gemm_f4_scaled(&down, s_dn, xq, xs, y, None, rows, None)
                .expect("down");
        };
        let time = |f: &mut dyn FnMut()| {
            f();
            exec.synchronize().expect("sync");
            let t = std::time::Instant::now();
            for _ in 0..3 {
                f();
            }
            exec.synchronize().expect("sync");
            t.elapsed().as_secs_f64() * 1e3 / 3.0
        };
        let t16 = time(&mut || a16(&mut x16, &mut act, &mut y16));
        let t4 = time(&mut || a4(&mut xq, &mut xs, &mut g, &mut u, &mut y4));
        let (h16, h4) = (
            exec.to_host(&y16).expect("y"),
            exec.to_host(&y4).expect("y4"),
        );
        let num: f64 = h4
            .iter()
            .zip(&h16)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        let den: f64 = h16.iter().map(|b| (*b as f64).powi(2)).sum();
        let l2 = (num / den).sqrt();
        let tflop = 3.0 * 2.0 * (rows * embd * ff) as f64 / 1e12;
        println!(
            "{rows} rows: W4A16 tile {t16:.2} ms ({:.0} TF/s), W4A4 {t4:.2} ms ({:.0} TF/s), \
             rel l2 {l2:.3}",
            tflop / (t16 / 1e3),
            tflop / (t4 / 1e3)
        );
        // the W4A4 class quantizes both GEMM inputs to nvfp4; on uniform
        // synthetic rows that noise is ~0.1 relative a stage (the e2m1
        // grid's own), compounded over two stages and the SwiGLU
        assert!(l2 < 0.25, "W4A4 parts from the W4A16 tile by l2 {l2:.3}");
    }
}

#[test]
fn ripple_nvfp4_lanes() {
    if !common::heavy() {
        return;
    }
    let Some(dir) = common::model("RIPPLE_NVFP4", common::RIPPLE_NVFP4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer");
    let stops = tok.stop_ids();
    assert!(
        stops.contains(&100265) && stops.contains(&100257),
        "<|im_end|> and <|endoftext|> both stop: {stops:?}"
    );
    let mut m = GpuLaguna::load_ripple_nvfp4(exec, &dir, 4096).expect("load");

    // the answer with the reasoning block closed up front (the model opens
    // its own otherwise, and a greedy think is long)
    let chat = tok
        .encode(
            "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.\
             <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        .expect("encode");
    // a log past the 1024-key window, and the same log with its last 40
    // tokens rewritten
    let mut text = String::from("<|im_start|>user\nHere is a shipping log:\n");
    for i in 0..70 {
        text.push_str(&format!(
            "Entry {i}: parcel P-{} from Leeds to Bergen, weight {} kg.\n",
            1000 + 37 * i,
            1 + i % 17
        ));
    }
    let log = tok.encode(&text).expect("encode");
    let mut long = log.clone();
    long.extend(
        tok.encode(
            "What did entry 7 weigh?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        .expect("encode"),
    );
    let mut rewritten = log[..log.len() - 40].to_vec();
    rewritten.extend(
        tok.encode("Where did entry 9 go?<|im_end|>\n<|im_start|>assistant\n")
            .expect("encode"),
    );
    assert!(long.len() > 1024 + 64, "long prompt {} tokens", long.len());

    // serial lane: W4A16 MLP GEMVs, exact-f32 BF16 GEMVs, a dense KV
    let n_gen = 32usize;
    let mut logits = Vec::new();
    for &t in &chat {
        logits = m.forward(t).expect("serial prefill");
    }
    let mut serial = Vec::new();
    let mut margins = Vec::new();
    while serial.len() < n_gen {
        let (t, gap) = top2(&logits);
        serial.push(t);
        margins.push(gap);
        if stops.contains(&t) {
            break;
        }
        logits = m.forward(t).expect("serial decode");
    }
    let stext = tok.decode(&serial, true).expect("decode");
    println!("serial ({} tokens): {stext:?}", serial.len());
    assert!(
        stext.contains("Paris"),
        "serial lane does not answer: {stext:?}"
    );
    let serial_tail = |m: &mut GpuLaguna, p: &[u32]| {
        m.reset();
        let mut l = Vec::new();
        for &t in p {
            l = m.forward(t).expect("serial tail");
        }
        l
    };
    let s_long = serial_tail(&mut m, &long);
    let s_rew = serial_tail(&mut m, &rewritten);

    // batched lane on the same weights (enable_batch drops the serial state)
    m.enable_batch(3).expect("enable_batch");
    let mut logits = m.forward_prefill(0, &chat).expect("batched prefill");
    let mut batched = Vec::new();
    while batched.len() < serial.len() {
        let (t, _) = top2(&logits);
        batched.push(t);
        if stops.contains(&t) {
            break;
        }
        let pos = (chat.len() + batched.len() - 1) as u32;
        logits = m.forward_batch(&[t], &[pos]).expect("batched decode");
    }
    let btext = tok.decode(&batched, true).expect("decode");
    println!("batched ({} tokens): {btext:?}", batched.len());
    match (0..serial.len().min(batched.len())).find(|&i| serial[i] != batched[i]) {
        Some(i) => {
            println!("lanes part at token {i} (serial margin {:.4})", margins[i]);
            assert!(
                margins[i] < 1.0,
                "lanes part at token {i} on a {:.3} margin - not a near-tie",
                margins[i]
            );
        }
        None => println!("lanes agree on all {} tokens", serial.len()),
    }

    let b_long = m.forward_prefill(1, &long).expect("batched long");
    same_pick(&s_long, &b_long, &format!("{}-token prefill", long.len()));

    let warm = m.forward_prefill(2, &rewritten).expect("warm rewritten");
    let reused = m.take_prefill_reused(2);
    println!(
        "rewritten {} tokens sharing {} with the long prompt: resumed at {reused}",
        rewritten.len(),
        log.len() - 40
    );
    assert!(
        reused > 0 && reused <= log.len() - 40,
        "the rewritten tail must resume behind the divergence (at {reused})"
    );
    same_pick(&s_rew, &warm, "resumed tail");
}

/// The DFlash2 drafter end to end on the batched lane (PADDOCK_HEAVY_TESTS +
/// both dirs): prefill, then the service's greedy round - draft the block,
/// verify pending + drafts, accept while the picks match, commit the
/// accepted rows' features - against the same lane's plain greedy decode.
/// The streams may part only on a near-tie of the plain stream (the verify
/// rows ride the multi-row classes); the drafter must earn its keep (mean
/// accepted drafts a round), and a round must leave the next one warm.
#[test]
fn ripple_dflash2_spec_greedy() {
    if !common::heavy() {
        return;
    }
    let (Some(dir), Some(ddir)) = (
        common::model("RIPPLE_NVFP4", common::RIPPLE_NVFP4),
        common::model("RIPPLE_DFLASH2", common::RIPPLE_DFLASH2),
    ) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer");
    let mut m = GpuLaguna::load_ripple_nvfp4(exec, &dir, 4096).expect("load");
    m.attach_dflash2(&ddir).expect("attach drafter");
    m.enable_batch(2).expect("enable_batch");
    let prompt = tok
        .encode(
            "<|im_start|>user\nWrite a Python function `is_prime(n: int) -> bool` with a \
             docstring and a short test at the bottom.<|im_end|>\n<|im_start|>assistant\n\
             <think>\n\n</think>\n\n",
        )
        .expect("encode");
    let n_gen = 160usize;
    let stops = tok.stop_ids();

    // the plain batched greedy stream (slot 0), with its margins
    let mut logits = m.forward_prefill(0, &prompt).expect("prefill");
    let mut plain = Vec::new();
    let mut margins = Vec::new();
    let t0 = std::time::Instant::now();
    while plain.len() < n_gen {
        let (t, gap) = top2(&logits);
        plain.push(t);
        margins.push(gap);
        if stops.contains(&t) {
            break;
        }
        let pos = (prompt.len() + plain.len() - 1) as u32;
        logits = m.forward_batch(&[t], &[pos]).expect("decode");
    }
    let plain_tps = plain.len() as f64 / t0.elapsed().as_secs_f64();

    // the spec stream (slot 1): the drafter drafts at the span's end
    let logits = m.forward_prefill(1, &prompt).expect("prefill again");
    let (mut pending, _) = top2(&logits);
    let mut pos = prompt.len();
    let (mut spec, mut rounds, mut accepted) = (Vec::new(), 0usize, 0usize);
    let t0 = std::time::Instant::now();
    while spec.len() < plain.len() {
        assert!(
            m.spec_ensure_warm(1, &[], (pos - 1) as u32).expect("warm"),
            "the drafter went cold at {pos} after {rounds} rounds"
        );
        let drafts = m
            .spec_draft_batch(&[(1, pending)], 15)
            .expect("draft")
            .and_then(|v| v.into_iter().next())
            .expect("a warm slot drafts");
        assert_eq!(drafts.len(), 15, "the whole block a round");
        let mut chunk = vec![pending];
        chunk.extend(&drafts);
        let picks = m
            .forward_spec_batch(&[(1, pos, chunk.clone())])
            .expect("verify")
            .expect("the round runs");
        let mut a = 0;
        while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
            a += 1;
        }
        spec.extend(&chunk[..=a]);
        rounds += 1;
        accepted += a;
        pending = picks[a];
        pos += a + 1;
        if chunk[..=a].iter().any(|t| stops.contains(t)) {
            break;
        }
    }
    let spec_tps = spec.len() as f64 / t0.elapsed().as_secs_f64();
    let mean = accepted as f64 / rounds as f64;
    println!(
        "plain {} tokens at {plain_tps:.2} tok/s; spec {} tokens in {rounds} rounds, \
         {mean:.2} drafts accepted a round, {spec_tps:.2} tok/s",
        plain.len(),
        spec.len()
    );
    println!("spec text: {:?}", tok.decode(&spec, true).expect("decode"));
    match (0..plain.len().min(spec.len())).find(|&i| plain[i] != spec[i]) {
        Some(i) => {
            println!("streams part at token {i} (plain margin {:.4})", margins[i]);
            assert!(
                margins[i] < 1.0,
                "streams part at token {i} on a {:.3} margin - not a near-tie",
                margins[i]
            );
        }
        None => println!(
            "streams agree on all {} tokens",
            plain.len().min(spec.len())
        ),
    }
    assert!(mean >= 2.0, "the drafter accepts {mean:.2} drafts a round");
}

#[test]
fn ripple_dflash2_tree_greedy() {
    if !common::heavy() {
        return;
    }
    let (Some(dir), Some(ddir)) = (
        common::model("RIPPLE_NVFP4", common::RIPPLE_NVFP4),
        common::model("RIPPLE_DFLASH2", common::RIPPLE_DFLASH2),
    ) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer");
    let mut m = GpuLaguna::load_ripple_nvfp4(exec, &dir, 4096).expect("load");
    m.attach_dflash2(&ddir).expect("attach drafter");
    // TREE_SLOTS (default 2): how many slots share every round
    let n: usize = std::env::var("TREE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    m.enable_batch(n + 1).expect("enable_batch");
    // TREE_LONG: a long system preamble ahead of the request, so the
    // prefill's feature append runs hundreds of rows wide
    let pre = if std::env::var_os("TREE_LONG").is_some() {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n",
            "You are a careful senior engineer. Prefer clear names, small functions, \
             docstrings, and tests that cover edge cases. "
                .repeat(16)
        )
    } else {
        String::new()
    };
    let prompt = tok
        .encode(&format!(
            "{pre}<|im_start|>user\nWrite a Python function `is_prime(n: int) -> bool` with a \
             docstring and a short test at the bottom.<|im_end|>\n<|im_start|>assistant\n\
             <think>\n\n</think>\n\n"
        ))
        .expect("encode");
    let n_gen = 160usize;
    let stops = tok.stop_ids();

    let mut logits = m.forward_prefill(0, &prompt).expect("prefill");
    let mut plain = Vec::new();
    let mut margins = Vec::new();
    while plain.len() < n_gen {
        let (t, gap) = top2(&logits);
        plain.push(t);
        margins.push(gap);
        if stops.contains(&t) {
            break;
        }
        let pos = (prompt.len() + plain.len() - 1) as u32;
        logits = m.forward_batch(&[t], &[pos]).expect("decode");
    }

    // two slots ride every round together (multi-run verify, multi-block
    // drafts and lattices, per-slot compaction), each must keep the stream
    let slots: Vec<usize> = (1..=n).collect();
    let mut pending = Vec::new();
    for &sl in &slots {
        let logits = m.forward_prefill(sl, &prompt).expect("prefill again");
        pending.push(top2(&logits).0);
    }
    let mut pos = vec![prompt.len(); n];
    let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut done = vec![false; n];
    let mut rounds = 0usize;
    let t0 = std::time::Instant::now();
    while !done.iter().all(|&d| d) {
        let live: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();
        for &i in &live {
            assert!(
                m.spec_ensure_warm(slots[i], &[], (pos[i] - 1) as u32)
                    .expect("warm")
            );
        }
        let pend: Vec<(usize, u32)> = live.iter().map(|&i| (slots[i], pending[i])).collect();
        // TREE_CHAIN: the same loop over the synchronous chain round
        let chain = std::env::var_os("TREE_CHAIN").is_some();
        let drafts = if chain {
            m.spec_draft_batch(&pend, 15)
                .expect("draft")
                .expect("warm slots draft")
        } else {
            let (depth, kept) = m
                .spec_draft_begin(&pend, 15)
                .expect("begin")
                .expect("the tree round arms");
            assert_eq!(depth, 15);
            assert!(kept.iter().all(|&k| k));
            Vec::new()
        };
        let reqs: Vec<(usize, usize, Vec<u32>)> = live
            .iter()
            .enumerate()
            .map(|(q, &i)| {
                let mut c = vec![pending[i]];
                c.extend(
                    drafts
                        .get(q)
                        .cloned()
                        .unwrap_or_else(|| vec![pending[i]; 15]),
                );
                (slots[i], pos[i], c)
            })
            .collect();
        let picks = m
            .forward_spec_batch(&reqs)
            .expect("verify")
            .expect("the round runs");
        let drafts = if chain {
            drafts
        } else {
            m.spec_draft_fetch().expect("fetch").expect("armed drafts")
        };
        for (q, &i) in live.iter().enumerate() {
            let p = &picks[q * 16..(q + 1) * 16];
            let mut c = reqs[q].2.clone();
            c[1..].copy_from_slice(&drafts[q]);
            let mut a = 0;
            while a + 1 < c.len() && c[a + 1] == p[a] {
                a += 1;
            }
            if std::env::var_os("TREE_TRACE").is_some() {
                println!("round {rounds} slot {i} pos {} accepted {a}", pos[i]);
            }
            streams[i].extend(&p[..=a]);
            pending[i] = p[a];
            pos[i] += a + 1;
            if streams[i].len() + 1 >= plain.len() || p[..=a].iter().any(|t| stops.contains(t)) {
                done[i] = true;
            }
        }
        rounds += 1;
    }
    let spec = streams[0].clone();
    let total: usize = streams.iter().map(|s| s.len()).sum();
    let tps = total as f64 / t0.elapsed().as_secs_f64();
    let per = spec.len() as f64 / rounds as f64;
    println!(
        "tree, {n} slots: {total} tokens in {rounds} rounds, {per:.2} a round a slot, {tps:.2} tok/s"
    );
    println!("tree text: {:?}", tok.decode(&spec, true).expect("decode"));
    for (si, s2) in streams.iter().enumerate() {
        // the first emitted token is the prefill's pick, the stream's first
        let stream: Vec<u32> = std::iter::once(plain[0])
            .chain(s2.iter().copied())
            .collect();
        match (0..plain.len().min(stream.len())).find(|&i| plain[i] != stream[i]) {
            Some(i) => {
                println!(
                    "slot {si}: streams part at token {i} (plain margin {:.4})",
                    margins[i]
                );
                assert!(
                    margins[i] < 1.0,
                    "slot {si}: streams part at token {i} on a {:.3} margin - not a near-tie",
                    margins[i]
                );
            }
            None => println!(
                "slot {si}: streams agree on all {} tokens",
                plain.len().min(stream.len())
            ),
        }
    }
    assert!(per >= 4.0, "the tree round carries {per:.2} tokens a round");
}
