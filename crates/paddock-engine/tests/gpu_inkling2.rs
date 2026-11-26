//! Inkling 2 Mini (Thinking Machines, `inkling2_loop`) on CUDA.
//!
//! - `inkling2_sconv_sum_matches_chain`: slot 1004 against the combine ->
//!   conv -> copy -> add chain it replaces, bit for bit. Light.
//! - `inkling2_rmsnorm_bf16_is_the_narrowed_norm`: slot 1007 against
//!   rmsnorm_batch + convert_f32_bf16, bit for bit. Light.
//! - `inkling2_msg_down_matches_cat_gemm`: slot 1008 against slot 978 +
//!   the bf16 ladder (f32 sum order only). Light.
//! - `inkling2_glue_*`: slots 977-982 against their definitions (the
//!   reference model's modules, written out here): block attention
//!   residuals, the loop controllers' gate input and mix, the attention's
//!   head prep, the short convolutions (static and dynamic taps, carried
//!   state over runs), the softcapped SwiGLU. Light.
//! - `inkling2_quant_*`: slot 983, the four-over-six NVFP4 and the MXFP8
//!   activation quantizers - codes and scales bit for bit against the rule
//!   written out on the host. Light.
//! - `inkling2_moe_gemvs`: slots 984-985 against the decoded codes, and
//!   the prompt-width arm (987-988 over moe_align's blocks) on the same wave. Light.
//! - `inkling2_moe_tc_matches_cc`: slots 991-992 (both classes on the
//!   block-scale tensor cores) against 987-988 on a wide wave. Light.
//! - `inkling2_moe_ms_matches_tc`: slots 1002-1003 (the NVFP4 class on
//!   64-pair blocks) against 991-992 at the model's shapes, bit for bit,
//!   with both arms timed at prompt widths. Light.
//! - `inkling2_attn_*`: slot 986, the relative-bias attention with sinks,
//!   sliding and full, against an f64 softmax. Light.
//! - `inkling2_draft_ops`: slots 994-999, the DSpark draft's ops, against
//!   their definitions. Light.
//! - `inkling2_kv_store_attn_*`: slots 989-990, the serving stores (paged
//!   and ring, f32 / f16 / fp8 rows) and the split attention over them, two
//!   sequences at once; slot 993's prompt-row tiles over the same stores. Light.

mod common;

use paddock_engine::gpu::{Inkling2Act, Inkling2AttnOut, Inkling2Experts, Inkling2Sorted};

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

fn bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}

fn maxrel(got: &[f32], want: &[f32], floor: f32) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs() / b.abs().max(floor))
        .fold(0.0, f32::max)
}

fn unit_rms(x: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * inv).collect()
}

const EPS: f32 = 1e-6;

#[test]
fn inkling2_glue_attn_res() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-986)");
        return;
    }
    let (rows, w) = (3usize, 2304usize);
    for (ns, nq) in [(0usize, 1usize), (1, 3), (3, 1), (4, 4), (5, 3)] {
        let states: Vec<Vec<f32>> = (0..ns)
            .map(|i| {
                det(rows * w, 10 + i as u64)
                    .iter()
                    .map(|v| v * (1.0 + i as f32))
                    .collect()
            })
            .collect();
        let partial = det(rows * w, 99);
        let query: Vec<f32> = det(nq * w, 7).iter().map(|v| v * 0.3).collect();
        let d_states: Vec<_> = states
            .iter()
            .map(|s| exec.to_device(s).expect("state"))
            .collect();
        let refs: Vec<_> = d_states.iter().collect();
        let d_p = exec.to_device(&partial).expect("partial");
        let d_q = exec.to_device(&query).expect("query");
        let mut d_o = exec.alloc(nq * rows * w).expect("out");
        exec.inkling2_attn_res(&refs, (&d_p, 0), &d_q, nq, &mut d_o, rows, w, rows * w)
            .expect("attn_res");
        let got = exec.to_host(&d_o).expect("out back");
        let mut want = vec![0.0f32; nq * rows * w];
        for r in 0..rows {
            let mut n: Vec<Vec<f32>> = states
                .iter()
                .map(|s| unit_rms(&s[r * w..(r + 1) * w], EPS))
                .collect();
            n.push(unit_rms(&partial[r * w..(r + 1) * w], EPS));
            for q in 0..nq {
                let qv = &query[q * w..(q + 1) * w];
                let lg: Vec<f64> = n
                    .iter()
                    .map(|s| {
                        s.iter()
                            .zip(qv)
                            .map(|(a, b)| (*a as f64) * (*b as f64))
                            .sum::<f64>()
                            / (w as f64).sqrt()
                    })
                    .collect();
                let m = lg.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = lg.iter().map(|l| (l - m).exp()).collect();
                let den: f64 = e.iter().sum();
                for d in 0..w {
                    want[q * rows * w + r * w + d] = n
                        .iter()
                        .zip(&e)
                        .map(|(s, p)| s[d] as f64 * p / den)
                        .sum::<f64>() as f32;
                }
            }
        }
        let rel = maxrel(&got, &want, 1e-2);
        println!("attn_res ns {ns} nq {nq}: max rel {rel:.2e}");
        assert!(
            rel < 1e-4,
            "attn_res ns {ns} nq {nq} parts from its definition by {rel}"
        );
    }
}

#[test]
fn inkling2_glue_msg_gate() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-986)");
        return;
    }
    let (rows, w, rank) = (2usize, 2304usize, 64usize);
    let h: Vec<f32> = det(rows * w, 3).iter().map(|v| v * 5.0).collect();
    let m: Vec<f32> = det(rows * w, 4).iter().map(|v| v * 0.5).collect();
    let d_h = exec.to_device(&h).expect("h");
    let d_m = exec.to_device(&m).expect("m");
    let mut d_c = exec.alloc(rows * 2 * w).expect("cat");
    exec.inkling2_msg_cat((&d_h, 0), (&d_m, 0), &mut d_c, rows, w)
        .expect("cat");
    let cat = exec.to_host(&d_c).expect("cat back");
    for r in 0..rows {
        let a = unit_rms(&h[r * w..(r + 1) * w], EPS);
        let b = unit_rms(&m[r * w..(r + 1) * w], EPS);
        for d in 0..w {
            for (got, want) in [(cat[r * 2 * w + d], a[d]), (cat[r * 2 * w + w + d], b[d])] {
                // the bf16 rounding of the same f32 (one ulp off at a tie at most)
                assert!(
                    (got - bf16(want)).abs() <= bf16(want).abs() * 0.0079,
                    "msg_cat row {r} col {d}: {got} vs {}",
                    bf16(want)
                );
            }
        }
    }
    // the mix over a z the down projection would give
    let z = det(rows * rank, 5);
    let fw: Vec<f32> = det(w * rank, 6).iter().map(|v| bf16(v * 0.2)).collect();
    let gw: Vec<f32> = det(w * rank, 8).iter().map(|v| bf16(v * 0.2)).collect();
    let d_z = exec.to_device(&z).expect("z");
    let d_f = exec.to_device_u8(&bf16_bytes(&fw)).expect("F");
    let d_g = exec.to_device_u8(&bf16_bytes(&gw)).expect("G");
    let mut d_o = exec.alloc(rows * w).expect("out");
    exec.inkling2_msg_mix(
        &d_z,
        &d_f,
        &d_g,
        (&d_h, 0),
        (&d_m, 0),
        &mut d_o,
        rows,
        w,
        rank,
    )
    .expect("mix");
    let got = exec.to_host(&d_o).expect("mix back");
    let mut want = vec![0.0f32; rows * w];
    for r in 0..rows {
        let zb: Vec<f32> = z[r * rank..(r + 1) * rank]
            .iter()
            .map(|v| bf16(*v))
            .collect();
        for j in 0..w {
            let f: f32 = (0..rank).map(|k| fw[j * rank + k] * zb[k]).sum();
            let g: f32 = (0..rank).map(|k| gw[j * rank + k] * zb[k]).sum();
            want[r * w + j] = (1.0 - f.tanh()) * h[r * w + j] + g.tanh() * m[r * w + j];
        }
    }
    let rel = maxrel(&got, &want, 1e-1);
    println!("msg_mix: max rel {rel:.2e}");
    assert!(rel < 1e-4, "msg_mix parts from its definition by {rel}");
}

#[test]
fn inkling2_glue_head_prep() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-986)");
        return;
    }
    // a segment of a fused output: 24 heads x 128 at offset 100, stride 6552
    let (rows, heads, d, stride, off) = (3usize, 24usize, 128usize, 6552usize, 100usize);
    let x: Vec<f32> = det(off + rows * stride, 11)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let scale: Vec<f32> = det(heads * d, 12)
        .iter()
        .map(|v| bf16(1.0 + 0.5 * v))
        .collect();
    let pos = [5u32, 8191, 70000];
    let d_s = exec.to_device_u8(&bf16_bytes(&scale)).expect("scale");
    let d_pos = exec.to_device_u32(&pos).expect("pos");
    for (case, sc, clip, rms, log) in [
        ("query, full", true, 22.627_417f32, false, true),
        ("query, sliding", true, 22.627_417, false, false),
        ("key", false, 22.627_417, false, false),
        ("value", false, 0.0, true, false),
    ] {
        let mut d_x = exec.to_device(&x).expect("x");
        exec.inkling2_head_prep(
            &mut d_x,
            off,
            rows,
            heads,
            d,
            stride,
            sc.then_some(&d_s),
            clip,
            rms,
            log.then_some(&d_pos),
            (0.1, 8192.0),
        )
        .expect("head_prep");
        let got = exec.to_host(&d_x).expect("back");
        let mut want = x.clone();
        for (r, &p) in pos.iter().enumerate().take(rows) {
            for h in 0..heads {
                let base = off + r * stride + h * d;
                let mut v: Vec<f32> = (0..d)
                    .map(|i| x[base + i] * if sc { scale[h * d + i] } else { 1.0 })
                    .collect();
                let n = v.iter().map(|a| a * a).sum::<f32>();
                let f = if rms {
                    1.0 / (n / d as f32 + EPS).sqrt()
                } else {
                    (clip / (n.sqrt() + 1e-6)).min(1.0)
                };
                let t = if log {
                    1.0 + 0.1 * ((p as f32 + 1.0) / 8192.0).max(1.0).ln()
                } else {
                    1.0
                };
                for a in v.iter_mut() {
                    *a *= f * t;
                }
                want[base..base + d].copy_from_slice(&v);
            }
        }
        let rel = maxrel(&got, &want, 1e-3);
        println!("head_prep {case}: max rel {rel:.2e}");
        assert!(
            rel < 2e-6,
            "head_prep {case} parts from its definition by {rel}"
        );
    }
}

/// The short conv by definition over one run with an incoming state.
fn sconv_ref(
    x: &[f32],
    c: usize,
    w: &[f32],
    dyn_: Option<(&[f32], usize)>,
    state: &[f32],
) -> Vec<f32> {
    let t_n = x.len() / c;
    let at = |t: isize, ch: usize| -> f32 {
        if t >= 0 {
            x[t as usize * c + ch]
        } else {
            state[(t + 3) as usize * c + ch]
        }
    };
    let mut out = vec![0.0f32; x.len()];
    for t in 0..t_n {
        for ch in 0..c {
            let cur = x[t * c + ch];
            out[t * c + ch] = match dyn_ {
                None => {
                    let y: f32 = (0..4)
                        .map(|j| w[ch * 4 + j] * at(t as isize - 3 + j as isize, ch))
                        .sum();
                    (cur + y) / 5f32.sqrt()
                }
                Some((dy, heads)) => {
                    let h = ch / (c / heads);
                    let y: f32 = (0..4)
                        .map(|lag| {
                            (w[lag * c + ch] + dy[t * heads * 4 + h * 4 + lag])
                                * at(t as isize - lag as isize, ch)
                        })
                        .sum();
                    (cur + y) / 3.0
                }
            };
        }
    }
    out
}

#[test]
fn inkling2_glue_sconv() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-986)");
        return;
    }
    for (case, c, heads) in [("static", 300usize, 0usize), ("dynamic", 64, 8)] {
        let dynamic = heads > 0;
        // two runs: 150 rows (three chunks) and 2 rows (shorter than the taps)
        let lens = [150usize, 2];
        let rows: usize = lens.iter().sum();
        let x = det(rows * c, 21);
        let w: Vec<f32> = det(4 * c, 22).iter().map(|v| bf16(v * 0.5)).collect();
        let dy: Vec<f32> = det(rows * heads.max(1) * 4, 23)
            .iter()
            .map(|v| v * 0.3)
            .collect();
        let state0 = det(2 * 3 * c, 24);
        let d_x = exec.to_device(&x).expect("x");
        let d_w = exec.to_device_u8(&bf16_bytes(&w)).expect("w");
        let d_dy = exec.to_device(&dy).expect("dyn");
        let mut d_st = exec.to_device(&state0).expect("state");
        let d_off = exec
            .to_device_u32(&[0, lens[0] as u32, rows as u32])
            .expect("off");
        let d_slot = exec.to_device_u32(&[1, 0]).expect("slot");
        let mut d_o = exec.alloc(rows * c).expect("out");
        exec.inkling2_sconv(
            (&d_x, 0, c),
            (&mut d_o, 0, c),
            c,
            &d_w,
            dynamic.then_some((&d_dy, 0, heads * 4, heads)),
            &mut d_st,
            (&d_off, &d_slot, 2),
            rows,
            150,
        )
        .expect("sconv");
        let got = exec.to_host(&d_o).expect("out back");
        let st = exec.to_host(&d_st).expect("state back");
        let mut want = Vec::new();
        let mut want_st = state0.clone();
        let mut r0 = 0;
        for (i, &n) in lens.iter().enumerate() {
            let slot = [1usize, 0][i];
            let s_in = &state0[slot * 3 * c..(slot + 1) * 3 * c];
            let xs = &x[r0 * c..(r0 + n) * c];
            let dys = &dy[r0 * heads.max(1) * 4..(r0 + n) * heads.max(1) * 4];
            want.extend(sconv_ref(xs, c, &w, dynamic.then_some((dys, heads)), s_in));
            // the new state: the run's last three inputs
            let mut hist: Vec<f32> = s_in.to_vec();
            hist.extend_from_slice(xs);
            want_st[slot * 3 * c..(slot + 1) * 3 * c].copy_from_slice(&hist[hist.len() - 3 * c..]);
            r0 += n;
        }
        let rel = maxrel(&got, &want, 1e-2);
        println!("sconv {case}: max rel {rel:.2e}");
        assert!(
            rel < 1e-5,
            "sconv {case} parts from its definition by {rel}"
        );
        assert_eq!(st, want_st, "sconv {case}: the carried state");
    }
}

/// Slot 1004 against the chain it replaces (zero + slot_combine, slot 981,
/// copy + add), bit for bit: the experts' 8 partials a row and the
/// attention's 1, onto a residual and alone, two runs over carried states.
#[test]
fn inkling2_sconv_sum_matches_chain() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2_sconv_sum() {
        common::missing("pack has no slot 1004");
        return;
    }
    let c = 2304usize;
    let lens = [150usize, 2];
    let rows: usize = lens.iter().sum();
    let w: Vec<f32> = det(4 * c, 32).iter().map(|v| bf16(v * 0.5)).collect();
    let d_w = exec.to_device_u8(&bf16_bytes(&w)).expect("w");
    let state0 = det(2 * 3 * c, 33);
    let d_off = exec
        .to_device_u32(&[0, lens[0] as u32, rows as u32])
        .expect("off");
    let d_slot = exec.to_device_u32(&[1, 0]).expect("slot");
    let d_res = exec.to_device(&det(rows * c, 34)).expect("res");
    for (k, with_res) in [(8usize, true), (1, true), (1, false)] {
        let d_in = exec
            .to_device(&det(rows * k * c, 35 + k as u64))
            .expect("in");
        // the chain
        let mut d_y = exec.alloc(rows * c).expect("y");
        if k == 8 {
            exec.zero_region(&mut d_y, 0, rows * c).expect("zero");
            exec.moe_slot_combine(&d_in, &mut d_y, c, k, rows)
                .expect("combine");
        } else {
            exec.copy_region(&d_in, 0, &mut d_y, 0, rows * c)
                .expect("copy in");
        }
        let mut st_a = exec.to_device(&state0).expect("state a");
        let mut d_yc = exec.alloc(rows * c).expect("yc");
        exec.inkling2_sconv(
            (&d_y, 0, c),
            (&mut d_yc, 0, c),
            c,
            &d_w,
            None,
            &mut st_a,
            (&d_off, &d_slot, 2),
            rows,
            150,
        )
        .expect("sconv");
        let mut d_want = exec.alloc(rows * c).expect("want");
        if with_res {
            exec.copy_region(&d_res, 0, &mut d_want, 0, rows * c)
                .expect("copy res");
            exec.add(&mut d_want, &d_yc, rows * c).expect("add");
        } else {
            exec.copy_region(&d_yc, 0, &mut d_want, 0, rows * c)
                .expect("copy yc");
        }
        // one pass
        let mut st_b = exec.to_device(&state0).expect("state b");
        let mut d_got = exec.alloc(rows * c).expect("got");
        exec.inkling2_sconv_sum(
            (&d_in, k),
            with_res.then_some(&d_res),
            &mut d_got,
            c,
            &d_w,
            &mut st_b,
            (&d_off, &d_slot, 2),
            rows,
            150,
        )
        .expect("sconv sum");
        let (want, got) = (
            exec.to_host(&d_want).expect("want back"),
            exec.to_host(&d_got).expect("got back"),
        );
        let diff = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff, 0,
            "k {k} res {with_res}: {diff} outputs part from the chain"
        );
        let (sa, sb) = (
            exec.to_host(&st_a).expect("state a back"),
            exec.to_host(&st_b).expect("state b back"),
        );
        assert!(
            sa.iter().zip(&sb).all(|(a, b)| a.to_bits() == b.to_bits()),
            "k {k}: the carried states part"
        );
        println!("sconv_sum k {k} res {with_res}: bit-identical to the chain");
    }
}

/// Slot 1008 is rmsnorm_batch narrowed to bf16, bit for bit, at prompt and
/// decode widths (the two width elections).
#[test]
fn inkling2_rmsnorm_bf16_is_the_narrowed_norm() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_rmsnorm_batch_bf16() {
        common::missing("pack has no slot 1007");
        return;
    }
    let w = 2304usize;
    let d_w = exec
        .to_device(&det(w, 51).iter().map(|v| 1.0 + 0.3 * v).collect::<Vec<_>>())
        .expect("w");
    for rows in [3usize, 2048] {
        let d_x = exec.to_device(&det(rows * w, 52 + rows as u64)).expect("x");
        let mut d_f = exec.alloc(rows * w).expect("f32 rows");
        exec.rmsnorm_batch(&d_x, &d_w, &mut d_f, w, 1e-6, rows)
            .expect("norm");
        let mut want = exec.stream_alloc_bf16(rows * w).expect("want");
        exec.convert_f32_bf16(&d_f, &mut want, rows * w)
            .expect("narrow");
        let mut got = exec.stream_alloc_bf16(rows * w).expect("got");
        exec.rmsnorm_batch_bf16_at(&d_x, 0, &d_w, &mut got, w, 1e-6, rows)
            .expect("norm16");
        let (a, b) = (
            exec.to_host_bf16(&want).expect("want back"),
            exec.to_host_bf16(&got).expect("got back"),
        );
        let bad = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            bad, 0,
            "rows {rows}: slot 1007 parts from the narrowed norm in {bad}"
        );
    }
}

/// Slot 1009 against slot 978's cat and the bf16 ladder over it: the same
/// bf16 rows, so z parts only by f32 sum order (the ladder K-splits), and
/// the bf16 z the mix reads stays all but identical.
#[test]
fn inkling2_msg_down_matches_cat_gemm() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops");
        return;
    }
    let (w, rank) = (2304usize, 64usize);
    let down = paddock_engine::gpu::QuantTensor {
        bytes: exec
            .to_device_u8(&bf16_bytes(
                &det(rank * 2 * w, 61)
                    .iter()
                    .map(|v| v * 0.05)
                    .collect::<Vec<_>>(),
            ))
            .expect("down"),
        ty: paddock_models::ggml_type::GgmlType::Bf16,
        dims: vec![2 * w, rank],
    };
    for rows in [70usize, 2048] {
        let d_h = exec.to_device(&det(rows * w, 62)).expect("h");
        let d_m = exec
            .to_device(
                &det(rows * w, 63)
                    .iter()
                    .map(|v| 3.0 * v)
                    .collect::<Vec<_>>(),
            )
            .expect("m");
        let mut d_cat = exec.alloc(rows * 2 * w).expect("cat");
        let mut d_z0 = exec.alloc(rows * rank).expect("z0");
        exec.inkling2_msg_cat((&d_h, 0), (&d_m, 0), &mut d_cat, rows, w)
            .expect("cat");
        exec.bf16_gemm_rows(&down, 0, rank, &d_cat, &mut d_z0, rows)
            .expect("ladder");
        let mut d_z1 = exec.alloc(rows * rank).expect("z1");
        if !exec
            .inkling2_msg_down((&d_h, 0), (&d_m, 0), &down, &mut d_z1, rows, w, rank)
            .expect("msg down")
        {
            common::missing("pack has no slot 1008");
            return;
        }
        let (a, b) = (
            exec.to_host(&d_z0).expect("z0 back"),
            exec.to_host(&d_z1).expect("z1 back"),
        );
        let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let rel = maxrel(&b, &a, scale * 1e-3);
        let same16 = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| bf16(**x) == bf16(**y))
            .count();
        println!(
            "msg_down rows {rows}: max rel {rel:.2e} (scale {scale:.2}), bf16 z {:.3}% identical",
            100.0 * same16 as f64 / a.len() as f64
        );
        // f32 order over K = 4608 products
        assert!(
            rel < 1e-4,
            "rows {rows}: slot 1008 parts from cat + gemm by {rel}"
        );
        assert!(
            same16 * 100 >= a.len() * 99,
            "rows {rows}: bf16 z parts too often"
        );
    }
}

#[test]
fn inkling2_glue_swiglu_cap() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-986)");
        return;
    }
    let n = 4099usize;
    let g: Vec<f32> = det(n, 31).iter().map(|v| 30.0 * v).collect();
    let u: Vec<f32> = det(n, 32).iter().map(|v| 80.0 * v).collect();
    let d_g = exec.to_device(&g).expect("g");
    let d_u = exec.to_device(&u).expect("u");
    let mut d_o = exec.alloc(n).expect("out");
    exec.inkling2_swiglu_cap(&d_g, &d_u, &mut d_o, n, (4.0, 25.0))
        .expect("swiglu");
    let got = exec.to_host(&d_o).expect("back");
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&g, &u)| 4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp()) * 25.0 * (u / 25.0).tanh())
        .collect();
    let rel = maxrel(&got, &want, 1.0);
    println!("swiglu_cap: max rel {rel:.2e}");
    assert!(rel < 1e-5, "swiglu_cap parts from its definition by {rel}");
}

// ---- the activation quantizers, written out --------------------------------

fn e4m3_val(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    s * if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// f32 -> e4m3, nearest with ties to the even code, saturating at 448.
fn e4m3_rne(x: f32) -> u8 {
    let a = x.abs().min(448.0);
    let mut best = 0u8;
    for c in 1..=0x7Eu8 {
        let (d, db) = ((e4m3_val(c) - a).abs(), (e4m3_val(best) - a).abs());
        if d < db || (d == db && c % 2 == 0) {
            best = c;
        }
    }
    // the hardware convert keeps the sign of a zero (-0 = 0x80)
    best | if x.is_sign_negative() { 0x80 } else { 0 }
}

fn e2m1(x: f32) -> u8 {
    let a = x.abs();
    let c = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    c | if x.is_sign_negative() { 8 } else { 0 }
}

fn e2m1_val(c: u8) -> f32 {
    let v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(c & 7) as usize];
    if c & 8 != 0 { -v } else { v }
}

/// One row, four-over-six: (codes, scales, outer).
fn nvfp4_row(x: &[f32]) -> (Vec<u8>, Vec<u8>, f32) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let enc = (1536.0f32 / amax).min(f32::MAX);
    let dec = 1.0f32 / enc;
    let outer = amax * (1.0f32 / 1536.0);
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    for b in x.chunks(16) {
        let bmax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let mut best: Option<(f32, u8, Vec<u8>)> = None;
        for div in [1.0f32 / 6.0, 0.25] {
            let s = e4m3_val(e4m3_rne((bmax * (enc * div)).min(448.0)));
            let inv = (1.0f32 / (s * dec)).min(f32::MAX);
            let c: Vec<u8> = b.iter().map(|v| e2m1(v * inv)).collect();
            let mut err: Vec<f32> = c
                .iter()
                .zip(b)
                .map(|(&q, &v)| ((e2m1_val(q) * s) * outer - v).abs())
                .collect();
            while err.len() > 1 {
                err = err.chunks(2).map(|p| p[0] + p[1]).collect();
            }
            if best.as_ref().is_none_or(|(e, ..)| err[0] < *e) {
                best = Some((err[0], e4m3_rne(s), c));
            }
        }
        let (_, sb, c) = best.expect("a candidate");
        scales.push(sb);
        codes.extend(c.chunks(2).map(|p| p[0] | (p[1] << 4)));
    }
    (codes, scales, outer)
}

fn mxfp8_row(x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let mut q = Vec::new();
    let mut ex = Vec::new();
    for b in x.chunks(32) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let raw = amax * (1.0f32 / 448.0);
        let mut e = ((raw.to_bits() + 0x007F_FFFF) & 0x7F80_0000) >> 23;
        e = if raw <= 2f32.powi(-127) {
            0
        } else {
            e.min(254)
        };
        let scale = f32::from_bits(e << 23).max(2f32.powi(-127));
        let inv = if e == 0 { 2f32.powi(127) } else { 1.0 / scale };
        q.extend(b.iter().map(|v| e4m3_rne((v * inv).clamp(-448.0, 448.0))));
        ex.push(e as u8);
    }
    (q, ex)
}

#[test]
fn inkling2_quant_bit_exact() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-986)");
        return;
    }
    let (rows, k) = (5usize, 2304usize);
    // magnitudes spread over decades, an outlier, a zero block
    let mut x: Vec<f32> = det(rows * k, 41)
        .iter()
        .enumerate()
        .map(|(i, v)| v * 10f32.powi((i % 7) as i32 - 3))
        .collect();
    x[17] = 900.0;
    for v in &mut x[k..k + 16] {
        *v = 0.0;
    }
    let gamma = [1.0f32, 0.37, 2.5, 0.01, 1.0];
    let d_x = exec.to_device(&x).expect("x");
    let d_g = exec.to_device(&gamma).expect("gamma");
    for (fmt, use_gamma, bf) in [
        (0u32, false, false),
        (0, true, true),
        (1, false, false),
        (1, true, true),
    ] {
        let mut act = Inkling2Act::alloc(&exec, rows, k).expect("act");
        exec.inkling2_quant(
            &d_x,
            k,
            use_gamma.then_some(&d_g),
            bf,
            fmt,
            &mut act,
            rows,
            k,
        )
        .expect("quant");
        let (qb, sb) = if fmt == 0 {
            (k / 2, k / 16)
        } else {
            (k, k / 32)
        };
        let q = exec.to_host_u8_len(&act.q, rows * qb).expect("q");
        let s = exec.to_host_u8_len(&act.s, rows * sb).expect("s");
        let o = exec.to_host(&act.outer).expect("outer");
        for r in 0..rows {
            let row: Vec<f32> = x[r * k..(r + 1) * k]
                .iter()
                .map(|v| {
                    let a = if use_gamma { v * gamma[r] } else { *v };
                    if bf { bf16(a) } else { a }
                })
                .collect();
            if fmt == 0 {
                let (wq, ws, wo) = nvfp4_row(&row);
                assert_eq!(&q[r * qb..(r + 1) * qb], &wq[..], "nvfp4 codes, row {r}");
                assert_eq!(&s[r * sb..(r + 1) * sb], &ws[..], "nvfp4 scales, row {r}");
                assert_eq!(o[r].to_bits(), wo.to_bits(), "nvfp4 outer, row {r}");
            } else {
                let (wq, we) = mxfp8_row(&row);
                assert_eq!(&q[r * qb..(r + 1) * qb], &wq[..], "mxfp8 values, row {r}");
                assert_eq!(
                    &s[r * sb..(r + 1) * sb],
                    &we[..],
                    "mxfp8 exponents, row {r}"
                );
            }
        }
        println!("quant fmt {fmt} gamma {use_gamma} bf16 {bf}: bit exact over {rows} rows");
    }
}

#[test]
fn inkling2_moe_gemvs() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-986)");
        return;
    }
    let (ne, hidden, ff, topk, rows) = (6usize, 256usize, 64usize, 3usize, 2usize);
    let idx: Vec<u32> = vec![4, 1, 5, 0, 4, 2];
    let gamma: Vec<f32> = vec![0.7, 2.1, 0.4, 1.3, 0.9, 0.2];
    let lcg = |seed: u64, n: usize| -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect()
    };
    let x: Vec<f32> = det(rows * hidden, 51);
    let d_x = exec.to_device(&x).expect("x");
    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_gam = exec.to_device(&gamma).expect("gamma");
    for fmt in [0u32, 1] {
        // weights: random codes; scales in a sane band (e4m3 0x28..0x38 or
        // ue8m0 120..127)
        let (w13, s13, w2, s2): (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) = if fmt == 0 {
            (
                lcg(1, ne * 2 * ff * hidden / 2),
                lcg(2, ne * 2 * ff * hidden / 16)
                    .iter()
                    .map(|b| 0x28 + b % 16)
                    .collect(),
                lcg(3, ne * hidden * ff / 2),
                lcg(4, ne * hidden * ff / 16)
                    .iter()
                    .map(|b| 0x28 + b % 16)
                    .collect(),
            )
        } else {
            let fin = |v: Vec<u8>| {
                v.into_iter()
                    .map(|b| if b & 0x7F == 0x7F { b ^ 1 } else { b })
                    .collect::<Vec<u8>>()
            };
            (
                fin(lcg(1, ne * 2 * ff * hidden)),
                lcg(2, ne * 2 * ff * hidden / 32)
                    .iter()
                    .map(|b| 118 + b % 8)
                    .collect(),
                fin(lcg(3, ne * hidden * ff)),
                lcg(4, ne * hidden * ff / 32)
                    .iter()
                    .map(|b| 118 + b % 8)
                    .collect(),
            )
        };
        let fac: Vec<f32> = (0..ne).map(|e| 0.5 + 0.1 * e as f32).collect();
        let experts = Inkling2Experts {
            fmt,
            w13: exec.to_device_u8(&w13).expect("w13"),
            s13: exec.to_device_u8(&s13).expect("s13"),
            f13: (fmt == 0).then(|| exec.to_device(&fac).expect("f13")),
            w2: exec.to_device_u8(&w2).expect("w2"),
            s2: exec.to_device_u8(&s2).expect("s2"),
            f2: (fmt == 0).then(|| exec.to_device(&fac).expect("f2")),
            n_expert: ne,
            hidden,
            ff,
        };
        let mut xa = Inkling2Act::alloc(&exec, rows, hidden).expect("xa");
        exec.inkling2_quant(&d_x, hidden, None, true, fmt, &mut xa, rows, hidden)
            .expect("quant x");
        let mut act = exec.alloc(rows * topk * ff).expect("act");
        exec.inkling2_moe_gu(&experts, &xa, &d_idx, topk, &mut act, rows, (4.0, 25.0))
            .expect("gu");
        let got_act = exec.to_host(&act).expect("act back");
        let mut aa = Inkling2Act::alloc(&exec, rows * topk, ff).expect("aa");
        exec.inkling2_quant(&act, ff, Some(&d_gam), true, fmt, &mut aa, rows * topk, ff)
            .expect("quant act");
        let mut out = exec.alloc(rows * hidden).expect("out");
        exec.inkling2_moe_dn(&experts, &aa, &d_idx, topk, &mut out, rows)
            .expect("dn");
        let got_out = exec.to_host(&out).expect("out back");
        // host: the decoded operands, dot in f64
        let deq = |q: &[u8], s: &[u8], o: Option<f32>, n: usize, row: usize| -> Vec<f64> {
            if fmt == 0 {
                (0..n)
                    .map(|i| {
                        let b = q[row * n / 2 + i / 2];
                        let c = if i % 2 == 0 { b & 0xF } else { b >> 4 };
                        e2m1_val(c) as f64
                            * e4m3_val(s[row * n / 16 + i / 16]) as f64
                            * o.unwrap_or(1.0) as f64
                    })
                    .collect()
            } else {
                (0..n)
                    .map(|i| {
                        e4m3_val(q[row * n + i]) as f64
                            * 2f64.powi(s[row * n / 32 + i / 32] as i32 - 127)
                    })
                    .collect()
            }
        };
        let xq = exec.to_host_u8_len(&xa.q, rows * hidden).expect("xq");
        let xs = exec.to_host_u8_len(&xa.s, rows * hidden / 16).expect("xs");
        let xo = exec.to_host(&xa.outer).expect("xo");
        let mut want_act = vec![0f32; rows * topk * ff];
        for p in 0..rows * topk {
            let (r, e) = (p / topk, idx[p] as usize);
            let xv = deq(&xq, &xs, (fmt == 0).then_some(xo[r]), hidden, r);
            for i in 0..ff {
                let dot = |row: usize| -> f64 {
                    let wv = deq(
                        &w13,
                        &s13,
                        (fmt == 0).then_some(fac[e]),
                        hidden,
                        e * 2 * ff + row,
                    );
                    wv.iter().zip(&xv).map(|(a, b)| a * b).sum()
                };
                let g = bf16(dot(2 * i) as f32);
                let u = bf16(dot(2 * i + 1) as f32);
                want_act[p * ff + i] =
                    4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp()) * 25.0 * (u / 25.0).tanh();
            }
        }
        // the bf16 of g / u can land a ulp apart; compare at that width
        let rel = maxrel(&got_act, &want_act, 1e-2);
        println!("moe gu fmt {fmt}: max rel {rel:.2e}");
        assert!(
            rel < 1.6e-2,
            "moe gu fmt {fmt} parts from the decoded dot by {rel}"
        );
        let aq = exec.to_host_u8_len(&aa.q, rows * topk * ff).expect("aq");
        let as_ = exec
            .to_host_u8_len(&aa.s, rows * topk * ff / 16)
            .expect("as");
        let ao = exec.to_host(&aa.outer).expect("ao");
        let mut want_out = vec![0f32; rows * hidden];
        for r in 0..rows {
            for o in 0..hidden {
                let mut sum = 0f64;
                for j in 0..topk {
                    let p = r * topk + j;
                    let e = idx[p] as usize;
                    let av = deq(&aq, &as_, (fmt == 0).then_some(ao[p]), ff, p);
                    let wv = deq(&w2, &s2, (fmt == 0).then_some(fac[e]), ff, e * hidden + o);
                    sum += wv.iter().zip(&av).map(|(a, b)| a * b).sum::<f64>();
                }
                want_out[r * hidden + o] = sum as f32;
            }
        }
        let scale = want_out.iter().fold(0f32, |m, v| m.max(v.abs()));
        let rel = maxrel(&got_out, &want_out, scale * 1e-3);
        println!("moe dn fmt {fmt}: max rel {rel:.2e} (scale {scale:.3})");
        assert!(
            rel < 1e-4,
            "moe dn fmt {fmt} parts from the decoded dot by {rel}"
        );

        // the prompt-width arm over the same wave: sorted blocks, one weight
        // tile a block, per-pair partials folded in slot order
        // planes for a wider wave than this one: the arms must launch over
        // the sorted wave's blocks only (the rest hold no rows of it)
        let mut sorted = Inkling2Sorted::alloc(&exec, 64 * rows, topk, ne).expect("sorted");
        exec.inkling2_sort(&d_idx, &mut sorted, rows, topk, ne, 32)
            .expect("sort");
        let mut act_g = exec.alloc(rows * topk * ff).expect("act grp");
        exec.inkling2_moe_gu_grp(&experts, &xa, &sorted, topk, &mut act_g, (4.0, 25.0))
            .expect("gu grp");
        let got_g = exec.to_host(&act_g).expect("act grp back");
        let rel = maxrel(&got_g, &got_act, 1e-2);
        println!("moe gu grp fmt {fmt}: max rel to the GEMV {rel:.2e}");
        assert!(
            rel < 1.6e-2,
            "moe gu grp fmt {fmt} parts from the GEMV by {rel}"
        );
        let mut part = exec.alloc(rows * topk * hidden).expect("part");
        exec.inkling2_moe_dn_grp(&experts, &aa, &sorted, topk, &mut part)
            .expect("dn grp");
        let mut out_g = exec.to_device(&vec![0f32; rows * hidden]).expect("out grp");
        exec.moe_slot_combine(&part, &mut out_g, hidden, topk, rows)
            .expect("combine");
        let got_og = exec.to_host(&out_g).expect("out grp back");
        let rel = maxrel(&got_og, &want_out, scale * 1e-3);
        println!("moe dn grp fmt {fmt}: max rel {rel:.2e}");
        assert!(
            rel < 1e-4,
            "moe dn grp fmt {fmt} parts from the decoded dot by {rel}"
        );
    }
}

/// Slots 991-992 (the experts on the block-scale tensor cores, NVFP4 and
/// MXFP8) against slots 987-988 on a wave wide enough to fill a block's
/// pair columns and several weight tiles: the products are exact in both, so
/// the down partials agree to f32 order; the gate|up output rounds through
/// bf16 like the reference, where an order-level difference can flip one
/// ulp.
#[test]
fn inkling2_moe_tc_matches_cc() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-992)");
        return;
    }
    let (ne, hidden, ff, topk, rows) = (8usize, 512usize, 256usize, 4usize, 48usize);
    let lcg = |seed: u64, n: usize| -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect()
    };
    // each token's topk distinct experts
    let idx: Vec<u32> = (0..rows)
        .flat_map(|r| (0..topk).map(move |j| ((r * 3 + j * 5) % ne) as u32))
        .collect();
    let gamma: Vec<f32> = det(rows * topk, 81).iter().map(|v| 0.6 + 0.5 * v).collect();
    let x: Vec<f32> = det(rows * hidden, 82);
    let fac: Vec<f32> = (0..ne).map(|e| 0.5 + 0.1 * e as f32).collect();
    let d_x = exec.to_device(&x).expect("x");
    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_gam = exec.to_device(&gamma).expect("gamma");
    let mut sorted = Inkling2Sorted::alloc(&exec, rows, topk, ne).expect("sorted");
    exec.inkling2_sort(&d_idx, &mut sorted, rows, topk, ne, 32)
        .expect("sort");
    let pairs = rows * topk;
    for fmt in [0u32, 1] {
        // NVFP4: random codes, e4m3 scales in a sane band; MXFP8: e4m3
        // codes without NaN, ue8m0 scales 118..125
        let (cb, sb) = if fmt == 0 { (2, 16) } else { (1, 32) };
        let codes = |seed, n| {
            let v = lcg(seed, n);
            if fmt == 0 {
                v
            } else {
                v.into_iter()
                    .map(|b| if b & 0x7F == 0x7F { b ^ 1 } else { b })
                    .collect()
            }
        };
        let scales = |seed, n| -> Vec<u8> {
            let v = lcg(seed, n);
            if fmt == 0 {
                v.iter().map(|b| 0x28 + b % 16).collect()
            } else {
                v.iter().map(|b| 118 + b % 8).collect()
            }
        };
        let experts = Inkling2Experts {
            fmt,
            w13: exec
                .to_device_u8(&codes(11, ne * 2 * ff * hidden / cb))
                .expect("w13"),
            s13: exec
                .to_device_u8(&scales(12, ne * 2 * ff * hidden / sb))
                .expect("s13"),
            f13: (fmt == 0).then(|| exec.to_device(&fac).expect("f13")),
            w2: exec
                .to_device_u8(&codes(13, ne * hidden * ff / cb))
                .expect("w2"),
            s2: exec
                .to_device_u8(&scales(14, ne * hidden * ff / sb))
                .expect("s2"),
            f2: (fmt == 0).then(|| exec.to_device(&fac).expect("f2")),
            n_expert: ne,
            hidden,
            ff,
        };
        let mut xa = Inkling2Act::alloc(&exec, rows, hidden).expect("xa");
        exec.inkling2_quant(&d_x, hidden, None, true, fmt, &mut xa, rows, hidden)
            .expect("quant x");
        let mut act_cc = exec.alloc(pairs * ff).expect("act cc");
        let mut act_tc = exec.alloc(pairs * ff).expect("act tc");
        exec.inkling2_moe_gu_cc(&experts, &xa, &sorted, topk, &mut act_cc, (4.0, 25.0))
            .expect("gu cc");
        if !exec
            .inkling2_moe_gu_tc(&experts, &xa, &sorted, topk, &mut act_tc, (4.0, 25.0))
            .expect("gu tc")
        {
            common::missing("no block-scale tensor cores on this die (slots 991-992 declined)");
            return;
        }
        let (a_cc, a_tc) = (
            exec.to_host(&act_cc).expect("cc"),
            exec.to_host(&act_tc).expect("tc"),
        );
        // an f32 order difference shows only where it flips a bf16 rounding
        // of the gate or up: rare, and one bf16 ulp of the input's size
        let same = a_cc.iter().zip(&a_tc).filter(|(a, b)| a == b).count();
        let scale = a_cc.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = a_cc
            .iter()
            .zip(&a_tc)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "fmt {fmt} gu tc vs cc: max |diff| {d:.2e} (scale {scale:.3}), {:.2}% identical",
            100.0 * same as f64 / a_cc.len() as f64
        );
        assert!(d < 1e-2 * scale, "fmt {fmt} gu tc parts from cc by {d}");
        assert!(
            same * 100 >= a_cc.len() * 99,
            "fmt {fmt} gu tc: only {same} of {} identical",
            a_cc.len()
        );
        // the down input from the CUDA-core act, so the down arms see one input
        let mut aa = Inkling2Act::alloc(&exec, pairs, ff).expect("aa");
        exec.inkling2_quant(&act_cc, ff, Some(&d_gam), true, fmt, &mut aa, pairs, ff)
            .expect("quant act");
        let mut part_cc = exec.alloc(pairs * hidden).expect("part cc");
        let mut part_tc = exec.alloc(pairs * hidden).expect("part tc");
        exec.inkling2_moe_dn_cc(&experts, &aa, &sorted, topk, &mut part_cc)
            .expect("dn cc");
        assert!(
            exec.inkling2_moe_dn_tc(&experts, &aa, &sorted, topk, &mut part_tc)
                .expect("dn tc")
        );
        let (p_cc, p_tc) = (
            exec.to_host(&part_cc).expect("cc"),
            exec.to_host(&part_tc).expect("tc"),
        );
        let scale = p_cc.iter().fold(0f32, |m, v| m.max(v.abs()));
        let rel = maxrel(&p_tc, &p_cc, scale * 1e-3);
        println!("fmt {fmt} dn tc vs cc: max rel {rel:.2e} (scale {scale:.3})");
        assert!(rel < 1e-4, "fmt {fmt} dn tc parts from cc by {rel}");
    }
}

#[test]
fn inkling2_moe_ms_matches_tc() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2_ms() {
        common::missing("no slots 1002-1003 or no block-scale tensor cores here");
        return;
    }
    // the model's planes and experts; routing skewed so experts hold from
    // a few pairs to several blocks' worth
    let (ne, hidden, ff, topk) = (128usize, 2304usize, 1152usize, 8usize);
    let lcg = |seed: u64, n: usize| -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect()
    };
    let fac: Vec<f32> = (0..ne).map(|e| 0.5 + 0.01 * e as f32).collect();
    let experts = Inkling2Experts {
        fmt: 0,
        w13: exec
            .to_device_u8(&lcg(21, ne * 2 * ff * hidden / 2))
            .expect("w13"),
        s13: exec
            .to_device_u8(
                &lcg(22, ne * 2 * ff * hidden / 16)
                    .iter()
                    .map(|b| 0x28 + b % 16)
                    .collect::<Vec<_>>(),
            )
            .expect("s13"),
        f13: Some(exec.to_device(&fac).expect("f13")),
        w2: exec
            .to_device_u8(&lcg(23, ne * hidden * ff / 2))
            .expect("w2"),
        s2: exec
            .to_device_u8(
                &lcg(24, ne * hidden * ff / 16)
                    .iter()
                    .map(|b| 0x28 + b % 16)
                    .collect::<Vec<_>>(),
            )
            .expect("s2"),
        f2: Some(exec.to_device(&fac).expect("f2")),
        n_expert: ne,
        hidden,
        ff,
    };
    for rows in [8usize, 16, 32, 64, 96, 256, 512, 1024, 2048] {
        let idx: Vec<u32> = (0..rows)
            .flat_map(|r| {
                let mut picks = Vec::with_capacity(topk);
                let mut h = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                while picks.len() < topk {
                    h = h
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    // squared draw: low experts popular
                    let u = (h >> 40) as f64 / (1u64 << 24) as f64;
                    let e = ((u * u) * ne as f64) as u32;
                    if !picks.contains(&e) {
                        picks.push(e);
                    }
                }
                picks
            })
            .collect();
        let pairs = rows * topk;
        let x = det(rows * hidden, 91 + rows as u64);
        let gamma: Vec<f32> = det(pairs, 92).iter().map(|v| 0.6 + 0.5 * v).collect();
        let d_x = exec.to_device(&x).expect("x");
        let d_idx = exec.to_device_u32(&idx).expect("idx");
        let d_gam = exec.to_device(&gamma).expect("gamma");
        let mut s32 = Inkling2Sorted::alloc(&exec, rows, topk, ne).expect("s32");
        let mut s64 = Inkling2Sorted::alloc(&exec, rows, topk, ne).expect("s64");
        exec.inkling2_sort(&d_idx, &mut s32, rows, topk, ne, 32)
            .expect("sort 32");
        exec.inkling2_sort(&d_idx, &mut s64, rows, topk, ne, 64)
            .expect("sort 64");
        let mut xa = Inkling2Act::alloc(&exec, rows, hidden).expect("xa");
        exec.inkling2_quant(&d_x, hidden, None, true, 0, &mut xa, rows, hidden)
            .expect("quant x");
        let mut act_tc = exec.alloc(pairs * ff).expect("act tc");
        let mut act_ms = exec.alloc(pairs * ff).expect("act ms");
        let caps = (4.0, 25.0);
        exec.inkling2_moe_gu_grp(&experts, &xa, &s32, topk, &mut act_tc, caps)
            .expect("gu tc");
        exec.inkling2_moe_gu_grp(&experts, &xa, &s64, topk, &mut act_ms, caps)
            .expect("gu ms");
        let (a_tc, a_ms) = (
            exec.to_host(&act_tc).expect("tc"),
            exec.to_host(&act_ms).expect("ms"),
        );
        let diff = a_tc
            .iter()
            .zip(&a_ms)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff,
            0,
            "rows {rows}: gu ms parts from tc in {diff} of {}",
            a_tc.len()
        );
        let mut aa = Inkling2Act::alloc(&exec, pairs, ff).expect("aa");
        exec.inkling2_quant(&act_tc, ff, Some(&d_gam), true, 0, &mut aa, pairs, ff)
            .expect("quant act");
        let mut p_tc = exec.alloc(pairs * hidden).expect("part tc");
        let mut p_ms = exec.alloc(pairs * hidden).expect("part ms");
        exec.inkling2_moe_dn_grp(&experts, &aa, &s32, topk, &mut p_tc)
            .expect("dn tc");
        exec.inkling2_moe_dn_grp(&experts, &aa, &s64, topk, &mut p_ms)
            .expect("dn ms");
        let (q_tc, q_ms) = (
            exec.to_host(&p_tc).expect("tc"),
            exec.to_host(&p_ms).expect("ms"),
        );
        let diff = q_tc
            .iter()
            .zip(&q_ms)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff,
            0,
            "rows {rows}: dn ms parts from tc in {diff} of {}",
            q_tc.len()
        );
        if exec.has_inkling2_ms16() {
            // slot 1006 + the bf16 read: slot 983's codes off the f32 act
            let mut act16 = exec.alloc(pairs * ff).expect("act16");
            exec.inkling2_moe_gu_ms16(&experts, &xa, &s64, topk, &mut act16, &d_gam, caps)
                .expect("gu ms16");
            let mut aa16 = Inkling2Act::alloc(&exec, pairs, ff).expect("aa16");
            exec.inkling2_quant16(&act16, 0, &mut aa16, pairs, ff)
                .expect("quant16");
            let planes = |a: &Inkling2Act| {
                (
                    exec.to_host_u8_len(&a.q, pairs * ff / 2).expect("codes"),
                    exec.to_host_u8_len(&a.s, pairs * ff / 16).expect("scales"),
                    exec.to_host(&a.outer).expect("outer"),
                )
            };
            let (q32, s32_, o32) = planes(&aa);
            let (q16, s16, o16) = planes(&aa16);
            assert_eq!(
                q32, q16,
                "rows {rows}: slot 1006's codes part from slot 983's"
            );
            assert_eq!(
                s32_, s16,
                "rows {rows}: slot 1006's scales part from slot 983's"
            );
            assert!(
                o32[..pairs]
                    .iter()
                    .zip(&o16[..pairs])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "rows {rows}: slot 1006's outer scales part"
            );
        }
        // timing, both arms on the same wave
        let time = |f: &mut dyn FnMut()| {
            f();
            exec.synchronize().expect("sync");
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f();
            }
            exec.synchronize().expect("sync");
            t.elapsed().as_secs_f64() * 1e6 / 20.0
        };
        let gu32 = time(&mut || {
            exec.inkling2_moe_gu_grp(&experts, &xa, &s32, topk, &mut act_tc, caps)
                .expect("gu");
        });
        let gu64 = time(&mut || {
            exec.inkling2_moe_gu_grp(&experts, &xa, &s64, topk, &mut act_ms, caps)
                .expect("gu");
        });
        let dn32 = time(&mut || {
            exec.inkling2_moe_dn_grp(&experts, &aa, &s32, topk, &mut p_tc)
                .expect("dn");
        });
        let dn64 = time(&mut || {
            exec.inkling2_moe_dn_grp(&experts, &aa, &s64, topk, &mut p_ms)
                .expect("dn");
        });
        println!(
            "rows {rows}: gu 32-pair {gu32:.0} us / 64-pair {gu64:.0} us, \
             dn {dn32:.0} / {dn64:.0} us (bit-identical)"
        );
    }
}

/// The attention by definition, f64: rows at p0.., k / v [pos][kv * 128].
#[allow(clippy::too_many_arguments)]
fn attn_ref(
    q: &[f32],
    rel: &[f32],
    sink: &[f32],
    k: &[f32],
    v: &[f32],
    proj: &[f32],
    rows: usize,
    p0: usize,
    heads: usize,
    kvh: usize,
    window: usize,
) -> Vec<f32> {
    let (hd, dr, ext) = (128usize, 16usize, 1024usize);
    let g = heads / kvh;
    let mut out = vec![0f32; rows * heads * hd];
    for r in 0..rows {
        let p = p0 + r;
        let lo = if window > 0 {
            (p + 1).saturating_sub(window)
        } else {
            0
        };
        for h in 0..heads {
            let kh = h / g;
            let qv = &q[(r * heads + h) * hd..(r * heads + h + 1) * hd];
            let rv = &rel[(r * heads + h) * dr..(r * heads + h + 1) * dr];
            let mut lg: Vec<f64> = (lo..=p)
                .map(|j| {
                    let kv = &k[j * kvh * hd + kh * hd..j * kvh * hd + (kh + 1) * hd];
                    let mut s = qv
                        .iter()
                        .zip(kv)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>()
                        / (hd as f64).sqrt();
                    let d = p - j;
                    if d < ext {
                        s += (0..dr)
                            .map(|i| rv[i] as f64 * proj[i * ext + d] as f64)
                            .sum::<f64>();
                    }
                    s
                })
                .collect();
            let sk = sink[r * heads + h] as f64;
            let m = lg.iter().cloned().fold(sk, f64::max);
            for l in &mut lg {
                *l = (*l - m).exp();
            }
            let den = lg.iter().sum::<f64>() + (sk - m).exp();
            for d in 0..hd {
                let a: f64 = (lo..=p)
                    .zip(&lg)
                    .map(|(j, e)| e * v[j * kvh * hd + kh * hd + d] as f64)
                    .sum();
                out[(r * heads + h) * hd + d] = (a / den) as f32;
            }
        }
    }
    out
}

#[test]
fn inkling2_attn_matches_definition() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-986)");
        return;
    }
    let (heads, kvh, hd, dr) = (24usize, 8usize, 128usize, 16usize);
    let (rows, p0) = (4usize, 1100usize);
    let keys = p0 + rows;
    // clipped queries and keys reach |q.k| ~ 45 / sqrt(128): scale them so
    let q: Vec<f32> = det(rows * heads * hd, 61).iter().map(|v| v * 2.0).collect();
    let k: Vec<f32> = det(keys * kvh * hd, 62).iter().map(|v| v * 2.0).collect();
    let v = det(keys * kvh * hd, 63);
    let rel = det(rows * heads * dr, 64);
    let sink: Vec<f32> = det(rows * heads, 65).iter().map(|v| 3.0 * v).collect();
    let proj: Vec<f32> = det(dr * 1024, 66).iter().map(|v| bf16(v * 0.5)).collect();
    let d_q = exec.to_device(&q).expect("q");
    let d_k = exec.to_device(&k).expect("k");
    let d_v = exec.to_device(&v).expect("v");
    let d_r = exec.to_device(&rel).expect("rel");
    let d_s = exec.to_device(&sink).expect("sink");
    let d_p = exec.to_device_u8(&bf16_bytes(&proj)).expect("proj");
    for window in [1024usize, 0] {
        let mut d_o = exec.alloc(rows * heads * hd).expect("out");
        exec.inkling2_attn(
            (&d_q, 0, heads * hd),
            (&d_r, 0, heads * dr),
            (&d_s, 0, heads),
            (&d_k, &d_v),
            &d_p,
            &mut d_o,
            rows,
            p0,
            (heads, kvh),
            (window, 1024),
            1.0 / (hd as f32).sqrt(),
        )
        .expect("attn");
        let got = exec.to_host(&d_o).expect("back");
        let want = attn_ref(&q, &rel, &sink, &k, &v, &proj, rows, p0, heads, kvh, window);
        let d = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("attn window {window}: max |diff| {d:.2e}");
        assert!(
            d < 2e-5,
            "attn window {window} parts from its definition by {d}"
        );
    }
}

/// Slots 989-990: two sequences' K / V through the store (paged over a
/// shuffled pool, or ring) in each class, then rows of both at their own
/// positions through the split attention - against slot 986's definition
/// over the rows the store holds (read back and decoded, its codes checked
/// against the rounding rule on the way).
#[test]
fn inkling2_kv_store_attn_matches_definition() {
    use paddock_engine::gpu::Inkling2KvLayout;
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-990)");
        return;
    }
    let (heads, kvh, hd, dr) = (24usize, 8usize, 128usize, 16usize);
    let kv_dim = kvh * hd;
    // slot 0 wraps its 2048-row ring; slot 1 is short
    let lens = [2100usize, 700];
    let ring = 2048usize;
    let bps = lens[0].div_ceil(16);
    let n_blocks = 2 * bps;
    // a shuffled block table: every slot's blocks scattered over the pool
    let mut perm: Vec<u32> = (0..n_blocks as u32).collect();
    let mut st = 0x2545_f491_4f6c_dd1du64;
    for i in (1..perm.len()).rev() {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        perm.swap(i, (st % (i as u64 + 1)) as usize);
    }
    let d_bt = exec.to_device_u32(&perm).expect("bt");
    let kv: Vec<Vec<f32>> = (0..2)
        .map(|s| {
            det(lens[s] * kv_dim, 70 + s as u64)
                .iter()
                .map(|v| v * 2.0)
                .collect()
        })
        .collect();
    let vv: Vec<Vec<f32>> = (0..2)
        .map(|s| det(lens[s] * kv_dim, 72 + s as u64))
        .collect();
    // every position of both slots, one store call
    let (mut slots, mut pos) = (Vec::new(), Vec::new());
    for (s, &l) in lens.iter().enumerate() {
        slots.extend(std::iter::repeat_n(s as u32, l));
        pos.extend(0..l as u32);
    }
    let all = slots.len();
    let d_slots = exec.to_device_u32(&slots).expect("slots");
    let d_pos = exec.to_device_u32(&pos).expect("pos");
    let d_ksrc = exec.to_device(&kv.concat()).expect("k");
    let d_vsrc = exec.to_device(&vv.concat()).expect("v");
    // the query rows: slot 0's last three, slot 1's last
    let rows = 4usize;
    let (qs, qp): (Vec<u32>, Vec<u32>) = (vec![0, 0, 0, 1], vec![2097, 2098, 2099, 699]);
    let d_qs = exec.to_device_u32(&qs).expect("qs");
    let d_qp = exec.to_device_u32(&qp).expect("qp");
    let q: Vec<f32> = det(rows * heads * hd, 74).iter().map(|v| v * 2.0).collect();
    let rel = det(rows * heads * dr, 75);
    let sink: Vec<f32> = det(rows * heads, 76).iter().map(|v| 3.0 * v).collect();
    let proj: Vec<f32> = det(dr * 1024, 77).iter().map(|v| bf16(v * 0.5)).collect();
    let d_q = exec.to_device(&q).expect("q");
    let d_r = exec.to_device(&rel).expect("rel");
    let d_s = exec.to_device(&sink).expect("sink");
    let d_p = exec.to_device_u8(&bf16_bytes(&proj)).expect("proj");
    // the prompt runs' 110 rows for slot 993
    let q2: Vec<f32> = det(110 * heads * hd, 78).iter().map(|v| v * 2.0).collect();
    let rel2 = det(110 * heads * dr, 79);
    let sink2: Vec<f32> = det(110 * heads, 80).iter().map(|v| 3.0 * v).collect();
    let d_q2 = exec.to_device(&q2).expect("q2");
    let d_r2 = exec.to_device(&rel2).expect("rel2");
    let d_s2 = exec.to_device(&sink2).expect("sink2");
    let mut part_o = exec.alloc(rows * heads * 16 * hd).expect("part o");
    let mut part_ml = exec.alloc(rows * heads * 16 * 2).expect("part ml");
    for dtype in [2u32, 0, 1] {
        let eb = [2usize, 1, 4][dtype as usize];
        for paged in [true, false] {
            let store_rows = if paged { n_blocks * 16 } else { 2 * ring };
            let layout = || {
                if paged {
                    Inkling2KvLayout::Paged(&d_bt, bps)
                } else {
                    Inkling2KvLayout::Ring(ring)
                }
            };
            let row_of = |s: usize, p: usize| {
                if paged {
                    perm[s * bps + p / 16] as usize * 16 + p % 16
                } else {
                    s * ring + p % ring
                }
            };
            let mut d_k = exec.alloc_u8(store_rows * kv_dim * eb).expect("k store");
            let mut d_v = exec.alloc_u8(store_rows * kv_dim * eb).expect("v store");
            for (src, dst) in [(&d_ksrc, &mut d_k), (&d_vsrc, &mut d_v)] {
                exec.inkling2_kv_store(
                    (src, 0, kv_dim),
                    (&d_slots, &d_pos),
                    dst,
                    layout(),
                    dtype,
                    kv_dim,
                    all,
                )
                .expect("store");
            }
            // the stored rows, decoded; their codes against the rounding rule
            let kb = exec.to_host_u8_len(&d_k, d_k.len()).expect("k back");
            let vb = exec.to_host_u8_len(&d_v, d_v.len()).expect("v back");
            let decode = |b: &[u8], s: usize, src: &[f32]| -> Vec<f32> {
                let mut out = Vec::with_capacity(lens[s] * kv_dim);
                for p in 0..lens[s] {
                    // a ring keeps only its last `ring` rows
                    if !paged && p + ring < lens[s] {
                        out.extend(std::iter::repeat_n(f32::NAN, kv_dim));
                        continue;
                    }
                    let base = row_of(s, p) * kv_dim * eb;
                    for i in 0..kv_dim {
                        let x = src[p * kv_dim + i];
                        let at = base + i * eb;
                        let y = match dtype {
                            2 => f32::from_le_bytes(b[at..at + 4].try_into().expect("4")),
                            0 => {
                                let h = half::f16::from_le_bytes([b[at], b[at + 1]]);
                                assert_eq!(
                                    h,
                                    half::f16::from_f32(x),
                                    "f16 code at ({s}, {p}, {i})"
                                );
                                h.to_f32()
                            }
                            _ => {
                                if i < 64 {
                                    assert_eq!(b[at], e4m3_rne(x), "e4m3 code at ({s}, {p}, {i})");
                                }
                                e4m3_val(b[at])
                            }
                        };
                        if dtype == 2 {
                            assert_eq!(y, x, "f32 row at ({s}, {p}, {i})");
                        }
                        out.push(y);
                    }
                }
                out
            };
            let hk: Vec<Vec<f32>> = (0..2).map(|s| decode(&kb, s, &kv[s])).collect();
            let hv: Vec<Vec<f32>> = (0..2).map(|s| decode(&vb, s, &vv[s])).collect();
            let window = if paged { 0 } else { 1024 };
            for splits in [1usize, 5] {
                let mut d_o = exec.alloc(rows * heads * hd).expect("out");
                exec.inkling2_attn_kv(
                    (&d_q, 0, heads * hd),
                    (&d_r, 0, heads * dr),
                    (&d_s, 0, heads),
                    (&d_k, &d_v),
                    layout(),
                    dtype,
                    (&d_qs, &d_qp),
                    &d_p,
                    &mut d_o,
                    Some((&mut part_o, &mut part_ml)),
                    rows,
                    (heads, kvh),
                    (window, 1024),
                    splits,
                    1.0 / (hd as f32).sqrt(),
                )
                .expect("attn kv");
                let got = exec.to_host(&d_o).expect("back");
                let mut d = 0f32;
                for r in 0..rows {
                    let s = qs[r] as usize;
                    let want = attn_ref(
                        &q[r * heads * hd..],
                        &rel[r * heads * dr..],
                        &sink[r * heads..],
                        &hk[s],
                        &hv[s],
                        &proj,
                        1,
                        qp[r] as usize,
                        heads,
                        kvh,
                        window,
                    );
                    let g = &got[r * heads * hd..(r + 1) * heads * hd];
                    d = g
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(d, f32::max);
                }
                println!(
                    "attn_kv class {dtype} {} splits {splits}: max |diff| {d:.2e}",
                    if paged { "paged" } else { "ring" }
                );
                assert!(
                    d < 2e-5,
                    "attn_kv class {dtype} paged {paged} splits {splits} parts by {d}"
                );
            }
            // slot 993 over the same stores: two prompt runs (slot 0's last
            // 60 rows, slot 1's last 50) in 32-row tiles; f16 queries and
            // probabilities, so a looser bound than the f32 kernel's
            if dtype < 2 && exec.has_inkling2_tile() {
                let runs = [(0usize, lens[0] - 60, 60usize), (1, lens[1] - 50, 50)];
                let mut tiles = Vec::new();
                let mut row = 0usize;
                for &(slot, p0, n) in &runs {
                    for j in (0..n).step_by(32) {
                        let k = 32.min(n - j);
                        tiles.extend([(row + j) as u32, k as u32, slot as u32, (p0 + j) as u32]);
                    }
                    row += n;
                }
                let d_tiles = exec.to_device_u32(&tiles).expect("tiles");
                let mut bank_t = vec![0f32; 1024 * dr];
                for i in 0..dr {
                    for d in 0..1024 {
                        bank_t[d * dr + i] = proj[i * 1024 + d];
                    }
                }
                let d_pt = exec.to_device_u8(&bf16_bytes(&bank_t)).expect("bank t");
                let mut d_o = exec.alloc(row * heads * hd).expect("tile out");
                assert!(
                    exec.inkling2_attn_tile(
                        (&d_q2, 0, heads * hd),
                        (&d_r2, 0, heads * dr),
                        (&d_s2, 0, heads),
                        (&d_k, &d_v),
                        layout(),
                        dtype,
                        (&d_tiles, tiles.len() / 4),
                        &d_pt,
                        Inkling2AttnOut::Plain(&mut d_o),
                        (heads, kvh),
                        (window, 1024),
                        1.0 / (hd as f32).sqrt(),
                    )
                    .expect("attn tile")
                );
                let got = exec.to_host(&d_o).expect("back");
                let (mut d, mut r) = (0f32, 0usize);
                for &(slot, p0, n) in &runs {
                    let want = attn_ref(
                        &q2[r * heads * hd..],
                        &rel2[r * heads * dr..],
                        &sink2[r * heads..],
                        &hk[slot],
                        &hv[slot],
                        &proj,
                        n,
                        p0,
                        heads,
                        kvh,
                        window,
                    );
                    let g = &got[r * heads * hd..(r + n) * heads * hd];
                    d = g
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(d, f32::max);
                    r += n;
                }
                println!(
                    "attn_tile class {dtype} {}: max |diff| {d:.2e}",
                    if paged { "paged" } else { "ring" }
                );
                assert!(
                    d < 1e-2,
                    "attn_tile class {dtype} paged {paged} parts by {d}"
                );
                // slot 1005: the gate folded in == the tile, then mul_sigmoid
                if exec.has_inkling2_tile_g() {
                    let d_g = exec.to_device(&det(row * heads * hd, 77)).expect("gate");
                    let mut d_og = exec.alloc(row * heads * hd).expect("gated out");
                    assert!(
                        exec.inkling2_attn_tile(
                            (&d_q2, 0, heads * hd),
                            (&d_r2, 0, heads * dr),
                            (&d_s2, 0, heads),
                            (&d_k, &d_v),
                            layout(),
                            dtype,
                            (&d_tiles, tiles.len() / 4),
                            &d_pt,
                            Inkling2AttnOut::Gated(&mut d_og, &d_g, heads * hd),
                            (heads, kvh),
                            (window, 1024),
                            1.0 / (hd as f32).sqrt(),
                        )
                        .expect("gated tile")
                    );
                    let mut d_o16 = exec.stream_alloc_bf16(row * heads * hd).expect("bf16 out");
                    assert!(
                        exec.inkling2_attn_tile(
                            (&d_q2, 0, heads * hd),
                            (&d_r2, 0, heads * dr),
                            (&d_s2, 0, heads),
                            (&d_k, &d_v),
                            layout(),
                            dtype,
                            (&d_tiles, tiles.len() / 4),
                            &d_pt,
                            Inkling2AttnOut::Gated16(&mut d_o16, &d_g, heads * hd),
                            (heads, kvh),
                            (window, 1024),
                            1.0 / (hd as f32).sqrt(),
                        )
                        .expect("gated bf16 tile")
                    );
                    let mut d_n16 = exec.stream_alloc_bf16(row * heads * hd).expect("narrowed");
                    exec.convert_f32_bf16(&d_og, &mut d_n16, row * heads * hd)
                        .expect("narrow");
                    let (h1, h2) = (
                        exec.to_host_bf16(&d_n16).expect("narrowed back"),
                        exec.to_host_bf16(&d_o16).expect("bf16 back"),
                    );
                    assert!(
                        h1.iter().zip(&h2).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "attn_tile_g bf16 class {dtype} paged {paged}: parts from the narrowed rows"
                    );
                    exec.mul_sigmoid(&mut d_o, &d_g, row * heads * hd)
                        .expect("mul sigmoid");
                    let (a, b) = (
                        exec.to_host(&d_o).expect("plain"),
                        exec.to_host(&d_og).expect("gated"),
                    );
                    assert!(
                        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                        "attn_tile_g class {dtype} paged {paged}: parts from tile + mul_sigmoid"
                    );
                }
                // slot 1009: the first run's tiles (60 rows from row 0) split
                // over keys and folded by slot 990's combine, against the
                // same tiles unsplit
                if exec.has_inkling2_tile_split() {
                    let n0 = runs[0].2;
                    let t0: Vec<u32> = tiles[..n0.div_ceil(32) * 4].to_vec();
                    let d_t0 = exec.to_device_u32(&t0).expect("tiles0");
                    let mut d_a = exec.alloc(row * heads * hd).expect("unsplit");
                    exec.inkling2_attn_tile(
                        (&d_q2, 0, heads * hd),
                        (&d_r2, 0, heads * dr),
                        (&d_s2, 0, heads),
                        (&d_k, &d_v),
                        layout(),
                        dtype,
                        (&d_t0, t0.len() / 4),
                        &d_pt,
                        Inkling2AttnOut::Plain(&mut d_a),
                        (heads, kvh),
                        (window, 1024),
                        1.0 / (hd as f32).sqrt(),
                    )
                    .expect("unsplit tiles");
                    for splits in [2usize, 5, 16] {
                        let mut d_b = exec.alloc(row * heads * hd).expect("split");
                        let mut po = exec.alloc(64 * heads * 16 * hd).expect("part o");
                        let mut pm = exec.alloc(64 * heads * 16 * 2).expect("part ml");
                        exec.inkling2_attn_tile_split(
                            (&d_q2, 0, heads * hd),
                            (&d_r2, 0, heads * dr),
                            (&d_s2, 0, heads),
                            (&d_k, &d_v),
                            layout(),
                            dtype,
                            (&d_t0, t0.len() / 4),
                            &d_pt,
                            &mut d_b,
                            (&mut po, &mut pm),
                            n0,
                            (heads, kvh),
                            (window, 1024),
                            1.0 / (hd as f32).sqrt(),
                            splits,
                        )
                        .expect("split tiles");
                        let (a, b) = (
                            exec.to_host(&d_a).expect("a"),
                            exec.to_host(&d_b).expect("b"),
                        );
                        let d = a[..n0 * heads * hd]
                            .iter()
                            .zip(&b[..n0 * heads * hd])
                            .map(|(x, y)| (x - y).abs())
                            .fold(0f32, f32::max);
                        println!(
                            "attn_tile_split class {dtype} {} splits {splits}: max |diff| to unsplit {d:.2e}",
                            if paged { "paged" } else { "ring" }
                        );
                        assert!(d < 1e-4, "attn_tile_split splits {splits} parts by {d}");
                    }
                }
            }
        }
    }
}

fn bf16r(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// Slots 994-999, the DSpark draft's ops, against their definitions: the
/// tap mix + norm, q / k norm + RoPE, the block attention over bf16 stores
/// (paged and windowed ring, split), the recurrent gate and the pick (ties
/// to the first index), the strided row copy.
#[test]
fn inkling2_draft_ops() {
    use paddock_engine::gpu::Inkling2KvLayout;
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2() || !exec.has_inkling2_draft() {
        common::missing("pack has no Inkling 2 draft ops (slots 994-999)");
        return;
    }
    let w = 2304usize;
    // ---- mix + norm (slot 994)
    let rows = 3usize;
    let feat = det(rows * 4 * w, 91);
    let alpha = [0.1f32, 0.2, 0.3, 0.4];
    let wn: Vec<f32> = det(w, 92).iter().map(|v| bf16(1.0 + 0.5 * v)).collect();
    let d_f = exec.to_device(&feat).expect("feat");
    let d_a = exec.to_device(&alpha).expect("alpha");
    let d_wn = exec.to_device_u8(&bf16_bytes(&wn)).expect("w");
    let mut d_o = exec.alloc(rows * w).expect("out");
    exec.inkling2_mix_norm((&d_f, 4 * w), 4, &d_a, &d_wn, &mut d_o, rows, w, EPS)
        .expect("mix");
    let got = exec.to_host(&d_o).expect("back");
    let mut d = 0f32;
    for r in 0..rows {
        let m: Vec<f32> = (0..w)
            .map(|i| (0..4).map(|t| feat[r * 4 * w + t * w + i] * alpha[t]).sum())
            .collect();
        let inv = 1.0 / (m.iter().map(|v| v * v).sum::<f32>() / w as f32 + EPS).sqrt();
        for i in 0..w {
            d = d.max((got[r * w + i] - bf16r(m[i] * inv * wn[i])).abs());
        }
    }
    println!("mix_norm: max |diff| {d:.2e}");
    assert!(d < 2e-2, "mix_norm parts by {d}");
    // ---- q / k norm + RoPE (slot 995)
    let (heads, xs) = (3usize, 3 * 128 + 64);
    let x = det(rows * xs, 93);
    let wq: Vec<f32> = det(128, 94).iter().map(|v| bf16(1.0 + 0.3 * v)).collect();
    let pos = [0u32, 17, 300_001];
    let inv: Vec<f32> = (0..64)
        .map(|i| 1.0 / 1.0e6f32.powf((2 * i) as f32 / 128.0))
        .collect();
    let mut d_x = exec.to_device(&x).expect("x");
    let d_wq = exec.to_device_u8(&bf16_bytes(&wq)).expect("wq");
    let d_pos = exec.to_device_u32(&pos).expect("pos");
    let d_inv = exec.to_device(&inv).expect("inv");
    exec.inkling2_qk_rope(
        (&mut d_x, xs, 64),
        rows,
        heads,
        Some(&d_wq),
        &d_pos,
        &d_inv,
        EPS,
    )
    .expect("rope");
    let got = exec.to_host(&d_x).expect("back");
    let mut d = 0f32;
    for r in 0..rows {
        for h in 0..heads {
            let o = r * xs + 64 + h * 128;
            let hv = &x[o..o + 128];
            let inv_rms = 1.0 / (hv.iter().map(|v| v * v).sum::<f32>() / 128.0 + EPS).sqrt();
            for i in 0..64 {
                let x1 = bf16r(hv[i] * inv_rms * wq[i]) as f64;
                let x2 = bf16r(hv[i + 64] * inv_rms * wq[i + 64]) as f64;
                let a = (pos[r] as f32 * inv[i]) as f64;
                let (o1, o2) = (x1 * a.cos() - x2 * a.sin(), x2 * a.cos() + x1 * a.sin());
                d = d.max((got[o + i] as f64 - o1).abs() as f32);
                d = d.max((got[o + i + 64] as f64 - o2).abs() as f32);
            }
        }
        // the 64 columns before the heads are untouched
        for i in 0..64 {
            assert_eq!(
                got[r * xs + i],
                x[r * xs + i],
                "rope wrote outside its heads"
            );
        }
    }
    println!("qk_rope: max |diff| {d:.2e}");
    assert!(d < 2e-2, "qk_rope parts by {d}");
    // ---- block attention over bf16 stores (slot 996)
    let (hq, kvh, hd, kv_dim) = (24usize, 8usize, 128usize, 1024usize);
    let lens = [1500usize, 300];
    let ring = 2048usize;
    let bps = 1600usize.div_ceil(16);
    let n_blocks = 2 * bps;
    let bt: Vec<u32> = (0..n_blocks as u32).rev().collect();
    let d_bt = exec.to_device_u32(&bt).expect("bt");
    let kv: Vec<Vec<f32>> = (0..2)
        .map(|s| det(lens[s] * kv_dim, 95 + s as u64))
        .collect();
    let vv: Vec<Vec<f32>> = (0..2)
        .map(|s| det(lens[s] * kv_dim, 97 + s as u64))
        .collect();
    let (mut sl, mut ps) = (Vec::new(), Vec::new());
    for (s, &l) in lens.iter().enumerate() {
        sl.extend(std::iter::repeat_n(s as u32, l));
        ps.extend(0..l as u32);
    }
    let d_sl = exec.to_device_u32(&sl).expect("slots");
    let d_ps = exec.to_device_u32(&ps).expect("pos");
    let d_ks = exec.to_device(&kv.concat()).expect("k");
    let d_vs = exec.to_device(&vv.concat()).expect("v");
    // blocks: slot 0 at 1492 (its last 8 rows), slot 1 at 292
    let blk = [0u32, 1492, 1, 292];
    let d_blk = exec.to_device_u32(&blk).expect("blk");
    let q: Vec<f32> = det(16 * hq * hd, 99).iter().map(|v| v * 0.5).collect();
    let d_q = exec.to_device(&q).expect("q");
    let mut part_o = exec.alloc(16 * hq * 16 * hd).expect("part o");
    let mut part_ml = exec.alloc(16 * hq * 16 * 2).expect("part ml");
    for paged in [true, false] {
        let rows_s = if paged { n_blocks * 16 } else { 2 * ring };
        let layout = || {
            if paged {
                Inkling2KvLayout::Paged(&d_bt, bps)
            } else {
                Inkling2KvLayout::Ring(ring)
            }
        };
        let mut d_k = exec.alloc_u8(rows_s * kv_dim * 2).expect("k store");
        let mut d_v = exec.alloc_u8(rows_s * kv_dim * 2).expect("v store");
        for (src, dst) in [(&d_ks, &mut d_k), (&d_vs, &mut d_v)] {
            exec.inkling2_kv_store(
                (src, 0, kv_dim),
                (&d_sl, &d_ps),
                dst,
                layout(),
                3,
                kv_dim,
                sl.len(),
            )
            .expect("store");
        }
        let window = if paged { 0 } else { 1024 };
        for splits in [1usize, 4] {
            let mut d_out = exec.alloc(16 * hq * hd).expect("out");
            exec.inkling2_draft_attn(
                (&d_q, 0, hq * hd),
                (&d_blk, 2),
                (&d_k, &d_v),
                layout(),
                3,
                (&mut d_out, hq * hd),
                Some((&mut part_o, &mut part_ml)),
                (hq, kvh),
                window,
                splits,
                1.0 / (hd as f32).sqrt(),
            )
            .expect("draft attn");
            let got = exec.to_host(&d_out).expect("back");
            let mut d = 0f64;
            for b in 0..2 {
                let (s, p0) = (blk[2 * b] as usize, blk[2 * b + 1] as usize);
                let hi = p0 + 8;
                let lo = if window > 0 {
                    hi.saturating_sub(window + 8)
                } else {
                    0
                };
                for j in 0..8 {
                    for h in 0..hq {
                        let kh = h / 3;
                        let qv = &q[((b * 8 + j) * hq + h) * hd..][..hd];
                        let lg: Vec<f64> = (lo..hi)
                            .map(|t| {
                                (0..hd)
                                    .map(|i| {
                                        qv[i] as f64 * bf16r(kv[s][t * kv_dim + kh * hd + i]) as f64
                                    })
                                    .sum::<f64>()
                                    / (hd as f64).sqrt()
                            })
                            .collect();
                        let m = lg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let e: Vec<f64> = lg.iter().map(|l| (l - m).exp()).collect();
                        let den: f64 = e.iter().sum();
                        for i in 0..hd {
                            let a: f64 = (lo..hi)
                                .zip(&e)
                                .map(|(t, p)| p * bf16r(vv[s][t * kv_dim + kh * hd + i]) as f64)
                                .sum();
                            let gv = got[((b * 8 + j) * hq + h) * hd + i] as f64;
                            d = d.max((gv - a / den).abs());
                        }
                    }
                }
            }
            println!(
                "draft_attn {} splits {splits}: max |diff| {d:.2e}",
                if paged { "paged" } else { "ring" }
            );
            assert!(
                d < 1e-4,
                "draft_attn paged {paged} splits {splits} parts by {d}"
            );
        }
    }
    // ---- the recurrent gate (slot 997)
    let (r_rows, rank) = (2usize, 256usize);
    let raw = det(r_rows * 3 * rank, 101);
    let ctx = det(r_rows * 3 * rank, 102);
    let st0: Vec<f32> = det(r_rows * 2 * rank, 103)
        .iter()
        .map(|v| bf16r(*v))
        .collect();
    let d_raw = exec.to_device(&raw).expect("raw");
    let d_ctx = exec.to_device(&ctx).expect("ctx");
    let mut d_st = exec.to_device(&st0).expect("state");
    let mut d_lat = exec.alloc(r_rows * rank).expect("latent");
    exec.inkling2_markov_gate(
        &d_raw,
        (&d_ctx, 0),
        (&mut d_st, 2 * rank),
        &mut d_lat,
        r_rows,
        rank,
    )
    .expect("gate");
    let st = exec.to_host(&d_st).expect("state back");
    let lat = exec.to_host(&d_lat).expect("latent back");
    let mut bad = 0;
    for r in 0..r_rows {
        for c in 0..rank {
            let at = |k: usize| {
                bf16r(raw[r * 3 * rank + k * rank + c] + ctx[r * 3 * rank + k * rank + c])
            };
            let g = bf16r(1.0 / (1.0 + (-at(0)).exp()));
            let cand = bf16r(at(1).tanh());
            let keep = bf16r(g * st0[r * 2 * rank + c]);
            let upd = bf16r(bf16r(1.0 - g) * cand);
            let want_s = bf16r(keep + upd);
            let want_l = bf16r(at(2).tanh());
            // one bf16 ulp of slack for the device's transcendental rounding
            let ulp = |a: f32, b: f32| (a - b).abs() > 1e-2 * a.abs().max(1e-3);
            bad += ulp(st[r * 2 * rank + c], want_s) as usize
                + ulp(lat[r * rank + c], want_l) as usize;
            assert_eq!(
                st[r * 2 * rank + rank + c],
                st0[r * 2 * rank + rank + c],
                "gate wrote past its state"
            );
        }
    }
    println!(
        "markov_gate: {bad} of {} values past one ulp",
        2 * r_rows * rank
    );
    assert_eq!(bad, 0, "markov_gate parts from its definition");
    // ---- the pick (slot 998): ties go to the first index
    let (pr, vocab) = (3usize, 50_000usize);
    let mut base = det(pr * vocab, 104)
        .iter()
        .map(|v| v * 40.0)
        .collect::<Vec<f32>>();
    let gemm = det(pr * vocab, 105);
    let fb: Vec<f32> = det(vocab, 106).iter().map(|v| bf16(v * 0.1)).collect();
    // row 1: two saturated entries tie after the softcap
    base[vocab + 777] = 500.0;
    base[vocab + 778] = 600.0;
    let d_base = exec.to_device(&base).expect("base");
    let d_gemm = exec.to_device(&gemm).expect("gemm");
    let d_fb = exec.to_device_u8(&bf16_bytes(&fb)).expect("fb");
    let mut d_best = exec.alloc_u8(8 * pr).expect("best");
    let mut d_ids = exec.alloc_u32(pr).expect("ids");
    exec.inkling2_markov_argmax(
        (&d_base, 0),
        &d_gemm,
        &d_fb,
        &mut d_best,
        &mut d_ids,
        pr,
        (vocab, vocab),
        30.0,
    )
    .expect("argmax");
    let ids = exec.to_host_u32(&d_ids).expect("ids back");
    for r in 0..pr {
        let mut best = (f32::NEG_INFINITY, 0usize);
        for i in 0..vocab {
            let bias = bf16r(gemm[r * vocab + i] + fb[i]);
            let x = bf16r(bf16r(base[r * vocab + i]) + bias);
            let l = bf16r((x / 30.0).tanh() * 30.0);
            if l > best.0 {
                best = (l, i);
            }
        }
        assert_eq!(
            ids[r] as usize, best.1,
            "row {r}: pick {} vs {}",
            ids[r], best.1
        );
    }
    assert_eq!(ids[1], 777, "the saturated tie goes to the first index");
    println!("markov_argmax: {ids:?}");
}

// ---- the model (heavy) -----------------------------------------------------

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn i32s(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

fn top1(l: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::MIN);
    let mut second = f32::MIN;
    for (i, &v) in l.iter().enumerate() {
        if v > best.1 {
            second = best.1;
            best = (i, v);
        } else if v > second {
            second = v;
        }
    }
    (best.0, best.1 - second)
}

/// The K / V class the heavy gates run (`INKLING2_KV`: f32 - the default,
/// the reference class - f16 or fp8).
fn kv_class() -> u32 {
    match std::env::var("INKLING2_KV").as_deref() {
        Ok("fp8") => 1,
        Ok("f16") => 0,
        _ => paddock_engine::gpu::INKLING2_KV_F32,
    }
}

/// The official release (`INKLING2_DIR`, else the data root's copy)
/// against a teacher-forced reference (`INKLING2_GOLDENS`, the reference
/// script's output: full logits for short prompts, the top 8 a row
/// otherwise): every prompt through the serial lane, top-1 at every
/// position. The model is chaotic at its own noise - the loop and the W4A4
/// experts amplify rounding - so the gate is statistical: independent
/// implementations of this checkpoint agree on ~96% of top-1s, and
/// the lane must sit inside that band (`INKLING2_FLOOR`, default 0.94). With
/// `INKLING2_OURS` the lane's top-1s land in that file, one line a prompt.
/// Short prompts also trace every visit's output and visit 0's stages.
#[test]
fn inkling2_serial_vs_reference() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-990)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS (the reference script's goldens)");
        return;
    };
    let floor: f64 = std::env::var("INKLING2_FLOOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.94);
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let t0 = std::time::Instant::now();
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        8192,
        2048,
        kv_class(),
    )
    .expect("load");
    println!("loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let mut dump = String::new();
    let (mut all_agree, mut all_n) = (0usize, 0usize);
    let mut s = 0;
    while let Some((_, ids)) = g.bytes(&format!("s{s}.ids")) {
        let ids: Vec<u32> = i32s(ids).into_iter().map(|t| t as u32).collect();
        // the reference's top-1 and its margin a row
        let refs: Vec<(usize, f32)> = if let Some((lt, lb)) = g.bytes(&format!("s{s}.logits")) {
            let vocab = lt.shape[1];
            let want = f32s(lb);
            (0..ids.len())
                .map(|r| top1(&want[r * vocab..(r + 1) * vocab]))
                .collect()
        } else {
            let (_, tb) = g.bytes(&format!("s{s}.top_ids")).expect("top ids");
            // logits or log-probs (vLLM's): a row's margins are the same
            let (_, lb) = g
                .bytes(&format!("s{s}.top_logits"))
                .or_else(|| g.bytes(&format!("s{s}.top_lp")))
                .expect("top logits");
            let ti = i32s(tb);
            let tl = f32s(lb);
            (0..ids.len())
                .map(|r| (ti[r * 8] as usize, tl[r * 8] - tl[r * 8 + 1]))
                .collect()
        };
        m.reset().expect("reset");
        let short = ids.len() <= 64;
        m.trace_rows = if short { 12.min(ids.len()) } else { 0 };
        m.trace.clear();
        m.stages.clear();
        let mut ours = vec![0usize; ids.len()];
        let mut misses = Vec::new();
        let t1 = std::time::Instant::now();
        m.forward_each(
            &ids,
            paddock_engine::gpu_model::inkling2::Logits::All,
            |r, l| {
                ours[r] = top1(l).0;
            },
        )
        .expect("forward");
        for (r, (&o, &(t, margin))) in ours.iter().zip(&refs).enumerate() {
            if o != t {
                misses.push((r, o, t, margin));
            }
        }
        let agree = ids.len() - misses.len();
        let near = misses.iter().filter(|m| m.3 < 0.5).count();
        println!(
            "seq {s}: {} tokens in {:.2} s, top-1 {agree}/{} ({:.2}%), {} of the {} misses within 0.5 of a reference tie",
            ids.len(),
            t1.elapsed().as_secs_f64(),
            ids.len(),
            100.0 * agree as f64 / ids.len() as f64,
            near,
            misses.len()
        );
        all_agree += agree;
        all_n += ids.len();
        dump.push_str(
            &ours
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" "),
        );
        dump.push('\n');
        if short {
            for (v, ours) in m.trace.iter().enumerate() {
                if let Some((_, hb)) = g.bytes(&format!("s{s}.h{:02}", v + 1)) {
                    let theirs = f32s(hb);
                    let n = ours.len().min(theirs.len());
                    let num: f64 = ours[..n]
                        .iter()
                        .zip(&theirs[..n])
                        .map(|(a, b)| (a * b) as f64)
                        .sum();
                    let na: f64 = ours[..n].iter().map(|a| (a * a) as f64).sum::<f64>().sqrt();
                    let nb: f64 = theirs[..n]
                        .iter()
                        .map(|b| (b * b) as f64)
                        .sum::<f64>()
                        .sqrt();
                    // the reference's last entry is its final normed output
                    let what = if v + 1 == 48 {
                        " (vs the reference's final norm)"
                    } else {
                        ""
                    };
                    println!("  visit {v:2}: cos {:.6}{what}", num / (na * nb));
                }
            }
        }
        s += 1;
    }
    assert!(s > 0, "no sequences in the goldens");
    if let Some(p) = std::env::var_os("INKLING2_OURS") {
        std::fs::write(p, dump).expect("write our top-1s");
    }
    let rate = all_agree as f64 / all_n as f64;
    println!(
        "all: top-1 {all_agree}/{all_n} = {:.2}% (floor {:.2}%)",
        100.0 * rate,
        100.0 * floor
    );
    assert!(
        rate >= floor,
        "top-1 agreement {rate:.4} under the floor {floor}"
    );
}

/// The serial lane's decode path (one row a step: the decode GEMVs, a
/// growing history) against its own prompt path: a greedy reply decoded
/// token by token, then the prompt + reply teacher-forced in one prefill -
/// each decoded token must be the prefill's top-1 at its position, parting
/// only near a tie (the two paths sum in different orders, and the model
/// amplifies that). `INKLING2_GOLDENS`' last prompt is the chat prompt.
#[test]
fn inkling2_decode_vs_prefill() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-990)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let mut s = 0;
    while g.bytes(&format!("s{}.ids", s + 1)).is_some() {
        s += 1;
    }
    let (_, ib) = g.bytes(&format!("s{s}.ids")).expect("ids");
    let prompt: Vec<u32> = i32s(ib).into_iter().map(|t| t as u32).collect();
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        4096,
        512,
        kv_class(),
    )
    .expect("load");
    let n_gen = 64;
    let mut logits = m.forward_last(&prompt).expect("prefill");
    let mut reply = Vec::new();
    let mut margins = Vec::new();
    let t0 = std::time::Instant::now();
    for _ in 0..n_gen {
        let (t, gap) = top1(&logits);
        reply.push(t as u32);
        margins.push(gap);
        if t as u32 == paddock_models::inkling2::EOS {
            break;
        }
        logits = m.forward_last(&[t as u32]).expect("decode");
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "decoded {} tokens in {dt:.2} s ({:.1} tok/s): {:?}",
        reply.len(),
        reply.len() as f64 / dt,
        reply
    );
    m.reset().expect("reset");
    let all: Vec<u32> = prompt
        .iter()
        .chain(&reply[..reply.len() - 1])
        .copied()
        .collect();
    let mut pf = vec![(0usize, 0f32); all.len()];
    m.forward_each(
        &all,
        paddock_engine::gpu_model::inkling2::Logits::All,
        |r, l| pf[r] = top1(l),
    )
    .expect("prefill");
    let mut parted = None;
    for (k, &t) in reply.iter().enumerate() {
        let (p, gap) = pf[prompt.len() - 1 + k];
        if p as u32 != t {
            println!(
                "  step {k}: decoded {t}, prefill {p} (prefill margin {gap:.3}, decode margin {:.3})",
                margins[k]
            );
            parted.get_or_insert((k, gap.min(margins[k])));
        }
    }
    if let Some((k, gap)) = parted {
        // past the first parting the streams are different texts
        assert!(
            gap < 0.5,
            "decode parts from the prompt path at step {k} off a near-tie (margin {gap})"
        );
    }
}

/// The batched lane against the single stream: four slots seated, three
/// prompts of different lengths in slots 0, 2 and 3 (slot 1 stays a hole).
/// Slot 0 prefills whole and decodes while 2 and 3 fill through chunked
/// mixed ticks; then all three decode together - every decode row
/// teacher-forced on its prompt's own single-stream greedy reply. Every row
/// (and each prompt's last logits) must take its single-stream top-1,
/// parting only near a tie (a wider tick splits the attention's keys
/// differently). Then the three prompts once more as one packed cohort, and
/// every block must come back to the pool.
#[test]
fn inkling2_batch_vs_single() {
    use paddock_engine::generator::Generator;
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() {
        common::missing("pack has no Inkling 2 ops (slots 977-990)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let seq = |s: usize, n: usize| -> Vec<u32> {
        let (_, ib) = g.bytes(&format!("s{s}.ids")).expect("ids");
        let ids: Vec<u32> = i32s(ib).into_iter().map(|t| t as u32).collect();
        ids[..n.min(ids.len())].to_vec()
    };
    let mut last = 0;
    while g.bytes(&format!("s{}.ids", last + 1)).is_some() {
        last += 1;
    }
    // (slot, prompt): the chat prompt and two long-document heads
    let prompts = [
        (0usize, seq(last, usize::MAX)),
        (2, seq(0, 700)),
        (3, seq(1, 1300)),
    ];
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        4096,
        512,
        kv_class(),
    )
    .expect("load");
    let n_gen = 32;
    // each prompt alone: its greedy reply and the top-1 / margin a step
    let mut single = Vec::new();
    let t0 = std::time::Instant::now();
    for (_, p) in &prompts {
        m.reset().expect("reset");
        let mut logits = m.forward_last(p).expect("prefill");
        let (mut toks, mut tops) = (Vec::new(), Vec::new());
        for _ in 0..n_gen {
            let (t, gap) = top1(&logits);
            tops.push((t, gap));
            toks.push(t as u32);
            logits = m.forward_last(&[t as u32]).expect("decode");
        }
        tops.push(top1(&logits));
        single.push((toks, tops));
    }
    let dt1 = t0.elapsed().as_secs_f64();
    let seated = m.enable_batch_impl(4).expect("enable batch");
    assert!(seated >= 4, "seated {seated} slots");
    let blocks0 = m.pool_free_blocks().expect("pool");
    let mut parted = Vec::new();
    let mut check = |k: usize, slot: usize, i: usize, got: (usize, f32), want: (usize, f32)| {
        if got.0 != want.0 {
            println!(
                "  slot {slot} step {k}: batch {} (margin {:.3}), single {} (margin {:.3})",
                got.0, got.1, want.0, want.1
            );
            parted.push((i, got.1.min(want.1)));
        }
    };
    // slot 0 prefills whole; slots 2 and 3 queue and fill through mixed
    // ticks (256 prompt rows a tick) while slot 0 decodes beside them
    let l = m
        .prefill_slot(prompts[0].0, &prompts[0].1)
        .expect("batch prefill");
    check(0, prompts[0].0, 0, top1(&l), single[0].1[0]);
    for (slot, p) in &prompts[1..] {
        m.prefill_begin(*slot, p.clone()).expect("queue");
    }
    let v = paddock_models::inkling2::VOCAB_REAL;
    // the next step to check a prompt (0 = its prompt's own logits)
    let mut step = [1usize, 0, 0];
    let mut mixed = 0;
    while !m.prefill_queue().is_empty() {
        let k = step[0] - 1;
        let dec = [(
            prompts[0].0,
            single[0].0[k],
            (prompts[0].1.len() + k) as u32,
        )];
        let (out, fin) = m.forward_mixed(&dec, 256).expect("mixed tick");
        check(
            step[0],
            prompts[0].0,
            0,
            top1(&out[..v]),
            single[0].1[step[0]],
        );
        step[0] += 1;
        for (slot, l, rows) in fin {
            let i = prompts
                .iter()
                .position(|(s, _)| *s == slot)
                .expect("a queued slot");
            assert_eq!(rows, prompts[i].1.len(), "slot {slot}'s prompt rows");
            check(0, slot, i, top1(&l), single[i].1[0]);
            step[i] = 1;
        }
        mixed += 1;
    }
    assert!(
        step.iter().all(|&s| s >= 1),
        "a queued prompt never finished"
    );
    // then every stream decodes together until each has its n_gen steps
    let t1 = std::time::Instant::now();
    let (mut ticks, mut rows) = (0usize, 0usize);
    while step.iter().any(|&s| s <= n_gen) {
        let (mut toks, mut pos) = (vec![0u32; 4], vec![0u32; 4]);
        for (i, (slot, p)) in prompts.iter().enumerate() {
            if step[i] <= n_gen {
                toks[*slot] = single[i].0[step[i] - 1];
                pos[*slot] = (p.len() + step[i] - 1) as u32;
            }
        }
        let out = m.decode_tick(&toks, &pos).expect("tick");
        assert!(
            out[v..2 * v].iter().all(|&x| x == 0.0),
            "the hole row has logits"
        );
        for (i, (slot, _)) in prompts.iter().enumerate() {
            if step[i] <= n_gen {
                let got = top1(&out[slot * v..(slot + 1) * v]);
                check(step[i], *slot, i, got, single[i].1[step[i]]);
                step[i] += 1;
                rows += 1;
            }
        }
        ticks += 1;
    }
    let dt3 = t1.elapsed().as_secs_f64();
    println!(
        "single: 3 x {n_gen} steps (prefill included) in {dt1:.2} s; batch: {mixed} mixed ticks, then {ticks} decode ticks ({rows} rows) in {dt3:.2} s ({:.1} tok/s)",
        rows as f64 / dt3
    );
    // the three prompts again as one cohort: packed into shared passes (a
    // 512-row chunk here, so the long ones straddle passes)
    let items: Vec<(usize, Vec<u32>)> = prompts.iter().map(|(s, p)| (*s, p.clone())).collect();
    let firsts = m.forward_prefill_batch(&items).expect("packed prefill");
    for (i, l) in firsts.iter().enumerate() {
        check(0, prompts[i].0, i, top1(l), single[i].1[0]);
    }
    m.release_inactive_slots(&[false; 4]);
    let free = m.pool_free_blocks().expect("pool");
    assert_eq!(free, blocks0, "the released slots kept blocks");
    println!("pool: {blocks0} blocks, all back after release");
    for (i, gap) in &parted {
        assert!(
            *gap < 0.5,
            "prompt {i} parts from its single stream off a near-tie (margin {gap})"
        );
    }
}

/// The release's DSpark draft against the target it drafts for: a greedy
/// reply decoded a token at a time, and before each step a draft round
/// anchored at the step's pending token - its 7 drafts scored against the
/// target's next 7 greedy tokens (the leading run that matches is what a
/// spec round would accept). The draft is trained on this target's own
/// distribution, so a working draft accepts several tokens a round; a broken
/// feature tap, RoPE or recurrent head accepts ~0.
#[test]
fn inkling2_draft_acceptance() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() || !exec.has_inkling2_draft() {
        common::missing("pack has no Inkling 2 draft ops (slots 994-999)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let mut last = 0;
    while g.bytes(&format!("s{}.ids", last + 1)).is_some() {
        last += 1;
    }
    let (_, ib) = g.bytes(&format!("s{last}.ids")).expect("ids");
    let prompt: Vec<u32> = i32s(ib).into_iter().map(|t| t as u32).collect();
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        4096,
        512,
        kv_class(),
    )
    .expect("load");
    assert!(m.has_draft(), "the release ships mtp.safetensors");
    let n_gen = 96;
    let mut logits = m.forward_last(&prompt).expect("prefill");
    let (mut toks, mut drafts) = (Vec::new(), Vec::new());
    let mut t_draft = 0f64;
    for _ in 0..n_gen {
        let (t, _) = top1(&logits);
        let t = t as u32;
        let pos = m.len();
        let t0 = std::time::Instant::now();
        drafts.push(m.draft_probe(0, t, pos).expect("draft"));
        t_draft += t0.elapsed().as_secs_f64();
        toks.push(t);
        if t == paddock_models::inkling2::EOS {
            break;
        }
        logits = m.forward_last(&[t]).expect("decode");
    }
    // round k drafts the tokens after toks[k]
    let mut acc = Vec::new();
    for (k, d) in drafts.iter().enumerate() {
        if k + 8 > toks.len() {
            break;
        }
        acc.push(
            d.iter()
                .zip(&toks[k + 1..])
                .take_while(|(a, b)| a == b)
                .count(),
        );
    }
    let mean = acc.iter().sum::<usize>() as f64 / acc.len() as f64;
    let mut hist = [0usize; 8];
    for &a in &acc {
        hist[a] += 1;
    }
    println!(
        "draft: {} rounds, mean accepted {mean:.2} of 7 (histogram 0..7 {hist:?}), {:.1} ms a round",
        acc.len(),
        1e3 * t_draft / drafts.len() as f64
    );
    for k in [0usize, 10, 40] {
        if k + 8 <= toks.len() {
            println!(
                "  round {k}: drafts {:?} / target {:?}",
                drafts[k],
                &toks[k + 1..k + 8]
            );
        }
    }
    assert!(mean >= 1.5, "the draft accepts {mean:.2} tokens a round");
}

/// Greedy speculative decoding with the release's DSpark draft, through the
/// Generator calls the service makes: warm (sync the slot), draft, verify the
/// chunk, accept the matching run, the bonus pick next - the short convs
/// rolled back over every rejected tail. Its reply, teacher-forced through
/// one prompt pass, must be that pass's top-1 at every position, parting
/// only near a tie (a wrong rollback parts at once, off a clear margin; the
/// looped model's own noise only at ties). Prints the drafts accepted a
/// round and the speed against plain greedy decoding.
#[test]
fn inkling2_spec_greedy() {
    use paddock_engine::generator::Generator;
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() || !exec.has_inkling2_draft() {
        common::missing("pack has no Inkling 2 draft ops (slots 994-1001)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let mut last = 0;
    while g.bytes(&format!("s{}.ids", last + 1)).is_some() {
        last += 1;
    }
    let (_, ib) = g.bytes(&format!("s{last}.ids")).expect("ids");
    let prompt: Vec<u32> = i32s(ib).into_iter().map(|t| t as u32).collect();
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        4096,
        512,
        kv_class(),
    )
    .expect("load");
    let n_gen = 160;
    // drafts a round (INKLING2_SPEC_K, default the block's 7)
    let spec_k: usize = std::env::var("INKLING2_SPEC_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    // plain greedy, for the speed
    let mut logits = m.forward_last(&prompt).expect("prefill");
    let t0 = std::time::Instant::now();
    for _ in 0..n_gen {
        let (t, _) = top1(&logits);
        logits = m.forward_last(&[t as u32]).expect("decode");
    }
    let dt_plain = t0.elapsed().as_secs_f64();
    // speculative greedy on slot 0
    m.reset().expect("reset");
    let first = Generator::forward_prefill(&mut m, 0, &prompt).expect("prefill");
    let mut pending = top1(&first).0 as u32;
    let mut pos = prompt.len();
    let mut spec = vec![pending];
    let (mut rounds, mut accepted) = (0usize, 0usize);
    let t1 = std::time::Instant::now();
    while spec.len() < n_gen {
        assert!(
            m.spec_ensure_warm(0, &[], (pos - 1) as u32).expect("warm"),
            "slot cold at {pos}"
        );
        let d = m
            .spec_draft_batch(&[(0, pending)], spec_k)
            .expect("draft")
            .expect("a draft");
        let mut chunk = vec![pending];
        chunk.extend(&d[0]);
        let picks = m
            .forward_spec_batch(&[(0, pos, chunk.clone())])
            .expect("verify")
            .expect("picks");
        let mut a = 0;
        while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
            a += 1;
        }
        spec.extend(&picks[..=a]);
        pending = picks[a];
        pos += a + 1;
        rounds += 1;
        accepted += a;
    }
    let dt_spec = t1.elapsed().as_secs_f64();
    spec.truncate(n_gen);
    println!(
        "plain {:.1} tok/s; spec {:.1} tok/s, {rounds} rounds, {:.2} drafts accepted a round",
        n_gen as f64 / dt_plain,
        n_gen as f64 / dt_spec,
        accepted as f64 / rounds as f64
    );
    // the reply teacher-forced through one prompt pass
    m.reset().expect("reset");
    let all: Vec<u32> = prompt.iter().chain(&spec[..n_gen - 1]).copied().collect();
    let mut pf = vec![(0usize, 0f32); all.len()];
    m.forward_each(
        &all,
        paddock_engine::gpu_model::inkling2::Logits::All,
        |r, l| pf[r] = top1(l),
    )
    .expect("teacher-forced pass");
    let mut parted = None;
    for (k, &t) in spec.iter().enumerate() {
        let (p, gap) = pf[prompt.len() - 1 + k];
        if p as u32 != t {
            println!("  token {k}: spec {t}, prompt pass {p} (margin {gap:.3})");
            parted.get_or_insert((k, gap));
        }
    }
    // decode-width rows (and verify chunks) and one long prompt pass sum in
    // different classes, and this model amplifies that: an isolated clear
    // parting is the model's (plain decoding parts there too); a broken
    // rollback parts nearly everywhere. The rollback itself is gated
    // exactly by `inkling2_spec_rollback`.
    let clear = spec
        .iter()
        .enumerate()
        .filter(|&(k, &t)| {
            let (p, gap) = pf[prompt.len() - 1 + k];
            p as u32 != t && gap >= 0.5
        })
        .count();
    if parted.is_none() {
        println!("  the prompt pass agrees on all {n_gen} tokens");
    }
    assert!(
        clear <= 2,
        "the spec reply parts from its prompt pass off a clear margin {clear} times"
    );
}

/// The rollback across speculative rounds, exactly: verify an 8-row chunk,
/// commit only its first k rows (the short convs roll back to them), then
/// verify the rejected rows again from position p + k - their logits must
/// be the first verify's for the same positions (same history, the convs'
/// carried rows rebuilt from the stash). A wrong rollback moves every row.
#[test]
fn inkling2_spec_rollback() {
    use paddock_engine::generator::Generator;
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() || !exec.has_inkling2_draft() {
        common::missing("pack has no Inkling 2 draft ops (slots 994-1001)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let (_, ib) = g.bytes("s0.ids").expect("ids");
    let text: Vec<u32> = i32s(ib).into_iter().map(|t| t as u32).collect();
    let (prompt, cont) = (&text[..300], &text[300..320]);
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        4096,
        512,
        kv_class(),
    )
    .expect("load");
    let v = paddock_models::inkling2::VOCAB_REAL;
    let pos = prompt.len();
    for k in [1usize, 3, 6] {
        m.reset().expect("reset");
        Generator::forward_prefill(&mut m, 0, prompt).expect("prefill");
        let c1 = cont[..8].to_vec();
        let l1 = m
            .forward_spec_verify(&[(0, pos, c1)])
            .expect("verify")
            .expect("logits");
        m.spec_commit(&[k as u32]).expect("commit");
        let c2 = cont[k..k + 8].to_vec();
        let l2 = m
            .forward_spec_verify(&[(0, pos + k, c2)])
            .expect("verify again")
            .expect("logits");
        let (mut d, mut scale) = (0f32, 0f32);
        for j in 0..8 - k {
            let (a, b) = (&l1[(k + j) * v..(k + j + 1) * v], &l2[j * v..(j + 1) * v]);
            assert_eq!(
                top1(a).0,
                top1(b).0,
                "k {k} row {j}: the re-verify picks otherwise"
            );
            d = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(d, f32::max);
            scale = a.iter().fold(scale, |s, x| s.max(x.abs()));
        }
        println!(
            "rollback to {k} of 8: re-verified rows max |diff| {d:.2e} (logit scale {scale:.1})"
        );
        assert!(d < 1e-2 * scale, "k {k}: the re-verified rows part by {d}");
        m.spec_commit(&[(8 - k) as u32]).expect("settle");
    }
}

/// Bench, not a gate (`INKLING2_BENCH=1` on top of the heavy switch): a
/// ~22K-token prompt, then verify passes at fixed widths (each rolled back)
/// and draft rounds timed, medians after warm-up (the graphs captured).
#[test]
fn inkling2_verify_bench() {
    use paddock_engine::generator::Generator;
    if !common::heavy() || std::env::var_os("INKLING2_BENCH").is_none() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let mut text = Vec::new();
    for s in 0..3 {
        let (_, ib) = g.bytes(&format!("s{s}.ids")).expect("ids");
        text.extend(i32s(ib).into_iter().map(|t| t as u32));
    }
    let ctx: usize = std::env::var("INKLING2_BENCH_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(22_000);
    let prompt: Vec<u32> = text.iter().copied().cycle().take(ctx).collect();
    let cont: Vec<u32> = text[1000..1016].to_vec();
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        32768,
        4096,
        kv_class(),
    )
    .expect("load");
    assert!(m.enable_batch_impl(1).expect("enable") >= 1);
    let t = std::time::Instant::now();
    let l = Generator::forward_prefill(&mut m, 0, &prompt).expect("prefill");
    println!(
        "prefill {} tokens in {:.2} s",
        prompt.len(),
        t.elapsed().as_secs_f64()
    );
    let pos = prompt.len();
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let widths: Vec<usize> = match std::env::var("INKLING2_BENCH_ROWS") {
        Ok(v) => v.split(',').filter_map(|x| x.parse().ok()).collect(),
        Err(_) => vec![1, 2, 4, 6, 8],
    };
    for rows in widths {
        let chunk = cont[..rows].to_vec();
        let mut ts = Vec::new();
        for it in 0..12 {
            let t = std::time::Instant::now();
            m.forward_spec_batch(&[(0, pos, chunk.clone())])
                .expect("verify")
                .expect("picks");
            if it >= 3 {
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            m.spec_commit(&[0]).expect("roll back");
        }
        println!("verify {rows} rows: {:.2} ms", med(&mut ts));
    }
    let pend = (0usize, top1(&l).0 as u32);
    for depth in [3usize, 7] {
        let mut ts = Vec::new();
        for it in 0..12 {
            let t = std::time::Instant::now();
            m.spec_draft_batch(&[pend], depth)
                .expect("draft")
                .expect("drafts");
            if it >= 3 {
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        println!("draft round depth {depth}: {:.2} ms", med(&mut ts));
    }
}

/// A draft round over several slots against each slot's round alone, from
/// the same states: every block's rows, its features and its recurrent steps
/// are its own, so the drafts are the same tokens - parting only at a pick
/// within a near-tie (the draft's GEMMs take other classes at 24 rows than
/// at 8, which moves a logit by a few bf16 ulps).
#[test]
fn inkling2_draft_batch_vs_single() {
    use paddock_engine::generator::Generator;
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() || !exec.has_inkling2_draft() {
        common::missing("pack has no Inkling 2 draft ops (slots 994-1001)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let seq = |s: usize, n: usize| -> Vec<u32> {
        let (_, ib) = g.bytes(&format!("s{s}.ids")).expect("ids");
        let ids: Vec<u32> = i32s(ib).into_iter().map(|t| t as u32).collect();
        ids[..n.min(ids.len())].to_vec()
    };
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        4096,
        512,
        kv_class(),
    )
    .expect("load");
    assert!(m.enable_batch_impl(4).expect("enable") >= 4);
    m.trace_draft_margins();
    let prompts = [(0usize, seq(0, 300)), (2, seq(1, 450)), (3, seq(2, 200))];
    let mut pend = Vec::new();
    for (slot, p) in &prompts {
        let l = Generator::forward_prefill(&mut m, *slot, p).expect("prefill");
        pend.push((*slot, top1(&l).0 as u32));
    }
    let multi = m
        .spec_draft_batch(&pend, 7)
        .expect("draft")
        .expect("drafts");
    let multi_m = m.draft_margins().expect("margins").to_vec();
    for (i, &pd) in pend.iter().enumerate() {
        let one = m
            .spec_draft_batch(&[pd], 7)
            .expect("draft")
            .expect("drafts");
        let one_m = m.draft_margins().expect("margins")[0].clone();
        println!(
            "slot {}: together {:?} / alone {:?}",
            pd.0, multi[i], one[0]
        );
        if let Some(k) = multi[i].iter().zip(&one[0]).position(|(a, b)| a != b) {
            let gap = multi_m[i][k].min(one_m[k]);
            println!("  parts at step {k} off a margin of {gap:.3}");
            assert!(
                gap < 0.5,
                "slot {}: drafts part with company off a clear margin {gap} at step {k}",
                pd.0
            );
        }
    }
    // the same prompts as one packed cohort (the service's simultaneous
    // admissions): the state the drafts read must be the same
    m.release_inactive_slots(&[false; 4]);
    let items: Vec<(usize, Vec<u32>)> = prompts.iter().map(|(s, p)| (*s, p.clone())).collect();
    let firsts = m.forward_prefill_batch(&items).expect("packed prefill");
    let pend2: Vec<(usize, u32)> = prompts
        .iter()
        .zip(&firsts)
        .map(|((s, _), l)| (*s, top1(l).0 as u32))
        .collect();
    let packed = m
        .spec_draft_batch(&pend2, 7)
        .expect("draft")
        .expect("drafts");
    let mut same = 0;
    for (i, (a, b)) in multi.iter().zip(&packed).enumerate() {
        println!("slot {}: individual {a:?} / packed {b:?}", pend[i].0);
        same += a.iter().zip(b).take_while(|(x, y)| x == y).count();
    }
    assert!(
        same >= 3 * 4,
        "packed prefill drafts part from individual ones ({same} leading matches)"
    );
}

/// Speculative rounds over three slots at once (warm each, one draft
/// batch, one verify of every chunk, accept per slot): each slot accepts
/// about as many drafts as the same prompt run alone. How often each reply
/// parts from one teacher-forced prompt pass is reported beside plain
/// decoding's (a batched verify rides other kernel classes than a solo
/// one, so the replies themselves may part).
#[test]
fn inkling2_spec_batch() {
    use paddock_engine::generator::Generator;
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() || !exec.has_inkling2_draft() {
        common::missing("pack has no Inkling 2 draft ops (slots 994-1001)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let seq = |s: usize, n: usize| -> Vec<u32> {
        let (_, ib) = g.bytes(&format!("s{s}.ids")).expect("ids");
        let ids: Vec<u32> = i32s(ib).into_iter().map(|t| t as u32).collect();
        ids[..n.min(ids.len())].to_vec()
    };
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        4096,
        512,
        kv_class(),
    )
    .expect("load");
    assert!(m.enable_batch_impl(4).expect("enable") >= 4);
    let prompts = [(0usize, seq(0, 300)), (2, seq(1, 450)), (3, seq(2, 200))];
    let rounds = 24;
    // run `which` slots together; per slot its tokens and accepted drafts
    let run = |m: &mut paddock_engine::gpu_model::inkling2::GpuInkling2,
               which: &[usize]|
     -> Vec<(Vec<u32>, usize)> {
        m.release_inactive_slots(&[false; 4]);
        let mut st: Vec<(usize, usize, u32, Vec<u32>, usize)> = Vec::new();
        for &i in which {
            let (slot, p) = &prompts[i];
            let l = Generator::forward_prefill(m, *slot, p).expect("prefill");
            st.push((*slot, p.len(), top1(&l).0 as u32, Vec::new(), 0));
        }
        for _ in 0..rounds {
            for s in &st {
                assert!(
                    m.spec_ensure_warm(s.0, &[], (s.1 - 1) as u32)
                        .expect("warm")
                );
            }
            let pend: Vec<(usize, u32)> = st.iter().map(|s| (s.0, s.2)).collect();
            let d = m
                .spec_draft_batch(&pend, 3)
                .expect("draft")
                .expect("drafts");
            let reqs: Vec<(usize, usize, Vec<u32>)> = st
                .iter()
                .zip(&d)
                .map(|(s, d)| {
                    let mut c = vec![s.2];
                    c.extend(d);
                    (s.0, s.1, c)
                })
                .collect();
            let picks = m.forward_spec_batch(&reqs).expect("verify").expect("picks");
            let mut base = 0;
            for (s, r) in st.iter_mut().zip(&reqs) {
                let c = &r.2;
                let mut a = 0;
                while a + 1 < c.len() && c[a + 1] == picks[base + a] {
                    a += 1;
                }
                s.3.extend(&picks[base..=base + a]);
                s.2 = picks[base + a];
                s.1 += a + 1;
                s.4 += a;
                base += c.len();
            }
        }
        st.into_iter().map(|s| (s.3, s.4)).collect()
    };
    let together = run(&mut m, &[0, 1, 2]);
    let alone: Vec<(Vec<u32>, usize)> = (0..3).map(|i| run(&mut m, &[i]).remove(0)).collect();
    // each reply teacher-forced through one prompt pass on slot 0: partings
    // only near ties (at most two clear ones - the classes differ)
    let clear_of =
        |m: &mut paddock_engine::gpu_model::inkling2::GpuInkling2, i: usize, reply: &[u32]| {
            // a fresh slot 0: blocks back AND the short convs from zero
            m.release_inactive_slots(&[false; 4]);
            m.reset().expect("reset");
            let p = &prompts[i].1;
            let all: Vec<u32> = p.iter().chain(&reply[..reply.len() - 1]).copied().collect();
            let mut pf = vec![(0usize, 0f32); all.len()];
            m.forward_each(
                &all,
                paddock_engine::gpu_model::inkling2::Logits::All,
                |r, l| pf[r] = top1(l),
            )
            .expect("teacher-forced pass");
            reply
                .iter()
                .enumerate()
                .filter(|&(k, &tok)| {
                    let (top, gap) = pf[p.len() - 1 + k];
                    top as u32 != tok && gap >= 0.5
                })
                .count()
        };
    // the baseline: plain greedy decode ticks of each prompt, as long as its
    // batched spec reply - how often decode-width rows part from one long
    // prompt pass on this text by themselves
    let v = paddock_models::inkling2::VOCAB_REAL;
    let mut plain = Vec::new();
    for (i, t) in together.iter().enumerate() {
        m.release_inactive_slots(&[false; 4]);
        let (slot, p) = &prompts[i];
        let l = Generator::forward_prefill(&mut m, *slot, p).expect("prefill");
        let mut tok = top1(&l).0 as u32;
        let mut reply = vec![tok];
        while reply.len() < t.0.len() {
            let (mut toks, mut pos) = (vec![0u32; 4], vec![0u32; 4]);
            toks[*slot] = tok;
            pos[*slot] = (p.len() + reply.len() - 1) as u32;
            let out = m.forward_batch(&toks, &pos).expect("tick");
            tok = top1(&out[slot * v..(slot + 1) * v]).0 as u32;
            reply.push(tok);
        }
        plain.push(reply);
    }
    for (i, t) in together.iter().enumerate() {
        let base = clear_of(&mut m, i, &plain[i]);
        let a = clear_of(&mut m, i, &alone[i].0);
        let clear = clear_of(&mut m, i, &t.0);
        // reported, not gated: a verify chunk rides the multi-row classes
        // (another f32 summation order than a decode row), which this looped
        // model amplifies into its own replies - the exactness of the spec
        // path is gated by `inkling2_spec_rollback`,
        // `inkling2_verify_rows_independent` and
        // `inkling2_draft_batch_vs_single`
        println!(
            "slot {}: clear partings from its prompt pass - plain decode {base}, spec alone {a}, spec batched {clear}",
            prompts[i].0
        );
    }
    for i in 0..3 {
        let alone = alone[i].clone();
        let t = &together[i];
        let agree = t.0.iter().zip(&alone.0).take_while(|(a, b)| a == b).count();
        println!(
            "slot {}: together {} drafts accepted, alone {}; replies agree on {agree} of {}",
            prompts[i].0,
            t.1,
            alone.1,
            t.0.len().min(alone.0.len())
        );
        println!("   together {:?}", &t.0[..t.0.len().min(16)]);
        println!("   alone    {:?}", &alone.0[..alone.0.len().min(16)]);
        assert!(
            t.1 * 2 >= alone.1,
            "slot {}: batched acceptance collapses ({} vs {})",
            prompts[i].0,
            t.1,
            alone.1
        );
    }
}

/// A verify row depends on its own token and the history only: the pending
/// token's logits from a 1-row verify, from a 4-row verify with arbitrary
/// drafts behind it, and from a plain decode tick must agree (the drafts are
/// later rows; nothing may flow back).
#[test]
fn inkling2_verify_rows_independent() {
    use paddock_engine::generator::Generator;
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() || !exec.has_inkling2_draft() {
        common::missing("pack has no Inkling 2 draft ops (slots 994-1001)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let (_, ib) = g.bytes("s1.ids").expect("ids");
    let text: Vec<u32> = i32s(ib).into_iter().map(|t| t as u32).collect();
    let prompt = &text[..450];
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        4096,
        512,
        kv_class(),
    )
    .expect("load");
    assert!(m.enable_batch_impl(4).expect("enable") >= 4);
    let v = paddock_models::inkling2::VOCAB_REAL;
    let l0 = Generator::forward_prefill(&mut m, 0, prompt).expect("prefill");
    let t = top1(&l0).0 as u32;
    let pos = prompt.len();
    let one = m
        .forward_spec_verify(&[(0, pos, vec![t])])
        .expect("verify")
        .expect("logits");
    m.spec_commit(&[0]).expect("back");
    let four = m
        .forward_spec_verify(&[(0, pos, vec![t, text[500], text[501], text[502]])])
        .expect("verify")
        .expect("logits");
    m.spec_commit(&[0]).expect("back");
    let four_b = m
        .forward_spec_verify(&[(0, pos, vec![t, text[700], text[33], text[901]])])
        .expect("verify")
        .expect("logits");
    m.spec_commit(&[0]).expect("back");
    let (mut toks, mut ps) = (vec![0u32; 4], vec![0u32; 4]);
    toks[0] = t;
    ps[0] = pos as u32;
    let dec = m.forward_batch(&toks, &ps).expect("tick");
    let diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let (d1, d2) = (diff(&one[..v], &four[..v]), diff(&one[..v], &dec[..v]));
    let d3 = diff(&four[..v], &four_b[..v]);
    println!(
        "pending row: 1-row verify vs 4-row {d1:.3e}, vs decode tick {d2:.3e}; 4-row with other drafts {d3:.3e}; top-1 {:?} {:?} {:?}",
        top1(&one[..v]),
        top1(&four[..v]),
        top1(&dec[..v])
    );
    // the same row through the same classes with other rows behind it: the
    // logits are the row's alone
    assert_eq!(
        d3, 0.0,
        "the pending row's logits move with the drafts behind it"
    );
    assert_eq!(d2, 0.0, "a 1-row verify is a decode tick");
}
