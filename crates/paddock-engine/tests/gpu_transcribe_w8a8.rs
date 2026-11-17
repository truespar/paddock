//! Nemotron 3.5 Transcribe's W8A8 speech-encoder kernels (slots 961-963)
//! against an exact host reference on identical e4m3 inputs: the GEMM's three
//! landings, the LayerNorm landed as e4m3 and the f16 -> e4m3 plane. The
//! whole-encoder gate can only be statistical (a static-scale quantizer is
//! discontinuous, so two W8A8 runs that differ anywhere drift apart to the
//! class's own noise over 32 blocks); this is where the kernels are held to
//! the arithmetic.
#![allow(clippy::unwrap_used)]

mod common;

use half::f16;
use paddock_engine::gpu::{GpuExecutor, TrW8Out};
use paddock_models::modelopt::e4m3_to_f32;

/// Deterministic generator (xorshift64*).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// an e4m3 code of moderate magnitude (exponent field 3..=10), random sign
    fn code(&mut self) -> u8 {
        let r = self.next();
        let exp = 3 + (r % 8) as u8;
        let mant = ((r >> 8) % 8) as u8;
        let sign = if (r >> 16) & 1 == 1 { 0x80 } else { 0 };
        sign | (exp << 3) | mant
    }
}

/// The reference quantizer's value: `v` clamped to +-448, rounded to the
/// nearest e4m3 value, ties to the even code - by search over every code.
fn e4m3(v: f64) -> f64 {
    let c = v.clamp(-448.0, 448.0);
    let mut best = (f64::INFINITY, 0u8);
    for b in 0u8..0x7f {
        let d = (f64::from(e4m3_to_f32(b)) - c.abs()).abs();
        if d < best.0 || (d == best.0 && b & 1 == 0) {
            best = (d, b);
        }
    }
    f64::from(e4m3_to_f32(best.1)).copysign(c)
}

/// The e4m3 step around `v`'s quantized value (the distance to a neighbour).
fn step(q: f64) -> f64 {
    let a = q.abs().max(2f64.powi(-6));
    2f64.powi(a.log2().floor() as i32 - 3)
}

fn gelu(v: f64) -> f64 {
    0.5 * v * (1.0 + libm_erf(v / std::f64::consts::SQRT_2))
}

/// erf to well past f32 resolution (Abramowitz-Stegun 7.1.26 is not enough;
/// a series / continued fraction pair is).
fn libm_erf(x: f64) -> f64 {
    let a = x.abs();
    let r = if a < 2.5 {
        // Maclaurin series
        let (mut term, mut sum) = (a, a);
        for n in 1..60 {
            term *= -a * a / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        // erfc continued fraction
        let mut f = 0.0;
        for n in (1..80).rev() {
            f = n as f64 / 2.0 / (a + f);
        }
        1.0 - (-a * a).exp() / std::f64::consts::PI.sqrt() / (a + f)
    };
    r.copysign(x)
}

fn setup() -> Option<std::sync::Arc<GpuExecutor>> {
    let e = common::gpu_arc()?;
    if !e.has_tr_w8a8() {
        common::missing("pack or die lacks the W8A8 speech-encoder kernels (slots 961-963)");
        return None;
    }
    Some(e)
}

#[test]
fn w8a8_gemm_landings_match_the_exact_product() {
    let Some(e) = setup() else {
        return;
    };
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    // the encoder's four shapes, ragged row counts (and one row)
    for &(inn, out, batch) in &[
        (1280usize, 3840usize, 300usize),
        (1280, 1280, 1),
        (1280, 5120, 257),
        (5120, 1280, 129),
    ] {
        let w: Vec<u8> = (0..out * inn).map(|_| rng.code()).collect();
        let x: Vec<u8> = (0..batch * inn).map(|_| rng.code()).collect();
        let bias: Vec<f32> = (0..out).map(|_| (rng.unit() - 0.5) as f32).collect();
        let resid: Vec<f32> = (0..batch * out)
            .map(|_| (rng.unit() - 0.5) as f32)
            .collect();
        let (sw, sx) = (0.0137f32, 0.0291f32);
        let inv_next = 1.0f32 / 0.0563f32;
        // exact sums and their condition (sum of |terms|)
        let wd: Vec<f64> = w.iter().map(|&b| f64::from(e4m3_to_f32(b))).collect();
        let xd: Vec<f64> = x.iter().map(|&b| f64::from(e4m3_to_f32(b))).collect();
        let mut exact = vec![0f64; batch * out];
        let mut cond = vec![0f64; batch * out];
        for n in 0..batch {
            for m in 0..out {
                let (wr, xr) = (&wd[m * inn..(m + 1) * inn], &xd[n * inn..(n + 1) * inn]);
                let (mut s, mut c) = (0f64, 0f64);
                for k in 0..inn {
                    s += wr[k] * xr[k];
                    c += (wr[k] * xr[k]).abs();
                }
                exact[n * out + m] = s;
                cond[n * out + m] = c;
            }
        }
        let scale = f64::from(sx) * f64::from(sw);
        // f32 sums of exact products: the error grows with the condition
        let tol = |i: usize| 4e-6 * cond[i] * scale + 1e-6;
        let dw = e.to_device_u8(&w).unwrap();
        let dx = e.to_device_u8(&x).unwrap();
        let db = e.to_device(&bias).unwrap();

        // 0: the f16 landing
        let mut y16 = e.alloc_f16(batch * out).unwrap();
        e.tr_w8a8_gemm(&dw, &dx, TrW8Out::H16(&mut y16), sw, sx, inn, out, batch)
            .unwrap();
        let got = e.to_host_f16_len(&y16, batch * out).unwrap();
        let mut worst = 0f64;
        for i in 0..batch * out {
            let want = exact[i] * scale;
            let d = (f64::from(got[i].to_f32()) - want).abs();
            // one f16 rounding of the f32 value
            let lim = tol(i) + want.abs() * 2f64.powi(-11) + 6e-8;
            worst = worst.max(d / lim);
            assert!(
                d <= lim,
                "[{inn}x{out}x{batch}] f16 landing {i}: {} vs {want}",
                got[i]
            );
        }

        // 1: the projection + bias onto an f32 residual
        let mut y = e.to_device(&resid).unwrap();
        e.tr_w8a8_gemm(
            &dw,
            &dx,
            TrW8Out::BiasRes(&mut y, &db),
            sw,
            sx,
            inn,
            out,
            batch,
        )
        .unwrap();
        let got = e.to_host_len(&y, batch * out).unwrap();
        for i in 0..batch * out {
            let want = f64::from(resid[i]) + exact[i] * scale + f64::from(bias[i % out]);
            let d = (f64::from(got[i]) - want).abs();
            assert!(
                d <= tol(i) + want.abs() * 2e-7,
                "[{inn}x{out}x{batch}] residual {i}: {} vs {want}",
                got[i]
            );
        }

        // 2: bias + exact GELU, quantized for the next linear: the code is
        // the reference's except where the value sits on a rounding boundary
        // within the sums' tolerance, and then one step away
        let mut yq = e.alloc_u8(batch * out).unwrap();
        e.tr_w8a8_gemm(
            &dw,
            &dx,
            TrW8Out::GeluQ(&mut yq, &db, inv_next),
            sw,
            sx,
            inn,
            out,
            batch,
        )
        .unwrap();
        let got = e.to_host_u8_len(&yq, batch * out).unwrap();
        let mut moved = 0usize;
        for i in 0..batch * out {
            let v = exact[i] * scale + f64::from(bias[i % out]);
            let r = gelu(v) * f64::from(inv_next);
            let want = e4m3(r);
            let g = f64::from(e4m3_to_f32(got[i]));
            if g != want {
                moved += 1;
                // a boundary case: both codes adjacent, and r within the
                // GELU-propagated tolerance of their midpoint
                let mid = 0.5 * (g + want);
                assert!(
                    (g - want).abs() <= step(want) * 1.0001
                        && (r - mid).abs()
                            <= (tol(i) * 1.2 + 1e-6) * f64::from(inv_next) * 1.2 + r.abs() * 1e-6,
                    "[{inn}x{out}x{batch}] gelu-q {i}: code {:#04x} = {g} vs {want} (r = {r})",
                    got[i]
                );
            }
        }
        println!(
            "[{inn} -> {out}, {batch} rows] f16 worst {worst:.2} of the bound; gelu-q {moved}/{} \
             codes on a boundary",
            batch * out
        );
    }
}

#[test]
fn w8a8_layernorm_and_f16_planes_quantize_as_the_reference() {
    let Some(e) = setup() else {
        return;
    };
    let mut rng = Rng(0x0dd_ba11_2468_0002);
    let (rows, n) = (37usize, 1280usize);
    let x: Vec<f32> = (0..rows * n)
        .map(|i| ((rng.unit() - 0.5) * 6.0 + (i % 7) as f64 * 0.1) as f32)
        .collect();
    let w: Vec<f32> = (0..n).map(|_| (0.5 + rng.unit()) as f32).collect();
    let b: Vec<f32> = (0..n).map(|_| ((rng.unit() - 0.5) * 0.2) as f32).collect();
    let inv = 1.0f32 / 0.0317f32;
    let (dx, dw, db) = (
        e.to_device(&x).unwrap(),
        e.to_device(&w).unwrap(),
        e.to_device(&b).unwrap(),
    );
    let mut q = e.alloc_u8(rows * n).unwrap();
    e.tr_ln_e4m3(&dx, (&dw, &db), &mut q, rows, n, 1e-5, inv)
        .unwrap();
    let got = e.to_host_u8_len(&q, rows * n).unwrap();
    let mut moved = 0usize;
    for r in 0..rows {
        let xr = &x[r * n..(r + 1) * n];
        let mean = xr.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let var = xr
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let is = 1.0 / (var + 1e-5).sqrt();
        for i in 0..n {
            let y = (f64::from(xr[i]) - mean) * is * f64::from(w[i]) + f64::from(b[i]);
            let rv = y * f64::from(inv);
            let want = e4m3(rv);
            let g = f64::from(e4m3_to_f32(got[r * n + i]));
            if g != want {
                moved += 1;
                let mid = 0.5 * (g + want);
                assert!(
                    (g - want).abs() <= step(want) * 1.0001
                        && (rv - mid).abs() <= 1e-5 * rv.abs() + 1e-6,
                    "ln row {r} col {i}: {g} vs {want} (value {rv})"
                );
            }
        }
    }
    println!("LayerNorm e4m3: {moved}/{} codes on a boundary", rows * n);
    assert!(moved * 1000 <= rows * n, "too many boundary cases: {moved}");

    // an f16 plane: the same f32 product and the same rounding - every code
    let h: Vec<f16> = (0..4096)
        .map(|_| f16::from_f64((rng.unit() - 0.5) * 40.0))
        .collect();
    let dh = e
        .to_device_f16(&h.iter().map(|v| v.to_f32()).collect::<Vec<_>>(), "plane")
        .unwrap();
    let mut q = e.alloc_u8(h.len()).unwrap();
    e.tr_quant_e4m3_h(&dh, &mut q, h.len(), inv).unwrap();
    let got = e.to_host_u8_len(&q, h.len()).unwrap();
    for (i, v) in h.iter().enumerate() {
        let want = e4m3(f64::from(v.to_f32() * inv));
        assert_eq!(f64::from(e4m3_to_f32(got[i])), want, "f16 plane {i}: {v}");
    }
}
