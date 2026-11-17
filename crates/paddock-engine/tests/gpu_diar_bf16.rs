//! The diarization encoder's BF16 class (slots 965-967, Transcribe's speaker
//! branch) held to the arithmetic: the bf16 GEMM's four landings against the
//! exact product of the same bf16 operands, and the LayerNorm and the
//! attention landed as bf16 against their own f32 landings rounded once -
//! the same values, so the same codes.
#![allow(clippy::unwrap_used)]

mod common;

use half::bf16;
use paddock_engine::gpu::{DiarB16Out, DiarEpi, DiarKeys, GpuExecutor};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn normal(&mut self) -> f64 {
        // Box-Muller
        let u = ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let v = ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn setup() -> Option<std::sync::Arc<GpuExecutor>> {
    let e = common::gpu_arc()?;
    if !e.has_diar_bf16() {
        common::missing("pack lacks the BF16-class diarization kernels (slots 965-967)");
        return None;
    }
    Some(e)
}

fn gelu(v: f64) -> f64 {
    0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a = x.abs();
    let r = if a < 2.5 {
        let (mut term, mut sum) = (a, a);
        for n in 1..60 {
            term *= -a * a / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        let mut f = 0.0;
        for n in (1..80).rev() {
            f = n as f64 / 2.0 / (a + f);
        }
        1.0 - (-a * a).exp() / std::f64::consts::PI.sqrt() / (a + f)
    };
    r.copysign(x)
}

#[test]
fn diar_bf16_gemm_landings_match_the_exact_product() {
    let Some(e) = setup() else {
        return;
    };
    let mut rng = Rng(0x0d1a_b16b_16b1_6001);
    // the encoder's projections at a stream's first windows, a window, the
    // ceiling, and one row
    for &(k, f, rows, epi) in &[
        (512usize, 1536usize, 300usize, DiarEpi::Rope),
        (512, 1536, 828, DiarEpi::Rope),
        (512, 512, 37, DiarEpi::Resid),
        (512, 512, 828, DiarEpi::Resid),
        (512, 2048, 684, DiarEpi::Gelu),
        (2048, 512, 1, DiarEpi::Resid),
        (2048, 512, 532, DiarEpi::Resid),
        (512, 512, 300, DiarEpi::Store),
    ] {
        let w: Vec<bf16> = (0..f * k)
            .map(|_| bf16::from_f64(rng.normal() * 0.05))
            .collect();
        let x: Vec<bf16> = (0..rows * k)
            .map(|_| bf16::from_f64(rng.normal()))
            .collect();
        let bias: Vec<f32> = (0..f).map(|_| (rng.normal() * 0.1) as f32).collect();
        let resid: Vec<f32> = (0..rows * f).map(|_| rng.normal() as f32).collect();
        let table: Vec<f32> = (0..rows * 32)
            .flat_map(|i| {
                let a = (i % 32) as f64 * 0.37 + (i / 32) as f64 * 0.01;
                [a.cos() as f32, a.sin() as f32]
            })
            .collect();
        // exact sums and their condition
        let mut exact = vec![0f64; rows * f];
        let mut cond = vec![0f64; rows * f];
        for r in 0..rows {
            for o in 0..f {
                let (mut s, mut c) = (0f64, 0f64);
                for kk in 0..k {
                    let p = f64::from(w[o * k + kk]) * f64::from(x[r * k + kk]);
                    s += p;
                    c += p.abs();
                }
                exact[r * f + o] = s;
                cond[r * f + o] = c;
            }
        }
        let tol = |i: usize| 4e-6 * cond[i] + 1e-6;
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        let dw = e.to_device_u8(&wb).unwrap();
        let dx = e.to_device_bf16(&x).unwrap();
        let db = e.to_device(&bias).unwrap();
        let what = format!("[{k} -> {f}, {rows} rows, {epi:?}]");
        match epi {
            DiarEpi::Gelu => {
                let mut y16 = e.stream_alloc_bf16(rows * f).unwrap();
                e.diar_b16_gemm(
                    &dx,
                    &dw,
                    Some(&db),
                    DiarB16Out::GeluB16(&mut y16),
                    (k, f, rows),
                    None,
                )
                .unwrap();
                let got: Vec<bf16> = e.stream.clone_dtoh(&y16).unwrap();
                let mut moved = 0usize;
                for i in 0..rows * f {
                    let v = gelu(exact[i] + f64::from(bias[i % f]));
                    let want = bf16::from_f64(v);
                    if got[i] != want {
                        moved += 1;
                        // the f32 sum's own error carried through the GELU
                        // (slope <= 1.13) plus one rounding to bf16: off the
                        // exact code only where that much moves it (near a
                        // boundary, or a cancelled value smaller than the
                        // sum's error)
                        let g = f64::from(got[i]);
                        assert!(
                            (g - v).abs() <= 1.2 * tol(i) + v.abs() * 2f64.powi(-8) * 1.01,
                            "{what} gelu {i}: {g} vs {} (value {v})",
                            f64::from(want)
                        );
                    }
                }
                println!("{what}: {moved}/{} codes off the exact one", rows * f);
            }
            DiarEpi::Rope => {
                let mut y = e.alloc(rows * f).unwrap();
                let mut q = e.alloc(rows * 512).unwrap();
                let dt = e.to_device(&table).unwrap();
                e.diar_b16_gemm(
                    &dx,
                    &dw,
                    Some(&db),
                    DiarB16Out::F32(&mut y, DiarEpi::Rope),
                    (k, f, rows),
                    Some((&mut q, &dt)),
                )
                .unwrap();
                let (yh, qh) = (
                    e.to_host_len(&y, rows * f).unwrap(),
                    e.to_host_len(&q, rows * 512).unwrap(),
                );
                for r in 0..rows {
                    for head in 0..24 {
                        for j in 0..32 {
                            let (ia, ic) = (r * f + head * 64 + j, r * f + head * 64 + j + 32);
                            let a = exact[ia] + f64::from(bias[head * 64 + j]);
                            let c = exact[ic] + f64::from(bias[head * 64 + j + 32]);
                            let (cs, sn) = (
                                f64::from(table[(r * 32 + j) * 2]),
                                f64::from(table[(r * 32 + j) * 2 + 1]),
                            );
                            let (wa, wc) = if head < 16 {
                                (a * cs - c * sn, c * cs + a * sn)
                            } else {
                                (a, c)
                            };
                            let (ga, gc) = if head < 8 {
                                (
                                    qh[r * 512 + head * 64 + j],
                                    qh[r * 512 + head * 64 + j + 32],
                                )
                            } else {
                                (yh[ia], yh[ic])
                            };
                            let lim = tol(ia) + tol(ic) + 4e-7 * (wa.abs() + wc.abs());
                            assert!(
                                (f64::from(ga) - wa).abs() <= lim
                                    && (f64::from(gc) - wc).abs() <= lim,
                                "{what} rope row {r} head {head} dim {j}: ({ga}, {gc}) vs ({wa}, {wc})"
                            );
                        }
                    }
                }
            }
            _ => {
                let mut y = e.to_device(&resid).unwrap();
                e.diar_b16_gemm(
                    &dx,
                    &dw,
                    Some(&db),
                    DiarB16Out::F32(&mut y, epi),
                    (k, f, rows),
                    None,
                )
                .unwrap();
                let got = e.to_host_len(&y, rows * f).unwrap();
                for i in 0..rows * f {
                    let z = exact[i] + f64::from(bias[i % f]);
                    let want = if epi == DiarEpi::Resid {
                        f64::from(resid[i]) + z
                    } else {
                        z
                    };
                    let d = (f64::from(got[i]) - want).abs();
                    assert!(
                        d <= tol(i) + want.abs() * 4e-7,
                        "{what} {i}: {} vs {want}",
                        got[i]
                    );
                }
            }
        }
    }
}

#[test]
fn diar_bf16_norm_and_attention_land_their_f32_values() {
    let Some(e) = setup() else {
        return;
    };
    let mut rng = Rng(0x0d1a_b16b_16b1_6002);
    let rows = 300usize;
    // LayerNorm: the f32 landing rounded once
    let x: Vec<f32> = (0..rows * 512)
        .map(|_| (rng.normal() * 3.0) as f32)
        .collect();
    let w: Vec<f32> = (0..512)
        .map(|_| (1.0 + rng.normal() * 0.1) as f32)
        .collect();
    let b: Vec<f32> = (0..512).map(|_| (rng.normal() * 0.1) as f32).collect();
    let (dx, dw, db) = (
        e.to_device(&x).unwrap(),
        e.to_device(&w).unwrap(),
        e.to_device(&b).unwrap(),
    );
    let mut y = e.alloc(rows * 512).unwrap();
    let mut y16 = e.stream_alloc_bf16(rows * 512).unwrap();
    e.diar_norm((&dx, 0), (&dw, &db), &mut y, rows).unwrap();
    e.diar_norm_b16((&dx, 0), (&dw, &db), &mut y16, rows)
        .unwrap();
    let (f32s, b16s): (Vec<f32>, Vec<bf16>) = (
        e.to_host_len(&y, rows * 512).unwrap(),
        e.stream.clone_dtoh(&y16).unwrap(),
    );
    for i in 0..rows * 512 {
        assert_eq!(b16s[i], bf16::from_f32(f32s[i]), "norm {i}");
    }
    // attention, one window and two packed: the f32 landing rounded once
    let q: Vec<f32> = (0..rows * 512).map(|_| rng.normal() as f32).collect();
    let qkv: Vec<f32> = (0..rows * 1536).map(|_| rng.normal() as f32).collect();
    let (dq, dkv) = (e.to_device(&q).unwrap(), e.to_device(&qkv).unwrap());
    let mut o = e.alloc(rows * 512).unwrap();
    let mut o16 = e.stream_alloc_bf16(rows * 512).unwrap();
    e.diar_attention(&dq, &dkv, &mut o, rows, 211).unwrap();
    e.diar_attention_b16(&dq, &dkv, &mut o16, rows, DiarKeys::Window(211))
        .unwrap();
    let (f32s, b16s): (Vec<f32>, Vec<bf16>) = (
        e.to_host_len(&o, rows * 512).unwrap(),
        e.stream.clone_dtoh(&o16).unwrap(),
    );
    for i in 0..rows * 512 {
        assert_eq!(b16s[i], bf16::from_f32(f32s[i]), "window attention {i}");
    }
    // two windows [0, 130) and [130, 300), tiles of up to 64 queries
    let mut tiles = Vec::new();
    for (first, len) in [(0u32, 130u32), (130, 170)] {
        let mut a = 0;
        while a < len {
            let n = (len - a).min(64);
            tiles.extend([first + a, n, first, len]);
            a += n;
        }
    }
    let n_tiles = tiles.len() / 4;
    let dt = e.to_device_u32(&tiles).unwrap();
    e.diar_attention_varlen(&dq, &dkv, &mut o, rows, &dt, n_tiles, 4)
        .unwrap();
    e.diar_attention_b16(
        &dq,
        &dkv,
        &mut o16,
        rows,
        DiarKeys::Packed {
            tiles: &dt,
            n_tiles,
            warps: 4,
        },
    )
    .unwrap();
    let (f32s, b16s): (Vec<f32>, Vec<bf16>) = (
        e.to_host_len(&o, rows * 512).unwrap(),
        e.stream.clone_dtoh(&o16).unwrap(),
    );
    for i in 0..rows * 512 {
        assert_eq!(b16s[i], bf16::from_f32(f32s[i]), "packed attention {i}");
    }
}
