//! The diarizer's stored-weight GEMM (slot 722) elects its tile arm from the
//! row count. Its per-element sequence is meant to be the same in every arm,
//! so a row's bits must not depend on how many rows share the call - the
//! property that lets a pass carry several diarizer windows at once. Each
//! served shape and epilogue at a 16-row-arm count against the same rows
//! inside a 619-row call. Pack only, no checkpoint.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::gpu::{DiarEpi, DiarWeights};

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn diar_gemm_rows_invariant_across_arms() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_diar_gemm() {
        common::missing("pack lacks slot 722");
        return;
    }
    let big = 619usize;
    let mut bad = Vec::new();
    for &(k, n, epi) in &[
        (512usize, 1536usize, DiarEpi::Rope),
        (512, 512, DiarEpi::Resid),
        (512, 2048, DiarEpi::Gelu),
        (2048, 512, DiarEpi::Resid),
        (512, 192, DiarEpi::Store),
        (1024, 512, DiarEpi::Store),
    ] {
        let w = exec
            .to_device_u8(&bf16_bytes(&det(n * k, 7 + n as u64)))
            .unwrap();
        let b = exec.to_device(&det(n, 11)).unwrap();
        let x = exec.to_device(&det(big * k, 13 + k as u64)).unwrap();
        let rope = exec.to_device(&det(big * 64, 17)).unwrap();
        let y0 = det(big * n, 19);
        for small in [1usize, 15, 33, 62, 97, 130] {
            let run = |m: usize| -> (Vec<f32>, Vec<f32>) {
                let mut y = exec.to_device(&y0).unwrap();
                let mut q = exec.to_device(&vec![0f32; big * 512]).unwrap();
                let r = if epi == DiarEpi::Rope {
                    Some((&mut q, &rope))
                } else {
                    None
                };
                exec.diar_gemm(
                    (&x, 0),
                    DiarWeights::Bf16(&w),
                    Some(&b),
                    &mut y,
                    (k, n, m),
                    epi,
                    r,
                )
                .unwrap();
                (exec.to_host(&y).unwrap(), exec.to_host(&q).unwrap())
            };
            let (ys, qs) = run(small);
            let (yb, qb) = run(big);
            let same =
                |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            let ok = same(&ys[..small * n], &yb[..small * n])
                && (epi != DiarEpi::Rope || same(&qs[..small * 512], &qb[..small * 512]));
            if !ok {
                bad.push(format!("k {k} n {n} {epi:?} at {small} rows"));
            }
        }
    }
    assert!(bad.is_empty(), "rows differ from the 619-row call: {bad:?}");
}
