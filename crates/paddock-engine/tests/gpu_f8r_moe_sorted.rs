//! The MTP block's FP8 experts over expert-sorted blocks (slots 951-953)
//! against the per-pick GEMVs they replace (945/946): random e4m3 planes at
//! Nemotron 3.5's expert shapes (128 routed of up 1856 x 2688 / down
//! 2688 x 1856, a 3712-wide shared expert, top-6), a draft step's rows and a
//! pass's (several 256-row chunks), every output bit-identical. Pack only.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::gpu::F8RowPlane;

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn plane(
    exec: &paddock_engine::gpu::GpuExecutor,
    experts: usize,
    out: usize,
    inn: usize,
    seed: u64,
) -> F8RowPlane {
    let mut g = lcg(seed);
    // finite e4m3 bytes only (0x7f / 0xff are NaN)
    let bytes: Vec<u8> = (0..experts * out * inn)
        .map(|_| {
            let b = (g() >> 33) as u8;
            if b & 0x7f == 0x7f { b & 0xfe } else { b }
        })
        .collect();
    let mut p = exec
        .fp8_ckpt_to_f8row(&bytes, 0.01, inn, experts * out)
        .unwrap();
    // a scale a row (each expert's own in the checkpoint)
    let scales: Vec<f32> = (0..experts * out)
        .map(|i| 0.005 + (i % 7) as f32 * 0.001)
        .collect();
    p.scale = exec.to_device(&scales).unwrap();
    p
}

#[test]
fn f8r_moe_sorted_matches_the_gemvs() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_f8r_moe_gemv() || !exec.has_f8r_moe_sorted() {
        common::missing("pack lacks slots 945-946 / 951-953");
        return;
    }
    let (e, k, inn, ff, sff) = (128usize, 6usize, 2688usize, 1856usize, 3712usize);
    let (up, down) = (plane(&exec, e, ff, inn, 1), plane(&exec, e, inn, ff, 2));
    let (su, sd) = (plane(&exec, 1, sff, inn, 3), plane(&exec, 1, inn, sff, 4));
    let chunk = 256usize;
    let nb = chunk * k / 8 + e.min(chunk * k) + 1;
    let mut srow = exec.alloc_u32(nb * 8).unwrap();
    let mut sslot = exec.alloc_u32(nb * 8).unwrap();
    let mut bexp = exec.alloc_u32(nb).unwrap();
    let mut part = exec.alloc(chunk * k * inn).unwrap();
    for rows in [1usize, 7, 32, 600] {
        let mut g = lcg(17 + rows as u64);
        let mut idx = Vec::with_capacity(rows * k);
        for _ in 0..rows {
            let mut picked = Vec::new();
            while picked.len() < k {
                let x = (g() % e as u64) as u32;
                if !picked.contains(&x) {
                    picked.push(x);
                }
            }
            idx.extend(picked);
        }
        let w: Vec<f32> = (0..rows * k)
            .map(|_| (g() % 1000) as f32 / 4000.0)
            .collect();
        let x: Vec<f32> = (0..rows * inn)
            .map(|_| ((g() % 2001) as f32 - 1000.0) / 500.0)
            .collect();
        let (di, dw, dx) = (
            exec.to_device_u32(&idx).unwrap(),
            exec.to_device(&w).unwrap(),
            exec.to_device(&x).unwrap(),
        );
        let sh_i = exec.to_device_u32(&vec![0u32; rows]).unwrap();
        let sh_w = exec.to_device(&vec![1.0f32; rows]).unwrap();
        // the GEMVs
        let mut act = exec.alloc(rows.max(chunk) * k * sff.max(ff)).unwrap();
        let mut y1 = exec.alloc(rows * inn).unwrap();
        exec.f8r_moe_up_relu2(&up, &di, &dx, &mut act, inn, ff, k, rows)
            .unwrap();
        exec.f8r_moe_down(&down, &di, &dw, &act, &mut y1, ff, inn, k, rows, false)
            .unwrap();
        exec.f8r_moe_up_relu2(&su, &sh_i, &dx, &mut act, inn, sff, 1, rows)
            .unwrap();
        exec.f8r_moe_down(&sd, &sh_i, &sh_w, &act, &mut y1, sff, inn, 1, rows, true)
            .unwrap();
        // the sorted blocks
        let mut y2 = exec.alloc(rows * inn).unwrap();
        exec.f8r_moe_sorted(
            (&up, &down),
            &di,
            &dw,
            &dx,
            &mut act,
            &mut part,
            &mut y2,
            (&mut srow, &mut sslot, &mut bexp),
            (inn, ff, inn, k, e),
            rows,
            chunk,
            false,
        )
        .unwrap();
        exec.f8r_moe_sorted(
            (&su, &sd),
            &sh_i,
            &sh_w,
            &dx,
            &mut act,
            &mut part,
            &mut y2,
            (&mut srow, &mut sslot, &mut bexp),
            (inn, sff, inn, 1, 1),
            rows,
            chunk,
            true,
        )
        .unwrap();
        let (a, b) = (exec.to_host(&y1).unwrap(), exec.to_host(&y2).unwrap());
        let bad = a
            .iter()
            .zip(&b)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        assert_eq!(bad, 0, "{rows} rows: {bad} outputs differ from the GEMVs");
        assert!(
            a.iter().all(|v| v.is_finite()),
            "{rows} rows: non-finite outputs"
        );
    }
}
