//! The prefill bf16 pair (slots 756 / 757) against the decode ladder's
//! unsplit tile: same mma per output tile, same k walk, the same
//! round-to-nearest narrowing (convert_f32_bf16 up front instead of in the
//! main loop), so every output must match BIT FOR BIT - plain and fused
//! q|k|v, on both prefill tiles (128x128 KT=64, and 128x256 on long-K planes
//! from 1024 rows), at ragged row counts. The ladder's K-split would regroup
//! sums on thin grids, so this binary turns it off for its one test. The
//! decode band's fused q|k|v multi-row GEMV (slot 773) rides along: it must
//! equal the plain multi-row GEMV over the fused plane bit for bit; so does
//! the fused sandwich norm (slot 777) against the three launches it replaces.
//! Light: synthetic planes, no checkpoint.

mod common;

use paddock_engine::gpu::QuantTensor;
use paddock_models::ggml_type::GgmlType;

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

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x * 0.05).to_le_bytes())
        .collect()
}

#[test]
fn bf16_prefill_pair_is_the_unsplit_tile() {
    // SAFETY: the only test in this binary, set before the pack's first read
    // of the switch (it is read once, at the first K-split election)
    unsafe { std::env::set_var("PADDOCK_NO_BF16_KSPLIT", "1") };
    let Some(exec) = common::gpu() else { return };
    if !exec.has_bf16_gemm_pf() {
        common::missing("pack has no bf16_gemm_pf (slot 756) - rebuild packs/cuda");
        return;
    }
    // (in, out): Kolibri's wo and shared-expert planes; the long-K wo takes
    // the 128x256 tile from 1024 rows
    for (in_dim, out) in [(6144usize, 2560usize), (2560, 512), (512, 2560)] {
        let w = QuantTensor {
            bytes: exec
                .to_device_u8(&bf16_bytes(&det(out * in_dim, 0x51 + out as u64)))
                .expect("w"),
            ty: GgmlType::Bf16,
            dims: vec![in_dim, out],
        };
        for rows in [128usize, 300, 1100, 2048] {
            let x = det(rows * in_dim, 0x77 + rows as u64);
            let d_x = exec.to_device(&x).expect("x");
            let mut x16 = exec.stream_alloc_bf16(rows * in_dim).expect("x16");
            exec.convert_f32_bf16(&d_x, &mut x16, rows * in_dim)
                .expect("narrow");
            let mut d_ref = exec.alloc(rows * out).expect("ref");
            let mut d_got = exec.alloc(rows * out).expect("got");
            exec.bf16_gemm(&w, None, &d_x, &mut d_ref, rows)
                .expect("ladder");
            exec.bf16_gemm_pf(&w, None, &x16, &mut d_got, rows)
                .expect("pf");
            let (want, have) = (
                exec.to_host(&d_ref).expect("ref host"),
                exec.to_host(&d_got).expect("got host"),
            );
            let diff = want
                .iter()
                .zip(&have)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                diff,
                0,
                "[{in_dim} -> {out}] x {rows}: {diff} of {} outputs differ from the ladder",
                want.len()
            );
        }
    }

    // fused q|k|v (Kolibri's 48 q / 4 kv heads at hd 128) against the plain
    // ladder over the same fused plane, sliced per segment
    let (hid, q_dim, kv_dim) = (2560usize, 6144usize, 512usize);
    let m = q_dim + 2 * kv_dim;
    let w = QuantTensor {
        bytes: exec
            .to_device_u8(&bf16_bytes(&det(m * hid, 0x9f)))
            .expect("w"),
        ty: GgmlType::Bf16,
        dims: vec![hid, m],
    };
    for rows in [128usize, 300, 2048] {
        let x = det(rows * hid, 0x33 + rows as u64);
        let d_x = exec.to_device(&x).expect("x");
        let mut x16 = exec.stream_alloc_bf16(rows * hid).expect("x16");
        exec.convert_f32_bf16(&d_x, &mut x16, rows * hid)
            .expect("narrow");
        let mut d_full = exec.alloc(rows * m).expect("full");
        exec.bf16_gemm(&w, None, &d_x, &mut d_full, rows)
            .expect("ladder");
        let full = exec.to_host(&d_full).expect("full host");
        let mut d_q = exec.alloc(rows * q_dim).expect("q");
        let mut d_k = exec.alloc(rows * kv_dim).expect("k");
        let mut d_v = exec.alloc(rows * kv_dim).expect("v");
        exec.bf16_qkv_gemm_pf(&w, &x16, &mut d_q, &mut d_k, &mut d_v, q_dim, kv_dim, rows)
            .expect("qkv pf");
        let got = [
            exec.to_host(&d_q).expect("q host"),
            exec.to_host(&d_k).expect("k host"),
            exec.to_host(&d_v).expect("v host"),
        ];
        for (p, (off, width)) in [(0, q_dim), (q_dim, kv_dim), (q_dim + kv_dim, kv_dim)]
            .into_iter()
            .enumerate()
        {
            for c in 0..rows {
                for r in 0..width {
                    let (want, have) = (full[c * m + off + r], got[p][c * width + r]);
                    assert_eq!(
                        have.to_bits(),
                        want.to_bits(),
                        "qkv x {rows}: plane {p} row {c} col {r}: pf {have} vs ladder {want}"
                    );
                }
            }
        }
    }
    // slot 773, the decode band: the fused q|k|v multi-row GEMV against the
    // plain multi-row GEMV over the same fused plane (the 2..=8 band's arm
    // for a 7168-row plane) - one row dot, so bit for bit
    if exec.has_bf16_qkv_gemv_mr() {
        for rows in [2usize, 3, 4, 8] {
            let x = det(rows * hid, 0x44 + rows as u64);
            let d_x = exec.to_device(&x).expect("x");
            let mut d_full = exec.alloc(rows * m).expect("full");
            exec.bf16_gemm(&w, None, &d_x, &mut d_full, rows)
                .expect("plain mr");
            let full = exec.to_host(&d_full).expect("full host");
            let mut d_q = exec.alloc(rows * q_dim).expect("q");
            let mut d_k = exec.alloc(rows * kv_dim).expect("k");
            let mut d_v = exec.alloc(rows * kv_dim).expect("v");
            assert!(
                exec.bf16_qkv_gemv_mr(&w, &d_x, &mut d_q, &mut d_k, &mut d_v, q_dim, kv_dim, rows)
                    .expect("qkv mr"),
                "slot 773 declined {rows} rows"
            );
            let got = [
                exec.to_host(&d_q).expect("q host"),
                exec.to_host(&d_k).expect("k host"),
                exec.to_host(&d_v).expect("v host"),
            ];
            for (p, (off, width)) in [(0, q_dim), (q_dim, kv_dim), (q_dim + kv_dim, kv_dim)]
                .into_iter()
                .enumerate()
            {
                for c in 0..rows {
                    for r in 0..width {
                        let (want, have) = (full[c * m + off + r], got[p][c * width + r]);
                        assert_eq!(
                            have.to_bits(),
                            want.to_bits(),
                            "qkv mr x {rows}: plane {p} row {c} col {r}: {have} vs {want}"
                        );
                    }
                }
            }
        }
    } else {
        common::missing("pack has no bf16_qkv_gemv_mr (slot 773) - rebuild packs/cuda");
    }
    // slot 777: the sandwich post-norm fused with the next norm against the
    // three launches it replaces (rmsnorm_add_scale + rmsnorm_batch +
    // convert_f32_bf16), f32 and bf16 outputs, bit for bit
    {
        let n = 2560usize;
        let wpost = exec.to_device(&det(n, 0x71)).expect("wpost");
        let wpre = exec.to_device(&det(n, 0x72)).expect("wpre");
        for rows in [256usize, 2048, 4100] {
            let x0 = det(rows * n, 0x80 + rows as u64);
            let pr = det(rows * n, 0x90 + rows as u64);
            let d_p = exec.to_device(&pr).expect("proj");
            // reference chain
            let mut xa = exec.to_device(&x0).expect("x");
            let mut xna = exec.alloc(rows * n).expect("xn");
            let mut h_a = exec.stream_alloc_bf16(rows * n).expect("x16");
            exec.rmsnorm_add_scale(&mut xa, &d_p, &wpost, n, 1e-6, 1.0, rows)
                .expect("add scale");
            exec.rmsnorm_batch(&xa, &wpre, &mut xna, n, 1e-6, rows)
                .expect("norm");
            exec.convert_f32_bf16(&xna, &mut h_a, rows * n)
                .expect("narrow");
            // fused
            let mut xb = exec.to_device(&x0).expect("x");
            let mut xnb = exec.alloc(rows * n).expect("xn");
            let mut h_b = exec.stream_alloc_bf16(rows * n).expect("x16");
            let ran = exec
                .rmsnorm_add_scale_norm(
                    &mut xb,
                    &d_p,
                    &wpost,
                    &wpre,
                    Some(&mut xnb),
                    Some(&mut h_b),
                    n,
                    1e-6,
                    1.0,
                    rows,
                )
                .expect("fused");
            if !ran {
                common::missing("pack has no rmsnorm_add_scale_norm (slot 777)");
                break;
            }
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(
                bits(exec.to_host(&xa).expect("xa")),
                bits(exec.to_host(&xb).expect("xb")),
                "fused norm x {rows}: residual differs"
            );
            assert_eq!(
                bits(exec.to_host(&xna).expect("xna")),
                bits(exec.to_host(&xnb).expect("xnb")),
                "fused norm x {rows}: xn differs"
            );
            let (ha, hb) = (
                exec.stream.clone_dtoh(&h_a).expect("ha"),
                exec.stream.clone_dtoh(&h_b).expect("hb"),
            );
            assert!(
                ha.iter().zip(&hb).all(|(a, b)| a.to_bits() == b.to_bits()),
                "fused norm x {rows}: bf16 rows differ"
            );
        }
    }
    println!("bf16 prefill pair: bit-identical to the unsplit ladder, plain and fused q|k|v");
}
