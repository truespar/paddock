//! The qwen-family tower's attention on halves - `pd_mrope_vision_qkv_h`
//! (slot 833) feeding `pd_vision_attn_h` (slot 620) - against the f32 chain it
//! replaces, bit for bit: `bias_add` on v, `mrope_vision_bias` on q and k, the
//! f32 mma attention per picture, then `convert_f32_f16` into the wo GEMM's
//! staging plane.
//!
//! Both sides are checked twice: the three landed planes against the f32
//! chain's planes rounded on the host (q scaled first - the product the f32
//! attention forms before its own round), and the attention output against
//! the f32 output rounded. Shapes are LightOnOCR-3's tower (hd 64, two
//! pictures through one grid.z launch, a row count that leaves a ragged last
//! key tile) and PaddleOCR-VL's (hd 72, padded to 80 inside the kernel), with
//! the towers' real merged-order (y, x) positions.
//!
//! Gated on: CUDA device + built pack.

mod common;

use half::f16;

fn det(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * amp
        })
        .collect()
}

/// The tower's positions for `b` pictures of a `pw` x `ph` patch grid, rows in
/// the merged 2x2-block order, axis-major `[4, rows]` = `[y, x, y, x]`.
fn positions(b: usize, pw: usize, ph: usize) -> Vec<u32> {
    let (n, rows) = (pw * ph, b * pw * ph);
    let mut pos = vec![0u32; 4 * rows];
    for bi in 0..b {
        let mut ptr = bi * n;
        for yb in (0..ph).step_by(2) {
            for xb in (0..pw).step_by(2) {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (py, px) = ((yb + dy) as u32, (xb + dx) as u32);
                        pos[ptr] = py;
                        pos[rows + ptr] = px;
                        pos[2 * rows + ptr] = py;
                        pos[3 * rows + ptr] = px;
                        ptr += 1;
                    }
                }
            }
        }
    }
    pos
}

fn same_bits(what: &str, got: &[f16], want: &[f16]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bad = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(bad, 0, "{what}: {bad} of {} halves differ", got.len());
}

#[test]
fn half_tower_attention_is_the_f32_chain_rounded() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_vision_qkv_h() {
        panic!("the pack has no slot 833 (mrope_vision_qkv_h) - rebuild packs/cuda");
    }
    // (pictures, patch grid w x h, heads, head_dim)
    for (b, pw, ph, heads, hd) in [
        (2usize, 30usize, 22usize, 16usize, 64usize),
        (1, 24, 18, 16, 72),
    ] {
        let (n, e) = (pw * ph, heads * hd);
        let rows = b * n;
        let tag = format!("{b} x {pw}x{ph} rows, {heads} heads x hd {hd}");
        let scale = 1.0 / (hd as f32).sqrt();
        let theta_scale = 10000f32.powf(-2.0 / (hd / 2) as f32);
        let seed = hd as u64 * 31 + b as u64;
        // projection planes at the spread a tower's GEMMs land, biases smaller
        let (q, k, v) = (
            det(rows * e, seed, 16.0),
            det(rows * e, seed + 1, 16.0),
            det(rows * e, seed + 2, 6.0),
        );
        let (bq, bk, bv) = (
            det(e, seed + 3, 2.0),
            det(e, seed + 4, 2.0),
            det(e, seed + 5, 2.0),
        );
        let d_pos = exec.to_device_u32(&positions(b, pw, ph)).unwrap();
        let (d_bq, d_bk, d_bv) = (
            exec.to_device(&bq).unwrap(),
            exec.to_device(&bk).unwrap(),
            exec.to_device(&bv).unwrap(),
        );

        // the f32 chain, as the tower ran it
        let mut fq = exec.to_device(&q).unwrap();
        let mut fk = exec.to_device(&k).unwrap();
        let mut fv = exec.to_device(&v).unwrap();
        exec.bias_add(&mut fv, &d_bv, rows, e).unwrap();
        exec.mrope_vision_bias(&mut fq, &d_bq, &d_pos, rows, heads, hd, theta_scale)
            .unwrap();
        exec.mrope_vision_bias(&mut fk, &d_bk, &d_pos, rows, heads, hd, theta_scale)
            .unwrap();
        let mut fa = exec.alloc(rows * e).unwrap();
        for bi in 0..b {
            exec.vision_attn_at(&fq, &fk, &fv, &mut fa, bi * n, n, heads, hd, scale)
                .unwrap();
        }
        let mut want = exec.alloc_f16(rows * e).unwrap();
        exec.convert_f32_f16(&fa, &mut want, rows * e).unwrap();
        let want = exec.to_host_f16_len(&want, rows * e).unwrap();

        // the half chain
        let (dq, dk, dv) = (
            exec.to_device(&q).unwrap(),
            exec.to_device(&k).unwrap(),
            exec.to_device(&v).unwrap(),
        );
        let mut q16 = exec.alloc_f16(rows * e).unwrap();
        let mut k16 = exec.alloc_f16(rows * e).unwrap();
        let mut v16 = exec.alloc_f16(rows * e).unwrap();
        exec.mrope_vision_qkv_h(
            (&dq, &dk, &dv),
            (&d_bq, &d_bk, &d_bv),
            &d_pos,
            (&mut q16, &mut k16, &mut v16),
            rows,
            heads,
            hd,
            theta_scale,
            scale,
        )
        .unwrap();
        let mut got = exec.alloc_f16(rows * e).unwrap();
        exec.vision_attn_h(&q16, &k16, &v16, &mut got, n, n, heads, hd, b)
            .unwrap();
        let got = exec.to_host_f16_len(&got, rows * e).unwrap();

        // the planes: each the f32 chain's value with its one round
        let round = |x: &[f32], mul: f32| -> Vec<f16> {
            x.iter().map(|v| f16::from_f32(v * mul)).collect()
        };
        let planes = [
            ("q", &q16, exec.to_host(&fq).unwrap(), scale),
            ("k", &k16, exec.to_host(&fk).unwrap(), 1.0),
            ("v", &v16, exec.to_host(&fv).unwrap(), 1.0),
        ];
        for (name, h, f, mul) in planes {
            let h = exec.to_host_f16_len(h, rows * e).unwrap();
            same_bits(&format!("{name} plane, {tag}"), &h, &round(&f, mul));
        }
        same_bits(&format!("attention, {tag}"), &got, &want);
        assert!(
            want.iter().any(|x| x.to_f32().abs() > 0.1),
            "{tag}: degenerate attention output"
        );
    }
}
