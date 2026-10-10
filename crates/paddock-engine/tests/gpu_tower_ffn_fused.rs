//! A tower FFN's seams folded into its GEMMs - `pd_f16_gemm_h_gelu_tanhf`
//! (slot 834) and `pd_f16_gemm_bias_res` (slot 835) - against the passes
//! they replace, bit for bit: `matvec_batch_f16` + `gelu_bias_f16` for the up
//! projection, `matvec_batch_f16` + `add_bias_res` for wo / down.
//!
//! Shapes are the qwen-family towers' (LightOnOCR-3: 1024 wide, 4096 FFN;
//! Qwen3.8 / PaddleOCR-VL: 1152, 4304) at row counts that walk every route
//! the f32 entry can take: a page's pass (the wide blocked ring), a small
//! picture (the narrow tile), an under-filled grid the f32 entry K-splits,
//! and decode-sized rows (the GEMV band). The last two are where the fused
//! landing cannot reproduce the f32 entry, so the pack runs the two passes
//! through the scratch plane there - the bits must hold either way.
//!
//! Gated on: CUDA device + built pack.

mod common;

use half::f16;
use paddock_engine::gpu::HalfTensor;

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

#[test]
fn fused_ffn_seams_are_the_unfused_passes() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_tower_ffn_fused() {
        panic!("the pack has no slots 834/835 (tower FFN seams) - rebuild packs/cuda");
    }
    // (embd, ffn, rows)
    for (e, ffn, rows) in [
        (1024usize, 4096usize, 11520usize),
        (1152, 4304, 5040),
        (1024, 4096, 700),
        (1152, 4304, 96),
        (1024, 4096, 40),
        (1152, 4304, 6),
    ] {
        let tag = format!("embd {e} ffn {ffn} rows {rows}");
        let seed = (e * 7 + ffn + rows) as u64;
        let half = |w: Vec<f32>, dims: Vec<usize>| HalfTensor {
            buf: exec.to_device_f16(&w, "w").unwrap(),
            dims,
        };
        // weights at a trained layer's spread, activations at a normed row's
        let up = half(det(e * ffn, seed, 0.12), vec![e, ffn]);
        let down = half(det(ffn * e, seed + 1, 0.06), vec![ffn, e]);
        let up_b = exec.to_device(&det(ffn, seed + 2, 0.5)).unwrap();
        let down_b = exec.to_device(&det(e, seed + 3, 0.5)).unwrap();
        let x16 = exec
            .to_device_f16(&det(rows * e, seed + 4, 4.0), "x")
            .unwrap();
        let resid = det(rows * e, seed + 5, 8.0);

        // up: the unfused pair, then the landing
        let mut up32 = exec.alloc(rows * ffn).unwrap();
        let mut want_h = exec.alloc_f16(rows * ffn).unwrap();
        exec.matvec_batch_f16(&up, &x16, &mut up32, rows).unwrap();
        exec.gelu_bias_f16(&up32, &up_b, &mut want_h, rows, ffn)
            .unwrap();
        let mut got_h = exec.alloc_f16(rows * ffn).unwrap();
        let mut scratch = exec.alloc(rows * ffn).unwrap();
        exec.matvec_batch_f16_gelu_tanh(&up, &x16, &mut got_h, &up_b, &mut scratch, rows)
            .unwrap();
        let want: Vec<f16> = exec.to_host_f16_len(&want_h, rows * ffn).unwrap();
        let got: Vec<f16> = exec.to_host_f16_len(&got_h, rows * ffn).unwrap();
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad,
            0,
            "up + GELU, {tag}: {bad} of {} halves differ",
            want.len()
        );

        // down onto the residual stream, from the GELU plane just landed
        let mut n32 = exec.alloc(rows * e).unwrap();
        let mut want_x = exec.to_device(&resid).unwrap();
        exec.matvec_batch_f16(&down, &want_h, &mut n32, rows)
            .unwrap();
        exec.add_bias_res(&mut want_x, &n32, &down_b, rows, e)
            .unwrap();
        let mut got_x = exec.to_device(&resid).unwrap();
        let mut scratch_e = exec.alloc(rows * e).unwrap();
        exec.matvec_batch_f16_bias_res(&down, &got_h, &mut got_x, &down_b, &mut scratch_e, rows)
            .unwrap();
        let (want, got) = (
            exec.to_host(&want_x).unwrap(),
            exec.to_host(&got_x).unwrap(),
        );
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad,
            0,
            "down + residual, {tag}: {bad} of {} differ",
            want.len()
        );
        assert!(
            want.iter().zip(&resid).any(|(a, r)| a != r),
            "{tag}: the residual did not move"
        );
    }
}
