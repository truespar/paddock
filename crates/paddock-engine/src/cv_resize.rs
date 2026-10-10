//! OpenCV-compatible uint8 `INTER_CUBIC` resizing - the reference
//! preprocessing for a checkpoint whose pipeline resizes with
//! `cv2.resize(..., interpolation=cv2.INTER_CUBIC)` (PaddleX's `Resize` with
//! `interp: 2`; PP-DocLayoutV3's 800 x 800 input).
//!
//! The exact path of OpenCV 4.10's `resize.cpp` on uint8, not an
//! approximation of it - like `pillow.rs`, because a resize is the first op
//! of a network and a +-1 pixel there is unattributable later:
//!
//! - per destination column / row, `f = (float)((d + 0.5) * scale - 0.5)`
//!   with `scale = src / dst` in double, `s = floor(f)`, `x = f - s`;
//! - the four cubic weights at A = -0.75 in float (OpenCV's own expression
//!   order), each rounded to a Q11 short (round to nearest even) - and not
//!   renormalised to sum to 2048;
//! - a horizontal pass in int32 over the four taps, source columns outside
//!   the image replicated from the edge;
//! - the vertical pass as OpenCV's SIMD kernel runs it: the four int32 rows
//!   to float, the Q11 row weights scaled by 2^-22, three fused
//!   multiply-adds and a multiply, round to nearest even, saturate to u8.
//!   Rows outside the image replicate the edge as the columns do.
//!
//! The vertical pass is where builds differ: the scalar tail of the same
//! kernel rounds an integer sum with `+ 2^21 >> 22`, and an x86 build without
//! FMA rounds the float sums twice. NEON (and AVX2 with FMA) is the fused
//! form here; every 800-wide RGB row is a whole number of vector lanes, so no
//! scalar tail runs at that width. Gated against `cv2.resize` 4.10 below.

/// Resize interleaved RGB8 `src` (`w` x `h`) to `dw` x `dh`.
pub fn resize_cubic_rgb8(src: &[u8], w: usize, h: usize, dw: usize, dh: usize) -> Vec<u8> {
    assert_eq!(src.len(), 3 * w * h, "expected tightly-packed RGB8");
    assert!(w > 0 && h > 0 && dw > 0 && dh > 0);
    let (xs, xw) = taps(dw, w);
    let (ys, yw) = taps(dh, h);
    // horizontal: one int32 row per source row the vertical pass reads
    let mut rows: Vec<Option<Vec<i32>>> = vec![None; h];
    let mut out = vec![0u8; 3 * dw * dh];
    for dy in 0..dh {
        for &sy in &ys[dy] {
            if rows[sy].is_none() {
                rows[sy] = Some(hrow(&src[3 * w * sy..3 * w * (sy + 1)], &xs, &xw));
            }
        }
        let scale = 1.0f32 / (2048.0 * 2048.0);
        let b = yw[dy].map(|q| f32::from(q) * scale);
        let r: [&Vec<i32>; 4] = ys[dy].map(|sy| rows[sy].as_ref().expect("filled above"));
        let orow = &mut out[3 * dw * dy..3 * dw * (dy + 1)];
        for (i, o) in orow.iter_mut().enumerate() {
            let acc = (r[3][i] as f32) * b[3];
            let acc = (r[2][i] as f32).mul_add(b[2], acc);
            let acc = (r[1][i] as f32).mul_add(b[1], acc);
            let acc = (r[0][i] as f32).mul_add(b[0], acc);
            // v_round, then v_pack (i16 saturate) and v_pack_u (u8 saturate)
            let v = acc.round_ties_even() as i32;
            *o = v.clamp(-32768, 32767).clamp(0, 255) as u8;
        }
        // drop rows no later destination row reads (rows only move down)
        let keep = ys[(dy + 1).min(dh - 1)][0];
        for row in rows.iter_mut().take(keep) {
            *row = None;
        }
    }
    out
}

/// Per destination index: the four clamped source indices and the four Q11
/// weights.
#[allow(clippy::type_complexity)]
fn taps(dst: usize, src: usize) -> (Vec<[usize; 4]>, Vec<[i16; 4]>) {
    let scale = src as f64 / dst as f64;
    let mut idx = Vec::with_capacity(dst);
    let mut wts = Vec::with_capacity(dst);
    for d in 0..dst {
        let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
        let s = f.floor();
        let x = f - s;
        let s = s as i64;
        let c = cubic(x);
        wts.push(c.map(|v| (v * 2048.0).round_ties_even() as i16));
        idx.push([0, 1, 2, 3].map(|k| (s - 1 + k).clamp(0, src as i64 - 1) as usize));
    }
    (idx, wts)
}

/// OpenCV's `interpolateCubic`, expression for expression.
fn cubic(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    let c3 = 1.0 - c0 - c1 - c2;
    [c0, c1, c2, c3]
}

/// One source row through the horizontal taps, all three channels.
fn hrow(row: &[u8], xs: &[[usize; 4]], xw: &[[i16; 4]]) -> Vec<i32> {
    let mut out = vec![0i32; 3 * xs.len()];
    for (dx, (s, w)) in xs.iter().zip(xw).enumerate() {
        for c in 0..3 {
            out[3 * dx + c] = (0..4)
                .map(|k| i32::from(row[3 * s[k] + c]) * i32::from(w[k]))
                .sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic, edge-rich RGB pattern both sides can build.
    fn pattern(w: usize, h: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(3 * w * h);
        for y in 0..h as u64 {
            for x in 0..w as u64 {
                for c in 0..3u64 {
                    v.push(((x * 7 + y * 13 + c * 101) ^ ((x * y) >> 3)) as u8);
                }
            }
        }
        v
    }

    fn fnv1a(b: &[u8]) -> u64 {
        b.iter().fold(0xcbf2_9ce4_8422_2325, |h, &v| {
            (h ^ u64::from(v)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Hashes of `cv2.resize(pattern, (dw, dh), interpolation=INTER_CUBIC)`
    /// from OpenCV 4.10.0 (aarch64, the PaddleX pin): a page down to the
    /// layout model's 800 x 800, odd sizes both ways, an upscale, and a
    /// non-square target.
    #[test]
    fn matches_opencv_4_10_byte_for_byte() {
        for (w, h, dw, dh, want) in [
            (
                1448usize,
                2048usize,
                800usize,
                800usize,
                0x52c7_c55f_294a_2491u64,
            ),
            (333, 517, 800, 800, 0x9bf0_393e_a4f6_9099),
            (900, 300, 800, 800, 0x90c3_ab87_f86b_90ac),
            (64, 48, 800, 800, 0x3af4_c794_1ebc_94ef),
            (1200, 1600, 640, 480, 0x9bae_0a57_2279_87c9),
        ] {
            let got = fnv1a(&resize_cubic_rgb8(&pattern(w, h), w, h, dw, dh));
            assert_eq!(got, want, "{w}x{h} -> {dw}x{dh}");
        }
    }

    #[test]
    fn a_flat_image_stays_flat() {
        let src = vec![77u8; 3 * 31 * 17];
        assert!(
            resize_cubic_rgb8(&src, 31, 17, 50, 9)
                .iter()
                .all(|&v| v == 77)
        );
    }
}
