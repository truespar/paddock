//! Pillow-compatible separable image resampling - the reference preprocessing
//! filter for every tower whose HF processor says `resample: 1` (LANCZOS) or
//! `resample: 3` (BICUBIC), and what llama.cpp's `mtmd-image.cpp` implements as
//! `resize_pillow`.
//!
//! This is the exact integer path, not an approximation of it: 22-bit
//! fixed-point weights, the `+0.5`-then-truncate rounding Pillow uses (which is
//! not `round()` - Pillow rounds once, `round()` would round twice), the `1 <<
//! 21` accumulator bias, and the u8 round-trip between the horizontal and
//! vertical passes. All of that is observable in the output bytes, and the
//! bytes are what the parity gate against llama-mtmd-cli compares through.
//!
//! Why exact and not "close enough": a resize is the first op in the tower, so
//! a ±1 LSB pixel difference feeds 50 layers of amplification before anyone
//! sees a token. Cheap to get right here, unattributable later.
//!
//! (`granite/preprocess.rs` carries an older f32-accumulate bicubic of its own,
//! documented there as an interim. It stays where it is: it is a passing gated
//! lane and moving its numerics belongs to a granite task, not a muse one.)

/// Which resampling kernel - the HF processor's `resample` value, an
/// architecture constant of the checkpoint rather than a quality knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// `resample: 1`, support radius 3. Muse Glimmer's processor_config.
    Lanczos3,
    /// `resample: 3`, support radius 2, Pillow's a = -0.5 (not torch/ggml's
    /// -0.75).
    Bicubic,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Lanczos3 => 3.0,
            Filter::Bicubic => 2.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Filter::Lanczos3 => {
                if !(-3.0..3.0).contains(&x) {
                    return 0.0;
                }
                fn sinc(v: f64) -> f64 {
                    if v == 0.0 {
                        return 1.0;
                    }
                    let pv = v * std::f64::consts::PI;
                    pv.sin() / pv
                }
                sinc(x) * sinc(x / 3.0)
            }
            Filter::Bicubic => {
                const A: f64 = -0.5;
                let x = x.abs();
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
        }
    }
}

/// 32 (i32) - 8 (u8 pixels) - 2 (accumulation headroom), Pillow's own split.
const PRECISION_BITS: u32 = 22;

/// Per-output-pixel taps: `(first_source_index, fixed_point_weights)`.
///
/// Pillow's `precompute_coeffs`. On a downscale the kernel is STRETCHED by the
/// scale factor - that stretch *is* the antialiasing, and it is why a plain
/// 6-tap Lanczos would alias badly shrinking a photo onto a 32-patch grid.
fn coeffs(in_size: usize, out_size: usize, filt: Filter) -> Vec<(usize, Vec<i32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filt.support() * filterscale;
    let ss = 1.0 / filterscale;
    let fxp = (1u64 << PRECISION_BITS) as f64;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            // C's `(int)` truncates toward zero; both bounds are then clamped,
            // which makes the truncation and a floor agree on every value that
            // survives the clamp.
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = (((center + support + 0.5) as i64).max(0) as usize).min(in_size);
            let mut w: Vec<f64> = (xmin..xmax)
                .map(|x| filt.weight(((x as f64 - center) + 0.5) * ss))
                .collect();
            let sum: f64 = w.iter().sum();
            if sum != 0.0 {
                for v in &mut w {
                    *v /= sum;
                }
            }
            // Pillow adds ±0.5 and truncates. `round()` here would round twice
            // and drift by one LSB on the ties.
            let q = w
                .iter()
                .map(|&v| (v * fxp + if v < 0.0 { -0.5 } else { 0.5 }) as i32)
                .collect();
            (xmin, q)
        })
        .collect()
}

#[inline]
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Resize tightly-packed RGB8 to `tw`×`th`. Horizontal pass then vertical,
/// with the intermediate stored as u8 exactly as Pillow's 8-bit path does.
///
/// Both passes run output-row-parallel on scoped threads past a size floor:
/// every output byte is a pure function of the read-only source plane and its
/// row's coefficients, so splitting `out` by rows is byte-identical to the
/// serial walk - same fixed-point math per pixel, no accumulation across
/// rows. An A4-class page (2.2 -> 1.0 Mpx bicubic) was ~100ms single-thread
/// inline on the serve admission path.
pub fn resize_rgb8(
    src: &[u8],
    sw: usize,
    sh: usize,
    tw: usize,
    th: usize,
    filt: Filter,
) -> Vec<u8> {
    assert_eq!(src.len(), 3 * sw * sh, "expected tightly-packed RGB8");
    assert!(sw > 0 && sh > 0 && tw > 0 && th > 0);
    if sw == tw && sh == th {
        return src.to_vec();
    }
    const BIAS: i32 = 1 << (PRECISION_BITS - 1);

    // one output row of the horizontal pass: src row y -> out[y]
    let hrow = |cx: &[(usize, Vec<i32>)], y: usize, out_row: &mut [u8]| {
        let row = y * sw * 3;
        for (x, (xmin, w)) in cx.iter().enumerate() {
            for c in 0..3 {
                let mut acc = BIAS;
                for (j, &k) in w.iter().enumerate() {
                    acc += src[row + (xmin + j) * 3 + c] as i32 * k;
                }
                out_row[x * 3 + c] = clip8(acc >> PRECISION_BITS);
            }
        }
    };

    let mid: Vec<u8> = if sw == tw {
        src.to_vec()
    } else {
        let cx = coeffs(sw, tw, filt);
        let mut out = vec![0u8; 3 * tw * sh];
        par_rows(&mut out, tw, sh, |y, out_row| hrow(&cx, y, out_row));
        out
    };

    if sh == th {
        return mid;
    }
    let cy = coeffs(sh, th, filt);
    let mut out = vec![0u8; 3 * tw * th];
    par_rows(&mut out, tw, th, |y, out_row| {
        let (ymin, w) = &cy[y];
        for x in 0..tw {
            for c in 0..3 {
                let mut acc = BIAS;
                for (j, &k) in w.iter().enumerate() {
                    acc += mid[((ymin + j) * tw + x) * 3 + c] as i32 * k;
                }
                out_row[x * 3 + c] = clip8(acc >> PRECISION_BITS);
            }
        }
    });
    out
}

/// Aspect-preserving Pillow bicubic with a centered black PAD_CEIL canvas.
/// Shared by CUDA and Metal OCR so page pixels have one arithmetic contract.
pub fn resize_pad_black(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    assert_eq!(rgb.len(), 3 * w * h);
    assert!(w > 0 && h > 0 && tw > 0 && th > 0);
    let scale = (tw as f32 / w as f32).min(th as f32 / h as f32);
    let nw = ((w as f32 * scale).ceil() as usize).min(tw).max(1);
    let nh = ((h as f32 * scale).ceil() as usize).min(th).max(1);
    let resized = resize_rgb8(rgb, w, h, nw, nh, Filter::Bicubic);
    let mut canvas = vec![0u8; 3 * tw * th];
    let ox = (tw - nw) / 2;
    let oy = (th - nh) / 2;
    for y in 0..nh {
        let s = y * nw * 3;
        let d = ((y + oy) * tw + ox) * 3;
        canvas[d..d + nw * 3].copy_from_slice(&resized[s..s + nw * 3]);
    }
    canvas
}

#[cfg(test)]
mod pad_tests {
    use super::*;

    #[test]
    fn shared_pad_ceil_keeps_identity_and_centers_black_letterboxing() {
        let src: Vec<u8> = (0..100 * 50).flat_map(|_| [255, 17, 3]).collect();
        assert_eq!(resize_pad_black(&src, 100, 50, 100, 50), src);
        let out = resize_pad_black(&src, 100, 50, 128, 96);
        assert_eq!(out.len(), 128 * 96 * 3);
        for (y, row) in out.as_chunks::<{ 128 * 3 }>().0.iter().enumerate() {
            let pixel = if (16..80).contains(&y) {
                [255, 17, 3]
            } else {
                [0; 3]
            };
            assert!(row.as_chunks::<3>().0.iter().all(|p| *p == pixel));
        }
    }
}

/// Run `f(y, row)` over every 3*`tw`-byte output row, fanning out to scoped
/// threads when the plane is big enough to pay for them. Small planes (probe
/// images, tests) stay on the caller's thread - identical bytes either way.
fn par_rows(out: &mut [u8], tw: usize, rows: usize, f: impl Fn(usize, &mut [u8]) + Sync) {
    let stride = 3 * tw;
    let nt = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(8);
    if nt < 2 || rows < 64 || rows * tw < 262_144 {
        for (y, row) in out.chunks_mut(stride).enumerate() {
            f(y, row);
        }
        return;
    }
    let band = rows.div_ceil(nt);
    std::thread::scope(|s| {
        for (b, chunk) in out.chunks_mut(band * stride).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (i, row) in chunk.chunks_mut(stride).enumerate() {
                    f(b * band + i, row);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_same_size() {
        let src: Vec<u8> = (0..3 * 9 * 5).map(|i| (i * 13 % 251) as u8).collect();
        assert_eq!(resize_rgb8(&src, 9, 5, 9, 5, Filter::Lanczos3), src);
    }

    /// Normalized taps mean a flat image survives a resize unchanged in both
    /// directions - the property the fixed-point rounding is most likely to
    /// break (a bias of one LSB per pass would show up here immediately).
    #[test]
    fn constant_image_is_preserved() {
        for (tw, th) in [(37usize, 11usize), (200, 300), (64, 64)] {
            let src = vec![173u8; 3 * 100 * 80];
            let out = resize_rgb8(&src, 100, 80, tw, th, Filter::Lanczos3);
            assert!(
                out.iter().all(|&v| v == 173),
                "lanczos {tw}x{th} did not preserve a flat image"
            );
            let out = resize_rgb8(&src, 100, 80, tw, th, Filter::Bicubic);
            assert!(out.iter().all(|&v| v == 173), "bicubic {tw}x{th}");
        }
    }

    /// A stretched kernel is the whole point of the downscale path: a 1-pixel
    /// checkerboard must average toward mid-grey, not alias into a pattern.
    /// Plain 6-tap Lanczos (no filterscale) fails this.
    #[test]
    fn downscale_antialiases() {
        let (sw, sh) = (256usize, 256usize);
        let mut src = vec![0u8; 3 * sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                for c in 0..3 {
                    src[(y * sw + x) * 3 + c] = v;
                }
            }
        }
        let out = resize_rgb8(&src, sw, sh, 32, 32, Filter::Lanczos3);
        // interior only: the edges see the clamped kernel
        for y in 2..30 {
            for x in 2..30 {
                let v = out[(y * 32 + x) * 3];
                assert!((110..=145).contains(&v), "aliased at {x},{y}: {v}");
            }
        }
    }

    /// Lanczos overshoots at a hard edge (that is the ringing it is known for)
    /// and the accumulator must CLAMP rather than wrap. A negative intermediate
    /// shifted right stays negative, so a missing clamp shows as a bright
    /// speck, not a dark one.
    #[test]
    fn ringing_clamps_instead_of_wrapping() {
        let (sw, sh) = (64usize, 8usize);
        let mut src = vec![0u8; 3 * sw * sh];
        for y in 0..sh {
            for x in (sw / 2)..sw {
                for c in 0..3 {
                    src[(y * sw + x) * 3 + c] = 255;
                }
            }
        }
        let out = resize_rgb8(&src, sw, sh, 128, 8, Filter::Lanczos3);
        // the step survives: left end black, right end white
        assert_eq!(out[0], 0);
        assert_eq!(out[(127) * 3], 255);
    }
}
