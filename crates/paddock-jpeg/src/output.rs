//! From component planes to RGB: the upsampler libjpeg elects per component
//! (jdsample.c `jinit_upsampler`), then its color conversion (jdcolor.c) -
//! and for CMYK, torchvision's conversion to RGB after libjpeg's.

use crate::{ColorSpace, Component};

/// A component plane: its samples and row stride.
struct Plane<'a> {
    px: &'a [u8],
    stride: usize,
    /// real samples a row and real rows (downsampled_width / _height)
    dw: usize,
    dh: usize,
}

impl Plane<'_> {
    #[inline]
    fn at(&self, r: usize, x: usize) -> i32 {
        i32::from(self.px[r * self.stride + x])
    }
    /// the row above / below r as the main controller provides it: the
    /// first and last real rows stand in past the image's edges
    #[inline]
    fn above(&self, r: usize) -> usize {
        r.saturating_sub(1)
    }
    #[inline]
    fn below(&self, r: usize) -> usize {
        (r + 1).min(self.dh - 1)
    }
}

/// libjpeg's h2v1 fancy upsampling of one row of `n` samples into `out`
/// (2n wide): 3/4 nearer + 1/4 further, biases 1 and 2.
fn h2v1_fancy_row(inp: &[i32], out: &mut [i32]) {
    let n = inp.len();
    out[0] = inp[0];
    out[1] = (inp[0] * 3 + inp[1] + 2) >> 2;
    for i in 1..n - 1 {
        let v = inp[i] * 3;
        out[2 * i] = (v + inp[i - 1] + 1) >> 2;
        out[2 * i + 1] = (v + inp[i + 1] + 2) >> 2;
    }
    let v = inp[n - 1];
    out[2 * n - 2] = (v * 3 + inp[n - 2] + 1) >> 2;
    out[2 * n - 1] = v;
}

/// libjpeg's h2v2 fancy upsampling of one output row from column sums
/// (3 * nearer row + further row, `n` of them): biases 8 and 7 by turns.
fn h2v2_fancy_row(sum: &[i32], out: &mut [i32]) {
    let n = sum.len();
    out[0] = (sum[0] * 4 + 8) >> 4;
    out[1] = (sum[0] * 3 + sum[1] + 7) >> 4;
    for i in 1..n - 1 {
        out[2 * i] = (sum[i] * 3 + sum[i - 1] + 8) >> 4;
        out[2 * i + 1] = (sum[i] * 3 + sum[i + 1] + 7) >> 4;
    }
    out[2 * n - 2] = (sum[n - 1] * 3 + sum[n - 2] + 8) >> 4;
    out[2 * n - 1] = (sum[n - 1] * 4 + 7) >> 4;
}

/// One component at full resolution (`width` x `height`).
fn upsample(
    c: &Component,
    p: &Plane<'_>,
    width: usize,
    height: usize,
    max_h: usize,
    max_v: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    let (h, v) = (c.h, c.v);
    if h == max_h && v == max_v {
        for y in 0..height {
            out[y * width..(y + 1) * width]
                .copy_from_slice(&p.px[y * p.stride..y * p.stride + width]);
        }
    } else if 2 * h == max_h && v == max_v && p.dw > 2 {
        let mut row = vec![0i32; p.dw];
        let mut up = vec![0i32; 2 * p.dw];
        for y in 0..height {
            for (x, r) in row.iter_mut().enumerate() {
                *r = p.at(y, x);
            }
            h2v1_fancy_row(&row, &mut up);
            for x in 0..width {
                out[y * width + x] = up[x] as u8;
            }
        }
    } else if h == max_h && 2 * v == max_v {
        // h1v2 fancy (no width condition): output row 2r leans on the row
        // above (bias 1), 2r + 1 on the row below (bias 2)
        for y in 0..height {
            let r = y / 2;
            let (other, bias) = if y % 2 == 0 {
                (p.above(r), 1)
            } else {
                (p.below(r), 2)
            };
            for x in 0..width {
                out[y * width + x] = ((p.at(r, x) * 3 + p.at(other, x) + bias) >> 2) as u8;
            }
        }
    } else if 2 * h == max_h && 2 * v == max_v && p.dw > 2 {
        let mut sum = vec![0i32; p.dw];
        let mut up = vec![0i32; 2 * p.dw];
        for y in 0..height {
            let r = y / 2;
            let other = if y % 2 == 0 { p.above(r) } else { p.below(r) };
            for (x, s) in sum.iter_mut().enumerate() {
                *s = p.at(r, x) * 3 + p.at(other, x);
            }
            h2v2_fancy_row(&sum, &mut up);
            for x in 0..width {
                out[y * width + x] = up[x] as u8;
            }
        }
    } else {
        // plain replication: h2v1 / h2v2 too narrow for fancy, and every
        // other integral factor (int_upsample)
        let (he, ve) = (max_h / h, max_v / v);
        for y in 0..height {
            for x in 0..width {
                out[y * width + x] = p.px[(y / ve) * p.stride + x / he];
            }
        }
    }
    out
}

/// jdcolor.c's YCbCr tables: 16-bit fixed point, `FIX(x) = x * 65536 + 0.5`
/// truncated, Cr->R and Cb->B rounded per entry, the green terms summed
/// before their one shift.
struct YccTables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i64; 256],
    cb_g: [i64; 256],
}

impl YccTables {
    fn new() -> Self {
        let fix = |x: f64| (x * 65536.0 + 0.5) as i64;
        let half = 1i64 << 15;
        let mut t = Self {
            cr_r: [0; 256],
            cb_b: [0; 256],
            cr_g: [0; 256],
            cb_g: [0; 256],
        };
        for i in 0..256 {
            let x = i as i64 - 128;
            t.cr_r[i] = ((fix(1.40200) * x + half) >> 16) as i32;
            t.cb_b[i] = ((fix(1.77200) * x + half) >> 16) as i32;
            t.cr_g[i] = -fix(0.71414) * x;
            t.cb_g[i] = -fix(0.34414) * x + half;
        }
        t
    }

    /// (R, G, B) before range limiting
    #[inline]
    fn rgb(&self, y: i32, cb: u8, cr: u8) -> (i32, i32, i32) {
        let (cb, cr) = (usize::from(cb), usize::from(cr));
        (
            y + self.cr_r[cr],
            y + ((self.cb_g[cb] + self.cr_g[cr]) >> 16) as i32,
            y + self.cb_b[cb],
        )
    }
}

#[inline]
fn clamp(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

/// torchvision's CMYK -> RGB (Pillow's formula, on libjpeg's CMYK).
#[inline]
fn cmyk_rgb(k: u8, cmy: u8) -> u8 {
    let (k, cmy) = (i32::from(k), i32::from(cmy));
    let v = k * cmy + 128;
    let v = ((v >> 8) + v) >> 8;
    clamp(k - v)
}

pub(crate) fn to_rgb(
    comps: &[Component],
    planes: &[(Vec<u8>, usize)],
    width: usize,
    height: usize,
    max_h: usize,
    max_v: usize,
    cs: ColorSpace,
) -> Vec<u8> {
    let full: Vec<Vec<u8>> = comps
        .iter()
        .zip(planes)
        .map(|(c, (px, stride))| {
            let p = Plane {
                px,
                stride: *stride,
                dw: c.dw,
                dh: c.dh,
            };
            upsample(c, &p, width, height, max_h, max_v)
        })
        .collect();
    let n = width * height;
    let mut rgb = vec![0u8; n * 3];
    match cs {
        ColorSpace::Gray => {
            for i in 0..n {
                rgb[3 * i..3 * i + 3].fill(full[0][i]);
            }
        }
        ColorSpace::Rgb => {
            for i in 0..n {
                rgb[3 * i] = full[0][i];
                rgb[3 * i + 1] = full[1][i];
                rgb[3 * i + 2] = full[2][i];
            }
        }
        ColorSpace::YCbCr => {
            let t = YccTables::new();
            for i in 0..n {
                let (r, g, b) = t.rgb(i32::from(full[0][i]), full[1][i], full[2][i]);
                rgb[3 * i] = clamp(r);
                rgb[3 * i + 1] = clamp(g);
                rgb[3 * i + 2] = clamp(b);
            }
        }
        ColorSpace::Cmyk | ColorSpace::Ycck => {
            let t = YccTables::new();
            for i in 0..n {
                let (c, m, y) = if cs == ColorSpace::Ycck {
                    // ycck_cmyk_convert: 255 - the YCbCr conversion
                    let (r, g, b) = t.rgb(i32::from(full[0][i]), full[1][i], full[2][i]);
                    (clamp(255 - r), clamp(255 - g), clamp(255 - b))
                } else {
                    (full[0][i], full[1][i], full[2][i])
                };
                let k = full[3][i];
                rgb[3 * i] = cmyk_rgb(k, 255 - c);
                rgb[3 * i + 1] = cmyk_rgb(k, 255 - m);
                rgb[3 * i + 2] = cmyk_rgb(k, 255 - y);
            }
        }
    }
    rgb
}
