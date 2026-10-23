//! Pixels and positions for Nemotron 3.5 Parse - the release's processor
//! (`hf_nemotron_parse_processor.py`) and its position field
//! (`hf_prefix_geometry.py`), reproduced so the tower sees the reference's
//! numbers, not an approximation of them.
//!
//! **Pixels** (host, byte-exact): `LongestMaxSizeHW` (downscale only, height
//! cap first, `int()` truncation - see `NemotronParseConfig::content_hw`),
//! Pillow BILINEAR through [`crate::pillow`]'s exact fixed-point path, then
//! white padding on the bottom and right up to the 16 x 64 grid. What leaves
//! here is the padded RGB8 canvas; normalization and patching run on the
//! device (`np_patch_u8`), so a page crosses the bus as 3 bytes a pixel.
//!
//! **Positions**: the 128 x 128 learned grid is resized to the CONTENT grid
//! (ceil(content / 16) a side) with torch's antialiased bilinear, then
//! sampled at every canvas patch centre - normalized to the content extent,
//! so padding patches clamp to the border values - with `grid_sample`
//! (bilinear, border, align_corners=False). The reference runs both in F32 on
//! the GPU (`positional_grid.float()` on the model's device), so this file
//! follows PyTorch's CUDA kernels (torch 2.13 `UpSampleBilinear2d.cu`,
//! `GridSampler.cuh`) operation for operation: the antialias span
//! (`upsample_gen2d_aa_out_frame`), the x-then-y accumulation order, and the
//! multiply-adds nvcc contracts into FMAs in both kernels (`mul_add` below
//! marks each one); the antialias WEIGHTS are the one place it is not literal
//! (see [`aa_axis`]). The elementwise ops
//! that build the sampling grid are separate eager kernels in the reference,
//! so they are NOT fused here. A field depends only on (content, canvas)
//! shape, so pages of one document share it - the tower caches by shape.

use crate::pillow::{Filter, resize_rgb8};
use paddock_models::nemotron_parse::NemotronParseConfig;

/// A page ready for the device: the padded canvas plus the two sizes the
/// position field and the decoder's box frame need.
pub struct PagePrep {
    /// RGB8 `[canvas_h][canvas_w][3]`
    pub canvas: Vec<u8>,
    /// (height, width) after the aspect-preserving downscale - the box frame
    pub content_hw: (usize, usize),
    /// (height, width) padded to the 16 x 64 grid
    pub canvas_hw: (usize, usize),
}

impl PagePrep {
    pub fn patch_rows(&self, patch: usize) -> usize {
        (self.canvas_hw.0 / patch) * (self.canvas_hw.1 / patch)
    }
}

/// The processor on one RGB8 page (`w` x `h`, tightly packed).
pub fn prep_page(rgb: &[u8], w: usize, h: usize, cfg: &NemotronParseConfig) -> PagePrep {
    assert_eq!(rgb.len(), 3 * w * h, "expected tightly-packed RGB8");
    let content_hw = cfg.content_hw(w, h);
    let (ch, cw) = content_hw;
    let resized = if (ch, cw) == (h, w) {
        // the processor returns the array untouched when nothing changes
        std::borrow::Cow::Borrowed(rgb)
    } else {
        std::borrow::Cow::Owned(resize_rgb8(rgb, w, h, cw, ch, Filter::Bilinear))
    };
    let canvas_hw = cfg.canvas_hw(content_hw);
    let (hh, ww) = canvas_hw;
    // np.pad(constant_values=255) on the bottom/right
    let mut canvas = vec![255u8; 3 * hh * ww];
    for y in 0..ch {
        canvas[y * ww * 3..y * ww * 3 + cw * 3]
            .copy_from_slice(&resized[y * cw * 3..(y + 1) * cw * 3]);
    }
    PagePrep {
        canvas,
        content_hw,
        canvas_hw,
    }
}

/// One axis of `upsample_gen2d_aa_out_frame`: per output index, the first
/// source index and the normalized weights. The SPAN is the kernel's own f32
/// arithmetic (`_compute_weights_span`: the int truncations decide which
/// taps exist, so they must match exactly); the WEIGHTS are computed in f64
/// and rounded once. Measured against torch 2.13's CUDA kernel on random
/// fields: its f32 weight math reproduced literally lands within 9.9e-7 of
/// it, the f64 weights within 4.5e-8 (one f32 ulp) - the kernel's divides are
/// not the host's IEEE ones bit for bit, and the closer of the two is taken.
/// Either is below the f32 rounding of the patch embedding (~20) the field is
/// added to, so the tower cannot see the difference.
fn aa_axis(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = in_size as f32 / out_size as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let invscale = if scale >= 1.0 {
        1.0 / scale as f64
    } else {
        1.0
    };
    (0..out_size)
        .map(|i| {
            let center = scale * (i as f32 + 0.5);
            let xmin = ((center - support + 0.5) as i32).max(0) as usize;
            let xmax = ((center + support + 0.5) as i32).min(in_size as i32) as usize;
            let c64 = center as f64;
            let w: Vec<f64> = (0..xmax - xmin)
                .map(|j| {
                    let x = (((j + xmin) as f64 - c64) + 0.5) * invscale;
                    let x = x.abs();
                    if x < 1.0 { 1.0 - x } else { 0.0 }
                })
                .collect();
            let total: f64 = w.iter().sum();
            let w = w
                .iter()
                .map(|&v| {
                    if total != 0.0 {
                        (v / total) as f32
                    } else {
                        v as f32
                    }
                })
                .collect();
            (xmin, w)
        })
        .collect()
}

/// `interpolate_aa_single_dim`: first product plain, the rest FMA-contracted.
#[inline]
fn aa_dot(src: impl Iterator<Item = f32>, w: &[f32]) -> f32 {
    let mut it = src.zip(w.iter());
    let Some((s0, &w0)) = it.next() else {
        return 0.0;
    };
    let mut acc = s0 * w0;
    for (s, &wt) in it {
        acc = s.mul_add(wt, acc);
    }
    acc
}

/// The position field one page adds to its patch embeddings, `[rows*cols][dim]`
/// row-major over canvas patches. `grid` is the checkpoint's `pos_embed`
/// (`[src * src][dim]`, token `y * src + x`). Content sides of one patch take
/// the reference's area-preserving pool instead (training's augmentation
/// crops); that path is torch's `adaptive_avg_pool2d`.
pub fn position_field(
    grid: &[f32],
    src: usize,
    dim: usize,
    content_hw: (usize, usize),
    canvas_hw: (usize, usize),
    patch: usize,
) -> Vec<f32> {
    assert_eq!(grid.len(), src * src * dim);
    let (ch, cw) = content_hw;
    let (rows, cols) = (canvas_hw.0 / patch, canvas_hw.1 / patch);
    let (crows, ccols) = (ch.div_ceil(patch), cw.div_ceil(patch));

    // ---- 1. the 128 x 128 field resized to the content grid, [crows][ccols][dim]
    let mut field = vec![0f32; crows * ccols * dim];
    if crows.min(ccols) == 1 {
        // adaptive_avg_pool2d: window [floor(i*in/out), ceil((i+1)*in/out))
        let span = |i: usize, o: usize| (i * src / o, ((i + 1) * src).div_ceil(o));
        for oy in 0..crows {
            let (y0, y1) = span(oy, crows);
            for ox in 0..ccols {
                let (x0, x1) = span(ox, ccols);
                let n = ((y1 - y0) * (x1 - x0)) as f32;
                let out = &mut field[(oy * ccols + ox) * dim..(oy * ccols + ox + 1) * dim];
                for (c, o) in out.iter_mut().enumerate() {
                    let mut s = 0f32;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            s += grid[(y * src + x) * dim + c];
                        }
                    }
                    *o = s / n;
                }
            }
        }
    } else {
        let wx = aa_axis(src, ccols);
        let wy = aa_axis(src, crows);
        let mut tmp = Vec::new();
        for (oy, (ymin, wyv)) in wy.iter().enumerate() {
            for (ox, (xmin, wxv)) in wx.iter().enumerate() {
                let out = &mut field[(oy * ccols + ox) * dim..(oy * ccols + ox + 1) * dim];
                for (c, o) in out.iter_mut().enumerate() {
                    // x first (each source row of the y span), then y
                    tmp.clear();
                    for yy in 0..wyv.len() {
                        let row = (ymin + yy) * src;
                        tmp.push(aa_dot(
                            (0..wxv.len()).map(|xx| grid[(row + xmin + xx) * dim + c]),
                            wxv,
                        ));
                    }
                    *o = aa_dot(tmp.iter().copied(), wyv);
                }
            }
        }
    }

    // ---- 2. grid_sample at each canvas patch centre, normalized to the content
    // the sampling grid exactly as the reference's eager ops build it
    let norm = |i: usize, extent: usize| -> f32 {
        let start = (i * patch) as f32;
        let ext = (extent as f32 - start).clamp(0.0, patch as f32);
        let v = (start + 0.5 * ext).min(extent as f32);
        v / extent as f32
    };
    // grid_sampler_compute_source_index (unnormalize with the contracted
    // fma, then the border clip)
    let src_index = |g: f32, size: usize| -> f32 {
        let c = (g + 1.0).mul_add(size as f32, -1.0) / 2.0;
        c.max(0.0).min((size - 1) as f32)
    };
    let mut out = vec![0f32; rows * cols * dim];
    for r in 0..rows {
        let gy = 2.0 * norm(r, ch) - 1.0;
        let iy = src_index(gy, crows);
        for cc in 0..cols {
            let gx = 2.0 * norm(cc, cw) - 1.0;
            let ix = src_index(gx, ccols);
            let (ix_nw, iy_nw) = (ix.floor() as i64, iy.floor() as i64);
            let (ix_se, iy_se) = (ix_nw + 1, iy_nw + 1);
            let nw = (ix_se as f32 - ix) * (iy_se as f32 - iy);
            let ne = (ix - ix_nw as f32) * (iy_se as f32 - iy);
            let sw = (ix_se as f32 - ix) * (iy - iy_nw as f32);
            let se = (ix - ix_nw as f32) * (iy - iy_nw as f32);
            let at = |y: i64, x: i64| -> Option<usize> {
                (y >= 0 && x >= 0 && (y as usize) < crows && (x as usize) < ccols)
                    .then(|| (y as usize * ccols + x as usize) * dim)
            };
            let taps = [
                (at(iy_nw, ix_nw), nw),
                (at(iy_nw, ix_se), ne),
                (at(iy_se, ix_nw), sw),
                (at(iy_se, ix_se), se),
            ];
            let o = &mut out[(r * cols + cc) * dim..(r * cols + cc + 1) * dim];
            for (c, v) in o.iter_mut().enumerate() {
                let mut acc = 0f32;
                for &(base, wt) in &taps {
                    if let Some(b) = base {
                        acc = field[b + c].mul_add(wt, acc);
                    }
                }
                *v = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When the content grid equals the source grid, the antialias resize is
    /// the identity (one tap of weight 1 per output) and every patch centre
    /// samples its own cell - the field is the stored grid, exactly.
    #[test]
    fn a_full_grid_page_reads_the_stored_field() {
        let (src, dim, p) = (8usize, 3usize, 16usize);
        let grid: Vec<f32> = (0..src * src * dim)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let f = position_field(&grid, src, dim, (src * p, src * p), (src * p, src * p), p);
        assert_eq!(f, grid);
    }

    /// Padding patches past the content clamp to the content's border values.
    #[test]
    fn padding_patches_take_the_border() {
        let (src, dim, p) = (8usize, 2usize, 16usize);
        let grid: Vec<f32> = (0..src * src * dim).map(|i| i as f32).collect();
        // content 8 x 6 patches on an 8 x 8 canvas
        let f = position_field(&grid, src, dim, (8 * p, 6 * p), (8 * p, 8 * p), p);
        for r in 0..8 {
            for c in 6..8 {
                assert_eq!(
                    &f[(r * 8 + c) * dim..(r * 8 + c + 1) * dim],
                    &f[(r * 8 + 5) * dim..(r * 8 + 6) * dim],
                    "row {r} col {c}"
                );
            }
        }
    }
}
