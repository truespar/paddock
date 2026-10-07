//! SAM 3's video-path ops around the tracker (`packs/cuda/src/sam3/video.cuh`,
//! slots 792-805): frames resampled the way Meta's loader does (Pillow's
//! bilinear) and normalized the way it stores them, mask bitplanes and their
//! pairwise overlaps, the hole and sprinkle fill, the resamplings and
//! non-overlap of the birth and memory chains, and the output masks at the
//! video's size.
//!
//! Planes are f32 logits, one object a plane at a fixed stride; `off` is an
//! element offset into the slice, so one allocation holds a frame's objects.
//! Buffers are checked against the geometry in Rust before any launch.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

/// Byte address of element `off` of a device slice of `T`.
fn at<T>(base: u64, off: usize) -> u64 {
    base + (off * std::mem::size_of::<T>()) as u64
}

/// How [`GpuExecutor::sam3_resize_f32`] filters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sam3Resize {
    /// torch's `upsample_bilinear2d`, `align_corners=False`
    Bilinear,
    /// torch's `_upsample_bilinear2d_aa` (`antialias=True`)
    Antialias,
}

impl GpuExecutor {
    /// Every video-path op (and the bank and memory path under it).
    pub fn has_sam3_video(&self) -> bool {
        let k = &self.kernels;
        self.has_sam3_bank()
            && k.sam3_mask_bits.is_some()
            && k.sam3_mask_pairs.is_some()
            && k.sam3_mask_clean.is_some()
            && k.sam3_mask_set.is_some()
            && k.sam3_mask_up2.is_some()
            && k.sam3_mask_owner.is_some()
            && k.sam3_mask_boxes.is_some()
            && k.sam3_pil_coeffs.is_some()
            && k.sam3_pil_pass.is_some()
            && k.sam3_patch_rows_norm.is_some()
            && k.sam3_mask_pick.is_some()
            && k.sam3_resize_f32.is_some()
            && k.sam3_mask_down4.is_some()
            && k.sam3_nonoverlap.is_some()
            && k.sam3_rle.is_some()
    }

    /// `n` planes of `px` values from `planes[off..]`, `stride` apart, as
    /// bitplanes `bits[n][ceil(px / 32)]` (bit set where > 0) and their
    /// areas. Slot 792.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_mask_bits(
        &self,
        planes: &CudaSlice<f32>,
        off: usize,
        stride: usize,
        n: usize,
        px: usize,
        bits: &mut CudaSlice<u32>,
        area: &mut CudaSlice<u32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_bits
            .ok_or(GpuError::MissingOp("sam3_mask_bits"))?;
        if n == 0 {
            return Ok(());
        }
        let words = px.div_ceil(32);
        if planes.len() < off + (n - 1) * stride + px || bits.len() < n * words || area.len() < n {
            return Err(oob("sam3_mask_bits: buffers under the planes"));
        }
        let (pp, _g1) = planes.device_ptr(&self.stream);
        let (bp, _g2) = bits.device_ptr_mut(&self.stream);
        let (ap, _g3) = area.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 792); bounds checked above
        check(unsafe {
            f(
                at::<f32>(pp, off) as *const _,
                bp as *mut _,
                ap as *mut _,
                stride as u64,
                n as u32,
                px as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The pairwise intersections of two bitplane sets (`words` a plane):
    /// `inter[i][j] = |a_i & b_j|`. Slot 793.
    pub fn sam3_mask_pairs(
        &self,
        a: &CudaSlice<u32>,
        b: &CudaSlice<u32>,
        words: usize,
        (na, nb): (usize, usize),
        inter: &mut CudaSlice<u32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_pairs
            .ok_or(GpuError::MissingOp("sam3_mask_pairs"))?;
        if na == 0 || nb == 0 {
            return Ok(());
        }
        if a.len() < na * words || b.len() < nb * words || inter.len() < na * nb {
            return Err(oob("sam3_mask_pairs: buffers under the sets"));
        }
        let (ap, _g1) = a.device_ptr(&self.stream);
        let (bp, _g2) = b.device_ptr(&self.stream);
        let (ip, _g3) = inter.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 793); bounds checked above
        check(unsafe {
            f(
                ap as *const _,
                bp as *const _,
                ip as *mut _,
                words as u32,
                na as u32,
                nb as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Meta's `fill_holes_in_mask_scores` on `n` `side x side` planes at
    /// `planes[off..]`, `stride` apart: background components of at most
    /// `max_area` take `hole`, then foreground ones of at most
    /// `min(fg / 2, max_area)` take `sprinkle`. `lab` / `area` are u32
    /// scratch of `n * side^2`, `tot` of `n`. Slot 794.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_mask_clean(
        &self,
        planes: &mut CudaSlice<f32>,
        off: usize,
        stride: usize,
        side: usize,
        n: usize,
        max_area: usize,
        (hole, sprinkle): (f32, f32),
        (lab, area, tot): (
            &mut CudaSlice<u32>,
            &mut CudaSlice<u32>,
            &mut CudaSlice<u32>,
        ),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_clean
            .ok_or(GpuError::MissingOp("sam3_mask_clean"))?;
        if n == 0 || max_area == 0 {
            return Ok(());
        }
        let px = side * side;
        if planes.len() < off + (n - 1) * stride + px
            || lab.len() < n * px
            || area.len() < n * px
            || tot.len() < n
        {
            return Err(oob("sam3_mask_clean: buffers under the planes"));
        }
        let (pp, _g1) = planes.device_ptr_mut(&self.stream);
        let (lp, _g2) = lab.device_ptr_mut(&self.stream);
        let (ap, _g3) = area.device_ptr_mut(&self.stream);
        let (tp, _g4) = tot.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 794); bounds checked above
        check(unsafe {
            f(
                at::<f32>(pp, off) as *mut _,
                lp as *mut _,
                ap as *mut _,
                tp as *mut _,
                stride as u64,
                side as u32,
                n as u32,
                max_area as u32,
                hole,
                sprinkle,
                self.stream_ptr(),
            )
        })
    }

    /// `plane[off..off + n]` set to `val`, or clamped to at most `val` when
    /// `clamp`. Slot 795.
    pub fn sam3_mask_set(
        &self,
        plane: &mut CudaSlice<f32>,
        off: usize,
        n: usize,
        val: f32,
        clamp: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_set
            .ok_or(GpuError::MissingOp("sam3_mask_set"))?;
        if plane.len() < off + n {
            return Err(oob("sam3_mask_set: buffer under the plane"));
        }
        let (pp, _g1) = plane.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 795); bounds checked above
        check(unsafe {
            f(
                at::<f32>(pp, off) as *mut _,
                n as u64,
                val,
                clamp as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Column `q` of the pixel-major `[side^2][nq]` logits at `logits[off..]`
    /// bilinear to `h x w`, u8 0/1 COLUMN-major into `out[out_off..]`:
    /// logit > 0 when `video` (Meta's video outputs), else sigmoid > 0.5.
    /// Slot 796.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_mask_up2(
        &self,
        logits: &CudaSlice<f32>,
        off: usize,
        (side, nq, q): (usize, usize, usize),
        out: &mut CudaSlice<u8>,
        out_off: usize,
        h: usize,
        w: usize,
        video: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_up2
            .ok_or(GpuError::MissingOp("sam3_mask_up2"))?;
        if q >= nq || logits.len() < off + side * side * nq || out.len() < out_off + h * w {
            return Err(oob("sam3_mask_up2: buffers under the mask geometry"));
        }
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 796); bounds checked above
        check(unsafe {
            f(
                at::<f32>(lp, off) as *const _,
                at::<u8>(op, out_off) as *mut _,
                side as u32,
                nq as u32,
                q as u32,
                h as u32,
                w as u32,
                video as u32,
                self.stream_ptr(),
            )
        })
    }

    /// One owner a pixel over `n` u8 masks of `px` pixels: the highest of
    /// `scores` among the masks covering it keeps it (the first on a tie),
    /// when that score is positive. Slot 797.
    pub fn sam3_mask_owner(
        &self,
        masks: &mut CudaSlice<u8>,
        scores: &CudaSlice<f32>,
        n: usize,
        px: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_owner
            .ok_or(GpuError::MissingOp("sam3_mask_owner"))?;
        if masks.len() < n * px || scores.len() < n {
            return Err(oob("sam3_mask_owner: buffers under the masks"));
        }
        let (mp, _g1) = masks.device_ptr_mut(&self.stream);
        let (sp, _g2) = scores.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 797); bounds checked above
        check(unsafe {
            f(
                mp as *mut _,
                sp as *const _,
                n as u32,
                px as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Each of `n` column-major `[w][h]` u8 masks' `{x0, y0, x1, y1, area}`
    /// into `out[n][5]`, extremes inclusive; an empty mask reads
    /// `{w, h, 0, 0, 0}`. Slot 798.
    pub fn sam3_mask_boxes(
        &self,
        masks: &CudaSlice<u8>,
        n: usize,
        h: usize,
        w: usize,
        out: &mut CudaSlice<u32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_boxes
            .ok_or(GpuError::MissingOp("sam3_mask_boxes"))?;
        if masks.len() < n * h * w || out.len() < n * 5 {
            return Err(oob("sam3_mask_boxes: buffers under the masks"));
        }
        let (mp, _g1) = masks.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 798); bounds checked above
        check(unsafe {
            f(
                mp as *const _,
                op as *mut _,
                n as u32,
                h as u32,
                w as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Pillow's bilinear taps for one axis, `in_size -> out_size`: `bounds`
    /// `[out][2]` (window start, taps), `kk` `[out][ksize]` 22-bit weights.
    /// `ksize` is [`pil_ksize`]'s. Slot 799.
    pub fn sam3_pil_coeffs(
        &self,
        in_size: usize,
        out_size: usize,
        bounds: &mut CudaSlice<i32>,
        kk: &mut CudaSlice<i32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_pil_coeffs
            .ok_or(GpuError::MissingOp("sam3_pil_coeffs"))?;
        let ksize = pil_ksize(in_size, out_size);
        if in_size == 0 || ksize > 64 || bounds.len() < 2 * out_size || kk.len() < out_size * ksize
        {
            return Err(oob("sam3_pil_coeffs: tables under the axis"));
        }
        let (bp, _g1) = bounds.device_ptr_mut(&self.stream);
        let (kp, _g2) = kk.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 799); bounds checked above
        check(unsafe {
            f(
                bp as *mut _,
                kp as *mut _,
                in_size as u32,
                out_size as u32,
                ksize as u32,
                self.stream_ptr(),
            )
        })
    }

    /// One Pillow pass over a u8 HWC picture with `ch` channels: `axis` 0
    /// takes `src [rows][in][ch]` to `dst [rows][out][ch]`, 1 takes
    /// `src [in][rows][ch]` to `dst [out][rows][ch]`, with the taps of
    /// [`Self::sam3_pil_coeffs`] for `in -> out`. Slot 800.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_pil_pass(
        &self,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<u8>,
        (bounds, kk): (&CudaSlice<i32>, &CudaSlice<i32>),
        rows: usize,
        (in_size, out_size): (usize, usize),
        ch: usize,
        axis: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_pil_pass
            .ok_or(GpuError::MissingOp("sam3_pil_pass"))?;
        let ksize = pil_ksize(in_size, out_size);
        if axis > 1
            || src.len() < rows * in_size * ch
            || dst.len() < rows * out_size * ch
            || bounds.len() < 2 * out_size
            || kk.len() < out_size * ksize
        {
            return Err(oob("sam3_pil_pass: buffers under the picture"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        let (bp, _g3) = bounds.device_ptr(&self.stream);
        let (kp, _g4) = kk.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 800); bounds checked above, and the taps
        // stay inside `in_size` by construction (799 clamps the windows)
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                bp as *const _,
                kp as *const _,
                rows as u32,
                in_size as u32,
                out_size as u32,
                ch as u32,
                ksize as u32,
                axis as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 751's patch stem with a normalization: `video` false is the
    /// picture processor's (751's own), true Meta's video frames' (fp16
    /// storage, fp16 normalize). Slot 801.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_patch_rows_norm(
        &self,
        pixels: &CudaSlice<u8>,
        out: &mut CudaSlice<f16>,
        pics: usize,
        side: usize,
        patch: usize,
        win: usize,
        ch: usize,
        kp: usize,
        video: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_patch_rows_norm
            .ok_or(GpuError::MissingOp("sam3_patch_rows_norm"))?;
        if patch == 0 || !side.is_multiple_of(patch) || kp < ch * patch * patch {
            return Err(oob("sam3_patch_rows_norm: geometry"));
        }
        let g = side / patch;
        if win == 0 || !g.is_multiple_of(win) {
            return Err(oob(
                "sam3_patch_rows_norm: the grid is not a whole number of windows",
            ));
        }
        if pixels.len() < pics * side * side * ch || out.len() < pics * g * g * kp {
            return Err(oob(
                "sam3_patch_rows_norm: buffers under the picture geometry",
            ));
        }
        let (pp, _g1) = pixels.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 801); bounds checked above
        check(unsafe {
            f(
                pp as *const _,
                op as *mut _,
                pics as u32,
                side as u32,
                patch as u32,
                win as u32,
                ch as u32,
                kp as u32,
                video as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Column `q` of the pixel-major `[px][nq]` plane at `src[src_off..]`
    /// into `dst[dst_off..dst_off + px]`. Slot 802.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_mask_pick(
        &self,
        src: &CudaSlice<f32>,
        src_off: usize,
        (px, nq, q): (usize, usize, usize),
        dst: &mut CudaSlice<f32>,
        dst_off: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_pick
            .ok_or(GpuError::MissingOp("sam3_mask_pick"))?;
        if q >= nq || src.len() < src_off + px * nq || dst.len() < dst_off + px {
            return Err(oob("sam3_mask_pick: buffers under the planes"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 802); bounds checked above
        check(unsafe {
            f(
                at::<f32>(sp, src_off) as *const _,
                at::<f32>(dp, dst_off) as *mut _,
                px as u32,
                nq as u32,
                q as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `n` f32 planes `ih x iw` (from `src[src_off..]`, `istride` apart)
    /// resized to `oh x ow` (into `dst[dst_off..]`, `ostride` apart) as
    /// torch's bilinear interpolation does, then optionally binarized:
    /// `bin = Some((thr, lo, hi))` writes `v > thr ? hi : lo`. Slot 803.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_resize_f32(
        &self,
        (src, src_off, istride): (&CudaSlice<f32>, usize, usize),
        (dst, dst_off, ostride): (&mut CudaSlice<f32>, usize, usize),
        n: usize,
        (ih, iw): (usize, usize),
        (oh, ow): (usize, usize),
        filter: Sam3Resize,
        bin: Option<(f32, f32, f32)>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_resize_f32
            .ok_or(GpuError::MissingOp("sam3_resize_f32"))?;
        if n == 0 {
            return Ok(());
        }
        if istride < ih * iw
            || ostride < oh * ow
            || src.len() < src_off + (n - 1) * istride + ih * iw
            || dst.len() < dst_off + (n - 1) * ostride + oh * ow
        {
            return Err(oob("sam3_resize_f32: buffers under the planes"));
        }
        let (thr, lo, hi) = bin.unwrap_or((0.0, 0.0, 0.0));
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 803); bounds checked above
        check(unsafe {
            f(
                at::<f32>(sp, src_off) as *const _,
                at::<f32>(dp, dst_off) as *mut _,
                n as u32,
                ih as u32,
                iw as u32,
                oh as u32,
                ow as u32,
                istride as u64,
                ostride as u64,
                (filter == Sam3Resize::Antialias) as u32,
                bin.is_some() as u32,
                thr,
                lo,
                hi,
                self.stream_ptr(),
            )
        })
    }

    /// The tracker's `mask_downsample` (Conv2d(1, 1, k4, s4) + bias, `w`
    /// `[16]`, `b` `[1]`) over `n` planes of `4s x 4s` (from `src[src_off..]`,
    /// `istride` apart) into `s x s` (`dst[dst_off..]`, `ostride` apart).
    /// Slot 804.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_mask_down4(
        &self,
        (src, src_off, istride): (&CudaSlice<f32>, usize, usize),
        (dst, dst_off, ostride): (&mut CudaSlice<f32>, usize, usize),
        (w, b): (&CudaSlice<f32>, &CudaSlice<f32>),
        n: usize,
        s: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_down4
            .ok_or(GpuError::MissingOp("sam3_mask_down4"))?;
        if n == 0 {
            return Ok(());
        }
        let (ip, op) = (16 * s * s, s * s);
        if istride < ip
            || ostride < op
            || w.len() < 16
            || b.is_empty()
            || src.len() < src_off + (n - 1) * istride + ip
            || dst.len() < dst_off + (n - 1) * ostride + op
        {
            return Err(oob("sam3_mask_down4: buffers under the planes"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (bp, _g4) = b.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 804); bounds checked above
        check(unsafe {
            f(
                at::<f32>(sp, src_off) as *const _,
                at::<f32>(dp, dst_off) as *mut _,
                wp as *const _,
                bp as *const _,
                n as u32,
                s as u32,
                istride as u64,
                ostride as u64,
                self.stream_ptr(),
            )
        })
    }

    /// Meta's pixel-wise non-overlap across `n` planes of `px` values at
    /// `planes[off..]`, `stride` apart: `counts` (`[n][2]`) gets each plane's
    /// area before (> 0) and after (kept and > 0); `write` clamps every
    /// pixel a plane does not win to at most -10, in place. Slot 805.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_nonoverlap(
        &self,
        planes: &mut CudaSlice<f32>,
        off: usize,
        stride: usize,
        n: usize,
        px: usize,
        counts: Option<&mut CudaSlice<u32>>,
        write: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_nonoverlap
            .ok_or(GpuError::MissingOp("sam3_nonoverlap"))?;
        if n == 0 {
            return Ok(());
        }
        if planes.len() < off + (n - 1) * stride + px
            || counts.as_ref().is_some_and(|c| c.len() < 2 * n)
        {
            return Err(oob("sam3_nonoverlap: buffers under the planes"));
        }
        let (pp, _g1) = planes.device_ptr_mut(&self.stream);
        let cg = counts.map(|c| c.device_ptr_mut(&self.stream));
        let cp = cg.as_ref().map_or(0, |(p, _)| *p);
        // SAFETY: ABI contract (slot 805); bounds checked above, a null
        // counts pointer only counts nothing
        check(unsafe {
            f(
                at::<f32>(pp, off) as *mut _,
                cp as *mut _,
                stride as u64,
                n as u32,
                px as u32,
                write as u32,
                self.stream_ptr(),
            )
        })
    }
}

/// Pillow's tap count for one axis, `in -> out`: `ceil(support) * 2 + 1`,
/// the support the scale (at least 1).
pub fn pil_ksize(in_size: usize, out_size: usize) -> usize {
    let scale = in_size as f64 / out_size.max(1) as f64;
    scale.max(1.0).ceil() as usize * 2 + 1
}
