//! Executor ops for Clef's image lane (slots 735-740,
//! `packs/cuda/src/clef_vision.cuh`): the image processor's uint8 resize and
//! patchify, and the vision tower's position embedding, rope and attention.
//! The tower's projections and norms are the backbone's GEMM (733) and the
//! head's LayerNorm (724).

use cudarc::driver::{CudaSlice, CudaViewMut, DevicePtr, DevicePtrMut};

use super::error::*;
use super::*;

/// Device memory for one resize axis's plan, held by the caller for good:
/// `idx` is xmin `[output]` | xsize `[output]` | the weight precision (u32),
/// `w` the int16 weights `[output][taps]` as little-endian byte pairs. A
/// resident plan, not one allocated per picture: a stream-ordered free and
/// a zeroing memset between programmatic-dependent launches can hand a plan
/// back to the pool while a kernel that already released its dependents is
/// still reading it.
pub struct ClefPlanMem {
    pub idx: CudaSlice<u32>,
    pub w: CudaSlice<u8>,
}

impl ClefPlanMem {
    /// Room for any axis up to `max_out` outputs and `max_in` inputs: the
    /// taps a row are at most 4 x the downscale + 5, so output x taps is
    /// under 4 * max_in + 5 * max_out.
    pub fn bytes(max_in: usize, max_out: usize) -> usize {
        4 * (2 * max_out + 1) + 2 * (4 * max_in + 5 * max_out)
    }
}

/// One planned axis of the resize (slot 735), in a [`ClefPlanMem`].
#[derive(Clone, Copy, Debug)]
pub struct ClefResamplePlan {
    pub taps: usize,
    pub input: usize,
    pub output: usize,
}

/// torch's `max_interp_size` for an antialiased bicubic axis from `input` to
/// `output`: the cubic's support (2, stretched by the downscale ratio)
/// rounded up, both sides, plus the center.
pub fn clef_resample_taps(input: usize, output: usize) -> usize {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    support.ceil() as usize * 2 + 1
}

impl GpuExecutor {
    /// Whether the pack carries Clef's image lane (735-740) beside the
    /// backbone's (see [`Self::has_clef`]).
    pub fn has_clef_vision(&self) -> bool {
        let k = &self.kernels;
        self.has_clef()
            && k.clef_resample_plan.is_some()
            && k.clef_resample.is_some()
            && k.clef_patchify.is_some()
            && k.clef_vpos.is_some()
            && k.clef_vrope.is_some()
            && k.clef_vattn.is_some()
    }

    /// Plan one resize axis from `input` to `output` samples into `mem`
    /// (slot 735).
    pub fn clef_resample_plan(
        &self,
        input: usize,
        output: usize,
        mem: &mut ClefPlanMem,
    ) -> Result<ClefResamplePlan, GpuError> {
        let f = self
            .kernels
            .clef_resample_plan
            .ok_or(GpuError::MissingOp("clef_resample_plan"))?;
        if input == 0 || output == 0 || input > u32::MAX as usize || output > u32::MAX as usize {
            return Err(oob("clef_resample_plan: an empty or oversized axis"));
        }
        let taps = clef_resample_taps(input, output);
        if mem.idx.len() < 2 * output + 1 || mem.w.len() < 2 * output * taps {
            return Err(oob("clef_resample_plan: the axis outgrows its plan memory"));
        }
        let (ip, _g1) = mem.idx.device_ptr_mut(&self.stream);
        let (wp, _g2) = mem.w.device_ptr_mut(&self.stream);
        let o4 = (output * 4) as u64;
        // SAFETY: ABI contract (slot 735); the plan memory holds `output`
        // indices of `taps` weights (checked above)
        check(unsafe {
            f(
                input as u32,
                output as u32,
                taps as u32,
                ip as *mut _,
                (ip + o4) as *mut _,
                wp as *mut _,
                (ip + 2 * o4) as *mut _,
                self.stream_ptr(),
            )
        })?;
        Ok(ClefResamplePlan {
            taps,
            input,
            output,
        })
    }

    /// One separable pass of the resize over interleaved RGB8 (slot 736):
    /// along the rows (`horiz`: `src [lines][plan.input][3]` ->
    /// `dst [lines][plan.output][3]`) or the columns (`src [plan.input]
    /// [lines][3]` -> `dst [plan.output][lines][3]`).
    pub fn clef_resample(
        &self,
        src: &impl DevicePtr<u8>,
        dst: &mut impl DevicePtrMut<u8>,
        lines: usize,
        horiz: bool,
        plan: &ClefResamplePlan,
        mem: &ClefPlanMem,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_resample
            .ok_or(GpuError::MissingOp("clef_resample"))?;
        if src.len() < lines * plan.input * 3
            || dst.len() < lines * plan.output * 3
            || lines > u32::MAX as usize
            || mem.idx.len() < 2 * plan.output + 1
            || mem.w.len() < 2 * plan.output * plan.taps
        {
            return Err(oob("clef_resample: buffers under the image"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        let (ip, _g3) = mem.idx.device_ptr(&self.stream);
        let (wp, _g4) = mem.w.device_ptr(&self.stream);
        let o4 = (plan.output * 4) as u64;
        let (xp, np, pp) = (ip, ip + o4, ip + 2 * o4);
        // SAFETY: ABI contract (slot 736); bounds checked above, every tap
        // the plan names lies inside the input axis by construction
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                lines as u32,
                plan.input as u32,
                plan.output as u32,
                u32::from(horiz),
                xp as *const _,
                np as *const _,
                wp as *const _,
                plan.taps as u32,
                pp as *const _,
                self.stream_ptr(),
            )
        })
    }

    /// Host bytes into the front of a device byte range (a view of a
    /// resident plane the caller borrows as staging).
    pub fn upload_u8_into(
        &self,
        host: &[u8],
        dst: &mut CudaViewMut<'_, u8>,
    ) -> Result<(), GpuError> {
        if dst.len() < host.len() {
            return Err(oob("upload_u8_into: the staging range is under the data"));
        }
        let mut dst = dst
            .try_slice_mut(0..host.len())
            .ok_or_else(|| oob("upload_u8_into: the staging range"))?;
        self.stream.memcpy_htod(host, &mut dst).map_err(drv)
    }

    /// The processor's pixel rows of one resized image (slot 737):
    /// `img [h][w][3]` -> rows `row_off ..` of `out` (`[patches][1536]`,
    /// 2 x 2 merge-window order).
    pub fn clef_patchify(
        &self,
        img: &impl DevicePtr<u8>,
        h: usize,
        w: usize,
        out: &mut CudaSlice<f32>,
        row_off: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_patchify
            .ok_or(GpuError::MissingOp("clef_patchify"))?;
        let patches = (h / 16) * (w / 16);
        if !h.is_multiple_of(32)
            || !w.is_multiple_of(32)
            || img.len() < h * w * 3
            || out.len() < (row_off + patches) * 1536
        {
            return Err(oob("clef_patchify: buffers under the image"));
        }
        let (ip, _g1) = img.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 737); bounds checked above
        check(unsafe {
            f(
                ip as *const _,
                h as u32,
                w as u32,
                (op + (row_off * 1536 * 4) as u64) as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// `x[r] +=` the learned position grid (BF16 `[side * side][d]`) read
    /// bilinearly at row r's patch (slot 738); `info` is `[rows][4]` =
    /// (patch row, patch column, grid rows, grid columns).
    #[allow(clippy::too_many_arguments)]
    pub fn clef_vpos(
        &self,
        x: &mut CudaSlice<f32>,
        table: &CudaSlice<u8>,
        info: &CudaSlice<u32>,
        rows: usize,
        side: usize,
        d: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_vpos
            .ok_or(GpuError::MissingOp("clef_vpos"))?;
        if x.len() < rows * d || table.len() < side * side * d * 2 || info.len() < 4 * rows {
            return Err(oob("clef_vpos: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (tp, _g2) = table.device_ptr(&self.stream);
        let (ip, _g3) = info.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 738); bounds checked above, grid points
        // inside [0, side - 1] by construction
        check(unsafe {
            f(
                xp as *mut _,
                tp as *const _,
                ip as *const _,
                rows as u32,
                side as u32,
                d as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The 2D vision rope on q and k of `rows` fused qkv rows (`[rows][3]
    /// [heads][72]`) from the `(cos, sin)` table `[max_pos][18]` (slot
    /// 739); `info` as [`Self::clef_vpos`]'s.
    pub fn clef_vrope(
        &self,
        qkv: &mut CudaSlice<f32>,
        heads: usize,
        rows: usize,
        info: &CudaSlice<u32>,
        table: &CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_vrope
            .ok_or(GpuError::MissingOp("clef_vrope"))?;
        let max_pos = table.len() / 36;
        if qkv.len() < rows * 3 * heads * 72 || info.len() < 4 * rows || max_pos == 0 {
            return Err(oob("clef_vrope: buffers under the geometry"));
        }
        let (xp, _g1) = qkv.device_ptr_mut(&self.stream);
        let (ip, _g2) = info.device_ptr(&self.stream);
        let (tp, _g3) = table.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 739); bounds checked above, positions
        // clamped to the table in the kernel
        check(unsafe {
            f(
                xp as *mut _,
                heads as u32,
                rows as u32,
                ip as *const _,
                tp as *const _,
                max_pos as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Bidirectional attention within each image (slot 740): image i is
    /// rows `cu[i] .. cu[i + 1]` of the fused qkv rows; the result
    /// `[rows][heads * 72]` into `out`. `max_len` bounds every image's rows.
    #[allow(clippy::too_many_arguments)]
    pub fn clef_vattn(
        &self,
        qkv: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        cu: &CudaSlice<u32>,
        segs: usize,
        rows: usize,
        max_len: usize,
        heads: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_vattn
            .ok_or(GpuError::MissingOp("clef_vattn"))?;
        if qkv.len() < rows * 3 * heads * 72
            || out.len() < rows * heads * 72
            || cu.len() < segs + 1
            || segs > 65535
        {
            return Err(oob("clef_vattn: buffers under the geometry"));
        }
        let (qp, _g1) = qkv.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        let (cp, _g3) = cu.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 740); bounds checked above, the
        // segments cover rows 0 .. rows by the caller's construction
        check(unsafe {
            f(
                qp as *const _,
                op as *mut _,
                cp as *const _,
                segs as u32,
                max_len as u32,
                heads as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }
}
