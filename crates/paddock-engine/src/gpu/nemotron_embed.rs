//! Nemotron 3.5 Embed / Rerank 8B ops (slots 895-901, the gated-SiLU landing
//! in `packs/cuda/src/gemm/f16_dense.cuh` and the family's glue in
//! `packs/cuda/src/nemotron_embed.cuh`). The graph is
//! `gpu_model::nemotron_embed`.
//!
//! Every buffer is checked against the geometry before a launch; what a
//! launch cannot be checked against (device-resident ids, positions and row
//! offsets) is the caller's to build in range, and each wrapper says so.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

/// Patch row width of the Pixtral stem: 3 x 14 x 14 = 588, padded to 592.
pub const NE_PATCH_K: usize = 592;
/// A tile descriptor is `(sequence << NE_TILE_SHIFT) | tile index`.
pub const NE_TILE_SHIFT: u32 = 12;
/// Query rows a block of [`GpuExecutor::ne_attn`] owns at `nh / nkv` query
/// heads a kv head: eight 16-row strips shared out over the group.
pub const fn ne_attn_rows(group: usize) -> usize {
    16 * 8 / group
}
/// [`GpuExecutor::ne_embed_rows`]: a source index with this bit set is a
/// picture row, else a token id.
pub const NE_EMB_PIC: u32 = 0x8000_0000;

impl GpuExecutor {
    /// Whether the pack carries the family's eight kernels.
    pub fn has_nemotron_embed(&self) -> bool {
        let k = &self.kernels;
        k.f16_gemm_h_swiglu.is_some()
            && k.ne_patch_u8.is_some()
            && k.ne_rms_h.is_some()
            && k.ne_embed_rows.is_some()
            && k.ne_rope_qk.is_some()
            && k.ne_attn.is_some()
            && k.ne_pool.is_some()
            && k.ne_add_rms_h.is_some()
    }

    /// Whether the pack carries torch's uint8 antialiased bicubic resize
    /// (Clef's slots 735/736), the picture processor's first step.
    pub fn has_clef_resample(&self) -> bool {
        self.kernels.clef_resample_plan.is_some() && self.kernels.clef_resample.is_some()
    }

    /// Slot 895: `y16 = silu(gate) * up` at f16 off a gate|up weight re-laid
    /// in 16-row blocks (gate rows 0-7, up rows 8-15 of each block): `w` is
    /// `[in, out]` with out = 2 x the FFN width, `y16` `[batch][out / 2]`.
    /// Ring-only, so a row's bits do not depend on `batch`.
    pub fn f16_gemm_h_swiglu(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y16: &mut CudaSlice<f16>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_swiglu
            .ok_or(GpuError::MissingOp("f16_gemm_h_swiglu"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if batch == 0 {
            return Ok(());
        }
        if !in_dim.is_multiple_of(8)
            || !out_dim.is_multiple_of(16)
            || w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y16.len() < batch * out_dim / 2
        {
            return Err(oob("f16_gemm_h_swiglu: buffers under the GEMM geometry"));
        }
        super::basic_ops::gemm_census("B-gemm-f16-h-swiglu", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (yp, _g3) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 895); bounds checked above
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 896: one picture's RGB8 canvas (`h` x `w`, both multiples of 14,
    /// `canvas_off` bytes into `canvas`) to its f16 patch rows
    /// (`[patches][592]`) at row `row_off` of `out`. `mean` / `std` are the
    /// processor's, already multiplied by 255.
    #[allow(clippy::too_many_arguments)]
    pub fn ne_patch_u8(
        &self,
        canvas: &CudaSlice<u8>,
        canvas_off: usize,
        out: &mut CudaSlice<f16>,
        row_off: usize,
        h: usize,
        w: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ne_patch_u8
            .ok_or(GpuError::MissingOp("ne_patch_u8"))?;
        if !h.is_multiple_of(14) || !w.is_multiple_of(14) {
            return Err(oob("ne_patch_u8: canvas sides must be multiples of 14"));
        }
        let n = (h / 14) * (w / 14);
        if canvas.len() < canvas_off + 3 * h * w || out.len() < (row_off + n) * NE_PATCH_K {
            return Err(oob("ne_patch_u8: buffers under the picture geometry"));
        }
        let (cp, _g1) = canvas.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 896); bounds checked above, offsets in bytes
        check(unsafe {
            f(
                (cp + canvas_off as u64) as *const _,
                (op + (row_off * NE_PATCH_K * 2) as u64) as *mut _,
                h as u32,
                w as u32,
                mean[0],
                mean[1],
                mean[2],
                std[0],
                std[1],
                std[2],
                self.stream_ptr(),
            )
        })
    }

    /// Slot 897: `out16[r] = f16(x[r] * rsqrt(mean(x[r]^2) + eps) * w)` over
    /// `rows` rows of `d` in {1024, 2048, 4096}.
    pub fn ne_rms_h(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out16: &mut CudaSlice<f16>,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ne_rms_h
            .ok_or(GpuError::MissingOp("ne_rms_h"))?;
        if !matches!(d, 1024 | 2048 | 4096)
            || x.len() < rows * d
            || out16.len() < rows * d
            || w.len() < d
        {
            return Err(oob("ne_rms_h: buffers under the row geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 897); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                op as *mut _,
                rows as u32,
                d as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 902: `x += proj` over `rows` f32 rows of `d`, then with `norm` =
    /// (w, out16) the RMSNorm of the updated rows into f16 (slot 897's
    /// arithmetic). d in {1024, 2048, 4096}.
    pub fn ne_add_rms_h(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        norm: Option<(&CudaSlice<f32>, &mut CudaSlice<f16>)>,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ne_add_rms_h
            .ok_or(GpuError::MissingOp("ne_add_rms_h"))?;
        if !matches!(d, 1024 | 2048 | 4096)
            || x.len() < rows * d
            || proj.len() < rows * d
            || norm
                .as_ref()
                .is_some_and(|(w, o)| w.len() < d || o.len() < rows * d)
        {
            return Err(oob("ne_add_rms_h: buffers under the row geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, op) = match norm {
            Some((w, o)) => (
                Some(w.device_ptr(&self.stream)),
                Some(o.device_ptr_mut(&self.stream)),
            ),
            None => (None, None),
        };
        // SAFETY: ABI contract (slot 902); bounds checked above, null w = the
        // add alone
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                op.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                rows as u32,
                d as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 898: `out[r]` = the bf16 embedding row `src[r]` of `emb` (bytes,
    /// `[vocab][d]`) widened, or - [`NE_EMB_PIC`] set - row `src[r] & !bit`
    /// of `pic` (`[*][d]` f32). The caller builds `src` in range: every id
    /// under `vocab`, every picture row inside `pic_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn ne_embed_rows(
        &self,
        src: &CudaSlice<u32>,
        emb: &CudaSlice<u8>,
        vocab: usize,
        pic: Option<(&CudaSlice<f32>, usize)>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        d: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ne_embed_rows
            .ok_or(GpuError::MissingOp("ne_embed_rows"))?;
        if !d.is_multiple_of(8)
            || src.len() < rows
            || emb.len() < vocab * d * 2
            || out.len() < rows * d
            || pic.is_some_and(|(p, n)| p.len() < n * d)
        {
            return Err(oob("ne_embed_rows: buffers under the row geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (ep, _g2) = emb.device_ptr(&self.stream);
        let pp = pic.map(|(p, _)| p.device_ptr(&self.stream));
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 898); bounds checked above, the indices
        // are the caller's (see the doc)
        check(unsafe {
            f(
                sp as *const _,
                ep as *const _,
                pp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                op as *mut _,
                rows as u32,
                d as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 899: rope in place on the first `nh + nkv` heads (q then k) of
    /// each row of the fused landing (`[rows][stride]` halves), from the
    /// `[table_rows][hd / 2]` tables at row `pos[r]`; q then times `qscale` x
    /// Llama 4's factor (`l4_beta` 0 = none). The caller builds every
    /// `pos[r] < table_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn ne_rope_qk(
        &self,
        qkv16: &mut CudaSlice<f16>,
        pos: &CudaSlice<u32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        table_rows: usize,
        rows: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        stride: usize,
        qscale: f32,
        l4: Option<(f32, usize)>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ne_rope_qk
            .ok_or(GpuError::MissingOp("ne_rope_qk"))?;
        let half = hd / 2;
        if !hd.is_multiple_of(4)
            || stride < (nh + nkv) * hd
            || qkv16.len() < rows * stride
            || pos.len() < rows
            || cos.len() < table_rows * half
            || sin.len() < table_rows * half
            || l4.is_some_and(|(_, o)| o == 0)
        {
            return Err(oob("ne_rope_qk: buffers under the row geometry"));
        }
        let (beta, orig) = l4.unwrap_or((0.0, 1));
        let (qp, _g1) = qkv16.device_ptr_mut(&self.stream);
        let (pp, _g2) = pos.device_ptr(&self.stream);
        let (cp, _g3) = cos.device_ptr(&self.stream);
        let (sp, _g4) = sin.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 899); bounds checked above, positions
        // are the caller's (see the doc)
        check(unsafe {
            f(
                qp as *mut _,
                pp as *const _,
                cp as *const _,
                sp as *const _,
                rows as u32,
                nh as u32,
                nkv as u32,
                hd as u32,
                stride as u32,
                qscale,
                beta,
                orig as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 900: bidirectional packed-varlen GQA attention off the fused
    /// landing (`[rows][(nh + 2 nkv) hd]`, q rotated and pre-scaled in the
    /// log2 domain) into `out16` `[rows][nh][hd]`. `cu` holds `n_seq + 1` row
    /// offsets, `tiles` one descriptor per [`ne_attn_rows`] query rows of
    /// each sequence. (hd, nh / nkv) in {(128, 4), (64, 1)}.
    #[allow(clippy::too_many_arguments)]
    pub fn ne_attn(
        &self,
        qkv16: &CudaSlice<f16>,
        cu: &CudaSlice<u32>,
        tiles: &CudaSlice<u32>,
        n_tiles: usize,
        out16: &mut CudaSlice<f16>,
        rows: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) -> Result<(), GpuError> {
        let f = self.kernels.ne_attn.ok_or(GpuError::MissingOp("ne_attn"))?;
        if nkv == 0
            || !nh.is_multiple_of(nkv)
            || !matches!((hd, nh / nkv), (128, 4) | (64, 1))
            || qkv16.len() < rows * (nh + 2 * nkv) * hd
            || out16.len() < rows * nh * hd
            || tiles.len() < n_tiles
        {
            return Err(oob("ne_attn: buffers under the attention geometry"));
        }
        let (qp, _g1) = qkv16.device_ptr(&self.stream);
        let (cp, _g2) = cu.device_ptr(&self.stream);
        let (tp, _g3) = tiles.device_ptr(&self.stream);
        let (op, _g4) = out16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 900); bounds checked above, cu / tiles
        // are the caller's packing of these rows
        check(unsafe {
            f(
                qp as *const _,
                cp as *const _,
                tp as *const _,
                n_tiles as u32,
                op as *mut _,
                nh as u32,
                nkv as u32,
                hd as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 901: per sequence (rows `cu[s]..cu[s + 1]` of `x`, `[rows][d]`
    /// f32) the f64 mean (into `mean_out` `[n_seq][d]` when given), then
    /// `out` = the first `dims` components unit-normed (`[n_seq][dims]`), or
    /// with `score` (`[d]` f32) `out[s]` = mean . score.
    #[allow(clippy::too_many_arguments)]
    pub fn ne_pool(
        &self,
        x: &CudaSlice<f32>,
        cu: &CudaSlice<u32>,
        score: Option<&CudaSlice<f32>>,
        mean_out: Option<&mut CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        n_seq: usize,
        rows: usize,
        d: usize,
        dims: usize,
    ) -> Result<(), GpuError> {
        let f = self.kernels.ne_pool.ok_or(GpuError::MissingOp("ne_pool"))?;
        let want = if score.is_some() { n_seq } else { n_seq * dims };
        if !d.is_multiple_of(128)
            || d > 4096
            || dims == 0
            || dims > d
            || x.len() < rows * d
            || cu.len() < n_seq + 1
            || score.as_ref().is_some_and(|s| s.len() < d)
            || mean_out.as_ref().is_some_and(|m| m.len() < n_seq * d)
            || out.len() < want
        {
            return Err(oob("ne_pool: buffers under the pool geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (cp, _g2) = cu.device_ptr(&self.stream);
        let sp = score.map(|s| s.device_ptr(&self.stream));
        let mp = mean_out.map(|m| m.device_ptr_mut(&self.stream));
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 901); bounds checked above, cu is the
        // caller's packing of these rows
        check(unsafe {
            f(
                xp as *const _,
                cp as *const _,
                sp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                mp.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                op as *mut _,
                n_seq as u32,
                d as u32,
                dims as u32,
                self.stream_ptr(),
            )
        })
    }
}
