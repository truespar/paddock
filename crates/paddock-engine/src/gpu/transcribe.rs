//! Nemotron 3.5 Transcribe's speech encoder glue (asr/transcribe.cuh, slots
//! 942-944): the per-feature mel normalization into the stacked f16 encoder
//! input, the per-head q/k LayerNorm with the half-dim rope, and the speaker
//! fusion. The blocks around them ride the tower kernels; the model side is
//! `gpu_model/nemotron/audio.rs`. Slots 945-946 are its MTP block's FP8
//! expert GEMVs (`gpu_model/nemotron/mtp.rs`); 958-960 the CTC timestamp
//! head; 961-963 the encoder's W8A8 class (e4m3 GEMM landings, LayerNorm and
//! f16 planes to e4m3 at a static scale).

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::check;
use super::{F8RowPlane, GpuError, GpuExecutor};

fn need(ok: bool, what: &str) -> Result<(), GpuError> {
    if ok {
        Ok(())
    } else {
        Err(GpuError::Unsupported(format!("transcribe: {what}")))
    }
}

impl GpuExecutor {
    /// Slot 949: the speaker activity codes on the device (the encode's
    /// readback-free speaker branch).
    pub fn has_tr_codes(&self) -> bool {
        self.kernels.tr_codes.is_some()
    }

    /// `codes [rows]` = each encoder row's speaker activity pattern from the
    /// diarizer's probabilities `[rows * 8][8]` starting at row `row0` of
    /// `probs` (a packed pass's window), rows from `valid` on 0 (slot 949).
    pub fn tr_codes(
        &self,
        probs: &CudaSlice<f32>,
        row0: usize,
        codes: &mut CudaSlice<u32>,
        rows: usize,
        valid: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .tr_codes
            .ok_or(GpuError::MissingOp("tr_codes"))?;
        need(
            valid <= rows && probs.len() >= (row0 + rows) * 64 && codes.len() >= rows,
            "codes buffers under the geometry",
        )?;
        let (pp, _g1) = probs.device_ptr(&self.stream);
        let (cp, _g2) = codes.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 949); bounds checked above
        check(unsafe {
            f(
                (pp + (row0 * 64 * 4) as u64) as *const _,
                cp as *mut _,
                rows as u32,
                valid as u32,
                self.stream_ptr(),
            )
        })
    }

    /// All three slots are in the pack.
    pub fn has_transcribe_audio(&self) -> bool {
        self.kernels.tr_mel_norm.is_some()
            && self.kernels.tr_qk_norm_rope.is_some()
            && self.kernels.tr_fuse.is_some()
    }

    /// `out [frames][128]` f16 = the first `valid` frames of `mel` normalized
    /// per bin over those frames, zero past them (slot 942).
    pub fn tr_mel_norm(
        &self,
        mel: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        valid: usize,
        frames: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .tr_mel_norm
            .ok_or(GpuError::MissingOp("tr_mel_norm"))?;
        need(
            valid <= frames && mel.len() >= valid * 128 && out.len() >= frames * 128,
            "mel_norm buffers under the geometry",
        )?;
        let (mp, _g1) = mel.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 942); bounds checked above
        check(unsafe {
            f(
                mp as *const _,
                op as *mut _,
                valid as u32,
                frames as u32,
                self.stream_ptr(),
            )
        })
    }

    /// In place over `qkv [rows][3][heads][128]` f16: LayerNorm on each q and
    /// k head, then the rope on its first 64 dims at `pos[row]` from `rope`
    /// (`[positions][32][cos, sin]` f32) (slot 943).
    #[allow(clippy::too_many_arguments)]
    pub fn tr_qk_norm_rope(
        &self,
        qkv: &mut CudaSlice<f16>,
        (wq, bq): (&CudaSlice<f32>, &CudaSlice<f32>),
        (wk, bk): (&CudaSlice<f32>, &CudaSlice<f32>),
        pos: &CudaSlice<u32>,
        rope: &CudaSlice<f32>,
        rows: usize,
        heads: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .tr_qk_norm_rope
            .ok_or(GpuError::MissingOp("tr_qk_norm_rope"))?;
        need(
            qkv.len() >= rows * 3 * heads * 128
                && [wq, bq, wk, bk].iter().all(|v| v.len() >= 128)
                && pos.len() >= rows,
            "qk_norm_rope buffers under the geometry",
        )?;
        let (qp, _g1) = qkv.device_ptr_mut(&self.stream);
        let (wqp, _g2) = wq.device_ptr(&self.stream);
        let (bqp, _g3) = bq.device_ptr(&self.stream);
        let (wkp, _g4) = wk.device_ptr(&self.stream);
        let (bkp, _g5) = bk.device_ptr(&self.stream);
        let (pp, _g6) = pos.device_ptr(&self.stream);
        let (rp, _g7) = rope.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 943); bounds checked above, the caller
        // keeps every position inside the rope table
        check(unsafe {
            f(
                qp as *mut _,
                wqp as *const _,
                bqp as *const _,
                wkp as *const _,
                bkp as *const _,
                pp as *const _,
                rp as *const _,
                rows as u32,
                heads as u32,
                128,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// `out [rows][n]` f16 = `LN(x) * w + b + table[codes[row]]` (slot 944):
    /// `table` holds 256 rows of `n`.
    #[allow(clippy::too_many_arguments)]
    pub fn tr_fuse(
        &self,
        x: &CudaSlice<f32>,
        (w, b): (&CudaSlice<f32>, &CudaSlice<f32>),
        codes: &CudaSlice<u32>,
        table: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self.kernels.tr_fuse.ok_or(GpuError::MissingOp("tr_fuse"))?;
        need(
            x.len() >= rows * n
                && w.len() >= n
                && b.len() >= n
                && codes.len() >= rows
                && table.len() >= 256 * n
                && out.len() >= rows * n,
            "fuse buffers under the geometry",
        )?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (cp, _g4) = codes.device_ptr(&self.stream);
        let (tp, _g5) = table.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 944); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                bp as *const _,
                cp as *const _,
                tp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Both FP8 expert GEMVs are in the pack.
    pub fn has_f8r_moe_gemv(&self) -> bool {
        self.kernels.f8r_moe_up_relu2.is_some() && self.kernels.f8r_moe_down.is_some()
    }

    /// `out [rows][k][ff] = relu(scale * W[idx * ff + o] . x[r])^2` over an FP8
    /// expert up plane (`[n_expert * ff][in]`) for every row's `k` picks
    /// (slot 945). `out` is f32, `x` `[rows][in]` f32.
    #[allow(clippy::too_many_arguments)]
    pub fn f8r_moe_up_relu2(
        &self,
        up: &F8RowPlane,
        idx: &CudaSlice<u32>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        k: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8r_moe_up_relu2
            .ok_or(GpuError::MissingOp("f8r_moe_up_relu2"))?;
        need(
            in_dim.is_multiple_of(16)
                && up.data.len() >= ff * in_dim
                && up.data.len().is_multiple_of(ff * in_dim)
                && up.scale.len() * in_dim == up.data.len()
                && idx.len() >= rows * k
                && x.len() >= rows * in_dim
                && out.len() >= rows * k * ff,
            "f8r_moe_up_relu2 buffers under the geometry",
        )?;
        let (dp, _g1) = up.data.device_ptr(&self.stream);
        let (sp, _g2) = up.scale.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (xp, _g4) = x.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 945); bounds checked above, every idx
        // names an expert of the plane (the router's top-k)
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                ip as *const _,
                xp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                k as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `out [rows][hidden] (+)= sum_j w[r][j] * scale * W[idx * hidden + o] .
    /// act[r][j]` over an FP8 expert down plane (`[n_expert * hidden][ff]`),
    /// the picks summed in order; `acc` adds onto `out` (slot 946).
    #[allow(clippy::too_many_arguments)]
    pub fn f8r_moe_down(
        &self,
        down: &F8RowPlane,
        idx: &CudaSlice<u32>,
        w: &CudaSlice<f32>,
        act: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        ff: usize,
        hidden: usize,
        k: usize,
        rows: usize,
        acc: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8r_moe_down
            .ok_or(GpuError::MissingOp("f8r_moe_down"))?;
        need(
            ff.is_multiple_of(16)
                && down.data.len() >= hidden * ff
                && down.scale.len() * ff == down.data.len()
                && idx.len() >= rows * k
                && w.len() >= rows * k
                && act.len() >= rows * k * ff
                && out.len() >= rows * hidden,
            "f8r_moe_down buffers under the geometry",
        )?;
        let (dp, _g1) = down.data.device_ptr(&self.stream);
        let (sp, _g2) = down.scale.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (wp, _g4) = w.device_ptr(&self.stream);
        let (ap, _g5) = act.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 946); bounds checked above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                ip as *const _,
                wp as *const _,
                ap as *const _,
                op as *mut _,
                ff as u32,
                hidden as u32,
                k as u32,
                rows as u32,
                u32::from(acc),
                self.stream_ptr(),
            )
        })
    }
}

impl GpuExecutor {
    /// Slots 951-953 and the BM-parameterized align they read.
    pub fn has_f8r_moe_sorted(&self) -> bool {
        self.kernels.f8r_moe_up_relu2_sorted.is_some()
            && self.kernels.f8r_moe_down_sorted.is_some()
            && self.kernels.f8r_moe_combine.is_some()
            && self.kernels.moe_align_bm.is_some()
    }

    /// The FP8 experts of [`Self::f8r_moe_up_relu2`] + [`Self::f8r_moe_down`]
    /// (their bits) over expert-sorted blocks of 8 (slots 951-953): `rows`
    /// rows of `x [rows][in]` with `k` picks each (`idx`, weights `w`), the
    /// pick sum into `out [rows][hidden]` (`acc` adds onto it). Rows run in
    /// chunks of `chunk`: `act` holds `chunk * k` up rows, `part` as many
    /// down rows, and the align planes one chunk's blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn f8r_moe_sorted(
        &self,
        (up, down): (&F8RowPlane, &F8RowPlane),
        idx: &CudaSlice<u32>,
        w: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        act: &mut CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        (srow, sslot, bexp): (
            &mut CudaSlice<u32>,
            &mut CudaSlice<u32>,
            &mut CudaSlice<u32>,
        ),
        (in_dim, ff, hidden, k, n_expert): (usize, usize, usize, usize, usize),
        rows: usize,
        chunk: usize,
        acc: bool,
    ) -> Result<(), GpuError> {
        let (Some(fa), Some(fu), Some(fd), Some(fc)) = (
            self.kernels.moe_align_bm,
            self.kernels.f8r_moe_up_relu2_sorted,
            self.kernels.f8r_moe_down_sorted,
            self.kernels.f8r_moe_combine,
        ) else {
            return Err(GpuError::MissingOp("f8r_moe_sorted"));
        };
        let bm = 8usize;
        let nb_max = chunk * k / bm + n_expert.min(chunk * k) + 1;
        need(
            in_dim.is_multiple_of(16)
                && ff.is_multiple_of(16)
                && up.scale.len() * in_dim == up.data.len()
                && down.scale.len() * ff == down.data.len()
                && up.data.len() >= n_expert * ff * in_dim
                && down.data.len() >= n_expert * hidden * ff
                && idx.len() >= rows * k
                && w.len() >= rows * k
                && x.len() >= rows * in_dim
                && act.len() >= chunk * k * ff
                && part.len() >= chunk * k * hidden
                && out.len() >= rows * hidden
                && bexp.len() >= nb_max
                && srow.len() >= nb_max * bm
                && sslot.len() >= nb_max * bm,
            "f8r_moe_sorted buffers under the geometry",
        )?;
        let (ud, _g1) = up.data.device_ptr(&self.stream);
        let (us, _g2) = up.scale.device_ptr(&self.stream);
        let (dd, _g3) = down.data.device_ptr(&self.stream);
        let (ds, _g4) = down.scale.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (wp, _g6) = w.device_ptr(&self.stream);
        let (xp, _g7) = x.device_ptr(&self.stream);
        let (ap, _g8) = act.device_ptr_mut(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        let (op, _g10) = out.device_ptr_mut(&self.stream);
        let (rp, _g11) = srow.device_ptr_mut(&self.stream);
        let (sp, _g12) = sslot.device_ptr_mut(&self.stream);
        let (bp, _g13) = bexp.device_ptr_mut(&self.stream);
        let mut r0 = 0usize;
        while r0 < rows {
            let n = chunk.min(rows - r0);
            let nb = n * k / bm + n_expert.min(n * k) + 1;
            // SAFETY: ABI contracts (the align, slots 951-953); bounds checked
            // above, every pointer offset stays inside its plane (r0 + n <=
            // rows), the routing picks name experts of the planes
            unsafe {
                check(fa(
                    (ip + (r0 * k * 4) as u64) as *const _,
                    rp as *mut _,
                    sp as *mut _,
                    bp as *mut _,
                    n as u32,
                    k as u32,
                    n_expert as u32,
                    bm as u32,
                    nb as u32,
                    self.stream_ptr(),
                ))?;
                check(fu(
                    ud as *const _,
                    us as *const _,
                    rp as *const _,
                    sp as *const _,
                    bp as *const _,
                    (xp + (r0 * in_dim * 4) as u64) as *const _,
                    ap as *mut _,
                    in_dim as u32,
                    ff as u32,
                    k as u32,
                    nb as u32,
                    self.stream_ptr(),
                ))?;
                check(fd(
                    dd as *const _,
                    ds as *const _,
                    rp as *const _,
                    sp as *const _,
                    bp as *const _,
                    ap as *const _,
                    pp as *mut _,
                    ff as u32,
                    hidden as u32,
                    k as u32,
                    nb as u32,
                    self.stream_ptr(),
                ))?;
                check(fc(
                    pp as *const _,
                    (wp + (r0 * k * 4) as u64) as *const _,
                    (op + (r0 * hidden * 4) as u64) as *mut _,
                    hidden as u32,
                    k as u32,
                    n as u32,
                    u32::from(acc),
                    self.stream_ptr(),
                ))?;
            }
            r0 += n;
        }
        Ok(())
    }

    /// Slots 958-960: the CTC timestamp head's own pieces.
    pub fn has_tr_ctc(&self) -> bool {
        self.kernels.tr_ctc_qk_norm_rope.is_some()
            && self.kernels.tr_ctc_attention.is_some()
            && self.kernels.tr_ctc_lse_gather.is_some()
    }

    /// Slot 958: per-head LayerNorm on q and k of fused [q|k|v] rows, then
    /// rope on each head's first `rot` dims from the (cos, sin) table.
    #[allow(clippy::too_many_arguments)]
    pub fn tr_ctc_qk_norm_rope(
        &self,
        qkv: &mut CudaSlice<f32>,
        qn: (&CudaSlice<f32>, &CudaSlice<f32>),
        kn: (&CudaSlice<f32>, &CudaSlice<f32>),
        rope: &CudaSlice<f32>,
        rows: usize,
        heads: usize,
        hd: usize,
        rot: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .tr_ctc_qk_norm_rope
            .ok_or(GpuError::MissingOp("tr_ctc_qk_norm_rope"))?;
        debug_assert!(qkv.len() >= rows * 3 * heads * hd && rope.len() >= rows * rot);
        let (xp, _g1) = qkv.device_ptr_mut(&self.stream);
        let (a, _g2) = qn.0.device_ptr(&self.stream);
        let (b, _g3) = qn.1.device_ptr(&self.stream);
        let (c, _g4) = kn.0.device_ptr(&self.stream);
        let (d, _g5) = kn.1.device_ptr(&self.stream);
        let (rp, _g6) = rope.device_ptr(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                xp as *mut _,
                a as *const _,
                b as *const _,
                c as *const _,
                d as *const _,
                rp as *const _,
                rows as u32,
                heads as u32,
                hd as u32,
                rot as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 959: unmasked f32 attention over `rows` fused [q|k|v] rows.
    pub fn tr_ctc_attention(
        &self,
        qkv: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        heads: usize,
        hd: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .tr_ctc_attention
            .ok_or(GpuError::MissingOp("tr_ctc_attention"))?;
        debug_assert!(qkv.len() >= rows * 3 * heads * hd && out.len() >= rows * heads * hd);
        let (qp, _g1) = qkv.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                qp as *const _,
                op as *mut _,
                rows as u32,
                heads as u32,
                hd as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 960: each row's log-softmax of `logits + bias` read at `cols`.
    #[allow(clippy::too_many_arguments)]
    pub fn tr_ctc_lse_gather(
        &self,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        rows: usize,
        classes: usize,
        cols: &CudaSlice<u32>,
        n_cols: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .tr_ctc_lse_gather
            .ok_or(GpuError::MissingOp("tr_ctc_lse_gather"))?;
        debug_assert!(logits.len() >= rows * classes && out.len() >= rows * n_cols);
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (cp, _g3) = cols.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                lp as *const _,
                bp as *const _,
                rows as u32,
                classes as u32,
                cp as *const _,
                n_cols as u32,
                op as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Slots 961-963: the speech encoder in the checkpoint's W8A8 class
    /// (e4m3 mma, so cc 8.9 and up).
    pub fn has_tr_w8a8(&self) -> bool {
        self.kernels.tr_w8a8_gemm.is_some()
            && self.kernels.tr_ln_e4m3.is_some()
            && self.kernels.tr_quant_e4m3_h.is_some()
            && self.compute_capability() >= (8, 9)
    }

    /// Slot 961: `w` e4m3 `[out][in]` (scale `sw`) times `x` e4m3
    /// `[batch][in]` (scale `sx`), f32 sums landed as `sx * (sw * acc)` into
    /// `y` per its kind. `in_dim` a multiple of 16.
    #[allow(clippy::too_many_arguments)]
    pub fn tr_w8a8_gemm(
        &self,
        w: &CudaSlice<u8>,
        x: &CudaSlice<u8>,
        y: TrW8Out<'_>,
        sw: f32,
        sx: f32,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .tr_w8a8_gemm
            .ok_or(GpuError::MissingOp("tr_w8a8_gemm"))?;
        need(
            in_dim.is_multiple_of(16),
            "W8A8 GEMM in_dim must be a multiple of 16",
        )?;
        need(
            w.len() >= out_dim * in_dim && x.len() >= batch * in_dim,
            "W8A8 GEMM operands under the geometry",
        )?;
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let n = batch * out_dim;
        let (yp, bias, inv, epi, _gy, _gb) = match y {
            TrW8Out::H16(o) => {
                need(o.len() >= n, "W8A8 GEMM f16 landing under the geometry")?;
                let (p, g) = o.device_ptr_mut(&self.stream);
                (p, 0, 0.0, 0u32, g, None)
            }
            TrW8Out::BiasRes(o, b) => {
                need(
                    o.len() >= n && b.len() >= out_dim,
                    "W8A8 GEMM residual under the geometry",
                )?;
                let (p, g) = o.device_ptr_mut(&self.stream);
                let (bp, gb) = b.device_ptr(&self.stream);
                (p, bp, 0.0, 1, g, Some(gb))
            }
            TrW8Out::GeluQ(o, b, inv) => {
                need(
                    o.len() >= n && b.len() >= out_dim,
                    "W8A8 GEMM e4m3 landing under the geometry",
                )?;
                let (p, g) = o.device_ptr_mut(&self.stream);
                let (bp, gb) = b.device_ptr(&self.stream);
                (p, bp, inv, 2, g, Some(gb))
            }
        };
        // SAFETY: ABI contract (slot 961); sizes checked above
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                bias as *const _,
                sw,
                sx,
                inv,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                epi,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 962: a row LayerNorm of `x` `[rows][n]` landed as e4m3 at `inv`
    /// = 1 / the consumer's input scale.
    #[allow(clippy::too_many_arguments)]
    pub fn tr_ln_e4m3(
        &self,
        x: &CudaSlice<f32>,
        norm: (&CudaSlice<f32>, &CudaSlice<f32>),
        out: &mut CudaSlice<u8>,
        rows: usize,
        n: usize,
        eps: f32,
        inv: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .tr_ln_e4m3
            .ok_or(GpuError::MissingOp("tr_ln_e4m3"))?;
        need(
            x.len() >= rows * n && out.len() >= rows * n && norm.0.len() >= n && norm.1.len() >= n,
            "LayerNorm e4m3 planes under the geometry",
        )?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = norm.0.device_ptr(&self.stream);
        let (bp, _g3) = norm.1.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 962); sizes checked above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                inv,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 963: the first `n` elements of an f16 plane to e4m3 at `inv`.
    pub fn tr_quant_e4m3_h(
        &self,
        input: &CudaSlice<f16>,
        out: &mut CudaSlice<u8>,
        n: usize,
        inv: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .tr_quant_e4m3_h
            .ok_or(GpuError::MissingOp("tr_quant_e4m3_h"))?;
        need(
            n.is_multiple_of(8) && input.len() >= n && out.len() >= n,
            "f16 -> e4m3 planes under the geometry",
        )?;
        let (ip, _g1) = input.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 963); sizes checked above
        check(unsafe {
            f(
                ip as *const _,
                op as *mut _,
                n as u64,
                inv,
                self.stream_ptr(),
            )
        })
    }
}

/// Where [`GpuExecutor::tr_w8a8_gemm`] lands its sums.
pub enum TrW8Out<'a> {
    /// f16 `[batch][out]`
    H16(&'a mut CudaSlice<f16>),
    /// `y += v + bias` on an f32 `[batch][out]` residual stream
    BiasRes(&'a mut CudaSlice<f32>, &'a CudaSlice<f32>),
    /// `e4m3(gelu(v + bias) * inv)`, `[batch][out]` - the next linear's input
    GeluQ(&'a mut CudaSlice<u8>, &'a CudaSlice<f32>, f32),
}
