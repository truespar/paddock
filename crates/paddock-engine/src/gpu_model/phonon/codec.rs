//! The streaming Mimi decoder pass: a batch of latents to 1920 samples a
//! frame. Every convolution reads its input rows and the raw rows the
//! previous pass left in its history, and leaves its own, so a stream decoded
//! in batches of any size gives the full-sequence causal result. Each runs on
//! the path elected for its layer (`ConvPath`): the 3xTF32 GEMM (698) over an
//! im2col plane, the same GEMM reading the stacked rows in place (917), or
//! the direct narrow conv (916).

use super::{CODEC_ROWS, Conv, ConvPath, D, GpuPhonon};
use crate::gpu::{GpuError, GpuExecutor, KumoEpi};
use cudarc::driver::CudaSlice;

/// One convolution over `rows` input rows of `x` (ELU on its input when
/// `elu`) into `y` - `[rows * stride][cout]`, accumulated onto y's contents
/// when `resid` - on its elected path, then its history update.
#[allow(clippy::too_many_arguments)]
fn conv(
    exec: &GpuExecutor,
    c: &mut Conv,
    x: &CudaSlice<f32>,
    stage: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
    elu: bool,
    resid: bool,
) -> Result<(), GpuError> {
    let n = c.stride * c.cout;
    match c.path {
        ConvPath::Im2col => {
            exec.ph_im2col(x, &c.hist, stage, 0, rows, c.cin, c.k, elu)?;
            let epi = if resid {
                KumoEpi::Resid
            } else {
                KumoEpi::Store
            };
            exec.kumo_gemm(
                (stage, 0),
                (&c.w, 0),
                Some((&c.b, 0)),
                y,
                None,
                (c.k * c.cin, n, rows),
                epi,
                1,
                (0, 0, 0),
            )?;
        }
        ConvPath::Rows => {
            // [act(hist) ; act(x)] once: the GEMM reads its taps in place
            let h = c.k - 1;
            if h > 0 {
                exec.ph_im2col(&c.hist, &c.hist, stage, 0, h, c.cin, 1, elu)?;
            }
            exec.ph_im2col(x, &c.hist, stage, h * c.cin, rows, c.cin, 1, elu)?;
            exec.ph_conv_gemm(stage, &c.w, &c.b, y, (rows, c.cin, n, c.k), resid)?;
        }
        ConvPath::Direct => {
            exec.ph_conv(x, &c.hist, &c.w, &c.b, y, (rows, c.cin, n, c.k), elu, resid)?;
        }
    }
    if c.k > 1 {
        exec.ph_history(x, &mut c.hist, rows, c.cin, c.k - 1)?;
    }
    Ok(())
}

impl GpuPhonon {
    /// Decode `cw.lat[0..frames]` (the latents the codec lane took over at
    /// the handoff) into `cw.p1[0..frames * 1920]`, all on the codec lane;
    /// the Mimi transformer's rows start at `*cw.tx.pos`.
    pub(super) fn codec_pass(&mut self, frames: usize) -> Result<(), GpuError> {
        {
            let exec = &self.codec;
            let m = &mut self.mimi;
            let w = &mut self.cw;
            // latent * std + mean, then the quantizer's 32 -> 512 projection
            exec.ph_point(
                Some(&w.lat),
                0,
                Some(&m.stats),
                &mut w.denorm,
                0,
                frames * 32,
                32,
                3,
                1.0,
            )?;
            exec.kumo_gemm(
                (&w.denorm, 0),
                (&m.quant_proj, 0),
                None,
                &mut w.proj,
                None,
                (32, D, frames),
                KumoEpi::Store,
                1,
                (0, 0, 0),
            )?;
            // x16 depthwise upsample to the transformer's 200 Hz; its history
            // is the batch's last projected row
            exec.ph_upsample(&w.proj, &m.up_hist, &m.upsample, &mut w.tx.x, frames)?;
            exec.copy_region(&w.proj, (frames - 1) * D, &mut m.up_hist, 0, D)?;
        }
        let mut rows = frames * CODEC_ROWS;
        self.wide_codec(rows)?;
        let exec = &self.codec;
        let w = &mut self.cw;
        let convs = &mut self.mimi.convs;
        // SEANet: conv k7; per stage ELU -> transposed conv (x6 / x5 / x4),
        // then the residual block (ELU -> conv k3 to half width -> ELU ->
        // conv k1 back, accumulated onto the stage's output - the skip);
        // finally ELU -> conv k3 to mono. The stage planes alternate p1 / p2.
        conv(
            exec,
            &mut convs[0],
            &w.tx.x,
            &mut w.stage,
            &mut w.p1,
            rows,
            false,
            false,
        )?;
        let mut cur_is_p1 = true;
        for s in 0..3 {
            let [up, half, back] = &mut convs[1 + 3 * s..4 + 3 * s] else {
                unreachable!("three convolutions a stage")
            };
            let (cur, skip) = if cur_is_p1 {
                (&mut w.p1, &mut w.p2)
            } else {
                (&mut w.p2, &mut w.p1)
            };
            conv(exec, up, cur, &mut w.stage, skip, rows, true, false)?;
            rows *= up.stride;
            conv(exec, half, skip, &mut w.stage, &mut w.p3, rows, true, false)?;
            conv(exec, back, &w.p3, &mut w.stage, skip, rows, true, true)?;
            cur_is_p1 = !cur_is_p1;
        }
        // three stages flip p1 -> p2 -> p1 -> p2: the last skip is p2, and
        // the mono output lands in p1
        conv(
            exec,
            &mut convs[10],
            &w.p2,
            &mut w.stage,
            &mut w.p1,
            rows,
            true,
            false,
        )
    }
}
