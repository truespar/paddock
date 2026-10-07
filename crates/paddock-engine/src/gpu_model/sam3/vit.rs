//! The image encoder's forward pass, for a batch of pictures at once:
//!
//!   u8 RGB [B, 1008, 1008, 3]
//!   -> patch stem: normalize + window-major im2row (GPU) -> GEMM
//!      -> + position table (one copy per window) -> ln_pre       [B*5184, 1024]
//!   -> 32 x pre-LN block, the planes between GEMMs at f16:
//!        LN -> fused qkv GEMM -> split + all three biases + rope
//!        -> attention (9 windows of 576 rows a picture, or the whole 5184)
//!        -> o GEMM -> x += o + b; LN -> fc1 GEMM (bias + tanh GELU in its
//!        landing) -> fc2 GEMM -> x += fc2 + b; next LN
//!   -> window-major -> raster f16                                [B, 72, 72, 1024]
//!   -> both necks (neck.rs)
//!
//! Every op is row-batched, so B is pure row count and there is one launch
//! train whatever B is. Nothing here allocates and nothing reads the host.

use cudarc::driver::CudaSlice;
use half::f16;

use super::load::LN_EPS;
use super::{GpuModelError, GpuSam3Vision, Sam3Plane};
use crate::gpu::{GpuExecutor, HalfTensor};

/// One tower GEMM onto the half interface. `land32` is `None` wherever the
/// f16-landing GEMM is the device's elected route, and the f32 plane a
/// non-elected device lands on first otherwise.
pub(super) fn gemm_h(
    exec: &GpuExecutor,
    w: &HalfTensor,
    x16: &CudaSlice<f16>,
    y16: &mut CudaSlice<f16>,
    rows: usize,
    land32: Option<&mut CudaSlice<f32>>,
) -> Result<(), GpuModelError> {
    match land32 {
        None => exec.matvec_batch_f16_h(w, x16, y16, rows)?,
        Some(y32) => {
            exec.matvec_batch_f16(w, x16, y32, rows)?;
            exec.convert_f32_f16(y32, y16, rows * w.dims[1])?;
        }
    }
    Ok(())
}

impl GpuSam3Vision {
    /// Encode `pics` pictures. `pixels` is `pics * 1008 * 1008 * 3` bytes,
    /// picture-major u8 RGB HWC, ALREADY at the model's input size (Meta's
    /// processor resizes to a square without keeping the aspect ratio).
    /// `tracker` also runs the tracker's neck and its two high-resolution
    /// 1x1s - what clicks, boxes and video need; concept prompts read only the
    /// detector's levels. Results stay on the device ([`Sam3Plane`]).
    pub fn encode(
        &mut self,
        pixels: &[u8],
        pics: usize,
        tracker: bool,
    ) -> Result<(), GpuModelError> {
        if pics == 0 {
            return Ok(());
        }
        if pics > self.ws.cap {
            return Err(GpuModelError::BatchTooLarge {
                got: pics,
                max: self.ws.cap,
            });
        }
        if pixels.len() != pics * self.picture_bytes() {
            let s = self.cfg.image_size;
            return Err(GpuModelError::Unsupported(format!(
                "sam3: {} bytes for {pics} picture(s), expected {} ({s}x{s}x3 u8 each)",
                pixels.len(),
                pics * self.picture_bytes()
            )));
        }
        self.exec.upload_u8(pixels, &mut self.ws.px)?;
        self.backbone(pics)?;
        self.necks(pics, tracker)?;
        Ok(())
    }

    /// The input plane, `[cap][1008][1008][3]` u8 - where a caller that
    /// resizes on the device (the request path) lands its pictures before
    /// [`Self::encode_staged`].
    pub fn input_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.ws.px
    }

    pub fn input(&self) -> &CudaSlice<u8> {
        &self.ws.px
    }

    /// [`Self::encode`] over pictures already in [`Self::input_mut`].
    pub fn encode_staged(&mut self, pics: usize, tracker: bool) -> Result<(), GpuModelError> {
        if pics == 0 {
            return Ok(());
        }
        if pics > self.ws.cap {
            return Err(GpuModelError::BatchTooLarge {
                got: pics,
                max: self.ws.cap,
            });
        }
        self.backbone(pics)?;
        self.necks(pics, tracker)?;
        Ok(())
    }

    fn backbone(&mut self, pics: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let cfg = &self.cfg;
        let (d, hd, heads) = (cfg.hidden, cfg.head_dim(), cfg.n_heads);
        let (t, wt) = (cfg.tokens(), cfg.window_tokens());
        let rows = pics * t;
        let scale = 1.0 / (hd as f32).sqrt();
        let ws = &mut self.ws;

        // ---- stem ----
        if self.video_frames {
            exec.sam3_patch_rows_norm(
                &ws.px,
                &mut ws.rows16,
                pics,
                cfg.image_size,
                cfg.patch,
                cfg.window,
                cfg.channels,
                self.kp,
                true,
            )?;
        } else {
            exec.sam3_patch_rows(
                &ws.px,
                &mut ws.rows16,
                pics,
                cfg.image_size,
                cfg.patch,
                cfg.window,
                cfg.channels,
                self.kp,
            )?;
        }
        exec.matvec_batch_f16(&self.patch_w, &ws.rows16, &mut ws.g32, rows)?;
        // window-major rows: row r's cell inside its window is r % 576, which
        // is exactly the tiled table's row
        exec.add_rows_bcast(&mut ws.g32, &self.pos, rows, wt, d)?;
        exec.layernorm(
            &ws.g32,
            &self.ln_pre.w,
            &self.ln_pre.b,
            &mut ws.x,
            rows,
            d,
            LN_EPS,
        )?;

        let first = &self.blocks[0].ln1;
        exec.whisper_ln_f16(&ws.x, &first.w, &first.b, &mut ws.n16, rows, d, LN_EPS)?;

        let n_layer = self.blocks.len();
        let fused_qkv = exec.f16_gemm_qkv_rope_elected() && hd == 64 && d % 128 == 0;
        for li in 0..n_layer {
            let blk = &self.blocks[li];
            // a window block's groups are its windows; a global block's group is
            // the whole picture. Same kernels, different group size and table.
            let (group, rope) = if blk.global {
                (t, &self.rope_glob)
            } else {
                (wt, &self.rope_win)
            };
            if fused_qkv {
                // the split's biases, rope and q scale in the GEMM landing
                exec.f16_gemm_qkv_rope(
                    &blk.wqkv,
                    &ws.n16,
                    &mut ws.q,
                    &mut ws.k,
                    &mut ws.v,
                    &blk.bqkv,
                    Some((&rope.0, &rope.1)),
                    d,
                    hd,
                    rows,
                    group,
                    scale,
                )?;
            } else {
                gemm_h(
                    &exec,
                    &blk.wqkv,
                    &ws.n16,
                    &mut ws.qkv,
                    rows,
                    ws.land32.as_mut(),
                )?;
                exec.sam3_qkv_split_rope_h(
                    &ws.qkv,
                    &blk.bq,
                    &blk.bk,
                    &blk.bv,
                    Some((&rope.0, &rope.1)),
                    &mut ws.q,
                    &mut ws.k,
                    &mut ws.v,
                    d,
                    hd,
                    rows,
                    group,
                    scale,
                )?;
            }
            exec.vision_attn_h(
                &ws.q,
                &ws.k,
                &ws.v,
                &mut ws.att,
                group,
                group,
                heads,
                hd,
                rows / group,
            )?;
            gemm_h(
                &exec,
                &blk.wo,
                &ws.att,
                &mut ws.proj,
                rows,
                ws.land32.as_mut(),
            )?;
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.bo,
                &self.ones,
                &blk.ln2.w,
                &blk.ln2.b,
                &mut ws.n16,
                rows,
                d,
                LN_EPS,
            )?;
            // fc1 + bias + tanh GELU in one landing - the same function Meta's
            // fused bf16 op computes at the same point (see the module note)
            exec.matvec_batch_f16_h_gelu_tanh(&blk.fc1, &ws.n16, &mut ws.ff, &blk.fc1_b, rows)?;
            gemm_h(
                &exec,
                &blk.fc2,
                &ws.ff,
                &mut ws.proj,
                rows,
                ws.land32.as_mut(),
            )?;
            // the seam also lands the next block's pre-norm; after the last
            // block there is no next norm (no ln_post in SAM 3), so it reuses
            // this block's own ln2 purely to satisfy the kernel - n16 is
            // overwritten by the raster exit right after
            let next = if li + 1 < n_layer {
                &self.blocks[li + 1].ln1
            } else {
                &blk.ln2
            };
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.fc2_b,
                &self.ones,
                &next.w,
                &next.b,
                &mut ws.n16,
                rows,
                d,
                LN_EPS,
            )?;
        }

        // ---- exit: the necks read a raster ----
        exec.sam3_rows_to_raster_h(&ws.x, &mut ws.n16, pics, cfg.grid(), cfg.window, d)?;
        Ok(())
    }

    /// Detector neck level `l` (0: 4x, 1: 2x, 2: 1x) of the last
    /// [`Self::encode`], on the device: f32 raster `[side^2][256]`. The
    /// detector reads these in place.
    pub fn det_level(&self, l: usize) -> &CudaSlice<f32> {
        &self.ws.det[l.min(2)]
    }

    /// Tracker neck level `l`, same shape - valid after an encode with the
    /// tracker or a [`Self::tracker_neck`].
    pub fn trk_level(&self, l: usize) -> &CudaSlice<f32> {
        &self.ws.trk[l.min(2)]
    }
    /// The tracker mask decoder's conv_s0 over the x4 level, `[288^2][32]`,
    /// and conv_s1 over the x2 level, `[144^2][64]` (the upscaling skips).
    pub fn trk_s0(&self) -> &CudaSlice<f32> {
        &self.ws.trk_s0
    }
    pub fn trk_s1(&self) -> &CudaSlice<f32> {
        &self.ws.trk_s1
    }

    /// Run the tracker neck (and its two 1x1s) for the pictures the last
    /// [`Self::encode_staged`] encoded without it: the trunk's raster is
    /// still in the workspace, so a click on a picture first asked about in
    /// words costs only this neck.
    pub fn tracker_neck(&mut self, pics: usize) -> Result<(), GpuModelError> {
        if pics == 0 || pics > self.ws.cap {
            return Err(GpuModelError::BatchTooLarge {
                got: pics,
                max: self.ws.cap,
            });
        }
        self.necks_tracker_only(pics)
    }

    /// Copy a plane of the last [`Self::encode`] pass to the host: `pics`
    /// pictures' worth, picture-major, in the plane's own layout (see
    /// [`Sam3Plane`]). The gates' view; the heads read the planes in place.
    pub fn read_plane(&self, plane: Sam3Plane, pics: usize) -> Result<Vec<f32>, GpuModelError> {
        let cfg = &self.cfg;
        let (t, d, f) = (cfg.tokens(), cfg.hidden, cfg.fpn_dim);
        let (buf, per) = match plane {
            Sam3Plane::Trunk => (&self.ws.x, t * d),
            Sam3Plane::Det(l) if l < 3 => (&self.ws.det[l], self.level_side(l).pow(2) * f),
            Sam3Plane::Trk(l) if l < 3 => (&self.ws.trk[l], self.level_side(l).pow(2) * f),
            Sam3Plane::TrkS0 => (&self.ws.trk_s0, self.level_side(0).pow(2) * (f / 8)),
            Sam3Plane::TrkS1 => (&self.ws.trk_s1, self.level_side(1).pow(2) * (f / 4)),
            _ => {
                return Err(GpuModelError::Unsupported(format!(
                    "sam3: no plane {plane:?}"
                )));
            }
        };
        if pics > self.ws.cap {
            return Err(GpuModelError::BatchTooLarge {
                got: pics,
                max: self.ws.cap,
            });
        }
        Ok(self.exec.to_host_len(buf, pics * per)?)
    }
}
