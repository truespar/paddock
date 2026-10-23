//! The picture path: Pixtral's processor and tower and Mistral 3's
//! projector, from a decoded picture to the body rows its `[IMG]` tokens
//! take.
//!
//! - the processor (the Transformers Torchvision backend, on the device):
//!   torch's uint8 antialiased bicubic resize to the floor-ratio target
//!   rounded up to whole 28-pixel tokens (slots 735/736 - integer weights and
//!   sums, byte for byte), then the fused rescale + normalize into 14 x 14
//!   patch rows (896);
//! - the tower: the patch conv as a GEMM, `ln_pre`, 24 pre-norm blocks
//!   (RMSNorm, fused q|k|v, the 2D rope from a per-pass angle table at each
//!   patch's own row (899), attention within each picture (900, one sequence
//!   a picture), o onto the residual, RMSNorm, the gated-SiLU FFN);
//! - the projector: RMSNorm, the 2 x 2 merge (each merged token's four patch
//!   rows gathered back to back into the re-laid merger GEMM), linear_1 with
//!   the exact GELU, linear_2 - f32 rows straight into the body's input.
//!
//! A pass packs as many pictures as [`TOWER_PATCHES`] holds; a request with
//! more runs several passes into the same output plane. Every op is
//! row-wise or per picture, so a picture's rows do not depend on what else
//! shared its pass.

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::nemotron_embed::NemotronEmbedConfig;

use crate::gpu::{ClefPlanMem, GpuExecutor, HalfTensor, NE_PATCH_K, NE_TILE_SHIFT, ne_attn_rows};
use crate::gpu_model::gpt_oss::GpuModelError;

/// Patch rows a tower pass holds: two of the largest pictures (1120 x 1120 =
/// 6400 patches) and room to spare, ~23 KB of planes a row.
pub const TOWER_PATCHES: usize = 16384;
/// The longest side a picture may arrive with (it bounds the resize plans;
/// the staging follows the pictures a pass actually carries).
pub const MAX_SIDE: usize = 16384;

pub(crate) struct VisionLayer {
    pub ln1: CudaSlice<f32>,
    pub qkv: HalfTensor,
    pub o: HalfTensor,
    pub ln2: CudaSlice<f32>,
    pub gu: HalfTensor,
    pub down: HalfTensor,
}

/// One picture as the request delivered it: RGB8, `w` x `h`, upright.
pub struct Picture<'a> {
    pub rgb: &'a [u8],
    pub w: usize,
    pub h: usize,
}

struct TowerScratch {
    patches: usize,
    /// the resize's final canvas, bytes
    fin: CudaSlice<u8>,
    rows16: CudaSlice<f16>,
    x: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    xn16: CudaSlice<f16>,
    qkv16: CudaSlice<f16>,
    attn16: CudaSlice<f16>,
    ff16: CudaSlice<f16>,
    pos: CudaSlice<u32>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    cu: CudaSlice<u32>,
    tiles: CudaSlice<u32>,
    midx: CudaSlice<u32>,
    m16: CudaSlice<f16>,
    y16: CudaSlice<f16>,
    g16: CudaSlice<f16>,
}

pub(crate) struct Tower {
    patch: HalfTensor,
    ln_pre: CudaSlice<f32>,
    layers: Vec<VisionLayer>,
    proj_norm: CudaSlice<f32>,
    merge: HalfTensor,
    l1: HalfTensor,
    l2: HalfTensor,
    /// the 2D rope's frequencies, row half then column half
    inv: Vec<f32>,
    plans: [ClefPlanMem; 2],
    zeros: CudaSlice<f32>,
    cfg: NemotronEmbedConfig,
    scratch: Option<TowerScratch>,
    /// the resize's staging (the arriving picture, then width-resized),
    /// grown to the largest picture seen and released with the scratch
    stage: Option<[CudaSlice<u8>; 2]>,
}

impl Tower {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        exec: &GpuExecutor,
        cfg: &NemotronEmbedConfig,
        patch: HalfTensor,
        ln_pre: CudaSlice<f32>,
        layers: Vec<VisionLayer>,
        proj_norm: CudaSlice<f32>,
        merge: HalfTensor,
        l1: HalfTensor,
        l2: HalfTensor,
    ) -> Result<Self, GpuModelError> {
        let plan = || -> Result<ClefPlanMem, GpuModelError> {
            let b = ClefPlanMem::bytes(MAX_SIDE, MAX_SIDE);
            Ok(ClefPlanMem {
                idx: exec.alloc_u32(2 * MAX_SIDE + 1)?,
                w: exec.alloc_u8(b)?,
            })
        };
        Ok(Self {
            patch,
            ln_pre,
            layers,
            proj_norm,
            merge,
            l1,
            l2,
            inv: cfg.vision_inv_freq(),
            plans: [plan()?, plan()?],
            zeros: exec.to_device(&vec![0.0f32; cfg.d_model.max(cfg.v_dim)])?,
            cfg: cfg.clone(),
            scratch: None,
            stage: None,
        })
    }

    /// Drop the pass planes (never the weights).
    pub fn release(&mut self) {
        self.scratch = None;
        self.stage = None;
    }

    fn ensure(&mut self, exec: &GpuExecutor, patches: usize) -> Result<(), GpuModelError> {
        if self.scratch.as_ref().is_some_and(|s| s.patches >= patches) {
            return Ok(());
        }
        let p = patches
            .next_multiple_of(1024)
            .min(TOWER_PATCHES)
            .max(patches);
        let c = &self.cfg;
        let (vd, vf) = (c.v_dim, c.v_ffn);
        let tokens = p / 4;
        self.scratch = None;
        self.scratch = Some(TowerScratch {
            patches: p,
            fin: exec.alloc_u8(3 * p * 196)?,
            rows16: exec.alloc_f16(p * NE_PATCH_K)?,
            x: exec.alloc(p * vd)?,
            proj: exec.alloc(p * vd)?,
            xn16: exec.alloc_f16(p * vd)?,
            qkv16: exec.alloc_f16(p * 3 * vd)?,
            attn16: exec.alloc_f16(p * vd)?,
            ff16: exec.alloc_f16(p * vf)?,
            pos: exec.alloc_u32(p)?,
            cos: exec.alloc(p * c.v_head_dim / 2)?,
            sin: exec.alloc(p * c.v_head_dim / 2)?,
            // a picture is at least 4 patches, and its tiles are one more than
            // its patches over a tile's rows at most
            cu: exec.alloc_u32(p / 4 + 2)?,
            tiles: exec.alloc_u32(p / ne_attn_rows(1) + p / 4 + 2)?,
            midx: exec.alloc_u32(p)?,
            m16: exec.alloc_f16(tokens * 4 * vd)?,
            y16: exec.alloc_f16(tokens * vd)?,
            g16: exec.alloc_f16(tokens * c.d_model)?,
        });
        Ok(())
    }

    /// Encode `pics` in order; picture i's merged tokens land back to back in
    /// `out` (`[*][d_model]` f32) from row `row0`. Returns the rows written.
    pub fn encode(
        &mut self,
        exec: &GpuExecutor,
        pics: &[Picture<'_>],
        out: &mut CudaSlice<f32>,
        row0: usize,
    ) -> Result<usize, GpuModelError> {
        let bad = |m: String| GpuModelError::Unsupported(format!("nemotron-embed picture: {m}"));
        let targets: Vec<(usize, usize)> = pics
            .iter()
            .map(|p| {
                if p.w > MAX_SIDE || p.h > MAX_SIDE {
                    return Err(bad(format!(
                        "a {} x {} picture: sides must be within {MAX_SIDE}",
                        p.w, p.h
                    )));
                }
                if p.rgb.len() != 3 * p.w * p.h {
                    return Err(bad("its bytes do not match its size".into()));
                }
                self.cfg.picture_target(p.w, p.h).map_err(bad)
            })
            .collect::<Result<_, _>>()?;
        let patches = |(tw, th): (usize, usize)| (tw / 14) * (th / 14);
        let mut row = row0;
        let mut i = 0usize;
        while i < pics.len() {
            // the pictures this pass takes: as many as fit, at least one
            let mut j = i;
            let mut n = 0usize;
            while j < pics.len() && (j == i || n + patches(targets[j]) <= TOWER_PATCHES) {
                n += patches(targets[j]);
                j += 1;
            }
            if n > TOWER_PATCHES {
                return Err(bad(format!(
                    "{n} patches are past one pass's {TOWER_PATCHES}"
                )));
            }
            row += self.pass(exec, &pics[i..j], &targets[i..j], out, row)?;
            i = j;
        }
        Ok(row - row0)
    }

    /// The processor for a pass's pictures: resize (the width pass, then the
    /// height pass, each only when its axis changes - torch's order), then
    /// the patch rows into `rows16`, pictures back to back.
    fn processor(
        &mut self,
        e: &GpuExecutor,
        pics: &[Picture<'_>],
        targets: &[(usize, usize)],
    ) -> Result<(), GpuModelError> {
        let mean = self.cfg.mean.map(|m| m * 255.0f32);
        let std = self.cfg.std.map(|s| s * 255.0f32);
        // staging for this pass's largest picture: as it arrived, and after
        // the width pass
        let raw_b = pics.iter().map(|p| 3 * p.w * p.h).max().unwrap_or(0);
        let mid_b = pics
            .iter()
            .zip(targets)
            .map(|(p, &(tw, _))| 3 * p.h * tw)
            .max()
            .unwrap_or(0);
        if self
            .stage
            .as_ref()
            .is_none_or(|[r, m]| r.len() < raw_b || m.len() < mid_b)
        {
            let (r0, m0) = self
                .stage
                .as_ref()
                .map_or((0, 0), |[r, m]| (r.len(), m.len()));
            self.stage = None;
            self.stage = Some([e.alloc_u8(raw_b.max(r0))?, e.alloc_u8(mid_b.max(m0))?]);
        }
        let s = self.scratch.as_mut().expect("tower scratch");
        let [raw, mid] = self.stage.as_mut().expect("tower staging");
        let mut row = 0usize;
        for (p, &(tw, th)) in pics.iter().zip(targets) {
            e.upload_u8_into(p.rgb, &mut raw.slice_mut(0..p.rgb.len()))?;
            let [pw, ph] = &mut self.plans;
            let mut at = 0u8;
            if tw != p.w {
                let plan = e.clef_resample_plan(p.w, tw, pw)?;
                e.clef_resample(&*raw, &mut *mid, p.h, true, &plan, pw)?;
                at = 1;
            }
            if th != p.h {
                let plan = e.clef_resample_plan(p.h, th, ph)?;
                if at == 1 {
                    e.clef_resample(&*mid, &mut s.fin, tw, false, &plan, ph)?;
                } else {
                    e.clef_resample(&*raw, &mut s.fin, tw, false, &plan, ph)?;
                }
                at = 2;
            }
            let canvas = match at {
                0 => &*raw,
                1 => &*mid,
                _ => &s.fin,
            };
            e.ne_patch_u8(canvas, 0, &mut s.rows16, row, th, tw, mean, std)?;
            row += (tw / 14) * (th / 14);
        }
        Ok(())
    }

    /// One picture through the processor alone: its f16 patch rows
    /// (`[patches][592]`) - the gates' probe of the pixel path.
    pub fn patch_rows(
        &mut self,
        e: &GpuExecutor,
        pic: &Picture<'_>,
    ) -> Result<Vec<f16>, GpuModelError> {
        let t = self
            .cfg
            .picture_target(pic.w, pic.h)
            .map_err(GpuModelError::Unsupported)?;
        let n = (t.0 / 14) * (t.1 / 14);
        self.ensure(e, n)?;
        self.processor(e, std::slice::from_ref(pic), &[t])?;
        let s = self.scratch.as_ref().expect("tower scratch");
        Ok(e.to_host_f16_len(&s.rows16, n * NE_PATCH_K)?)
    }

    /// One tower pass over pictures whose patches fit [`TOWER_PATCHES`].
    fn pass(
        &mut self,
        e: &GpuExecutor,
        pics: &[Picture<'_>],
        targets: &[(usize, usize)],
        out: &mut CudaSlice<f32>,
        out_row: usize,
    ) -> Result<usize, GpuModelError> {
        let n: usize = targets.iter().map(|&(w, h)| (w / 14) * (h / 14)).sum();
        self.ensure(e, n)?;
        let c = self.cfg.clone();
        let (vd, nh, hd) = (c.v_dim, c.v_heads, c.v_head_dim);
        let half = hd / 2;
        // host tables for the pass: per patch its rope angles' cos / sin, the
        // pictures' row offsets and attention tiles, the merge gather
        let mut cos = Vec::with_capacity(n * half);
        let mut sin = Vec::with_capacity(n * half);
        let mut cu = vec![0u32];
        let mut tiles = Vec::new();
        let mut midx = Vec::with_capacity(n);
        let mut ang = vec![0f32; half];
        let per = ne_attn_rows(1);
        for (k, &(tw, th)) in targets.iter().enumerate() {
            let (pw, ph) = (tw / 14, th / 14);
            let first = *cu.last().expect("cu") as usize;
            for r in 0..ph {
                for col in 0..pw {
                    c.vision_rope_angles(&self.inv, r, col, &mut ang);
                    for &a in &ang {
                        cos.push((a as f64).cos() as f32);
                        sin.push((a as f64).sin() as f32);
                    }
                }
            }
            // merged token (i, j): patches (2i + dy, 2j + dx), dy-major
            for i in 0..ph / 2 {
                for j in 0..pw / 2 {
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        midx.push((first + (2 * i + dy) * pw + 2 * j + dx) as u32);
                    }
                }
            }
            for t in 0..(pw * ph).div_ceil(per) {
                tiles.push(((k as u32) << NE_TILE_SHIFT) | t as u32);
            }
            cu.push((first + pw * ph) as u32);
        }
        let pos: Vec<u32> = (0..n as u32).collect();
        let tokens = n / 4;
        {
            let s = self.scratch.as_mut().expect("tower scratch");
            e.upload_u32(&pos, &mut s.pos)?;
            e.upload_u32(&cu, &mut s.cu)?;
            e.upload_u32(&tiles, &mut s.tiles)?;
            e.upload_u32(&midx, &mut s.midx)?;
            e.upload_f32(&cos, &mut s.cos)?;
            e.upload_f32(&sin, &mut s.sin)?;
        }
        self.processor(e, pics, targets)?;
        let s = self.scratch.as_mut().expect("tower scratch");

        // the tower
        e.f16_gemm_ring_rows(&self.patch, &s.rows16, 0, &mut s.x, 0, n, None)?;
        let eps = paddock_models::nemotron_embed::VISION_RMS_EPS;
        e.rmsnorm_batch_inplace(&mut s.x, &self.ln_pre, vd, eps, n)?;
        let qscale = std::f32::consts::LOG2_E / (hd as f32).sqrt();
        e.ne_rms_h(&s.x, &self.layers[0].ln1, &mut s.xn16, n, vd, eps)?;
        for (li, l) in self.layers.iter().enumerate() {
            e.matvec_batch_f16_h(&l.qkv, &s.xn16, &mut s.qkv16, n)?;
            e.ne_rope_qk(
                &mut s.qkv16,
                &s.pos,
                &s.cos,
                &s.sin,
                n,
                n,
                nh,
                nh,
                hd,
                3 * vd,
                qscale,
                None,
            )?;
            e.ne_attn(
                &s.qkv16,
                &s.cu,
                &s.tiles,
                tiles.len(),
                &mut s.attn16,
                n,
                nh,
                nh,
                hd,
            )?;
            e.f16_gemm_ring_rows(&l.o, &s.attn16, 0, &mut s.proj, 0, n, None)?;
            e.ne_add_rms_h(&mut s.x, &s.proj, Some((&l.ln2, &mut s.xn16)), n, vd, eps)?;
            e.f16_gemm_h_swiglu(&l.gu, &s.xn16, &mut s.ff16, n)?;
            e.f16_gemm_ring_rows(&l.down, &s.ff16, 0, &mut s.proj, 0, n, None)?;
            // the residual add, then the next block's norm - or the
            // projector's (text eps) after the last
            let (w, eps_next) = match self.layers.get(li + 1) {
                Some(next) => (&next.ln1, eps),
                None => (&self.proj_norm, c.rms_eps),
            };
            e.ne_add_rms_h(&mut s.x, &s.proj, Some((w, &mut s.xn16)), n, vd, eps_next)?;
        }

        // the projector (its norm is in xn16)
        e.gather_rows_f16(&s.xn16, &s.midx, &mut s.m16, n, vd)?;
        e.matvec_batch_f16_h(&self.merge, &s.m16, &mut s.y16, tokens)?;
        e.matvec_batch_f16_h_gelu(&self.l1, &s.y16, &mut s.g16, &self.zeros, tokens)?;
        e.f16_gemm_ring_rows(&self.l2, &s.g16, 0, out, out_row, tokens, None)?;
        Ok(tokens)
    }
}
