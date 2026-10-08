//! Pictures and audio in an EmbeddingGemma 2 sequence: the token layout the
//! model's processor builds, the image processor's resize, and the towers
//! that turn a medium into rows of the 512-wide input.
//!
//! The layout (Hugging Face `EmbeddingGemma2Processor`, llama.cpp's mtmd
//! agrees): a picture is `<|image>` + N x `<|image|>` + `<image|>`, a clip
//! `<|audio>` + N x `<|audio|>` + `<audio|>`, spliced where the medium sits
//! in the text, inside the sequence's `<bos>` ... `<eos>`. The soft tokens'
//! rows are the tower's output, entered WITHOUT the sqrt(512) the token rows
//! get (`build_inp_embd`'s raw-embedding path, HF's `masked_scatter`), and
//! the per-layer inputs are projected from those rows like any other.
//!
//! Pictures: Hugging Face's Gemma 4 image processor - aspect-preserving
//! resize to the largest size whose sides are whole output tokens (48 px:
//! 16-px patches, 3x3 pooled) inside the soft-token budget (280 by default),
//! torchvision's uint8 antialiased bicubic (`resample=BICUBIC,
//! antialias=True`, the fast processor - reproduced byte for byte by the
//! Clef lane's resampler, slots 735/736), then 1/255; the tower applies its
//! own 2x - 1. The tower is Gemma 4's vision tower at the E2B/E4B geometry
//! (768 wide, 16 layers), served by [`VisionModel`].

use std::sync::Arc;

use crate::gpu::{ClefPlanMem, GpuError, GpuExecutor};
use crate::gpu_model::gemma4::vision::{Resized, VisionModel, VisionOutput};

pub use crate::encoder::embedding_gemma2::{
    AUDIO_TOKEN, BOA_TOKEN, BOI_TOKEN, DEFAULT_IMAGE_TOKENS, EOA_TOKEN, EOI_TOKEN, IMAGE_TOKEN,
    IMAGE_TOKEN_BUDGETS, Run, VIDEO_TOKEN, image_target, image_tokens, placeholder_runs,
};

/// The picture tower plus its resize: plan memory for the two resample axes
/// and the device staging for one picture's three planes (raw, width-resized,
/// final), grown to the largest picture seen.
pub struct ImageTower {
    exec: Arc<GpuExecutor>,
    tower: VisionModel,
    budget: usize,
    plans: [ClefPlanMem; 2],
    stage: [cudarc::driver::CudaSlice<u8>; 3],
}

/// The largest picture side the resampler plans for (input or output).
const PLAN_MAX: usize = 8192;

impl ImageTower {
    pub fn new(
        exec: Arc<GpuExecutor>,
        tower: VisionModel,
        budget: usize,
    ) -> Result<Self, GpuError> {
        if !IMAGE_TOKEN_BUDGETS.contains(&budget) {
            return Err(GpuError::Driver(format!(
                "image token budget {budget} is not one of {IMAGE_TOKEN_BUDGETS:?}"
            )));
        }
        if !exec.has_clef_vision() {
            return Err(GpuError::MissingOp("clef_resample (slots 735/736)"));
        }
        // the Clef lane's sizing: 2 x out + 1 indices, (4 x in + 5 x out)
        // int16 weights an axis
        let plan = || -> Result<ClefPlanMem, GpuError> {
            Ok(ClefPlanMem {
                idx: exec.alloc_u32(2 * PLAN_MAX + 1)?,
                w: exec.alloc_u8(2 * (4 * PLAN_MAX + 5 * PLAN_MAX))?,
            })
        };
        let plans = [plan()?, plan()?];
        // an encoder embeds pictures back to back: keep the tower's scratch
        // between them (released with the rest on idle, `release`)
        tower.keep_scratch(true);
        let stage = [exec.alloc_u8(1)?, exec.alloc_u8(1)?, exec.alloc_u8(1)?];
        Ok(Self {
            exec,
            tower,
            budget,
            plans,
            stage,
        })
    }

    /// Drop what is kept between pictures (the owner's idle release).
    pub fn release(&mut self) {
        self.tower.release_scratch();
    }

    /// Soft tokens a `w` x `h` picture takes on this endpoint.
    pub fn tokens_for(&self, w: usize, h: usize) -> Result<usize, String> {
        image_tokens(w, h, self.budget)
    }

    /// Resident weight bytes of the tower.
    pub fn weight_bytes(&self) -> usize {
        self.tower.weight_bytes()
    }

    fn grow(&mut self, i: usize, n: usize) -> Result<(), GpuError> {
        if self.stage[i].len() < n {
            self.stage[i] = self.exec.alloc_u8(n)?;
        }
        Ok(())
    }

    /// One picture (interleaved RGB8) -> its [tokens][512] input rows: the
    /// processor's resize, then the tower from the resized plane where it
    /// lies on the device (the fused tower patchifies it there).
    pub fn encode(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<VisionOutput, GpuError> {
        let (at, tw, th) = self.resize_on_device(rgb, w, h)?;
        self.tower
            .encode_resized(Resized::Device(&self.stage[at]), tw, th)
    }

    /// [`Self::encode`] through the tower's unfused chain - the reference the
    /// fused pass is held to.
    pub fn encode_unfused(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<VisionOutput, GpuError> {
        let (at, tw, th) = self.resize_on_device(rgb, w, h)?;
        self.tower
            .encode_resized_unfused(Resized::Device(&self.stage[at]), tw, th)
    }

    /// The processor's resize, downloaded: (RGB8, w, h).
    pub fn resize(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(Vec<u8>, usize, usize), GpuError> {
        let (at, tw, th) = self.resize_on_device(rgb, w, h)?;
        Ok((
            self.exec.to_host_u8_len(&self.stage[at], th * tw * 3)?,
            tw,
            th,
        ))
    }

    /// The processor's resize on the device - width pass, then height, each
    /// only when that side changes (torchvision's order). Returns which stage
    /// plane holds the result, and its size.
    fn resize_on_device(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(usize, usize, usize), GpuError> {
        let bad = |m: String| GpuError::Driver(m);
        if rgb.len() != w * h * 3 {
            return Err(bad(format!("a {w} x {h} picture with {} bytes", rgb.len())));
        }
        let (tw, th) = image_target(w, h, self.budget).map_err(bad)?;
        if w.max(h).max(tw).max(th) > PLAN_MAX {
            return Err(bad(format!(
                "a {w} x {h} picture (to {tw} x {th}) is past the {PLAN_MAX}-pixel resampler"
            )));
        }
        let e = self.exec.clone();
        self.grow(0, w * h * 3)?;
        self.grow(1, h * tw * 3)?;
        self.grow(2, th * tw * 3)?;
        {
            let mut raw = self.stage[0].slice_mut(0..w * h * 3);
            e.upload_u8_into(rgb, &mut raw)?;
        }
        let [s0, s1, s2] = &mut self.stage;
        let [pw, ph] = &mut self.plans;
        // which plane holds the picture after the passes (0 raw, 1 mid, 2 fin)
        let mut at = 0u8;
        if tw != w {
            let plan = e.clef_resample_plan(w, tw, pw)?;
            e.clef_resample(s0, s1, h, true, &plan, pw)?;
            at = 1;
        }
        if th != h {
            let plan = e.clef_resample_plan(h, th, ph)?;
            if at == 1 {
                e.clef_resample(s1, s2, tw, false, &plan, ph)?;
            } else {
                e.clef_resample(s0, s2, tw, false, &plan, ph)?;
            }
            at = 2;
        }
        Ok((at as usize, tw, th))
    }
}
