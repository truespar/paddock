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
use crate::gpu_model::gemma4::vision::{VisionModel, VisionOutput};

/// `<|image|>`: a picture's soft token.
pub const IMAGE_TOKEN: u32 = 258880;
/// `<|audio|>`: an audio clip's soft token.
pub const AUDIO_TOKEN: u32 = 258881;
/// `<image|>`: closes a picture.
pub const EOI_TOKEN: u32 = 258882;
/// `<audio|>`: closes a clip.
pub const EOA_TOKEN: u32 = 258883;
/// `<|video|>`: a video frame's soft token (frames are not served yet).
pub const VIDEO_TOKEN: u32 = 258884;
/// `<|image>`: opens a picture.
pub const BOI_TOKEN: u32 = 255999;
/// `<|audio>`: opens a clip.
pub const BOA_TOKEN: u32 = 256000;

/// The soft-token budgets the processor supports (`_SUPPORTED_SOFT_TOKENS`).
pub const IMAGE_TOKEN_BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];
/// The checkpoint's own (`vision_soft_tokens_per_image`).
pub const DEFAULT_IMAGE_TOKENS: usize = 280;
const PATCH: usize = 16;
const POOL: usize = 3;

/// One run of soft tokens in a sequence: where it starts, how long it is,
/// and which medium's token it repeats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub token: u32,
    pub start: usize,
    pub len: usize,
}

/// The sequence's media soft-token runs in order (`<|image|>`, `<|audio|>`
/// and `<|video|>` repeats). Two media never touch - each run sits between
/// its medium's open and close tokens - so a run is a maximal repeat.
pub fn placeholder_runs(seq: &[u32]) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    for (i, &t) in seq.iter().enumerate() {
        if !matches!(t, IMAGE_TOKEN | AUDIO_TOKEN | VIDEO_TOKEN) {
            continue;
        }
        match out.last_mut() {
            Some(r) if r.token == t && r.start + r.len == i => r.len += 1,
            _ => out.push(Run {
                token: t,
                start: i,
                len: 1,
            }),
        }
    }
    out
}

/// The processor's resize target for a `w` x `h` picture at a soft-token
/// budget, as (width, height) - `get_aspect_ratio_preserving_size`, ported
/// line for line (its float math in f64, as Python's).
pub fn image_target(w: usize, h: usize, budget: usize) -> Result<(usize, usize), String> {
    if w == 0 || h == 0 {
        return Err("an empty picture".into());
    }
    let max_patches = budget * POOL * POOL;
    let target_px = (max_patches * PATCH * PATCH) as f64;
    let factor = (target_px / (h * w) as f64).sqrt();
    let side = PATCH * POOL;
    let mut th = (factor * h as f64 / side as f64).floor() as usize * side;
    let mut tw = (factor * w as f64 / side as f64).floor() as usize * side;
    if th == 0 && tw == 0 {
        return Err("the picture is too small to resize to whole tokens".into());
    }
    let max_side = (max_patches / (POOL * POOL)) * side;
    if th == 0 {
        th = side;
        tw = ((w as f64 / h as f64).floor() as usize * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((h as f64 / w as f64).floor() as usize * side).min(max_side);
    }
    if (th * tw) as f64 > target_px {
        return Err(format!(
            "a {w} x {h} picture resizes past the {budget}-token budget"
        ));
    }
    Ok((tw, th))
}

/// Soft tokens a `w` x `h` picture takes at `budget`.
pub fn image_tokens(w: usize, h: usize, budget: usize) -> Result<usize, String> {
    let (tw, th) = image_target(w, h, budget)?;
    Ok((tw / PATCH) * (th / PATCH) / (POOL * POOL))
}

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
        let stage = [exec.alloc_u8(1)?, exec.alloc_u8(1)?, exec.alloc_u8(1)?];
        Ok(Self {
            exec,
            tower,
            budget,
            plans,
            stage,
        })
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
    /// processor's resize, patchify, the tower.
    pub fn encode(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<VisionOutput, GpuError> {
        let (resized, tw, th) = self.resize(rgb, w, h)?;
        let (patches, gw, gh) = self.tower.patches_from_rgb(&resized, tw, th);
        self.tower.encode(&patches, gw, gh)
    }

    /// The processor's resize on the device - width pass, then height, each
    /// only when that side changes (torchvision's order) - as (RGB8, w, h).
    pub fn resize(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(Vec<u8>, usize, usize), GpuError> {
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
        let plane = match at {
            0 => &self.stage[0],
            1 => &self.stage[1],
            _ => &self.stage[2],
        };
        Ok((e.to_host_u8_len(plane, th * tw * 3)?, tw, th))
    }
}
