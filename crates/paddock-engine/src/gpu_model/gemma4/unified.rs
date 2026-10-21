//! Gemma 4 12B's media input - the "unified" (encoder-free) checkpoint.
//!
//! There is no vision tower and no audio tower: the mmproj (`gemma4uv` +
//! `gemma4ua`) holds two small embedders, and their rows go straight into
//! the residual stream like gemma4v's projected rows do.
//!
//! - a picture is resized by Hugging Face's Gemma 4 image processor (the same
//!   aspect-preserving fit to whole 48 px tokens inside the soft-token budget,
//!   uint8 antialiased bicubic - EmbeddingGemma 2's resize, on the Clef
//!   resampler), cut into 48 px patches (raster order, 6912 values each in
//!   the file's CHW order), then LayerNorm -> Linear 6912 -> 3840 (+bias) ->
//!   LayerNorm -> + pos_x[col] + pos_y[row] -> LayerNorm -> weightless RMS
//!   norm -> Linear 3840 -> 3840
//! - a clip is cut into 640-sample (40 ms) frames, one token each, no
//!   downsampling: weightless RMS norm -> Linear 640 -> 3840
//!
//! LayerNorms take torch's default 1e-5 (llama.cpp's gemma4uv graph agrees),
//! the RMS norms the file's 1e-6. Every projection is slot 722 (F32 class:
//! the activation split three ways in bf16 against the stored bf16 weight -
//! the file's F32 patch plane is bf16-exact, checked at load, so it is kept
//! as bf16). Pictures attend bidirectionally inside their run in the LLM
//! (the caller's attention bounds); clips stay causal.

use std::sync::{Arc, Mutex};

use cudarc::driver::CudaSlice;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::encoder::embedding_gemma2::{IMAGE_TOKEN_BUDGETS, image_target, image_tokens};
use crate::gpu::{ClefPlanMem, DiarEpi, DiarWeights, G4U_FRAME, G4U_PATCH, GpuError, GpuExecutor};

/// The model patch side: 16 px patches merged 3 x 3 by the processor.
const SIDE: usize = 48;
/// torch's LayerNorm default - the embedder's three norms are nn.LayerNorm.
const LN_EPS: f32 = 1e-5;
/// The largest picture side the resampler plans for (input or output).
const PLAN_MAX: usize = 8192;

fn drv(m: impl Into<String>) -> GpuError {
    GpuError::Driver(m.into())
}

/// One tensor of the mmproj as (dims, f32 values) - F32 or BF16 storage.
fn host_f32(map: &MappedGguf, name: &str) -> Result<(Vec<usize>, Vec<f32>), GpuError> {
    use paddock_models::ggml_type::GgmlType;
    let (info, bytes) = map.tensor_bytes(name)?;
    let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    let v = match info.ggml_type {
        GgmlType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        GgmlType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        ty => return Err(drv(format!("{name}: unhandled type {ty:?}"))),
    };
    Ok((dims, v))
}

/// A `[n][k]` projection plane as bf16 bytes for slot 722. BF16 storage is
/// taken as stored; F32 storage must be bf16-exact (the 12B's patch plane
/// is: the checkpoint is bf16, the converter only widened it) - anything else
/// would need a precision class this embedder does not have, so it refuses.
fn bf16_plane(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    k: usize,
    n: usize,
) -> Result<CudaSlice<u8>, GpuError> {
    use paddock_models::ggml_type::GgmlType;
    let (info, bytes) = map.tensor_bytes(name)?;
    let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    if dims != [k, n] {
        return Err(drv(format!("{name}: dims {dims:?}, expected [{k}, {n}]")));
    }
    let plane: Vec<u8> = match info.ggml_type {
        GgmlType::Bf16 => bytes.to_vec(),
        GgmlType::F32 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for c in bytes.as_chunks::<4>().0 {
                let bits = u32::from_le_bytes(*c);
                if bits & 0xffff != 0 {
                    return Err(drv(format!(
                        "{name}: an F32 value that is not bf16-exact - not the 12B's \
                         converted checkpoint"
                    )));
                }
                out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            }
            out
        }
        ty => return Err(drv(format!("{name}: unhandled type {ty:?}"))),
    };
    exec.to_device_u8(&plane)
}

/// A norm's (weight, bias) pair, `n` wide.
fn ln_pair(
    exec: &GpuExecutor,
    map: &MappedGguf,
    k: u32,
    n: usize,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>), GpuError> {
    let one = |what: &str| -> Result<CudaSlice<f32>, GpuError> {
        let name = format!("v.patch_norm.{k}.{what}");
        let (dims, v) = host_f32(map, &name)?;
        if dims != [n] {
            return Err(drv(format!("{name}: dims {dims:?}, expected [{n}]")));
        }
        exec.to_device(&v)
    };
    Ok((one("weight")?, one("bias")?))
}

/// Device staging, grown to the largest picture / clip seen. Behind a mutex
/// because a tower encodes through `&self`.
struct Stage {
    plans: [ClefPlanMem; 2],
    planes: [CudaSlice<u8>; 3],
    patches: CudaSlice<f32>,
    hid: CudaSlice<f32>,
    samples: CudaSlice<f32>,
    frames: CudaSlice<f32>,
}

/// The picture + audio embedders of one unified mmproj.
pub struct UnifiedEmbedder {
    exec: Arc<GpuExecutor>,
    /// soft tokens a picture resizes to (the processor's `max_soft_tokens`)
    budget: usize,
    embd: usize,
    eps_rms: f32,
    ln1: (CudaSlice<f32>, CudaSlice<f32>),
    patch_w: CudaSlice<u8>,
    patch_b: CudaSlice<f32>,
    ln2: (CudaSlice<f32>, CudaSlice<f32>),
    /// `[2][pos_size][embd]`: the x table, then the y table
    pos: CudaSlice<f32>,
    pos_size: usize,
    ln3: (CudaSlice<f32>, CudaSlice<f32>),
    proj: CudaSlice<u8>,
    /// the audio projection; None when the mmproj has no `gemma4ua` half or
    /// the endpoint turned audio off
    audio: Option<CudaSlice<u8>>,
    weight_bytes: usize,
    stage: Mutex<Stage>,
}

/// Whether an mmproj is the unified one (its picture projector is
/// `gemma4uv`). The 12B's text file is plain `gemma4`, so the projector is
/// the only thing that tells the two picture paths apart.
pub fn is_unified(map: &MappedGguf) -> bool {
    matches!(
        map.gguf().metadata.get("clip.vision.projector_type"),
        Some(Value::Str(s)) if s == "gemma4uv"
    )
}

impl UnifiedEmbedder {
    /// `max_image_tokens` is the endpoint's picture budget (None = the
    /// checkpoint's 280; one of the processor's 70 / 140 / 280 / 560 / 1120).
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_image_tokens: Option<usize>,
        audio: bool,
    ) -> Result<Self, GpuError> {
        if !exec.has_g4u() || !exec.has_clef_vision() {
            return Err(GpuError::MissingOp(
                "g4u_patches / g4u_post / g4u_audio (slots 862-864) + clef_resample",
            ));
        }
        let budget = max_image_tokens.unwrap_or(280);
        if !IMAGE_TOKEN_BUDGETS.contains(&budget) {
            return Err(drv(format!(
                "max_image_tokens {budget}: the 12B's processor takes one of \
                 {IMAGE_TOKEN_BUDGETS:?}"
            )));
        }
        let embd = match map.gguf().metadata.get("clip.vision.projection_dim") {
            Some(v) => v.as_u64().unwrap_or(0) as usize,
            None => 0,
        };
        if embd == 0 {
            return Err(drv("mmproj missing clip.vision.projection_dim"));
        }
        let eps_rms = match map
            .gguf()
            .metadata
            .get("clip.vision.attention.layer_norm_epsilon")
        {
            Some(Value::F32(f)) => *f,
            _ => 1e-6,
        };
        let ln1 = ln_pair(&exec, map, 1, G4U_PATCH)?;
        let patch_w = bf16_plane(&exec, map, "v.patch_embd.weight", G4U_PATCH, embd)?;
        let (bd, bv) = host_f32(map, "v.patch_embd.bias")?;
        if bd != [embd] {
            return Err(drv(format!("v.patch_embd.bias: dims {bd:?}")));
        }
        let patch_b = exec.to_device(&bv)?;
        let ln2 = ln_pair(&exec, map, 2, embd)?;
        let ln3 = ln_pair(&exec, map, 3, embd)?;
        let (pd, pv) = host_f32(map, "v.position_embd.weight")?;
        if pd.len() != 3 || pd[0] != embd || pd[2] != 2 {
            return Err(drv(format!("v.position_embd.weight: dims {pd:?}")));
        }
        let pos_size = pd[1];
        let pos = exec.to_device(&pv)?;
        let proj = bf16_plane(&exec, map, "mm.input_projection.weight", embd, embd)?;
        let has_audio = matches!(
            map.gguf().metadata.get("clip.audio.projector_type"),
            Some(Value::Str(s)) if s == "gemma4ua"
        );
        let audio = if audio && has_audio {
            Some(bf16_plane(
                &exec,
                map,
                "mm.a.input_projection.weight",
                G4U_FRAME,
                embd,
            )?)
        } else {
            None
        };
        let weight_bytes = 4 * (2 * G4U_PATCH + 6 * embd + 2 * pos_size * embd + embd)
            + patch_w.len()
            + proj.len()
            + audio.as_ref().map_or(0, |a| a.len());
        let plan = || -> Result<ClefPlanMem, GpuError> {
            Ok(ClefPlanMem {
                idx: exec.alloc_u32(2 * PLAN_MAX + 1)?,
                w: exec.alloc_u8(2 * (4 * PLAN_MAX + 5 * PLAN_MAX))?,
            })
        };
        let stage = Stage {
            plans: [plan()?, plan()?],
            planes: [exec.alloc_u8(1)?, exec.alloc_u8(1)?, exec.alloc_u8(1)?],
            patches: exec.alloc(1)?,
            hid: exec.alloc(1)?,
            samples: exec.alloc(1)?,
            frames: exec.alloc(1)?,
        };
        tracing::info!(
            budget,
            audio = audio.is_some(),
            "Gemma 4 unified media embedders attached (encoder-free mmproj)"
        );
        Ok(Self {
            exec,
            budget,
            embd,
            eps_rms,
            ln1,
            patch_w,
            patch_b,
            ln2,
            pos,
            pos_size,
            ln3,
            proj,
            audio,
            weight_bytes,
            stage: Mutex::new(stage),
        })
    }

    pub fn llm_embd(&self) -> usize {
        self.embd
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// The processor always fills its budget (it scales up as well as down),
    /// so a picture costs the budget give or take the 48 px flooring: the
    /// budget is both bounds.
    pub fn budget(&self) -> crate::generator::VisionBudget {
        let px = (self.budget * SIDE * SIDE) as u64;
        crate::generator::VisionBudget {
            max_pixels: px,
            min_pixels: px,
            max_edge: None,
            pixels_per_token: (SIDE * SIDE) as u64,
            max_tokens: self.budget as u32,
            min_tokens: self.budget as u32,
        }
    }

    /// Soft tokens a `w` x `h` picture encodes to (0 for a picture the
    /// processor cannot take - the encode then names why).
    pub fn tokens_for(&self, w: usize, h: usize) -> usize {
        image_tokens(w, h, self.budget).unwrap_or(0)
    }

    /// Rows a clip of `n` samples takes: one a 640-sample frame, the last one
    /// padded (the feature extractor's `ceil`).
    pub fn audio_rows(n: usize) -> usize {
        n.div_ceil(G4U_FRAME)
    }

    fn grow_u8(exec: &GpuExecutor, b: &mut CudaSlice<u8>, n: usize) -> Result<(), GpuError> {
        if b.len() < n {
            *b = exec.alloc_u8(n)?;
        }
        Ok(())
    }

    fn grow_f32(exec: &GpuExecutor, b: &mut CudaSlice<f32>, n: usize) -> Result<(), GpuError> {
        if b.len() < n {
            *b = exec.alloc(n)?;
        }
        Ok(())
    }

    /// One RGB8 picture -> its `[n][embd]` rows.
    pub fn encode_picture(
        &self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(CudaSlice<f32>, usize), GpuError> {
        if rgb.len() != w * h * 3 {
            return Err(drv(format!("a {w} x {h} picture with {} bytes", rgb.len())));
        }
        let (tw, th) = image_target(w, h, self.budget).map_err(drv)?;
        if w.max(h).max(tw).max(th) > PLAN_MAX {
            return Err(drv(format!(
                "a {w} x {h} picture (to {tw} x {th}) is past the {PLAN_MAX}-pixel resampler"
            )));
        }
        let (gw, gh) = (tw / SIDE, th / SIDE);
        if gw > self.pos_size || gh > self.pos_size {
            return Err(drv(format!(
                "a {gw} x {gh} patch grid is past the {}-entry position tables",
                self.pos_size
            )));
        }
        let n = gw * gh;
        let e = &self.exec;
        let mut guard = self
            .stage
            .lock()
            .map_err(|_| drv("embedder stage poisoned"))?;
        let st = &mut *guard;
        // the processor's resize: width pass then height, each only when that
        // side changes (torchvision's order)
        Self::grow_u8(e, &mut st.planes[0], w * h * 3)?;
        Self::grow_u8(e, &mut st.planes[1], h * tw * 3)?;
        Self::grow_u8(e, &mut st.planes[2], th * tw * 3)?;
        {
            let mut raw = st.planes[0].slice_mut(0..w * h * 3);
            e.upload_u8_into(rgb, &mut raw)?;
        }
        let [s0, s1, s2] = &mut st.planes;
        let [pw, ph] = &mut st.plans;
        let mut at = 0usize;
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
            0 => &st.planes[0],
            1 => &st.planes[1],
            _ => &st.planes[2],
        };
        Self::grow_f32(e, &mut st.patches, n * G4U_PATCH)?;
        Self::grow_f32(e, &mut st.hid, n * self.embd)?;
        e.g4u_patches(
            plane,
            tw,
            gw,
            n,
            &self.ln1.0,
            &self.ln1.1,
            &mut st.patches,
            LN_EPS,
        )?;
        e.diar_gemm(
            (&st.patches, 0),
            DiarWeights::Bf16(&self.patch_w),
            Some(&self.patch_b),
            &mut st.hid,
            (G4U_PATCH, self.embd, n),
            DiarEpi::Store,
            None,
        )?;
        e.g4u_post(
            &mut st.hid,
            (&self.ln2.0, &self.ln2.1),
            &self.pos,
            self.pos_size,
            gw,
            (&self.ln3.0, &self.ln3.1),
            n,
            self.embd,
            (LN_EPS, self.eps_rms),
        )?;
        let mut out = e.alloc(n * self.embd)?;
        e.diar_gemm(
            (&st.hid, 0),
            DiarWeights::Bf16(&self.proj),
            None,
            &mut out,
            (self.embd, self.embd, n),
            DiarEpi::Store,
            None,
        )?;
        Ok((out, n))
    }

    /// Rows `f0 .. f1` of a 16 kHz clip -> `[f1 - f0][embd]`. Frames are
    /// independent, so a pass encodes only the frames it splices.
    pub fn encode_audio(
        &self,
        samples: &[f32],
        f0: usize,
        f1: usize,
    ) -> Result<CudaSlice<f32>, GpuError> {
        let proj = self
            .audio
            .as_ref()
            .ok_or_else(|| drv("this endpoint serves no audio (the mmproj's audio half is off)"))?;
        let t = f1.saturating_sub(f0);
        if t == 0 || f1 > Self::audio_rows(samples.len()) {
            return Err(drv(format!(
                "audio frames {f0}..{f1} of a {}-sample clip",
                samples.len()
            )));
        }
        let lo = f0 * G4U_FRAME;
        let hi = (f1 * G4U_FRAME).min(samples.len());
        let e = &self.exec;
        let mut guard = self
            .stage
            .lock()
            .map_err(|_| drv("embedder stage poisoned"))?;
        let st = &mut *guard;
        Self::grow_f32(e, &mut st.samples, hi - lo)?;
        Self::grow_f32(e, &mut st.frames, t * G4U_FRAME)?;
        e.upload_f32_at(&samples[lo..hi], &mut st.samples, 0)?;
        e.g4u_audio(&st.samples, hi - lo, t, &mut st.frames, self.eps_rms)?;
        let mut out = e.alloc(t * self.embd)?;
        e.diar_gemm(
            (&st.frames, 0),
            DiarWeights::Bf16(proj),
            None,
            &mut out,
            (G4U_FRAME, self.embd, t),
            DiarEpi::Store,
            None,
        )?;
        Ok(out)
    }
}
