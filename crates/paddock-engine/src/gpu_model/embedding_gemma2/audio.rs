//! EmbeddingGemma 2's audio tower: Gemma 4's audio Conformer ("gemma4a" in
//! the mmproj) from 16 kHz mono samples to rows of the 512-wide text input.
//!
//! Per clip, all on the device (kernels: slots 813-819, the shared f16 and
//! f32 GEMMs): the processor's log-mel frontend, two stride-2 conv layers
//! (4x fewer rows - 25 a second), the f32 input projection, 12 Conformer
//! blocks (half-step FFN, chunked local attention with relative positions,
//! light conv, half-step FFN, closing norm), the 1024 -> 1536 output
//! projection, the embedder's weightless norm and its projection to 512.
//!
//! Every linear but the relative-key one is CLIPPED: its input and output
//! are clamped to bounds the checkpoint ships (`input_min` ... per linear),
//! and those bounds keep every GEMM input under |33| - so the GEMMs run on
//! f16 activations against the file's bf16 weights narrowed to f16 (checked
//! on upload), accumulating in f32, as the picture tower does.
//!
//! The frontend is Hugging Face's `Gemma4AudioFeatureExtractor`: frames of
//! 320 samples (the clip left-padded by 160) every 160, periodic Hann,
//! 512-point FFT magnitudes, 128 HTK mel bins over 0-8 kHz, `ln(mel +
//! 1e-3)`. llama.cpp floors with `ln(max(mel, 1e-3))` instead;
//! [`AudioTower::set_llama_floor`] switches to it so a test can hold the
//! tower to llama.cpp's vectors with the frontend difference taken out.
//!
//! A clip is at most 30 s: the processor truncates past that (its
//! `max_length`) and llama.cpp splits there, so past 30 s the two references
//! disagree and the endpoint refuses instead of picking one.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::mapped::MappedGguf;

use super::GpuModelError;
use crate::gpu::{
    EG2A_MEL, EG2A_WIDTH, Eg2aMelTables, Eg2aSeam, GpuError, GpuExecutor, HalfTensor,
};
use crate::gpu_model::qwen35::vision::host_f32;

/// Audio samples a second (the processor's `sampling_rate`).
pub const AUDIO_RATE: usize = 16_000;
/// The longest clip served: 30 s, the processor's `max_length`.
pub const AUDIO_MAX_SAMPLES: usize = 30 * AUDIO_RATE;
const HOP: usize = 160;
const WIN: usize = 320;
const BLOCKS: usize = 12;
const FF: usize = 4 * EG2A_WIDTH;
const OUT: usize = 1536;
const TEXT: usize = 512;
const EPS: f32 = 1e-6;

/// Frontend frames of an `n`-sample clip: the processor's unfold of 321
/// samples every 160 over the clip left-padded by 160.
pub fn audio_frames(n: usize) -> Result<usize, String> {
    if n > AUDIO_MAX_SAMPLES {
        return Err(format!(
            "an audio clip is at most 30 s ({AUDIO_MAX_SAMPLES} samples at 16 kHz); this one has {n} - split it into several parts"
        ));
    }
    if n + HOP < WIN + 1 {
        return Err(format!(
            "an audio clip of {n} samples is shorter than one 20 ms frame"
        ));
    }
    Ok((n + HOP - (WIN + 1)) / HOP + 1)
}

/// Soft tokens an `n`-sample clip takes: the frames through two stride-2
/// convs (`ceil` twice) - the processor's `replace_audio_token`.
pub fn audio_tokens(n: usize) -> Result<usize, String> {
    let f = audio_frames(n)?;
    Ok(((f - 1) / 2) / 2 + 1)
}

/// A clipped linear: the f16 plane `[in, out]` and its clamp bounds.
struct Lin {
    w: HalfTensor,
    inp: (f32, f32),
    out: (f32, f32),
}

struct Block {
    ffn1: (CudaSlice<f32>, Lin, Lin, CudaSlice<f32>),
    attn_pre: CudaSlice<f32>,
    /// q|k|v fused `[1024, 3072]`, one input clamp (the three share it).
    qkv: HalfTensor,
    qkv_in: (f32, f32),
    /// q, k, v output clamps and the output linear's input clamp.
    lims: [f32; 8],
    attn_o: Lin,
    attn_post: CudaSlice<f32>,
    /// The layer's relative keys `[13][1024]`, row p at distance 12 - p.
    rel: CudaSlice<f32>,
    /// softplus(per_dim_scale) `[128]` (the GGUF stores it folded).
    pds: CudaSlice<f32>,
    conv_pre: CudaSlice<f32>,
    pw1: Lin,
    dw: CudaSlice<f32>,
    conv_norm: CudaSlice<f32>,
    pw2: Lin,
    ffn2: (CudaSlice<f32>, Lin, Lin, CudaSlice<f32>),
    ln2: CudaSlice<f32>,
}

/// One clip's rows `[tokens][512]`, ready for the text model.
pub struct AudioOutput {
    pub embd: CudaSlice<f32>,
    pub n_tokens: usize,
}

pub struct AudioTower {
    exec: Arc<GpuExecutor>,
    mel: Eg2aMelTables,
    conv: [(CudaSlice<f32>, CudaSlice<f32>); 2],
    in_proj: CudaSlice<f32>,
    blocks: Vec<Block>,
    out_proj: HalfTensor,
    out_bias: CudaSlice<f32>,
    mm_proj: HalfTensor,
    llama_floor: bool,
    weight_bytes: usize,
}

impl AudioTower {
    /// The gemma4a tower out of an EmbeddingGemma 2 mmproj.
    pub fn load(exec: Arc<GpuExecutor>, map: &MappedGguf) -> Result<Self, GpuModelError> {
        let meta = |k: &str| {
            map.gguf()
                .metadata
                .get(k)
                .and_then(paddock_models::gguf::Value::as_u64)
        };
        let shape = (
            meta("clip.audio.block_count"),
            meta("clip.audio.embedding_length"),
            meta("clip.audio.attention.head_count"),
            meta("clip.audio.num_mel_bins"),
            meta("clip.audio.projection_dim"),
        );
        if shape != (Some(12), Some(1024), Some(8), Some(128), Some(512)) {
            return Err(GpuModelError::MissingMeta(format!(
                "the gemma4a tower this lane serves is 12 x 1024 wide, 8 heads, 128 mel, into 512 (the mmproj says {shape:?})"
            )));
        }
        if !exec.has_eg2_audio() {
            return Err(GpuError::MissingOp("eg2a_* (slots 813-819)").into());
        }
        let scalar = |name: String| -> Result<f32, GpuModelError> {
            let (v, _) = host_f32(map, &name)?;
            v.first()
                .copied()
                .ok_or_else(|| GpuModelError::MissingMeta(format!("{name} is empty")))
        };
        let f32s = |name: &str| -> Result<CudaSlice<f32>, GpuModelError> {
            Ok(exec.to_device(&host_f32(map, name)?.0)?)
        };
        let lin = |name: String| -> Result<Lin, GpuModelError> {
            Ok(Lin {
                w: exec.upload_f16(map, &format!("{name}.weight"))?,
                inp: (
                    scalar(format!("{name}.input_min"))?,
                    scalar(format!("{name}.input_max"))?,
                ),
                out: (
                    scalar(format!("{name}.output_min"))?,
                    scalar(format!("{name}.output_max"))?,
                ),
            })
        };

        // the relative positions' sinusoid, [13][1024] = [sin | cos] of
        // position 12 - p, in f32 the way the reference builds it
        let half = EG2A_WIDTH / 2;
        let inc = (10_000f64.ln() / (half - 1) as f64) as f32;
        let inv: Vec<f32> = (0..half).map(|i| (i as f32 * -inc).exp()).collect();
        let mut pos = vec![0f32; 13 * EG2A_WIDTH];
        for p in 0..13 {
            for i in 0..half {
                let t = (12 - p) as f32 * inv[i];
                pos[p * EG2A_WIDTH + i] = t.sin();
                pos[p * EG2A_WIDTH + half + i] = t.cos();
            }
        }
        let pos = exec.to_device(&pos)?;

        let mut blocks = Vec::with_capacity(BLOCKS);
        for b in 0..BLOCKS {
            let p = |s: &str| format!("a.blk.{b}.{s}");
            let ffn =
                |sfx: &str| -> Result<(CudaSlice<f32>, Lin, Lin, CudaSlice<f32>), GpuModelError> {
                    Ok((
                        f32s(&p(&format!("ffn_norm{sfx}.weight")))?,
                        lin(p(&format!("ffn_up{sfx}")))?,
                        lin(p(&format!("ffn_down{sfx}")))?,
                        f32s(&p(&format!("ffn_post_norm{sfx}.weight")))?,
                    ))
                };
            let (q, k, v) = (lin(p("attn_q"))?, lin(p("attn_k"))?, lin(p("attn_v"))?);
            if q.inp != k.inp || q.inp != v.inp {
                return Err(GpuModelError::MissingMeta(format!(
                    "block {b}: q/k/v input clamps differ; this lane fuses them into one GEMM"
                )));
            }
            // q|k|v rows stacked: the planes are [out][in], so the fused
            // plane is the three one after another
            let mut fused = Vec::with_capacity(3 * EG2A_WIDTH * EG2A_WIDTH);
            for s in ["attn_q", "attn_k", "attn_v"] {
                fused.extend(host_f32(map, &p(&format!("{s}.weight")))?.0);
            }
            let qkv = HalfTensor {
                buf: exec.to_device_f16(&fused, "a.blk.attn_qkv")?,
                dims: vec![EG2A_WIDTH, 3 * EG2A_WIDTH],
            };
            let attn_o = lin(p("attn_out"))?;
            // relative keys: r = W_rel . sinusoid, once, in f32
            let w_rel = f32s(&p("attn_k_rel.weight"))?;
            let mut rel = exec.alloc(13 * EG2A_WIDTH)?;
            exec.gemm_f32(&w_rel, EG2A_WIDTH, EG2A_WIDTH, &pos, &mut rel, 13)?;
            blocks.push(Block {
                ffn1: ffn("")?,
                attn_pre: f32s(&p("attn_pre_norm.weight"))?,
                lims: [
                    q.out.0,
                    q.out.1,
                    k.out.0,
                    k.out.1,
                    v.out.0,
                    v.out.1,
                    attn_o.inp.0,
                    attn_o.inp.1,
                ],
                qkv_in: q.inp,
                qkv,
                attn_o,
                attn_post: f32s(&p("attn_post_norm.weight"))?,
                rel,
                pds: f32s(&p("per_dim_scale.weight"))?,
                // llama.cpp's converter swaps the two conv norms' names:
                // `conv_norm` is the module's pre-norm, `norm_conv` the one
                // after the depthwise conv (its clip loader reads them so)
                conv_pre: f32s(&p("conv_norm.weight"))?,
                pw1: lin(p("conv_pw1"))?,
                dw: f32s(&p("conv_dw.weight"))?,
                conv_norm: f32s(&p("norm_conv.weight"))?,
                pw2: lin(p("conv_pw2"))?,
                ffn2: ffn("_1")?,
                ln2: f32s(&p("ln2.weight"))?,
            });
        }

        // frontend tables: periodic Hann, HTK filterbank and its spans, twiddles
        let window: Vec<f32> = crate::audio::dsp::hann_periodic(WIN)
            .iter()
            .map(|&v| v as f32)
            .collect();
        let fb: Vec<f32> = crate::audio::dsp::mel_filterbank(
            EG2A_MEL,
            512,
            AUDIO_RATE as f64,
            0.0,
            8000.0,
            crate::audio::dsp::MelScale::Htk,
            false,
        )
        .iter()
        .map(|&v| v as f32)
        .collect();
        let spans: Vec<u32> = fb
            .chunks(257)
            .flat_map(|row| {
                let first = row.iter().position(|&v| v != 0.).unwrap_or(0) as u32;
                let end = row.iter().rposition(|&v| v != 0.).map_or(0, |i| i + 1) as u32;
                [first, end.max(first)]
            })
            .collect();
        let twiddle: Vec<f32> = (0..256)
            .flat_map(|i| {
                let (s, c) = (-2.0 * std::f64::consts::PI * i as f64 / 512.).sin_cos();
                [c as f32, s as f32]
            })
            .collect();
        let mel = Eg2aMelTables {
            window: exec.to_device(&window)?,
            fb: exec.to_device(&fb)?,
            spans: exec.to_device_u32(&spans)?,
            twiddle: exec.to_device(&twiddle)?,
        };

        let conv = [
            (f32s("a.conv1d.0.weight")?, f32s("a.conv1d.0.norm.weight")?),
            (f32s("a.conv1d.1.weight")?, f32s("a.conv1d.1.norm.weight")?),
        ];
        let in_proj = f32s("a.input_projection.weight")?;
        let out_proj = exec.upload_f16(map, "a.pre_encode.out.weight")?;
        let out_bias = f32s("a.pre_encode.out.bias")?;
        let mm_proj = exec.upload_f16(map, "mm.a.input_projection.weight")?;
        if out_proj.dims != [EG2A_WIDTH, OUT] || mm_proj.dims != [OUT, TEXT] {
            return Err(GpuModelError::MissingMeta(format!(
                "audio output projections {:?} / {:?}",
                out_proj.dims, mm_proj.dims
            )));
        }
        let weight_bytes = blocks
            .iter()
            .map(|b| {
                b.qkv.bytes()
                    + b.attn_o.w.bytes()
                    + b.pw1.w.bytes()
                    + b.pw2.w.bytes()
                    + [&b.ffn1, &b.ffn2]
                        .iter()
                        .map(|f| f.1.w.bytes() + f.2.w.bytes())
                        .sum::<usize>()
            })
            .sum::<usize>()
            + out_proj.bytes()
            + mm_proj.bytes()
            + EG2A_WIDTH * EG2A_WIDTH * 4;
        exec.synchronize()?;
        Ok(Self {
            exec,
            mel,
            conv,
            in_proj,
            blocks,
            out_proj,
            out_bias,
            mm_proj,
            llama_floor: false,
            weight_bytes,
        })
    }

    /// Resident bytes of the GEMM planes.
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// Test instrument: llama.cpp's `ln(max(mel, 1e-3))` frontend floor in
    /// place of the processor's `ln(mel + 1e-3)` (see the module note).
    pub fn set_llama_floor(&mut self, on: bool) {
        self.llama_floor = on;
    }

    /// Test instrument: the frontend's log-mel `[frames][128]` of a clip.
    pub fn log_mel(&self, samples: &[f32]) -> Result<Vec<f32>, GpuError> {
        let e = &self.exec;
        let frames = audio_frames(samples.len()).map_err(GpuError::Driver)?;
        let pcm = e.to_device(samples)?;
        let mut mel = e.alloc(frames * EG2A_MEL)?;
        e.eg2a_mel(
            &pcm,
            samples.len(),
            &self.mel,
            &mut mel,
            frames,
            self.llama_floor,
        )?;
        e.to_host_len(&mel, frames * EG2A_MEL)
    }

    /// One clip (mono 16 kHz) -> its rows `[tokens][512]`.
    pub fn encode(&self, samples: &[f32]) -> Result<AudioOutput, GpuError> {
        let e = &self.exec;
        let frames = audio_frames(samples.len()).map_err(GpuError::Driver)?;
        let pcm = e.to_device(samples)?;
        let mut mel = e.alloc(frames * EG2A_MEL)?;
        e.eg2a_mel(
            &pcm,
            samples.len(),
            &self.mel,
            &mut mel,
            frames,
            self.llama_floor,
        )?;
        // two stride-2 convs: [frames][128][1] -> [t1][64][128] -> [t][32][32]
        let t1 = (frames - 1) / 2 + 1;
        let t = (t1 - 1) / 2 + 1;
        let mut c0 = e.alloc(t1 * 64 * 128)?;
        e.eg2a_sscp(
            &mel,
            &self.conv[0].0,
            &self.conv[0].1,
            &mut c0,
            frames,
            EG2A_MEL,
            1,
            128,
            EPS,
        )?;
        let mut c1 = e.alloc(t * EG2A_WIDTH)?;
        e.eg2a_sscp(
            &c0,
            &self.conv[1].0,
            &self.conv[1].1,
            &mut c1,
            t1,
            64,
            128,
            32,
            EPS,
        )?;
        drop((mel, c0, pcm));
        let mut x = e.alloc(t * EG2A_WIDTH)?;
        e.gemm_f32(&self.in_proj, EG2A_WIDTH, EG2A_WIDTH, &c1, &mut x, t)?;

        let mut x16 = e.alloc_f16(t * FF)?;
        let mut h16 = e.alloc_f16(t * FF)?;
        let mut wide = e.alloc(t * FF)?;
        let mut y = e.alloc(t * EG2A_WIDTH)?;
        let first = &self.blocks[0];
        e.eg2a_rows(
            &mut x,
            Eg2aSeam {
                y: None,
                out_w: None,
                next: Some((true, Some(&first.ffn1.0), first.ffn1.1.inp, &mut x16)),
            },
            t,
            EPS,
        )?;
        for (i, b) in self.blocks.iter().enumerate() {
            // FFN1, half step; then the attention's pre-norm
            self.ffn(&b.ffn1, &mut x16, &mut h16, &mut wide, &mut y, t)?;
            e.eg2a_rows(
                &mut x,
                Eg2aSeam {
                    y: Some((&y, b.ffn1.2.out, Some(&b.ffn1.3), 0.5)),
                    out_w: None,
                    next: Some((true, Some(&b.attn_pre), b.qkv_in, &mut x16)),
                },
                t,
                EPS,
            )?;
            // attention; then the conv module's pre-norm
            e.matvec_batch_f16(&b.qkv, &x16, &mut wide, t)?;
            e.eg2a_attn(
                &wide, &b.rel, &b.pds, &mut h16, t, &b.lims, Q_SCALE, K_SCALE, 50.0,
            )?;
            e.matvec_batch_f16(&b.attn_o.w, &h16, &mut y, t)?;
            e.eg2a_rows(
                &mut x,
                Eg2aSeam {
                    y: Some((&y, b.attn_o.out, Some(&b.attn_post), 1.0)),
                    out_w: None,
                    next: Some((true, Some(&b.conv_pre), b.pw1.inp, &mut x16)),
                },
                t,
                EPS,
            )?;
            // light conv; then FFN2's pre-norm
            e.matvec_batch_f16(&b.pw1.w, &x16, &mut wide, t)?;
            e.eg2a_conv(
                &wide,
                &b.dw,
                &b.conv_norm,
                &mut h16,
                t,
                b.pw1.out,
                b.pw2.inp,
                EPS,
            )?;
            e.matvec_batch_f16(&b.pw2.w, &h16, &mut y, t)?;
            e.eg2a_rows(
                &mut x,
                Eg2aSeam {
                    y: Some((&y, b.pw2.out, None, 1.0)),
                    out_w: None,
                    next: Some((true, Some(&b.ffn2.0), b.ffn2.1.inp, &mut x16)),
                },
                t,
                EPS,
            )?;
            // FFN2, half step, the block's closing norm; then the next
            // block's FFN1 pre-norm, or the output projection's plain input
            self.ffn(&b.ffn2, &mut x16, &mut h16, &mut wide, &mut y, t)?;
            let next = match self.blocks.get(i + 1) {
                Some(n) => (true, Some(&n.ffn1.0), n.ffn1.1.inp),
                None => (false, None, (f32::NEG_INFINITY, f32::INFINITY)),
            };
            e.eg2a_rows(
                &mut x,
                Eg2aSeam {
                    y: Some((&y, b.ffn2.2.out, Some(&b.ffn2.3), 0.5)),
                    out_w: Some(&b.ln2),
                    next: Some((next.0, next.1, next.2, &mut x16)),
                },
                t,
                EPS,
            )?;
        }
        // 1024 -> 1536 + bias, the embedder's weightless norm, 1536 -> 512
        let mut o = e.alloc(t * OUT)?;
        e.matvec_batch_f16(&self.out_proj, &x16, &mut o, t)?;
        e.eg2a_out(&o, &self.out_bias, &mut h16, t, OUT, EPS)?;
        let mut embd = e.alloc(t * TEXT)?;
        e.matvec_batch_f16(&self.mm_proj, &h16, &mut embd, t)?;
        Ok(AudioOutput { embd, n_tokens: t })
    }

    /// A half-step FFN's GEMMs: `x16` (its clamped pre-normed input) ->
    /// up -> clamp, SiLU, clamp -> down into `y` (unclamped; the seam
    /// clamps it).
    fn ffn(
        &self,
        f: &(CudaSlice<f32>, Lin, Lin, CudaSlice<f32>),
        x16: &mut CudaSlice<f16>,
        h16: &mut CudaSlice<f16>,
        wide: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        t: usize,
    ) -> Result<(), GpuError> {
        let e = &self.exec;
        e.matvec_batch_f16(&f.1.w, x16, wide, t)?;
        e.eg2a_act(wide, h16, t * FF, f.1.out, true, f.2.inp)?;
        e.matvec_batch_f16(&f.2.w, h16, y, t)
    }
}

/// head_dim^-0.5 / ln 2 and ln(1 + e) / ln 2, the reference's scalings.
const Q_SCALE: f32 = (0.088_388_347_648_318_44 / std::f64::consts::LN_2) as f32;
const K_SCALE: f32 = (1.313_261_687_518_222_8 / std::f64::consts::LN_2) as f32;
