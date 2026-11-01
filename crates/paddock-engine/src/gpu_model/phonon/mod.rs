//! Gradium Phonon 31M English on CUDA: FlowLM (a 6-layer 512-wide causal
//! transformer over the voice prompt, the text and the generated frames, an
//! EOS head and a one-step LSD flow head that turns scaled noise into each
//! 80 ms frame's 32-d latent) and the streaming Mimi decoder (a 2-layer
//! windowed transformer at 200 Hz, then SEANet: conv, three ELU / transposed
//! conv / residual stages x6 x5 x4, conv) - 1920 samples a frame at 24 kHz.
//!
//! **Numerics.** F32 class throughout, gated against the model's F64
//! equations: FlowLM's BF16 matrices widen exactly, products, sums and
//! activations are F32, Mimi is F32. The multi-row passes (the prompt, the
//! voice prompt, Mimi's transformer and every Mimi convolution as an im2col
//! GEMM) run on the pack's F32-class tensor-core GEMMs - 722 for FlowLM's
//! stored BF16 planes, 698's 3xTF32 for Mimi's F32 ones - and the one-row
//! frame step on Phonon's GEMV (slot 908). Every kernel's per-element
//! sequence is fixed by shape, so a stream decoded in batches of any size is
//! the frame-by-frame stream to the bit.
//!
//! **The frame step** is ~55 MB of BF16 weights for 80 ms of audio: a weight
//! stream with a 46-pass dependency chain. Its GEMVs carry the input
//! LayerNorm / adaLN modulation in their prologue and the residual, gate,
//! activation - and for the qkv projection the rope and K/V cache writes -
//! in their epilogue, load their weights ahead of the wait on the previous
//! pass (every pass a programmatic dependent), and keep the flow head's
//! ~17 MB resident in L2 across frames while the transformer's weights and
//! the K/V cache stream past it.
//!
//! **State.** The voice prompt's K/V is computed once at load into the
//! FlowLM caches' first rows, which no request writes - a request only
//! resets its position. Mimi's transformer caches are rings (window 251 in a
//! 512-row ring) and its convolutions keep their earlier raw input rows.
//! Positions live in device memory, so every pass is position-free: one
//! captured graph per batch size samples a batch's frames, one decodes it,
//! and a request is a prompt pass plus graph replays.
//!
//! **Two lanes.** FlowLM samples on the executor's stream (the greatest
//! priority); Mimi decodes on a forked lane at the least priority, with its
//! own transformer planes. Batch b's decode runs beside batch b + 1's
//! sampling - a weight-stream chain of short kernels beside a few compute-
//! bound GEMMs, so the decoder fills the SMs the chain leaves idle. The
//! lanes meet at two event-guarded points: the latents hand-off (FlowLM
//! writes it only once Mimi has taken the last one) and the PCM, double
//! buffered so batch b's audio is read while batch b + 1 decodes. The
//! request's first frame is decoded before anything else is launched, so
//! nothing competes with the first audio.
//!
//! The request lifecycle (bounded queue, cancellation, seeded noise, the stop
//! rule) is `crate::phonon`'s, shared with Metal.

mod codec;
mod frame;
mod load;

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::sys::CUstreamCaptureMode;
use cudarc::driver::{CudaEvent, CudaSlice};
use paddock_models::phonon::{FLOW_CACHE, FRAME_SAMPLES, MAX_FRAMES, MAX_TEXT_TOKENS};
use paddock_tokenizer::GgufTokenizer;

use crate::gpu::{CapturedGraph, GpuError, GpuExecutor, PhEpi, PhExtra, PhPro, PhWeight};
use crate::phonon::{Backend, Noise, Options};

/// Frames a decoder pass carries (the Metal lane's batch, so the stop
/// boundary falls the same way).
pub const CODEC_BATCH: usize = 8;
const D: usize = 512;
const FF: usize = 2048;
/// The transformers' LayerNorm epsilon, and the flow head's.
const LN_EPS: f32 = 1e-5;
const FLOW_EPS: f32 = 1e-6;
/// The stacked adaLN modulations: six res blocks (shift, scale, gate) and
/// the final layer (shift, scale).
const BLOCK_MOD: usize = 3 * D;
const MODULATION: usize = 6 * BLOCK_MOD + 2 * D;
/// A frame's conditioning row: cond_embed's 512 outputs, the EOS logit
/// beside them, padded to a 16 B row.
const COND_STRIDE: usize = D + 4;
/// Mimi's transformer ring and its attention window.
const CODEC_RING: usize = 512;
const CODEC_WINDOW: usize = 251;
/// Attention splits (the kernel's 64 keys each): FlowLM's whole cache, and
/// Mimi's window wherever it starts.
const FLOW_SPLITS: usize = FLOW_CACHE.div_ceil(64);
const CODEC_SPLITS: usize = CODEC_WINDOW.div_ceil(64);
/// Mimi rows a frame at its transformer (200 Hz).
const CODEC_ROWS: usize = 16;
/// The longest pass the transformer planes carry: the prompt (text + BOS);
/// the voice prompt (at most 250 frames) and a decoder batch are shorter.
const MAX_ROWS: usize = MAX_TEXT_TOKENS + 1;

/// One transformer layer's planes (FlowLM BF16, Mimi F32; no biases).
struct Layer {
    norm1: (CudaSlice<f32>, CudaSlice<f32>),
    norm2: (CudaSlice<f32>, CudaSlice<f32>),
    in_proj: PhWeight,
    out_proj: PhWeight,
    linear1: PhWeight,
    linear2: PhWeight,
    /// Mimi's layer scales (FlowLM has none)
    scale: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
}

/// A layer's K/V, `[cap][512]` f32 (a ring for Mimi, linear for FlowLM).
struct Cache {
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
}

/// A linear with its bias.
struct Lin {
    w: PhWeight,
    b: Option<CudaSlice<f32>>,
}

/// A flow-head residual block.
struct Block {
    ln: (CudaSlice<f32>, CudaSlice<f32>),
    mlp0: Lin,
    mlp2: Lin,
}

/// FlowLM: transformer, output heads, flow head and the per-model constants.
struct FlowLm {
    layers: Vec<Layer>,
    caches: Vec<Cache>,
    out_norm: (CudaSlice<f32>, CudaSlice<f32>),
    /// `input_linear` with the speaker/duration condition as its bias - the
    /// reference adds the condition to the bias-free product, the same one
    /// rounding
    input_linear: Lin,
    input_proj: Lin,
    /// cond_embed's rows then out_eos's: one product, SiLU on the first 512
    /// outputs (cond_embed's bias carries the time embedding), the EOS logit
    /// plain at 512
    cond_eos: Lin,
    modulation: Lin,
    blocks: Vec<Block>,
    final_linear: Lin,
    /// the text embedding table `[4001][512]`
    embed: CudaSlice<f32>,
    bos: CudaSlice<f32>,
    /// the projected voice prompt `[voice_rows][512]` (its K/V is cached)
    voice: CudaSlice<f32>,
    voice_rows: usize,
    /// gate probes of the load-time constants
    condition: CudaSlice<f32>,
    time: CudaSlice<f32>,
}

/// How a Mimi convolution runs - elected per layer at load (by shape, never
/// by batch size, so a stream's samples do not depend on how it is cut).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvPath {
    /// the k-times expanded im2col plane, then the 3xTF32 GEMM (wide inputs,
    /// where the plane is small beside the GEMM's work)
    Im2col,
    /// `[hist ; act(x)]` stacked once, the 3xTF32 GEMM reading its im2col
    /// rows in place (slot 917) - the same bits as Im2col, without the plane
    Rows,
    /// a few outputs over many rows (the mono output): straight from the
    /// staged rows (slot 916)
    Direct,
}

/// A Mimi convolution as a GEMM over its im2col rows: weights `[N][K]` (a
/// plain conv `[cout][k * cin]`, a transposed one phase-packed `[s *
/// cout][2 cin]`), bias `[N]`, the raw input history it keeps.
struct Conv {
    path: ConvPath,
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    hist: CudaSlice<f32>,
    cin: usize,
    cout: usize,
    /// taps over the input rows (a transposed conv: 2)
    k: usize,
    /// output rows an input row (a plain conv: 1)
    stride: usize,
}

/// The Mimi decoder.
struct Mimi {
    /// `[std ; mean]` of the latents (32 each)
    stats: CudaSlice<f32>,
    quant_proj: CudaSlice<f32>,
    upsample: CudaSlice<f32>,
    up_hist: CudaSlice<f32>,
    layers: Vec<Layer>,
    caches: Vec<Cache>,
    convs: Vec<Conv>,
}

/// One transformer pass's planes and the position its rows start at:
/// FlowLM's sized for the prompt, Mimi's for a decoder batch (128 rows).
struct Tx {
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    att: CudaSlice<f32>,
    part: CudaSlice<f32>,
    ctr: CudaSlice<u32>,
    wide: CudaSlice<f32>,
    pos: CudaSlice<u32>,
}

/// FlowLM's scratch, sized once at load (a request allocates nothing).
struct Work {
    tx: Tx,
    /// `[CODEC_BATCH][COND_STRIDE]`: a frame's conditioning row + EOS
    cond: CudaSlice<f32>,
    modulation: CudaSlice<f32>,
    /// `[CODEC_BATCH][512]`: the flow state of each frame of a batch
    flow: CudaSlice<f32>,
    tmp: CudaSlice<f32>,
    noise: CudaSlice<f32>,
    latents: CudaSlice<f32>,
    /// the latent before the batch's first frame
    prev: CudaSlice<f32>,
    ids: CudaSlice<u32>,
    /// the batch's latents as handed to the decoder lane
    handoff: CudaSlice<f32>,
}

/// Mimi's scratch on its own lane.
struct CodecWork {
    tx: Tx,
    /// the latents being decoded (taken over from `Work::handoff`)
    lat: CudaSlice<f32>,
    denorm: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    p1: CudaSlice<f32>,
    p2: CudaSlice<f32>,
    p3: CudaSlice<f32>,
    stage: CudaSlice<f32>,
    /// a batch's PCM, double buffered between the decoder and the reader
    pcm: [CudaSlice<f32>; 2],
}

/// A sampling graph's shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Pass {
    /// the request's first frame (sampled from the prompt's last row)
    first: bool,
    frames: usize,
}

/// A decoded batch on its way to the reader: its PCM buffer, the event
/// that fires once it is written, and how many of its frames are speech.
struct Decoded {
    done: CudaEvent,
    buf: usize,
    frames: usize,
}

pub struct GpuPhonon {
    exec: Arc<GpuExecutor>,
    /// the decoder lane: the executor forked onto a least-priority stream
    codec: Arc<GpuExecutor>,
    tokenizer: GgufTokenizer,
    flow: FlowLm,
    mimi: Mimi,
    ws: Work,
    cw: CodecWork,
    sample_graphs: HashMap<Pass, CapturedGraph>,
    decode_graphs: HashMap<usize, CapturedGraph>,
    /// the hand-off of the batch sampled last (the decoder lane waits on it)
    handed: Option<CudaEvent>,
    /// the decoder lane has taken the last hand-off (FlowLM may write the
    /// next one)
    taken: Option<CudaEvent>,
    /// the PCM buffer the next decode writes
    pcm_next: usize,
    /// host mirrors of the device positions: the next FlowLM row, the next
    /// Mimi transformer row
    flow_pos: usize,
    codec_pos: usize,
    weight_bytes: u64,
    workspace_bytes: u64,
}

impl GpuPhonon {
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    /// Start a request: its text rows then BOS through FlowLM after the
    /// cached voice prompt. The prompt's last row is copied to row 0 of
    /// FlowLM's `x`, where every frame's flow head reads.
    pub fn reset(&mut self, text: &str) -> Result<(), String> {
        let ids = self
            .tokenizer
            .encode(text.trim())
            .map_err(|e| format!("Phonon: {e}"))?;
        // the tokenizer has 4000 entries; embedding row 4000 is conditioner
        // padding, never a text token
        if ids.is_empty() || ids.len() > MAX_TEXT_TOKENS || ids.iter().any(|&id| id >= 4000) {
            return Err(format!(
                "input must tokenize to 1-{MAX_TEXT_TOKENS} text tokens"
            ));
        }
        self.reset_ids(&ids).map_err(|e| format!("Phonon: {e}"))
    }

    fn reset_ids(&mut self, ids: &[u32]) -> Result<(), GpuError> {
        let exec = self.exec.clone();
        let n = ids.len();
        exec.upload_u32(ids, &mut self.ws.ids)?;
        exec.embed_gather_batch(&self.flow.embed, &self.ws.ids, &mut self.ws.tx.x, D, n)?;
        exec.copy_region(&self.flow.bos, 0, &mut self.ws.tx.x, n * D, D)?;
        self.flow_pos = self.flow.voice_rows;
        self.codec_pos = 0;
        exec.upload_u32(&[self.flow_pos as u32], &mut self.ws.tx.pos)?;
        // the decoder's state, on its own lane (after any decode a cancelled
        // request left there)
        let codec = &self.codec;
        let m = &mut self.mimi;
        codec.zero_region(&mut m.up_hist, 0, D)?;
        for c in &mut m.convs {
            let len = c.hist.len();
            codec.zero_region(&mut c.hist, 0, len)?;
        }
        self.wide_flow(n + 1)?;
        self.flow_pos += n + 1;
        exec.copy_region(&self.ws.tx.x, n * D, &mut self.ws.tmp, 0, D)?;
        exec.copy_region(&self.ws.tmp, 0, &mut self.ws.tx.x, 0, D)
    }

    /// Run a pass from its captured graph (capturing it on first use) or as
    /// plain launches; `codec` picks the lane the pass and its graph live on.
    fn replay(
        &mut self,
        graph: bool,
        codec: bool,
        key: Pass,
        pass: impl FnOnce(&mut Self) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        if !graph {
            return pass(self);
        }
        let have = if codec {
            self.decode_graphs.contains_key(&key.frames)
        } else {
            self.sample_graphs.contains_key(&key)
        };
        if !have {
            let lane = if codec {
                self.codec.clone()
            } else {
                self.exec.clone()
            };
            lane.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| GpuError::Driver(format!("Phonon begin_capture: {e}")))?;
            let rec = pass(self);
            let graph = crate::gpu::end_capture_no_flags(&lane.stream)
                .map_err(|e| GpuError::Driver(format!("Phonon end_capture: {e}")));
            rec?;
            let graph = graph?
                .ok_or_else(|| GpuError::Driver("Phonon capture produced no graph".into()))?;
            if codec {
                self.decode_graphs.insert(key.frames, graph);
            } else {
                self.sample_graphs.insert(key, graph);
            }
        }
        let graph = if codec {
            &self.decode_graphs[&key.frames]
        } else {
            &self.sample_graphs[&key]
        };
        graph
            .launch()
            .map_err(|e| GpuError::Driver(format!("Phonon graph launch: {e}")))
    }

    /// Sample `noises.len()` frames on FlowLM's lane and hand their latents
    /// over; nothing waits on the host.
    fn launch_sample(
        &mut self,
        noises: &[[f32; 32]],
        first: bool,
        graph: bool,
    ) -> Result<(), GpuError> {
        let n = noises.len();
        self.exec
            .upload_f32(noises.as_flattened(), &mut self.ws.noise)?;
        self.exec
            .upload_u32(&[self.flow_pos as u32], &mut self.ws.tx.pos)?;
        self.replay(graph, false, Pass { first, frames: n }, |m| {
            m.sample_pass(n, first)
        })?;
        // the hand-off: only once the decoder lane has taken the last one
        if let Some(ev) = self.taken.take() {
            self.exec.wait_event(&ev)?;
        }
        self.exec
            .copy_region(&self.ws.latents, 0, &mut self.ws.handoff, 0, n * 32)?;
        self.handed = Some(self.exec.record_event()?);
        if !first {
            self.flow_pos += n;
        }
        Ok(())
    }

    /// The last sampled batch's EOS logits (waits for its sampling).
    fn read_eos(&self, n: usize) -> Result<Vec<f32>, GpuError> {
        let cond = self.exec.to_host_len(&self.ws.cond, n * COND_STRIDE)?;
        Ok((0..n).map(|i| cond[i * COND_STRIDE + D]).collect())
    }

    /// Decode the last hand-off's first `frames` latents on the decoder lane
    /// into the next PCM buffer; returns the batch's ticket.
    fn launch_decode(&mut self, frames: usize, graph: bool) -> Result<Decoded, GpuError> {
        let codec = self.codec.clone();
        let handed = self
            .handed
            .take()
            .ok_or_else(|| GpuError::Driver("Phonon: a decode with nothing sampled".into()))?;
        codec.wait_event(&handed)?;
        codec.copy_region(&self.ws.handoff, 0, &mut self.cw.lat, 0, frames * 32)?;
        self.taken = Some(codec.record_event()?);
        codec.upload_u32(&[self.codec_pos as u32], &mut self.cw.tx.pos)?;
        self.replay(
            graph,
            true,
            Pass {
                first: false,
                frames,
            },
            |m| m.codec_pass(frames),
        )?;
        let buf = self.pcm_next;
        self.pcm_next ^= 1;
        codec.copy_region(
            &self.cw.p1,
            0,
            &mut self.cw.pcm[buf],
            0,
            frames * FRAME_SAMPLES,
        )?;
        self.codec_pos += frames * CODEC_ROWS;
        Ok(Decoded {
            done: codec.record_event()?,
            buf,
            frames,
        })
    }

    /// A decoded batch's speech frames (waits for that batch only).
    fn read_pcm(&self, d: &Decoded) -> Result<Vec<f32>, GpuError> {
        self.codec
            .to_host_len_after(&d.done, &self.cw.pcm[d.buf], d.frames * FRAME_SAMPLES)
    }

    /// `noises.len()` FlowLM frames without decoding (gate probe, plain
    /// launches); returns their EOS logits. `first` marks the request's
    /// first frame, sampled alone.
    pub fn sample_batch(&mut self, noises: &[[f32; 32]], first: bool) -> Result<Vec<f32>, String> {
        let n = noises.len();
        if !(1..=CODEC_BATCH).contains(&n) || (first && n != 1) {
            return Err("Phonon: frame batch out of range".into());
        }
        let run = |m: &mut Self| -> Result<Vec<f32>, GpuError> {
            m.launch_sample(noises, first, false)?;
            m.read_eos(n)
        };
        let eos = run(self).map_err(|e| format!("Phonon: {e}"))?;
        check_finite(&eos, &[])?;
        Ok(eos)
    }

    /// Decode the last sampled batch's first `frames` latents (gate probe,
    /// plain launches).
    pub fn decode_batch(&mut self, frames: usize) -> Result<Vec<f32>, String> {
        if !(1..=CODEC_BATCH).contains(&frames) {
            return Err("Phonon: decoder batch out of range".into());
        }
        let run = |m: &mut Self| -> Result<Vec<f32>, GpuError> {
            let d = m.launch_decode(frames, false)?;
            m.read_pcm(&d)
        };
        let pcm = run(self).map_err(|e| format!("Phonon: {e}"))?;
        check_finite(&[], &pcm)?;
        Ok(pcm)
    }

    /// Replace the latent the next batch's first frame reads (gate probe:
    /// teacher forcing, the reference's latent in place of the sampled one).
    pub fn force_prev(&mut self, latent: &[f32]) -> Result<(), String> {
        if latent.len() != 32 {
            return Err("Phonon: a latent is 32 values".into());
        }
        self.exec
            .upload_f32(latent, &mut self.ws.prev)
            .map_err(|e| e.to_string())
    }

    /// The last sampled batch's first `n` latents (gate probe).
    pub fn latents(&self, n: usize) -> Result<Vec<f32>, String> {
        self.exec
            .to_host_len(&self.ws.latents, n * 32)
            .map_err(|e| e.to_string())
    }

    /// The gate probes of the load-time constants: condition, time, the
    /// projected voice prompt and BOS.
    pub fn constants(&self) -> Result<[Vec<f32>; 4], String> {
        let e = |e: GpuError| e.to_string();
        let f = &self.flow;
        Ok([
            self.exec.to_host_len(&f.condition, D).map_err(e)?,
            self.exec.to_host_len(&f.time, D).map_err(e)?,
            self.exec
                .to_host_len(&f.voice, f.voice_rows * D)
                .map_err(e)?,
            self.exec.to_host_len(&f.bos, D).map_err(e)?,
        ])
    }

    /// The gate probe of FlowLM's output: `out_norm` of the row the flow
    /// head reads (an identity product through the norm prologue).
    pub fn hidden(&mut self) -> Result<Vec<f32>, String> {
        let e = |e: GpuError| e.to_string();
        let mut eye = vec![0f32; D * D];
        for i in 0..D {
            eye[i * D + i] = 1.0;
        }
        let eye = PhWeight::F32(self.exec.to_device(&eye).map_err(e)?);
        let f = &self.flow;
        self.exec
            .ph_linear(
                &self.ws.tx.x,
                0,
                PhPro::Ln(&f.out_norm.0, &f.out_norm.1),
                &eye,
                None,
                PhEpi::None,
                &mut self.ws.tmp,
                0,
                (1, D, D),
                LN_EPS,
                PhExtra::default(),
            )
            .map_err(e)?;
        self.exec.to_host_len(&self.ws.tmp, D).map_err(e)
    }

    /// The request after its reset, batch by batch: batch b + 1 is sampled
    /// while batch b decodes and while the host reads and emits batch b's
    /// audio. Every batch decodes all the frames it sampled; the ones past
    /// the stop are dropped - the decoder is causal, so they never touch the
    /// samples before them.
    fn stream(
        &mut self,
        noise: &mut Noise,
        tail: usize,
        emit: &mut dyn FnMut(Vec<f32>) -> bool,
    ) -> Result<(), GpuError> {
        // the first frame alone, decoded before anything else is launched
        self.launch_sample(&[noise.frame()], true, true)?;
        let first = self.launch_decode(1, true)?;
        let eos = self.read_eos(1)?;
        let pcm = self.read_pcm(&first)?;
        check_finite(&eos, &pcm).map_err(GpuError::Driver)?;
        if !emit(pcm) {
            return Ok(());
        }
        let mut generated = 1;
        let mut stop = if eos[0] > 4. {
            1 + tail
        } else {
            MAX_FRAMES + 1
        };
        let mut pending: Option<Decoded> = None;
        loop {
            let next = if generated < MAX_FRAMES && generated < stop {
                let count = CODEC_BATCH
                    .min(MAX_FRAMES - generated)
                    .min(stop - generated);
                let noises: Vec<_> = (0..count).map(|_| noise.frame()).collect();
                self.launch_sample(&noises, false, true)?;
                Some((count, self.launch_decode(count, true)?))
            } else {
                None
            };
            // the previous batch's audio while this one samples and decodes
            if let Some(d) = pending.take() {
                let pcm = self.read_pcm(&d)?;
                check_finite(&[], &pcm).map_err(GpuError::Driver)?;
                for chunk in pcm.as_chunks::<FRAME_SAMPLES>().0 {
                    if !emit(chunk.to_vec()) {
                        return Ok(());
                    }
                }
            }
            let Some((count, mut d)) = next else {
                break;
            };
            let eos = self.read_eos(count)?;
            check_finite(&eos, &[]).map_err(GpuError::Driver)?;
            if let Some(i) = eos.iter().position(|&v| v > 4.) {
                stop = stop.min(generated + i + 1 + tail);
            }
            d.frames = count.min(stop - generated);
            generated += d.frames;
            pending = Some(d);
        }
        if generated != stop {
            return Err(GpuError::Driver(
                "Phonon reached its 60-second utterance limit before end of speech; split the \
                 input into shorter sentences"
                    .into(),
            ));
        }
        Ok(())
    }
}

fn check_finite(eos: &[f32], pcm: &[f32]) -> Result<(), String> {
    if eos.iter().any(|v| !v.is_finite()) {
        return Err("Phonon produced a non-finite end-of-speech logit".into());
    }
    if pcm.iter().any(|v| !v.is_finite()) {
        return Err("Phonon produced non-finite audio".into());
    }
    Ok(())
}

impl Backend for GpuPhonon {
    fn generate(
        &mut self,
        text: &str,
        options: Options,
        emit: &mut dyn FnMut(Vec<f32>) -> bool,
    ) -> Result<(), String> {
        crate::phonon::validate_text(text)?;
        options.validate()?;
        self.reset(text)?;
        let mut noise = Noise::new(options);
        let tail = if text.split_whitespace().count() <= 4 {
            3
        } else {
            1
        };
        self.stream(&mut noise, tail, emit).map_err(|e| match e {
            GpuError::Driver(m) if m.starts_with("Phonon") => m,
            e => format!("Phonon: {e}"),
        })
    }
}
