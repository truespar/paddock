//! Nemotron 3.5 ASR on the native CUDA engine: NVIDIA's cache-aware
//! streaming FastConformer encoder with language-ID prompting and an RNN-T
//! greedy decoder, served from NVIDIA's own GGUF (Q8_0 projections as
//! stored).
//!
//! One code path serves a file and a live stream: a [`Session`] takes PCM as
//! it arrives and encodes whole chunks (`lookahead + 1` encoder frames, 80 ms
//! each) in windows of up to [`MAX_WINDOW`] frames, each window's attention
//! keys reaching back through 56-slot K/V rings and its convolutions through
//! 8-slot rings; `finish` encodes the rest under the recording's end. A
//! file is a session fed at once. The model's mask and causal convolutions
//! make the windowing invisible to the arithmetic - every row is computed
//! from the same inputs in the same order wherever a window boundary falls -
//! so a stream's tokens are the file's to the bit.
//!
//! The subsampler is causal and reaches 14 mel frames back, so a window
//! recomputes those frames from the session's PCM tail instead of carrying
//! three stage caches. Precision is F32 class everywhere (the house GEMM on
//! the stored weights; nothing narrowed); the reference is the model's F64
//! evaluation on the same weights.

mod forward;

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::nemotron_asr::{
    self as arch, BINS, Checkpoint, D_FF, D_MODEL, LAYERS, MEL, PE_CENTER, PRED, PROMPT_HIDDEN,
    PROMPTS, SUB_CHANNELS, SUB_OUT, VOCAB,
};

use crate::gpu::{ClefQ8, GpuExecutor};

pub use crate::gpu_model::gpt_oss::GpuModelError;

/// Encoder frames of the largest window: 224 = 17.92 s, a whole number of
/// chunks at every published lookahead (1, 2, 4, 7, 14 frames).
pub const MAX_WINDOW: usize = 224;
/// Rows of the largest pass: the windows of several streams stacked (four
/// whole windows, or a few hundred live chunks).
pub const MAX_PASS: usize = 4 * MAX_WINDOW;
/// Streams one pass holds.
pub const MAX_SEGMENTS: usize = 256;
/// The relative-key table's distances: -13 ..= 69 (a 14-frame chunk's
/// widest spread over 56 frames of left context).
const REL_MIN: i64 = -13;
const REL_ROWS: usize = 83;
/// K/V and convolution ring slots.
const KV_SLOTS: usize = 56;
const CONV_SLOTS: usize = 8;
/// Mel frames, stem rows and stage-two rows a window of `r` frames reads.
const fn mel_rows(r: usize) -> usize {
    8 * r + 7
}
const fn stem_rows(r: usize) -> usize {
    4 * r + 3
}
const fn stage2_rows(r: usize) -> usize {
    2 * r + 1
}
/// PCM samples a window's frames read: the first frame's centered window
/// starts 256 before its hop, and the preemphasis one sample before that.
const MAX_PCM: usize = (mel_rows(MAX_WINDOW) - 1) * arch::HOP + 512 + 2;

struct Norm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

struct Layer {
    ff1_norm: Norm,
    ff1_up: ClefQ8,
    ff1_down: ClefQ8,
    att_norm: Norm,
    /// q | k | v rows, one product
    qkv: ClefQ8,
    out: ClefQ8,
    bias_u: CudaSlice<f32>,
    bias_v: CudaSlice<f32>,
    /// `[83][1024]`: the position projection of the stored table's rows at
    /// distances -13 ..= 69
    rel: CudaSlice<f32>,
    conv_norm: Norm,
    /// pointwise_conv1 with rows interleaved (2j value j, 2j + 1 its gate)
    pw1: ClefQ8,
    /// `[1024][9]` widened from F16
    dw: CudaSlice<f32>,
    dw_norm: Norm,
    pw2: ClefQ8,
    ff2_norm: Norm,
    ff2_up: ClefQ8,
    ff2_down: ClefQ8,
    out_norm: Norm,
}

/// The pass planes, sized once: the subsampler's for one stream's
/// [`MAX_WINDOW`]-frame window at a time, the rest for [`MAX_PASS`] rows.
struct Workspace {
    pcm: CudaSlice<f32>,
    mel: CudaSlice<f32>,
    s1: CudaSlice<f32>,
    s2a: CudaSlice<f32>,
    s2b: CudaSlice<f32>,
    s3a: CudaSlice<f32>,
    s3b: CudaSlice<f32>,
    flat: CudaSlice<f32>,
    x: CudaSlice<f32>,
    n: CudaSlice<f32>,
    h: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    att: CudaSlice<f32>,
    g: CudaSlice<f32>,
    d: CudaSlice<f32>,
    f1: CudaSlice<f32>,
    fused: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    tokens: CudaSlice<u32>,
    frames: CudaSlice<u32>,
    part: CudaSlice<u8>,
    segs: CudaSlice<u8>,
    jobs: CudaSlice<u8>,
}

impl Workspace {
    fn floats() -> usize {
        let (r, m) = (MAX_WINDOW, MAX_PASS);
        MAX_PCM
            + mel_rows(r) * MEL
            + stem_rows(r) * 65 * SUB_CHANNELS
            + 2 * stage2_rows(r) * 33 * SUB_CHANNELS
            + 2 * r * 17 * SUB_CHANNELS
            + m * SUB_OUT
            + m * (6 * D_MODEL + D_FF + 3 * D_MODEL + PROMPT_HIDDEN + PRED)
    }

    fn bytes() -> u64 {
        4 * Self::floats() as u64
            + 8 * (MAX_PASS * arch::MAX_SYMBOLS) as u64
            + (crate::gpu::NASR_PART_BYTES
                + MAX_SEGMENTS * std::mem::size_of::<crate::gpu::NasrSeg>()
                + crate::gpu::NASR_JOBS * std::mem::size_of::<crate::gpu::NasrJob>())
                as u64
    }

    fn new(exec: &GpuExecutor) -> Result<Self, GpuModelError> {
        let (r, m) = (MAX_WINDOW, MAX_PASS);
        let a = |n: usize| exec.alloc(n);
        Ok(Self {
            pcm: a(MAX_PCM)?,
            mel: a(mel_rows(r) * MEL)?,
            s1: a(stem_rows(r) * 65 * SUB_CHANNELS)?,
            s2a: a(stage2_rows(r) * 33 * SUB_CHANNELS)?,
            s2b: a(stage2_rows(r) * 33 * SUB_CHANNELS)?,
            s3a: a(r * 17 * SUB_CHANNELS)?,
            s3b: a(r * 17 * SUB_CHANNELS)?,
            flat: a(m * SUB_OUT)?,
            x: a(m * D_MODEL)?,
            n: a(m * D_MODEL)?,
            h: a(m * D_FF)?,
            qkv: a(m * 3 * D_MODEL)?,
            att: a(m * D_MODEL)?,
            g: a(m * D_MODEL)?,
            d: a(m * D_MODEL)?,
            f1: a(m * PROMPT_HIDDEN)?,
            fused: a(m * D_MODEL)?,
            proj: a(m * PRED)?,
            tokens: exec.alloc_u32(m * arch::MAX_SYMBOLS)?,
            frames: exec.alloc_u32(m * arch::MAX_SYMBOLS)?,
            part: exec.alloc_u8(crate::gpu::NASR_PART_BYTES)?,
            segs: exec.alloc_u8(MAX_SEGMENTS * std::mem::size_of::<crate::gpu::NasrSeg>())?,
            jobs: exec
                .alloc_u8(crate::gpu::NASR_JOBS * std::mem::size_of::<crate::gpu::NasrJob>())?,
        })
    }
}

/// Q8_0 rows `[n][k]` as the GGUF stores them (`k / 32` 34-byte blocks a
/// row), reordered by `order` (output row i = stored row order[i]) and
/// repacked for slot 853 - values bit for bit.
fn q8_rows(
    exec: &GpuExecutor,
    parts: &[&[u8]],
    k: usize,
    order: Option<&dyn Fn(usize) -> usize>,
) -> Result<ClefQ8, GpuModelError> {
    let row = k / 32 * 34;
    let all: Vec<u8> = parts.concat();
    let n = all.len() / row;
    let blocks = match order {
        None => all,
        Some(f) => {
            let mut v = Vec::with_capacity(all.len());
            for i in 0..n {
                let s = f(i);
                v.extend_from_slice(&all[s * row..(s + 1) * row]);
            }
            v
        }
    };
    let (q, scale) = ClefQ8::repack(&blocks, n, k);
    Ok(ClefQ8 {
        q: exec.to_device_u8(&q)?,
        scale: exec.to_device_u8(&scale)?,
        k,
        n,
    })
}

/// Host copies of a pass's planes, every window appended in order:
/// `sub` (the subsampler's output), `layer_<i>` for each block asked for,
/// `fused` (the prompt projector's) and `proj` (the joint's encoder half).
/// For the reference gate; a traced pass synchronizes after each plane.
#[derive(Default)]
pub struct Trace {
    pub layers: Vec<usize>,
    pub planes: Vec<(String, Vec<f32>)>,
}

impl Trace {
    fn add(&mut self, name: &str, v: Vec<f32>) {
        match self.planes.iter_mut().find(|(n, _)| n == name) {
            Some((_, p)) => p.extend(v),
            None => self.planes.push((name.to_owned(), v)),
        }
    }

    pub fn plane(&self, name: &str) -> Option<&[f32]> {
        self.planes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.as_slice())
    }
}

/// The loaded model.
pub struct GpuNemotronAsr {
    /// planes to keep for the reference gate (None = no tracing)
    pub trace: Option<Trace>,
    exec: Arc<GpuExecutor>,
    weight_bytes: u64,
    // frontend constants (the diarization frontend's planes)
    window: CudaSlice<f32>,
    fb: CudaSlice<f32>,
    spans: CudaSlice<u32>,
    twiddle: CudaSlice<f32>,
    // subsampler
    stem: (CudaSlice<f32>, CudaSlice<f32>),
    dw1: (CudaSlice<f32>, CudaSlice<f32>),
    pw1: (CudaSlice<f32>, CudaSlice<f32>),
    dw2: (CudaSlice<f32>, CudaSlice<f32>),
    pw2: (CudaSlice<f32>, CudaSlice<f32>),
    sub_out: ClefQ8,
    sub_out_b: CudaSlice<f32>,
    layers: Vec<Layer>,
    // prompt projector: W1 without its one-hot columns, the one-hot column
    // of every prompt folded into a bias row `[128][2048]`
    prompt1: CudaSlice<f32>,
    prompt1_b: CudaSlice<f32>,
    prompt2: CudaSlice<f32>,
    prompt2_b: CudaSlice<f32>,
    enc: ClefQ8,
    enc_b: CudaSlice<f32>,
    // decoder
    head: ClefQ8,
    head_b: CudaSlice<f32>,
    ih: [(ClefQ8, CudaSlice<f32>); 2],
    hh: [(ClefQ8, CudaSlice<f32>); 2],
    pred: ClefQ8,
    pred_b: CudaSlice<f32>,
    embed: CudaSlice<u8>,
    ws: Workspace,
    prompts: Vec<(String, u32)>,
    vocab: Vec<String>,
}

impl GpuNemotronAsr {
    /// Load NVIDIA's GGUF. The whole inventory is checked before the first
    /// allocation.
    pub fn load(exec: Arc<GpuExecutor>, path: &Path) -> Result<Self, GpuModelError> {
        if !exec.has_nemotron_asr() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates the Nemotron 3.5 ASR lane (slots 853-861) - \
                 rebuild or update the pack"
                    .into(),
            ));
        }
        let bad = |e: String| GpuModelError::Unsupported(format!("Nemotron 3.5 ASR: {e}"));
        let ck = Checkpoint::open(path).map_err(bad)?;
        let stored = ck.validate().map_err(bad)?;
        exec.vram_load_gate(
            stored + Workspace::bytes() + Session::device_bytes() + (16 << 20),
            "Nemotron 3.5 ASR",
        )
        .map_err(GpuModelError::WontFit)?;
        // one stream, one owning thread - must precede every alloc
        exec.disable_event_tracking();
        let weight_bytes = std::cell::Cell::new(0u64);
        let up = |v: &[f32]| {
            weight_bytes.set(weight_bytes.get() + 4 * v.len() as u64);
            exec.to_device(v)
        };
        let vals = |n: &str, shape: &[usize]| ck.values(n, shape).map_err(bad);
        let raw = |n: &str, shape: &[usize]| ck.raw(n, shape).map(|r| r.bytes).map_err(bad);
        let q8 = |parts: &[&[u8]], k: usize, order: Option<&dyn Fn(usize) -> usize>| {
            let w = q8_rows(&exec, parts, k, order)?;
            weight_bytes.set(weight_bytes.get() + w.bytes());
            Ok::<_, GpuModelError>(w)
        };
        let proj =
            |n: &str, out: usize, k: usize| q8(&[raw(&format!("{n}.weight"), &[out, k])?], k, None);
        let norm = |n: &str| -> Result<Norm, GpuModelError> {
            Ok(Norm {
                w: up(&vals(&format!("{n}.weight"), &[D_MODEL])?)?,
                b: up(&vals(&format!("{n}.bias"), &[D_MODEL])?)?,
            })
        };
        let conv = |i: usize,
                    shape: &[usize]|
         -> Result<(CudaSlice<f32>, CudaSlice<f32>), GpuModelError> {
            let p = format!("encoder.pre_encode.conv.{i}");
            Ok((
                up(&vals(&format!("{p}.weight"), shape)?)?,
                up(&vals(&format!("{p}.bias"), &[SUB_CHANNELS])?)?,
            ))
        };
        let c = SUB_CHANNELS;
        let stem = conv(0, &[c, 1, 3, 3])?;
        let dw1 = conv(2, &[c, 1, 3, 3])?;
        let pw1 = conv(3, &[c, c, 1, 1])?;
        let dw2 = conv(5, &[c, 1, 3, 3])?;
        let pw2 = conv(6, &[c, c, 1, 1])?;
        let sub_out = proj("encoder.pre_encode.out", D_MODEL, SUB_OUT)?;
        let sub_out_b = up(&vals("encoder.pre_encode.out.bias", &[D_MODEL])?)?;
        // the stored relative-position table's rows at distances -13 ..= 69,
        // row d - REL_MIN (stored row PE_CENTER - d)
        let pe = vals("encoder.pos_enc.pe", &[arch::PE_ROWS, D_MODEL])?;
        let mut pe_sel = vec![0f32; REL_ROWS * D_MODEL];
        for (r, row) in pe_sel.as_chunks_mut::<D_MODEL>().0.iter_mut().enumerate() {
            let src = (PE_CENTER as i64 - (r as i64 + REL_MIN)) as usize;
            row.copy_from_slice(&pe[src * D_MODEL..(src + 1) * D_MODEL]);
        }
        let pe_dev = exec.to_device(&pe_sel)?;
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("encoder.layers.{i}");
            let a = |n: &str| raw(&format!("{p}.self_attn.{n}.weight"), &[D_MODEL, D_MODEL]);
            let qkv = q8(
                &[a("linear_q")?, a("linear_k")?, a("linear_v")?],
                D_MODEL,
                None,
            )?;
            let pos = proj(&format!("{p}.self_attn.linear_pos"), D_MODEL, D_MODEL)?;
            let mut rel = exec.alloc(REL_ROWS * D_MODEL)?;
            exec.nasr_gemm(
                (&pe_dev, 0),
                &pos,
                None,
                &mut rel,
                REL_ROWS,
                crate::gpu::NasrEpi::Store,
            )?;
            weight_bytes.set(weight_bytes.get() + 4 * (REL_ROWS * D_MODEL) as u64);
            // GLU rows interleaved: output row 2j = value j, 2j + 1 = gate j
            let pw1 = q8(
                &[raw(
                    &format!("{p}.conv.pointwise_conv1.weight"),
                    &[2 * D_MODEL, D_MODEL],
                )?],
                D_MODEL,
                Some(&|i: usize| {
                    if i.is_multiple_of(2) {
                        i / 2
                    } else {
                        D_MODEL + i / 2
                    }
                }),
            )?;
            layers.push(Layer {
                ff1_norm: norm(&format!("{p}.norm_feed_forward1"))?,
                ff1_up: proj(&format!("{p}.feed_forward1.linear1"), D_FF, D_MODEL)?,
                ff1_down: proj(&format!("{p}.feed_forward1.linear2"), D_MODEL, D_FF)?,
                att_norm: norm(&format!("{p}.norm_self_att"))?,
                qkv,
                out: proj(&format!("{p}.self_attn.linear_out"), D_MODEL, D_MODEL)?,
                bias_u: up(&vals(&format!("{p}.self_attn.pos_bias_u"), &[8, 128])?)?,
                bias_v: up(&vals(&format!("{p}.self_attn.pos_bias_v"), &[8, 128])?)?,
                rel,
                conv_norm: norm(&format!("{p}.norm_conv"))?,
                pw1,
                dw: up(&vals(
                    &format!("{p}.conv.depthwise_conv.weight"),
                    &[D_MODEL, 1, 9],
                )?)?,
                dw_norm: norm(&format!("{p}.conv.batch_norm"))?,
                pw2: proj(&format!("{p}.conv.pointwise_conv2"), D_MODEL, D_MODEL)?,
                ff2_norm: norm(&format!("{p}.norm_feed_forward2"))?,
                ff2_up: proj(&format!("{p}.feed_forward2.linear1"), D_FF, D_MODEL)?,
                ff2_down: proj(&format!("{p}.feed_forward2.linear2"), D_MODEL, D_FF)?,
                out_norm: norm(&format!("{p}.norm_out"))?,
            });
        }
        drop(pe_dev);
        // prompt projector: [x | onehot] . W1^T = x . W1[:, :1024]^T +
        // W1[:, 1024 + p] - the one-hot's column added to the bias at load
        let w1 = vals(
            "prompt_kernel.0.weight",
            &[PROMPT_HIDDEN, D_MODEL + PROMPTS],
        )?;
        let b1 = vals("prompt_kernel.0.bias", &[PROMPT_HIDDEN])?;
        let mut prompt1 = Vec::with_capacity(PROMPT_HIDDEN * D_MODEL);
        for row in w1.as_chunks::<{ D_MODEL + PROMPTS }>().0 {
            prompt1.extend_from_slice(&row[..D_MODEL]);
        }
        let mut prompt1_b = vec![0f32; PROMPTS * PROMPT_HIDDEN];
        for (p, out) in prompt1_b
            .as_chunks_mut::<PROMPT_HIDDEN>()
            .0
            .iter_mut()
            .enumerate()
        {
            for (n, o) in out.iter_mut().enumerate() {
                *o = b1[n] + w1[n * (D_MODEL + PROMPTS) + D_MODEL + p];
            }
        }
        let prompt1 = up(&prompt1)?;
        let prompt1_b = up(&prompt1_b)?;
        let prompt2 = up(&vals("prompt_kernel.2.weight", &[D_MODEL, PROMPT_HIDDEN])?)?;
        let prompt2_b = up(&vals("prompt_kernel.2.bias", &[D_MODEL])?)?;
        let enc = proj("joint.enc", PRED, D_MODEL)?;
        let enc_b = up(&vals("joint.enc.bias", &[PRED])?)?;
        let head = proj("joint.joint_net.2", VOCAB, PRED)?;
        let head_b = up(&vals("joint.joint_net.2.bias", &[VOCAB])?)?;
        // gate rows interleaved: row 4 j + g = gate g (i, f, g, o) of unit j,
        // so an 8-row tile is two units' four gates (the greedy walk's
        // layout)
        let gate_row = |i: usize| (i % 4) * PRED + i / 4;
        let lstm = |g: &str, l: usize| -> Result<(ClefQ8, CudaSlice<f32>), GpuModelError> {
            let n = format!("decoder.prediction.dec_rnn.lstm.{g}_l{l}");
            let w = q8(
                &[raw(&format!("{n}.weight"), &[4 * PRED, PRED])?],
                PRED,
                Some(&gate_row),
            )?;
            let b = vals(&format!("{n}.bias"), &[4 * PRED])?;
            let b: Vec<f32> = (0..4 * PRED).map(|i| b[gate_row(i)]).collect();
            Ok((w, up(&b)?))
        };
        let ih = [lstm("ih", 0)?, lstm("ih", 1)?];
        let hh = [lstm("hh", 0)?, lstm("hh", 1)?];
        let pred = proj("joint.pred", PRED, PRED)?;
        let pred_b = up(&vals("joint.pred.bias", &[PRED])?)?;
        let embed_raw = raw("decoder.prediction.embed.weight", &[VOCAB, PRED])?;
        weight_bytes.set(weight_bytes.get() + embed_raw.len() as u64);
        let embed = exec.to_device_u8(embed_raw)?;
        // the frontend's planes, as the diarization lane builds them
        let filterbank = vals("preprocessor.fb", &[MEL, BINS])?;
        let mut padded = vec![0f32; arch::N_FFT];
        padded[56..456].copy_from_slice(&ck.window());
        let spans: Vec<u32> = filterbank
            .as_chunks::<BINS>()
            .0
            .iter()
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
        let window = up(&padded)?;
        let fb = up(&filterbank)?;
        let spans = exec.to_device_u32(&spans)?;
        let twiddle = up(&twiddle)?;
        let ws = Workspace::new(&exec)?;
        exec.synchronize()?;
        let weight_bytes = weight_bytes.get();
        Ok(Self {
            trace: None,
            exec,
            weight_bytes,
            window,
            fb,
            spans,
            twiddle,
            stem,
            dw1,
            pw1,
            dw2,
            pw2,
            sub_out,
            sub_out_b,
            layers,
            prompt1,
            prompt1_b,
            prompt2,
            prompt2_b,
            enc,
            enc_b,
            head,
            head_b,
            ih,
            hh,
            pred,
            pred_b,
            embed,
            ws,
            prompts: ck.prompts().to_vec(),
            vocab: ck.vocab().to_vec(),
        })
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// The window planes, allocated once at load.
    pub fn workspace_bytes(&self) -> u64 {
        Workspace::bytes()
    }

    pub fn executor(&self) -> &Arc<GpuExecutor> {
        &self.exec
    }

    /// The prompt index of a language key (`de-DE`, `de`, `auto`).
    pub fn prompt(&self, key: &str) -> Option<u32> {
        self.prompts
            .iter()
            .find(|(k, _)| k == key)
            .map(|&(_, id)| id)
    }

    /// The prompt dictionary as the checkpoint stores it.
    pub fn prompts(&self) -> &[(String, u32)] {
        &self.prompts
    }

    /// The vocabulary (index = token id).
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    /// A new stream at `lookahead` (one of [`arch::LOOKAHEADS`]) under
    /// `prompt`.
    pub fn session(&self, lookahead: usize, prompt: u32) -> Result<Session, GpuModelError> {
        if !arch::LOOKAHEADS.contains(&lookahead) {
            return Err(GpuModelError::Unsupported(format!(
                "lookahead {lookahead} (the model publishes {:?})",
                arch::LOOKAHEADS
            )));
        }
        if prompt as usize >= PROMPTS {
            return Err(GpuModelError::Unsupported(format!("prompt {prompt}")));
        }
        Session::new(&self.exec, lookahead, prompt)
    }

    /// A whole recording: one session fed at once.
    pub fn transcribe(
        &mut self,
        pcm: &[f32],
        lookahead: usize,
        prompt: u32,
    ) -> Result<Session, GpuModelError> {
        let mut s = self.session(lookahead, prompt)?;
        self.push(&mut s, pcm)?;
        self.finish(&mut s)?;
        Ok(s)
    }
}

/// One stream's state: the PCM tail the next window reads, the K/V and
/// convolution rings, the decoder state, and what it has decoded.
pub struct Session {
    lookahead: usize,
    prompt: u32,
    /// samples `pcm_base ..` of the stream
    pcm: Vec<f32>,
    pcm_base: usize,
    received: usize,
    /// encoder frames already encoded and decoded
    done: usize,
    /// the recording is complete: windows read past its end as padding
    finishing: bool,
    finished: bool,
    /// `[24][2][56][1024]`: each layer's K then V ring
    kv: CudaSlice<f32>,
    /// `[24][8][1024]`: each layer's GLU-row ring
    conv: CudaSlice<f32>,
    dec: CudaSlice<u8>,
    primed: bool,
    /// emitted tokens and their absolute encoder frames (80 ms each)
    pub tokens: Vec<u32>,
    pub frames: Vec<u32>,
}

impl Session {
    fn device_bytes() -> u64 {
        4 * (LAYERS * (2 * KV_SLOTS + CONV_SLOTS) * D_MODEL) as u64
            + crate::gpu::NASR_STATE_BYTES as u64
    }

    fn new(exec: &GpuExecutor, lookahead: usize, prompt: u32) -> Result<Self, GpuModelError> {
        Ok(Self {
            lookahead,
            prompt,
            pcm: Vec::new(),
            pcm_base: 0,
            received: 0,
            done: 0,
            finishing: false,
            finished: false,
            kv: exec.alloc(LAYERS * 2 * KV_SLOTS * D_MODEL)?,
            conv: exec.alloc(LAYERS * CONV_SLOTS * D_MODEL)?,
            dec: exec.alloc_u8(crate::gpu::NASR_STATE_BYTES)?,
            primed: false,
            tokens: Vec::new(),
            frames: Vec::new(),
        })
    }

    pub fn lookahead(&self) -> usize {
        self.lookahead
    }

    /// Encoder frames encoded so far.
    pub fn encoded_frames(&self) -> usize {
        self.done
    }

    /// Samples received so far.
    pub fn samples(&self) -> usize {
        self.received
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}
