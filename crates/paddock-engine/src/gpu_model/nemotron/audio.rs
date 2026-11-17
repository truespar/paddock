//! Nemotron 3.5 Transcribe's speech encoder: the rows that replace a
//! recording's `<|audio|>` placeholders. NeMo's ParallelExpertEncoder in its
//! online inference (what vLLM's NeMo plugin serves and the Transformers
//! port reproduces), on one shared log-mel:
//!
//! - the speech branch: the mel normalized per bin over the recording, cut
//!   into windows of 500 encoder rows (80 ms each) with 50 rows of context on
//!   each side, every window through a 32-block pre-LN transformer (d 1280,
//!   10 heads of 128, LayerNorm on each q / k head, rope on the first half of
//!   each head, exact GELU), only its core rows kept;
//! - the speaker branch: the checkpoint's frozen Nemotron 3 Diarization (the
//!   diarization lane's own planes and kernels) over the raw mel at the
//!   encoder's window geometry, its 10-ms speaker probabilities averaged to
//!   the encoder rate and thresholded at 0.5;
//! - the fusion: LayerNorm(speech row) + LayerNorm_8(activity) through the
//!   fixed sinusoid speaker table, then the 1280 -> 2688 projector.
//!
//! Precision class - the served one is the reference's: the speech branch's
//! 128 linears run the checkpoint's own ModelOpt FP8 W8A8 with static
//! per-tensor scales - e4m3 weights as shipped, each linear's input
//! quantized to e4m3 with its `input_scale` where it is produced (the
//! LayerNorms, the attention's output, the FFN up's GELU), e4m3 x e4m3 into
//! f32 tensor-core sums - exactly what vLLM's NeMo plugin serves (its
//! `FP8Linear`), on the f32 residual stream; for a recording inside one
//! diarizer window the speaker branch's encoder runs bf16 too - both GEMM
//! operands bf16, landed so by their producers (the plugin serves the
//! perception module in bf16) - while a longer recording, which walks the
//! streaming speaker cache, keeps the diarization lane's F32 class
//! (`speaker_class`). The finer class this lane ran first - f16 activations
//! against the dequantized weights rounded to f16, the speaker branch in the
//! F32 class throughout - stays behind PADDOCK_TR_ENC_F16 as the
//! A/B. Everything else (norms, rope, attention, the mel statistics, the
//! speaker head) is f32 or f64 arithmetic, rounded once into f16 or bf16 where a
//! GEMM or the attention reads it.

use std::sync::{Arc, Mutex};

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::modelopt::{e4m3_to_f32, fp8_view};
use paddock_models::safetensors::ShardedSafetensors;

use crate::diarization::OfflineGeometry;
use crate::gpu::{ENC_ATTN_QTILE, ENC_ATTN_TILE_SHIFT, GpuExecutor, HalfTensor, TrW8Out};
use crate::gpu_model::diarization::GpuDiarization;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::f32_tensor;

/// Speech encoder width, heads, head width, FFN width, blocks.
const D: usize = 1280;
const HEADS: usize = 10;
const HD: usize = 128;
const FF: usize = 5120;
const LAYERS: usize = 32;
/// mel bins, and frames an encoder row stacks
const MEL: usize = 128;
const STACK: usize = 8;
/// The online inference's window: core rows and context rows a side.
const WIN: usize = paddock_models::transcribe::WINDOW;
const CTX: usize = paddock_models::transcribe::CONTEXT;
const MAX_WIN: usize = WIN + 2 * CTX;
/// Rows one encoder pass packs (whole windows; one window is at most
/// [`MAX_WIN`]).
const PASS_ROWS: usize = 4096;
const EPS: f32 = 1e-5;
/// The speaker branch at the encoder's geometry (NeMo's ParallelExpertEncoder
/// online defaults, which the checkpoint does not override).
pub(crate) const DIAR: OfflineGeometry = OfflineGeometry {
    chunk: WIN,
    left: CTX,
    right: CTX,
    fifo: paddock_models::transcribe::DIAR_FIFO,
    update: paddock_models::transcribe::DIAR_UPDATE,
    cache: paddock_models::transcribe::DIAR_CACHE,
};

fn bad(what: impl std::fmt::Display) -> GpuModelError {
    GpuModelError::Unsupported(format!("Nemotron 3.5 Transcribe audio: {what}"))
}

struct Norm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

struct Block {
    n1: Norm,
    qn: Norm,
    kn: Norm,
    out_b: CudaSlice<f32>,
    n2: Norm,
    fc1_b: CudaSlice<f32>,
    fc2_b: CudaSlice<f32>,
    /// the four linears: fused [q;k;v] `[D -> 3D]`, out `[D -> D]`, fc1
    /// `[D -> FF]`, fc2 `[FF -> D]`
    lin: Lins,
}

/// A block's linears in one of the two classes (module docs).
enum Lins {
    W8 {
        qkv: F8Lin,
        out: F8Lin,
        fc1: F8Lin,
        fc2: F8Lin,
    },
    F16 {
        qkv: HalfTensor,
        out: HalfTensor,
        fc1: HalfTensor,
        fc2: HalfTensor,
    },
}

/// One checkpoint FP8 linear as shipped: e4m3 `[out][in]`, its weight
/// scale, its static input scale and that scale's f32 reciprocal (the
/// reference quantizer multiplies by it).
struct F8Lin {
    w: CudaSlice<u8>,
    sw: f32,
    sx: f32,
    inv: f32,
}

/// The planes one encoder pass reads and writes, sized for [`PASS_ROWS`].
struct PassWs {
    in16: CudaSlice<f16>,
    x: CudaSlice<f32>,
    t: CudaSlice<f32>,
    qkv16: CudaSlice<f16>,
    att16: CudaSlice<f16>,
    /// the linears' inputs: e4m3 (W8A8) or f16 (the A/B class)
    lin_in: LinIn,
    pos: CudaSlice<u32>,
    cu: CudaSlice<u32>,
    tiles: CudaSlice<u32>,
}

enum LinIn {
    /// the normed rows, the attention's output, the FFN's activation
    W8 {
        h8: CudaSlice<u8>,
        a8: CudaSlice<u8>,
        ff8: CudaSlice<u8>,
    },
    F16 {
        h16: CudaSlice<f16>,
        ff16: CudaSlice<f16>,
    },
}

impl PassWs {
    fn new(e: &GpuExecutor, w8: bool) -> Result<Self, GpuModelError> {
        let r = PASS_ROWS;
        let seqs = r / (2 * CTX) + 2;
        let lin_in = if w8 {
            LinIn::W8 {
                h8: e.alloc_u8(r * D)?,
                a8: e.alloc_u8(r * D)?,
                ff8: e.alloc_u8(r * FF)?,
            }
        } else {
            LinIn::F16 {
                h16: e.alloc_f16(r * D)?,
                ff16: e.alloc_f16(r * FF)?,
            }
        };
        Ok(Self {
            in16: e.alloc_f16(r * STACK * MEL)?,
            x: e.alloc(r * D)?,
            t: e.alloc(r * D)?,
            qkv16: e.alloc_f16(r * 3 * D)?,
            att16: e.alloc_f16(r * D)?,
            lin_in,
            pos: e.alloc_u32(r)?,
            cu: e.alloc_u32(seqs + 1)?,
            tiles: e.alloc_u32(r / ENC_ATTN_QTILE + seqs)?,
        })
    }

    fn bytes(w8: bool) -> u64 {
        let r = PASS_ROWS as u64;
        let seqs = r / (2 * CTX as u64) + 2;
        let lin_in = if w8 {
            r * (2 * D + FF) as u64
        } else {
            2 * r * (D + FF) as u64
        };
        2 * r * (STACK * MEL) as u64
            + 4 * 2 * r * D as u64
            + 2 * r * (3 * D + D) as u64
            + lin_in
            + 4 * (r + seqs + 1 + r / ENC_ATTN_QTILE as u64 + seqs)
    }
}

/// A recording's projected rows on the device: `n` rows of the language
/// model's hidden width, one per `<|audio|>` placeholder.
pub(crate) struct AudioRows {
    pub rows: CudaSlice<f32>,
    pub n: usize,
}

pub(crate) struct TranscribeAudio {
    exec: Arc<GpuExecutor>,
    diar: GpuDiarization,
    pre: HalfTensor,
    embed: Norm,
    blocks: Vec<Block>,
    last: Norm,
    asr_norm: Norm,
    /// LayerNorm_8(activity pattern) @ the sinusoid table, `[256][D]`
    infusion: CudaSlice<f32>,
    proj: HalfTensor,
    proj_b: CudaSlice<f32>,
    /// the rope's (cos, sin) at every position of a window, `[MAX_WIN][32][2]`
    rope: CudaSlice<f32>,
    hidden: usize,
    ws: PassWs,
    pub weight_bytes: u64,
    /// the CTC timestamp head and the aligner's tables, when the checkpoint
    /// ships them
    ctc: Option<super::ctc::CtcHead>,
    pub(crate) ctc_align: Option<Arc<super::ctc::CtcAlign>>,
}

/// The rope table: inv_freq = 1 / 10000^(2i / 64) in f32 as NeMo builds it,
/// angle = pos * inv_freq rounded to f32, cos / sin of that angle evaluated
/// in f64 and rounded once.
fn rope_table() -> Vec<f32> {
    let inv: Vec<f32> = (0..32)
        .map(|i| (1.0 / 10000f64.powf(2.0 * i as f64 / 64.0)) as f32)
        .collect();
    let mut t = Vec::with_capacity(MAX_WIN * 64);
    for pos in 0..MAX_WIN {
        for &f in &inv {
            let angle = f64::from(pos as f32 * f);
            t.push(angle.cos() as f32);
            t.push(angle.sin() as f32);
        }
    }
    t
}

/// The 256 speaker infusion rows: for every 0/1 activity pattern of the
/// eight speakers, the 8-wide LayerNorm (weight `w8`, bias `b8`) through
/// NeMo's fixed sinusoid table `[8][D]` (sin on even dims, cos on odd, the
/// 10000^(-2i/D) frequencies), all in f64.
fn infusion_table(w8: &[f32], b8: &[f32]) -> Vec<f32> {
    let mut kernel = vec![[0f64; D]; 8];
    for (s, row) in kernel.iter_mut().enumerate() {
        for i in 0..D / 2 {
            // NeMo forms the frequency in f32: exp(2i * -(ln 1e4 / D))
            let div = ((2 * i) as f32 * -(10000f32.ln() / D as f32)).exp();
            let a = f64::from(s as f32 * div);
            row[2 * i] = a.sin();
            row[2 * i + 1] = a.cos();
        }
    }
    let mut out = Vec::with_capacity(256 * D);
    for code in 0..256u32 {
        let a: Vec<f64> = (0..8).map(|s| f64::from((code >> s) & 1)).collect();
        let mean = a.iter().sum::<f64>() / 8.0;
        let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 8.0;
        let inv = 1.0 / (var + f64::from(EPS)).sqrt();
        let y: Vec<f64> = (0..8)
            .map(|s| (a[s] - mean) * inv * f64::from(w8[s]) + f64::from(b8[s]))
            .collect();
        out.extend((0..D).map(|d| (0..8).map(|s| y[s] * kernel[s][d]).sum::<f64>() as f32));
    }
    out
}

impl TranscribeAudio {
    /// Load the perception tensors from the checkpoint's shards `st` (the
    /// speech LM's `hidden` is the projector's width).
    pub(crate) fn load(
        exec: Arc<GpuExecutor>,
        st: &ShardedSafetensors,
        dir: &std::path::Path,
        hidden: usize,
    ) -> Result<Self, GpuModelError> {
        if !exec.has_transcribe_audio()
            || !exec.has_diarization()
            || !exec.has_text_encoder()
            || !exec.has_deepencoder_fused()
        {
            return Err(bad(
                "this kernel pack predates the speech encoder (slots 942-944) - rebuild or \
                 update the pack",
            ));
        }
        paddock_models::transcribe::check_perception(dir).map_err(bad)?;
        // The speech branch runs on its own lane: an admission's encode is
        // enqueued there and the language model's next passes run beside it -
        // the compute-bound encoder against the experts' DRAM stream - with the
        // first pass that reads its rows waiting on the encode's event
        // (`ready`). Serial between passes it held ~16% of a c32 serve's wall.
        // The lane's dispatch priority: equal to the language model's by
        // default (a least-priority lane only ran in the passes' gaps);
        // PADDOCK_TR_LANE=least|main for the A/B (main = no lane at all).
        let exec = match paddock_models::dev_var!("PADDOCK_TR_LANE").as_deref() {
            Ok("main") => exec,
            Ok("least") => Arc::new(exec.fork_stream_at(false)?),
            _ => Arc::new(exec.fork_stream_at(true)?),
        };
        // a K-split f16 GEMM's slices hand partials over cross-CTA flags and
        // assume the grid owns the machine - with the language model resident
        // beside the lane, keep them unsplit (the qwen4_exp fork's rule)
        if exec.has_f16_ksplit_set() {
            exec.f16_ksplit_set(false)?;
        }
        let mut weight_bytes = 0u64;
        let pfx = "perception.encoder.asr_encoder";
        let vec_f32 =
            |name: &str, n: usize, wb: &mut u64| -> Result<CudaSlice<f32>, GpuModelError> {
                *wb += 4 * n as u64;
                Ok(exec.to_device(&f32_tensor(st, name, n)?)?)
            };
        let norm = |name: &str, n: usize, wb: &mut u64| -> Result<Norm, GpuModelError> {
            Ok(Norm {
                w: vec_f32(&format!("{name}.weight"), n, wb)?,
                b: vec_f32(&format!("{name}.bias"), n, wb)?,
            })
        };
        // a bf16 plane `[out][in]` to f16 (bf16 widens exactly, the narrow is
        // the one rounding)
        let half_bf16 = |name: &str,
                         out: usize,
                         inn: usize,
                         wb: &mut u64|
         -> Result<HalfTensor, GpuModelError> {
            let v = f32_tensor(st, name, out * inn)?;
            *wb += 2 * v.len() as u64;
            Ok(HalfTensor {
                buf: exec.to_device_f16(&v, name)?,
                dims: vec![inn, out],
            })
        };
        // The linears' class (module docs): W8A8 as shipped wherever the die
        // has e4m3 mma and the pack the landings; PADDOCK_TR_ENC_F16 for the
        // f16 class's A/B. A pack without the W8A8 slots on such a die is
        // refused rather than served in the other class unannounced.
        let w8a8 = paddock_models::dev_var_os!("PADDOCK_TR_ENC_F16").is_none()
            && exec.compute_capability() >= (8, 9);
        if w8a8 && !exec.has_tr_w8a8() {
            return Err(bad(
                "this kernel pack predates the W8A8 speech encoder (slots 961-963) - rebuild or \
                 update the pack",
            ));
        }
        // an FP8 plane `[out][in]`: the e4m3 bytes and scales as shipped
        let fp8 = |name: &str, out: usize, inn: usize| {
            let v = fp8_view(st, name).map_err(|e| bad(format!("{name}: {e}")))?;
            if (v.n, v.k) != (out, inn) {
                return Err(bad(format!(
                    "{name} is [{}, {}], expected [{out}, {inn}]",
                    v.n, v.k
                )));
            }
            if !(v.input_scale.is_finite() && v.input_scale > 0.0) {
                return Err(bad(format!("{name}: input_scale {}", v.input_scale)));
            }
            Ok(v)
        };
        let lin =
            |name: &str, out: usize, inn: usize, wb: &mut u64| -> Result<F8Lin, GpuModelError> {
                let v = fp8(name, out, inn)?;
                let mut w = exec.alloc_u8(out * inn)?;
                exec.upload_u8(v.weight, &mut w)?;
                *wb += (out * inn) as u64;
                Ok(F8Lin {
                    w,
                    sw: v.weight_scale,
                    sx: v.input_scale,
                    inv: 1.0 / v.input_scale,
                })
            };
        // the A/B class: e4m3 * the per-tensor scale, one rounding to f16
        let half_fp8 = |name: &str,
                        out: usize,
                        inn: usize,
                        wb: &mut u64|
         -> Result<HalfTensor, GpuModelError> {
            let v = fp8(name, out, inn)?;
            let w: Vec<f32> = v
                .weight
                .iter()
                .map(|&b| e4m3_to_f32(b) * v.weight_scale)
                .collect();
            *wb += 2 * w.len() as u64;
            Ok(HalfTensor {
                buf: exec.to_device_f16(&w, name)?,
                dims: vec![inn, out],
            })
        };
        let pre = half_bf16(
            &format!("{pfx}.pre_encode.proj.weight"),
            D,
            STACK * MEL,
            &mut weight_bytes,
        )?;
        let embed = norm(&format!("{pfx}.embed_norm"), D, &mut weight_bytes)?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("{pfx}.layers.{i}");
            let (qkv, out, fc1, fc2) = (
                format!("{p}.attn.w_qkv"),
                format!("{p}.attn.out_proj"),
                format!("{p}.ffn.net.0"),
                format!("{p}.ffn.net.3"),
            );
            let wb = &mut weight_bytes;
            let lin = if w8a8 {
                Lins::W8 {
                    qkv: lin(&qkv, 3 * D, D, wb)?,
                    out: lin(&out, D, D, wb)?,
                    fc1: lin(&fc1, FF, D, wb)?,
                    fc2: lin(&fc2, D, FF, wb)?,
                }
            } else {
                Lins::F16 {
                    qkv: half_fp8(&qkv, 3 * D, D, wb)?,
                    out: half_fp8(&out, D, D, wb)?,
                    fc1: half_fp8(&fc1, FF, D, wb)?,
                    fc2: half_fp8(&fc2, D, FF, wb)?,
                }
            };
            blocks.push(Block {
                n1: norm(&format!("{p}.norm1"), D, &mut weight_bytes)?,
                qn: norm(&format!("{p}.attn.q_norm"), HD, &mut weight_bytes)?,
                kn: norm(&format!("{p}.attn.k_norm"), HD, &mut weight_bytes)?,
                out_b: vec_f32(&format!("{out}.bias"), D, &mut weight_bytes)?,
                n2: norm(&format!("{p}.norm2"), D, &mut weight_bytes)?,
                fc1_b: vec_f32(&format!("{fc1}.bias"), FF, &mut weight_bytes)?,
                fc2_b: vec_f32(&format!("{fc2}.bias"), D, &mut weight_bytes)?,
                lin,
            });
        }
        let last = norm(&format!("{pfx}.final_norm"), D, &mut weight_bytes)?;
        let asr_norm = norm("perception.encoder.asr_norm", D, &mut weight_bytes)?;
        let w8 = f32_tensor(st, "perception.encoder.diar_norm.weight", 8)?;
        let b8 = f32_tensor(st, "perception.encoder.diar_norm.bias", 8)?;
        let infusion = exec.to_device(&infusion_table(&w8, &b8))?;
        weight_bytes += 4 * (256 * D) as u64;
        let proj = half_bf16("perception.proj.weight", hidden, D, &mut weight_bytes)?;
        let proj_b = vec_f32("perception.proj.bias", hidden, &mut weight_bytes)?;
        let rope = exec.to_device(&rope_table())?;
        weight_bytes += 4 * (MAX_WIN * 64) as u64;
        let perception = ShardedSafetensors::open_dir(dir)
            .map_err(|e| bad(format!("perception shards: {e}")))?;
        let mut diar = GpuDiarization::load_perception(exec.clone(), perception, DIAR.max_rows())?;
        // the served class runs the speaker branch of a recording inside one
        // window in bf16 (vLLM's plugin serves the perception module in
        // bf16; picked per recording, see `speaker_class`); the f16 A/B keeps
        // the diarization lane's F32 class throughout
        if w8a8 {
            diar.enable_bf16_class()?;
        }
        weight_bytes += diar.weight_bytes();
        let ws = PassWs::new(&exec, w8a8)?;
        let (ctc, ctc_align) = match super::ctc::CtcHead::load(&exec, st, dir)? {
            Some((head, align)) => {
                weight_bytes += head.weight_bytes;
                (Some(head), Some(Arc::new(align)))
            }
            None => (None, None),
        };
        Ok(Self {
            exec,
            diar,
            pre,
            embed,
            blocks,
            last,
            asr_norm,
            infusion,
            proj,
            proj_b,
            rope,
            hidden,
            ws,
            weight_bytes,
            ctc,
            ctc_align,
        })
    }

    /// The resident pass planes (the diarizer's and the encoder's).
    /// The speaker branch's class for a recording of `rows` encoder rows. In
    /// the served class a recording inside one diarizer window runs bf16; a
    /// longer one walks the streaming driver's speaker cache, whose frame
    /// picks are a discrete selection - in bf16 they part from the exact
    /// model's (GB10 2026-11-15: 6-8% of the activity patterns of a 95 s and
    /// a 200 s recording against 0, probabilities moved up to 0.4) - so it
    /// keeps the F32 class. Picked by the recording's length alone, so its
    /// bits never depend on what it was packed with or which driver ran it.
    /// PADDOCK_TR_DIAR_F32 keeps the F32 class for the A/B.
    fn speaker_class(&mut self, rows: usize) -> Result<(), GpuModelError> {
        let bf16 = self.w8a8()
            && rows <= DIAR.chunk
            && paddock_models::dev_var_os!("PADDOCK_TR_DIAR_F32").is_none();
        self.diar.set_bf16_class(bf16)
    }

    /// Whether the speech branch's linears run the checkpoint's W8A8 class.
    pub(crate) fn w8a8(&self) -> bool {
        self.blocks
            .first()
            .is_some_and(|b| matches!(b.lin, Lins::W8 { .. }))
    }

    pub(crate) fn workspace_bytes(&self) -> u64 {
        self.diar.workspace_bytes() + PassWs::bytes(self.w8a8())
    }

    /// An event on the speech lane after everything enqueued so far: the
    /// language model's stream waits on it before it reads rows an encode
    /// produced (device-side; nothing blocks the host).
    pub(crate) fn ready(&self) -> Result<cudarc::driver::CudaEvent, GpuModelError> {
        Ok(self.exec.record_event()?)
    }

    /// One recording (mono 16-kHz f32) to its projected rows, keeping the
    /// intermediate planes the gates compare when `keep` is set.
    pub(crate) fn encode_probe(
        &mut self,
        pcm: &[f32],
        keep: bool,
    ) -> Result<AudioProbe, GpuModelError> {
        let mut recs = self.prepare_all(&[pcm], keep, &[false])?;
        self.run_windows(&mut recs)?;
        self.finish(recs.pop().expect("one recording"))
    }

    /// Several recordings at once: each one's mel and speaker branch, then
    /// the speech branch's windows of ALL of them packed into shared passes
    /// (a 10-s clip alone gives the GEMMs ~130 rows; a wave of them fills
    /// the tile grids), then each one's fusion and projector. Every
    /// recording's rows are the bits it gets alone - the encoder's sums are
    /// row-local and its attention per window. The ones flagged in
    /// `capture` also keep what their word timestamps need (when the
    /// checkpoint ships the head).
    pub(crate) fn encode_many(
        &mut self,
        pcms: &[&[f32]],
        capture: &[bool],
    ) -> Result<Vec<(AudioRows, Option<super::ctc::Capture>)>, GpuModelError> {
        let capture: Vec<bool> = capture.iter().map(|&c| c && self.ctc.is_some()).collect();
        let mut recs = self.prepare_all(pcms, false, &capture)?;
        self.run_windows(&mut recs)?;
        recs.into_iter()
            .map(|r| self.finish(r).map(|p| (p.rows, p.capture)))
            .collect()
    }

    /// The head's log probabilities at `cols` for a capture's states.
    pub(crate) fn ctc_columns(
        &self,
        states: &CudaSlice<f32>,
        t_enc: usize,
        cols: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        let head = self
            .ctc
            .as_ref()
            .ok_or_else(|| bad("no CTC timestamp head"))?;
        head.columns(&self.exec, states, t_enc, cols)
    }

    /// Every recording's [`Self::prepare`], with the speaker branch of the
    /// ones inside one diarizer window run together: their rows packed into
    /// shared passes (slot 950) - a recording alone gave the diarizer's ~226
    /// launches a clip of ~150 rows each, a sliver of the die - and each
    /// one's activity patterns read from its rows of the packed output. A
    /// recording's bits are its own window's (`predict_packed_device`).
    fn prepare_all(
        &mut self,
        pcms: &[&[f32]],
        keep: bool,
        capture: &[bool],
    ) -> Result<Vec<Rec>, GpuModelError> {
        let exec = self.exec.clone();
        let e = &*exec;
        let pack = e.has_tr_codes()
            && e.has_diar_attention_varlen()
            && paddock_models::dev_var_os!("PADDOCK_TR_NO_DIAR_PACK").is_none()
            && paddock_models::dev_var_os!("PADDOCK_TR_HOST_DIAR").is_none();
        let cap = self.diar.max_rows();
        let mut recs = Vec::with_capacity(pcms.len());
        // (recording, its diarizer input rows, rows, keys)
        let mut queued: Vec<(usize, CudaSlice<f32>, usize, usize)> = Vec::new();
        for (i, pcm) in pcms.iter().enumerate() {
            let (_, _, rows, n_tok) = paddock_models::transcribe::geometry(pcm.len());
            if pack && rows <= DIAR.chunk && rows <= cap {
                let (rec, emb) = self.prepare_head(pcm, keep, capture[i])?;
                queued.push((i, emb, rows, n_tok.clamp(1, rows)));
                recs.push(rec);
            } else {
                recs.push(self.prepare(pcm, keep, capture[i])?);
            }
        }
        // greedy packs in arrival order, one spare row between windows
        let mut at = 0usize;
        while at < queued.len() {
            let mut end = at;
            let mut used = 0usize;
            while end < queued.len()
                && end - at < crate::gpu_model::diarization::MAX_PACKED
                && used + queued[end].2 + usize::from(end > at) <= cap
            {
                used += queued[end].2 + usize::from(end > at);
                end += 1;
            }
            let group = &queued[at..end];
            let embs: Vec<&CudaSlice<f32>> = group.iter().map(|q| &q.1).collect();
            let wins: Vec<(usize, usize)> = group.iter().map(|q| (q.2, q.3)).collect();
            // every packed recording fits one window: the class they were
            // pre-encoded in (a longer recording prepared since may have
            // switched it)
            self.speaker_class(DIAR.chunk)?;
            let (probs, firsts) = self.diar.predict_packed_device(&embs, &wins)?;
            for (j, &(i, _, rows, keys)) in group.iter().enumerate() {
                let mut dc = e.alloc_u32(rows)?;
                e.tr_codes(probs, firsts[j], &mut dc, rows, keys)?;
                recs[i].codes = Codes::Dev(dc);
                if keep || recs[i].capture {
                    let flat = e.to_host_len(probs, (firsts[j] + rows) * 64)?;
                    let mut p: Vec<[f32; 8]> = flat[firsts[j] * 64..].as_chunks::<8>().0.to_vec();
                    p[keys * STACK..].fill([0.; 8]);
                    recs[i].probs = Some(p);
                }
            }
            at = end;
        }
        Ok(recs)
    }

    /// [`Self::prepare`] up to the speaker branch: the mel, the normalized
    /// copy and the diarizer's input rows on the device (its patterns come
    /// from the packed pass).
    fn prepare_head(
        &mut self,
        pcm: &[f32],
        keep: bool,
        capture: bool,
    ) -> Result<(Rec, CudaSlice<f32>), GpuModelError> {
        if pcm.len() < 160 || pcm.iter().any(|v| !v.is_finite()) {
            return Err(bad("expected at least 10 ms of finite mono 16-kHz audio"));
        }
        let (valid, frames, rows, n_tok) = paddock_models::transcribe::geometry(pcm.len());
        let exec = self.exec.clone();
        let e = &*exec;
        let mut mel = e.alloc(frames * MEL)?;
        self.diar.features_into(pcm, &mut mel, frames)?;
        let mut norm16 = e.alloc_f16(frames * MEL)?;
        e.tr_mel_norm(&mel, &mut norm16, valid, frames)?;
        let mut emb = e.alloc(rows * 512)?;
        self.speaker_class(rows)?;
        self.diar.pre_encode_rows_into(&mel, 0, rows, &mut emb)?;
        let mel_host = if keep {
            Some(e.to_host_len(&mel, frames * MEL)?)
        } else {
            None
        };
        Ok((
            Rec {
                rows,
                n_tok,
                samples: pcm.len(),
                valid,
                capture,
                norm16,
                states: e.alloc(rows * D)?,
                codes: Codes::Host(Vec::new()),
                probs: None,
                mel: mel_host,
            },
            emb,
        ))
    }

    /// One recording's mel, its normalized copy, its speaker branch and its
    /// state plane, ready for the speech branch's windows.
    fn prepare(&mut self, pcm: &[f32], keep: bool, capture: bool) -> Result<Rec, GpuModelError> {
        if pcm.len() < 160 || pcm.iter().any(|v| !v.is_finite()) {
            return Err(bad("expected at least 10 ms of finite mono 16-kHz audio"));
        }
        let (valid, frames, rows, n_tok) = paddock_models::transcribe::geometry(pcm.len());
        let exec = self.exec.clone();
        let e = &*exec;
        // the shared log-mel and the speech branch's normalized copy
        let mut mel = e.alloc(frames * MEL)?;
        self.diar.features_into(pcm, &mut mel, frames)?;
        let mut norm16 = e.alloc_f16(frames * MEL)?;
        e.tr_mel_norm(&mel, &mut norm16, valid, frames)?;
        // the speaker branch: encoder-rate 0/1 activity patterns. A recording
        // inside one diarizer window (40 s) is the offline driver's single
        // predict over every row - nothing for the host to decide, so it stays
        // on the device with the patterns from slot 949; longer ones walk the
        // driver's FIFO and cache on the host.
        self.speaker_class(rows)?;
        let one_window = rows <= DIAR.chunk
            && rows <= self.diar.max_rows()
            && e.has_tr_codes()
            && paddock_models::dev_var_os!("PADDOCK_TR_HOST_DIAR").is_none();
        let (codes, probs) = if one_window {
            let mut emb = e.alloc(rows * 512)?;
            self.diar.pre_encode_rows_into(&mel, 0, rows, &mut emb)?;
            let probs = self.diar.predict_device(&emb, rows, n_tok.clamp(1, rows))?;
            let mut dc = e.alloc_u32(rows)?;
            e.tr_codes(probs, 0, &mut dc, rows, n_tok.min(rows))?;
            let host = if keep || capture {
                let flat = e.to_host_len(probs, rows * 64)?;
                let mut p: Vec<[f32; 8]> = flat.as_chunks::<8>().0.to_vec();
                p[n_tok.min(rows) * STACK..].fill([0.; 8]);
                Some(p)
            } else {
                None
            };
            (Codes::Dev(dc), host)
        } else {
            let embeds = self.diar.pre_encode_rows(&mel, 0, rows)?;
            let (probs, _) =
                crate::diarization::offline_probabilities(&mut self.diar, &embeds, n_tok, DIAR)
                    .map_err(bad)?;
            let codes: Vec<u32> = probs
                .as_chunks::<STACK>()
                .0
                .iter()
                .map(|r| {
                    let mut code = 0u32;
                    for s in 0..8 {
                        let p = r.iter().map(|f| f[s]).sum::<f32>() / STACK as f32;
                        if p > 0.5 {
                            code |= 1 << s;
                        }
                    }
                    code
                })
                .collect();
            (Codes::Host(codes), (keep || capture).then_some(probs))
        };
        let mel_host = if keep {
            Some(e.to_host_len(&mel, frames * MEL)?)
        } else {
            None
        };
        Ok(Rec {
            rows,
            n_tok,
            samples: pcm.len(),
            valid,
            capture,
            norm16,
            states: e.alloc(rows * D)?,
            codes,
            probs,
            mel: mel_host,
        })
    }

    /// The speech branch over every recording's windows, whole windows
    /// packed up to [`PASS_ROWS`] a pass.
    fn run_windows(&mut self, recs: &mut [Rec]) -> Result<(), GpuModelError> {
        let wins: Vec<(usize, Window)> = recs
            .iter()
            .enumerate()
            .flat_map(|(i, r)| windows(r.rows, r.n_tok).into_iter().map(move |w| (i, w)))
            .collect();
        let mut wi = 0usize;
        while wi < wins.len() {
            let mut used = 0usize;
            let first = wi;
            while wi < wins.len() && used + wins[wi].1.len() <= PASS_ROWS {
                used += wins[wi].1.len();
                wi += 1;
            }
            self.pass(recs, &wins[first..wi])?;
        }
        Ok(())
    }

    /// One recording's fusion and projector.
    fn finish(&mut self, mut rec: Rec) -> Result<AudioProbe, GpuModelError> {
        let exec = self.exec.clone();
        let e = &*exec;
        let (rows, n_tok) = (rec.rows, rec.n_tok);
        let uploaded;
        let dcodes = match &rec.codes {
            Codes::Host(v) => {
                uploaded = e.to_device_u32(v)?;
                &uploaded
            }
            Codes::Dev(d) => d,
        };
        let mut fused16 = e.alloc_f16(rows * D)?;
        e.tr_fuse(
            &rec.states,
            (&self.asr_norm.w, &self.asr_norm.b),
            dcodes,
            &self.infusion,
            &mut fused16,
            rows,
            D,
            EPS,
        )?;
        let mut out = e.alloc(n_tok.max(1) * self.hidden)?;
        e.matvec_batch_f16(&self.proj, &fused16, &mut out, n_tok)?;
        e.add_rows_bcast(&mut out, &self.proj_b, n_tok, 1, self.hidden)?;
        let capture = match (&rec.probs, &self.ctc_align) {
            (Some(probs), Some(align)) if rec.capture => {
                // the states plane goes with the capture; a gate's stages
                // read their own copy first
                let states = if rec.mel.is_some() {
                    e.to_device(&e.to_host_len(&rec.states, rows * D)?)?
                } else {
                    std::mem::replace(&mut rec.states, e.alloc(1)?)
                };
                Some(super::ctc::Capture::new(
                    states,
                    n_tok,
                    probs,
                    rec.valid,
                    rec.samples,
                    align.params.vad_threshold,
                ))
            }
            _ => None,
        };
        let stages = match (rec.mel, rec.probs) {
            (Some(mel), Some(probs)) => Some(ProbeStages {
                mel,
                states: e.to_host_len(&rec.states, rows * D)?,
                probs,
                codes: match rec.codes {
                    Codes::Host(v) => v,
                    Codes::Dev(d) => e.to_host_u32(&d)?,
                },
            }),
            _ => None,
        };
        Ok(AudioProbe {
            rows: AudioRows {
                rows: out,
                n: n_tok,
            },
            stages,
            capture,
        })
    }

    /// One encoder pass over whole windows (each tagged with its recording):
    /// the packed rows through the 32 blocks, each window's core rows into
    /// its recording's states.
    fn pass(&mut self, recs: &mut [Rec], wins: &[(usize, Window)]) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let e = &*exec;
        let ws = &mut self.ws;
        let rows: usize = wins.iter().map(|(_, w)| w.len()).sum();
        if rows == 0 {
            return Ok(());
        }
        let mut cu = Vec::with_capacity(wins.len() + 1);
        let mut pos = Vec::with_capacity(rows);
        let mut tiles = Vec::new();
        let mut base = 0u32;
        for (s, (ri, w)) in wins.iter().enumerate() {
            let len = w.len();
            // the stacked mel rows of the window (8 frames of 128 a row)
            e.copy_region_f16(
                &recs[*ri].norm16,
                w.ctx0 * STACK * MEL,
                &mut ws.in16,
                base as usize * STACK * MEL,
                len * STACK * MEL,
            )?;
            cu.push(base);
            pos.extend(0..len as u32);
            for t in 0..len.div_ceil(ENC_ATTN_QTILE) {
                tiles.push(((s as u32) << ENC_ATTN_TILE_SHIFT) | t as u32);
            }
            base += len as u32;
        }
        cu.push(base);
        e.upload_u32(&cu, &mut ws.cu)?;
        e.upload_u32(&pos, &mut ws.pos)?;
        e.upload_u32(&tiles, &mut ws.tiles)?;

        // stacked frames -> 1280, the embedding LayerNorm
        e.matvec_batch_f16(&self.pre, &ws.in16, &mut ws.t, rows)?;
        e.layernorm(&ws.t, &self.embed.w, &self.embed.b, &mut ws.x, rows, D, EPS)?;
        for b in &self.blocks {
            match (&b.lin, &mut ws.lin_in) {
                (Lins::W8 { qkv, out, fc1, fc2 }, LinIn::W8 { h8, a8, ff8 }) => {
                    e.tr_ln_e4m3(&ws.x, (&b.n1.w, &b.n1.b), h8, rows, D, EPS, qkv.inv)?;
                    let o = TrW8Out::H16(&mut ws.qkv16);
                    e.tr_w8a8_gemm(&qkv.w, h8, o, qkv.sw, qkv.sx, D, 3 * D, rows)?;
                    attend(
                        e,
                        b,
                        &mut ws.qkv16,
                        &mut ws.att16,
                        &ws.pos,
                        &ws.cu,
                        &ws.tiles,
                        &self.rope,
                        rows,
                        tiles.len(),
                    )?;
                    e.tr_quant_e4m3_h(&ws.att16, a8, rows * D, out.inv)?;
                    let o = TrW8Out::BiasRes(&mut ws.x, &b.out_b);
                    e.tr_w8a8_gemm(&out.w, a8, o, out.sw, out.sx, D, D, rows)?;
                    e.tr_ln_e4m3(&ws.x, (&b.n2.w, &b.n2.b), h8, rows, D, EPS, fc1.inv)?;
                    let o = TrW8Out::GeluQ(ff8, &b.fc1_b, fc2.inv);
                    e.tr_w8a8_gemm(&fc1.w, h8, o, fc1.sw, fc1.sx, D, FF, rows)?;
                    let o = TrW8Out::BiasRes(&mut ws.x, &b.fc2_b);
                    e.tr_w8a8_gemm(&fc2.w, ff8, o, fc2.sw, fc2.sx, FF, D, rows)?;
                }
                (Lins::F16 { qkv, out, fc1, fc2 }, LinIn::F16 { h16, ff16 }) => {
                    e.layernorm_f16(&ws.x, &b.n1.w, &b.n1.b, h16, rows, D, EPS)?;
                    e.matvec_batch_f16_h(qkv, h16, &mut ws.qkv16, rows)?;
                    attend(
                        e,
                        b,
                        &mut ws.qkv16,
                        &mut ws.att16,
                        &ws.pos,
                        &ws.cu,
                        &ws.tiles,
                        &self.rope,
                        rows,
                        tiles.len(),
                    )?;
                    e.matvec_batch_f16_bias_res(
                        out, &ws.att16, &mut ws.x, &b.out_b, &mut ws.t, rows,
                    )?;
                    e.layernorm_f16(&ws.x, &b.n2.w, &b.n2.b, h16, rows, D, EPS)?;
                    e.matvec_batch_f16_h_gelu(fc1, h16, ff16, &b.fc1_b, rows)?;
                    e.matvec_batch_f16_bias_res(fc2, ff16, &mut ws.x, &b.fc2_b, &mut ws.t, rows)?;
                }
                _ => unreachable!("the planes and the workspace are built in one class"),
            }
        }
        e.layernorm(&ws.x, &self.last.w, &self.last.b, &mut ws.t, rows, D, EPS)?;
        // each window's core rows to their place in the recording
        let mut at = 0usize;
        for (ri, w) in wins {
            let skip = w.start - w.ctx0;
            e.copy_region(
                &ws.t,
                (at + skip) * D,
                &mut recs[*ri].states,
                w.start * D,
                (w.end - w.start) * D,
            )?;
            at += w.len();
        }
        Ok(())
    }
}

/// A block's attention in either class: the per-head q/k LayerNorm with the
/// half-dim rope on the f16 q|k|v rows, then the windows' bidirectional
/// attention into `att16`.
#[allow(clippy::too_many_arguments)]
fn attend(
    e: &GpuExecutor,
    b: &Block,
    qkv16: &mut CudaSlice<f16>,
    att16: &mut CudaSlice<f16>,
    pos: &CudaSlice<u32>,
    cu: &CudaSlice<u32>,
    tiles: &CudaSlice<u32>,
    rope: &CudaSlice<f32>,
    rows: usize,
    n_tiles: usize,
) -> Result<(), GpuModelError> {
    e.tr_qk_norm_rope(
        qkv16,
        (&b.qn.w, &b.qn.b),
        (&b.kn.w, &b.kn.b),
        pos,
        rope,
        rows,
        HEADS,
        EPS,
    )?;
    e.enc_attn_h(
        qkv16, cu, tiles, n_tiles, None, None, att16, rows, HEADS, HD, 0,
    )?;
    Ok(())
}

/// One recording between its preparation and its projector.
/// A recording's speaker activity patterns: built on the device (one
/// diarizer window, the readback-free encode) or by the host driver.
enum Codes {
    Host(Vec<u32>),
    Dev(CudaSlice<u32>),
}

struct Rec {
    rows: usize,
    n_tok: usize,
    /// the recording's samples and valid mel frames (the timestamps' clock)
    samples: usize,
    valid: usize,
    /// keep what the word timestamps need
    capture: bool,
    norm16: CudaSlice<f16>,
    states: CudaSlice<f32>,
    codes: Codes,
    probs: Option<Vec<[f32; 8]>>,
    mel: Option<Vec<f32>>,
}

/// One window of the online inference, in encoder rows: its core `start ..
/// end` and the rows it runs over `ctx0 .. ctx1` (core plus context, cut at
/// the recording's valid rows - padding rows are neither keys nor kept).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Window {
    start: usize,
    end: usize,
    ctx0: usize,
    ctx1: usize,
}

impl Window {
    fn len(&self) -> usize {
        self.ctx1 - self.ctx0
    }
}

/// The windows over `rows` encoder rows of which the first `valid` are real.
fn windows(rows: usize, valid: usize) -> Vec<Window> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < rows.min(valid) {
        let end = (start + WIN).min(rows).min(valid);
        out.push(Window {
            start,
            end,
            ctx0: start.saturating_sub(CTX),
            ctx1: (end + CTX).min(rows).min(valid),
        });
        start += WIN;
    }
    out
}

/// The gates' view of one encode: the projected rows and, when kept, the
/// stages a reference can be held against.
/// The speech branch on its own host thread as well as its own lane. An
/// encode is ~1,200 launches (the speaker branch's ~226 a recording, the
/// speech branch's ~265 a wave); issued from the engine thread they held it
/// for ~12 ms an admission with nothing queued on the language model's
/// stream. Here the engine hands the recordings over and goes on; the reply
/// carries the rows and the lane's event, and the first pass that reads the
/// rows waits on that event device-side.
pub(crate) struct AudioWorker {
    /// the queue; shared with the aligner handle the server holds, and
    /// emptied on drop so that handle never keeps the thread alive
    tx: Arc<Mutex<Option<std::sync::mpsc::Sender<AudioJob>>>>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// what opted-in recordings keep until their transcripts are aligned
    captures: Arc<Mutex<super::ctc::Captures>>,
    align: Option<Arc<super::ctc::CtcAlign>>,
    /// the speech branch's class (W8A8 as shipped, or the f16 A/B)
    pub(crate) w8a8: bool,
}

enum AudioJob {
    Encode {
        pcms: Vec<Vec<f32>>,
        /// keep the gates' stages (one recording)
        keep: bool,
        /// a recording's capture id when its request asked for word
        /// timestamps
        captures: Vec<Option<u64>>,
        reply: std::sync::mpsc::Sender<Result<Encoded, String>>,
    },
    /// the CTC head over a capture's states, for the server's aligner
    Head(super::ctc::HeadJob),
}

/// A finished encode: each recording's rows, the kept stages of a probe,
/// and the lane's event after it.
pub(crate) struct Encoded {
    pub rows: Vec<AudioRows>,
    pub stages: Option<ProbeStages>,
    pub ready: Arc<cudarc::driver::CudaEvent>,
}

/// An encode handed to the worker, not yet collected.
pub(crate) struct PendingEncode(std::sync::mpsc::Receiver<Result<Encoded, String>>);

impl PendingEncode {
    /// The reply if the worker has sent it (never blocks).
    pub(crate) fn try_take(&self) -> Option<Result<Encoded, GpuModelError>> {
        match self.0.try_recv() {
            Ok(r) => Some(r.map_err(bad)),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                Some(Err(bad("the speech encoder thread is gone")))
            }
        }
    }

    /// Block until the worker replies (usually it already has: the encode
    /// ran beside the passes before the one that needs it).
    pub(crate) fn wait(self) -> Result<Encoded, GpuModelError> {
        self.0
            .recv()
            .map_err(|_| bad("the speech encoder thread is gone"))?
            .map_err(bad)
    }
}

impl AudioWorker {
    pub(crate) fn spawn(audio: TranscribeAudio) -> Result<Self, GpuModelError> {
        let (tx, rx) = std::sync::mpsc::channel::<AudioJob>();
        let captures = Arc::new(Mutex::new(super::ctc::Captures::default()));
        let align = audio.ctc_align.clone();
        let w8a8 = audio.w8a8();
        let registry = captures.clone();
        let handle = std::thread::Builder::new()
            .name("paddock-speech".into())
            .spawn(move || {
                let mut audio = audio;
                if audio.exec.stream.context().bind_to_thread().is_err() {
                    return;
                }
                while let Ok(job) = rx.recv() {
                    match job {
                        AudioJob::Encode {
                            pcms,
                            keep,
                            captures,
                            reply,
                        } => {
                            let r = audio.serve_encode(&pcms, keep, &captures, &registry);
                            let _ = reply.send(r.map_err(|e| e.to_string()));
                        }
                        AudioJob::Head(job) => {
                            let r = audio.ctc_columns(&job.states, job.t_enc, &job.cols);
                            let _ = job.reply.send(r.map_err(|e| e.to_string()));
                        }
                    }
                }
            })
            .map_err(|e| bad(format!("speech encoder thread: {e}")))?;
        Ok(Self {
            tx: Arc::new(Mutex::new(Some(tx))),
            handle: Some(handle),
            captures,
            align,
            w8a8,
        })
    }

    fn send(&self, job: AudioJob) -> Result<(), GpuModelError> {
        send(&self.tx, job).map_err(bad)
    }

    /// Hand recordings to the worker; collect with [`PendingEncode::wait`].
    /// `captures` names the recordings whose requests asked for word
    /// timestamps (empty = none did).
    pub(crate) fn submit(
        &self,
        pcms: Vec<Vec<f32>>,
        keep: bool,
        captures: Vec<Option<u64>>,
    ) -> Result<PendingEncode, GpuModelError> {
        let (reply, rx) = std::sync::mpsc::channel();
        self.send(AudioJob::Encode {
            pcms,
            keep,
            captures,
            reply,
        })?;
        Ok(PendingEncode(rx))
    }

    /// [`Self::submit`] and wait.
    pub(crate) fn encode(
        &self,
        pcms: Vec<Vec<f32>>,
        keep: bool,
        captures: Vec<Option<u64>>,
    ) -> Result<Encoded, GpuModelError> {
        self.submit(pcms, keep, captures)?.wait()
    }

    /// The CTC head over states already on the device, waited on.
    pub(crate) fn head_columns(
        &self,
        states: CudaSlice<f32>,
        t_enc: usize,
        cols: Vec<u32>,
    ) -> Result<Vec<f32>, GpuModelError> {
        let (reply, rx) = std::sync::mpsc::channel();
        self.send(AudioJob::Head(super::ctc::HeadJob {
            states,
            t_enc,
            cols,
            reply,
        }))?;
        rx.recv()
            .map_err(|_| bad("the speech encoder thread is gone"))?
            .map_err(bad)
    }

    /// The server's word-timestamp handle, when the checkpoint ships the
    /// CTC head.
    pub(crate) fn aligner(&self) -> Option<Arc<dyn crate::generator::MediaAligner>> {
        let align = self.align.clone()?;
        let tx = self.tx.clone();
        Some(Arc::new(super::ctc::TranscribeAligner {
            captures: self.captures.clone(),
            align,
            head: Box::new(move |job| send(&tx, AudioJob::Head(job))),
        }))
    }

    /// Whether word timestamps can be served (the head loaded).
    pub(crate) fn has_ctc(&self) -> bool {
        self.align.is_some()
    }
}

fn send(
    tx: &Mutex<Option<std::sync::mpsc::Sender<AudioJob>>>,
    job: AudioJob,
) -> Result<(), String> {
    tx.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .ok_or("the speech encoder is shut down")?
        .send(job)
        .map_err(|_| "the speech encoder thread is gone".to_owned())
}

impl TranscribeAudio {
    /// One encode job on the worker: the rows, and the captures of the
    /// recordings that asked, in the registry before the reply goes (the
    /// transcript that reads one can only follow the reply).
    fn serve_encode(
        &mut self,
        pcms: &[Vec<f32>],
        keep: bool,
        captures: &[Option<u64>],
        registry: &Mutex<super::ctc::Captures>,
    ) -> Result<Encoded, GpuModelError> {
        let pcms: Vec<&[f32]> = pcms.iter().map(|p| p.as_slice()).collect();
        let (rows, stages) = if keep {
            let p = self.encode_probe(pcms[0], true)?;
            (vec![p.rows], p.stages)
        } else {
            let want: Vec<bool> = (0..pcms.len())
                .map(|i| captures.get(i).is_some_and(Option::is_some))
                .collect();
            let mut rows = Vec::with_capacity(pcms.len());
            for (i, (r, cap)) in self.encode_many(&pcms, &want)?.into_iter().enumerate() {
                if let (Some(cap), Some(Some(id))) = (cap, captures.get(i)) {
                    registry
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(*id, cap);
                }
                rows.push(r);
            }
            (rows, None)
        };
        let ready = Arc::new(self.ready()?);
        Ok(Encoded {
            rows,
            stages,
            ready,
        })
    }
}

impl Drop for AudioWorker {
    fn drop(&mut self) {
        // closing the queue ends the thread after its current job (the
        // server's aligner handle shares the slot, so it closes too)
        self.tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub(crate) struct AudioProbe {
    pub rows: AudioRows,
    pub stages: Option<ProbeStages>,
    /// what the word timestamps keep, for a recording that asked
    pub capture: Option<super::ctc::Capture>,
}

pub struct ProbeStages {
    /// the shared log-mel `[frames][128]`
    pub mel: Vec<f32>,
    /// the speech branch's states `[rows][1280]` before the fusion
    pub states: Vec<f32>,
    /// the speaker branch's 10-ms probabilities `[rows * 8]`
    pub probs: Vec<[f32; 8]>,
    /// the encoder-rate activity patterns `[rows]`
    pub codes: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_the_valid_rows() {
        // a long recording's windows: 500-row cores, 50 rows of context
        let w = windows(1240, 1239);
        assert_eq!(
            w,
            vec![
                Window {
                    start: 0,
                    end: 500,
                    ctx0: 0,
                    ctx1: 550
                },
                Window {
                    start: 500,
                    end: 1000,
                    ctx0: 450,
                    ctx1: 1050
                },
                Window {
                    start: 1000,
                    end: 1239,
                    ctx0: 950,
                    ctx1: 1239
                },
            ]
        );
        assert_eq!(DIAR.max_rows(), 828);
    }

    #[test]
    fn silent_and_full_patterns_infuse_the_bias_alone() {
        // a constant pattern has zero variance: the 8-wide norm is its bias
        let w8 = [1.0f32; 8];
        let b8 = [0.25f32, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let t = infusion_table(&w8, &b8);
        for code in [0usize, 255] {
            let row = &t[code * D..(code + 1) * D];
            // row = 0.25 * sinusoid[0] - 0.5 * sinusoid[1]
            assert!((row[0] - (0.25 * 0.0 - 0.5 * 1f64.sin()) as f32).abs() < 1e-6);
            assert!((row[1] - (0.25 * 1.0 - 0.5 * 1f64.cos()) as f32).abs() < 1e-6);
        }
    }
}
