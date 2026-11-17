//! Transcribe's word timestamps on the device side: the checkpoint's CTC
//! timestamp head (`model-ctc-bf16.safetensors`, F32 weights), what a
//! request's encode keeps for it, and the handle the server aligns a
//! finished transcript through (`crate::ctc_align` holds the alignment).
//!
//! The head is deferred: an opted-in recording keeps its speech branch's
//! final-norm states and its speaker branch's probabilities, and only once
//! the transcript is known does the head run, window by window as the
//! encoder ran (500 encoder rows with 50 of context each side, rope
//! positions restarting a window), reading back just the transcript's
//! columns of its log-softmax.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaSlice;
use paddock_models::safetensors::ShardedSafetensors;
use paddock_tokenizer::spm::SpmBpe;

use crate::ctc_align::{self, AlignParams, SPEAKERS};
use crate::gpu::GpuExecutor;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::f32_tensor;

const D: usize = 1280;
const HEADS: usize = 16;
const HD: usize = 80;
const ROT: usize = 40;
const FF: usize = 640;
const CORE: usize = 500;
const CTX: usize = 50;
/// CTC rows a window's head pass covers at most (twice core + context).
const MAX_ROWS: usize = 2 * (CORE + 2 * CTX);
const EPS: f32 = 1e-5;
const RATE: f64 = 16_000.0;

fn bad(what: impl std::fmt::Display) -> GpuModelError {
    GpuModelError::Unsupported(format!("Transcribe CTC timestamps: {what}"))
}

struct Norm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

struct Layer {
    n1: Norm,
    qkv: CudaSlice<f32>,
    qn: Norm,
    kn: Norm,
    out: CudaSlice<f32>,
    out_b: CudaSlice<f32>,
    n2: Norm,
    fc1: CudaSlice<f32>,
    fc1_b: CudaSlice<f32>,
    fc2: CudaSlice<f32>,
    fc2_b: CudaSlice<f32>,
}

/// The head, resident on the speech lane.
pub(crate) struct CtcHead {
    /// the ConvT x2 as one GEMM: row k*1280 + o = weight[:, o, k]
    up: CudaSlice<f32>,
    up_b: CudaSlice<f32>,
    embed: Norm,
    layers: Vec<Layer>,
    fin: Norm,
    cls: CudaSlice<f32>,
    cls_b: CudaSlice<f32>,
    classes: usize,
    residual: f32,
    rope: CudaSlice<f32>,
    pub weight_bytes: u64,
}

/// The tokenizer and hyperparameters the alignment needs, shared with the
/// server's handle.
pub(crate) struct CtcAlign {
    pub spm: SpmBpe,
    pub params: AlignParams,
    /// the language model's control tokens a transcript may still carry
    pub specials: Vec<String>,
}

impl CtcHead {
    /// Load the head and the aligner's tables, or None when the checkpoint
    /// ships no head.
    pub(crate) fn load(
        exec: &GpuExecutor,
        st: &ShardedSafetensors,
        dir: &std::path::Path,
    ) -> Result<Option<(Self, CtcAlign)>, GpuModelError> {
        let p = "ctc_timestamp.decoder";
        if st.bytes(&format!("{p}.upsample.weight")).is_none() {
            return Ok(None);
        }
        if !exec.has_tr_ctc() {
            return Err(bad(
                "this kernel pack predates the CTC head (slots 958-960)",
            ));
        }
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .map_err(|e| bad(format!("config.json: {e}")))?,
        )
        .map_err(|e| bad(format!("config.json: {e}")))?;
        let ctc = &config["ctc_timestamps"];
        let params = AlignParams::from_config(ctc).map_err(bad)?;
        let classes = ctc["decoder_config"]["num_classes"]
            .as_u64()
            .ok_or_else(|| bad("decoder_config.num_classes"))? as usize
            + 1;
        let (_, proto) = st
            .bytes("ctc_timestamp.tokenizer.model_proto")
            .ok_or_else(|| bad("no tokenizer.model_proto"))?;
        let spm = SpmBpe::from_proto(proto).map_err(bad)?;
        if spm.piece_count() + 1 != classes {
            return Err(bad(format!(
                "{} pieces for {classes} classes with blank",
                spm.piece_count()
            )));
        }
        let mut wb = 0u64;
        let mut vec = |name: &str, n: usize| -> Result<CudaSlice<f32>, GpuModelError> {
            wb += 4 * n as u64;
            Ok(exec.to_device(&f32_tensor(st, name, n)?)?)
        };
        let up_raw = f32_tensor(st, &format!("{p}.upsample.weight"), D * D * 2)?;
        // [in][out][k] -> [k*D + out][in]
        let mut up = vec![0f32; 2 * D * D];
        for i in 0..D {
            for o in 0..D {
                for k in 0..2 {
                    up[(k * D + o) * D + i] = up_raw[(i * D + o) * 2 + k];
                }
            }
        }
        let ub = f32_tensor(st, &format!("{p}.upsample.bias"), D)?;
        let up_b: Vec<f32> = ub.iter().chain(&ub).copied().collect();
        let t = format!("{p}.transformer");
        let mut layers = Vec::new();
        for l in 0..2 {
            let q = format!("{t}.layers.{l}");
            layers.push(Layer {
                n1: Norm {
                    w: vec(&format!("{q}.norm1.weight"), D)?,
                    b: vec(&format!("{q}.norm1.bias"), D)?,
                },
                qkv: vec(&format!("{q}.attn.w_qkv.weight"), 3 * D * D)?,
                qn: Norm {
                    w: vec(&format!("{q}.attn.q_norm.weight"), HD)?,
                    b: vec(&format!("{q}.attn.q_norm.bias"), HD)?,
                },
                kn: Norm {
                    w: vec(&format!("{q}.attn.k_norm.weight"), HD)?,
                    b: vec(&format!("{q}.attn.k_norm.bias"), HD)?,
                },
                out: vec(&format!("{q}.attn.out_proj.weight"), D * D)?,
                out_b: vec(&format!("{q}.attn.out_proj.bias"), D)?,
                n2: Norm {
                    w: vec(&format!("{q}.norm2.weight"), D)?,
                    b: vec(&format!("{q}.norm2.bias"), D)?,
                },
                fc1: vec(&format!("{q}.ffn.net.0.weight"), FF * D)?,
                fc1_b: vec(&format!("{q}.ffn.net.0.bias"), FF)?,
                fc2: vec(&format!("{q}.ffn.net.3.weight"), D * FF)?,
                fc2_b: vec(&format!("{q}.ffn.net.3.bias"), D)?,
            });
        }
        let embed = Norm {
            w: vec(&format!("{t}.embed_norm.weight"), D)?,
            b: vec(&format!("{t}.embed_norm.bias"), D)?,
        };
        let fin = Norm {
            w: vec(&format!("{t}.final_norm.weight"), D)?,
            b: vec(&format!("{t}.final_norm.bias"), D)?,
        };
        let cls = vec(&format!("{p}.decoder_layers.0.weight"), classes * D)?;
        let cls_b = vec(&format!("{p}.decoder_layers.0.bias"), classes)?;
        let residual = f32_tensor(st, &format!("{p}.residual_scale"), 1)?[0];
        // rope (cos, sin) per window position: inv_freq = 1 / 1e4^(2i/40),
        // the angle position * inv_freq, in f32 as the reference builds it
        let mut rope = Vec::with_capacity(MAX_ROWS * ROT);
        for pos in 0..MAX_ROWS {
            for i in 0..ROT / 2 {
                let inv = 1.0f32 / 10000f32.powf((2 * i) as f32 / ROT as f32);
                let a = pos as f32 * inv;
                rope.push(a.cos());
                rope.push(a.sin());
            }
        }
        let head = Self {
            up: exec.to_device(&up)?,
            up_b: exec.to_device(&up_b)?,
            embed,
            layers,
            fin,
            cls,
            cls_b,
            classes,
            residual,
            rope: exec.to_device(&rope)?,
            weight_bytes: wb + 4 * (up.len() + up_b.len() + rope.len()) as u64,
        };
        let specials = special_tokens(dir);
        Ok(Some((
            head,
            CtcAlign {
                spm,
                params,
                specials,
            },
        )))
    }

    /// The head's log probabilities at `cols` for every CTC row of a
    /// recording's `t_enc` states ([t_enc][1280] on the device), host
    /// `[2 * t_enc][cols]`.
    pub(crate) fn columns(
        &self,
        e: &GpuExecutor,
        states: &CudaSlice<f32>,
        t_enc: usize,
        cols: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        let nc = cols.len();
        let mut out = vec![0f32; 2 * t_enc * nc];
        if t_enc == 0 || nc == 0 {
            return Ok(out);
        }
        let dcols = e.to_device_u32(cols)?;
        let r = MAX_ROWS;
        let mut x = e.alloc((CORE + 2 * CTX) * D)?;
        let mut u = e.alloc(r * D)?;
        let mut h = e.alloc(r * D)?;
        let mut a = e.alloc(r * D)?;
        let mut qkv = e.alloc(r * 3 * D)?;
        let mut att = e.alloc(r * D)?;
        let mut m = e.alloc(r * FF)?;
        let mut z = e.alloc(r * D)?;
        let mut logits = e.alloc(r * self.classes)?;
        let mut lp = e.alloc(r * nc)?;
        for start in (0..t_enc).step_by(CORE) {
            let end = (start + CORE).min(t_enc);
            let (cs, ce) = (start.saturating_sub(CTX), (end + CTX).min(t_enc));
            let tw = ce - cs;
            let rows = 2 * tw;
            e.copy_region(states, cs * D, &mut x, 0, tw * D)?;
            // upsample: [tw][2560] read as [2 * tw][1280]
            e.gemm_f32(&self.up, D, 2 * D, &x, &mut u, tw)?;
            e.bias_add(&mut u, &self.up_b, tw, 2 * D)?;
            e.layernorm(&u, &self.embed.w, &self.embed.b, &mut h, rows, D, EPS)?;
            for l in &self.layers {
                e.layernorm(&h, &l.n1.w, &l.n1.b, &mut a, rows, D, EPS)?;
                e.gemm_f32(&l.qkv, D, 3 * D, &a, &mut qkv, rows)?;
                e.tr_ctc_qk_norm_rope(
                    &mut qkv,
                    (&l.qn.w, &l.qn.b),
                    (&l.kn.w, &l.kn.b),
                    &self.rope,
                    rows,
                    HEADS,
                    HD,
                    ROT,
                    EPS,
                )?;
                e.tr_ctc_attention(&qkv, &mut att, rows, HEADS, HD)?;
                e.gemm_f32(&l.out, D, D, &att, &mut z, rows)?;
                e.bias_add(&mut z, &l.out_b, rows, D)?;
                e.scale_add(&mut h, &z, 1.0, rows * D)?;
                e.layernorm(&h, &l.n2.w, &l.n2.b, &mut a, rows, D, EPS)?;
                e.gemm_f32(&l.fc1, D, FF, &a, &mut m, rows)?;
                e.bias_add(&mut m, &l.fc1_b, rows, FF)?;
                e.gelu_erf(&mut m, rows * FF)?;
                e.gemm_f32(&l.fc2, FF, D, &m, &mut z, rows)?;
                e.bias_add(&mut z, &l.fc2_b, rows, D)?;
                e.scale_add(&mut h, &z, 1.0, rows * D)?;
            }
            e.layernorm(&h, &self.fin.w, &self.fin.b, &mut z, rows, D, EPS)?;
            // the residual input is the upsampled states, not their norm
            e.scale_add(&mut u, &z, self.residual, rows * D)?;
            e.gemm_f32(&self.cls, D, self.classes, &u, &mut logits, rows)?;
            e.tr_ctc_lse_gather(
                &logits,
                &self.cls_b,
                rows,
                self.classes,
                &dcols,
                nc,
                &mut lp,
            )?;
            let (k0, kn) = (2 * (start - cs), 2 * (end - start));
            let host = e.to_host_len(&lp, (k0 + kn) * nc)?;
            out[2 * start * nc..2 * end * nc].copy_from_slice(&host[k0 * nc..]);
        }
        Ok(out)
    }
}

/// The language model's control tokens (the tokenizer config's special-
/// token entries) - what the reference cleans out of a transcript.
fn special_tokens(dir: &std::path::Path) -> Vec<String> {
    let Ok(raw) = std::fs::read(dir.join("tokenizer_config.json")) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&raw) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for k in ["bos_token", "eos_token", "unk_token", "pad_token"] {
        match &v[k] {
            serde_json::Value::String(s) => out.push(s.clone()),
            serde_json::Value::Object(o) => {
                if let Some(s) = o.get("content").and_then(|c| c.as_str()) {
                    out.push(s.to_owned());
                }
            }
            _ => {}
        }
    }
    if let Some(a) = v["additional_special_tokens"].as_array() {
        out.extend(a.iter().filter_map(|t| t.as_str().map(str::to_owned)));
    }
    out.sort();
    out.dedup();
    out
}

/// What an opted-in recording keeps until its transcript is aligned.
pub(crate) struct Capture {
    /// the speech branch's final-norm states `[t_enc][1280]`
    pub states: CudaSlice<f32>,
    pub t_enc: usize,
    /// the 10 ms speaker probabilities mean-pooled by 8 (the last group
    /// over the frames present)
    pub speaker80: Vec<[f32; SPEAKERS]>,
    /// one bit per speaker a 10 ms frame: probability > vad_threshold
    pub labels10: Vec<u8>,
    pub duration: f64,
    at: std::time::Instant,
}

impl Capture {
    /// From an encode's planes: `probs` = the speaker branch's 10 ms rows,
    /// the first `t_mel` valid.
    pub(crate) fn new(
        states: CudaSlice<f32>,
        t_enc: usize,
        probs: &[[f32; SPEAKERS]],
        t_mel: usize,
        samples: usize,
        vad: f64,
    ) -> Self {
        let t_mel = t_mel.min(probs.len());
        let speaker80 = (0..t_enc)
            .map(|t| {
                let (a, b) = (8 * t, (8 * t + 8).min(t_mel));
                let mut o = [0f32; SPEAKERS];
                if b > a {
                    for s in 0..SPEAKERS {
                        let sum: f32 = probs[a..b].iter().map(|p| p[s]).sum();
                        o[s] = sum / (b - a) as f32;
                    }
                }
                o
            })
            .collect();
        let v = vad as f32;
        let labels10 = probs[..t_mel]
            .iter()
            .map(|p| (0..SPEAKERS).fold(0u8, |m, s| if p[s] > v { m | 1 << s } else { m }))
            .collect();
        Self {
            states,
            t_enc,
            speaker80,
            labels10,
            duration: samples as f64 / RATE,
            at: std::time::Instant::now(),
        }
    }

    fn bytes(&self) -> usize {
        self.t_enc * D * 4 + self.labels10.len() + self.speaker80.len() * 32
    }
}

/// Captures waiting for their transcripts. An entry nobody aligns goes after
/// `TTL`, and the oldest go first past `MAX_BYTES` (their requests then get
/// `capture_unavailable`, the reference's word for it).
#[derive(Default)]
pub(crate) struct Captures {
    map: HashMap<u64, Capture>,
}

const TTL: std::time::Duration = std::time::Duration::from_secs(600);
const MAX_BYTES: usize = 2 << 30;

impl Captures {
    pub(crate) fn insert(&mut self, id: u64, c: Capture) {
        let now = std::time::Instant::now();
        self.map.retain(|_, v| now.duration_since(v.at) < TTL);
        self.map.insert(id, c);
        while self.map.values().map(Capture::bytes).sum::<usize>() > MAX_BYTES {
            let Some(old) = self.map.iter().min_by_key(|(_, v)| v.at).map(|(k, _)| *k) else {
                break;
            };
            self.map.remove(&old);
        }
    }

    pub(crate) fn take(&mut self, id: u64) -> Option<Capture> {
        self.map.remove(&id)
    }
}

/// A head pass the speech lane runs for the server's aligner.
pub(crate) struct HeadJob {
    pub states: CudaSlice<f32>,
    pub t_enc: usize,
    pub cols: Vec<u32>,
    pub reply: std::sync::mpsc::Sender<Result<Vec<f32>, String>>,
}

/// The server's handle ([`crate::generator::MediaAligner`]).
pub(crate) struct TranscribeAligner {
    pub captures: Arc<Mutex<Captures>>,
    pub align: Arc<CtcAlign>,
    /// hands a head pass to the speech lane's thread
    pub head: Box<dyn Fn(HeadJob) -> Result<(), String> + Send + Sync>,
}

impl crate::generator::MediaAligner for TranscribeAligner {
    fn align(&self, id: u64, transcript: &str) -> serde_json::Value {
        let cap = self
            .captures
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take(id);
        let Some(cap) = cap else {
            return ctc_align::empty(Some("no_capture"), &[]);
        };
        let dia = ctc_align::segments(&cap.labels10, cap.duration);
        let text = ctc_align::clean(transcript, &self.align.specials);
        if text.is_empty() {
            return ctc_align::empty(None, &dia);
        }
        let blank = self.align.spm.piece_count() as u32;
        let frames = 2 * cap.t_enc;
        let failed = |why: &str| {
            tracing::warn!(why, "Transcribe CTC timestamps: alignment failed");
            ctc_align::empty(Some("alignment_failed"), &dia)
        };
        let prep = match ctc_align::prepare(&text, &self.align.spm, blank, frames) {
            Ok(p) => p,
            Err(e) => return failed(&e),
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let job = HeadJob {
            states: cap.states,
            t_enc: cap.t_enc,
            cols: prep.columns.clone(),
            reply: tx,
        };
        if let Err(e) = (self.head)(job) {
            return failed(&e);
        }
        let logp = match rx.recv() {
            Ok(Ok(l)) => l,
            Ok(Err(e)) => return failed(&e),
            Err(_) => return failed("the speech lane is gone"),
        };
        let inputs = ctc_align::Inputs {
            logp: &logp,
            ctc_frames: frames,
            speaker80: &cap.speaker80,
            labels10: &cap.labels10,
            duration: cap.duration,
        };
        match ctc_align::align(&prep, &inputs, &self.align.params) {
            Ok(v) => v,
            Err(e) => failed(&e),
        }
    }

    fn forget(&self, id: u64) {
        self.captures
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take(id);
    }
}
