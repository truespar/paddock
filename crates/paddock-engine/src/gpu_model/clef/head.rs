//! The joint schema head (`JointSchemaHead` in the reference's
//! `joint_schema_model.py`) over a pass's final hidden rows. Its projections
//! run on the diarization lane's stored-weight GEMM (slot 722: the
//! activation split three ways in bf16 against the exact bf16 weight, F32
//! class), everything else on Clef's own F32 passes (slots 724-731); see
//! `packs/cuda/src/clef.cuh`.
//!
//! Per request, as the reference walks it: LayerNorm the hidden rows; the
//! memory is their projection; a question is the mean of its instruction's
//! rows, an option the mean of its text's rows plus the mean of its tokens'
//! `lm_head` rows; options route evidence from the memory (two layers),
//! summarize into their field, the fields decode against the memory (four
//! `TransformerDecoderLayer`s, norm first); every option is scored against
//! its field. Requests never meet: every attention runs over its own
//! request's keys.

use std::path::Path;

use cudarc::driver::CudaSlice;
use paddock_models::clef::{ClefHeadConfig, HEAD_WEIGHTS};
use paddock_models::safetensors::{SafetensorsFile, StDtype};

use super::GpuClef;
use crate::gpu::{ClefScoreIn, DiarEpi, DiarWeights, GpuExecutor};
use crate::gpu_model::gpt_oss::GpuModelError;

/// Most questions, options and requests the head takes in one pass.
pub const MAX_QUESTIONS: usize = 1024;
pub const MAX_OPTIONS: usize = 4096;
pub const MAX_REQUESTS: usize = 256;

/// LayerNorm eps of every norm in the head (torch's default).
const LN_EPS: f32 = 1e-5;

struct Norm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

/// A linear layer's weights as stored: BF16 rows `[n][k]`, or a GGUF's
/// Q8_0 repacked to int8 rows and F32 block scales `[k/32][n]` (slot 722's
/// two weight classes).
pub(super) enum LinW {
    Bf16(CudaSlice<u8>),
    Q8(CudaSlice<u8>, CudaSlice<f32>),
}

/// A linear layer: its weights, F32 bias.
pub(super) struct Lin {
    pub(super) w: LinW,
    pub(super) b: Option<CudaSlice<f32>>,
    pub(super) k: usize,
    pub(super) n: usize,
}

/// `nn.MultiheadAttention` with its packed in-projection split: q rows, then
/// k and v rows together (one GEMM over the keys' plane).
struct Mha {
    q: Lin,
    kv: Lin,
    o: Lin,
}

struct Evidence {
    query_norm: Norm,
    memory_norm: Norm,
    attn: Mha,
    ff_norm: Norm,
    ff1: Lin,
    ff2: Lin,
}

struct Decoder {
    norm1: Norm,
    norm2: Norm,
    norm3: Norm,
    /// self-attention: q, k and v in one GEMM (the packed in_proj)
    sa_qkv: Lin,
    sa_o: Lin,
    ca: Mha,
    ff1: Lin,
    ff2: Lin,
}

pub(super) struct HeadW {
    cfg: ClefHeadConfig,
    hidden_norm: Norm,
    memory: Lin,
    question: Lin,
    option_question: Lin,
    global: Lin,
    option_context: Lin,
    option_lexical: Lin,
    type_emb: CudaSlice<f32>,
    evidence: Vec<Evidence>,
    option_summary_norm: Norm,
    layers: Vec<Decoder>,
    field_norm: Norm,
    option_norm: Norm,
    scorer0: Lin,
    scorer3_w: CudaSlice<f32>,
    scorer3_b: f32,
    prior_scale: f32,
    joint_scale: f32,
    gate: f32,
    pub(super) bytes: u64,
}

/// Head planes sized for the per-pass maxima.
pub(super) struct HeadWs {
    qvec: CudaSlice<f32>,
    glob: CudaSlice<f32>,
    ctx: CudaSlice<f32>,
    lex: CudaSlice<f32>,
    wide: CudaSlice<f32>,
    oq: CudaSlice<f32>,
    tq: CudaSlice<f32>,
    att: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    qo: CudaSlice<f32>,
    fields: CudaSlice<f32>,
    summ: CudaSlice<f32>,
    gproj: CudaSlice<f32>,
    onorm: CudaSlice<f32>,
    hid: CudaSlice<f32>,
    logits: CudaSlice<f32>,
}

fn widen(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

impl HeadW {
    pub(super) fn load(
        e: &GpuExecutor,
        dir: &Path,
        cfg: &ClefHeadConfig,
        hidden: usize,
    ) -> Result<Self, GpuModelError> {
        let bad = |m: String| GpuModelError::Unsupported(format!("Clef head: {m}"));
        let st = SafetensorsFile::open(&dir.join(HEAD_WEIGHTS)).map_err(|e| bad(e.to_string()))?;
        let bytes = std::cell::Cell::new(0u64);
        let raw = |name: &str, shape: &[usize]| -> Result<&[u8], GpuModelError> {
            let (t, b) = st
                .bytes(name)
                .ok_or_else(|| bad(format!("{name}: tensor missing")))?;
            if t.dtype != StDtype::Bf16 || t.shape != shape {
                return Err(bad(format!(
                    "{name}: {:?} {:?}, expected BF16 {shape:?}",
                    t.dtype, t.shape
                )));
            }
            Ok(b)
        };
        let f32v = |name: &str, shape: &[usize]| -> Result<CudaSlice<f32>, GpuModelError> {
            let v = widen(raw(name, shape)?);
            bytes.set(bytes.get() + 4 * v.len() as u64);
            Ok(e.to_device(&v)?)
        };
        // `name` is the module; torch's MultiheadAttention stores its packed
        // projection as `in_proj_weight` / `in_proj_bias`, everything else
        // as `.weight` / `.bias`
        let lin_rows =
            |name: &str, rows: std::ops::Range<usize>, k: usize, bias: bool, full_n: usize| {
                let (wn, bn) = match name.strip_suffix(".in_proj") {
                    Some(m) => (format!("{m}.in_proj_weight"), format!("{m}.in_proj_bias")),
                    None => (format!("{name}.weight"), format!("{name}.bias")),
                };
                let w = raw(&wn, &[full_n, k])?;
                let w = &w[rows.start * k * 2..rows.end * k * 2];
                bytes.set(bytes.get() + w.len() as u64);
                let b = if bias {
                    let all = widen(raw(&bn, &[full_n])?);
                    let v = all[rows.clone()].to_vec();
                    bytes.set(bytes.get() + 4 * v.len() as u64);
                    Some(e.to_device(&v)?)
                } else {
                    None
                };
                Ok::<_, GpuModelError>(Lin {
                    w: LinW::Bf16(e.to_device_u8(w)?),
                    b,
                    k,
                    n: rows.len(),
                })
            };
        let lin = |name: &str, n: usize, k: usize, bias: bool| lin_rows(name, 0..n, k, bias, n);
        let norm = |name: &str, d: usize| -> Result<Norm, GpuModelError> {
            Ok(Norm {
                w: f32v(&format!("{name}.weight"), &[d])?,
                b: f32v(&format!("{name}.bias"), &[d])?,
            })
        };
        let (w, ff) = (cfg.width, cfg.feedforward);
        let mha = |name: &str| -> Result<Mha, GpuModelError> {
            // in_proj_weight / in_proj_bias, q k v stacked
            let base = format!("{name}.in_proj");
            Ok(Mha {
                q: lin_rows(&base, 0..w, w, true, 3 * w)?,
                kv: lin_rows(&base, w..3 * w, w, true, 3 * w)?,
                o: lin(&format!("{name}.out_proj"), w, w, true)?,
            })
        };
        let evidence = (0..cfg.routing_layers)
            .map(|i| {
                let p = format!("evidence_layers.{i}");
                Ok(Evidence {
                    query_norm: norm(&format!("{p}.query_norm"), w)?,
                    memory_norm: norm(&format!("{p}.memory_norm"), w)?,
                    attn: mha(&format!("{p}.attention"))?,
                    ff_norm: norm(&format!("{p}.feedforward_norm"), w)?,
                    ff1: lin(&format!("{p}.feedforward.0"), ff, w, true)?,
                    ff2: lin(&format!("{p}.feedforward.3"), w, ff, true)?,
                })
            })
            .collect::<Result<Vec<_>, GpuModelError>>()?;
        let layers = (0..cfg.layers)
            .map(|i| {
                let p = format!("layers.{i}");
                Ok(Decoder {
                    norm1: norm(&format!("{p}.norm1"), w)?,
                    norm2: norm(&format!("{p}.norm2"), w)?,
                    norm3: norm(&format!("{p}.norm3"), w)?,
                    sa_qkv: lin_rows(&format!("{p}.self_attn.in_proj"), 0..3 * w, w, true, 3 * w)?,
                    sa_o: lin(&format!("{p}.self_attn.out_proj"), w, w, true)?,
                    ca: mha(&format!("{p}.multihead_attn"))?,
                    ff1: lin(&format!("{p}.linear1"), ff, w, true)?,
                    ff2: lin(&format!("{p}.linear2"), w, ff, true)?,
                })
            })
            .collect::<Result<Vec<_>, GpuModelError>>()?;
        let scalar = |name: &str| -> Result<f32, GpuModelError> { Ok(widen(raw(name, &[])?)[0]) };
        let s3 = widen(raw("residual_scorer.3.weight", &[1, w])?);
        let s3b = widen(raw("residual_scorer.3.bias", &[1])?)[0];
        // the reference's own arithmetic on its three scalars, once: the
        // clamp(max = ln 100) and exp of the two scales, the gate's sigmoid
        let cap = 100f32.ln();
        let prior_scale = scalar("prior_logit_scale")?.min(cap).exp();
        let joint_scale = scalar("joint_logit_scale")?.min(cap).exp();
        let g = scalar("residual_gate")?;
        let gate = 1.0 / (1.0 + (-g).exp());
        let head = Self {
            hidden_norm: norm("hidden_norm", hidden)?,
            memory: lin("memory_projection", w, hidden, false)?,
            question: lin("question_projection", w, hidden, false)?,
            option_question: lin("option_question_projection", w, hidden, false)?,
            global: lin("global_projection", w, hidden, false)?,
            option_context: lin("option_context_projection", w, hidden, false)?,
            option_lexical: lin("option_lexical_projection", w, hidden, false)?,
            type_emb: f32v("type_embedding.weight", &[3, w])?,
            evidence,
            option_summary_norm: norm("option_summary_norm", w)?,
            layers,
            field_norm: norm("field_norm", w)?,
            option_norm: norm("option_norm", w)?,
            scorer0: lin("residual_scorer.0", w, 4 * w, true)?,
            scorer3_w: e.to_device(&s3)?,
            scorer3_b: s3b,
            prior_scale,
            joint_scale,
            gate,
            cfg: cfg.clone(),
            bytes: 0,
        };
        Ok(Self {
            bytes: bytes.get(),
            ..head
        })
    }
}

impl HeadW {
    /// The head off a Clef GGUF (llama.cpp's `clef` schema): `decision.*`,
    /// `token_types` and the `dec.blk.N` blocks - routing blocks first, then
    /// the decoder layers - with `nn.MultiheadAttention`'s packed projection
    /// split into q, k and v. Q8_0 weights stay Q8_0 (slot 722's int8 rows
    /// and F32 block scales), BF16 stays BF16, the rest F32 as stored; the
    /// three scalars are stored as the values used (clamp/exp/sigmoid done).
    pub(super) fn load_gguf(
        e: &GpuExecutor,
        src: &super::gguf::Src<'_>,
        cfg: &ClefHeadConfig,
        hidden: usize,
    ) -> Result<Self, GpuModelError> {
        let bytes = std::cell::Cell::new(0u64);
        let up = |v: &[f32]| -> Result<CudaSlice<f32>, GpuModelError> {
            bytes.set(bytes.get() + 4 * v.len() as u64);
            Ok(e.to_device(v)?)
        };
        let vec = |name: &str, n: usize| up(&src.f32s(name, &[n])?);
        let norm = |name: &str, d: usize| -> Result<Norm, GpuModelError> {
            Ok(Norm {
                w: vec(&format!("{name}.weight"), d)?,
                b: vec(&format!("{name}.bias"), d)?,
            })
        };
        // `bases` stacked by rows, each `k -> n`, with or without biases
        let lin = |bases: &[&str], k: usize, n: usize, bias: bool| -> Result<Lin, GpuModelError> {
            let names: Vec<String> = bases.iter().map(|b| format!("{b}.weight")).collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let ns = vec![n; bases.len()];
            let (ty, raw) = src.rows(&refs, k, &ns, false)?;
            let rows = n * bases.len();
            let w = if ty == paddock_models::ggml_type::GgmlType::Q8_0 {
                let (q, scale) = crate::gpu::ClefQ8::repack(&raw, rows, k);
                let scale: Vec<f32> = scale
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::f16::from_le_bytes(*c).to_f32())
                    .collect();
                bytes.set(bytes.get() + q.len() as u64);
                LinW::Q8(e.to_device_u8(&q)?, up(&scale)?)
            } else {
                bytes.set(bytes.get() + raw.len() as u64);
                LinW::Bf16(e.to_device_u8(&raw)?)
            };
            let b = if bias {
                let mut v = Vec::with_capacity(rows);
                for base in bases {
                    v.extend(src.f32s(&format!("{base}.bias"), &[n])?);
                }
                Some(up(&v)?)
            } else {
                None
            };
            Ok(Lin { w, b, k, n: rows })
        };
        let (w, ff) = (cfg.width, cfg.feedforward);
        let blk = |i: usize, n: &str| format!("dec.blk.{i}.{n}");
        let cross = |i: usize| -> Result<Mha, GpuModelError> {
            Ok(Mha {
                q: lin(&[&blk(i, "cross_attn_q")], w, w, true)?,
                kv: lin(
                    &[&blk(i, "cross_attn_k"), &blk(i, "cross_attn_v")],
                    w,
                    w,
                    true,
                )?,
                o: lin(&[&blk(i, "cross_attn_o")], w, w, true)?,
            })
        };
        let evidence = (0..cfg.routing_layers)
            .map(|i| {
                Ok(Evidence {
                    query_norm: norm(&blk(i, "cross_attn_norm"), w)?,
                    memory_norm: norm(&blk(i, "cross_attn_norm_kv"), w)?,
                    attn: cross(i)?,
                    ff_norm: norm(&blk(i, "ffn_norm"), w)?,
                    ff1: lin(&[&blk(i, "ffn_up")], w, ff, true)?,
                    ff2: lin(&[&blk(i, "ffn_down")], ff, w, true)?,
                })
            })
            .collect::<Result<Vec<_>, GpuModelError>>()?;
        let layers = (cfg.routing_layers..cfg.routing_layers + cfg.layers)
            .map(|i| {
                Ok(Decoder {
                    norm1: norm(&blk(i, "attn_norm"), w)?,
                    norm2: norm(&blk(i, "cross_attn_norm"), w)?,
                    norm3: norm(&blk(i, "ffn_norm"), w)?,
                    sa_qkv: lin(
                        &[&blk(i, "attn_q"), &blk(i, "attn_k"), &blk(i, "attn_v")],
                        w,
                        w,
                        true,
                    )?,
                    sa_o: lin(&[&blk(i, "attn_o")], w, w, true)?,
                    ca: cross(i)?,
                    ff1: lin(&[&blk(i, "ffn_up")], w, ff, true)?,
                    ff2: lin(&[&blk(i, "ffn_down")], ff, w, true)?,
                })
            })
            .collect::<Result<Vec<_>, GpuModelError>>()?;
        let scales = src.f32s("decision.scales", &[3])?;
        let s3 = src.f32s("decision.scorer_out.weight", &[w, 1])?;
        let s3b = src.f32s("decision.scorer_out.bias", &[1])?[0];
        let proj = |name: &str| lin(&[&format!("decision.{name}")], hidden, w, false);
        let head = Self {
            hidden_norm: norm("decision.hidden_norm", hidden)?,
            memory: proj("proj_memory")?,
            question: proj("proj_question")?,
            option_question: proj("proj_option_question")?,
            global: proj("proj_global")?,
            option_context: proj("proj_option_context")?,
            option_lexical: proj("proj_option_lexical")?,
            type_emb: up(&src.f32s("token_types.weight", &[w, 3])?)?,
            evidence,
            option_summary_norm: norm("decision.option_summary_norm", w)?,
            layers,
            field_norm: norm("decision.field_norm", w)?,
            option_norm: norm("decision.option_norm", w)?,
            scorer0: lin(&["decision.scorer"], 4 * w, w, true)?,
            scorer3_w: up(&s3)?,
            scorer3_b: s3b,
            prior_scale: scales[0],
            joint_scale: scales[1],
            gate: scales[2],
            cfg: cfg.clone(),
            bytes: 0,
        };
        Ok(Self {
            bytes: bytes.get(),
            ..head
        })
    }
}

impl HeadWs {
    pub(super) fn floats(cfg: &ClefHeadConfig, hidden: usize) -> usize {
        let (mq, mo, mr, w) = (MAX_QUESTIONS, MAX_OPTIONS, MAX_REQUESTS, cfg.width);
        let wide = mo.max(mq) * hidden.max(cfg.feedforward).max(4 * w);
        mq * hidden
            + mr * hidden
            + 2 * mo * hidden
            + wide
            + mo * w * 4
            + 2 * mo.max(mq) * w
            + mq * 3 * w
            + 3 * mq * w
            + mr * w
            + mo
            + mq * w
    }

    pub(super) fn new(
        e: &GpuExecutor,
        cfg: &ClefHeadConfig,
        hidden: usize,
    ) -> Result<Self, GpuModelError> {
        let (mq, mo, mr, w) = (MAX_QUESTIONS, MAX_OPTIONS, MAX_REQUESTS, cfg.width);
        let p = |n: usize| e.alloc(n);
        Ok(Self {
            qvec: p(mq * hidden)?,
            glob: p(mr * hidden)?,
            ctx: p(mo * hidden)?,
            lex: p(mo * hidden)?,
            wide: p(mo.max(mq) * hidden.max(cfg.feedforward).max(4 * w))?,
            oq: p(mo * w)?,
            tq: p(mo.max(mq) * w)?,
            att: p(mo.max(mq) * w)?,
            qkv: p(mq * 3 * w)?,
            qo: p(mq * w)?,
            fields: p(mq * w)?,
            summ: p(mq * w)?,
            gproj: p(mr * w)?,
            onorm: p(mo * w)?,
            hid: p(mo * w)?,
            logits: p(mo)?,
        })
    }
}

pub use crate::clef_decision::{ClefImage, ClefLogits, ClefQuestion, ClefRequest};

/// `y = epi(x . W + b)` over `m` rows, `x` from element `xo`.
fn gemm(
    e: &GpuExecutor,
    (x, xo): (&CudaSlice<f32>, usize),
    l: &Lin,
    y: &mut CudaSlice<f32>,
    m: usize,
    epi: DiarEpi,
) -> Result<(), GpuModelError> {
    let w = match &l.w {
        LinW::Bf16(w) => DiarWeights::Bf16(w),
        LinW::Q8(q, scale) => DiarWeights::Q8 { q, scale },
    };
    e.diar_gemm((x, xo), w, l.b.as_ref(), y, (l.k, l.n, m), epi, None)?;
    Ok(())
}

impl GpuClef {
    /// Run the backbone and the head over `reqs`, packed back to back.
    pub fn forward(&mut self, reqs: &[ClefRequest<'_>]) -> Result<ClefLogits, GpuModelError> {
        self.pass(reqs, None)
    }

    /// The head alone over given final hidden rows (`[rows][hidden]`, the
    /// requests back to back) - the head's own gate against the reference's
    /// hidden state, with the backbone's numerics out of the picture.
    pub fn head_on_hidden(
        &mut self,
        reqs: &[ClefRequest<'_>],
        hidden: &[f32],
    ) -> Result<ClefLogits, GpuModelError> {
        self.pass(reqs, Some(hidden))
    }

    fn pass(
        &mut self,
        reqs: &[ClefRequest<'_>],
        hidden: Option<&[f32]>,
    ) -> Result<ClefLogits, GpuModelError> {
        let bad = |m: String| GpuModelError::Unsupported(format!("Clef pass: {m}"));
        let nr = reqs.len();
        let nq: usize = reqs.iter().map(|r| r.questions.len()).sum();
        let no: usize = reqs
            .iter()
            .flat_map(|r| r.questions.iter())
            .map(|q| q.options.len())
            .sum();
        if nr == 0 || nr > MAX_REQUESTS || nq == 0 || nq > MAX_QUESTIONS || no > MAX_OPTIONS {
            return Err(bad(format!(
                "{nr} requests / {nq} questions / {no} options (at most {MAX_REQUESTS} / \
                 {MAX_QUESTIONS} / {MAX_OPTIONS} a pass)"
            )));
        }
        // the pass's ids and runs, and every index plane the head walks
        let mut ids = Vec::new();
        let mut runs = Vec::with_capacity(nr);
        let (mut qspans, mut ospans, mut lex_ids, mut lex_spans) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut glob_spans, mut qof, mut rof, mut qtypes, mut qopts) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut opt_mem, mut q_mem, mut q_self) = (Vec::new(), Vec::new(), Vec::new());
        let (mut images, mut pass_images) = (Vec::new(), Vec::new());
        let vocab = self.cfg.vocab as u32;
        for (ri, r) in reqs.iter().enumerate() {
            let (s, n) = (ids.len(), r.ids.len());
            if n == 0 || r.questions.is_empty() {
                return Err(bad(format!("request {ri} has no rows or no questions")));
            }
            if r.ids.iter().any(|&t| t >= vocab) {
                return Err(bad(format!("request {ri}: a token id past the vocabulary")));
            }
            if !r.images.is_empty() {
                // every image on its own run of <|image_pad|> rows, and no
                // pad row without an image (the reference refuses a count
                // mismatch too)
                let Some(v) = self.vision.as_ref() else {
                    return Err(bad(format!(
                        "request {ri} carries images; this model's vision tower is not loaded"
                    )));
                };
                let pad = v.cfg.image_token;
                let mut next = 0usize;
                for im in r.images {
                    let t = im.tokens();
                    let (rh, rw) = im.resized;
                    if rh == 0
                        || rw == 0
                        || !rh.is_multiple_of(32)
                        || !rw.is_multiple_of(32)
                        || im.row < next
                        || im.row + t > n
                        || r.ids[im.row..im.row + t].iter().any(|&x| x != pad)
                        || im.rgb.len() != im.width * im.height * 3
                    {
                        return Err(bad(format!("request {ri}: an image off its rows")));
                    }
                    next = im.row + t;
                    pass_images.push(super::PassImage {
                        row: s + im.row,
                        grid: (rh / 32, rw / 32),
                    });
                    images.push(im);
                }
                let want: usize = r.images.iter().map(|i| i.tokens()).sum();
                if r.ids.iter().filter(|&&x| x == pad).count() != want {
                    return Err(bad(format!(
                        "request {ri}: image tokens and image features do not match"
                    )));
                }
            }
            runs.push((s, n));
            ids.extend_from_slice(r.ids);
            glob_spans.extend([(s + n - 1) as u32, (s + n) as u32]);
            let first_q = qtypes.len();
            for q in r.questions {
                let ok = |(a, b): (usize, usize)| a < b && b <= n;
                if !ok(q.span)
                    || q.options.is_empty()
                    || !q.options.iter().all(|&o| ok(o))
                    || q.qtype > 2
                {
                    return Err(bad(format!(
                        "request {ri}: a question span outside its rows"
                    )));
                }
                let qi = qtypes.len() as u32;
                qspans.extend([(s + q.span.0) as u32, (s + q.span.1) as u32]);
                qtypes.push(q.qtype);
                rof.push(ri as u32);
                qopts.extend([qof.len() as u32, q.options.len() as u32]);
                q_mem.extend([s as u32, n as u32]);
                for &(a, b) in &q.options {
                    ospans.extend([(s + a) as u32, (s + b) as u32]);
                    let l0 = lex_ids.len() as u32;
                    lex_ids.extend_from_slice(&r.ids[a..b]);
                    lex_spans.extend([l0, lex_ids.len() as u32]);
                    qof.push(qi);
                    opt_mem.extend([s as u32, n as u32]);
                }
            }
            let nq_r = (qtypes.len() - first_q) as u32;
            for _ in first_q..qtypes.len() {
                q_self.extend([first_q as u32, nq_r]);
            }
        }
        match hidden {
            None => {
                if let Some(v) = self.vision.as_mut().filter(|_| !images.is_empty()) {
                    let ws = &mut self.ws;
                    v.encode(
                        &self.exec,
                        &images,
                        &mut ws.wide,
                        &mut ws.dq,
                        &mut ws.ffn_g,
                        &mut ws.xn,
                        &mut ws.x,
                    )?;
                }
                self.backbone(&super::ClefPass {
                    ids: &ids,
                    runs: &runs,
                    images: &pass_images,
                })?
            }
            Some(h) => {
                if h.len() != ids.len() * self.cfg.hidden {
                    return Err(bad("hidden rows do not match the requests".into()));
                }
                let d = self.exec.to_device(h)?;
                self.exec.copy_region(&d, 0, &mut self.ws.xn, 0, h.len())?;
            }
        }

        let e = self.exec.clone();
        let up = |v: &[u32]| e.to_device_u32(v);
        let (d_qspans, d_ospans, d_lex_ids, d_lex_spans) =
            (up(&qspans)?, up(&ospans)?, up(&lex_ids)?, up(&lex_spans)?);
        let (d_glob, d_qof, d_rof, d_qtypes, d_qopts) = (
            up(&glob_spans)?,
            up(&qof)?,
            up(&rof)?,
            up(&qtypes)?,
            up(&qopts)?,
        );
        let (d_opt_mem, d_q_mem, d_q_self) = (up(&opt_mem)?, up(&q_mem)?, up(&q_self)?);

        let h = &self.head;
        let bw = &mut self.ws;
        let hw = &mut self.head_ws;
        let t = ids.len();
        let (dd, w) = (self.cfg.hidden, h.cfg.width);
        let heads = h.cfg.heads;
        let scale = 1.0 / ((w / heads) as f32).sqrt();
        let max_opts = reqs
            .iter()
            .flat_map(|r| r.questions.iter())
            .map(|q| q.options.len())
            .max()
            .unwrap_or(1);

        // normalized hidden rows -> bw.x; memory -> bw.k
        e.clef_norm(
            &bw.xn,
            &h.hidden_norm.w,
            &h.hidden_norm.b,
            &mut bw.x,
            dd,
            t,
            LN_EPS,
        )?;
        gemm(&e, (&bw.x, 0), &h.memory, &mut bw.k, t, DiarEpi::Store)?;
        e.clef_span_mean(&bw.x, dd, dd, &d_qspans, nq, &mut hw.qvec, t)?;
        e.clef_span_mean(&bw.x, dd, dd, &d_glob, nr, &mut hw.glob, t)?;
        e.clef_span_mean(&bw.x, dd, dd, &d_ospans, no, &mut hw.ctx, t)?;
        // the output embedding as stored: BF16 rows (726) or Q8_0 (742)
        if self.lm_head.ty == paddock_models::ggml_type::GgmlType::Q8_0 {
            e.clef_lex_mean_q8(&self.lm_head, &d_lex_ids, &d_lex_spans, no, &mut hw.lex)?;
        } else {
            e.clef_lex_mean(&self.lm_head, &d_lex_ids, &d_lex_spans, no, &mut hw.lex)?;
        }

        // option queries: context + lexical + their question's projection
        gemm(
            &e,
            (&hw.ctx, 0),
            &h.option_context,
            &mut hw.oq,
            no,
            DiarEpi::Store,
        )?;
        gemm(
            &e,
            (&hw.lex, 0),
            &h.option_lexical,
            &mut hw.oq,
            no,
            DiarEpi::Resid,
        )?;
        gemm(
            &e,
            (&hw.qvec, 0),
            &h.option_question,
            &mut hw.qo,
            nq,
            DiarEpi::Store,
        )?;
        e.clef_gather_add(&mut hw.oq, &hw.qo, &d_qof, no, w)?;

        // evidence routing: options attend their request's normed memory
        for ev in &h.evidence {
            e.clef_norm(
                &hw.oq,
                &ev.query_norm.w,
                &ev.query_norm.b,
                &mut hw.tq,
                w,
                no,
                LN_EPS,
            )?;
            e.clef_norm(
                &bw.k,
                &ev.memory_norm.w,
                &ev.memory_norm.b,
                &mut bw.kn,
                w,
                t,
                LN_EPS,
            )?;
            gemm(&e, (&hw.tq, 0), &ev.attn.q, &mut hw.att, no, DiarEpi::Store)?;
            gemm(
                &e,
                (&bw.kn, 0),
                &ev.attn.kv,
                &mut bw.wide,
                t,
                DiarEpi::Store,
            )?;
            e.clef_attention(
                (&hw.att, 0, w),
                (&bw.wide, 0, 2 * w),
                (&bw.wide, w, 2 * w),
                &mut hw.tq,
                w,
                &d_opt_mem,
                no,
                heads,
                scale,
            )?;
            gemm(&e, (&hw.tq, 0), &ev.attn.o, &mut hw.oq, no, DiarEpi::Resid)?;
            e.clef_norm(
                &hw.oq,
                &ev.ff_norm.w,
                &ev.ff_norm.b,
                &mut hw.tq,
                w,
                no,
                LN_EPS,
            )?;
            gemm(&e, (&hw.tq, 0), &ev.ff1, &mut hw.wide, no, DiarEpi::Gelu)?;
            gemm(&e, (&hw.wide, 0), &ev.ff2, &mut hw.oq, no, DiarEpi::Resid)?;
        }

        // fields: question projection + its routed option summary (normed) +
        // the request's global projection + the type embedding
        gemm(
            &e,
            (&hw.qvec, 0),
            &h.question,
            &mut hw.fields,
            nq,
            DiarEpi::Store,
        )?;
        e.clef_route(&hw.oq, &hw.fields, &d_qopts, nq, w, max_opts, &mut hw.summ)?;
        e.clef_norm(
            &hw.summ,
            &h.option_summary_norm.w,
            &h.option_summary_norm.b,
            &mut hw.tq,
            w,
            nq,
            LN_EPS,
        )?;
        e.add(&mut hw.fields, &hw.tq, nq * w)?;
        gemm(
            &e,
            (&hw.glob, 0),
            &h.global,
            &mut hw.gproj,
            nr,
            DiarEpi::Store,
        )?;
        e.clef_gather_add(&mut hw.fields, &hw.gproj, &d_rof, nq, w)?;
        e.clef_gather_add(&mut hw.fields, &h.type_emb, &d_qtypes, nq, w)?;

        // the decoder layers over the fields
        for l in &h.layers {
            e.clef_norm(
                &hw.fields, &l.norm1.w, &l.norm1.b, &mut hw.tq, w, nq, LN_EPS,
            )?;
            gemm(&e, (&hw.tq, 0), &l.sa_qkv, &mut hw.qkv, nq, DiarEpi::Store)?;
            e.clef_attention(
                (&hw.qkv, 0, 3 * w),
                (&hw.qkv, w, 3 * w),
                (&hw.qkv, 2 * w, 3 * w),
                &mut hw.att,
                w,
                &d_q_self,
                nq,
                heads,
                scale,
            )?;
            gemm(
                &e,
                (&hw.att, 0),
                &l.sa_o,
                &mut hw.fields,
                nq,
                DiarEpi::Resid,
            )?;
            e.clef_norm(
                &hw.fields, &l.norm2.w, &l.norm2.b, &mut hw.tq, w, nq, LN_EPS,
            )?;
            gemm(&e, (&hw.tq, 0), &l.ca.q, &mut hw.att, nq, DiarEpi::Store)?;
            gemm(&e, (&bw.k, 0), &l.ca.kv, &mut bw.wide, t, DiarEpi::Store)?;
            e.clef_attention(
                (&hw.att, 0, w),
                (&bw.wide, 0, 2 * w),
                (&bw.wide, w, 2 * w),
                &mut hw.tq,
                w,
                &d_q_mem,
                nq,
                heads,
                scale,
            )?;
            gemm(&e, (&hw.tq, 0), &l.ca.o, &mut hw.fields, nq, DiarEpi::Resid)?;
            e.clef_norm(
                &hw.fields, &l.norm3.w, &l.norm3.b, &mut hw.tq, w, nq, LN_EPS,
            )?;
            gemm(&e, (&hw.tq, 0), &l.ff1, &mut hw.wide, nq, DiarEpi::Gelu)?;
            gemm(
                &e,
                (&hw.wide, 0),
                &l.ff2,
                &mut hw.fields,
                nq,
                DiarEpi::Resid,
            )?;
        }

        // scoring
        e.clef_norm(
            &hw.fields,
            &h.field_norm.w,
            &h.field_norm.b,
            &mut hw.summ,
            w,
            nq,
            LN_EPS,
        )?;
        e.clef_norm(
            &hw.oq,
            &h.option_norm.w,
            &h.option_norm.b,
            &mut hw.onorm,
            w,
            no,
            LN_EPS,
        )?;
        e.clef_features(&hw.summ, &hw.onorm, &d_qof, no, w, &mut hw.wide)?;
        gemm(
            &e,
            (&hw.wide, 0),
            &h.scorer0,
            &mut hw.hid,
            no,
            DiarEpi::Gelu,
        )?;
        e.clef_score(
            &ClefScoreIn {
                lex: &hw.lex,
                qvec: &hw.qvec,
                glob: &hw.glob,
                qof: &d_qof,
                rof: &d_rof,
                fields: &hw.summ,
                opts: &hw.onorm,
                hid: &hw.hid,
                w3: &h.scorer3_w,
                b3: h.scorer3_b,
                dd,
                w,
                prior_scale: h.prior_scale,
                joint_scale: h.joint_scale,
                gate: h.gate,
            },
            no,
            &mut hw.logits,
        )?;
        let flat = e.to_host_len(&hw.logits, no)?;
        let mut out = Vec::with_capacity(nr);
        let mut k = 0usize;
        for r in reqs {
            let mut per_q = Vec::with_capacity(r.questions.len());
            for q in r.questions {
                per_q.push(flat[k..k + q.options.len()].to_vec());
                k += q.options.len();
            }
            out.push(per_q);
        }
        Ok(out)
    }
}
