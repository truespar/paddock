//! One group's read: the prompt the canvas answers, the answer template and
//! its slots, the seeded canvas, and the reads - one, or several in parallel
//! when the first is unsettled - folded into one answer per question.

use axum::http::StatusCode;
use axum::response::Response;
use futures::future::join_all;
use serde_json::{Value, json};
use tokio::sync::mpsc::unbounded_channel;

use paddock_engine::generator::CanvasReadOut;
use paddock_engine::sampler::SamplingParams;
use paddock_engine::service::{CanvasReadRequest, GenRequest, MmChunk, TokenEvent};

use super::schema::{Format, Kind, Question, answer_text};
use super::{err, invalid};
use crate::serving::ServingModel;

/// The canvas grows in steps of this many positions.
pub const CANVAS_STEP: usize = 16;
/// `<turn|>` closes the model's answer inside the canvas.
const TURN_CLOSE: u32 = 106;
/// `<pad>` fills the canvas past the template.
const PAD: u32 = 0;

pub type Fail = Box<Response>;

/// What the canvas is read against: the prompt's text ids (what the engine
/// is handed as `prompt`), plus its interleaved chunks when it carries
/// images. `extend` appends to both, so a continuation (earlier answer
/// lines, a written thought) lands after the images as well.
#[derive(Clone)]
pub struct Prompt {
    pub ids: Vec<u32>,
    pub mm: Option<Vec<MmChunk>>,
}

impl Prompt {
    pub fn extend(&mut self, more: &[u32]) {
        self.ids.extend_from_slice(more);
        if let Some(chunks) = self.mm.as_mut() {
            match chunks.last_mut() {
                Some(MmChunk::Text(t)) => t.extend_from_slice(more),
                _ => chunks.push(MmChunk::Text(more.to_vec())),
            }
        }
    }
}

fn encode(model: &ServingModel, text: &str) -> Result<Vec<u32>, String> {
    model.tokenizer.encode(text).map_err(|e| e.to_string())
}

/// The prompt: system + user rendered through the model's own template,
/// thinking on or off, tokenized, BOS-led the way the chat path does it.
/// Images go ahead of the state in the user turn and are split in at the
/// template's image slots exactly as a chat request's are.
pub fn render(
    model: &ServingModel,
    sys: &str,
    state: &str,
    images: &[MmChunk],
    thinking: bool,
) -> Result<Prompt, String> {
    let template = model
        .chat_template
        .as_deref()
        .ok_or("this model has no chat template")?;
    let user = if images.is_empty() {
        json!(state)
    } else {
        let mut parts: Vec<Value> = images
            .iter()
            .map(|_| json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}))
            .collect();
        parts.push(json!({"type": "text", "text": state}));
        Value::Array(parts)
    };
    let messages = crate::chat_template::normalize_messages(&[
        json!({"role": "system", "content": sys}),
        json!({"role": "user", "content": user}),
    ]);
    let kwargs = json!({"enable_thinking": thinking});
    let text = crate::chat_template::render_with_specials(
        template,
        &messages,
        None,
        Some(&kwargs),
        &model.template_specials(),
    )?;
    let mut ids = encode(model, &text)?;
    if let Some(bos) = model.bos
        && ids.first() != Some(&bos)
    {
        ids.insert(0, bos);
    }
    if images.is_empty() {
        return Ok(Prompt { ids, mm: None });
    }
    let pad = model
        .image_pad_id
        .ok_or("this model has no image slot token")?;
    let mm = crate::chat::build_mm_chunks(&ids, pad, images.to_vec(), None)?;
    Ok(Prompt {
        ids: mm.text_ids,
        mm: Some(mm.chunks),
    })
}

/// The thought channel's tags: the empty block the model opens every answer
/// with when thinking is off (seeded ahead of the template - measured, the
/// slot otherwise wants a newline), and the two halves a written thought
/// sits between.
pub struct Channel {
    pub open: Vec<u32>,
    pub close: Vec<u32>,
}

impl Channel {
    pub fn of(model: &ServingModel) -> Result<Self, String> {
        Ok(Self {
            open: encode(model, "<|channel>thought\n")?,
            close: encode(model, "<channel|>")?,
        })
    }

    pub fn scaffold(&self) -> Vec<u32> {
        let mut s = self.open.clone();
        s.extend_from_slice(&self.close);
        s
    }
}

/// Where a question's label sits in the tokenized template, and per label
/// every id that answers with it - the slot's own token first, then the same
/// label with its leading space toggled, then the option's own NAME - each
/// as written, capitalized and lower-cased, bare and after a space - wherever
/// that is one token. The model is free
/// to put its mass on any of them: asked "bug or question?" with the options
/// lettered A and B, it writes " bug" at the slot on nearly every read, and
/// counting the letters alone left 99.9% of the answer in `outside` while
/// the letters' leftovers still named the right option.
pub struct Slot {
    pub pos: usize,
    pub spellings: Vec<Vec<u32>>,
}

/// Tokenize the answer template and find each question's slot. The example
/// server's rule: every label must tokenize the template to the same length
/// as the base and differ from it at exactly one position, the same position
/// for all of a question's labels. `lead` is the text ahead of the first
/// answer - the join when earlier answers are already in the prompt, so the
/// ids match one joint template.
pub fn resolve(
    model: &ServingModel,
    qs: &[&Question],
    lead: &str,
    fmt: Format,
) -> Result<(Vec<u32>, Vec<Slot>), String> {
    let text = |picks: &[usize]| format!("{lead}{}", answer_text(qs, picks, fmt));
    let base_pick = vec![0usize; qs.len()];
    let base = encode(model, &text(&base_pick))?;
    let mut slots = Vec::with_capacity(qs.len());
    for (qi, q) in qs.iter().enumerate() {
        let mut pos: Option<usize> = None;
        let mut ids = vec![0u32; q.labels.len()];
        for (li, slot_id) in ids.iter_mut().enumerate().skip(1) {
            let mut pick = base_pick.clone();
            pick[qi] = li;
            let e = encode(model, &text(&pick))?;
            if e.len() != base.len() {
                return Err(format!(
                    "question {:?}: label {:?} is not a single token in the answer template",
                    q.id, q.labels[li]
                ));
            }
            let diff: Vec<usize> = (0..e.len()).filter(|&i| e[i] != base[i]).collect();
            match (diff.as_slice(), pos) {
                ([p], None) => pos = Some(*p),
                ([p], Some(known)) if *p == known => {}
                _ => {
                    return Err(format!(
                        "question {:?}: its labels do not share one template position",
                        q.id
                    ));
                }
            }
            *slot_id = e[pos.expect("set")];
        }
        let Some(p) = pos else {
            return Err(format!(
                "question {:?}: its labels tokenize identically",
                q.id
            ));
        };
        ids[0] = base[p];
        let mut uniq = ids.clone();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() != ids.len() {
            return Err(format!(
                "question {:?}: two labels tokenize to the same id",
                q.id
            ));
        }
        let spellings = spellings(model, q, &ids)?;
        slots.push(Slot { pos: p, spellings });
    }
    Ok((base, slots))
}

/// Each label's ids: its slot id, plus the label and the option's name - as
/// written, capitalized and lower-cased, each alone and after a space -
/// where that is a single token. An id two
/// labels of one question would share is dropped from both (it cannot tell
/// them apart) - except a label's own slot id, which is unique by
/// construction.
fn capitalized(w: &str) -> String {
    let mut c = w.chars();
    c.next()
        .map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
}

fn spellings(
    model: &ServingModel,
    q: &Question,
    slot_ids: &[u32],
) -> Result<Vec<Vec<u32>>, String> {
    let mut all: Vec<Vec<u32>> = Vec::with_capacity(q.labels.len());
    for ((label, name), &id) in q.labels.iter().zip(&q.names).zip(slot_ids) {
        let mut ids = vec![id];
        // the label and the name as written, capitalized and lower-cased
        // ("yes" / " Yes", "bug" / " Bug"), each bare and after a space
        let mut words = vec![label.clone(), name.clone()];
        for w in [label, name] {
            for form in [capitalized(w), w.to_lowercase()] {
                if !words.contains(&form) {
                    words.push(form);
                }
            }
        }
        for w in &words {
            for text in [w.clone(), format!(" {w}")] {
                if let [one] = encode(model, &text)?.as_slice()
                    && !ids.contains(one)
                {
                    ids.push(*one);
                }
            }
        }
        all.push(ids);
    }
    let owners = |id: u32, all: &[Vec<u32>]| all.iter().filter(|v| v.contains(&id)).count();
    let snapshot = all.clone();
    for (li, ids) in all.iter_mut().enumerate() {
        ids.retain(|&id| id == slot_ids[li] || owners(id, &snapshot) == 1);
    }
    Ok(all)
}

/// Everything one group needs to be read, resolved once and reused by every
/// noise draw.
pub struct Plan {
    pub prompt: Prompt,
    /// the canvas' first ids: the empty thought block, or nothing when the
    /// prompt already ends the thought channel
    pub head: Vec<u32>,
    pub template: Vec<u32>,
    pub slots: Vec<Slot>,
    /// every label id of every slot, each once - what the engine gathers
    pub union: Vec<u32>,
    pub width: usize,
}

impl Plan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &ServingModel,
        prompt: Prompt,
        head: Vec<u32>,
        qs: &[&Question],
        lead: &str,
        fmt: Format,
        width_cap: usize,
        max_ctx: usize,
    ) -> Result<Self, String> {
        let (template, slots) = resolve(model, qs, lead, fmt)?;
        let need = head.len() + template.len() + 1;
        let width = need.next_multiple_of(CANVAS_STEP);
        if width > width_cap {
            return Err(format!(
                "the answer template needs {need} canvas positions; this model reads up to \
                 {width_cap} - mark questions `alone`, set `chunk_rows`, or ask fewer per call"
            ));
        }
        if prompt.ids.len() + width > max_ctx {
            return Err(format!(
                "the prompt is {} tokens and the canvas {width}; the window is {max_ctx}",
                prompt.ids.len()
            ));
        }
        let mut union: Vec<u32> = Vec::new();
        for s in &slots {
            for ids in &s.spellings {
                for &id in ids {
                    if !union.contains(&id) {
                        union.push(id);
                    }
                }
            }
        }
        Ok(Self {
            prompt,
            head,
            template,
            slots,
            union,
            width,
        })
    }

    /// The canvas: template behind the head, `<turn|>`, pad, and uniform
    /// noise at the answer slots from a splitmix64 stream on the seed.
    fn canvas(&self, vocab: u64, seed: u64) -> Vec<u32> {
        let mut canvas = self.head.clone();
        canvas.extend_from_slice(&self.template);
        canvas.push(TURN_CLOSE);
        canvas.resize(self.width, PAD);
        let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
        for s in &self.slots {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            canvas[self.head.len() + s.pos] = (z % vocab.max(1)) as u32;
        }
        canvas
    }

    /// Every canvas position but the answer slots: what a multi-step read
    /// holds between its steps.
    fn pinned(&self) -> Vec<u32> {
        (0..self.width as u32)
            .filter(|&p| {
                !self
                    .slots
                    .iter()
                    .any(|s| self.head.len() + s.pos == p as usize)
            })
            .collect()
    }
}

/// One read's outcome at one slot.
struct SlotRead {
    prompt_tokens: usize,
    /// normalized over the question's labels, each label's spellings summed
    probs: Vec<f32>,
    /// the raw label mass before normalizing (1 - this = outside the labels)
    mass: f32,
    /// the ANSWER's entropy: over the labels, each with its spellings
    /// summed, plus everything outside them as one more outcome. This is
    /// what says whether the reads can disagree - a slot splitting its mass
    /// between " blue" and "Blue" is settled on blue.
    entropy: f32,
    /// the slot's entropy over the whole vocabulary, as the engine measured
    /// it - high for a settled answer with several spellings in play
    slot_entropy: f32,
    /// the slot's argmax over the whole vocabulary - where the mass
    /// outside the labels went, when it is not a label
    argmax: u32,
}

/// One read: seed the canvas, submit, collect.
async fn read_once(
    model: &ServingModel,
    plan: &Plan,
    steps: u32,
    seed: u64,
) -> Result<Vec<SlotRead>, Fail> {
    let vocab = model.tokenizer.vocab_size() as u64;
    let canvas = plan.canvas(vocab, seed);
    let (reply_tx, reply_rx) = std::sync::mpsc::channel::<CanvasReadOut>();
    let (tx, mut rx) = unbounded_channel();
    let req = GenRequest {
        prompt: plan.prompt.ids.clone(),
        max_tokens: 1,
        // the seed keys a multi-step read's intermediate re-noise, so every
        // draw of a decision denoises differently
        sampler: SamplingParams {
            seed,
            ..Default::default()
        },
        stop_tokens: Vec::new(),
        events: tx,
        mm_chunks: plan.prompt.mm.clone(),
        constraint: None,
        logprobs: None,
        submitted: None,
        canvas_read: Some(CanvasReadRequest {
            canvas,
            label_ids: plan.union.clone(),
            steps,
            pinned: if steps > 1 { plan.pinned() } else { Vec::new() },
            reply: reply_tx,
        }),
        user_turn: false,
    };
    if let Err(e) = model.engine.submit(req) {
        return Err(Box::new(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            e,
        )));
    }
    let mut prompt_tokens = plan.prompt.ids.len();
    loop {
        match rx.recv().await {
            Some(TokenEvent::Prefilled { rows, .. }) => prompt_tokens = rows as usize,
            Some(TokenEvent::Done(..)) => break,
            Some(TokenEvent::Error(e)) => return Err(Box::new(crate::chat::engine_err(&e))),
            Some(_) => {}
            None => {
                return Err(Box::new(err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "the engine closed the read without answering",
                )));
            }
        }
    }
    let Ok(out) = reply_rx.try_recv() else {
        return Err(Box::new(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "the engine finished the read without a distribution",
        )));
    };
    let k = plan.union.len();
    Ok(plan
        .slots
        .iter()
        .map(|s| {
            let pos = plan.head.len() + s.pos;
            let raw: Vec<f32> = s
                .spellings
                .iter()
                .map(|ids| {
                    ids.iter()
                        .map(|id| {
                            let j = plan.union.iter().position(|u| u == id).expect("in union");
                            out.probs.get(pos * k + j).copied().unwrap_or(0.0)
                        })
                        .sum()
                })
                .collect();
            let mass: f32 = raw.iter().sum();
            let probs = if mass > 0.0 {
                raw.iter().map(|p| p / mass).collect()
            } else {
                vec![1.0 / raw.len() as f32; raw.len()]
            };
            SlotRead {
                prompt_tokens,
                entropy: answer_entropy(&raw),
                probs,
                mass,
                slot_entropy: out.entropy.get(pos).copied().unwrap_or(f32::NAN),
                argmax: out.argmax.get(pos).copied().unwrap_or(0),
            }
        })
        .collect())
}

/// The entropy of an answer, in nats: each label's raw mass (its spellings
/// summed) as one outcome and the rest of the vocabulary as one more. The
/// outside bucket keeps a slot that is not answering from reading as
/// settled; lumping it is the conservative side, since the mass there is
/// reported on its own as `outside`.
fn answer_entropy(raw: &[f32]) -> f32 {
    let term = |p: f32| if p > 0.0 { -p * p.ln() } else { 0.0 };
    let mass: f32 = raw.iter().sum();
    raw.iter().map(|&p| term(p)).sum::<f32>() + term((1.0 - mass).max(0.0))
}

/// How many reads, and when to take more.
#[derive(Clone, Copy)]
pub enum Samples {
    /// read once; when any answer's entropy (`answer_entropy`) is above
    /// `threshold`, read `max - 1` more with fresh noise, all at once
    Auto { max: usize, threshold: f32 },
    /// read exactly this many, all at once
    Fixed(usize),
}

/// A group's answers, one per question, and how they were read.
pub struct GroupOut {
    pub prompt_tokens: usize,
    pub folded: Vec<Folded>,
    pub reads: usize,
    /// the first read's per-question entropy, and whether it called for more
    pub first_entropy: Vec<f32>,
    pub extended: bool,
    pub width: usize,
}

/// Read a group: `samples` decides how many draws; every draw after the
/// first of an auto read, and every draw of a fixed one, is submitted at
/// once so the engine batches them into the same ticks.
pub async fn read_group(
    model: &ServingModel,
    plan: &Plan,
    qs: &[&Question],
    samples: Samples,
    steps: u32,
    seed: u64,
) -> Result<GroupOut, Fail> {
    let draw = |k: usize| read_once(model, plan, steps, seed.wrapping_add(k as u64 * 7919));
    let (reads, first_entropy, extended) = match samples {
        Samples::Fixed(n) => {
            let reads = collect(join_all((0..n).map(draw)).await)?;
            let first = reads[0].iter().map(|r| r.entropy).collect();
            (reads, first, false)
        }
        Samples::Auto { max, threshold } => {
            let first = draw(0).await?;
            let first_entropy: Vec<f32> = first.iter().map(|r| r.entropy).collect();
            let extended = max > 1
                && first
                    .iter()
                    .any(|r| r.entropy.is_nan() || r.entropy > threshold);
            let mut reads = vec![first];
            if extended {
                reads.extend(collect(join_all((1..max).map(draw)).await)?);
            }
            (reads, first_entropy, extended)
        }
    };
    let folded = qs
        .iter()
        .enumerate()
        .map(|(qi, q)| {
            fold(
                q,
                &reads.iter().map(|r| &r[qi]).collect::<Vec<_>>(),
                plan.head.len() + plan.slots[qi].pos,
            )
        })
        .collect();
    Ok(GroupOut {
        prompt_tokens: reads
            .iter()
            .flat_map(|r| r.iter())
            .map(|r| r.prompt_tokens)
            .max()
            .unwrap_or(0),
        folded,
        reads: reads.len(),
        first_entropy,
        extended,
        width: plan.width,
    })
}

fn collect(results: Vec<Result<Vec<SlotRead>, Fail>>) -> Result<Vec<Vec<SlotRead>>, Fail> {
    results.into_iter().collect()
}

/// Jev's confidence: how far the winning probability sits above uniform,
/// scaled so that 1/n reads 0 and 1 reads 1. Averaged probabilities sum to
/// one, so the clamp only guards float noise.
pub fn margin(p_top: f32, n: usize) -> f32 {
    if n < 2 {
        return 1.0;
    }
    let n = n as f32;
    ((n * p_top - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

/// The standard error of a mean over `xs` (sample variance, n - 1), None
/// for a single read.
fn stderr(xs: &[f32]) -> Option<f32> {
    let n = xs.len();
    if n < 2 {
        return None;
    }
    let mean = xs.iter().sum::<f32>() / n as f32;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
    Some((var / n as f32).sqrt())
}

/// One question's reads, folded.
pub struct Folded {
    pub mean: Vec<f32>,
    pub top: usize,
    pub confidence: f32,
    pub agreement: f32,
    pub mass: f32,
    /// the answer's entropy, averaged over the reads
    pub entropy: f32,
    /// the slot's whole-vocabulary entropy, averaged over the reads
    pub slot_entropy: f32,
    /// standard error of the winning label's probability over the reads
    pub stderr: Option<f32>,
    /// a score's fractional value and its standard error
    pub score: Option<(f32, Option<f32>)>,
    pub position: usize,
    per_read: Vec<Value>,
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn fold(q: &Question, reads: &[&SlotRead], position: usize) -> Folded {
    let n = q.labels.len();
    let k = reads.len() as f32;
    let mut mean = vec![0f32; n];
    let mut picks = vec![0usize; n];
    let (mut mass, mut entropy, mut slot_entropy) = (0f32, 0f32, 0f32);
    let mut per_read = Vec::with_capacity(reads.len());
    for r in reads {
        for (m, p) in mean.iter_mut().zip(&r.probs) {
            *m += p;
        }
        let top = argmax(&r.probs);
        picks[top] += 1;
        mass += r.mass;
        entropy += r.entropy;
        slot_entropy += r.slot_entropy;
        per_read.push(json!({
            "pick": q.names[top],
            "confidence": margin(r.probs[top], n),
            "entropy": r.entropy,
            "slot_entropy": r.slot_entropy,
            "argmax": r.argmax,
        }));
    }
    for m in mean.iter_mut() {
        *m /= k;
    }
    let top = argmax(&mean);
    let score = (q.kind == Kind::Score).then(|| {
        let value = |p: &[f32]| p.iter().enumerate().map(|(i, p)| i as f32 * p).sum::<f32>();
        let per: Vec<f32> = reads.iter().map(|r| value(&r.probs)).collect();
        (value(&mean), stderr(&per))
    });
    Folded {
        confidence: margin(mean[top], n),
        agreement: picks[top] as f32 / k,
        mass: mass / k,
        entropy: entropy / k,
        slot_entropy: slot_entropy / k,
        stderr: stderr(&reads.iter().map(|r| r.probs[top]).collect::<Vec<_>>()),
        score,
        position,
        per_read,
        mean,
        top,
    }
}

impl Folded {
    /// The answer in Jev's shape, with the runner's own fields beside it.
    pub fn answer(&self, q: &Question) -> Value {
        let mut a = match q.kind {
            Kind::Noul => json!({"type": "noul", "noul": self.mean[0]}),
            Kind::Choice => json!({
                "type": "choice",
                "choice": q.names[self.top],
                "probabilities": q.names.iter().zip(&self.mean).map(|(n, p)| (n.clone(), json!(p))).collect::<serde_json::Map<_, _>>(),
            }),
            Kind::Score => json!({
                "type": "score",
                "score": self.score.map_or(0.0, |s| s.0),
                "level": q.names[self.top],
                "legend": q.names.iter().enumerate().map(|(i, n)| (i.to_string(), json!(n))).collect::<serde_json::Map<_, _>>(),
                "probabilities": self.mean.iter().enumerate().map(|(i, p)| (i.to_string(), json!(p))).collect::<serde_json::Map<_, _>>(),
            }),
        };
        let o = a.as_object_mut().expect("object");
        o.insert("confidence".into(), json!(self.confidence));
        o.insert("agreement".into(), json!(self.agreement));
        o.insert("outside".into(), json!(1.0 - self.mass));
        if let Some(se) = self.stderr {
            o.insert("stderr".into(), json!(se));
        }
        if let Some((_, Some(se))) = self.score {
            o.insert("score_stderr".into(), json!(se));
        }
        a
    }

    /// The name `ask_if` compares against: yes/no, an option, a level.
    pub fn name<'a>(&self, q: &'a Question) -> &'a str {
        &q.names[self.top]
    }

    pub fn diagnostics(&self, q: &Question) -> Value {
        json!({
            "id": q.id,
            "label": q.labels[self.top],
            "position": self.position,
            "entropy": self.entropy,
            "slot_entropy": self.slot_entropy,
            "label_mass": self.mass,
            "reads": self.per_read,
        })
    }
}

/// A thought written before the read: the model's own generation with the
/// thinking marker on, up to `budget` tokens, cut at the channel close. The
/// read then runs with the thought in its prompt, so the answer slots
/// condition on it; every noise draw of the decision shares it.
pub struct Thought {
    pub ids: Vec<u32>,
    pub closed: bool,
    pub ms: f64,
}

pub async fn think(
    model: &ServingModel,
    prompt: &Prompt,
    close: &[u32],
    budget: usize,
    seed: u64,
) -> Result<Thought, Fail> {
    let t0 = std::time::Instant::now();
    let (tx, mut rx) = unbounded_channel();
    let req = GenRequest {
        prompt: prompt.ids.clone(),
        max_tokens: budget,
        // any positive temperature is the model's own schedule on a canvas
        sampler: SamplingParams {
            temperature: 1.0,
            seed,
            ..Default::default()
        },
        stop_tokens: close.to_vec(),
        events: tx,
        mm_chunks: prompt.mm.clone(),
        constraint: None,
        logprobs: None,
        submitted: None,
        canvas_read: None,
        user_turn: false,
    };
    if let Err(e) = model.engine.submit(req) {
        return Err(Box::new(err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            e,
        )));
    }
    let mut ids = Vec::new();
    let mut closed = false;
    loop {
        match rx.recv().await {
            Some(TokenEvent::Token { id, .. }) => {
                if close.contains(&id) {
                    closed = true;
                } else if !closed {
                    ids.push(id);
                }
            }
            Some(TokenEvent::Done(..)) | None => break,
            Some(TokenEvent::Error(e)) => return Err(Box::new(crate::chat::engine_err(&e))),
            Some(_) => {}
        }
    }
    // a stop token ends the run without being emitted on some lanes: a
    // thought that stopped short of its budget was closed by the model
    if !closed && ids.len() < budget {
        closed = true;
    }
    Ok(Thought {
        ids,
        closed,
        ms: t0.elapsed().as_secs_f64() * 1e3,
    })
}

/// The refusal a plan error becomes.
pub fn refuse(e: String) -> Fail {
    Box::new(invalid(e))
}

#[cfg(test)]
mod tests {
    use super::answer_entropy;

    #[test]
    fn answer_entropy_counts_labels_and_one_outside_bucket() {
        // settled on one label, whatever spellings carried it
        assert!(answer_entropy(&[1.0, 0.0]) < 1e-6);
        // a coin flip between two labels
        assert!((answer_entropy(&[0.5, 0.5]) - std::f32::consts::LN_2).abs() < 1e-6);
        // half the mass outside the labels is as unsettled as a coin flip
        assert!((answer_entropy(&[0.5, 0.0]) - std::f32::consts::LN_2).abs() < 1e-6);
    }
}
