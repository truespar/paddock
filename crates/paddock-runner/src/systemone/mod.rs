//! `POST /v1/systemone` - structured decisions on a block-diffusion model
//! (DiffusionGemma), in the Jev shape vLLM's structured-diffusion example
//! server speaks: `state` + a map of `questions` in, a typed probabilistic
//! answer per question out - `noul` (a yes/no probability), `choice` (one
//! option with a distribution over all of them), `score` (an ordered level
//! set with a fractional score).
//!
//! A runner serving a decision model (Laya, Clef) answers the same endpoint
//! with its own reader - see `laya/mod.rs`, `clef/mod.rs`; everything below
//! is the canvas one.
//!
//! How it reads: the questions and their allowed labels go into a system
//! turn, the state (and any images, ahead of it) into the user turn, both
//! rendered through the model's own chat template; the answer template (`id:
//! label` per question) is tokenized and seeded into a canvas behind the
//! empty thought channel, with ONLY the answer slots left as noise; one
//! forward at temperature 1 gives every slot its full-vocab distribution,
//! from which each question's label probabilities are read. Nothing is
//! generated, nothing committed. A choice answers with a letter and a score
//! with a digit, so option and level names of any length work - the names
//! stand beside their labels in the system turn.
//!
//! Beyond one pass, from the same example server's playbook:
//!
//! - `samples`: `"auto"` (default) reads once and, when any answer's entropy
//!   (over its labels, spellings merged, plus the mass outside them) is
//!   above `auto_threshold` (0.1), fires `auto_max - 1` (3) more draws with
//!   fresh slot noise AT ONCE, so the engine batches them; an integer reads
//!   that many, all at once. Answers carry the mean, `agreement`, and the
//!   `stderr` of the winning probability over the draws.
//! - `steps`: up to 8 denoising steps with the template pinned, so the answer
//!   slots condition on each other's partly settled values before the read.
//! - `depends_on` / `ask_if` / `alone`: questions run in stages by their
//!   dependencies; a later stage reads with the earlier answers prefilled
//!   (a text state) or restated in the state (an image one); `ask_if` skips a
//!   question whose condition failed (its answer is null); `alone` gets a
//!   read of its own.
//! - A schema whose template does not fit the canvas runs in chunks, in
//!   parallel (`sequential: true` in order, earlier answers prefilled); past
//!   ten questions the template runs ids into labels to save rows.
//! - `think`: the model writes up to N tokens of thought first, and the read
//!   conditions on it.
//! - `images`: data URLs (or a multipart body with the JSON in `request` and
//!   each image as a file part), on a model loaded with its vision companion.
//!
//! What the answer carries beyond the example server: the mass the model put
//! OUTSIDE the label set at each slot (every spelling of each label counted -
//! the slot's token and the label with its leading space toggled), read
//! from the full normalized plane, which a top-k logprobs read cannot see;
//! and `confidence` in Jev's documented measure, `(n * max - 1) / (n - 1)`
//! over the label distribution (0 at uniform, 1 at certainty), on every
//! type, so a client written against Jev's act / review / human bands reads
//! it the same way.

pub mod clef;
pub mod laya;
pub mod pyjson;
mod read;
mod schema;

use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::future::join_all;
use serde_json::{Map, Value, json};

use paddock_api::ErrorBody;
use paddock_engine::service::MmChunk;

use crate::routes::AppState;
use crate::serving::ServingModel;
use read::{Channel, Fail, GroupOut, Plan, Prompt, Samples, Thought, refuse};
use schema::{Format, Order, Question};

pub use schema::MAX_QUESTIONS;

/// Reads per question, fixed or auto.
pub const MAX_SAMPLES: usize = 32;
/// Images one decision takes.
const MAX_IMAGES: usize = 16;
/// The longest thought a read conditions on.
const MAX_THINK: usize = 4096;
/// A body past this is refused before parsing (images ride inline).
const MAX_BODY: usize = 64 << 20;

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

fn invalid(msg: impl Into<String>) -> Response {
    err(StatusCode::UNPROCESSABLE_ENTITY, "validation_error", msg)
}

/// The body as JSON bytes, plus the images a multipart body carried as file
/// parts (as data URLs, decoded with the inline ones).
async fn read_body(req: Request) -> Result<(Vec<u8>, Vec<String>), Fail> {
    let multipart = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("multipart/form-data"));
    if !multipart {
        let bytes = axum::body::to_bytes(req.into_body(), MAX_BODY)
            .await
            .map_err(|e| Box::new(invalid(format!("the body could not be read: {e}"))))?;
        return Ok((bytes.to_vec(), Vec::new()));
    }
    use base64::Engine as _;
    let mut mp = Multipart::from_request(req, &())
        .await
        .map_err(|e| Box::new(invalid(format!("multipart body: {e}"))))?;
    let mut json_part = None;
    let mut images = Vec::new();
    while let Some(field) = mp
        .next_field()
        .await
        .map_err(|e| Box::new(invalid(format!("multipart body: {e}"))))?
    {
        let name = field.name().unwrap_or_default().to_owned();
        let ct = field.content_type().unwrap_or_default().to_owned();
        let data = field
            .bytes()
            .await
            .map_err(|e| Box::new(invalid(format!("multipart part {name:?}: {e}"))))?;
        if name == "request" {
            json_part = Some(data.to_vec());
        } else {
            let ct = if ct.is_empty() {
                "application/octet-stream"
            } else {
                ct.as_str()
            };
            images.push(format!(
                "data:{ct};base64,{}",
                base64::engine::general_purpose::STANDARD.encode(&data)
            ));
        }
    }
    let Some(json_part) = json_part else {
        return Err(Box::new(invalid(
            "multipart body: the JSON request goes in a part named \"request\"",
        )));
    };
    Ok((json_part, images))
}

/// `images` in the JSON body: data URLs, or objects with a `url`, or with
/// base64 `data` and its `media_type`.
pub(super) fn json_images(v: Option<&Value>) -> Result<Vec<String>, String> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(Vec::new());
    };
    let Some(list) = v.as_array() else {
        return Err("images: a list of data:image/... URLs".into());
    };
    list.iter()
        .enumerate()
        .map(|(i, im)| match im {
            Value::String(s) => Ok(s.clone()),
            Value::Object(o) => {
                if let Some(u) = o.get("url").and_then(Value::as_str) {
                    return Ok(u.to_owned());
                }
                let data = o.get("data").and_then(Value::as_str);
                let mt = o
                    .get("media_type")
                    .or_else(|| o.get("mime_type"))
                    .and_then(Value::as_str);
                match (data, mt) {
                    (Some(d), Some(m)) => Ok(format!("data:{m};base64,{d}")),
                    _ => Err(format!(
                        "images[{i}]: a data:image/... URL, or an object with url, or with data \
                         and media_type"
                    )),
                }
            }
            _ => Err(format!("images[{i}]: a data:image/... URL")),
        })
        .collect()
}

/// Everything a decision's groups share.
struct Ctx<'a> {
    model: &'a ServingModel,
    state: String,
    images: Vec<MmChunk>,
    instructions: Option<String>,
    fmt: Format,
    chained: bool,
    shared_prompt: bool,
    asked: Vec<&'a Question>,
    sys_full: String,
    channel: Channel,
    think: usize,
    samples: Samples,
    steps: u32,
    seed: u64,
    width_cap: usize,
    max_ctx: usize,
    /// a chained text decision's prompt prefix: the full question list, the
    /// state, and the thought channel (the written thought, or the empty
    /// block) - every stage continues it
    base: Option<Prompt>,
}

/// A group's outcome with the thought its read conditioned on, if it wrote
/// one.
struct Ran {
    out: GroupOut,
    thought: Option<Thought>,
    prompt_tokens: usize,
}

impl Ctx<'_> {
    fn text_state(&self) -> bool {
        self.images.is_empty()
    }

    /// One group's read. `lines` are the earlier answers as template text
    /// (prefilled into a text prompt), `earlier` the same as `id: name`
    /// (restated in an image one).
    async fn run(
        &self,
        group: &[&Question],
        k: usize,
        conditioned: bool,
        lines: &[String],
        earlier: &[String],
    ) -> Result<Ran, Fail> {
        let join = self.fmt.join();
        let seed = self.seed.wrapping_add(104_729 * k as u64);
        let mut thought = None;
        let (prompt, head, lead) = if self.text_state() && (conditioned || self.base.is_some()) {
            // a chained text decision: continue the shared prefix, with the
            // earlier answers written after the thought channel
            let mut p = self
                .base
                .clone()
                .expect("chained text decisions carry a base");
            if conditioned {
                let more = self
                    .model
                    .tokenizer
                    .encode(&lines.join(join))
                    .map_err(|e| refuse(e.to_string()))?;
                p.extend(&more);
                (p, Vec::new(), join)
            } else {
                (p, Vec::new(), "")
            }
        } else if conditioned {
            // an image state cannot be continued by token ids through its
            // template, so the earlier answers are restated in the state
            let state = format!("{}\n\nAnswers so far:\n{}", self.state, earlier.join("\n"));
            let p = read::render(self.model, &self.sys_full, &state, &self.images, false)
                .map_err(refuse)?;
            (p, self.channel.scaffold(), "")
        } else {
            let sys = if self.chained || self.shared_prompt {
                if self.shared_prompt && !self.chained {
                    schema::system_text(&self.asked, self.instructions.as_deref(), self.fmt, true)
                } else {
                    self.sys_full.clone()
                }
            } else {
                schema::system_text(group, self.instructions.as_deref(), self.fmt, false)
            };
            if self.think > 0 {
                let mut p = read::render(self.model, &sys, &self.state, &self.images, true)
                    .map_err(refuse)?;
                p.extend(&self.channel.open);
                let t = read::think(self.model, &p, &self.channel.close, self.think, seed).await?;
                p.extend(&t.ids);
                p.extend(&self.channel.close);
                thought = Some(t);
                (p, Vec::new(), "")
            } else {
                let p = read::render(self.model, &sys, &self.state, &self.images, false)
                    .map_err(refuse)?;
                (p, self.channel.scaffold(), "")
            }
        };
        let plan = Plan::new(
            self.model,
            prompt,
            head,
            group,
            lead,
            self.fmt,
            self.width_cap,
            self.max_ctx,
        )
        .map_err(refuse)?;
        let out =
            read::read_group(self.model, &plan, group, self.samples, self.steps, seed).await?;
        Ok(Ran {
            prompt_tokens: out.prompt_tokens,
            out,
            thought,
        })
    }
}

fn thought_json(model: &ServingModel, t: &Thought) -> Value {
    json!({
        "text": model.tokenizer.decode(&t.ids, true).unwrap_or_default(),
        "tokens": t.ids.len(),
        "closed": t.closed,
        "ms": t.ms,
    })
}

pub async fn handle(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let t0 = Instant::now();
    // a decision model answers the same endpoint with its own reader
    if let Some(cm) = state.clef.as_ref() {
        let (raw, file_images) = match read_body(req).await {
            Ok(v) => v,
            Err(r) => return *r,
        };
        let body: Value = match serde_json::from_slice(&raw) {
            Ok(v) => v,
            Err(e) => return invalid(format!("the body is not JSON: {e}")),
        };
        let Some(body) = body.as_object() else {
            return invalid("the body must be a JSON object");
        };
        return match clef::decide(cm, body, &raw, file_images, t0).await {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(r) => *r,
        };
    }
    if let Some(lm) = state.laya.as_ref() {
        let (raw, file_images) = match read_body(req).await {
            Ok(v) => v,
            Err(r) => return *r,
        };
        let body: Value = match serde_json::from_slice(&raw) {
            Ok(v) => v,
            Err(e) => return invalid(format!("the body is not JSON: {e}")),
        };
        let Some(body) = body.as_object() else {
            return invalid("the body must be a JSON object");
        };
        return match laya::decide(lm, body, &raw, file_images.len(), t0).await {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(r) => *r,
        };
    }
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no chat model is loaded",
        );
    };
    let width_cap = model.engine.canvas_width();
    if width_cap == 0 {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "structured decisions need a block-diffusion model (DiffusionGemma); this model \
             generates one token at a time",
        );
    }
    let (raw, file_images) = match read_body(req).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let body: Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(e) => return invalid(format!("the body is not JSON: {e}")),
    };
    let Some(body) = body.as_object() else {
        return invalid("the body must be a JSON object");
    };
    match decide(model, body, &raw, file_images, width_cap, state.max_ctx, t0).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(r) => *r,
    }
}

async fn decide(
    model: &ServingModel,
    body: &Map<String, Value>,
    raw: &[u8],
    file_images: Vec<String>,
    width_cap: usize,
    max_ctx: usize,
    t0: Instant,
) -> Result<Value, Fail> {
    let bad = |m: String| refuse(m);
    let all = schema::parse_questions(body, &Order::read(raw)).map_err(bad)?;
    let state_text = match body.get("state") {
        None | Some(Value::Null) => return Err(bad("state: required".into())),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    };
    let instructions = match body.get("instructions") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(bad("instructions: a string".into())),
    };
    // images: multipart file parts first, then the JSON list
    let mut urls = file_images;
    urls.extend(json_images(body.get("images")).map_err(bad)?);
    if urls.len() > MAX_IMAGES {
        return Err(bad(format!(
            "images: {} given, at most {MAX_IMAGES}",
            urls.len()
        )));
    }
    if !urls.is_empty() && !(model.supports_vision && model.engine.canvas_images()) {
        return Err(bad(
            "images: this model was started without its vision companion (mmproj), or this \
             backend does not take images on a structured read yet"
                .into(),
        ));
    }
    let images: Vec<MmChunk> = urls
        .iter()
        .map(|u| {
            crate::chat::decode_image_url(
                u,
                model.engine.vision_budget(),
                crate::chat::ImageDetail::Auto,
            )
            .map(crate::chat::RequestImage::into_chunk)
        })
        .collect::<Result<_, _>>()
        .map_err(|e| bad(format!("images: {e}")))?;
    let seed = body.get("seed").and_then(Value::as_u64).unwrap_or(42);
    let max_steps = model.engine.canvas_read_steps().max(1);
    let steps = match body.get("steps") {
        None | Some(Value::Null) => 1,
        Some(v) => match v.as_u64() {
            Some(n) if n >= 1 && n <= u64::from(max_steps) => n as u32,
            _ => {
                return Err(bad(if max_steps > 1 {
                    format!("steps: 1 to {max_steps} denoising steps")
                } else {
                    "steps: this backend serves one-step reads only".into()
                }));
            }
        },
    };
    let samples = match body.get("samples") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s == "auto" => None,
        Some(v) => match v.as_u64() {
            Some(n) if (1..=MAX_SAMPLES as u64).contains(&n) => Some(n as usize),
            _ => return Err(bad(format!("samples: \"auto\" or 1..={MAX_SAMPLES}"))),
        },
    };
    let samples = match samples {
        Some(n) => Samples::Fixed(n),
        None => Samples::Auto {
            max: body
                .get("auto_max")
                .and_then(Value::as_u64)
                .map_or(4usize, |n| (n as usize).clamp(1, MAX_SAMPLES)),
            threshold: body
                .get("auto_threshold")
                .and_then(Value::as_f64)
                .unwrap_or(0.1) as f32,
        },
    };
    let think = match body.get("think") {
        None | Some(Value::Null) => 0,
        Some(v) => match v.as_u64() {
            Some(n) if n as usize <= MAX_THINK => n as usize,
            _ => {
                return Err(bad(format!(
                    "think: a thought budget in tokens, 0 to {MAX_THINK}"
                )));
            }
        },
    };
    let sequential = match body.get("sequential") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad("sequential: true or false".into())),
    };
    let shared_prompt = match body.get("chunk_prompt").and_then(Value::as_str) {
        None | Some("own") => false,
        Some("shared") => true,
        Some(_) => return Err(bad("chunk_prompt: \"own\" or \"shared\"".into())),
    };
    let chunk_rows = match body.get("chunk_rows") {
        None | Some(Value::Null) => width_cap,
        Some(v) => match v.as_u64() {
            Some(n) if n >= 8 => (n as usize).min(width_cap),
            _ => return Err(bad("chunk_rows: an integer of at least 8".into())),
        },
    };
    // `ask`: the subset one call answers, with everything it depends on
    let asked: Vec<&Question> = match body.get("ask") {
        None | Some(Value::Null) => all.iter().collect(),
        Some(Value::Array(ids)) if !ids.is_empty() => {
            let ids: Vec<&str> = ids.iter().filter_map(Value::as_str).collect();
            if let Some(unknown) = ids.iter().find(|id| !all.iter().any(|q| q.id == **id)) {
                return Err(bad(format!("ask: {unknown:?} is not a question here")));
            }
            let asked: Vec<&Question> = all
                .iter()
                .filter(|q| ids.contains(&q.id.as_str()))
                .collect();
            for q in &asked {
                if let Some(d) = q.depends_on.iter().find(|d| !ids.contains(&d.as_str())) {
                    return Err(bad(format!(
                        "ask: {:?} is asked but not {d:?}, which it depends on",
                        q.id
                    )));
                }
            }
            asked
        }
        Some(_) => return Err(bad("ask: a non-empty list of question ids".into())),
    };
    let levels = schema::schedule(&asked).map_err(bad)?;
    let fmt = Format::for_count(all.len());
    let text_state = images.is_empty();
    let chained = levels.len() > 1 || sequential;
    let channel = Channel::of(model).map_err(bad)?;
    let sys_full =
        schema::system_text(&asked, instructions.as_deref(), fmt, !text_state && chained);

    // a chained text decision continues ONE prompt: the full question list,
    // the state, then the thought channel - written, or the empty block
    let mut decision_thought = None;
    let base = if chained && text_state {
        let mut p = read::render(model, &sys_full, &state_text, &[], think > 0).map_err(bad)?;
        if think > 0 {
            p.extend(&channel.open);
            let t = read::think(model, &p, &channel.close, think, seed).await?;
            p.extend(&t.ids);
            p.extend(&channel.close);
            decision_thought = Some(t);
        } else {
            p.extend(&channel.scaffold());
        }
        Some(p)
    } else {
        None
    };
    let ctx = Ctx {
        model,
        state: state_text,
        images,
        instructions,
        fmt,
        chained,
        shared_prompt,
        asked: asked.clone(),
        sys_full,
        channel,
        think: if base.is_some() { 0 } else { think },
        samples,
        steps,
        seed,
        width_cap,
        max_ctx,
        base,
    };
    let scaffold_len = ctx.channel.scaffold().len();
    let rows_of = |g: &[&Question]| {
        let picks = vec![0usize; g.len()];
        model
            .tokenizer
            .encode(&schema::answer_text(g, &picks, fmt))
            .map_or(usize::MAX / 2, |ids| scaffold_len + ids.len() + 1)
    };

    let mut answered: Vec<(&Question, read::Folded)> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    let mut earlier: Vec<String> = Vec::new();
    let mut skipped = Map::new();
    let mut stages: Vec<Value> = Vec::new();
    let mut chunks: Vec<Value> = Vec::new();
    let mut sample_diag: Vec<Value> = Vec::new();
    let mut thoughts: Vec<Value> = decision_thought
        .iter()
        .map(|t| thought_json(model, t))
        .collect();
    let mut thought_tokens = decision_thought.as_ref().map_or(0, |t| t.ids.len());
    let (mut width, mut total_reads, mut max_reads, mut prompt_tokens) =
        (0usize, 0usize, 0usize, 0usize);
    let mut k = 0usize;
    for level in &levels {
        // ask_if: a question whose condition failed is not read at all
        let mut stage: Vec<&Question> = Vec::new();
        for q in level {
            let failed = q.ask_if.iter().find_map(|(dep, vals)| {
                let got = answered
                    .iter()
                    .find(|(a, _)| &a.id == dep)
                    .map(|(a, f)| f.name(a));
                (!got.is_some_and(|g| vals.iter().any(|v| v == g))).then_some((dep, got, vals))
            });
            if let Some((dep, got, vals)) = failed {
                skipped.insert(
                    q.id.clone(),
                    json!({"because": dep, "was": got, "wanted": vals}),
                );
                continue;
            }
            stage.push(q);
        }
        if stage.is_empty() {
            continue;
        }
        stages.push(json!(stage.iter().map(|q| &q.id).collect::<Vec<_>>()));
        let groups = schema::chunk_groups(&stage, chunk_rows, rows_of);
        let conditioned = !lines.is_empty();
        let mut ran: Vec<(Vec<&Question>, Ran)> = Vec::with_capacity(groups.len());
        if sequential || groups.len() == 1 {
            for g in groups {
                // a sequential chunk reads with every earlier chunk's answers
                let cond = !lines.is_empty();
                let r = ctx.run(&g, k, cond, &lines, &earlier).await?;
                k += 1;
                // a sequential chunk's successor reads with its answers
                if sequential {
                    lines.push(schema::answer_text(
                        &g,
                        &r.out.folded.iter().map(|f| f.top).collect::<Vec<_>>(),
                        fmt,
                    ));
                    earlier.extend(
                        g.iter()
                            .zip(&r.out.folded)
                            .map(|(q, f)| format!("{}: {}", q.id, f.name(q))),
                    );
                }
                ran.push((g, r));
            }
        } else {
            let base_k = k;
            let results = join_all(
                groups
                    .iter()
                    .enumerate()
                    .map(|(i, g)| ctx.run(g, base_k + i, conditioned, &lines, &earlier)),
            )
            .await;
            k += groups.len();
            for (g, r) in groups.into_iter().zip(results) {
                ran.push((g, r?));
            }
        }
        for (g, r) in ran {
            chunks.push(json!(g.iter().map(|q| &q.id).collect::<Vec<_>>()));
            width = width.max(r.out.width);
            total_reads += r.out.reads;
            max_reads = max_reads.max(r.out.reads);
            prompt_tokens = prompt_tokens.max(r.prompt_tokens);
            if let Some(t) = &r.thought {
                thought_tokens += t.ids.len();
                thoughts.push(thought_json(model, t));
            }
            sample_diag.push(json!({
                "chunk": g.iter().map(|q| &q.id).collect::<Vec<_>>(),
                "reads": r.out.reads,
                "extended": r.out.extended,
                "first_read_entropy": g.iter().zip(&r.out.first_entropy).map(|(q, e)| (q.id.clone(), json!(e))).collect::<Map<_, _>>(),
            }));
            if !sequential {
                lines.push(schema::answer_text(
                    &g,
                    &r.out.folded.iter().map(|f| f.top).collect::<Vec<_>>(),
                    fmt,
                ));
                earlier.extend(
                    g.iter()
                        .zip(&r.out.folded)
                        .map(|(q, f)| format!("{}: {}", q.id, f.name(q))),
                );
            }
            answered.extend(g.into_iter().zip(r.out.folded));
        }
    }

    // the answers in the caller's order; a skipped question answers null
    let mut answers = Map::new();
    let mut diag_q = Vec::with_capacity(answered.len());
    for q in &asked {
        match answered.iter().find(|(a, _)| a.id == q.id) {
            Some((q, f)) => {
                answers.insert(q.id.clone(), f.answer(q));
                let mut d = f.diagnostics(q);
                // each read's argmax as text: a slot that is unsettled while
                // every read agrees has its mass on a token that is not a
                // label, and this says which
                if let Some(reads) = d.get_mut("reads").and_then(Value::as_array_mut) {
                    for r in reads {
                        if let Some(id) = r.get("argmax").and_then(Value::as_u64) {
                            let text = model
                                .tokenizer
                                .decode(&[id as u32], false)
                                .unwrap_or_default();
                            r["argmax"] = json!(text);
                        }
                    }
                }
                diag_q.push(d);
            }
            None => {
                answers.insert(q.id.clone(), Value::Null);
            }
        }
    }
    let conditioning = if stages.len() <= 1 && !sequential {
        Value::Null
    } else if text_state {
        json!("prefill")
    } else {
        json!("restated")
    };
    let total_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(json!({
        "model": model.id,
        "answers": answers,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": thought_tokens},
        "diagnostics": {
            "reads": max_reads,
            "canvas": width,
            "steps": steps,
            "format": match fmt { Format::Lines => "lines", Format::Indexed => "indexed" },
            "images": ctx.images.len(),
            "stages": stages,
            "chunks": chunks,
            "skipped": skipped,
            "conditioning": conditioning,
            "samples": sample_diag,
            "thought": match thoughts.len() { 0 => Value::Null, 1 => thoughts.remove(0), _ => Value::Array(thoughts) },
            "questions": diag_q,
            "timing": {"total_ms": total_ms, "reads": total_reads},
        },
    }))
}
