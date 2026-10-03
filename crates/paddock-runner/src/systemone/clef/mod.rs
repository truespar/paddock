//! `POST /v1/systemone` on Clef (Cloudflare's decision models): one joint
//! sequence per request - the state, then every question with its options -
//! through a Qwen3.5 backbone, and a joint schema head that scores every
//! option of every question in the same pass. Nothing is generated.
//!
//! The request is the reference's (`joint_schema_model.py` `systemone`):
//! `state` (any JSON value), `questions` (id -> `{type, instructions?,
//! criteria}`), with Paddock's `ask` (answer only these ids) and
//! `max_state_tokens` (read only the state's first tokens - the reference
//! cuts a state that does not fit silently; here a state that does not fit
//! is refused unless the caller asks for the cut). The answers follow the
//! endpoint's contract, the same for every backend: `confidence` is Jev's
//! margin `(n * max - 1) / (n - 1)`, the reference's own confidence (the top
//! probability) is `answer_confidence`, and nothing is rounded (the
//! reference rounds to four places).
//!
//! Images (`images`, or multipart file parts) are the reference's too: each
//! decoded to the reference decoder's RGB (a JPEG as libjpeg-turbo decodes
//! it, alpha dropped) - but turned upright by its EXIF orientation first,
//! like every picture Paddock reads; the reference's decode ignores the tag
//! and reads a phone photo sideways - sized by the processor's
//! `smart_resize` within its
//! pixel bounds - or the caller's, `media_kwargs` `size` /
//! `min_pixels` + `max_pixels`, as the reference reads them - and placed
//! after `STATE:`. The engine resizes and encodes them on the GPU. Videos
//! are not read.

pub mod encode;
mod images;
pub mod question;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Instant;

use serde_json::{Map, Value, json};

use paddock_engine::clef_decision::{ClefDecider, ClefImage, ClefJob, ClefQuestion};

use super::pyjson::{self, PyVal};
use super::read::{Fail, margin, refuse};
use encode::{ClefTok, MAX_LENGTH};
use question::{Kind, Question};

/// Images one decision takes (the canvas backend's bound).
const MAX_IMAGES: usize = 16;
/// The processor's resize factor: patch 16 x merge 2.
const FACTOR: u64 = 32;

/// What the runner needs of the vision tower: the `<|image_pad|>` id and
/// the processor's pixel bounds.
#[derive(Clone, Copy, Debug)]
pub struct ClefVision {
    pub pad: u32,
    pub min_pixels: u64,
    pub max_pixels: u64,
}

/// A loaded Clef checkpoint: its decision thread and its tokenizer.
pub struct ClefModel {
    pub id: String,
    pub decider: ClefDecider,
    tok: Arc<ClefTok>,
    /// Some when the engine loaded the vision tower
    vision: Option<ClefVision>,
}

impl ClefModel {
    pub fn new(id: String, decider: ClefDecider, tok: ClefTok, vision: Option<ClefVision>) -> Self {
        let vision = vision.filter(|_| decider.info().images);
        Self {
            id,
            decider,
            tok: Arc::new(tok),
            vision,
        }
    }

    /// The longest sequence one request may be.
    pub fn max_len(&self) -> usize {
        MAX_LENGTH.min(self.decider.info().max_rows)
    }

    /// What the endpoint takes, for the model card - the canvas backend's
    /// keys, with this backend's values.
    pub fn caps(&self) -> Value {
        json!({
            "backend": "clef",
            "max_questions": question::MAX_QUESTIONS,
            "max_options": question::MAX_OPTIONS,
            "max_samples": 1,
            "max_steps": 1,
            "images": self.vision.is_some(),
            "max_images": if self.vision.is_some() { MAX_IMAGES } else { 0 },
            "conditional": false,
            "think": false,
            "types": ["noul", "choice", "score"],
            "max_tokens": self.max_len(),
        })
    }
}

fn bad(msg: impl Into<String>) -> Fail {
    refuse(msg.into())
}

/// Options this backend cannot honour are refused by name - silently
/// ignoring one would answer a question the caller did not ask.
fn refuse_unsupported(body: &Map<String, Value>, images: usize, vision: bool) -> Result<(), Fail> {
    if images > 0 && !vision {
        return Err(bad(
            "images: this Clef runner reads text only (its vision tower is not loaded - the \
             kernel pack predates Clef's image lane)",
        ));
    }
    if body
        .get("videos")
        .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
    {
        return Err(bad("videos: Clef's video input is not read on this runner"));
    }
    if body
        .get("think")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 0)
    {
        return Err(bad(
            "think: Clef writes no thought - every field is decided in one pass",
        ));
    }
    if body
        .get("steps")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 1)
    {
        return Err(bad("steps: Clef decides in one pass; steps is 1"));
    }
    match body.get("samples") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s == "auto" => {}
        Some(v) if v.as_u64() == Some(1) => {}
        Some(_) => {
            return Err(bad(
                "samples: Clef is deterministic - a second read returns the same answer",
            ));
        }
    }
    if body.get("sequential").and_then(Value::as_bool) == Some(true) {
        return Err(bad(
            "sequential: Clef decides every field jointly, in one sequence",
        ));
    }
    if body.get("instructions").is_some_and(|v| !v.is_null()) {
        return Err(bad(
            "instructions: Clef takes instructions per question, in each question's own \
             `instructions`",
        ));
    }
    if let Some(qs) = body.get("questions").and_then(Value::as_object)
        && let Some((id, k)) = qs.iter().find_map(|(id, q)| {
            ["depends_on", "ask_if", "alone"]
                .into_iter()
                .find(|k| q.get(*k).is_some_and(|v| !v.is_null()))
                .map(|k| (id, k))
        })
    {
        return Err(bad(format!(
            "question {id:?}: {k} - Clef decides every field jointly in one pass; conditional \
             questions are not supported on this model"
        )));
    }
    Ok(())
}

/// The processor's pixel bounds for this request: the checkpoint's, or the
/// caller's `media_kwargs` as the reference applies them - `size` with both
/// `shortest_edge` and `longest_edge`, or `min_pixels` and `max_pixels`
/// together (which win over `size`). A lone `min_pixels` / `max_pixels` is
/// refused: the reference drops it without a word.
fn pixel_bounds(body: &Map<String, Value>, v: &ClefVision) -> Result<(u64, u64), Fail> {
    let empty = Map::new();
    let m = match body.get("media_kwargs") {
        None | Some(Value::Null) => &empty,
        Some(Value::Object(m)) => m,
        Some(_) => return Err(bad("media_kwargs: an object")),
    };
    let int = |k: &str, x: &Value| {
        x.as_u64()
            .filter(|&n| n > 0)
            .ok_or_else(|| bad(format!("media_kwargs.{k}: a positive integer")))
    };
    let (mut lo, mut hi) = (v.min_pixels, v.max_pixels);
    for (k, x) in m {
        match k.as_str() {
            "size" => {
                let o = x.as_object().ok_or_else(|| {
                    bad("media_kwargs.size: an object with shortest_edge and longest_edge")
                })?;
                if let Some(u) = o
                    .keys()
                    .find(|k| *k != "shortest_edge" && *k != "longest_edge")
                {
                    return Err(bad(format!(
                        "media_kwargs.size.{u}: the processor's size is shortest_edge and \
                         longest_edge (pixel counts)"
                    )));
                }
                match (o.get("shortest_edge"), o.get("longest_edge")) {
                    (Some(a), Some(b)) => {
                        lo = int("size.shortest_edge", a)?;
                        hi = int("size.longest_edge", b)?;
                    }
                    _ => {
                        return Err(bad(
                            "media_kwargs.size: needs both shortest_edge and longest_edge",
                        ));
                    }
                }
            }
            "min_pixels" | "max_pixels" => {}
            other => {
                return Err(bad(format!(
                    "media_kwargs.{other}: Clef reads size, or min_pixels with max_pixels"
                )));
            }
        }
    }
    match (m.get("min_pixels"), m.get("max_pixels")) {
        (Some(a), Some(b)) => {
            lo = int("min_pixels", a)?;
            hi = int("max_pixels", b)?;
        }
        (None, None) => {}
        _ => {
            return Err(bad(
                "media_kwargs: min_pixels and max_pixels take effect only together (the \
                 reference ignores one alone); send both, or size",
            ));
        }
    }
    if lo > hi {
        return Err(bad(
            "media_kwargs: the smallest pixel count is above the largest",
        ));
    }
    // A media override cannot make more image rows than the model's entire
    // context. Check before resizing, token expansion or pixel allocation.
    let cap = MAX_LENGTH as u64 * FACTOR * FACTOR;
    if lo == 0 || hi < FACTOR * FACTOR || hi > cap {
        return Err(bad(format!(
            "media_kwargs: pixel bounds must be positive and max_pixels within {}..={cap}",
            FACTOR * FACTOR
        )));
    }
    Ok((lo, hi))
}

fn decision_failure(e: String) -> Fail {
    let busy = e.starts_with("decision queue is full");
    let mut response = super::err(
        if busy {
            axum::http::StatusCode::SERVICE_UNAVAILABLE
        } else {
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        },
        if busy {
            "overloaded_error"
        } else {
            "server_error"
        },
        e,
    );
    if busy {
        response.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            axum::http::HeaderValue::from_static("1"),
        );
    }
    Box::new(response)
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = z.iter().map(|&x| f64::from(x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| (x / s) as f32).collect()
}

/// A level's legend entry as text: a string as it is, anything else as JSON
/// (what the Studio's scale labels show).
fn level_text(v: &PyVal) -> String {
    match v {
        PyVal::Str(s) => s.clone(),
        other => other.dumps(),
    }
}

/// The answer's name: yes / no, the chosen option's id, the level's text.
fn label(q: &Question, probs: &[f32], a: &Value) -> String {
    match q.kind {
        Kind::Noul => if probs[0] >= 0.5 { "yes" } else { "no" }.into(),
        Kind::Choice => a["choice"].as_str().unwrap_or_default().to_owned(),
        Kind::Score => a["level"].as_str().unwrap_or_default().to_owned(),
    }
}

/// One question's answer from its option probabilities (encoding order).
fn answer(q: &Question, probs: &[f32]) -> Value {
    let k = probs.len();
    let (top, pmax) = match q.kind {
        // the reference's choice is the first maximum in the CALLER's order
        Kind::Choice => q.answer_order.iter().map(|&i| (i, probs[i])).fold(
            (q.answer_order[0], f32::NEG_INFINITY),
            |a, b| {
                if b.1 > a.1 { b } else { a }
            },
        ),
        _ => probs
            .iter()
            .copied()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a }),
    };
    let mut a = match q.kind {
        Kind::Noul => json!({"type": "noul", "noul": probs[0]}),
        Kind::Choice => json!({
            "type": "choice",
            "choice": q.options[top].id,
            "probabilities": q.answer_order.iter()
                .map(|&i| (q.options[i].id.clone(), json!(probs[i]))).collect::<Map<_, _>>(),
        }),
        Kind::Score => json!({
            "type": "score",
            "score": probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum::<f32>(),
            "level": level_text(&q.levels[top]),
            "legend": q.levels.iter().enumerate()
                .map(|(i, l)| (i.to_string(), l.to_json())).collect::<Map<_, _>>(),
            "probabilities": probs.iter().enumerate()
                .map(|(i, p)| (i.to_string(), json!(p))).collect::<Map<_, _>>(),
        }),
    };
    let o = a.as_object_mut().expect("object");
    o.insert("confidence".into(), json!(margin(pmax, k)));
    o.insert("answer_confidence".into(), json!(pmax));
    a
}

pub async fn decide(
    model: &ClefModel,
    body: &Map<String, Value>,
    raw: &[u8],
    file_images: Vec<String>,
    t0: Instant,
) -> Result<Value, Fail> {
    // images: multipart file parts first, then the JSON list
    let mut urls = file_images;
    urls.extend(super::json_images(body.get("images")).map_err(bad)?);
    refuse_unsupported(body, urls.len(), model.vision.is_some())?;
    let state = match pyjson::field(raw, "state") {
        None | Some(PyVal::Null) => return Err(bad("state: required")),
        Some(s) => s,
    };
    let Some(qs) = pyjson::field(raw, "questions") else {
        return Err(bad("questions: needs a non-empty map of id -> question"));
    };
    let all = question::parse_all(&qs).map_err(bad)?;
    let asked: Vec<Question> = match body.get("ask") {
        None | Some(Value::Null) => all,
        Some(Value::Array(ids)) if !ids.is_empty() => {
            let ids: Vec<&str> = ids.iter().filter_map(Value::as_str).collect();
            if let Some(u) = ids.iter().find(|id| !all.iter().any(|q| q.id == **id)) {
                return Err(bad(format!("ask: {u:?} is not a question here")));
            }
            all.into_iter()
                .filter(|q| ids.contains(&q.id.as_str()))
                .collect()
        }
        Some(_) => return Err(bad("ask: a non-empty list of question ids")),
    };
    let max_state_tokens = match body.get("max_state_tokens") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .ok_or_else(|| bad("max_state_tokens: a non-negative integer"))?
                as usize,
        ),
    };
    let (mut images, reservation) = match model.vision.as_ref() {
        Some(v) if !urls.is_empty() => {
            images::read_images(&model.decider, urls, pixel_bounds(body, v)?).await?
        }
        _ => (Vec::new(), None),
    };
    let tokens: Vec<usize> = images.iter().map(ClefImage::tokens).collect();
    let (media, at) = match model.vision.as_ref() {
        Some(v) => encode::media_ids(&model.tok, &tokens, v.pad).map_err(bad)?,
        None => (Vec::new(), Vec::new()),
    };
    let text = question::render(&state);
    let enc = encode::encode(
        &model.tok,
        &asked,
        &text,
        (&media, &at),
        model.max_len(),
        max_state_tokens,
    )
    .map_err(bad)?;
    for (im, &row) in images.iter_mut().zip(&enc.image_rows) {
        im.row = row;
    }
    let image_diag: Vec<Value> = images
        .iter()
        .map(|im| {
            json!({
                "width": im.width,
                "height": im.height,
                "resized": [im.resized.1, im.resized.0],
                "tokens": im.tokens(),
            })
        })
        .collect();
    let job = ClefJob {
        ids: enc.ids.clone(),
        questions: asked
            .iter()
            .zip(&enc.spans)
            .map(|(q, s)| ClefQuestion {
                qtype: q.kind.qtype(),
                span: s.question,
                options: s.options.clone(),
            })
            .collect(),
        images,
    };
    let reply = model
        .decider
        .decide_reserved(job, reservation)
        .await
        .map_err(decision_failure)?;

    let mut answers = Map::new();
    let mut diag_q = Vec::with_capacity(asked.len());
    for (q, logits) in asked.iter().zip(&reply.logits) {
        let probs = softmax(logits);
        let a = answer(q, &probs);
        // the answer's entropy over its options, in nats (the endpoint's
        // `entropy` on every backend)
        let entropy: f32 = probs.iter().map(|p| -p * p.clamp(1e-12, 1.0).ln()).sum();
        diag_q.push(json!({
            "id": q.id,
            "label": label(q, &probs, &a),
            "entropy": entropy,
            "options": q.options.len(),
            "option_ids": q.options.iter().map(|o| &o.id).collect::<Vec<_>>(),
            "logits": logits,
        }));
        answers.insert(q.id.clone(), a);
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(json!({
        "model": model.id,
        "answers": answers,
        "usage": {"input_tokens": enc.ids.len(), "output_tokens": 0},
        "diagnostics": {
            "backend": "clef",
            "reads": 1,
            "tokens": enc.ids.len(),
            "state_tokens": enc.state_tokens,
            "state_read": enc.state_kept,
            "images": image_diag.len(),
            "pictures": image_diag,
            "questions": diag_q,
            "timing": {
                "total_ms": total_ms,
                "gpu_ms": reply.gpu_ms,
                "pass_requests": reply.pass_requests,
            },
        },
    }))
}
