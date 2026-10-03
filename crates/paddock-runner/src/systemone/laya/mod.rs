//! `POST /v1/systemone` on a Laya decision model - the same Jev shape the
//! block-diffusion backend serves (`state` + typed `questions` in, one typed
//! probabilistic answer per question out), answered by a model built for it:
//! ModernBERT under a trained head that scores each option at its own
//! `[MASK]`, one bidirectional pass per question, nothing generated.
//!
//! What a request goes through:
//!   1. the router picks the checkpoint - `model` names one, `lang` says
//!      whether the text is English, else the reference's own script and
//!      language detection over the state (`lang.rs`);
//!   2. every question becomes one sequence exactly as the reference builds
//!      it (`sequence.rs`) - or several windows over a state too long for
//!      one, where the reference would silently cut it;
//!   3. the questions of a stage go to the engine as one request, where they
//!      are packed with whatever else is queued into one pass;
//!   4. each question's option logits are divided by the checkpoint's fitted
//!      temperature (clamped as the reference clamps it) and softmaxed.
//!
//! The answer is Laya's, with the fields every `/v1/systemone` backend here
//! agrees on: `confidence` is Jev's `(n * max - 1) / (n - 1)` - the same
//! measure the canvas backend reports, so a client's act / review / human
//! bands read one scale on either - and `answer_confidence` is Laya's
//! calibrated one (max p, what the temperatures were fitted to). Laya's own
//! `confidence` (1 - normalised entropy for choice / score) is in the
//! diagnostics as `entropy_confidence`. `action.act_probability` is the act
//! head's, which the model card itself calls unreliable (AUROC 0.30) - it is
//! reported because it is the model's output, not recommended.
//!
//! Deterministic by construction: no sampling, and the engine is batch
//! invariant, so a second read returns the same bits - `samples` beyond one
//! is refused rather than silently repeated, as are the canvas backend's
//! `steps`, `think` and `images`, which this model has no way to honour.

pub mod lang;
pub use super::pyjson;
pub mod question;
pub mod sequence;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Instant;

use serde_json::{Map, Value, json};

use paddock_engine::decision::{Decider, DecisionRequest, DecisionSeq};
use paddock_models::laya::Checkpoint;

use super::read::{Fail, margin, refuse};
use pyjson::PyVal;
use question::{Kind, Question};
use sequence::{LayaTok, Plan, Prefix};

/// The reference server's cap on a state (`laya/serve.py`), in characters.
pub const MAX_STATE_CHARS: usize = 50_000;

/// A loaded Laya bundle: the engine thread and a tokenizer per checkpoint.
pub struct LayaModel {
    pub id: String,
    pub decider: Decider,
    toks: Vec<(Checkpoint, Arc<LayaTok>)>,
}

impl LayaModel {
    pub fn new(id: String, decider: Decider, toks: Vec<(Checkpoint, Arc<LayaTok>)>) -> Self {
        Self { id, decider, toks }
    }

    pub fn checkpoints(&self) -> Vec<Checkpoint> {
        self.toks.iter().map(|(c, _)| *c).collect()
    }

    fn tok(&self, c: Checkpoint) -> Option<&Arc<LayaTok>> {
        self.toks.iter().find(|(k, _)| *k == c).map(|(_, t)| t)
    }

    /// What the endpoint takes, for the model card - the canvas backend's
    /// keys, with this backend's values.
    pub fn caps(&self) -> Value {
        let info = self.decider.info();
        json!({
            "backend": "laya",
            "max_questions": question::MAX_QUESTIONS,
            "max_samples": 1,
            "max_steps": 1,
            "images": false,
            "conditional": true,
            "think": false,
            "types": ["noul", "choice", "score"],
            "max_options": question::MAX_CHOICE_OPTIONS,
            "max_state_chars": MAX_STATE_CHARS,
            "checkpoints": info.checkpoints.iter().map(|(c, cfg)| json!({
                "name": c.name(),
                "max_len": cfg.max_len,
                "head_max_len": cfg.head_max_len,
                "encoder_layers": cfg.encoder.n_layer,
                "hidden": cfg.encoder.hidden,
            })).collect::<Vec<_>>(),
        })
    }
}

/// `serialize_state`: a string as it is, anything else `json.dumps`ed.
fn serialize(state: &PyVal) -> String {
    match state {
        PyVal::Str(s) => s.clone(),
        other => other.dumps(),
    }
}

fn softmax(z: &[f32], t: f32) -> Vec<f32> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) / t;
    let e: Vec<f32> = z.iter().map(|x| (x / t - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .map_or(0, |(i, _)| i)
}

/// One question's read, after the window choice.
struct Read {
    probs: Vec<f32>,
    logits: Vec<f32>,
    act: Vec<f32>,
    temperature: f32,
    tokens: Vec<usize>,
    /// (chosen, count, token_start, token_end) when read in windows
    window: Option<(usize, usize, usize, usize)>,
    options_cut: Option<usize>,
}

impl Read {
    fn top(&self) -> usize {
        argmax(&self.probs)
    }

    fn name<'a>(&self, q: &'a Question) -> &'a str {
        match q.kind {
            // names are [no, yes]; probs are [false, true]
            Kind::Noul => {
                if self.probs[1] >= 0.5 {
                    "yes"
                } else {
                    "no"
                }
            }
            _ => &q.names[self.top()],
        }
    }

    fn answer(&self, q: &Question) -> Value {
        let k = self.probs.len();
        let top = self.top();
        let pmax = self.probs[top];
        let mut a = match q.kind {
            Kind::Choice => json!({
                "type": "choice",
                "choice": q.labels[top].to_json(),
                "probabilities": q.labels.iter().zip(&self.probs)
                    .map(|(l, p)| (l.py_str(), json!(p))).collect::<Map<_, _>>(),
            }),
            Kind::Score => json!({
                "type": "score",
                "score": self.probs.iter().enumerate().map(|(i, p)| i as f32 * p).sum::<f32>(),
                "level": q.names[top],
                "legend": q.levels.iter().enumerate()
                    .map(|(i, l)| (i.to_string(), l.to_json())).collect::<Map<_, _>>(),
                "probabilities": self.probs.iter().enumerate()
                    .map(|(i, p)| (i.to_string(), json!(p))).collect::<Map<_, _>>(),
            }),
            Kind::Noul => json!({"type": "noul", "noul": self.probs[1]}),
        };
        let o = a.as_object_mut().expect("object");
        o.insert("confidence".into(), json!(margin(pmax, k)));
        o.insert("answer_confidence".into(), json!(pmax));
        o.insert(
            "action".into(),
            json!({"act_probability": self.act.first().copied().unwrap_or(0.0)}),
        );
        a
    }

    fn diagnostics(&self, q: &Question) -> Value {
        let k = self.probs.len();
        // the answer's entropy over its options, in nats - the measure the
        // canvas backend reports as `entropy` (there with the outside mass as
        // one more outcome; this model has no outside)
        let h: f32 = self
            .probs
            .iter()
            .map(|p| -p * p.clamp(1e-12, 1.0).ln())
            .sum();
        let entropy_conf = if k < 2 {
            1.0
        } else {
            (1.0 - h / (k as f32).ln()).clamp(0.0, 1.0)
        };
        json!({
            "id": q.id,
            "label": self.name(q),
            "entropy": h,
            "options": k,
            "tokens": self.tokens,
            "temperature": self.temperature,
            "logits": self.logits,
            "entropy_confidence": entropy_conf,
            "act": self.act,
            "window": self.window.map(|(i, n, s, e)| json!({
                "index": i, "count": n, "token_start": s, "token_end": e,
            })),
            "options_cut_to": self.options_cut,
        })
    }
}

fn bad(msg: impl Into<String>) -> Fail {
    refuse(msg.into())
}

/// Options that belong to the canvas backend and cannot be honoured here
/// are refused by name - silently ignoring one would answer a question the
/// caller did not ask.
fn refuse_canvas_options(body: &Map<String, Value>, file_images: usize) -> Result<(), Fail> {
    let images = file_images
        + body
            .get("images")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
    if images > 0 {
        return Err(bad(
            "images: this model reads text only - Laya has no vision tower",
        ));
    }
    if body
        .get("think")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 0)
    {
        return Err(bad(
            "think: Laya writes no thought - every question is one bidirectional pass",
        ));
    }
    if body
        .get("steps")
        .and_then(Value::as_u64)
        .is_some_and(|n| n > 1)
    {
        return Err(bad("steps: Laya reads in one pass; steps is 1"));
    }
    match body.get("samples") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s == "auto" => {}
        Some(v) if v.as_u64() == Some(1) => {}
        Some(_) => {
            return Err(bad(
                "samples: Laya is deterministic - a second read returns the same answer, so one \
                 read (or \"auto\") is the only choice",
            ));
        }
    }
    if body.get("sequential").and_then(Value::as_bool) == Some(true) {
        return Err(bad(
            "sequential: Laya reads each question on its own; it has no way to read one with \
             another's answer written in",
        ));
    }
    if body.get("instructions").is_some_and(|v| !v.is_null()) {
        return Err(bad(
            "instructions: Laya takes instructions per question, in each question's own \
             `instructions`",
        ));
    }
    Ok(())
}

/// Which checkpoint answers, and why.
fn route(
    model: &LayaModel,
    body: &Map<String, Value>,
    state: &PyVal,
) -> Result<(Checkpoint, Value), Fail> {
    let loaded = model.checkpoints();
    let have = |c: Checkpoint| loaded.contains(&c);
    let missing = |c: Checkpoint, why: &str| {
        bad(format!(
            "{why}, and the {} checkpoint is not installed with this model",
            c.name()
        ))
    };
    if let Some(name) = body.get("model").and_then(Value::as_str)
        && let Some(c) = Checkpoint::parse(name)
    {
        if !have(c) {
            return Err(missing(c, &format!("model {name:?} names it")));
        }
        return Ok((
            c,
            json!({"model": c.name(), "reason": format!("explicit model={name:?}"), "detection": null}),
        ));
    }
    if let Some(code) = body.get("lang").and_then(Value::as_str)
        && let Some(en) = lang::english_from_code(code)
    {
        let c = if en {
            Checkpoint::English
        } else {
            Checkpoint::Multilingual
        };
        if !have(c) {
            return Err(missing(c, &format!("lang {code:?} asks for it")));
        }
        return Ok((
            c,
            json!({"model": c.name(), "reason": format!("explicit lang={code:?}"), "detection": null}),
        ));
    }
    let (multi, reason, det) = lang::route_text(state);
    let c = if multi {
        Checkpoint::Multilingual
    } else {
        Checkpoint::English
    };
    if !have(c) {
        return Err(missing(
            c,
            &format!("this state reads as non-English ({reason})"),
        ));
    }
    Ok((
        c,
        json!({"model": c.name(), "reason": reason, "detection": det.to_json()}),
    ))
}

pub async fn decide(
    model: &LayaModel,
    body: &Map<String, Value>,
    raw: &[u8],
    file_images: usize,
    t0: Instant,
) -> Result<Value, Fail> {
    refuse_canvas_options(body, file_images)?;
    let state = match pyjson::field(raw, "state") {
        None | Some(PyVal::Null) => return Err(bad("state: required")),
        Some(s) => s,
    };
    let text = serialize(&state);
    let chars = text.chars().count();
    if chars > MAX_STATE_CHARS {
        return Err(bad(format!(
            "state too large ({chars} > {MAX_STATE_CHARS} chars)"
        )));
    }
    let Some(qs) = pyjson::field(raw, "questions") else {
        return Err(bad("questions: needs a non-empty map of id -> question"));
    };
    let all = question::parse_all(&qs).map_err(bad)?;
    let asked: Vec<&Question> = match body.get("ask") {
        None | Some(Value::Null) => all.iter().collect(),
        Some(Value::Array(ids)) if !ids.is_empty() => {
            let ids: Vec<&str> = ids.iter().filter_map(Value::as_str).collect();
            if let Some(u) = ids.iter().find(|id| !all.iter().any(|q| q.id == **id)) {
                return Err(bad(format!("ask: {u:?} is not a question here")));
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
        Some(_) => return Err(bad("ask: a non-empty list of question ids")),
    };
    let levels = question::schedule(&asked).map_err(bad)?;
    let (ck, routing) = route(model, body, &state)?;
    let tok = model
        .tok(ck)
        .expect("routed to a loaded checkpoint")
        .clone();
    let cfg = model
        .decider
        .info()
        .config(ck)
        .expect("routed to a loaded checkpoint")
        .clone();
    let state_ids = tok.encode(&text).map_err(bad)?;

    let mut reads: Vec<(&Question, Read)> = Vec::new();
    let mut skipped = Map::new();
    let mut stages: Vec<Value> = Vec::new();
    let (mut tokens, mut passes, mut gpu_ms) = (0usize, 0usize, 0f64);
    for level in &levels {
        let mut stage: Vec<&Question> = Vec::new();
        for q in level {
            let failed = q.ask_if.iter().find_map(|(dep, vals)| {
                let got = reads
                    .iter()
                    .find(|(a, _)| &a.id == dep)
                    .map(|(a, r)| r.name(a));
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

        // every question's sequences (windows included), one engine request
        let mut seqs: Vec<DecisionSeq> = Vec::new();
        let mut spans: Vec<(usize, usize, Plan, Option<usize>)> = Vec::new();
        for q in &stage {
            let p = Prefix::build(&tok, q, cfg.head_max_len).map_err(bad)?;
            let room = p.room(cfg.max_len);
            if p.markers.len() != sequence::option_count(q)
                || p.markers.iter().any(|&m| m as usize >= cfg.max_len)
                || room == 0
            {
                return Err(bad(format!(
                    "question {:?}: its instructions and options fill the {}-token window \
                     (options exceed head_max_len={}); shorten them or split the question",
                    q.id, cfg.max_len, cfg.head_max_len
                )));
            }
            let plan = Plan::for_state(state_ids.len(), room);
            let first = seqs.len();
            for &(s, e) in &plan.windows {
                seqs.push(DecisionSeq {
                    ids: p.with_state(&state_ids[s..e], tok.sep),
                    markers: p.markers.clone(),
                    qtype: q.kind.qtype(),
                });
            }
            spans.push((first, plan.windows.len(), plan, p.options_cut));
        }
        // each sequence's full length (question + options + its state window)
        let seq_lens: Vec<usize> = seqs.iter().map(|s| s.ids.len()).collect();
        let reply = model
            .decider
            .decide(DecisionRequest {
                checkpoint: ck,
                seqs,
            })
            .await
            .map_err(|e| {
                let busy = e.starts_with("decision queue is full");
                Box::new(super::err(
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
                    format!("the decision pass failed: {e}"),
                ))
            })?;
        tokens += reply.tokens;
        passes = passes.max(reply.passes);
        gpu_ms += reply.gpu_ms;

        for (q, (first, n, plan, cut)) in stage.iter().zip(spans) {
            let k = sequence::option_count(q);
            let t = cfg.temperature_for(q.kind.qtype(), k);
            let per: Vec<Vec<f32>> = (first..first + n)
                .map(|i| softmax(&reply.logits[i], t))
                .collect();
            // predict_long's aggregation: evidence anywhere for a noul (the
            // strongest window), the most confident window for the rest
            let pick = (0..n)
                .max_by(|&a, &b| {
                    let key = |i: usize| match q.kind {
                        Kind::Noul => per[i][1],
                        _ => per[i].iter().copied().fold(0f32, f32::max),
                    };
                    key(a).total_cmp(&key(b)).then(b.cmp(&a))
                })
                .unwrap_or(0);
            let (s, e) = plan.windows[pick];
            reads.push((
                q,
                Read {
                    probs: per[pick].clone(),
                    logits: reply.logits[first + pick].clone(),
                    act: reply.act[first + pick].clone(),
                    temperature: t,
                    tokens: seq_lens[first..first + n].to_vec(),
                    window: (n > 1).then_some((pick, n, s, e)),
                    options_cut: cut,
                },
            ));
        }
    }

    let mut answers = Map::new();
    let mut diag_q = Vec::with_capacity(reads.len());
    for q in &asked {
        match reads.iter().find(|(a, _)| a.id == q.id) {
            Some((q, r)) => {
                answers.insert(q.id.clone(), r.answer(q));
                diag_q.push(r.diagnostics(q));
            }
            None => {
                answers.insert(q.id.clone(), Value::Null);
            }
        }
    }
    let windowed = reads.iter().any(|(_, r)| r.window.is_some());
    let total_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(json!({
        "model": model.id,
        "answers": answers,
        "usage": {"input_tokens": tokens, "output_tokens": 0},
        "routing": routing,
        "diagnostics": {
            "backend": "laya",
            "checkpoint": ck.name(),
            "reads": 1,
            "state_tokens": state_ids.len(),
            "windowed": windowed,
            "stages": stages,
            "skipped": skipped,
            "conditioning": if stages.len() > 1 { json!("none") } else { Value::Null },
            "questions": diag_q,
            "timing": {"total_ms": total_ms, "gpu_ms": gpu_ms, "passes": passes},
        },
    }))
}
