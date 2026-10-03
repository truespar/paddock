//! Clef's question dialect, as its reference reads one
//! (`joint_schema_model.py`: `question_options`, `encode_record`,
//! `systemone`, `systemone_answer`):
//!
//! - `noul`: options `true`, `false` in that order, described by the
//!   reference's two default sentences unless `criteria` (an object) replaces
//!   one; any other criteria key is ignored, as the reference ignores it.
//! - `choice`: `criteria` is a non-empty object of option id -> description;
//!   the options are ENCODED sorted by id, and ANSWERED in the caller's order
//!   (the reference's answer walks `question["criteria"]`).
//! - `score`: `criteria` is a non-empty list of level descriptions; option
//!   ids are the indices.
//!
//! `instructions` is optional - the question id stands in when it is absent,
//! null or empty - and any JSON value is rendered the reference's way. A
//! description that is null is left out of the option's text, as the
//! reference leaves out `None`.

use super::super::pyjson::PyVal;

/// Most questions in one request. The reference has no cap but the 16K-token
/// sequence; this keeps the head's per-pass tables bounded.
pub const MAX_QUESTIONS: usize = 256;
/// Most options over all questions of one request.
pub const MAX_OPTIONS: usize = 1024;

const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE: &str = "The proposition is false or the answer is no.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Noul => "noul",
            Kind::Choice => "choice",
            Kind::Score => "score",
        }
    }

    /// The head's type embedding row (`QUESTION_TYPES`).
    pub fn qtype(self) -> u32 {
        match self {
            Kind::Noul => 0,
            Kind::Choice => 1,
            Kind::Score => 2,
        }
    }
}

/// One option as the sequence carries it.
#[derive(Clone, Debug)]
pub struct Opt {
    pub id: String,
    /// `render({"option_id": id, "description": d})`, the description left
    /// out when null
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub kind: Kind,
    /// the instruction as the sequence carries it
    pub instruction: String,
    /// encoding order
    pub options: Vec<Opt>,
    /// answer order: for each option in the caller's order, its index in
    /// `options` (identity but for `choice`)
    pub answer_order: Vec<usize>,
    /// a score question's level descriptions, for the legend
    pub levels: Vec<PyVal>,
}

/// `render(value)`: a string as it is, anything else as sorted compact JSON.
pub fn render(v: &PyVal) -> String {
    match v {
        PyVal::Str(s) => s.clone(),
        other => other.dumps_sorted_compact(),
    }
}

fn option(id: &str, description: Option<&PyVal>) -> Opt {
    let mut kv = vec![("option_id".to_owned(), PyVal::Str(id.to_owned()))];
    if let Some(d) = description.filter(|d| !matches!(d, PyVal::Null)) {
        kv.push(("description".to_owned(), d.clone()));
    }
    Opt {
        id: id.to_owned(),
        text: render(&PyVal::Dict(kv)),
    }
}

fn parse_one(id: &str, q: &PyVal) -> Result<Question, String> {
    let PyVal::Dict(_) = q else {
        return Err(format!("question {id:?}: must be an object"));
    };
    if id.is_empty() {
        return Err("question ids must not be empty".into());
    }
    let kind = match q.get("type") {
        Some(PyVal::Str(t)) if t == "noul" => Kind::Noul,
        Some(PyVal::Str(t)) if t == "choice" => Kind::Choice,
        Some(PyVal::Str(t)) if t == "score" => Kind::Score,
        _ => {
            return Err(format!(
                "question {id:?}: type must be noul, choice, or score"
            ));
        }
    };
    let instruction = match q.get("instructions") {
        None | Some(PyVal::Null) => id.to_owned(),
        Some(PyVal::Str(s)) if s.is_empty() => id.to_owned(),
        Some(v) => render(v),
    };
    let criteria = q.get("criteria");
    let (options, answer_order, levels) = match kind {
        Kind::Noul => {
            let over = match criteria {
                None | Some(PyVal::Null) => None,
                Some(PyVal::Dict(kv)) => Some(kv),
                // the reference's `criteria or {}` takes any empty value
                Some(PyVal::List(l)) if l.is_empty() => None,
                Some(PyVal::Str(s)) if s.is_empty() => None,
                Some(_) => {
                    return Err(format!(
                        "question {id:?}: a noul question's criteria is an object with \
                         optional \"true\" and \"false\" descriptions"
                    ));
                }
            };
            let desc = |key: &str, default: &str| -> PyVal {
                over.and_then(|kv| kv.iter().find(|(k, _)| k == key))
                    .map_or_else(|| PyVal::Str(default.to_owned()), |(_, v)| v.clone())
            };
            let t = desc("true", NOUL_TRUE);
            let f = desc("false", NOUL_FALSE);
            (
                vec![option("true", Some(&t)), option("false", Some(&f))],
                vec![0, 1],
                Vec::new(),
            )
        }
        Kind::Choice => {
            let kv = match criteria {
                Some(PyVal::Dict(kv)) if !kv.is_empty() => kv,
                _ => {
                    return Err(format!(
                        "question {id:?}: criteria must be a non-empty object of option id -> \
                         description"
                    ));
                }
            };
            let mut sorted: Vec<usize> = (0..kv.len()).collect();
            sorted.sort_by(|&a, &b| kv[a].0.cmp(&kv[b].0));
            let options: Vec<Opt> = sorted
                .iter()
                .map(|&i| option(&kv[i].0, Some(&kv[i].1)))
                .collect();
            // caller order -> position in the sorted encoding
            let mut answer_order = vec![0; kv.len()];
            for (pos, &i) in sorted.iter().enumerate() {
                answer_order[i] = pos;
            }
            (options, answer_order, Vec::new())
        }
        Kind::Score => {
            let levels = match criteria {
                Some(PyVal::List(l)) if !l.is_empty() => l.clone(),
                _ => {
                    return Err(format!(
                        "question {id:?}: criteria must be a non-empty list of level \
                         descriptions"
                    ));
                }
            };
            let options = levels
                .iter()
                .enumerate()
                .map(|(i, d)| option(&i.to_string(), Some(d)))
                .collect();
            (options, (0..levels.len()).collect(), levels)
        }
    };
    Ok(Question {
        id: id.to_owned(),
        kind,
        instruction,
        options,
        answer_order,
        levels,
    })
}

/// Every question of a request, in the caller's order.
pub fn parse_all(qs: &PyVal) -> Result<Vec<Question>, String> {
    let kv = match qs {
        PyVal::Dict(kv) if !kv.is_empty() => kv,
        _ => return Err("questions: at least one question is required".into()),
    };
    if kv.len() > MAX_QUESTIONS {
        return Err(format!(
            "questions: {} questions, at most {MAX_QUESTIONS} in one request",
            kv.len()
        ));
    }
    let out: Vec<Question> = kv
        .iter()
        .map(|(id, q)| parse_one(id, q))
        .collect::<Result<_, _>>()?;
    let options: usize = out.iter().map(|q| q.options.len()).sum();
    if options > MAX_OPTIONS {
        return Err(format!(
            "questions: {options} options in all, at most {MAX_OPTIONS} in one request"
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::super::pyjson::parse;
    use super::*;

    #[test]
    fn options_follow_the_reference() {
        let qs = parse(
            r#"{"t": {"type": "choice", "criteria": {"b": null, "a": {"desc": "x", "w": 2}, "c": "Other"}},
                "n": {"type": "noul", "criteria": {"false": "No.", "extra": "ignored"}},
                "s": {"type": "score", "instructions": "", "criteria": ["Low", "High"]},
                "j": {"type": "noul", "instructions": {"ask": "Which?", "hint": [1, 2]}}}"#,
        )
        .unwrap();
        let all = parse_all(&qs).unwrap();
        let t = &all[0];
        assert_eq!(t.instruction, "t");
        let ids: Vec<_> = t.options.iter().map(|o| o.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        assert_eq!(
            t.options[0].text,
            r#"{"description":{"desc":"x","w":2},"option_id":"a"}"#
        );
        assert_eq!(t.options[1].text, r#"{"option_id":"b"}"#);
        // caller order b, a, c -> sorted positions 1, 0, 2
        assert_eq!(t.answer_order, [1, 0, 2]);
        let n = &all[1];
        assert_eq!(
            n.options[0].text,
            format!(r#"{{"description":"{NOUL_TRUE}","option_id":"true"}}"#)
        );
        assert_eq!(
            n.options[1].text,
            r#"{"description":"No.","option_id":"false"}"#
        );
        assert_eq!(all[2].instruction, "s");
        assert_eq!(
            all[2].options[1].text,
            r#"{"description":"High","option_id":"1"}"#
        );
        assert_eq!(all[3].instruction, r#"{"ask":"Which?","hint":[1,2]}"#);
    }

    #[test]
    fn refusals_name_the_question() {
        for (body, want) in [
            (r#"{}"#, "at least one question"),
            (r#"{"q": {"type": "bool"}}"#, "type must be"),
            (
                r#"{"q": {"type": "choice", "criteria": {}}}"#,
                "non-empty object",
            ),
            (
                r#"{"q": {"type": "choice", "criteria": ["a"]}}"#,
                "non-empty object",
            ),
            (
                r#"{"q": {"type": "score", "criteria": {"a": 1}}}"#,
                "non-empty list",
            ),
            (
                r#"{"q": {"type": "noul", "criteria": ["x"]}}"#,
                "optional \"true\"",
            ),
            (r#"{"": {"type": "noul"}}"#, "must not be empty"),
        ] {
            let err = parse_all(&parse(body).unwrap()).unwrap_err();
            assert!(err.contains(want), "{body}: {err}");
        }
    }
}
