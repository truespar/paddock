//! Parse Inkling 2 assistant output into content, reasoning and tool calls.
//!
//! The turn is a SEQUENCE of PARTS, each a model message carrying one kind of
//! content (the shipped `chat_template.jinja`):
//!
//! ```text
//! <|message_model|><|content_thinking|>THOUGHT<|end_message|>
//! <|message_model|><|content_text|>ANSWER<|end_message|>
//! <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
//! <|content_model_end_sampling|>
//! ```
//!
//! `<|content_thinking|>` is reasoning; `<|content_text|>`,
//! `<|content_tool_error|>` and `<|content_invoke_tool_text|>` are text the
//! client sees (the tokenizer's own response template routes them to
//! `content`); `<|content_invoke_tool_json|>` is a call whose body is the
//! whole JSON object, arguments already typed. Parallel calls are
//! consecutive parts. `<|content_model_end_sampling|>` is the eos (a stop
//! token - it never reaches the text).
//!
//! The generation prompt ends at the previous message's `<|end_message|>`,
//! so the model opens every part itself, the first included. All markers
//! are single special tokens (`<|message_model|>` 200001, `<|content_text|>`
//! 200004, `<|content_thinking|>` 200008, `<|end_message|>` 200010,
//! `<|content_invoke_tool_json|>` 200049, ...): they decode atomically and
//! cannot split across streaming deltas. The text that can is a part's
//! header (a tool name between `<|message_model|>` and its content marker),
//! which is why nothing is classified until the content marker lands.

use serde_json::Value;

use crate::parsers::ToolHints;
pub use crate::parsers::{Parsed, ToolCallRaw};

pub(crate) const MODEL: &str = "<|message_model|>";
pub(crate) const END: &str = "<|end_message|>";
pub(crate) const EOS: &str = "<|content_model_end_sampling|>";
pub(crate) const THINKING: &str = "<|content_thinking|>";
pub(crate) const TEXT: &str = "<|content_text|>";
pub(crate) const CALL_JSON: &str = "<|content_invoke_tool_json|>";
const CALL_TEXT: &str = "<|content_invoke_tool_text|>";
const TOOL_ERROR: &str = "<|content_tool_error|>";

/// Part kinds by content marker.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Reasoning,
    Content,
    Call,
}

const KINDS: [(&str, Kind); 5] = [
    (THINKING, Kind::Reasoning),
    (TEXT, Kind::Content),
    (TOOL_ERROR, Kind::Content),
    (CALL_TEXT, Kind::Content),
    (CALL_JSON, Kind::Call),
];

/// The longest a part header (a tool name) may run before its content
/// marker. Text past it without one is the model having left the part
/// syntax; it stays visible as content. A fixed byte bound keeps the verdict
/// stable across streaming re-parses (the first HEADER_MAX + 1 bytes after a
/// `<|message_model|>` never change once written).
const HEADER_MAX: usize = 128;

/// Parse one assistant turn. `hints: None` = the request declared no tools ->
/// call parts stay visible text (see `parsers::tool_hints`).
pub fn parse(text: &str, hints: Option<&ToolHints>) -> Parsed {
    let mut out = Parsed::default();
    let mut content = String::new();
    let mut reasoning = String::new();
    let text = text.split(EOS).next().unwrap_or("");
    let mut cur = text;
    // text ahead of the first part: the model skipped the part syntax, and
    // what it wrote is its answer (this dialect never drops model text)
    let lead = cur.find(MODEL).unwrap_or(cur.len());
    push(&mut content, &cur[..lead]);
    cur = &cur[lead..];
    while let Some(rest) = cur.strip_prefix(MODEL) {
        // the header runs to the earliest content marker
        let marker = KINDS
            .iter()
            .filter_map(|(m, k)| rest.find(m).map(|i| (i, *m, *k)))
            .min_by_key(|&(i, ..)| i)
            .filter(|&(i, ..)| i <= HEADER_MAX);
        let Some((h, m, kind)) = marker else {
            // no content marker yet: a header still being typed (nothing to
            // classify) - or, past HEADER_MAX, prose the model wrote outside
            // the syntax, which stays visible
            if rest.len() > HEADER_MAX {
                let end = rest.find(MODEL).unwrap_or(rest.len());
                push(&mut content, rest[..end].trim());
                cur = &rest[end..];
                continue;
            }
            break;
        };
        let name = rest[..h].trim();
        let body_start = &rest[h + m.len()..];
        // the body ends at its `<|end_message|>`; a missing one (still
        // generating, or max_tokens) runs to the end - or to the next part,
        // should the model have opened one without closing this
        let (end, closed) = match (body_start.find(END), body_start.find(MODEL)) {
            (Some(e), Some(n)) if n < e => (n, false),
            (Some(e), _) => (e, true),
            (None, Some(n)) => (n, false),
            (None, None) => (body_start.len(), false),
        };
        let body = &body_start[..end];
        match kind {
            Kind::Reasoning => push(&mut reasoning, body),
            Kind::Content => push(&mut content, body),
            Kind::Call => {
                // a call is only a call once its part closed and its JSON
                // parsed; until then it streams nothing (an unterminated
                // call is not text the client should see half of)
                match hints {
                    Some(_) if closed => {
                        if let Some(call) = read_call(body, name) {
                            out.tool_calls.push(call);
                            out.complete_calls = out.tool_calls.len();
                        } else {
                            push(&mut content, body.trim());
                        }
                    }
                    Some(_) => {}
                    None => push(&mut content, body.trim()),
                }
            }
        }
        cur = &body_start[end..];
        if let Some(r) = cur.strip_prefix(END) {
            cur = r;
        }
        // stray text between parts (outside any part) is still model text
        let next = cur.find(MODEL).unwrap_or(cur.len());
        push(&mut content, cur[..next].trim());
        cur = &cur[next..];
    }
    if !reasoning.is_empty() {
        out.reasoning = Some(reasoning);
    }
    if !content.is_empty() {
        out.content = Some(content);
    }
    out
}

/// One call body: `{"name": NAME, "args": {...}}` (the template's canonical
/// form; the header names the same function). The arguments arrive typed -
/// nothing to coerce against the schema.
fn read_call(body: &str, header: &str) -> Option<ToolCallRaw> {
    let v: Value = serde_json::from_str(body.trim()).ok()?;
    let name = v
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .or((!header.is_empty()).then_some(header))?
        .to_owned();
    let args = match v.get("args").or_else(|| v.get("arguments")) {
        Some(a @ Value::Object(_)) => a.to_string(),
        None | Some(Value::Null) => "{}".to_owned(),
        Some(_) => return None,
    };
    Some(ToolCallRaw {
        name,
        arguments: args,
    })
}

/// Append one part body to a channel; separate parts join with a newline,
/// empty ones add nothing (the accumulated string stays a growing PREFIX
/// across streaming re-parses).
fn push(dst: &mut String, s: &str) {
    if s.is_empty() {
        return;
    }
    if !dst.is_empty() {
        dst.push('\n');
    }
    dst.push_str(s);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hints() -> ToolHints {
        ToolHints::default()
    }

    #[test]
    fn reasoning_then_answer() {
        let t = "<|message_model|><|content_thinking|>The user wants X.<|end_message|>\
                 <|message_model|><|content_text|>Here is X.<|end_message|>";
        let p = parse(t, None);
        assert_eq!(p.reasoning.as_deref(), Some("The user wants X."));
        assert_eq!(p.content.as_deref(), Some("Here is X."));
        assert!(p.tool_calls.is_empty());
    }

    #[test]
    fn parallel_calls_with_typed_args() {
        let t = "<|message_model|><|content_thinking|>two lookups<|end_message|>\
                 <|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":3}}<|end_message|>\
                 <|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":{}}<|end_message|>";
        let p = parse(t, Some(&hints()));
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(p.complete_calls, 2);
        assert_eq!(p.tool_calls[0].name, "get_weather");
        assert_eq!(p.tool_calls[0].arguments, "{\"city\":\"Paris\",\"days\":3}");
        assert_eq!(p.tool_calls[1].arguments, "{}");
        assert_eq!(p.content, None);
        assert_eq!(p.finish_reason(), "tool_calls");
    }

    #[test]
    fn streaming_prefixes_hold() {
        let full = "<|message_model|><|content_thinking|>Let me think about it<|end_message|>\
                    <|message_model|><|content_text|>The answer is 42.<|end_message|>";
        // every prefix a stream can show parses, and the channels only ever
        // grow - markers are single tokens, so no cut lands inside one
        let inside: Vec<bool> = {
            let mut v = vec![false; full.len() + 1];
            for m in [MODEL, END, THINKING, TEXT] {
                for (i, _) in full.match_indices(m) {
                    for x in &mut v[i + 1..i + m.len()] {
                        *x = true;
                    }
                }
            }
            v
        };
        let mut last = (String::new(), String::new());
        for end in (1..=full.len()).filter(|&e| full.is_char_boundary(e) && !inside[e]) {
            let p = parse(&full[..end], None);
            let r = p.reasoning.unwrap_or_default();
            let c = p.content.unwrap_or_default();
            assert!(
                r.starts_with(&last.0),
                "reasoning shrank at {end}: {r:?} after {:?}",
                last.0
            );
            assert!(
                c.starts_with(&last.1),
                "content shrank at {end}: {c:?} after {:?}",
                last.1
            );
            last = (r, c);
        }
        assert_eq!(last.0, "Let me think about it");
        assert_eq!(last.1, "The answer is 42.");
    }

    #[test]
    fn an_open_call_streams_nothing() {
        let t = "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"ci";
        let p = parse(t, Some(&hints()));
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content, None);
    }

    #[test]
    fn no_tools_declared_keeps_the_call_visible() {
        let t = "<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>";
        let p = parse(t, None);
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some("{\"name\":\"f\",\"args\":{}}"));
    }

    /// The release's own template through our renderer against
    /// transformers' rendering of the same conversations
    /// (`INKLING2_TEMPLATE_FIXTURE`: [{messages, tools, kwargs, expected}],
    /// the template from `INKLING2_DIR`); skipped where either is absent.
    #[test]
    fn the_template_renders_like_transformers() {
        let (Some(fix), Some(dir)) = (
            std::env::var_os("INKLING2_TEMPLATE_FIXTURE"),
            std::env::var_os("INKLING2_DIR"),
        ) else {
            eprintln!("SKIP: INKLING2_TEMPLATE_FIXTURE / INKLING2_DIR not set");
            return;
        };
        let tpl = std::fs::read_to_string(std::path::Path::new(&dir).join("chat_template.jinja"))
            .expect("template");
        let cases: Vec<Value> =
            serde_json::from_str(&std::fs::read_to_string(fix).expect("fixture")).expect("json");
        for (i, c) in cases.iter().enumerate() {
            let msgs = c["messages"].as_array().expect("messages");
            let tools = c["tools"].as_array().map(|t| t.as_slice());
            let kw = c["kwargs"]
                .as_object()
                .filter(|o| !o.is_empty())
                .map(|_| &c["kwargs"]);
            let got = crate::chat_template::render(&tpl, msgs, tools, kw).expect("renders");
            assert_eq!(got, c["expected"].as_str().expect("expected"), "case {i}");
        }
    }

    #[test]
    fn text_outside_the_parts_is_kept() {
        let p = parse("plain answer", None);
        assert_eq!(p.content.as_deref(), Some("plain answer"));
        let long = format!("<|message_model|>{}", "x".repeat(200));
        assert!(
            parse(&long, None).content.is_some(),
            "a runaway header stays visible"
        );
    }
}
