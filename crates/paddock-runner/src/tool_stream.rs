//! Tool calls that stream their arguments while the model writes them.
//!
//! The wire protocols all carry arguments as JSON text a client concatenates
//! (Anthropic `input_json_delta`, OpenAI `tool_calls[].function.arguments`),
//! so a fragment may go out only when no later token can change it: what is
//! sent must stay a PREFIX of the call's final `arguments`. The Qwen XML
//! dialect (`<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...`)
//! is not JSON, but its JSON form can be built in generation order:
//!
//! * the name settles at the `>` closing `<function=NAME`;
//! * a parameter's key settles at the `>` closing `<parameter=KEY`;
//! * a value the request's schema declares a string streams as it arrives,
//!   JSON-escaped, holding back a tail that could still be a closing tag
//!   (`</parameter>`, `</function>`, `</tool_call>`, and `</think>` while the
//!   call sits in a still-open think region) plus the one newline the
//!   template wraps values in;
//! * any other value (a number, an object, an undeclared key) is coerced
//!   against the schema, so it goes out whole once its tag closes.
//!
//! The same builder renders the final arguments the non-streaming parse
//! returns ([`qwen_call_json`] with nothing growing), so the streamed text
//! and the final arguments cannot disagree. Big string arguments are exactly
//! what agents write - a whole file in `Write.content`, an `Edit` pair, a
//! heredoc in `Bash.command` - and at a slow decode they used to arrive as
//! minutes of silence followed by one blob.
//!
//! Dialects without an incremental reader keep the old shape: each call goes
//! out whole, but as soon as its block closes rather than at end of turn.

use crate::parsers::{Dialect, Parsed, ToolHints, coerce, holdback};

pub(crate) const THINK: &str = "<think>";
pub(crate) const THINK_END: &str = "</think>";
pub(crate) const TOOL_CALL: &str = "<tool_call>";
pub(crate) const TOOL_CALL_END: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";
const P_OPEN: &str = "<parameter=";
const P_CLOSE: &str = "</parameter>";

/// What could still end a value that is being written.
const STOPS: &[&str] = &[P_CLOSE, FN_CLOSE, TOOL_CALL_END];
const STOPS_IN_THINK: &[&str] = &[P_CLOSE, FN_CLOSE, TOOL_CALL_END, THINK_END];

/// One `<function=NAME>` block as `(name, arguments JSON)`, None until the
/// name is complete (or when it is empty - such a block is not a call).
///
/// `growing: None` - the block is final: the whole arguments object, the
/// value of an unclosed parameter running to the block's end. This is the
/// parse every response uses.
/// `growing: Some(stops)` - more text may follow: the result stops where the
/// text stops settling it (an open string value's held-back tail, an open
/// value of any other type, the closing brace), and is a prefix of what the
/// final parse will return. The bool says the arguments are complete
/// (`</function>` seen) - then the result is the final one either way.
///
/// Values keep generation order and a repeated key repeats in the text (a
/// JSON reader takes the last), so a prefix never has to be taken back.
pub(crate) fn qwen_call_json(
    block: &str,
    hints: &ToolHints,
    growing: Option<&[&str]>,
) -> Option<(String, String, bool)> {
    let f = block.find(FN_OPEN)?;
    let after = &block[f + FN_OPEN.len()..];
    let name_end = after.find('>')?;
    let name = after[..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let mut body = &after[name_end + 1..];
    let closed = match body.find(FN_CLOSE) {
        Some(e) => {
            body = &body[..e];
            true
        }
        None => false,
    };
    // past `</function>` nothing can change the arguments
    let growing = growing.filter(|_| !closed);
    let param_hints = hints.get(name);
    let mut out = String::from("{");
    let mut cur = body;
    while let Some(p) = cur.find(P_OPEN) {
        let after_p = &cur[p + P_OPEN.len()..];
        let Some(k_end) = after_p.find('>') else {
            break;
        };
        let key = after_p[..k_end].trim();
        let vstart = &after_p[k_end + 1..];
        let (raw, next, done) = match vstart.find(P_CLOSE) {
            Some(e) => (&vstart[..e], &vstart[e + P_CLOSE.len()..], true),
            None => (vstart, "", false),
        };
        let declared_string = param_hints.and_then(|h| h.get(key)).copied();
        let sep = if out.len() > 1 { "," } else { "" };
        match growing {
            Some(stops) if !done => {
                // the value still being written: only a declared string has
                // a settled prefix (anything else is coerced when it closes)
                if declared_string == Some(true) {
                    let quoted = json_text(&serde_json::Value::String(settled(raw, stops).into()));
                    out.push_str(sep);
                    out.push_str(&json_text(&serde_json::Value::String(key.into())));
                    out.push(':');
                    out.push_str(&quoted[..quoted.len() - 1]);
                }
                return Some((name.to_owned(), out, false));
            }
            _ => {
                // the template wraps values in single newlines; inner ones are data
                let val = raw.strip_prefix('\n').unwrap_or(raw);
                let val = val.strip_suffix('\n').unwrap_or(val);
                out.push_str(sep);
                out.push_str(&json_text(&serde_json::Value::String(key.into())));
                out.push(':');
                out.push_str(&json_text(&coerce(val, declared_string)));
            }
        }
        cur = next;
    }
    if growing.is_some() {
        // another parameter or `</function>` may still come
        return Some((name.to_owned(), out, false));
    }
    out.push('}');
    Some((name.to_owned(), out, closed))
}

fn json_text(v: &serde_json::Value) -> String {
    v.to_string()
}

/// The part of an open value no later text can change: without a tail that
/// could still become a closing tag, without the template's leading newline
/// (known once the first byte is in) and without a trailing newline (the one
/// the template puts before the closing tag, should it come next).
fn settled<'a>(raw: &'a str, stops: &[&str]) -> &'a str {
    // markers are ASCII, so the cut always lands on a char boundary
    let s = &raw[..raw.len() - holdback(raw, stops)];
    if s.is_empty() {
        return s;
    }
    let s = s.strip_prefix('\n').unwrap_or(s);
    s.strip_suffix('\n').unwrap_or(s)
}

/// One call as far as the text so far settles it.
#[derive(Debug, PartialEq)]
pub(crate) struct CallProgress {
    pub name: String,
    /// a prefix of the call's final `arguments` - all of them once `complete`
    pub args: String,
    pub complete: bool,
}

/// Every call in a Qwen XML turn so far, in the order `qwen_parse` returns
/// them: the reasoning region's calls, then the answer region's.
fn qwen_progress(text: &str, thinking_open: bool, hints: &ToolHints) -> Vec<CallProgress> {
    let mut out = Vec::new();
    match text.find(THINK_END) {
        // a closed think region is final; the text after it still grows
        Some(i) => {
            scan(&text[..i], hints, None, &mut out);
            scan(&text[i + THINK_END.len()..], hints, Some(STOPS), &mut out);
        }
        // still inside the think block: a `</think>` can yet cut a call short
        None if thinking_open || text.trim_start().starts_with(THINK) => {
            scan(text, hints, Some(STOPS_IN_THINK), &mut out)
        }
        None => scan(text, hints, Some(STOPS), &mut out),
    }
    out
}

/// The `<tool_call>` blocks of one region. `growing`: the region's end is
/// not final yet - its last unclosed block may still be written.
fn scan(region: &str, hints: &ToolHints, growing: Option<&[&str]>, out: &mut Vec<CallProgress>) {
    let mut cur = region;
    while let Some(s) = cur.find(TOOL_CALL) {
        let after = &cur[s + TOOL_CALL.len()..];
        let (block, next, closed) = match after.find(TOOL_CALL_END) {
            Some(e) => (&after[..e], &after[e + TOOL_CALL_END.len()..], true),
            None => (after, "", false),
        };
        let grows = growing.filter(|_| !closed);
        if let Some((name, args, done)) = qwen_call_json(block, hints, grows) {
            out.push(CallProgress {
                name,
                args,
                complete: grows.is_none() || done,
            });
        }
        cur = next;
    }
}

/// What a stream sends for its tool calls, in order: a call starts (its
/// name is known), its arguments grow, it stops (they are complete).
#[derive(Debug, PartialEq)]
pub(crate) enum ToolEv {
    Start { k: usize, name: String },
    Args { k: usize, json: String },
    Stop { k: usize },
}

/// Per-stream tool-call state: which calls started, how much of the open
/// one's arguments went out, which stopped.
#[derive(Debug, Default)]
pub(crate) struct ToolStream {
    started: usize,
    stopped: usize,
    sent: String,
}

impl ToolStream {
    /// Mid-stream: the events this tick's text settles. `parsed` is the
    /// tick's parse of the same text (dialects without an incremental reader
    /// send each call whole once its block closes).
    pub(crate) fn step(
        &mut self,
        dialect: Dialect,
        text: &str,
        thinking_open: bool,
        hints: Option<&ToolHints>,
        single: bool,
        parsed: &Parsed,
    ) -> Vec<ToolEv> {
        let Some(hints) = hints else {
            return Vec::new();
        };
        let mut progress = match dialect {
            Dialect::QwenXml => qwen_progress(text, thinking_open, hints),
            _ => whole(parsed, parsed.complete_calls),
        };
        if single {
            progress.truncate(1);
        }
        self.advance(&progress)
    }

    /// End of turn: every call of the final parse, whole - the open one
    /// finishes, any never started goes out complete.
    pub(crate) fn finish(&mut self, parsed: &Parsed) -> Vec<ToolEv> {
        self.advance(&whole(parsed, parsed.tool_calls.len()))
    }

    /// Calls that have started streaming (a client-facing count).
    pub(crate) fn started(&self) -> usize {
        self.started
    }

    fn advance(&mut self, progress: &[CallProgress]) -> Vec<ToolEv> {
        let mut evs = Vec::new();
        for (k, p) in progress.iter().enumerate().skip(self.stopped) {
            if k >= self.started {
                evs.push(ToolEv::Start {
                    k,
                    name: p.name.clone(),
                });
                self.started = k + 1;
                self.sent.clear();
            }
            if p.args.len() > self.sent.len() {
                match p.args.strip_prefix(self.sent.as_str()) {
                    Some(more) => {
                        evs.push(ToolEv::Args {
                            k,
                            json: more.to_owned(),
                        });
                        self.sent = p.args.clone();
                    }
                    // the builder's contract is prefix stability; a client
                    // given a broken prefix cannot be repaired, so say so
                    None => tracing::warn!(
                        call = k,
                        sent = self.sent.len(),
                        "tool-call stream: arguments diverged from the streamed prefix"
                    ),
                }
            }
            if !p.complete {
                break;
            }
            evs.push(ToolEv::Stop { k });
            self.stopped = k + 1;
        }
        evs
    }
}

/// One event as a chat-completions `delta` (None: nothing to send - a call's
/// end shows only in the finish chunk). OpenAI's shape: the first fragment of
/// a call carries its index, id, type and name with empty arguments, later
/// ones the index and an arguments piece. Legacy `functions` clients read
/// `function_call` instead and have room for one call (parallel calls are
/// pinned off for them).
pub(crate) fn chat_delta(ev: ToolEv, legacy: bool) -> Option<serde_json::Value> {
    use serde_json::json;
    Some(match (ev, legacy) {
        (ToolEv::Start { name, .. }, true) => {
            json!({"function_call": {"name": name, "arguments": ""}})
        }
        (ToolEv::Start { k, name }, false) => json!({"tool_calls": [{
            "index": k, "id": format!("call_{}", uuid::Uuid::new_v4().simple()), "type": "function",
            "function": {"name": name, "arguments": ""}}]}),
        (ToolEv::Args { json, .. }, true) => json!({"function_call": {"arguments": json}}),
        (ToolEv::Args { k, json }, false) => {
            json!({"tool_calls": [{"index": k, "function": {"arguments": json}}]})
        }
        (ToolEv::Stop { .. }, _) => return None,
    })
}

/// The first `n` calls of a parse, each complete.
fn whole(parsed: &Parsed, n: usize) -> Vec<CallProgress> {
    parsed.tool_calls[..n.min(parsed.tool_calls.len())]
        .iter()
        .map(|tc| CallProgress {
            name: tc.name.clone(),
            args: tc.arguments.clone(),
            complete: true,
        })
        .collect()
}

#[cfg(test)]
#[path = "tool_stream_tests.rs"]
mod tests;
