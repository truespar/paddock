//! Output items of a streamed `/v1/responses` turn that are not plain text
//! deltas: closing the reasoning and message items, and the function_call
//! items, which stream their arguments while the model writes them (see
//! `tool_stream`). Items go out in generation order, so the reasoning and
//! message items close when the first call starts. Prose a model writes
//! after a call (rare: reasoning resumed past a call made inside a think
//! region) has no open item left to stream into; it is in the completed
//! response's output, where non-streaming clients read it too.

use serde_json::{Value, json};

use crate::tool_stream::ToolEv;

/// (event name, event data) pairs, ready for the SSE writer.
pub(crate) type Events = Vec<(&'static str, Value)>;

/// The stream's `sequence_number` counter.
#[derive(Default)]
pub(crate) struct Seq(u64);

impl Seq {
    pub(crate) fn next(&mut self) -> u64 {
        let s = self.0;
        self.0 += 1;
        s
    }
}

/// The reasoning item's closing pair, carrying its whole text.
pub(crate) fn reasoning_done(sq: &mut Seq, id: &str, idx: usize, text: &str) -> Events {
    vec![
        (
            "response.reasoning_text.done",
            json!({
            "type":"response.reasoning_text.done","sequence_number":sq.next(),
            "item_id":id,"output_index":idx,"content_index":0,"text":text}),
        ),
        (
            "response.output_item.done",
            json!({
            "type":"response.output_item.done","sequence_number":sq.next(),"output_index":idx,
            "item":{"type":"reasoning","id":id,"summary":[],
                    "content":[{"type":"reasoning_text","text":text}]}}),
        ),
    ]
}

/// The message item's closing events; the done event carries the full
/// logprobs run, and the part itself carries it too when the include asked.
pub(crate) fn message_done(
    sq: &mut Seq,
    id: &str,
    idx: usize,
    text: &str,
    lp_all: &[Value],
    want_logprobs: bool,
) -> Events {
    let mut part = json!({"type":"output_text","text":text,"annotations":[]});
    if want_logprobs {
        part["logprobs"] = json!(lp_all);
    }
    vec![
        (
            "response.output_text.done",
            json!({
            "type":"response.output_text.done","sequence_number":sq.next(),
            "item_id":id,"output_index":idx,"content_index":0,
            "text":text,"logprobs":lp_all}),
        ),
        (
            "response.content_part.done",
            json!({
            "type":"response.content_part.done","sequence_number":sq.next(),
            "item_id":id,"output_index":idx,"content_index":0,"part":part.clone()}),
        ),
        (
            "response.output_item.done",
            json!({
            "type":"response.output_item.done","sequence_number":sq.next(),"output_index":idx,
            "item":{"type":"message","id":id,"role":"assistant","status":"completed",
                    "content":[part]}}),
        ),
    ]
}

/// The function_call item being written.
struct Call {
    fc_id: String,
    call_id: String,
    idx: usize,
    name: String,
    args: String,
}

#[derive(Default)]
pub(crate) struct Calls {
    open: Option<Call>,
    completed: Vec<Value>,
}

impl Calls {
    /// The terminal snapshot must reuse the exact item and call IDs emitted
    /// on the wire, including a call completed only by ToolStream::finish.
    pub(crate) fn into_items(self) -> Vec<Value> {
        self.completed
    }

    /// One tool event as Responses events; `first`: the output index of
    /// call 0 (after the reasoning and message items, when there are any).
    pub(crate) fn event(&mut self, ev: ToolEv, first: usize, sq: &mut Seq) -> Events {
        match ev {
            ToolEv::Start { k, name } => {
                let c = Call {
                    fc_id: format!("fc_{}", uuid::Uuid::new_v4().simple()),
                    call_id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                    idx: first + k,
                    name,
                    args: String::new(),
                };
                let out = vec![(
                    "response.output_item.added",
                    json!({
                    "type":"response.output_item.added","sequence_number":sq.next(),"output_index":c.idx,
                    "item":{"type":"function_call","id":c.fc_id,"call_id":c.call_id,
                            "name":c.name,"arguments":"","status":"in_progress"}}),
                )];
                self.open = Some(c);
                out
            }
            ToolEv::Args { json, .. } => match &mut self.open {
                Some(c) => {
                    c.args.push_str(&json);
                    vec![(
                        "response.function_call_arguments.delta",
                        json!({
                        "type":"response.function_call_arguments.delta","sequence_number":sq.next(),
                        "item_id":c.fc_id,"output_index":c.idx,"delta":json}),
                    )]
                }
                None => Vec::new(),
            },
            ToolEv::Stop { .. } => match self.open.take() {
                Some(c) => {
                    let item = json!({"type":"function_call","id":c.fc_id,"call_id":c.call_id,
                        "name":c.name,"arguments":c.args,"status":"completed"});
                    self.completed.push(item.clone());
                    vec![
                        (
                            "response.function_call_arguments.done",
                            json!({
                        "type":"response.function_call_arguments.done","sequence_number":sq.next(),
                        "item_id":c.fc_id,"output_index":c.idx,"arguments":c.args}),
                        ),
                        (
                            "response.output_item.done",
                            json!({
                        "type":"response.output_item.done","sequence_number":sq.next(),"output_index":c.idx,
                        "item":item}),
                        ),
                    ]
                }
                None => Vec::new(),
            },
        }
    }
}

/// One SSE event: `event: NAME` with the JSON body.
pub(crate) fn ev(
    name: &str,
    data: Value,
) -> Result<axum::response::sse::Event, std::convert::Infallible> {
    Ok(axum::response::sse::Event::default()
        .event(name)
        .data(data.to_string()))
}

/// Emit-safe prefix length: hold back a partial marker tail, on a char
/// boundary.
pub(crate) fn safe_len(s: &str, markers: &[&str]) -> usize {
    let mut n = s.len() - crate::parsers::holdback(s, markers);
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    n
}
