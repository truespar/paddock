//! The content blocks of one streamed Anthropic message, in the order the
//! model writes them: thinking, text and tool_use blocks open as their first
//! bytes are known and close when the next kind begins. A tool call streams
//! its `input_json_delta`s while it is generated (see `tool_stream`), so a
//! block can follow a tool_use block - text after a call, or reasoning
//! resumed after a call written inside a think region.

use serde_json::{Value, json};

use crate::tool_stream::ToolEv;

/// (event name, event data) pairs, ready for the SSE writer.
pub(crate) type Events = Vec<(&'static str, Value)>;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Thinking,
    Text,
    Tool,
}

pub(crate) struct AnthBlocks {
    index: usize,
    open: Option<Kind>,
    /// thinking.display "omitted": thinking blocks open and close empty
    omit_thinking: bool,
    /// a tool block went out: whitespace alone opens nothing after it (the
    /// newlines between parallel calls are framing, not an answer)
    after_tool: bool,
    /// whitespace waiting for a non-blank delta of the same kind
    pending_ws: Option<(Kind, String)>,
    /// prose that arrived while a call was open, replayed once it stops
    held: Vec<(Kind, String)>,
}

impl AnthBlocks {
    /// `index`: the first block's index (a compaction block may lead).
    pub(crate) fn new(index: usize, omit_thinking: bool) -> AnthBlocks {
        AnthBlocks {
            index,
            open: None,
            omit_thinking,
            after_tool: false,
            pending_ws: None,
            held: Vec::new(),
        }
    }

    pub(crate) fn thinking(&mut self, delta: &str) -> Events {
        self.prose(Kind::Thinking, delta)
    }

    pub(crate) fn text(&mut self, delta: &str) -> Events {
        self.prose(Kind::Text, delta)
    }

    fn prose(&mut self, kind: Kind, delta: &str) -> Events {
        let mut out = Vec::new();
        if delta.is_empty() {
            return out;
        }
        if self.open == Some(Kind::Tool) {
            self.held.push((kind, delta.to_owned()));
            return out;
        }
        let mut delta = delta.to_owned();
        if self.open != Some(kind) {
            if self.after_tool && delta.trim().is_empty() {
                match &mut self.pending_ws {
                    Some((k, ws)) if *k == kind => ws.push_str(&delta),
                    _ => self.pending_ws = Some((kind, delta)),
                }
                return out;
            }
            if let Some((k, ws)) = self.pending_ws.take()
                && k == kind
            {
                delta.insert_str(0, &ws);
            }
            self.close_into(&mut out);
            self.open = Some(kind);
            let block = match kind {
                Kind::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
                _ => json!({"type": "text", "text": ""}),
            };
            out.push((
                "content_block_start",
                json!({
                "type": "content_block_start", "index": self.index, "content_block": block}),
            ));
        }
        let d = match kind {
            // the block opens and closes, but no thinking text goes over the wire
            Kind::Thinking if self.omit_thinking => return out,
            Kind::Thinking => json!({"type": "thinking_delta", "thinking": delta}),
            _ => json!({"type": "text_delta", "text": delta}),
        };
        out.push((
            "content_block_delta",
            json!({
            "type": "content_block_delta", "index": self.index, "delta": d}),
        ));
        out
    }

    pub(crate) fn tool(&mut self, ev: ToolEv) -> Events {
        let mut out = Vec::new();
        match ev {
            ToolEv::Start { name, .. } => {
                self.close_into(&mut out);
                self.pending_ws = None;
                let id = format!("toolu_{}", uuid::Uuid::new_v4().simple());
                out.push((
                    "content_block_start",
                    json!({
                    "type": "content_block_start", "index": self.index,
                    "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}}),
                ));
                self.open = Some(Kind::Tool);
            }
            ToolEv::Args { json, .. } => out.push((
                "content_block_delta",
                json!({
                "type": "content_block_delta", "index": self.index,
                "delta": {"type": "input_json_delta", "partial_json": json}}),
            )),
            ToolEv::Stop { .. } => {
                self.close_into(&mut out);
                self.after_tool = true;
                for (kind, d) in std::mem::take(&mut self.held) {
                    out.extend(self.prose(kind, &d));
                }
            }
        }
        out
    }

    /// End of message: close whatever is open.
    pub(crate) fn close(&mut self) -> Events {
        let mut out = Vec::new();
        self.close_into(&mut out);
        out
    }

    fn close_into(&mut self, out: &mut Events) {
        if self.open.take().is_some() {
            out.push((
                "content_block_stop",
                json!({
                "type": "content_block_stop", "index": self.index}),
            ));
            self.index += 1;
        }
    }
}
