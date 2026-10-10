//! Pictures inside tool results. An agent's file-read tool returns an image
//! as image blocks inside its `tool_result` (Claude Code's Read on a .png),
//! and the conversion to the chat shape used to keep only the result's text:
//! the model saw an empty tool reply and invented what the picture showed
//! (12 of 12 delivery notes read back wrong on Qwen3.8-27B, 2026-10-09).
//!
//! Where the model can see and its template renders image parts in a tool
//! reply, the reply becomes text + image parts and the picture reaches the
//! tower like a user's would. Everywhere else each picture becomes a text note
//! in its place, so the model knows a picture was there and that it cannot
//! see it - an agent's turn should not die on a 400 for a file it read (a
//! user's own image on a text-only endpoint still gets the honest refusal).

use serde_json::{Value, json};

/// A tool result's content as the chat `tool` message carries it. `see`: the
/// endpoint serves vision AND its template renders tool images. A result
/// without pictures is the plain text it always was (prompt bytes unchanged).
pub(crate) fn tool_content(
    content: Option<&Value>,
    see: bool,
    image_part: impl Fn(&Value) -> Result<Value, String>,
) -> Result<Value, String> {
    let Some(Value::Array(blocks)) = content else {
        return Ok(Value::String(content.map(text_of).unwrap_or_default()));
    };
    if !blocks.iter().any(is_image) {
        return Ok(Value::String(text_of(content.unwrap_or(&Value::Null))));
    }
    if see {
        let mut parts = Vec::new();
        for b in blocks {
            if is_image(b) {
                parts.push(image_part(
                    b.get("source").ok_or("image block needs source")?,
                )?);
            } else if let Some(t) = b.get("text").and_then(Value::as_str) {
                parts.push(json!({"type": "text", "text": t}));
            }
        }
        return Ok(Value::Array(parts));
    }
    Ok(Value::String(noted(blocks)))
}

/// A Responses `function_call_output` output as the `tool` message carries it:
/// the text, or text + image parts when `input_image` items ride along (the
/// API allows them there) - [`blind`] turns those into notes where they
/// cannot be shown.
pub(crate) fn output_content(output: Option<&Value>) -> Value {
    match output {
        Some(Value::Array(parts)) if parts.iter().any(is_image) => Value::Array(
            parts
                .iter()
                .filter_map(|p| match p.get("text").and_then(Value::as_str) {
                    _ if is_image(p) => Some(p.clone()),
                    Some(t) => Some(json!({"type": "text", "text": t})),
                    None => None,
                })
                .collect(),
        ),
        Some(v) => Value::String(text_of(v)),
        None => Value::String(String::new()),
    }
}

/// The conversation with every tool message's pictures as text notes, for
/// an endpoint that cannot show them; None when there are none.
pub(crate) fn blind(messages: &[Value]) -> Option<Vec<Value>> {
    let pictured = |m: &Value| {
        m["role"] == "tool"
            && m["content"]
                .as_array()
                .is_some_and(|c| c.iter().any(is_image))
    };
    if !messages.iter().any(pictured) {
        return None;
    }
    Some(
        messages
            .iter()
            .map(|m| match m["content"].as_array() {
                Some(c) if pictured(m) => {
                    let mut m = m.clone();
                    m["content"] = Value::String(noted(c));
                    m
                }
                _ => m.clone(),
            })
            .collect(),
    )
}

/// The parts' text with a note where each picture was.
fn noted(parts: &[Value]) -> String {
    let mut out = String::new();
    for b in parts {
        if is_image(b) {
            out.push_str("[image not shown: this model cannot see images in tool results]");
        } else if let Some(t) = b.get("text").and_then(Value::as_str) {
            out.push_str(t);
        }
    }
    out
}

/// An Anthropic `image` block, a Responses `input_image` item or a chat
/// `image_url` part.
fn is_image(b: &Value) -> bool {
    matches!(
        b.get("type").and_then(Value::as_str),
        Some("image" | "input_image" | "image_url")
    )
}

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    fn part(src: &serde_json::Value) -> Result<serde_json::Value, String> {
        Ok(json!({"type": "image_url", "image_url": {"url": src["data"]}}))
    }

    #[test]
    fn pictures_reach_a_model_that_sees_and_a_note_stands_in_otherwise() {
        let img = json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QQ=="}});
        let c = json!([{"type": "text", "text": "file:"}, img]);
        let seen = super::tool_content(Some(&c), true, part).unwrap();
        assert_eq!(
            seen,
            json!([{"type": "text", "text": "file:"},
                                {"type": "image_url", "image_url": {"url": "QQ=="}}])
        );
        let blind = super::tool_content(Some(&c), false, part).unwrap();
        assert!(blind.as_str().unwrap().starts_with("file:[image not shown"));
        // text-only results keep their old shape exactly
        let t = json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]);
        assert_eq!(
            super::tool_content(Some(&t), true, part).unwrap(),
            json!("ab")
        );
        assert_eq!(
            super::tool_content(Some(&json!("x")), true, part).unwrap(),
            json!("x")
        );
        assert_eq!(super::tool_content(None, true, part).unwrap(), json!(""));
    }

    #[test]
    fn a_function_call_output_keeps_its_pictures_until_an_endpoint_cannot_show_them() {
        let out = json!([{"type": "input_text", "text": "shot:"},
                         {"type": "input_image", "image_url": "data:image/png;base64,QQ=="}]);
        let content = super::output_content(Some(&out));
        assert_eq!(content[1]["type"], "input_image");
        let msgs = vec![
            json!({"role": "user", "content": "look"}),
            json!({"role": "tool", "tool_call_id": "c", "content": content}),
        ];
        let blind = super::blind(&msgs).expect("a picture to note");
        assert!(
            blind[1]["content"]
                .as_str()
                .unwrap()
                .starts_with("shot:[image not shown")
        );
        assert_eq!(blind[0], msgs[0]);
        assert!(super::blind(&blind).is_none());
        assert_eq!(super::output_content(Some(&json!("plain"))), json!("plain"));
    }
}
