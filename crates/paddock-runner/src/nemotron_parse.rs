//! Nemotron 3.5 Parse's request mapping: the chat template's rendering, the
//! `ocr` request object, its two named prompts and the block parse of its
//! output.
//!
//! The checkpoint is a page parser, not a chat model: it reads ONE page
//! picture and a task prompt "composed from the supported control tokens"
//! (the model card) - `</s><s>` then one switch from each group: boxes
//! (`<predict_bbox>` / `<no_bbox>`), classes (`<predict_classes>` /
//! `<no_classes>`), the text form (`<output_markdown>` / `<output_plain>` /
//! `<output_ocr>` / `<output_no_text>`) and text inside pictures
//! (`<predict_text_in_pic>` / `<predict_no_text_in_pic>`). It answers with
//! elements of `<x_A><y_B>text<x_C><y_D><class_Name>` (boxes normalized to
//! the page's content, x on 1024 bins, y on 1280; tables as LaTeX
//! `tabular`), the parts a prompt switched off left out.
//!
//! The release's chat template is `{% for message in messages %}{{
//! message['content'] }}{% endfor %}`: every message's content, in order,
//! whatever its role, nothing between them. That is how a request renders
//! here too:
//! - each message's text parts concatenate in order; a message carrying the
//!   page puts the picture placeholder `<pad>` (the checkpoint's
//!   `image_token_index`, 1) at its start - the only layout the checkpoint
//!   was trained on is picture first, and it is where vLLM puts an implicit
//!   placeholder too (without the newline vLLM glues after it, which the
//!   Transformers contract does not have). A `<pad>` the caller wrote
//!   marks the picture's place instead;
//! - so system text, earlier turns and a trailing assistant message (an
//!   output prefix to continue from) all reach the model exactly as the
//!   template renders them, and any switch combination passes verbatim.
//!
//! Two exceptions, both the cross-family document-parser contract: an
//! explicit `ocr.mode` (document = the published contract prompt, layout =
//! NVIDIA's detection pass) replaces the caller's text, and a request whose
//! text carries no control token at all - only the page, or free text this
//! model has no reading for - gets the document prompt, with any text it
//! dropped echoed as `dropped_text`. Output is the model's bytes with every
//! coordinate and class token kept (the contract's
//! `skip_special_tokens=False`; those ids are ordinary added tokens, so
//! detokenization keeps them).

use serde_json::{Value, json};

use crate::deepseek_ocr::{OcrResolved, Region};

/// The engine arch string this module serves.
pub const ARCH: &str = "nemotron-parse";

const DOCUMENT_PROMPT: &str = paddock_models::nemotron_parse::CONTRACT_PROMPT;
const LAYOUT_PROMPT: &str =
    "</s><s><predict_bbox><predict_classes><output_no_text><predict_no_text_in_pic>";
/// The checkpoint's picture placeholder (`<pad>`, id 1).
pub const IMAGE_MARKER: &str = "<pad>";
/// The long edge PDF pages render to: the processor's height cap, so an A4
/// page reaches it at 2048 x 1448 and is never resampled.
pub const PDF_EDGE: u32 = 2048;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NpMode {
    Document,
    Layout,
}

impl NpMode {
    fn parse(s: &str) -> Result<NpMode, String> {
        match s {
            "document" => Ok(NpMode::Document),
            "layout" => Ok(NpMode::Layout),
            other => Err(format!(
                "invalid ocr.mode {other:?} (expected document or layout)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            NpMode::Document => "document",
            NpMode::Layout => "layout",
        }
    }

    fn prompt(self) -> &'static str {
        match self {
            NpMode::Document => DOCUMENT_PROMPT,
            NpMode::Layout => LAYOUT_PROMPT,
        }
    }

    pub const ALL: [NpMode; 2] = [NpMode::Document, NpMode::Layout];
}

pub fn caps_json() -> Value {
    json!({
        "modes": NpMode::ALL.map(NpMode::as_str),
        "crops": [],
        "grounding": true,
    })
}

pub fn parse_opts(v: &Value) -> Result<Option<NpMode>, String> {
    let Some(obj) = v.as_object() else {
        return Err("ocr must be a JSON object".into());
    };
    let mut mode = None;
    for (k, v) in obj {
        match k.as_str() {
            "mode" => {
                let s = v.as_str().ok_or("ocr.mode must be a string")?;
                mode = Some(NpMode::parse(s)?);
            }
            other => {
                return Err(format!(
                    "unknown ocr field {other:?} for this family (only `mode`: document or layout)"
                ));
            }
        }
    }
    Ok(mode)
}

/// The two accepted channels, same precedence as the other parsers.
pub fn opts_from_request(
    top: Option<&Value>,
    kwargs: Option<&Value>,
) -> Result<Option<NpMode>, String> {
    let kw = kwargs
        .and_then(|k| k.as_object())
        .and_then(|k| k.get("ocr"));
    match (top, kw) {
        (Some(t), other) => {
            if other.is_some() {
                tracing::warn!(
                    "both top-level `ocr` and `chat_template_kwargs.ocr` sent - using the \
                     top-level object"
                );
            }
            parse_opts(t)
        }
        (None, Some(k)) => parse_opts(k),
        (None, None) => Ok(None),
    }
}

/// Resolve one request and rewrite the turn the template will render.
pub fn resolve(
    messages: &mut [Value],
    mode: Option<NpMode>,
    pages: usize,
) -> Result<Option<OcrResolved>, String> {
    if pages == 0 {
        if mode.is_some() {
            return Err(
                "the ocr request object applies to page requests - this request has no image \
                 (attach the page as an image or PDF)"
                    .into(),
            );
        }
        return Ok(None);
    }
    if pages > 1 {
        return Err(format!(
            "Nemotron 3.5 Parse reads one page a request, this one carries {pages} - send each \
             page on its own (a PDF part can pick one with `pages`)"
        ));
    }
    // the template's rendering, message by message
    let rendered: Vec<(String, bool)> = messages.iter().map(render_message).collect();
    let all: String = rendered.iter().map(|(t, _)| t.as_str()).collect();
    let marks = all.matches(IMAGE_MARKER).count();
    let switched = has_task_switch(&all);
    let caller_text = all.replace(IMAGE_MARKER, "");
    let (resolved, verbatim, dropped) = match mode {
        Some(m) => (Some(m), false, !caller_text.trim().is_empty()),
        None if switched => {
            // named by the prompt side alone: an assistant message is an
            // output prefix the reading continues from, not part of the task
            let prompt_side: String = messages
                .iter()
                .zip(&rendered)
                .filter(|(m, _)| m.get("role").and_then(Value::as_str) != Some("assistant"))
                .map(|(_, (t, _))| t.replace(IMAGE_MARKER, ""))
                .collect();
            let named = match prompt_side.trim() {
                p if p == DOCUMENT_PROMPT => Some(NpMode::Document),
                p if p == LAYOUT_PROMPT => Some(NpMode::Layout),
                _ => None,
            };
            (named, true, false)
        }
        None => {
            let had = !caller_text.trim().is_empty();
            if had {
                tracing::warn!(
                    "nemotron-parse: the request's text carries none of the checkpoint's task \
                     switches - reading the page with the document prompt instead"
                );
            }
            (Some(NpMode::Document), false, had)
        }
    };
    if verbatim {
        if marks > 1 {
            return Err(format!(
                "the picture placeholder {IMAGE_MARKER} appears {marks} times - Nemotron 3.5 \
                 Parse reads one page a request"
            ));
        }
        // a page that rode no message part (none to lead) goes at the start
        // of the last user turn, the picture-first layout
        let lead_at = if marks == 0 {
            Some(
                messages
                    .iter()
                    .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                    .ok_or("nemotron-parse: the request has no user message")?,
            )
        } else {
            None
        };
        for (i, (m, (text, _))) in messages.iter_mut().zip(&rendered).enumerate() {
            let content = if lead_at == Some(i) {
                format!("{IMAGE_MARKER}{text}")
            } else {
                text.clone()
            };
            if let Some(obj) = m.as_object_mut() {
                obj.insert("content".into(), Value::String(content));
            }
        }
    } else {
        let prompt = resolved.expect("a named prompt").prompt();
        // one user turn: the picture, then the prompt; every other message
        // renders empty (the template concatenates all contents)
        let last_user = messages
            .iter()
            .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            .ok_or("nemotron-parse: the request has no user message")?;
        for (i, m) in messages.iter_mut().enumerate() {
            if let Some(obj) = m.as_object_mut() {
                let content = if i == last_user {
                    format!("{IMAGE_MARKER}{prompt}")
                } else {
                    String::new()
                };
                obj.insert("content".into(), Value::String(content));
            }
        }
    }
    let pass_through = verbatim;
    Ok(Some(OcrResolved {
        mode: resolved.map(NpMode::as_str),
        crop: "page",
        force_base: false,
        grounding: true,
        pages,
        views: pages,
        tiles: 0,
        image_tokens: 0,
        pass_through,
        dropped_text: dropped,
        ngram: (0, 0),
        repeat_stop: false,
    }))
}

/// The control tokens a task prompt is composed from (one per group).
const TASK_SWITCHES: [&str; 10] = [
    "<predict_bbox>",
    "<no_bbox>",
    "<predict_classes>",
    "<no_classes>",
    "<output_markdown>",
    "<output_plain>",
    "<output_ocr>",
    "<output_no_text>",
    "<predict_text_in_pic>",
    "<predict_no_text_in_pic>",
];

fn has_task_switch(text: &str) -> bool {
    TASK_SWITCHES.iter().any(|t| text.contains(t))
}

/// One message as the template renders it: its text parts in order, and
/// whether it carries the page (an image part, or a PDF part the runner
/// rasterizes) - in which case the placeholder leads its content unless the
/// caller's text already places it.
fn render_message(m: &Value) -> (String, bool) {
    match m.get("content") {
        Some(Value::String(s)) => (s.clone(), false),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            let mut page = false;
            for p in parts {
                let ty = p.get("type").and_then(Value::as_str);
                if p.get("image").is_some()
                    || p.get("image_url").is_some()
                    || matches!(ty, Some("image" | "file" | "input_file" | "document"))
                {
                    page = true;
                } else if matches!(ty, Some("text" | "input_text" | "output_text"))
                    && let Some(t) = p.get("text").and_then(Value::as_str)
                {
                    text.push_str(t);
                }
            }
            if page && !text.contains(IMAGE_MARKER) {
                text.insert_str(0, IMAGE_MARKER);
            }
            (text, page)
        }
        _ => (String::new(), false),
    }
}

/// `<x_0.0654>` -> 0.0654, the coordinate's own value.
fn coord(tok: &str, axis: char) -> Option<f64> {
    tok.strip_prefix('<')?
        .strip_prefix(axis)?
        .strip_prefix('_')?
        .strip_suffix('>')?
        .parse::<f64>()
        .ok()
}

/// Split `s` at its leading `<...>` token, if it starts with one.
fn take_tag(s: &str) -> Option<(&str, &str)> {
    if !s.starts_with('<') {
        return None;
    }
    let end = s.find('>')?;
    Some((&s[..=end], &s[end + 1..]))
}

/// The elements of a Nemotron Parse answer as regions: boxes on the 0..1000
/// grid of the page content (the convention every parser's regions share),
/// label = the class name, text = what sits between the two corners. A
/// prompt with `<no_classes>` writes no class token - such an element gets
/// the label `element`; one with `<no_bbox>` writes no boxes, so it has no
/// regions (its text is the answer); `<output_no_text>` / `<output_ocr>`
/// elements have no text. A malformed tail (a read cut at the token cap) is
/// dropped, not guessed.
pub fn parse_blocks(raw: &str) -> Vec<Region> {
    let mut out = Vec::new();
    let mut rest = raw;
    let grid = |v: f64| ((v * 1000.0).round() as i64).clamp(0, 1000);
    while let Some(i) = rest.find("<x_") {
        rest = &rest[i..];
        let Some((xt, r1)) = take_tag(rest) else {
            break;
        };
        let Some((yt, r2)) = take_tag(r1) else { break };
        let (Some(x0), Some(y0)) = (coord(xt, 'x'), coord(yt, 'y')) else {
            rest = &rest[1..];
            continue;
        };
        // the element's text runs to its closing corner
        let Some(j) = r2.find("<x_") else { break };
        let text = &r2[..j];
        let r3 = &r2[j..];
        let Some((xt2, r4)) = take_tag(r3) else { break };
        let Some((yt2, r5)) = take_tag(r4) else { break };
        let (Some(x1), Some(y1)) = (coord(xt2, 'x'), coord(yt2, 'y')) else {
            rest = &r2[j..];
            continue;
        };
        let (label, after) = match take_tag(r5)
            .and_then(|(ct, r6)| Some((ct.strip_prefix("<class_")?.strip_suffix('>')?, r6)))
        {
            Some((c, r6)) => (c, r6),
            // no class token follows: a `<no_classes>` answer (or the cap cut
            // it - the corners are whole, so the box stands)
            None => ("element", r5),
        };
        let text = text.trim();
        out.push(Region {
            label: label.to_owned(),
            boxes: vec![[grid(x0), grid(y0), grid(x1), grid(y1)]],
            text: (!text.is_empty()).then(|| text.to_owned()),
            quads: Vec::new(),
            continues: false,
        });
        rest = after;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_parse_with_classes_and_text() {
        let raw = "<x_0.0654><y_0.0484>NORDVIK BYGG AB<x_0.3174><y_0.0617><class_Page-header>\n\n\
                   <x_0.0596><y_0.2336>\\begin{tabular}{cc}\na & b\\\\\n\\end{tabular}<x_0.9346><y_0.8141><class_Table>\n\n\
                   <x_0.9014><y_0.95>2 (2)<x_0.9316><y_0.9609><class_Text></s>";
        let rs = parse_blocks(raw);
        assert_eq!(rs.len(), 3);
        assert_eq!(rs[0].label, "Page-header");
        assert_eq!(rs[0].boxes[0], [65, 48, 317, 62]);
        assert_eq!(rs[0].text.as_deref(), Some("NORDVIK BYGG AB"));
        assert_eq!(rs[1].label, "Table");
        assert!(
            rs[1]
                .text
                .as_deref()
                .is_some_and(|t| t.starts_with("\\begin{tabular}"))
        );
        assert_eq!(rs[2].label, "Text");
    }

    #[test]
    fn a_cut_tail_is_dropped() {
        let raw = "<x_0.1><y_0.1>done<x_0.2><y_0.2><class_Text><x_0.3><y_0.3>cut mid-elem";
        assert_eq!(parse_blocks(raw).len(), 1);
    }

    #[test]
    fn the_turn_is_rewritten_picture_first() {
        let mut msgs = vec![
            json!({"role": "system", "content": "be helpful"}),
            json!({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "extract tables"}]}),
        ];
        let r = resolve(&mut msgs, None, 1)
            .expect("resolve")
            .expect("a page");
        assert_eq!(r.mode, Some("document"));
        assert!(r.dropped_text);
        assert_eq!(msgs[0]["content"], "");
        assert_eq!(msgs[1]["content"], format!("<pad>{DOCUMENT_PROMPT}"));
        let mut msgs = vec![json!({"role": "user", "content": [{"type": "image"}]})];
        let r = resolve(&mut msgs, Some(NpMode::Layout), 1)
            .expect("resolve")
            .expect("a page");
        assert_eq!(r.mode, Some("layout"));
        assert!(!r.dropped_text);
        assert_eq!(msgs[0]["content"], format!("<pad>{LAYOUT_PROMPT}"));
    }

    #[test]
    fn answers_without_classes_or_text_still_give_regions() {
        let raw = "<x_0.0801><y_0.0609>**Across**<x_0.4502><y_0.0875>\n\n\
                   <x_0.0791><y_0.1086>Only about<x_0.9><y_0.2>\n\n";
        let rs = parse_blocks(raw);
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[0].label, "element");
        assert_eq!(rs[0].text.as_deref(), Some("**Across**"));
        let raw = "<x_0.08><y_0.06><x_0.45><y_0.08><class_Title>\n\n\
                   <x_0.08><y_0.11><x_0.88><y_0.19><class_Text>";
        let rs = parse_blocks(raw);
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[1].label, "Text");
        assert!(rs[1].text.is_none());
        // no boxes, no regions
        assert!(parse_blocks("171<class_Page-header>\n\n## Across<class_Title>").is_empty());
    }

    #[test]
    fn the_template_renders_every_message_verbatim() {
        let custom = "</s><s><no_bbox><predict_classes><output_plain><predict_text_in_pic>";
        // NVIDIA's client order (text, then the page) with a system message:
        // every content in order, the picture at the start of its message
        let mut msgs = vec![
            json!({"role": "system", "content": "be precise"}),
            json!({"role": "user", "content": [
                {"type": "text", "text": custom},
                {"type": "image_url", "image_url": {"url": "data:"}}
            ]}),
        ];
        let r = resolve(&mut msgs, None, 1)
            .expect("resolve")
            .expect("a page");
        assert_eq!(r.mode, None);
        assert!(r.pass_through && !r.dropped_text);
        assert_eq!(msgs[0]["content"], "be precise");
        assert_eq!(msgs[1]["content"], format!("<pad>{custom}"));
        // a trailing assistant message is an output prefix to continue from
        let mut msgs = vec![
            json!({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DOCUMENT_PROMPT}]}),
            json!({"role": "assistant", "content": "<x_0.08><y_0.06>"}),
        ];
        let r = resolve(&mut msgs, None, 1)
            .expect("resolve")
            .expect("a page");
        assert_eq!(r.mode, Some("document"));
        assert_eq!(msgs[0]["content"], format!("<pad>{DOCUMENT_PROMPT}"));
        assert_eq!(msgs[1]["content"], "<x_0.08><y_0.06>");
        // the caller's own placeholder (vLLM's offline example) places it
        let mine = format!("<pad>\n{DOCUMENT_PROMPT}");
        let mut msgs = vec![json!({"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": mine.clone()}
        ]})];
        resolve(&mut msgs, None, 1).expect("resolve");
        assert_eq!(msgs[0]["content"], mine);
        // two placeholders for one page
        let mut msgs = vec![json!({"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": format!("<pad><pad>{DOCUMENT_PROMPT}")}
        ]})];
        assert!(resolve(&mut msgs, None, 1).is_err());
    }

    #[test]
    fn more_than_one_page_is_refused() {
        let mut msgs =
            vec![json!({"role": "user", "content": [{"type": "image"}, {"type": "image"}]})];
        assert!(resolve(&mut msgs, None, 2).is_err());
    }
}
