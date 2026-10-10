//! LightOnOCR-3 instruction mapping: the family's `ocr` request object, its
//! page geometry and the grounding-block parse.
//!
//! LightOnOCR-3 0.8B and 4B are stock Qwen3.5 vision-language graphs
//! (`Qwen3_5ForConditionalGeneration`) that LightOn fine-tuned into page
//! readers, so the engine serves them as `qwen35` unchanged and everything
//! that makes them OCR models lives here. The switch is the checkpoint's own
//! identity (`general.basename`), never the arch - the arch is the same
//! string Qwen3.5-9B carries. The 1B keeps LightOnOCR-2's Pixtral graph and
//! is not served.
//!
//! The interface, read from LightOn's model cards and their client
//! (lightonai/LightOnOCR: `client.py`, `grounding.py`, `models.py`,
//! `render.py`):
//!
//! - **plain** - the page image alone, no text at all -> markdown with HTML
//!   tables and LaTeX. An empty text part renders the same bytes as none
//!   (checked against transformers 5.13's `apply_chat_template`).
//! - **grounding** - the image followed by the text `grounding` -> one
//!   `![label](x1,y1,x2,y2) text` block per layout element, boxes on a
//!   0..=1000 grid per axis; images get a short description, charts an HTML
//!   table of their data.
//!
//! "Other instructions are out of distribution: use the empty prompt or
//! `grounding`" (the cards). So an image with no text derives plain, an
//! explicit `ocr.mode` wins over caller text (echoed as `dropped_text`), and
//! any other caller text still passes through verbatim - the conformance
//! floor every document parser here keeps - but is logged as off-distribution.
//!
//! Not done here: LightOn's client repairs escaped math delimiters
//! (`\$x\$` -> `$x$`) after the fact. That is a client post-process, and the
//! response content stays the model's own bytes, as vLLM returns them.

use serde_json::{Value, json};

use crate::deepseek_ocr::{OcrResolved, Region, body_text, set_body_text};

/// `general.basename` LightOn's checkpoints convert to.
const BASENAME: &str = "LightOnOCR-3";

/// The word that selects grounding - byte-exact to the client's
/// `GROUNDING_PROMPT`.
const GROUNDING_PROMPT: &str = "grounding";

/// Longest page edge the 0.8B and 4B were trained and evaluated at. Their
/// processor applies no resize beyond Qwen's 32-px grid, so this IS the
/// inference resolution: the client downscales anything larger (Pillow
/// LANCZOS, never upscaling) and renders PDFs straight to it.
pub const PAGE_EDGE: u32 = 2048;

/// The client's `fit`, byte for byte: shrink so the longest edge is at most
/// `edge`, never upscaling, each side `max(1, round(side * scale))` with
/// Python's round (half to even), resampled with Pillow's LANCZOS - our exact
/// integer port of it. Runs before the request's token budget, so a page the
/// client would have sent at 2048 px reaches the tower at 2048 px (2048 x 2048
/// is exactly the 4096-token `auto` cap, so `auto` never binds after it).
pub fn fit_page(rgb: image::RgbImage, edge: u32) -> image::RgbImage {
    let (w, h) = rgb.dimensions();
    let scale = f64::from(edge) / f64::from(w.max(h));
    if scale >= 1.0 {
        return rgb;
    }
    let side = |v: u32| ((f64::from(v) * scale).round_ties_even() as u32).max(1);
    let (tw, th) = (side(w), side(h));
    let out = paddock_engine::pillow::resize_rgb8(
        rgb.as_raw(),
        w as usize,
        h as usize,
        tw as usize,
        th as usize,
        paddock_engine::pillow::Filter::Lanczos3,
    );
    image::RgbImage::from_raw(tw, th, out).expect("resize_rgb8 returns exactly tw * th * 3 bytes")
}

/// Is this checkpoint LightOnOCR-3 on the Qwen3.5 graph? `general.name`
/// ("LightOnOCR 3 4B") is checked too, as the field that survives a
/// re-conversion from a renamed directory.
pub fn is_checkpoint(arch: &str, g: &paddock_models::gguf::GgufFile) -> bool {
    let field = |k: &str| match g.metadata.get(k) {
        Some(paddock_models::gguf::Value::Str(s)) => Some(s.to_ascii_lowercase()),
        _ => None,
    };
    arch == "qwen35"
        && (field("general.basename").is_some_and(|b| b == BASENAME.to_ascii_lowercase())
            || field("general.name")
                .is_some_and(|n| n.starts_with("lightonocr 3") || n.starts_with("lightonocr-3")))
}

/// The community MLX config drops the fine-tune's identity. Require both
/// its validated small tied Qwen graph and an explicit LightOn package name,
/// never infer OCR from architecture/size alone. A quantization subfolder is
/// allowed; arbitrary ancestors are not (a Qwen model inside an OCR workspace
/// must remain a chat model).
pub fn is_hf_checkpoint(dir: &std::path::Path, cfg: &paddock_models::mlx::QwenConfig) -> bool {
    if !cfg.tied {
        return false;
    }
    let name = |p: &std::path::Path| {
        p.file_name()
            .and_then(|n| n.to_str())
            .map(str::to_ascii_lowercase)
    };
    let Some(mut basename) = name(dir) else {
        return false;
    };
    if basename == "4bit" {
        let Some(parent) = dir.parent().and_then(name) else {
            return false;
        };
        basename = parent;
    }
    let prefix = match cfg.width {
        1024 => "lightonocr-3-0.8b",
        2560 => "lightonocr-3-4b",
        _ => return false,
    };
    basename
        .strip_prefix(prefix)
        .is_some_and(|tail| tail.is_empty() || tail.starts_with('-'))
}

/// The two modes; the wire names are the client's own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoMode {
    Plain,
    Grounding,
}

impl LoMode {
    fn parse(s: &str) -> Result<LoMode, String> {
        match s {
            "plain" => Ok(LoMode::Plain),
            "grounding" => Ok(LoMode::Grounding),
            other => Err(format!(
                "invalid ocr.mode {other:?} (expected plain or grounding)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            LoMode::Plain => "plain",
            LoMode::Grounding => "grounding",
        }
    }

    /// The text the checkpoint conditions on: none for plain, the bare word
    /// for grounding.
    fn canonical(self) -> &'static str {
        match self {
            LoMode::Plain => "",
            LoMode::Grounding => GROUNDING_PROMPT,
        }
    }

    pub const ALL: [LoMode; 2] = [LoMode::Plain, LoMode::Grounding];
}

/// The capability object, the same shape every document parser publishes.
pub fn caps_json() -> Value {
    json!({
        "modes": LoMode::ALL.map(LoMode::as_str),
        "crops": [],
        "grounding": true,
    })
}

/// Parse one `ocr` object: `mode` is the whole interface.
pub fn parse_opts(v: &Value) -> Result<Option<LoMode>, String> {
    let Some(obj) = v.as_object() else {
        return Err("ocr must be a JSON object".into());
    };
    let mut mode = None;
    for (k, v) in obj {
        match k.as_str() {
            "mode" => {
                let s = v.as_str().ok_or("ocr.mode must be a string")?;
                mode = Some(LoMode::parse(s)?);
            }
            other => {
                return Err(format!(
                    "unknown ocr field {other:?} for this family (only `mode` - grounding is \
                     a mode here, and there are no crop classes)"
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
) -> Result<Option<LoMode>, String> {
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

/// Drop every text part of the last non-system message - plain mode's
/// prompt is the image alone.
fn clear_body_text(messages: &mut [Value]) {
    let Some(last) = messages
        .iter_mut()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) != Some("system"))
    else {
        return;
    };
    match last.get_mut("content") {
        Some(Value::Array(parts)) => {
            parts.retain(|p| p.get("type").and_then(Value::as_str) != Some("text"));
        }
        Some(Value::String(s)) => s.clear(),
        _ => {}
    }
}

/// Resolve one request. An image with no text is plain; text that is exactly
/// `grounding` is grounding as sent; an explicit mode replaces the caller's
/// text with its canonical form; anything else passes through.
pub fn resolve(
    messages: &mut [Value],
    mode: Option<LoMode>,
    pages: usize,
) -> Result<Option<OcrResolved>, String> {
    if pages == 0 {
        if mode.is_some() {
            return Err(
                "the ocr request object applies to image requests - this request has no image \
                 (attach the page as an image or PDF)"
                    .into(),
            );
        }
        return Ok(None);
    }
    let text = body_text(messages);
    let has_text = !text.trim().is_empty();
    let resolved_mode = match mode {
        Some(m) => Some(m),
        None if !has_text => Some(LoMode::Plain),
        None if text == GROUNDING_PROMPT => Some(LoMode::Grounding),
        None => None,
    };
    let pass_through = resolved_mode.is_none();
    if pass_through {
        tracing::warn!(
            "LightOnOCR-3 was trained on the empty prompt and `grounding` only - other text \
             passes through verbatim but is outside its training distribution"
        );
    }

    let mut dropped_text = false;
    if let Some(m) = mode {
        dropped_text = has_text && text != m.canonical();
        if dropped_text {
            tracing::warn!(
                mode = m.as_str(),
                "explicit ocr.mode replaced the request's own text - echoed as dropped_text"
            );
        }
        match m {
            LoMode::Plain => clear_body_text(messages),
            LoMode::Grounding => set_body_text(messages, GROUNDING_PROMPT, true),
        }
    }
    if pages > 1 {
        tracing::warn!(
            pages,
            "LightOnOCR-3 reads one page per request - send each page on its own for \
             per-page boxes and the trained conditioning"
        );
    }

    let resolved = OcrResolved {
        mode: resolved_mode.map(LoMode::as_str),
        crop: "base",
        force_base: false,
        grounding: resolved_mode == Some(LoMode::Grounding),
        pages,
        views: pages,
        tiles: 0,
        image_tokens: 0,
        pass_through,
        dropped_text,
        ngram: (0, 0),
        repeat_stop: false,
    };
    tracing::info!(
        mode = resolved.mode.unwrap_or("pass-through"),
        pages,
        pass_through,
        dropped_text,
        "lightonocr request resolved"
    );
    Ok(Some(resolved))
}

/// One `![label](x1,y1,x2,y2)` marker at the head of `s`: (label, continues,
/// box, bytes consumed). The grammar is LightOn's `MARKER` regex written out,
/// `!\[([A-Za-z_][\w-]*?)(\+?)\]\(\s*(\d+)\s*,...\s*\)[ \t]*`, so the
/// spaces and tabs after the closing bracket belong to the marker, and the
/// block text starts after them.
fn marker(s: &str) -> Option<(&str, bool, [i64; 4], usize)> {
    let b = s.as_bytes();
    if !s.starts_with("![") {
        return None;
    }
    let label_start = 2;
    let first = s[label_start..].chars().next()?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    let mut i = label_start + first.len_utf8();
    while let Some(c) = s[i..].chars().next() {
        if c.is_alphanumeric() || c == '_' || c == '-' {
            i += c.len_utf8();
        } else {
            break;
        }
    }
    let label = &s[label_start..i];
    let continues = b.get(i) == Some(&b'+');
    if continues {
        i += 1;
    }
    if b.get(i) != Some(&b']') || b.get(i + 1) != Some(&b'(') {
        return None;
    }
    i += 2;
    let skip_ws = |i: &mut usize| {
        while let Some(c) = s[*i..].chars().next().filter(|c| c.is_whitespace()) {
            *i += c.len_utf8();
        }
    };
    let mut v = [0i64; 4];
    for (n, slot) in v.iter_mut().enumerate() {
        skip_ws(&mut i);
        let d0 = i;
        while b.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        *slot = s[d0..i].parse().ok()?;
        skip_ws(&mut i);
        let close = if n == 3 { b')' } else { b',' };
        if b.get(i) != Some(&close) {
            return None;
        }
        i += 1;
    }
    while matches!(b.get(i), Some(b' ' | b'\t')) {
        i += 1;
    }
    Some((label, continues, v, i))
}

/// Parse grounding output into regions, LightOn's `parse_blocks` exactly:
/// each marker owns the text up to the next one (a table or list spans
/// lines, and the text may start on the line after the marker), stripped.
/// A `+` on the label marks a block continuing the previous one - a
/// paragraph flowing into the next column - and rides as `continues`.
///
/// Boxes are rescaled from the checkpoint's 0..=1000 grid onto the wire's
/// 0..=999 (deepseek's), the same rescale paddleocr's spotting does, so
/// `regions` reads identically across families. A partial marker at a
/// mid-stream tail does not match, so the parse is safe on a cut stream.
pub fn parse_blocks(raw: &str) -> Vec<Region> {
    let mut marks = Vec::new();
    let mut at = 0;
    while let Some(off) = raw[at..].find("![") {
        let start = at + off;
        match marker(&raw[start..]) {
            Some((label, continues, v, len)) => {
                marks.push((start, start + len, label, continues, v));
                at = start + len;
            }
            None => at = start + 1,
        }
    }
    let s = |v: i64| (v.clamp(0, 1000) * 999 + 500) / 1000;
    marks
        .iter()
        .enumerate()
        .map(|(n, &(_, end, label, continues, v))| {
            let stop = marks.get(n + 1).map_or(raw.len(), |m| m.0);
            let text = raw[end..stop].trim();
            Region {
                label: label.to_owned(),
                boxes: vec![[s(v[0]), s(v[1]), s(v[2]), s(v[3])]],
                text: (!text.is_empty()).then(|| text.to_owned()),
                quads: Vec::new(),
                continues,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_identity_requires_the_right_package_and_geometry() {
        let mut cfg = paddock_models::mlx::QwenConfig {
            width: 1024,
            ff: 3584,
            layers: 24,
            heads: 8,
            kv_heads: 2,
            value_heads: 16,
            tied: true,
            context: 262144,
            eps: 1e-6,
            rope: 10_000_000.,
        };
        for path in [
            "/models/LightOnOCR-3-0.8B-MLX/4bit",
            "/models/LightOnOCR-3-0.8B-MLX-4bit",
        ] {
            assert!(is_hf_checkpoint(std::path::Path::new(path), &cfg));
        }
        for path in [
            "/models/LightOnOCR-3-4B-MLX/4bit",
            "/models/qwen3.5-0.8b",
            "/LightOnOCR-3-0.8B-MLX/qwen/4bit",
            "/models/LightOnOCR-3-0.8Bigger",
        ] {
            assert!(!is_hf_checkpoint(std::path::Path::new(path), &cfg));
        }
        cfg.tied = false;
        assert!(!is_hf_checkpoint(
            std::path::Path::new("LightOnOCR-3-0.8B-MLX"),
            &cfg
        ));
    }

    fn user(parts: Value) -> Vec<Value> {
        vec![json!({"role": "user", "content": parts})]
    }

    fn texts(msgs: &[Value]) -> Vec<String> {
        msgs[0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["type"] == "text")
            .map(|p| p["text"].as_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn page_fit_matches_the_client() {
        let img = |w, h| image::RgbImage::from_pixel(w, h, image::Rgb([200, 30, 90]));
        // never upscales, and an image already at the edge is untouched
        assert_eq!(
            fit_page(img(1200, 900), PAGE_EDGE).dimensions(),
            (1200, 900)
        );
        assert_eq!(
            fit_page(img(2048, 1448), PAGE_EDGE).dimensions(),
            (2048, 1448)
        );
        // a photo: 4000 x 3000 -> 2048 x 1536
        assert_eq!(
            fit_page(img(4000, 3000), PAGE_EDGE).dimensions(),
            (2048, 1536)
        );
        // Python's round is half-to-even: 1025 * 0.5 = 512.5 -> 512, not 513
        assert_eq!(
            fit_page(img(4096, 1025), PAGE_EDGE).dimensions(),
            (2048, 512)
        );
        assert_eq!(
            fit_page(img(1027, 4096), PAGE_EDGE).dimensions(),
            (514, 2048)
        );
        // a thin strip keeps one row
        assert_eq!(fit_page(img(8192, 1), PAGE_EDGE).dimensions(), (2048, 1));
        // a flat colour stays flat through the Lanczos taps
        assert!(
            fit_page(img(3000, 2000), PAGE_EDGE)
                .pixels()
                .all(|p| p.0 == [200, 30, 90])
        );
    }

    #[test]
    fn an_image_alone_is_plain_and_untouched() {
        let mut msgs = user(json!([{"type": "image"}]));
        let before = msgs.clone();
        let r = resolve(&mut msgs, None, 1).unwrap().unwrap();
        assert_eq!(r.mode, Some("plain"));
        assert!(!r.grounding && !r.pass_through && !r.dropped_text);
        assert_eq!(msgs, before, "plain's prompt is the image alone");
    }

    #[test]
    fn the_bare_word_is_grounding_as_sent() {
        let mut msgs = user(json!([{"type": "image"}, {"type": "text", "text": "grounding"}]));
        let before = msgs.clone();
        let r = resolve(&mut msgs, None, 1).unwrap().unwrap();
        assert_eq!(r.mode, Some("grounding"));
        assert!(r.grounding && !r.pass_through);
        assert_eq!(msgs, before);
    }

    #[test]
    fn explicit_grounding_replaces_text_after_the_image() {
        let mut msgs = user(json!([{"type": "text", "text": "read this"}, {"type": "image"}]));
        let r = resolve(&mut msgs, Some(LoMode::Grounding), 1)
            .unwrap()
            .unwrap();
        assert!(r.grounding && r.dropped_text);
        assert_eq!(texts(&msgs), ["grounding"]);
        // the reference order: image part first, the word after it
        assert_eq!(msgs[0]["content"][0]["type"], "image");
    }

    #[test]
    fn explicit_plain_clears_text() {
        let mut msgs = user(json!([{"type": "image"}, {"type": "text", "text": "grounding"}]));
        let r = resolve(&mut msgs, Some(LoMode::Plain), 1).unwrap().unwrap();
        assert_eq!(r.mode, Some("plain"));
        assert!(r.dropped_text && !r.grounding);
        assert!(texts(&msgs).is_empty());
    }

    #[test]
    fn the_same_word_sent_twice_is_not_a_drop() {
        let mut msgs = user(json!([{"type": "image"}, {"type": "text", "text": "grounding"}]));
        let r = resolve(&mut msgs, Some(LoMode::Grounding), 1)
            .unwrap()
            .unwrap();
        assert!(!r.dropped_text);
        assert_eq!(texts(&msgs), ["grounding"]);
    }

    #[test]
    fn other_text_passes_through() {
        let mut msgs = user(json!([{"type": "image"}, {"type": "text", "text": "Grounding"}]));
        let before = msgs.clone();
        let r = resolve(&mut msgs, None, 1).unwrap().unwrap();
        assert_eq!(r.mode, None);
        assert!(r.pass_through && !r.grounding);
        assert_eq!(msgs, before);
    }

    #[test]
    fn unknown_fields_and_text_only_are_refused() {
        assert!(parse_opts(&json!({"mode": "plain", "crop": "base"})).is_err());
        assert!(parse_opts(&json!({"mode": "document"})).is_err());
        assert_eq!(
            parse_opts(&json!({"mode": "grounding"})).unwrap(),
            Some(LoMode::Grounding)
        );
        let mut msgs = user(json!([{"type": "text", "text": "hi"}]));
        assert!(resolve(&mut msgs, Some(LoMode::Plain), 0).is_err());
        assert!(resolve(&mut msgs, None, 0).unwrap().is_none());
    }

    // grounding output in the cards' own shape: a title, a paragraph whose
    // text starts on the next line, a continuation and a chart with a table
    const GROUNDED: &str = "![title](102,61,898,95) Quarterly Report\n\
        ![text](102,120,480,410)\nRevenue grew in every region.\n\
        ![text+](520,120,898,300) The north led.\n\
        ![chart](102,450,898,800) <table><tr><td>Q1</td><td>4.2</td></tr></table>\n\
        ![page_number](480,960,520,980) 3";

    #[test]
    fn grounding_blocks_parse_like_the_reference() {
        let rs = parse_blocks(GROUNDED);
        let labels: Vec<_> = rs.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["title", "text", "text", "chart", "page_number"]);
        assert_eq!(rs[0].text.as_deref(), Some("Quarterly Report"));
        assert_eq!(rs[1].text.as_deref(), Some("Revenue grew in every region."));
        assert!(rs[2].continues && !rs[1].continues);
        assert!(rs[3].text.as_deref().unwrap().starts_with("<table>"));
        // 0..=1000 onto 0..=999: 102 -> 102, 898 -> 897, 1000 -> 999
        assert_eq!(rs[0].boxes, vec![[102, 61, 897, 95]]);
        assert_eq!(
            parse_blocks("![image](0,0,1000,1000)")[0].boxes,
            vec![[0, 0, 999, 999]]
        );
        // ...and the shared entry point reaches it
        assert!(crate::deepseek_ocr::regions_json(GROUNDED).is_some());
    }

    #[test]
    fn spaces_inside_the_box_and_empty_blocks() {
        let rs = parse_blocks("![image]( 10 , 20,30 ,40 )\t\n![caption](10,45,30,50) Fig. 1");
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[0].boxes, vec![[10, 20, 30, 40]]);
        assert_eq!(rs[0].text, None, "an undescribed image has no text");
        assert_eq!(rs[1].text.as_deref(), Some("Fig. 1"));
    }

    #[test]
    fn markdown_images_and_cut_markers_are_not_regions() {
        assert!(parse_blocks("see ![logo](logo.png) here").is_empty());
        assert!(
            parse_blocks("![text](10,20,30").is_empty(),
            "mid-stream tail"
        );
        assert!(
            parse_blocks("![9lives](1,2,3,4)").is_empty(),
            "a label starts with a letter or underscore"
        );
        assert!(parse_blocks("plain transcript, no markers").is_empty());
        // a bad marker does not swallow the good one after it
        let rs = parse_blocks("![x](1,2) junk ![text](1,2,3,4) ok");
        assert_eq!(rs.len(), 1);
        assert_eq!(rs[0].text.as_deref(), Some("ok"));
    }
}
