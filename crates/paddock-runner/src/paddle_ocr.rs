//! PaddleOCR-VL instruction mapping: the family's `ocr` request
//! object - its six official task prompts as an advertised vocabulary, with
//! the same four-layer precedence deepseek_ocr established (pass-through /
//! explicit mode / derived default / no intent sniffing).
//!
//! This family is SIMPLER than deepseek's: no crop classes, no grounding
//! token, no sampling default - its whole interface is which task prompt
//! precedes the decode. The canonical strings are byte-exact to the
//! checkpoint's own usage (the template fixtures encode id-exact against the
//! reference processor, tests/paddleocr_template.rs), because the
//! conditioning is measured against the reference and "close" is a
//! different prompt.
//!
//! The resolved outcome reuses `deepseek_ocr::OcrResolved` - one echo shape
//! on the wire for every document parser, so a client reads `response.ocr`
//! without knowing the family.

use serde_json::{Value, json};

use crate::deepseek_ocr::{OcrResolved, Region, body_text, set_body_text};

/// The six official tasks, plus the authors' document pipeline. Wire names
/// are ours (short, stable); canonical strings are the checkpoint's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PoMode {
    /// the page through PP-DocLayoutV3 first, each region read with its
    /// label's task (`crate::paddle_layout`) - served where the layout
    /// companion is installed
    Document,
    Ocr,
    Table,
    Formula,
    Chart,
    Spotting,
    Seal,
}

impl PoMode {
    fn parse(s: &str) -> Result<PoMode, String> {
        Ok(match s {
            "document" => PoMode::Document,
            "ocr" => PoMode::Ocr,
            "table" => PoMode::Table,
            "formula" => PoMode::Formula,
            "chart" => PoMode::Chart,
            "spotting" => PoMode::Spotting,
            "seal" => PoMode::Seal,
            other => {
                return Err(format!(
                    "invalid ocr.mode {other:?} (expected one of document, ocr, table, formula, \
                     chart, spotting, seal)"
                ));
            }
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PoMode::Document => "document",
            PoMode::Ocr => "ocr",
            PoMode::Table => "table",
            PoMode::Formula => "formula",
            PoMode::Chart => "chart",
            PoMode::Spotting => "spotting",
            PoMode::Seal => "seal",
        }
    }

    /// The checkpoint's own task prompt for this mode (document mode's
    /// regions carry their own; its page prompt is never rendered).
    fn canonical(self) -> &'static str {
        match self {
            PoMode::Document | PoMode::Ocr => "OCR:",
            PoMode::Table => "Table Recognition:",
            PoMode::Formula => "Formula Recognition:",
            PoMode::Chart => "Chart Recognition:",
            PoMode::Spotting => "Spotting:",
            PoMode::Seal => "Seal Recognition:",
        }
    }

    pub const ALL: [PoMode; 6] = [
        PoMode::Ocr,
        PoMode::Table,
        PoMode::Formula,
        PoMode::Chart,
        PoMode::Spotting,
        PoMode::Seal,
    ];
}

/// The capability object - same shape as deepseek's so one client reads
/// both; this family has no crop classes and no grounded-region parse.
/// `document` is advertised only where the layout companion is loaded.
pub fn caps_json(document: bool) -> Value {
    let mut modes: Vec<&str> = Vec::new();
    if document {
        modes.push(PoMode::Document.as_str());
    }
    modes.extend(PoMode::ALL.map(PoMode::as_str));
    json!({
        "modes": modes,
        "crops": [],
        "grounding": false,
    })
}

/// The repetition stop's n-gram and window: the DeepSeek-OCR reference's
/// guard sizes (35 within 128), here ending the read instead of banning.
const REPEAT_STOP: (usize, usize) = (35, 128);

/// What one request asks of this family.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct PoOpts {
    pub mode: Option<PoMode>,
    /// `ocr.repetition_stop` - None keeps the default (armed).
    pub repetition_stop: Option<bool>,
}

/// Parse one `ocr` object for this family: `mode`, and `repetition_stop`
/// (on by default) - its interface has no crops, grounding or sampling
/// knobs, and accepting them silently would advertise behaviour that does
/// not exist.
pub fn parse_opts(v: &Value) -> Result<PoOpts, String> {
    let Some(obj) = v.as_object() else {
        return Err("ocr must be a JSON object".into());
    };
    let mut opts = PoOpts::default();
    for (k, v) in obj {
        match k.as_str() {
            "mode" => {
                let s = v.as_str().ok_or("ocr.mode must be a string")?;
                opts.mode = Some(PoMode::parse(s)?);
            }
            "repetition_stop" => {
                opts.repetition_stop =
                    Some(v.as_bool().ok_or("ocr.repetition_stop must be a boolean")?);
            }
            other => {
                return Err(format!(
                    "unknown ocr field {other:?} for this family (only `mode` and \
                     `repetition_stop` - it has no crop classes, grounding or sampling knobs)"
                ));
            }
        }
    }
    Ok(opts)
}

/// The two accepted channels, same precedence as deepseek's from_request.
pub fn opts_from_request(top: Option<&Value>, kwargs: Option<&Value>) -> Result<PoOpts, String> {
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
        (None, None) => Ok(PoOpts::default()),
    }
}

/// Resolve one request: pass-through text stands, an explicit mode's
/// canonical prompt wins (dropped text echoed, never silent), and an empty
/// body defaults to the document pipeline where it is served (`document`:
/// Ok), else to `OCR:` - the checkpoint is task-prompt-conditioned, so a
/// bare image without any prompt is off its training distribution, and its
/// authors read pages region by region, not whole. `document` carries why
/// the pipeline is not served here when it is not; asking for it then is a
/// 400 that says so.
///
/// The repetition stop is armed unless the request turns it off. Greedy
/// decoding of a whole page can fall into a loop that runs to the token cap
/// (the checkpoint ships no repetition control: its own pipeline reads
/// layout-detected regions, not pages); the stop ends the read at the
/// loop's first repeated 35-token n-gram, so everything before it is the
/// unguarded output byte for byte, and a page that never loops is untouched.
pub fn resolve(
    messages: &mut [Value],
    opts: PoOpts,
    pages: usize,
    document: Result<(), &str>,
) -> Result<Option<OcrResolved>, String> {
    let (mode, stop) = (opts.mode, opts.repetition_stop.unwrap_or(true));
    if mode == Some(PoMode::Document)
        && let Err(why) = document
    {
        return Err(format!("ocr.mode \"document\" is not served here: {why}"));
    }
    if pages == 0 {
        if opts != PoOpts::default() {
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
    let pass_through = has_text && mode.is_none();

    let resolved_mode = match mode {
        Some(m) => Some(m),
        None if !has_text && document.is_ok() => Some(PoMode::Document),
        None if !has_text => Some(PoMode::Ocr),
        None => None,
    };

    let mut dropped_text = false;
    if let Some(m) = resolved_mode {
        if mode.is_some() && has_text {
            dropped_text = true;
            tracing::warn!(
                mode = m.as_str(),
                "explicit ocr.mode replaced the request's own text - echoed as dropped_text"
            );
        }
        // the document pipeline renders each region's own prompt
        if m != PoMode::Document {
            set_body_text(messages, m.canonical(), mode.is_some());
        }
    }

    let resolved = OcrResolved {
        mode: resolved_mode.map(PoMode::as_str),
        // no crop concept: every page is read whole
        crop: "base",
        force_base: false,
        grounding: false,
        pages,
        views: pages,
        tiles: 0,
        image_tokens: 0,
        pass_through,
        dropped_text,
        // off is off - never the ban the same sizes would arm
        ngram: if stop { REPEAT_STOP } else { (0, 0) },
        repeat_stop: stop,
    };
    tracing::info!(
        mode = resolved.mode.unwrap_or("pass-through"),
        pages,
        pass_through,
        dropped_text,
        "paddleocr request resolved"
    );
    Ok(Some(resolved))
}

/// Parse Spotting output into regions. The format was PROBED on the live
/// checkpoint (greedy, two images with text at known pixel
/// positions), never guessed:
///
/// ```text
/// The quick brown fox<|LOC_59|><|LOC_69|><|LOC_488|><|LOC_69|><|LOC_488|><|LOC_99|><|LOC_59|><|LOC_99|>\n
/// ```
///
/// One line per detected text INSTANCE (line-level: a wrapped paragraph came
/// back as three instances): the text, then EIGHT `<|LOC_n|>` tokens - a
/// 4-corner quadrilateral (TL, TR, BR, BL), each coordinate an integer on a
/// 0..=1000 grid normalized per axis (verified against a non-square image:
/// y=1000px of 1200 -> LOC_836). The tokenizer also ships
/// `<|LOC_BEGIN|>`/`<|LOC_END|>`/`<|LOC_SEP|>` structural tokens that neither
/// probe triggered; they are tolerated as delimiters so an output that does
/// use them still parses.
///
/// Coordinates are rescaled onto the wire's 0-999 space (deepseek's grid) so
/// `regions` reads identically across families; each quad also flattens to
/// its axis-aligned hull in `boxes` for clients that only draw rectangles.
/// A trailing run of fewer than 8 values is dropped, not guessed - that also
/// makes the parse safe to run mid-stream on a partial tail.
pub fn parse_spotting(raw: &str) -> Vec<Region> {
    if !raw.contains("<|LOC_") {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cur = raw;
    while let Some(i) = cur.find("<|LOC_") {
        let text = cur[..i].trim();
        // consume the whole marker cluster: numeric LOC tokens + structural
        // separators, nothing else
        let mut rest = &cur[i..];
        let mut vals: Vec<i64> = Vec::new();
        loop {
            if let Some(r) = rest
                .strip_prefix("<|LOC_BEGIN|>")
                .or_else(|| rest.strip_prefix("<|LOC_END|>"))
                .or_else(|| rest.strip_prefix("<|LOC_SEP|>"))
            {
                rest = r;
                continue;
            }
            let Some(body) = rest.strip_prefix("<|LOC_") else {
                break;
            };
            let Some(e) = body.find("|>") else { break };
            let Ok(n) = body[..e].parse::<i64>() else {
                break;
            };
            vals.push(n);
            rest = &body[e + 2..];
        }
        cur = rest;
        if text.is_empty() {
            continue;
        }
        let mut quads = Vec::new();
        let mut boxes = Vec::new();
        for q in vals.as_chunks::<8>().0 {
            // native 0..=1000 grid -> the wire's 0..=999
            let s = |v: i64| (v.clamp(0, 1000) * 999 + 500) / 1000;
            let quad = [
                s(q[0]),
                s(q[1]),
                s(q[2]),
                s(q[3]),
                s(q[4]),
                s(q[5]),
                s(q[6]),
                s(q[7]),
            ];
            let xs = [quad[0], quad[2], quad[4], quad[6]];
            let ys = [quad[1], quad[3], quad[5], quad[7]];
            boxes.push([
                xs[0].min(xs[1]).min(xs[2]).min(xs[3]),
                ys[0].min(ys[1]).min(ys[2]).min(ys[3]),
                xs[0].max(xs[1]).max(xs[2]).max(xs[3]),
                ys[0].max(ys[1]).max(ys[2]).max(ys[3]),
            ]);
            quads.push(quad);
        }
        if !boxes.is_empty() {
            out.push(Region {
                label: "text".to_owned(),
                boxes,
                text: Some(text.to_owned()),
                quads,
                continues: false,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_DOC: Result<(), &str> = Err("no layout model");

    #[test]
    fn a_bare_page_reads_as_a_document_where_the_pipeline_serves() {
        let mut msgs = user("");
        let r = resolve(&mut msgs, PoOpts::default(), 1, Ok(()))
            .unwrap()
            .unwrap();
        assert_eq!(r.mode, Some("document"));
        // the page prompt is never rendered: the regions carry their own
        assert!(!body_text(&msgs).contains("OCR:"));
        // text still passes through, and the other modes still resolve
        let mut msgs = user("Table Recognition:");
        let r = resolve(&mut msgs, PoOpts::default(), 1, Ok(()))
            .unwrap()
            .unwrap();
        assert!(r.pass_through);
        // asked for where it is not served: a 400 naming why
        let opts = parse_opts(&json!({"mode": "document"})).unwrap();
        let e = resolve(&mut user(""), opts, 1, NO_DOC).unwrap_err();
        assert!(e.contains("no layout model"), "{e}");
        assert_eq!(caps_json(false)["modes"][0], "ocr");
        assert_eq!(caps_json(true)["modes"][0], "document");
    }

    fn user(text: &str) -> Vec<Value> {
        vec![json!({"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]})]
    }

    #[test]
    fn empty_body_defaults_to_ocr() {
        let mut msgs = user("");
        let r = resolve(&mut msgs, PoOpts::default(), 1, NO_DOC)
            .unwrap()
            .unwrap();
        assert_eq!(r.mode, Some("ocr"));
        assert!(!r.pass_through && !r.dropped_text);
        assert!(body_text(&msgs).contains("OCR:"));
    }

    #[test]
    fn pass_through_text_stands() {
        let mut msgs = user("Table Recognition:");
        let r = resolve(&mut msgs, PoOpts::default(), 1, NO_DOC)
            .unwrap()
            .unwrap();
        assert_eq!(r.mode, None);
        assert!(r.pass_through);
        assert_eq!(body_text(&msgs).trim(), "Table Recognition:");
    }

    #[test]
    fn explicit_mode_wins_and_echoes_drop() {
        let mut msgs = user("make a markdown");
        let r = resolve(
            &mut msgs,
            PoOpts {
                mode: Some(PoMode::Table),
                ..PoOpts::default()
            },
            1,
            NO_DOC,
        )
        .unwrap()
        .unwrap();
        assert_eq!(r.mode, Some("table"));
        assert!(r.dropped_text);
        assert_eq!(body_text(&msgs).trim(), "Table Recognition:");
    }

    #[test]
    fn unknown_field_is_a_loud_400() {
        assert!(parse_opts(&json!({"mode": "ocr", "crop": "base"})).is_err());
        assert!(parse_opts(&json!({"grounding": true})).is_err());
        assert!(parse_opts(&json!({"mode": "ocr"})).unwrap().mode.is_some());
    }

    #[test]
    fn text_only_with_ocr_object_is_refused() {
        let mut msgs = user("hello");
        assert!(
            resolve(
                &mut msgs,
                PoOpts {
                    mode: Some(PoMode::Ocr),
                    ..PoOpts::default()
                },
                0,
                NO_DOC,
            )
            .is_err()
        );
    }

    // live-probe fixture: greedy spotting output on a 1000×1400
    // synthetic page with text drawn at known pixel positions
    const SPOT: &str = "INVOICE 2026-001<|LOC_81|><|LOC_76|><|LOC_626|><|LOC_76|><|LOC_626|><|LOC_115|><|LOC_81|><|LOC_115|>\nTotal: 1234,56 kr<|LOC_78|><|LOC_289|><|LOC_390|><|LOC_289|><|LOC_390|><|LOC_317|><|LOC_78|><|LOC_317|>\nPADDOCK<|LOC_599|><|LOC_896|><|LOC_801|><|LOC_896|><|LOC_801|><|LOC_921|><|LOC_599|><|LOC_921|>";

    #[test]
    fn the_repetition_stop_is_armed_by_default_and_turns_off_whole() {
        use paddock_engine::service::FinishReason;
        let mut msgs = user("");
        let r = resolve(&mut msgs, PoOpts::default(), 1, NO_DOC)
            .unwrap()
            .unwrap();
        assert_eq!(r.guard(), (35, 128, true), "a stop, never the ban");
        let echo = r.echo_at(Some(FinishReason::Repetition));
        assert_eq!(echo["repetition_stop"]["fired"], true);
        assert_eq!(echo["no_repeat_ngram"]["size"], 0, "nothing is masked");
        assert_eq!(
            r.echo_at(Some(FinishReason::Stop))["repetition_stop"]["fired"],
            false
        );

        let opts = parse_opts(&json!({"repetition_stop": false})).unwrap();
        let mut msgs = user("");
        let r = resolve(&mut msgs, opts, 1, NO_DOC).unwrap().unwrap();
        assert_eq!(r.guard(), (0, 0, false), "off is off, not the ban");
        assert!(r.echo_at(None).get("repetition_stop").is_none());
        assert!(parse_opts(&json!({"repetition_stop": "no"})).is_err());
    }

    #[test]
    fn spotting_lines_parse_to_text_regions() {
        let rs = parse_spotting(SPOT);
        assert_eq!(rs.len(), 3);
        assert_eq!(rs[0].text.as_deref(), Some("INVOICE 2026-001"));
        assert_eq!(rs[2].text.as_deref(), Some("PADDOCK"));
        // native 0..=1000 rescaled onto the wire's 0..=999: 81 -> 81, 896 -> 895
        assert_eq!(rs[0].boxes, vec![[81, 76, 625, 115]]);
        assert_eq!(rs[2].boxes, vec![[598, 895, 800, 920]]);
        // the quad rides along, TL/TR/BR/BL order preserved
        assert_eq!(rs[0].quads, vec![[81, 76, 625, 76, 625, 115, 81, 115]]);
        // and the shared entry point reaches it (deepseek markup absent)
        assert!(crate::deepseek_ocr::regions_json(SPOT).is_some());
    }

    #[test]
    fn spotting_partial_tail_is_dropped_not_guessed() {
        // a mid-stream cut: 5 of 8 LOC tokens - no region until the run closes
        let cut = "Header<|LOC_10|><|LOC_20|><|LOC_30|><|LOC_20|><|LOC_30|>";
        assert!(parse_spotting(cut).is_empty());
        // structural separators are tolerated as delimiters
        let sep = "A<|LOC_BEGIN|><|LOC_1|><|LOC_2|><|LOC_3|><|LOC_2|><|LOC_3|><|LOC_4|><|LOC_1|><|LOC_4|><|LOC_END|>";
        let rs = parse_spotting(sep);
        assert_eq!(rs.len(), 1);
        assert_eq!(rs[0].text.as_deref(), Some("A"));
        // plain text never conjures regions
        assert!(parse_spotting("no markers here").is_empty());
    }
}
