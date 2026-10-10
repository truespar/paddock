//! DeepSeek-OCR family instruction mapping: the `ocr` request
//! object, prompt-vocabulary resolution, the family's sampling default, and
//! the grounding-region parse.
//!
//! Four layers, strict precedence:
//!
//! 1. **Pass-through** - text the caller wrote is used verbatim (the
//!    conformance floor: someone following the official README or another
//!    engine's recipe gets byte-identical conditioning, which is what the parity gate
//!    measures).
//! 2. **The `ocr` request object** - the honest knob. Accepted at the TOP
//!    LEVEL of all three APIs (the documented form) and inside
//!    `chat_template_kwargs.ocr` where that channel exists (what an
//!    unmodified OpenAI SDK reaches via `extra_body`). An explicit
//!    `ocr.mode` wins: the canonical task string is used, and any text the
//!    caller also sent is reported dropped in the echo - never silently.
//! 3. **Derived defaults** - no `ocr` object and no text: one image parses
//!    as a document (gundam crop), several as multi-page (base). Derived
//!    from the request shape, never sniffed from prose.
//! 4. Natural-language intent detection is explicitly not done.
//!
//! Everything the server decides is echoed back in an `ocr` extension field
//! on the response (and logged), so a bench artifact can stamp the serve
//! configuration from server state rather than intent.

use paddock_engine::service::FinishReason;
use serde_json::{Value, json};

/// The family's five task modes. The canonical strings are BYTE-EXACT to the
/// checkpoint (README + the `infer()` conversation comments in
/// `modeling_unlimitedocr.py`) - trailing spaces and leading newlines
/// included, because the conditioning is measured against the reference and
/// "close" is a different prompt.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OcrMode {
    /// `<image>document parsing.` - the headline mode: structured document
    /// text,每 block prefixed with a `<|det|>type [bbox]<|/det|>` record.
    Document,
    /// `<image>Multi page parsing.` - multi-page / PDF, base sizing only.
    Multipage,
    /// `<image>\nFree OCR. ` - plain text, no structure.
    Free,
    /// `<image>\n<|grounding|>Given the layout of the image. ` - layout with
    /// `<|ref|>label<|/ref|><|det|>[[boxes]]<|/det|>` regions.
    Layout,
    /// `<image>\nParse the figure. ` - figure/chart parsing.
    Figure,
}

impl OcrMode {
    fn parse(s: &str) -> Result<OcrMode, String> {
        Ok(match s {
            "document" => OcrMode::Document,
            "multipage" => OcrMode::Multipage,
            "free" => OcrMode::Free,
            "layout" => OcrMode::Layout,
            "figure" => OcrMode::Figure,
            other => {
                return Err(format!(
                    "invalid ocr.mode {other:?} (expected one of document, multipage, free, \
                     layout, figure)"
                ));
            }
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            OcrMode::Document => "document",
            OcrMode::Multipage => "multipage",
            OcrMode::Free => "free",
            OcrMode::Layout => "layout",
            OcrMode::Figure => "figure",
        }
    }

    /// The canonical task string - what follows the `<image>` marker run.
    fn canonical(self) -> &'static str {
        match self {
            OcrMode::Document => "document parsing.",
            OcrMode::Multipage => "Multi page parsing.",
            OcrMode::Free => "\nFree OCR. ",
            OcrMode::Layout => "\n<|grounding|>Given the layout of the image. ",
            OcrMode::Figure => "\nParse the figure. ",
        }
    }

    /// Every mode, in the vocabulary's documented order - the advertised list
    /// and the `parse` vocabulary come from the same enum, so they cannot
    /// drift apart.
    pub const ALL: [OcrMode; 5] = [
        OcrMode::Document,
        OcrMode::Multipage,
        OcrMode::Free,
        OcrMode::Layout,
        OcrMode::Figure,
    ];
}

/// The `ocr` request surface as a capability object, advertised on
/// `/api/server` and in the `/v1/models` capabilities when this family
/// serves. The mode names are the model's interface (same reasoning as
/// `task_tags`): `supported_parameters` can say an `ocr` object exists, but
/// only a listed vocabulary lets a client offer the modes as controls
/// instead of leaving them discoverable by 400.
pub fn caps_json() -> Value {
    json!({
        "modes": OcrMode::ALL.map(OcrMode::as_str),
        // OcrOpts::parse's crop vocabulary ("auto" derives from the request
        // shape) - kept in step by the round-trip test below
        "crops": ["auto", "gundam", "base"],
        // grounded decodes append the parsed `ocr.regions` extension
        // (0-999-normalized [x1,y1,x2,y2] boxes) to the terminal response
        "grounding": true,
    })
}

/// `ocr.crop` - the reference's `crop_mode`, SGLang's per-request
/// `images_config.image_mode`. `Auto` derives from the request shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OcrCrop {
    #[default]
    Auto,
    Gundam,
    Base,
}

/// The parsed `ocr` request object, every field optional.
#[derive(Default, Debug, Clone)]
pub struct OcrOpts {
    pub mode: Option<OcrMode>,
    /// `true` additionally arms the grounding-region parse; when the resolved
    /// text lacks the `<|grounding|>` token it is prepended (the family's
    /// grounded-prompt shape is `<image>\n<|grounding|>TASK`).
    pub grounding: Option<bool>,
    pub crop: OcrCrop,
    /// `no_repeat_ngram_size` - the reference's own kwarg name. 0 disables.
    pub ngram_size: Option<usize>,
    /// `ngram_window` - 0 disables.
    pub ngram_window: Option<usize>,
}

impl OcrOpts {
    /// Parse one `ocr` object. Unknown keys are a loud 400, matching the
    /// wire structs' deny_unknown_fields stance - a typo'd knob must never
    /// silently serve default behaviour.
    pub fn parse(v: &Value) -> Result<OcrOpts, String> {
        let Some(obj) = v.as_object() else {
            return Err("ocr must be a JSON object".into());
        };
        let mut o = OcrOpts::default();
        for (k, v) in obj {
            match k.as_str() {
                "mode" => {
                    let s = v.as_str().ok_or("ocr.mode must be a string")?;
                    o.mode = Some(OcrMode::parse(s)?);
                }
                "grounding" => {
                    o.grounding = Some(v.as_bool().ok_or("ocr.grounding must be a boolean")?);
                }
                "crop" => {
                    o.crop = match v.as_str().ok_or("ocr.crop must be a string")? {
                        "auto" => OcrCrop::Auto,
                        "gundam" => OcrCrop::Gundam,
                        "base" => OcrCrop::Base,
                        other => {
                            return Err(format!(
                                "invalid ocr.crop {other:?} (expected auto, gundam or base)"
                            ));
                        }
                    };
                }
                "no_repeat_ngram_size" => {
                    o.ngram_size = Some(usize_field(v, "ocr.no_repeat_ngram_size")?);
                }
                "ngram_window" => {
                    o.ngram_window = Some(usize_field(v, "ocr.ngram_window")?);
                }
                other => {
                    return Err(format!(
                        "unknown ocr field {other:?} (expected mode, grounding, crop, \
                         no_repeat_ngram_size, ngram_window)"
                    ));
                }
            }
        }
        Ok(o)
    }

    /// The two accepted channels. The top-level field is the documented form
    /// and wins; a `chat_template_kwargs.ocr` alongside it is ignored with a
    /// warning rather than merged - two half-applied objects would be the
    /// silent-failure shape.
    pub fn from_request(
        top: Option<&Value>,
        kwargs: Option<&Value>,
    ) -> Result<Option<OcrOpts>, String> {
        let kw = kwargs
            .and_then(|k| k.as_object())
            .and_then(|k| k.get("ocr"));
        match (top, kw) {
            (Some(t), other) => {
                if other.is_some() {
                    tracing::warn!(
                        "both top-level `ocr` and `chat_template_kwargs.ocr` sent - \
                         using the top-level object"
                    );
                }
                Ok(Some(OcrOpts::parse(t)?))
            }
            (None, Some(k)) => Ok(Some(OcrOpts::parse(k)?)),
            (None, None) => Ok(None),
        }
    }
}

fn usize_field(v: &Value, name: &str) -> Result<usize, String> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| format!("{name} must be a non-negative integer"))
}

/// What the server actually resolved for one request - the echo/log payload,
/// plus the two values the serving path consumes (`ngram`, `force_base`).
#[derive(Debug, Clone)]
pub struct OcrResolved {
    /// The resolved mode's WIRE NAME (a string so both document-parser
    /// families construct this - paddle_ocr's vocabulary is not this enum).
    /// None = pass-through (the caller's text, unclassified by design).
    pub mode: Option<&'static str>,
    /// The crop class that will actually run.
    pub crop: &'static str,
    /// Prepend an `OcrCrop(Base)` directive chunk (single image forced base;
    /// multi-image is base engine-side already).
    pub force_base: bool,
    /// The grounding-region parse is armed for this request's output.
    pub grounding: bool,
    pub pages: usize,
    /// Tower views the request will encode (global + crops, or pages).
    pub views: usize,
    /// Crop tiles (gundam grid), 0 in base mode / small-image bail.
    pub tiles: usize,
    /// Image-id tokens the marker(s) expand to - same math the engine runs.
    pub image_tokens: usize,
    pub pass_through: bool,
    /// An explicit ocr.mode replaced text the caller also sent.
    pub dropped_text: bool,
    /// (n, window) for the sampler; (0, 0) = off.
    pub ngram: (usize, usize),
    /// The n-gram guard STOPS the read where it would have banned (the
    /// sampler's stop mode) instead of masking - PaddleOCR-VL's default.
    pub repeat_stop: bool,
}

impl OcrResolved {
    /// The sampler's `no_repeat_ngram`: the guard sizes and whether it stops.
    pub fn guard(&self) -> (usize, usize, bool) {
        (self.ngram.0, self.ngram.1, self.repeat_stop)
    }

    /// The response extension object (without `regions`, which is appended
    /// from the finished output when grounding is armed). `finish` says
    /// whether the repetition stop ended the read; None while it runs.
    pub fn echo_at(&self, finish: Option<FinishReason>) -> Value {
        let ban = if self.repeat_stop { (0, 0) } else { self.ngram };
        let mut echo = json!({
            "mode": self.mode,
            "crop": self.crop,
            "grounding": self.grounding,
            "pages": self.pages,
            "views": self.views,
            "tiles": self.tiles,
            "image_tokens": self.image_tokens,
            "pass_through": self.pass_through,
            "dropped_text": self.dropped_text,
            "no_repeat_ngram": { "size": ban.0, "window": ban.1 },
        });
        if self.repeat_stop {
            echo["repetition_stop"] = json!({
                "size": self.ngram.0,
                "window": self.ngram.1,
                "fired": finish == Some(FinishReason::Repetition),
            });
        }
        echo
    }
}

/// The reference's ngram defaults (README): size 35 always; window 128 for a
/// single page, 1024 for multi-page.
const NGRAM_SIZE: usize = 35;
const NGRAM_WINDOW_SINGLE: usize = 128;
const NGRAM_WINDOW_MULTI: usize = 1024;

/// Resolve one OCR request. `messages` is the NORMALIZED message array the
/// chat template will render (mutated in place when a canonical string or
/// grounding token is injected); `sizes` is one (w, h) per decoded image in
/// request order; `max_tiles` is the served tower's tile budget.
///
/// Returns `Ok(None)` for a text-only request with no `ocr` object - plain
/// chat on this model, nothing to resolve. A text-only request with an `ocr`
/// object is a 400: every knob in it conditions an image parse.
pub fn resolve(
    messages: &mut [Value],
    opts: Option<OcrOpts>,
    sizes: &[(usize, usize)],
    max_tiles: usize,
) -> Result<Option<OcrResolved>, String> {
    let pages = sizes.len();
    if pages == 0 {
        if opts.is_some() {
            return Err(
                "the ocr request object applies to image requests - this request has no image \
                 (attach the page as an image or PDF)"
                    .into(),
            );
        }
        return Ok(None);
    }
    let opts = opts.unwrap_or_default();

    let text = body_text(messages);
    let has_text = !text.trim().is_empty();
    let pass_through = has_text && opts.mode.is_none();

    // mode: explicit wins; empty text derives from the request shape;
    // otherwise the caller's text passes through verbatim (reference parity -
    // silently replacing what someone wrote is worse than off-vocabulary
    // conditioning, so pass-through is never rewritten).
    let mode = match opts.mode {
        Some(m) => Some(m),
        None if !has_text => Some(if pages > 1 {
            OcrMode::Multipage
        } else {
            OcrMode::Document
        }),
        None => None,
    };
    let multipage = pages > 1 || mode == Some(OcrMode::Multipage);

    // crop: gundam is single-image only (the reference mandates non-crop for
    // multi-image, and multipage mode is the non-crop configuration).
    if opts.crop == OcrCrop::Gundam && pages > 1 {
        return Err(format!(
            "ocr.crop \"gundam\" needs a single image - this family parses {pages} images in \
             base mode only (the reference mandates non-crop for multi-image)"
        ));
    }
    if opts.crop == OcrCrop::Gundam && mode == Some(OcrMode::Multipage) {
        return Err(
            "ocr.crop \"gundam\" contradicts ocr.mode \"multipage\" - multi-page parsing runs \
             in base mode"
                .into(),
        );
    }
    let force_base = pages == 1 && (opts.crop == OcrCrop::Base || mode == Some(OcrMode::Multipage));

    // grounding: layout mode's canonical string carries the token already;
    // an explicit `grounding: true` on any other prompt prepends it (the
    // family's grounded shape is `<image>\n<|grounding|>TASK`). Either arms
    // the region parse, as does pass-through text that brought its own token.
    let grounding =
        opts.grounding == Some(true) || mode == Some(OcrMode::Layout) || text.contains(GROUNDING);

    // the prompt mutation
    let mut dropped_text = false;
    if let Some(m) = mode {
        let mut task = m.canonical().to_owned();
        if opts.grounding == Some(true) && !task.contains(GROUNDING) {
            task = format!("\n{GROUNDING}{}", task.trim_start_matches('\n'));
        }
        if opts.mode.is_some() && has_text {
            // decided behaviour: explicit ocr.mode wins -> canonical string;
            // the replacement is echoed and logged, never silent
            dropped_text = true;
            tracing::warn!(
                mode = m.as_str(),
                "explicit ocr.mode replaced the request's own text - echoed as dropped_text"
            );
        }
        set_body_text(messages, &task, opts.mode.is_some());
    } else if opts.grounding == Some(true) && !text.contains(GROUNDING) {
        prepend_grounding(messages);
    }

    // the sampling default, unless overridden; both halves > 0 or it is off
    // (the reference's own gate)
    let size = opts.ngram_size.unwrap_or(NGRAM_SIZE);
    let window = opts.ngram_window.unwrap_or(if multipage {
        NGRAM_WINDOW_MULTI
    } else {
        NGRAM_WINDOW_SINGLE
    });
    let ngram = if size == 0 || window == 0 {
        (0, 0)
    } else {
        (size, window)
    };

    // geometry echo - the exact planner the engine runs
    let layout =
        paddock_engine::gpu_model::deepseek_ocr::tiling::Layout::plan(sizes, max_tiles, force_base);
    let resolved = OcrResolved {
        mode: mode.map(OcrMode::as_str),
        crop: if layout.grid.is_some() {
            "gundam"
        } else {
            "base"
        },
        force_base,
        grounding,
        pages,
        views: layout.views,
        tiles: layout.grid.map_or(0, |g| g.tiles()),
        image_tokens: layout.image_tokens,
        pass_through,
        dropped_text,
        ngram,
        repeat_stop: false,
    };
    tracing::info!(
        mode = resolved.mode.unwrap_or("pass-through"),
        crop = resolved.crop,
        grounding = resolved.grounding,
        pages = resolved.pages,
        views = resolved.views,
        tiles = resolved.tiles,
        pass_through = resolved.pass_through,
        dropped_text = resolved.dropped_text,
        ngram_size = resolved.ngram.0,
        ngram_window = resolved.ngram.1,
        "ocr request resolved"
    );
    Ok(Some(resolved))
}

const GROUNDING: &str = "<|grounding|>";

/// The body text the family template will render: every non-system message's
/// text content, concatenated in order - the same walk the template does,
/// minus the image markers.
pub(crate) fn body_text(messages: &[Value]) -> String {
    let mut out = String::new();
    for m in messages {
        if m.get("role").and_then(Value::as_str) == Some("system") {
            continue;
        }
        match m.get("content") {
            Some(Value::String(s)) => out.push_str(s),
            Some(Value::Array(parts)) => {
                for p in parts {
                    if p.get("type").and_then(Value::as_str) == Some("text")
                        && let Some(t) = p.get("text").and_then(Value::as_str)
                    {
                        out.push_str(t);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Write the resolved task string into the last non-system message: replace
/// its text parts when an explicit mode overrode caller text (`replace`),
/// else append the derived string after whatever is there (which is nothing -
/// the derived arm only runs on empty text).
pub(crate) fn set_body_text(messages: &mut [Value], task: &str, replace: bool) {
    let Some(last) = messages
        .iter_mut()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) != Some("system"))
    else {
        return;
    };
    match last.get_mut("content") {
        Some(Value::Array(parts)) => {
            if replace {
                parts.retain(|p| p.get("type").and_then(Value::as_str) != Some("text"));
            }
            parts.push(json!({"type": "text", "text": task}));
        }
        Some(Value::String(s)) => {
            if replace {
                s.clear();
            }
            s.push_str(task);
        }
        _ => {
            if let Some(obj) = last.as_object_mut() {
                obj.insert("content".into(), json!([{"type": "text", "text": task}]));
            }
        }
    }
}

/// Prepend `\n<|grounding|>` to the first text part of the last message that
/// carries text - the family's grounded-prompt position (right after the
/// image marker run).
fn prepend_grounding(messages: &mut [Value]) {
    for m in messages.iter_mut().rev() {
        if m.get("role").and_then(Value::as_str) == Some("system") {
            continue;
        }
        match m.get_mut("content") {
            Some(Value::String(s)) if !s.trim().is_empty() => {
                *s = format!("\n{GROUNDING}{}", s.trim_start_matches('\n'));
                return;
            }
            Some(Value::Array(parts)) => {
                for p in parts.iter_mut() {
                    if p.get("type").and_then(Value::as_str) == Some("text")
                        && let Some(t) = p.get("text").and_then(Value::as_str)
                        && !t.trim().is_empty()
                    {
                        let new = format!("\n{GROUNDING}{}", t.trim_start_matches('\n'));
                        p["text"] = Value::String(new);
                        return;
                    }
                }
            }
            _ => {}
        }
    }
}

/// One grounded region parsed from the model's output. Coordinates are the
/// model's own 0-999-normalized integers - scale by `image_dim / 999` to get
/// pixels (the reference's `draw_bounding_boxes` does exactly that). Every
/// family's regions land in this space on the wire; a family with a different
/// native grid (paddleocr's 0-1000 LOC vocabulary) is rescaled at parse time,
/// so a client never needs a per-family denominator.
#[derive(Debug, PartialEq)]
pub struct Region {
    pub label: String,
    pub boxes: Vec<[i64; 4]>,
    /// The block's own text (document mode: what follows its det record until
    /// the next one). None on grounding records, whose label is the content.
    /// Carried so a client can link a box to the words inside it - the
    /// association exists in the stream and flattening it away was a bug
    pub text: Option<String>,
    /// Full 4-corner quadrilaterals `[x1,y1,x2,y2,x3,y3,x4,y4]`, parallel to
    /// `boxes` (each box is its quad's axis-aligned hull). Only the spotting
    /// parse fills these - the format supports rotated text and flattening a
    /// quad to its hull loses that - deepseek's rectangle forms leave it
    /// empty and the wire omits it.
    pub quads: Vec<[i64; 8]>,
    /// The block continues the previous one (LightOnOCR-3's `label+`: a
    /// paragraph flowing into the next column). Only that parse sets it, and
    /// the wire carries it only when true.
    pub continues: bool,
}

impl Region {
    pub fn to_json(&self) -> Value {
        let mut v = json!({"label": self.label, "boxes": self.boxes});
        if let Some(t) = &self.text {
            v["text"] = json!(t);
        }
        if !self.quads.is_empty() {
            v["quads"] = json!(self.quads);
        }
        if self.continues {
            v["continues"] = json!(true);
        }
        v
    }
}

/// Parse grounded regions from RAW output text (decoded with special tokens -
/// the markup rides on `<|ref|>`/`<|det|>` specials that a plain content
/// decode may strip). Mirrors the reference's `re_match` exactly, both
/// forms and in its order - every `<|ref|>label<|/ref|><|det|>boxes<|/det|>`
/// match first, then every bare `<|det|>type [box]<|/det|>` block record:
///
/// ```text
/// <|ref|>title<|/ref|><|det|>[[68, 69, 385, 100]]<|/det|>       (grounding)
/// <|det|>title [68, 69, 385, 100]<|/det|>Quarterly Report       (document)
/// ```
pub fn parse_regions(raw: &str) -> Vec<Region> {
    let mut out = Vec::new();
    // form 1: <|ref|>(.*?)<|/ref|><|det|>(.*?)<|/det|>, non-greedy
    let mut cur = raw;
    while let Some(i) = cur.find("<|ref|>") {
        let after = &cur[i + "<|ref|>".len()..];
        let Some(e) = after.find("<|/ref|>") else {
            break;
        };
        let label = &after[..e];
        let rest = &after[e + "<|/ref|>".len()..];
        // the reference regex requires <|det|> IMMEDIATELY after <|/ref|>
        let Some(body) = rest.strip_prefix("<|det|>") else {
            cur = rest;
            continue;
        };
        let Some(de) = body.find("<|/det|>") else {
            break;
        };
        if let Some(boxes) = parse_boxes(&body[..de]) {
            out.push(Region {
                label: label.trim().to_owned(),
                boxes,
                text: None,
                quads: vec![],
                continues: false,
            });
        }
        cur = &body[de + "<|/det|>".len()..];
    }
    // form 2: <|det|>\s*label\s*[box]\s*<|/det|> - the document-mode block
    // records (a det whose body is a bare coordinate list belongs to form 1
    // and fails the label scan here, so the two passes never double-count)
    let mut cur = raw;
    while let Some(i) = cur.find("<|det|>") {
        let body_start = &cur[i + "<|det|>".len()..];
        let Some(de) = body_start.find("<|/det|>") else {
            break;
        };
        let body = &body_start[..de];
        cur = &body_start[de + "<|/det|>".len()..];
        let t = body.trim();
        let Some(bracket) = t.find('[') else { continue };
        let label = t[..bracket].trim();
        // the reference's label class: [A-Za-z_][\w-]*
        let mut chars = label.chars();
        let head_ok = chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        if !head_ok
            || !label
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            continue;
        }
        if let Some(boxes) = parse_boxes(&t[bracket..]) {
            // the block's text: everything after this det record up to the
            // next marker (document mode has exactly this shape)
            let span_end = ["<|det|>", "<|ref|>"]
                .iter()
                .filter_map(|m| cur.find(m))
                .min()
                .unwrap_or(cur.len());
            let span = cur[..span_end].trim();
            out.push(Region {
                label: label.to_owned(),
                boxes,
                text: (!span.is_empty()).then(|| span.to_owned()),
                quads: vec![],
                continues: false,
            });
        }
    }
    out
}

/// `[x1, y1, x2, y2]` or `[[...], [...]]` - the reference `eval()`s this; a
/// Python list-of-numbers literal is valid JSON, so parse it as that and
/// accept both nestings like `extract_coordinates_and_label` does.
fn parse_boxes(s: &str) -> Option<Vec<[i64; 4]>> {
    let v: Value = serde_json::from_str(s.trim()).ok()?;
    let arr = v.as_array()?;
    let one = |b: &[Value]| -> Option<[i64; 4]> {
        if b.len() != 4 {
            return None;
        }
        let mut o = [0i64; 4];
        for (d, s) in o.iter_mut().zip(b) {
            *d = s.as_i64().or_else(|| s.as_f64().map(|f| f as i64))?;
        }
        Some(o)
    };
    if arr.iter().all(Value::is_number) {
        return Some(vec![one(arr)?]);
    }
    arr.iter()
        .map(|b| b.as_array().and_then(|b| one(b)))
        .collect::<Option<Vec<_>>>()
        .filter(|v| !v.is_empty())
}

/// Regions as the response extension array, or None when nothing parsed.
/// Tries this family's det/ref markup first, then paddleocr's spotting
/// `<|LOC_n|>` lines, then LightOnOCR-3's `![label](x1,y1,x2,y2)` blocks -
/// the marker vocabularies are disjoint, so one entry point serves every
/// attach site and the parse itself stays the truth test.
pub fn regions_json(raw: &str) -> Option<Value> {
    let mut rs = parse_regions(raw);
    if rs.is_empty() {
        rs = crate::paddle_ocr::parse_spotting(raw);
    }
    if rs.is_empty() {
        rs = crate::lighton_ocr::parse_blocks(raw);
    }
    (!rs.is_empty()).then(|| Value::Array(rs.iter().map(Region::to_json).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_TILES: usize = 32;

    fn user(parts: Value) -> Value {
        json!({"role": "user", "content": parts})
    }

    fn text_of(messages: &[Value]) -> String {
        body_text(messages)
    }

    #[test]
    fn derived_default_single_image_is_document_gundam() {
        // the battery-page shape: one image, no text, no ocr object
        let mut msgs = vec![user(json!([{"type": "image"}]))];
        let r = resolve(&mut msgs, None, &[(1240, 1754)], MAX_TILES)
            .expect("resolve")
            .expect("images present");
        assert_eq!(r.mode, Some("document"));
        assert_eq!((r.crop, r.force_base), ("gundam", false));
        assert_eq!((r.pages, r.views, r.tiles), (1, 7, 6));
        assert_eq!(r.image_tokens, 903, "the arbiter-measured battery geometry");
        assert_eq!(r.ngram, (35, 128));
        assert!(!r.pass_through && !r.grounding && !r.dropped_text);
        assert_eq!(text_of(&msgs), "document parsing.");
    }

    #[test]
    fn derived_default_multi_image_is_multipage_base() {
        let mut msgs = vec![user(json!([{"type": "image"}, {"type": "image"}]))];
        let r = resolve(&mut msgs, None, &[(1240, 1754), (800, 600)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert_eq!(r.mode, Some("multipage"));
        assert_eq!(
            (r.crop, r.force_base),
            ("base", false),
            "N pages: engine derives base"
        );
        assert_eq!((r.pages, r.views, r.tiles), (2, 2, 0));
        assert_eq!(r.ngram, (35, 1024));
        assert_eq!(text_of(&msgs), "Multi page parsing.");
    }

    #[test]
    fn caller_text_passes_through_verbatim() {
        let mut msgs = vec![user(
            json!([{"type": "image"}, {"type": "text", "text": "document parsing."}]),
        )];
        let before = msgs.clone();
        let r = resolve(&mut msgs, None, &[(1240, 1754)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert!(r.pass_through);
        assert_eq!(r.mode, None, "pass-through is never classified");
        assert_eq!(msgs, before, "verbatim means untouched");
        assert_eq!(r.ngram, (35, 128), "the sampling default still applies");
    }

    #[test]
    fn explicit_mode_wins_and_reports_dropped_text() {
        let mut msgs = vec![user(
            json!([{"type": "image"}, {"type": "text", "text": "my own words"}]),
        )];
        let opts = OcrOpts {
            mode: Some(OcrMode::Free),
            ..Default::default()
        };
        let r = resolve(&mut msgs, Some(opts), &[(1000, 1000)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert_eq!(r.mode, Some("free"));
        assert!(r.dropped_text, "replacement is echoed, never silent");
        assert_eq!(
            text_of(&msgs),
            "\nFree OCR. ",
            "byte-exact canonical incl. trailing space"
        );
    }

    #[test]
    fn layout_mode_is_grounded_and_multipage_forces_base() {
        let mut msgs = vec![user(json!([{"type": "image"}]))];
        let opts = OcrOpts {
            mode: Some(OcrMode::Layout),
            ..Default::default()
        };
        let r = resolve(&mut msgs, Some(opts), &[(1240, 1754)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert!(r.grounding, "layout IS the grounded mode");
        assert!(text_of(&msgs).contains(GROUNDING));
        assert!(!r.force_base);

        // multipage mode on one image runs base - the README pairs them
        let mut msgs = vec![user(json!([{"type": "image"}]))];
        let opts = OcrOpts {
            mode: Some(OcrMode::Multipage),
            ..Default::default()
        };
        let r = resolve(&mut msgs, Some(opts), &[(1240, 1754)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert!(r.force_base);
        assert_eq!((r.crop, r.views, r.image_tokens), ("base", 1, 273));
        assert_eq!(r.ngram, (35, 1024), "multipage mode takes the long window");
    }

    #[test]
    fn grounding_flag_composes_with_canonical_and_pass_through() {
        // derived document + grounding: token prepended in the family shape
        let mut msgs = vec![user(json!([{"type": "image"}]))];
        let opts = OcrOpts {
            grounding: Some(true),
            ..Default::default()
        };
        let r = resolve(&mut msgs, Some(opts.clone()), &[(1240, 1754)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert!(r.grounding);
        assert_eq!(text_of(&msgs), "\n<|grounding|>document parsing.");

        // pass-through + grounding: the caller's words survive, prefixed
        let mut msgs = vec![user(
            json!([{"type": "image"}, {"type": "text", "text": "find the tables"}]),
        )];
        let r = resolve(&mut msgs, Some(opts), &[(1240, 1754)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert!(r.grounding && r.pass_through);
        assert_eq!(text_of(&msgs), "\n<|grounding|>find the tables");
    }

    #[test]
    fn crop_overrides_and_refusals() {
        // explicit base on one image: forced, and the echo geometry says so
        let mut msgs = vec![user(json!([{"type": "image"}]))];
        let opts = OcrOpts {
            crop: OcrCrop::Base,
            ..Default::default()
        };
        let r = resolve(&mut msgs, Some(opts), &[(1240, 1754)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert!(r.force_base);
        assert_eq!(
            (r.crop, r.views, r.tiles, r.image_tokens),
            ("base", 1, 0, 273)
        );

        // gundam on many images is the documented refusal
        let mut msgs = vec![user(json!([{"type": "image"}, {"type": "image"}]))];
        let opts = OcrOpts {
            crop: OcrCrop::Gundam,
            ..Default::default()
        };
        let err = resolve(&mut msgs, Some(opts), &[(100, 100), (100, 100)], MAX_TILES)
            .expect_err("must refuse");
        assert!(err.contains("base mode only"), "{err}");
    }

    #[test]
    fn ngram_overrides_and_off_switch() {
        let mut msgs = vec![user(json!([{"type": "image"}]))];
        let opts = OcrOpts {
            ngram_size: Some(20),
            ngram_window: Some(256),
            ..Default::default()
        };
        let r = resolve(&mut msgs, Some(opts), &[(640, 480)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert_eq!(r.ngram, (20, 256));

        let mut msgs = vec![user(json!([{"type": "image"}]))];
        let opts = OcrOpts {
            ngram_size: Some(0),
            ..Default::default()
        };
        let r = resolve(&mut msgs, Some(opts), &[(640, 480)], MAX_TILES)
            .expect("resolve")
            .expect("images");
        assert_eq!(
            r.ngram,
            (0, 0),
            "size 0 disables, like the reference's gate"
        );
    }

    #[test]
    fn text_only_requests_stay_plain_chat() {
        let mut msgs = vec![user(json!("what is the capital of France?"))];
        assert!(
            resolve(&mut msgs, None, &[], MAX_TILES)
                .expect("ok")
                .is_none()
        );
        // ...but an ocr object without an image is a loud 400
        let opts = OcrOpts {
            mode: Some(OcrMode::Document),
            ..Default::default()
        };
        assert!(resolve(&mut msgs, Some(opts), &[], MAX_TILES).is_err());
    }

    #[test]
    fn opts_parse_rejects_unknowns_and_bad_values() {
        assert!(OcrOpts::parse(&json!({"mode": "document"})).is_ok());
        assert!(OcrOpts::parse(&json!({"mode": "documnet"})).is_err());
        assert!(
            OcrOpts::parse(&json!({"crops": "base"})).is_err(),
            "typo'd key must 400"
        );
        assert!(OcrOpts::parse(&json!({"grounding": "yes"})).is_err());
        assert!(OcrOpts::parse(&json!("document")).is_err());
        // both channels: top level wins
        let top = json!({"mode": "free"});
        let kw = json!({"ocr": {"mode": "layout"}});
        let o = OcrOpts::from_request(Some(&top), Some(&kw))
            .expect("ok")
            .expect("some");
        assert_eq!(o.mode, Some(OcrMode::Free));
        let o = OcrOpts::from_request(None, Some(&kw))
            .expect("ok")
            .expect("some");
        assert_eq!(o.mode, Some(OcrMode::Layout));
    }

    #[test]
    fn regions_parse_both_reference_forms() {
        // the document-mode block record (what the battery page emits)
        let doc = "<|det|>title [68, 69, 385, 100]<|/det|>Quarterly Report\n\
                   <|det|>text [67, 136, 661, 209]<|/det|>Revenue for the quarter...";
        let rs = parse_regions(doc);
        assert_eq!(rs.len(), 2);
        assert_eq!(
            rs[0],
            Region {
                label: "title".into(),
                boxes: vec![[68, 69, 385, 100]],
                text: Some("Quarterly Report".into()),
                quads: vec![],
                continues: false,
            }
        );
        assert_eq!(rs[1].label, "text");
        // each block keeps its own words - the box↔text association
        assert_eq!(rs[1].text.as_deref(), Some("Revenue for the quarter..."));

        // the grounding form, incl. the multi-box nesting eval() accepts
        let grounded = "<|ref|>table<|/ref|><|det|>[[68, 244, 434, 340]]<|/det|> and \
                        <|ref|>image<|/ref|><|det|>[[1,2,3,4], [5,6,7,8]]<|/det|>";
        let rs = parse_regions(grounded);
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[0].label, "table");
        assert_eq!(rs[0].text, None);
        assert_eq!(rs[1].boxes.len(), 2);

        // malformed bodies parse to nothing rather than garbage
        assert!(parse_regions("<|det|>title [68, 69<|/det|>x").is_empty());
        assert!(parse_regions("plain text with [1,2,3,4] but no markers").is_empty());
        assert!(regions_json("no markup at all").is_none());
    }

    #[test]
    fn advertised_caps_round_trip_through_the_parsers() {
        // Every advertised mode and crop must be accepted verbatim by the
        // request parser - the capability object is a promise about what
        // OcrOpts::parse takes, and a listed string the parser rejects is the
        // silent-drift failure this test exists to catch.
        let caps = caps_json();
        let modes = caps["modes"].as_array().expect("modes array");
        assert_eq!(modes.len(), OcrMode::ALL.len());
        for m in modes {
            let o = OcrOpts::parse(&json!({"mode": m})).expect("advertised mode parses");
            assert_eq!(
                o.mode.expect("mode set").as_str(),
                m.as_str().expect("string")
            );
        }
        for c in caps["crops"].as_array().expect("crops array") {
            OcrOpts::parse(&json!({"crop": c})).expect("advertised crop parses");
        }
        assert_eq!(caps["grounding"], json!(true));
    }
}
