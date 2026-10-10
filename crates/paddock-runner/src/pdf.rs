//! PDF -> per-page RGB8 images for the vision path.
//!
//! Server-side rasterization via **pdfium** (`paddock-pdfium`, our own
//! in-process binding - Not a `pdftoppm`/`mutool` subprocess, so it honors the
//! no-shell-fan-out doctrine). Mirrors hq's `hq-ocr-service` rasterizer, but
//! emits interleaved RGB8 straight into the engine's `MmChunk::Image { rgb, w,
//! h }` shape (no PNG round-trip) - the vision tower wants raw pixels.
//!
//! pdfium is statically linked: our own build from
//! `packs/pdfium/build/`, part of the binary. It used to be an optional sidecar
//! library searched for at startup, which meant PDF rendering could be quietly
//! absent; now the only question left is whether the MODEL can read page
//! images. Rendering is CPU-bound and must run off the request executor /
//! inference thread - callers wrap [`render`] in `tokio::task::spawn_blocking`.
//! pdfium's core is not reentrant, so every call goes through one process-wide
//! instance behind a mutex. That serialization used to come free with
//! `pdfium-render`'s `thread_safe` feature; `paddock-pdfium` deliberately does
//! not serialize for you - a library should not impose a lock its caller may
//! already hold - so the mutex lives here, in the one place that calls pdfium.
//! Both entry points below run under `spawn_blocking`, i.e. on arbitrary
//! threads, which is exactly why it is not optional.
//!
//! macOS uses its native CoreGraphics rasterizer when PDFium is not linked.
//! Both paths run in-process on the blocking pool; text extraction uses sift.

#[cfg(feature = "pdf")]
use std::sync::{Mutex, MutexGuard, OnceLock};

#[cfg(feature = "pdf")]
use paddock_pdfium::Pdfium;

#[cfg(all(target_os = "macos", not(feature = "pdf")))]
#[path = "pdf_macos.rs"]
mod native;
#[cfg(all(target_os = "macos", not(feature = "pdf")))]
pub(crate) use native::{render, render_page_rgb};

/// Rasterization knobs (built from [`crate::config::Config`]).
#[derive(Debug, Clone)]
pub struct PdfConfig {
    /// Soft cap: render at most this many pages; the rest are dropped with
    /// `truncated = true` (callers surface "N of M pages" - never silent).
    pub max_pages: usize,
    /// Target long-edge (px) per page; per-page DPI derived from it, capped at
    /// `max_dpi`. 1568 matches the Qwen vision sweet spot.
    pub long_edge: u32,
    /// The DPI ceiling that keeps an odd tiny page from rendering huge - 300
    /// unless the serving checkpoint was trained on pages rendered to a fixed
    /// edge whatever their size (LightOnOCR-3), which lifts it. The pdfium
    /// path honours it; the macOS PDFKit path keeps its own 300.
    pub max_dpi: f32,
}

impl PdfConfig {
    pub fn from_config(cfg: &crate::config::Config) -> Self {
        Self {
            max_pages: cfg.pdf_max_pages,
            long_edge: cfg.pdf_page_long_edge,
            max_dpi: 300.0,
        }
    }
}

/// One rendered page as interleaved RGB8 - drops straight into
/// `MmChunk::Image { rgb, w, h }`.
pub struct PdfPage {
    pub rgb: Vec<u8>,
    pub w: usize,
    pub h: usize,
}

/// Which pages of a multi-page attachment the caller wants - resolved per
/// part from the `pages` field ("2-4" / "3" / "2-" / "all"), falling back to
/// the `max_pages` cap (= `First`), else everything. 1-based inclusive,
/// clamped to the document by the consumer; a START past the end is a loud
/// error there, never a silent empty result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageSel {
    All,
    /// pages 1..=n (the `max_pages` cap).
    First(usize),
    /// pages a..=b; `b == usize::MAX` means "to the end" (`"2-"`).
    Range(usize, usize),
}

impl PageSel {
    /// Resolve against a document's real page count: the concrete 1-based
    /// inclusive (start, end). Errors when the range starts past the end -
    /// `label` names the file in the message.
    pub(crate) fn resolve(self, total: usize, label: &str) -> Result<(usize, usize), String> {
        match self {
            PageSel::All => Ok((1, total)),
            PageSel::First(n) => Ok((1, n.clamp(1, total))),
            PageSel::Range(a, b) => {
                if a > total {
                    let s = if total == 1 { "" } else { "s" };
                    return Err(format!(
                        "{label} has only {total} page{s} - the requested pages start at {a}"
                    ));
                }
                Ok((a, b.min(total)))
            }
        }
    }
}

/// Parse the `pages` grammar: `"all"` | `"3"` | `"2-4"` | `"2-"` (to the end)
/// | `"-4"` (from the start). Anything else is a loud error.
pub(crate) fn parse_pages(s: &str) -> Result<PageSel, String> {
    let bad = || {
        format!("pages {s:?} is not valid - use \"2-4\", \"3\", \"2-\" (to the end), or \"all\"")
    };
    let t = s.trim();
    if t.eq_ignore_ascii_case("all") {
        return Ok(PageSel::All);
    }
    let (a, b) = match t.split_once('-') {
        None => {
            let n: usize = t.parse().map_err(|_| bad())?;
            (n, n)
        }
        Some((x, y)) => {
            let a = if x.trim().is_empty() {
                1
            } else {
                x.trim().parse().map_err(|_| bad())?
            };
            let b = if y.trim().is_empty() {
                usize::MAX
            } else {
                y.trim().parse().map_err(|_| bad())?
            };
            (a, b)
        }
    };
    if a == 0 || b < a {
        return Err(bad());
    }
    Ok(PageSel::Range(a, b))
}

/// Part-level `pages` range, if present.
pub(crate) fn part_pages(part: &Value) -> Result<Option<PageSel>, String> {
    match part.get("pages") {
        None => Ok(None),
        Some(v) => {
            let s = v
                .as_str()
                .ok_or("pages on a content part must be a string like \"2-4\" or \"all\"")?;
            parse_pages(s).map(Some)
        }
    }
}

/// The page selection of one content part: its `pages` range wins, then its
/// `max_pages` cap, then the request-level cap, then everything.
pub(crate) fn part_page_sel(part: &Value, req_max: Option<usize>) -> Result<PageSel, String> {
    if let Some(sel) = part_pages(part)? {
        return Ok(sel);
    }
    Ok(match part_max_pages(part)?.or(req_max) {
        Some(n) => PageSel::First(n),
        None => PageSel::All,
    })
}

/// The outcome of rasterizing a document. `pages.len() <= total_pages`.
pub struct RenderedPdf {
    /// One RGB8 image per rendered page, in reading order.
    pub pages: Vec<PdfPage>,
    /// Pages in the source PDF (>= `pages.len()`).
    pub total_pages: usize,
    /// 1-based number of the first rendered page (page markers must show the
    /// real page numbers when a range starts past 1).
    pub first_page: usize,
    /// True when `pages` is a strict subset of the document - callers must
    /// surface which pages are shown (in-prompt note + UI), never silently.
    pub truncated: bool,
    /// True when the SERVER's rendering ceiling (`cfg.max_pages`) cut the
    /// render below what the caller's own selection resolved to. Distinct
    /// from `truncated`: a user asking pages "2-4" of a 10-page file is
    /// intent, not a clip. Callers with no in-band disclosure channel (the
    /// OCR family's bare-page route) turn this into a loud error.
    pub ceiling_clipped: bool,
}

// Default builds link PDFium in. The Metal bring-up build may omit it.
#[derive(Debug, thiserror::Error)]
pub enum PdfError {
    #[error("PDF image rendering is not included in this build; text extraction remains available")]
    Unavailable,
    #[error("could not parse the PDF: {0}")]
    Load(String),
    #[error("{0}")]
    Pages(String),
    #[error("the PDF has no pages")]
    Empty,
    #[error("failed to render page {0}: {1}")]
    Render(usize, String),
}

// pdfium is linked into this binary (build.rs, packs/pdfium/build/), so there
// is no library to find and no bind that can fail - `FPDF_InitLibrary` is a
// call, not a dlopen. Still cached for the process because pdfium's global
// init must happen exactly once.
#[cfg(feature = "pdf")]
static PDFIUM: OnceLock<Mutex<Pdfium>> = OnceLock::new();

/// The one pdfium, locked. A poisoned mutex means a previous render panicked
/// inside pdfium; every document is loaded and closed within a single lock
/// hold, so there is nothing of OURS left half-built - recover rather than
/// poison every later request in the process.
#[cfg(feature = "pdf")]
fn bind() -> MutexGuard<'static, Pdfium> {
    PDFIUM
        .get_or_init(|| Mutex::new(Pdfium::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Is PDF rasterization included in this build?
///
/// Kept as a function rather than deleted at every call site: what varies is
/// whether the MODEL can consume page images, and the callers read better
/// asking both questions the same way.
pub fn available(_cfg: &PdfConfig) -> bool {
    cfg!(any(feature = "pdf", target_os = "macos"))
}

#[cfg(not(any(feature = "pdf", target_os = "macos")))]
pub(crate) fn render(
    _bytes: &[u8],
    _cfg: &PdfConfig,
    _sel: PageSel,
) -> Result<RenderedPdf, PdfError> {
    Err(PdfError::Unavailable)
}

#[cfg(not(any(feature = "pdf", target_os = "macos")))]
pub(crate) fn render_page_rgb(_bytes: &[u8], _page: u32, _dpi: f32) -> Option<(Vec<u8>, u32, u32)> {
    None
}

/// Rasterize `bytes` to per-page RGB8, `sel` picking which pages (the
/// server's `cfg.max_pages` ceiling still bounds how many - a VRAM/latency
/// guard, whatever the caller asked for). **CPU-bound and blocking** - call
/// under `tokio::task::spawn_blocking`, never on an async/inference thread.
#[cfg(feature = "pdf")]
pub(crate) fn render(bytes: &[u8], cfg: &PdfConfig, sel: PageSel) -> Result<RenderedPdf, PdfError> {
    let pdfium = bind();
    let doc = pdfium
        .load(bytes)
        .map_err(|e| PdfError::Load(e.to_string()))?;
    let total = doc.page_count();
    if total == 0 {
        return Err(PdfError::Empty);
    }
    let (start, want_end) = sel.resolve(total, "the PDF").map_err(PdfError::Pages)?;
    let end = want_end.min(start + cfg.max_pages.max(1) - 1);
    // `want_end` is already clamped to the document, so falling below it here
    // can only be the server ceiling
    let ceiling_clipped = end < want_end;
    let take = end - start + 1;
    let truncated = take < total;

    let mut pages = Vec::with_capacity(take);
    for idx in (start - 1)..end {
        let (pw, ph) = doc
            .page_size(idx)
            .ok_or_else(|| PdfError::Render(idx, "page not found".into()))?;
        let long_pts = pw.max(ph);
        // pixels = points * dpi/72; DPI capped (300 by default) to bound odd
        // tiny-page PDFs.
        let dpi = ((cfg.long_edge as f32) / long_pts * 72.0).min(cfg.max_dpi);
        // interleaved RGB8 straight to the engine's image shape - no PNG hop
        let bitmap = doc
            .render(
                idx,
                (pw * dpi / 72.0).max(1.0) as u32,
                (ph * dpi / 72.0).max(1.0) as u32,
            )
            .map_err(|e| PdfError::Render(idx, e.to_string()))?;
        pages.push(PdfPage {
            rgb: bitmap.rgb,
            w: bitmap.width as usize,
            h: bitmap.height as usize,
        });
    }
    Ok(RenderedPdf {
        pages,
        total_pages: total,
        first_page: start,
        truncated,
        ceiling_clipped,
    })
}

/// Render a single 0-based `page` to interleaved RGB8 at ~`dpi`, for the
/// forensic render-vs-scan comparison (paddock-forensics `PageRenderer`). Uses
/// the one process-wide pdfium (`bind`), so it must run under
/// `spawn_blocking` like [`render`]. `None` on any load/render failure.
#[cfg(feature = "pdf")]
pub(crate) fn render_page_rgb(bytes: &[u8], page: u32, dpi: f32) -> Option<(Vec<u8>, u32, u32)> {
    let pdfium = bind();
    let doc = pdfium.load(bytes).ok()?;
    let (pw, ph) = doc.page_size(page as usize)?;
    let bitmap = doc
        .render(
            page as usize,
            (pw * dpi / 72.0).max(1.0) as u32,
            (ph * dpi / 72.0).max(1.0) as u32,
        )
        .ok()?;
    Some((bitmap.rgb, bitmap.width, bitmap.height))
}

// ─── content-part expansion (PDF part -> N image parts) ──────────────────────
//
// A PDF arrives as one content part; the vision path wants N page images. We
// rasterize and REPLACE the PDF part with N chat-shaped `image_url` (PNG
// data-URI) parts - so the exact multi-image flow (`find_images` ->
// `build_mm_chunks`) carries it, no parallel path. Page markers + a loud
// truncation note ride alongside so the MODEL knows what (and how much) it got.

use serde_json::{Value, json};

/// What expansion did - for the startup/telemetry log and response metadata.
/// Carries the multi-page TIFF lane too (same page accounting, same
/// disclosure duty), so `chat::expand_attachments` logs one summary.
#[derive(Debug, Default, Clone)]
pub struct PdfSummary {
    /// Number of PDF parts expanded.
    pub pdfs: usize,
    /// Number of multi-page TIFF parts expanded (`crate::tiffdoc`).
    pub tiffs: usize,
    /// Sum of source page counts across all expanded documents.
    pub total_pages: usize,
    /// Sum of pages actually rendered (<= total when a cap truncated).
    pub rendered_pages: usize,
    /// Any document exceeded the page cap.
    pub truncated: bool,
}

impl PdfSummary {
    pub fn any(&self) -> bool {
        self.pdfs > 0 || self.tiffs > 0
    }

    /// Fold another lane's summary into this one (the TIFF lane accumulates
    /// in place; the PDF lanes return their own - one log line either way).
    pub fn absorb(&mut self, other: &PdfSummary) {
        self.pdfs += other.pdfs;
        self.tiffs += other.tiffs;
        self.total_pages += other.total_pages;
        self.rendered_pages += other.rendered_pages;
        self.truncated |= other.truncated;
    }
}

/// A recognized PDF content part: its base64 payload + the caller's name for
/// it (filename, or an Anthropic `document`'s `title`) - the name feeds the
/// injected header so the model can refer to the file the way the user does.
pub(crate) struct PdfPart<'a> {
    /// Data-URI or raw base64; [`decode_pdf_payload`] handles both.
    pub data: &'a str,
    pub filename: Option<&'a str>,
}

/// The PDF payload of a content part, if it is a PDF. Recognizes OpenAI
/// chat `file` (`file.file_data`), Responses `input_file` (`file_data`), and
/// Anthropic `document` (`source: {type: base64, media_type: application/pdf,
/// data}`). For file/input_file, detection is an `application/pdf` data-URI or
/// a `.pdf` filename; for `document`, the declared `media_type`.
pub(crate) fn pdf_part(part: &Value) -> Option<PdfPart<'_>> {
    fn looks_pdf(data: &str, filename: Option<&str>) -> bool {
        data.starts_with("data:application/pdf")
            || filename.is_some_and(|f| f.to_ascii_lowercase().ends_with(".pdf"))
    }
    match part.get("type").and_then(Value::as_str)? {
        "file" => {
            let f = part.get("file")?;
            let data = f.get("file_data").and_then(Value::as_str)?;
            let filename = f.get("filename").and_then(Value::as_str);
            looks_pdf(data, filename).then_some(PdfPart { data, filename })
        }
        "input_file" => {
            let data = part.get("file_data").and_then(Value::as_str)?;
            let filename = part.get("filename").and_then(Value::as_str);
            looks_pdf(data, filename).then_some(PdfPart { data, filename })
        }
        "document" => {
            let src = part.get("source")?;
            let is_pdf_b64 = src.get("type").and_then(Value::as_str) == Some("base64")
                && src.get("media_type").and_then(Value::as_str) == Some("application/pdf");
            let data = is_pdf_b64
                .then(|| src.get("data").and_then(Value::as_str))
                .flatten()?;
            Some(PdfPart {
                data,
                filename: part.get("title").and_then(Value::as_str),
            })
        }
        _ => None,
    }
}

/// Does any message carry a PDF content part? Cheap sync scan so the common
/// no-PDF request skips `spawn_blocking` entirely.
pub fn has_pdf_parts(messages: &[Value]) -> bool {
    messages.iter().any(|m| {
        m.get("content")
            .and_then(Value::as_array)
            .is_some_and(|parts| parts.iter().any(|p| pdf_part(p).is_some()))
    })
}

pub(crate) fn decode_pdf_payload(data: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let b64 = match data.strip_prefix("data:") {
        Some(rest) => {
            rest.split_once(',')
                .ok_or("malformed data: URI in file attachment")?
                .1
        }
        None => data,
    };
    base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("file attachment base64: {e}"))
}

/// Part-level `max_pages`: a flat key on the content part itself, overriding
/// the request-level extension for this attachment only - the settings belong
/// to the file, not the prompt. Shared with the TIFF lane.
pub(crate) fn part_max_pages(part: &Value) -> Result<Option<usize>, String> {
    match part.get("max_pages") {
        None => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) if n >= 1 => Ok(Some(n as usize)),
            _ => Err("max_pages on a content part must be a positive integer".into()),
        },
    }
}

/// Part-level `pdf_mode` on a file-shaped part - same override semantics.
fn part_pdf_mode(part: &Value) -> Result<Option<crate::chat::PdfMode>, String> {
    use crate::chat::PdfMode;
    match part.get("pdf_mode") {
        None => Ok(None),
        Some(v) => match v.as_str() {
            Some("render") => Ok(Some(PdfMode::Render)),
            Some("text") => Ok(Some(PdfMode::Text)),
            other => Err(format!(
                "pdf_mode on a content part must be \"render\" or \"text\", got {:?}",
                other.unwrap_or("a non-string")
            )),
        },
    }
}

/// One page image -> a chat-shaped `image_url` part (lossless PNG data-URI).
/// Shared with the multi-page TIFF expansion (`crate::tiffdoc`), which speaks
/// the same page-part dialect.
pub(crate) fn png_image_part(img: image::RgbImage) -> Result<Value, String> {
    use base64::Engine as _;
    use std::io::Cursor;
    let mut png = Vec::with_capacity(64 * 1024);
    img.write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| format!("PNG encode: {e}"))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
    Ok(json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}}))
}

/// A rendered PDF page -> the shared PNG part shape.
fn page_image_part(page: PdfPage) -> Result<Value, String> {
    let img = image::RgbImage::from_raw(page.w as u32, page.h as u32, page.rgb)
        .ok_or("rendered page RGB size mismatch")?;
    png_image_part(img)
}

/// Replace every PDF content part with what its RESOLVED route produces -
/// rendered page images (`image_url` parts + `[page k]` markers + a loud
/// truncation note), or the sift-extracted text part. The route and page cap
/// are resolved per PART: a flat `pdf_mode`/`max_pages` on the part wins,
/// then the request-level extension, then auto (render where `can_render`) -
/// so two files in one prompt can take different routes. An explicit
/// "render" this server can't do is an error NAMING the file, never a silent
/// downgrade. `with_meta` adds the metadata block (Title/Author/dates) to
/// the note - same block the text path injects, so the two routes read
/// alike. Non-PDF parts/messages pass through untouched. **Blocking**
/// (rasterizes/parses) - call under `spawn_blocking`.
///
/// `plain_pages` (the deepseek2-ocr class): the raster route emits BARE page
/// images - no attachment note, no `[page k]` markers. That family has a
/// fixed prompt vocabulary (framing text is off-vocabulary conditioning for
/// a document parser) and its derived canonical task string engages only on
/// an empty body. With no in-band disclosure channel, a server-ceiling clip
/// becomes a loud error instead of a silent partial parse; the caller's own
/// page selection stays intent, not truncation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn expand_in_messages(
    mut messages: Vec<Value>,
    cfg: &PdfConfig,
    with_meta: bool,
    max_ctx: usize,
    req_max: Option<usize>,
    req_mode: Option<crate::chat::PdfMode>,
    can_render: bool,
    no_render_why: &str,
    plain_pages: bool,
) -> Result<(Vec<Value>, PdfSummary), String> {
    use crate::chat::PdfMode;
    let mut summary = PdfSummary::default();
    for msg in &mut messages {
        let Some(parts) = msg.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        if !parts.iter().any(|p| pdf_part(p).is_some()) {
            continue;
        }
        let taken = std::mem::take(parts);
        let mut out = Vec::with_capacity(taken.len() + 4);
        for part in taken {
            let Some(pdf) = pdf_part(&part) else {
                out.push(part);
                continue;
            };
            let filename = pdf.filename.map(str::to_owned);
            let sel = part_page_sel(&part, req_max)?;
            let raster = match part_pdf_mode(&part)?.or(req_mode) {
                Some(PdfMode::Text) => false,
                Some(PdfMode::Render) => {
                    if !can_render {
                        let name = filename.as_deref().unwrap_or("the attached PDF");
                        return Err(format!(
                            "pdf_mode \"render\" is not possible for {name:?} - {no_render_why}; \
                             use \"text\" or omit pdf_mode"
                        ));
                    }
                    true
                }
                None => can_render,
            };
            let bytes = decode_pdf_payload(pdf.data)?;
            if !raster {
                // the text route: any model reads a PDF's text layer; the
                // caller's page selection slices it with an in-text
                // disclosure. The server rendering ceiling is a VRAM/latency
                // guard and does not apply.
                let extracted =
                    crate::doc::extract_text(&bytes, filename.as_deref(), with_meta, max_ctx, sel)?;
                summary.pdfs += 1;
                summary.total_pages += extracted.total_pages;
                summary.rendered_pages += extracted.taken_pages;
                summary.truncated |= extracted.taken_pages < extracted.total_pages;
                out.push(json!({"type": "text", "text": extracted.text}));
                continue;
            }
            // raster route: the server's own ceiling still protects VRAM
            let rendered = render(&bytes, cfg, sel).map_err(|e| e.to_string())?;
            let n = rendered.pages.len();
            summary.pdfs += 1;
            summary.total_pages += rendered.total_pages;
            summary.rendered_pages += n;
            summary.truncated |= rendered.truncated;
            let total = rendered.total_pages;
            if plain_pages {
                if rendered.ceiling_clipped {
                    let name = filename.as_deref().unwrap_or("the attached PDF");
                    return Err(format!(
                        "{name} renders {total} pages but this server parses at most {cap} per \
                         request - select pages (e.g. \"pages\": \"1-{cap}\") and send the rest \
                         separately, or raise the server's PDF page ceiling",
                        cap = cfg.max_pages.max(1),
                    ));
                }
                for page in rendered.pages {
                    out.push(page_image_part(page)?);
                }
                continue;
            }
            let s = if total == 1 { "" } else { "s" };
            let mut note = match &filename {
                Some(f) => format!("[Attached file: {f} - PDF, {total} page{s}]"),
                None => format!("[Attached PDF - {total} page{s}]"),
            };
            let (a, b) = (rendered.first_page, rendered.first_page + n.max(1) - 1);
            if rendered.truncated {
                // loud, never silent: the model is told which pages it got
                note.push_str(&if a == b {
                    format!("\n[Only page {a} of {total} is shown below.]")
                } else {
                    format!("\n[Only pages {a}-{b} of {total} are shown below.]")
                });
            }
            if with_meta {
                for line in crate::doc::pdf_meta_lines(&bytes) {
                    note.push('\n');
                    note.push_str(&line);
                }
            }
            out.push(json!({"type": "text", "text": note}));
            for (i, page) in rendered.pages.into_iter().enumerate() {
                out.push(json!({"type": "text", "text": format!("[page {}]", a + i)}));
                out.push(page_image_part(page)?);
            }
        }
        // `parts` was emptied by mem::take; restore the expanded content
        if let Some(parts) = msg.get_mut("content").and_then(Value::as_array_mut) {
            *parts = out;
        }
    }
    Ok((messages, summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(any(feature = "pdf", target_os = "macos")))]
    fn omitted_rasterizer_is_reported_not_silently_accepted() {
        let cfg = PdfConfig {
            max_pages: 1,
            long_edge: 512,
            max_dpi: 300.0,
        };
        assert!(!available(&cfg));
        assert!(matches!(
            render(b"%PDF", &cfg, PageSel::All),
            Err(PdfError::Unavailable)
        ));
        assert!(render_page_rgb(b"%PDF", 0, 72.0).is_none());
    }

    /// A minimal valid multi-page PDF with blank pages of the given size (pts),
    /// xref offsets computed correctly so it needs no pdfium recovery path.
    fn tiny_pdf(n_pages: usize, w: u32, h: u32) -> Vec<u8> {
        let mut objs: Vec<String> = Vec::new();
        objs.push("<</Type/Catalog/Pages 2 0 R>>".to_string()); // obj 1
        let kids: Vec<String> = (0..n_pages).map(|i| format!("{} 0 R", 3 + i)).collect();
        objs.push(format!(
            "<</Type/Pages/Kids[{}]/Count {}>>",
            kids.join(" "),
            n_pages
        )); // obj 2
        for _ in 0..n_pages {
            objs.push(format!("<</Type/Page/Parent 2 0 R/MediaBox[0 0 {w} {h}]>>"));
        }
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"%PDF-1.4\n");
        let mut offsets = Vec::with_capacity(objs.len());
        for (i, body) in objs.iter().enumerate() {
            offsets.push(buf.len());
            buf.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", i + 1, body).as_bytes());
        }
        let xref_off = buf.len();
        let n = objs.len() + 1; // + the free object 0
        buf.extend_from_slice(format!("xref\n0 {n}\n").as_bytes());
        buf.extend_from_slice(b"0000000000 65535 f \n"); // 20-byte free entry
        for off in &offsets {
            buf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes()); // 20-byte entries
        }
        buf.extend_from_slice(
            format!("trailer\n<</Size {n}/Root 1 0 R>>\nstartxref\n{xref_off}\n%%EOF").as_bytes(),
        );
        buf
    }

    #[test]
    #[cfg(feature = "pdf")]
    fn renders_pages_honors_cap_and_reports_truncation() {
        let cfg = PdfConfig {
            max_pages: 2,
            long_edge: 256,
            max_dpi: 300.0,
        };
        // 3 portrait pages (200×300 pts) with a 2-page cap
        let out = render(&tiny_pdf(3, 200, 300), &cfg, PageSel::All).expect("render");
        assert_eq!(out.total_pages, 3, "all source pages counted");
        assert!(out.truncated, "3 pages under a cap of 2 must truncate");
        assert_eq!(out.pages.len(), 2, "cap honored");
        for p in &out.pages {
            assert!(p.w > 0 && p.h > 0);
            assert_eq!(p.rgb.len(), p.w * p.h * 3, "interleaved RGB8, w*h*3");
            assert!(p.h >= p.w, "portrait aspect preserved");
            // long edge scaled toward the 256 target (dpi well under the 300 cap)
            assert!(p.h <= 258, "long edge near target, got {}", p.h);
        }
    }

    #[test]
    #[cfg(feature = "pdf")]
    fn single_page_no_truncation() {
        let cfg = PdfConfig {
            max_pages: 20,
            long_edge: 512,
            max_dpi: 300.0,
        };
        let out = render(&tiny_pdf(1, 300, 200), &cfg, PageSel::All).expect("render");
        assert_eq!(out.total_pages, 1);
        assert!(!out.truncated);
        assert_eq!(out.pages.len(), 1);
        assert!(
            out.pages[0].w >= out.pages[0].h,
            "landscape aspect preserved"
        );
    }

    /// TEXT actually GETS DRAWN - the test the other render cases cannot do.
    ///
    /// Everything else here rasterizes BLANK pages and checks their geometry,
    /// which a build with broken font handling would pass without complaint.
    /// This one draws Helvetica through pdfium's font stack (on Windows that
    /// goes out to GDI, which is why the runner links gdi32) and looks at the
    /// pixels: there must be ink, it must not be a solid block, and it must be
    /// where we asked for it.
    ///
    /// Worth the extra assertions because our build turns things off -
    /// pdf_use_skia=false, use_custom_libcxx=false, no XFA, no V8 - and a
    /// rasterizer that silently renders nothing is exactly the failure those
    /// switches could produce.
    #[test]
    #[cfg(feature = "pdf")]
    fn text_is_really_rendered_not_a_blank_page() {
        let cfg = PdfConfig {
            max_pages: 4,
            long_edge: 792, // 1:1 with the 612x792 page, so PDF units == pixels
            max_dpi: 300.0,
        };
        // Drawn at (72, 720) in PDF space: 1 inch in from the left, and 720 up
        // from the BOTTOM, so it lands near the top-left of the raster.
        let doc = crate::doc::tests::text_pdf(&["HELLO PADDOCK"], "");
        let out = render(&doc, &cfg, PageSel::All).expect("render");
        let page = &out.pages[0];

        let ink = page
            .rgb
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|px| px[0] < 200)
            .count();
        let total = page.w * page.h;
        assert!(
            ink > 50,
            "no ink on the page - pdfium rendered nothing. {ink} dark pixels of {total}"
        );
        assert!(
            ink < total / 10,
            "page is mostly dark; that is not rendered text. {ink} of {total}"
        );

        // ...and in the right place. Row 0 is the TOP of the raster, and the
        // text sits 720/792 of the way up the page, so it belongs in the top
        // ~15% of rows. This is what separates "drew something" from "drew the
        // text where the content stream said to".
        let band_end = page.h * 15 / 100;
        let in_band = (0..band_end)
            .flat_map(|y| (0..page.w).map(move |x| (y, x)))
            .filter(|(y, x)| page.rgb[(y * page.w + x) * 3] < 200)
            .count();
        assert!(
            in_band * 2 > ink,
            "ink is not where the text was placed: {in_band} of {ink} dark pixels \
             in the top {band_end} rows"
        );
    }

    /// Bytes that are not a PDF are a Load error, not a panic or a hang.
    ///
    /// Replaces the old `not_configured_when_no_lib`, which asserted a state
    /// that can no longer be reached: pdfium is linked in, so "no library" is
    /// unrepresentable. This covers what is still true - and, incidentally,
    /// proves the statically linked pdfium really is callable, since a truncated
    /// header has to reach pdfium's parser to be rejected.
    #[test]
    #[cfg(feature = "pdf")]
    fn garbage_bytes_are_a_load_error() {
        let cfg = PdfConfig {
            max_pages: 4,
            long_edge: 512,
            max_dpi: 300.0,
        };
        assert!(matches!(
            render(b"%PDF-1.4", &cfg, PageSel::All),
            Err(PdfError::Load(_))
        ));
    }

    fn pdf_data_uri(bytes: &[u8]) -> String {
        use base64::Engine as _;
        format!(
            "data:application/pdf;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    #[test]
    fn has_pdf_parts_detects_all_shapes() {
        let uri = pdf_data_uri(b"%PDF-1.4");
        // OpenAI chat `file`
        let chat = json!([{"role":"user","content":[
            {"type":"file","file":{"filename":"a.pdf","file_data": uri}}]}]);
        // Responses `input_file`
        let resp = json!([{"role":"user","content":[
            {"type":"input_file","filename":"a.pdf","file_data": uri}]}]);
        // Anthropic `document` (raw base64 source)
        let anth = json!([{"role":"user","content":[
            {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]);
        // no PDF: a plain image part, and an image document (not application/pdf)
        let img = json!([{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]);
        let img_doc = json!([{"role":"user","content":[
            {"type":"document","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]);
        assert!(has_pdf_parts(chat.as_array().unwrap()));
        assert!(has_pdf_parts(resp.as_array().unwrap()));
        assert!(has_pdf_parts(anth.as_array().unwrap()));
        assert!(!has_pdf_parts(img.as_array().unwrap()));
        assert!(
            !has_pdf_parts(img_doc.as_array().unwrap()),
            "non-pdf document ignored"
        );
    }

    #[test]
    #[cfg(feature = "pdf")]
    fn expands_pdf_part_into_page_images_with_truncation() {
        let cfg = PdfConfig {
            max_pages: 2,
            long_edge: 256,
            max_dpi: 300.0,
        };
        let uri = pdf_data_uri(&tiny_pdf(3, 200, 300));
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "summarize this"},
                {"type": "input_file", "filename": "doc.pdf", "file_data": uri},
            ]
        })];
        let (out, summary) =
            expand_in_messages(messages, &cfg, true, 8192, None, None, true, "test", false)
                .expect("expand");
        assert_eq!(summary.pdfs, 1);
        assert_eq!(summary.total_pages, 3);
        assert_eq!(summary.rendered_pages, 2);
        assert!(summary.truncated);

        let parts = out[0]["content"].as_array().unwrap();
        // original text + truncation note + 2×([page k] marker + image) = 6
        assert_eq!(parts.len(), 6, "parts: {parts:#?}");
        assert_eq!(parts[0]["text"], "summarize this");
        assert!(
            parts[1]["text"].as_str().unwrap().contains("3 pages"),
            "truncation note names the total"
        );
        // the PDF part is gone; two image_url parts took its place
        let imgs = parts
            .iter()
            .filter(|p| p.get("image_url").is_some())
            .count();
        assert_eq!(imgs, 2, "one image_url per rendered page");
        assert!(
            parts.iter().all(|p| pdf_part(p).is_none()),
            "no PDF parts remain"
        );
        // find_images must now see exactly the two pages
        let urls = crate::chat::find_images(&out).expect("find");
        assert_eq!(urls.len(), 2);
    }

    /// The deepseek2-ocr route: bare page images (no note, no `[page k]`
    /// markers - the body must stay empty so the derived canonical task
    /// string engages), a user page selection is honored quietly, and a
    /// server-ceiling clip is a loud error instead of the disclosure note.
    #[test]
    #[cfg(feature = "pdf")]
    fn plain_pages_emits_bare_images_and_refuses_the_ceiling_clip() {
        let cfg = PdfConfig {
            max_pages: 4,
            long_edge: 256,
            max_dpi: 300.0,
        };
        let uri = pdf_data_uri(&tiny_pdf(3, 200, 300));
        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "input_file", "filename": "doc.pdf", "file_data": uri}]
        })];
        let (out, summary) =
            expand_in_messages(messages, &cfg, true, 8192, None, None, true, "test", true)
                .expect("expand");
        assert_eq!(summary.rendered_pages, 3);
        let parts = out[0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 3, "three bare page images: {parts:#?}");
        assert!(
            parts.iter().all(|p| p.get("image_url").is_some()),
            "no text parts at all"
        );

        // the user's own selection is intent, not truncation - still quiet
        let uri = pdf_data_uri(&tiny_pdf(3, 200, 300));
        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "input_file", "filename": "doc.pdf", "file_data": uri,
                         "pages": "2-3"}]
        })];
        let (out, _) =
            expand_in_messages(messages, &cfg, true, 8192, None, None, true, "test", true)
                .expect("expand");
        assert_eq!(
            out[0]["content"].as_array().unwrap().len(),
            2,
            "pages 2-3 only"
        );

        // the SERVER ceiling cutting below the ask has no in-band disclosure
        // on this route - loud error naming the cap
        let tight = PdfConfig {
            max_pages: 2,
            long_edge: 256,
            max_dpi: 300.0,
        };
        let uri = pdf_data_uri(&tiny_pdf(3, 200, 300));
        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "input_file", "filename": "doc.pdf", "file_data": uri}]
        })];
        let err = expand_in_messages(messages, &tight, true, 8192, None, None, true, "test", true)
            .unwrap_err();
        assert!(
            err.contains("doc.pdf") && err.contains("at most 2"),
            "{err}"
        );
        // and the same 3 pages under the tight ceiling on the CHAT route
        // still take the note path (regression guard for the split)
        let uri = pdf_data_uri(&tiny_pdf(3, 200, 300));
        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "input_file", "filename": "doc.pdf", "file_data": uri}]
        })];
        let (out, _) = expand_in_messages(
            messages, &tight, true, 8192, None, None, true, "test", false,
        )
        .expect("chat route discloses instead");
        let texts: Vec<&str> = out[0]["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("Only pages 1-2 of 3")),
            "{texts:?}"
        );
    }

    #[test]
    fn non_pdf_messages_pass_through_untouched() {
        let cfg = PdfConfig {
            max_pages: 4,
            long_edge: 512,
            max_dpi: 300.0,
        };
        let messages = vec![json!({"role":"user","content":[{"type":"text","text":"hi"}]})];
        let (out, summary) = expand_in_messages(
            messages.clone(),
            &cfg,
            true,
            8192,
            None,
            None,
            false,
            "test",
            false,
        )
        .expect("expand");
        assert!(!summary.any());
        assert_eq!(
            out, messages,
            "no PDF => identical messages (render never called)"
        );
    }

    /// Two files in one prompt, each with its own part-level cap - the flat
    /// `max_pages` key on the part beats the request-level extension. Runs on
    /// the text route (can_render=false), so no pdfium needed.
    #[test]
    #[cfg(feature = "pdf")]
    fn part_level_max_pages_overrides_per_file() {
        let cfg = PdfConfig {
            max_pages: 20,
            long_edge: 512,
            max_dpi: 300.0,
        };
        let uri = pdf_data_uri(&crate::doc::tests::text_pdf(&["One", "Two", "Three"], ""));
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "input_file", "filename": "a.pdf", "file_data": uri, "max_pages": 1},
                {"type": "input_file", "filename": "b.pdf", "file_data": uri},
            ]
        })];
        // request-level cap 2: file a's own cap 1 wins for a, 2 applies to b
        let (out, summary) = expand_in_messages(
            messages,
            &cfg,
            false,
            8192,
            Some(2),
            None,
            false,
            "test",
            false,
        )
        .expect("expand");
        assert_eq!(summary.pdfs, 2);
        assert_eq!(summary.total_pages, 6);
        assert_eq!(
            summary.rendered_pages,
            1 + 2,
            "per-file caps: 1 for a, request's 2 for b"
        );
        assert!(summary.truncated);
        let parts = out[0]["content"].as_array().unwrap();
        let texts: Vec<&str> = parts.iter().filter_map(|p| p["text"].as_str()).collect();
        assert!(
            texts[0].contains("Only page 1 of 3"),
            "a capped to its own 1: {}",
            texts[0]
        );
        assert!(
            texts[1].contains("Only pages 1-2 of 3"),
            "b capped to the request's 2: {}",
            texts[1]
        );
    }

    #[test]
    fn pages_grammar_parses_and_refuses() {
        assert_eq!(parse_pages("all").unwrap(), PageSel::All);
        assert_eq!(parse_pages("ALL").unwrap(), PageSel::All);
        assert_eq!(parse_pages("3").unwrap(), PageSel::Range(3, 3));
        assert_eq!(parse_pages("2-4").unwrap(), PageSel::Range(2, 4));
        assert_eq!(parse_pages(" 2 - 4 ").unwrap(), PageSel::Range(2, 4));
        assert_eq!(parse_pages("2-").unwrap(), PageSel::Range(2, usize::MAX));
        assert_eq!(parse_pages("-4").unwrap(), PageSel::Range(1, 4));
        for bad in ["", "0", "4-2", "x", "1-2-3", "2..4"] {
            assert!(parse_pages(bad).is_err(), "{bad:?} must be refused");
        }
    }

    /// A part-level `pages` RANGE beats the caps and keeps real page numbers
    /// on the text route.
    #[test]
    #[cfg(feature = "pdf")]
    fn part_level_pages_range_slices_the_middle() {
        let cfg = PdfConfig {
            max_pages: 20,
            long_edge: 512,
            max_dpi: 300.0,
        };
        let uri = pdf_data_uri(&crate::doc::tests::text_pdf(&["One", "Two", "Three"], ""));
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "input_file", "filename": "a.pdf", "file_data": uri,
                 "pages": "2-3", "max_pages": 1},
            ]
        })];
        // pages beats both the part's own max_pages and the request's cap
        let (out, summary) = expand_in_messages(
            messages,
            &cfg,
            false,
            8192,
            Some(1),
            None,
            false,
            "test",
            false,
        )
        .expect("expand");
        assert_eq!(summary.rendered_pages, 2);
        let text = out[0]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("Only pages 2-3 of 3"), "{text}");
        assert!(text.contains("[page 2]"), "real page numbers: {text}");
        assert!(!text.contains("One"), "page 1 skipped: {text}");
    }

    /// A part-level "render" the server cannot honor is an error NAMING the
    /// file - never a silent downgrade to text.
    #[test]
    fn part_level_render_refused_names_the_file() {
        let cfg = PdfConfig {
            max_pages: 20,
            long_edge: 512,
            max_dpi: 300.0,
        };
        let uri = pdf_data_uri(&tiny_pdf(1, 200, 300));
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "input_file", "filename": "scan.pdf", "file_data": uri, "pdf_mode": "render"},
            ]
        })];
        let err = expand_in_messages(
            messages, &cfg, false, 8192, None, None, false, "no tower", false,
        )
        .unwrap_err();
        assert!(err.contains("scan.pdf"), "{err}");
        assert!(err.contains("no tower"), "{err}");
        // and a junk part-level value is refused, not ignored
        let uri = pdf_data_uri(&tiny_pdf(1, 200, 300));
        let bad = vec![json!({"role":"user","content":[
            {"type": "input_file", "filename": "x.pdf", "file_data": uri, "max_pages": 0}]})];
        let err = expand_in_messages(bad, &cfg, false, 8192, None, None, false, "test", false)
            .unwrap_err();
        assert!(err.contains("positive integer"), "{err}");
    }
}
