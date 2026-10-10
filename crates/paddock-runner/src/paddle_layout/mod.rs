//! PaddleOCR-VL's document pipeline - `ocr.mode = "document"`: the page
//! read the way the checkpoint's authors read it (PaddleX's PaddleOCR-VL 1.6
//! pipeline), not as one whole-page decode.
//!
//! 1. PP-DocLayoutV3 (the companion, `paddock_engine::doclayout`) finds the
//!    page's regions, labels them and orders them;
//! 2. the boxes are filtered, cropped and the adjacent text blocks merged
//!    (`prep`);
//! 3. every region that is not a picture is read by the recognizer with its
//!    label's task prompt - all of a page's regions submitted together, so
//!    they batch;
//! 4. each answer is cleaned (`text`) and the page assembled as Markdown in
//!    reading order (`markdown`), pictures as references to their crops.
//!
//! Each region is a request of its own to the engine with the pipeline's
//! own settings: greedy, 4,096 new tokens (or the caller's `max_tokens`,
//! applied per region), the default pixel budget, and this family's
//! repetition stop. The response is one chat completion: the Markdown as
//! content, usage summed over the regions, and the `ocr` echo carrying every
//! region (label, box, text). Streaming sends each block as soon as it and
//! everything before it is read.

mod markdown;
mod prep;
mod respond;
#[cfg(test)]
mod tests;
mod text;

pub(crate) use respond::respond;

use std::sync::Arc;

use image::RgbImage;
use paddock_engine::sampler::SamplingParams;
use paddock_engine::service::{FinishReason, GenRequest, MmChunk, TokenEvent};
use serde_json::{Value, json};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::routes::AppState;
use crate::serving::ServingModel;

/// The pipeline's own `max_new_tokens` for a region.
pub(crate) const REGION_TOKENS: usize = 4096;

/// The repetition stop each region reads under - the family's default.
const REPEAT_STOP: (usize, usize, bool) = (35, 128, true);

/// One decoded page.
pub(crate) struct Page {
    pub rgb: Vec<u8>,
    pub w: usize,
    pub h: usize,
}

/// What a document read reports as it goes.
pub(crate) enum DocEvent {
    /// the next piece of Markdown, its separator included
    Text(String),
    Done(Summary),
    Error(String),
}

/// The whole read's facts, for usage and the `ocr` echo.
pub(crate) struct Summary {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub regions: Vec<Value>,
    /// regions the repetition stop ended
    pub stopped: usize,
    /// regions cut at the token cap
    pub truncated: usize,
    pub layout_ms: f64,
}

/// A region's request in flight.
struct Pending {
    label: &'static str,
    bbox: [i32; 4],
    /// a merged group's first block (its crop carries the group's text)
    group: Option<usize>,
    /// why a readable region was not read
    unread: Option<&'static str>,
    /// its outline in page pixels
    polygon: Option<paddock_engine::gpu_model::doclayout::polygon::Polygon>,
    rx: Option<UnboundedReceiver<TokenEvent>>,
    prompt: usize,
}

/// A page's regions in flight, in reading order.
struct PagePlan {
    page: usize,
    w: usize,
    h: usize,
    blocks: Vec<Pending>,
}

/// The prompt ids of one task over one image slot, rendered through the
/// model's own template exactly as a chat request for it would be.
fn task_ids(model: &ServingModel, task: &str) -> Result<Vec<u32>, String> {
    let template = model
        .chat_template
        .as_deref()
        .ok_or("this model has no chat template")?;
    let messages = [json!({"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": task},
    ]})];
    let kwargs = model.template_defaults(None);
    let prompt = crate::chat_template::render_with_specials(
        template,
        &messages,
        None,
        kwargs.as_ref(),
        &model.template_specials(),
    )?;
    let mut ids = model.tokenizer.encode(&prompt).map_err(|e| e.to_string())?;
    if let Some(bos) = model.bos
        && ids.first() != Some(&bos)
    {
        ids.insert(0, bos);
    }
    Ok(ids)
}

/// Submit one region: its crop under its task's prompt.
fn submit(
    model: &ServingModel,
    ids: &[u32],
    crop: RgbImage,
    cap: usize,
    max_ctx: usize,
) -> Result<(UnboundedReceiver<TokenEvent>, usize), String> {
    let pad = model.image_pad_id.ok_or("model has no image pad token")?;
    let (w, h) = (crop.width() as usize, crop.height() as usize);
    let media = vec![MmChunk::Image {
        rgb: crop.into_raw(),
        w,
        h,
    }];
    let mm = crate::chat::build_mm_chunks(ids, pad, media, None)?;
    if let Some(e) = crate::chat::context_gate(model, mm.text_ids.len(), Some(&mm.chunks), max_ctx)
    {
        return Err(format!("a region does not fit the context window: {e}"));
    }
    let (tx, rx) = unbounded_channel();
    let prompt = mm.text_ids.len();
    model.engine.submit(GenRequest {
        prompt: mm.text_ids,
        max_tokens: cap,
        sampler: SamplingParams {
            no_repeat_ngram: REPEAT_STOP,
            ..SamplingParams::default()
        },
        stop_tokens: model.stop_tokens.clone(),
        events: tx,
        mm_chunks: Some(mm.chunks),
        constraint: None,
        logprobs: None,
        submitted: None,
        canvas_read: None,
        user_turn: false,
    })?;
    Ok((rx, prompt))
}

/// Detect, prepare and submit every page in order, handing each page's
/// plan on as soon as its regions are in the engine.
async fn produce(
    state: &AppState,
    pages: Vec<Page>,
    cap: usize,
    plans: UnboundedSender<Result<PagePlan, String>>,
) -> f64 {
    let Some(model) = state.serving.as_ref() else {
        return 0.0;
    };
    let Some(layout) = model.layout.as_ref() else {
        let _ = plans.send(Err("no layout model is loaded".into()));
        return 0.0;
    };
    let mut ids: Vec<(&'static str, Vec<u32>)> = Vec::new();
    let mut layout_ms = 0.0;
    for (p, page) in pages.into_iter().enumerate() {
        let t0 = std::time::Instant::now();
        let boxes = match layout.page(page.rgb.clone(), page.w, page.h).await {
            Ok(b) => b,
            Err(e) => {
                let _ = plans.send(Err(format!("layout detection failed: {e}")));
                return layout_ms;
            }
        };
        layout_ms += t0.elapsed().as_secs_f64() * 1e3;
        let Some(img) = RgbImage::from_raw(page.w as u32, page.h as u32, page.rgb) else {
            let _ = plans.send(Err("page buffer does not match its size".into()));
            return layout_ms;
        };
        let kept = prep::filter_overlap(&boxes);
        let blocks = prep::merge(prep::blocks(&img, &kept));
        let mut plan = PagePlan {
            page: p,
            w: page.w,
            h: page.h,
            blocks: Vec::with_capacity(blocks.len()),
        };
        for b in &blocks {
            let mut pending = Pending {
                label: b.label,
                bbox: b.bbox,
                group: b.group,
                unread: None,
                polygon: b.polygon.clone(),
                rx: None,
                prompt: 0,
            };
            if let Some((crop, task)) = prep::request(b) {
                // the recognizer's resize refuses past 200:1 (the reference
                // raises and loses the page); the region stays unread
                let (cw, ch) = (crop.width().max(1), crop.height().max(1));
                if cw.max(ch) as f64 / cw.min(ch) as f64 > 200.0 {
                    pending.unread = Some("aspect ratio past 200:1");
                    plan.blocks.push(pending);
                    continue;
                }
                if !ids.iter().any(|(t, _)| *t == task) {
                    match task_ids(model, task) {
                        Ok(v) => ids.push((task, v)),
                        Err(e) => {
                            let _ = plans.send(Err(e));
                            return layout_ms;
                        }
                    }
                }
                let task_prompt = &ids.iter().find(|(t, _)| *t == task).expect("rendered").1;
                match submit(model, task_prompt, crop, cap, state.max_ctx) {
                    Ok((rx, n)) => {
                        pending.rx = Some(rx);
                        pending.prompt = n;
                    }
                    Err(e) => {
                        let _ = plans.send(Err(e));
                        return layout_ms;
                    }
                }
            }
            plan.blocks.push(pending);
        }
        if plans.send(Ok(plan)).is_err() {
            // the reader hung up: dropping the plan cancels its regions
            return layout_ms;
        }
    }
    layout_ms
}

/// The wire's 0..999 grid (the shared `regions` space of every parser).
fn grid(v: i32, side: usize) -> i64 {
    ((i64::from(v) * 999 + side as i64 / 2) / side.max(1) as i64).clamp(0, 999)
}

/// One region on the wire: the shared `regions` shape (label, the box on
/// the 0..999 grid, text) plus its page, its box and outline in page pixels,
/// a picture's crop name and a merged block's group.
fn region_json(plan: &PagePlan, p: &Pending, content: String) -> Value {
    let b = p.bbox;
    let picture = prep::prompt(p.label).is_none();
    let mut v = crate::deepseek_ocr::Region {
        label: p.label.to_owned(),
        boxes: vec![[
            grid(b[0], plan.w),
            grid(b[1], plan.h),
            grid(b[2], plan.w),
            grid(b[3], plan.h),
        ]],
        text: (!picture).then_some(content),
        quads: Vec::new(),
        continues: false,
    }
    .to_json();
    v["page"] = json!(plan.page);
    v["bbox"] = json!(b);
    if let Some(p) = &p.polygon {
        v["polygon"] = json!(p);
    }
    if picture {
        v["image"] = json!(markdown::image_path(p.label, b));
    }
    if let Some(g) = p.group {
        v["group"] = json!(g);
    }
    if let Some(why) = p.unread {
        v["unread"] = json!(why);
    }
    v
}

/// Start a document read: the Markdown and the summary arrive on the
/// returned channel in reading order.
pub(crate) fn start(
    state: Arc<AppState>,
    pages: Vec<Page>,
    cap: usize,
) -> UnboundedReceiver<DocEvent> {
    let (out, rx) = unbounded_channel();
    let (plan_tx, mut plan_rx) = unbounded_channel();
    let producer_state = state.clone();
    let producer = tokio::spawn(async move { produce(&producer_state, pages, cap, plan_tx).await });
    tokio::spawn(async move {
        let Some(model) = state.serving.as_ref() else {
            return;
        };
        let mut sum = Summary {
            prompt_tokens: 0,
            completion_tokens: 0,
            regions: Vec::new(),
            stopped: 0,
            truncated: 0,
            layout_ms: 0.0,
        };
        while let Some(plan) = plan_rx.recv().await {
            let mut plan = match plan {
                Ok(p) => p,
                Err(e) => {
                    let _ = out.send(DocEvent::Error(e));
                    return;
                }
            };
            // pages join with a blank line; within a page a block's piece is
            // preceded by one once the page's Markdown is non-empty (an empty
            // piece - a merged block's twin - leaves an empty page empty)
            let mut page_open = false;
            if plan.page > 0 && out.send(DocEvent::Text("\n\n".into())).is_err() {
                return;
            }
            for mut pending in std::mem::take(&mut plan.blocks) {
                let mut content = String::new();
                if let Some(mut rx) = pending.rx.take() {
                    let (mut ids, mut rows, mut reason) =
                        (Vec::new(), 0usize, FinishReason::Length);
                    let mut terminal = 0;
                    while let Some(ev) = rx.recv().await {
                        match ev {
                            TokenEvent::Prefilled { rows: r, .. } => rows = rows.max(r as usize),
                            TokenEvent::Token { id, .. } => ids.push(id),
                            TokenEvent::Done(r, stats) => {
                                reason = r;
                                terminal = stats.terminal_tokens();
                                break;
                            }
                            TokenEvent::Error(e) => {
                                let _ = out.send(DocEvent::Error(e.to_string()));
                                return;
                            }
                        }
                    }
                    sum.prompt_tokens += pending.prompt.max(rows);
                    sum.completion_tokens += ids.len() + terminal;
                    sum.stopped += usize::from(reason == FinishReason::Repetition);
                    sum.truncated += usize::from(reason == FinishReason::Length);
                    let raw = model.tokenizer.decode(&ids, true).unwrap_or_default();
                    content = text::clean(pending.label, raw.trim());
                }
                let block = markdown::MdBlock {
                    label: pending.label,
                    content: &content,
                    bbox: pending.bbox,
                };
                if let Some(piece) = markdown::render(&block, plan.w) {
                    let sep = if page_open { "\n\n" } else { "" };
                    page_open |= !piece.is_empty();
                    if out.send(DocEvent::Text(format!("{sep}{piece}"))).is_err() {
                        return;
                    }
                }
                sum.regions.push(region_json(&plan, &pending, content));
            }
        }
        sum.layout_ms = producer.await.unwrap_or(0.0);
        let _ = out.send(DocEvent::Done(sum));
    });
    rx
}
