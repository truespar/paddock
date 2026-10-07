//! POST /v1/masks - promptable segmentation (Meta's SAM 3): a picture and a
//! prompt in, masks out, each with its score, its box in the picture's pixels
//! and its mask as COCO RLE. Two tasks, told apart by what the prompt holds:
//!
//! - a CONCEPT in words (`text`), exemplar boxes (`boxes`, in the picture's
//!   pixels, positive or negative), or both: every instance the model keeps,
//!   best first. A box-only prompt is encoded with the word "visual", as
//!   Meta's processor does it. A concept past 30 tokens is refused rather
//!   than truncated (Meta truncates silently).
//! - CLICKS (`points`, positive or negative) and/or an `object_box`: the one
//!   object under them, as up to three candidate masks best first, scored by
//!   the model's predicted IoU; `refine_id` refines the answer next time.
//!
//! The picture arrives as a `data:` URI and is decoded the way the reference
//! decoded it (JPEG through `paddock_jpeg`, libjpeg-turbo's bytes), upright by
//! EXIF so the masks line up with the picture as the caller sees it. The
//! resize to the model's 1008 x 1008 happens on the device, to the bit of
//! torchvision's.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use paddock_api::ErrorBody;
use paddock_api::masks::{MaskInstance, MaskRequest, MaskResponse, MaskTimings, Rle};
use paddock_engine::gpu_model::sam3::{
    PVS_MAX_POINTS, Sam3Box, Sam3Click, Sam3PointRequest, Sam3Request,
};
use paddock_engine::masks::{MaskError, MaskPrompt, MaskRequest as EngineRequest};
use paddock_tokenizer::sam3::{Sam3TokenizeError, Sam3Tokens};

use crate::routes::AppState;

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

fn bad(msg: impl Into<String>) -> Response {
    err(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

fn server_error(msg: impl Into<String>) -> Response {
    err(StatusCode::INTERNAL_SERVER_ERROR, "server_error", msg)
}

/// A box in the picture's pixels that is a box: finite, x0 < x1, y0 < y1.
fn valid_box(b: &[f32; 4]) -> bool {
    let [x0, y0, x1, y1] = *b;
    x1 > x0 && y1 > y0 && b.iter().all(|v| v.is_finite())
}

/// The prompt as checked, before the picture is decoded.
enum Task {
    Concept { tokens: Sam3Tokens, threshold: f32 },
    Points,
}

pub async fn handle(State(state): State<Arc<AppState>>, Json(req): Json<MaskRequest>) -> Response {
    let Some(model) = state.masker.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no promptable segmentation model is loaded (start paddock with a SAM 3 \
             checkpoint directory as `model`)",
        );
    };
    if let Some(m) = req.model.as_deref()
        && m != model.id
    {
        return err(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("this endpoint serves `{}`, not `{m}`", model.id),
        );
    }
    let info = model.masker.info().clone();
    let text = req.text.as_deref().map(str::trim).filter(|t| !t.is_empty());
    let clicks = !req.points.is_empty() || req.object_box.is_some();

    let task = if clicks {
        if text.is_some() || !req.boxes.is_empty() {
            return bad(
                "clicks (`points` / `object_box`) and a concept (`text` / `boxes`) are two \
                 different tasks - send one of them",
            );
        }
        if !info.clicks {
            return bad(
                "this endpoint's kernel pack predates click prompts - update it to use them",
            );
        }
        if req.points.len() > PVS_MAX_POINTS {
            return bad(format!(
                "{} clicks; this endpoint takes at most {PVS_MAX_POINTS}",
                req.points.len()
            ));
        }
        if let Some(p) = req
            .points
            .iter()
            .find(|p| p.xy.iter().any(|v| !v.is_finite()))
        {
            return bad(format!("click {:?} is not a point", p.xy));
        }
        if let Some(b) = req.object_box.as_ref().filter(|b| !valid_box(b)) {
            return bad(format!("object_box {b:?} is not x0 < x1, y0 < y1"));
        }
        Task::Points
    } else {
        let threshold = req.threshold.unwrap_or(0.5);
        if !(0.0..1.0).contains(&threshold) {
            return bad(format!("threshold {threshold} (want 0 <= t < 1)"));
        }
        if text.is_none() && req.boxes.is_empty() {
            return bad(
                "give a concept (`text`), at least one exemplar box (`boxes`), or both - or \
                 click on one object (`points`, `object_box`)",
            );
        }
        if req.boxes.len() > info.max_boxes {
            return bad(format!(
                "{} exemplar boxes; this endpoint takes at most {}",
                req.boxes.len(),
                info.max_boxes
            ));
        }
        if let Some(b) = req.boxes.iter().find(|b| !valid_box(&b.xyxy)) {
            return bad(format!("box {:?} is not x0 < x1, y0 < y1", b.xyxy));
        }
        let tokens = match model.tokenizer.encode(text.unwrap_or("visual")) {
            Ok(t) => t,
            Err(e @ Sam3TokenizeError::TooLong { .. }) => return bad(e.to_string()),
            Err(e) => return server_error(e.to_string()),
        };
        Task::Concept { tokens, threshold }
    };
    let refine = match req.refine.as_deref() {
        None => None,
        Some(_) if !clicks => {
            return bad("`refine` refines a click answer - send it with clicks");
        }
        Some(r) => match r.parse::<u64>() {
            Ok(id) => Some(id),
            Err(_) => {
                return bad(format!(
                    "refine {r:?} is not a refine_id this endpoint gave out"
                ));
            }
        },
    };

    // decode off the async runtime: a 24 MP JPEG is tens of milliseconds
    let image = req.image;
    let max_px = info.max_pixels as u64;
    let decoded = tokio::task::spawn_blocking(move || {
        crate::reference_image::decode_image_url_reference_limited(&image, max_px)
    })
    .await;
    let (rgb, width, height) = match decoded {
        Ok(Ok(d)) => d,
        Ok(Err(e)) => return bad(e),
        Err(e) => return server_error(e.to_string()),
    };

    let prompt = match task {
        Task::Concept { tokens, threshold } => {
            // exemplar boxes: the picture's xyxy pixels -> Meta's normalized cxcywh
            let (wf, hf) = (width as f32, height as f32);
            let boxes = req
                .boxes
                .iter()
                .map(|b| {
                    let [x0, y0, x1, y1] = b.xyxy;
                    Sam3Box {
                        cx: (x0 + x1) * 0.5 / wf,
                        cy: (y0 + y1) * 0.5 / hf,
                        w: (x1 - x0) / wf,
                        h: (y1 - y0) / hf,
                        positive: b.positive,
                    }
                })
                .collect();
            MaskPrompt::Concept(Sam3Request {
                ids: tokens.ids,
                valid: tokens.valid,
                boxes,
                threshold,
            })
        }
        Task::Points => MaskPrompt::Points(Sam3PointRequest {
            clicks: req
                .points
                .iter()
                .map(|p| Sam3Click {
                    x: p.xy[0],
                    y: p.xy[1],
                    positive: p.positive,
                })
                .collect(),
            bbox: req.object_box,
            multimask: req.multimask,
            refine,
        }),
    };

    let out = match model
        .masker
        .segment(EngineRequest {
            rgb,
            width,
            height,
            prompt,
        })
        .await
    {
        Ok(o) => o,
        Err(MaskError::Request(m)) => return bad(m),
        // a picture request never touches a session
        Err(e @ (MaskError::NoSession(_) | MaskError::Busy(_))) => {
            return server_error(format!("{e:?}"));
        }
        Err(MaskError::Engine(m)) => return server_error(m),
    };
    let take = req.max_instances.unwrap_or(usize::MAX);
    let instances = out
        .instances
        .into_iter()
        .take(take)
        .map(|i| MaskInstance {
            score: i.score,
            xyxy: i.bbox,
            area: i.area,
            mask: Rle {
                size: [out.height, out.width],
                counts: i.rle,
            },
        })
        .collect();
    Json(MaskResponse {
        object: "masks".into(),
        model: model.id.clone(),
        width: out.width,
        height: out.height,
        presence: out.presence,
        instances,
        prompt_tokens: out.tokens,
        timings: MaskTimings {
            resize_ms: out.timings.resize_ms,
            encode_ms: out.timings.encode_ms,
            prompt_ms: out.timings.prompt_ms,
            detect_ms: out.timings.detect_ms,
            masks_ms: out.timings.masks_ms,
            image_reused: out.timings.image_reused,
            text_reused: out.timings.text_reused,
        },
        refine_id: out.refine_id.map(|id| id.to_string()),
    })
    .into_response()
}
