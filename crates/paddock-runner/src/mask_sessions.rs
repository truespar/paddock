//! `/v1/masks/sessions` - a concept tracked through a video sent a frame at a
//! time (Meta's SAM 3 video predictor: its detector on every frame, its
//! tracker carrying each object from frame to frame under one id).
//!
//! - `POST /v1/masks/sessions` `{text | texts, frames?, preview?}` starts
//!   one - `texts` tracks up to four concepts, the frame read once;
//! - `POST /v1/masks/sessions/{id}/frames` `{image}` sends the next frame
//!   (every frame the first one's size) and answers with the frames whose
//!   output is final now and, unless the session declined it, the frame
//!   just sent as it stands;
//! - `POST /v1/masks/sessions/{id}/finish` ends the video: the frames still
//!   held, final;
//! - `DELETE /v1/masks/sessions/{id}` drops it.
//!
//! An output is final once Meta's hot start has passed over it, 15 frames
//! later: a new object that turns out to be noise in that window is removed
//! from the frames it was seen on too. The preview is what a live view
//! shows meanwhile. A session lives on the model's thread; four can be live
//! at once, and one idle for two minutes is dropped.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use paddock_api::ErrorBody;
use paddock_api::masks::{
    MaskFinishResponse, MaskFrameRequest, MaskFrameResponse, MaskSessionRequest,
    MaskSessionResponse, MaskVideoFrame, MaskVideoObject, Rle,
};
use paddock_engine::gpu_model::sam3::{HOTSTART_DELAY, MAX_CONCEPTS, Sam3VideoFrame};
use paddock_engine::masks::{MaskError, SESSION_IDLE, VideoParams};
use paddock_tokenizer::sam3::Sam3TokenizeError;

use crate::routes::AppState;
use crate::serving::MaskModel;

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

fn bad(msg: impl Into<String>) -> Response {
    err(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

fn engine_err(e: MaskError) -> Response {
    match e {
        MaskError::Request(m) => bad(m),
        MaskError::NoSession(id) => err(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!(
                "no video session {id} - it was never started, has finished, or sat idle \
                 past {} s",
                SESSION_IDLE.as_secs()
            ),
        ),
        MaskError::Busy(m) => err(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", m),
        MaskError::Engine(m) => err(StatusCode::INTERNAL_SERVER_ERROR, "server_error", m),
    }
}

fn model(state: &AppState) -> Result<&MaskModel, Box<Response>> {
    state.masker.as_ref().ok_or_else(|| {
        Box::new(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no promptable segmentation model is loaded (start paddock with a SAM 3 \
             checkpoint directory as `model`)",
        ))
    })
}

fn session_id(id: &str) -> Result<u64, Box<Response>> {
    id.parse::<u64>().map_err(|_| {
        Box::new(err(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("{id:?} is not a video session this endpoint started"),
        ))
    })
}

/// The engine's frame as the wire's: boxes as pixel edges (the extremes are
/// inclusive), the masks COCO RLE at the video's size.
fn wire_frame(f: Sam3VideoFrame) -> MaskVideoFrame {
    let size = [f.height, f.width];
    MaskVideoFrame {
        frame: f.frame,
        objects: f
            .objects
            .into_iter()
            .map(|o| {
                let [x0, y0, x1, y1] = o.bbox_px;
                MaskVideoObject {
                    id: o.id,
                    concept: o.concept,
                    score: o.prob,
                    xyxy: [x0 as f32, y0 as f32, (x1 + 1) as f32, (y1 + 1) as f32],
                    area: o.area,
                    mask: Rle {
                        size,
                        counts: o.rle,
                    },
                }
            })
            .collect(),
    }
}

pub async fn create(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MaskSessionRequest>,
) -> Response {
    let model = match model(&state) {
        Ok(m) => m,
        Err(r) => return *r,
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
    if let Some(why) = model.masker.info().video_off.as_deref() {
        return bad(why.to_owned());
    }
    let texts: Vec<String> = match (req.text, req.texts) {
        (Some(_), Some(_)) => return bad("give `text` or `texts`, not both"),
        (Some(t), None) => vec![t],
        (None, Some(t)) => t,
        (None, None) => {
            return bad(
                "a video session tracks a concept: give it `text` (\"person\", \"red car\")",
            );
        }
    };
    let texts: Vec<String> = texts.iter().map(|t| t.trim().to_owned()).collect();
    if texts.is_empty() || texts.len() > MAX_CONCEPTS {
        return bad(format!(
            "{} concepts - a session tracks 1 to {MAX_CONCEPTS}",
            texts.len()
        ));
    }
    if let Some(i) = texts.iter().position(|t| t.is_empty()) {
        return bad(format!("concept {i} is empty"));
    }
    if req.frames == Some(0) {
        return bad("`frames` is the video's length - at least 1");
    }
    let mut concepts = Vec::with_capacity(texts.len());
    for text in &texts {
        match model.tokenizer.encode(text) {
            Ok(t) => concepts.push((t.ids, t.valid)),
            Err(e @ Sam3TokenizeError::TooLong { .. }) => return bad(e.to_string()),
            Err(e) => {
                return err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    e.to_string(),
                );
            }
        }
    }
    let id = match model
        .masker
        .video_start(VideoParams {
            concepts,
            num_frames: req.frames,
            preview: req.preview.unwrap_or(true),
        })
        .await
    {
        Ok(id) => id,
        Err(e) => return engine_err(e),
    };
    Json(MaskSessionResponse {
        object: "masks.session".into(),
        id: id.to_string(),
        model: model.id.clone(),
        concepts: texts,
        hold_frames: HOTSTART_DELAY,
        idle_seconds: SESSION_IDLE.as_secs(),
    })
    .into_response()
}

pub async fn frame(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<MaskFrameRequest>,
) -> Response {
    let model = match model(&state) {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let id = match session_id(&id) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    // decode off the async runtime, as a picture request does
    let image = req.image;
    let max_px = model.masker.info().max_pixels as u64;
    let decoded = tokio::task::spawn_blocking(move || {
        crate::reference_image::decode_image_url_reference_limited(&image, max_px)
    })
    .await;
    let (rgb, width, height) = match decoded {
        Ok(Ok(d)) => d,
        Ok(Err(e)) => return bad(e),
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                e.to_string(),
            );
        }
    };
    let out = match model.masker.video_frame(id, rgb, width, height).await {
        Ok(o) => o,
        Err(e) => return engine_err(e),
    };
    Json(MaskFrameResponse {
        object: "masks.frame".into(),
        frame: out.frame,
        width,
        height,
        frames: out.frames.into_iter().map(wire_frame).collect(),
        preview: out.preview.map(wire_frame),
        ms: out.ms,
    })
    .into_response()
}

pub async fn finish(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let model = match model(&state) {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let id = match session_id(&id) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match model.masker.video_finish(id).await {
        Ok(frames) => Json(MaskFinishResponse {
            object: "masks.frames".into(),
            frames: frames.into_iter().map(wire_frame).collect(),
        })
        .into_response(),
        Err(e) => engine_err(e),
    }
}

pub async fn delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let model = match model(&state) {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let id = match session_id(&id) {
        Ok(id) => id,
        Err(r) => return *r,
    };
    match model.masker.video_drop(id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => engine_err(e),
    }
}
