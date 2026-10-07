//! Promptable-segmentation wire types - `POST /v1/masks` (SAM 3).
//!
//! A Paddock surface: no OpenAI or Anthropic API asks "find every instance of
//! this in the picture". The request is one picture (a `data:` URI, like every
//! image input here) and one of two prompts:
//!
//! - a CONCEPT - words (`text`), exemplar boxes (`boxes`), or both: every
//!   instance the model keeps comes back, best first, each with its score,
//!   its box in the picture's pixels and its mask;
//! - CLICKS on one object (`points`, positive or negative) and/or its box
//!   (`object_box`): that object comes back as up to three candidate masks,
//!   best first, scored by the model's predicted IoU, and `refine_id` lets a
//!   follow-up request refine the answer with more clicks.
//!
//! A VIDEO is a session (`/v1/masks/sessions`): a concept, then its frames
//! one at a time, every object tracked under one id from frame to frame.
//! Each frame answers with the frames whose output is final - Meta holds an
//! output for its 15-frame hot start, so a new object that turns out to be
//! noise disappears from the frames it was seen on - and, by default, the
//! frame just sent as it stands now.
//!
//! Masks are COCO run-length encoding, the format SAM's own tooling and
//! pycocotools speak: `size` [height, width] and `counts`, the run lengths of
//! the COLUMN-major pixel sequence starting with a (possibly empty) run of
//! zeros. Uncompressed (a list of numbers, not COCO's compact string) - what
//! `pycocotools.mask.frPyObjects` reads, and a few kilobytes for a real
//! object where a 1080p bitmap would be 260 KB of base64.

use serde::{Deserialize, Serialize};

/// An exemplar box in the picture's pixels: "more things like this one" when
/// positive, "not like this one" when not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskBox {
    /// x0, y0, x1, y1
    #[serde(rename = "box")]
    pub xyxy: [f32; 4],
    /// default true
    #[serde(default = "yes")]
    pub positive: bool,
}

fn yes() -> bool {
    true
}

/// A click on the object, in the picture's pixels: on it (positive) or off it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskPoint {
    /// x, y
    #[serde(rename = "point")]
    pub xy: [f32; 2],
    /// default true
    #[serde(default = "yes")]
    pub positive: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaskRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// the picture, as a base64 `data:` URI
    pub image: String,
    /// a concept: a short noun phrase ("red car", "shoe")
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub boxes: Vec<MaskBox>,
    /// keep instances scoring above this (default 0.5, Meta's processor's)
    #[serde(default)]
    pub threshold: Option<f32>,
    /// at most this many instances, best first (default: all kept)
    #[serde(default)]
    pub max_instances: Option<usize>,
    /// clicks on ONE object - the other task: that object's mask, not every
    /// instance of a concept (never together with `text` / `boxes`)
    #[serde(default)]
    pub points: Vec<MaskPoint>,
    /// the object's box, x0 y0 x1 y1 in the picture's pixels (with or
    /// without clicks)
    #[serde(default)]
    pub object_box: Option<[f32; 4]>,
    /// three candidates or one (default: three for a lone click)
    #[serde(default)]
    pub multimask: Option<bool>,
    /// a previous click answer's `refine_id` on the same picture: its best
    /// mask becomes part of this prompt
    #[serde(default)]
    pub refine: Option<String>,
}

/// COCO RLE, uncompressed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rle {
    /// [height, width]
    pub size: [usize; 2],
    pub counts: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskInstance {
    /// a concept: sigmoid(logit) * sigmoid(presence), the model's confidence
    /// this is one; clicks: the model's predicted IoU of this mask
    pub score: f32,
    /// x0, y0, x1, y1 in the picture's pixels
    #[serde(rename = "box")]
    pub xyxy: [f32; 4],
    /// mask pixels set
    pub area: u64,
    pub mask: Rle,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaskTimings {
    pub resize_ms: f64,
    pub encode_ms: f64,
    pub prompt_ms: f64,
    pub detect_ms: f64,
    pub masks_ms: f64,
    /// the picture was the previous request's, byte for byte: its encoding
    /// was reused (resize and encode are 0)
    #[serde(default)]
    pub image_reused: bool,
    /// the prompt was the previous request's: the text encoding was reused
    #[serde(default)]
    pub text_reused: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskResponse {
    pub object: String, // "masks"
    pub model: String,
    pub width: usize,
    pub height: usize,
    /// a concept: how sure the model is it appears in the picture at all;
    /// clicks: how sure it is there is an object under them
    pub presence: f32,
    pub instances: Vec<MaskInstance>,
    /// prompt tokens the text tower read (start and end markers included)
    pub prompt_tokens: usize,
    pub timings: MaskTimings,
    /// a click answer's handle - send it back as `refine` with more clicks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refine_id: Option<String>,
}

// ---------------------------------------------------------------- video

/// `POST /v1/masks/sessions` - track a concept through a video sent a frame
/// at a time (Meta's SAM 3 video predictor: detection on every frame, its
/// tracker carrying each object from frame to frame). `texts` tracks
/// several concepts at once, each as its own session would, the frame read
/// once for all of them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaskSessionRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// the concept: a short noun phrase ("person", "red car")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// several concepts, in place of `text`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texts: Option<Vec<String>>,
    /// the video's length in frames, when it is known (a clip under 16
    /// frames tracks a little differently; a camera never knows)
    #[serde(default)]
    pub frames: Option<u32>,
    /// answer every frame with its provisional output too (default true):
    /// what the frame shows now, before its hold ends
    #[serde(default)]
    pub preview: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskSessionResponse {
    pub object: String, // "masks.session"
    pub id: String,
    pub model: String,
    /// the concepts tracked, in the order an object's `concept` counts
    pub concepts: Vec<String>,
    /// frames an output is held before it is final (Meta's hot start: a new
    /// object that turns out to be noise is removed from frames already
    /// seen)
    pub hold_frames: u32,
    /// a session with no frame for this long is dropped
    pub idle_seconds: u64,
}

/// `POST /v1/masks/sessions/{id}/frames` - the session's next frame, every
/// frame the first one's size.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaskFrameRequest {
    /// the frame, as a base64 `data:` URI
    pub image: String,
}

/// One tracked object on one frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskVideoObject {
    /// the object's id, the same on every frame it is tracked on and
    /// unique across the session's concepts
    pub id: i64,
    /// which of the session's `concepts` it was found for
    pub concept: usize,
    /// the detection score the object was born with
    pub score: f32,
    /// x0, y0, x1, y1 in the frame's pixels (pixel edges)
    #[serde(rename = "box")]
    pub xyxy: [f32; 4],
    pub area: u64,
    pub mask: Rle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskVideoFrame {
    /// the frame's index in the session, from 0
    pub frame: u32,
    pub objects: Vec<MaskVideoObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskFrameResponse {
    pub object: String, // "masks.frame"
    /// the index of the frame just sent
    pub frame: u32,
    pub width: usize,
    pub height: usize,
    /// frames whose output is final now, oldest first (one a frame once the
    /// hold has filled)
    pub frames: Vec<MaskVideoFrame>,
    /// the frame just sent as it stands now, when the session asked for it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<MaskVideoFrame>,
    /// the engine's time on this frame
    pub ms: f64,
}

/// `POST /v1/masks/sessions/{id}/finish` - the end of the video: the frames
/// still held, final. The session is gone after it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskFinishResponse {
    pub object: String, // "masks.frames"
    pub frames: Vec<MaskVideoFrame>,
}
