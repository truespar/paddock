//! Decoded-frame transport, like our video masking API. No remote fetches,
//! subprocess codecs or implicit audio extraction. Clients sample the movie;
//! the backend owns the model's resize, frame tokens and optional timestamps.
//! Reference: HF EmbeddingGemma2Processor.replace_video_token (5.19.0).
use paddock_engine::{encoder::embedding_gemma2::MAX_VIDEO_FRAMES, service::MmChunk};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Video {
    frames: Vec<Frame>,
    #[serde(default)]
    add_timestamps: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frame {
    image_url: String,
    #[serde(default)]
    timestamp: Option<f64>,
}
fn parse(part: &serde_json::Value) -> Result<Video, String> {
    let value = part
        .get("input_video")
        .ok_or("an input_video part needs input_video.frames")?;
    let video: Video = serde_json::from_value(value.clone()).map_err(|e| format!("video: {e}"))?;
    if video.frames.is_empty() || video.frames.len() > MAX_VIDEO_FRAMES {
        return Err(format!(
            "video requires 1..{MAX_VIDEO_FRAMES} sampled frames"
        ));
    }
    let mut previous = None;
    for frame in &video.frames {
        if video.add_timestamps && frame.timestamp.is_none() {
            return Err("timestamped video needs a timestamp in seconds for every frame".into());
        }
        if let Some(t) = frame.timestamp {
            if !t.is_finite() || !(0.0..86400.0).contains(&t) || previous.is_some_and(|p| t <= p) {
                return Err(
                    "video timestamps must be finite, increasing seconds in [0, 86400)".into(),
                );
            }
            previous = Some(t);
        } else if previous.is_some() {
            return Err("supply timestamps for every frame or none".into());
        }
    }
    if video.frames.iter().any(|f| f.timestamp.is_some())
        && video.frames.iter().any(|f| f.timestamp.is_none())
    {
        return Err("supply timestamps for every frame or none".into());
    }
    Ok(video)
}
pub(super) fn decode(
    part: &serde_json::Value,
    pixels: &mut u64,
) -> Result<Vec<(String, MmChunk)>, String> {
    let video = parse(part)?;
    let mut result = Vec::with_capacity(video.frames.len());
    let mut shape = None;
    for (i, frame) in video.frames.iter().enumerate() {
        let (rgb, w, h) =
            crate::reference_image::decode_image_url_reference_limited(&frame.image_url, *pixels)?;
        if w > 8192 || h > 8192 || shape.is_some_and(|s| s != (w, h)) {
            return Err(
                "video frames must have the same upright size and edges no larger than 8192".into(),
            );
        }
        shape = Some((w, h));
        *pixels = pixels
            .checked_sub((w * h) as u64)
            .ok_or("video exceeds the request pixel budget")?;
        let before = if video.add_timestamps {
            let t = frame.timestamp.expect("validated timestamps") as u64;
            format!(
                "{}{:02}:{:02} ",
                if i == 0 { "" } else { " " },
                t / 60,
                t % 60
            )
        } else {
            String::new()
        };
        result.push((before, MmChunk::Image { rgb, w, h }));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn video_contract_rejects_ambiguous_or_unbounded_frames() {
        let frame = |t| json!({"image_url":"data:image/png;base64,invalid", "timestamp":t});
        for frames in [
            vec![],
            vec![frame(0); 33],
            vec![frame(2), frame(1)],
            vec![frame(-1)],
            vec![frame(86400)],
            vec![frame(0), frame(0)],
            vec![frame(0), json!({"image_url":"x"})],
        ] {
            assert!(parse(&json!({"input_video":{"frames":frames}})).is_err());
        }
        assert!(parse(&json!({"input_video":{"frames":[frame(0),frame(1)]}})).is_ok());
        assert!(
            parse(&json!({"input_video":{"frames":[{"image_url":"x"}],"add_timestamps":true}}))
                .is_err()
        );
        assert!(
            parse(&json!({"input_video":{"frames":[{"image_url":"x"}],"audio":true}})).is_err()
        );
    }
}
