//! Backend-neutral EmbeddingGemma 2 processor geometry. No device code or tensor math.
/// `<|image|>`: a picture's soft token.
pub const IMAGE_TOKEN: u32 = 258880;
/// `<|audio|>`: an audio clip's soft token.
pub const AUDIO_TOKEN: u32 = 258881;
/// `<image|>`: closes a picture.
pub const EOI_TOKEN: u32 = 258882;
/// `<audio|>`: closes a clip.
pub const EOA_TOKEN: u32 = 258883;
/// `<|video|>`: a video frame's soft token, distinct from a still image.
pub const VIDEO_TOKEN: u32 = 258884;
/// Upstream video processor defaults, independent of the still-image budget.
pub const VIDEO_FRAME_TOKENS: usize = 70;
pub const MAX_VIDEO_FRAMES: usize = 32;
/// `<|image>`: opens a picture.
pub const BOI_TOKEN: u32 = 255999;
/// `<|audio>`: opens a clip.
pub const BOA_TOKEN: u32 = 256000;

/// The soft-token budgets the processor supports (`_SUPPORTED_SOFT_TOKENS`).
pub const IMAGE_TOKEN_BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];
/// The checkpoint's own (`vision_soft_tokens_per_image`).
pub const DEFAULT_IMAGE_TOKENS: usize = 280;
const PATCH: usize = 16;
const POOL: usize = 3;

/// One run of soft tokens in a sequence: where it starts, how long it is,
/// and which medium's token it repeats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub token: u32,
    pub start: usize,
    pub len: usize,
}

/// The sequence's media soft-token runs in order (`<|image|>`, `<|audio|>`
/// and `<|video|>` repeats). Two media never touch - each run sits between
/// its medium's open and close tokens - so a run is a maximal repeat.
pub fn placeholder_runs(seq: &[u32]) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    for (i, &t) in seq.iter().enumerate() {
        if !matches!(t, IMAGE_TOKEN | AUDIO_TOKEN | VIDEO_TOKEN) {
            continue;
        }
        match out.last_mut() {
            Some(r) if r.token == t && r.start + r.len == i => r.len += 1,
            _ => out.push(Run {
                token: t,
                start: i,
                len: 1,
            }),
        }
    }
    out
}

/// The processor's resize target for a `w` x `h` picture at a soft-token
/// budget, as (width, height) - `get_aspect_ratio_preserving_size`, ported
/// line for line (its float math in f64, as Python's).
pub fn image_target(w: usize, h: usize, budget: usize) -> Result<(usize, usize), String> {
    if w == 0 || h == 0 || w > 8192 || h > 8192 {
        return Err("picture edges must be within 1..8192".into());
    }
    if !IMAGE_TOKEN_BUDGETS.contains(&budget) {
        return Err("unsupported picture token budget".into());
    }
    let max_patches = budget * POOL * POOL;
    let target_px = (max_patches * PATCH * PATCH) as f64;
    let factor = (target_px / (h * w) as f64).sqrt();
    let side = PATCH * POOL;
    let mut th = (factor * h as f64 / side as f64).floor() as usize * side;
    let mut tw = (factor * w as f64 / side as f64).floor() as usize * side;
    if th == 0 && tw == 0 {
        return Err("the picture is too small to resize to whole tokens".into());
    }
    let max_side = (max_patches / (POOL * POOL)) * side;
    if th == 0 {
        th = side;
        tw = ((w as f64 / h as f64).floor() as usize * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((h as f64 / w as f64).floor() as usize * side).min(max_side);
    }
    if (th * tw) as f64 > target_px {
        return Err(format!(
            "a {w} x {h} picture resizes past the {budget}-token budget"
        ));
    }
    Ok((tw, th))
}

/// Soft tokens a `w` x `h` picture takes at `budget`.
pub fn image_tokens(w: usize, h: usize, budget: usize) -> Result<usize, String> {
    let (tw, th) = image_target(w, h, budget)?;
    Ok((tw / PATCH) * (th / PATCH) / (POOL * POOL))
}

/// Audio processor rate and maximum clip duration.
pub const AUDIO_RATE: usize = 16_000;
pub const AUDIO_MAX_SAMPLES: usize = 30 * AUDIO_RATE;
const HOP: usize = 160;
const WIN: usize = 320;

/// Frontend frames of an `n`-sample clip: the processor's unfold of 321
/// samples every 160 over the clip left-padded by 160.
pub fn audio_frames(n: usize) -> Result<usize, String> {
    if n > AUDIO_MAX_SAMPLES {
        return Err(format!(
            "an audio clip is at most 30 s ({AUDIO_MAX_SAMPLES} samples at 16 kHz); this one has {n} - split it into several parts"
        ));
    }
    if n + HOP < WIN + 1 {
        return Err(format!(
            "an audio clip of {n} samples is shorter than one 20 ms frame"
        ));
    }
    Ok((n + HOP - (WIN + 1)) / HOP + 1)
}

/// Soft tokens an `n`-sample clip takes: the frames through two stride-2
/// convs (`ceil` twice) - the processor's `replace_audio_token`.
pub fn audio_tokens(n: usize) -> Result<usize, String> {
    let f = audio_frames(n)?;
    Ok(((f - 1) / 2) / 2 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn embedding_media_geometry_is_bounded_and_shared() {
        assert_eq!(image_target(1200, 900, 280).unwrap(), (912, 672));
        assert_eq!(image_tokens(420, 760, 280).unwrap(), 264);
        for (w, h, b) in [
            (0, 1, 280),
            (usize::MAX, 1, 280),
            (1, usize::MAX, 280),
            (48, 48, usize::MAX),
            (48, 48, 281),
        ] {
            assert!(image_target(w, h, b).is_err());
        }
        for budget in IMAGE_TOKEN_BUDGETS {
            for (w, h) in [(1, 8192), (8192, 1), (8192, 8192), (1, 1)] {
                let (tw, th) = image_target(w, h, budget).unwrap();
                assert!(tw.is_multiple_of(48) && th.is_multiple_of(48));
                assert!(tw * th / 2304 <= budget);
            }
        }
        assert_eq!(audio_tokens(480000).unwrap(), 750);
        assert!(audio_frames(480001).is_err());
        assert!(audio_frames(usize::MAX).is_err());
        assert!(audio_frames(160).is_err());
        assert_eq!(audio_tokens(161).unwrap(), 1);
    }
}
