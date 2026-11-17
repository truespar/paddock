//! granite-vision multimodal prefill: interleaved text/image chunks become one
//! batch slot's prompt.
//!
//! Compared with gemma4's splice this is simpler in two ways and harder in one.
//!
//! **Simpler - no marker tokens.** llama.cpp's mtmd sets `img_beg = "<image>"`
//! and `img_end = ""` for granite: the placeholder is the whole marker. The
//! template renders one `<image>`, and the picture's rows take its place. There
//! is no begin/end pair to wrap the span in, so nothing here mirrors gemma4's
//! `<|image>` ... `<image|>` bracketing.
//!
//! **Simpler - image rows are CAUSAL.** gemma4v sets
//! `mtmd_decode_use_non_causal`, so its soft tokens all attend to the span's
//! last position and the splice has to carry a separate attention-bound vector.
//! Granite does not, so image rows are ordinary prefill rows: their position is
//! their attention bound, and `rows_pass_body` needs no override at all.
//!
//! **Harder - the row count is not a constant.** AnyRes picks a tile grid from
//! the aspect ratio, so one picture is 144 rows (anything that fits a single
//! 384px tile) up to ~2.5k (a max-grid 3840×384 strip). Two consequences run
//! through everything below:
//!
//! 1. The row plan is built first, from pure host arithmetic
//!    ([`super::preprocess::AnyResPlan`]), so admission reserves the real KV and
//!    an oversized prompt is refused before the vision tower runs.
//! 2. Every image is registered at the exact row it occupies. Get that wrong
//!    and `deepstack::plan` resolves nothing, the placeholder embedding survives
//!    into the stack, and the model answers fluently about a picture it never
//!    saw. Two independent checks below exist for exactly that failure: the
//!    encoder's row count must equal the planned one, and the registry's
//!    coverage of each pass must equal the rows the plan reserved.
//!
//! Image prompts do take the prefix cache. They cannot
//! be keyed on their row tokens - every image row carries the same `<image>`
//! id, so two different pictures would look like the same prefix - so this
//! lane builds a parallel KEY vector alongside the row stream, with each image
//! row keyed on the picture's content hash and its offset within the picture.
//! See `prefix.rs`; `plan_rows` produces both vectors together so they cannot
//! drift apart.

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::service::MmChunk;

use super::GpuGranite;
use super::batch::pf_rows;

/// Where one media item sits in the row plan: `(chunk index, first row, row
/// count)`.
type Placement = (usize, usize, usize);

/// The interleaved row stream and the media items' placements, from chunks
/// alone - no device, no pixels, no samples.
///
/// `media` resolves one non-text chunk to `(placeholder id, row count,
/// content hash)`: the AnyRes row count for a picture, or the projector's
/// 10-tokens-per-second rule for a clip. It is the only thing that differs
/// between the vision and speech lanes here; everything below - dense
/// positions, content-derived radix keys, placement records - is shared.
///
/// Rows are `(slot, position, token)`, positions dense from 0, and a media
/// item's rows all carry its placeholder id. That id is overwritten before
/// the first layer, so its VALUE never reaches the math - but it is still the
/// right id to write: `embed_rows` gathers it, so it has to be in range, and
/// if the coverage guard below ever failed to fire, the rows would hold
/// exactly what an unfilled placeholder holds upstream rather than some
/// arbitrary token's embedding.
fn plan_rows(
    slot: usize,
    chunks: &[MmChunk],
    media: impl Fn(&MmChunk) -> Result<(u32, usize, u64), GpuError>,
) -> Result<(Vec<(u32, u32, u32)>, Vec<u32>, Vec<Placement>), GpuError> {
    let mut rows: Vec<(u32, u32, u32)> = Vec::new();
    let mut keys: Vec<u32> = Vec::new();
    let mut placed: Vec<Placement> = Vec::new();
    for (ci, ch) in chunks.iter().enumerate() {
        let base = rows.len();
        match ch {
            MmChunk::Text(ids) => {
                rows.extend(
                    ids.iter()
                        .enumerate()
                        .map(|(j, &t)| (slot as u32, (base + j) as u32, t)),
                );
                keys.extend_from_slice(ids);
            }
            _ => {
                let (pad, n, h64) = media(ch)?;
                rows.extend((0..n).map(|j| (slot as u32, (base + j) as u32, pad)));
                // The radix key for a media row is derived from the item's
                // CONTENT, never from `pad` - every image (or audio) row of
                // every prompt carries the same placeholder id, so keying on
                // it would serve one picture's KV for another. See prefix.rs.
                keys.extend((0..n).map(|j| super::prefix::image_key_row(h64, j)));
                placed.push((ci, base, n));
            }
        }
    }
    debug_assert_eq!(rows.len(), keys.len(), "one key per row");
    Ok((rows, keys, placed))
}

/// The first row where the plan's reserved image runs and the registry's
/// coverage disagree, if any.
///
/// Checked ROW by ROW rather than by count, and over the whole prompt at
/// admission. Two reasons that shape is the right one:
///
/// - Equal totals can hide a row that is reserved but uncovered sitting
///   against one that is covered but unreserved. Either shifts a picture.
/// - Because the predicate is per row, a later chunk cut cannot introduce a
///   disagreement this missed - which is what lets the chunked lane advance an
///   image prompt in pieces with no per-pass guard of its own.
///
/// `rows` must be the whole prompt, so that a row's index is its position.
fn coverage_drift(
    rows: &[(u32, u32, u32)],
    placed: &[Placement],
    hit: impl Fn(u32, usize) -> bool,
) -> Option<usize> {
    rows.iter().enumerate().find_map(|(i, &(slot, pos, _))| {
        let reserved = placed.iter().any(|&(_, b, n)| i >= b && i < b + n);
        (reserved != hit(slot, pos as usize)).then_some(i)
    })
}

impl GpuGranite {
    /// Prefill an interleaved text/image prompt into `slot`. Returns the last
    /// row's logits and the slot's KV row count - image rows included, so it
    /// differs from the prompt's token count and the service must use this as
    /// the slot's position.
    pub(crate) fn multimodal_prefill_slot(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        self.multimodal_prefill_slot_pre(slot, chunks, None)
    }

    /// [`Self::multimodal_prefill_slot`], optionally with this prompt's images
    /// already encoded - one per image chunk, in chunk order.
    ///
    /// That is how a batched admission wave hands its work over: every pending
    /// request's pictures go through one tower pass first (`encode_wave`), then
    /// each slot prefills against features that already exist. `None` keeps the
    /// self-contained path, where each image is encoded (or cache-hit) here.
    pub(crate) fn multimodal_prefill_slot_pre(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
        pre: Option<&[std::sync::Arc<super::vision::MediaFeatures>]>,
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let (rows, keys) = self.mm_admit(slot, chunks, pre)?;
        let start = self.prefill_resume_rows(slot, &keys, rows.len())?;
        self.hist_reset(slot, &keys[..start]);
        let mut last_len = 0usize;
        for chunk in rows[start..].chunks(pf_rows()) {
            last_len = chunk.len();
            self.rows_pass_body(chunk, 0)?;
        }
        let logits = self.head_row(last_len - 1)?;
        self.prefix_insert(slot, &keys);
        Ok((logits, rows.len()))
    }

    /// Queue an interleaved text/image prompt for STALL-FREE chunked prefill,
    /// the same queue text prompts use. Returns the slot's KV row count.
    ///
    /// The images are encoded and registered here, at admission, and the queue
    /// carries the resulting row stream - image rows included, each holding the
    /// `<image>` placeholder id. Everything downstream then treats them as
    /// ordinary prefill rows: `rows_pass_body` resolves each row's vision
    /// features from its `(slot, position)`, so a chunk boundary landing inside
    /// a picture resumes at the right source row (`deepstack::plan_spans`, with
    /// a test for exactly that cut).
    ///
    /// This is what stops one image holding the tick. A max-grid AnyRes picture
    /// is ~2.5k rows on its own, so the blocking whole-prompt path could freeze
    /// every live decode stream for several chunks' worth of work - the thing
    /// Sarathi-style chunked prefill exists to prevent
    pub(crate) fn multimodal_prefill_begin(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
        pre: Option<&[std::sync::Arc<super::vision::MediaFeatures>]>,
    ) -> Result<usize, GpuModelError> {
        let (rows, keys) = self.mm_admit(slot, chunks, pre)?;
        let n = rows.len();
        let cursor = self.prefill_resume_rows(slot, &keys, n)?;
        // A queued entry for this slot is STALE (the scheduler keeps one chunk
        // per live slot - a duplicate means the old request died and the slot
        // was reused): evict rather than wedge the slot. Same rule as
        // `prefill_begin_impl`, and `mm_admit` has already dropped the old
        // sequence's images.
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.push(super::batch::ChunkedPrefill {
            slot,
            tokens: rows.into_iter().map(|r| r.2).collect(),
            cursor,
            keys,
        });
        Ok(n)
    }

    /// Everything both mm lanes do before any row moves: plan, admit KV,
    /// encode + register each picture at the row it owns, and prove the
    /// registry agrees with the plan. Returns the prompt's row stream and its
    /// radix KEY vector (content-derived at image rows - see prefix.rs).
    fn mm_admit(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
        pre: Option<&[std::sync::Arc<super::vision::MediaFeatures>]>,
    ) -> Result<(Vec<(u32, u32, u32)>, Vec<u32>), GpuModelError> {
        if self.vision.is_none() && self.audio.is_none() {
            return Err(GpuModelError::Unsupported(
                "granite was loaded without an mmproj - set `mmproj` in the config to enable \
                 image or audio input"
                    .into(),
            ));
        }
        if self.batch.is_none() {
            return Err(GpuModelError::Unsupported(
                "granite multimodal needs the batch lane (max_batch > 1): the serial batch-1 \
                 lane walks one token at a time with no row/position table, so there is \
                 nowhere to address media features from"
                    .into(),
            ));
        }

        // ── pass 1: the row plan, pure host arithmetic ──────────────────────
        // Both rules run without touching a pixel or a sample, so the whole
        // prompt's shape is known before any VRAM moves. The placeholder ids
        // are checked by `attach_vision`/`attach_audio`, so a None here is a
        // real invariant break rather than a fallback - but the mm lane is
        // where a wrong id would corrupt rows, so it names the reason.
        let v = self.vision.as_ref();
        let a = self.audio.as_ref();
        let (img_pad, aud_pad) = (self.img_pad_id, self.audio_pad_id);
        let (rows, keys, placed) = plan_rows(slot, chunks, |ch| match ch {
            MmChunk::Image { rgb, w, h } => {
                let v = v.ok_or_else(|| {
                    GpuError::Driver("image request but this granite has no vision tower".into())
                })?;
                let pad = img_pad.ok_or_else(|| {
                    GpuError::Driver("granite-vision: no `<image>` token in the vocab".into())
                })?;
                Ok((
                    pad,
                    v.image_tokens(*w, *h)?,
                    super::deepstack::img_key(rgb, *w, *h),
                ))
            }
            MmChunk::Audio { samples, .. } => {
                let a = a.ok_or_else(|| {
                    GpuError::Driver("audio request but this granite has no audio tower".into())
                })?;
                let pad = aud_pad.ok_or_else(|| {
                    GpuError::Driver("granite-speech: no `<|audio|>` token in the vocab".into())
                })?;
                Ok((
                    pad,
                    a.tokens_for_samples(samples.len()),
                    super::deepstack::audio_key(samples),
                ))
            }
            MmChunk::OcrCrop(_) => Err(GpuError::Driver(
                "OCR crop directive on granite - routing bug".into(),
            )),
            MmChunk::CtcTimestamps { .. } => Err(GpuError::Driver(
                "CTC timestamp directive on granite - routing bug".into(),
            )),
            MmChunk::VisionPixels { .. } => Err(GpuError::Driver(
                "pixel-budget directive on granite - routing bug".into(),
            )),
            MmChunk::Text(_) => unreachable!("text chunks never reach the media resolver"),
        })?;

        // A pre-encoded wave must line up one-for-one with this prompt's media
        // chunks, or the pictures would be placed against the wrong rows -
        // silently, which is the whole failure class this lane guards.
        if let Some(p) = pre
            && p.len() != placed.len()
        {
            return Err(GpuModelError::Unsupported(format!(
                "granite-vision: {} pre-encoded images for a prompt with {} image chunks",
                p.len(),
                placed.len()
            )));
        }

        // ── admission, before the tower runs ────────────────────────────────
        // Drops the slot's previous sequence (its media with it) and backs
        // every block the prompt needs, so a prompt the pool cannot hold fails
        // without having burnt an encode first.
        self.admit_rows(slot, rows.len())?;

        // ── pass 2: encode and register, at the exact row each item owns ────
        for (k, &(ci, base, want)) in placed.iter().enumerate() {
            let (got, what) = match &chunks[ci] {
                MmChunk::Image { rgb, w, h } => (
                    match pre {
                        Some(p) => self.place_feats(slot, base, std::sync::Arc::clone(&p[k])),
                        None => self.place_image(slot, base, rgb, *w, *h)?,
                    },
                    format!("{w}×{h} image"),
                ),
                MmChunk::Audio { samples, mel } => {
                    // The frontend normally runs on the runner's blocking pool
                    //  - computing it here would put the host DSP
                    // back on the engine thread, where it serializes every
                    // other slot's tick behind this clip.
                    let owned;
                    let mel = match mel {
                        Some(m) => m,
                        None => {
                            owned = crate::audio::granite::speech_features(samples)
                                .map_err(GpuModelError::Unsupported)?;
                            &owned
                        }
                    };
                    (
                        self.place_audio(slot, base, mel)?,
                        format!("{:.2}s clip", samples.len() as f64 / 16000.0),
                    )
                }
                MmChunk::Text(_)
                | MmChunk::OcrCrop(_)
                | MmChunk::VisionPixels { .. }
                | MmChunk::CtcTimestamps { .. } => {
                    unreachable!("placement indexes a media chunk")
                }
            };
            if got != want {
                return Err(GpuModelError::Unsupported(format!(
                    "granite: {what} planned {want} rows but the encoder packed {got} - the \
                     prompt's reserved run and the packed features disagree, which would shift \
                     every token after it"
                )));
            }
        }

        // ── the drift guard, over the whole prompt, before any row moves ────
        // Independent of the row plan that built the stream: every row the plan
        // reserved for a media item must be a row the registry resolves, and
        // vice versa. They can only differ if a position drifted, and the drift
        // is otherwise silent - the stack runs, the answer reads fine, the
        // picture (or the speech) is simply not in it. Cheap: one integer walk
        // over rows that are about to move gigabytes of weights.
        if let Some(i) = coverage_drift(&rows, &placed, |s, p| self.media.hits(s, p)) {
            return Err(GpuModelError::Unsupported(format!(
                "granite: row {i} of {} disagrees - the plan reserves it for a media item and \
                 the registry does not, or the reverse. The registered position and the prompt \
                 layout have drifted",
                rows.len()
            )));
        }
        Ok((rows, keys))
    }
}

#[cfg(test)]
mod tests {
    use super::{Placement, coverage_drift, plan_rows};
    use crate::service::MmChunk;

    const PAD: u32 = 100266; // <image> on granite-vision-4.1-4b
    const APAD: u32 = 100352; // <|audio|> on granite-speech-4.1-2b

    fn txt(n: usize, first: u32) -> MmChunk {
        MmChunk::Text((0..n).map(|i| first + i as u32).collect())
    }

    /// Pixels are irrelevant to the plan - only w/h are, and only through the
    /// row rule. Sizes double as the row count via the fake below.
    fn img(w: usize, h: usize) -> MmChunk {
        MmChunk::Image {
            rgb: Vec::new(),
            w,
            h,
        }
    }

    fn clip(samples: usize) -> MmChunk {
        MmChunk::Audio {
            samples: vec![0.25; samples],
            mel: None,
        }
    }

    /// The plan under test, with stand-in row rules: a picture's `w` is its
    /// row count and a clip's row count is the real projector rule, so a test
    /// states the shape it wants directly. The AnyRes rule is proven
    /// separately against the HF processor in `preprocess.rs`, and the audio
    /// rule against the upstream formula in `crate::audio::granite`.
    fn plan(slot: usize, chunks: &[MmChunk]) -> (Vec<(u32, u32, u32)>, Vec<u32>, Vec<Placement>) {
        plan_rows(slot, chunks, |ch| match ch {
            MmChunk::Image { rgb, w, h } => Ok((
                PAD,
                *w,
                crate::gpu_model::granite::deepstack::img_key(rgb, *w, *h),
            )),
            MmChunk::Audio { samples, .. } => Ok((
                APAD,
                crate::audio::granite::audio_tokens_for_samples(samples.len()),
                crate::gpu_model::granite::deepstack::audio_key(samples),
            )),
            MmChunk::Text(_)
            | MmChunk::OcrCrop(_)
            | MmChunk::VisionPixels { .. }
            | MmChunk::CtcTimestamps { .. } => {
                unreachable!()
            }
        })
        .expect("plan")
    }

    /// The registry, as it looks after a correct registration pass: every row
    /// a placement covers reports as an image row.
    fn hit_from(placed: &[Placement], slot: u32, shift: isize) -> impl Fn(u32, usize) -> bool + '_ {
        move |s, p| {
            s == slot
                && placed.iter().any(|&(_, b, n)| {
                    let b = b as isize + shift;
                    (p as isize) >= b && (p as isize) < b + n as isize
                })
        }
    }

    #[test]
    fn positions_are_dense_and_image_rows_carry_the_placeholder() {
        let chunks = [txt(3, 10), img(5, 1), txt(2, 20)];
        let (rows, _keys, placed) = plan(1, &chunks);
        assert_eq!(rows.len(), 10);
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.0, 1, "every row belongs to the asked-for slot");
            assert_eq!(r.1 as usize, i, "positions dense from 0");
        }
        let toks: Vec<u32> = rows.iter().map(|r| r.2).collect();
        assert_eq!(toks, vec![10, 11, 12, PAD, PAD, PAD, PAD, PAD, 20, 21]);
        assert_eq!(placed, vec![(1, 3, 5)]);
    }

    /// The count the service uses as the slot's KV position is the ROW count,
    /// which is nothing like the token count once a picture expands.
    #[test]
    fn one_placeholder_expands_to_the_whole_anyres_run() {
        let chunks = [txt(4, 1), img(2353, 1), txt(1, 9)];
        let (rows, _keys, _placed) = plan(0, &chunks);
        assert_eq!(rows.len(), 4 + 2353 + 1);
    }

    #[test]
    fn several_images_land_at_their_own_rows() {
        let chunks = [
            img(144, 1),
            txt(2, 50),
            img(300, 1),
            txt(1, 60),
            img(144, 1),
        ];
        let (rows, _keys, placed) = plan(0, &chunks);
        assert_eq!(placed, vec![(0, 0, 144), (2, 146, 300), (4, 447, 144)]);
        assert_eq!(rows.len(), 591);
        // and the placements really do point at the placeholder runs
        for &(_, b, n) in &placed {
            assert!(
                rows[b..b + n].iter().all(|r| r.2 == PAD),
                "run at {b} is not all pad"
            );
        }
        assert_eq!(rows.iter().filter(|r| r.2 == PAD).count(), 588);
    }

    /// The property at the plan level: two prompts that are identical in
    /// row tokens (same text, same placeholder run, same length) but hold
    /// different pictures must produce different radix keys. Keying on the row
    /// tokens is exactly what would serve one picture's KV for another.
    #[test]
    fn different_pictures_give_different_radix_keys() {
        let a = [
            txt(4, 1),
            MmChunk::Image {
                rgb: vec![1, 2, 3],
                w: 64,
                h: 1,
            },
        ];
        let b = [
            txt(4, 1),
            MmChunk::Image {
                rgb: vec![1, 2, 4],
                w: 64,
                h: 1,
            },
        ];
        let (rows_a, keys_a, _) = plan(0, &a);
        let (rows_b, keys_b, _) = plan(0, &b);
        assert_eq!(
            rows_a, rows_b,
            "the ROW streams are indistinguishable - that is the trap"
        );
        assert_ne!(keys_a, keys_b, "the KEY streams must not be");
        assert_eq!(
            &keys_a[..4],
            &keys_b[..4],
            "the shared text prefix still matches"
        );
    }

    /// ...and the same picture must key identically, or a document
    /// conversation never hits.
    #[test]
    fn the_same_picture_gives_the_same_radix_keys() {
        let a = [
            txt(4, 1),
            MmChunk::Image {
                rgb: vec![9, 9, 9],
                w: 64,
                h: 1,
            },
        ];
        let b = [
            txt(4, 1),
            MmChunk::Image {
                rgb: vec![9, 9, 9],
                w: 64,
                h: 1,
            },
        ];
        assert_eq!(plan(0, &a).1, plan(0, &b).1);
    }

    /// Text rows key as themselves, so a text-only prompt's key vector is its
    /// token vector - the two lanes must agree or a prompt served through the
    /// mm lane could never hit one served through the text lane.
    #[test]
    fn text_rows_key_as_their_own_tokens() {
        let chunks = [txt(6, 100)];
        let (rows, keys, _) = plan(0, &chunks);
        let toks: Vec<u32> = rows.iter().map(|r| r.2).collect();
        assert_eq!(keys, toks);
    }

    /// A correct registration passes, over a prompt with several images and
    /// text on both sides of each.
    #[test]
    fn a_correct_registration_shows_no_drift() {
        let chunks = [txt(7, 1), img(300, 1), txt(5, 90), img(144, 1), txt(3, 70)];
        let (rows, _keys, placed) = plan(2, &chunks);
        assert_eq!(
            coverage_drift(&rows, &placed, hit_from(&placed, 2, 0)),
            None
        );
    }

    /// The guard has to FIRE when the registration drifts - otherwise it is
    /// decoration. One row of shift is the realistic failure (an off-by-one in
    /// the position handed to `place_image`), and it is invisible downstream:
    /// the stack runs and the answer still reads fine.
    #[test]
    fn a_one_row_shift_in_the_registry_is_caught() {
        let chunks = [txt(7, 1), img(300, 1), txt(5, 90)];
        let (rows, _keys, placed) = plan(0, &chunks);
        let at = coverage_drift(&rows, &placed, hit_from(&placed, 0, 1));
        assert_eq!(
            at,
            Some(7),
            "the image's first row is reserved but no longer covered"
        );
    }

    /// The failure a per-COUNT check could miss: one row reserved-but-uncovered
    /// against one covered-but-unreserved cancels out in any total, and both
    /// halves shift a picture. This is why the guard is per row.
    #[test]
    fn offsetting_errors_do_not_cancel_out() {
        let chunks = [txt(4, 1), img(10, 1), txt(4, 90)];
        let (rows, _keys, placed) = plan(0, &chunks);
        // registry covers 5..15 where the plan reserves 4..14 - same COUNT
        let shifted = hit_from(&placed, 0, 1);
        assert_eq!(
            rows.iter()
                .filter(|&&(s, p, _)| shifted(s, p as usize))
                .count(),
            10
        );
        assert!(coverage_drift(&rows, &placed, shifted).is_some());
    }

    /// A registry that resolves nothing - the failure that leaves the
    /// placeholder embedding in the residual stream and answers about a
    /// picture the model never saw.
    #[test]
    fn an_unregistered_image_is_caught() {
        let chunks = [txt(2, 1), img(144, 1)];
        let (rows, _keys, placed) = plan(0, &chunks);
        assert_eq!(coverage_drift(&rows, &placed, |_, _| false), Some(2));
    }

    #[test]
    fn a_text_only_chunk_list_reserves_nothing() {
        let chunks = [txt(6, 1)];
        let (rows, _keys, placed) = plan(0, &chunks);
        assert!(placed.is_empty());
        assert_eq!(coverage_drift(&rows, &placed, |_, _| false), None);
    }

    /// granite-speech rides the same plan: a clip reserves its projector rows
    /// under the `<|audio|>` placeholder, exactly as a picture does under
    /// `<image>`.
    ///
    /// 16000 samples is 1 s, which is 50 encoder frames - and the projector
    /// pads up to whole 15-frame windows, so it is 4 windows × 3 queries = 12
    /// rows, not the 10 the steady-state rate would suggest. The "10 tokens
    /// per second" figure is asymptotic; every clip pays the round-up.
    #[test]
    fn a_clip_reserves_its_own_placeholder_run() {
        let chunks = [txt(3, 10), clip(16000), txt(2, 20)];
        let (rows, _keys, placed) = plan(1, &chunks);
        assert_eq!(
            placed,
            vec![(1, 3, 12)],
            "1 s of speech rounds up to 4 windows"
        );
        assert_eq!(rows.len(), 17);
        let toks: Vec<u32> = rows.iter().map(|r| r.2).collect();
        assert_eq!(toks[..3], [10, 11, 12]);
        assert!(
            toks[3..15].iter().all(|&t| t == APAD),
            "the clip's whole run is the pad"
        );
        assert_eq!(toks[15..], [20, 21]);
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.1 as usize, i, "positions dense from 0 across the clip");
        }
    }

    /// The property for audio, and the reason clips are content-keyed:
    /// two prompts with the same words and same-LENGTH but different speech
    /// are indistinguishable in the row stream. Keying on the placeholder
    /// would answer one caller's question about the other's recording.
    #[test]
    fn different_clips_give_different_radix_keys() {
        let a = [
            txt(4, 1),
            MmChunk::Audio {
                samples: vec![0.1; 16000],
                mel: None,
            },
        ];
        let b = [
            txt(4, 1),
            MmChunk::Audio {
                samples: vec![0.2; 16000],
                mel: None,
            },
        ];
        let (rows_a, keys_a, _) = plan(0, &a);
        let (rows_b, keys_b, _) = plan(0, &b);
        assert_eq!(
            rows_a, rows_b,
            "the ROW streams are indistinguishable - that is the trap"
        );
        assert_ne!(keys_a, keys_b, "the KEY streams must not be");
        assert_eq!(
            &keys_a[..4],
            &keys_b[..4],
            "the shared text prefix still matches"
        );
        // ...and the same clip keys identically, or a follow-up question about
        // the same recording never resumes past it
        assert_eq!(plan(0, &a).1, keys_a);
    }

    /// An unregistered clip is the audio spelling of the silent failure: the
    /// `<|audio|>` embedding stays in the residual stream and the model
    /// transcribes fluent nothing.
    #[test]
    fn an_unregistered_clip_is_caught() {
        let chunks = [txt(2, 1), clip(48000)];
        let (rows, _keys, placed) = plan(0, &chunks);
        assert_eq!(placed, vec![(1, 2, 30)]);
        assert_eq!(coverage_drift(&rows, &placed, |_, _| false), Some(2));
        assert_eq!(
            coverage_drift(&rows, &placed, hit_from(&placed, 0, 1)),
            Some(2)
        );
    }

    /// The chunked lane advances this row stream in pieces, so the guard's
    /// promise has to hold under any cut: a per-row check that passes over the
    /// whole prompt cannot fail over a slice of it.
    #[test]
    fn no_cut_can_disagree_once_the_whole_prompt_agrees() {
        let chunks = [txt(7, 1), img(300, 1), txt(5, 90), img(144, 1)];
        let (rows, _keys, placed) = plan(0, &chunks);
        let hit = hit_from(&placed, 0, 0);
        assert_eq!(coverage_drift(&rows, &placed, &hit), None);
        for cut in [1usize, 17, 128, 256, 4096] {
            for chunk in rows.chunks(cut) {
                // same predicate, same placements - a slice re-checked against
                // its own absolute row indices still agrees
                let bad = chunk.iter().find(|&&(s, p, _)| {
                    let i = p as usize;
                    placed.iter().any(|&(_, b, n)| i >= b && i < b + n) != hit(s, i)
                });
                assert!(bad.is_none(), "cut {cut} disagreed at {bad:?}");
            }
        }
    }
}
