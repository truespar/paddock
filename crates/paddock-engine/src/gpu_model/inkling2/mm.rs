//! Picture prompts: a request's interleaved text / picture chunks laid out
//! as one prompt and prefilled through the ordinary chunked queue, each
//! picture's rows written over its placeholder rows right after a pass's
//! embedding (forward.rs `walk`).
//!
//! A picture becomes the processor's expansion of its one
//! `<|unused_200054|>`: that id once per 40 x 40 cell (the chat template
//! already wraps it in `<|content_image|> .. <|end_message|>`). The runner
//! cuts the rendered prompt at the placeholder, so a picture chunk here
//! carries no id of its own.
//!
//! A wave's pictures go through the tower at admission (passes of up to
//! `vision::PASS_ROWS` rows); their prompts then advance under the mixed
//! ticks' row budget like any text prompt, a chunk boundary cutting a
//! picture's rows anywhere.

use std::sync::Arc;

use paddock_models::inkling2 as ik;
use paddock_models::inkling2::vision as iv;

use super::GpuInkling2;
use super::vision::{Picture, PictureRows};
use crate::generator::{MmAdmit, VisionBudget};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::service::MmChunk;

/// Picture rows a pass writes: rows `[from, from + n)` of `pic` over pass
/// rows `[row, row + n)`.
pub(crate) struct Span {
    pub row: usize,
    pub pic: Arc<PictureRows>,
    pub from: usize,
    pub n: usize,
}

/// The parts of `pics` (each `(first prompt row, rows)`) that prompt rows
/// `[lo, hi)` cover, at pass rows from `base`.
pub(crate) fn spans(
    pics: &[(usize, Arc<PictureRows>)],
    lo: usize,
    hi: usize,
    base: usize,
) -> Vec<Span> {
    pics.iter()
        .filter_map(|(at, p)| {
            let (a, b) = ((*at).max(lo), (at + p.n).min(hi));
            (a < b).then(|| Span {
                row: base + a - lo,
                pic: p.clone(),
                from: a - at,
                n: b - a,
            })
        })
        .collect()
}

/// A request's prompt with each picture expanded to its placeholder run,
/// and where each run starts (pictures in order).
fn layout(chunks: &[MmChunk]) -> Result<(Vec<u32>, Vec<usize>, Vec<Picture<'_>>), String> {
    let mut ids = Vec::new();
    let mut at = Vec::new();
    let mut pics = Vec::new();
    for c in chunks {
        match c {
            MmChunk::Text(t) => ids.extend_from_slice(t),
            MmChunk::Image { rgb, w, h } => {
                let p = Picture { rgb, w: *w, h: *h };
                at.push(ids.len());
                ids.extend(std::iter::repeat_n(ik::IMAGE_TOKEN, p.tokens()));
                pics.push(p);
            }
            _ => return Err("Inkling 2 takes text and pictures".into()),
        }
    }
    Ok((ids, at, pics))
}

impl GpuInkling2 {
    /// Lay out `chunks`, encode its pictures, and queue the prompt on `slot`.
    fn queue_mm(&mut self, slot: usize, chunks: &[MmChunk]) -> Result<(), GpuModelError> {
        let (ids, at, pics) = layout(chunks).map_err(GpuModelError::Unsupported)?;
        let tower = self.vision.as_mut().ok_or_else(|| {
            GpuModelError::Unsupported("this endpoint serves Inkling 2 without pictures".into())
        })?;
        let rows = tower.encode(&pics)?;
        self.prefill_begin_impl(slot, ids)?;
        let p = self
            .queue
            .back_mut()
            .expect("prefill_begin_impl queued the prompt");
        p.pics = at.into_iter().zip(rows.into_iter().map(Arc::new)).collect();
        Ok(())
    }

    /// Queue a wave of picture prompts on the chunked queue.
    pub(crate) fn prefill_begin_mm_impl(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        items
            .into_iter()
            .map(|(slot, chunks)| {
                let r = match self.queue_mm(slot, &chunks) {
                    Ok(()) => MmAdmit::Queued,
                    Err(e) => MmAdmit::Failed(crate::generator::GenError::Backend(e.to_string())),
                };
                (slot, r)
            })
            .collect()
    }

    /// A picture prompt prefilled whole, now: its last logits and the rows
    /// its slot holds. The other queued prompts wait it out (they are set
    /// aside, not advanced).
    pub(crate) fn prefill_mm_slot_impl(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let others = std::mem::take(&mut self.queue);
        let r = self.queue_mm(slot, chunks).and_then(|()| {
            loop {
                let chunk = self.scratch.rows;
                let (_, mut done) = self.forward_mixed_impl(&[], chunk)?;
                if let Some((_, logits, rows)) = done.pop() {
                    break Ok((logits, rows));
                }
                if self.queue.is_empty() {
                    break Err(GpuModelError::Unsupported(
                        "inkling2: a picture prompt left the queue unfinished".into(),
                    ));
                }
            }
        });
        if r.is_err() {
            self.queue.clear();
        }
        let mine = std::mem::replace(&mut self.queue, others);
        debug_assert!(mine.is_empty());
        r
    }

    /// [`GpuInkling2::forward_each`] over a picture prompt on the single
    /// stream (slot 0, from its current position): the pictures through the
    /// tower, then the laid-out prompt a pass at a time with their rows over
    /// the placeholders'. Returns the prompt as laid out (each picture its
    /// placeholder run). The gates' teacher-forced entry.
    pub fn forward_each_mm(
        &mut self,
        chunks: &[MmChunk],
        want: super::Logits,
        mut each: impl FnMut(usize, &[f32]),
    ) -> Result<Vec<u32>, GpuModelError> {
        let (ids, at, pics) = layout(chunks).map_err(GpuModelError::Unsupported)?;
        let tower = self.vision.as_mut().ok_or_else(|| {
            GpuModelError::Unsupported("this endpoint serves Inkling 2 without pictures".into())
        })?;
        let rows = tower.encode(&pics)?;
        let pics: Vec<(usize, Arc<PictureRows>)> =
            at.into_iter().zip(rows.into_iter().map(Arc::new)).collect();
        let chunk = self.scratch.rows;
        let n = ids.len();
        let mut i = 0;
        while i < n {
            let k = chunk.min(n - i);
            let w = match want {
                super::Logits::All => super::Logits::All,
                super::Logits::Last if i + k == n => super::Logits::Last,
                _ => super::Logits::None,
            };
            let seg = super::forward::Seg {
                slot: 0,
                ids: &ids[i..i + k],
                want: w,
            };
            self.overlay = spans(&pics, i, i + k, 0);
            let r = self.forward_segs(&[seg], &mut |_, r, l| each(i + r, l));
            self.overlay.clear();
            r?;
            i += k;
        }
        Ok(ids)
    }

    /// What a picture costs this endpoint (see `Generator::vision_budget`):
    /// 1600 pixels a token and no resize of its own - the processor pads to
    /// whole cells and never upsamples. The pixel ceiling leaves room for
    /// that padding (a side rounds up to a whole cell, so a fitted picture
    /// costs up to (w + h) / 40 + 1 tokens over its area's share) under the
    /// edge cap, so a picture fitted to it never passes the tower's 8192.
    pub(crate) fn vision_budget_impl(&self) -> Option<VisionBudget> {
        self.vision.as_ref()?;
        let px = (iv::CELL * iv::CELL) as u64;
        Some(VisionBudget {
            max_pixels: (iv::MAX_TOKENS as u64 - 256) * px,
            min_pixels: 1,
            max_edge: Some(8000),
            pixels_per_token: px,
            max_tokens: iv::MAX_TOKENS as u32,
            min_tokens: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_expands_to_one_placeholder_a_cell() {
        let rgb = vec![0u8; 81 * 40 * 3];
        let chunks = [
            MmChunk::Text(vec![1, 2]),
            MmChunk::Image {
                rgb: rgb.clone(),
                w: 81,
                h: 40,
            },
            MmChunk::Text(vec![3]),
        ];
        let (ids, at, pics) = layout(&chunks).unwrap();
        // 81 x 40 is three cells across (the last padded), one down
        assert_eq!(
            ids,
            [1, 2, ik::IMAGE_TOKEN, ik::IMAGE_TOKEN, ik::IMAGE_TOKEN, 3]
        );
        assert_eq!(at, [2]);
        assert_eq!(pics[0].grid(), (1, 3));
    }

    #[test]
    fn the_budget_ceiling_keeps_a_fitted_picture_inside_the_tower() {
        // the worst aspect the edge cap allows at the pixel ceiling
        let max_px = (iv::MAX_TOKENS as u64 - 256) * 1600;
        let w = 8000usize;
        let h = (max_px / w as u64) as usize;
        let (r, c) = iv::grid(w, h);
        assert!(r * c <= iv::MAX_TOKENS, "{r} x {c}");
        // and a square one
        let s = (max_px as f64).sqrt() as usize;
        let (r, c) = iv::grid(s, s);
        assert!(r * c <= iv::MAX_TOKENS, "{r} x {c}");
    }
}
