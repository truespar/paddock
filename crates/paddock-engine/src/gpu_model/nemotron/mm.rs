//! Picture prompts (Nemotron 3.5 Super VL): the request's interleaved
//! text / picture chunks laid out as one row sequence and prefilled through
//! the ordinary passes, each picture's rows written over its placeholder
//! rows right after the pass's embedding gather.
//!
//! A picture becomes the processor's expansion of its `<image>`: `<img>`,
//! the picture's rows (they carry the `<image>` id, which the gather then
//! overwrites), `</img>`. The chat template renders the bare `<image>` the
//! runner splits at; the wrapping is the processor's, so it lives here.
//!
//! **The prefix cache stays out of it.** Every picture row carries the same
//! placeholder id, so two different pictures under the same text would key
//! the same radix path - a picture prompt neither resumes from the cache nor
//! inserts into it, and its slot's reply tracking is released (the reply
//! checkpoints would key the same rows).

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::service::MmChunk;

use super::vision::PictureRows;
use super::*;

/// Picture rows a prefill pass carries. After the pass's embedding gather
/// each span's rows overwrite the placeholder rows.
#[derive(Default)]
pub(crate) struct MmOverlay {
    pub pics: Vec<PictureRows>,
    /// (first row of the span in the pass, picture, first picture row, rows)
    pub spans: Vec<(usize, usize, usize, usize)>,
}

/// One picture through the tower alone (the gates' probe): its tile
/// (patches across, down), the normalized patch rows as the patch GEMM reads
/// them, and the language-model rows that replace its placeholders.
pub struct PictureProbe {
    pub tile: (usize, usize),
    pub patches: Vec<half::f16>,
    pub rows: Vec<f32>,
}

/// A picture prompt as rows: the ids (picture rows carry the placeholder)
/// and where each picture's run starts.
struct MmRows {
    tokens: Vec<u32>,
    pics: Vec<PictureRows>,
    /// (first row, picture, rows)
    at: Vec<(usize, usize, usize)>,
}

/// The parts of every picture run inside sequence rows `[base, base + len)`,
/// as pass spans starting at pass row `row0`.
fn spans_for(
    at: &[(usize, usize, usize)],
    base: usize,
    len: usize,
    row0: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let mut out = Vec::new();
    for &(first, pi, n) in at {
        let lo = first.max(base);
        let hi = (first + n).min(base + len);
        if lo < hi {
            out.push((row0 + lo - base, pi, lo - first, hi - lo));
        }
    }
    out
}

impl GpuNemotron {
    /// Write the overlay's spans over `x`'s rows (`hidden` wide).
    pub(super) fn apply_overlay(
        exec: &GpuExecutor,
        mm: Option<&MmOverlay>,
        x: &mut cudarc::driver::CudaSlice<f32>,
        hidden: usize,
    ) -> Result<(), GpuError> {
        let Some(mm) = mm else {
            return Ok(());
        };
        for &(dst, pi, src, n) in &mm.spans {
            exec.copy_region(&mm.pics[pi].rows, src * hidden, x, dst * hidden, n * hidden)?;
        }
        Ok(())
    }

    /// Encode the request's pictures and lay its chunks out as rows.
    fn mm_layout(&mut self, chunks: &[MmChunk]) -> Result<MmRows, GpuModelError> {
        let vis = self.vision.as_mut().ok_or_else(|| {
            GpuModelError::Unsupported("this checkpoint has no picture tower".into())
        })?;
        let n_pics = chunks
            .iter()
            .filter(|c| matches!(c, MmChunk::Image { .. }))
            .count();
        let (start, end, image) = (vis.cfg.start_id, vis.cfg.end_id, vis.cfg.image_id);
        let mut rows = MmRows {
            tokens: Vec::new(),
            pics: Vec::with_capacity(n_pics),
            at: Vec::with_capacity(n_pics),
        };
        for c in chunks {
            match c {
                MmChunk::Text(ids) => rows.tokens.extend_from_slice(ids),
                MmChunk::Image { rgb, w, h } => {
                    let pic = vis.encode(rgb, *w, *h, n_pics)?;
                    rows.tokens.push(start);
                    rows.at.push((rows.tokens.len(), rows.pics.len(), pic.n));
                    rows.tokens.extend(std::iter::repeat_n(image, pic.n));
                    rows.tokens.push(end);
                    rows.pics.push(pic);
                }
                _ => {
                    return Err(GpuModelError::Unsupported(
                        "Nemotron 3.5 Super VL takes text and pictures here (no audio or \
                         processor directives)"
                            .into(),
                    ));
                }
            }
        }
        if rows.tokens.len() > self.max_ctx {
            return Err(GpuModelError::ContextExceeded {
                got: rows.tokens.len(),
                max: self.max_ctx,
            });
        }
        Ok(rows)
    }

    /// The serial lane's picture prompt: fresh state, the rows through the
    /// chunked bulk prefill. Returns the last row's logits and the row count.
    pub(crate) fn mm_prefill_serial(
        &mut self,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let rows = self.mm_layout(chunks)?;
        crate::generator::Generator::reset(self);
        self.ensure_decode()?;
        self.ensure_prefill()?;
        let n = rows.tokens.len();
        let mut ov = MmOverlay {
            pics: rows.pics,
            spans: Vec::new(),
        };
        let mut done = 0usize;
        let mut logits = None;
        for chunk in rows.tokens.chunks(self.prefill_chunk) {
            ov.spans = spans_for(&rows.at, done, chunk.len(), 0);
            self.mm = Some(ov);
            let r = self.prefill_chunk(chunk, done + chunk.len() == n);
            ov = self.mm.take().expect("overlay");
            logits = r?;
            done += chunk.len();
        }
        Ok((logits.expect("non-empty prompt"), n))
    }

    /// The batch lane's picture prompt into `slot`: admitted fresh (no cache
    /// resume), the rows through the ordinary passes. Returns the last row's
    /// logits and the slot's row count.
    pub(crate) fn mm_prefill_slot(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let rows = self.mm_layout(chunks)?;
        let n = rows.tokens.len();
        self.admit_rows(slot, n)?;
        self.last_reused[slot] = 0;
        self.ensure_rows(&[slot as u32], &[(n - 1) as u32])?;
        self.reply_release(slot);
        let mut ov = MmOverlay {
            pics: rows.pics,
            spans: Vec::new(),
        };
        let mut base = 0usize;
        let mut last_len = 0usize;
        for chunk in rows.tokens.chunks(self.prefill_chunk) {
            let pass: Vec<(u32, u32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(j, &t)| (slot as u32, (base + j) as u32, t))
                .collect();
            ov.spans = spans_for(&rows.at, base, chunk.len(), 0);
            self.mm = Some(ov);
            let r = self.rows_pass_body(&pass, 0, Vec::new());
            ov = self.mm.take().expect("overlay");
            r?;
            base += chunk.len();
            last_len = chunk.len();
        }
        Ok((self.head_row(last_len - 1)?, n))
    }

    /// Whether this checkpoint carries the picture tower.
    pub fn has_pictures(&self) -> bool {
        self.vision.is_some()
    }

    /// Encode one picture (`rgb` interleaved, `w x h`; one of `n_pictures`
    /// in its request) and read every stage back.
    pub fn picture_probe(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
        n_pictures: usize,
    ) -> Result<PictureProbe, GpuModelError> {
        let vis = self.vision.as_mut().ok_or_else(|| {
            GpuModelError::Unsupported("this checkpoint has no picture tower".into())
        })?;
        let tile = vis.tile(w, h, n_pictures);
        let pic = vis.encode(rgb, w, h, n_pictures)?;
        let patches = vis.last_patches(tile.0 * tile.1)?;
        let rows = self.exec.to_host_len(&pic.rows, pic.n * self.hp.hidden)?;
        Ok(PictureProbe {
            tile,
            patches,
            rows,
        })
    }

    /// What a picture costs this endpoint (see `Generator::vision_budget`).
    pub(crate) fn vision_budget_impl(&self) -> Option<crate::generator::VisionBudget> {
        let vis = self.vision.as_ref()?;
        let c = &vis.cfg;
        let px = (c.patch * c.patch) as u64;
        let per_token = px * (c.shuffle * c.shuffle) as u64;
        Some(crate::generator::VisionBudget {
            max_pixels: c.max_patches as u64 * px,
            min_pixels: c.min_patches as u64 * px,
            max_edge: None,
            pixels_per_token: per_token,
            max_tokens: (c.max_patches / (c.shuffle * c.shuffle)) as u32,
            min_tokens: (c.min_patches / (c.shuffle * c.shuffle)) as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_cut_picture_runs_at_pass_bounds() {
        // picture 0 at rows 3..7, picture 1 at rows 10..12
        let at = [(3usize, 0usize, 4usize), (10, 1, 2)];
        // a pass over rows 5..11: the tail of 0, the head of 1
        assert_eq!(spans_for(&at, 5, 6, 0), vec![(0, 0, 2, 2), (5, 1, 0, 1)]);
        assert_eq!(spans_for(&at, 0, 3, 0), vec![]);
        // the whole prompt in one pass that starts at pass row 7
        assert_eq!(spans_for(&at, 0, 20, 7), vec![(10, 0, 0, 4), (17, 1, 0, 2)]);
    }
}
