//! Picture prompts (Nemotron 3.5 Super VL) and recording prompts (Nemotron
//! 3.5 Transcribe): the request's interleaved text / media chunks laid out
//! as one row sequence and prefilled through the ordinary passes, each
//! media item's rows written over its placeholder rows right after the
//! pass's embedding gather.
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

/// A recording prompt on the chunked queue: its recordings' rows and where
/// each run sits in the prompt - (first prompt row, recording, rows).
pub(crate) struct QueuedMedia {
    pub pics: Vec<PictureRows>,
    pub at: Vec<(usize, usize, usize)>,
    /// the speech lane's event after this prompt's encode: the first pass
    /// that reads the rows waits on it (device-side), then it clears
    pub ready: Option<std::sync::Arc<cudarc::driver::CudaEvent>>,
    /// the wave's encode on the speech worker and this prompt's recordings
    /// in it, collected into `pics` by the first pass that lends them
    pub encode: Option<(
        std::sync::Arc<std::sync::Mutex<SharedEncode>>,
        std::ops::Range<usize>,
    )>,
}

/// One admission wave's encode, shared by its prompts: the first to need its
/// rows collects the worker's reply, each takes its own recordings.
pub(crate) struct SharedEncode {
    pending: Option<super::audio::PendingEncode>,
    rows: Vec<Option<super::audio::AudioRows>>,
    ready: Option<std::sync::Arc<cudarc::driver::CudaEvent>>,
    /// the worker's refusal, kept for the first prompt that lends
    failed: Option<String>,
}

impl SharedEncode {
    /// Whether a pass can take this wave's rows without waiting: the worker
    /// has replied and the lane has run the encode (never blocks). A wave
    /// that failed reads ready - its first lend reports the failure.
    pub(crate) fn landed(&mut self, exec: &crate::gpu::GpuExecutor) -> bool {
        if let Some(p) = &self.pending {
            match p.try_take() {
                None => return false,
                Some(Ok(e)) => {
                    self.rows = e.rows.into_iter().map(Some).collect();
                    self.ready = Some(e.ready);
                }
                Some(Err(e)) => self.failed = Some(e.to_string()),
            }
            self.pending = None;
        }
        self.ready.as_ref().is_none_or(|ev| exec.event_done(ev))
    }

    fn take(
        &mut self,
        range: std::ops::Range<usize>,
    ) -> Result<(Vec<PictureRows>, std::sync::Arc<cudarc::driver::CudaEvent>), GpuModelError> {
        if let Some(p) = self.pending.take() {
            let e = p.wait()?;
            self.rows = e.rows.into_iter().map(Some).collect();
            self.ready = Some(e.ready);
        }
        if let Some(e) = &self.failed {
            return Err(GpuModelError::Unsupported(e.clone()));
        }
        let ready = self.ready.clone().expect("collected encode");
        let pics = range
            .map(|i| {
                let r = self.rows[i].take().expect("a recording's rows lent once");
                PictureRows {
                    rows: r.rows,
                    n: r.n,
                }
            })
            .collect();
        Ok((pics, ready))
    }
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

    /// Encode the request's pictures (or recordings) and lay its chunks out
    /// as rows.
    fn mm_layout(&mut self, chunks: &[MmChunk]) -> Result<MmRows, GpuModelError> {
        if self.audio.is_some() {
            return self.audio_layout(chunks);
        }
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

    /// Transcribe: each recording becomes its encoder's rows over that many
    /// placeholder rows. The template renders one `<|audio|>` a recording and
    /// the runner splits there; the expansion to one placeholder per 80 ms is
    /// the processor's, so it lives here. The placeholder id is past the
    /// vocabulary (the tokenizer appends it), so its rows gather token 0 -
    /// Transformers' lookup too - before the overlay replaces them.
    fn audio_layout(&mut self, chunks: &[MmChunk]) -> Result<MmRows, GpuModelError> {
        let enc = self.audio.as_ref().expect("speech encoder");
        let (pcms, caps) = recordings(chunks, enc.has_ctc()).map_err(GpuModelError::Unsupported)?;
        let n_rec = pcms.len();
        let done = enc.encode(pcms, false, caps)?;
        // the caller's passes read the rows next (speech lane, device-side)
        self.exec.wait_event(&done.ready)?;
        let mut recs = done.rows.into_iter();
        let mut rows = MmRows {
            tokens: Vec::new(),
            pics: Vec::with_capacity(n_rec),
            at: Vec::with_capacity(n_rec),
        };
        for c in chunks {
            match c {
                MmChunk::Text(ids) => rows.tokens.extend_from_slice(ids),
                MmChunk::Audio { .. } => {
                    let rec = recs.next().expect("one encode a recording");
                    rows.at.push((rows.tokens.len(), rows.pics.len(), rec.n));
                    rows.tokens.extend(std::iter::repeat_n(0, rec.n));
                    rows.pics.push(PictureRows {
                        rows: rec.rows,
                        n: rec.n,
                    });
                }
                MmChunk::CtcTimestamps { .. } => {}
                _ => unreachable!("refused by recordings()"),
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

    /// A wave of recording prompts into their slots together (Transcribe):
    /// every recording through ONE speech-encoder call (their windows share
    /// passes), then every prompt's rows concatenated into shared prefill
    /// passes - the coalesced text prefill's shape - each recording's rows
    /// over its placeholders wherever a pass cut falls. Per item, order kept:
    /// the last row's logits and the slot's row count. Like a single
    /// recording prompt, none resumes from the prefix cache or inserts.
    pub(crate) fn mm_prefill_batch(
        &mut self,
        items: &[(usize, Vec<MmChunk>)],
    ) -> Result<Vec<(Vec<f32>, usize)>, GpuModelError> {
        // every recording of the wave, encoded together
        let enc = self.audio.as_ref().expect("speech encoder");
        let (mut pcms, mut caps) = (Vec::new(), Vec::new());
        for (_, chunks) in items {
            let (p, c) = recordings(chunks, enc.has_ctc()).map_err(GpuModelError::Unsupported)?;
            pcms.extend(p);
            caps.extend(c);
        }
        let done = enc.encode(pcms, false, caps)?;
        // the wave's passes read the rows next: their stream waits on the
        // speech lane (device-side)
        self.exec.wait_event(&done.ready)?;
        let recs = done.rows;
        // the wave's row stream: (slot, pos, tok) per row, the recordings'
        // runs as (first stream row, recording, rows)
        let mut stream: Vec<(u32, u32, u32)> = Vec::new();
        let mut at: Vec<(usize, usize, usize)> = Vec::new();
        let mut last_row = Vec::with_capacity(items.len());
        let mut lens = Vec::with_capacity(items.len());
        let mut ri = 0usize;
        for (slot, chunks) in items {
            let mut pos = 0u32;
            for c in chunks {
                match c {
                    MmChunk::Text(ids) => {
                        for &t in ids {
                            stream.push((*slot as u32, pos, t));
                            pos += 1;
                        }
                    }
                    MmChunk::Audio { .. } => {
                        let n = recs[ri].n;
                        at.push((stream.len(), ri, n));
                        for _ in 0..n {
                            stream.push((*slot as u32, pos, 0));
                            pos += 1;
                        }
                        ri += 1;
                    }
                    MmChunk::CtcTimestamps { .. } => {}
                    _ => unreachable!("refused above"),
                }
            }
            if pos as usize > self.max_ctx {
                return Err(GpuModelError::ContextExceeded {
                    got: pos as usize,
                    max: self.max_ctx,
                });
            }
            last_row.push(stream.len() - 1);
            lens.push(pos as usize);
        }
        for ((slot, _), &n) in items.iter().zip(&lens) {
            self.admit_rows(*slot, n)?;
            self.last_reused[*slot] = 0;
            self.reply_release(*slot);
        }
        let slots: Vec<u32> = items.iter().map(|(s, _)| *s as u32).collect();
        let ends: Vec<u32> = lens.iter().map(|&n| (n - 1) as u32).collect();
        self.ensure_rows(&slots, &ends)?;
        let mut ov = MmOverlay {
            pics: recs
                .into_iter()
                .map(|r| PictureRows {
                    rows: r.rows,
                    n: r.n,
                })
                .collect(),
            spans: Vec::new(),
        };
        let mut out: Vec<(Vec<f32>, usize)> = vec![(Vec::new(), 0); items.len()];
        let mut base = 0usize;
        for chunk in stream.chunks(self.prefill_chunk) {
            let r = chunk.len();
            ov.spans = spans_for(&at, base, r, 0);
            self.mm = Some(ov);
            let pass = self.rows_pass_body(chunk, 0, Vec::new());
            ov = self.mm.take().expect("overlay");
            pass?;
            // finishers read inside the pass (the next chunk's gather
            // overwrites the rows), ascending - head_row bounces through x[0]
            let mut fin: Vec<(usize, usize)> = last_row
                .iter()
                .enumerate()
                .filter(|&(_, &lr)| lr >= base && lr < base + r)
                .map(|(it, &lr)| (lr - base, it))
                .collect();
            fin.sort_unstable();
            for (row, it) in fin {
                out[it] = (self.head_row(row)?, lens[it]);
            }
            base += r;
        }
        Ok(out)
    }

    /// Admit a wave of recording prompts onto the CHUNKED queue (Transcribe):
    /// every recording through one speech-encoder call, then each prompt
    /// queued with its rows as an ordinary chunked prompt - the mixed ticks
    /// carry its rows beside the decode band, so a long recording's prefill
    /// never freezes the streams already decoding. Like every recording
    /// prompt it resumes nothing from the prefix cache and inserts nothing.
    pub(crate) fn prefill_begin_audio(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, crate::generator::MmAdmit)> {
        use crate::generator::{GenError, MmAdmit};
        let fail = |items: Vec<(usize, Vec<MmChunk>)>, msg: String| {
            items
                .into_iter()
                .map(|(k, _)| (k, MmAdmit::Failed(GenError::Backend(msg.clone()))))
                .collect::<Vec<_>>()
        };
        let enc = self.audio.as_ref().expect("speech encoder");
        let (mut pcms, mut caps) = (Vec::new(), Vec::new());
        for (_, chunks) in &items {
            let (p, c) = match recordings(chunks, enc.has_ctc()) {
                Ok(r) => r,
                Err(e) => return fail(items, e),
            };
            // the encoder's own refusal, up front: once handed to the worker
            // a recording can no longer fail its prompt
            if p.iter()
                .any(|s| s.len() < 160 || s.iter().any(|v| !v.is_finite()))
            {
                return fail(
                    items,
                    "expected at least 10 ms of finite mono 16-kHz audio".into(),
                );
            }
            pcms.extend(p);
            caps.extend(c);
        }
        // handed to the speech worker and NOT waited on: the passes before
        // a prompt's first chunk run beside the encode, and that chunk's pass
        // collects its rows and waits on the lane's event (lend_media). A
        // recording's row count is its length's, so the prompt queues now.
        let n_recs = pcms.len();
        let pending = match enc.submit(pcms, false, caps) {
            Ok(p) => p,
            Err(e) => return fail(items, e.to_string()),
        };
        let shared = std::sync::Arc::new(std::sync::Mutex::new(SharedEncode {
            pending: Some(pending),
            rows: Vec::with_capacity(n_recs),
            ready: None,
            failed: None,
        }));
        let mut ri = 0usize;
        let mut out = Vec::with_capacity(items.len());
        for (slot, chunks) in items {
            let mut tokens = Vec::new();
            let mut media = QueuedMedia {
                pics: Vec::new(),
                at: Vec::new(),
                ready: None,
                encode: None,
            };
            let first = ri;
            for c in &chunks {
                match c {
                    MmChunk::Text(ids) => tokens.extend_from_slice(ids),
                    MmChunk::Audio { samples, .. } => {
                        let n = paddock_models::transcribe::audio_tokens(samples.len());
                        media.at.push((tokens.len(), ri - first, n));
                        tokens.extend(std::iter::repeat_n(0, n));
                        ri += 1;
                    }
                    MmChunk::CtcTimestamps { .. } => {}
                    _ => unreachable!("refused above"),
                }
            }
            media.encode = Some((shared.clone(), first..ri));
            let verdict = (|| {
                if tokens.len() > self.max_ctx {
                    return Err(GpuModelError::ContextExceeded {
                        got: tokens.len(),
                        max: self.max_ctx,
                    });
                }
                self.chunked.retain(|c| c.slot != slot);
                self.admit_rows(slot, tokens.len())?;
                self.last_reused[slot] = 0;
                // the prompt's pages up front: a text prompt gets them from
                // its prefix resume, which a recording prompt skips, and a
                // mixed tick backs only its decode rows
                self.ensure_rows(&[slot as u32], &[(tokens.len() - 1) as u32])?;
                self.reply_release(slot);
                Ok(())
            })();
            match verdict {
                Ok(()) => {
                    self.chunked.push(batch::ChunkedPrefill {
                        slot,
                        keys: tokens.clone(),
                        tokens,
                        cursor: 0,
                        media: Some(media),
                    });
                    out.push((slot, MmAdmit::Queued));
                }
                Err(e) => out.push((slot, MmAdmit::Failed(GenError::Backend(e.to_string())))),
            }
        }
        out
    }

    /// Lend the media rows of the recording prompts a tick's take carries to
    /// the overlay, spans at the tick's own row offsets (the decode band
    /// first). Returns which entries lent how many items, for
    /// [`Self::return_media`].
    pub(super) fn lend_media(
        &mut self,
        take: &[(usize, usize, bool)],
        dec_n: usize,
    ) -> Result<Vec<(usize, usize)>, GpuModelError> {
        let mut lent = Vec::new();
        let mut ov = MmOverlay::default();
        let mut row = dec_n;
        for &(qi, n, _) in take {
            let c = &mut self.chunked[qi];
            if let Some(m) = c.media.as_mut() {
                if let Some((sh, range)) = m.encode.take() {
                    let (pics, ready) = sh
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take(range)?;
                    m.pics = pics;
                    m.ready = Some(ready);
                }
                // the encode that made these rows ran on the speech lane
                if let Some(ev) = m.ready.take() {
                    self.exec.wait_event(&ev)?;
                }
                let base = ov.pics.len();
                for (dst, pi, src, cnt) in spans_for(&m.at, c.cursor, n, row) {
                    ov.spans.push((dst, base + pi, src, cnt));
                }
                lent.push((qi, m.pics.len()));
                ov.pics.append(&mut m.pics);
            }
            row += n;
        }
        if !lent.is_empty() {
            self.mm = Some(ov);
        }
        Ok(lent)
    }

    /// Hand the lent media rows back to their entries.
    pub(super) fn return_media(&mut self, lent: Vec<(usize, usize)>) {
        if lent.is_empty() {
            return;
        }
        let mut ov = self.mm.take().expect("lent overlay");
        let mut pics = ov.pics.drain(..);
        for (qi, n) in lent {
            let m = self.chunked[qi].media.as_mut().expect("lending entry");
            m.pics.extend(pics.by_ref().take(n));
        }
    }

    /// Whether this checkpoint carries the picture tower.
    pub fn has_pictures(&self) -> bool {
        self.vision.is_some()
    }

    /// Whether this checkpoint carries the speech encoder.
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Block until every queued recording's encode has landed (the gates
    /// pin which pass a prompt rides).
    #[doc(hidden)]
    pub fn settle_encodes_probe(&mut self) -> Result<(), GpuModelError> {
        for c in &self.chunked {
            if let Some((sh, _)) = c.media.as_ref().and_then(|m| m.encode.as_ref()) {
                let mut g = sh.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(p) = g.pending.take() {
                    let e = p.wait()?;
                    g.rows = e.rows.into_iter().map(Some).collect();
                    g.ready = Some(e.ready);
                }
                if let Some(ev) = &g.ready {
                    ev.synchronize().map_err(crate::gpu::from_driver)?;
                }
            }
        }
        Ok(())
    }

    /// Several recordings through the speech encoder together (the gates'
    /// probe): each one's language-model rows.
    #[doc(hidden)]
    pub fn audio_rows_probe(&mut self, pcms: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, GpuModelError> {
        let hidden = self.hp.hidden;
        let enc = self.audio.as_ref().ok_or_else(|| {
            GpuModelError::Unsupported("this checkpoint has no speech encoder".into())
        })?;
        let done = enc.encode(pcms.to_vec(), false, Vec::new())?;
        self.exec.wait_event(&done.ready)?;
        done.rows
            .iter()
            .map(|r| Ok(self.exec.to_host_len(&r.rows, r.n * hidden)?))
            .collect()
    }

    /// One recording through the speech encoder alone (the gates' probe):
    /// the language-model rows that replace its placeholders and the
    /// stages before them.
    pub fn audio_probe(
        &mut self,
        pcm: &[f32],
    ) -> Result<(Vec<f32>, super::ProbeStages), GpuModelError> {
        let hidden = self.hp.hidden;
        let enc = self.audio.as_ref().ok_or_else(|| {
            GpuModelError::Unsupported("this checkpoint has no speech encoder".into())
        })?;
        let mut p = enc.encode(vec![pcm.to_vec()], true, Vec::new())?;
        self.exec.wait_event(&p.ready)?;
        let r = p.rows.pop().expect("the probe's recording");
        let rows = self.exec.to_host_len(&r.rows, r.n * hidden)?;
        Ok((rows, p.stages.expect("kept stages")))
    }

    /// Whether the speech branch runs the checkpoint's W8A8 class (the gates
    /// pick their reference and bands by it).
    #[doc(hidden)]
    pub fn speech_w8a8_probe(&self) -> bool {
        self.audio.as_ref().is_some_and(|a| a.w8a8)
    }

    /// The CTC timestamp head over given speech-branch states `[t_enc][1280]`
    /// (the gates' probe): host `[2 * t_enc][cols]` log probabilities.
    #[doc(hidden)]
    pub fn ctc_columns_probe(
        &mut self,
        states: &[f32],
        cols: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        let enc = self.audio.as_ref().filter(|a| a.has_ctc()).ok_or_else(|| {
            GpuModelError::Unsupported("this checkpoint has no CTC timestamp head".into())
        })?;
        let dev = self.exec.to_device(states)?;
        // the speech lane reads it on its own stream
        self.exec.synchronize()?;
        enc.head_columns(dev, states.len() / 1280, cols.to_vec())
    }

    /// One recording through the speech encoder with its capture kept, then
    /// `transcript` aligned against it (the gates' probe): the
    /// `ctc_timestamps` object a request asking for it gets.
    #[doc(hidden)]
    pub fn ctc_timestamps_probe(
        &mut self,
        pcm: &[f32],
        transcript: &str,
    ) -> Result<serde_json::Value, GpuModelError> {
        let enc = self.audio.as_ref().ok_or_else(|| {
            GpuModelError::Unsupported("this checkpoint has no speech encoder".into())
        })?;
        let aligner = enc.aligner().ok_or_else(|| {
            GpuModelError::Unsupported("this checkpoint has no CTC timestamp head".into())
        })?;
        let id = u64::MAX;
        let done = enc.encode(vec![pcm.to_vec()], false, vec![Some(id)])?;
        self.exec.wait_event(&done.ready)?;
        Ok(aligner.align(id, transcript))
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

/// One Transcribe request's recordings in order and, for each, the capture
/// id its word timestamps are kept under: a `CtcTimestamps` directive names
/// the request's one recording (the reference's contract - word timestamps
/// take a single audio a request).
fn recordings(chunks: &[MmChunk], ctc: bool) -> Result<(Vec<Vec<f32>>, Vec<Option<u64>>), String> {
    let mut pcms = Vec::new();
    let mut id = None;
    for c in chunks {
        match c {
            MmChunk::Text(_) => {}
            MmChunk::Audio { samples, .. } => pcms.push(samples.clone()),
            MmChunk::CtcTimestamps { id: i } => id = Some(*i),
            _ => return Err("Nemotron 3.5 Transcribe takes text and audio here".into()),
        }
    }
    let caps = match id {
        None => vec![None; pcms.len()],
        Some(_) if !ctc => {
            return Err("this checkpoint ships no CTC timestamp head".into());
        }
        Some(i) if pcms.len() == 1 => vec![Some(i)],
        Some(_) => return Err("CTC timestamps take exactly one audio input a request".into()),
    };
    Ok((pcms, caps))
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
