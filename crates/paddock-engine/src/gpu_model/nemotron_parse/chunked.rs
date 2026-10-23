//! Admission, the encoder budget and the mixed tick.
//!
//! A request is one page: its rows are the rendered prompt's, in order - the
//! text before the picture (`pre` tokens), the picture's prefix (`n_img`
//! rows, the tower + neck output), then the rest of the text. The checkpoint
//! was trained on the picture first (its modeling code concatenates
//! `(vision, text_embeddings)`), which is what the runner renders for a
//! plain page request; a chat that puts text ahead of the page (a system
//! message - the template concatenates every message) keeps it there, as
//! vLLM's placeholder splice does. A request with no picture prefills its
//! text alone (the serial warm-up and text probes - the model's output is
//! meaningless without a page, but the pass is legal).
//!
//! - **Pixels** are prepared on a worker thread at admission (the Pillow
//!   resize of a 300 dpi scan is tens of ms) - `MmAdmit::Encoding` until the
//!   tower has run.
//! - **The encoder budget is one tower pass a tick**: the front entry, plus
//!   every following prep-ready page that still fits the pass's rows. A pass
//!   packs pages of different sizes (see `vision.rs`), so unlike the
//!   same-dims grouping of other towers any ready page can ride.
//! - **The mixed tick** runs the decode band and as many queued prefill rows
//!   as the scratch holds in one pass; picture rows are spliced into the
//!   embedding staging plane from the page's prefix plane by (slot, position).

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, TryRecvError};

use cudarc::driver::CudaSlice;

use crate::generator::{GenError, MmAdmit};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::granite::batch::pf_rows;
use crate::service::MmChunk;

use super::batch::PfCuts;
use super::load::GpuNemotronParse;
use super::preprocess::{PagePrep, prep_page};

fn gen_err(e: GpuModelError) -> GenError {
    GenError::Backend(e.to_string())
}

/// A prompt on the chunked queue.
pub(crate) struct ChunkedPrefill {
    pub slot: usize,
    /// text tokens ahead of the picture
    pub pre: usize,
    /// picture rows (0 = a text-only prompt), after `pre` text rows
    pub n_img: usize,
    pub text: Vec<u32>,
    /// next row to compute
    pub cursor: usize,
    /// the page's prefix rows `[n_img][d_model]` (None when n_img == 0)
    pub plane: Option<CudaSlice<f32>>,
}

impl ChunkedPrefill {
    pub fn rows(&self) -> usize {
        self.n_img + self.text.len()
    }

    /// The token of row `p`: text before the picture, the picture's
    /// placeholder over its rows, text after.
    fn token(&self, p: usize, img: u32) -> u32 {
        if p < self.pre {
            self.text[p]
        } else if p < self.pre + self.n_img {
            img
        } else {
            self.text[p - self.n_img]
        }
    }
}

/// A page waiting for the tower.
pub(crate) struct PageEnc {
    slot: usize,
    text: Vec<u32>,
    /// text tokens ahead of the picture
    pre: usize,
    rx: Receiver<PagePrep>,
    got: Option<PagePrep>,
    /// prep worker gone without a result (it panicked)
    lost: bool,
    /// request died while queued - still reports, queues nothing
    dead: bool,
}

/// A tower pass running on the tower's lane: its pages' requests, the
/// planes it is writing and the event that says it is done.
pub(crate) struct Flight {
    members: Vec<(usize, Vec<u32>, usize, bool)>, // (slot, text, pre, dead)
    planes: Vec<super::vision::PrefixPlane>,
    ev: cudarc::driver::CudaEvent,
}

/// The one picture of a request, all its text in order, and how many of
/// those tokens come before the picture.
#[allow(clippy::type_complexity)]
fn split_request(
    chunks: &[MmChunk],
) -> Result<(Option<(&[u8], usize, usize)>, Vec<u32>, usize), GpuModelError> {
    let mut img = None;
    let mut text = Vec::new();
    let mut pre = 0usize;
    for c in chunks {
        match c {
            MmChunk::Text(t) => {
                text.extend_from_slice(t);
                if img.is_none() {
                    pre = text.len();
                }
            }
            MmChunk::Image { rgb, w, h } => {
                if img.is_some() {
                    return Err(GpuModelError::Unsupported(
                        "nemotron-parse reads one page a request - send each page on its own"
                            .into(),
                    ));
                }
                img = Some((rgb.as_slice(), *w, *h));
            }
            MmChunk::VisionPixels { .. } => {} // the processor fixes the size itself
            MmChunk::Audio { .. } | MmChunk::OcrCrop(_) => {
                return Err(GpuModelError::Unsupported(
                    "nemotron-parse takes a page picture and text - routing bug".into(),
                ));
            }
        }
    }
    // a text-only prompt has nothing to sit ahead of
    Ok((img, text, if img.is_some() { pre } else { 0 }))
}

/// The row stream of a pass: `(slot, position, token)`.
type PassRows = Vec<(u32, u32, u32)>;

impl GpuNemotronParse {
    /// Rows a request will prefill (picture prefix + text) - host math only.
    fn plan_rows(&self, img: Option<(&[u8], usize, usize)>, text: usize) -> usize {
        img.map_or(0, |(_, w, h)| {
            self.cfg
                .prefix_rows(self.cfg.canvas_hw(self.cfg.content_hw(w, h)))
        }) + text
    }

    /// Queue a TEXT prompt for chunked prefill.
    pub(crate) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuModelError> {
        self.chunked.retain(|c| c.slot != slot);
        self.enc.retain(|e| e.slot != slot);
        self.admit(slot, tokens.len(), &tokens)?;
        self.chunked.push(ChunkedPrefill {
            slot,
            pre: 0,
            n_img: 0,
            text: tokens,
            cursor: 0,
            plane: None,
        });
        Ok(())
    }

    pub(crate) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        let n = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.len() != n
    }

    pub(crate) fn chunk_release(&mut self, occupied: &[bool]) {
        let occ = |s: usize| occupied.get(s).copied().unwrap_or(false);
        self.chunked.retain(|c| occ(c.slot));
        for e in &mut self.enc {
            if !occ(e.slot) {
                e.dead = true;
            }
        }
        if let Some(f) = self.flight.as_mut() {
            for m in &mut f.members {
                if !occ(m.0) {
                    m.3 = true;
                }
            }
        }
    }

    pub(crate) fn prefill_begin_multimodal_impl(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        items
            .into_iter()
            .map(|(slot, chunks)| match self.enc_admit(slot, chunks) {
                Ok(a) => (slot, a),
                Err(e) => (slot, MmAdmit::Failed(gen_err(e))),
            })
            .collect()
    }

    fn enc_admit(&mut self, slot: usize, chunks: Vec<MmChunk>) -> Result<MmAdmit, GpuModelError> {
        self.chunked.retain(|c| c.slot != slot);
        self.enc.retain(|e| e.slot != slot);
        if let Some(f) = self.flight.as_mut() {
            for m in &mut f.members {
                if m.0 == slot {
                    m.3 = true;
                }
            }
        }
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        if slot >= n_slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} >= enabled {n_slots}"
            )));
        }
        let (img, text, pre) = split_request(&chunks)?;
        let n_rows = self.plan_rows(img, text.len());
        if n_rows == 0 || n_rows > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "prompt {n_rows} rows out of range (max_ctx {})",
                self.max_ctx
            )));
        }
        let Some((rgb, w, h)) = img else {
            self.prefill_begin_impl(slot, text)?;
            return Ok(MmAdmit::Queued);
        };
        if rgb.len() != 3 * w * h || w == 0 || h == 0 {
            return Err(GpuModelError::Unsupported(format!(
                "page {w}x{h}: {} bytes is not RGB8",
                rgb.len()
            )));
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let rgb = rgb.to_vec();
        let cfg = self.cfg.clone();
        std::thread::spawn(move || {
            let _ = tx.send(prep_page(&rgb, w, h, &cfg));
        });
        self.enc.push_back(PageEnc {
            slot,
            text,
            pre,
            rx,
            got: None,
            lost: false,
            dead: false,
        });
        Ok(MmAdmit::Encoding)
    }

    /// Spend the tick's tower pass on the queue front (+ every ready page that
    /// fits behind it); report what queued.
    pub(crate) fn encode_step_impl(&mut self) -> Vec<(usize, MmAdmit)> {
        let mut out = Vec::new();
        // dead entries report first (the scheduler's encoding set only clears
        // on a report)
        self.enc.retain(|e| {
            if e.dead {
                out.push((e.slot, MmAdmit::Queued));
            }
            !e.dead
        });
        for e in &mut self.enc {
            if e.got.is_none() && !e.lost {
                match e.rx.try_recv() {
                    Ok(p) => e.got = Some(p),
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => e.lost = true,
                }
            }
        }
        // a lost prep fails its request
        let mut keep = VecDeque::new();
        while let Some(e) = self.enc.pop_front() {
            if e.lost {
                out.push((
                    e.slot,
                    MmAdmit::Failed(GenError::Backend("nemotron-parse page prep failed".into())),
                ));
            } else {
                keep.push_back(e);
            }
        }
        self.enc = keep;
        // a pass in flight: done -> its pages queue; running -> the tick goes
        // to decode while the tower's lane keeps working
        if let Some(f) = self.flight.take() {
            if !f.ev.is_complete() {
                self.flight = Some(f);
                return out;
            }
            for ((slot, text, pre, dead), plane) in f.members.into_iter().zip(f.planes) {
                if dead {
                    out.push((slot, MmAdmit::Queued));
                    continue;
                }
                let n_rows = plane.rows + text.len();
                match self.admit(slot, n_rows, &text) {
                    Ok(()) => {
                        self.chunked.push(ChunkedPrefill {
                            slot,
                            pre,
                            n_img: plane.rows,
                            text,
                            cursor: 0,
                            plane: Some(plane.plane),
                        });
                        out.push((slot, MmAdmit::Queued));
                    }
                    Err(err) => out.push((slot, MmAdmit::Failed(gen_err(err)))),
                }
            }
        }
        if self.enc.front().is_none_or(|e| e.got.is_none()) {
            return out; // the front's pixels are not ready - the tick goes to decode
        }
        // the pass: the front, then ready pages in order while rows fit
        let cap = self.tower.pass_rows();
        let mut take = 0usize;
        let mut rows = 0usize;
        for e in &self.enc {
            let Some(p) = &e.got else { break };
            let r = self.tower.page_rows(p);
            if take > 0 && rows + r > cap {
                break;
            }
            rows += r;
            take += 1;
        }
        let group: Vec<PageEnc> = self.enc.drain(..take).collect();
        let pages: Vec<&PagePrep> = group
            .iter()
            .map(|e| e.got.as_ref().expect("ready"))
            .collect();
        let launched = self
            .tower
            .encode(&pages)
            .and_then(|planes| Ok((planes, self.tower.record()?)));
        match launched {
            Ok((planes, ev)) => {
                self.flight = Some(Flight {
                    members: group
                        .into_iter()
                        .map(|e| (e.slot, e.text, e.pre, e.dead))
                        .collect(),
                    planes,
                    ev,
                });
            }
            Err(err) => {
                let msg = err.to_string();
                for e in &group {
                    out.push((e.slot, MmAdmit::Failed(GenError::Backend(msg.clone()))));
                }
            }
        }
        out
    }

    /// FIFO over the queue, up to `budget` rows; the last prompt may split.
    fn plan_chunk(&self, budget: usize) -> (PassRows, Vec<(usize, usize, bool)>) {
        let mut rows: PassRows = Vec::new();
        let mut take = Vec::new();
        let cap = budget.clamp(1, pf_rows());
        let img = self.cfg.image_token_id;
        for (qi, c) in self.chunked.iter().enumerate() {
            if rows.len() >= cap {
                break;
            }
            let remaining = c.rows() - c.cursor;
            let n = remaining.min(cap - rows.len()).max(1);
            for p in c.cursor..c.cursor + n {
                rows.push((c.slot as u32, p as u32, c.token(p, img)));
            }
            take.push((qi, n, n == remaining));
        }
        (rows, take)
    }

    /// One pass over `chunk` rows, the leading `dec` the decode band: embed,
    /// splice picture rows, norm, the layer walk.
    pub(super) fn rows_pass_body(
        &mut self,
        chunk: &PassRows,
        dec: usize,
    ) -> Result<(), GpuModelError> {
        let r = chunk.len();
        let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
        let slots: Vec<u32> = chunk.iter().map(|x| x.0).collect();
        let pos: Vec<u32> = chunk.iter().map(|x| x.1).collect();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, x) in chunk.iter().enumerate().skip(dec) {
            match runs.last_mut() {
                Some((off, n)) if chunk[*off].0 == x.0 => *n += 1,
                _ => runs.push((i, 1)),
            }
        }
        self.ensure_rows(&slots, &pos)?;
        self.upload_rows(&toks, &pos, &slots)?;
        if self.inv {
            self.upload_band_groups(&slots[..dec], &pos[..dec])?;
        }
        self.seen_band(dec)?;
        self.embed_rows(r)?;
        self.splice_picture_rows(chunk, dec)?;
        self.embed_norm(r)?;
        self.layer_walk(r, Some(&PfCuts { dec, runs }))
    }

    /// Overwrite picture rows' staged embeddings with their prefix rows.
    fn splice_picture_rows(&mut self, chunk: &PassRows, dec: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let dm = self.cfg.d_model;
        let (chunked, batch) = (&self.chunked, &mut self.batch);
        let bs = batch.as_mut().expect("batch enabled");
        let mut i = dec;
        while i < chunk.len() {
            let (s, p) = (chunk[i].0 as usize, chunk[i].1 as usize);
            let Some(e) = chunked.iter().find(|c| c.slot == s) else {
                i += 1;
                continue;
            };
            let pic = e.pre..e.pre + e.n_img;
            if !pic.contains(&p) {
                i += 1;
                continue;
            }
            // one contiguous run of this page's picture rows
            let mut n = 1usize;
            while i + n < chunk.len()
                && chunk[i + n].0 as usize == s
                && chunk[i + n].1 as usize == p + n
                && pic.contains(&(p + n))
            {
                n += 1;
            }
            let plane = e.plane.as_ref().expect("a picture entry carries its plane");
            exec.copy_region(plane, (p - e.pre) * dm, &mut bs.sc.xe, i * dm, n * dm)?;
            i += n;
        }
        Ok(())
    }

    /// Advance cursors; finished prompts leave the queue.
    fn commit_chunk(
        &mut self,
        take: &[(usize, usize, bool)],
        finished_raw: Vec<(usize, Vec<f32>)>,
    ) -> Vec<(usize, Vec<f32>, usize)> {
        for &(qi, n, _) in take {
            self.chunked[qi].cursor += n;
        }
        let mut out = Vec::new();
        for (qi, logits) in finished_raw {
            let (slot, n_rows) = (self.chunked[qi].slot, self.chunked[qi].rows());
            if slot == 0
                && let Some(bs) = self.batch.as_mut()
            {
                bs.slot0_pos = n_rows;
            }
            self.chunked[qi].text = Vec::new();
            self.chunked[qi].pre = 0;
            self.chunked[qi].n_img = 0; // marks it finished
            self.chunked[qi].plane = None;
            out.push((slot, logits, n_rows));
        }
        self.chunked.retain(|c| c.rows() > 0);
        out
    }

    /// Decode rows first (one band), then as much of the queue as fits.
    fn fuse_rows(
        &self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> (
        PassRows,
        usize,
        Vec<(usize, usize, usize)>,
        Vec<(usize, usize, bool)>,
    ) {
        let mut rows: PassRows = decodes.iter().map(|&(s, t, p)| (s as u32, p, t)).collect();
        let dec_n = rows.len();
        let bs = self.batch.as_ref().expect("batch enabled");
        let room = bs.cap.saturating_sub(dec_n);
        let (chunk_rows, take) = if room == 0 {
            (Vec::new(), Vec::new())
        } else {
            self.plan_chunk(budget.min(room))
        };
        rows.extend_from_slice(&chunk_rows);
        // finishers: (pass row, queue index, slot)
        let mut fin = Vec::new();
        let mut off = dec_n;
        for &(qi, n, done) in &take {
            if done {
                fin.push((off + n - 1, qi, self.chunked[qi].slot));
            }
            off += n;
        }
        (rows, dec_n, fin, take)
    }

    pub(crate) fn forward_mixed_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        if self.chunked.is_empty() {
            if decodes.is_empty() {
                return Ok((Vec::new(), Vec::new()));
            }
            let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            self.batch_step_slots(&toks, &pos, &slots)?;
            return Ok((self.read_batch_logits(decodes.len())?, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        self.rows_pass_body(&rows, dec_n)?;
        let mut dec_logits = Vec::new();
        if dec_n > 0 {
            self.head_rows(dec_n, None, 0)?;
            let (ks, kp): (Vec<u32>, Vec<u32>) = rows[..dec_n].iter().map(|x| (x.0, x.1)).unzip();
            self.keep_h_rows(&ks, &kp)?;
            dec_logits = self.read_batch_logits(dec_n)?;
        }
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi, slot) in &fin {
            finished_raw.push((qi, self.head_row(row, slot, rows[row].1)?));
        }
        Ok((dec_logits, self.commit_chunk(&take, finished_raw)))
    }

    pub(crate) fn forward_mixed_sampled_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        use crate::generator::{FinishSample, SampledStep};
        let empty = || SampledStep {
            ids: Vec::new(),
            host_rows: Vec::new(),
        };
        if self.chunked.is_empty() {
            if decodes.is_empty() {
                return Ok((empty(), Vec::new()));
            }
            let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            self.batch_step_slots(&toks, &pos, &slots)?;
            return Ok((self.sample_head_rows(decodes.len(), plans)?, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        self.rows_pass_body(&rows, dec_n)?;
        let step = if dec_n > 0 {
            self.head_rows(dec_n, None, 0)?;
            let (ks, kp): (Vec<u32>, Vec<u32>) = rows[..dec_n].iter().map(|x| (x.0, x.1)).unzip();
            self.keep_h_rows(&ks, &kp)?;
            self.sample_head_rows(dec_n, plans)?
        } else {
            empty()
        };
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi, slot) in &fin {
            finished_raw.push((qi, self.head_row(row, slot, rows[row].1)?));
        }
        let finished = self
            .commit_chunk(&take, finished_raw)
            .into_iter()
            .map(|(slot, logits, n)| (slot, FinishSample::Logits(logits), n))
            .collect();
        Ok((step, finished))
    }

    /// Blocking prefill of one request into `slot` (the wave path and the
    /// serial surface): prep + tower inline, then chunked passes to the end.
    pub(crate) fn prefill_now(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        self.chunked.retain(|c| c.slot != slot);
        self.enc.retain(|e| e.slot != slot);
        let (img, text, pre) = split_request(chunks)?;
        let entry = match img {
            Some((rgb, w, h)) => {
                let page = prep_page(rgb, w, h, &self.cfg);
                let plane = self.tower.encode(&[&page])?.pop().expect("one page");
                self.tower.sync()?;
                let n_rows = plane.rows + text.len();
                self.admit(slot, n_rows, &text)?;
                ChunkedPrefill {
                    slot,
                    pre,
                    n_img: plane.rows,
                    text,
                    cursor: 0,
                    plane: Some(plane.plane),
                }
            }
            None => {
                self.admit(slot, text.len(), &text)?;
                ChunkedPrefill {
                    slot,
                    pre: 0,
                    n_img: 0,
                    text,
                    cursor: 0,
                    plane: None,
                }
            }
        };
        let n_rows = entry.rows();
        // run it alone: park the rest of the queue for the duration
        let parked = std::mem::replace(&mut self.chunked, vec![entry]);
        let res = (|| {
            loop {
                let (rows, take) = self.plan_chunk(pf_rows());
                self.rows_pass_body(&rows, 0)?;
                let done = take[0].2;
                self.chunked[0].cursor += take[0].1;
                if done {
                    let last = rows.len() - 1;
                    return self.head_row(last, slot, rows[last].1);
                }
            }
        })();
        self.chunked = parked;
        if slot == 0
            && let Some(bs) = self.batch.as_mut()
        {
            bs.slot0_pos = n_rows;
        }
        Ok((res?, n_rows))
    }
}
