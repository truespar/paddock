//! Stall-free chunked prefill + the ENCODER BUDGET for PaddleOCR-VL
//! - deepseek-ocr's shape with this family's deltas:
//!
//! - A persistent [`ChunkedPrefill`] queue whose entries carry the mm ROW
//!   PLAN (image rows as [`Row::Image`]), the request's 3-axis position plan
//!   (a chunk cut anywhere slices its M-RoPE stream from it), and the
//!   assembled per-image projector planes. Mixed ticks advance a row budget
//!   per tick (`forward_mixed`); `rows_pass_body` splices image rows from the
//!   planes by (slot, position).
//! - An encode queue ([`PoEnc`]) spending one tower call per tick. This
//!   family's encode ladder is trivially flat - one NaViT pass per image, no
//!   crops/global split - so the budget unit is simply "one image".
//!   Host preprocess (the bit-exact PIL-parity resize) runs on worker
//!   threads spawned at admission; `encode_step` drains the channel
//!   NON-blocking and falls back to inline prep - the pipeline accelerates,
//!   never gates. `MmAdmit::Encoding` until the row plan is queued.
//!
//! ## The adoption-ordering invariant (deepseek-ocr's live-caught clobber)
//!
//! The slot table may adopt the radix match only once the scheduler can no
//! longer route hole rows at the slot - i.e. in the same `encode_step` that
//! reports `Queued` (or on the admission fast path, whose verdict puts the
//! slot into the scheduler's chunking set before its next decode section).
//! While a slot is `Encoding`, dense ticks feed it HOLE rows whose KV append
//! still runs - an early adoption turns those into writes on radix-shared
//! blocks. So: admission PROBES (read-only) to decide prep/encode work; the
//! real admit+resume happens at completion (`enc_finish`), and a completion
//! resume SHORTER than the probe basis (eviction in between) re-enters the
//! encode lap for the newly-uncovered images.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};

use cudarc::driver::CudaSlice;

use crate::generator::{GenError, MmAdmit};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::granite::batch::pf_rows;
use crate::service::MmChunk;

use super::IMAGE_TOKEN;
use super::forward::Positions;
use super::load::GpuPaddleOcrVl;
use super::multimodal::{MmPlan, Prepped, Row, images_of, mm_plan, prep_image};

fn gen_err(e: GpuModelError) -> GenError {
    GenError::Backend(e.to_string())
}

/// A prompt on the stall-free queue: rows advance inside mixed ticks. Text
/// entries carry an arange position plan and no planes; image rows resolve
/// from `planes[img]` at embed time (`splice_queued_rows`).
pub(crate) struct ChunkedPrefill {
    pub slot: usize,
    /// the ROW stream, one per KV row
    pub rows: Vec<Row>,
    /// the RADIX key vector - content-derived at image rows, kept so the
    /// insert on completion keys the way the match did
    pub keys: Vec<u32>,
    /// 3-axis positions for every row + the continuation counter - fixes the
    /// slot's M-RoPE delta at completion
    pub pos: Positions,
    /// next row to compute; starts at the resume point
    pub cursor: usize,
    /// assembled [image_tokens, embd] planes, one per image; None = the
    /// image sits entirely inside the adopted prefix
    pub planes: Vec<Option<CudaSlice<f32>>>,
}

/// One image prompt mid-encode: the owned request, its plan, the prep
/// pipeline's receiving end, and the per-image cursor.
pub(crate) struct PoEnc {
    slot: usize,
    /// Owned request - Arc because prep threads need the pixels too.
    chunks: Arc<Vec<MmChunk>>,
    plan: MmPlan,
    /// EXPECTED resume position, from the admission-time read-only probe.
    /// Drives prep/encode skips only; the real cursor comes from the
    /// completion-time admit+resume (the adoption-ordering invariant).
    start: usize,
    /// request died while queued - reports Queued at its turn so the
    /// scheduler clears its encoding set, but nothing is queued
    dead: bool,
    rx: Receiver<(usize, Prepped)>,
    /// channel disconnected: every prep worker has exited
    rx_dead: bool,
    got: Vec<Option<Prepped>>,
    /// per image: prep job id into `got` (None = no job spawned)
    jobs: Vec<Option<usize>>,
    /// encode cursor: next image to consider
    img: usize,
    planes: Vec<Option<CudaSlice<f32>>>,
}

/// What the prep pipeline has for one job right now.
enum Fetch {
    Ready(Prepped),
    /// worker still running - yield the tick, retry next
    Wait,
    /// no job was spawned for this need, or its worker died: prep inline
    Inline,
}

fn fetch(e: &mut PoEnc, j: Option<usize>) -> Fetch {
    // drain whatever has landed; Disconnected = every worker exited (each
    // sends at most once, so a missing result past that point never arrives)
    loop {
        match e.rx.try_recv() {
            Ok((i, p)) => e.got[i] = Some(p),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                e.rx_dead = true;
                break;
            }
        }
    }
    match j {
        None => Fetch::Inline,
        Some(j) => match e.got[j].take() {
            Some(p) => Fetch::Ready(p),
            None if e.rx_dead => Fetch::Inline,
            None => Fetch::Wait,
        },
    }
}

/// True when every remaining image of `e` is assembled or sits inside the
/// expected prefix - finishing it costs no tower call.
fn enc_assembled(e: &PoEnc) -> bool {
    (e.img..e.plan.img_start.len())
        .all(|k| e.planes[k].is_some() || e.plan.img_start[k] + e.plan.img_tokens[k] <= e.start)
}

/// Group cap for the batched-tower admission lap: bounds the tower pass's
/// transient buffers, and eight is the widest wave --max-batch 8 can form.
const ENC_GROUP_MAX: usize = 8;

/// Arange position plan for a text prompt (all three axes = row index).
fn text_positions(n: usize) -> Positions {
    let a: Vec<u32> = (0..n as u32).collect();
    Positions {
        t: a.clone(),
        h: a.clone(),
        w: a,
        next: n as u32,
    }
}

/// What `fuse_rows` hands the tick, in this order: the row stream as
/// `(slot, token, pos)`, the per-row M-RoPE positions, the decode-row count
/// at the head of the stream, the finisher rows as `(row, slot)`, and the
/// prefill entries taken as `(queue index, rows consumed, finished)`.
type FusedRows = (
    Vec<(u32, u32, u32)>,
    Vec<u32>,
    usize,
    Vec<(usize, usize)>,
    Vec<(usize, usize, bool)>,
);

impl GpuPaddleOcrVl {
    // ── the chunked queue (text + queued-image entries) ─────────────────────

    /// Queue a TEXT prompt for chunked prefill: admit, adopt the radix match,
    /// push the row plan.
    pub(crate) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuModelError> {
        // a queued entry for this slot is STALE (the old request died and the
        // slot was reused): evict rather than wedge the slot
        self.chunked.retain(|c| c.slot != slot);
        self.enc.retain(|e| e.slot != slot);
        self.admit(slot, tokens.len())?;
        let cursor = self.prefill_resume_rows(slot, &tokens, tokens.len())?;
        self.chunked.push(ChunkedPrefill {
            slot,
            rows: tokens.iter().map(|&t| Row::Token(t)).collect(),
            pos: text_positions(tokens.len()),
            keys: tokens,
            cursor,
            planes: Vec::new(),
        });
        Ok(())
    }

    /// Drop slot's in-flight chunked prefill (client hung up mid-prompt).
    pub(crate) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        let n = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.len() != n
    }

    /// Drop dead slots' queue entries. Encode entries are MARKED rather than
    /// removed: the scheduler's encoding set only clears on an `encode_step`
    /// report, so a silently vanished entry would wedge the slot.
    pub(crate) fn chunk_release(&mut self, occupied: &[bool]) {
        let occ = |s: usize| occupied.get(s).copied().unwrap_or(false);
        self.chunked.retain(|c| occ(c.slot));
        for e in &mut self.enc {
            if !occ(e.slot) {
                e.dead = true;
            }
        }
    }

    /// Overwrite queued image rows' placeholder embeddings from their entry's
    /// feature planes, by (slot, position) - called by `rows_pass_body` right
    /// after the embed gather. Decode-band rows (i < dec) sit past every
    /// span; rows of slots with no queue entry (the wave path splices itself)
    /// resolve to nothing.
    pub(super) fn splice_queued_rows(
        &mut self,
        chunk: &[(u32, u32, u32)],
        dec: usize,
    ) -> Result<(), GpuModelError> {
        if self.chunked.is_empty() || chunk.len() == dec {
            return Ok(());
        }
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        // disjoint field borrows: planes read from the queue, rows written
        // into the batch scratch
        let (chunked, batch) = (&self.chunked, &mut self.batch);
        let bs = batch.as_mut().expect("batch enabled");
        let mut i = dec;
        while i < chunk.len() {
            // one contiguous same-slot run
            let s = chunk[i].0;
            let mut len = 1usize;
            while i + len < chunk.len() && chunk[i + len].0 == s {
                len += 1;
            }
            if let Some(e) = chunked.iter().find(|c| c.slot == s as usize) {
                let mut j = 0usize;
                while j < len {
                    let p = chunk[i + j].1 as usize;
                    if let Row::Image(img, prow) = e.rows[p] {
                        // contiguous plane rows coalesce into one dtod
                        let mut n = 1usize;
                        while j + n < len
                            && matches!(
                                e.rows[chunk[i + j + n].1 as usize],
                                Row::Image(i2, r2) if i2 == img && r2 == prow + n
                            )
                        {
                            n += 1;
                        }
                        let plane = e.planes[img]
                            .as_ref()
                            .expect("plane exists for any image row the queue computes");
                        exec.copy_region(
                            plane,
                            prow * embd,
                            &mut bs.sc.x,
                            (i + j) * embd,
                            n * embd,
                        )?;
                        j += n;
                    } else {
                        j += 1;
                    }
                }
            }
            i += len;
        }
        Ok(())
    }

    /// Pick this tick's chunk rows: FIFO over the queue, up to `budget` rows,
    /// splitting the last prompt if it does not fit. Returns the row stream,
    /// its per-row M-RoPE triples in row order (t/h/w columns, transposed by
    /// the caller), and (queue index, rows taken, finishes?) per entry.
    fn plan_chunk(
        &self,
        budget: usize,
    ) -> (
        Vec<(u32, u32, u32)>,
        Vec<[u32; 3]>,
        Vec<(usize, usize, bool)>,
    ) {
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut mr: Vec<[u32; 3]> = Vec::new();
        let mut take: Vec<(usize, usize, bool)> = Vec::new();
        if self.chunked.is_empty() {
            return (rows, mr, take);
        }
        let cap = budget.clamp(1, pf_rows());
        for (qi, c) in self.chunked.iter().enumerate() {
            if rows.len() >= cap {
                break;
            }
            let remaining = c.rows.len() - c.cursor;
            let n = remaining.min(cap - rows.len()).max(1);
            for j in 0..n {
                let p = c.cursor + j;
                let t = match c.rows[p] {
                    Row::Token(t) => t,
                    Row::Image(..) => IMAGE_TOKEN,
                };
                rows.push((c.slot as u32, p as u32, t));
                mr.push([c.pos.t[p], c.pos.h[p], c.pos.w[p]]);
            }
            take.push((qi, n, n == remaining));
        }
        (rows, mr, take)
    }

    /// Advance cursors; finished prompts publish their prefix, fix the slot's
    /// M-RoPE delta, and leave the queue.
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
            let (slot, n_rows) = (self.chunked[qi].slot, self.chunked[qi].rows.len());
            let keys = std::mem::take(&mut self.chunked[qi].keys);
            self.prefix_insert(slot, &keys);
            let next = self.chunked[qi].pos.next;
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.mrope_delta[slot] = next as i64 - n_rows as i64;
            if slot == 0 {
                bs.slot0_pos = n_rows;
            }
            self.chunked[qi].rows = Vec::new(); // marks it finished
            out.push((slot, logits, n_rows));
        }
        self.chunked.retain(|c| !c.rows.is_empty());
        out
    }

    /// Build the fused tick's row stream: decode rows first (one band, M-RoPE
    /// at pos+delta), then as much of the prefill queue as the scratch
    /// capacity allows (M-RoPE from each entry's plan).
    fn fuse_rows(&self, decodes: &[(usize, u32, u32)], budget: usize) -> FusedRows {
        let mut rows: Vec<(u32, u32, u32)> =
            decodes.iter().map(|&(s, t, p)| (s as u32, p, t)).collect();
        let dec_n = rows.len();
        let bs = self.batch.as_ref().expect("batch enabled");
        let mut mr: Vec<[u32; 3]> = decodes
            .iter()
            .map(|&(s, _, p)| {
                let m = (p as i64 + bs.mrope_delta[s]) as u32;
                [m, m, m]
            })
            .collect();
        // decode rows and the chunk share one scratch plane, bounded by
        // BatchState::cap - the band never eats chunk rows
        let room = bs.cap.saturating_sub(dec_n);
        let (chunk_rows, chunk_mr, take) = if room == 0 {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            self.plan_chunk(budget.min(room))
        };
        rows.extend_from_slice(&chunk_rows);
        mr.extend_from_slice(&chunk_mr);
        // transpose the per-row triples into the axis-major [4, r] layout
        let r = rows.len();
        let mut mrope = vec![0u32; 4 * r];
        for (i, m) in mr.iter().enumerate() {
            mrope[i] = m[0];
            mrope[r + i] = m[1];
            mrope[2 * r + i] = m[2];
            mrope[3 * r + i] = m[0];
        }
        let mut fin: Vec<(usize, usize)> = Vec::new();
        let mut off = dec_n;
        for &(qi, n, done) in &take {
            if done {
                fin.push((off + n - 1, qi));
            }
            off += n;
        }
        (rows, mrope, dec_n, fin, take)
    }

    /// One fused mixed tick: decode rows and the prefill chunk in a single
    /// weight-amortized pass.
    pub(crate) fn forward_mixed_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        if self.chunked.is_empty() {
            if decodes.is_empty() {
                return Ok((Vec::new(), Vec::new()));
            }
            // plain decode tick on its captured graph
            let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            self.batch_step_slots(&toks, &pos, &slots)?;
            return Ok((self.read_batch_logits(decodes.len())?, Vec::new()));
        }
        let (rows, mrope, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        self.rows_pass_body(&rows, &mrope, dec_n)?;
        // decode rows first: bulk norm+head over rows 0..dec_n. Must precede
        // the finisher heads - head_row bounces its row through x[0] and
        // rewrites head_logits[0..vocab].
        let mut dec_logits = Vec::new();
        if dec_n > 0 {
            self.head_rows(dec_n)?;
            dec_logits = self.read_batch_logits(dec_n)?;
        }
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi) in &fin {
            finished_raw.push((qi, self.head_row(row)?));
        }
        let finished = self.commit_chunk(&take, finished_raw);
        Ok((dec_logits, finished))
    }

    /// `forward_mixed_impl` with device sampling for the decode rows: same
    /// row stream, same finisher path, but the decode band's ids come back as
    /// dec_n u32s instead of the [dec_n, vocab] logits plane. Finishers stay
    /// on the logits path (`FinishSample::Logits` is legal for a Device plan;
    /// the peek is uncommitted).
    pub(crate) fn forward_mixed_sampled_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        _fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        use crate::generator::{FinishSample, SampledStep};
        if self.chunked.is_empty() {
            if decodes.is_empty() {
                return Ok((
                    SampledStep {
                        ids: Vec::new(),
                        host_rows: Vec::new(),
                    },
                    Vec::new(),
                ));
            }
            let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            self.batch_step_slots(&toks, &pos, &slots)?;
            return Ok((self.sample_head_rows(decodes.len(), plans)?, Vec::new()));
        }
        let t_tick = std::time::Instant::now();
        let (rows, mrope, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        let n_rows = rows.len();
        self.rows_pass_body(&rows, &mrope, dec_n)?;
        let step = if dec_n > 0 {
            self.head_rows(dec_n)?;
            self.sample_head_rows(dec_n, plans)?
        } else {
            SampledStep {
                ids: Vec::new(),
                host_rows: Vec::new(),
            }
        };
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi) in &fin {
            finished_raw.push((qi, self.head_row(row)?));
        }
        if n_rows > dec_n && paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
            eprintln!(
                "req-trace: mixed tick {:.1} ms ({} rows, {} decode, {} finish), done at {}",
                t_tick.elapsed().as_secs_f64() * 1e3,
                n_rows,
                dec_n,
                fin.len(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_micros())
            );
        }
        let finished = self
            .commit_chunk(&take, finished_raw)
            .into_iter()
            .map(|(slot, logits, n)| (slot, FinishSample::Logits(logits), n))
            .collect();
        Ok((step, finished))
    }

    // ── the encoder budget (image admission) ────────────────────────────────

    /// Admit a wave of image prompts: plan + probe per slot, spawn the prep
    /// jobs, and either queue immediately (every image inside the probed
    /// prefix - the radix fast path) or park the entry on the encode queue
    /// and answer `Encoding`.
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
        // A queued entry for this slot is STALE. The old entry's pending
        // report is swallowed here, which is safe only because the new
        // admission re-enters the encode queue and reports in its place -
        // the fast path below is gated on `had_stale` for exactly that
        // reason (only encode_step reports remove from the scheduler's
        // encoding set).
        self.chunked.retain(|c| c.slot != slot);
        let n_enc = self.enc.len();
        self.enc.retain(|e| e.slot != slot);
        let had_stale = self.enc.len() != n_enc;
        if self.vision.is_none() {
            return Err(GpuModelError::Unsupported(
                "paddleocr-vl: image request but no mmproj attached".into(),
            ));
        }
        let t_plan = std::time::Instant::now();
        let plan = mm_plan(&chunks, self.default_budget())?;
        if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
            eprintln!(
                "req-trace: mm_plan slot {slot} {:.1} ms ({} rows), done at {}",
                t_plan.elapsed().as_secs_f64() * 1e3,
                plan.n_rows,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_micros())
            );
        }
        // admission validates BOUNDS only - the slot's KV state stays
        // untouched until the entry queues (the adoption-ordering invariant:
        // hole rows may tick at this slot for as long as it is Encoding)
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        if slot >= n_slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} >= enabled {n_slots}"
            )));
        }
        if plan.n_rows == 0 || plan.n_rows > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "prompt {} rows out of range (max_ctx {})",
                plan.n_rows, self.max_ctx
            )));
        }
        // read-only probe: decides what is worth preprocessing/encoding
        let start = self.prefix_probe(&plan.keys);
        let n_img = plan.img_start.len();

        if !had_stale && (0..n_img).all(|k| plan.img_start[k] + plan.img_tokens[k] <= start) {
            // every image sits inside the cached prefix: no tower, no prep -
            // adopt and queue now. Adopting here is safe (unlike in the
            // Encoding state) because the Queued verdict puts this slot into
            // the scheduler's chunking set before its next decode section.
            self.admit(slot, plan.n_rows)?;
            let cursor = self.prefill_resume_rows(slot, &plan.keys, plan.n_rows)?;
            self.chunked.push(ChunkedPrefill {
                slot,
                rows: plan.rows,
                keys: plan.keys,
                pos: plan.pos,
                cursor,
                planes: (0..n_img).map(|_| None).collect(),
            });
            return Ok(MmAdmit::Queued);
        }

        // prep jobs from the probe view, one thread per needed image. The
        // completion-time resume usually only LENGTHENS the match; when
        // eviction shrinks it the encode lap re-enters and the inline
        // fallback covers the gap.
        let img_ci: Vec<usize> = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| matches!(c, MmChunk::Image { .. }))
            .map(|(ci, _)| ci)
            .collect();
        let budget = plan.budget;
        let chunks = Arc::new(chunks);
        let (tx, rx) = std::sync::mpsc::channel();
        let mut jobs: Vec<Option<usize>> = vec![None; n_img];
        let mut next = 0usize;
        for k in 0..n_img {
            if plan.img_start[k] + plan.img_tokens[k] <= start {
                continue;
            }
            let (tx, chunks, ci) = (tx.clone(), Arc::clone(&chunks), img_ci[k]);
            let id = next;
            next += 1;
            jobs[k] = Some(id);
            // A/B pin: PADDOCK_OCR_SYNC_PREP=1 preps inline on this thread
            // (triage isolation of the worker threads)
            if paddock_models::dev_var_os!("PADDOCK_OCR_SYNC_PREP").is_some() {
                let MmChunk::Image { rgb, w, h } = &chunks[ci] else {
                    continue;
                };
                if let Ok(p) = prep_image(rgb, *w, *h, budget) {
                    let _ = tx.send((id, p));
                }
                continue;
            }
            let trace_slot = slot;
            std::thread::spawn(move || {
                let MmChunk::Image { rgb, w, h } = &chunks[ci] else {
                    return;
                };
                let trace = paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some();
                let t0 = std::time::Instant::now();
                // a failed prep falls to the inline path (which reports the
                // real error); a dead receiver means the entry died first
                if let Ok(p) = prep_image(rgb, *w, *h, budget) {
                    if trace {
                        eprintln!(
                            "req-trace: prep slot {trace_slot} job {id} {:.1} ms \
                             ({w}x{h} -> {}x{}), done at {}",
                            t0.elapsed().as_secs_f64() * 1e3,
                            p.w,
                            p.h,
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_or(0, |d| d.as_micros())
                        );
                    }
                    let _ = tx.send((id, p));
                }
            });
        }
        drop(tx);

        self.enc.push_back(PoEnc {
            slot,
            chunks,
            plan,
            start,
            dead: false,
            rx,
            rx_dead: false,
            got: (0..next).map(|_| None).collect(),
            jobs,
            img: 0,
            planes: (0..n_img).map(|_| None).collect(),
        });
        Ok(MmAdmit::Encoding)
    }

    /// Spend one encoder budget on the queue. Called once per tick while
    /// `encoding_pending`; empty result = still going.
    ///
    /// The budget unit is still one tower pass per tick, but the pass is the
    /// batched-group lap: the front's next image rides together
    /// with every other queued entry's prep-ready image of the same dims.
    /// Entries whose planes are all assembled then drain in the same tick -
    /// finishing them is resume/queue bookkeeping, not tower work. The old
    /// strictly-serial form turned a synchronized c8 arrival wave into eight
    /// tower ticks and a ~700ms TTFT plateau.
    pub(crate) fn encode_step_impl(&mut self) -> Vec<(usize, MmAdmit)> {
        let mut out = Vec::new();
        let mut towered = false;
        loop {
            // dead entries report and drop first - no budget for the
            // departed, but the scheduler must hear or the slot wedges in
            // its encoding set
            while self.enc.front().is_some_and(|e| e.dead) {
                let e = self.enc.pop_front().expect("front checked");
                out.push((e.slot, MmAdmit::Queued));
            }
            let Some(mut e) = self.enc.pop_front() else {
                return out;
            };
            if towered && !enc_assembled(&e) {
                // the tick's tower budget is spent - only zero-cost entries
                // may still finish this tick
                self.enc.push_front(e);
                return out;
            }
            if !towered && !enc_assembled(&e) {
                self.enc_group_lap(&mut e);
                towered = true;
            }
            match self.enc_advance(&mut e) {
                Ok(true) => match self.enc_finish(&mut e) {
                    Ok(Some(cursor)) => {
                        let slot = e.slot;
                        self.chunked.push(ChunkedPrefill {
                            slot,
                            rows: std::mem::take(&mut e.plan.rows),
                            keys: std::mem::take(&mut e.plan.keys),
                            pos: std::mem::replace(
                                &mut e.plan.pos,
                                Positions {
                                    t: Vec::new(),
                                    h: Vec::new(),
                                    w: Vec::new(),
                                    next: 0,
                                },
                            ),
                            cursor,
                            planes: std::mem::take(&mut e.planes),
                        });
                        out.push((slot, MmAdmit::Queued));
                    }
                    // the real resume fell short of the probe basis (eviction
                    // while we encoded): the encode lap re-enters for the
                    // images the shorter prefix no longer covers
                    Ok(None) => {
                        self.enc.push_front(e);
                        return out;
                    }
                    Err(err) => out.push((e.slot, MmAdmit::Failed(gen_err(err)))),
                },
                Ok(false) => {
                    self.enc.push_front(e);
                    return out;
                }
                Err(err) => out.push((e.slot, MmAdmit::Failed(gen_err(err)))),
            }
        }
    }

    /// The batched-tower lap: gather the front's next needed image plus every
    /// other queued entry's prep-ready image of the same dims into one
    /// `encode_batch` pass. Per-row tower math does not depend on batch
    /// placement (per-image attention windows - the tower's own contract), so
    /// the group's planes are bit-identical to serial single-image calls.
    /// A prep is only TAKEN once its dims are confirmed in-group; a taken
    /// prep whose encode fails falls back through the entry's own
    /// Wait->Disconnected->Inline path, so nothing wedges.
    fn enc_group_lap(&mut self, front: &mut PoEnc) {
        let n_front = front.plan.img_start.len();
        let mut fk = front.img;
        while fk < n_front
            && (front.planes[fk].is_some()
                || front.plan.img_start[fk] + front.plan.img_tokens[fk] <= front.start)
        {
            fk += 1;
        }
        if fk >= n_front {
            return; // fully assembled - nothing to batch
        }
        // the front's prep must be ready; Wait/Inline laps stay with
        // enc_advance exactly as before
        let Fetch::Ready(fp) = fetch(front, front.jobs[fk]) else {
            return;
        };
        let (w, h) = (fp.w, fp.h);
        // rows, not just pictures, bound a pass: the workspace is sized once
        // for TOWER_PASS_ROWS, and eight maximum-size pictures would be four
        // times that
        let rows = self
            .vision
            .as_ref()
            .expect("vision attached")
            .patch_rows(w, h);

        let mut members: Vec<(usize, usize, Prepped)> = Vec::new(); // (queue idx, k, prep)
        for (qi, e) in self.enc.iter_mut().enumerate() {
            if members.len() + 1 >= ENC_GROUP_MAX
                || (members.len() + 2) * rows > super::vision::TOWER_PASS_ROWS
            {
                break;
            }
            if e.dead {
                continue;
            }
            let n = e.plan.img_start.len();
            let mut k = e.img;
            while k < n
                && (e.planes[k].is_some() || e.plan.img_start[k] + e.plan.img_tokens[k] <= e.start)
            {
                k += 1;
            }
            if k >= n {
                continue;
            }
            let Some(j) = e.jobs[k] else { continue };
            // drain the channel, then PEEK - take only on a dims match
            loop {
                match e.rx.try_recv() {
                    Ok((i, p)) => e.got[i] = Some(p),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        e.rx_dead = true;
                        break;
                    }
                }
            }
            if e.got[j].as_ref().is_some_and(|p| p.w == w && p.h == h) {
                members.push((qi, k, e.got[j].take().expect("peeked ready")));
            }
        }

        // one tower pass for the whole group. `vision` is borrowed directly
        // (not through an &mut self helper) so the queue borrows above stay
        // legal - disjoint fields.
        let vis = self.vision.as_mut().expect("vision attached");
        let mut group: Vec<(&[f32], usize, usize)> = Vec::with_capacity(members.len() + 1);
        group.push((&fp.planar, w, h));
        for (_, _, p) in &members {
            group.push((&p.planar, p.w, p.h));
        }
        let t_enc = std::time::Instant::now();
        let outs = match vis.encode_batch(&group) {
            Ok(o) => o,
            Err(_) => return, // fail-soft: entries re-enter via Inline
        };
        if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
            eprintln!(
                "req-trace: group encode x{} {:.1} ms, done at {}",
                group.len(),
                t_enc.elapsed().as_secs_f64() * 1e3,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_micros())
            );
        }
        let mut it = outs.into_iter();
        let fo = it.next().expect("front output");
        // a grid mismatch is a plan/tower fault - leave the plane unset and
        // the entry's own inline lap surfaces the real error
        if (fo.ny, fo.nx) == (front.plan.grids[fk].ny, front.plan.grids[fk].nx) {
            front.planes[fk] = Some(fo.embd);
        }
        for ((qi, k, _), o) in members.into_iter().zip(it) {
            let e = &mut self.enc[qi];
            if (o.ny, o.nx) == (e.plan.grids[k].ny, e.plan.grids[k].nx) {
                e.planes[k] = Some(o.embd);
            }
        }
    }

    /// Every plane is assembled: adopt the radix match and take the cursor -
    /// the only place an Encoding entry may touch the slot's KV state.
    /// `Ok(None)` = the real resume came up short of the probe basis; the
    /// caller re-enters the encode lap with the smaller expectation.
    fn enc_finish(&mut self, e: &mut PoEnc) -> Result<Option<usize>, GpuModelError> {
        self.admit(e.slot, e.plan.n_rows)?;
        let cursor = self.prefill_resume_rows(e.slot, &e.plan.keys, e.plan.n_rows)?;
        if cursor < e.start {
            // UN-adopt before going back to Encoding: the slot is hole-row
            // territory again the moment this entry yields the tick
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.tables[e.slot].clear(&mut bs.pool);
            e.start = cursor;
            e.img = 0;
            return Ok(None);
        }
        Ok(Some(cursor))
    }

    /// Advance the front entry: at most one tower call per invocation (the
    /// encoder-budget unit - one image, this family's whole ladder); skips
    /// ride free. Ok(true) = every plane assembled against the PROBE view.
    fn enc_advance(&mut self, e: &mut PoEnc) -> Result<bool, GpuModelError> {
        let n_img = e.plan.img_start.len();
        loop {
            if e.img >= n_img {
                return Ok(true);
            }
            let k = e.img;
            if e.planes[k].is_some() || e.plan.img_start[k] + e.plan.img_tokens[k] <= e.start {
                // assembled on an earlier lap, or never read (inside the
                // expected prefix)
                e.img += 1;
                continue;
            }
            let prepped = match fetch(e, e.jobs[k]) {
                Fetch::Ready(p) => p,
                Fetch::Wait => return Ok(false),
                Fetch::Inline => {
                    let (rgb, w, h) = images_of(&e.chunks)[k];
                    prep_image(rgb, w, h, e.plan.budget)?
                }
            };
            e.planes[k] = Some(self.encode_plan_image(&e.plan, k, &prepped)?);
            e.img += 1;
            // one tower call spent - remaining images take later ticks
            return Ok(e.img >= n_img);
        }
    }
}
