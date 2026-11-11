//! Stall-free chunked prefill + the ENCODER BUDGET for DeepSeek-OCR
//! (door B): image prompts ride the same mixed ticks text does,
//! and the tower encode is spent one call per tick - decode streams never
//! freeze for a page admission.
//!
//! Three pieces, all granite/ASR shapes with this family's deltas:
//!
//! - A persistent [`ChunkedPrefill`] queue (qwen3-asr's), whose entries carry
//!   the mm ROW PLAN (image rows as [`Row::Image`]) plus the assembled
//!   per-image feature planes. Mixed ticks advance a row budget per tick
//!   (`forward_mixed`), and `rows_pass_body` splices image rows from the
//!   planes by (slot, position) - a chunk cut inside a picture is just
//!   another cut.
//! - An encode queue ([`OcrEnc`], granite's WaveEncode shape made per SLOT -
//!   this tower shares nothing across requests, so a page queues the moment
//!   its own encodes finish instead of waiting out wave-mates; this is also
//!   what dissolves the classic path's 1+3 arrival-split artifact). One
//!   `encode_step` spends one tower call (~25-50 ms: a ≤[`MAX_VIEWS`]-view
//!   crops chunk or the global view), staging into the persistent `mm_src`
//!   slab across ticks; the per-image gather rides the same tick as its last
//!   encode. `MmAdmit::Encoding` until the row plan is queued.
//! - The admission wave's preprocess pipeline, made cross-tick: prep jobs
//!   spawn at admission on their own threads (the wave's scoped workers
//!   cannot outlive one call) and land on a per-entry channel; `encode_step`
//!   drains it NON-blocking - a prep still in flight just yields the tick,
//!   and a dead worker (or a need the admission view didn't foresee) falls
//!   back to inline prep. The pipeline accelerates, never gates.
//!
//! ## Ring + radix across ticks
//!
//! Prefill rows keep wpos == apos == pos across every cut because
//! `ring[slot].prefill_len` is only set at COMPLETION (the finisher tick,
//! right after `prefix_insert`) - mid-queue rows see a disarmed ring, decode
//! rows of other slots see their own armed state, and `rows_pass_body`
//! derives both per row. An aborted or dying entry never inserts: the radix
//! only ever sees completed prompts.
//!
//! ## The adoption-ordering invariant (a real clobber, caught live)
//!
//! The slot table may adopt the radix match only once the scheduler can no
//! longer route hole rows at the slot - i.e. in the same `encode_step` that
//! reports `Queued` (or on the admission fast path, whose verdict puts the
//! slot into the scheduler's chunking set before its next decode section).
//! The scheduler's dense decode step feeds occupied-but-unprefilled slots
//! as HOLE rows `(token 0, pos 0)` whose sampled output is ignored - but
//! whose KV APPEND still runs, at row 0 of whatever block the slot's table
//! maps there. Granite survives that because it admits KV only at encode
//! completion, so a hole lands in a scratch block the admission then
//! clears. An early adoption turned the same hole into a write on the
//! radix-shared block 0: every encode tick overwrote the cached page's
//! first K/V row, and the resumed transcript flipped a det coordinate
//! (385->383) against its own cold pass. The wave path passed the same probe
//! on the same binary, which is what indicted the chunked lane.
//! So: admission PROBES (read-only) to decide prep work; adoption
//! and the cursor come from a real admit+resume at completion. A completion
//! resume SHORTER than the probe basis (eviction in between) re-enters
//! assembly for the newly-uncovered images - the pipeline accelerates,
//! never gates.
//!
//! The completion-time resume also buys the sequential half of the wave
//! path's same-wave dedupe for free: a same-prefix prompt that COMPLETED
//! while this one encoded lengthens the match and the cursor skips it. The
//! simultaneous half is deliberately traded for overlap: two identical
//! pages admitted together each run their own tower here, where the serial
//! wave path had the second resume off the first's insert.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};

use cudarc::driver::CudaSlice;

use crate::generator::{GenError, MmAdmit};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::granite::batch::pf_rows;
use crate::service::MmChunk;

use super::load::GpuDeepseekOcr;
use super::multimodal::{
    IMAGE_TOKEN_ID, MmPlan, PrepSpec, PreppedImage, Row, images_of, mm_plan, prep_image, prep_need,
};

fn gen_err(e: GpuModelError) -> GenError {
    GenError::Backend(e.to_string())
}

/// A row stream as token ids, image rows as the image token - what the router
/// lane's page layout is built from.
fn row_ids(rows: &[Row]) -> Vec<u32> {
    rows.iter()
        .map(|row| match row {
            Row::Token(t) => *t,
            Row::Image(..) => IMAGE_TOKEN_ID,
        })
        .collect()
}

/// A prompt on the stall-free queue: rows advance inside mixed ticks. Text
/// entries carry no planes; image rows resolve from `planes[img]` at embed
/// time (`splice_queued_rows`).
pub(crate) struct ChunkedPrefill {
    pub slot: usize,
    /// the ROW stream, one per KV row
    pub rows: Vec<Row>,
    /// the RADIX key vector - content-derived at image rows. Kept so the
    /// insert on completion keys the way the match did; a prompt inserted
    /// under different keys than it matched under would never hit itself.
    pub keys: Vec<u32>,
    /// next row to compute; starts at the resume point, so a radix hit costs
    /// no rows
    pub cursor: usize,
    /// assembled [image_tokens, embd] planes, one per image; None = the
    /// image sits entirely inside the adopted prefix and no row of it is
    /// ever computed
    pub planes: Vec<Option<CudaSlice<f32>>>,
}

/// Where the front entry's current image is in its encode ladder.
#[derive(Clone, Copy)]
enum EncStep {
    /// next crops chunk to encode (index into the prepped chunk list)
    Crops(usize),
    Global,
    Gather,
}

/// One image prompt mid-encode: the owned request, its plan, the prep
/// pipeline's receiving end, and the encode cursor. Everything the finished
/// entry hands the chunked queue (rows/keys/planes) accumulates here.
pub(crate) struct OcrEnc {
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
    rx: Receiver<(usize, PreppedImage)>,
    /// channel disconnected: every prep worker has exited
    rx_dead: bool,
    got: Vec<Option<PreppedImage>>,
    /// per image: (crops job id, global job id) into `got`
    jobs: Vec<(Option<usize>, Option<usize>)>,
    /// encode cursor: current image / step within it
    img: usize,
    step: EncStep,
    /// the current image's prepped crops chunks, fetched once
    cur_crops: Option<Vec<(usize, usize, Vec<u8>)>>,
    planes: Vec<Option<CudaSlice<f32>>>,
}

/// What the prep pipeline has for one job right now.
enum Fetch {
    Ready(PreppedImage),
    /// worker still running - yield the tick, retry next
    Wait,
    /// no job was spawned for this need, or its worker died: prep inline
    Inline,
}

fn fetch(e: &mut OcrEnc, j: Option<usize>) -> Fetch {
    // drain whatever has landed; Disconnected = every worker exited (each
    // sends exactly once, so a missing result past that point never arrives)
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

impl GpuDeepseekOcr {
    // ── the chunked queue (text + queued-image entries) ─────────────────────

    /// Queue a TEXT prompt for chunked prefill: admit, adopt the radix match,
    /// push the row plan. Rare on this family - chat without an image - but
    /// the scheduler routes it here once `supports_chunked_prefill` holds.
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
        self.dash_admit(slot, &tokens);
        let cursor = self.prefill_resume_rows(slot, &tokens, tokens.len())?;
        self.chunked.push(ChunkedPrefill {
            slot,
            rows: tokens.iter().map(|&t| Row::Token(t)).collect(),
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
    /// report, so a silently vanished entry would wedge the slot for whatever
    /// request reuses it (granite's wave_release contract).
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
    /// span; rows of slots with no queue entry (the classic paths) resolve to
    /// nothing.
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
    /// splitting the last prompt if it does not fit. Returns the row stream
    /// and (queue index, rows taken, finishes?) per touched entry.
    fn plan_chunk(&self, budget: usize) -> (Vec<(u32, u32, u32)>, Vec<(usize, usize, bool)>) {
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut take: Vec<(usize, usize, bool)> = Vec::new();
        if self.chunked.is_empty() {
            return (rows, take);
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
                    Row::Image(..) => IMAGE_TOKEN_ID,
                };
                rows.push((c.slot as u32, p as u32, t));
            }
            take.push((qi, n, n == remaining));
        }
        (rows, take)
    }

    /// Advance cursors; finished prompts publish their prefix, arm the ring,
    /// and leave the queue. The ring mark here - after the finisher's pass,
    /// before its first decode tick - is what keeps wpos == apos == pos true
    /// for every prefill row across every cut.
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
            if paddock_models::dev_var_os!("PADDOCK_OCR_CHUNK_DUMP").is_some()
                && let Ok(b0) = self.dump_hash_blk0()
            {
                eprintln!("ocr commit slot={slot} n={n_rows} blk0={b0:016x}");
            }
            let keys = std::mem::take(&mut self.chunked[qi].keys);
            self.prefix_insert(slot, &keys);
            let base = self.ring_base(slot, n_rows);
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.ring[slot].prefill_len = Some(base);
            if slot == 0 {
                bs.slot0_pos = n_rows;
            }
            self.chunked[qi].rows = Vec::new(); // marks it finished
            out.push((slot, logits, n_rows));
        }
        self.chunked.retain(|c| !c.rows.is_empty());
        out
    }

    /// Build the fused tick's row stream: decode rows first (one band), then
    /// as much of the prefill queue as the scratch capacity allows.
    fn fuse_rows(
        &self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> (
        Vec<(u32, u32, u32)>,
        usize,
        Vec<(usize, usize)>,
        Vec<(usize, usize, bool)>,
    ) {
        let mut rows: Vec<(u32, u32, u32)> =
            decodes.iter().map(|&(s, t, p)| (s as u32, p, t)).collect();
        let dec_n = rows.len();
        // decode rows and the chunk share one scratch plane, bounded by
        // BatchState::cap - the band never eats chunk rows
        let room = self
            .batch
            .as_ref()
            .expect("batch enabled")
            .cap
            .saturating_sub(dec_n);
        let (chunk_rows, take) = if room == 0 {
            (Vec::new(), Vec::new())
        } else {
            self.plan_chunk(budget.min(room))
        };
        rows.extend_from_slice(&chunk_rows);
        let mut fin: Vec<(usize, usize)> = Vec::new();
        let mut off = dec_n;
        for &(qi, n, done) in &take {
            if done {
                fin.push((off + n - 1, qi));
            }
            off += n;
        }
        (rows, dec_n, fin, take)
    }

    /// One fused mixed tick: decode rows and the prefill chunk in a single
    /// weight-amortized pass. `rows_pass_body` handles the whole band -
    /// ring-mapped positions for armed decode slots, the queue splice for
    /// image rows - so this is qwen3-asr's tick with the splice for free.
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
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        self.rows_pass_body(&rows, dec_n)?;
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
            self.dash_head_guard(self.chunked[qi].slot)?;
            finished_raw.push((qi, self.head_row(row)?));
        }
        let finished = self.commit_chunk(&take, finished_raw);
        Ok((dec_logits, finished))
    }

    /// `forward_mixed_impl` with device sampling for the decode rows
    /// - same row stream, same finisher path, but the decode band's ids
    ///   come back as dec_n u32s instead of the [dec_n, vocab] logits plane.
    ///   Finishers deliberately stay on the logits path - under the family's
    ///   always-armed ngram guard their peeked plans are Host anyway, and
    ///   `FinishSample::Logits` is legal for a Device plan (the peek is
    ///   uncommitted; the trait documents this fallback as correct).
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
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        self.rows_pass_body(&rows, dec_n)?;
        // decode rows first, same ordering constraint as the unsampled twin:
        // head_row bounces its row through x[0] and rewrites head_logits[0..]
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
            self.dash_head_guard(self.chunked[qi].slot)?;
            finished_raw.push((qi, self.head_row(row)?));
        }
        let finished = self
            .commit_chunk(&take, finished_raw)
            .into_iter()
            .map(|(slot, logits, n)| (slot, FinishSample::Logits(logits), n))
            .collect();
        Ok((step, finished))
    }

    // ── the encoder budget (image admission) ────────────────────────────────

    /// Admit a wave of image prompts: plan + admit + adopt per slot, spawn
    /// the prep jobs, and either queue immediately (every image inside the
    /// adopted prefix - the radix fast path) or park the entry on the encode
    /// queue and answer `Encoding`.
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
        // A queued entry for this slot is STALE (the old request died and the
        // slot was reused). The old entry's pending report is swallowed here,
        // which is safe only because the new admission re-enters the encode
        // queue and reports in its place: the scheduler's encoding set is a
        // set, so the one report clears both inserts. The fast path below is
        // gated on `had_stale` for exactly that reason - a Queued verdict
        // from this call would leave the stale membership behind forever
        // (only encode_step reports remove from the set).
        self.chunked.retain(|c| c.slot != slot);
        let n_enc = self.enc.len();
        self.enc.retain(|e| e.slot != slot);
        let had_stale = self.enc.len() != n_enc;
        let vis = self.vision.as_ref().ok_or_else(|| {
            GpuModelError::Unsupported("deepseek-ocr: image request but no mmproj attached".into())
        })?;
        let max_tiles = self.tile_cap(vis.hp.max_tiles);
        let plan = mm_plan(&chunks, max_tiles)?;
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
            self.dash_admit(slot, &row_ids(&plan.rows));
            let cursor = self.prefill_resume_rows(slot, &plan.keys, plan.n_rows)?;
            self.chunked.push(ChunkedPrefill {
                slot,
                rows: plan.rows,
                keys: plan.keys,
                cursor,
                planes: (0..n_img).map(|_| None).collect(),
            });
            return Ok(MmAdmit::Queued);
        }

        // prep jobs from the probe view, one thread per source (a request is
        // ≤ 2 jobs/image and admission waves are small; the wave path's
        // scoped pool cannot outlive this call). The completion-time resume
        // usually only LENGTHENS the match, so this view rarely under-preps;
        // when eviction shrinks it the assembly loop re-enters and the
        // inline fallback covers the gap.
        let img_ci: Vec<usize> = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| matches!(c, MmChunk::Image { .. }))
            .map(|(ci, _)| ci)
            .collect();
        let chunks = Arc::new(chunks);
        let (tx, rx) = std::sync::mpsc::channel();
        let mut jobs: Vec<(Option<usize>, Option<usize>)> = vec![(None, None); n_img];
        let mut next = 0usize;
        for k in 0..n_img {
            if plan.img_start[k] + plan.img_tokens[k] <= start {
                continue;
            }
            let need_from = start.saturating_sub(plan.img_start[k]);
            let blocks = &plan.layout.blocks[plan.img_blocks[k].clone()];
            let (need_crops, need_global) = prep_need(blocks, plan.layout.grids[k], need_from);
            let spec0 = self.prep_spec(plan.layout.mode, plan.layout.grids[k]);
            for crops_job in [true, false] {
                if !(if crops_job { need_crops } else { need_global }) {
                    continue;
                }
                let spec = if crops_job {
                    PrepSpec {
                        need_crops: true,
                        ..spec0
                    }
                } else {
                    PrepSpec {
                        need_global: true,
                        ..spec0
                    }
                };
                let (tx, chunks, ci) = (tx.clone(), Arc::clone(&chunks), img_ci[k]);
                let id = next;
                next += 1;
                if crops_job {
                    jobs[k].0 = Some(id);
                } else {
                    jobs[k].1 = Some(id);
                }
                // A/B pin: PADDOCK_OCR_SYNC_PREP=1 preps inline on this
                // thread (triage isolation of the worker threads)
                if paddock_models::dev_var_os!("PADDOCK_OCR_SYNC_PREP").is_some() {
                    let MmChunk::Image { rgb, w, h } = &chunks[ci] else {
                        continue;
                    };
                    let _ = tx.send((id, prep_image(rgb, *w, *h, &spec)));
                    continue;
                }
                std::thread::spawn(move || {
                    let MmChunk::Image { rgb, w, h } = &chunks[ci] else {
                        return;
                    };
                    // a dead receiver just means the entry died first
                    let _ = tx.send((id, prep_image(rgb, *w, *h, &spec)));
                });
            }
        }
        drop(tx);

        self.enc.push_back(OcrEnc {
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
            step: EncStep::Crops(0),
            cur_crops: None,
            planes: (0..n_img).map(|_| None).collect(),
        });
        Ok(MmAdmit::Encoding)
    }

    /// Spend one encoder budget on the front entry. Called once per tick
    /// while `encoding_pending`; empty result = still going.
    pub(crate) fn encode_step_impl(&mut self) -> Vec<(usize, MmAdmit)> {
        let mut out = Vec::new();
        // dead entries report and drop first - no budget for the departed,
        // but the scheduler must hear or the slot wedges in its encoding set
        while self.enc.front().is_some_and(|e| e.dead) {
            let e = self.enc.pop_front().expect("front checked");
            out.push((e.slot, MmAdmit::Queued));
        }
        let Some(mut e) = self.enc.pop_front() else {
            return out;
        };
        match self.enc_advance(&mut e) {
            Ok(true) => match self.enc_finish(&mut e) {
                Ok(Some(cursor)) => {
                    let slot = e.slot;
                    self.dash_admit(slot, &row_ids(&e.plan.rows));
                    self.chunked.push(ChunkedPrefill {
                        slot,
                        rows: std::mem::take(&mut e.plan.rows),
                        keys: std::mem::take(&mut e.plan.keys),
                        cursor,
                        planes: std::mem::take(&mut e.planes),
                    });
                    out.push((slot, MmAdmit::Queued));
                }
                // the real resume fell short of the probe basis (eviction
                // while we encoded): assembly re-enters for the images the
                // shorter prefix no longer covers
                Ok(None) => self.enc.push_front(e),
                Err(err) => out.push((e.slot, MmAdmit::Failed(gen_err(err)))),
            },
            Ok(false) => self.enc.push_front(e),
            Err(err) => out.push((e.slot, MmAdmit::Failed(gen_err(err)))),
        }
        out
    }

    /// Every plane is assembled: adopt the radix match and take the cursor -
    /// the only place an Encoding entry may touch the slot's KV state (the
    /// adoption-ordering invariant; the Queued verdict this enables is what
    /// shields the slot from hole rows from here on). `Ok(None)` = the real
    /// resume came up short of the probe basis; the caller re-enters
    /// assembly with the smaller expectation.
    fn enc_finish(&mut self, e: &mut OcrEnc) -> Result<Option<usize>, GpuModelError> {
        self.admit(e.slot, e.plan.n_rows)?;
        let cursor = self.prefill_resume_rows(e.slot, &e.plan.keys, e.plan.n_rows)?;
        if cursor < e.start {
            // UN-adopt before going back to Encoding: the slot is hole-row
            // territory again the moment this entry yields the tick
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.tables[e.slot].clear(&mut bs.pool);
            e.start = cursor;
            e.img = 0;
            e.step = EncStep::Crops(0);
            return Ok(None);
        }
        Ok(Some(cursor))
    }

    /// Advance the front entry: at most one tower call per invocation; skips
    /// and per-image gathers ride free. Ok(true) = every plane assembled
    /// against the PROBE view - the caller's `enc_finish` then does the real
    /// adoption and may send assembly back here with a smaller `start`.
    fn enc_advance(&mut self, e: &mut OcrEnc) -> Result<bool, GpuModelError> {
        let n_img = e.plan.img_start.len();
        let mut spent = false;
        loop {
            if e.img >= n_img {
                return Ok(true);
            }
            let k = e.img;
            if e.planes[k].is_some() {
                // assembled on an earlier lap (the enc_finish re-entry)
                e.img += 1;
                continue;
            }
            if e.plan.img_start[k] + e.plan.img_tokens[k] <= e.start {
                // never read: the whole image sits inside the expected prefix
                e.img += 1;
                continue;
            }
            let need_from = e.start.saturating_sub(e.plan.img_start[k]);
            let blocks = &e.plan.layout.blocks[e.plan.img_blocks[k].clone()];
            let grid = e.plan.layout.grids[k];
            let (need_crops, need_global) = prep_need(blocks, grid, need_from);
            let (base_px, tile_px) = e.plan.layout.mode.px();
            match e.step {
                EncStep::Crops(i) => {
                    if !need_crops {
                        e.step = EncStep::Global;
                        continue;
                    }
                    if e.cur_crops.is_none() {
                        let jc = e.jobs[k].0;
                        match fetch(e, jc) {
                            Fetch::Ready(p) => e.cur_crops = Some(p.crops),
                            Fetch::Wait => return Ok(false),
                            Fetch::Inline => {
                                let spec = self.prep_spec(e.plan.layout.mode, grid);
                                let (rgb, w, h) = images_of(&e.chunks)[k];
                                e.cur_crops = Some(
                                    prep_image(
                                        rgb,
                                        w,
                                        h,
                                        &PrepSpec {
                                            need_crops: true,
                                            ..spec
                                        },
                                    )
                                    .crops,
                                );
                            }
                        }
                    }
                    let crops = e.cur_crops.as_ref().expect("fetched above");
                    if i >= crops.len() {
                        e.cur_crops = None;
                        e.step = EncStep::Global;
                        continue;
                    }
                    if spent {
                        return Ok(false);
                    }
                    let (c0, n) = (crops[i].0, crops[i].1);
                    self.stage_crops_chunk(&crops[i].2, c0, n, tile_px)?;
                    spent = true;
                    e.step = EncStep::Crops(i + 1);
                }
                EncStep::Global => {
                    if !need_global {
                        e.step = EncStep::Gather;
                        continue;
                    }
                    if spent {
                        return Ok(false);
                    }
                    let jg = e.jobs[k].1;
                    let pr = match fetch(e, jg) {
                        Fetch::Ready(p) => p.global,
                        Fetch::Wait => return Ok(false),
                        Fetch::Inline => {
                            let spec = self.prep_spec(e.plan.layout.mode, grid);
                            let (rgb, w, h) = images_of(&e.chunks)[k];
                            prep_image(
                                rgb,
                                w,
                                h,
                                &PrepSpec {
                                    need_global: true,
                                    ..spec
                                },
                            )
                            .global
                        }
                    };
                    let pr = pr.expect("need_global spec yields a plane");
                    self.stage_global(&pr, base_px, tile_px, grid)?;
                    spent = true;
                    e.step = EncStep::Gather;
                }
                EncStep::Gather => {
                    // launch-only: rides the same tick as the image's last
                    // encode instead of costing one of its own
                    let plane = self.finish_image_plane(blocks, grid, e.plan.layout.mode)?;
                    e.planes[k] = Some(plane);
                    e.img += 1;
                    e.step = EncStep::Crops(0);
                }
            }
        }
    }
}
