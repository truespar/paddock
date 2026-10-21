//! Picture prompts RIDING the unified tick: a picture prompt's prefill span
//! and the live decode rows in one forward.
//!
//! The planner (`mm_steps`) used to run every picture prefill pass itself, one
//! exclusive unit per `Generator::encode_step`, while every decoding slot
//! waited for it. On the document-recovery workload (48 slots, one ~1,800-row
//! picture prompt admitted about every second) those passes held 40-60% of
//! the GPU and decode got the rest: the unified tick fuses TEXT prefill spans
//! with the decode rows (one weight read for both - the decode riders), and
//! the picture path never joined it. Nothing about a picture row needs a pass
//! of its own: its attention bound is its own index (raster-causal, see
//! `build_mm_layout`), the recurrence is causal, and the unified walk already
//! applies per-row mRoPE. What a picture share needs beyond a text share is
//!
//!   1. its mRoPE rows from the layout instead of the position identity,
//!   2. the picture embeddings copied over its placeholder rows after the
//!      token embed,
//!   3. its checkpoints published through the picture-keyed radix entries
//!      (`mm_prefix_publish`) and the decode mRoPE delta set on finish.
//!
//! That is all this module adds. The share build, cut landing, KV append,
//! DeltaNet resume, finisher logits and sampling are the unified tick's own,
//! unchanged, so a picture prompt now chunks like a text one
//! (`PADDOCK_UNIFIED_MIXED_ROWS` sizes the span beside the riders).
//!
//! A span may end INSIDE a picture: its rows are causal, so a chunk boundary
//! is as arbitrary there as in text, and the splice copies the sub-range.
//! Checkpoints still land outside pictures (`mm_ckpt_cuts`), because a resume
//! restores a prefix the radix keys by whole pictures. Group jobs (a cold
//! burst's small prompts, one batched pass) keep the planner's own pass.
//!
//! The overlap pump stays on for vision serves too (`overlap_ready`). The pipe
//! corruption first seen beside picture spans - exact-output text probes lost
//! 55 of 144 replies at 48 slots on an RTX PRO 6000 (sm_120, Qwen3.8-27B
//! NVFP4, 2026-10-18) - was the pack sampler's process-wide scratch racing
//! between the two lanes, and reproduced with text alone; the per-stream
//! scratch fixed it. The fused tick with picture shares and 47 riders is
//! exact (the 107-page gate 1064/1068 at 48 slots, every difference a known
//! near-tie). Bisect knobs: PADDOCK_MM_RIDE_NOSPLICE / NOMROPE / NOCUTS.
//! Kill: PADDOCK_NO_MM_RIDE=1 (the planner runs its passes exclusively
//! again).

use cudarc::driver::CudaSlice;

use super::pictures::Picture;
use super::*;
use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;

/// The multimodal extras one batched-prefill share carries through
/// `prefill_batch_pass` (parallel to `items`): the request-relative
/// [4, take] axis-major mrope rows, the per-row attention bound (causal,
/// image rows included), the (row offset, rows) embedding splices with their
/// encoded outputs, and the request's final llama-position for the decode
/// mrope delta.
pub(super) struct MmShareCtx {
    pub(super) mrope: Vec<u32>,
    pub(super) bound: Vec<u32>,
    pub(super) splices: Vec<(usize, usize)>,
    /// the request's pictures while its group's pass runs, empty otherwise
    pub(super) images: Vec<Picture>,
    pub(super) final_mrope_pos: usize,
}

/// What a picture prompt queued for the unified tick carries beside its
/// placeholder ids (`ChunkedPrefill::tokens`).
pub(super) struct MmRide {
    /// axis-major `[4, t_len]` mRoPE positions of the whole prompt
    pub(super) mrope: Vec<u32>,
    pub(super) t_len: usize,
    /// `(row offset, rows, embeddings)` per picture at or past the resume
    /// point - the ones a span can still splice; earlier pictures are in KV
    pub(super) splices: Vec<(usize, usize, Picture)>,
    /// the prompt's radix keys (a picture row is keyed by its picture)
    pub(super) keys: Vec<u32>,
    /// checkpoint boundaries, page-aligned and outside every picture; the
    /// share build lands a span on each, and `mm_ride_landed` publishes there
    pub(super) cuts: Vec<usize>,
    /// the llama-position the first decoded token continues from
    pub(super) final_mrope_pos: usize,
}

/// Row type of a unified share: (chunk index, slot, done, take, finishing,
/// span tokens) - the tuple `unified_launch_core` builds.
type Share = (usize, usize, usize, usize, bool, Vec<u32>);

/// Copy each picture's embeddings over its placeholder rows of the mm shares
/// in the tick's residual plane - the sub-range a span covers when it ends
/// inside a picture. Rows `b..` are the shares in order, `take` rows each.
/// A free function so it can borrow `chunked` beside the scratch the caller
/// already holds mutably.
pub(super) fn splice(
    exec: &crate::gpu::GpuExecutor,
    chunked: &[ChunkedPrefill],
    shares: &[Share],
    b: usize,
    d_x: &mut CudaSlice<f32>,
    embd: usize,
) -> Result<(), GpuError> {
    // PADDOCK_MM_RIDE_NOSPLICE=1: leave the placeholder embeddings in place (a
    // bisect arm - the pictures come out blank, so only the OTHER rows'
    // correctness is the measurement)
    if paddock_models::dev_var_os!("PADDOCK_MM_RIDE_NOSPLICE").is_some() {
        return Ok(());
    }
    let mut rb = b;
    for &(idx, _, done, take, _, _) in shares {
        if let Some(m) = &chunked[idx].mm {
            let (a, e) = (done, done + take);
            for (off, n, pic) in &m.splices {
                let (lo, hi) = (a.max(*off), e.min(off + n));
                if lo < hi {
                    exec.copy_region(
                        &pic.embd,
                        (lo - off) * embd,
                        d_x,
                        (rb + lo - a) * embd,
                        (hi - lo) * embd,
                    )?;
                }
            }
        }
        rb += take;
    }
    Ok(())
}

impl GpuQwen35 {
    /// Whether planned picture prompts ride the unified tick on this serve:
    /// the fused tick is elected (PADDOCK_UNIFIED, the qwen35 default stack),
    /// the serve is paged with a prefix cache (what `mm_prefix_publish` and
    /// the planner's resume assume), and the kill is unset.
    pub(super) fn mm_ride_on(&self) -> bool {
        // supports_chunked_prefill: riding means queueing the prompt as a
        // unified chunk, and the chunk queue only exists on a chunked-prefill
        // serve (only prefill_begin_hinted fills it otherwise). Picture
        // prompts ride only where text does.
        std::env::var_os("PADDOCK_UNIFIED").is_some()
            && self.supports_chunked_prefill()
            && paddock_models::dev_var_os!("PADDOCK_NO_MM_RIDE").is_none()
            && self
                .batch
                .as_ref()
                .is_some_and(|b| b.paged && b.paged_prefix.is_some())
    }

    /// Hand a planned slot job to the unified queue: the whole remaining
    /// prompt as one chunk, resumed at `job.pos` (the planner matched and
    /// restored the prefix and backed the slot's KV table), with every
    /// picture from there on encoded, landing a span on each checkpoint cut
    /// the planner computed. The job's data moves into the chunk; the job
    /// is spent afterwards.
    pub(super) fn mm_ride_enqueue(
        &mut self,
        job: &mut super::mm_steps::MmSlotJob,
    ) -> Result<(), GpuModelError> {
        let slot = job.slot();
        if self.chunked.iter().any(|c| c.slot == slot) {
            return Err(GpuModelError::Unsupported(
                "slot already has a chunked prefill in flight".into(),
            ));
        }
        let (lay, keys, pictures, pos, passes) = job.take_ride_parts();
        let t_len = lay.t_len;
        // PADDOCK_MM_RIDE_NOCUTS=1: no checkpoint cuts at all (bisect arm:
        // every span then runs whole and nothing resumes from this prompt)
        let cuts: Vec<usize> = if paddock_models::dev_var_os!("PADDOCK_MM_RIDE_NOCUTS").is_some() {
            Vec::new()
        } else {
            passes
                .iter()
                .filter(|&&(end, ck)| ck && end < t_len)
                .map(|&(end, _)| end)
                .collect()
        };
        let mut splices: Vec<(usize, usize, Picture)> = Vec::new();
        for (&(off, n), pic) in lay.splices.iter().zip(pictures) {
            if off + n <= pos {
                continue; // already in KV through the resumed prefix
            }
            debug_assert!(off >= pos, "a resume point inside a picture");
            splices.push((
                off,
                n,
                pic.expect("every picture past the resume point is encoded"),
            ));
        }
        if paddock_models::dev_var_os!("PADDOCK_UNIFIED_SHAPE_LOG").is_some() {
            let sp: Vec<(usize, usize)> = splices.iter().map(|(o, n, _)| (*o, *n)).collect();
            eprintln!(
                "[mmride] enqueue slot={slot} pos={pos} t_len={t_len} cuts={cuts:?} splices={sp:?}"
            );
        }
        self.enqueue_chunk(ChunkedPrefill {
            slot,
            tokens: lay.ids,
            done: pos,
            hints: Vec::new(),
            mm: Some(Box::new(MmRide {
                mrope: lay.mrope,
                t_len,
                splices,
                keys,
                cuts,
                final_mrope_pos: lay.final_mrope_pos,
            })),
        });
        Ok(())
    }

    /// Write the mm shares' mRoPE rows over the text identity in the tick's
    /// axis-major `[4, r]` host plane (rows `b..` are the shares in order).
    pub(super) fn mm_ride_mrope(&self, shares: &[Share], b: usize, r: usize, mrope_h: &mut [u32]) {
        // PADDOCK_MM_RIDE_NOMROPE=1: keep the text identity for picture rows
        // (bisect arm, same caveat as NOSPLICE)
        if paddock_models::dev_var_os!("PADDOCK_MM_RIDE_NOMROPE").is_some() {
            return;
        }
        let mut rb = b;
        for &(idx, _, done, take, _, _) in shares {
            if let Some(m) = &self.chunked[idx].mm {
                for ax in 0..4 {
                    let src = ax * m.t_len + done;
                    mrope_h[ax * r + rb..ax * r + rb + take]
                        .copy_from_slice(&m.mrope[src..src + take]);
                }
            }
            rb += take;
        }
    }

    /// A span of an mm chunk landed at row `landed` (the tick is done, so the
    /// slot's DeltaNet state is the state there): publish the picture-keyed
    /// checkpoint when it is a cut, or on finish cache the full pages and set
    /// the decode mRoPE delta the picture rows displaced - what the planner's
    /// own pass did after `mm_prefill_span`.
    pub(super) fn mm_ride_landed(
        &mut self,
        idx: usize,
        slot: usize,
        landed: usize,
        finishing: bool,
    ) -> Result<(), GpuModelError> {
        let (keys, t_len, cut, fmp) = {
            let m = self.chunked[idx].mm.as_ref().expect("an mm chunk");
            (
                m.keys.clone(),
                m.t_len,
                m.cuts.contains(&landed),
                m.final_mrope_pos,
            )
        };
        if paddock_models::dev_var_os!("PADDOCK_UNIFIED_SHAPE_LOG").is_some() {
            eprintln!(
                "[mmride] landed slot={slot} landed={landed} finishing={finishing} cut={cut}"
            );
        }
        if finishing {
            self.mm_prefix_publish(slot, &keys, t_len / BLOCK_TOKENS * BLOCK_TOKENS, false)?;
            self.batch.as_mut().expect("batch").mrope_delta[slot] = fmp as i64 - t_len as i64;
        } else if cut {
            self.mm_prefix_publish(slot, &keys, landed, true)?;
        }
        Ok(())
    }
}
