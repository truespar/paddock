//! The unified prefill span's public half: launch without waiting, poll,
//! finish - and the error exit every tick that launched one takes (the
//! layer walk itself is `unified_launch_core` / `unified_finish_core` in
//! batch.rs).

use super::*;
use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;

impl GpuQwen35 {
    /// Launch a prefill-only unified span (no decode rows) without waiting
    /// for it: the layer walk and finisher sampling are enqueued on the main
    /// lane and an event marks completion. Returns false (launching nothing)
    /// when the chunk queue is empty. While the span is in flight the
    /// overlap scheduler pumps decode-pipe ticks on the decode lane -
    /// disjoint slots, and finishers sample in the span-side d_fin_*
    /// buffers, so nothing touches the decode graph's rows. Admission may
    /// APPEND to the chunk queue during the flight; `unified_span_finish`
    /// must run before any other unified/prefill/mixed call.
    pub fn unified_span_launch(
        &mut self,
        budget: usize,
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<bool, GpuModelError> {
        if self.chunked.is_empty() {
            return Ok(false);
        }
        self.unified_launch_core(&[], budget, &[], fin_plans)?;
        let flushed = self.dflash_flush_pending_append().map_err(Into::into);
        self.unified_span_settle(flushed)?;
        // The overlap scheduler begins its decode pipe BEFORE the span
        // (pipe-first). A shared-scratch grow inside the launch drops every
        // batch graph - the in-flight replay finishes (exec destroy is
        // deferred), but the pipe's next tick indexes graphs[b], so recapture
        // it here (decode lane, against the decode arena, not the scratch).
        let regraph = match self.pipe.as_ref().map(|p| p.b) {
            Some(b) => self.with_decode_lane(|m| m.ensure_batch_graph(b)),
            None => Ok(()),
        };
        self.unified_span_settle(regraph)?;
        Ok(true)
    }

    /// Non-blocking: has the in-flight span's GPU work (incl. finisher
    /// sampling) completed? True when no span is in flight.
    pub fn unified_span_done(&self) -> bool {
        self.unified_inflight
            .as_ref()
            .is_none_or(|s| self.exec.event_done(&s.ev))
    }

    /// Complete the in-flight span: drain the main lane, read finisher ids,
    /// run the warm/prefix hooks, advance the chunk queue. Returns the
    /// finished prompts.
    pub fn unified_span_finish(
        &mut self,
    ) -> Result<Vec<(usize, crate::generator::FinishSample, usize)>, GpuModelError> {
        let (_step, finished) = self.unified_finish_core()?;
        Ok(finished)
    }

    /// A tick that fails while its span is in flight: drain the span and drop
    /// its results, so the error leaves no span behind. Returning with one in
    /// flight left `unified_inflight` set; the service failed the tick's
    /// slots and its next launch hit "unified span already in flight", a
    /// panic on the engine thread. The span has advanced the chunk queue, so
    /// the tick cannot be retried: pool exhaustion becomes a plain failure,
    /// since the service's preempt-and-retry answer would keep slots whose
    /// prompts just finished with nothing left to report them. With no span
    /// in flight the error passes through unchanged.
    pub fn unified_span_abandon(&mut self, e: GpuModelError) -> GpuModelError {
        if self.unified_inflight.is_none() {
            return e;
        }
        if let Err(f) = self.unified_finish_core() {
            tracing::warn!("unified span: drain after a failed tick: {f}");
        }
        match e {
            GpuModelError::PoolExhausted => GpuError::Driver(
                "KV pool exhausted after the prefill span launched; the tick cannot be retried"
                    .into(),
            )
            .into(),
            e => e,
        }
    }

    /// `r` itself, or - an error - the span abandoned first.
    pub(super) fn unified_span_settle<T>(
        &mut self,
        r: Result<T, GpuModelError>,
    ) -> Result<T, GpuModelError> {
        r.map_err(|e| self.unified_span_abandon(e))
    }
}
