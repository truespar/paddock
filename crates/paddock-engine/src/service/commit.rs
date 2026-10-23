//! Committing a slot's next token: the device-sampled row's id, or the host
//! pick from its logits row - and, for the device rows of a slot inside an
//! open reasoning block, the stop masks the backend needs before it samples
//! (pack slot 850, `sampler/reasoning.rs`).

use super::{EngineError, Slot, TokenEvent, compute_logprobs, pick_next};
use crate::generator::{Generator, RowSample};

/// Hand the backend the stop masks of a device-sampled call's rows (row i
/// is `plans[i]`, drawn for `slot(i)`) - nothing when no row draws inside an
/// open reasoning block. Call right before the call; take the pressure back
/// right after it (`Generator::take_stop_pressure`).
pub(super) fn send_stop_masks<'a>(
    g: &mut dyn Generator,
    plans: &[RowSample],
    slot: impl Fn(usize) -> Option<&'a Slot>,
) {
    let m: Vec<_> = plans
        .iter()
        .enumerate()
        .map(|(i, p)| match p {
            RowSample::Device(_) => slot(i).and_then(|s| s.sampler.stop_mask(&s.stop_tokens, 0)),
            _ => None,
        })
        .collect();
    if m.iter().any(Option::is_some) {
        g.set_stop_masks(m);
    }
}

/// The same for a device verify round: its flat rows, each slot's chunk
/// after the last, the j-th row of a chunk masked as the slot's pick j ahead.
pub(super) fn send_verify_masks(
    g: &mut dyn Generator,
    slots: &[Option<Slot>],
    reqs: &[(usize, usize, Vec<u32>)],
) {
    let m: Vec<_> = reqs
        .iter()
        .flat_map(|(k, _, c)| {
            let s = slots[*k].as_ref();
            (0..c.len() as u32).map(move |j| s.and_then(|s| s.sampler.stop_mask(&s.stop_tokens, j)))
        })
        .collect();
    if m.iter().any(Option::is_some) {
        g.set_stop_masks(m);
    }
}

/// Commit a device-sampled token (the phase-3 body for GPU-picked rows: no
/// host logits exist - constraint/logprob slots never take this path), with
/// its row's stop pressure (false where no stop mask rode the row).
/// PADDOCK_TOK_TRACE=1: one line per committed device token (slot, position
/// it lands at, id, the phase that produced it) - the instrument that places a
/// wrong token at the tick that committed it.
pub(super) fn tok_trace(k: usize, slot: &Option<Slot>, id: u32, src: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_TOK_TRACE").is_some()) {
        return;
    }
    if let Some(s) = slot {
        eprintln!("[tok] slot={k} pos={} id={id} src={src}", s.pos);
    }
}

pub(super) fn commit_device_token(slot_opt: &mut Option<Slot>, next: u32, pressure: bool) {
    let Some(slot) = slot_opt.as_mut() else {
        return;
    };
    // A slot with no KV behind it has no token to commit - whatever came back
    // for its row is a hole. Guarded here as well as at the plan, because the
    // plan is built in several places and this is the one door they all use.
    if !slot.prefilled {
        return;
    }
    slot.pos += 1; // the pending token is now committed at its position
    slot.sampler.note_drawn(next, pressure);
    slot.draft.push(next);
    if !slot.accept(next, None) {
        *slot_opt = None;
    }
}

/// Sample + commit one slot's next token from its decode-tick logits row
/// (the phase-3 body, shared by the serial and scoped-thread paths).
pub(super) fn sample_slot_row(slot_opt: &mut Option<Slot>, row: &mut [f32]) {
    let Some(slot) = slot_opt.as_mut() else {
        return;
    };
    // Same guard as `commit_device_token`: an unprefilled slot's row is a hole,
    // and sampling it would stream a token drawn from nothing.
    if !slot.prefilled {
        return;
    }
    slot.pos += 1; // the pending token is now committed at its position
    let raw = slot.logprobs.map(|_| row.to_vec());
    let next = match pick_next(
        &mut slot.sampler,
        row,
        &slot.history,
        &mut slot.constraint,
        &slot.stop_tokens,
    ) {
        Ok(t) => t,
        Err(e) => {
            let _ = slot
                .events
                .send(TokenEvent::Error(EngineError::internal(&e)));
            *slot_opt = None;
            return;
        }
    };
    let lp = slot
        .logprobs
        .map(|n| compute_logprobs(raw.as_ref().expect("snapshot"), next, n));
    // keep the drafter's history gapless even on dense ticks (a
    // greedy slot may ride spec rounds later; gapped history only
    // degrades proposals, but why degrade)
    slot.draft.push(next);
    if !slot.accept(next, lp) {
        *slot_opt = None;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::{FinishReason, GenRequest, ShutdownCtl, TokenEvent, run_batched};
    use crate::generator::{GenError, Generator, RowSample, SampledStep};
    use crate::metrics::EngineMetrics;
    use crate::sampler::{ReasoningStop, SamplingParams, StopMask};

    const EOS: u32 = 0;
    const CLOSE: u32 = 7;

    /// A backend that masks stops on device the way slot 850 does: a device
    /// row that carries a mask draws text (2) and reports stop pressure; one
    /// without draws the stop - the model asking to finish. Host rows (and
    /// the prefill) put EOS on top.
    #[derive(Default)]
    struct Masked {
        masks: Vec<Option<StopMask>>,
        pressure: Vec<bool>,
        masked_rows: usize,
    }

    fn eos_on_top() -> Vec<f32> {
        vec![9.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    impl Generator for Masked {
        fn reset(&mut self) {}
        fn vocab(&self) -> usize {
            8
        }
        fn forward(&mut self, _: u32) -> Result<Vec<f32>, GenError> {
            Ok(eos_on_top())
        }
        fn forward_prefill_batch(
            &mut self,
            items: &[(usize, Vec<u32>)],
        ) -> Result<Vec<Vec<f32>>, GenError> {
            Ok(items.iter().map(|_| eos_on_top()).collect())
        }
        fn supports_device_sampling(&self) -> bool {
            true
        }
        fn forward_batch_sampled(
            &mut self,
            tokens: &[u32],
            _: &[u32],
            plans: &[RowSample],
        ) -> Result<SampledStep, GenError> {
            let masks = std::mem::take(&mut self.masks);
            let mut step = SampledStep {
                ids: vec![0; tokens.len()],
                host_rows: Vec::new(),
            };
            self.pressure = vec![false; tokens.len()];
            for (i, p) in plans.iter().enumerate() {
                match p {
                    RowSample::Device(_) if masks.get(i).is_some_and(Option::is_some) => {
                        let m = masks[i].expect("mask");
                        assert_eq!((m.n, m.stops[0], m.inv_t), (1, EOS, 0.0));
                        step.ids[i] = 2;
                        self.pressure[i] = true;
                        self.masked_rows += 1;
                    }
                    RowSample::Device(_) => step.ids[i] = EOS,
                    RowSample::Host => step.host_rows.push((i, eos_on_top())),
                    RowSample::Hole => {}
                }
            }
            Ok(step)
        }
        fn supports_stop_mask(&self) -> bool {
            true
        }
        fn set_stop_masks(&mut self, masks: Vec<Option<StopMask>>) {
            self.masks = masks;
        }
        fn take_stop_pressure(&mut self) -> Vec<bool> {
            self.masks.clear();
            std::mem::take(&mut self.pressure)
        }
    }

    /// A greedy reply inside an open think block: the prefill's host pick
    /// masks EOS (pressure 1), two masked device ticks draw text with
    /// pressure (2, 3 - the run), the exit is forced token by token on the
    /// host, and after the close the device's unmasked stop ends the reply.
    #[test]
    fn device_rows_inside_the_block_carry_the_mask_and_its_pressure() {
        let (send, requests) = std::sync::mpsc::channel();
        let (events, mut rx) = tokio::sync::mpsc::unbounded_channel();
        send.send(GenRequest {
            prompt: vec![1; 4],
            max_tokens: 32,
            sampler: SamplingParams {
                temperature: 0.0,
                reasoning: Some(ReasoningStop {
                    leave: vec![CLOSE],
                    exit: vec![5, 6, CLOSE],
                }),
                ..SamplingParams::default()
            },
            stop_tokens: vec![EOS],
            events,
            mm_chunks: None,
            constraint: None,
            logprobs: None,
            submitted: None,
            canvas_read: None,
            user_turn: false,
            background: false,
        })
        .unwrap();
        drop(send);
        let mut g = Masked::default();
        let metrics = Arc::new(EngineMetrics::default());
        run_batched(&mut g, &requests, 2, 8, &metrics, &ShutdownCtl::new(), None);
        let (mut ids, mut done) = (Vec::new(), None);
        while let Ok(e) = rx.try_recv() {
            match e {
                TokenEvent::Token { id, .. } => ids.push(id),
                TokenEvent::Done(r, _) => done = Some(r),
                TokenEvent::Error(e) => panic!("{}", e.message),
                _ => {}
            }
        }
        assert_eq!(ids, vec![1, 2, 2, 5, 6, CLOSE]);
        assert!(
            matches!(done, Some(FinishReason::Stop)),
            "the stop after the block ends it"
        );
        assert_eq!(
            g.masked_rows, 2,
            "only the in-block device ticks carried a mask"
        );
    }
}
