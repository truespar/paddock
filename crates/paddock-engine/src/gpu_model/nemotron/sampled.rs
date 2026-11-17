//! Device sampling on the batch lane and the pipelined decode ticks (the
//! host overlaps tick t's readback with tick t+1's launch). Split out of
//! `batch.rs` (which had crossed the 2,500-line ceiling).

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;

use super::batch::*;
use super::*;

impl GpuNemotron {
    /// Pack per-row sampler params (inv_t, u, mode, pad). Host/Hole rows
    /// stay mode 0 = untouched.
    /// TruncCat rows pack mode 5 (top_k 1..=64) or mode 6 (k-less -
    /// nemotron's own election) + the tpar side plane (Some iff any).
    pub(super) fn pack_samp_par(
        plans: &[crate::generator::RowSample],
    ) -> (Vec<u32>, Option<Vec<u32>>, bool, bool) {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        let mut tpar = vec![0u32; plans.len() * 4];
        // which trunc chains a tick actually needs: nemotron's own election
        // is pure mode 6, so the mode-5 launches would all early-return -
        // skip whole chains, not rows (a launch is the unit of waste here)
        let (mut any5, mut any6) = (false, false);
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => par[i * 4 + 2] = 1,
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                RowSample::Device(DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    let mode5 = *k >= 1 && *k <= 64;
                    par[i * 4 + 2] = if mode5 { 5 } else { 6 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    if mode5 { any5 = true } else { any6 = true }
                }
                // RS plans are spec-only; nemotron's batch lane has no
                // drafter yet
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        (par, (any5 || any6).then_some(tpar), any5, any6)
    }

    /// device-truncation engagement witness (bisect-trap law): once per process.
    pub(super) fn trunc_dev_witness(rows: usize) {
        static DEV: std::sync::Once = std::sync::Once::new();
        DEV.call_once(|| {
            eprintln!("[trunc-dev6] engaged: r={rows} (nemotron device truncation sampling)");
        });
    }

    /// TruncCat rows execute fully on device (slots 435+436).
    pub(crate) fn device_trunc_supported(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows_t() && self.exec.has_sample_rows_p()
    }

    pub(crate) fn supports_device_sampling_impl(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows()
    }

    /// Sample head_logits rows 0..r on device with `plans`; only Host-plan
    /// rows pay a vocab-row readback. Assumes the head already ran. `masked`:
    /// these rows are the call's rows 0..r, the ones the service's stop masks
    /// index (a finisher's lone row is not).
    pub(super) fn sample_head_rows(
        &mut self,
        r: usize,
        plans: &[crate::generator::RowSample],
        masked: bool,
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        assert_eq!(plans.len(), r, "one plan per row");
        let exec = self.exec.clone();
        let vocab = self.hp.vocab;
        let (par, tpar, any5, any6) = Self::pack_samp_par(plans);
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("d_par slice".into()))?;
            exec.htod(&par, &mut v)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_tpar
                    .try_slice_mut(0..r * 4)
                    .ok_or_else(|| GpuError::Driver("d_tpar slice".into()))?;
                exec.htod(t, &mut v)?;
            }
            if masked {
                let at: Vec<usize> = (0..r).collect();
                self.stop
                    .launch(&exec, &mut sc.head_logits, 0, vocab, &at)?;
            }
            exec.sample_rows_at(&sc.head_logits, &sc.d_par, 0, &mut sc.d_out, 0, r, vocab)?;
            if tpar.is_some() {
                Self::trunc_dev_witness(r);
                if any5 {
                    exec.sample_rows_t_at(
                        &sc.head_logits,
                        &sc.d_par,
                        0,
                        &sc.d_tpar,
                        0,
                        &mut sc.d_out,
                        0,
                        r,
                        vocab,
                    )?;
                }
                if any6 {
                    exec.sample_rows_p_at(
                        &sc.head_logits,
                        &sc.d_par,
                        0,
                        &sc.d_tpar,
                        0,
                        &mut sc.d_out,
                        0,
                        r,
                        vocab,
                    )?;
                }
            }
        }
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
        let ids_view = sc
            .d_out
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("d_out slice".into()))?;
        let ids = exec.stream.clone_dtoh(&ids_view).map_err(drv)?;
        let mut host_rows = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                let v = sc
                    .head_logits
                    .try_slice(i * vocab..(i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("host row slice".into()))?;
                host_rows.push((i, exec.stream.clone_dtoh(&v).map_err(drv)?));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// Device-sampled decode tick: graph replay + sample_rows.
    pub(crate) fn forward_batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        self.forward_batch_sampled_slots(tokens, positions, None, plans)
    }

    pub(crate) fn forward_batch_sampled_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        let r = tokens.len();
        let owned: Vec<u32> = (0..r as u32).collect();
        let ident: &[u32] = slots.unwrap_or(&owned);
        assert_eq!(ident.len(), r, "one slot per row");
        self.ensure_rows(ident, positions)?;
        self.upload_rows(tokens, positions, ident)?;
        for i in 0..r {
            self.reply_feed(ident[i] as usize, positions[i], tokens[i]);
        }
        self.step_replay(r)?;
        self.reply_after_rows(ident, positions)?;
        self.dflash_note_ticks(ident, positions);
        let step = self.sample_head_rows(r, plans, true)?;
        self.mtp_append_ticks(ident, positions)?;
        Ok(step)
    }

    pub(crate) fn supports_decode_pipe_batch(&self) -> bool {
        self.exec.has_sample_rows()
            && self.exec.has_pipe_advance()
            && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none()
            // the in-file MTP's h chain needs host staging around every
            // tick - incompatible with the pipe's fire-and-forget replays.
            // Spec-on serving decodes through spec rounds instead; --no-spec
            // serves never load the block, so the pipe survives there.
            && !self.mtp_active()
    }

    pub(super) fn pipe_launch_tick_b(
        &mut self,
        plans: &[crate::generator::RowSample],
        advance: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.hp.vocab;
        let (b, tick) = {
            let p = self.pipe_b.as_ref().expect("pipe active");
            (p.b, p.tick)
        };
        // back every row's this-tick write position before anything mutates -
        // a growth error leaves the rings/inputs untouched
        let (slots_v, pos_v) = {
            let (pos0, slot_map) = {
                let p = self.pipe_b.as_ref().expect("pipe active");
                (p.pos0.clone(), p.slots.clone())
            };
            let slots_v: Vec<u32> = (0..b as u32)
                .map(|i| slot_map.as_ref().map_or(i, |s| s[i as usize]))
                .collect();
            let pos_v: Vec<u32> = pos0.iter().map(|&p0| p0 + tick as u32).collect();
            self.ensure_rows(&slots_v, &pos_v)?;
            (slots_v, pos_v)
        };
        let ring = tick % 2;
        let (par, tpar, any5, any6) = Self::pack_samp_par(plans);
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let off = ring * n_slots * 4;
            let mut v = sc
                .d_pipe_par
                .try_slice_mut(off..off + b * 4)
                .ok_or_else(|| GpuError::Driver("d_pipe_par slice".into()))?;
            exec.htod(&par, &mut v)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_pipe_tpar
                    .try_slice_mut(off..off + b * 4)
                    .ok_or_else(|| GpuError::Driver("d_pipe_tpar slice".into()))?;
                exec.htod(t, &mut v)?;
            }
        }
        if advance {
            // tokens <- previous ring's sampled ids, positions += 1, on device
            let prev = (tick + 1) % 2;
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let (out, tok, pos) = (&sc.d_pipe_out, &mut sc.d_tok, &mut sc.d_pos);
            exec.pipe_advance(out, prev * n_slots, tok, pos, b)?;
        }
        self.step_replay(b)?;
        // stage F: the snapshot copies ride the stream behind this tick; the
        // ids that complete their pages arrive with the next host read
        self.reply_after_rows(&slots_v, &pos_v)?;
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            exec.sample_rows_at(
                &sc.head_logits,
                &sc.d_pipe_par,
                ring * n_slots * 4,
                &mut sc.d_pipe_out,
                ring * n_slots,
                b,
                vocab,
            )?;
            // trunc rows draw into the same out ring - pipe_advance
            // feeds their ids forward exactly like mode-1/2 rows
            if tpar.is_some() {
                Self::trunc_dev_witness(b);
                if any5 {
                    exec.sample_rows_t_at(
                        &sc.head_logits,
                        &sc.d_pipe_par,
                        ring * n_slots * 4,
                        &sc.d_pipe_tpar,
                        ring * n_slots * 4,
                        &mut sc.d_pipe_out,
                        ring * n_slots,
                        b,
                        vocab,
                    )?;
                }
                if any6 {
                    exec.sample_rows_p_at(
                        &sc.head_logits,
                        &sc.d_pipe_par,
                        ring * n_slots * 4,
                        &sc.d_pipe_tpar,
                        ring * n_slots * 4,
                        &mut sc.d_pipe_out,
                        ring * n_slots,
                        b,
                        vocab,
                    )?;
                }
            }
        }
        let ev = exec.record_event()?;
        self.pipe_b.as_mut().expect("pipe active").ev[ring] = Some(ev);
        Ok(())
    }

    pub(crate) fn decode_pipe_begin_b(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        assert_eq!(positions.len(), b, "one position per row");
        if !self.supports_decode_pipe_batch() {
            return Err(GpuModelError::Config("decode pipe unsupported".into()));
        }
        match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) if b > bs.n_slots => {
                return Err(GpuModelError::BatchTooLarge {
                    got: b,
                    max: bs.n_slots,
                });
            }
            _ => {}
        }
        if let Some(s) = slots {
            assert_eq!(s.len(), b, "one slot per row");
        }
        assert!(self.pipe_b.is_none(), "decode pipe already active");
        // tick-0 inputs land in the fixed graph buffers (advance=false keeps
        // them); ensure_rows runs inside pipe_launch_tick_b at tick 0
        let ident: Vec<u32> = (0..b as u32).collect();
        self.upload_rows(tokens, positions, slots.unwrap_or(&ident))?;
        for i in 0..b {
            let slot = slots.map_or(i, |s| s[i] as usize);
            self.reply_feed(slot, positions[i], tokens[i]);
        }
        // the drafter's coverage follows the pipe's input rows as the host
        // learns them: tick 0's here, tick k's when tick k-1's ids come back
        let ident: Vec<u32> = (0..b as u32).collect();
        self.dflash_note_ticks(slots.unwrap_or(&ident), positions);
        self.pipe_b = Some(PipeB {
            b,
            tick: 0,
            ev: [None, None],
            pos0: positions.to_vec(),
            slots: slots.map(<[u32]>::to_vec),
        });
        if let Err(e) = self.pipe_launch_tick_b(plans, false) {
            self.pipe_b_abort();
            return Err(e);
        }
        Ok(())
    }

    /// Enqueue the next tick and return the OLDEST in-flight tick's ids, read
    /// via the copy stream while the new tick executes.
    pub(crate) fn decode_pipe_next_b(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let (b, j) = {
            let p = self
                .pipe_b
                .as_ref()
                .ok_or_else(|| GpuModelError::Config("decode_pipe_next without begin".into()))?;
            (p.b, p.tick)
        };
        assert_eq!(plans.len(), b, "one plan per row");
        self.pipe_b.as_mut().expect("pipe active").tick = j + 1;
        if let Err(e) = self.pipe_launch_tick_b(plans, true) {
            self.pipe_b_abort();
            return Err(e);
        }
        let ring = j % 2;
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        let r = {
            let sc = &self.batch.as_ref().expect("batch enabled").sc;
            let ev = self.pipe_b.as_ref().expect("pipe active").ev[ring]
                .as_ref()
                .expect("in-flight event");
            exec.to_host_u32_after(ev, &sc.d_pipe_out, ring * n_slots, b)
        };
        match r {
            Ok(ids) => {
                let (pos0, slots) = {
                    let p = self.pipe_b.as_ref().expect("pipe active");
                    (p.pos0.clone(), p.slots.clone())
                };
                self.reply_pipe_ids(&ids, &pos0, slots.as_deref(), j + 1);
                // these ids are the inputs of tick j + 1, launched above -
                // its features are in the drafter cache once it runs
                let ident: Vec<u32> = (0..ids.len() as u32).collect();
                let at: Vec<u32> = pos0.iter().map(|&p| p + j as u32 + 1).collect();
                self.dflash_note_ticks(slots.as_deref().unwrap_or(&ident), &at);
                Ok(ids)
            }
            Err(e) => {
                self.pipe_b_abort();
                Err(e.into())
            }
        }
    }

    /// End the pipe: return the last in-flight tick's ids. The fixed input
    /// buffers are stale after this - every other path re-uploads them.
    pub(crate) fn decode_pipe_drain_b(&mut self) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let st = self
            .pipe_b
            .take()
            .ok_or_else(|| GpuModelError::Config("decode_pipe_drain without begin".into()))?;
        let ring = st.tick % 2;
        let n_slots = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .n_slots;
        let ev = st.ev[ring].as_ref().expect("in-flight event");
        let r = {
            let sc = &self.batch.as_ref().expect("batch enabled").sc;
            exec.to_host_u32_after(ev, &sc.d_pipe_out, ring * n_slots, st.b)
        };
        match r {
            Ok(ids) => {
                self.reply_pipe_ids(&ids, &st.pos0, st.slots.as_deref(), st.tick + 1);
                Ok(ids)
            }
            Err(e) => {
                let _ = exec.synchronize(); // state gone - quiesce ring readers
                Err(e.into())
            }
        }
    }

    /// Kill an in-flight batch pipe (error/reset/re-enable): quiesce so
    /// nothing still reads the rings, then drop the state.
    pub(crate) fn pipe_b_abort(&mut self) {
        if self.pipe_b.take().is_some() {
            let _ = self.exec.synchronize();
        }
    }
}
