//! The device-sampled decode tick: the rows' logits reduced to ids on device
//! (`pd_sample_rows` and its truncation twins), with the reasoning stop mask
//! (pack slot 850) launched on the rows the service masked, ahead of every
//! pick - device, host-head or the small-batch host tail alike.

use super::*;

impl GpuQwen35 {
    /// True when the pack ships the fused row-sampling kernel - lets the
    /// scheduler take `forward_batch_sampled` and skip the [B, vocab] readback.
    /// K-quant models report false: the batched serving paths are Q8_0-class
    /// until the stage-2 W4A8 kernels, so they serve on the serial spine.
    pub fn supports_device_sampling(&self) -> bool {
        // sampling reads d_logits - the k-quant W4A8 head fills the same buffer
        self.exec.has_sample_rows()
    }

    /// `forward_batch` + fused on-device sampling (the gpt-oss path ported to
    /// qwen35): `Device` rows come back as bare token ids (b x 4 bytes) instead
    /// of the full `[b, vocab]` logits readback; `Host` rows (penalties,
    /// constraints, logprobs) still get their own logits row copied back. The
    /// sampling kernel is model-agnostic - it reads the same `d_logits` buffer
    /// `forward_batch` fills, so this is bit-identical to host argmax on greedy
    /// rows.
    pub fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        self.batch_sampled_impl(tokens, positions, None, plans)
    }

    /// `forward_batch_sampled` with an optional explicit slot mapping (see
    /// `launch_batch_step_slots`). `None` = identity (pure decode); `Some` =
    /// the mixed tick's compacted decode rows at arbitrary slots. Runs on the
    /// overlap decode lane when forked.
    pub(super) fn batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        // Split decode tick (the c128 lane): every fast decode
        // arm (mma_ks ladder, f8t plane, W8 tile class) is bounded b <= 64
        // (pack guards + tile shapes), so a 65..128-row tick fell to slow
        // eager lanes (~62ms vs 10.4ms at 64). At b > 64 any <=64 kernel
        // needs two weight passes anyway - run the tick as <=64-row halves
        // through the fast graphed path instead: rows are independent
        // (per-slot KV/state, per-row glue/sampling), so per-row results
        // are the 64-row tick's exact outputs. Requires explicit slots
        // (identity mapping cannot survive a split); serve always passes
        // them. Kill: PADDOCK_NO_DEC_SPLIT.
        let b = tokens.len();
        let bmax = f8t_dec_bmax();
        if b > bmax && paddock_models::dev_var_os!("PADDOCK_NO_DEC_SPLIT").is_none() {
            // identity mapping is materialized so the halves stay correct
            // (the None fast path cannot survive a split) - this None case
            // was the last b=128 hole: forward_batch_sampled passes None,
            // and the first post-warmup tick captured the mma graph.
            let ident: Vec<u32>;
            let sl: &[u32] = match slots {
                Some(s) => s,
                None => {
                    ident = (0..b as u32).collect();
                    &ident
                }
            };
            let mut ids = Vec::with_capacity(b);
            let mut host_rows = Vec::new();
            let mut b0 = 0usize;
            while b0 < b {
                let bn = (b - b0).min(bmax);
                let step = self.with_decode_lane(|m| {
                    m.batch_sampled_inner(
                        &tokens[b0..b0 + bn],
                        &positions[b0..b0 + bn],
                        Some(&sl[b0..b0 + bn]),
                        &plans[b0..b0 + bn],
                        b0,
                    )
                })?;
                ids.extend(step.ids);
                host_rows.extend(step.host_rows.into_iter().map(|(r, l)| (r + b0, l)));
                b0 += bn;
            }
            return Ok(crate::generator::SampledStep { ids, host_rows });
        }
        self.with_decode_lane(|m| m.batch_sampled_inner(tokens, positions, slots, plans, 0))
    }

    /// `at0`: the call row (the service's plan and stop-mask index) of this
    /// tick's row 0 - a split tick's second half starts past the first.
    fn batch_sampled_inner(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
        at0: usize,
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        use crate::sampler::DevicePlan;
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        // pack per-row sampler params; the kernel skips mode-0 rows (holes AND
        // host rows - the host reads the latter's logits itself)
        // small-b: the one-block-per-row prefilter is DRAM-latency-bound
        // (c1 -13% pre-fallback); the classic full-row readback + host
        // top-64 is cheaper there and distribution-identical
        let small_b = b <= 2;
        // P67: full-device sampling (mode 5) whenever the pack ships slot
        // 435 - the token lands in d_samp_out like modes 1/2, no head
        // readback, no host tail. Old packs keep mode 4 (host-head).
        let dev_full = !small_b && exec.has_sample_rows_t();
        let mut par = vec![0u32; b * 4];
        let mut tpar = vec![0u32; b * 4];
        let mut any_trunc = false;
        let mut any_base = false;
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => {
                    par[i * 4 + 2] = 1;
                    any_base = true;
                }
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                    any_base = true;
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
                    par[i * 4 + 2] = if dev_full { 5 } else { 4 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    any_trunc = true;
                }
                // RS plans are gemma4-only (supports_spec_rs); skip-safe
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        {
            let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
            let mut v = bs.d_samp_par.slice_mut(0..b * 4);
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if any_trunc && dev_full {
                let mut t = bs.d_samp_tpar.slice_mut(0..b * 4);
                exec.stream.memcpy_htod(&tpar, &mut t).map_err(drv)?;
            }
        }
        self.launch_batch_step_slots(tokens, positions, slots)?;
        // DFlash: this tick's rows just walked (taps baked in the step
        // graph) - fuse and ring-append before the ids readback syncs.
        if self.dflash.as_ref().is_some_and(|d| d.state.is_some()) && !super::dflash::fuse_off() {
            let ident: Vec<u32>;
            let slots_v: &[u32] = match slots {
                Some(s) => s,
                None => {
                    ident = (0..b as u32).collect();
                    &ident
                }
            };
            self.dflash_append_features(tokens, positions, slots_v, None)?;
        }
        // stage F: the rows' states advanced in this pass (stream-ordered
        // behind it, ahead of the ids readback below)
        self.reply_after_rows(tokens, positions, slots)?;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        let at: Vec<usize> = (at0..at0 + b).collect();
        self.stop.launch(&exec, &mut bs.d_logits, 0, vocab, &at)?;
        // folded launches: this site writes mode 5 for every trunc row (no
        // mode 6 exists here), so the p chain was 11 launches of nothing
        let fold_off = Self::samp_fold_off();
        if any_base || fold_off {
            exec.sample_rows(&bs.d_logits, &bs.d_samp_par, &mut bs.d_samp_out, b, vocab)?;
        }
        if any_trunc && dev_full {
            // engagement witness (bisect-trap law): once per process
            static DEV5: std::sync::Once = std::sync::Once::new();
            DEV5.call_once(|| {
                eprintln!("[trunc-dev5] engaged: b={b} (full-device truncation sampling)");
            });
            Self::samp_fold_witness(any_base, true, false);
            exec.sample_rows_t(
                &bs.d_logits,
                &bs.d_samp_par,
                &bs.d_samp_tpar,
                &mut bs.d_samp_out,
                b,
                vocab,
            )?;
            if fold_off {
                exec.sample_rows_p(
                    &bs.d_logits,
                    &bs.d_samp_par,
                    &bs.d_samp_tpar,
                    &mut bs.d_samp_out,
                    b,
                    vocab,
                )?;
            }
        } else if any_trunc && !small_b {
            exec.topk_rows(
                &bs.d_logits,
                &bs.d_samp_par,
                &mut bs.d_samp_head,
                b,
                vocab,
                64,
            )?;
        }
        let ids_view = bs
            .d_samp_out
            .try_slice(0..b)
            .ok_or_else(|| GpuError::Driver("samp_out slice out of range".into()))?;
        let mut ids = exec
            .stream
            .clone_dtoh(&ids_view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        if any_trunc && dev_full {
            // mode 5: ids already carry the device-sampled tokens
        } else if any_trunc && !small_b {
            let hv = bs
                .d_samp_head
                .try_slice(0..b * 128)
                .ok_or_else(|| GpuError::Driver("samp_head slice out of range".into()))?;
            let head = exec
                .stream
                .clone_dtoh(&hv)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            trunc_fill_ids(&head, plans, &mut ids);
        } else if any_trunc {
            for (i, p) in plans.iter().enumerate() {
                if let RowSample::Device(DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                }) = *p
                {
                    let view = bs
                        .d_logits
                        .try_slice(i * vocab..(i + 1) * vocab)
                        .ok_or_else(|| GpuError::Driver("logits row slice out of range".into()))?;
                    let row = exec
                        .stream
                        .clone_dtoh(&view)
                        .map_err(|e| GpuError::Driver(e.to_string()))?;
                    let head = host_top64(&row);
                    ids[i] = crate::sampler::sample_trunc_head(&head, inv_t, u, k, top_p, min_p);
                }
            }
        }
        let mut host_rows = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                let view = bs
                    .d_logits
                    .try_slice(i * vocab..(i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("logits row slice out of range".into()))?;
                let row = exec
                    .stream
                    .clone_dtoh(&view)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                host_rows.push((i, row));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// The reasoning stop mask over a verify round's padded block rows (block
    /// i's row j is the service's flat row `off_i + j`; pad rows carry none),
    /// right after the walk's head and before any pick reads the logits. The
    /// greedy round's argmax ran inside the walk, so it runs again over the
    /// masked rows when any row was masked.
    pub(in crate::gpu_model::qwen35) fn spec_stop_mask(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        k1: usize,
        greedy: bool,
    ) -> Result<(), GpuModelError> {
        let mut at = Vec::with_capacity(reqs.len() * k1);
        let mut off = 0usize;
        for (_, _, c) in reqs {
            at.extend(off..off + c.len());
            at.extend(std::iter::repeat_n(usize::MAX, k1 - c.len()));
            off += c.len();
        }
        let (exec, vocab) = (self.exec.clone(), self.vocab);
        let sb = self.spec_batch.as_mut().expect("spec batch");
        if self
            .stop
            .launch(&exec, &mut sb.d_logits_chunk, 0, vocab, &at)?
            && greedy
        {
            exec.argmax_rows(&sb.d_logits_chunk, &mut sb.d_row_tok, at.len(), vocab)?;
        }
        Ok(())
    }

    /// A verify round's accept walk over its padded block rows, exactly as the
    /// service walks it - cut where a reasoning row's mask stops holding: the
    /// rows each slot commits, and the real rows' picks the service gets back.
    pub(in crate::gpu_model::qwen35) fn spec_walk(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        padded: &[u32],
        picks_all: &[u32],
        k1: usize,
    ) -> Result<(Vec<u32>, Vec<u32>), GpuModelError> {
        self.stop.read(&self.exec)?;
        let mut committed: Vec<u32> = Vec::with_capacity(reqs.len());
        let mut picks_out: Vec<u32> = Vec::new();
        let mut off = 0usize;
        for (i, (_, _, chunk)) in reqs.iter().enumerate() {
            let base = i * k1;
            let mut a = 0usize;
            while a + 1 < chunk.len() && padded[base + a + 1] == picks_all[base + a] {
                a += 1;
            }
            let row = &picks_all[base..base + chunk.len()];
            committed.push((a.min(self.stop.cut(off, row)) + 1) as u32);
            picks_out.extend_from_slice(row);
            off += chunk.len();
        }
        Ok((committed, picks_out))
    }

    /// The reasoning stop mask rides every device-sampled call above (pack
    /// slot 850): `Generator::supports_stop_mask` and its two hand-offs.
    pub fn stop_mask_supported(&self) -> bool {
        self.supports_device_sampling() && self.exec.has_stop_mask_rows()
    }

    pub fn set_stop_masks(&mut self, masks: Vec<Option<crate::sampler::StopMask>>) {
        self.stop.set(masks);
    }

    pub fn take_stop_pressure(&mut self) -> Vec<bool> {
        self.stop.take(&self.exec)
    }
}

impl super::GpuQwen35 {
    /// Which shares are FUSED CKPT shares (their tail share of the same
    /// chunk follows immediately in the same tick), and which d_ckpt_stage
    /// blob each owns. Single source of truth: the launch-side per-layer
    /// stage copies and the finish-side staged snapshot both recompute this
    /// from the shares list (deterministic - adjacency + the 2-blob cap).
    pub(super) fn fused_stage_map(
        shares: &[(usize, usize, usize, usize, bool, Vec<u32>)],
    ) -> Vec<Option<usize>> {
        let mut map = vec![None; shares.len()];
        let mut stg = 0usize;
        for i in 0..shares.len().saturating_sub(1) {
            if shares[i].0 == shares[i + 1].0 && stg < 2 {
                map[i] = Some(stg);
                stg += 1;
            }
        }
        map
    }
}

/// P65: fill TruncCat rows' ids from the device top-64 head plane
/// (`head` = the dtoh'd [rows, 64, 2] u32 (id, raw-logit bits) plane;
/// `plans[i]` indexes row i). Sentinel id u32::MAX = n < 64 padding.
pub(in crate::gpu_model::qwen35) fn trunc_fill_ids(
    head: &[u32],
    plans: &[crate::generator::RowSample],
    ids: &mut [u32],
) {
    use crate::generator::RowSample;
    use crate::sampler::DevicePlan;
    // engagement witness (bisect-trap law): once per process
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let n = plans
            .iter()
            .filter(|p| matches!(p, RowSample::Device(DevicePlan::TruncCat { .. })))
            .count();
        eprintln!("[trunc-head] engaged: rows={n} (device top-64 + host head sampling)");
    });
    for (i, p) in plans.iter().enumerate() {
        if let RowSample::Device(DevicePlan::TruncCat {
            inv_t,
            u,
            k,
            top_p,
            min_p,
        }) = *p
        {
            let base = i * 128;
            let pairs: Vec<(u32, f32)> = (0..64usize)
                .filter_map(|j| {
                    let id = head[base + j * 2];
                    (id != u32::MAX).then(|| (id, f32::from_bits(head[base + j * 2 + 1])))
                })
                .collect();
            if !pairs.is_empty() {
                ids[i] = crate::sampler::sample_trunc_head(&pairs, inv_t, u, k, top_p, min_p);
            }
        }
    }
}

/// P65 small-b twin of the device prefilter: top-64 by f32 total order from
/// a full host row (identical head to pd_topk_rows modulo boundary-tie
/// choice, which was never contractual). Used at b <= 2 where the one-block
/// binary-search kernel is DRAM-latency-bound (c1 measured -13% before this
/// fallback existed).
pub(in crate::gpu_model::qwen35) fn host_top64(row: &[f32]) -> Vec<(u32, f32)> {
    let k = 64.min(row.len());
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, |&a, &b| row[b as usize].total_cmp(&row[a as usize]));
        idx.truncate(k);
    }
    idx.into_iter().map(|i| (i, row[i as usize])).collect()
}

/// Pack per-row sampler params exactly like `batch_sampled_impl` (Host/Hole
/// are skip - the pipe never runs with Host rows). Returns (par, tpar,
/// any_base, any5, any6): TruncCat rows pack mode 5 - the pipe only ever
/// receives them when the pack ships slot 435 (the service gates on
/// supports_device_trunc), so full-device is the only pipe form.
pub(in crate::gpu_model::qwen35) fn pack_samp_par(
    plans: &[crate::generator::RowSample],
) -> (Vec<u32>, Vec<u32>, bool, bool, bool) {
    use crate::generator::RowSample;
    use crate::sampler::DevicePlan;
    let mut par = vec![0u32; plans.len() * 4];
    let mut tpar = vec![0u32; plans.len() * 4];
    let (mut any_base, mut any5, mut any6) = (false, false, false);
    for (i, p) in plans.iter().enumerate() {
        match p {
            RowSample::Hole | RowSample::Host => {}
            RowSample::Device(DevicePlan::Greedy) => {
                par[i * 4 + 2] = 1;
                any_base = true;
            }
            RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                par[i * 4] = inv_t.to_bits();
                par[i * 4 + 1] = u.to_bits();
                par[i * 4 + 2] = 2;
                any_base = true;
            }
            // P67b: full-device truncation sampling in the pipe
            RowSample::Device(DevicePlan::TruncCat {
                inv_t,
                u,
                k,
                top_p,
                min_p,
            }) => {
                par[i * 4] = inv_t.to_bits();
                par[i * 4 + 1] = u.to_bits();
                par[i * 4 + 2] = if *k >= 1 && *k <= 64 { 5 } else { 6 };
                tpar[i * 4] = *k;
                tpar[i * 4 + 1] = top_p.to_bits();
                tpar[i * 4 + 2] = min_p.to_bits();
                if *k >= 1 && *k <= 64 {
                    any5 = true
                } else {
                    any6 = true
                }
            }
            // RS plans are gemma4-only (supports_spec_rs); skip-safe
            RowSample::Device(DevicePlan::RsVerify { .. })
            | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
        }
    }
    (par, tpar, any_base, any5, any6)
}
