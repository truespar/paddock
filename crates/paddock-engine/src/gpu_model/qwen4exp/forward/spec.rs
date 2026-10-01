//! Speculative verify for Flash-Next: one ragged walk over every live slot's
//! `[pending, drafts..]`, a pick per row, and an exact commit of the accepted
//! prefix. Two rounds share all of that and differ only in where a pick comes
//! from - `argmax_rows` for a pure-greedy cohort (`verify_round`), or
//! `sample_rows` with the service's pre-drawn per-row plans when any slot
//! samples (`verify_round_plans`, which is what lets temperature > 0
//! speculate at all).
//!
//! The walk is the prefill wave's (`Phase::PrefillRuns`, each chunk a run that
//! continues its slot) with the head over every row. It advances the carried
//! state through all of a chunk's rows, so a rejecting round puts back what
//! the rejected rows changed:
//! - GDN recurrence: restored from the copy taken before the walk, then
//!   re-advanced over the accepted rows from the q/k/v/g/beta the walk
//!   captured. The recurrence is token-serial, so that is the state an
//!   accepted-length walk leaves.
//! - GDN conv window: rebuilt as the last k-1 rows of [pre-round window ;
//!   accepted conv-input rows].
//! - PLE conv ring: the ring slots the rejected positions overwrote come back
//!   from the pre-round copy. A chunk never exceeds the ring, so no accepted
//!   position shares a slot with a rejected one.
//! - PLE n-gram stream: truncated.
//! - attention KV: nothing - cells past the accept are rewritten before any
//!   read, the argument every paged lane makes.
//!
//! A round that closes a page takes the reply checkpoint a decode tick takes
//! there (`reply_after_spec`), at the exact closing row: when the round ends
//! on the boundary the committed state is the checkpoint's, and when it runs
//! past it the state at the boundary is rebuilt from the same planes a
//! rollback reads (`reply_snapshot_mid`).
//!
//! Rows are decode-exact on the GGUF lane (`verify_exact_on`): every
//! row-count-sensitive op that lane runs - its `Kq` planes and attention -
//! takes the decode tick's own reduction for that row, and the replay
//! re-advances the recurrence through the tick's kernel too, so a spec stream
//! is the greedy stream bit for bit rather than to the last ulp (a near-tie
//! does not survive the last ulp - measured on real text, 2026-09-14).
//!
//! NOT on the safetensors MX/NVFP4 export: its bf16 hyper-connection planes,
//! MXFP8 dense planes and W4A4 experts pick kernels by row count (the hc down
//! is fused with its silu at one row and segmented above it, for one), and
//! `row_exact` does not reach them. Measured 2026-09-30: no verify row's
//! logits bit-identical to the tick's (0.014 off on the first row), so a
//! greedy spec stream parts from plain decode at a near tie - different
//! numerics of the same model, not a wrong answer. The target is
//! batch-invariant decode-band kernels (one reduction order per row at every
//! width); `verify_rows_carry_the_decode_logits` is the gate that says so.
//!
//! This is save-and-replay, not per-row state snapshots inside the recurrence
//! kernel (qwen35's `gated_delta_recurrent_snap` shape): one state copy per
//! layer per round instead of one per row, and the replay launches only when a
//! round rejects - or runs past a page boundary, for the reply checkpoint
//! there (at most the chunk's rows, once per 16 tokens of reply). The snapshot
//! kernel is the SOTA form and the next step if these copies show up in a
//! profile.

use super::*;
use crate::gpu::GpuError;

/// Widest chunk (pending token + drafts) a round verifies: the 9-row PLE ring.
pub(crate) const VERIFY_MAX_CHUNK: usize = 9;

/// Verify rows agree with the decode tick bit for bit - the default.
/// `PADDOCK_Q38FN_VERIFY_EXACT=0` keeps the batched kernel classes (A/B only).
pub(crate) fn verify_exact_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PADDOCK_Q38FN_VERIFY_EXACT").as_deref() != Ok("0"))
}

/// One GDN row staged at offset 0, for the row-exact walk and its replay: the
/// decode recurrence entry takes a slot vector and reads its operands from row
/// 0, and the walk's own planes must stay whole for the capture.
pub(crate) struct VerifyRows {
    pub(crate) q: CudaSlice<f32>,
    pub(crate) k: CudaSlice<f32>,
    pub(crate) v: CudaSlice<f32>,
    pub(crate) g: CudaSlice<f32>,
    pub(crate) b: CudaSlice<f32>,
    pub(crate) z: CudaSlice<f32>,
    pub(crate) core: CudaSlice<f32>,
    /// the un-normed recurrence output when the fused norm declines
    pub(crate) attn: CudaSlice<f32>,
    pub(crate) slot: CudaSlice<u32>,
    /// one-run table for the replay's runs walk (slot 599)
    pub(crate) run_off: CudaSlice<u32>,
    pub(crate) run_len: CudaSlice<u32>,
}

pub(crate) struct Verify {
    /// Set for the verify walk only: device_walk captures the GDN inputs and
    /// runs the head over every row.
    pub(crate) active: bool,
    rows_cap: usize,
    pub(crate) logits: CudaSlice<f32>,
    picks: CudaSlice<u32>,
    /// Per-row sampler planes for the DEVICE-SAMPLED round (`verify_round`
    /// with plans): `[rows, 4]` each, the same words the dense tick packs -
    /// `samp_par` = {inv_t bits, u bits, mode, pad} and `samp_tpar` =
    /// {k, top_p bits, min_p bits, pad} for the truncation modes. Allocated
    /// with the rest of the round's planes rather than per round: 4 u32 a row
    /// against a `[rows, vocab]` logits plane is nothing, and a round that
    /// allocates cannot be the hot path.
    samp_par: CudaSlice<u32>,
    samp_tpar: CudaSlice<u32>,
    // per GDN layer, [rows_cap, width]: what a rejecting commit replays
    cap_q: Vec<Option<CudaSlice<f32>>>,
    cap_k: Vec<Option<CudaSlice<f32>>>,
    cap_v: Vec<Option<CudaSlice<f32>>>,
    cap_g: Vec<Option<CudaSlice<f32>>>,
    cap_b: Vec<Option<CudaSlice<f32>>>,
    cap_qkv: Vec<Option<CudaSlice<f32>>>,
    // pre-round carried state, per slot region
    sh_state: Vec<Option<CudaSlice<f32>>>,
    sh_win: Vec<Option<CudaSlice<f32>>>,
    sh_ring: Option<CudaSlice<f32>>,
    /// window rebuild staging (the live window is source and destination)
    bounce: CudaSlice<f32>,
    /// one layer's recurrence re-advanced to a page boundary inside the
    /// round, for the reply checkpoint there (`reply_snapshot_mid`)
    snap_state: CudaSlice<f32>,
    pub(crate) rows: VerifyRows,
}

impl Verify {
    fn new(
        e: &GpuExecutor,
        c: &Qwen4ExpConfig,
        slots: usize,
        max_tokens: usize,
    ) -> Result<Self, GpuModelError> {
        let rows = (slots * VERIFY_MAX_CHUNK).min(max_tokens);
        let hv = c.gdn_v_heads;
        let (kdim, vdim, qr) = (hv * c.gdn_k_dim, hv * c.gdn_v_dim, c.gdn_qkv_rows());
        let st = hv * c.gdn_k_dim * c.gdn_v_dim;
        let wl = (c.gdn_conv - 1) * qr;
        let per = |len: usize, gdn: bool| -> Result<Option<CudaSlice<f32>>, GpuError> {
            if gdn {
                e.alloc(len).map(Some)
            } else {
                Ok(None)
            }
        };
        let mut v = Verify {
            active: false,
            rows_cap: rows,
            logits: e.alloc(rows * c.vocab)?,
            picks: e.alloc_u32(rows)?,
            samp_par: e.alloc_u32(rows * 4)?,
            samp_tpar: e.alloc_u32(rows * 4)?,
            cap_q: Vec::with_capacity(c.n_layer),
            cap_k: Vec::with_capacity(c.n_layer),
            cap_v: Vec::with_capacity(c.n_layer),
            cap_g: Vec::with_capacity(c.n_layer),
            cap_b: Vec::with_capacity(c.n_layer),
            cap_qkv: Vec::with_capacity(c.n_layer),
            sh_state: Vec::with_capacity(c.n_layer),
            sh_win: Vec::with_capacity(c.n_layer),
            sh_ring: if c.ple_layers.is_empty() {
                None
            } else {
                Some(e.alloc(slots * (c.ple_conv - 1) * PLE_DILATION * c.hc_width())?)
            },
            bounce: e.alloc(wl)?,
            snap_state: e.alloc(st)?,
            rows: VerifyRows {
                q: e.alloc(kdim)?,
                k: e.alloc(kdim)?,
                v: e.alloc(vdim)?,
                g: e.alloc(hv)?,
                b: e.alloc(hv)?,
                z: e.alloc(c.gdn_z_rows())?,
                core: e.alloc(vdim)?,
                attn: e.alloc(kdim.max(vdim))?,
                slot: e.alloc_u32(1)?,
                run_off: e.alloc_u32(1)?,
                run_len: e.alloc_u32(1)?,
            },
        };
        for li in 0..c.n_layer {
            let g = c.blocks[li] == Qwen4ExpBlock::Gdn;
            v.cap_q.push(per(rows * kdim, g)?);
            v.cap_k.push(per(rows * kdim, g)?);
            v.cap_v.push(per(rows * vdim, g)?);
            v.cap_g.push(per(rows * hv, g)?);
            v.cap_b.push(per(rows * hv, g)?);
            v.cap_qkv.push(per(rows * qr, g)?);
            v.sh_state.push(per(slots * st, g)?);
            v.sh_win.push(per(slots * wl, g)?);
        }
        Ok(v)
    }

    /// Called by device_walk right after GDN layer `li` of a verify walk.
    pub(crate) fn capture_gdn(
        &mut self,
        e: &GpuExecutor,
        c: &Qwen4ExpConfig,
        sc: &Scratch,
        li: usize,
        n: usize,
    ) -> Result<(), GpuModelError> {
        let hv = c.gdn_v_heads;
        let (kdim, vdim, qr) = (hv * c.gdn_k_dim, hv * c.gdn_v_dim, c.gdn_qkv_rows());
        e.copy_region(&sc.d_dq, 0, gdn_plane_mut(&mut self.cap_q, li), 0, n * kdim)?;
        e.copy_region(&sc.d_dk, 0, gdn_plane_mut(&mut self.cap_k, li), 0, n * kdim)?;
        e.copy_region(&sc.d_dv, 0, gdn_plane_mut(&mut self.cap_v, li), 0, n * vdim)?;
        e.copy_region(&sc.d_g, 0, gdn_plane_mut(&mut self.cap_g, li), 0, n * hv)?;
        e.copy_region(&sc.d_beta, 0, gdn_plane_mut(&mut self.cap_b, li), 0, n * hv)?;
        e.copy_region(
            &sc.d_qkv,
            0,
            gdn_plane_mut(&mut self.cap_qkv, li),
            0,
            n * qr,
        )?;
        Ok(())
    }
}

/// A per-layer plane that exists on GDN layers only.
fn gdn_plane(v: &[Option<CudaSlice<f32>>], li: usize) -> &CudaSlice<f32> {
    v[li].as_ref().expect("GDN layer verify plane")
}

fn gdn_plane_mut(v: &mut [Option<CudaSlice<f32>>], li: usize) -> &mut CudaSlice<f32> {
    v[li].as_mut().expect("GDN layer verify plane")
}

impl Qwen4ExpGpu {
    /// The greedy verify round (`Generator::forward_spec_batch`): returns the
    /// flat per-row picks, with every slot committed to its accepted prefix.
    /// `Ok(None)` declines (a chunk wider than the ring, or past the rows the
    /// planes hold).
    pub(super) fn verify_round(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        self.verify_round_impl(reqs, None)
    }

    /// The DEVICE-SAMPLED verify round (`Generator::forward_spec_batch_plans`).
    ///
    /// Same walk, same commit; the only difference is where a row's pick comes
    /// from - `sample_rows` with the service's pre-drawn per-row plan instead
    /// of `argmax_rows`. That is what lets a sampling request speculate at
    /// all: the greedy round demands `is_pure_greedy()`, so before this existed
    /// every temperature > 0 serve declined here (the Generator default is
    /// `Ok(None)`) and fell back to the dense tick. Measured on the standing
    /// aiperf scenarios, which all send temperature 0.7: 0 tokens drafted, and
    /// the attached head's 2.74 GiB came out of the headroom the MoE expert
    /// cache sizes from - it cost 2-3% and bought nothing (2026-09-17).
    ///
    /// **Why it stays exact.** The head drafts greedily (`mtp_draft` argmaxes),
    /// so a chunk's drafts are a deterministic function of the accepted
    /// prefix. For deterministic drafts, "sample every verify row with that
    /// row's own plan, accept while the sample equals the next draft, and emit
    /// the first mismatching row's sample" draws from exactly the target
    /// distribution - the standard rejection-sampling argument with q a point
    /// mass, and the same rule the dense tick would have drawn with the same
    /// uniform. The accept walk below is unchanged and re-derives the prefix
    /// from these picks, so the committed state cannot disagree with the
    /// tokens the service streams.
    ///
    /// Rejection-sampling plans (`RsVerify`/`RsTrunc`) are declined: the row
    /// sampler skips them (mode 0) and their resolve kernel is the drafter's
    /// own K-candidate machinery, which this family does not record. They
    /// cannot reach us today (the service only draws them for backends that
    /// answer `supports_spec_rs*`), and declining beats leaving a pick plane
    /// untouched.
    ///
    /// Still greedy-only: the mixed tick (`forward_mixed_spec_plans`, a
    /// prefill chunk riding with spec rows). Without it a mixed tick takes the
    /// dense route, which costs a round rather than correctness.
    pub(super) fn verify_round_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        self.verify_round_impl(reqs, Some(plans))
    }

    fn verify_round_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: Option<&[crate::sampler::DevicePlan]>,
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        let timing = std::env::var_os("PADDOCK_Q38FN_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let Some(runs) = self.verify_walk(reqs, plans)? else {
            return Ok(None);
        };
        let total: usize = runs.iter().map(|r| r.len).sum();
        let picks: Vec<u32> = {
            let Self {
                exec, verify, cfg, ..
            } = self;
            let vf: &mut Verify = verify.as_mut().expect("built");
            match plans {
                None => exec.argmax_rows(&vf.logits, &mut vf.picks, total, cfg.vocab)?,
                Some(plans) => {
                    // The dense tick's packer is the one place the modes are
                    // spelled, and a verify row is an ordinary sampled row -
                    // so the two paths cannot drift apart on what a plan means.
                    let rows: Vec<crate::generator::RowSample> = plans
                        .iter()
                        .map(|&p| crate::generator::RowSample::Device(p))
                        .collect();
                    let (par, tpar) = Self::pack_samp_par(&rows);
                    {
                        let mut v = vf
                            .samp_par
                            .try_slice_mut(0..total * 4)
                            .ok_or_else(|| GpuError::Driver("verify samp_par slice".into()))?;
                        exec.stream
                            .memcpy_htod(&par, &mut v)
                            .map_err(crate::gpu::from_driver)?;
                    }
                    if let Some(t) = &tpar {
                        let mut v = vf
                            .samp_tpar
                            .try_slice_mut(0..total * 4)
                            .ok_or_else(|| GpuError::Driver("verify samp_tpar slice".into()))?;
                        exec.stream
                            .memcpy_htod(t, &mut v)
                            .map_err(crate::gpu::from_driver)?;
                    }
                    exec.sample_rows(&vf.logits, &vf.samp_par, &mut vf.picks, total, cfg.vocab)?;
                    if tpar.is_some() {
                        exec.sample_rows_t(
                            &vf.logits,
                            &vf.samp_par,
                            &vf.samp_tpar,
                            &mut vf.picks,
                            total,
                            cfg.vocab,
                        )?;
                        exec.sample_rows_p(
                            &vf.logits,
                            &vf.samp_par,
                            &vf.samp_tpar,
                            &mut vf.picks,
                            total,
                            cfg.vocab,
                        )?;
                    }
                    // Engagement witness (the bisect-trap law): a path that
                    // silently never runs is what this whole lane just cost a
                    // session to find. Once per process, naming the modes.
                    static ENGAGED: std::sync::Once = std::sync::Once::new();
                    ENGAGED.call_once(|| {
                        eprintln!(
                            "[q4x-spec-plans] device-sampled verify engaged: {total} rows{}",
                            if tpar.is_some() {
                                " (truncation rows present)"
                            } else {
                                ""
                            }
                        );
                    });
                }
            }
            let view = vf
                .picks
                .try_slice(0..total)
                .ok_or_else(|| GpuError::Driver("verify picks view".into()))?;
            exec.stream
                .clone_dtoh(&view)
                .map_err(crate::gpu::from_driver)?
        };
        let d_walk = t0.elapsed();
        // the service's accept walk, re-derived so the state commit can never
        // disagree with the tokens it streams
        let counts: Vec<usize> = runs
            .iter()
            .zip(reqs)
            .map(|(r, (_, _, chunk))| {
                let mut a = 0usize;
                while a + 1 < chunk.len() && chunk[a + 1] == picks[r.off + a] {
                    a += 1;
                }
                a + 1
            })
            .collect();
        self.verify_commit(&runs, &counts)?;
        if timing {
            self.exec.synchronize()?;
            eprintln!(
                "[spec-verify] rows {total} committed {} | walk+picks {:7.2} ms commit {:7.2} ms",
                counts.iter().sum::<usize>(),
                d_walk.as_secs_f64() * 1e3,
                (t0.elapsed() - d_walk).as_secs_f64() * 1e3
            );
        }
        Ok(Some(picks))
    }

    /// The HOST-SAMPLED verify round, phase 1 (`Generator::forward_spec_verify`):
    /// the same walk, then the raw row logits ([rows, vocab], request order)
    /// for the service to sample - each row through the slot's own sampler
    /// and, for a constrained slot, through its grammar machine. That is the
    /// round every tool-carrying request speculates through (a constraint
    /// keeps a slot out of the greedy and device rounds, whose acceptance
    /// resolves where no machine can sit); without it every one of them
    /// declined here into the service's cooldown and decoded dense - Claude
    /// Code's requests all carry tools (measured on its replayed requests:
    /// the MTP head bought 25.1 vs 25.0 tok/s). The round stays open until
    /// [`Self::verify_close`] commits what the service accepted.
    pub(super) fn verify_open(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<f32>>, GpuModelError> {
        if self.spec_open.take().is_some() {
            // the service closes every round it opens; one left open means
            // its commit never came - refuse rather than walk on top of it
            return Err(GpuModelError::Unsupported(
                "verify: the previous sampled round was never committed".into(),
            ));
        }
        let Some(runs) = self.verify_walk(reqs, None)? else {
            return Ok(None);
        };
        let total: usize = runs.iter().map(|r| r.len).sum();
        let logits = self.exec.to_host_len(
            &self.verify.as_ref().expect("built").logits,
            total * self.cfg.vocab,
        )?;
        self.spec_open = Some(runs);
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            eprintln!("[q4x-spec-sampled] host-sampled verify engaged: {total} rows");
        });
        Ok(Some(logits))
    }

    /// The host-sampled round, phase 2 (`Generator::spec_commit`): commit
    /// `committed[i]` rows of request i of the open round - the accepted
    /// drafts plus the row whose sample replaced the first mismatch.
    pub(super) fn verify_close(&mut self, committed: &[u32]) -> Result<(), GpuModelError> {
        let runs = self.spec_open.take().ok_or_else(|| {
            GpuModelError::Unsupported("spec_commit without an open verify round".into())
        })?;
        if committed.len() != runs.len() {
            return Err(GpuModelError::Unsupported(format!(
                "spec_commit: {} counts for {} requests",
                committed.len(),
                runs.len()
            )));
        }
        let counts: Vec<usize> = committed.iter().map(|&c| c as usize).collect();
        for (r, &c) in runs.iter().zip(&counts) {
            if c == 0 || c > r.len {
                return Err(GpuModelError::Unsupported(format!(
                    "spec_commit: slot {} commits {c} of {} rows",
                    r.slot, r.len
                )));
            }
        }
        self.verify_commit(&runs, &counts)
    }

    /// Preflight, save the slots' carried state, and walk the ragged chunks
    /// as one `PrefillRuns` walk that captures each row's logits and the GDN
    /// inputs a rollback replays. `Ok(None)` declines (nothing touched).
    fn verify_walk(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: Option<&[crate::sampler::DevicePlan]>,
    ) -> Result<Option<Vec<Run>>, GpuModelError> {
        if reqs.is_empty() {
            return Ok(None);
        }
        let total: usize = reqs.iter().map(|r| r.2.len()).sum();
        let mut seen = vec![false; self.slots];
        for (slot, pos, chunk) in reqs {
            let (slot, pos) = (*slot, *pos);
            if slot >= self.slots || seen[slot] {
                return Err(GpuModelError::Unsupported(format!(
                    "verify: slot {slot} out of range or repeated"
                )));
            }
            seen[slot] = true;
            if pos == 0 || self.pos[slot] != pos || self.stream[slot].len() != pos + 2 {
                return Err(GpuModelError::Unsupported(format!(
                    "verify: slot {slot}: scheduler says position {pos}, model is at {} \
                     (stream {})",
                    self.pos[slot],
                    self.stream[slot].len()
                )));
            }
            if chunk.is_empty()
                || chunk.len() > VERIFY_MAX_CHUNK
                || pos + chunk.len() > self.max_tokens
            {
                return Ok(None);
            }
        }
        // Plans preflight, before the state copies: one plan per verify row in
        // request order (the service's flat layout), and every mode this round
        // will spell has to have a kernel in the loaded pack. A pack without
        // the truncation samplers is a decline, not an error - the round falls
        // back to the dense tick and the serve keeps going.
        if let Some(plans) = plans {
            if plans.len() != total {
                return Err(GpuModelError::Unsupported(format!(
                    "verify: {} plans for {total} verify rows",
                    plans.len()
                )));
            }
            if !self.exec.has_sample_rows() {
                return Ok(None);
            }
            let mut trunc = false;
            for p in plans {
                match p {
                    crate::sampler::DevicePlan::Greedy
                    | crate::sampler::DevicePlan::Categorical { .. } => {}
                    crate::sampler::DevicePlan::TruncCat { .. } => trunc = true,
                    // see the doc comment: mode-0 rows would keep whatever the
                    // pick plane held
                    crate::sampler::DevicePlan::RsVerify { .. }
                    | crate::sampler::DevicePlan::RsTrunc { .. } => return Ok(None),
                }
            }
            if trunc && !(self.exec.has_sample_rows_t() && self.exec.has_sample_rows_p()) {
                return Ok(None);
            }
        }
        if self.verify.is_none() {
            let v = Verify::new(&self.exec, &self.cfg, self.slots, self.max_tokens)?;
            self.verify = Some(Box::new(v));
        }
        if total > self.verify.as_ref().expect("built").rows_cap {
            return Ok(None);
        }
        for (slot, _, _) in reqs {
            self.verify_save(*slot)?;
        }
        let mut runs = Vec::with_capacity(reqs.len());
        let mut ids = Vec::with_capacity(total);
        for (slot, pos, chunk) in reqs {
            self.stream[*slot].extend(chunk.iter().map(|&t| t as i64));
            runs.push(Run {
                slot: *slot,
                off: ids.len(),
                len: chunk.len(),
                row0: *pos,
            });
            ids.extend_from_slice(chunk);
        }
        self.cur_slots = runs
            .iter()
            .flat_map(|r| std::iter::repeat_n(r.slot, r.len))
            .collect();
        self.stage_inputs_runs_ids(&ids, &runs, 0)?;
        self.cur_runs = runs.clone();
        self.walk_qsa = self.qsa_for_runs(&runs);
        self.verify.as_mut().expect("built").active = true;
        let walked = self.device_walk(total, Phase::PrefillRuns);
        self.verify.as_mut().expect("built").active = false;
        self.cur_runs.clear();
        walked?;
        Ok(Some(runs))
    }

    /// Commit `counts[i]` rows of verify run i: the drafter catches up on
    /// them, and a run that did not keep every row rolls its carried state
    /// back to the state after them.
    fn verify_commit(&mut self, runs: &[Run], counts: &[usize]) -> Result<(), GpuModelError> {
        for (r, &c) in runs.iter().zip(counts) {
            self.mtp_note_verify(r.slot, r.off, r.row0, c)?;
            if c < r.len {
                self.verify_rollback(r.slot, r.off, r.row0, c, r.len)?;
            }
            self.pos[r.slot] = r.row0 + c;
        }
        self.mtp_flush()?;
        self.reply_after_spec(runs, counts)
    }

    /// A round committed `counts[i]` rows of run i: the reply checkpoint a
    /// decode tick takes at every page it closes, for the page this round
    /// closed (a round commits at most `VERIFY_MAX_CHUNK` rows, so one page
    /// at most). When the round ends on the boundary the committed state is
    /// the checkpoint's; when it runs past it, the state at the boundary is
    /// rebuilt from the round's own planes. Without this a speculating reply
    /// checkpointed only on its non-spec ticks - with MTP on, next to never -
    /// and the next turn re-prefilled the whole reply.
    fn reply_after_spec(&mut self, runs: &[Run], counts: &[usize]) -> Result<(), GpuModelError> {
        const { assert!(VERIFY_MAX_CHUNK <= BLOCK_TOKENS) };
        for (r, &c) in runs.iter().zip(counts) {
            self.qsa_dense_guard(r.slot);
            if !self.reply_track[r.slot] {
                continue;
            }
            let cut = (r.row0 + c) / BLOCK_TOKENS * BLOCK_TOKENS;
            if cut <= r.row0 {
                continue;
            }
            if cut == r.row0 + c {
                self.reply_snapshot(r.slot)?;
            } else {
                self.reply_snapshot_mid(r, cut)?;
            }
        }
        Ok(())
    }

    /// The reply checkpoint at `cut`, a page boundary run `r` committed
    /// `cut - r.row0` rows into and then passed. The state there is rebuilt
    /// in the prefix cache's staging blob from what the round keeps for a
    /// rollback, and committed into a reserved checkpoint's pages: each GDN
    /// recurrence is the pre-round copy re-advanced over those rows through
    /// the rollback's decode-exact replay, each conv window the last k-1 rows
    /// of [pre-round window ; those rows' conv inputs], and the PLE ring the
    /// live one with the slots of the positions past `cut` taken back from
    /// the pre-round copy (a chunk never exceeds the ring, so every such slot
    /// held a position before the round). The live state is not touched.
    fn reply_snapshot_mid(&mut self, r: &Run, cut: usize) -> Result<(), GpuModelError> {
        let (slot, pos0, off) = (r.slot, r.row0, r.off);
        let n = cut - pos0;
        if self.stream[slot].len() < cut + 2 {
            self.reply_track[slot] = false;
            return Ok(());
        }
        {
            let Self {
                exec: e,
                cfg: c,
                ple_win,
                sc,
                verify,
                prefix,
                ..
            } = self;
            let Some(pc) = prefix.as_mut() else {
                return Ok(());
            };
            let vf: &mut Verify = verify.as_mut().expect("a round just ran");
            let sink = pc.ckpt_sink();
            let (hv, kd) = (c.gdn_v_heads, c.gdn_k_dim);
            let (kdim, vdim) = (hv * kd, hv * c.gdn_v_dim);
            let st = hv * kd * c.gdn_v_dim;
            let (qr, km1) = (c.gdn_qkv_rows(), c.gdn_conv - 1);
            let wl = km1 * qr;
            let keep_old = km1.saturating_sub(n);
            let take_new = km1 - keep_old;
            if verify_exact_on() {
                // the replay's one-run table: this run's first n rows, onto
                // the scratch plane's only slot
                e.upload_u32(&[0], &mut vf.rows.slot)?;
                e.upload_u32(&[off as u32], &mut vf.rows.run_off)?;
                e.upload_u32(&[n as u32], &mut vf.rows.run_len)?;
            }
            let mut ord = 0usize;
            for li in 0..c.n_layer {
                if vf.sh_state[li].is_none() {
                    continue;
                }
                e.copy_region(
                    gdn_plane(&vf.sh_state, li),
                    slot * st,
                    &mut vf.snap_state,
                    0,
                    st,
                )?;
                if verify_exact_on() {
                    let walked = e.gated_delta_recurrent_runs_slots(
                        gdn_plane(&vf.cap_q, li),
                        gdn_plane(&vf.cap_k, li),
                        gdn_plane(&vf.cap_v, li),
                        gdn_plane(&vf.cap_g, li),
                        gdn_plane(&vf.cap_b, li),
                        &mut vf.snap_state,
                        &mut sc.d_dattn,
                        &vf.rows.run_off,
                        &vf.rows.run_len,
                        &vf.rows.slot,
                        None,
                        1,
                        hv,
                        kd,
                    )?;
                    for t in 0..if walked { 0 } else { n } {
                        let rw = &mut vf.rows;
                        let row = off + t;
                        e.copy_region(gdn_plane(&vf.cap_q, li), row * kdim, &mut rw.q, 0, kdim)?;
                        e.copy_region(gdn_plane(&vf.cap_k, li), row * kdim, &mut rw.k, 0, kdim)?;
                        e.copy_region(gdn_plane(&vf.cap_v, li), row * vdim, &mut rw.v, 0, vdim)?;
                        e.copy_region(gdn_plane(&vf.cap_g, li), row * hv, &mut rw.g, 0, hv)?;
                        e.copy_region(gdn_plane(&vf.cap_b, li), row * hv, &mut rw.b, 0, hv)?;
                        e.gated_delta_recurrent_slots(
                            &rw.q,
                            &rw.k,
                            &rw.v,
                            &rw.g,
                            &rw.b,
                            &rw.slot,
                            &mut vf.snap_state,
                            &mut rw.attn,
                            1,
                            hv,
                            kd,
                        )?;
                    }
                } else {
                    let plane = |v| gdn_plane(v, li);
                    e.copy_region(plane(&vf.cap_q), off * kdim, &mut sc.d_dq, 0, n * kdim)?;
                    e.copy_region(plane(&vf.cap_k), off * kdim, &mut sc.d_dk, 0, n * kdim)?;
                    e.copy_region(plane(&vf.cap_v), off * vdim, &mut sc.d_dv, 0, n * vdim)?;
                    e.copy_region(plane(&vf.cap_g), off * hv, &mut sc.d_g, 0, n * hv)?;
                    e.copy_region(plane(&vf.cap_b), off * hv, &mut sc.d_beta, 0, n * hv)?;
                    e.gated_delta_recurrent_at(
                        &sc.d_dq,
                        &sc.d_dk,
                        &sc.d_dv,
                        &sc.d_g,
                        &sc.d_beta,
                        &mut vf.snap_state,
                        0,
                        &mut sc.d_dattn,
                        n,
                        hv,
                        kd,
                    )?;
                }
                let so = sink.state_off(0, ord);
                e.copy_region(&vf.snap_state, 0, &mut *sink.pool, so, st)?;
                let wo = sink.win_off(0, ord);
                for j in 0..keep_old {
                    e.copy_region(
                        gdn_plane(&vf.sh_win, li),
                        slot * wl + (n + j) * qr,
                        &mut *sink.pool,
                        wo + j * qr,
                        qr,
                    )?;
                }
                e.copy_region(
                    gdn_plane(&vf.cap_qkv, li),
                    (off + n - take_new) * qr,
                    &mut *sink.pool,
                    wo + keep_old * qr,
                    take_new * qr,
                )?;
                ord += 1;
            }
            if let (Some(ring), Some(sh)) = (ple_win.as_ref(), vf.sh_ring.as_ref()) {
                let hw = c.hc_width();
                let wrows = (c.ple_conv - 1) * PLE_DILATION;
                let pbase = slot * wrows * hw;
                let po = sink.ple_off(0, ord);
                e.copy_region(ring, pbase, &mut *sink.pool, po, wrows * hw)?;
                for q in cut..pos0 + r.len {
                    let ri = q % wrows;
                    e.copy_region(sh, pbase + ri * hw, &mut *sink.pool, po + ri * hw, hw)?;
                }
            }
        }
        let tokens: Vec<u32> = self.stream[slot][2..2 + cut]
            .iter()
            .map(|&t| t as u32)
            .collect();
        let Some(pc) = self.prefix.as_mut() else {
            return Ok(());
        };
        let Some(idx) = pc.reserve_ckpt(&mut self.pages) else {
            return Ok(());
        };
        if let Err(err) = pc.commit_staged(&self.exec, &self.pages, 0, idx) {
            pc.recycle_ckpt(idx, &mut self.pages);
            return Err(err);
        }
        // the reply's pages up to the cut under the radix, then the
        // checkpoint on the node that ends there
        if let Err(err) = self.prefix_publish(slot, &tokens, cut, false) {
            self.recycle_reserved(&[(cut, idx)]);
            return Err(err);
        }
        let pc = self.prefix.as_mut().expect("checked above");
        if pc.attach_reply(&tokens, cut, idx, &mut self.pages) {
            self.reply_filed(slot, &tokens, cut, idx);
        }
        Ok(())
    }

    /// Copy `slot`'s carried state before a verify walk moves it.
    fn verify_save(&mut self, slot: usize) -> Result<(), GpuModelError> {
        let Self {
            exec: e,
            cfg: c,
            recur,
            gdn_win,
            ple_win,
            verify,
            ..
        } = self;
        let vf: &mut Verify = verify.as_mut().expect("built");
        let st = c.gdn_v_heads * c.gdn_k_dim * c.gdn_v_dim;
        let wl = (c.gdn_conv - 1) * c.gdn_qkv_rows();
        for li in 0..c.n_layer {
            if let (Some(s), Some(d)) = (recur[li].as_ref(), vf.sh_state[li].as_mut()) {
                e.copy_region(s, slot * st, d, slot * st, st)?;
            }
            if let (Some(s), Some(d)) = (gdn_win[li].as_ref(), vf.sh_win[li].as_mut()) {
                e.copy_region(s, slot * wl, d, slot * wl, wl)?;
            }
        }
        if let (Some(s), Some(d)) = (ple_win.as_ref(), vf.sh_ring.as_mut()) {
            let pl = (c.ple_conv - 1) * PLE_DILATION * c.hc_width();
            e.copy_region(s, slot * pl, d, slot * pl, pl)?;
        }
        Ok(())
    }

    /// Put `slot` back to the state after `acc` of the `len` rows it walked
    /// from `pos` (run rows at `off` of the capture planes).
    fn verify_rollback(
        &mut self,
        slot: usize,
        off: usize,
        pos: usize,
        acc: usize,
        len: usize,
    ) -> Result<(), GpuModelError> {
        let Self {
            exec: e,
            cfg: c,
            recur,
            gdn_win,
            ple_win,
            sc,
            verify,
            stream,
            ..
        } = self;
        let vf: &mut Verify = verify.as_mut().expect("built");
        let (hv, kd) = (c.gdn_v_heads, c.gdn_k_dim);
        let (kdim, vdim) = (hv * kd, hv * c.gdn_v_dim);
        let st = hv * kd * c.gdn_v_dim;
        let (qr, km1) = (c.gdn_qkv_rows(), c.gdn_conv - 1);
        let wl = km1 * qr;
        // pre-round rows the window keeps, and accepted rows it takes
        let keep_old = km1.saturating_sub(acc);
        let take_new = km1 - keep_old;
        for li in 0..c.n_layer {
            let (Some(state), Some(win)) = (recur[li].as_mut(), gdn_win[li].as_mut()) else {
                continue;
            };
            let plane = |v| gdn_plane(v, li);
            e.copy_region(plane(&vf.sh_state), slot * st, state, slot * st, st)?;
            if verify_exact_on() {
                // the walk advanced each row through the decode tick's entry,
                // so the replay does too: the state left is the one `acc`
                // decode ticks leave, bit for bit
                e.upload_u32(&[slot as u32], &mut vf.rows.slot)?;
                e.upload_u32(&[off as u32], &mut vf.rows.run_off)?;
                e.upload_u32(&[acc as u32], &mut vf.rows.run_len)?;
                // one launch over the accepted rows, straight off the capture
                // planes (slot 599, no norm: only the state is kept); the
                // per-row replay below is its fallback
                let walked = e.gated_delta_recurrent_runs_slots(
                    gdn_plane(&vf.cap_q, li),
                    gdn_plane(&vf.cap_k, li),
                    gdn_plane(&vf.cap_v, li),
                    gdn_plane(&vf.cap_g, li),
                    gdn_plane(&vf.cap_b, li),
                    state,
                    &mut sc.d_dattn,
                    &vf.rows.run_off,
                    &vf.rows.run_len,
                    &vf.rows.slot,
                    None,
                    1,
                    hv,
                    kd,
                )?;
                for t in 0..if walked { 0 } else { acc } {
                    let rw = &mut vf.rows;
                    e.copy_region(
                        gdn_plane(&vf.cap_q, li),
                        (off + t) * kdim,
                        &mut rw.q,
                        0,
                        kdim,
                    )?;
                    e.copy_region(
                        gdn_plane(&vf.cap_k, li),
                        (off + t) * kdim,
                        &mut rw.k,
                        0,
                        kdim,
                    )?;
                    e.copy_region(
                        gdn_plane(&vf.cap_v, li),
                        (off + t) * vdim,
                        &mut rw.v,
                        0,
                        vdim,
                    )?;
                    e.copy_region(gdn_plane(&vf.cap_g, li), (off + t) * hv, &mut rw.g, 0, hv)?;
                    e.copy_region(gdn_plane(&vf.cap_b, li), (off + t) * hv, &mut rw.b, 0, hv)?;
                    e.gated_delta_recurrent_slots(
                        &rw.q,
                        &rw.k,
                        &rw.v,
                        &rw.g,
                        &rw.b,
                        &rw.slot,
                        state,
                        &mut rw.attn,
                        1,
                        hv,
                        kd,
                    )?;
                }
            } else {
                e.copy_region(plane(&vf.cap_q), off * kdim, &mut sc.d_dq, 0, acc * kdim)?;
                e.copy_region(plane(&vf.cap_k), off * kdim, &mut sc.d_dk, 0, acc * kdim)?;
                e.copy_region(plane(&vf.cap_v), off * vdim, &mut sc.d_dv, 0, acc * vdim)?;
                e.copy_region(plane(&vf.cap_g), off * hv, &mut sc.d_g, 0, acc * hv)?;
                e.copy_region(plane(&vf.cap_b), off * hv, &mut sc.d_beta, 0, acc * hv)?;
                e.gated_delta_recurrent_at(
                    &sc.d_dq,
                    &sc.d_dk,
                    &sc.d_dv,
                    &sc.d_g,
                    &sc.d_beta,
                    state,
                    slot * hv * kd * kd,
                    &mut sc.d_dattn,
                    acc,
                    hv,
                    kd,
                )?;
            }
            for j in 0..keep_old {
                e.copy_region(
                    plane(&vf.sh_win),
                    slot * wl + (acc + j) * qr,
                    &mut vf.bounce,
                    j * qr,
                    qr,
                )?;
            }
            e.copy_region(
                plane(&vf.cap_qkv),
                (off + acc - take_new) * qr,
                &mut vf.bounce,
                keep_old * qr,
                take_new * qr,
            )?;
            e.copy_region(&vf.bounce, 0, win, slot * wl, wl)?;
        }
        if let (Some(ring), Some(sh)) = (ple_win.as_mut(), vf.sh_ring.as_ref()) {
            let hw = c.hc_width();
            let wrows = (c.ple_conv - 1) * PLE_DILATION;
            let pbase = slot * wrows * hw;
            for q in pos + acc..pos + len {
                let ri = q % wrows;
                e.copy_region(sh, pbase + ri * hw, ring, pbase + ri * hw, hw)?;
            }
        }
        stream[slot].truncate(2 + pos + acc);
        Ok(())
    }
}
