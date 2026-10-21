//! The pipelined pure-decode lane: `decode_pipe_begin` enqueues tick 0 and
//! each `decode_pipe_next` enqueues tick t+1 before returning tick t's
//! device-sampled ids (one replay of the shared step graph a tick, inputs
//! advanced on device from the previous out ring), `decode_pipe_drain` ends
//! it, `pipe_abort` kills it after an error.

use super::*;

impl GpuQwen35 {
    /// Capture the fixed-B batched decode step into a replayable graph, cached
    /// per batch size. Shared by the classic single-tick path and the pipelined
    /// path - both replay the same graph, so the pipe is bit-identical to
    /// `forward_batch_sampled`. Grid-stable for a given B (loop bounds like KV
    /// length come from the device position buffer), so one capture replays at
    /// every position.
    pub(in crate::gpu_model::qwen35) fn ensure_batch_graph(
        &mut self,
        b: usize,
    ) -> Result<(), GpuModelError> {
        if self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .graphs
            .contains_key(&b)
        {
            return Ok(());
        }
        if b > f8t_dec_bmax() {
            tracing::warn!(
                "ensure_batch_graph TRIPWIRE b={b} caller:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
        }
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        // capture against the DEDICATED decode arena (see enable_batch): the
        // graph must not bake the shared prefill scratch's addresses
        let saved = self.scratch.take();
        self.scratch = self.decode_arena.take();
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        let rec = self.record_batch_step(b);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        self.decode_arena = self.scratch.take();
        self.scratch = saved;
        rec?; // surface a record failure only after capture is cleanly ended
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("batch capture produced no graph".into()))?;
        self.batch
            .as_mut()
            .expect("batch enabled")
            .graphs
            .insert(b, SendGraph(graph));
        Ok(())
    }

    /// `launch_batch_step` with an optional explicit slot mapping. `None` = the
    /// identity mapping (row i drives slot i - the pure batched-decode path).
    /// `Some(slots)` maps row i to `slots[i]`: the batched-step kernels already
    /// take `bs.d_slots` for per-slot KV/state/pool indexing, so a mixed tick's
    /// compacted decode rows (arbitrary slots) ride the same captured graph - we
    /// just write the real slots into `bs.d_slots` before the replay and restore
    /// identity after (so a following pure-decode tick is unaffected).
    pub(super) fn launch_batch_step_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
    ) -> Result<(), GpuModelError> {
        self.with_decode_lane(|m| m.launch_batch_step_slots_inner(tokens, positions, slots))
    }

    pub(super) fn launch_batch_step_slots_inner(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
    ) -> Result<(), GpuModelError> {
        let b = tokens.len();
        assert_eq!(b, positions.len());
        if let Some(s) = slots {
            assert_eq!(s.len(), b, "one slot per row");
        }
        assert!(self.batch.is_some(), "enable_batch first");
        assert!(b >= 1 && b <= self.batch.as_ref().expect("batch enabled").max_batch);
        self.ensure_scratch(b)?;
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        {
            let bs = self.batch.as_mut().expect("batch enabled");
            // mrope delta is per-SLOT: identity indexes by row i, slot-explicit
            // by the row's actual slot (text sequences carry delta 0 either way).
            let mrope_host: Vec<u32> = (0..4)
                .flat_map(|_| {
                    positions.iter().enumerate().map(|(i, &p)| {
                        let d = match slots {
                            Some(s) => bs.mrope_delta[s[i] as usize],
                            None => bs.mrope_delta[i],
                        };
                        (p as i64 + d) as u32
                    })
                })
                .collect();
            exec.stream
                .memcpy_htod(tokens, &mut bs.d_tokens)
                .map_err(drv)?;
            exec.stream
                .memcpy_htod(positions, &mut bs.d_pos)
                .map_err(drv)?;
            if let Some(s) = slots {
                let mut v = bs.d_slots.slice_mut(0..b);
                exec.stream.memcpy_htod(s, &mut v).map_err(drv)?;
            }
            // d_mrope holds 4*max_batch; write the leading 4*b axis-major view
            let mut staged = vec![0u32; 4 * b];
            staged.copy_from_slice(&mrope_host);
            exec.stream
                .memcpy_htod(&staged, &mut bs.d_mrope)
                .map_err(drv)?;
        }

        // P5 budget pool: grow each row's ACTUAL slot block table to back the
        // token it writes this step (positions[i]) before the (captured) compute
        // reads the device table. The upload is outside the graph, so growth is
        // replay-safe. No-op in identity/dense mode.
        if self.batch.as_ref().expect("batch enabled").pool.is_some() {
            for (i, &p) in positions.iter().enumerate() {
                let slot = slots.map_or(i, |s| s[i] as usize);
                self.ensure_slot_blocks(slot, p as usize)?;
            }
        }

        // Capture the fixed-B step once, then replay per token - the compute is
        // grid-stable for a given B (loop bounds like the KV length come from the
        // device position buffer, exactly as the single-stream graph relies on).
        self.ensure_batch_graph(b)?;
        self.batch.as_ref().expect("batch enabled").graphs[&b]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("batch graph launch: {e}")))?;
        // restore the identity d_slots (stream-ordered after the graph read it)
        // so a following pure-decode tick sees row i -> slot i again.
        if slots.is_some() {
            let ident: Vec<u32> = (0..b as u32).collect();
            let bs = self.batch.as_mut().expect("batch enabled");
            let mut v = bs.d_slots.slice_mut(0..b);
            exec.stream.memcpy_htod(&ident, &mut v).map_err(drv)?;
        }
        exec.synchronize()?;
        Ok(())
    }

    /// True when the pipelined pure-decode path can run: device sampling + the
    /// advance kernel, with no pin forcing a different dispatch. Runs in
    /// identity-KV mode AND under a FULLY-BACKED budget pool (capacity >=
    /// max_batch x blocks_per_slot - the enable_batch hard-reject guarantees
    /// this unless the user opted into oversubscription): per-tick block growth
    /// then cannot PoolExhaust (worst case it evicts radix pages), so
    /// `pipe_launch_tick` grows tables infallibly before each replay, exactly
    /// like the classic path does per step. Oversubscribed pools keep the
    /// classic tick (growth can fail there and the pipe has no clean unwind -
    /// the tick's KV/DeltaNet state advances before its ids are read back).
    /// Measured on an RTX PRO 6000, the pool gate cost ~7 ms/step of host
    /// serialization at b=32 (83% GPU busy on the classic tick).
    pub fn supports_decode_pipe(&self) -> bool {
        // A DFlash drafter needs every decode tick's feature append (the
        // pipe is device-driven and cannot append - the ring stalls behind
        // the cursor and the drafter goes permanently cold; see
        // dflash_ensure_warm). Classic ticks carry decode while attached.
        self.dflash.is_none()
            && self.exec.has_sample_rows()
            && self.exec.has_pipe_advance()
            // k-quant rides the pipe since the kq batch step is the
            // same captured-graph body (bmmq!'s dp4a/ks rungs are stream-ordered
            // and capture-safe - the ks launcher's device-attr statics warm on
            // the pre-capture run, like Q8's). Greedy A/B pipe-vs-classic gated.
            //
            // The static full-backing pool requirement is gone - it
            // silently disabled the pipe for every oversubscribed serve (the
            // FP8 headline stack runs PADDOCK_KV_OVERSUBSCRIBE=1), which the
            // 128x128c32 investigation exposed after five flat scheduling
            // fixes landed into dead code. Oversubscription safety moved to
            // the scheduler's DYNAMIC headroom gate: the pipe begins/continues
            // only while pool_free_blocks() covers a worst-case tick of block
            // growth plus the admission watermark (service.rs pipe loop), so
            // ensure_slot_blocks inside pipe_launch_tick can never PoolExhaust
            // mid-flight.
            && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none()
    }

    /// True when the overlap decode lane is forked (PADDOCK_OVERLAP at
    /// enable_batch) and the pipe machinery is available - the scheduler's
    /// gate for the overlapped span+decode tick (route B).
    pub fn overlap_ready(&self) -> bool {
        // On for every serve, vision included. The corruption once blamed on
        // picture spans under the pump was the pack sampler's process-wide
        // scratch racing between the two lanes (stage2_sample.cuh, PdSrScr);
        // it reproduced with text alone on an untouched build.
        self.overlap_exec.is_some() && self.supports_decode_pipe()
    }

    /// Launch fold for the sampler chains (nemotron's (any5,any6)
    /// fold ported): launch only the chains this tick's rows need. The
    /// elected qwen truncation (top_k 20) is pure mode 5, so without this
    /// every tick also paid the 11-launch mode-6 nucleus chain (and the
    /// base modes-1/2 kernel) for zero rows. PADDOCK_NO_SAMP_FOLD=1
    /// restores launch-always for A/B legs. Behavior-identical: a skipped
    /// chain had no rows of its mode, so it wrote nothing.
    pub(super) fn samp_fold_off() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_SAMP_FOLD").is_some())
    }

    /// Engagement witness (bisect-trap law): once per process, the first
    /// folded trunc tick prints which chains it actually launched.
    pub(super) fn samp_fold_witness(any_base: bool, any5: bool, any6: bool) {
        static W: std::sync::Once = std::sync::Once::new();
        W.call_once(|| {
            eprintln!("[samp-fold] engaged: base={any_base} t5={any5} p6={any6}");
        });
    }

    /// Host-build the section-major mrope buffer for a pipe tick - positions are
    /// deterministic (`pos0[i] + tick`), so RoPE needs no device->host roundtrip.
    /// Byte-identical layout to `launch_batch_step_slots`' host build.
    pub(super) fn pipe_mrope_host(pos0: &[u32], delta: &[i64], tick: u64) -> Vec<u32> {
        let b = pos0.len();
        let mut v = vec![0u32; 4 * b];
        for sec in 0..4 {
            for i in 0..b {
                v[sec * b + i] = (pos0[i] as i64 + tick as i64 + delta[i]) as u32;
            }
        }
        v
    }

    /// Enqueue one pipelined tick: advance the device-dependent inputs (token =
    /// previous tick's sampled id, position += 1) from the previous out ring,
    /// rebuild mrope host-side, replay the shared step graph (no sync), sample
    /// on device into this tick's out ring, and record its readiness event.
    fn pipe_launch_tick(
        &mut self,
        plans: &[crate::generator::RowSample],
        advance: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let (b, tick) = {
            let p = self.pipe.as_ref().expect("pipe active");
            (p.b, p.tick)
        };
        // P5 budget pool: back the position this tick writes (pos0[i] + tick -
        // deterministic, same arithmetic as pipe_mrope_host) before any pipe
        // state mutates, so a growth error leaves the rings/inputs untouched.
        // The pipe only runs on a fully-backed pool (supports_decode_pipe), so
        // this cannot PoolExhaust; the table htod inside ensure_slot_blocks is
        // stream-ordered ahead of the graph replay below. Row i drives its
        // mapped slot (identity unless the pipe carries an explicit set).
        // No-op in identity/dense mode.
        if self.batch.as_ref().expect("batch enabled").pool.is_some() {
            let (pos0, slot_map) = {
                let p = self.pipe.as_ref().expect("pipe active");
                (p.pos0.clone(), p.slots.clone())
            };
            for (i, &p0) in pos0.iter().enumerate().take(b) {
                let slot = slot_map.as_ref().map_or(i, |s| s[i] as usize);
                self.ensure_slot_blocks(slot, p0 as usize + tick as usize)?;
            }
        }
        let ring = (tick % 2) as usize;
        let (par, tpar, any_base, any5, any6) = pack_samp_par(plans);
        let fold_off = Self::samp_fold_off();
        let any_trunc = any5 || any6;
        let mrope = {
            let p = self.pipe.as_ref().expect("pipe active");
            Self::pipe_mrope_host(&p.pos0, &p.delta, tick)
        };
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let max_batch = self.batch.as_ref().expect("batch enabled").max_batch;
        // sampler params into this tick's ring plane
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let off = ring * max_batch * 4;
            let mut v = bs.d_pipe_par.slice_mut(off..off + b * 4);
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if any_trunc {
                let mut t = bs.d_pipe_tpar.slice_mut(off..off + b * 4);
                exec.stream.memcpy_htod(&tpar, &mut t).map_err(drv)?;
            }
        }
        // advance the device-dependent inputs from the previous ring's out plane
        if advance {
            let prev = ((tick + 1) % 2) as usize;
            let bs = self.batch.as_mut().expect("batch enabled");
            let (out, tok, pos) = (&bs.d_pipe_out, &mut bs.d_tokens, &mut bs.d_pos);
            exec.pipe_advance(out, prev * max_batch, tok, pos, b)?;
        }
        // rebuild mrope (deterministic positions) - stream-ordered before the
        // replay, so the captured graph reads this tick's RoPE positions
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let mut v = bs.d_mrope.slice_mut(0..4 * b);
            exec.stream.memcpy_htod(&mrope, &mut v).map_err(drv)?;
        }
        // replay the same per-B step graph the classic path uses
        {
            let g = &self.batch.as_ref().expect("batch enabled").graphs[&b].0;
            if paddock_models::dev_var_os!("PADDOCK_PIPE_CHECK").is_some()
                && g.stream_ptr() != exec.stream.cu_stream() as usize
            {
                eprintln!(
                    "[pipe-check] graph b={b} replays on stream {:#x} but the tick runs on {:#x}",
                    g.stream_ptr(),
                    exec.stream.cu_stream() as usize
                );
            }
            g.launch()
                .map_err(|e| GpuError::Driver(format!("pipe step graph launch: {e}")))?;
        }
        if paddock_models::dev_var_os!("PADDOCK_PIPE_CHECK2").is_some() {
            exec.synchronize()?; // A/B: does the sampler overtake the graph?
        }
        // fused device sampling into this tick's out ring plane (folded:
        // launch only the chains this tick's row modes need)
        if any_base || fold_off {
            let bs = self.batch.as_mut().expect("batch enabled");
            let (logits, par_buf, out) = (&bs.d_logits, &bs.d_pipe_par, &mut bs.d_pipe_out);
            exec.sample_rows_at(
                logits,
                par_buf,
                ring * max_batch * 4,
                out,
                ring * max_batch,
                b,
                vocab,
            )?;
        }
        if any_trunc {
            Self::samp_fold_witness(any_base, any5, any6);
            // P67b: mode-5 rows sample in the same out ring plane - the
            // pipe_advance of tick N+1 consumes them device-side unchanged.
            // Engagement witness (bisect-trap law; its absence cost a
            // diagnosis round on): once per process.
            static PIPE5: std::sync::Once = std::sync::Once::new();
            PIPE5.call_once(|| {
                eprintln!("[trunc-pipe] engaged: b={b} (mode-5 rows in the decode pipe)");
            });
            let bs = self.batch.as_mut().expect("batch enabled");
            let (logits, par_buf, tpar_buf) = (&bs.d_logits, &bs.d_pipe_par, &bs.d_pipe_tpar);
            if any5 || fold_off {
                exec.sample_rows_t_at(
                    logits,
                    par_buf,
                    ring * max_batch * 4,
                    tpar_buf,
                    ring * max_batch * 4,
                    &mut bs.d_pipe_out,
                    ring * max_batch,
                    b,
                    vocab,
                )?;
            }
            if any6 || fold_off {
                exec.sample_rows_p_at(
                    logits,
                    par_buf,
                    ring * max_batch * 4,
                    tpar_buf,
                    ring * max_batch * 4,
                    &mut bs.d_pipe_out,
                    ring * max_batch,
                    b,
                    vocab,
                )?;
            }
        }
        let ev = exec.record_event()?;
        self.pipe.as_mut().expect("pipe active").ev[ring] = Some(ev);
        Ok(())
    }

    /// Begin a pipelined pure-decode over the same rows `forward_batch_sampled`
    /// would take (all Device/Hole plans - no Host rows). No ids come back yet;
    /// the first `decode_pipe_next` returns tick 0's while tick 1 runs.
    pub fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        self.with_decode_lane(|m| m.decode_pipe_begin_inner(tokens, positions, None, plans))
    }

    /// `decode_pipe_begin` over an ARBITRARY slot set: row i drives
    /// `slots[i]` instead of slot i. The overlap scheduler uses this for the
    /// churn-phase decode set (the non-chunking slots - never contiguous).
    /// The mapping is written to `d_slots` once here; the captured graph
    /// reads `d_slots` per replay, so every tick sees it. Identity is
    /// restored at drain/abort.
    pub fn decode_pipe_begin_slots(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        self.with_decode_lane(|m| m.decode_pipe_begin_inner(tokens, positions, Some(slots), plans))
    }

    fn decode_pipe_begin_inner(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        assert_eq!(positions.len(), b, "one position per row");
        if !self.supports_decode_pipe() {
            return Err(GpuModelError::Unsupported("decode pipe".into()));
        }
        // Split-decode companion (see batch_sampled_impl): the pipe graph
        // cannot split, so above the fast arms' b <= 64 bound it would
        // replay slow eager lanes (~62ms ticks at 128 rows). Decline; the
        // service falls back to classic sampled ticks, which split.
        if b > f8t_dec_bmax() && paddock_models::dev_var_os!("PADDOCK_NO_DEC_SPLIT").is_none() {
            return Err(GpuModelError::Unsupported(
                "decode pipe above the fast-arm bound".into(),
            ));
        }
        match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) if b > bs.max_batch => {
                return Err(GpuModelError::BatchTooLarge {
                    got: b,
                    max: bs.max_batch,
                });
            }
            _ => {}
        }
        if let Some(s) = slots {
            assert_eq!(s.len(), b, "one slot per row");
        }
        assert!(self.pipe.is_none(), "decode pipe already active");
        self.ensure_scratch(b)?;
        self.ensure_batch_graph(b)?;
        // A/B instrument (PADDOCK_PUMP_SERIAL): drain the OTHER lane before the
        // pipe's first tick - the host flow of the overlap scheduler is kept,
        // only the GPU overlap between the span and the pipe is removed.
        if paddock_models::dev_var_os!("PADDOCK_PUMP_SERIAL").is_some()
            && let Some(other) = self.overlap_exec.as_ref()
        {
            other.synchronize()?;
        }
        let delta: Vec<i64> = {
            let bs = self.batch.as_ref().expect("batch enabled");
            (0..b)
                .map(|i| bs.mrope_delta[slots.map_or(i, |s| s[i] as usize)])
                .collect()
        };
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        // tick-0 inputs land in the fixed graph buffers (advance=false won't set
        // them); d_slots is identity (pure decode) unless an explicit mapping
        // rides the whole pipe - written once here, restored at drain/abort.
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let mut vt = bs.d_tokens.slice_mut(0..b);
            exec.stream.memcpy_htod(tokens, &mut vt).map_err(drv)?;
            let mut vp = bs.d_pos.slice_mut(0..b);
            exec.stream.memcpy_htod(positions, &mut vp).map_err(drv)?;
            if let Some(s) = slots {
                let mut vs = bs.d_slots.slice_mut(0..b);
                exec.stream.memcpy_htod(s, &mut vs).map_err(drv)?;
            }
        }
        self.pipe = Some(PipeStateQ {
            b,
            tick: 0,
            ev: [None, None],
            pos0: positions.to_vec(),
            delta,
            slots: slots.map(<[u32]>::to_vec),
        });
        if let Err(e) = self.pipe_launch_tick(plans, false) {
            self.pipe_abort();
            return Err(e);
        }
        Ok(())
    }

    /// Enqueue the next tick (its inputs advance on device from the previous
    /// tick's sampler output) and return the ids of the OLDEST in-flight tick,
    /// read via the side stream while the new tick executes. `plans[i]` must be
    /// Device or Hole, same rows as begin.
    pub fn decode_pipe_next(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuModelError> {
        self.with_decode_lane(|m| m.decode_pipe_next_inner(plans))
    }

    fn decode_pipe_next_inner(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let (b, j) = {
            let p = self.pipe.as_ref().ok_or_else(|| {
                GpuModelError::Unsupported("decode_pipe_next without begin".into())
            })?;
            (p.b, p.tick)
        };
        assert_eq!(plans.len(), b, "one plan per row");
        self.pipe.as_mut().expect("pipe active").tick = j + 1;
        if let Err(e) = self.pipe_launch_tick(plans, true) {
            self.pipe_abort();
            return Err(e);
        }
        let ring = (j % 2) as usize;
        let max_batch = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .max_batch;
        let r = {
            let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
            let ev = self.pipe.as_ref().expect("pipe active").ev[ring]
                .as_ref()
                .expect("in-flight tick event");
            let ids = exec.to_host_u32_after(ev, &bs.d_pipe_out, ring * max_batch, b);
            // PADDOCK_PIPE_CHECK=1: re-read the same ring after a full lane drain
            // and report rows whose id changed - a changed id means the first
            // read preceded the sampler's write (the event did not cover it).
            if let Ok(first) = &ids
                && paddock_models::dev_var_os!("PADDOCK_PIPE_CHECK").is_some()
            {
                let _ = exec.synchronize();
                if let Ok(again) = exec
                    .stream
                    .clone_dtoh(&bs.d_pipe_out.slice(ring * max_batch..ring * max_batch + b))
                {
                    let diff: Vec<(usize, u32, u32)> = first
                        .iter()
                        .zip(again.iter())
                        .enumerate()
                        .filter(|(_, (a, c))| a != c)
                        .map(|(i, (a, c))| (i, *a, *c))
                        .collect();
                    if !diff.is_empty() {
                        eprintln!(
                            "[pipe-check] tick {j} ring {ring} b={b}: premature ids {diff:?}"
                        );
                    }
                }
            }
            ids
        };
        match r {
            Ok(ids) => Ok(ids),
            Err(e) => {
                self.pipe_abort();
                Err(e.into())
            }
        }
    }

    /// End the pipe: return the last in-flight tick's ids without enqueueing
    /// more work. Must be called before any other forward call - the fixed
    /// input buffers are left stale and every other path re-uploads them.
    pub fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GpuModelError> {
        self.with_decode_lane(|m| m.decode_pipe_drain_inner())
    }

    fn decode_pipe_drain_inner(&mut self) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let st = self
            .pipe
            .take()
            .ok_or_else(|| GpuModelError::Unsupported("decode_pipe_drain without begin".into()))?;
        let ring = (st.tick % 2) as usize;
        let max_batch = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .max_batch;
        let ev = st.ev[ring].as_ref().expect("in-flight tick event");
        let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
        let r = exec.to_host_u32_after(ev, &bs.d_pipe_out, ring * max_batch, st.b);
        // a slot-mapped pipe leaves real slots in d_slots - restore identity
        // (stream-ordered after the last replay's reads on this same stream)
        // so following pure-decode ticks see row i -> slot i again.
        if st.slots.is_some() {
            let ident: Vec<u32> = (0..st.b as u32).collect();
            let bs = self.batch.as_mut().expect("batch enabled");
            let mut v = bs.d_slots.slice_mut(0..st.b);
            let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
            exec.stream.memcpy_htod(&ident, &mut v).map_err(drv)?;
        }
        match r {
            Ok(ids) => Ok(ids),
            Err(e) => {
                let _ = exec.synchronize(); // state gone - quiesce readers of the rings
                Err(e.into())
            }
        }
    }

    /// Kill an in-flight pipe after an error (or on reset): quiesce the stream so
    /// nothing still reads the pipe buffers, then drop the state.
    pub(in crate::gpu_model::qwen35) fn pipe_abort(&mut self) {
        // quiesce the lane the pipe actually ran on
        self.with_decode_lane(|m| {
            if let Some(st) = m.pipe.take() {
                let _ = m.exec.synchronize();
                // best-effort identity restore after a slot-mapped pipe (the
                // stream is quiesced, so this can't race a replay)
                if st.slots.is_some()
                    && let Some(bs) = m.batch.as_mut()
                {
                    let ident: Vec<u32> = (0..st.b as u32).collect();
                    let mut v = bs.d_slots.slice_mut(0..st.b);
                    let _ = m.exec.stream.memcpy_htod(&ident, &mut v);
                }
            }
        })
    }
}
