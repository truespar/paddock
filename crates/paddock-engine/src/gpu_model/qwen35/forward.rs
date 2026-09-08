//! Qwen3.5/3.6 single-seq step/prefill, graph capture, core forward + geometry.

use super::*;
use crate::gpu::{GpuError, KvDtype, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use cudarc::driver::sys::CUstreamCaptureMode;

impl GpuQwen35 {
    /// Clear all per-sequence state for a fresh sequence (Generator seam). The
    /// decode buffers keep their allocations - and with them every captured
    /// graph, whose baked addresses stay valid across sequences. What actually
    /// resets: position bookkeeping, the DeltaNet recurrent states, the conv
    /// windows, the MTP h carry, and the generated-token cursor. The KV caches
    /// need no clearing: every attention read is position-bounded, and the
    /// prefill f16 path multiplies masked (stale) rows by exact zeros.
    /// Select the KV cache element type (default [`KvDtype::Fp16`], greedy-exact).
    /// [`KvDtype::Fp8E4m3`] is a lossy opt-in throughput/memory mode (halves KV
    /// bytes; greedy-robust gated, same class as the gpt-oss fp8 KV gate). Drops
    /// every lazily-built state so caches, scratch, and captured graphs
    /// re-materialize at the new element size - call it before any decode/serving.
    /// Widest out_dim the K-split mma serving/spec rung may see - sizes the
    /// d_ks_part buffers (8 z-planes x 64-row envelope x this).
    /// Ticket words for one FFN half (gate or up): BM=64 puts two CTAs on
    /// every 128-row box. gate owns [0, this), up owns [this, 2*this).
    pub(super) fn lin_tick_half(&self) -> usize {
        2 * self.ff.div_ceil(128)
    }

    /// Ticket-region offset of the FFN `down` plane. gate and up run as two
    /// INDEPENDENT launches (that is the whole point - see `f8lin_gemv_at`),
    /// so they cannot share a counter region.
    pub(super) fn lin_tick_dn_off(&self) -> usize {
        2 * self.lin_tick_half()
    }

    /// Total lin-GEMV ticket words: one region per concurrent launch.
    pub(super) fn lin_tick_len(&self) -> usize {
        self.lin_tick_dn_off() + 2 * self.embd.div_ceil(128)
    }

    // (lin_part_half lived here: a partials stride for running gate|up as two
    // INDEPENDENT lin-GEMV launches. That variant was measured and lost to the
    // fused form, so both calls share part_off 0 on one stream and the helper
    // had no callers. Removed rather than left as a sizing contract nothing
    // honours.)

    pub(super) fn ks_out_max(&self) -> usize {
        // 2*ff covers the fused gate|up plane (nz stays 1 on <=272-SM dies,
        // but the part scratch must fit any future die's split choice)
        let ff_max = if self.bs_gu.iter().any(Option::is_some) {
            2 * self.ff
        } else {
            self.ff
        };
        ff_max
            .max(2 * self.embd)
            .max(self.conv_dim)
            .max(2 * self.n_heads * self.head_dim)
            // f8t attention lane: the K-split partials land the FUSED plane's
            // whole out row, which is wider than either term above -
            // [2q|k|v] for full-attn, [conv|value] for DeltaNet.
            .max(2 * self.n_heads * self.head_dim + 2 * self.n_kv_heads * self.head_dim)
            .max(self.conv_dim + self.value_dim)
    }

    pub fn set_kv_dtype(&mut self, dtype: KvDtype) {
        self.pipe_abort(); // pipe events/rings point into the old BatchState
        self.kv_dtype = dtype;
        self.decode = None;
        self.batch = None;
        self.spec = None;
        self.spec_batch = None;
        self.scratch = None;
        self.history.clear();
    }

    pub fn reset(&mut self) {
        self.pipe_abort();
        self.history.clear();
        let Some(ds) = self.decode.as_mut() else {
            return;
        };
        ds.pos = 0;
        ds.mrope_pos = 0;
        let zero_err = "reset: zeroing sequence state";
        for r in ds.recur.iter_mut().flatten() {
            self.exec.stream.memset_zeros(r).expect(zero_err);
        }
        for w in ds.conv_win.iter_mut().flatten() {
            self.exec.stream.memset_zeros(w).expect(zero_err);
        }
        self.exec
            .stream
            .memset_zeros(&mut ds.pending_h)
            .expect(zero_err);
        self.exec
            .stream
            .memset_zeros(&mut ds.d_step)
            .expect(zero_err);
    }

    /// Allocate the persistent per-layer decode state (zeroed) if absent.
    pub(super) fn ensure_decode(&mut self) -> Result<(), GpuModelError> {
        if self.decode.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let (mut kv_k, mut kv_v, mut recur, mut conv_win) = (
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
        );
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Full(_) => {
                    kv_k.push(Some(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?));
                    kv_v.push(Some(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?));
                    recur.push(None);
                    conv_win.push(None);
                }
                Mixer::Linear(_) => {
                    kv_k.push(None);
                    kv_v.push(None);
                    recur.push(Some(
                        e.alloc(self.n_v_heads * self.state_size * self.state_size)?,
                    ));
                    conv_win.push(Some(e.alloc((self.conv_k - 1) * self.conv_dim)?));
                }
            }
        }
        // Fixed device-resident inputs (d_slots is a constant [0]).
        let d_token = e.alloc_u32(1)?;
        let d_pos = e.alloc_u32(1)?;
        let d_slots = e.alloc_u32(1)?; // zeroed by alloc -> slot 0
        let d_mrope = e.alloc_u32(4)?;
        let d_out = e.alloc_u32(GEN_CHUNK)?;
        let d_step = e.alloc_u32(1)?;
        let d_pmax = e.alloc(ARGMAX_PARTS)?;
        let d_pidx = e.alloc_u32(ARGMAX_PARTS)?;
        let (mtp_kv_k, mtp_kv_v) = if self.mtp.is_some() {
            (
                Some(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?),
                Some(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?),
            )
        } else {
            (None, None)
        };
        let pending_h = e.alloc(self.embd)?; // zeroed - the position-0 h input
        // Prefill inputs at fixed addresses (graph-capturable). Positions carry
        // 0..max_ctx once and for all - prefill is always a fresh sequence, so
        // every prompt length reads a valid prefix. Slots stay zeroed (slot 0).
        let d_pf_tokens = e.alloc_u32(self.max_ctx)?;
        let mut d_pf_pos = e.alloc_u32(self.max_ctx)?;
        let pf_pos_host: Vec<u32> = (0..self.max_ctx as u32).collect();
        e.stream
            .memcpy_htod(&pf_pos_host, &mut d_pf_pos)
            .map_err(|x| GpuError::Driver(x.to_string()))?;
        let d_pf_slots = e.alloc_u32(self.max_ctx)?;
        let d_pf_mrope = e.alloc_u32(4 * self.max_ctx)?;
        self.decode = Some(DecodeState {
            pos: 0,
            mrope_pos: 0,
            kv_k,
            kv_v,
            recur,
            conv_win,
            d_token,
            d_pos,
            d_slots,
            d_mrope,
            d_out,
            d_step,
            d_pmax,
            d_pidx,
            mtp_kv_k,
            mtp_kv_v,
            pending_h,
            graph: None,
            graph_gen: None,
            d_pf_tokens,
            d_pf_pos,
            d_pf_slots,
            d_pf_mrope,
            pf_graphs: std::collections::HashMap::new(),
        });
        Ok(())
    }

    /// One incremental decode token: advance the persistent per-layer state and
    /// return the next-token logits [vocab]. O(1) in context length. Structurally
    /// identical to `forward_full`'s per-layer body at t=1, but the DeltaNet state
    /// and conv window persist (via `conv_step` + non-zeroed recurrence) and the
    /// full-attn KV cache grows by one row instead of being rebuilt.
    fn step(&mut self, token: u32) -> Result<Vec<f32>, GpuModelError> {
        self.ensure_scratch(1)?;
        self.ensure_decode()?;
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let pos = self.decode.as_ref().expect("decode").pos;
        assert!(
            pos < self.max_ctx,
            "decode position {pos} exceeds max_ctx {}",
            self.max_ctx
        );

        // Push this token's inputs into the fixed device buffers (stream-ordered,
        // outside the graph - only their contents change per token).
        {
            let ds = self.decode.as_mut().expect("decode");
            exec.stream
                .memcpy_htod(&[token], &mut ds.d_token)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            exec.stream
                .memcpy_htod(&[pos as u32], &mut ds.d_pos)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            let mp = ds.mrope_pos as u32;
            exec.stream
                .memcpy_htod(&[mp; 4], &mut ds.d_mrope)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }

        // Capture the per-token compute into a CUDA graph on the first step of a
        // sequence, then replay it for every subsequent token. Each step launches
        // the identical kernels over the identical (persistent) buffers with fixed
        // grids; only the device-resident inputs above change, plus the full-attn
        // loop bound - which the kernel reads from `d_pos` on device, so the grid is
        // position-independent and one capture is valid at every position. Replaying
        // the graph collapses ~480 per-token kernel launches into a single submit.
        if self.decode.as_ref().expect("decode").graph.is_none() {
            self.capture_graph()?;
        }
        self.decode
            .as_ref()
            .expect("decode")
            .graph
            .as_ref()
            .expect("graph captured above")
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("graph launch: {e}")))?;

        let logits = exec.to_host(&self.scratch.as_ref().expect("scratch").d_logits)?;
        debug_assert_eq!(logits.len(), vocab);
        {
            let ds = self.decode.as_mut().expect("decode");
            ds.pos += 1;
            ds.mrope_pos += 1;
        }
        Ok(logits)
    }

    /// Capture `record_step` into a replayable CUDA graph (stored on the decode
    /// state). Called once per sequence - the capture itself only *records* the
    /// launches (nothing executes), so the caller still `launch()`es the returned
    /// graph to run this first token. Invalidated (recaptured) after `reset` or any
    /// scratch reallocation, since those move the buffer addresses the graph baked in.
    fn capture_graph(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        // Quiesce the stream so no in-flight work is folded into the capture.
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        // Record the per-token launches. On error we still must end the capture to
        // leave the stream in a non-capturing state before surfacing it.
        let rec = self.record_step();
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        rec?; // surface a record_step failure only after capture is cleanly ended
        let graph = graph?.ok_or_else(|| GpuError::Driver("capture produced no graph".into()))?;
        self.decode.as_mut().expect("decode").graph = Some(SendGraph(graph));
        Ok(())
    }

    /// Capture `record_step` plus the on-device argmax+advance epilogue into a second
    /// graph - the graph-resident generation loop. Each replay computes this token's
    /// logits, then (all on device) picks the next token into `d_token`, appends it to
    /// `d_out[d_step++]`, and advances `d_pos`/`d_mrope`, so the host can replay it
    /// back-to-back with zero per-token round-trip.
    pub(super) fn capture_graph_gen(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        let rec = self.record_step().and_then(|()| self.record_advance());
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        rec?;
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("gen capture produced no graph".into()))?;
        self.decode.as_mut().expect("decode").graph_gen = Some(SendGraph(graph));
        Ok(())
    }

    /// The `Generator::forward_prefill_stream` entry: one stream, no scheduler.
    /// Batched (the served configuration at every width, max_batch=1
    /// included): the stream owns slot 0 and takes the same slot prefill the
    /// scheduler drives per request, so the service warm-up exercises the
    /// lane that will serve and the decode pipe that follows it finds its
    /// prompt in slot 0. Routing this through the serial `prefill` instead
    /// cost a dense max_ctx KV per full-attention layer + the MTP layer from
    /// `ensure_decode` - 17 GiB at 262k on the k-quant lanes, allocated by
    /// the warm-up after the plan audit had closed at zero unplanned and
    /// never released (measured 2026-09-07: UD-Q3_K_XL 1x262k 58.5 GiB on
    /// a 42.3 GiB plan; on Q8_0 the reclaim guard below refused it, so the
    /// warm-up silently did nothing at all). Serial (no batch lane): the
    /// token-by-token spine is the only path and stays the parity reference.
    pub fn prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GpuModelError> {
        if self.batch.is_some() {
            // run_request reaches this only with no request live (the
            // service warm-up before the loop; the scheduler's hand-offs
            // happen at active == 0), so every slot is idle: return what a
            // previous stream left in slot 0 before reusing it
            self.release_inactive_slots(&[]);
            return self.forward_prefill_slot(0, tokens);
        }
        self.prefill(tokens)
    }

    /// Serial-lane prefill (no batch lane: the parity reference, the ppl gate,
    /// the load tests): run the whole `tokens` chunk through the backbone with one
    /// GEMM pass per weight (batch-tiled - the weight is read once per 16 tokens,
    /// not per token; llama's pp lever), writing the persistent per-layer state (KV
    /// rows at slot 0, DeltaNet recurrent state, conv window) exactly as `step`-ing
    /// each token would. The per-token math is bit-identical to the incremental
    /// path: the GEMM at each row uses the same chunk order as `q8_0_gemv_repacked`,
    /// attention/norm/recurrence reuse the identical batched kernels, and the
    /// prefill conv from a zero window equals `conv_step`'s zero-window chain.
    /// Returns the last token's logits. Requires a fresh sequence (pos == 0).
    /// Allocates the dense serial decode state (`ensure_decode`, a max_ctx KV
    /// per full-attention layer) - a served generator enters through
    /// `prefill_stream`, never here.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GpuModelError> {
        let t_len = tokens.len();
        assert!(t_len > 0, "empty prefill");
        // This path's `Ffn::Dense` arm reads gate/up/down directly -- it has no
        // e4m3 arm and no bs_f8ffn consult. When load.rs reclaimed the Q8_0
        // twins those planes became 32-byte stubs, so every caller here (the
        // ppl gate, qwen35_profile, tests/gpu_qwen35_load) would have measured
        // garbage without saying so. Refuse loudly instead: the reclaim is
        // what these harnesses have to opt out of, and the message says how.
        if let Some(Ffn::Dense { gate, .. }) = self.layers.first().map(|l| &l.ffn)
            && let QuantW::Q8(q) = gate
            && q.data.len() == 32
            && q.dims.iter().product::<usize>() > 32
        {
            return Err(GpuModelError::Unsupported(
                "qwen35 prefill(): the Q8_0 dense-FFN planes were reclaimed at load \
                         (e4m3 twins serve every SERVING band), and this path has no e4m3 arm \
                         -- it would read 32-byte stubs and return garbage. Re-run with \
                         PADDOCK_QWEN35_F8_FFN_PF_MIN=2 to keep the Q8_0 planes resident."
                    .into(),
            ));
        }
        self.ensure_decode()?;
        assert_eq!(
            self.decode.as_ref().expect("decode").pos,
            0,
            "prefill requires a fresh sequence (reset first)"
        );
        assert!(
            t_len <= self.max_ctx,
            "prompt {t_len} exceeds max_ctx {}",
            self.max_ctx
        );
        self.ensure_scratch(t_len)?;

        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        // Per-prompt inputs land in the fixed decode-state buffers, outside any
        // graph - only their contents change per prompt. Positions are already
        // 0..max_ctx (fresh sequence); slots stay zeroed (slot 0); mrope is the
        // text layout (all four axes = token index) but axis-major [4, t_len],
        // so its layout depends on the length - re-uploaded each call.
        {
            let ds = self.decode.as_mut().expect("decode");
            let mrope_host: Vec<u32> = (0..4).flat_map(|_| 0..t_len as u32).collect();
            let mut tok_view = ds.d_pf_tokens.slice_mut(0..t_len);
            exec.stream
                .memcpy_htod(tokens, &mut tok_view)
                .map_err(drv)?;
            let mut mrope_view = ds.d_pf_mrope.slice_mut(0..4 * t_len);
            exec.stream
                .memcpy_htod(&mrope_host, &mut mrope_view)
                .map_err(drv)?;
        }

        // Replay the captured pass for this prompt length, capturing on first
        // sight - the capture records exactly the launches the eager pass would
        // issue, then the instantiated graph runs them, collapsing ~700 kernel
        // submits into one (P6k measured the eager launch tax at ~3.5 ms/pp512;
        // llama replays one graph). PADDOCK_NO_PREFILL_GRAPH=1 pins eager; the
        // numerics-pinning A/B envs also force eager - a cached graph baked the
        // DEFAULT dispatch at record time and would silently ignore them.
        let eager = paddock_models::dev_var_os!("PADDOCK_NO_PREFILL_GRAPH").is_some()
            || paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_DN").is_some()
            || paddock_models::dev_var!("PADDOCK_PREFILL_ATTN").is_ok_and(|v| !v.is_empty());
        if eager {
            self.record_prefill(t_len)?;
        } else {
            if !self
                .decode
                .as_ref()
                .expect("decode")
                .pf_graphs
                .contains_key(&t_len)
            {
                self.capture_prefill_graph(t_len)?;
            }
            self.decode.as_ref().expect("decode").pf_graphs[&t_len]
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("prefill graph launch: {e}")))?;
        }

        let logits = exec.to_host(&self.scratch.as_ref().expect("scratch").d_logits)?;
        {
            let ds = self.decode.as_mut().expect("decode");
            ds.pos = t_len;
            ds.mrope_pos = t_len;
        }
        Ok(logits)
    }

    /// Capture `record_prefill` into a replayable graph, cached per prompt
    /// length (the launch grids bake in t_len). Same contract as
    /// `capture_graph`: the capture only records, the caller launches.
    fn capture_prefill_graph(&mut self, t_len: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        let rec = self.record_prefill(t_len);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        rec?;
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("prefill capture produced no graph".into()))?;
        let ds = self.decode.as_mut().expect("decode");
        // Chat traffic can produce many distinct lengths - bound the cache.
        if ds.pf_graphs.len() >= 16 {
            ds.pf_graphs.clear();
        }
        ds.pf_graphs.insert(t_len, SendGraph(graph));
        Ok(())
    }

    /// Record one full prefill pass (embed -> layers -> out_norm -> lm_head) onto
    /// the stream - the body shared by the eager path and graph capture. Every
    /// input comes from a fixed-address device buffer (`d_pf_*`), so a capture
    /// of these launches replays correctly for any prompt of the same length.
    fn record_prefill(&mut self, t_len: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (state_size, n_k_heads, n_v_heads, conv_k) =
            (self.state_size, self.n_k_heads, self.n_v_heads, self.conv_k);
        let (conv_dim, ff, max_ctx) = (self.conv_dim, self.ff, self.max_ctx);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kq_res = self.kq_resident;

        let sinks = &self.sinks;
        let layers = &self.layers;
        let tok_embd = &self.tok_embd;
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");

        embed_any(&exec, tok_embd, &ds.d_pf_tokens, &mut sc.d_x, embd, t_len)?;

        // Per-tensor prefill matmul dispatch: Q8_0 keeps the exact existing
        // int8-MMA ladder; the k-quant arm rides the stage-2 W4A8 GEMM (int8
        // tensor cores off the 4-6.6 bpw streams - same activation class as
        // the Q8 mmq route). PADDOCK_KQ_F32_PREFILL=1 pins the exact-f32
        // interim instead (parity triage: exact values, ~4-5x slower).
        let kq_f32 = paddock_models::dev_var_os!("PADDOCK_KQ_F32_PREFILL").is_some();
        // pre sites (input = the fused-quantized attn/post-norm rows): the
        // batch>64 fused kernel wrote d_yq (mmq layout, W4A8 tile GEMM);
        // small prefills wrote the strided staging (dp4a GEMM) - the _any
        // helpers pick by the same batch rule the quantize used.
        macro_rules! pmm_pre {
            ($w:expr, $y:expr) => {
                match $w {
                    QuantW::Kq(k) if kq_f32 => {
                        kq_gemm(&exec, k, &sc.d_xn, &mut sc.d_wdq, $y, t_len)?
                    }
                    w => prefill_mm_pre_any(
                        &exec,
                        w,
                        &sc.d_pxq,
                        &sc.d_pxs,
                        &sc.d_yq,
                        &mut sc.d_xsums,
                        &mut sc.d_ssums,
                        &mut sc.d_skfix,
                        $y,
                        t_len,
                    )?,
                }
            };
        }
        macro_rules! pmm {
            ($w:expr, $x:expr, $y:expr) => {
                match $w {
                    QuantW::Kq(k) if kq_f32 => kq_gemm(&exec, k, $x, &mut sc.d_wdq, $y, t_len)?,
                    w => prefill_mm_any(
                        &exec,
                        w,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        &mut sc.d_xsums,
                        &mut sc.d_ssums,
                        &mut sc.d_skfix,
                        $x,
                        $y,
                        t_len,
                    )?,
                }
            };
        }

        // dev triage hook twin (see the per-layer print below): sub-step norms
        // on layer 0 only, eager mode.
        let dbg_sub = paddock_models::dev_var_os!("PADDOCK_DEBUG_LAYER_NORMS").is_some();
        macro_rules! dbg_norm {
            ($li:expr, $tag:expr, $buf:expr, $n:expr) => {
                if dbg_sub && $li == 0 {
                    let h = exec.to_host($buf)?;
                    let s = h[..$n]
                        .iter()
                        .map(|v| (*v as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    tracing::info!("dbg sub {:>10} |.| = {s:.6e}", $tag);
                }
            };
        }

        for (li, layer) in layers.iter().enumerate() {
            // attn_norm fused with the qkv/in_qkv quantize (P6k); xn only
            // materializes for Linear mixers (alpha/beta still read it) - and
            // on the k-quant f32-interim fallback (its arms read the f32 rows).
            let keep_xn = matches!(&layer.mixer, Mixer::Linear(_)) || (kq_res && kq_f32);
            prefill_add_norm_quant(
                &exec,
                &mut sc.d_x,
                None,
                false,
                &layer.attn_norm.buf,
                &mut sc.d_xn,
                keep_xn,
                &mut sc.d_pxq,
                &mut sc.d_pxs,
                &mut sc.d_yq,
                embd,
                t_len,
                eps,
            )?;
            dbg_norm!(li, "x_embed", &sc.d_x, t_len * embd);
            dbg_norm!(li, "xn", &sc.d_xn, t_len * embd);
            match &layer.mixer {
                Mixer::Full(w) => {
                    pmm_pre!(&w.wq, &mut sc.d_qg);
                    exec.split_qg(
                        &sc.d_qg,
                        &mut sc.d_q,
                        &mut sc.d_gate,
                        t_len,
                        n_heads,
                        head_dim,
                    )?;
                    pmm_pre!(&w.wk, &mut sc.d_k);
                    pmm_pre!(&w.wv, &mut sc.d_v);
                    exec.rmsnorm_batch(
                        &sc.d_q,
                        &w.q_norm.buf,
                        &mut sc.d_qn,
                        head_dim,
                        eps,
                        t_len * n_heads,
                    )?;
                    exec.rmsnorm_batch(
                        &sc.d_k,
                        &w.k_norm.buf,
                        &mut sc.d_kn,
                        head_dim,
                        eps,
                        t_len * n_kv_heads,
                    )?;
                    exec.mrope(
                        &mut sc.d_qn,
                        &ds.d_pf_mrope,
                        t_len,
                        n_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.d_kn,
                        &ds.d_pf_mrope,
                        t_len,
                        n_kv_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    exec.kv_append_batch(
                        &sc.d_kn,
                        ds.kv_k[li].as_mut().expect("full-attn layer KV"),
                        &ds.d_pf_pos,
                        Some(&ds.d_pf_slots),
                        kv_dim,
                        max_ctx,
                        t_len,
                        self.kv_dtype,
                    )?;
                    exec.kv_append_batch(
                        &sc.d_v,
                        ds.kv_v[li].as_mut().expect("full-attn layer KV"),
                        &ds.d_pf_pos,
                        Some(&ds.d_pf_slots),
                        kv_dim,
                        max_ctx,
                        t_len,
                        self.kv_dtype,
                    )?;
                    prefill_attn(
                        &exec,
                        &sc.d_qn,
                        ds.kv_k[li].as_ref().expect("full-attn layer KV"),
                        ds.kv_v[li].as_ref().expect("full-attn layer KV"),
                        sinks,
                        &mut sc.d_attn,
                        &ds.d_pf_pos,
                        &ds.d_pf_slots,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        t_len,
                        scale,
                        self.kv_dtype,
                        None,
                        Some((&mut sc.d_attn_o, &mut sc.d_attn_ml)),
                    )?;
                    exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, t_len * q_dim)?;
                    pmm!(&w.wo, &sc.d_attn, &mut sc.d_proj);
                }
                Mixer::Linear(w) => {
                    // input quantized by the fused attn_norm above (P6k)
                    pmm_pre!(&w.in_qkv, &mut sc.d_mixed);
                    dbg_norm!(li, "mixed", &sc.d_mixed, t_len * conv_dim);
                    let vb16 = dn_vb16(&exec, t_len, state_size)
                        && paddock_models::dev_var_os!("PADDOCK_DBG_NORM").is_none();
                    if vb16 {
                        exec.causal_conv1d_silu_qkv_b16_at(
                            &sc.d_mixed,
                            &w.conv_w.buf,
                            &mut sc.d_dq,
                            &mut sc.d_dk,
                            &mut sc.d_dv,
                            0,
                            0,
                            t_len,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                            conv_k,
                        )?;
                    } else if exec.has_conv_silu_qkv()
                        && paddock_models::dev_var_os!("PADDOCK_DBG_NORM").is_none()
                    {
                        // fused conv+split+norm (bit-exact composition); the
                        // dbg_norm probe keeps the two-kernel chain since it
                        // wants the d_conv intermediate
                        exec.causal_conv1d_silu_qkv_at(
                            &sc.d_mixed,
                            &w.conv_w.buf,
                            &mut sc.d_dq,
                            &mut sc.d_dk,
                            &mut sc.d_dv,
                            0,
                            0,
                            t_len,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                            conv_k,
                        )?;
                    } else {
                        exec.causal_conv1d_silu(
                            &sc.d_mixed,
                            &w.conv_w.buf,
                            &mut sc.d_conv,
                            t_len,
                            conv_dim,
                            conv_k,
                        )?;
                        dbg_norm!(li, "conv", &sc.d_conv, t_len * conv_dim);
                    }
                    // Leave the conv window holding the last k-1 pre-conv inputs so
                    // incremental decode continues seamlessly. Window rows are
                    // oldest-first == the tail rows of d_mixed (contiguous copy).
                    {
                        let win = ds.conv_win[li].as_mut().expect("DeltaNet layer conv");
                        let km1 = conv_k - 1;
                        if t_len >= km1 {
                            exec.copy_region(
                                &sc.d_mixed,
                                (t_len - km1) * conv_dim,
                                win,
                                0,
                                km1 * conv_dim,
                            )?;
                        } else {
                            // shorter than the window: it starts zeroed, fill the tail
                            exec.copy_region(
                                &sc.d_mixed,
                                0,
                                win,
                                (km1 - t_len) * conv_dim,
                                t_len * conv_dim,
                            )?;
                        }
                    }
                    if exec.has_conv_silu_qkv()
                        && paddock_models::dev_var_os!("PADDOCK_DBG_NORM").is_none()
                    {
                    } else {
                        exec.deltanet_split_gqa_norm(
                            &sc.d_conv,
                            &mut sc.d_dq,
                            &mut sc.d_dk,
                            &mut sc.d_dv,
                            t_len,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                        )?;
                    }
                    // Non-Q8 alpha/beta have no repacked pair - the ab plane is
                    // the only path (all spans); Q8 keeps its measured 1024 gate.
                    let use_ab = w.alpha_w.is_none() || t_len >= ab_f32_min_rows();
                    if let Some(ab) = w.ab_f32.as_ref().filter(|_| use_ab) {
                        // x2-v3: one f32-plane decay GEMM (64-col tile, x read once) +
                        // fused-layout gate; same values, tiled order (PPL-gated opt-in)
                        ab_gate(
                            &exec,
                            ab,
                            &sc.d_xn,
                            &mut sc.d_ab,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            t_len,
                            n_v_heads,
                        )?;
                    } else {
                        let aw = w.alpha_w.as_ref().expect("Q8 alpha (pair path)");
                        let bw = w.beta_w.as_ref().expect("Q8 beta (pair path)");
                        if exec.has_q8_0_gemm_repacked_x2() {
                            // fused pair: x staged once for both decay projections
                            // (bit-exact per output vs the two separate calls)
                            exec.q8_0_gemm_repacked_x2(
                                aw,
                                bw,
                                &sc.d_xn,
                                &mut sc.d_a,
                                &mut sc.d_b,
                                t_len,
                            )?;
                        } else {
                            exec.q8_0_gemm_repacked(aw, None, &sc.d_xn, &mut sc.d_a, t_len)?;
                            exec.q8_0_gemm_repacked(bw, None, &sc.d_xn, &mut sc.d_b, t_len)?;
                        }
                        exec.delta_gate(
                            &sc.d_a,
                            &sc.d_b,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            t_len,
                            n_v_heads,
                        )?;
                    }
                    dbg_norm!(li, "g", &sc.d_g, t_len * n_v_heads);
                    dbg_norm!(li, "beta", &sc.d_beta, t_len * n_v_heads);
                    prefill_delta_recurrent(
                        &exec,
                        sc,
                        ds.recur[li].as_mut().expect("DeltaNet layer state"),
                        0,
                        t_len,
                        n_v_heads,
                        state_size,
                        vb16,
                    )?;
                    dbg_norm!(li, "dattn", &sc.d_dattn, t_len * n_v_heads * state_size);
                    // d_xn/d_yq untouched since in_qkv's prefill_quant: reuse
                    pmm_pre!(&w.gate_w, &mut sc.d_z);
                    dbg_norm!(li, "z", &sc.d_z, t_len * n_v_heads * state_size);
                    exec.gated_rmsnorm(
                        &sc.d_dattn,
                        &sc.d_z,
                        &w.ssm_norm.buf,
                        &mut sc.d_core,
                        t_len * n_v_heads,
                        state_size,
                        eps,
                    )?;
                    dbg_norm!(li, "core", &sc.d_core, t_len * n_v_heads * state_size);
                    pmm!(&w.out_w, &sc.d_core, &mut sc.d_proj);
                    dbg_norm!(li, "mix_proj", &sc.d_proj, t_len * embd);
                }
            }
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    // residual add + post_norm + gate/up quantize in one pass (P6k);
                    // xn skipped where possible - the ffn quantize is its only
                    // consumer here (the k-quant f32-interim fallback reads xn)
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        false,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        kq_res && kq_f32,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        t_len,
                        eps,
                    )?;
                    pmm_pre!(gate, &mut sc.d_ffn_gate);
                    pmm_pre!(up, &mut sc.d_ffn_up);
                    dbg_norm!(li, "ffn_gate", &sc.d_ffn_gate, t_len * ff);
                    dbg_norm!(li, "ffn_up", &sc.d_ffn_up, t_len * ff);
                    match down {
                        QuantW::Kq(k) if kq_f32 => {
                            // exact-f32 fallback: swiglu explicitly, then the
                            // interim GEMM off the f32 activation
                            exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, t_len * ff)?;
                            kq_gemm(
                                &exec,
                                k,
                                &sc.d_ffn_gate,
                                &mut sc.d_wdq,
                                &mut sc.d_proj,
                                t_len,
                            )?;
                        }
                        w => prefill_ffn_down_any(
                            &exec,
                            w,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            t_len,
                        )?,
                    }
                    dbg_norm!(li, "ffn_proj", &sc.d_proj, t_len * embd);
                }
                Ffn::Nvf4Dense { gu, down } => {
                    // off the f32 xn - write_xn=true, the int8 staging
                    // outputs are unused (the chain quantizes to nvf4 itself
                    // on the W4A4 arm, and consumes f32 below the band)
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        false,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        true,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        t_len,
                        eps,
                    )?;
                    nvf4_ffn(
                        &exec,
                        gu,
                        down,
                        &sc.d_xn,
                        &mut sc.d_pxq,
                        &mut sc.d_nvs,
                        &mut sc.d_nv4part,
                        &mut sc.d_ffn_gate,
                        &mut sc.d_ffn_up,
                        &mut sc.d_ffn_gu,
                        &mut sc.d_swq_q,
                        &mut sc.d_swq_s,
                        &mut sc.d_proj,
                        ff,
                        t_len,
                    )?;
                    dbg_norm!(li, "ffn_gate", &sc.d_ffn_gate, t_len * ff);
                    dbg_norm!(li, "ffn_up", &sc.d_ffn_up, t_len * ff);
                    dbg_norm!(li, "ffn_proj", &sc.d_proj, t_len * embd);
                }
                Ffn::Moe(w) => {
                    // MoE needs the f32 xn for the router + shared expert -
                    // write_xn=true (the fused quantize output is unused here;
                    // moe_ffn quantizes into its own staging)
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        false,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        true,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        t_len,
                        eps,
                    )?;
                    moe_ffn(
                        &exec,
                        w,
                        self.moe.expect("moe dims"),
                        embd,
                        t_len,
                        true,
                        &sc.d_xn,
                        &mut sc.d_moe_xq,
                        &mut sc.d_moe_xs,
                        &mut sc.d_ssums,
                        &mut sc.d_moe_xs8,
                        &mut sc.d_moe_fs8,
                        &mut sc.d_moe_logits,
                        &sc.d_zero_bias,
                        &mut sc.d_moe_idx,
                        &mut sc.d_moe_w,
                        &mut sc.d_moe_fused,
                        &mut sc.d_moe_fq,
                        &mut sc.d_moe_fs,
                        &mut sc.d_moe_srow,
                        &mut sc.d_moe_sslot,
                        &mut sc.d_moe_bexp,
                        &mut sc.d_moe_part,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        &mut sc.d_skfix,
                        &mut sc.d_ffn_gate,
                        &mut sc.d_ffn_up,
                        &mut sc.d_mixed,
                        &mut sc.d_proj,
                    )?;
                }
            }
            exec.add(&mut sc.d_x, &sc.d_proj, t_len * embd)?;
            // dev triage hook (needs PADDOCK_NO_PREFILL_GRAPH=1 - readback is
            // capture-illegal): per-layer residual norms, to layer-localize
            // numeric wiring bugs by diffing two quant classes side by side.
            if paddock_models::dev_var_os!("PADDOCK_DEBUG_LAYER_NORMS").is_some() {
                let h = exec.to_host(&sc.d_x)?;
                let n = h[..t_len * embd]
                    .iter()
                    .map(|v| (*v as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                tracing::info!("dbg layer {li:>2} |x| = {n:.6e}");
            }
        }

        // h_nextn rows (post-out_norm hidden at every position - the MTP head's h
        // inputs), then lm_head on the last row only.
        exec.rmsnorm_batch(&sc.d_x, &self.out_norm.buf, &mut sc.d_h, embd, eps, t_len)?;
        exec.copy_region(&sc.d_h, (t_len - 1) * embd, &mut sc.d_xn, 0, embd)?;
        if let Some(p) = super::head_f8(self.out_f8.as_ref(), 1) {
            // f8 head - the Q8_0 twin is dropped at load (REPLACE lane)
            super::head_f8_gemm(
                &exec,
                p,
                &sc.d_xn,
                &mut sc.d_pxq,
                &mut sc.d_exs,
                &mut sc.d_head_part,
                &mut sc.d_logits,
                1,
            )?;
        } else {
            super::stub_guard(&self.output, "record_prefill head")?;
            gemv_any(&exec, &self.output, &sc.d_xn, &mut sc.d_logits)?;
        }
        Ok(())
    }

    /// The greedy epilogue (device argmax of the final logits -> advance token/pos/
    /// mrope + append to the output ring). Captured at the tail of `graph_gen`.
    fn record_advance(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let sc = self.scratch.as_ref().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");
        let d_logits = &sc.d_logits;
        exec.argmax_advance(
            d_logits,
            vocab,
            &mut ds.d_pmax,
            &mut ds.d_pidx,
            &mut ds.d_token,
            &mut ds.d_pos,
            &mut ds.d_mrope,
            &mut ds.d_out,
            &mut ds.d_step,
        )?;
        Ok(())
    }

    /// The per-token compute, using device-resident inputs (`ds.d_token/d_pos/
    /// d_mrope/d_slots`). Only capturable ops (kernel launches + device-to-device
    /// copies; no host sync or allocation) so a CUDA graph can capture it.
    fn record_step(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (state_size, n_k_heads, n_v_heads, conv_k) =
            (self.state_size, self.n_k_heads, self.n_v_heads, self.conv_k);
        let (conv_dim, ff, max_ctx) = (self.conv_dim, self.ff, self.max_ctx);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let sinks = &self.sinks;
        let layers = &self.layers;
        let tok_embd = &self.tok_embd;
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");

        // embed the single token (device-resident id) into the residual stream
        embed_any(&exec, tok_embd, &ds.d_token, &mut sc.d_x, embd, 1)?;

        // b=1 lin-GEMV arm, same election as the batched path
        let lin_gemv_on =
            exec.has_f8lin_gemv() && paddock_models::dev_var_os!("PADDOCK_NO_LIN_GEMV").is_none();

        for (li, layer) in layers.iter().enumerate() {
            exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
            match &layer.mixer {
                Mixer::Full(w) => {
                    super::stub_guard(&w.wq, "forward.rs serial wq")?;
                    gemv_any(&exec, &w.wq, &sc.d_xn, &mut sc.d_qg)?;
                    exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, 1, n_heads, head_dim)?;
                    super::stub_guard(&w.wk, "forward.rs serial wk")?;
                    gemv_any(&exec, &w.wk, &sc.d_xn, &mut sc.d_k)?;
                    super::stub_guard(&w.wv, "forward.rs serial wv")?;
                    gemv_any(&exec, &w.wv, &sc.d_xn, &mut sc.d_v)?;
                    exec.rmsnorm_batch(
                        &sc.d_q,
                        &w.q_norm.buf,
                        &mut sc.d_qn,
                        head_dim,
                        eps,
                        n_heads,
                    )?;
                    exec.rmsnorm_batch(
                        &sc.d_k,
                        &w.k_norm.buf,
                        &mut sc.d_kn,
                        head_dim,
                        eps,
                        n_kv_heads,
                    )?;
                    exec.mrope(
                        &mut sc.d_qn,
                        &ds.d_mrope,
                        1,
                        n_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.d_kn,
                        &ds.d_mrope,
                        1,
                        n_kv_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    exec.kv_append_batch(
                        &sc.d_kn,
                        ds.kv_k[li].as_mut().expect("full-attn layer KV"),
                        &ds.d_pos,
                        Some(&ds.d_slots),
                        kv_dim,
                        max_ctx,
                        1,
                        self.kv_dtype,
                    )?;
                    exec.kv_append_batch(
                        &sc.d_v,
                        ds.kv_v[li].as_mut().expect("full-attn layer KV"),
                        &ds.d_pos,
                        Some(&ds.d_slots),
                        kv_dim,
                        max_ctx,
                        1,
                        self.kv_dtype,
                    )?;
                    // FlashDecoding split via attn_decode_dispatch. An A6000 trial of a
                    // fixed 8-way split REGRESSED at tg128 - that die
                    // keeps the single-pass walk (attn_splits gates on sm_count). On
                    // GB202 the unsplit 16-block grid measured 25-29 GB/s at every ctx
                    // (kbench: 2x at ctx=128 split, 10x at 8k) so big dies split always.
                    attn_decode_dispatch(
                        &exec,
                        &sc.d_qn,
                        ds.kv_k[li].as_ref().expect("full-attn layer KV"),
                        ds.kv_v[li].as_ref().expect("full-attn layer KV"),
                        sinks,
                        &mut sc.d_attn_o,
                        &mut sc.d_attn_ml,
                        &mut sc.d_attn,
                        &ds.d_pos,
                        Some(&ds.d_slots),
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        1,
                        scale,
                        self.kv_dtype,
                        None,
                    )?;
                    exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, q_dim)?;
                    super::stub_guard(&w.wo, "forward.rs serial wo")?;
                    gemv_any(&exec, &w.wo, &sc.d_attn, &mut sc.d_proj)?;
                }
                Mixer::Linear(w) => {
                    super::stub_guard(&w.in_qkv, "forward.rs serial in_qkv")?;
                    gemv_any(&exec, &w.in_qkv, &sc.d_xn, &mut sc.d_mixed)?;
                    exec.conv_step(
                        ds.conv_win[li].as_mut().expect("DeltaNet layer conv"),
                        &sc.d_mixed,
                        &w.conv_w.buf,
                        &mut sc.d_conv,
                        conv_dim,
                        conv_k,
                    )?;
                    exec.deltanet_split_gqa_norm(
                        &sc.d_conv,
                        &mut sc.d_dq,
                        &mut sc.d_dk,
                        &mut sc.d_dv,
                        1,
                        n_k_heads,
                        n_v_heads,
                        state_size,
                    )?;
                    // Fused: alpha·x, beta·x, and the gate math in one launch (the
                    // alpha/beta GEMVs are latency-bound out=n_v_heads skinny projs).
                    // Non-Q8 alpha/beta (UD F16) ride the f32 ab plane instead -
                    // same math, two launches.
                    if let (Some(aw), Some(bw)) = (&w.alpha_w, &w.beta_w) {
                        exec.deltanet_alpha_beta_gate(
                            aw,
                            bw,
                            &sc.d_xn,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            n_v_heads,
                        )?;
                    } else {
                        let ab = w
                            .ab_f32
                            .as_ref()
                            .expect("non-Q8 alpha/beta load the ab plane");
                        ab_gate(
                            &exec,
                            ab,
                            &sc.d_xn,
                            &mut sc.d_ab,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            1,
                            n_v_heads,
                        )?;
                    }
                    exec.gated_delta_recurrent_v2(
                        &sc.d_dq,
                        &sc.d_dk,
                        &sc.d_dv,
                        &sc.d_g,
                        &sc.d_beta,
                        None,
                        ds.recur[li].as_mut().expect("DeltaNet layer state"),
                        0,
                        None,
                        &mut sc.d_dattn,
                        1,
                        1,
                        n_v_heads,
                        state_size,
                    )?;
                    super::stub_guard(&w.gate_w, "forward.rs serial gate_w")?;
                    gemv_any(&exec, &w.gate_w, &sc.d_xn, &mut sc.d_z)?;
                    exec.gated_rmsnorm(
                        &sc.d_dattn,
                        &sc.d_z,
                        &w.ssm_norm.buf,
                        &mut sc.d_core,
                        n_v_heads,
                        state_size,
                        eps,
                    )?;
                    super::stub_guard(&w.out_w, "forward.rs serial out_w")?;
                    gemv_any(&exec, &w.out_w, &sc.d_core, &mut sc.d_proj)?;
                }
            }
            exec.add_rmsnorm_batch(
                &mut sc.d_x,
                &sc.d_proj,
                &layer.post_norm.buf,
                &mut sc.d_xn,
                embd,
                eps,
                1,
            )?;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    // sm_100 tcgen05 arm (PADDOCK_QWEN_F8T). Stands on its own:
                    // it must not sit under the F8D_B1 / bs_f8ffn arms, which
                    // are PPL-scoring and opt-in-plane modes respectively. When
                    // the tile planes are live the whole dense FFN decode rides
                    // f8t_gemm instead of the warp-level GEMV chain. gemma4's
                    // identical lane on this die, same Q8_0 file: 2.3x byte
                    // floor on f8t vs 4.7x on the warp path.
                    if let Some([gu_t, dn_t]) = self.bs_f8t_ffn.get(li).and_then(|o| o.as_ref()) {
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            embd,
                            1,
                        )?;
                        exec.f8t_gemm(
                            gu_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut sc.d_ffn_up,
                            &mut sc.d_f8_part,
                            embd,
                            2 * ff,
                            1,
                        )?;
                        exec.swiglu_fused(&sc.d_f8_part, &mut sc.d_ffn_gate, ff, 1)?;
                        exec.quantize_e4m3_row(
                            &sc.d_ffn_gate,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            ff,
                            1,
                        )?;
                        exec.f8t_gemm(
                            dn_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            embd,
                            1,
                        )?;
                        exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
                        continue;
                    }
                    // checkpoint-exact fp8 layer (the f8row class): three
                    // f32-in row GEMVs, no staging - see F8RowFfn.
                    if let Some(p) = self.bs_f8row_ffn.get(li).and_then(|o| o.as_ref()) {
                        super::ops::ffn_f8row_gemv(
                            &exec,
                            p,
                            &sc.d_xn,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                        )?;
                        exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
                        continue;
                    }
                    // PADDOCK_F8D_B1: the PPL-gate scoring mode - route the
                    // b=1 teacher-forced decode through the same e4m3 ks
                    // rung the b>=8 serving chain uses (BN16 tile at b=1),
                    // so qwen35_ppl measures exactly the serving numerics.
                    // Never a serving default (the b=1 GEMV is the fast path).
                    static F8D_B1: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    let f8b1 = *F8D_B1
                        .get_or_init(|| paddock_models::dev_var_os!("PADDOCK_F8D_B1").is_some());
                    if let Some([gu8, d8]) = self
                        .bs_f8ffn_bs
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .or_else(|| self.bs_f8ffn.get(li).and_then(|o| o.as_ref()))
                        .filter(|_| f8b1)
                    {
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, gu8.1)?;
                        // fused 2ff landing rides d_f8_part (8x out_max >= 2ff;
                        // nz=1 at 544 tiles so the ks part plane goes unused -
                        // d_ffn_up stands in as the never-touched part arg)
                        static F8R1: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let f8r1 = *F8R1.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_some()
                        });
                        if f8r1 {
                            exec.f8r_gemm_mma_ks(
                                &gu8.0,
                                gu8.1,
                                gu8.2,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_up,
                                &mut sc.d_f8_part,
                                1,
                            )?;
                        } else {
                            exec.f8d_gemm_mma_ks(
                                &gu8.0,
                                gu8.1,
                                gu8.2,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_up,
                                &mut sc.d_f8_part,
                                1,
                            )?;
                        }
                        if exec.has_swiglu_fused_e4m3() {
                            exec.swiglu_fused_e4m3(
                                &sc.d_f8_part,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                ff,
                                1,
                            )?;
                        } else {
                            exec.swiglu_fused(&sc.d_f8_part, &mut sc.d_ffn_gate, ff, 1)?;
                            exec.quantize_e4m3(&sc.d_ffn_gate, &mut sc.d_pxq, &mut sc.d_exs, ff)?;
                        }
                        if f8r1 {
                            exec.f8r_gemm_mma_ks(
                                &d8.0,
                                d8.1,
                                d8.2,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_f8_part,
                                &mut sc.d_proj,
                                1,
                            )?;
                        } else {
                            exec.f8d_gemm_mma_ks(
                                &d8.0,
                                d8.1,
                                d8.2,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_f8_part,
                                &mut sc.d_proj,
                                1,
                            )?;
                        }
                        exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
                        continue;
                    }
                    // b=1 lin arm (non-KV-overhead R2.4): the serial path is a
                    // live reader of the Q8_0 FFN planes - its own f8 arm above
                    // is opt-in (PADDOCK_F8D_B1) because routing b=1 onto the
                    // width GEMM loses 9%. The lin GEMV is built for exactly
                    // this shape, so wiring it here is what makes the Q8
                    // reclaim safe: with this, no path reads those planes.
                    // ticket=None pins nz=1 (no combine, y written directly);
                    // this is the max_batch=1 fallback, not the hot lane.
                    if lin_gemv_on
                        && let Some([gu8, d8]) = self
                            .bs_f8ffn
                            .get(li)
                            .and_then(|o| o.as_ref())
                            .filter(|p| p[0].0.is_lin() && p[1].0.is_lin())
                    {
                        exec.f8lin_gemv(
                            &gu8.0,
                            &sc.d_xn,
                            &mut sc.d_ffn_up,
                            &mut sc.d_f8_part,
                            None,
                            gu8.1,
                            gu8.2,
                        )?;
                        exec.swiglu_fused(&sc.d_f8_part, &mut sc.d_ffn_gate, ff, 1)?;
                        exec.f8lin_gemv(
                            &d8.0,
                            &sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            None,
                            d8.1,
                            d8.2,
                        )?;
                        exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
                        continue;
                    }
                    // NOTE: gate+up were tried fused (pd_q8_0_ffn_gate_up_swiglu) but it
                    // REGRESSED - these GEMVs are memory-bound (profiles at ~90% DRAM),
                    // and halving the block count starves the DRAM system of concurrency. Keep
                    // them split; only latency-bound small ops benefit from fusion.
                    gemv_any(&exec, gate, &sc.d_xn, &mut sc.d_ffn_gate)?;
                    gemv_any(&exec, up, &sc.d_xn, &mut sc.d_ffn_up)?;
                    exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, ff)?;
                    gemv_any(&exec, down, &sc.d_ffn_gate, &mut sc.d_proj)?;
                }
                Ffn::Nvf4Dense { gu, down } => {
                    // f8t tile arm first, off the planes load.rs builds from
                    // the NVFP4 checkpoint's own values (see the Dense arm
                    // above - same chain, same planes).
                    if let Some([gu_t, dn_t]) = self.bs_f8t_ffn.get(li).and_then(|o| o.as_ref()) {
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            embd,
                            1,
                        )?;
                        exec.f8t_gemm(
                            gu_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut sc.d_ffn_up,
                            &mut sc.d_f8_part,
                            embd,
                            2 * ff,
                            1,
                        )?;
                        exec.swiglu_fused(&sc.d_f8_part, &mut sc.d_ffn_gate, ff, 1)?;
                        exec.quantize_e4m3_row(
                            &sc.d_ffn_gate,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            ff,
                            1,
                        )?;
                        exec.f8t_gemm(
                            dn_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            embd,
                            1,
                        )?;
                    } else {
                        // W4A16 serial decode: rows=1 rides nvf4_ffn's exact
                        // fallback - same split-GEMV shape as the Q8 chain (the
                        // fused-gate note above applies - memory-bound)
                        nvf4_ffn(
                            &exec,
                            gu,
                            down,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_nvs,
                            &mut sc.d_nv4part,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_ffn_gu,
                            &mut sc.d_swq_q,
                            &mut sc.d_swq_s,
                            &mut sc.d_proj,
                            ff,
                            1,
                        )?;
                    }
                }
                Ffn::Moe(w) => {
                    moe_ffn(
                        &exec,
                        w,
                        self.moe.expect("moe dims"),
                        embd,
                        1,
                        true,
                        &sc.d_xn,
                        &mut sc.d_moe_xq,
                        &mut sc.d_moe_xs,
                        &mut sc.d_ssums,
                        &mut sc.d_moe_xs8,
                        &mut sc.d_moe_fs8,
                        &mut sc.d_moe_logits,
                        &sc.d_zero_bias,
                        &mut sc.d_moe_idx,
                        &mut sc.d_moe_w,
                        &mut sc.d_moe_fused,
                        &mut sc.d_moe_fq,
                        &mut sc.d_moe_fs,
                        &mut sc.d_moe_srow,
                        &mut sc.d_moe_sslot,
                        &mut sc.d_moe_bexp,
                        &mut sc.d_moe_part,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        &mut sc.d_skfix,
                        &mut sc.d_ffn_gate,
                        &mut sc.d_ffn_up,
                        &mut sc.d_mixed,
                        &mut sc.d_proj,
                    )?;
                }
            }
            exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
        }

        exec.rmsnorm_batch(&sc.d_x, &self.out_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
        // single token: d_xn's first row is the last position - lm_head directly.
        if let Some(p) = super::head_f8(self.out_f8.as_ref(), 1) {
            // f8 head - the Q8_0 twin is dropped at load (REPLACE lane)
            super::head_f8_gemm(
                &exec,
                p,
                &sc.d_xn,
                &mut sc.d_pxq,
                &mut sc.d_exs,
                &mut sc.d_head_part,
                &mut sc.d_logits,
                1,
            )?;
        } else {
            super::stub_guard(&self.output, "record_prefill head")?;
            gemv_any(&exec, &self.output, &sc.d_xn, &mut sc.d_logits)?;
        }
        Ok(())
    }

    /// Incremental decode step (Generator seam): advance the persistent per-layer
    /// state by exactly this one token and return its next-token logits. O(1) per
    /// token (constant work regardless of context length) - this is the fast path
    /// that replaces the O(T^2) recompute; correctness is pinned by the b9895
    /// greedy-parity gate.
    pub fn forward_one(&mut self, token: u32) -> Result<Vec<f32>, GpuModelError> {
        self.history.push(token);
        self.step(token)
    }

    /// True when a `rows`-row pass fits the CURRENT scratch (no realloc) -
    /// the overlapped-admission guard (see `Generator::prefill_scratch_fits`).
    pub fn prefill_scratch_fits(&self, rows: usize) -> bool {
        matches!(&self.scratch, Some(sc) if sc.cap >= rows)
    }

    /// (Re)allocate the per-pass scratch if it can't hold `t` tokens.
    pub(super) fn ensure_scratch(&mut self, t: usize) -> Result<(), GpuModelError> {
        if matches!(&self.scratch, Some(sc) if sc.cap >= t) {
            return Ok(());
        }
        // A (re)allocation moves the scratch buffer addresses a captured graph baked
        // in, so drop any graph - it will be recaptured against the new scratch.
        if let Some(ds) = self.decode.as_mut() {
            ds.graph = None;
            ds.graph_gen = None;
            ds.pf_graphs.clear();
        }
        if let Some(bs) = self.batch.as_mut() {
            bs.graphs.clear();
            bs.pf_pass_graphs.clear();
        }
        if let Some(sb) = self.spec_batch.as_mut() {
            sb.graph_draft.clear();
            sb.graph_verify.clear();
            sb.graph_commit.clear();
        }
        // Growth headroom: the unified tick's row count r = decode_rows +
        // prefill_share creeps by one as each admission joins decode, and a
        // bare cap = r turned every such tick into a multi-GB realloc + full
        // graph drop (observed as cap=4097,4098,... at ~600 ms intervals in
        // the pf8 serving log). Pad by max_batch so one grow covers every
        // future decode-row count at this prefill share.
        let headroom = self.batch.as_ref().map(|b| b.max_batch).unwrap_or(0);
        let cap = t.max(64) + headroom;
        let e = &self.exec;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        // Widest activation any prefill quantize (q8 / mmq) will see: the
        // projection INPUT dims differ per weight - wo reads q_dim, out_w reads
        // value_dim, ffn down reads ff/moe_ff - and several exceed embd. The
        // activation staging (d_x16/d_pxq/d_pxs/d_yq) must cover the true max, or
        // pd_quantize_q8_mmq (grid = ceil(in_dim/128) × round_up(batch,128), stride
        // 144 B/chunk) writes past the buffer. That OOB was long latent: the classic
        // prefill's large cap left pool slack after d_yq so the spill was benign;
        // the fused tick's small cap (unified_prefill_rows) put a live buffer there
        // and the spill corrupted it (garbled decode / illegal-address crash).
        let qw = self
            .embd
            .max(2 * self.embd) // eh_proj input (spec concat) - sums plane cover
            .max(self.ff)
            .max(q_dim)
            .max(self.value_dim)
            .max(self.moe.map_or(0, |m| m.moe_ff));
        let v_scratch0 = cudarc::driver::result::mem_get_info()
            .map(|(f, _)| f)
            .unwrap_or(0);
        // the f8row FFN lane stages per-row e4m3 at the wave's own row count
        // through the same pair (d_f8t_q/d_f8t_rs), so it sizes them to cap too
        let f8row_lane = self.bs_f8row_ffn.iter().any(|o| o.is_some());
        self.scratch = Some(Scratch {
            cap,
            d_x: e.alloc(cap * self.embd)?,
            d_xn: e.alloc(cap * self.embd)?,
            d_proj: e.alloc(cap * self.embd)?,
            // widened past 2*q_dim: the Phase A fused-plane prefill path
            // lands the whole [q|gate|k|v] one-GEMM row (2q + 2kv) here
            d_qg: e.alloc(cap * (2 * q_dim + 2 * kv_dim))?,
            d_q: e.alloc(cap * q_dim)?,
            d_qn: e.alloc(cap * q_dim)?,
            d_gate: e.alloc(cap * q_dim)?,
            d_k: e.alloc(cap * kv_dim)?,
            d_kn: e.alloc(cap * kv_dim)?,
            d_v: e.alloc(cap * kv_dim)?,
            d_attn: e.alloc(cap * q_dim)?,
            // split-attention partials: splitting engages only while base =
            // n_heads*rows < 2*fill_blocks, so partial rows are bounded by
            // 2*fill*MAX_ATTN_SPLITS regardless of batch/row count (~37 MB at
            // 188 SMs, head_dim 256).
            d_attn_o: e.alloc(
                2 * attn_fill_blocks(super::attn_boundary_sms(e.sm_count()))
                    * MAX_ATTN_SPLITS
                    * self.head_dim,
            )?,
            d_attn_ml: e.alloc(
                2 * attn_fill_blocks(super::attn_boundary_sms(e.sm_count())) * MAX_ATTN_SPLITS * 2,
            )?,
            d_mixed: e.alloc(cap * self.conv_dim)?,
            d_conv: e.alloc(cap * self.conv_dim)?,
            d_dq: e.alloc(cap * self.value_dim)?,
            d_dk: e.alloc(cap * self.value_dim)?,
            d_dv: e.alloc(cap * self.value_dim)?,
            // compact q/k: HK * s bf16 per row = (conv_dim - value_dim) / 2 halves
            d_dqc: e.alloc(if super::dn_qkc(e) {
                cap * (self.conv_dim - self.value_dim) / 4
            } else {
                1
            })?,
            d_dkc: e.alloc(if super::dn_qkc(e) {
                cap * (self.conv_dim - self.value_dim) / 4
            } else {
                1
            })?,
            d_row0s_zero: e.alloc_u32(cap)?,
            d_a: e.alloc(cap * self.n_v_heads)?,
            d_b: e.alloc(cap * self.n_v_heads)?,
            d_ab: e.alloc(cap * 2 * self.n_v_heads)?,
            d_g: e.alloc(cap * self.n_v_heads)?,
            d_beta: e.alloc(cap * self.n_v_heads)?,
            d_dattn: e.alloc(cap * self.value_dim)?,
            d_z: e.alloc(cap * self.value_dim)?,
            d_core: e.alloc(cap * self.value_dim)?,
            // +32 chunks over cap/64: the varlen chunked-GDN launch packs
            // every span's chunks side by side, and each span's partial
            // last chunk pads up to one extra chunk (32 spans max)
            d_dnc_dw: e.alloc((cap.div_ceil(64) + 32) * self.n_v_heads * 64 * self.state_size)?,
            d_dnc_du: e.alloc((cap.div_ceil(64) + 32) * self.n_v_heads * 64 * self.state_size)?,
            d_dnc_coef: e.alloc((cap.div_ceil(64) + 32) * self.n_v_heads * 64 * 64)?,
            d_dnc_cg: e.alloc_f64((cap.div_ceil(64) + 32) * self.n_v_heads * 64)?,
            d_ffn_gate: e.alloc(cap * self.ff)?,
            // 128 rows: the wide band (>= 128) never lands (SWQ epilogue) - unless
            // PADDOCK_NO_NVF4_SWQ pins the landing path for A/B, then every width
            d_ffn_gu: e.alloc(if self.nvf4_gu_fused() {
                let rows = if paddock_models::dev_var_os!("PADDOCK_NO_NVF4_SWQ").is_some() {
                    cap
                } else {
                    cap.min(128)
                };
                rows * 2 * self.ff
            } else {
                1
            })?,
            d_swq_q: e.alloc_i8(if self.nvf4_gu_fused() {
                cap * self.ff / 2
            } else {
                1
            })?,
            d_swq_s: e.alloc_u8(if self.nvf4_gu_fused() {
                cap * self.ff / 16
            } else {
                1
            })?,
            d_moe_logits: e.alloc(self.moe.map_or(1, |m| cap * m.n_expert))?,
            d_moe_idx: e.alloc_u32(self.moe.map_or(1, |m| cap * m.n_active))?,
            d_moe_w: e.alloc(self.moe.map_or(1, |m| cap * m.n_active))?,
            // fused rows are SORTED-padded: max_blocks*BM >= cap*n_active
            // (per-expert tails pad up to BM-1 rows each). Sized for the WIDEST
            // block tile the MoE mma can pick (BM=64, the wider prefill variant)
            // so both BM=32 and BM=64 layouts fit; the 64-padding is the coarser
            // (larger) bound, a safe superset of the 32 case.
            d_moe_fused: e.alloc(self.moe.map_or(1, |m| {
                (cap * m.n_active + m.n_expert * 63).div_ceil(64) * 64 * m.moe_ff
            }))?,
            d_moe_fq: e.alloc_i8(self.moe.map_or(1, |m| {
                (cap * m.n_active + m.n_expert * 63).div_ceil(64) * 64 * m.moe_ff
            }))?,
            d_moe_fs: e.alloc(self.moe.map_or(1, |m| {
                (cap * m.n_active + m.n_expert * 63).div_ceil(64) * 64 * m.moe_ff / 32
            }))?,
            d_moe_fs8: e.alloc_u8(self.moe.map_or(1, |m| {
                (cap * m.n_active + m.n_expert * 63).div_ceil(64) * 64 * m.moe_ff / 32
            }))?,
            d_moe_xq: e.alloc_i8(if self.moe.is_some() {
                cap * self.embd
            } else {
                1
            })?,
            d_moe_xs: e.alloc(if self.moe.is_some() {
                cap * self.embd / 32
            } else {
                1
            })?,
            d_moe_xs8: e.alloc_u8(if self.moe.is_some() {
                cap * self.embd / 32
            } else {
                1
            })?,
            d_zero_bias: e.alloc(self.moe.map_or(1, |m| m.n_expert))?,
            d_moe_srow: e.alloc_u32(self.moe.map_or(1, |m| {
                (cap * m.n_active + m.n_expert * 63).div_ceil(64) * 64
            }))?,
            d_moe_sslot: e.alloc_u32(self.moe.map_or(1, |m| {
                (cap * m.n_active + m.n_expert * 63).div_ceil(64) * 64
            }))?,
            d_moe_bexp: e.alloc_u32(self.moe.map_or(1, |m| {
                // block_expert is [max_blocks] (no BM factor). Unlike srow/sslot/
                // fused above - whose *64 sizing already dominates the BM=32 need -
                // this array must hold the LARGER block count, and BM=32 packs more
                // blocks than BM=64: max_blocks ≈ cap*n_active/bm + n_expert, so the
                // smaller tile wins. Size for BM=32 (the default path); undersizing
                // OOBs pd_moe_align's block_expert write.
                (cap * m.n_active + m.n_expert * 31).div_ceil(32)
            }))?,
            d_moe_part: e.alloc(self.moe.map_or(1, |m| cap * m.n_active * self.embd))?,
            d_ffn_up: e.alloc(cap * self.ff)?,
            d_f8_part: e.alloc(8 * self.ks_out_max())?,
            d_pxq: e.alloc_i8(cap * qw)?,
            // f8t needs its own activation scratch, not d_pxq: f8t_gemm's TMA
            // boxes are 64 rows wide and read past `batch`, so the buffer must
            // hold >= 64 rows regardless of cap (cap is max_batch, 8 in the
            // smoke lane - d_pxq would be read out of bounds). Stale tail rows
            // only feed D columns the epilogue never stores. The chunk arm
            // extends the same contract to tc5r, whose TMA pad is 256 rows -
            // size to the arm's ceiling rounded up to that pad.
            // the PREFILL fp4 arm lands `down`'s ff-wide e4m3
            // activations here at the wave's own row count, not the decode
            // band's - so the buffer has to follow cap the way d_pxq already
            // does. Same magnitude as d_pxq (cap * qw), and it is only taken
            d_f8t_q: e.alloc_i8(
                super::batch::f8t_chunk_rmax()
                    .next_multiple_of(256)
                    .max(64)
                    .max(if f8row_lane {
                        cap.next_multiple_of(256)
                    } else {
                        0
                    })
                    * qw,
            )?,
            d_f8t_rs: e.alloc(
                super::batch::f8t_chunk_rmax()
                    .next_multiple_of(256)
                    .max(64)
                    .max(if f8row_lane {
                        cap.next_multiple_of(256)
                    } else {
                        0
                    }),
            )?,
            d_pxs: e.alloc(cap * qw / 32)?,
            d_exs: e.alloc_u8(cap * qw / 32)?,
            d_nvs: e.alloc_u8(cap * qw / 16)?,
            // split-K partials for the checkpoint W4A4 GEMM (
            // the split engages only on tile grids under 64 CTAs,
            // which in this graph is the FFN down plane (out = embd) at
            // decode-band batches (rows <= 128) - sk(4) x 128 x embd f32
            d_nv4part: e.alloc(
                if self
                    .layers
                    .iter()
                    .any(|l| matches!(l.ffn, Ffn::Nvf4Dense { .. }))
                {
                    4 * 128 * self.embd
                } else {
                    1
                },
            )?,
            d_yq: e.alloc_u8(qw.div_ceil(128) * cap.next_multiple_of(128) * 144)?,
            d_xsums: e.alloc(if self.kq_resident {
                qw.div_ceil(128) * cap.next_multiple_of(128) * 4
            } else {
                1
            })?,
            // k-quant per-16 sums plane. The sorted kq-MoE down stage sums the
            // SORTED fq rows - max_blocks*BM exceeds cap*n_active (per-expert
            // tails pad up to BM-1 rows each), so the MoE bound rides the same
            // 64-padded superset as d_moe_fq above.
            d_ssums: e.alloc(if self.kq_resident {
                (cap * qw / 16).max(self.moe.map_or(0, |m| {
                    (cap * m.n_active + m.n_expert * 63).div_ceil(64) * 64 * m.moe_ff / 16
                }))
            } else {
                1
            })?,
            d_skfix: e.alloc(256 * 128 * 128 + 256)?, // +256: stream-k fold flags tail
            d_logits: e.alloc(self.vocab)?,
            // K-split partials for the ONE-ROW f8 lm_head on the paths that own
            // no BatchState: multimodal.rs's vision prefill and forward.rs's
            // record_prefill/record_step (the ppl + profile + test harnesses).
            // f8_gemm_lin's ABI wants >= 8 * out_dim * batch f32, and the head
            // is the widest out_dim there is, so bs.d_ks_part's 8*64*ks_out_max
            // sizing (which tops out at 2*ff and never saw the vocab) is not
            // what these sites can borrow. 8 * vocab f32 is ~8 MB against a
            // 41 GB resident model - the price of letting the head REPLACE lane
            // hold on every path instead of keeping a 1.35 GB Q8 twin alive for
            // three call sites.
            d_head_part: e.alloc(8 * self.vocab)?,
            d_h: e.alloc(cap * self.embd)?,
            d_wdq: e.alloc(
                if self.kq_resident
                    && paddock_models::dev_var_os!("PADDOCK_KQ_F32_PREFILL").is_some()
                {
                    self.kq_max_elems.max(1)
                } else {
                    1
                },
            )?,
        });
        let v_scratch1 = cudarc::driver::result::mem_get_info()
            .map(|(f, _)| f)
            .unwrap_or(0);
        tracing::info!(
            "qwen35 VRAM  prefill scratch (cap={cap} rows)     {:>7.2} GB",
            v_scratch0.saturating_sub(v_scratch1) as f64 / 1e9
        );
        Ok(())
    }

    /// Allocate the speculative-decoding scratch (rollback + MTP staging) if absent.
    pub(super) fn ensure_spec(&mut self) -> Result<(), GpuModelError> {
        if self.spec.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let (mut recur_snap, mut conv_ext) = (
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
        );
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Linear(_) => {
                    recur_snap.push(Some(e.alloc(
                        SPEC_ROWS * self.n_v_heads * self.state_size * self.state_size,
                    )?));
                    conv_ext.push(Some(
                        e.alloc((self.conv_k - 1 + SPEC_ROWS) * self.conv_dim)?,
                    ));
                }
                Mixer::Full(_) => {
                    recur_snap.push(None);
                    conv_ext.push(None);
                }
            }
        }
        self.spec = Some(SpecState {
            recur_snap,
            conv_ext,
            d_logits_chunk: e.alloc(SPEC_ROWS * self.vocab)?,
            d_h_chunk: e.alloc(SPEC_ROWS * self.embd)?,
            d_concat: e.alloc(WARM_CHUNK * 2 * self.embd)?,
            d_e: e.alloc(WARM_CHUNK * self.embd)?,
            d_en: e.alloc(WARM_CHUNK * self.embd)?,
            d_hn: e.alloc(WARM_CHUNK * self.embd)?,
            d_hin: e.alloc(WARM_CHUNK * self.embd)?,
            d_hout: e.alloc(WARM_CHUNK * self.embd)?,
            d_mtp_tok: e.alloc_u32(WARM_CHUNK)?,
            d_xq: e.alloc_i8(SPEC_ROWS * self.ff.max(2 * self.embd))?,
            d_ks_part: e.alloc(8 * 64 * self.ks_out_max())?,
            d_xs: e.alloc(SPEC_ROWS * self.ff.max(2 * self.embd) / 32)?,
        });
        Ok(())
    }

    /// Greedy decode `max_new` tokens from `prompt`. Prefill runs the host `step`
    /// path (short, and it captures the base graph); generation runs the
    /// **graph-resident** loop: the captured `graph_gen` (compute + on-device
    /// argmax/advance) is replayed back-to-back in chunks with no per-token host
    /// round-trip - only one sync + one small readback per `GEN_CHUNK` tokens. This
    /// removes the ~1 ms/token host bubble (sync + 993 KB logits D2H + CPU argmax)
    /// that a per-token loop pays. `stop` ends early (at chunk granularity, trimmed).
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        stop: Option<u32>,
    ) -> Result<Vec<u32>, GpuModelError> {
        assert!(!prompt.is_empty(), "empty prompt");
        assert!(max_new > 0, "max_new must be > 0");
        self.reset();
        // Batched prefill: one GEMM pass per weight over the whole prompt.
        let last = self.prefill(prompt)?;
        let mut out = Vec::with_capacity(max_new);
        let token0 = argmax(&last);
        out.push(token0);
        if Some(token0) == stop || max_new == 1 {
            return Ok(out);
        }

        let p = self.decode.as_ref().expect("decode").pos; // = prompt.len()
        assert!(
            p + max_new <= self.max_ctx,
            "context {p} + {max_new} exceeds max_ctx {}",
            self.max_ctx
        );
        let exec = self.exec.clone();
        // Seed the device inputs for the first generation replay: process token0 at
        // position p. Subsequent tokens/positions are produced on-device by graph_gen.
        {
            let ds = self.decode.as_mut().expect("decode");
            let mp = ds.mrope_pos as u32;
            let e = |x: cudarc::driver::DriverError| crate::gpu::from_driver(x);
            exec.stream
                .memcpy_htod(&[token0], &mut ds.d_token)
                .map_err(e)?;
            exec.stream
                .memcpy_htod(&[p as u32], &mut ds.d_pos)
                .map_err(e)?;
            exec.stream
                .memcpy_htod(&[mp; 4], &mut ds.d_mrope)
                .map_err(e)?;
        }
        if self.decode.as_ref().expect("decode").graph_gen.is_none() {
            self.capture_graph_gen()?;
        }

        // Graph-resident generation: launch chunks of replays, then read the chunk's
        // tokens out in one shot. Each replay appends its token to d_out[d_step++].
        let target = max_new - 1; // token0 already emitted
        let mut produced = 0usize;
        while produced < target {
            let k = (target - produced).min(GEN_CHUNK);
            {
                let ds = self.decode.as_mut().expect("decode");
                let e = |x: cudarc::driver::DriverError| crate::gpu::from_driver(x);
                exec.stream
                    .memcpy_htod(&[0u32], &mut ds.d_step)
                    .map_err(e)?; // reset ring
                let g = ds.graph_gen.as_ref().expect("gen graph captured above");
                for _ in 0..k {
                    g.0.launch()
                        .map_err(|x| GpuError::Driver(format!("gen launch: {x}")))?;
                }
            }
            let ids = exec.to_host_u32(&self.decode.as_ref().expect("decode").d_out)?;
            for &id in ids.iter().take(k) {
                out.push(id);
                produced += 1;
                if Some(id) == stop {
                    self.decode.as_mut().expect("decode").pos = p + produced;
                    return Ok(out);
                }
            }
        }
        self.decode.as_mut().expect("decode").pos = p + produced;
        Ok(out)
    }

    /// Low-variance per-token GPU-time probe for kernel tuning. Prefills `prompt`
    /// (capturing the base per-token graph), warms to steady boost clock, then times
    /// `iters` back-to-back graph replays and returns the **min** ms/token over a few
    /// batches (min = the peak-clock batch, reproducible even though we can't lock
    /// clocks). Each replay is identical constant work (fixed device position), so
    /// this isolates kernel time from a throughput benchmark's boost/thermal noise.
    pub fn bench_decode_ms(
        &mut self,
        prompt: &[u32],
        warmup: usize,
        iters: usize,
    ) -> Result<f64, GpuModelError> {
        assert!(!prompt.is_empty(), "empty prompt");
        self.reset();
        for &t in prompt {
            self.step(t)?; // prefill: captures the base graph + fills state
        }
        let exec = self.exec.clone();
        let drv = |x: cudarc::driver::DriverError| crate::gpu::from_driver(x);
        let launch = |m: &Self| -> Result<(), GpuModelError> {
            m.decode
                .as_ref()
                .expect("decode")
                .graph
                .as_ref()
                .expect("step captured the base graph")
                .0
                .launch()
                .map_err(|x| GpuError::Driver(format!("bench launch: {x}")))?;
            Ok(())
        };
        for _ in 0..warmup {
            launch(self)?;
        }
        exec.stream.synchronize().map_err(drv)?;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            exec.stream.synchronize().map_err(drv)?;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch(self)?;
            }
            exec.stream.synchronize().map_err(drv)?;
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            best = best.min(ms);
        }
        Ok(best)
    }

    /// Number of full-attention (KV-cached) layers vs DeltaNet (recurrent) layers.
    pub fn layer_counts(&self) -> (usize, usize) {
        let full = self.layers.iter().filter(|l| l.is_full()).count();
        (full, self.n_layers - full)
    }

    /// Human-readable geometry line - used by the load test / diagnostics.
    pub fn geometry(&self) -> String {
        let (full, linear) = self.layer_counts();
        format!(
            "qwen35: {} layers ({linear} DeltaNet + {full} full-attn), embd {}, \
             {}Q/{}KV hd {}, ff {}, DeltaNet(s{} k{} v{} conv{} dim{}), \
             n_rot {} sections {:?}, vocab {}",
            self.n_layers,
            self.embd,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.ff,
            self.state_size,
            self.n_k_heads,
            self.n_v_heads,
            self.conv_k,
            self.conv_dim,
            self.n_rot,
            self.sections,
            self.vocab,
        )
    }
}
