//! Qwen3.5/3.6 chunked prefill + MTP speculative decode (B=1 and batch).

use super::*;
use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

impl GpuQwen35 {
    /// Transient device buffers for a batched pass at positions `start..start+r`
    /// (text mrope: all four axes = position). Not graph-captured - alloc is fine.
    fn chunk_inputs(
        &self,
        start: usize,
        r: usize,
    ) -> Result<(CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u32>), GpuModelError> {
        let exec = &self.exec;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let pos_host: Vec<u32> = (start as u32..(start + r) as u32).collect();
        let mrope_host: Vec<u32> = (0..4).flat_map(|_| pos_host.iter().copied()).collect();
        let mut d_pos = exec.alloc_u32(r)?;
        let d_slots = exec.alloc_u32(r)?; // zeroed -> slot 0
        let mut d_mrope = exec.alloc_u32(4 * r)?;
        exec.stream
            .memcpy_htod(&pos_host, &mut d_pos)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&mrope_host, &mut d_mrope)
            .map_err(drv)?;
        Ok((d_pos, d_slots, d_mrope))
    }

    /// Target verify pass for speculative decoding: run `tokens` (1 committed +
    /// drafts) through the backbone at positions `pos..pos+r` without committing -
    /// the recurrence advances but snapshots every token's state
    /// (`spec.recur_snap`), the conv window is read via the extended-rows buffer
    /// (`spec.conv_ext`) and left untouched, and `ds.pos` stays put until
    /// `commit_chunk` rolls state to the accepted row. Emits logits + h for every
    /// row (`spec.d_logits_chunk` / `d_h_chunk`).
    fn forward_chunk(&mut self, tokens: &[u32]) -> Result<(), GpuModelError> {
        let r = tokens.len();
        assert!(
            r > 0 && r <= SPEC_ROWS,
            "chunk of {r} rows exceeds SPEC_ROWS"
        );
        self.ensure_scratch(r.max(1))?;
        self.ensure_spec()?;
        let start = self.decode.as_ref().expect("decode").pos;
        assert!(start + r <= self.max_ctx, "chunk exceeds max_ctx");
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let (d_pos, d_slots, d_mrope) = self.chunk_inputs(start, r)?;
        let mut d_tokens = exec.alloc_u32(r)?;
        exec.stream
            .memcpy_htod(tokens, &mut d_tokens)
            .map_err(drv)?;

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
        let km1 = conv_k - 1;

        let sinks = &self.sinks;
        let layers = &self.layers;
        let tok_embd = &self.tok_embd;
        // e4m3 dense-FFN twins. This WALK had no f8 ARM at all, and the Q8_0
        // reclaim stubs exactly the planes its `Ffn::Dense` arm
        // reads to 32 bytes -- so on the default build every `--spec auto`
        // request died with CUDA_ERROR_ILLEGAL_ADDRESS. The reclaim's consumer
        // audit listed decode/prefill/vision/forward.rs and simply did not
        // include spec.rs; the fix that followed it was then verified on the
        // nospec variant only, so nothing caught it. Bound here, before the
        // &mut borrows below, the same way record_spec_verify does it.
        let bs_f8ffn = &self.bs_f8ffn;
        let bs_f8ffn_bs = &self.bs_f8ffn_bs;
        // the checkpoint-exact fp8 layers (F8RowFfn) build no lin twin
        let bs_f8row = &self.bs_f8row_ffn;
        let out_f8 = self.out_f8.as_ref();
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");
        let sp = self.spec.as_mut().expect("spec");

        embed_any(&exec, tok_embd, &d_tokens, &mut sc.d_x, embd, r)?;

        for (li, layer) in layers.iter().enumerate() {
            exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, r)?;
            match &layer.mixer {
                Mixer::Full(w) => {
                    super::stub_guard(&w.wq, "spec.rs forward_chunk wq")?;
                    mmq(
                        &exec,
                        &w.wq,
                        &sc.d_xn,
                        &mut sp.d_xq,
                        &mut sp.d_xs,
                        &mut sc.d_ssums,
                        &mut sp.d_ks_part,
                        &mut sc.d_qg,
                        r,
                    )?;
                    exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                    super::stub_guard(&w.wk, "spec.rs forward_chunk wk")?;
                    mmq_pre_any(
                        &exec,
                        &w.wk,
                        &sp.d_xq,
                        &sp.d_xs,
                        &mut sc.d_ssums,
                        &mut sp.d_ks_part,
                        &mut sc.d_k,
                        r,
                    )?;
                    super::stub_guard(&w.wv, "spec.rs forward_chunk wv")?;
                    mmq_pre_any(
                        &exec,
                        &w.wv,
                        &sp.d_xq,
                        &sp.d_xs,
                        &mut sc.d_ssums,
                        &mut sp.d_ks_part,
                        &mut sc.d_v,
                        r,
                    )?;
                    exec.rmsnorm_batch(
                        &sc.d_q,
                        &w.q_norm.buf,
                        &mut sc.d_qn,
                        head_dim,
                        eps,
                        r * n_heads,
                    )?;
                    exec.rmsnorm_batch(
                        &sc.d_k,
                        &w.k_norm.buf,
                        &mut sc.d_kn,
                        head_dim,
                        eps,
                        r * n_kv_heads,
                    )?;
                    exec.mrope(
                        &mut sc.d_qn,
                        &d_mrope,
                        r,
                        n_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.d_kn,
                        &d_mrope,
                        r,
                        n_kv_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    exec.kv_append_batch(
                        &sc.d_kn,
                        ds.kv_k[li].as_mut().expect("full-attn layer KV"),
                        &d_pos,
                        Some(&d_slots),
                        kv_dim,
                        max_ctx,
                        r,
                        self.kv_dtype,
                    )?;
                    exec.kv_append_batch(
                        &sc.d_v,
                        ds.kv_v[li].as_mut().expect("full-attn layer KV"),
                        &d_pos,
                        Some(&d_slots),
                        kv_dim,
                        max_ctx,
                        r,
                        self.kv_dtype,
                    )?;
                    attn_decode_dispatch(
                        &exec,
                        &sc.d_qn,
                        ds.kv_k[li].as_ref().expect("full-attn layer KV"),
                        ds.kv_v[li].as_ref().expect("full-attn layer KV"),
                        sinks,
                        &mut sc.d_attn_o,
                        &mut sc.d_attn_ml,
                        &mut sc.d_attn,
                        &d_pos,
                        Some(&d_slots),
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        r,
                        scale,
                        self.kv_dtype,
                        None,
                    )?;
                    exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, r * q_dim)?;
                    super::stub_guard(&w.wo, "spec.rs forward_chunk wo")?;
                    mmq(
                        &exec,
                        &w.wo,
                        &sc.d_attn,
                        &mut sp.d_xq,
                        &mut sp.d_xs,
                        &mut sc.d_ssums,
                        &mut sp.d_ks_part,
                        &mut sc.d_proj,
                        r,
                    )?;
                }
                Mixer::Linear(w) => {
                    super::stub_guard(&w.in_qkv, "spec.rs forward_chunk in_qkv")?;
                    mmq(
                        &exec,
                        &w.in_qkv,
                        &sc.d_xn,
                        &mut sp.d_xq,
                        &mut sp.d_xs,
                        &mut sc.d_ssums,
                        &mut sp.d_ks_part,
                        &mut sc.d_mixed,
                        r,
                    )?;
                    // extended pre-conv rows: window prefix + this chunk's mixed rows.
                    {
                        let ext = sp.conv_ext[li].as_mut().expect("DeltaNet layer conv ext");
                        exec.copy_region(
                            ds.conv_win[li].as_ref().expect("DeltaNet layer conv"),
                            0,
                            ext,
                            0,
                            km1 * conv_dim,
                        )?;
                        exec.copy_region(&sc.d_mixed, 0, ext, km1 * conv_dim, r * conv_dim)?;
                    }
                    // conv over the extended rows; keep only the real tokens' outputs
                    // (rows km1..) - d_mixed is free again (its rows live in ext now).
                    exec.causal_conv1d_silu(
                        sp.conv_ext[li].as_ref().expect("DeltaNet layer conv ext"),
                        &w.conv_w.buf,
                        &mut sc.d_conv,
                        km1 + r,
                        conv_dim,
                        conv_k,
                    )?;
                    exec.copy_region(&sc.d_conv, km1 * conv_dim, &mut sc.d_mixed, 0, r * conv_dim)?;
                    exec.deltanet_split_gqa_norm(
                        &sc.d_mixed,
                        &mut sc.d_dq,
                        &mut sc.d_dk,
                        &mut sc.d_dv,
                        r,
                        n_k_heads,
                        n_v_heads,
                        state_size,
                    )?;
                    if let (Some(aw), Some(bw)) = (w.alpha_w.as_ref(), w.beta_w.as_ref()) {
                        mmq_pre(
                            &exec,
                            aw,
                            &sp.d_xq,
                            &sp.d_xs,
                            &mut sp.d_ks_part,
                            &mut sc.d_a,
                            r,
                        )?;
                        mmq_pre(
                            &exec,
                            bw,
                            &sp.d_xq,
                            &sp.d_xs,
                            &mut sp.d_ks_part,
                            &mut sc.d_b,
                            r,
                        )?;
                        exec.delta_gate(
                            &sc.d_a,
                            &sc.d_b,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            r,
                            n_v_heads,
                        )?;
                    } else {
                        // non-Q8 alpha/beta (k-quant files): the exact f32 ab
                        // route - P6b decay-numerics rule, same as serving
                        let ab = w.ab_f32.as_ref().expect("ab_f32 (alpha_w is None)");
                        ab_gate(
                            &exec,
                            ab,
                            &sc.d_xn,
                            &mut sc.d_ab,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            r,
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
                        Some(sp.recur_snap[li].as_mut().expect("DeltaNet layer snapshot")),
                        &mut sc.d_dattn,
                        1,
                        r,
                        n_v_heads,
                        state_size,
                    )?;
                    super::stub_guard(&w.gate_w, "spec.rs forward_chunk gate_w")?;
                    mmq_pre_any(
                        &exec,
                        &w.gate_w,
                        &sp.d_xq,
                        &sp.d_xs,
                        &mut sc.d_ssums,
                        &mut sp.d_ks_part,
                        &mut sc.d_z,
                        r,
                    )?;
                    exec.gated_rmsnorm(
                        &sc.d_dattn,
                        &sc.d_z,
                        &w.ssm_norm.buf,
                        &mut sc.d_core,
                        r * n_v_heads,
                        state_size,
                        eps,
                    )?;
                    super::stub_guard(&w.out_w, "spec.rs forward_chunk out_w")?;
                    mmq(
                        &exec,
                        &w.out_w,
                        &sc.d_core,
                        &mut sp.d_xq,
                        &mut sp.d_xs,
                        &mut sc.d_ssums,
                        &mut sp.d_ks_part,
                        &mut sc.d_proj,
                        r,
                    )?;
                }
            }
            exec.add_rmsnorm_batch(
                &mut sc.d_x,
                &sc.d_proj,
                &layer.post_norm.buf,
                &mut sc.d_xn,
                embd,
                eps,
                r,
            )?;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    // f8 first, on the PREFILL floor -- this walk is a chunk
                    // prefill, so it is `r > f8_ffn_pf_min()` like every other
                    // prefill arm, not a bare literal. With the elected 0 that
                    // covers every r >= 1, which is what the reclaim's
                    // precondition demands before it stubs anything.
                    let f8 = bs_f8ffn_bs
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .or_else(|| bs_f8ffn.get(li).and_then(|o| o.as_ref()))
                        .filter(|_| r > super::f8_ffn_pf_min());
                    if let Some(pr) = bs_f8row.get(li).and_then(|o| o.as_ref()) {
                        // checkpoint-exact fp8 layer: every width armed, no
                        // twin, so the Q8_0 seats below are stubs
                        super::ops::ffn_f8row_rows(
                            &exec,
                            pr,
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else if let Some([gu8, d8]) = f8 {
                        // ROW-SLICED off the fused gate|up plane, the same
                        // shape prefix.rs's prefill arm runs (gate = rows
                        // [0,ffh), up = rows [ffh,2ffh)) - this walk has only
                        // the main scratch, no d_fused_land (that lives on
                        // SpecBatchState, not SpecState).
                        let ffh = gu8.2 / 2;
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * gu8.1)?;
                        exec.f8_gemm_w8(
                            &gu8.0,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_ffn_gate,
                            gu8.1,
                            ffh,
                            r,
                        )?;
                        exec.f8_gemm_w8(
                            &gu8.0,
                            ffh,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_ffn_up,
                            gu8.1,
                            ffh,
                            r,
                        )?;
                        exec.quantize_e4m3_swiglu(
                            &sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * d8.1,
                        )?;
                        exec.f8_gemm_w8(
                            &d8.0,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_proj,
                            d8.1,
                            d8.2,
                            r,
                        )?;
                    } else {
                        super::stub_guard(gate, "spec.rs forward_chunk dense FFN")?;
                        mmq(
                            &exec,
                            gate,
                            &sc.d_xn,
                            &mut sp.d_xq,
                            &mut sp.d_xs,
                            &mut sc.d_ssums,
                            &mut sp.d_ks_part,
                            &mut sc.d_ffn_gate,
                            r,
                        )?;
                        mmq_pre_any(
                            &exec,
                            up,
                            &sp.d_xq,
                            &sp.d_xs,
                            &mut sc.d_ssums,
                            &mut sp.d_ks_part,
                            &mut sc.d_ffn_up,
                            r,
                        )?;
                        exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, r * ff)?;
                        mmq(
                            &exec,
                            down,
                            &sc.d_ffn_gate,
                            &mut sp.d_xq,
                            &mut sp.d_xs,
                            &mut sc.d_ssums,
                            &mut sp.d_ks_part,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
                Ffn::Nvf4Dense { gu, down } => {
                    // exact family, not W4A4 - spec single-class rule (see
                    // the note on the batch-draft site above /)
                    nvf4_gu_swiglu(
                        &exec,
                        gu,
                        &sc.d_xn,
                        &mut sc.d_ffn_gate,
                        &mut sc.d_ffn_up,
                        &mut sc.d_ffn_gu,
                        ff,
                        r,
                    )?;
                    nvf4_mm(&exec, down, &sc.d_ffn_gate, &mut sc.d_proj, r)?;
                }
                Ffn::Moe(w) => {
                    // sorted_ok=true: the historic `false` pinned the
                    // token-batched dp4a MoE so batched verify matched the
                    // single-slot dev refs bit-for-bit - but it made verify the
                    // whole round (@ B=32: gate_up_dp4a 836us + down_dp4a
                    // 362us per layer ≈ 48ms of the 80ms round, vs the mma pair's
                    // 116+62us on the dense path). Serving holds no cross-batch
                    // numeric-class invariant (the dense path itself switches MoE
                    // class with batch), and per-(B,K) the class stays
                    // fixed/deterministic. Also lets the fp4 grouped MoE engage
                    // at verify under the serving env - the same class the dense
                    // decode runs, i.e. the acceptance-relevant one.
                    moe_ffn(
                        &exec,
                        w,
                        self.moe.expect("moe dims"),
                        embd,
                        r,
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
            exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
        }

        exec.rmsnorm_batch(&sc.d_x, &self.out_norm.buf, &mut sp.d_h_chunk, embd, eps, r)?;
        // head: had no f8 arm, so it read the Q8_0 plane the REPLACE lane
        // drops. Same omission as this walk's FFN arm (see above).
        if let Some(p) = super::head_f8(out_f8, r) {
            super::head_f8_gemm(
                &exec,
                p,
                &sp.d_h_chunk,
                &mut sc.d_pxq,
                &mut sc.d_exs,
                &mut sp.d_ks_part,
                &mut sp.d_logits_chunk,
                r,
            )?;
        } else {
            super::stub_guard(&self.output, "spec.rs forward_chunk head")?;
            mmq(
                &exec,
                &self.output,
                &sp.d_h_chunk,
                &mut sp.d_xq,
                &mut sp.d_xs,
                &mut sc.d_ssums,
                &mut sp.d_ks_part,
                &mut sp.d_logits_chunk,
                r,
            )?;
        }
        Ok(())
    }

    /// Commit the verify chunk at `committed` rows (accepted drafts + the
    /// correction context): recurrent state rolls back to the snapshot of the last
    /// committed row (no-op copy skipped when the whole chunk stands), the conv
    /// window re-slices from the extended rows, and the position advances. KV needs
    /// no rollback (stale cells past `pos` are overwritten before any later read).
    fn commit_chunk(&mut self, committed: usize, chunk_rows: usize) -> Result<(), GpuModelError> {
        assert!(committed >= 1 && committed <= chunk_rows);
        let exec = self.exec.clone();
        let (n_v, s, conv_dim, km1) = (
            self.n_v_heads,
            self.state_size,
            self.conv_dim,
            self.conv_k - 1,
        );
        let ds = self.decode.as_mut().expect("decode");
        let sp = self.spec.as_mut().expect("spec");
        let state_elems = n_v * s * s;
        for li in 0..self.n_layers {
            let Some(snap) = sp.recur_snap[li].as_ref() else {
                continue;
            };
            if committed < chunk_rows {
                exec.copy_region(
                    snap,
                    (committed - 1) * state_elems,
                    ds.recur[li].as_mut().expect("DeltaNet layer state"),
                    0,
                    state_elems,
                )?;
            }
            // window after `committed` rows = ext rows [committed, committed+km1)
            exec.copy_region(
                sp.conv_ext[li].as_ref().expect("DeltaNet layer conv ext"),
                committed * conv_dim,
                ds.conv_win[li].as_mut().expect("DeltaNet layer conv"),
                0,
                km1 * conv_dim,
            )?;
        }
        ds.pos += committed;
        ds.mrope_pos += committed;
        Ok(())
    }

    /// One MTP head pass over `r` rows: input = eh_proj(enorm(embed(tok)) ||
    /// hnorm(h)) already in `sc.d_x`; runs the block's full-attn layer (own KV) +
    /// FFN in-place (b9895 graph_mtp order). `head_norm_out`: also produce the
    /// post-shared_head_norm rows into `spec.d_hout` (the draft-time h chain and
    /// lm_head input).
    fn mtp_block_pass(
        &mut self,
        r: usize,
        pos_bufs: Option<(&CudaSlice<u32>, &CudaSlice<u32>, &CudaSlice<u32>)>,
        head_norm_out: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (ff, max_ctx) = (self.ff, self.max_ctx);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let sinks = &self.sinks;
        let m = self.mtp.as_ref().expect("mtp weights");
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");
        let sp = self.spec.as_mut().expect("spec");
        // None = the decode state's device inputs (the device-resident draft loop,
        // r == 1: argmax_advance bumps them between steps with no host round-trip).
        let (d_pos, d_slots, d_mrope) = match pos_bufs {
            Some(b) => b,
            None => (&ds.d_pos, &ds.d_slots, &ds.d_mrope),
        };

        exec.rmsnorm_batch(&sc.d_x, &m.attn_norm.buf, &mut sc.d_xn, embd, eps, r)?;
        mm(&exec, &m.attn.wq, &sc.d_xn, &mut sc.d_qg, r)?;
        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
        mm(&exec, &m.attn.wk, &sc.d_xn, &mut sc.d_k, r)?;
        mm(&exec, &m.attn.wv, &sc.d_xn, &mut sc.d_v, r)?;
        exec.rmsnorm_batch(
            &sc.d_q,
            &m.attn.q_norm.buf,
            &mut sc.d_qn,
            head_dim,
            eps,
            r * n_heads,
        )?;
        exec.rmsnorm_batch(
            &sc.d_k,
            &m.attn.k_norm.buf,
            &mut sc.d_kn,
            head_dim,
            eps,
            r * n_kv_heads,
        )?;
        exec.mrope(
            &mut sc.d_qn,
            d_mrope,
            r,
            n_heads,
            head_dim,
            n_rot,
            yarn,
            sections,
        )?;
        exec.mrope(
            &mut sc.d_kn,
            d_mrope,
            r,
            n_kv_heads,
            head_dim,
            n_rot,
            yarn,
            sections,
        )?;
        exec.kv_append_batch(
            &sc.d_kn,
            ds.mtp_kv_k.as_mut().expect("MTP block KV cache"),
            d_pos,
            Some(d_slots),
            kv_dim,
            max_ctx,
            r,
            self.kv_dtype,
        )?;
        exec.kv_append_batch(
            &sc.d_v,
            ds.mtp_kv_v.as_mut().expect("MTP block KV cache"),
            d_pos,
            Some(d_slots),
            kv_dim,
            max_ctx,
            r,
            self.kv_dtype,
        )?;
        attn_decode_dispatch(
            &exec,
            &sc.d_qn,
            ds.mtp_kv_k.as_ref().expect("MTP block KV cache"),
            ds.mtp_kv_v.as_ref().expect("MTP block KV cache"),
            sinks,
            &mut sc.d_attn_o,
            &mut sc.d_attn_ml,
            &mut sc.d_attn,
            d_pos,
            Some(d_slots),
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            kv_dim,
            r,
            scale,
            self.kv_dtype,
            None,
        )?;
        exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, r * q_dim)?;
        mm(&exec, &m.attn.wo, &sc.d_attn, &mut sc.d_proj, r)?;
        exec.add_rmsnorm_batch(
            &mut sc.d_x,
            &sc.d_proj,
            &m.post_norm.buf,
            &mut sc.d_xn,
            embd,
            eps,
            r,
        )?;
        match &m.ffn {
            Ffn::Dense { gate, up, down } => {
                mm(&exec, gate, &sc.d_xn, &mut sc.d_ffn_gate, r)?;
                mm(&exec, up, &sc.d_xn, &mut sc.d_ffn_up, r)?;
                exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, r * ff)?;
                mm(&exec, down, &sc.d_ffn_gate, &mut sc.d_proj, r)?;
            }
            Ffn::Nvf4Dense { gu, down } => {
                // stays on the exact scalar/tc family, not the W4A4 arm
                // the spec paths keep one numeric class so
                // draft and verify rows agree - flipping verify to lossy
                // e2m1 activations without an acceptance measurement risks
                // the fp8-KV-costs-acceptance failure mode
                nvf4_gu_swiglu(
                    &exec,
                    gu,
                    &sc.d_xn,
                    &mut sc.d_ffn_gate,
                    &mut sc.d_ffn_up,
                    &mut sc.d_ffn_gu,
                    ff,
                    r,
                )?;
                nvf4_mm(&exec, down, &sc.d_ffn_gate, &mut sc.d_proj, r)?;
            }
            Ffn::Moe(w) => {
                // sorted_ok=false: the MTP draft rows share the spec paths'
                // single-class rule
                moe_ffn(
                    &exec,
                    w,
                    self.moe.expect("moe dims"),
                    embd,
                    r,
                    false,
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
        exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
        if head_norm_out {
            exec.rmsnorm_batch(&sc.d_x, &m.head_norm.buf, &mut sp.d_hout, embd, eps, r)?;
        }
        Ok(())
    }

    /// Prepare the MTP inputs for `r` rows (tokens on device in `spec.d_mtp_tok`,
    /// h rows already in `spec.d_hin`) and project into `sc.d_x`:
    /// eh_proj(enorm(embed(tok)) || hnorm(h)), concat e-first per b9895.
    fn mtp_project_inputs(
        &mut self,
        r: usize,
        from_decode_token: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, eps) = (self.embd, self.rms_eps);
        let m = self.mtp.as_ref().expect("mtp weights");
        let tok_embd = &self.tok_embd;
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_ref().expect("decode");
        let sp = self.spec.as_mut().expect("spec");
        // the draft loop keeps the current token on device (argmax_advance wrote it)
        let toks = if from_decode_token {
            &ds.d_token
        } else {
            &sp.d_mtp_tok
        };
        embed_any(&exec, tok_embd, toks, &mut sp.d_e, embd, r)?;
        exec.rmsnorm_batch(&sp.d_e, &m.enorm.buf, &mut sp.d_en, embd, eps, r)?;
        exec.rmsnorm_batch(&sp.d_hin, &m.hnorm.buf, &mut sp.d_hn, embd, eps, r)?;
        for i in 0..r {
            exec.copy_region(&sp.d_en, i * embd, &mut sp.d_concat, i * 2 * embd, embd)?;
            exec.copy_region(
                &sp.d_hn,
                i * embd,
                &mut sp.d_concat,
                i * 2 * embd + embd,
                embd,
            )?;
        }
        mm(&exec, &m.eh_proj, &sp.d_concat, &mut sc.d_x, r)?;
        Ok(())
    }

    /// MTP process/catch-up (b9895 `process()`): decode `tokens` (positions
    /// `start..start+n`) through the MTP block to warm its KV, with the h inputs
    /// shifted right by one - row 0 pairs with `pending_h`, row i with h_src row
    /// i-1 - then advance `pending_h` to h_src's last row. `h_src(buffer, row0)`
    /// selects where the target h rows live (prefill's `sc.d_h` or the verify
    /// chunk's `spec.d_h_chunk`).
    fn mtp_process(
        &mut self,
        tokens: &[u32],
        start: usize,
        h_from_prefill: bool,
    ) -> Result<(), GpuModelError> {
        let n = tokens.len();
        assert!(n > 0);
        // the warm loop's mtp_block_pass_b reads the round's table MIRROR -
        // warm runs outside rounds (prefill hooks), so refresh it here or
        // the drafter KV appends land on whatever pages the last round saw
        self.stage_spec_tables()?;
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let embd = self.embd;
        let mut done = 0usize;
        while done < n {
            let r = (n - done).min(WARM_CHUNK);
            // stage h inputs: row 0 = pending_h (done==0) or h_src[done-1]; rows
            // 1..r = h_src[done .. done+r-1]
            {
                let sc = self.scratch.as_ref().expect("scratch");
                let ds = self.decode.as_ref().expect("decode");
                let sp = self.spec.as_mut().expect("spec");
                let h_src = if h_from_prefill {
                    &sc.d_h
                } else {
                    &sp.d_h_chunk
                };
                if done == 0 {
                    exec.copy_region(&ds.pending_h, 0, &mut sp.d_hin, 0, embd)?;
                } else {
                    exec.copy_region(h_src, (done - 1) * embd, &mut sp.d_hin, 0, embd)?;
                }
                if r > 1 {
                    exec.copy_region(h_src, done * embd, &mut sp.d_hin, embd, (r - 1) * embd)?;
                }
                exec.stream
                    .memcpy_htod(&tokens[done..done + r], &mut sp.d_mtp_tok)
                    .map_err(drv)?;
            }
            let (d_pos, d_slots, d_mrope) = self.chunk_inputs(start + done, r)?;
            self.mtp_project_inputs(r, false)?;
            self.mtp_block_pass(r, Some((&d_pos, &d_slots, &d_mrope)), false)?;
            done += r;
        }
        // pending_h := last target h row of this batch
        {
            let sc = self.scratch.as_ref().expect("scratch");
            let sp = self.spec.as_ref().expect("spec");
            let ds = self.decode.as_mut().expect("decode");
            let h_src = if h_from_prefill {
                &sc.d_h
            } else {
                &sp.d_h_chunk
            };
            exec.copy_region(h_src, (n - 1) * embd, &mut ds.pending_h, 0, embd)?;
        }
        Ok(())
    }

    /// Greedily draft up to `k` tokens with the MTP head (b9895 `draft()`, single
    /// non-chained head): step i decodes (tok, h) at position `pos+i`, where the
    /// first pair is (id_last, pending_h) and each next pair is (drafted token,
    /// the head's own post-shared_head_norm output). DEVICE-RESIDENT: the token
    /// stays on device between steps (`argmax_advance` picks it, bumps pos/mrope,
    /// and appends into the output ring) - one small readback for the whole loop
    /// instead of a logits download + full sync per draft.
    fn mtp_draft(&mut self, id_last: u32, k: usize) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let (embd, vocab) = (self.embd, self.vocab);
        // seed the device loop state: current token, position, step counter
        {
            let ds = self.decode.as_mut().expect("decode");
            let pos0 = ds.pos as u32;
            let mp = ds.mrope_pos as u32;
            exec.stream
                .memcpy_htod(&[id_last], &mut ds.d_token)
                .map_err(drv)?;
            exec.stream
                .memcpy_htod(&[pos0], &mut ds.d_pos)
                .map_err(drv)?;
            exec.stream
                .memcpy_htod(&[mp; 4], &mut ds.d_mrope)
                .map_err(drv)?;
            exec.stream
                .memcpy_htod(&[0u32], &mut ds.d_step)
                .map_err(drv)?;
        }
        for i in 0..k {
            {
                let ds = self.decode.as_ref().expect("decode");
                let sp = self.spec.as_mut().expect("spec");
                if i == 0 {
                    exec.copy_region(&ds.pending_h, 0, &mut sp.d_hin, 0, embd)?;
                } else {
                    // chain the head's own h output from the previous step
                    let (src, dst) = (&sp.d_hout, &mut sp.d_hin);
                    exec.copy_region(src, 0, dst, 0, embd)?;
                }
            }
            self.mtp_project_inputs(1, true)?;
            self.mtp_block_pass(1, None, true)?;
            // draft logits = shared lm_head over the head-normed row, then pick the
            // next token on device (writes d_token, bumps d_pos/d_mrope, appends
            // d_out[d_step++]) - no host round-trip inside the loop.
            {
                let sc = self.scratch.as_mut().expect("scratch");
                let sp = self.spec.as_ref().expect("spec");
                if let Some(p) = super::head_f8(self.out_f8.as_ref(), 1) {
                    super::head_f8_gemm(
                        &exec,
                        p,
                        &sp.d_hout,
                        &mut sc.d_pxq,
                        &mut sc.d_exs,
                        &mut sc.d_head_part,
                        &mut sc.d_logits,
                        1,
                    )?;
                } else {
                    super::stub_guard(&self.output, "spec.rs draft-chain head")?;
                    gemv_any(&exec, &self.output, &sp.d_hout, &mut sc.d_logits)?;
                }
            }
            {
                let sc = self.scratch.as_ref().expect("scratch");
                let ds = self.decode.as_mut().expect("decode");
                exec.argmax_advance(
                    &sc.d_logits,
                    vocab,
                    &mut ds.d_pmax,
                    &mut ds.d_pidx,
                    &mut ds.d_token,
                    &mut ds.d_pos,
                    &mut ds.d_mrope,
                    &mut ds.d_out,
                    &mut ds.d_step,
                )?;
            }
        }
        let ids = exec.to_host_u32(&self.decode.as_ref().expect("decode").d_out)?;
        Ok(ids[..k].to_vec())
    }

    /// Greedy decode with MTP speculative decoding: draft `n_draft` tokens per
    /// round with the nextn head, verify them all in one batched target pass, and
    /// emit the accepted run + the target's correction token. Greedy verification
    /// makes the output BIT-IDENTICAL to `generate_greedy` - the draft only
    /// changes how many tokens each weight-read pass yields.
    pub fn generate_greedy_spec(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        stop: Option<u32>,
        n_draft: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        assert!(self.mtp.is_some(), "model has no nextn/MTP block");
        assert!(!prompt.is_empty() && max_new > 0);
        assert!(
            (1..SPEC_ROWS).contains(&n_draft),
            "n_draft+1 must fit SPEC_ROWS"
        );
        self.reset();
        let last = self.prefill(prompt)?; // also fills sc.d_h (all-row h_nextn)
        self.ensure_spec()?;
        // warm the MTP block's KV over the prompt (h shifted right; row 0 = zeros)
        self.mtp_process(prompt, 0, true)?;

        let exec = self.exec.clone();
        let vocab = self.vocab;
        let mut out = Vec::with_capacity(max_new);
        out.push(argmax(&last));
        if Some(out[0]) == stop || max_new == 1 {
            return Ok(out);
        }

        let debug = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some();
        let (mut t_draft, mut t_verify, mut t_post) = (0f64, 0f64, 0f64);
        let mut rounds = 0usize;
        while out.len() < max_new {
            let id_last = *out.last().expect("non-empty: token0 pushed above");
            let p = self.decode.as_ref().expect("decode").pos;
            // cap the chunk so we never verify past max_ctx
            let k = n_draft.min(self.max_ctx - p - 1);
            if k == 0 {
                break;
            }
            rounds += 1;
            let t0 = std::time::Instant::now();
            let drafts = self.mtp_draft(id_last, k)?;
            if debug {
                t_draft += t0.elapsed().as_secs_f64();
            }
            let t1 = std::time::Instant::now();
            let mut chunk = Vec::with_capacity(k + 1);
            chunk.push(id_last);
            chunk.extend_from_slice(&drafts);
            self.forward_chunk(&chunk)?;
            let all = exec.to_host(&self.spec.as_ref().expect("spec").d_logits_chunk)?;
            if debug {
                t_verify += t1.elapsed().as_secs_f64();
            }
            let t2 = std::time::Instant::now();
            let targets: Vec<u32> = (0..chunk.len())
                .map(|i| argmax(&all[i * vocab..(i + 1) * vocab]))
                .collect();
            // accept drafts while they match the target's own next token
            let mut a = 0usize;
            while a < drafts.len() && drafts[a] == targets[a] {
                a += 1;
            }
            let committed = a + 1; // chunk rows 0..=a become context
            if paddock_models::dev_var_os!("PADDOCK_SPEC_TRACE").is_some() {
                tracing::info!(
                    "TRACE single round {rounds}: pos {p} drafts {drafts:?} targets {targets:?} committed {committed}"
                );
            }
            self.commit_chunk(committed, chunk.len())?;
            // MTP catch-up over the committed rows (h = the TARGET's rows)
            self.mtp_process(&chunk[..committed], p, false)?;
            if debug {
                self.exec.synchronize()?;
                t_post += t2.elapsed().as_secs_f64();
            }
            // emit the a accepted drafts + the correction/bonus token
            for &t in targets.iter().take(committed) {
                out.push(t);
                if Some(t) == stop || out.len() >= max_new {
                    if debug {
                        tracing::info!(
                            "spec k={n_draft}: {rounds} rounds, {:.0} tok emitted; per-round draft {:.1}ms verify {:.1}ms post {:.1}ms",
                            out.len() as f64,
                            t_draft * 1e3 / rounds as f64,
                            t_verify * 1e3 / rounds as f64,
                            t_post * 1e3 / rounds as f64
                        );
                    }
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }

    /// Allocate the per-slot (batched) speculative-decoding state for `batch`
    /// concurrent sequences drafting `n_draft` tokens per round. Requires
    /// `enable_batch` (the backbone KV/state/conv slots) and the MTP block.
    pub fn enable_spec_batch(&mut self, batch: usize, n_draft: usize) -> Result<(), GpuModelError> {
        assert!(self.mtp.is_some(), "model has no nextn/MTP block");
        assert!(self.batch.is_some(), "enable_batch first");
        assert!(batch >= 1 && batch <= self.batch.as_ref().expect("enable_batch first").max_batch);
        assert!(n_draft >= 1);
        let k1 = n_draft + 1;
        self.ensure_scratch(batch * k1)?;
        // The spec ROUND runs on the decode arena (its graphs bake arena
        // addresses - see capture_spec_graph), so the mixed tick can overlap
        // the round with a span that owns the shared scratch. The arena was
        // built max_batch rows (enable_batch); the round needs batch*k1 -
        // regrow it via the same swap-build so the layout stays byte-
        // identical to the shared arena's.
        if self
            .decode_arena
            .as_ref()
            .is_some_and(|a| a.cap < batch * k1)
        {
            let saved = self.scratch.take();
            self.scratch = self.decode_arena.take();
            self.ensure_scratch(batch * k1)?;
            self.decode_arena = self.scratch.take();
            self.scratch = saved;
            // the decode graphs baked the old arena's addresses (see
            // ensure_batch_graph) - a regrow reallocates, so drop them and
            // let the next decode tick recapture (same rule as
            // pf_pass_graphs on prefill-buffer regrow)
            self.batch
                .as_mut()
                .expect("enable_batch first")
                .graphs
                .clear();
        }
        let e = &self.exec;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let state_elems = self.n_v_heads * self.state_size * self.state_size;
        let km1 = self.conv_k - 1;
        // Snapshot-free verify (dflash): stash the round's
        // split/gate planes per layer instead of per-token state snapshots -
        // the commit recomputes the accepted-prefix state from round-start
        // (gated_delta_commit_walk), bit-exact vs the snapshot restore. The
        // snapshots were ~87% of the ~1.15 GiB/spec-row draft state (the
        // 14-row width cap); the stash is ~128x smaller (s=128). The legacy
        // path stays for packs without slot 462/463 and for the
        // PADDOCK_QWEN35_SPEC_SNAPSHOT bring-up A/B.
        let snapshot_verify = self.spec_snapshot_verify();
        let vplane = state_elems / self.state_size; // n_v_heads * s per (row, pos)
        let (mut recur_snap, mut conv_ext) = (
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
        );
        let (mut vstash_k, mut vstash_v, mut vstash_g, mut vstash_beta) = (
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
            Vec::with_capacity(self.n_layers),
        );
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Linear(_) => {
                    recur_snap.push(
                        snapshot_verify
                            .then(|| e.alloc(batch * k1 * state_elems))
                            .transpose()?,
                    );
                    conv_ext.push(Some(e.alloc(batch * (km1 + k1) * self.conv_dim)?));
                    vstash_k.push(
                        (!snapshot_verify)
                            .then(|| e.alloc(batch * k1 * vplane))
                            .transpose()?,
                    );
                    vstash_v.push(
                        (!snapshot_verify)
                            .then(|| e.alloc(batch * k1 * vplane))
                            .transpose()?,
                    );
                    vstash_g.push(
                        (!snapshot_verify)
                            .then(|| e.alloc(batch * k1 * self.n_v_heads))
                            .transpose()?,
                    );
                    vstash_beta.push(
                        (!snapshot_verify)
                            .then(|| e.alloc(batch * k1 * self.n_v_heads))
                            .transpose()?,
                    );
                }
                Mixer::Full(_) => {
                    recur_snap.push(None);
                    conv_ext.push(None);
                    vstash_k.push(None);
                    vstash_v.push(None);
                    vstash_g.push(None);
                    vstash_beta.push(None);
                }
            }
        }
        let rmax = (batch * k1).max(WARM_CHUNK);
        // Drafter KV store size. Paged serves get a POOL STRIPE: the same
        // [n_blocks, BLOCK_TOKENS, kv_dim] layout as one backbone full-attn
        // layer's store (copy its byte count exactly), addressed by the same
        // combined block tables - prefix-cache adoption then restores drafter
        // rows with the pages (see mtp_block_pass_b). The stripe covers the
        // whole pool (all slots share it via block ids), so it sizes off the
        // pool, not `batch`. Dense mode (max_batch<=1, no prefix cache) keeps
        // the slot-strided [batch, max_ctx] store.
        let mtp_kv_bytes = {
            let bs = self.batch.as_ref().expect("enable_batch first");
            if bs.d_block_tables.is_some() && bs.paged {
                bs.kv_k
                    .iter()
                    .flatten()
                    .next()
                    .map(|b| b.len())
                    .expect("paged serve has a full-attn KV store")
            } else {
                batch * self.max_ctx * kv_dim * kv_bytes
            }
        };
        let spec_tables = {
            let bs = self.batch.as_ref().expect("enable_batch first");
            if bs.d_block_tables.is_some() && bs.paged {
                Some(e.alloc_u32(bs.block_table_host.len())?)
            } else {
                None
            }
        };
        // fused-landing width: widest of gate|up, fused qkv, fused in_qkv|gate
        // (the same shapes the dense elections land in bs.d_gu_fused/d_dn_fused)
        let fused_w = {
            let mut fw = 2 * self.ff;
            for layer in &self.layers {
                match &layer.mixer {
                    Mixer::Full(w) => {
                        fw = fw.max(w.wq.dims()[1] + w.wk.dims()[1] + w.wv.dims()[1]);
                    }
                    Mixer::Linear(w) => {
                        // +128: a DN tile plane built with the alpha||beta
                        // fold is nc + nz + 128 wide (load.rs, fuse_ab). The
                        // verify walk lands that plane here, so the width has
                        // to cover it or the tile arm cannot be elected.
                        fw = fw.max(w.in_qkv.dims()[1] + w.gate_w.dims()[1] + 128);
                    }
                }
            }
            fw
        };
        // Chain records at the MTP election when a block drafter widened the
        // alloc depth (see SpecBatchState::chain_depth); plain MTP unchanged.
        let chain_depth = if self.dflash.is_some() {
            n_draft.min(self.serve_spec_k_mtp())
        } else {
            n_draft
        };
        self.spec_batch = Some(SpecBatchState {
            chain_depth,
            batch,
            alloc_batch: batch,
            n_draft,
            pos: vec![0; batch],
            round_slots: (0..batch as u32).collect(),
            round_k1: n_draft + 1,
            d_spec_tables: spec_tables,
            d_fused_land: e.alloc(batch * k1 * fused_w)?,
            d_pending_hb: e.alloc(batch * self.embd)?,
            mtp_warm: vec![false; batch],
            mtp_toks: vec![Vec::new(); batch],
            mtp_kv_k: e.alloc_u8(mtp_kv_bytes)?,
            mtp_kv_v: e.alloc_u8(mtp_kv_bytes)?,
            pending_h: e.alloc(batch * self.embd)?,
            recur_snap,
            vstash_k,
            vstash_v,
            vstash_g,
            vstash_beta,
            warm_stash: None,
            conv_ext,
            d_logits_chunk: e.alloc(batch * k1 * self.vocab)?,
            d_h_chunk: e.alloc(batch * k1 * self.embd)?,
            d_row_tok: e.alloc_u32(batch * k1)?,
            d_draft: e.alloc_u32(n_draft * batch)?,
            d_asm_meta: e.alloc_u32(5 * batch)?,
            chain_lens: vec![0; batch],
            d_round_slots: e.alloc_u32(batch)?,
            d_committed: e.alloc_u32(batch)?,
            d_concat: e.alloc(rmax * 2 * self.embd)?,
            d_e: e.alloc(rmax * self.embd)?,
            d_en: e.alloc(rmax * self.embd)?,
            d_hn: e.alloc(rmax * self.embd)?,
            d_hin: e.alloc(rmax * self.embd)?,
            d_hout: e.alloc(rmax * self.embd)?,
            d_mtp_tok: e.alloc_u32(rmax)?,
            d_pos_rows: e.alloc_u32(rmax)?,
            d_slots_rows: e.alloc_u32(rmax)?,
            d_mrope_rows: e.alloc_u32(4 * rmax)?,
            max_pos_row: 0,
            d_xq: e.alloc_i8(rmax * self.ff.max(2 * self.embd))?,
            d_xs: e.alloc(rmax * self.ff.max(2 * self.embd) / 32)?,
            // split-K partials for the verify walk's f8d/mmq GEMMs: nz (<= 8)
            // x verify rows x out_dim. Was a fixed 64 rows (the 32/64-row
            // budget envelope) - with the f8d launcher now taking any batch
            // (rung D), the walk runs batch x k1 rows and this must hold
            // them all; 64 stays the floor so narrow serves are unchanged.
            d_ks_part: e.alloc(8 * (batch * k1).max(64) * self.ks_out_max())?,
            d_samp_par_chunk: e.alloc_u32(rmax * 4)?,
            d_samp_out_chunk: e.alloc_u32(rmax)?,
            d_samp_tpar_chunk: e.alloc_u32(rmax * 4)?,
            graph_draft: std::collections::HashMap::new(),
            graph_verify: std::collections::HashMap::new(),
            graph_commit: std::collections::HashMap::new(),
        });
        Ok(())
    }

    /// Stage per-row position/slot/mrope device inputs for a spec-batch pass
    /// (mrope is axis-major [4, r]: the text position plus the slot's
    /// multimodal mrope delta; the device draft loop bumps pos and mrope
    /// together, so the delta stays applied through drafted steps).
    fn stage_spec_rows(&mut self, pos: &[u32], slots: &[u32]) -> Result<(), GpuModelError> {
        assert_eq!(pos.len(), slots.len());
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let delta = &self.batch.as_ref().expect("batch").mrope_delta;
        let row: Vec<u32> = pos
            .iter()
            .zip(slots)
            .map(|(&p, &s)| (p as i64 + delta[s as usize]) as u32)
            .collect();
        let mrope: Vec<u32> = (0..4).flat_map(|_| row.iter().copied()).collect();
        let sb = self.spec_batch.as_mut().expect("spec batch");
        sb.max_pos_row = pos.iter().copied().max().unwrap_or(0);
        exec.stream
            .memcpy_htod(pos, &mut sb.d_pos_rows)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(slots, &mut sb.d_slots_rows)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&mrope, &mut sb.d_mrope_rows)
            .map_err(drv)?;
        Ok(())
    }

    /// Batched twin of `mtp_project_inputs`: tokens in `spec_batch.d_mtp_tok`,
    /// h rows in `spec_batch.d_hin` -> eh_proj(enorm(embed(tok)) || hnorm(h))
    /// into `sc.d_x` for `r` rows.
    fn mtp_project_inputs_b(&mut self, r: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, eps) = (self.embd, self.rms_eps);
        let m = self.mtp.as_ref().expect("mtp weights");
        let tok_embd = &self.tok_embd;
        let sc = self.scratch.as_mut().expect("scratch");
        let sb = self.spec_batch.as_mut().expect("spec batch");
        embed_any(&exec, tok_embd, &sb.d_mtp_tok, &mut sb.d_e, embd, r)?;
        exec.rmsnorm_batch(&sb.d_e, &m.enorm.buf, &mut sb.d_en, embd, eps, r)?;
        exec.rmsnorm_batch(&sb.d_hin, &m.hnorm.buf, &mut sb.d_hn, embd, eps, r)?;
        for i in 0..r {
            exec.copy_region(&sb.d_en, i * embd, &mut sb.d_concat, i * 2 * embd, embd)?;
            exec.copy_region(
                &sb.d_hn,
                i * embd,
                &mut sb.d_concat,
                i * 2 * embd + embd,
                embd,
            )?;
        }
        mmq(
            &exec,
            &m.eh_proj,
            &sb.d_concat,
            &mut sb.d_xq,
            &mut sb.d_xs,
            &mut sc.d_ssums,
            &mut sb.d_ks_part,
            &mut sc.d_x,
            r,
        )?;
        Ok(())
    }

    /// Batched twin of `mtp_block_pass`: the MTP transformer layer over `r`
    /// rows using the PER-SLOT MTP KV (`spec_batch.mtp_kv_*`) and the staged
    /// per-row position/slot/mrope inputs (`stage_spec_rows`).
    fn mtp_block_pass_b(&mut self, r: usize, head_norm_out: bool) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (ff, max_ctx) = (self.ff, self.max_ctx);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let sinks = &self.sinks;
        let m = self.mtp.as_ref().expect("mtp weights");
        // Paged serves route the drafter KV through the same combined block
        // tables as the backbone (mtp_kv_* is then a pool STRIPE - one more
        // [n_blocks, BLOCK_TOKENS, kv_dim] store like a full-attn layer's),
        // so a prefix-cache adoption restores the drafter's rows with the
        // pages and cross-slot hits keep drafting at full fidelity. Before
        // this, the drafter KV was slot-strided dense: every cross-slot radix
        // hit (repeated prompts, agentic prefix reuse) lost the drafter's
        // history and served dense, more than halving spec throughput from
        // the second repeat on.
        let bs = self.batch.as_ref().expect("batch");
        let (is_paged, bps) = (bs.paged, bs.blocks_per_slot);
        let sc = self.scratch.as_mut().expect("scratch");
        let sb = self.spec_batch.as_mut().expect("spec batch");
        // paged reads go through the round's table MIRROR (d_spec_tables) so
        // the span's live-table growth uploads can't race them mid-overlap;
        // staged fresh at every entry point (stage_spec_tables)
        let paged = sb
            .d_spec_tables
            .as_ref()
            .filter(|_| is_paged)
            .map(|bt| (bt, bps));

        exec.rmsnorm_batch(&sc.d_x, &m.attn_norm.buf, &mut sc.d_xn, embd, eps, r)?;
        // dp4a class throughout: the MTP block only shapes DRAFTS, and drafts
        // only move round boundaries (the verify numerics decide the emitted
        // stream), so the faster activation-quantized weight read is free
        mmq(
            &exec,
            &m.attn.wq,
            &sc.d_xn,
            &mut sb.d_xq,
            &mut sb.d_xs,
            &mut sc.d_ssums,
            &mut sb.d_ks_part,
            &mut sc.d_qg,
            r,
        )?;
        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
        mmq_pre_any(
            &exec,
            &m.attn.wk,
            &sb.d_xq,
            &sb.d_xs,
            &mut sc.d_ssums,
            &mut sb.d_ks_part,
            &mut sc.d_k,
            r,
        )?;
        mmq_pre_any(
            &exec,
            &m.attn.wv,
            &sb.d_xq,
            &sb.d_xs,
            &mut sc.d_ssums,
            &mut sb.d_ks_part,
            &mut sc.d_v,
            r,
        )?;
        exec.rmsnorm_batch(
            &sc.d_q,
            &m.attn.q_norm.buf,
            &mut sc.d_qn,
            head_dim,
            eps,
            r * n_heads,
        )?;
        exec.rmsnorm_batch(
            &sc.d_k,
            &m.attn.k_norm.buf,
            &mut sc.d_kn,
            head_dim,
            eps,
            r * n_kv_heads,
        )?;
        exec.mrope(
            &mut sc.d_qn,
            &sb.d_mrope_rows,
            r,
            n_heads,
            head_dim,
            n_rot,
            yarn,
            sections,
        )?;
        exec.mrope(
            &mut sc.d_kn,
            &sb.d_mrope_rows,
            r,
            n_kv_heads,
            head_dim,
            n_rot,
            yarn,
            sections,
        )?;
        if let Some((bt, bps)) = paged {
            exec.kv_append_batch_paged(
                &sc.d_kn,
                &mut sb.mtp_kv_k,
                &sb.d_pos_rows,
                Some(&sb.d_slots_rows),
                bt,
                bps,
                kv_dim,
                r,
                self.kv_dtype,
            )?;
            exec.kv_append_batch_paged(
                &sc.d_v,
                &mut sb.mtp_kv_v,
                &sb.d_pos_rows,
                Some(&sb.d_slots_rows),
                bt,
                bps,
                kv_dim,
                r,
                self.kv_dtype,
            )?;
        } else {
            exec.kv_append_batch(
                &sc.d_kn,
                &mut sb.mtp_kv_k,
                &sb.d_pos_rows,
                Some(&sb.d_slots_rows),
                kv_dim,
                max_ctx,
                r,
                self.kv_dtype,
            )?;
            exec.kv_append_batch(
                &sc.d_v,
                &mut sb.mtp_kv_v,
                &sb.d_pos_rows,
                Some(&sb.d_slots_rows),
                kv_dim,
                max_ctx,
                r,
                self.kv_dtype,
            )?;
        }
        attn_decode_dispatch(
            &exec,
            &sc.d_qn,
            &sb.mtp_kv_k,
            &sb.mtp_kv_v,
            sinks,
            &mut sc.d_attn_o,
            &mut sc.d_attn_ml,
            &mut sc.d_attn,
            &sb.d_pos_rows,
            Some(&sb.d_slots_rows),
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            kv_dim,
            r,
            scale,
            self.kv_dtype,
            paged,
        )?;
        exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, r * q_dim)?;
        mmq(
            &exec,
            &m.attn.wo,
            &sc.d_attn,
            &mut sb.d_xq,
            &mut sb.d_xs,
            &mut sc.d_ssums,
            &mut sb.d_ks_part,
            &mut sc.d_proj,
            r,
        )?;
        exec.add_rmsnorm_batch(
            &mut sc.d_x,
            &sc.d_proj,
            &m.post_norm.buf,
            &mut sc.d_xn,
            embd,
            eps,
            r,
        )?;
        match &m.ffn {
            Ffn::Dense { gate, up, down } => {
                mmq(
                    &exec,
                    gate,
                    &sc.d_xn,
                    &mut sb.d_xq,
                    &mut sb.d_xs,
                    &mut sc.d_ssums,
                    &mut sb.d_ks_part,
                    &mut sc.d_ffn_gate,
                    r,
                )?;
                mmq_pre_any(
                    &exec,
                    up,
                    &sb.d_xq,
                    &sb.d_xs,
                    &mut sc.d_ssums,
                    &mut sb.d_ks_part,
                    &mut sc.d_ffn_up,
                    r,
                )?;
                exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, r * ff)?;
                mmq(
                    &exec,
                    down,
                    &sc.d_ffn_gate,
                    &mut sb.d_xq,
                    &mut sb.d_xs,
                    &mut sc.d_ssums,
                    &mut sb.d_ks_part,
                    &mut sc.d_proj,
                    r,
                )?;
            }
            Ffn::Nvf4Dense { gu, down } => {
                // W4A4 class election for the batched DRAFTER: the batch
                // decode path serves W4A4 above the row band, so the old
                // exact-family pin here was both slower -
                // the nvfp4 lane's spec c1 ran at its no-spec rate, the
                // exact chain ate the entire drafting gain - and a class
                // mismatch against the class the drafter's own history rows
                // were served with. Draft + batched verify flip together
                // (record_spec_verify has the twin) so they agree with each
                // other and with the served no-spec class; acceptance is
                // measured on the flip (the old comment's recorded fear).
                // nvf4_ffn falls back to the exact chain below the row band.
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
                    r,
                )?;
            }
            Ffn::Moe(w) => {
                moe_ffn(
                    &exec,
                    w,
                    self.moe.expect("moe dims"),
                    embd,
                    r,
                    false,
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
        exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
        if head_norm_out {
            exec.rmsnorm_batch(&sc.d_x, &m.head_norm.buf, &mut sb.d_hout, embd, eps, r)?;
        }
        Ok(())
    }

    /// Warm slot `slot`'s MTP KV over its freshly prefilled prompt (b9895
    /// `process()` semantics, h shifted right by one; row 0 pairs with the
    /// slot's pending_h - zeros for a fresh sequence). Must run immediately
    /// after `forward_prefill_slot` for the same slot: it reads the prompt's
    /// post-out_norm h rows from `sc.d_h` before the next prefill overwrites
    /// them. Leaves pending_h[slot] = h of the prompt's last position.
    /// `start` = the absolute position of `tokens[0]` (0 for a whole fresh
    /// prompt; a later chunk's offset when warming per prefill chunk - the
    /// previous chunk's warm left pending_h[slot] = h of position start-1,
    /// which row 0 pairs with). `h_off` = the ROW offset of this span's h
    /// rows inside sc.d_h (0 for the serial paths; the share's row base for
    /// the batched-cohort prefill, whose d_h holds the whole group).
    pub(super) fn mtp_warm_slot(
        &mut self,
        slot: usize,
        tokens: &[u32],
        start: usize,
        h_off: usize,
    ) -> Result<(), GpuModelError> {
        let n = tokens.len();
        assert!(n > 0);
        // the warm loop's mtp_block_pass_b reads the round's table MIRROR -
        // warm runs outside rounds (prefill hooks), so refresh it here or
        // the drafter KV appends land on whatever pages the last round saw
        self.stage_spec_tables()?;
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let embd = self.embd;
        let mut done = 0usize;
        while done < n {
            let r = (n - done).min(WARM_CHUNK);
            {
                let sc = self.scratch.as_ref().expect("scratch");
                let sb = self.spec_batch.as_mut().expect("spec batch");
                if done == 0 {
                    exec.copy_region(&sb.pending_h, slot * embd, &mut sb.d_hin, 0, embd)?;
                } else {
                    exec.copy_region(&sc.d_h, (h_off + done - 1) * embd, &mut sb.d_hin, 0, embd)?;
                }
                if r > 1 {
                    exec.copy_region(
                        &sc.d_h,
                        (h_off + done) * embd,
                        &mut sb.d_hin,
                        embd,
                        (r - 1) * embd,
                    )?;
                }
                exec.stream
                    .memcpy_htod(&tokens[done..done + r], &mut sb.d_mtp_tok)
                    .map_err(drv)?;
            }
            let pos: Vec<u32> = ((start + done) as u32..(start + done + r) as u32).collect();
            let slots = vec![slot as u32; r];
            self.stage_spec_rows(&pos, &slots)?;
            self.mtp_project_inputs_b(r)?;
            self.mtp_block_pass_b(r, false)?;
            done += r;
        }
        {
            let sc = self.scratch.as_ref().expect("scratch");
            let sb = self.spec_batch.as_mut().expect("spec batch");
            exec.copy_region(
                &sc.d_h,
                (h_off + n - 1) * embd,
                &mut sb.pending_h,
                slot * embd,
                embd,
            )?;
        }
        Ok(())
    }

    /// Capture one round-phase recording as a CUDA graph (the shared sync ->
    /// begin_capture -> record -> end_capture pattern of the decode/batch graphs).
    fn capture_spec_graph(
        &mut self,
        record: fn(&mut Self) -> Result<(), GpuModelError>,
        what: &'static str,
    ) -> Result<SendGraph, GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("{what} pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("{what} begin_capture: {e}")))?;
        let rec = record(self);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("{what} end_capture: {e}")));
        rec?;
        let graph =
            graph?.ok_or_else(|| GpuError::Driver(format!("{what} capture produced no graph")))?;
        Ok(SendGraph(graph))
    }

    /// Batched greedy draft: every slot drafts `n_draft` tokens with the MTP
    /// head in lock-step - step i runs the MTP block over B rows (slot b at
    /// position pos[b]+i), picks each row's token on device (`argmax_rows`),
    /// and chains (token, hout) into step i+1. `last[b]` = slot b's last
    /// emitted token. Returns the drafts i-major: ids[i*B + b].
    ///
    /// The K unrolled steps replay as one captured graph: tokens/positions are
    /// staged before launch, positions advance on device between steps
    /// (`bump_rows_u32`), and the token/h chains are device-resident already.
    fn mtp_draft_launch_inner(&mut self, last: &[u32]) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let b = self.spec_batch.as_ref().expect("spec batch").batch;
        assert_eq!(last.len(), b);
        // block i serves round_slots[i] - positions/slots staged per true
        // slot, and pending_h gathered block-ordered into d_pending_hb (the
        // draft graph's step-0 h read is a contiguous copy baked at capture)
        let (pos, slots): (Vec<u32>, Vec<u32>) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            sb.round_slots[..b]
                .iter()
                .map(|&s| (sb.pos[s as usize] as u32, s))
                .unzip()
        };
        {
            let embd = self.embd;
            let sb = self.spec_batch.as_mut().expect("spec batch");
            exec.stream
                .memcpy_htod(last, &mut sb.d_mtp_tok)
                .map_err(drv)?;
            for (i, &s) in slots.iter().enumerate() {
                exec.copy_region(
                    &sb.pending_h,
                    s as usize * embd,
                    &mut sb.d_pending_hb,
                    i * embd,
                    embd,
                )?;
            }
        }
        self.stage_spec_rows(&pos, &slots)?;
        if paddock_models::dev_var_os!("PADDOCK_SPEC_NOGRAPH").is_some() {
            self.record_spec_draft()?; // eager A/B path - identical launches
        } else {
            // the draft chain's shape keys on live alone (it unrolls n_draft
            // steps regardless of the ROUND's ragged k1) - constant second key
            let live = (b, 0usize);
            if !self
                .spec_batch
                .as_ref()
                .expect("spec batch")
                .graph_draft
                .contains_key(&live)
            {
                let g = self.capture_spec_graph(Self::record_spec_draft, "spec draft")?;
                self.spec_batch
                    .as_mut()
                    .expect("spec batch")
                    .graph_draft
                    .insert(live, g);
            }
            self.spec_batch.as_ref().expect("spec batch").graph_draft[&live]
                .0
                .launch_on(&self.exec.stream)
                .map_err(|e| GpuError::Driver(format!("spec draft graph launch: {e}")))?;
        }
        Ok(())
    }

    /// Synchronous draft: launch the chain and read the ids back (the
    /// classic path; the async round leaves them on device - see
    /// spec_draft_begin_mtp).
    fn mtp_draft_b(&mut self, last: &[u32]) -> Result<Vec<u32>, GpuModelError> {
        self.mtp_draft_launch(last)?;
        let sb = self.spec_batch.as_ref().expect("spec batch");
        self.exec
            .to_host_u32(&sb.d_draft)
            .map_err(GpuModelError::from)
    }

    /// Arena-scoped wrappers: the spec round's four GPU phases run with the
    /// decode arena active (see with_spec_arena).
    fn mtp_draft_launch(&mut self, last: &[u32]) -> Result<(), GpuModelError> {
        self.with_spec_arena(|m| m.mtp_draft_launch_inner(last))
    }
    fn forward_chunk_b(&mut self, chunk: &[u32]) -> Result<(), GpuModelError> {
        self.with_spec_arena(|m| m.forward_chunk_b_inner(chunk))
    }
    fn commit_chunk_b(&mut self, chunk: &[u32], committed: &[u32]) -> Result<(), GpuModelError> {
        self.with_spec_arena(|m| m.commit_chunk_b_inner(chunk, committed))
    }
    fn mtp_catchup_b(
        &mut self,
        chunk: &[u32],
        committed: &[u32],
        pos_before: &[usize],
    ) -> Result<(), GpuModelError> {
        self.with_spec_arena(|m| m.mtp_catchup_b_inner(chunk, committed, pos_before))
    }

    /// Run `f` with the DECODE ARENA as the active scratch: the spec round's
    /// graphs bake arena addresses (capture happens inside these regions),
    /// and its eager pieces must address the same buffers - which is what
    /// lets a mixed tick overlap the round with a span that owns the shared
    /// scratch. Reentrant (the inner call sees `decode_arena` already taken
    /// and no-ops); a build without an arena (serial/dense) runs on the
    /// shared scratch exactly as before. The prefill WARM hooks deliberately
    /// do not swap - they read the prompt's h rows from the shared scratch.
    fn with_spec_arena<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, GpuModelError>,
    ) -> Result<T, GpuModelError> {
        if self.decode_arena.is_none() {
            return f(self);
        }
        let saved = self.scratch.take();
        self.scratch = self.decode_arena.take();
        let r = f(self);
        self.decode_arena = self.scratch.take();
        self.scratch = saved;
        r
    }

    /// The K-step draft loop, capture-safe (kernels + D2D copies only): expects
    /// `d_mtp_tok` = last tokens and `d_pos_rows`/`d_slots_rows`/`d_mrope_rows`
    /// staged for step 0.
    fn record_spec_draft(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, vocab) = (self.embd, self.vocab);
        // chain_depth, not n_draft: with a block drafter attached the alloc
        // depth is the BLOCK's k; the chain itself records the MTP election
        // (see SpecBatchState::chain_depth).
        let (b, k) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            (sb.batch, sb.chain_depth)
        };
        for i in 0..k {
            {
                let sb = self.spec_batch.as_mut().expect("spec batch");
                if i == 0 {
                    // d_hin <- the block-gathered pending_h staging (see
                    // d_pending_hb: pending_h itself is TRUE-slot-strided,
                    // this copy is contiguous and baked at capture)
                    exec.copy_region(&sb.d_pending_hb, 0, &mut sb.d_hin, 0, b * embd)?;
                } else {
                    // token chain: argmax_rows already wrote d_mtp_tok[..B];
                    // h chain: the head's own post-shared_head_norm rows;
                    // positions advance on device
                    exec.bump_rows_u32(&mut sb.d_pos_rows, &mut sb.d_mrope_rows, b)?;
                    exec.copy_region(&sb.d_hout, 0, &mut sb.d_hin, 0, b * embd)?;
                }
            }
            self.mtp_project_inputs_b(b)?;
            self.mtp_block_pass_b(b, true)?;
            {
                let out_f8 = self.out_f8.as_ref();
                let sc = self.scratch.as_mut().expect("scratch");
                let sb = self.spec_batch.as_mut().expect("spec batch");
                // lm_head: the biggest single weight read of the draft step
                // (drafts are class-free, see mtp_block_pass_b). f8d head at
                // batched widths (1.58ms/step on mt_dp4a vs the dense head's
                // ~0.9ms) - dp4a below the b>=8 boundary.
                // was `b >= 8`, a bare literal that left b < 8 on the Q8_0
                // head -- which the REPLACE lane drops at load. One election.
                if let Some((p8, pi, po)) = super::head_f8(out_f8, b) {
                    exec.quantize_e4m3(&sb.d_hout, &mut sc.d_pxq, &mut sc.d_exs, b * pi)?;
                    exec.f8d_gemm_mma_ks(
                        p8,
                        *pi,
                        *po,
                        &sc.d_pxq,
                        &sc.d_exs,
                        &mut sb.d_ks_part,
                        &mut sb.d_logits_chunk,
                        b,
                    )?;
                } else {
                    super::stub_guard(&self.output, "spec.rs draft-step head")?;
                    mmq(
                        &exec,
                        &self.output,
                        &sb.d_hout,
                        &mut sb.d_xq,
                        &mut sb.d_xs,
                        &mut sc.d_ssums,
                        &mut sb.d_ks_part,
                        &mut sb.d_logits_chunk,
                        b,
                    )?;
                }
                exec.argmax_rows(&sb.d_logits_chunk, &mut sb.d_mtp_tok, b, vocab)?;
                exec.copy_region(&sb.d_mtp_tok, 0, &mut sb.d_draft, i * b, b)?;
            }
        }
        Ok(())
    }

    /// Refresh the round's block-table mirror (`d_spec_tables`) from the host
    /// truth, on the CURRENT exec lane. Called at every drafter/round entry
    /// point after its ensure_slot_blocks calls (so the host table is final
    /// for the rows about to be read) - the mirror is what the round's paged
    /// kernels read, so the live table's growth re-uploads (which the span
    /// issues on the main lane mid-overlap) can never race them.
    fn stage_spec_tables(&mut self) -> Result<(), GpuModelError> {
        let Some(bs) = self.batch.as_ref() else {
            return Ok(());
        };
        let host: &[u32] = &bs.block_table_host;
        let Some(sb) = self.spec_batch.as_mut() else {
            return Ok(());
        };
        let Some(dst) = sb.d_spec_tables.as_mut() else {
            return Ok(());
        };
        self.exec
            .stream
            .memcpy_htod(host, dst)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        Ok(())
    }

    /// Batched verify pass: `chunk` is B×(K+1) tokens slot-major (row b*(K+1)
    /// = slot b's last emitted token, then its K drafts) run through the
    /// backbone at per-slot positions without committing - the v2 recurrence
    /// snapshots every (slot, row) state, the conv windows are read through
    /// per-slot extended-row buffers and left untouched, and `pos` stays put
    /// until `commit_chunk_b`. Leaves logits/h for every row plus the device
    /// argmax picks in `spec_batch.d_row_tok`.
    fn forward_chunk_b_inner(&mut self, chunk: &[u32]) -> Result<(), GpuModelError> {
        let (b, k1) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            (sb.batch, sb.round_k1)
        };
        let rows = b * k1;
        assert_eq!(chunk.len(), rows);
        self.ensure_scratch(rows)?;
        // P5 pool: back every verify row (pos..pos+k1, pads included) with real
        // blocks before the graph runs - the paged append/attention read the
        // device table at replay, and growth uploads outside any capture (the
        // ensure_slot_blocks contract). The verify used to write dense
        // slot*max_ctx offsets straight into pool storage, which is only
        // coincidentally correct while a sequence's blocks are identity-mapped
        // (always true on fresh bench runs, hence every parity gate passing) -
        // once the allocator's order diverged, the verify attended another
        // REQUEST'S resident KV: the cross-request contamination root cause.
        {
            let rs: Vec<u32> =
                self.spec_batch.as_ref().expect("spec batch").round_slots[..b].to_vec();
            for &s in &rs {
                let p = self.spec_batch.as_ref().expect("spec batch").pos[s as usize];
                self.ensure_slot_blocks(s as usize, p + k1)?;
            }
        }
        self.stage_spec_tables()?;
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        // per-row positions/slots (block-major, true slot per block via
        // round_slots) + the chunk tokens
        let (pos_rows, slot_rows): (Vec<u32>, Vec<u32>) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            let mut p = Vec::with_capacity(rows);
            let mut s = Vec::with_capacity(rows);
            for &sl in &sb.round_slots[..b] {
                assert!(
                    sb.pos[sl as usize] + k1 <= self.max_ctx,
                    "chunk exceeds max_ctx"
                );
                for j in 0..k1 {
                    p.push((sb.pos[sl as usize] + j) as u32);
                    s.push(sl);
                }
            }
            (p, s)
        };
        self.stage_spec_rows(&pos_rows, &slot_rows)?;
        {
            let sb = self.spec_batch.as_mut().expect("spec batch");
            exec.stream
                .memcpy_htod(chunk, &mut sb.d_mtp_tok)
                .map_err(drv)?;
        }
        // ARMED async chain: the host `chunk` rows carry
        // placeholder VALUES for drafted positions - assemble the real
        // verify tokens on device from d_draft (pd_spec_toks). Meta per
        // verify block: pend = the block's row 0 (always real), srow = the
        // block's position in the CHAIN's slot list (chain-cold blocks get
        // nd=0 -> every drafted row pads with pend, matching the length-1
        // chunk the service built), nd = real drafted rows
        // (chain_lens[i]-1, staged by the driver), clen = k1 so the pad
        // rows take the last draft - byte-matching the synchronous path's
        // repeat-last-token padding.
        if let Some((chain_slots, k_use)) = self.spec_chain.clone()
            && exec.has_spec_toks()
        {
            let rr = chain_slots.len();
            let mut meta = vec![0u32; 5 * b];
            {
                let sb = self.spec_batch.as_ref().expect("spec batch");
                for i in 0..b {
                    let sl = sb.round_slots[i];
                    let ci = chain_slots.iter().position(|&s| s == sl);
                    let nd = match ci {
                        Some(_) => (sb.chain_lens[i].saturating_sub(1) as usize).min(k_use) as u32,
                        None => 0,
                    };
                    meta[i] = chunk[i * k1];
                    meta[b + i] = ci.unwrap_or(0) as u32;
                    meta[2 * b + i] = nd;
                    meta[3 * b + i] = k1 as u32;
                    meta[4 * b + i] = (i * k1) as u32;
                }
            }
            {
                let sb = self.spec_batch.as_mut().expect("spec batch");
                let mut v = sb.d_asm_meta.slice_mut(0..5 * b);
                exec.stream.memcpy_htod(&meta, &mut v).map_err(drv)?;
            }
            let sb = self.spec_batch.as_mut().expect("spec batch");
            let (meta_b, draft_b) = (&sb.d_asm_meta, &sb.d_draft);
            exec.spec_toks(meta_b, draft_b, &mut sb.d_mtp_tok, b, k1, rr)?;
        }
        // the verify + commit graphs route DeltaNet state/conv per block
        // through the round's own d_round_slots (bs.d_slots belongs to the
        // main-lane decode/mixed passes - sharing it would race when the
        // round overlaps a span).
        {
            let sb = self.spec_batch.as_mut().expect("spec batch");
            let rs: Vec<u32> = sb.round_slots[..b].to_vec();
            let mut v = sb.d_round_slots.slice_mut(0..b);
            exec.stream.memcpy_htod(&rs, &mut v).map_err(drv)?;
        }
        // ~15 launches x 65 layers replay as one graph; every per-row input
        // (tokens, positions, slots, mrope) is read from the staged device
        // buffers, so the capture is round-invariant for fixed (B, K)
        if paddock_models::dev_var_os!("PADDOCK_SPEC_NOGRAPH").is_some() {
            self.record_spec_verify()?; // eager A/B path - identical launches
            return Ok(());
        }
        let live = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            (sb.batch, sb.round_k1)
        };
        if !self
            .spec_batch
            .as_ref()
            .expect("spec batch")
            .graph_verify
            .contains_key(&live)
        {
            let g = self.capture_spec_graph(Self::record_spec_verify, "spec verify")?;
            self.spec_batch
                .as_mut()
                .expect("spec batch")
                .graph_verify
                .insert(live, g);
        }
        self.spec_batch.as_ref().expect("spec batch").graph_verify[&live]
            .0
            .launch_on(&self.exec.stream)
            .map_err(|e| GpuError::Driver(format!("spec verify graph launch: {e}")))?;
        Ok(())
    }

    /// The verify-pass compute (capture-safe): B×(K+1) staged rows through the
    /// backbone with per-slot KV/conv/state routing, snapshots for the ragged
    /// commit, then row logits + device argmax picks.
    fn record_spec_verify(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (b, k1) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            (sb.batch, sb.round_k1)
        };
        let rows = b * k1;
        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (state_size, n_k_heads, n_v_heads, conv_k) =
            (self.state_size, self.n_k_heads, self.n_v_heads, self.conv_k);
        let (conv_dim, ff, max_ctx, vocab) = (self.conv_dim, self.ff, self.max_ctx, self.vocab);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let km1 = conv_k - 1;
        let r = rows;

        let sinks = &self.sinks;
        let layers = &self.layers;
        let tok_embd = &self.tok_embd;
        // f8 class mirror: the verify decides the EMITTED stream, so it must
        // run the same numeric class the dense decode path serves at this
        // width (the Q8 twin of the nvfp4 flip below) - and the f8d lane is
        // ~2x the Q8 int8-mma route at verify rows (41.2ms -> ~20ms of round
        // GEMM). Elections mirror record_batch_step exactly:
        // rows >= 8 with the plane loaded, Q8 chain below the boundary.
        let bs_w8 = &self.bs_w8;
        let bs_f8ffn = &self.bs_f8ffn;
        let bs_f8ffn_bs = &self.bs_f8ffn_bs;
        // the checkpoint-exact fp8 layers (F8RowFfn) build no lin twin
        let bs_f8row = &self.bs_f8row_ffn;
        // the Nvf4Dense verify arm's own planes - see the FFN match
        // below. Bound here with the others because `sc` takes &mut self next.
        let bs_f8t_ffn = &self.bs_f8t_ffn;
        // the ATTENTION projections' tile planes. The verify
        // walk ran them on f8d/mmq - the f8 mma and k-quant dp4a classes - while
        // decode has served them off these planes through pd_f8cut_gemm since
        // profiling a spec round: pd_f8_gemm_mma 107 launches/round x 37.52 us
        // = 4.0 ms + ~1.6 ms of k-quant int8, against 1.3 ms for the same three
        // projections in the decode census. int8 is the one class B200 de-rates
        // (1148 TOPS vs ~7.5 PF e4m3). Kill: PADDOCK_NO_QWEN35_SPEC_F8T.
        let bs_f8t_attn = &self.bs_f8t_attn;
        let spec_f8t = paddock_models::dev_var_os!("PADDOCK_NO_QWEN35_SPEC_F8T").is_none();
        let out_f8 = self.out_f8.as_ref();
        let f8_lane = !self.bs_f8ffn.is_empty();
        let sc = self.scratch.as_mut().expect("scratch");
        let bs = self.batch.as_mut().expect("batch");
        let sb = self.spec_batch.as_mut().expect("spec batch");

        // DFlash: tap the verify rows' residuals so the accepted rows'
        // features can ring-append post-accept (dflash_spec_commit). Baked
        // into the verify graph - the drafter arms before any capture.
        let mut dtap = self
            .dflash
            .as_mut()
            .filter(|d| d.state.is_some() && !super::dflash::fuse_off());

        embed_any(&exec, tok_embd, &sb.d_mtp_tok, &mut sc.d_x, embd, r)?;

        for (li, layer) in layers.iter().enumerate() {
            if let Some(df) = dtap.as_mut()
                && let Some(band) = df.target_layers.iter().position(|&t| t == li)
            {
                super::dflash::tap_band(&exec, df, &sc.d_x, band, embd, r)?;
            }
            exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, r)?;
            match &layer.mixer {
                Mixer::Full(w) => {
                    // was a bare `r >= 8`: the verify chunk is k+1 rows, so at
                    // c1 (k~2-4) every projection fell to the Q8_0 planes --
                    // invisible while both residencies exist, fatal under the
                    // projection REPLACE. Same literal-8 family as this walk's
                    // FFN and head gates (both since fixed).
                    let l8d = bs_w8
                        .get(li)
                        .filter(|_| r > super::w8_min_batch() && f8_lane);
                    let f8_qkv = l8d.is_some_and(|l| l.wq.is_some());
                    // set when the qkv arm took the tile lane; wo follows it
                    // (one plane pair, one precision class per layer)
                    let mut f8t_wo: Option<&crate::gpu::F8TilePlane> = None;
                    // tile lane first, exactly as batch.rs elects it. Same
                    // landing buffer and same slices, so nothing downstream
                    // moves - only the GEMM class does.
                    let f8t_qkv = bs_f8t_attn
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| spec_f8t && r <= 64);
                    if let Some([qkv_t, wo_t]) = f8t_qkv {
                        static W: std::sync::Once = std::sync::Once::new();
                        W.call_once(|| {
                            eprintln!(
                                "[spec-f8t] engaged: verify projections on the tile lane (rows={r})"
                            )
                        });
                        let (nq, nk2, nv2) = (w.wq.dims()[1], w.wk.dims()[1], w.wv.dims()[1]);
                        let nt = nq + nk2 + nv2;
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            embd,
                            r,
                        )?;
                        exec.f8t_gemm(
                            qkv_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut sb.d_ks_part,
                            &mut sb.d_fused_land,
                            embd,
                            nt,
                            r,
                        )?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_qg, nt, 0, nq, r)?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_k, nt, nq, nk2, r)?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_v, nt, nq + nk2, nv2, r)?;
                        f8t_wo = Some(wo_t);
                    } else if f8_qkv {
                        let l8 = l8d.expect("f8_qkv checked l8d");
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        let (nq, nk2, nv2) = (w.wq.dims()[1], w.wk.dims()[1], w.wv.dims()[1]);
                        let nt = nq + nk2 + nv2;
                        exec.f8d_gemm_mma_ks(
                            l8.wq.as_ref().expect("f8_qkv checked wq"),
                            w.wq.dims()[0],
                            nt,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sb.d_ks_part,
                            &mut sb.d_fused_land,
                            r,
                        )?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_qg, nt, 0, nq, r)?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_k, nt, nq, nk2, r)?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_v, nt, nq + nk2, nv2, r)?;
                    } else {
                        super::stub_guard(&w.wq, "spec.rs verify wq")?;
                        mmq(
                            &exec,
                            &w.wq,
                            &sc.d_xn,
                            &mut sb.d_xq,
                            &mut sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_qg,
                            r,
                        )?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        super::stub_guard(&w.wk, "spec.rs verify wk")?;
                        mmq_pre_any(
                            &exec,
                            &w.wk,
                            &sb.d_xq,
                            &sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_k,
                            r,
                        )?;
                        super::stub_guard(&w.wv, "spec.rs verify wv")?;
                        mmq_pre_any(
                            &exec,
                            &w.wv,
                            &sb.d_xq,
                            &sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_v,
                            r,
                        )?;
                    }
                    exec.rmsnorm_batch(
                        &sc.d_q,
                        &w.q_norm.buf,
                        &mut sc.d_qn,
                        head_dim,
                        eps,
                        r * n_heads,
                    )?;
                    exec.rmsnorm_batch(
                        &sc.d_k,
                        &w.k_norm.buf,
                        &mut sc.d_kn,
                        head_dim,
                        eps,
                        r * n_kv_heads,
                    )?;
                    exec.mrope(
                        &mut sc.d_qn,
                        &sb.d_mrope_rows,
                        r,
                        n_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.d_kn,
                        &sb.d_mrope_rows,
                        r,
                        n_kv_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    // Pool-aware routing: mirror the dense tick.
                    // The former unconditional dense append/attend here is the
                    // contamination bug - see the ensure_slot_blocks note in
                    // forward_chunk_b.
                    if bs.paged {
                        // round mirror, not the live table - see mtp_block_pass_b
                        let bt = sb.d_spec_tables.as_ref().expect("paged serve spec tables");
                        let bps = bs.blocks_per_slot;
                        exec.kv_append_batch_paged(
                            &sc.d_kn,
                            bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                            &sb.d_pos_rows,
                            Some(&sb.d_slots_rows),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            self.kv_dtype,
                        )?;
                        exec.kv_append_batch_paged(
                            &sc.d_v,
                            bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                            &sb.d_pos_rows,
                            Some(&sb.d_slots_rows),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            self.kv_dtype,
                        )?;
                    } else {
                        exec.kv_append_batch(
                            &sc.d_kn,
                            bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                            &sb.d_pos_rows,
                            Some(&sb.d_slots_rows),
                            kv_dim,
                            max_ctx,
                            r,
                            self.kv_dtype,
                        )?;
                        exec.kv_append_batch(
                            &sc.d_v,
                            bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                            &sb.d_pos_rows,
                            Some(&sb.d_slots_rows),
                            kv_dim,
                            max_ctx,
                            r,
                            self.kv_dtype,
                        )?;
                    }
                    // rung E1: the block-shared verify walk (k1 rows per
                    // slot block share one KV pass) where it engages; the
                    // per-row decode dispatch otherwise
                    let paged_bt = sb
                        .d_spec_tables
                        .as_ref()
                        .filter(|_| bs.paged)
                        .map(|bt| (bt, bs.blocks_per_slot));
                    let shared = super::ops::attn_verify_dispatch(
                        &exec,
                        &sc.d_qn,
                        bs.kv_k[li].as_ref().expect("full-attn layer KV"),
                        bs.kv_v[li].as_ref().expect("full-attn layer KV"),
                        sinks,
                        &mut sc.d_attn_o,
                        &mut sc.d_attn_ml,
                        &mut sc.d_attn,
                        &sb.d_pos_rows,
                        &sb.d_slots_rows,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_dim,
                        r,
                        k1,
                        scale,
                        self.kv_dtype,
                        paged_bt,
                        sb.max_pos_row as usize,
                    )?;
                    if !shared {
                        attn_decode_dispatch(
                            &exec,
                            &sc.d_qn,
                            bs.kv_k[li].as_ref().expect("full-attn layer KV"),
                            bs.kv_v[li].as_ref().expect("full-attn layer KV"),
                            sinks,
                            &mut sc.d_attn_o,
                            &mut sc.d_attn_ml,
                            &mut sc.d_attn,
                            &sb.d_pos_rows,
                            Some(&sb.d_slots_rows),
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_dim,
                            r,
                            scale,
                            self.kv_dtype,
                            paged_bt,
                        )?;
                    }
                    exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, r * q_dim)?;
                    if let Some(wo_t) = f8t_wo {
                        exec.quantize_e4m3_row(
                            &sc.d_attn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            w.wo.dims()[0],
                            r,
                        )?;
                        exec.f8t_gemm(
                            wo_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut sb.d_ks_part,
                            &mut sc.d_proj,
                            w.wo.dims()[0],
                            w.wo.dims()[1],
                            r,
                        )?;
                    } else if let Some(w8o) = l8d.and_then(|l| l.wo.as_ref()).filter(|_| f8_qkv) {
                        exec.quantize_e4m3(
                            &sc.d_attn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * w.wo.dims()[0],
                        )?;
                        exec.f8d_gemm_mma_ks(
                            w8o,
                            w.wo.dims()[0],
                            w.wo.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sb.d_ks_part,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else {
                        super::stub_guard(&w.wo, "spec.rs verify wo")?;
                        mmq(
                            &exec,
                            &w.wo,
                            &sc.d_attn,
                            &mut sb.d_xq,
                            &mut sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
                Mixer::Linear(w) => {
                    // was a bare `r >= 8`: the verify chunk is k+1 rows, so at
                    // c1 (k~2-4) every projection fell to the Q8_0 planes --
                    // invisible while both residencies exist, fatal under the
                    // projection REPLACE. Same literal-8 family as this walk's
                    // FFN and head gates (both since fixed).
                    let l8d = bs_w8
                        .get(li)
                        .filter(|_| r > super::w8_min_batch() && f8_lane);
                    let f8_dn = l8d.is_some_and(|l| l.in_qkv.is_some());
                    // tile lane for the merged in_qkv|gate_w plane, same
                    // landing and same two slices as the f8d arm below. Taken
                    // only when the plane is exactly nc+nz_ wide: a plane
                    // carrying the +128 alpha||beta fold would need a wider
                    // landing and a fourth slice, which is a separate change.
                    let mut f8t_ow: Option<&crate::gpu::F8TilePlane> = None;
                    // set when the tile plane carried alpha||beta: d_a/d_b come
                    // out of the same GEMM and their two mmq_pre calls are dead
                    let mut dn_ab_folded = false;
                    let (nin0, nc0) = (w.in_qkv.dims()[0], w.in_qkv.dims()[1]);
                    let nz0 = w.gate_w.dims()[1];
                    let f8t_dn = bs_f8t_attn.get(li).and_then(|o| o.as_ref()).filter(|p| {
                        let tot = p[0].scale.len();
                        spec_f8t && r <= 64 && (tot == nc0 + nz0 || tot == nc0 + nz0 + 128)
                    });
                    if let Some([in_t, ow_t]) = f8t_dn {
                        // the plane's scale length is its out_dim; +128 marks
                        // the alpha||beta fold (load.rs pushes fuse_ab when the
                        // export ships them Q8_0, which this one does)
                        let tot = in_t.scale.len();
                        dn_ab_folded = tot == nc0 + nz0 + 128;
                        static W: std::sync::Once = std::sync::Once::new();
                        W.call_once(|| eprintln!("[spec-f8t] engaged: verify DN in_proj on the tile lane (rows={r} folded_ab={dn_ab_folded})"));
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            nin0,
                            r,
                        )?;
                        exec.f8t_gemm(
                            in_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut sb.d_ks_part,
                            &mut sb.d_fused_land,
                            nin0,
                            tot,
                            r,
                        )?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_mixed, tot, 0, nc0, r)?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_z, tot, nc0, nz0, r)?;
                        if dn_ab_folded {
                            exec.row_slice(
                                &sb.d_fused_land,
                                &mut sc.d_a,
                                tot,
                                nc0 + nz0,
                                n_v_heads,
                                r,
                            )?;
                            exec.row_slice(
                                &sb.d_fused_land,
                                &mut sc.d_b,
                                tot,
                                nc0 + nz0 + n_v_heads,
                                n_v_heads,
                                r,
                            )?;
                        } else {
                            // alpha/beta ride the Q8 xq, exactly as the f8d arm
                            // does - the fused GEMM did not stage them
                            exec.quantize_q8(&sc.d_xn, &mut sb.d_xq, &mut sb.d_xs, r * embd)?;
                        }
                        f8t_ow = Some(ow_t);
                    } else if f8_dn {
                        let l8 = l8d.expect("f8_dn checked l8d");
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        let (nin, nc) = (w.in_qkv.dims()[0], w.in_qkv.dims()[1]);
                        let nz_ = w.gate_w.dims()[1];
                        // merged in_qkv|gate_w plane: one GEMM + slice, exactly
                        // the dense f8_dn arm (gate_w's own GEMM is skipped)
                        exec.f8d_gemm_mma_ks(
                            l8.in_qkv.as_ref().expect("f8_dn checked in_qkv"),
                            nin,
                            nc + nz_,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sb.d_ks_part,
                            &mut sb.d_fused_land,
                            r,
                        )?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_mixed, nc + nz_, 0, nc, r)?;
                        exec.row_slice(&sb.d_fused_land, &mut sc.d_z, nc + nz_, nc, nz_, r)?;
                        // alpha/beta below still ride the Q8 xq (the dense
                        // f8_dn arm does the same via bmm!) - stage it here
                        // since the fused GEMM no longer does
                        exec.quantize_q8(&sc.d_xn, &mut sb.d_xq, &mut sb.d_xs, r * embd)?;
                    } else {
                        super::stub_guard(&w.in_qkv, "spec.rs verify in_qkv")?;
                        mmq(
                            &exec,
                            &w.in_qkv,
                            &sc.d_xn,
                            &mut sb.d_xq,
                            &mut sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_mixed,
                            r,
                        )?;
                    }
                    // per-slot extended rows: window(slot) ++ chunk mixed rows,
                    // then causal conv emitting only the real rows
                    exec.conv_ext_build_slots(
                        bs.conv_win[li].as_ref().expect("DeltaNet layer conv"),
                        &sb.d_round_slots,
                        &sc.d_mixed,
                        sb.conv_ext[li].as_mut().expect("DeltaNet layer conv ext"),
                        b,
                        km1,
                        k1,
                        conv_dim,
                    )?;
                    exec.conv_chunk_ext(
                        sb.conv_ext[li].as_ref().expect("DeltaNet layer conv ext"),
                        &w.conv_w.buf,
                        &mut sc.d_conv,
                        b,
                        km1,
                        k1,
                        conv_dim,
                        conv_k,
                    )?;
                    if sb.vstash_k[li].is_some() {
                        // Snapshot-free verify (dflash): split k/v and
                        // the gates land in PER-LAYER stash planes that
                        // survive to commit, and the recurrence runs in hold
                        // mode - identical out[] values, no snapshots, live
                        // state stays at round-start. commit_walk recomputes
                        // the accepted prefix from these same planes.
                        exec.deltanet_split_gqa_norm(
                            &sc.d_conv,
                            &mut sc.d_dq,
                            sb.vstash_k[li].as_mut().expect("snapshot-free verify"),
                            sb.vstash_v[li].as_mut().expect("snapshot-free verify"),
                            r,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                        )?;
                        if let (Some(aw), Some(bw)) = (w.alpha_w.as_ref(), w.beta_w.as_ref()) {
                            if !dn_ab_folded {
                                mmq_pre(
                                    &exec,
                                    aw,
                                    &sb.d_xq,
                                    &sb.d_xs,
                                    &mut sb.d_ks_part,
                                    &mut sc.d_a,
                                    r,
                                )?;
                                mmq_pre(
                                    &exec,
                                    bw,
                                    &sb.d_xq,
                                    &sb.d_xs,
                                    &mut sb.d_ks_part,
                                    &mut sc.d_b,
                                    r,
                                )?;
                            }
                            exec.delta_gate(
                                &sc.d_a,
                                &sc.d_b,
                                &w.ssm_a.buf,
                                &w.dt_bias.buf,
                                sb.vstash_g[li].as_mut().expect("snapshot-free verify"),
                                sb.vstash_beta[li].as_mut().expect("snapshot-free verify"),
                                r,
                                n_v_heads,
                            )?;
                        } else {
                            let ab = w.ab_f32.as_ref().expect("ab_f32 (alpha_w is None)");
                            ab_gate(
                                &exec,
                                ab,
                                &sc.d_xn,
                                &mut sc.d_ab,
                                &w.ssm_a.buf,
                                &w.dt_bias.buf,
                                sb.vstash_g[li].as_mut().expect("snapshot-free verify"),
                                sb.vstash_beta[li].as_mut().expect("snapshot-free verify"),
                                r,
                                n_v_heads,
                            )?;
                        }
                        exec.gated_delta_verify_hold(
                            &sc.d_dq,
                            sb.vstash_k[li].as_ref().expect("snapshot-free verify"),
                            sb.vstash_v[li].as_ref().expect("snapshot-free verify"),
                            sb.vstash_g[li].as_ref().expect("snapshot-free verify"),
                            sb.vstash_beta[li].as_ref().expect("snapshot-free verify"),
                            Some(&sb.d_round_slots),
                            bs.recur[li].as_ref().expect("DeltaNet layer state"),
                            &mut sc.d_dattn,
                            b,
                            k1,
                            n_v_heads,
                            state_size,
                        )?;
                    } else {
                        exec.deltanet_split_gqa_norm(
                            &sc.d_conv,
                            &mut sc.d_dq,
                            &mut sc.d_dk,
                            &mut sc.d_dv,
                            r,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                        )?;
                        if let (Some(aw), Some(bw)) = (w.alpha_w.as_ref(), w.beta_w.as_ref()) {
                            mmq_pre(
                                &exec,
                                aw,
                                &sb.d_xq,
                                &sb.d_xs,
                                &mut sb.d_ks_part,
                                &mut sc.d_a,
                                r,
                            )?;
                            mmq_pre(
                                &exec,
                                bw,
                                &sb.d_xq,
                                &sb.d_xs,
                                &mut sb.d_ks_part,
                                &mut sc.d_b,
                                r,
                            )?;
                            exec.delta_gate(
                                &sc.d_a,
                                &sc.d_b,
                                &w.ssm_a.buf,
                                &w.dt_bias.buf,
                                &mut sc.d_g,
                                &mut sc.d_beta,
                                r,
                                n_v_heads,
                            )?;
                        } else {
                            // non-Q8 alpha/beta (k-quant files): the exact f32 ab
                            // route - P6b decay-numerics rule, same as serving
                            let ab = w.ab_f32.as_ref().expect("ab_f32 (alpha_w is None)");
                            ab_gate(
                                &exec,
                                ab,
                                &sc.d_xn,
                                &mut sc.d_ab,
                                &w.ssm_a.buf,
                                &w.dt_bias.buf,
                                &mut sc.d_g,
                                &mut sc.d_beta,
                                r,
                                n_v_heads,
                            )?;
                        }
                        exec.gated_delta_recurrent_v2(
                            &sc.d_dq,
                            &sc.d_dk,
                            &sc.d_dv,
                            &sc.d_g,
                            &sc.d_beta,
                            Some(&sb.d_round_slots),
                            bs.recur[li].as_mut().expect("DeltaNet layer state"),
                            0,
                            Some(sb.recur_snap[li].as_mut().expect("snapshot-verify mode")),
                            &mut sc.d_dattn,
                            b,
                            k1,
                            n_v_heads,
                            state_size,
                        )?;
                    }
                    // the tile arm slices d_z out of the fused plane just
                    // like the f8d arm, so it must suppress this GEMM too -
                    // otherwise gate_w runs twice and the slower answer wins
                    if !f8_dn && f8t_ow.is_none() {
                        super::stub_guard(&w.gate_w, "spec.rs verify gate_w")?;
                        mmq_pre_any(
                            &exec,
                            &w.gate_w,
                            &sb.d_xq,
                            &sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_z,
                            r,
                        )?;
                    }
                    exec.gated_rmsnorm(
                        &sc.d_dattn,
                        &sc.d_z,
                        &w.ssm_norm.buf,
                        &mut sc.d_core,
                        r * n_v_heads,
                        state_size,
                        eps,
                    )?;
                    if let Some(w8o) = l8d.and_then(|l| l.out_w.as_ref()).filter(|_| f8_dn) {
                        exec.quantize_e4m3(
                            &sc.d_core,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * w.out_w.dims()[0],
                        )?;
                        exec.f8d_gemm_mma_ks(
                            w8o,
                            w.out_w.dims()[0],
                            w.out_w.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sb.d_ks_part,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else if let Some(ow_t) = f8t_ow {
                        exec.quantize_e4m3_row(
                            &sc.d_core,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            w.out_w.dims()[0],
                            r,
                        )?;
                        exec.f8t_gemm(
                            ow_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut sb.d_ks_part,
                            &mut sc.d_proj,
                            w.out_w.dims()[0],
                            w.out_w.dims()[1],
                            r,
                        )?;
                    } else {
                        super::stub_guard(&w.out_w, "spec.rs verify out_w")?;
                        mmq(
                            &exec,
                            &w.out_w,
                            &sc.d_core,
                            &mut sb.d_xq,
                            &mut sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
            }
            exec.add_rmsnorm_batch(
                &mut sc.d_x,
                &sc.d_proj,
                &layer.post_norm.buf,
                &mut sc.d_xn,
                embd,
                eps,
                r,
            )?;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    // the tile lane, ahead of the f8d arm - the third and last
                    // place in this walk still electing a class the decode
                    // path abandoned. Same shape as the Nvf4Dense arm below.
                    let f8t_d = bs_f8t_ffn
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| spec_f8t && r <= 64);
                    // was a bare `r >= 8`, which left r < 8 on the Q8_0 arm --
                    // i.e. on the planes the reclaim stubs. Same literal-8 bug
                    // the FFN/head floors were split to kill; the verify walk
                    // is a prefill-class pass, so it takes the prefill floor.
                    let f8 = bs_f8ffn_bs
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .or_else(|| bs_f8ffn.get(li).and_then(|o| o.as_ref()))
                        .filter(|_| r > super::f8_ffn_pf_min());
                    if let Some(pr) = bs_f8row.get(li).and_then(|o| o.as_ref()) {
                        // checkpoint-exact fp8 layer (F8RowFfn): the same
                        // class the decode tick serves these layers with -
                        // the spec single-class rule holds by construction.
                        // Missing this site made every verify fall to the
                        // stubbed Q8_0 seats, and cost the wide spec lane
                        // ~40% of its throughput.
                        super::ops::ffn_f8row_rows(
                            &exec,
                            pr,
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else if let Some([gu_t, dn_t]) = f8t_d {
                        static W: std::sync::Once = std::sync::Once::new();
                        W.call_once(|| {
                            eprintln!(
                                "[spec-f8t] engaged: verify Dense FFN on the tile lane (rows={r})"
                            )
                        });
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            embd,
                            r,
                        )?;
                        exec.f8t_gemm(
                            gu_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_ffn_gate,
                            embd,
                            2 * ff,
                            r,
                        )?;
                        exec.swiglu_fused(&sc.d_ffn_gate, &mut sc.d_ffn_up, ff, r)?;
                        exec.quantize_e4m3_row(
                            &sc.d_ffn_up,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            ff,
                            r,
                        )?;
                        exec.f8t_gemm(
                            dn_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            ff,
                            embd,
                            r,
                        )?;
                    } else if let Some([gu8, d8]) = f8 {
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * gu8.1)?;
                        exec.f8d_gemm_mma_ks(
                            &gu8.0,
                            gu8.1,
                            gu8.2,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sb.d_ks_part,
                            &mut sb.d_fused_land,
                            r,
                        )?;
                        if exec.has_swiglu_fused_e4m3() {
                            exec.swiglu_fused_e4m3(
                                &sb.d_fused_land,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                ff,
                                r,
                            )?;
                        } else {
                            exec.swiglu_fused(&sb.d_fused_land, &mut sc.d_ffn_gate, ff, r)?;
                            exec.quantize_e4m3(
                                &sc.d_ffn_gate,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                r * d8.1,
                            )?;
                        }
                        exec.f8d_gemm_mma_ks(
                            &d8.0,
                            d8.1,
                            d8.2,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sb.d_ks_part,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else {
                        super::stub_guard(gate, "spec.rs record_spec_verify dense FFN")?;
                        mmq(
                            &exec,
                            gate,
                            &sc.d_xn,
                            &mut sb.d_xq,
                            &mut sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_ffn_gate,
                            r,
                        )?;
                        mmq_pre_any(
                            &exec,
                            up,
                            &sb.d_xq,
                            &sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_ffn_up,
                            r,
                        )?;
                        exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, r * ff)?;
                        mmq(
                            &exec,
                            down,
                            &sc.d_ffn_gate,
                            &mut sb.d_xq,
                            &mut sb.d_xs,
                            &mut sc.d_ssums,
                            &mut sb.d_ks_part,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
                Ffn::Nvf4Dense { gu, down } => {
                    // the verify walk gets the same arm the batch
                    // decode path elects (batch.rs, Ffn::Nvf4Dense). It did
                    // not, and on this die that was the whole spec lane:
                    // `nvf4_ffn`'s fast half is gated on has_nvf4_gemm_f4(),
                    // whose kernel family is `mma.sync ... kind::mxf4` under a
                    // __CUDA_ARCH__ gate satisfied only by sm_120a - so every
                    // cc-10 B200 fell to the W4A16 software-dequant chain.
                    // Profiling a spec round: pd_nvf4_gemm_tcp was 56.1% of
                    // all GPU time, 25704 launches x 133.59 us = 168/round
                    // (56 fp4 layers x 3 GEMMs) = 22.4 ms of FFN per verify
                    // round, against ~3.1 ms for this arm. The batch lane
                    // measured the same swap at ~2.7x on this checkpoint.
                    //
                    // NUMERICS: this IMPROVES the class match the note above
                    // was protecting. Decode already serves
                    // nv4cut+f8t; it was VERIFY that had drifted off-class.
                    // Kill: PADDOCK_NO_QWEN35_SPEC_NV4CUT.
                    static SPEC_NV4: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    let spec_nv4 = *SPEC_NV4.get_or_init(|| {
                        paddock_models::dev_var_os!("PADDOCK_NO_QWEN35_SPEC_NV4CUT").is_none()
                    });
                    // r > 64 keeps the old chain: that is f8t_gemm's elected
                    // band (the batch arm carries the same bound) and widening
                    // it is a separate measurement.
                    let f8t4 = bs_f8t_ffn
                        .get(li)
                        .and_then(|o| o.as_ref())
                        .filter(|_| spec_nv4 && r <= 64);
                    if let Some([gu_t, dn_t]) = f8t4 {
                        exec.quantize_e4m3_row(
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            embd,
                            r,
                        )?;
                        exec.f8t_gemm(
                            gu_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_ffn_gate,
                            embd,
                            2 * ff,
                            r,
                        )?;
                        exec.swiglu_fused(&sc.d_ffn_gate, &mut sc.d_ffn_up, ff, r)?;
                        exec.quantize_e4m3_row(
                            &sc.d_ffn_up,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            ff,
                            r,
                        )?;
                        exec.f8t_gemm(
                            dn_t,
                            &sc.d_f8t_q,
                            &sc.d_f8t_rs,
                            &mut bs.d_ks_part,
                            &mut sc.d_proj,
                            ff,
                            embd,
                            r,
                        )?;
                    } else {
                        // W4A4 class election for the batched VERIFY - the
                        // twin of the batch-draft flip (see mtp_block_pass_b
                        // for the full note).
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
                            r,
                        )?;
                    }
                }
                Ffn::Moe(w) => {
                    // sorted_ok=true: the historic `false` pinned the
                    // token-batched dp4a MoE so batched verify matched the
                    // single-slot dev refs bit-for-bit - but it made verify the
                    // whole round (@ B=32: gate_up_dp4a 836us + down_dp4a
                    // 362us per layer ≈ 48ms of the 80ms round, vs the mma pair's
                    // 116+62us on the dense path). Serving holds no cross-batch
                    // numeric-class invariant (the dense path itself switches MoE
                    // class with batch), and per-(B,K) the class stays
                    // fixed/deterministic. Also lets the fp4 grouped MoE engage
                    // at verify under the serving env - the same class the dense
                    // decode runs, i.e. the acceptance-relevant one.
                    moe_ffn(
                        &exec,
                        w,
                        self.moe.expect("moe dims"),
                        embd,
                        r,
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
            exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
        }

        exec.rmsnorm_batch(&sc.d_x, &self.out_norm.buf, &mut sb.d_h_chunk, embd, eps, r)?;
        // was `r >= 8`; see the draft-step site above.
        if let Some((p8, pi, po)) = super::head_f8(out_f8, r) {
            // f8d lm head at batched widths - the dense path's own b>=8 class
            exec.quantize_e4m3(&sb.d_h_chunk, &mut sc.d_pxq, &mut sc.d_exs, r * pi)?;
            exec.f8d_gemm_mma_ks(
                p8,
                *pi,
                *po,
                &sc.d_pxq,
                &sc.d_exs,
                &mut sb.d_ks_part,
                &mut sb.d_logits_chunk,
                r,
            )?;
        } else {
            super::stub_guard(&self.output, "spec.rs verify-chunk head")?;
            mmq(
                &exec,
                &self.output,
                &sb.d_h_chunk,
                &mut sb.d_xq,
                &mut sb.d_xs,
                &mut sc.d_ssums,
                &mut sb.d_ks_part,
                &mut sb.d_logits_chunk,
                r,
            )?;
        }
        exec.argmax_rows(&sb.d_logits_chunk, &mut sb.d_row_tok, r, vocab)?;
        Ok(())
    }

    /// Ragged per-slot commit: slot b keeps `committed[b]` chunk rows
    /// (accepted drafts + the correction token's context). Short slots roll
    /// their recurrent state back to the snapshot of their last committed row;
    /// every slot's conv window re-slices from its extended rows; positions
    /// advance per slot. KV needs no rollback (stale rows past pos are
    /// overwritten before any later read).
    fn commit_chunk_b_inner(
        &mut self,
        chunk: &[u32],
        committed: &[u32],
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        {
            let sb = self.spec_batch.as_mut().expect("spec batch");
            let (b, k1) = (sb.batch, sb.round_k1);
            assert_eq!(committed.len(), b);
            assert_eq!(chunk.len(), b * k1);
            for (i, &c) in committed.iter().enumerate() {
                assert!(c >= 1 && c as usize <= k1);
                // block i's true slot (round_slots); chunk rows stay block-major
                let s = sb.round_slots[i] as usize;
                sb.pos[s] += c as usize;
                // token shadow tracks the committed rows (mtp_catchup_b feeds
                // exactly these through the MTP block right after)
                sb.mtp_toks[s].extend_from_slice(&chunk[i * k1..i * k1 + c as usize]);
            }
            exec.stream
                .memcpy_htod(committed, &mut sb.d_committed)
                .map_err(drv)?;
        }
        // the two ragged kernels per DeltaNet layer read committed[] from the
        // staged device buffer, so the launch sequence is round-invariant
        if paddock_models::dev_var_os!("PADDOCK_SPEC_NOGRAPH").is_some() {
            self.record_spec_commit()?; // eager A/B path - identical launches
            return Ok(());
        }
        let live = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            (sb.batch, sb.round_k1)
        };
        if !self
            .spec_batch
            .as_ref()
            .expect("spec batch")
            .graph_commit
            .contains_key(&live)
        {
            let g = self.capture_spec_graph(Self::record_spec_commit, "spec commit")?;
            self.spec_batch
                .as_mut()
                .expect("spec batch")
                .graph_commit
                .insert(live, g);
        }
        self.spec_batch.as_ref().expect("spec batch").graph_commit[&live]
            .0
            .launch_on(&self.exec.stream)
            .map_err(|e| GpuError::Driver(format!("spec commit graph launch: {e}")))?;
        Ok(())
    }

    /// The per-layer commit kernels (capture-safe): state rollback to the
    /// snapshot of each slot's last committed row + conv-window re-slice, both
    /// indexed by the staged `d_committed`.
    fn record_spec_commit(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (n_v, s, conv_dim, km1) = (
            self.n_v_heads,
            self.state_size,
            self.conv_dim,
            self.conv_k - 1,
        );
        let n_layers = self.n_layers;
        let bs = self.batch.as_mut().expect("batch");
        let sb = self.spec_batch.as_mut().expect("spec batch");
        let (b, k1) = (sb.batch, sb.round_k1);
        for li in 0..n_layers {
            if sb.vstash_k[li].is_some() {
                // Snapshot-free commit (dflash): the live state is
                // still at ROUND-START (verify ran in hold mode) - recompute
                // it forward over each row's accepted prefix from the stashed
                // split/gate planes. Same fixed f32 op order as the verify
                // walk, so the result is bit-exact vs the snapshot the old
                // restore picked; committed[b] == 0 leaves the state alone.
                exec.gated_delta_commit_walk(
                    sb.vstash_k[li].as_ref().expect("snapshot-free verify"),
                    sb.vstash_v[li].as_ref().expect("snapshot-free verify"),
                    sb.vstash_g[li].as_ref().expect("snapshot-free verify"),
                    sb.vstash_beta[li].as_ref().expect("snapshot-free verify"),
                    Some(&sb.d_round_slots),
                    &sb.d_committed,
                    bs.recur[li].as_mut().expect("DeltaNet layer state"),
                    b,
                    k1,
                    n_v,
                    s,
                )?;
            } else if let Some(snap) = sb.recur_snap[li].as_ref() {
                exec.state_restore_slots(
                    bs.recur[li].as_mut().expect("DeltaNet layer state"),
                    snap,
                    &sb.d_round_slots,
                    &sb.d_committed,
                    b,
                    k1,
                    n_v,
                    s,
                )?;
            } else {
                continue; // full-attention layer: no recurrent state
            }
            exec.conv_commit_slots(
                sb.conv_ext[li].as_ref().expect("DeltaNet layer conv ext"),
                bs.conv_win[li].as_mut().expect("DeltaNet layer conv"),
                &sb.d_round_slots,
                &sb.d_committed,
                b,
                km1,
                k1,
                conv_dim,
            )?;
        }
        Ok(())
    }

    /// Batched MTP catch-up after a commit: feed every slot's committed chunk
    /// rows through the MTP block (per-slot KV) with the TARGET's h rows
    /// shifted right by one - row (b, 0) pairs with pending_h[b], row (b, j)
    /// with h_chunk[b][j-1] - then advance pending_h[b] to the slot's last
    /// committed h. One ragged pass over sum(committed) rows.
    fn mtp_catchup_b_inner(
        &mut self,
        chunk: &[u32],
        committed: &[u32],
        pos_before: &[usize],
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let embd = self.embd;
        let (b, k1) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            (sb.batch, sb.round_k1)
        };
        // ragged row plan (block-major; true slot per block via round_slots)
        let mut toks = Vec::new();
        let mut pos = Vec::new();
        let mut slots = Vec::new();
        {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            for i in 0..b {
                let c = committed[i] as usize;
                let s = sb.round_slots[i];
                for j in 0..c {
                    toks.push(chunk[i * k1 + j]);
                    pos.push((pos_before[i] + j) as u32);
                    slots.push(s);
                }
            }
        }
        let rows = toks.len();
        assert!(rows <= b * k1);
        {
            let sb = self.spec_batch.as_mut().expect("spec batch");
            exec.stream
                .memcpy_htod(&toks, &mut sb.d_mtp_tok)
                .map_err(drv)?;
            // h inputs, shifted right within each block; pending_h is
            // TRUE-slot-strided, the chunk h/logits buffers are block-strided
            let mut row = 0usize;
            for (i, &ci) in committed[..b].iter().enumerate() {
                let c = ci as usize;
                let s = sb.round_slots[i] as usize;
                for j in 0..c {
                    if j == 0 {
                        exec.copy_region(&sb.pending_h, s * embd, &mut sb.d_hin, row * embd, embd)?;
                    } else {
                        exec.copy_region(
                            &sb.d_h_chunk,
                            (i * k1 + j - 1) * embd,
                            &mut sb.d_hin,
                            row * embd,
                            embd,
                        )?;
                    }
                    row += 1;
                }
            }
        }
        self.stage_spec_rows(&pos, &slots)?;
        self.mtp_project_inputs_b(rows)?;
        self.mtp_block_pass_b(rows, false)?;
        {
            let sb = self.spec_batch.as_mut().expect("spec batch");
            for (i, &ci) in committed[..b].iter().enumerate() {
                let c = ci as usize;
                let s = sb.round_slots[i] as usize;
                exec.copy_region(
                    &sb.d_h_chunk,
                    (i * k1 + c - 1) * embd,
                    &mut sb.pending_h,
                    s * embd,
                    embd,
                )?;
            }
        }
        Ok(())
    }

    /// Greedy decode of B concurrent prompts with PER-SLOT MTP speculative
    /// decoding: every round drafts n_draft tokens per slot (MTP head over B
    /// rows per step), verifies all slots in one backbone pass over B×(K+1)
    /// rows, and commits each slot's accepted run + correction token. Greedy
    /// verification keeps every stream BIT-IDENTICAL to the non-spec path -
    /// the batched draft only changes how many tokens each weight-read pass
    /// yields per slot. Requires `enable_batch`; slots 0..prompts.len().
    pub fn generate_greedy_spec_batch(
        &mut self,
        prompts: &[Vec<u32>],
        max_new: usize,
        n_draft: usize,
    ) -> Result<Vec<Vec<u32>>, GpuModelError> {
        let b = prompts.len();
        assert!(b >= 1 && max_new >= 1);
        assert!(self.mtp.is_some(), "model has no nextn/MTP block");
        let need_alloc = match self.spec_batch.as_ref() {
            Some(sb) => sb.batch != b || sb.n_draft != n_draft,
            None => true,
        };
        if need_alloc {
            self.enable_spec_batch(b, n_draft)?;
        }
        let exec = self.exec.clone();
        // consume only what the chain graph actually recorded - with a block
        // drafter attached that is the MTP election, not the alloc depth
        // (rows chain_depth..n_draft of d_draft are never written then)
        let n_draft = self.spec_batch.as_ref().expect("spec batch").chain_depth;
        let k1 = n_draft + 1;

        // prefill each slot, then warm its MTP KV while sc.d_h still holds the
        // slot's prompt h rows; pending_h starts zeroed for fresh sequences
        {
            let sb = self.spec_batch.as_mut().expect("spec batch");
            let zeros = vec![0f32; b * self.embd];
            let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
            exec.stream
                .memcpy_htod(&zeros, &mut sb.pending_h)
                .map_err(drv)?;
        }
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(b);
        for (slot, prompt) in prompts.iter().enumerate() {
            assert!(!prompt.is_empty());
            let logits = self.forward_prefill_slot(slot, prompt)?;
            self.mtp_warm_slot(slot, prompt, 0, 0)?;
            let sb = self.spec_batch.as_mut().expect("spec batch");
            sb.pos[slot] = prompt.len();
            sb.mtp_warm[slot] = true;
            sb.mtp_toks[slot] = prompt.clone();
            out.push(vec![argmax(&logits)]);
        }

        // PADDOCK_SPEC_PHASE_TIME: per-phase wall attribution. The round already
        // host-syncs at the draft->verify and verify->commit boundaries (the two
        // to_host_u32 calls), so the only added sync is after commit+catchup.
        let phase_time = paddock_models::dev_var_os!("PADDOCK_SPEC_PHASE_TIME").is_some();
        let (mut t_draft, mut t_verify, mut t_tail) = (
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
        );
        let (mut n_rounds, mut n_committed) = (0usize, 0usize);

        while out.iter().any(|o| o.len() < max_new) {
            // stop cleanly if any slot is about to run out of context
            {
                let sb = self.spec_batch.as_ref().expect("spec batch");
                if (0..b).any(|s| sb.pos[s] + k1 > self.max_ctx) {
                    break;
                }
            }
            let t0 = std::time::Instant::now();
            let last: Vec<u32> = out
                .iter()
                .map(|o| *o.last().expect("non-empty: seeded with the prefill pick"))
                .collect();
            // the pooled drafter stripe appends at pos..pos+k during the draft
            // graph - back those rows with real blocks before it launches (the
            // verify's own ensure runs too late for the draft's writes)
            {
                let pos: Vec<usize> =
                    self.spec_batch.as_ref().expect("spec batch").pos[..b].to_vec();
                for (slot, &p) in pos.iter().enumerate() {
                    self.ensure_slot_blocks(slot, p + k1)?;
                }
            }
            let drafts = self.mtp_draft_b(&last)?; // i-major [n_draft, B]
            if phase_time {
                t_draft += t0.elapsed(); // mtp_draft_b ends host-synced (to_host_u32)
            }

            let t1 = std::time::Instant::now();
            // slot-major chunk rows: [last_b, draft_b0, .., draft_b(K-1)]
            let mut chunk = Vec::with_capacity(b * k1);
            for slot in 0..b {
                chunk.push(last[slot]);
                for i in 0..n_draft {
                    chunk.push(drafts[i * b + slot]);
                }
            }
            let pos_before: Vec<usize> = self.spec_batch.as_ref().expect("spec batch").pos.clone();
            self.forward_chunk_b(&chunk)?;
            let row_toks =
                exec.to_host_u32(&self.spec_batch.as_ref().expect("spec batch").d_row_tok)?;
            if phase_time {
                t_verify += t1.elapsed();
            }
            let t2 = std::time::Instant::now();

            // per-slot greedy acceptance: drafts stand while they match the
            // target's own next-token picks
            let mut committed = Vec::with_capacity(b);
            for slot in 0..b {
                let mut a = 0usize;
                while a < n_draft && chunk[slot * k1 + a + 1] == row_toks[slot * k1 + a] {
                    a += 1;
                }
                committed.push((a + 1) as u32);
            }
            if paddock_models::dev_var_os!("PADDOCK_SPEC_TRACE").is_some() {
                for slot in 0..b {
                    tracing::info!(
                        "TRACE batch slot {slot}: pos {} chunk {:?} targets {:?} committed {}",
                        pos_before[slot],
                        &chunk[slot * k1..(slot + 1) * k1],
                        &row_toks[slot * k1..(slot + 1) * k1],
                        committed[slot]
                    );
                }
            }
            self.commit_chunk_b(&chunk, &committed)?;
            self.mtp_catchup_b(&chunk, &committed, &pos_before)?;
            if phase_time {
                exec.synchronize()?;
                t_tail += t2.elapsed();
                n_rounds += 1;
                n_committed += committed.iter().sum::<u32>() as usize;
            }

            for slot in 0..b {
                let c = committed[slot] as usize;
                for j in 0..c {
                    if out[slot].len() < max_new {
                        out[slot].push(row_toks[slot * k1 + j]);
                    }
                }
            }
        }
        if phase_time && n_rounds > 0 {
            let per = |d: std::time::Duration| d.as_secs_f64() * 1e3 / n_rounds as f64;
            tracing::info!(
                "SPEC PHASE B={b} K={n_draft}: rounds {n_rounds}, committed/round {:.2}; \
                 per round draft {:.2}ms verify {:.2}ms commit+catchup {:.2}ms  (total {:.2}ms)",
                n_committed as f64 / n_rounds as f64,
                per(t_draft),
                per(t_verify),
                per(t_tail),
                per(t_draft + t_verify + t_tail),
            );
        }
        Ok(out)
    }

    // ---------------- serving MTP spec (greedy) ----------------
    // Env-gated PADDOCK_QWEN35_SPEC. Spec state is allocated once at
    // (SPEC_LIVE_MAX, K); rounds run over the first `live` slots by setting
    // sb.batch = live (all state arrays index slot-relative, so only row
    // counts change) with the round graphs re-captured when live changes.
    // The MTP KV/h warm rides the serial prefill path (see the hook in
    // prefill_slot_chunk); a slot whose position desyncs (dense ticks in
    // between, prefix resume) serves dense until its next fresh prefill.

    pub(crate) fn serve_spec_on(&self) -> bool {
        // k-quant models load the MTP block too - the spec matmuls ride
        // mmq's W4A8 rungs, so the same env gate governs both.
        // A sideloaded DFlash drafter is the second drafter class.
        (self.mtp.is_some() || self.dflash.is_some())
            && std::env::var_os("PADDOCK_QWEN35_SPEC").is_some()
    }

    /// Legacy snapshot-rollback verify? Default is the snapshot-free pair
    /// (verify_hold + commit_walk, pack slots 462/463, dflash): the
    /// per-token snapshots were ~87% of the ~1.15 GiB/spec-row draft state
    /// and O(k1) state writes per round - the shape that made wide-batch
    /// spec lose. PADDOCK_QWEN35_SPEC_SNAPSHOT=1 restores the old path (bring-up
    /// bit-cmp A/B); a pack without the new slots falls back automatically.
    pub(super) fn spec_snapshot_verify(&self) -> bool {
        paddock_models::dev_var_os!("PADDOCK_QWEN35_SPEC_SNAPSHOT").is_some()
            || !self.exec.has_gated_delta_commit_walk()
    }

    /// Draft-state live preference fed to the width sizer (which degrades
    /// live before width under VRAM pressure - the election here is a CAP,
    /// not a reservation). Drafter-conditional (rung B): 32 with a
    /// block drafter attached - the batched round + 64-row budget make
    /// width drafting pay (b32r64: c16 856 / c32 913 vs dense 838) and the
    /// snapshot-free verify prices 32 rows at ~1.45 GiB on 27B. MTP-only
    /// lanes (qwen3.5-9b, qwen3.6) keep 8: their chain at width measured
    /// worse than dense (cap-16 A/B, service.rs:1850 history) and no width
    /// re-measurement exists for them - raise only with evidence.
    /// Widths at which speculation is allowed to engage at all.
    ///
    /// A drafter attaching used to raise this to 32 unconditionally. On sm_100
    /// that was worth two board cells, because **every threshold in this
    /// chain had been elected spec-vs-spec** - `dflash_live_max` went
    /// 1 -> 8 -> 32 on arguments that only ever beat the other spec arm.
    /// Nothing in the policy ever compared speculating against not
    /// speculating, and on a die where speculation always won that omission
    /// was free.
    ///
    /// It is not free here. qwen3.8 nvfp4 on B200, our spec against our own
    /// no-spec, with the spec path as fast as it has ever been:
    ///
    ///   c1 2.84x  c4 2.17x  c8 1.57x  |  c16 0.45x  c32 0.44x  imax 0.39x
    ///
    /// The crossover sits between c8 and c16 and it is not close on either
    /// side. Capping there roughly DOUBLES wide-batch throughput - by
    /// declining to speculate, not by making speculation faster.
    ///
    /// Die-aware rather than a blanket flip: sm_120 is measured at 32
    /// and its target decode is ~3x slower, so the verify tick still amortises
    /// there. The honest version of this is an election that MEASURES spec
    /// against no-spec per width instead of carrying a constant per die; this
    /// default is the interim, and the fact that it had to be measured on each
    /// die separately is the whole lesson.
    pub(super) fn serve_spec_live_max(&self) -> usize {
        paddock_models::dev_var!("PADDOCK_QWEN35_SPEC_LIVE_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                if self.dflash.is_none() {
                    return 8;
                }
                if self.exec.compute_capability().0 == 12 {
                    32
                } else {
                    8
                }
            })
    }

    /// Hybrid drafter threshold: DFlash2 rounds at live <= this, the MTP
    /// chain above. This was 1 at first, because a mixed-load hybrid at
    /// threshold 4 collapsed - but that measured a binary before the
    /// per-round k-miss floor (spec_k_miss_floor_mtp), the ring-rides-radix
    /// pool, and the pool-stripe finisher-warmth rungs. A same-build A/B
    /// under continuous admissions has T=8 clearly ahead of T=1 at c4 and c8
    /// with c1 unchanged (live=1 routes here either way), and no collapse
    /// remains. NOTE the row-budget interaction: serve_spec_k_budget folds
    /// block rounds to k=3 at live 8, and that is correct today - a
    /// full-block probe (SPEC_MAX_ROWS=64 + DEEP_LIVE_MAX=8, k=7 at live 8
    /// = 64 verify rows) measured worse than the k=3 fold.
    /// Depth at width only pays once the batched block round (one drafter
    /// forward over live x 8 rows) replaces the per-row round train - which
    /// is what raised this from 8 to 32: the drafter forward batches all rows
    /// (graphs keyed (n, rows)), the chain arm kills the per-round dtoh, the
    /// snapshot-free verify prices 32 rows at ~1.45 GiB, and
    /// max_blocks/maintenance moved to 32 with it. On that shape (64-row
    /// serving budget, k=3 at 9..16 / k=1 at 17..32) c16 gains ~45% and c32
    /// beats dense at 91.5%/token. Above 32 the MTP chain path takes over
    /// (and the
    /// serving plans gate disengages spec anyway).
    pub(crate) fn dflash_live_max() -> usize {
        paddock_models::dev_var!("PADDOCK_QWEN35_DFLASH_LIVE_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32)
    }

    /// The MTP chain's own draft depth - the pre-hybrid serve_spec_k logic.
    /// serve_spec_k() returns the ALLOC bound (max of both drafters' depths)
    /// once a DFlash drafter is attached, so chain sites clamp with this.
    pub(super) fn serve_spec_k_mtp(&self) -> usize {
        let default = if self.moe.is_none() && self.embd <= 4096 {
            4
        } else {
            3
        };
        paddock_models::dev_var!("PADDOCK_QWEN35_SPEC_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
            .clamp(1, 4)
    }

    pub(super) fn serve_spec_k(&self) -> usize {
        // Dense-MTP default 4 (was 3): with the near-full acceptance of the
        // class-aligned draft chain the 4th draft is nearly free on the small
        // dense model. MoE models stay at 3: their MTP block runs the routed
        // experts per draft step (pricier chain, and the k1=5 spec state
        // squeezes serving width hard).
        // Measured boundary, single-stream A/Bs:
        //   9B  (dense, embd 4096): k=4 is a clear win      -> 4
        //   27B (dense, embd 5120): k=4 regresses, k=3 restores -> 3
        //   35B (MoE): k=4 regressed at every width         -> 3
        // Only the small dense model repays depth 4 today; dense && embd<=4096
        // is the discriminator that holds across the measured set.
        let default = if self.moe.is_none() && self.embd <= 4096 {
            4
        } else {
            3
        };
        // DFlash2: one drafter forward covers the whole block, so draft depth
        // is nearly free - default to block-1 (7). Verify rows are not free
        // (live * (k+1) target rows), so the env override still wins for
        // depth A/Bs; the clamp widens to the drafter's own block.
        if let Some(df) = self.dflash.as_ref() {
            return paddock_models::dev_var!("PADDOCK_QWEN35_SPEC_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(df.block - 1)
                .clamp(1, df.block - 1);
        }
        paddock_models::dev_var!("PADDOCK_QWEN35_SPEC_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
            .clamp(1, 4)
    }

    /// One-time lazy allocation of the serving spec state.
    pub(super) fn ensure_serve_spec(&mut self) -> Result<(), GpuModelError> {
        if self.spec_batch.is_some() {
            return Ok(());
        }
        let max_batch = self.batch.as_ref().map(|b| b.max_batch).unwrap_or(1);
        let live_max = self
            .spec_live_vram_cap
            .unwrap_or_else(|| self.serve_spec_live_max())
            .clamp(1, max_batch);
        self.enable_spec_batch(live_max, self.serve_spec_k())
    }

    /// Point the round machinery at the first `live` slots. Round graphs are
    /// cached per live COUNT (they bake row counts) - nothing is dropped
    /// here. The old drop-on-change behavior re-instantiated all three
    /// graphs on every live transition, and at width the live count churns
    /// every finish/admission: the spec round cost ~98 ms against a 24.5 ms
    /// dense tick and the controller learned the poisoned latencies.
    fn spec_set_live(&mut self, live: usize) {
        let sb = self.spec_batch.as_mut().expect("spec batch");
        assert!(live >= 1 && live <= sb.alloc_batch);
        sb.batch = live;
    }

    /// Re-sync a warm slot whose MTP state fell behind the decode cursor. The
    /// prefill hook warms through the prompt, but dense/mixed ticks (Phase 2m,
    /// which preempts spec while any slot chunks) then commit tokens that only
    /// advance the backbone KV - `sb.pos`/MTP-KV stay at the warm boundary. In
    /// a cohort, early-finishing slots take several such ticks before spec
    /// resumes, so `slot.pos` runs ahead of `sb.pos` and every spec round
    /// declines on the pos check (a single-slot c1 never sees this - it decodes
    /// spec from tick 0). `committed` is the slot's committed token sequence
    /// (`history[0..slot.pos]`); we re-warm just the gap `committed[sb.pos..]`,
    /// which is cheap (a few backbone rows) and paid once per dense interruption
    /// since spec commits keep `sb.pos` synced thereafter. `pending_h` is
    /// untouched by dense ticks, so the MTP chain is intact - only the warm
    /// flag needs restoring. Returns whether the slot is warm through `slot.pos`.
    /// Scheduler hint (see `Generator::spec_warm_hint`): gates the eager
    /// prefill warm hooks. Live > spec cap -> warming is unrecoverable cost.
    pub fn set_spec_warm_wanted(&mut self, on: bool) {
        self.spec_warm_wanted = on;
    }

    /// See the `spec_ring_probed` field.
    pub(crate) fn set_spec_ring_probed(&mut self, on: bool) {
        self.spec_ring_probed = on;
    }

    /// Effective spec-round live capacity (see `Generator::spec_live_cap`):
    /// the allocated draft-state batch once it exists, else what
    /// ensure_serve_spec would allocate (VRAM-degraded cap or the env
    /// default). The 35B-Q8-on-48GB case: alloc 1, but the service's own
    /// engagement cap is 4 - without this clamp, ramp windows at live 2..4
    /// warm slots and draft-decline every mixed tick, and ensure_warm re-warms
    /// an ever-growing gap that no round ever re-syncs (measured c32 collapse
    /// 440 -> 67 t/s).
    pub fn spec_live_cap_mtp(&self) -> usize {
        if !self.serve_spec_on() {
            return usize::MAX; // spec off - the cap never gates anything
        }
        self.spec_batch
            .as_ref()
            .map(|sb| sb.alloc_batch)
            .unwrap_or_else(|| {
                self.spec_live_vram_cap
                    .unwrap_or_else(|| self.serve_spec_live_max())
            })
    }

    pub fn spec_ensure_warm_mtp(
        &mut self,
        slot: usize,
        committed: &[u32],
    ) -> Result<bool, GpuModelError> {
        let dbg = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some();
        if !self.serve_spec_on() {
            if dbg {
                tracing::info!(
                    "[spec-warm] slot {slot}: serve_spec_on=false (mtp={})",
                    self.mtp.is_some()
                );
            }
            return Ok(false);
        }
        self.ensure_serve_spec()?;
        let target = committed.len();
        let (warm, w, alloc) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            (
                sb.mtp_warm.get(slot).copied().unwrap_or(false),
                sb.pos.get(slot).copied().unwrap_or(0),
                sb.alloc_batch,
            )
        };
        if slot >= alloc || !warm || w > target {
            if dbg {
                tracing::info!(
                    "[spec-warm] slot {slot}: COLD (alloc={alloc} warm={warm} pos={w} target={target})"
                );
            }
            return Ok(false); // cold slot (long prompt / prefix resume) - serve dense
        }
        if w == target {
            return Ok(true); // already synced (steady spec decode - the hot path)
        }
        // Gap bound: the gap re-warm exists for dense-interlude
        // desyncs - a few tokens per preempted tick. An unbounded gap means a
        // prefill lane advanced the slot far past its warm point (a 64K prompt
        // whose chunks stop warming at warm_max leaves warm=true with the
        // cursor at ~16K): re-prefilling tens of thousands of rows serially
        // inside a tick stalled every neighbor for seconds and overran the
        // conv-ext staging assert ("resume chunk 61843 rows outgrew"), killing
        // the engine thread. Past the bound the slot takes the documented
        // cold-slot path: serve dense, re-warm at its next fresh prefill.
        const WARM_GAP_MAX: usize = 512;
        if target - w > WARM_GAP_MAX {
            self.spec_batch.as_mut().expect("spec batch").mtp_warm[slot] = false;
            if dbg {
                tracing::info!(
                    "[spec-warm] slot {slot}: gap {} > {WARM_GAP_MAX} - going cold until next prefill",
                    target - w
                );
            }
            return Ok(false);
        }
        // Prime chain state is already valid (warm && sb.pos==w), so
        // prefill_slot_chunk's internal warm hook extends the MTP KV over the
        // gap and bumps sb.pos to `target`. The backbone re-run's KV rewrite
        // stays in the documented bounded-nudge class and its h rows feed only
        // the drafter (verify re-judges) - but its STATE side-effects must be
        // fenced (see spec_warm_rewarm).
        self.spec_warm_rewarm(slot, &committed[w..target], w)?;
        let sb = self.spec_batch.as_ref().expect("spec batch");
        Ok(sb.mtp_warm[slot] && sb.pos[slot] == target)
    }

    /// Gap re-warm with the backbone's stateful side-effects fenced off.
    /// `prefill_slot_chunk` re-runs the backbone over tokens
    /// the slot's live DeltaNet state has already consumed. The recurrence is
    /// not idempotent, and the conv path resumes from the window at `target`
    /// rather than `w` - so every re-warm double-advanced the state through
    /// the gap and re-sliced the window from the wrong position. Under mixed
    /// ticks (a neighbor slot chunking - the exact condition that CREATES
    /// warm gaps) this compounded per tick and the model repeated its own
    /// recent fragment ("OBSIDIANIAN", digit loops). Invisible at c1: no
    /// interleaved ticks, no gap, no re-warm. Save the slot's state + conv
    /// windows, run the warm, restore both - even if the warm errors.
    fn spec_warm_rewarm(
        &mut self,
        slot: usize,
        tokens: &[u32],
        w: usize,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (n_v, s) = (self.n_v_heads, self.state_size);
        let (km1, conv_dim) = (self.conv_k - 1, self.conv_dim);
        let state_elems = n_v * s * s;
        let per_layer = state_elems + km1 * conv_dim;
        let linear: Vec<usize> = self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l.mixer, Mixer::Linear(_)))
            .map(|(li, _)| li)
            .collect();
        if self
            .spec_batch
            .as_ref()
            .expect("spec batch")
            .warm_stash
            .is_none()
        {
            let buf = exec.alloc(per_layer * linear.len())?;
            self.spec_batch.as_mut().expect("spec batch").warm_stash = Some(buf);
        }
        {
            let bs = self.batch.as_ref().expect("batch");
            let sb = self.spec_batch.as_mut().expect("spec batch");
            let stash = sb.warm_stash.as_mut().expect("warm stash allocated above");
            for (j, &li) in linear.iter().enumerate() {
                let so = j * per_layer;
                exec.copy_region(
                    bs.recur[li].as_ref().expect("DeltaNet layer state"),
                    slot * state_elems,
                    stash,
                    so,
                    state_elems,
                )?;
                exec.copy_region(
                    bs.conv_win[li].as_ref().expect("DeltaNet layer conv"),
                    slot * km1 * conv_dim,
                    stash,
                    so + state_elems,
                    km1 * conv_dim,
                )?;
            }
        }
        let ran = self.prefill_slot_chunk(slot, tokens, w).map(|_| ());
        {
            let bs = self.batch.as_mut().expect("batch");
            let sb = self.spec_batch.as_ref().expect("spec batch");
            let stash = sb.warm_stash.as_ref().expect("warm stash allocated above");
            for (j, &li) in linear.iter().enumerate() {
                let so = j * per_layer;
                exec.copy_region(
                    stash,
                    so,
                    bs.recur[li].as_mut().expect("DeltaNet layer state"),
                    slot * state_elems,
                    state_elems,
                )?;
                exec.copy_region(
                    stash,
                    so + state_elems,
                    bs.conv_win[li].as_mut().expect("DeltaNet layer conv"),
                    slot * km1 * conv_dim,
                    km1 * conv_dim,
                )?;
            }
        }
        ran
    }

    /// Per-round post-miss k floor for the service controller. DFlash2
    /// drafts its whole block in one forward - depth is nearly free, and the
    /// classic post-miss rule made it spend most rounds re-climbing (the
    /// measured cross-leg k death-spiral, c4 457 -> 159). The MTP chain pays
    /// per depth, so it keeps the classic floor. Answered by which drafter
    /// actually ran (spec_round_dflash), replacing the attach-time
    /// PADDOCK_SPEC_K_MISS_FLOOR env default that pinned floor 7 on MTP
    /// rounds too (measured as the attached-serve residual: c4 290 vs 361).
    pub fn spec_k_miss_floor_mtp(&self) -> Option<usize> {
        if self.spec_round_dflash {
            self.dflash.as_ref().map(|d| d.block - 1)
        } else {
            None
        }
    }

    /// Serving draft hook: MTP drafts for the live slots (must be contiguous
    /// 0..n and warm). None = the service falls back to its n-gram drafter.
    pub fn spec_draft_batch_mtp(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GpuModelError> {
        // HYBRID routing: DFlash2 drafts the round at low live (block drafts
        // pay hugely there), the MTP chain takes it above the threshold. A
        // dflash decline (cold ring) falls through to the MTP path.
        //
        // ONE-SHOT: the ring-probed flag belongs to the round that follows
        // the scheduler's probe pass and is consumed here. Left standing, it
        // leaked into the sampled host-walk round (service.rs, live <= 4),
        // which drafts with no warm pass: when the warm slot finished and
        // the replacements were all ring-cold, the block lane declined and
        // the stale flag refused the MTP fallback the old code took (fresh
        // slots' cursors are prefill-synced, so it was legitimate). It cost
        // ~9% at c4 while c1 and c8+ held.
        let ring_probed = std::mem::take(&mut self.spec_ring_probed);
        if self.dflash_attached()
            && pendings.len() <= Self::dflash_live_max()
            && let Some(out) = self.dflash_draft_batch(pendings, k)?
        {
            self.spec_round_dflash = true;
            return Ok(Some(out));
        }
        self.spec_round_dflash = false;
        let k = k.min(self.serve_spec_k_mtp());
        if !self.serve_spec_on() || pendings.is_empty() || k == 0 {
            return Ok(None);
        }
        let dbg = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some();
        if ring_probed {
            // The scheduler ring-probed this round (a ring was warm, so the
            // block drafter was expected to take it) and the block lane still
            // declined - e.g. every warm slot sits at max_ctx. The MTP cursors
            // were not gap-synced for this round; a chain round from a stale
            // cursor desyncs the chain, so decline (dense tick) rather than
            // draft unsynced. The next full warm pass re-arms the chain.
            if dbg {
                tracing::info!("[spec-draft] decline: ring-probed round, MTP cursors unsynced");
            }
            return Ok(None);
        }
        let n = pendings.len();
        {
            let Some(sb) = self.spec_batch.as_ref() else {
                if dbg {
                    tracing::info!("[spec-draft] decline: no spec_batch state");
                }
                return Ok(None);
            };
            if n > sb.alloc_batch {
                if dbg {
                    tracing::info!("[spec-draft] decline: n={n} > alloc={}", sb.alloc_batch);
                }
                return Ok(None);
            }
            // RAGGED slot sets accepted: rounds no longer require
            // pendings[i].slot == i. The old contiguity decline sent every
            // churned round down the mixed tick's chunk lane, where the verify
            // executed as a 2-6-row eager 65-layer Q8 pass (~40 ms/round - the
            // c4-c32 spec loss). Only alloc coverage + warmth gate now.
            for &(slot, _) in pendings.iter() {
                if slot >= sb.alloc_batch || !sb.mtp_warm[slot] {
                    if dbg {
                        tracing::info!(
                            "[spec-draft] decline: slot {slot} alloc={} warm={}",
                            sb.alloc_batch,
                            sb.mtp_warm.get(slot).copied().unwrap_or(false)
                        );
                    }
                    return Ok(None);
                }
            }
        }
        let k_use = k.min(self.spec_batch.as_ref().expect("spec batch").n_draft);
        self.spec_set_live(n);
        self.spec_batch.as_mut().expect("spec batch").round_slots[..n]
            .copy_from_slice(&pendings.iter().map(|&(s, _)| s as u32).collect::<Vec<_>>());
        let last: Vec<u32> = pendings.iter().map(|&(_, t)| t).collect();
        let td = std::time::Instant::now();
        // the pooled drafter stripe appends at pos..pos+k during the draft
        // graph - back those rows with real blocks before it launches (the
        // verify's own ensure runs too late for the draft's writes)
        {
            let sbk = self.spec_batch.as_ref().expect("spec batch");
            let k1e = sbk.n_draft + 1;
            let sp: Vec<(usize, usize)> = pendings.iter().map(|&(s, _)| (s, sbk.pos[s])).collect();
            for (s, p) in sp {
                self.ensure_slot_blocks(s, p + k1e)?;
            }
        }
        self.stage_spec_tables()?;
        let drafts = self.mtp_draft_b(&last)?; // i-major [n_draft, n]
        if dbg {
            tracing::info!(
                "[spec-round-t] draft_chain={}us (k_use={k_use})",
                td.elapsed().as_micros()
            );
        }
        Ok(Some(
            (0..n)
                .map(|s| (0..k_use).map(|i| drafts[i * n + s]).collect())
                .collect(),
        ))
    }

    /// Async draft chain, phase 1: LAUNCH the draft
    /// graph and return without the ids readback - the measured ~5.2 ms
    /// host stall per round on the chain->verify boundary becomes queued
    /// stream work. The verify assembles its token rows on device from
    /// `d_draft` (pd_spec_toks, see forward_chunk_b); real values surface
    /// post-verify via the drivers' peek and the service's
    /// `spec_draft_fetch`. Same eligibility as the synchronous round;
    /// `Ok(None)` = fall back to `spec_draft_batch_mtp`.
    /// Async block round, phase 1 - the dflash twin of the chain:
    /// stage + launch the block-draft graph
    /// and copy its picks DEVICE-SIDE into the chain's `d_draft` (slot 464),
    /// then arm `spec_chain`. The armed verify assembles its tokens on
    /// device exactly as for MTP chains; the host readback happens
    /// post-verify via the chain peek - the per-round draft dtoh sync that
    /// the synchronous dflash path pays is gone. `Ok(None)` = ineligible
    /// (cold ring, alloc miss, old pack): the caller falls through to the
    /// MTP chain arm or the synchronous paths, exactly as before.
    fn dflash_draft_begin(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<(usize, Vec<bool>)>, GpuModelError> {
        if k == 0
            || pendings.is_empty()
            || !self.dflash_armed()
            || !self.exec.has_dflash_chain_picks()
            || !self.exec.has_spec_toks()
            || paddock_models::dev_var_os!("PADDOCK_NO_DFLASH_CHAIN").is_some()
        {
            return Ok(None);
        }
        // Rung A: screen the round's block capacity before any
        // staging work. dflash_draft_launch re-checks it, but by then a
        // declining width round has already paid ensure_serve_spec +
        // per-slot ensure_slot_blocks + stage_tables + spec_set_live -
        // every TICK (the w32k7 probe measured c32 at 37 t/s from this
        // arm-and-decline loop; the isolated decline shape cost ~25% vs
        // the engaged round). Decline here, where it costs a comparison.
        if pendings.len() > self.dflash_round_cap() {
            return Ok(None);
        }
        self.ensure_serve_spec()?;
        let n = pendings.len();
        {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            if n > sb.alloc_batch || pendings.iter().any(|&(s, _)| s >= sb.alloc_batch) {
                return Ok(None);
            }
        }
        let (block, feat) = {
            let df = self.dflash.as_ref().expect("armed");
            (df.block, df.state.as_ref().expect("armed").feat.clone())
        };
        let rows = k.min(block - 1) + 1;
        let k_use = rows - 1;
        if k_use == 0 {
            return Ok(None);
        }
        // ALL-or-nothing eligibility (the MTP begin's shape): one cold ring
        // sends the whole round to the MTP chain arm, whose warm seam
        // re-warms - partial dflash rounds keep the synchronous path.
        let mut reqs: Vec<(usize, usize, u32)> = Vec::with_capacity(n);
        for &(slot, tok) in pendings {
            let Some(&(_, e)) = feat.get(slot) else {
                return Ok(None);
            };
            let p = e as usize;
            if p + rows > self.max_ctx || !self.dflash_warm(slot, p) {
                return Ok(None);
            }
            reqs.push((slot, p, tok));
        }
        // Paged ring backing for the draft graph's own appends + the mirror
        // refresh before the replay (same rule as the synchronous round).
        if self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .is_some_and(|st| st.paged)
        {
            for &(slot, p, _) in &reqs {
                self.ensure_slot_blocks(slot, p + rows)?;
            }
            self.dflash_stage_tables()?;
        }
        // Chain plumbing: the armed verify indexes d_draft/d_asm_meta by the
        // CURRENT live count and round slots.
        self.spec_set_live(n);
        let slots: Vec<u32> = pendings.iter().map(|&(s, _)| s as u32).collect();
        self.spec_batch.as_mut().expect("spec batch").round_slots[..n].copy_from_slice(&slots);
        if !self.dflash_draft_launch(&reqs, rows)? {
            return Ok(None);
        }
        {
            let exec = self.exec.clone();
            let st = self
                .dflash
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .expect("armed");
            let sb = self.spec_batch.as_mut().expect("spec batch");
            exec.dflash_chain_picks(&st.d_out, &mut sb.d_draft, n, rows, k_use)?;
        }
        self.spec_round_dflash = true;
        self.spec_chain = Some((slots, k_use));
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!("[dflash-chain] ARMED n={n} rows={rows} k_use={k_use}");
        }
        Ok(Some((k_use, vec![true; n])))
    }

    pub fn spec_draft_begin_mtp(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<(usize, Vec<bool>)>, GpuModelError> {
        // Hybrid routing at/below the threshold: try the ASYNC block round
        // first (- same begin/fetch contract as the MTP chain, no
        // draft readback in the round); a decline (cold ring, old pack)
        // falls to the synchronous dflash path via the batch hook, exactly
        // as before this arm existed.
        if self.dflash_attached() && pendings.len() <= Self::dflash_live_max() {
            if let Some(armed) = self.dflash_draft_begin(pendings, k)? {
                return Ok(Some(armed));
            }
            return Ok(None);
        }
        self.spec_round_dflash = false;
        let k = k.min(self.serve_spec_k_mtp());
        if !self.serve_spec_on() || pendings.is_empty() || k == 0 || !self.exec.has_spec_toks() {
            return Ok(None);
        }
        self.ensure_serve_spec()?;
        let n = pendings.len();
        {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            if n > sb.alloc_batch {
                return Ok(None);
            }
            for &(slot, _) in pendings {
                if slot >= sb.alloc_batch || !sb.mtp_warm[slot] {
                    return Ok(None);
                }
            }
        }
        let k_use = k.min(self.spec_batch.as_ref().expect("spec batch").n_draft);
        self.spec_set_live(n);
        let slots: Vec<u32> = pendings.iter().map(|&(s, _)| s as u32).collect();
        self.spec_batch.as_mut().expect("spec batch").round_slots[..n].copy_from_slice(&slots);
        // block the drafter stripe's pool writes before the graph launches
        {
            let sbk = self.spec_batch.as_ref().expect("spec batch");
            let k1e = sbk.n_draft + 1;
            let sp: Vec<(usize, usize)> = pendings.iter().map(|&(s, _)| (s, sbk.pos[s])).collect();
            for (s, p) in sp {
                self.ensure_slot_blocks(s, p + k1e)?;
            }
        }
        self.stage_spec_tables()?;
        let last: Vec<u32> = pendings.iter().map(|&(_, t)| t).collect();
        self.mtp_draft_launch(&last)?;
        self.spec_chain = Some((slots, k_use));
        Ok(Some((k_use, vec![true; n])))
    }

    /// Read the armed chain's draft plane without disarming (drivers call
    /// this post-verify - the picks readback already synced the stream, so
    /// this is a cheap device->host copy of n_draft×n u32).
    fn spec_chain_peek(&self) -> Result<Vec<Vec<u32>>, GpuModelError> {
        let (slots, k_use) = self.spec_chain.as_ref().expect("armed chain");
        let n = slots.len();
        let sb = self.spec_batch.as_ref().expect("spec batch");
        let flat = self.exec.to_host_u32(&sb.d_draft)?;
        Ok((0..n)
            .map(|b| (0..*k_use).map(|i| flat[i * n + b]).collect())
            .collect())
    }

    /// Async draft chain, phase 2: the armed chain's drafts in pendings
    /// order; disarms. `Ok(None)` when nothing is armed.
    pub fn spec_draft_fetch_mtp(&mut self) -> Result<Option<Vec<Vec<u32>>>, GpuModelError> {
        if self.spec_chain.is_none() {
            return Ok(None);
        }
        let out = self.spec_chain_peek()?;
        self.spec_chain = None;
        Ok(Some(out))
    }

    /// The plans round's eligibility - the single source of truth, shared by
    /// the driver and the mixed tick's pre-check (whose decline contract is
    /// "chunk untouched", so the span must not launch for a round that would
    /// decline). Mutates exactly as the in-driver check always did: a
    /// pos-desync cools the slot. Ragged slot sets accepted; alloc coverage
    /// replaces the old contiguity requirement (slot is the GLOBAL id and
    /// may exceed alloc_batch - those cannot hold drafter state).
    fn spec_round_precheck(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
        k1: usize,
        dbg: bool,
    ) -> bool {
        use crate::sampler::DevicePlan;
        let n = reqs.len();
        let dfl = self.spec_round_dflash && self.dflash_armed();
        for (i, (slot, start, chunk)) in reqs.iter().enumerate() {
            // DFlash warmth is ring coverage, not the MTP chain: the drafter
            // is warm iff its feature ring ends exactly at `start`. On pass,
            // sync sb.pos to the service's cursor - the verify machinery
            // (pos_rows, ensure_slot_blocks, the commit advance) reads it.
            if dfl {
                let sb = self.spec_batch.as_ref().expect("spec batch");
                if *slot >= sb.alloc_batch
                    || chunk.is_empty()
                    || chunk.len() > k1
                    || *start + k1 > self.max_ctx
                    || !self.dflash_warm(*slot, *start)
                {
                    if dbg {
                        tracing::info!(
                            "[spec-plans] DECLINE(dflash) n={n} row{i} slot={slot}                              start={start} chunk={} warm={}",
                            chunk.len(),
                            self.dflash_warm(*slot, *start)
                        );
                    }
                    return false;
                }
                self.spec_batch.as_mut().expect("spec batch").pos[*slot] = *start;
                continue;
            }
            let sb = self.spec_batch.as_ref().expect("spec batch");
            if *slot >= sb.alloc_batch
                || chunk.is_empty()
                || chunk.len() > k1
                || *start + k1 > self.max_ctx
                || !sb.mtp_warm[*slot]
            {
                if dbg {
                    tracing::info!(
                        "[spec-plans] DECLINE n={n} row{i} slot={slot} over_alloc={} \
                         chunk={} k1={k1} start={start} warm={} maxctx={}",
                        *slot >= sb.alloc_batch,
                        chunk.len(),
                        sb.mtp_warm.get(*slot).copied().unwrap_or(false),
                        self.max_ctx
                    );
                }
                return false;
            }
            if sb.pos[*slot] != *start {
                if dbg {
                    tracing::info!(
                        "[spec-plans] DECLINE n={n} row{i} slot={slot} POS-DESYNC \
                         sb.pos={} != start={start}",
                        sb.pos[*slot]
                    );
                }
                self.spec_batch.as_mut().expect("spec batch").mtp_warm[*slot] = false;
                return false;
            }
        }
        // truncation stage (b): TruncCat verify rows pack mode 5 + the tpar side
        // plane and sample fully on device (pd_sample_rows_t on the chunk
        // buffers) - the service admits them to this round only when
        // `supports_device_trunc` holds. Belt-and-braces: without slot 435
        // a TruncCat row's pick would be stale garbage, so decline the
        // round outright rather than mis-accept.
        let dev_full = self.exec.has_sample_rows_t();
        if !dev_full
            && plans
                .iter()
                .any(|p| matches!(p, DevicePlan::TruncCat { .. }))
        {
            return false;
        }
        true
    }

    /// Mixed spec tick: the prefill span and the spec round in
    /// one stream-ordered tick - the span's eager layer walk is ENQUEUED
    /// (unified_span_launch, event-marked, no host wait), the round's
    /// draft/verify/commit graphs queue up BEHIND it on the same stream, and
    /// the round's picks readback is the single host boundary for both.
    /// Slots are disjoint by construction (a chunking slot is never a decode
    /// row), the round's conv/recurrence touch only its slots, and every
    /// buffer hand-off is stream-ordered (the round's staging htod's queue
    /// after the span's kernels; forward_chunk_b's ensure_scratch drains
    /// first in the rare realloc case). Before this existed the service's
    /// mixed-spec block hit the trait default and fell to the PLAIN mixed
    /// tick - decode rows rode chunk-in-flight ticks UNSPECULATED (~half of
    /// width decode under continuous admissions), and a "spec-first
    /// arbitration" A/B unknowingly measured exactly that silent fallback
    /// rather than two-forward drafting.
    pub fn forward_mixed_spec_plans_mtp(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        budget: usize,
        plans: &[crate::sampler::DevicePlan],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            Option<Vec<u32>>,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        // v2 OVERLAPPED: the round runs on the DECODE LANE
        // against the decode arena while the span runs on MAIN against the
        // shared scratch - v1 serialized round -> span and the width cells
        // (c16/c32/imax) paid the full round latency on every mixed tick.
        // Safety inventory: slots are disjoint by construction (a chunking
        // slot is never a decode row); the round's scratch is the arena
        // (with_spec_arena), its slot map is d_round_slots, its paged reads
        // go through the d_spec_tables mirror; the span's table-growth
        // uploads touch only the live table. Two event fences: ev_pre orders
        // [pre-tick draft graph + the hoisted pool growth below] before the
        // lane's round, recorded before the span enqueues so the lane never
        // waits on span work; ev_round orders the round's un-synced tail
        // (commit/catchup kernels) before main's finish hooks touch drafter
        // state.
        //
        // DFlash drafter armed: fall back to the v1 sequential order (round
        // then span). Both the round's verify taps and the span's own taps
        // write the drafter's one fusion accumulator (zacc) - overlapped
        // they would race it across lanes and cross-fuse each other's rows.
        // The overlap bought ~2% on the MTP width cells; correctness first.
        if self.spec_round_dflash && self.dflash_armed() {
            let picks = self.forward_spec_batch_plans_mtp(reqs, plans)?;
            if picks.is_none() {
                return Ok((None, Vec::new()));
            }
            let finished = if self.unified_span_launch(budget, fin_plans)? {
                self.unified_span_finish()?
            } else {
                Vec::new()
            };
            return Ok((picks, finished));
        }
        // Eligibility runs here first (same code the driver runs -
        // spec_round_precheck is the single source of truth) because the
        // service's decline contract is "chunk untouched": once the span
        // launches, a round decline would double-advance the chunk through
        // the service's plain-mixed fallback.
        if !self.serve_spec_on() || reqs.is_empty() {
            return Ok((None, Vec::new()));
        }
        self.ensure_serve_spec()?;
        let k1 = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            if reqs.len() > sb.alloc_batch {
                return Ok((None, Vec::new()));
            }
            sb.n_draft + 1
        };
        let dbg = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some();
        if !self.spec_round_precheck(reqs, plans, k1, dbg) {
            return Ok((None, Vec::new()));
        }
        // Hoist the round's pool growth ahead of the span: the verify's own
        // ensure_slot_blocks would otherwise upload the live table from the
        // LANE while the span's kernels read it on MAIN. After this loop the
        // in-round ensures are no-ops (commit/catchup never write past
        // pos + k1).
        for (slot, _, _) in reqs {
            let p = self.spec_batch.as_ref().expect("spec batch").pos[*slot];
            self.ensure_slot_blocks(*slot, p + k1)?;
        }
        let ev_pre = self.exec.record_event()?;
        let launched = self.unified_span_launch(budget, fin_plans)?;
        let round = self.with_decode_lane(|m| -> Result<_, GpuModelError> {
            m.exec.wait_event(&ev_pre)?;
            let picks = m.forward_spec_batch_plans_mtp(reqs, plans)?;
            let ev_round = m.exec.record_event()?;
            Ok((picks, ev_round))
        });
        let (picks, ev_round) = round?;
        self.exec.wait_event(&ev_round)?;
        let finished = if launched {
            self.unified_span_finish()?
        } else {
            Vec::new()
        };
        let Some(picks) = picks else {
            // unreachable by construction: the driver re-runs the same
            // precheck on the same state. If it ever fires the span has
            // advanced under a declined round - fail loud, never let the
            // service's plain-mixed fallback double-advance the chunk.
            tracing::error!(
                "spec round declined after passing precheck - mixed tick invariant broken"
            );
            debug_assert!(false, "spec round declined after precheck");
            return Err(GpuError::Driver(
                "spec round declined after precheck in the overlapped mixed tick".into(),
            )
            .into());
        };
        Ok((Some(picks), finished))
    }

    /// Armed-round bookkeeping, pre-verify: per-block real chunk lengths for
    /// the device token assembly (see forward_chunk_b).
    fn spec_chain_stage_lens(&mut self, reqs: &[(usize, usize, Vec<u32>)]) {
        if self.spec_chain.is_none() {
            return;
        }
        let sb = self.spec_batch.as_mut().expect("spec batch");
        for (i, r) in reqs.iter().enumerate() {
            sb.chain_lens[i] = r.2.len() as u32;
        }
    }

    /// Armed-round bookkeeping, post-verify: overwrite the placeholder draft
    /// values in `padded` with the real chain ids (peeked - the picks
    /// readback already synced), reproducing pd_spec_toks' pad rule exactly,
    /// so the host accept walk and the commit/catchup token records see what
    /// the verify actually saw.
    fn spec_chain_rebuild(
        &self,
        padded: &mut [u32],
        reqs: &[(usize, usize, Vec<u32>)],
        k1: usize,
    ) -> Result<(), GpuModelError> {
        let Some((chain_slots, k_use)) = self.spec_chain.as_ref() else {
            return Ok(());
        };
        let drafts = self.spec_chain_peek()?;
        for (i, (slot, _, chunk)) in reqs.iter().enumerate() {
            let Some(ci) = chain_slots.iter().position(|&s| s == *slot as u32) else {
                continue; // chain-cold block: its length-1 chunk is already real
            };
            let nd = (chunk.len().saturating_sub(1)).min(*k_use);
            for j in 0..nd {
                padded[i * k1 + 1 + j] = drafts[ci][j];
            }
            let lastv = if nd > 0 {
                drafts[ci][nd - 1]
            } else {
                padded[i * k1]
            };
            for j in (1 + nd)..k1 {
                padded[i * k1 + j] = lastv;
            }
        }
        Ok(())
    }

    /// Serving spec round: verify the service's chunks, commit accepted rows
    /// into the backbone + MTP state, return per-row greedy picks in the
    /// service's flat layout. Declines (None) when a slot is cold or its
    /// position desynced - the service cools down and retries later.
    pub fn forward_spec_batch_mtp(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        if !self.serve_spec_on() || reqs.is_empty() {
            return Ok(None);
        }
        self.ensure_serve_spec()?;
        let n = reqs.len();
        let k1 = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            if n > sb.alloc_batch {
                return Ok(None);
            }
            sb.n_draft + 1
        };
        // eligibility validates against the ALLOC bound; the round itself
        // runs RAGGED at this round's own k1 (shadowed after eligibility)
        let mut eligible = true;
        let dfl = self.spec_round_dflash && self.dflash_armed();
        // ragged slot sets accepted - see spec_draft_batch_mtp
        for (slot, start, chunk) in reqs.iter() {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            if *slot >= sb.alloc_batch
                || chunk.is_empty()
                || chunk.len() > k1
                || *start + k1 > self.max_ctx
                || (!dfl && !sb.mtp_warm[*slot])
                || (dfl && !self.dflash_warm(*slot, *start))
            {
                eligible = false;
                break;
            }
            if dfl {
                // ring end == start proven above; sync the verify cursor
                // (pos_rows / ensure_slot_blocks / the commit advance read it)
                self.spec_batch.as_mut().expect("spec batch").pos[*slot] = *start;
                continue;
            }
            if sb.pos[*slot] != *start {
                // dense ticks advanced this slot without MTP catchup: its
                // draft KV is stale for good - cold until the next prefill
                self.spec_batch.as_mut().expect("spec batch").mtp_warm[*slot] = false;
                eligible = false;
                break;
            }
        }
        if !eligible {
            return Ok(None);
        }
        // RAGGED k: this round's k1 = its widest chunk (min 2 so the
        // kernels keep their >=1-draft shapes; <= alloc, checked above)
        let k1 = reqs.iter().map(|r| r.2.len()).max().unwrap_or(2).max(2);
        self.spec_batch.as_mut().expect("spec batch").round_k1 = k1;
        self.spec_set_live(n);
        self.spec_batch.as_mut().expect("spec batch").round_slots[..n]
            .copy_from_slice(&reqs.iter().map(|r| r.0 as u32).collect::<Vec<_>>());
        // block-major rows padded to k1 (repeat the last token; acceptance is
        // capped at the real chunk length so pad rows never commit)
        let mut padded: Vec<u32> = Vec::with_capacity(n * k1);
        for (_, _, chunk) in reqs {
            padded.extend_from_slice(chunk);
            let t = *chunk.last().expect("non-empty chunk checked above");
            padded.extend(std::iter::repeat_n(t, k1 - chunk.len()));
        }
        // per-BLOCK positions (block i = reqs[i], true slot reqs[i].0)
        let pos_before: Vec<usize> = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            reqs.iter().map(|r| sb.pos[r.0]).collect()
        };
        if paddock_models::dev_var_os!("PADDOCK_SPEC_TRACE").is_some() {
            // pre-verify bracket (contamination hunt): what state does this
            // round START from, and where does d_slots route it?
            if let Some(li) = self
                .layers
                .iter()
                .position(|l| matches!(l.mixer, Mixer::Linear(_)))
            {
                let se = self.n_v_heads * self.state_size * self.state_size;
                if let Some(bs) = self.batch.as_ref() {
                    let slots_host = self.exec.to_host_u32(&bs.d_slots).unwrap_or_default();
                    let fp = |o: usize| -> f64 {
                        bs.recur[li]
                            .as_ref()
                            .and_then(|b| b.try_slice(o..o + 256))
                            .and_then(|v| self.exec.stream.clone_dtoh(&v).ok())
                            .map(|h| h.iter().map(|&x| x as f64).sum())
                            .unwrap_or(f64::NAN)
                    };
                    tracing::info!(
                        "TRACE pre-verify d_slots {:?} fp_slot0 {:.6} fp_slot1 {:.6}",
                        &slots_host[..slots_host.len().min(4)],
                        fp(0),
                        fp(se),
                    );
                }
            }
        }
        let dbg2 = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some();
        let tv = std::time::Instant::now();
        self.spec_chain_stage_lens(reqs);
        self.forward_chunk_b(&padded)?;
        let picks_all = self
            .exec
            .to_host_u32(&self.spec_batch.as_ref().expect("spec batch").d_row_tok)?;
        self.spec_chain_rebuild(&mut padded, reqs, k1)?;
        let t_verify = tv.elapsed().as_micros();
        // acceptance identical to the service rule, over the real chunk only
        // (walk `padded` - identical values to reqs on sync rounds, the real
        // chain ids on armed async rounds)
        let mut committed: Vec<u32> = Vec::with_capacity(n);
        let mut picks_out: Vec<u32> = Vec::new();
        for (i, (_, _, chunk)) in reqs.iter().enumerate() {
            let base = i * k1;
            let mut a = 0usize;
            while a + 1 < chunk.len() && padded[base + a + 1] == picks_all[base + a] {
                a += 1;
            }
            committed.push((a + 1) as u32);
            picks_out.extend_from_slice(&picks_all[base..base + chunk.len()]);
        }
        if paddock_models::dev_var_os!("PADDOCK_SPEC_TRACE").is_some() {
            // token-level round trace (from the contamination hunt): the
            // serving round only logged counts, which cannot say WHOSE tokens
            // a bad round carried
            for (i, (slot, start, chunk)) in reqs.iter().enumerate() {
                let base = i * k1;
                tracing::info!(
                    "TRACE serve slot {slot} start {start}: chunk {:?} picks {:?} pos_before {}",
                    chunk,
                    &picks_all[base..base + chunk.len()],
                    pos_before[i],
                );
            }
            // slot-0 recurrent-state fingerprint of the first DeltaNet layer,
            // read from the batch state the verify consumed this round. If a
            // new request's round 1 logs the same sum as the previous
            // request's last round, the serving prefill never rewrote
            // bs.recur and the verify is decoding someone else's state.
            if let Some(li) = self
                .layers
                .iter()
                .position(|l| matches!(l.mixer, Mixer::Linear(_)))
            {
                let n = 256usize;
                if let Some(buf) = self.batch.as_ref().and_then(|bs| bs.recur[li].as_ref())
                    && let Some(view) = buf.try_slice(0..n)
                    && let Ok(host) = self.exec.stream.clone_dtoh(&view)
                {
                    let sum: f64 = host.iter().map(|&x| x as f64).sum();
                    tracing::info!("TRACE state fp layer {li} slot0: {sum:.6}");
                }
            }
        }
        let tc = std::time::Instant::now();
        self.commit_chunk_b(&padded, &committed)?;
        let t_commit = tc.elapsed().as_micros();
        let tk = std::time::Instant::now();
        if self.spec_round_dflash && self.dflash_armed() {
            // drafter state is the feature ring: append the ACCEPTED rows'
            // fused features (tapped during the verify walk) instead of the
            // MTP catchup pass
            self.dflash_spec_commit(reqs, &padded, &committed, k1)?;
        } else {
            self.mtp_catchup_b(&padded, &committed, &pos_before)?;
            // MTP round with the block drafter attached: the ring still needs
            // the committed rows' features (the verify tapped them regardless
            // of who drafted) - without this append the ring freezes at the
            // first MTP round and the hybrid can never hand back to the block
            // drafter (measured: ring stuck at (176,180) while decode ran to
            // 304, forcing MTP for the rest of the session).
            if self.dflash_armed() {
                self.dflash_spec_commit(reqs, &padded, &committed, k1)?;
            }
        }
        if dbg2 {
            tracing::info!(
                "[spec-round-t] verify+readback={t_verify}us commit={t_commit}us catchup={}us",
                tk.elapsed().as_micros()
            );
        }
        Ok(Some(picks_out))
    }

    /// Device-sampled spec round: verify, then `sample_rows` over the padded
    /// rows with the service's pre-drawn plans (pad rows get Greedy - their
    /// picks are never read), acceptance + commit + catchup internally.
    /// Identical structure to the greedy round; only the pick source differs.
    /// No logits readback - the sampled path that scales to the full row
    /// budget (c8-class). Samples into the spec-owned d_samp_*_chunk (sized for
    /// the full n*(K+1) verify), so unlike the greedy round it has no max_batch
    /// row ceiling. PADDOCK_SPEC_DEBUG=1 logs per-round engagement/decline.
    pub fn forward_spec_batch_plans_mtp(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        use crate::sampler::DevicePlan;
        if !self.serve_spec_on() || reqs.is_empty() {
            return Ok(None);
        }
        self.ensure_serve_spec()?;
        let n = reqs.len();
        let (k1, vocab) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            if n > sb.alloc_batch {
                return Ok(None);
            }
            (sb.n_draft + 1, self.vocab)
        };
        // n*k1 verify rows fit the spec-owned d_samp_*_chunk (sized R_max =
        // max(alloc_batch*k1, WARM_CHUNK)); n<=alloc_batch was checked above so
        // n*k1 <= alloc_batch*k1 <= R_max always - no max_batch ceiling here.
        assert_eq!(plans.len(), reqs.iter().map(|r| r.2.len()).sum::<usize>());
        let dbg = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some();
        if !self.spec_round_precheck(reqs, plans, k1, dbg) {
            return Ok(None);
        }
        // RAGGED k: this round's k1 = its widest chunk (min 2 so the
        // kernels keep their >=1-draft shapes; <= alloc, checked above)
        let k1 = reqs.iter().map(|r| r.2.len()).max().unwrap_or(2).max(2);
        self.spec_batch.as_mut().expect("spec batch").round_k1 = k1;
        self.spec_set_live(n);
        self.spec_batch.as_mut().expect("spec batch").round_slots[..n]
            .copy_from_slice(&reqs.iter().map(|r| r.0 as u32).collect::<Vec<_>>());
        let mut padded: Vec<u32> = Vec::with_capacity(n * k1);
        // per padded row: sampler params for sample_rows (mode 1 greedy pads)
        let mut par = vec![0u32; n * k1 * 4];
        let mut tpar = vec![0u32; n * k1 * 4];
        let mut any_trunc = false;
        // Rung G: drafted rows of nucleus-sampling slots resolve under the
        // K-candidate rejection sampler (mode 7) only when this round's
        // drafts came from the sampled selector walk (spec_round_rs) - any
        // other draft shape leaves no valid q, so those plans serve as the
        // classic mode-5 rule with u1, which is lossless with any draft.
        let rs_round = self.spec_round_dflash && self.spec_round_rs;
        let mut any_rs = false;
        let mut flat = 0usize;
        for (i, (_, _, chunk)) in reqs.iter().enumerate() {
            for j in 0..k1 {
                let row = i * k1 + j;
                if j < chunk.len() {
                    match plans[flat + j] {
                        DevicePlan::Greedy => par[row * 4 + 2] = 1,
                        DevicePlan::Categorical { inv_t, u } => {
                            par[row * 4] = inv_t.to_bits();
                            par[row * 4 + 1] = u.to_bits();
                            par[row * 4 + 2] = 2;
                        }
                        // P67 mode 5: zero-host truncation draw on the
                        // verify row, same plane layout as the batched tick
                        DevicePlan::TruncCat {
                            inv_t,
                            u,
                            k,
                            top_p,
                            min_p,
                        } => {
                            par[row * 4] = inv_t.to_bits();
                            par[row * 4 + 1] = u.to_bits();
                            par[row * 4 + 2] = if (1..=64).contains(&k) { 5 } else { 6 };
                            tpar[row * 4] = k;
                            tpar[row * 4 + 1] = top_p.to_bits();
                            tpar[row * 4 + 2] = min_p.to_bits();
                            any_trunc = true;
                        }
                        // the full-softmax RS plan is gemma4's; here it is
                        // an untruncated temperature>0 slot - the classic
                        // categorical rule with its first uniform
                        DevicePlan::RsVerify { inv_t, u1, .. } => {
                            par[row * 4] = inv_t.to_bits();
                            par[row * 4 + 1] = u1.to_bits();
                            par[row * 4 + 2] = 2;
                        }
                        DevicePlan::RsTrunc {
                            inv_t,
                            u1,
                            u2,
                            k,
                            top_p,
                            min_p,
                        } => {
                            par[row * 4] = inv_t.to_bits();
                            par[row * 4 + 1] = u1.to_bits();
                            tpar[row * 4] = k;
                            tpar[row * 4 + 1] = top_p.to_bits();
                            tpar[row * 4 + 2] = min_p.to_bits();
                            // a drafted row by the service's contract, but
                            // the resolve only has a draft to judge while
                            // j+1 is inside the (ragged) chunk
                            if rs_round && j + 1 < chunk.len() && (1..=64).contains(&k) {
                                par[row * 4 + 2] = 7;
                                par[row * 4 + 3] = u2.to_bits();
                                any_rs = true;
                            } else {
                                par[row * 4 + 2] = if (1..=64).contains(&k) { 5 } else { 6 };
                                any_trunc = true;
                            }
                        }
                    }
                } else {
                    par[row * 4 + 2] = 1; // pad row: greedy, pick unread
                }
            }
            flat += chunk.len();
            padded.extend_from_slice(chunk);
            let t = *chunk.last().expect("non-empty chunk checked above");
            padded.extend(std::iter::repeat_n(t, k1 - chunk.len()));
        }
        // per-BLOCK positions (block i = reqs[i], true slot reqs[i].0)
        let pos_before: Vec<usize> = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            reqs.iter().map(|r| sb.pos[r.0]).collect()
        };
        self.spec_chain_stage_lens(reqs);
        self.forward_chunk_b(&padded)?;
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        {
            let sb = self.spec_batch.as_mut().expect("spec batch");
            let mut v = sb.d_samp_par_chunk.slice_mut(0..n * k1 * 4);
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if any_trunc || any_rs {
                let mut v = sb.d_samp_tpar_chunk.slice_mut(0..n * k1 * 4);
                exec.stream.memcpy_htod(&tpar, &mut v).map_err(drv)?;
            }
        }
        {
            let sb = self.spec_batch.as_mut().expect("spec batch");
            let (logits, par_buf, out_buf) = (
                &sb.d_logits_chunk,
                &sb.d_samp_par_chunk,
                &mut sb.d_samp_out_chunk,
            );
            exec.sample_rows(logits, par_buf, out_buf, n * k1, vocab)?;
            if any_trunc {
                // engagement witness (bisect-trap law): once per process
                static SPEC5: std::sync::Once = std::sync::Once::new();
                SPEC5.call_once(|| {
                    eprintln!(
                        "[trunc-spec] engaged: rows={} (mode-5 verify rows in the spec round)",
                        n * k1
                    );
                });
                exec.sample_rows_t(
                    logits,
                    par_buf,
                    &sb.d_samp_tpar_chunk,
                    out_buf,
                    n * k1,
                    vocab,
                )?;
                exec.sample_rows_p(
                    logits,
                    par_buf,
                    &sb.d_samp_tpar_chunk,
                    out_buf,
                    n * k1,
                    vocab,
                )?;
            }
        }
        if any_rs {
            // Rung G: the mode-7 rows - accept the draft at row j+1 with
            // probability min(1, p/q) against the mode-5 nucleus (the same
            // distribution the mode-5 rows above draw from), residual pick on
            // reject; lands in the sampled-ids plane so the accept walk below
            // is unchanged. q/cand are the drafter's planes of this round
            // (spec_round_rs), indexed through the chain meta's block->chain
            // row map and the drafter's rows per block.
            static RS7: std::sync::Once = std::sync::Once::new();
            RS7.call_once(|| {
                eprintln!("[dflash-rs] engaged: rejection-sampling verify rows in the spec round (k1={k1})");
            });
            let drows = self.spec_chain.as_ref().map_or(k1, |(_, ku)| ku + 1);
            let top_k = self
                .dflash
                .as_ref()
                .and_then(|d| d.selector.as_ref().map(|(_, _, k)| *k))
                .expect("rs round without a selector");
            let st = self
                .dflash
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .expect("armed");
            let sb = self.spec_batch.as_mut().expect("spec batch");
            exec.dflash_rs_resolve(
                &sb.d_logits_chunk,
                &sb.d_samp_par_chunk,
                &sb.d_samp_tpar_chunk,
                &sb.d_asm_meta,
                &sb.d_mtp_tok,
                &st.sel_ids,
                &st.q16,
                &mut sb.d_samp_out_chunk,
                n * k1,
                n,
                k1,
                drows,
                top_k,
                vocab,
            )?;
        }
        let picks_all = exec.to_host_u32(
            &self
                .spec_batch
                .as_ref()
                .expect("spec batch")
                .d_samp_out_chunk,
        )?;
        self.spec_chain_rebuild(&mut padded, reqs, k1)?;
        let mut committed: Vec<u32> = Vec::with_capacity(n);
        let mut picks_out: Vec<u32> = Vec::new();
        // accept walks `padded` - identical to reqs on sync rounds, the real
        // chain ids on armed async rounds (placeholders never accept-match)
        for (i, (_, _, chunk)) in reqs.iter().enumerate() {
            let base = i * k1;
            let mut a = 0usize;
            while a + 1 < chunk.len() && padded[base + a + 1] == picks_all[base + a] {
                a += 1;
            }
            committed.push((a + 1) as u32);
            picks_out.extend_from_slice(&picks_all[base..base + chunk.len()]);
        }
        self.commit_chunk_b(&padded, &committed)?;
        if self.spec_round_dflash && self.dflash_armed() {
            // drafter state is the feature ring: append the ACCEPTED rows'
            // fused features (tapped during the verify walk) instead of the
            // MTP catchup pass
            self.dflash_spec_commit(reqs, &padded, &committed, k1)?;
        } else {
            self.mtp_catchup_b(&padded, &committed, &pos_before)?;
            // MTP round with the block drafter attached: the ring still needs
            // the committed rows' features (the verify tapped them regardless
            // of who drafted) - without this append the ring freezes at the
            // first MTP round and the hybrid can never hand back to the block
            // drafter (measured: ring stuck at (176,180) while decode ran to
            // 304, forcing MTP for the rest of the session).
            if self.dflash_armed() {
                self.dflash_spec_commit(reqs, &padded, &committed, k1)?;
            }
        }
        if dbg {
            let acc: u32 = committed.iter().sum();
            let c0 = &reqs[0].2;
            let p0 = &picks_all[..c0.len().min(picks_all.len())];
            tracing::info!(
                "[spec-plans] ROUND n={n} k1={k1} committed={committed:?} \
                 accepted_tok={acc} (mean {:.2}/slot) row0 chunk={c0:?} picks={p0:?}",
                acc as f64 / n as f64
            );
        }
        Ok(Some(picks_out))
    }

    /// Sampled-spec phase 1: same eligibility + verify as the greedy round,
    /// but returns the RAW row logits (request order, real chunk rows only)
    /// and stashes the round for `spec_commit_mtp`. The service samples rows
    /// with each slot's own sampler - exact rejection sampling for our
    /// deterministic MTP drafts (see sampler::is_spec_safe).
    pub fn forward_spec_verify_mtp(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<f32>>, GpuModelError> {
        if !self.serve_spec_on() || reqs.is_empty() {
            return Ok(None);
        }
        self.ensure_serve_spec()?;
        assert!(self.spec_pending.is_none(), "unclosed sampled spec round");
        let n = reqs.len();
        let (k1, vocab) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            if n > sb.alloc_batch {
                return Ok(None);
            }
            (sb.n_draft + 1, self.vocab)
        };
        let mut eligible = true;
        // ragged slot sets accepted - see spec_draft_batch_mtp
        for (slot, start, chunk) in reqs.iter() {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            if *slot >= sb.alloc_batch
                || chunk.is_empty()
                || chunk.len() > k1
                || *start + k1 > self.max_ctx
                || !sb.mtp_warm[*slot]
            {
                eligible = false;
                break;
            }
            if sb.pos[*slot] != *start {
                self.spec_batch.as_mut().expect("spec batch").mtp_warm[*slot] = false;
                eligible = false;
                break;
            }
        }
        if !eligible {
            return Ok(None);
        }
        // RAGGED k: this round's k1 = its widest chunk (min 2 so the
        // kernels keep their >=1-draft shapes; <= alloc, checked above)
        let k1 = reqs.iter().map(|r| r.2.len()).max().unwrap_or(2).max(2);
        self.spec_batch.as_mut().expect("spec batch").round_k1 = k1;
        self.spec_set_live(n);
        let slots: Vec<u32> = reqs.iter().map(|r| r.0 as u32).collect();
        self.spec_batch.as_mut().expect("spec batch").round_slots[..n].copy_from_slice(&slots);
        let mut padded: Vec<u32> = Vec::with_capacity(n * k1);
        for (_, _, chunk) in reqs {
            padded.extend_from_slice(chunk);
            let t = *chunk.last().expect("non-empty chunk checked above");
            padded.extend(std::iter::repeat_n(t, k1 - chunk.len()));
        }
        // per-BLOCK positions (block i = reqs[i], true slot reqs[i].0)
        let pos_before: Vec<usize> = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            reqs.iter().map(|r| sb.pos[r.0]).collect()
        };
        self.spec_chain_stage_lens(reqs);
        self.forward_chunk_b(&padded)?;
        let all = self.exec.to_host_len(
            &self.spec_batch.as_ref().expect("spec batch").d_logits_chunk,
            n * k1 * vocab,
        )?;
        self.spec_chain_rebuild(&mut padded, reqs, k1)?;
        let mut out: Vec<f32> =
            Vec::with_capacity(reqs.iter().map(|r| r.2.len()).sum::<usize>() * vocab);
        for (i, (_, _, chunk)) in reqs.iter().enumerate() {
            let base = i * k1 * vocab;
            out.extend_from_slice(&all[base..base + chunk.len() * vocab]);
        }
        self.spec_pending = Some((padded, pos_before, slots));
        Ok(Some(out))
    }

    /// Sampled-spec phase 2: commit the service's accepted row counts for the
    /// stashed round (state rollback + MTP catchup + pending_h advance).
    pub fn spec_commit_mtp(&mut self, committed: &[u32]) -> Result<(), GpuModelError> {
        let (padded, pos_before, slots) = self
            .spec_pending
            .take()
            .expect("no open sampled spec round");
        assert_eq!(committed.len(), pos_before.len());
        // restore the stashed round's block->slot map (host mirror + the
        // round's own device buffer the commit graph reads)
        {
            let exec = self.exec.clone();
            let sb = self.spec_batch.as_mut().expect("spec batch");
            sb.round_slots[..slots.len()].copy_from_slice(&slots);
            debug_assert_eq!(sb.batch, slots.len(), "live changed inside an open round");
            let mut v = sb.d_round_slots.slice_mut(0..slots.len());
            exec.stream
                .memcpy_htod(&slots, &mut v)
                .map_err(crate::gpu::from_driver)?;
        }
        self.commit_chunk_b(&padded, committed)?;
        self.mtp_catchup_b(&padded, committed, &pos_before)?;
        // Same ring-append rule as the one-phase drivers: an MTP round with
        // the block drafter attached must keep the feature ring current
        // (chunks aren't stashed, but dflash_spec_commit only reads slot,
        // start, and the k1 shape - rebuild that much).
        if self.dflash_armed() {
            let k1 = padded.len() / slots.len().max(1);
            let reqs: Vec<(usize, usize, Vec<u32>)> = slots
                .iter()
                .zip(&pos_before)
                .map(|(&s, &p)| (s as usize, p, Vec::new()))
                .collect();
            self.dflash_spec_commit(&reqs, &padded, committed, k1)?;
        }
        Ok(())
    }
}
