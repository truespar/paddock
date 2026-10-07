//! One pass of the layer stack over a tick's rows - every mixer arm (mamba,
//! attention, MoE) in every row class the batch lane runs: one-row decode,
//! a decode tick's band, a spec verify round's runs, prefill chunks and the
//! fused mixed tick. Split out of `batch.rs` (which had crossed the
//! 2,500-line ceiling); the scratch it walks and the elections it reads
//! stay there.

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;

use super::batch::*;
use super::*;
use crate::gpu_model::qwen35::{gemv_any, mmq_pre, prefill_quant};

impl GpuNemotron {
    /// The whole-stack walk over r rows. `cuts`: Some = prefill mode (append
    /// the whole chunk, attend + advance recurrent state per same-slot run);
    /// None = decode mode (every row is one new token of its slot, the
    /// stage-A batched step kernels advance the arenas by d_slots).
    ///
    /// Compute classes, keyed on MODE (granite's law - every prefill row
    /// takes the same rungs at any r, so a warm-resume tail reproduces the
    /// cold chunk's bytes): prefill rows ride the serial bulk-prefill lane's
    /// W8A8 f8row GEMM / gemm_f32 / sorted-tile bs MoE, all arbiter-gated by
    /// the bulk-prefill parity gate. A PURE r==1 decode tick rides the
    /// serial decode twins (f8r GEMV, bf16 GEMV, fused mt MoE) so the c1
    /// battery stays in the serial lane's numeric class; r>1 decode takes
    /// the batched GEMM class - attention projections in the twins' bf16
    /// class when the pack carries them (the serial GEMV twins' own class),
    /// f32 otherwise; prefill rows always the exact-f32 planes.
    /// `verify` (spec core): the mamba advance runs the spec
    /// verify's non-committing discipline - conv on the slot's SCRATCH
    /// window (the live window stays pre-round for the commit rebuild),
    /// scan on the live state with per-row snapshots, and the xBC rows
    /// snapshot so a partial accept can rebuild the window. Everything else
    /// (KV, attention, MoE, residuals) is byte-identical to the plain walk -
    /// KV needs no rollback (stale cells past the accept are overwritten
    /// before any later read). Requires `bs.verify` populated.
    pub(super) fn layer_walk(
        &mut self,
        r: usize,
        cuts: Option<&PfCuts>,
        verify: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let (embd, eps) = (hp.hidden, hp.eps);
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let q_dim = hp.n_heads * hp.head_dim;
        let (nh, n_kv, hd) = (hp.n_heads, hp.n_kv_heads, hp.head_dim);
        let d_inner = hp.d_inner();
        let conv_dim = hp.conv_dim();
        let in_rows = hp.in_proj_rows();
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * conv_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let kv_dtype = self.kv_dtype;
        let pf = cuts.is_some();
        // pure single-row decode: the serial lane's kernel twins
        let dec1 = r == 1 && !pf;
        // attention projections in the bf16 twins' class: decode rows, and
        // verify rows too - a verify walk carries cuts only for the mamba
        // runs' sequential advance, its rows are decode rows. The f32 prefill
        // class re-read twice the bytes every round (8 of a round's 42 ms at
        // depth 7 with the drafter's own f32 planes, GB10 2026-09-26).
        let proj16 = !pf || verify;
        // rows standing in for decode steps run the W16 decode class (see
        // w16_class) - one-row decode included
        // the class is the NVFP4 lane's (w16_lane): the GGUF lane keeps its own
        let w16_lane = self.w16_lane();
        let w16 = (!pf || verify) && w16_lane && w16_class(&exec, r);
        // KV splits for the batched decode attention - see
        // attn_split_election (the head-packed arm, both KV classes, one
        // live CTA per SM; the scratch is sized from the same function).
        let (hp16, ns) = attn_split_election(nh, n_kv, hd, exec.sm_count(), r);
        // f16 always rides the wmma tile; fp8 rides its raw-e4m3 hd128 arm
        // (pack group set {4,6,8,9,16} - nemotron is G=16). PADDOCK_NO_NPF8
        // pins fp8 back onto the scalar paged walk (granite's A/B precedent).
        let pf8_ok = match kv_dtype {
            KvDtype::Fp16 => true,
            KvDtype::Fp8E4m3 => {
                n_kv > 0
                    && nh == n_kv * 16
                    && paddock_models::dev_var_os!("PADDOCK_NO_NPF8").is_none()
            }
        };
        let wmma_pf = pf
            && hd == 128
            && pf8_ok
            && exec.has_attn_prefill_f16_paged()
            && paddock_models::dev_var_os!("PADDOCK_NO_WMMA_PREFILL").is_none();
        let bs = self.batch.as_mut().expect("batch enabled");
        let bps = bs.bps;
        // A mixed tick's decode band - rows [0, dec), one per decoding slot,
        // beside a prefill chunk - runs the same class as a decode tick's
        // rows, so a stream's bits do not depend on the prompt being
        // admitted beside it. The class's ops cannot share a launch with the
        // chunk's prefill lanes, so each takes the chunk's lane over all the
        // tick's rows and the W16 kernel then recomputes the band's rows over
        // it: the projections overwrite, attention takes the band's rows
        // alone, and the MoE holds the band's residual rows while the chunk's
        // fold runs over every row. The band's planes stream a second time -
        // about one decode tick per mixed tick. The mamba advance already
        // splits (the band steps, the chunk walks), and norms and the head
        // are per row. 0 = no band (or a pack without the class's segmented
        // q|k|v and routing front). PADDOCK_NO_NEMO_W16_BAND pins the band to
        // the chunk's lanes for the A/B.
        let band = match cuts {
            Some(c)
                if !verify
                    && w16_band_on()
                    && c.dec > 0
                    && c.dec <= bs.sc.w16_rows
                    && w16_lane
                    && w16_class(&exec, c.dec)
                    && exec.has_dense_w16_seg()
                    && exec.has_moe_route_w16() =>
            {
                c.dec
            }
            _ => 0,
        };
        // Verify rows attend through the multi-row split partial: every run's
        // context streams once per group of <= 8 rows and splits across the
        // die. The per-run prefill tile ran one CTA per (kv head, 4-token
        // tile) - an 8-row verify at 24K was 4 CTAs, 2.0 ms a layer (GB10
        // 2026-09-26). The groups upload once per walk. PADDOCK_NO_ATTN_ROWS
        // pins the prefill tile for the A/B.
        let rows_attn: Option<(usize, usize, usize)> = match cuts {
            Some(c)
                if verify
                    && hd == 128
                    && n_kv > 0
                    && nh % n_kv == 0
                    && nh / n_kv <= 16
                    && exec.has_attn_rows_partial()
                    && paddock_models::dev_var_os!("PADDOCK_NO_ATTN_ROWS").is_none() =>
            {
                // slot 745 takes six rows a group (two warps each)
                let cap = if bs.sc.w16_kh {
                    ROWS_GROUP_KH
                } else {
                    ROWS_GROUP
                };
                let (law, ns) = (bs.sc.w16_split, bs.sc.w16_ns);
                let groups = if law == 0 || law & 0xc000_0000 != 0 {
                    // a key-dependent law (744 pow2, 745 TILE): groups stay
                    // inside one split-size bucket
                    let runs = c.runs.iter().zip(&c.run_pos);
                    rows_groups_law(
                        runs.map(|(&(o, l, _), &p)| (o, l, p)),
                        |n| w16_split_size(law, ns, n),
                        cap,
                    )
                } else {
                    rows_groups_cap(c.runs.iter().map(|&(off, len, _)| (off, len)), cap)
                };
                let ng = groups.len() / 2;
                let gmax = groups.chunks(2).map(|g| g[1] as usize).max().unwrap_or(1);
                let vp = bs.verify.as_mut().expect("verify planes");
                let mut v = vp
                    .d_groups
                    .try_slice_mut(0..groups.len())
                    .ok_or_else(|| GpuError::Driver("verify groups".into()))?;
                exec.stream
                    .memcpy_htod(&groups, &mut v)
                    .map_err(crate::gpu::from_driver)?;
                Some((ng, rows_split(n_kv, ng, exec.sm_count()), gmax))
            }
            _ => None,
        };
        // running mamba-layer index: the checkpoint blob offset for a layer
        // is mi * (state + win), matching the pool/restore layout
        let mut mi = 0usize;

        // Glue rung: every MoE layer's prologue is add(x += proj)
        // + rmsnorm + quantize_nvf4, three latency-bound launches, and the
        // checkpoint's pattern makes that 23 of them per decode tick. One
        // fused row-per-CTA kernel does all three, so the previous layer's
        // trailing add is hoisted into it. Only the bs arm consumes the nvf4
        // planes, so dec1 (which takes the mt path) keeps the plain chain.
        // Bit-exact - PADDOCK_NO_GLUE_FUSE restores the three launches for the
        // A/B on one binary.
        let glue_fuse = !dec1
            && paddock_models::dev_var_os!("PADDOCK_NO_GLUE_FUSE").is_none()
            && exec.has_add_rmsnorm_quant_nvf4();
        let mut fused_pro = false;
        for (li, layer) in self.layers.iter().enumerate() {
            let pro_done = std::mem::take(&mut fused_pro);
            let sc = &mut bs.sc;
            // DFlash aux tap: d_x here is the post-block residual of layer
            // li-1 - copy it into the drafter's aux band when li-1 is a
            // target layer (the last target layer taps after the loop)
            if li > 0
                && let Some(df) = self.dflash.as_mut()
                && let Some(st) = df.state.as_mut()
                && let Some(ai) = df.target_layers.iter().position(|&t| t == li - 1)
            {
                exec.copy_region(&sc.d_x, 0, &mut st.aux[ai], 0, r * embd)?;
            }
            if !pro_done {
                exec.rmsnorm_batch(&sc.d_x, &layer.norm.buf, &mut sc.d_xn, embd, eps, r)?;
            }
            match &layer.mixer {
                Mixer::Mamba(w) => {
                    match &w.in_proj {
                        LinW::F8(p) => {
                            if w16 {
                                exec.dense_w16_e4m3(
                                    p,
                                    embd,
                                    in_rows,
                                    &sc.d_xn,
                                    &mut sc.d_x16h,
                                    &mut sc.d_zxbcdt,
                                    in_rows,
                                    r,
                                )?;
                            } else if dec1 {
                                exec.f8r_gemv(p, &sc.d_xn, &mut sc.d_zxbcdt, embd, in_rows)?;
                            } else {
                                exec.quantize_e4m3_row(
                                    &sc.d_xn,
                                    &mut sc.d_xq,
                                    &mut sc.d_xrs,
                                    embd,
                                    r,
                                )?;
                                exec.f8row_gemm(
                                    p,
                                    &sc.d_xq,
                                    &sc.d_xrs,
                                    &mut sc.d_zxbcdt,
                                    embd,
                                    in_rows,
                                    r,
                                )?;
                                if band > 0 {
                                    exec.dense_w16_e4m3(
                                        p,
                                        embd,
                                        in_rows,
                                        &sc.d_xn,
                                        &mut sc.d_x16h,
                                        &mut sc.d_zxbcdt,
                                        in_rows,
                                        band,
                                    )?;
                                }
                            }
                        }
                        // GGUF lane: repacked GEMV at r=1 (the serial decode
                        // class), the int8 mmq ladder above it
                        LinW::Qw(q) => {
                            if dec1 {
                                gemv_any(&exec, q, &sc.d_xn, &mut sc.d_zxbcdt)?;
                            } else {
                                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                                prefill_quant(
                                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_xn, embd, r,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    q,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_zxbcdt,
                                    r,
                                    pf,
                                )?;
                            }
                        }
                    }
                    let win = bs.conv_win[li].as_mut().expect("conv arena");
                    let ssm = bs.ssm[li].as_mut().expect("ssm arena");
                    match cuts {
                        None => {
                            // every row advances its own slot's arena
                            exec.mamba_conv_step_batch(
                                win,
                                &sc.d_zxbcdt,
                                d_inner,
                                in_rows,
                                &sc.d_slots,
                                &w.conv_w,
                                &w.conv_b,
                                &mut sc.d_conv,
                                conv_dim,
                                hp.d_conv,
                                r,
                            )?;
                            ssm.scan_step_batch(
                                &exec,
                                &sc.d_conv,
                                &sc.d_zxbcdt,
                                d_inner + conv_dim,
                                in_rows,
                                &sc.d_slots,
                                &w.a,
                                &w.d,
                                &w.dt_bias,
                                &mut sc.d_y,
                                r,
                                hp.mamba_heads,
                                hp.mamba_head_dim,
                                hp.d_state,
                                hp.n_groups,
                            )?;
                        }
                        Some(c) => {
                            if c.dec > 0 {
                                exec.mamba_conv_step_batch(
                                    win,
                                    &sc.d_zxbcdt,
                                    d_inner,
                                    in_rows,
                                    &sc.d_slots,
                                    &w.conv_w,
                                    &w.conv_b,
                                    &mut sc.d_conv,
                                    conv_dim,
                                    hp.d_conv,
                                    c.dec,
                                )?;
                                ssm.scan_step_batch(
                                    &exec,
                                    &sc.d_conv,
                                    &sc.d_zxbcdt,
                                    d_inner + conv_dim,
                                    in_rows,
                                    &sc.d_slots,
                                    &w.a,
                                    &w.d,
                                    &w.dt_bias,
                                    &mut sc.d_y,
                                    c.dec,
                                    hp.mamba_heads,
                                    hp.mamba_head_dim,
                                    hp.d_state,
                                    hp.n_groups,
                                )?;
                            }
                            if verify {
                                // spec verify: conv on the slot's SCRATCH
                                // window (live stays pre-round), scan on the
                                // live state with per-row snapshots, xBC rows
                                // snapshotted for the window rebuild
                                let vp = bs.verify.as_mut().expect("verify planes");
                                let vw = vp.vwin[li].as_mut().expect("vwin");
                                let keep_row = crate::gpu::mamba2_keep_row(
                                    hp.mamba_heads,
                                    hp.mamba_head_dim,
                                    hp.d_state,
                                    hp.n_groups,
                                );
                                for &(off, len, slot) in &c.runs {
                                    let s = slot as usize;
                                    exec.copy_region(
                                        win,
                                        s * win_elems,
                                        vw,
                                        s * win_elems,
                                        win_elems,
                                    )?;
                                    exec.mamba_conv_seq_at(
                                        vw,
                                        s * win_elems,
                                        &sc.d_zxbcdt,
                                        off * in_rows + d_inner,
                                        in_rows,
                                        &w.conv_w,
                                        &w.conv_b,
                                        &mut sc.d_conv,
                                        off * conv_dim,
                                        conv_dim,
                                        hp.d_conv,
                                        len,
                                    )?;
                                    if vp.rescan {
                                        // the live state stays pre-round;
                                        // the commit replays the accepted
                                        // rows from the kept ones
                                        ssm.scan_seq_keep_at(
                                            &exec,
                                            s * state_elems,
                                            &sc.d_conv,
                                            off * conv_dim,
                                            &sc.d_zxbcdt,
                                            off * in_rows + d_inner + conv_dim,
                                            in_rows,
                                            &w.a,
                                            &w.d,
                                            &w.dt_bias,
                                            &mut sc.d_y,
                                            off * d_inner,
                                            vp.keep[li].as_mut().expect("keep"),
                                            off * keep_row,
                                            len,
                                            hp.mamba_heads,
                                            hp.mamba_head_dim,
                                            hp.d_state,
                                            hp.n_groups,
                                        )?;
                                        continue;
                                    }
                                    ssm.scan_seq_snap_at(
                                        &exec,
                                        s * state_elems,
                                        &sc.d_conv,
                                        off * conv_dim,
                                        &sc.d_zxbcdt,
                                        off * in_rows + d_inner + conv_dim,
                                        in_rows,
                                        &w.a,
                                        &w.d,
                                        &w.dt_bias,
                                        &mut sc.d_y,
                                        off * d_inner,
                                        vp.snap[li].as_mut().expect("snap"),
                                        off * state_elems,
                                        len,
                                        hp.mamba_heads,
                                        hp.mamba_head_dim,
                                        hp.d_state,
                                        hp.n_groups,
                                    )?;
                                }
                                exec.copy_rows_strided(
                                    &sc.d_zxbcdt,
                                    d_inner,
                                    in_rows,
                                    vp.xbc[li].as_mut().expect("xbc"),
                                    0,
                                    conv_dim,
                                    r,
                                )?;
                                // falls through to the arm's shared tail
                                // (gated norm + out_proj + residual)
                            } else {
                                // each chunk run advances its slot's arena
                                // sequentially - the run walk is the whole reason
                                // runs carry their slot. Checkpoint break rows
                                // split the ADVANCE only (never the pass): the
                                // state at the break is copied into the staging
                                // blob before the tail of the run continues.
                                for &(off, len, slot) in &c.runs {
                                    let s = slot as usize;
                                    let mut seg = off;
                                    for &(brow, stg) in
                                        c.breaks.iter().filter(|&&(b, _)| b > off && b <= off + len)
                                    {
                                        if brow > seg {
                                            exec.mamba_conv_seq_at(
                                                win,
                                                s * win_elems,
                                                &sc.d_zxbcdt,
                                                seg * in_rows + d_inner,
                                                in_rows,
                                                &w.conv_w,
                                                &w.conv_b,
                                                &mut sc.d_conv,
                                                seg * conv_dim,
                                                conv_dim,
                                                hp.d_conv,
                                                brow - seg,
                                            )?;
                                            ssm.scan_seq_at(
                                                &exec,
                                                s * state_elems,
                                                &sc.d_conv,
                                                seg * conv_dim,
                                                &sc.d_zxbcdt,
                                                seg * in_rows + d_inner + conv_dim,
                                                in_rows,
                                                &w.a,
                                                &w.d,
                                                &w.dt_bias,
                                                &mut sc.d_y,
                                                seg * d_inner,
                                                brow - seg,
                                                hp.mamba_heads,
                                                hp.mamba_head_dim,
                                                hp.d_state,
                                                hp.n_groups,
                                            )?;
                                            seg = brow;
                                        }
                                        let blob = mi * (state_elems + win_elems);
                                        let stage_buf = &mut bs.d_ckpt_stage[stg];
                                        ssm.save_to_blob(
                                            &exec,
                                            s * state_elems,
                                            stage_buf,
                                            blob,
                                            state_elems,
                                        )?;
                                        exec.copy_region(
                                            win,
                                            s * win_elems,
                                            stage_buf,
                                            blob + state_elems,
                                            win_elems,
                                        )?;
                                    }
                                    if off + len > seg {
                                        exec.mamba_conv_seq_at(
                                            win,
                                            s * win_elems,
                                            &sc.d_zxbcdt,
                                            seg * in_rows + d_inner,
                                            in_rows,
                                            &w.conv_w,
                                            &w.conv_b,
                                            &mut sc.d_conv,
                                            seg * conv_dim,
                                            conv_dim,
                                            hp.d_conv,
                                            off + len - seg,
                                        )?;
                                        ssm.scan_seq_at(
                                            &exec,
                                            s * state_elems,
                                            &sc.d_conv,
                                            seg * conv_dim,
                                            &sc.d_zxbcdt,
                                            seg * in_rows + d_inner + conv_dim,
                                            in_rows,
                                            &w.a,
                                            &w.d,
                                            &w.dt_bias,
                                            &mut sc.d_y,
                                            seg * d_inner,
                                            off + len - seg,
                                            hp.mamba_heads,
                                            hp.mamba_head_dim,
                                            hp.d_state,
                                            hp.n_groups,
                                        )?;
                                    }
                                }
                            }
                        }
                    }
                    exec.mamba_rmsnorm_gated_g(
                        &sc.d_y,
                        &sc.d_zxbcdt,
                        0,
                        in_rows,
                        &w.norm_w,
                        &mut sc.d_yn,
                        r,
                        d_inner,
                        hp.n_groups,
                        eps,
                    )?;
                    match &w.out_proj {
                        LinW::F8(p) => {
                            if w16 {
                                exec.dense_w16_e4m3(
                                    p,
                                    d_inner,
                                    embd,
                                    &sc.d_yn,
                                    &mut sc.d_x16h,
                                    &mut sc.d_proj,
                                    embd,
                                    r,
                                )?;
                            } else if dec1 {
                                exec.f8r_gemv(p, &sc.d_yn, &mut sc.d_proj, d_inner, embd)?;
                            } else {
                                exec.quantize_e4m3_row(
                                    &sc.d_yn,
                                    &mut sc.d_xq,
                                    &mut sc.d_xrs,
                                    d_inner,
                                    r,
                                )?;
                                exec.f8row_gemm(
                                    p,
                                    &sc.d_xq,
                                    &sc.d_xrs,
                                    &mut sc.d_proj,
                                    d_inner,
                                    embd,
                                    r,
                                )?;
                                if band > 0 {
                                    exec.dense_w16_e4m3(
                                        p,
                                        d_inner,
                                        embd,
                                        &sc.d_yn,
                                        &mut sc.d_x16h,
                                        &mut sc.d_proj,
                                        embd,
                                        band,
                                    )?;
                                }
                            }
                        }
                        LinW::Qw(q) => {
                            if dec1 {
                                gemv_any(&exec, q, &sc.d_yn, &mut sc.d_proj)?;
                            } else {
                                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                                prefill_quant(
                                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_yn, d_inner, r,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    q,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_proj,
                                    r,
                                    pf,
                                )?;
                            }
                        }
                    }
                    mi += 1;
                }
                Mixer::Attn(w) => {
                    // NoPE - no rotary anywhere, projections go to the pool
                    // as computed
                    match w {
                        AttnWeights::F32 {
                            wq, wk, wv, bf16, ..
                        } => {
                            if let (true, Some(b), true) = (w16, bf16, exec.has_dense_w16_seg()) {
                                // the fused plane in one launch
                                exec.dense_w16_bf16_seg(
                                    &b.wqkv,
                                    b.q_dim,
                                    b.kv_dim,
                                    &sc.d_xn,
                                    &mut sc.d_x16b,
                                    &mut sc.d_q,
                                    &mut sc.d_k,
                                    &mut sc.d_v,
                                    r,
                                )?;
                            } else if let (true, Some(b)) = (w16, bf16) {
                                // a pack without the segmented launch: one
                                // cast serves the three segments, same class
                                exec.convert_f32_bf16(&sc.d_xn, &mut sc.d_x16b, r * embd)?;
                                exec.dense_w16_bf16_pre(
                                    &b.wqkv,
                                    0,
                                    b.q_dim,
                                    &sc.d_x16b,
                                    &mut sc.d_q,
                                    b.q_dim,
                                    r,
                                )?;
                                exec.dense_w16_bf16_pre(
                                    &b.wqkv,
                                    b.q_dim,
                                    b.kv_dim,
                                    &sc.d_x16b,
                                    &mut sc.d_k,
                                    b.kv_dim,
                                    r,
                                )?;
                                exec.dense_w16_bf16_pre(
                                    &b.wqkv,
                                    b.q_dim + b.kv_dim,
                                    b.kv_dim,
                                    &sc.d_x16b,
                                    &mut sc.d_v,
                                    b.kv_dim,
                                    r,
                                )?;
                            } else if dec1 {
                                if let Some(b) = bf16 {
                                    exec.bf16_gemv_rows(
                                        &b.wqkv,
                                        0,
                                        b.q_dim,
                                        &sc.d_xn,
                                        &mut sc.d_q,
                                    )?;
                                    exec.bf16_gemv_rows(
                                        &b.wqkv,
                                        b.q_dim,
                                        b.kv_dim,
                                        &sc.d_xn,
                                        &mut sc.d_k,
                                    )?;
                                    exec.bf16_gemv_rows(
                                        &b.wqkv,
                                        b.q_dim + b.kv_dim,
                                        b.kv_dim,
                                        &sc.d_xn,
                                        &mut sc.d_v,
                                    )?;
                                } else {
                                    exec.matvec_f32_batch(wq, &sc.d_xn, &mut sc.d_q, 1)?;
                                    exec.matvec_f32_batch(wk, &sc.d_xn, &mut sc.d_k, 1)?;
                                    exec.matvec_f32_batch(wv, &sc.d_xn, &mut sc.d_v, 1)?;
                                }
                            } else {
                                match (proj16, bf16) {
                                    // batched decode/verify rows: the twins'
                                    // bf16 class (half the plane bytes - the
                                    // c32 ledger had these f32 GEMMs at
                                    // 8.6% of GPU time). One fused launch past
                                    // the mr band - the thin k/v rows ride the
                                    // q grid instead of starving on their own
                                    // (thin-k/v rung). Prefill stays
                                    // on the exact-f32 planes, the
                                    // arbiter-gated class.
                                    (true, Some(b)) => {
                                        super::attn_qkv_batch(
                                            &exec,
                                            b,
                                            &sc.d_xn,
                                            &mut sc.d_q,
                                            &mut sc.d_k,
                                            &mut sc.d_v,
                                            r,
                                        )?;
                                    }
                                    _ => {
                                        exec.gemm_f32(
                                            &wq.buf,
                                            embd,
                                            q_dim,
                                            &sc.d_xn,
                                            &mut sc.d_q,
                                            r,
                                        )?;
                                        exec.gemm_f32(
                                            &wk.buf,
                                            embd,
                                            kv_dim,
                                            &sc.d_xn,
                                            &mut sc.d_k,
                                            r,
                                        )?;
                                        exec.gemm_f32(
                                            &wv.buf,
                                            embd,
                                            kv_dim,
                                            &sc.d_xn,
                                            &mut sc.d_v,
                                            r,
                                        )?;
                                        if let (true, Some(b)) = (band > 0, bf16) {
                                            exec.dense_w16_bf16_seg(
                                                &b.wqkv,
                                                b.q_dim,
                                                b.kv_dim,
                                                &sc.d_xn,
                                                &mut sc.d_x16b,
                                                &mut sc.d_q,
                                                &mut sc.d_k,
                                                &mut sc.d_v,
                                                band,
                                            )?;
                                        }
                                    }
                                }
                            }
                        }
                        AttnWeights::Qw { wq, wk, wv, .. } => {
                            if dec1 {
                                gemv_any(&exec, wq, &sc.d_xn, &mut sc.d_q)?;
                                gemv_any(&exec, wk, &sc.d_xn, &mut sc.d_k)?;
                                gemv_any(&exec, wv, &sc.d_xn, &mut sc.d_v)?;
                            } else {
                                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                                prefill_quant(
                                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_xn, embd, r,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    wq,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_q,
                                    r,
                                    pf,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    wk,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_k,
                                    r,
                                    pf,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    wv,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_v,
                                    r,
                                    pf,
                                )?;
                            }
                        }
                    }
                    let kvs = bs.kv[li].as_mut().expect("paged kv");
                    exec.kv_append_batch_paged(
                        &sc.d_k,
                        &mut kvs.k,
                        &sc.d_pos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        kv_dim,
                        r,
                        kv_dtype,
                    )?;
                    exec.kv_append_batch_paged(
                        &sc.d_v,
                        &mut kvs.v,
                        &sc.d_pos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        kv_dim,
                        r,
                        kv_dtype,
                    )?;
                    match cuts {
                        None if w16 && r <= sc.w16_rows => {
                            // the W16 class: a group per row on the fixed
                            // key splits - the law a verify round attends by
                            exec.attn_rows_partial_fixed(
                                &sc.d_q,
                                &kvs.k,
                                &kvs.v,
                                &mut sc.d_w16o,
                                &mut sc.d_w16ml,
                                &sc.d_pos,
                                &sc.d_slots,
                                &sc.d_w16_groups,
                                r,
                                Some((&bs.d_bt, bps)),
                                0,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                r,
                                sc.w16_ns,
                                0,
                                scale,
                                kv_dtype,
                                sc.w16_split,
                                sc.w16_kh.then_some(1),
                            )?;
                            exec.attn_combine_batch(
                                &sc.d_w16o,
                                &sc.d_w16ml,
                                &sc.d_sinks,
                                &mut sc.d_attn,
                                nh,
                                hd,
                                sc.w16_ns,
                                r,
                            )?;
                        }
                        None => {
                            // the head-packed arm stays split at ns = 1:
                            // the unsplit kernel is the per-q-head walk
                            if (ns > 1 || hp16) && exec.has_attn_partial_batch_paged() {
                                exec.attn_partial_batch_paged(
                                    &sc.d_q,
                                    &kvs.k,
                                    &kvs.v,
                                    &mut sc.attn_o,
                                    &mut sc.attn_ml,
                                    &sc.d_pos,
                                    Some(&sc.d_slots),
                                    &bs.d_bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    0,
                                    ns,
                                    r,
                                    scale,
                                    kv_dtype,
                                )?;
                                exec.attn_combine_batch(
                                    &sc.attn_o,
                                    &sc.attn_ml,
                                    &sc.d_sinks,
                                    &mut sc.d_attn,
                                    nh,
                                    hd,
                                    ns,
                                    r,
                                )?;
                            } else {
                                exec.attn_decode_batch_paged(
                                    &sc.d_q,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.d_sinks,
                                    &mut sc.d_attn,
                                    &sc.d_pos,
                                    Some(&sc.d_slots),
                                    &bs.d_bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    0,
                                    r,
                                    scale,
                                    kv_dtype,
                                )?;
                            }
                        }
                        Some(c) => {
                            if band > 0 {
                                // the band attends as a decode tick's rows do:
                                // a group per row on the fixed key splits
                                exec.attn_rows_partial_fixed(
                                    &sc.d_q,
                                    &kvs.k,
                                    &kvs.v,
                                    &mut sc.d_w16o,
                                    &mut sc.d_w16ml,
                                    &sc.d_pos,
                                    &sc.d_slots,
                                    &sc.d_w16_groups,
                                    band,
                                    Some((&bs.d_bt, bps)),
                                    0,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    band,
                                    sc.w16_ns,
                                    0,
                                    scale,
                                    kv_dtype,
                                    sc.w16_split,
                                    sc.w16_kh.then_some(1),
                                )?;
                                exec.attn_combine_batch(
                                    &sc.d_w16o,
                                    &sc.d_w16ml,
                                    &sc.d_sinks,
                                    &mut sc.d_attn,
                                    nh,
                                    hd,
                                    sc.w16_ns,
                                    band,
                                )?;
                            } else if c.dec > 0 {
                                exec.attn_decode_batch_rows_paged(
                                    &sc.d_q,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.d_sinks,
                                    &mut sc.d_attn,
                                    &sc.d_pos,
                                    Some(&sc.d_slots),
                                    &bs.d_bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    0,
                                    0,
                                    c.dec,
                                    scale,
                                    kv_dtype,
                                )?;
                            }
                            if let (Some((ng, _, gmax)), true) =
                                (rows_attn, w16 && r <= sc.w16_rows)
                            {
                                // the W16 class: the decode ticks' split law
                                let vp = bs.verify.as_ref().expect("verify planes");
                                exec.attn_rows_partial_fixed(
                                    &sc.d_q,
                                    &kvs.k,
                                    &kvs.v,
                                    &mut sc.d_w16o,
                                    &mut sc.d_w16ml,
                                    &sc.d_pos,
                                    &sc.d_slots,
                                    &vp.d_groups,
                                    ng,
                                    Some((&bs.d_bt, bps)),
                                    0,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    r,
                                    sc.w16_ns,
                                    0,
                                    scale,
                                    kv_dtype,
                                    sc.w16_split,
                                    sc.w16_kh.then_some(gmax),
                                )?;
                                exec.attn_combine_batch(
                                    &sc.d_w16o,
                                    &sc.d_w16ml,
                                    &sc.d_sinks,
                                    &mut sc.d_attn,
                                    nh,
                                    hd,
                                    sc.w16_ns,
                                    r,
                                )?;
                            } else if let Some((ng, nsr, _)) = rows_attn {
                                let vp = bs.verify.as_mut().expect("verify planes");
                                exec.attn_rows_partial(
                                    &sc.d_q,
                                    &kvs.k,
                                    &kvs.v,
                                    &mut vp.attn_o,
                                    &mut vp.attn_ml,
                                    &sc.d_pos,
                                    &sc.d_slots,
                                    &vp.d_groups,
                                    ng,
                                    Some((&bs.d_bt, bps)),
                                    0,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    r,
                                    nsr,
                                    0,
                                    scale,
                                    kv_dtype,
                                )?;
                                exec.attn_combine_batch(
                                    &vp.attn_o,
                                    &vp.attn_ml,
                                    &sc.d_sinks,
                                    &mut sc.d_attn,
                                    nh,
                                    hd,
                                    nsr,
                                    r,
                                )?;
                            }
                            // verify rows went through the multi-row partial above
                            let tile_runs: &[(usize, usize, u32)] =
                                if rows_attn.is_some() { &[] } else { &c.runs };
                            for &(off, len, _slot) in tile_runs {
                                if wmma_pf {
                                    exec.attn_prefill_f16_paged_at(
                                        &sc.d_q,
                                        &kvs.k,
                                        &kvs.v,
                                        &sc.d_sinks,
                                        &mut sc.d_attn,
                                        &sc.d_pos,
                                        &sc.d_slots,
                                        off,
                                        &bs.d_bt,
                                        bps,
                                        nh,
                                        n_kv,
                                        hd,
                                        kv_dim,
                                        0,
                                        len,
                                        scale,
                                        kv_dtype,
                                    )?;
                                } else if len > 24 && exec.has_attn_prefill_paged() {
                                    exec.attn_prefill_rows_paged(
                                        &sc.d_q,
                                        &kvs.k,
                                        &kvs.v,
                                        &sc.d_sinks,
                                        &mut sc.d_attn,
                                        &sc.d_pos,
                                        &sc.d_slots,
                                        &bs.d_bt,
                                        bps,
                                        nh,
                                        n_kv,
                                        hd,
                                        kv_dim,
                                        0,
                                        off,
                                        len,
                                        scale,
                                        kv_dtype,
                                    )?;
                                } else {
                                    exec.attn_decode_batch_rows_paged(
                                        &sc.d_q,
                                        &kvs.k,
                                        &kvs.v,
                                        &sc.d_sinks,
                                        &mut sc.d_attn,
                                        &sc.d_pos,
                                        Some(&sc.d_slots),
                                        &bs.d_bt,
                                        bps,
                                        nh,
                                        n_kv,
                                        hd,
                                        kv_dim,
                                        0,
                                        off,
                                        len,
                                        scale,
                                        kv_dtype,
                                    )?;
                                }
                            }
                        }
                    }
                    match w {
                        AttnWeights::F32 { wo, bf16, .. } => {
                            if let (true, Some(b)) = (w16, bf16) {
                                exec.dense_w16_bf16(
                                    &b.wo,
                                    0,
                                    embd,
                                    &sc.d_attn,
                                    &mut sc.d_x16b,
                                    &mut sc.d_proj,
                                    embd,
                                    r,
                                )?;
                            } else if dec1 {
                                if let Some(b) = bf16 {
                                    exec.bf16_gemv(&b.wo, None, &sc.d_attn, &mut sc.d_proj)?;
                                } else {
                                    exec.matvec_f32_batch(wo, &sc.d_attn, &mut sc.d_proj, 1)?;
                                }
                            } else {
                                match (proj16, bf16) {
                                    // batched decode: bf16 twin class (see
                                    // the QKV arm above)
                                    (true, Some(b)) => {
                                        exec.bf16_gemm(&b.wo, None, &sc.d_attn, &mut sc.d_proj, r)?;
                                    }
                                    _ => {
                                        exec.gemm_f32(
                                            &wo.buf,
                                            q_dim,
                                            embd,
                                            &sc.d_attn,
                                            &mut sc.d_proj,
                                            r,
                                        )?;
                                        if let (true, Some(b)) = (band > 0, bf16) {
                                            exec.dense_w16_bf16(
                                                &b.wo,
                                                0,
                                                embd,
                                                &sc.d_attn,
                                                &mut sc.d_x16b,
                                                &mut sc.d_proj,
                                                embd,
                                                band,
                                            )?;
                                        }
                                    }
                                }
                            }
                        }
                        AttnWeights::Qw { wo, .. } => {
                            if dec1 {
                                gemv_any(&exec, wo, &sc.d_attn, &mut sc.d_proj)?;
                            } else {
                                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                                prefill_quant(
                                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_attn, q_dim, r,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    wo,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_proj,
                                    r,
                                    pf,
                                )?;
                            }
                        }
                    }
                }
                Mixer::Moe(w) => {
                    // shared fold-: the loader registers the
                    // shared expert as ns_sh pseudo-experts appended to the
                    // NVFP4 routed planes; the r>1 path then widens the topk
                    // rows by ns_sh constant picks and serves everything in
                    // one sorted-tile pair (the separate 1-block shared pass
                    // ran at 10-12% of the stream roof). dec1 and the Q8
                    // lane keep the plain k-wide rows.
                    let (ns_sh, moe_tiled) = match &w.planes {
                        MoePlanes::Nvf4 { up, .. } => (
                            up.n_expert - hp.n_expert,
                            up.layout == crate::gpu::Nvf4MoeLayout::Tiled64,
                        ),
                        _ => (0, false),
                    };
                    // the W16 class's experts: plain k-wide topk rows through
                    // the routing front, the shared expert on its own planes
                    // (as one-row decode ran it)
                    if w16 && moe_tiled {
                        moe_w16_rows(&exec, &hp, w, sc, r)?;
                        continue;
                    }
                    // skinny-tile decode election: tiled
                    // planes + a pure-decode tick route the ROUTED experts
                    // through the BM=8 pair (fill is ~2.4 at c32; 32-wide
                    // blocks are ~7.5% live) and the shared expert through
                    // the WIDE tiled pair on its resident planes - a BM=8
                    // fold-in would split the always-full shared pseudo-
                    // experts into ceil(r/8) blocks and re-read their strips.
                    // The tiny shared grid rides the routed grid's PDL tail
                    // (the rung-17 law working for us, not against).
                    let skinny = moe_tiled && !pf && !dec1;
                    // BM=8 fold of the shared expert (lever 14, GB10 c8):
                    // at r <= 8 every pseudo-expert is one block, so the
                    // strip re-read the comment above fears does not
                    // happen, and the separate wide pair - 28.6 + 26.1 us
                    // per layer PDL-accounted, 2.5x its 7.5 MB byte floor,
                    // 1.26 ms of a 25.4 ms c8 tick - becomes two more
                    // blocks of a launch that streams at ~93% of the roof.
                    // PADDOCK_NO_NEMO_SH_FOLD8=1 keeps the separate pair.
                    let fold8 = skinny && ns_sh > 0 && r <= 8 && sh_fold8_on();
                    exec.matvec_f32_batch(&w.router, &sc.d_xn, &mut sc.d_logits_r, r)?;
                    if ns_sh > 0 && !dec1 && (!skinny || fold8) {
                        exec.moe_topk_sigmoid_batch_sh(
                            &sc.d_logits_r,
                            &w.bias.buf,
                            hp.routed_scale,
                            hp.n_expert,
                            hp.n_active,
                            ns_sh,
                            hp.n_expert,
                            &mut sc.d_idx,
                            &mut sc.d_w,
                            r,
                        )?;
                    } else {
                        exec.moe_topk_sigmoid_batch(
                            &sc.d_logits_r,
                            &w.bias.buf,
                            hp.routed_scale,
                            hp.n_expert,
                            hp.n_active,
                            &mut sc.d_idx,
                            &mut sc.d_w,
                            r,
                        )?;
                    }
                    //  diagnostic (PADDOCK_MOE_UNIQ=path), the task
                    // The MoE rung's attribution instrument: the real
                    // uniq-routed-experts-per-(tick,layer) histogram -
                    // pairs walks the full idx rows (k routed picks + the
                    // ns_sh shared folds in the _sh lane; the kernel's
                    // 128-bit bitmap skips ids >= 128, so uniq = ROUTED
                    // uniq and the shared folds only ride the pairs
                    // totals). Sits before the arm branches so every route
                    // is measured; launch-only, so captured decode graphs
                    // bake it in and it keeps counting on replays.
                    if sc.moe_uniq_dev != 0 {
                        let kw = if ns_sh > 0 && !dec1 && (!skinny || fold8) {
                            hp.n_active + ns_sh
                        } else {
                            hp.n_active
                        };
                        exec.moe_uniq_hist(&sc.d_idx, r * kw, hp.n_expert, sc.moe_uniq_dev)?;
                    }
                    let MoePlanes::Nvf4 {
                        up,
                        down,
                        sh_up,
                        sh_down,
                    } = &w.planes
                    else {
                        // GGUF lane: same class split as the serial spine -
                        // r=1 decode on the token-batched dp4a relu2 pair
                        // (write + one add; falls through to the residual
                        // add), r>1 on the sorted tiles folding straight
                        // into the residual
                        let MoePlanes::Q8 {
                            up,
                            down,
                            sh_up,
                            sh_down,
                        } = &w.planes
                        else {
                            unreachable!("nemotron MoePlanes is Nvf4 or Q8");
                        };
                        let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                        if dec1 {
                            exec.quantize_q8(&sc.d_xn, &mut s8.xq, &mut s8.xs, embd)?;
                            exec.q8_0_moe_up_relu2(
                                up,
                                &sc.d_idx,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.act_r,
                                hp.n_active,
                                1,
                            )?;
                            exec.quantize_q8(
                                &s8.act_r,
                                &mut s8.fq_r1,
                                &mut s8.fs_r1,
                                hp.n_active * hp.moe_ff,
                            )?;
                            exec.q8_0_moe_down(
                                down,
                                &sc.d_idx,
                                &sc.d_w,
                                &s8.fq_r1,
                                &s8.fs_r1,
                                &mut sc.d_proj,
                                hp.n_active,
                                1,
                            )?;
                            exec.q8_0_moe_up_relu2(
                                sh_up,
                                &sc.d_sh_idx,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.act_s,
                                1,
                                1,
                            )?;
                            exec.quantize_q8(
                                &s8.act_s,
                                &mut s8.fq_s1,
                                &mut s8.fs_s1,
                                hp.shared_ff,
                            )?;
                            exec.q8_0_moe_down(
                                sh_down,
                                &sc.d_sh_idx,
                                &sc.d_sh_w,
                                &s8.fq_s1,
                                &s8.fs_s1,
                                &mut s8.shproj,
                                1,
                                1,
                            )?;
                            exec.add(&mut sc.d_proj, &s8.shproj, embd)?;
                        } else if !pf && r <= MOE_DEC2_MAX_ROWS && moe_dec2_ok(&exec) {
                            // DECODE BAND. Two separate
                            // shape mistakes shared one arm before this:
                            //
                            //  - the ROUTED experts rode the sorted BM=32
                            //    tile, which at r=4/top-6 over 128 experts
                            //    puts one real row in a 32-row block. It is
                            //    not just wasted flops: the tile measured
                            //    FLAT at ~144 GB/s where the same bytes on
                            //    the dec2 pair (warp per output row, no pad,
                            //    no align, no combine) stream at ~660.
                            //  - the shared expert is not an expert at all.
                            //    Every row uses it, so it is a plain dense
                            //    FFN, and it belongs on the same q8 ladder
                            //    the dense projections took at the time: one
                            //    weight pass for the whole tick instead of
                            //    the 1-block align+tile pair, which was
                            //    another flat ~117 GB/s. The only thing that
                            //    was missing is the activation - hence
                            //    quantize_q8_relu2, which folds relu(x)^2
                            //    into the quantize between up and down and
                            //    is bit-identical to doing it in f32.
                            //
                            // The epilogue quantize shrinks with the routed
                            // plane too: nb*32*moe_ff -> r*n_active*moe_ff,
                            // 24x fewer elements at r=4 (it was 16% of the
                            // tick in the profile).
                            exec.quantize_q8(&sc.d_xn, &mut s8.xq, &mut s8.xs, r * embd)?;
                            exec.q8_0_moe_up_relu2_dec2(
                                up,
                                &sc.d_idx,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.fu_r,
                                hp.n_active,
                                r,
                                0,
                            )?;
                            exec.quantize_q8(
                                &s8.fu_r,
                                &mut s8.fq_r,
                                &mut s8.fs_r,
                                r * hp.n_active * hp.moe_ff,
                            )?;
                            exec.q8_0_moe_dn_dec2(
                                down,
                                &sc.d_idx,
                                &sc.d_w,
                                &s8.fq_r,
                                &s8.fs_r,
                                &mut sc.d_proj,
                                hp.n_active,
                                r,
                            )?;
                            mmq_pre(
                                &exec,
                                sh_up,
                                &s8.xq,
                                &s8.xs,
                                &mut sc.d_part,
                                &mut s8.fu_s,
                                r,
                            )?;
                            exec.quantize_q8_relu2(
                                &s8.fu_s,
                                &mut s8.fq_s,
                                &mut s8.fs_s,
                                r * hp.shared_ff,
                            )?;
                            mmq_pre(
                                &exec,
                                sh_down,
                                &s8.fq_s,
                                &s8.fs_s,
                                &mut sc.d_part,
                                &mut s8.shproj,
                                r,
                            )?;
                            exec.add(&mut sc.d_proj, &s8.shproj, r * embd)?;
                        } else {
                            exec.quantize_q8(&sc.d_xn, &mut s8.xq, &mut s8.xs, r * embd)?;
                            // the int8-MMA pair where the pack has it, the
                            // dp4a pair + quantize otherwise - bitwise either
                            // way (see nemo_qmma_on)
                            let qmma = nemo_qmma_on(&exec);
                            let nbr = moe_live_blocks(r, hp.n_active, hp.n_expert, sc.nb_r);
                            exec.moe_align(
                                &sc.d_idx,
                                &mut sc.d_srow,
                                &mut sc.d_sslot,
                                &mut sc.d_bexp,
                                r,
                                hp.n_active,
                                hp.n_expert,
                                nbr,
                            )?;
                            if qmma {
                                exec.q8_0_moe_up_relu2_mma(
                                    up,
                                    &sc.d_srow,
                                    &sc.d_bexp,
                                    &s8.xq,
                                    &s8.xs,
                                    &mut s8.fq_r,
                                    &mut s8.fs_r,
                                    nbr,
                                    32,
                                )?;
                                exec.q8_0_moe_down_mma(
                                    down,
                                    &sc.d_srow,
                                    &sc.d_sslot,
                                    &sc.d_bexp,
                                    &sc.d_w,
                                    &s8.fq_r,
                                    &s8.fs_r,
                                    &mut sc.d_part,
                                    hp.n_active,
                                    nbr,
                                    32,
                                )?;
                            } else {
                                exec.q8_0_moe_up_relu2_sorted(
                                    up,
                                    &sc.d_srow,
                                    &sc.d_bexp,
                                    &s8.xq,
                                    &s8.xs,
                                    &mut s8.fu_r,
                                    nbr,
                                )?;
                                exec.quantize_q8(
                                    &s8.fu_r,
                                    &mut s8.fq_r,
                                    &mut s8.fs_r,
                                    nbr * 32 * hp.moe_ff,
                                )?;
                                exec.q8_0_moe_down_sorted(
                                    down,
                                    &sc.d_srow,
                                    &sc.d_sslot,
                                    &sc.d_bexp,
                                    &sc.d_w,
                                    &s8.fq_r,
                                    &s8.fs_r,
                                    &mut sc.d_part,
                                    hp.n_active,
                                    nbr,
                                )?;
                            }
                            exec.moe_slot_combine(&sc.d_part, &mut sc.d_x, embd, hp.n_active, r)?;
                            let nbs = moe_live_blocks(r, 1, 1, sc.nb_s);
                            exec.moe_align(
                                &sc.d_sh_idx,
                                &mut sc.d_srow_s,
                                &mut sc.d_sslot_s,
                                &mut sc.d_bexp_s,
                                r,
                                1,
                                1,
                                nbs,
                            )?;
                            if qmma {
                                // one expert, every row: the 32-row blocks
                                // are full, so this is the dense FFN on the
                                // tensor cores
                                exec.q8_0_moe_up_relu2_mma(
                                    sh_up,
                                    &sc.d_srow_s,
                                    &sc.d_bexp_s,
                                    &s8.xq,
                                    &s8.xs,
                                    &mut s8.fq_s,
                                    &mut s8.fs_s,
                                    nbs,
                                    32,
                                )?;
                                exec.q8_0_moe_down_mma(
                                    sh_down,
                                    &sc.d_srow_s,
                                    &sc.d_sslot_s,
                                    &sc.d_bexp_s,
                                    &sc.d_sh_w,
                                    &s8.fq_s,
                                    &s8.fs_s,
                                    &mut sc.d_proj,
                                    1,
                                    nbs,
                                    32,
                                )?;
                            } else {
                                exec.q8_0_moe_up_relu2_sorted(
                                    sh_up,
                                    &sc.d_srow_s,
                                    &sc.d_bexp_s,
                                    &s8.xq,
                                    &s8.xs,
                                    &mut s8.fu_s,
                                    nbs,
                                )?;
                                exec.quantize_q8(
                                    &s8.fu_s,
                                    &mut s8.fq_s,
                                    &mut s8.fs_s,
                                    nbs * 32 * hp.shared_ff,
                                )?;
                                exec.q8_0_moe_down_sorted(
                                    sh_down,
                                    &sc.d_srow_s,
                                    &sc.d_sslot_s,
                                    &sc.d_bexp_s,
                                    &sc.d_sh_w,
                                    &s8.fq_s,
                                    &s8.fs_s,
                                    &mut sc.d_proj,
                                    1,
                                    nbs,
                                )?;
                            }
                            exec.moe_slot_combine(&sc.d_proj, &mut sc.d_x, embd, 1, r)?;
                            continue;
                        }
                        // residual add for the two arms that leave their
                        // whole MoE output in d_proj (dec1 and the decode
                        // band); the sorted arm folds through slot_combine
                        // and skips this with its own continue
                        exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
                        continue;
                    };
                    if dec1 {
                        // the serial decode's fused wave-dense pair + the
                        // fixed ascending-slot fold into the residual
                        // (tiled planes ride the regrouped _mtt twins)
                        if moe_tiled {
                            exec.nvf4_moe_up_relu2_mtt(
                                up,
                                sh_up,
                                &sc.d_idx,
                                &sc.d_xn,
                                &mut sc.d_act,
                                hp.n_active,
                            )?;
                            exec.nvf4_moe_down_part_tt(
                                down,
                                sh_down,
                                &sc.d_idx,
                                &sc.d_w,
                                &sc.d_act,
                                &mut sc.d_part7,
                                hp.n_active,
                            )?;
                        } else {
                            exec.nvf4_moe_up_relu2_mt(
                                up,
                                sh_up,
                                &sc.d_idx,
                                &sc.d_xn,
                                &mut sc.d_act,
                                hp.n_active,
                            )?;
                            exec.nvf4_moe_down_part(
                                down,
                                sh_down,
                                &sc.d_idx,
                                &sc.d_w,
                                &sc.d_act,
                                &mut sc.d_part7,
                                hp.n_active,
                            )?;
                        }
                        exec.moe_slot_combine(&sc.d_part7, &mut sc.d_x, embd, hp.n_active + 1, 1)?;
                        continue;
                    }
                    // sorted-tile mxf4nvf4 MMA class (the serial bulk
                    // prefill's rung-2 lane at r rows; BM=32 pad waste at
                    // small r is the accepted first cut). With the shared
                    // fold-in (ns_sh > 0) the shared expert's pseudo-expert
                    // blocks ride the same align + pair launch - its picks
                    // occupy slots n_active.., and the fixed-order combine
                    // sums the down K halves (the sanctioned split-K
                    // regroup, same class as the slot fold itself).
                    if skinny {
                        // routed experts: plain topk rows, BM=8 blocks; with
                        // the fold the shared pseudo-experts ride along as
                        // picks n_active.. (one block each at r <= 8)
                        let ns_f = if fold8 { ns_sh } else { 0 };
                        let kw = hp.n_active + ns_f;
                        let np = if fold8 { kw } else { hp.n_active + 1 };
                        if !pro_done {
                            exec.quantize_nvf4(&sc.d_xn, &mut sc.d_xq4, &mut sc.d_xs4, r * embd)?;
                        }
                        let nbr = moe_live_blocks_bm8(r, kw, hp.n_expert + ns_f, sc.nb_r);
                        exec.moe_align_bm(
                            &sc.d_idx,
                            &mut sc.d_srow,
                            &mut sc.d_sslot,
                            &mut sc.d_bexp,
                            r,
                            kw,
                            hp.n_expert + ns_f,
                            8,
                            nbr,
                        )?;
                        exec.nvf4_moe_up_relu2_st(
                            up,
                            &sc.d_srow,
                            &sc.d_bexp,
                            &sc.d_xq4,
                            &sc.d_xs4,
                            &mut sc.d_fq,
                            &mut sc.d_fs,
                            nbr,
                            8,
                        )?;
                        exec.nvf4_moe_down_st(
                            down,
                            &sc.d_srow,
                            &sc.d_sslot,
                            &sc.d_bexp,
                            Some(&sc.d_w),
                            &sc.d_fq,
                            &sc.d_fs,
                            &mut sc.d_part,
                            kw,
                            np,
                            0,
                            nbr,
                            8,
                        )?;
                        if !fold8 {
                            // shared expert: resident sh planes, WIDE tiled
                            // pair (full 32-blocks; the 1-block grid
                            // overlaps the routed grid's drain under PDL)
                            let nbs = moe_live_blocks(r, 1, 1, sc.nb_s);
                            exec.moe_align(
                                &sc.d_sh_idx,
                                &mut sc.d_srow_s,
                                &mut sc.d_sslot_s,
                                &mut sc.d_bexp_s,
                                r,
                                1,
                                1,
                                nbs,
                            )?;
                            exec.nvf4_moe_up_relu2_st(
                                sh_up,
                                &sc.d_srow_s,
                                &sc.d_bexp_s,
                                &sc.d_xq4,
                                &sc.d_xs4,
                                &mut sc.d_fq_s,
                                &mut sc.d_fs_s,
                                nbs,
                                32,
                            )?;
                            exec.nvf4_moe_down_st(
                                sh_down,
                                &sc.d_srow_s,
                                &sc.d_sslot_s,
                                &sc.d_bexp_s,
                                None,
                                &sc.d_fq_s,
                                &sc.d_fs_s,
                                &mut sc.d_part,
                                1,
                                np,
                                hp.n_active,
                                nbs,
                                32,
                            )?;
                        }
                        exec.moe_slot_combine(&sc.d_part, &mut sc.d_x, embd, np, r)?;
                        continue;
                    }
                    let kw = hp.n_active + ns_sh;
                    let np = if ns_sh > 0 { kw } else { hp.n_active + 1 };
                    if !pro_done {
                        exec.quantize_nvf4(&sc.d_xn, &mut sc.d_xq4, &mut sc.d_xs4, r * embd)?;
                    }
                    // same live-block extent as the Q8 arm - this one has no
                    // capacity-sized epilogue quantize (up_bs writes fq/fs per
                    // block), so all it drops is pad CTAs. UNMEASURED on
                    // sm_120: nvf4 needs a Blackwell die.
                    let nbr = moe_live_blocks(r, kw, hp.n_expert + ns_sh, sc.nb_r);
                    exec.moe_align(
                        &sc.d_idx,
                        &mut sc.d_srow,
                        &mut sc.d_sslot,
                        &mut sc.d_bexp,
                        r,
                        kw,
                        hp.n_expert + ns_sh,
                        nbr,
                    )?;
                    if moe_tiled {
                        exec.nvf4_moe_up_relu2_st(
                            up,
                            &sc.d_srow,
                            &sc.d_bexp,
                            &sc.d_xq4,
                            &sc.d_xs4,
                            &mut sc.d_fq,
                            &mut sc.d_fs,
                            nbr,
                            32,
                        )?;
                        exec.nvf4_moe_down_st(
                            down,
                            &sc.d_srow,
                            &sc.d_sslot,
                            &sc.d_bexp,
                            Some(&sc.d_w),
                            &sc.d_fq,
                            &sc.d_fs,
                            &mut sc.d_part,
                            kw,
                            np,
                            0,
                            nbr,
                            32,
                        )?;
                    } else {
                        exec.nvf4_moe_up_relu2_bs(
                            up,
                            &sc.d_srow,
                            &sc.d_bexp,
                            &sc.d_xq4,
                            &sc.d_xs4,
                            &mut sc.d_fq,
                            &mut sc.d_fs,
                            nbr,
                        )?;
                        exec.nvf4_moe_down_bs(
                            down,
                            &sc.d_srow,
                            &sc.d_sslot,
                            &sc.d_bexp,
                            Some(&sc.d_w),
                            &sc.d_fq,
                            &sc.d_fs,
                            &mut sc.d_part,
                            kw,
                            np,
                            0,
                            nbr,
                        )?;
                    }
                    if ns_sh == 0 {
                        // No fold-in (shared_ff not a clean multiple of
                        // moe_ff): the separate 1-block shared pass. Its grid
                        // is (1, rt) - 29 and 21 CTAs on a 188-SM die at ~20%
                        // of peak DRAM - which looks like an obvious
                        // underfill rung and is not one: a delete-the-work
                        // probe measured its WALL cost at ~zero. It rides
                        // entirely in the routed pair's PDL shadow. Do not
                        // build a K-split for it without new evidence.
                        let nbs = moe_live_blocks(r, 1, 1, sc.nb_s);
                        exec.moe_align(
                            &sc.d_sh_idx,
                            &mut sc.d_srow_s,
                            &mut sc.d_sslot_s,
                            &mut sc.d_bexp_s,
                            r,
                            1,
                            1,
                            nbs,
                        )?;
                        if moe_tiled {
                            exec.nvf4_moe_up_relu2_st(
                                sh_up,
                                &sc.d_srow_s,
                                &sc.d_bexp_s,
                                &sc.d_xq4,
                                &sc.d_xs4,
                                &mut sc.d_fq_s,
                                &mut sc.d_fs_s,
                                nbs,
                                32,
                            )?;
                            exec.nvf4_moe_down_st(
                                sh_down,
                                &sc.d_srow_s,
                                &sc.d_sslot_s,
                                &sc.d_bexp_s,
                                None,
                                &sc.d_fq_s,
                                &sc.d_fs_s,
                                &mut sc.d_part,
                                1,
                                np,
                                hp.n_active,
                                nbs,
                                32,
                            )?;
                        } else {
                            exec.nvf4_moe_up_relu2_bs(
                                sh_up,
                                &sc.d_srow_s,
                                &sc.d_bexp_s,
                                &sc.d_xq4,
                                &sc.d_xs4,
                                &mut sc.d_fq_s,
                                &mut sc.d_fs_s,
                                nbs,
                            )?;
                            exec.nvf4_moe_down_bs(
                                sh_down,
                                &sc.d_srow_s,
                                &sc.d_sslot_s,
                                &sc.d_bexp_s,
                                None,
                                &sc.d_fq_s,
                                &sc.d_fs_s,
                                &mut sc.d_part,
                                1,
                                np,
                                hp.n_active,
                                nbs,
                            )?;
                        }
                    }
                    // a mixed tick's decode band: its residual rows ride out
                    // the chunk's fold over every row, then take the class's
                    // own experts
                    let band_moe = band > 0 && moe_tiled;
                    if band_moe {
                        exec.copy_region(&sc.d_x, 0, &mut sc.d_band_x, 0, band * embd)?;
                    }
                    exec.moe_slot_combine(&sc.d_part, &mut sc.d_x, embd, np, r)?;
                    if band_moe {
                        exec.copy_region(&sc.d_band_x, 0, &mut sc.d_x, 0, band * embd)?;
                        moe_w16_rows(&exec, &hp, w, sc, band)?;
                    }
                    continue;
                }
            }
            // Hoist this add into the next layer's prologue when that layer is
            // an nvf4 MoE on the bs arm - the shape the fused kernel serves,
            // and the one every non-MoE layer in this checkpoint precedes.
            let next_bs_moe = glue_fuse
                && matches!(
                    self.layers.get(li + 1).map(|l| &l.mixer),
                    Some(Mixer::Moe(w)) if matches!(w.planes, MoePlanes::Nvf4 { .. })
                );
            if next_bs_moe {
                let next_w = &self.layers[li + 1].norm.buf;
                exec.add_rmsnorm_quant_nvf4_batch(
                    &mut bs.sc.d_x,
                    Some(&bs.sc.d_proj),
                    next_w,
                    &mut bs.sc.d_xn,
                    &mut bs.sc.d_xq4,
                    &mut bs.sc.d_xs4,
                    embd,
                    eps,
                    r,
                )?;
                fused_pro = true;
            } else {
                exec.add(&mut bs.sc.d_x, &bs.sc.d_proj, r * embd)?;
            }
        }
        // last aux tap: the final layer's post-block residual
        if let Some(df) = self.dflash.as_mut()
            && let Some(st) = df.state.as_mut()
            && let Some(ai) = df
                .target_layers
                .iter()
                .position(|&t| t == self.hp.n_layer - 1)
        {
            let sc = &bs.sc;
            exec.copy_region(&sc.d_x, 0, &mut st.aux[ai], 0, r * embd)?;
        }
        Ok(())
    }
}

/// The W16 class's MoE over rows [0, rows): the routing front (the router,
/// sigmoid top-k, the rows' bf16 cast and the 32-row sorting in one launch -
/// on a pack without it, the four launches it replaces), the W4A16 expert
/// pair with the shared expert over every row, and the [rows][k + 1]
/// partials folded in slot order into the residual. A decode tick's rows and
/// a mixed tick's decode band both come here.
fn moe_w16_rows(
    exec: &GpuExecutor,
    hp: &NemotronConfig,
    w: &MoeWeights,
    sc: &mut NemoBatchScratch,
    rows: usize,
) -> Result<(), GpuModelError> {
    let MoePlanes::Nvf4 {
        up,
        down,
        sh_up,
        sh_down,
    } = &w.planes
    else {
        unreachable!("the W16 class serves the NVFP4 tiled planes");
    };
    let (embd, k) = (hp.hidden, hp.n_active);
    let nbr = moe_live_blocks(rows, k, hp.n_expert, sc.nb_r);
    if exec.has_moe_route_w16() {
        exec.moe_route_w16(
            &w.router,
            &sc.d_xn,
            &mut sc.d_logits_r,
            &w.bias.buf,
            hp.routed_scale,
            k,
            &mut sc.d_x16b,
            &mut sc.d_idx,
            &mut sc.d_w,
            &mut sc.d_srow,
            &mut sc.d_sslot,
            &mut sc.d_bexp,
            nbr,
            rows,
            &mut sc.d_route_tickets,
        )?;
    } else {
        exec.matvec_f32_batch(&w.router, &sc.d_xn, &mut sc.d_logits_r, rows)?;
        exec.moe_topk_sigmoid_batch(
            &sc.d_logits_r,
            &w.bias.buf,
            hp.routed_scale,
            hp.n_expert,
            k,
            &mut sc.d_idx,
            &mut sc.d_w,
            rows,
        )?;
        exec.convert_f32_bf16(&sc.d_xn, &mut sc.d_x16b, rows * embd)?;
        exec.moe_align(
            &sc.d_idx,
            &mut sc.d_srow,
            &mut sc.d_sslot,
            &mut sc.d_bexp,
            rows,
            k,
            hp.n_expert,
            nbr,
        )?;
    }
    // the uniq-routed-experts instrument (PADDOCK_MOE_UNIQ) counts these
    // routes too
    if sc.moe_uniq_dev != 0 {
        exec.moe_uniq_hist(&sc.d_idx, rows * k, hp.n_expert, sc.moe_uniq_dev)?;
    }
    exec.nvf4_moe_up_relu2_w16(
        up,
        sh_up,
        &sc.d_srow,
        &sc.d_sslot,
        &sc.d_bexp,
        &sc.d_x16b,
        &mut sc.d_act16,
        k,
        nbr,
        rows,
    )?;
    exec.nvf4_moe_down_part_w16(
        down,
        sh_down,
        &sc.d_srow,
        &sc.d_sslot,
        &sc.d_bexp,
        &sc.d_w,
        &sc.d_act16,
        &mut sc.d_part,
        k,
        nbr,
        rows,
    )?;
    exec.moe_slot_combine(&sc.d_part, &mut sc.d_x, embd, k + 1, rows)?;
    Ok(())
}

/// The GGUF lane's sorted MoE arm on the int8 MMA (slot 743's relu^2 up +
/// the shared mma down) instead of the dp4a sorted pair + quantize. Bitwise:
/// the same exact k32 integer dots, the same scale fold in ascending K, the
/// same relu^2 and per-32 rounding, the same per-(token, slot) partials -
/// only the math units change. The dp4a pair ran ~4 TOPS here: an 11.7K-row
/// append at 247K depth spent 6.6 of its 16.2 s in it (GB10, Claude Code
/// replay), routed and shared expert alike. PADDOCK_NO_NEMO_QMMA pins the
/// dp4a pair for the A/B; a pack without slot 743 keeps it too.
fn nemo_qmma_on(exec: &GpuExecutor) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    exec.has_q8_moe_relu2_mma()
        && !*OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_NEMO_QMMA").is_some())
}

fn w16_band_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_NEMO_W16_BAND").is_none())
}
