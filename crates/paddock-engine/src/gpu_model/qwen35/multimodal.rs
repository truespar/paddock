//! Qwen3.5/3.6 vision attach, image cache, multimodal prefill.

use super::*;
use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use paddock_models::mapped::MappedGguf;

/// The prefix radix's identity of one picture (its image-row keys): its raw
/// bytes plus its dimensions. The picture store keys on a 256-bit digest of
/// the same inputs (`pictures::picture_key`), so the two can only disagree on
/// a 64-bit collision here - and the radix needs rows, not bytes, to hold.
pub(super) fn image_content_hash(rgb: &[u8], w: usize, h: usize) -> u64 {
    hash_bytes(rgb) ^ (w as u64).wrapping_mul(31) ^ (h as u64).wrapping_mul(131)
}

impl GpuQwen35 {
    /// Attach the vision tower from its separate mmproj GGUF (same device).
    pub fn attach_vision(&mut self, mmproj: &MappedGguf) -> Result<(), GpuModelError> {
        let vm = crate::gpu_model::qwen35::vision::VisionModel::load(self.exec.clone(), mmproj)?;
        self.vision = Some(vm);
        if self.batch.is_some() {
            self.cap_picture_at_pass();
        }
        Ok(())
    }

    /// One picture must fit one planned prefill pass (see
    /// `VisionModel::cap_image_tokens`), and the picture store holds what the
    /// plan reserved for that pass (`picture_budget_bytes`); run wherever
    /// either side of the pair - the tower or the elected chunk - arrives
    /// second.
    pub(super) fn cap_picture_at_pass(&mut self) {
        let pass = self.prefill_chunk_rows;
        let budget = self.picture_budget_bytes(pass);
        self.pictures.set_budget(budget);
        if let Some(v) = self.vision.as_mut()
            && let Some((was, now)) = v.cap_image_tokens(pass)
        {
            tracing::info!(
                "qwen35 vision: one picture is capped at {now} tokens, the planned prefill pass \
                 (the checkpoint allows {was}) - its rows prefill together, and a longer \
                 picture would grow the serving scratch past the plan"
            );
        }
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// The attached tower, for the callers that need to ask it something
    /// (the vision input budget).
    pub fn vision_model(&self) -> Option<&super::vision::VisionModel> {
        self.vision.as_ref()
    }

    /// Vision-tower outputs a re-sent image would recompute, served from the
    /// cache instead (test/telemetry hook).
    pub fn image_cache_reuses(&self) -> u64 {
        self.pictures.reused
    }

    /// Bytes the picture store holds, and its budget (test/telemetry hook).
    pub fn picture_store_bytes(&self) -> (u64, u64) {
        (self.pictures.bytes(), self.pictures.budget())
    }

    /// Borrow every picture in the chunk list (in order), encoded or served
    /// from the picture store. The serial lane's whole-prompt prefill takes
    /// them all at once; the batched lanes ask per pass instead.
    pub(super) fn encode_all_images(
        &mut self,
        chunks: &[crate::service::MmChunk],
    ) -> Result<Vec<super::pictures::Picture>, GpuModelError> {
        let want: Vec<(&[u8], usize, usize)> = chunks
            .iter()
            .filter_map(|c| match c {
                crate::service::MmChunk::Image { rgb, w, h } => Some((rgb.as_slice(), *w, *h)),
                _ => None,
            })
            .collect();
        if want.is_empty() {
            return Err(GpuModelError::Unsupported(
                "multimodal prompt has no image".into(),
            ));
        }
        self.encode_pictures(&want)
    }

    /// Exclusive multimodal prefill from interleaved text/image chunks: resets
    /// all sequence state (concurrent slots do not survive), preprocesses +
    /// encodes each image, prefills text+embeddings from position 0, and returns
    /// the last row's logits plus the ROW COUNT it prefilled (image rows
    /// included - what usage must report). Decode then continues
    /// on `forward_one` (the diverged mrope position is carried by the decode
    /// state). Handles any number of interleaved images.
    pub fn forward_multimodal_chunks(
        &mut self,
        chunks: &[crate::service::MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        if self.vision.is_none() {
            return Err(GpuModelError::Unsupported(
                "qwen35 was loaded without an mmproj - configure `mmproj` to enable image input"
                    .into(),
            ));
        }
        // borrowed pictures (store-served or freshly encoded), held apart from
        // `self` so reset/prefill can take &mut self
        let images = self.encode_all_images(chunks)?;

        self.reset();
        for c in chunks {
            if let crate::service::MmChunk::Text(t) = c {
                self.history.extend_from_slice(t);
            }
        }
        let logits = self.prefill_multimodal(chunks, &images)?;
        // the prefill parked its own row count in the decode cursor - read it
        // back rather than recomputing the layout, so the number reported to
        // the client is provably the number that was prefilled
        let rows = self
            .decode
            .as_ref()
            .expect("prefill built the decode state")
            .pos;
        Ok((logits, rows))
    }

    /// Multimodal prefill: text tokens with each image's embeddings injected
    /// between them (any number of interleaved images). Matches b9895 mtmd
    /// semantics per image:
    /// - image rows carry the vision-tower output instead of token embeddings;
    /// - M-RoPE: text rows use the sequential llama-position on all four axes;
    ///   image rows share t = the running cursor and vary h/w over the merged grid;
    /// - the llama-position after each image advances by max(grid_x, grid_y);
    /// - the causal mask compares llama-positions, so all rows of one image
    ///   (equal t) see each other - emulated on our row-indexed KV by setting
    ///   each image row's attention bound to that image block's last row.
    ///   Returns the last row's logits; decode continues incrementally.
    pub fn prefill_multimodal(
        &mut self,
        chunks: &[crate::service::MmChunk],
        images: &[super::pictures::Picture],
    ) -> Result<Vec<f32>, GpuModelError> {
        // token ids (image spans are `0` placeholders, overwritten by the vision
        // embeddings below), the mRoPE grid, and the equal-t image visibility
        // bound - one ordered walk over the chunks, any number of images.
        let grids: Vec<(usize, usize)> = images.iter().map(|v| (v.nx, v.ny)).collect();
        let MmLayout {
            ids,
            mrope,
            bound,
            splices,
            t_len,
            final_mrope_pos,
        } = build_mm_layout(chunks, &grids)?;
        assert!(t_len > 0);
        self.ensure_decode()?;
        assert_eq!(
            self.decode.as_ref().expect("decode state ensured").pos,
            0,
            "prefill requires a fresh sequence"
        );
        assert!(
            t_len <= self.max_ctx,
            "prompt {t_len} exceeds max_ctx {}",
            self.max_ctx
        );
        self.ensure_scratch(t_len)?;

        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        let mut d_tokens = exec.alloc_u32(t_len)?;
        let mut d_rows = exec.alloc_u32(t_len)?;
        let d_slots = exec.alloc_u32(t_len)?;
        let mut d_bound = exec.alloc_u32(t_len)?;
        let mut d_mrope = exec.alloc_u32(4 * t_len)?;
        let rows_host: Vec<u32> = (0..t_len as u32).collect();
        exec.stream.memcpy_htod(&ids, &mut d_tokens).map_err(drv)?;
        exec.stream
            .memcpy_htod(&rows_host, &mut d_rows)
            .map_err(drv)?;
        exec.stream.memcpy_htod(&bound, &mut d_bound).map_err(drv)?;
        exec.stream.memcpy_htod(&mrope, &mut d_mrope).map_err(drv)?;

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
        let r = t_len;

        let sinks = &self.sinks;
        let layers = &self.layers;
        let tok_embd = &self.tok_embd;
        let rot = self.rot.as_ref();
        let bs_f8ffn_p = &self.bs_f8ffn;
        let bs_f8row_p = &self.bs_f8row_ffn;
        // PROJECTION e4m3 planes. This walk had no w8 arm at all -- every image
        // prefill ran its projections on the Q8_0 planes regardless of the
        // floor, which is invisible while both residencies exist and fatal the
        // moment the projection REPLACE drops the Q8 twins. prefix.rs and
        // batch.rs have had this arm since the w8 lane landed; the vision walk
        // was simply never wired, the same way its FFN arm was on the wrong
        // floor until the fix.
        let bs_w8_all = &self.bs_w8;
        let w8_min = super::w8_min_batch();
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");

        embed_any(&exec, tok_embd, &d_tokens, &mut sc.d_x, embd, r, rot)?;
        // inject each image's embeddings over its placeholder rows
        for (k, &(off, n)) in splices.iter().enumerate() {
            exec.copy_region(&images[k].embd, 0, &mut sc.d_x, off * embd, n * embd)?;
        }

        for (li, layer) in layers.iter().enumerate() {
            // attn_norm fused with the qkv/in_qkv quantize (P6k); xn only
            // materializes for Linear mixers (alpha/beta still read it)
            // the w8 arms quantize from xn, so they need it materialized --
            // prefix.rs has the same `|| lw8.is_some()` for exactly this reason.
            // Without it the e4m3 projections would read an unwritten buffer.
            let lw8 = bs_w8_all.get(li).filter(|_| r > w8_min);
            let keep_xn = matches!(&layer.mixer, Mixer::Linear(_)) || lw8.is_some();
            prefill_add_norm_quant_rot(
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
                r,
                eps,
                rot,
            )?;
            match &layer.mixer {
                Mixer::Full(w) => {
                    // fused qkv plane, row-sliced at 0 / nq / nq+nk -- the same
                    // offsets prefix.rs uses (one plane, three consumers)
                    if let Some(l8) = lw8.filter(|l| l.wq.is_some()) {
                        let p8 = l8.wq.as_ref().expect("wq checked in the filter");
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        exec.f8_gemm_w8(
                            p8,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_qg,
                            w.wq.dims()[0],
                            w.wq.dims()[1],
                            r,
                        )?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        exec.f8_gemm_w8(
                            p8,
                            w.wq.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_k,
                            w.wk.dims()[0],
                            w.wk.dims()[1],
                            r,
                        )?;
                        exec.f8_gemm_w8(
                            p8,
                            w.wq.dims()[1] + w.wk.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_v,
                            w.wv.dims()[0],
                            w.wv.dims()[1],
                            r,
                        )?;
                    } else {
                        super::stub_guard(&w.wq, "multimodal.rs vision qkv")?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.wq,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_qg,
                            r,
                        )?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.wk,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_k,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.wv,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
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
                        ds.kv_k[li]
                            .as_mut()
                            .expect("full-attn layer keeps a KV cache"),
                        &d_rows,
                        Some(&d_slots),
                        kv_dim,
                        max_ctx,
                        r,
                        self.kv_dtype,
                    )?;
                    exec.kv_append_batch(
                        &sc.d_v,
                        ds.kv_v[li]
                            .as_mut()
                            .expect("full-attn layer keeps a KV cache"),
                        &d_rows,
                        Some(&d_slots),
                        kv_dim,
                        max_ctx,
                        r,
                        self.kv_dtype,
                    )?;
                    prefill_attn(
                        &exec,
                        &sc.d_qn,
                        ds.kv_k[li]
                            .as_ref()
                            .expect("full-attn layer keeps a KV cache"),
                        ds.kv_v[li]
                            .as_ref()
                            .expect("full-attn layer keeps a KV cache"),
                        sinks,
                        &mut sc.d_attn,
                        &d_bound,
                        &d_slots,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        r,
                        scale,
                        self.kv_dtype,
                        None,
                        Some((&mut sc.d_attn_o, &mut sc.d_attn_ml)),
                    )?;
                    exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, r * q_dim)?;
                    if let Some(l8) = lw8.and_then(|l| l.wo.as_ref()) {
                        exec.quantize_e4m3(
                            &sc.d_attn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * w.wo.dims()[0],
                        )?;
                        exec.f8_gemm_w8(
                            l8,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_proj,
                            w.wo.dims()[0],
                            w.wo.dims()[1],
                            r,
                        )?;
                    } else {
                        super::stub_guard(&w.wo, "multimodal.rs vision wo")?;
                        rotate_opt(rot, &exec, &mut sc.d_attn, q_dim, r)?;
                        prefill_mm_any(
                            &exec,
                            &w.wo,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &sc.d_attn,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
                Mixer::Linear(w) => {
                    // input quantized by the fused attn_norm above (P6k)
                    if let Some(l8) = lw8.and_then(|l| l.in_qkv.as_ref()) {
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        exec.f8_gemm_w8(
                            l8,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_mixed,
                            w.in_qkv.dims()[0],
                            w.in_qkv.dims()[1],
                            r,
                        )?;
                    } else {
                        super::stub_guard(&w.in_qkv, "multimodal.rs vision in_qkv")?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.in_qkv,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_mixed,
                            r,
                        )?;
                    }
                    exec.causal_conv1d_silu(
                        &sc.d_mixed,
                        &w.conv_w.buf,
                        &mut sc.d_conv,
                        r,
                        conv_dim,
                        conv_k,
                    )?;
                    {
                        let win = ds.conv_win[li]
                            .as_mut()
                            .expect("DeltaNet layer keeps a conv window");
                        let km1 = conv_k - 1;
                        if r >= km1 {
                            exec.copy_region(
                                &sc.d_mixed,
                                (r - km1) * conv_dim,
                                win,
                                0,
                                km1 * conv_dim,
                            )?;
                        } else {
                            exec.copy_region(
                                &sc.d_mixed,
                                0,
                                win,
                                (km1 - r) * conv_dim,
                                r * conv_dim,
                            )?;
                        }
                    }
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
                    // alpha/beta on the exact f32 path (P6b decay-numerics rule) -
                    // they feed g, the decay multiplying the whole recurrent state
                    if let Some(ab) = w
                        .ab_f32
                        .as_ref()
                        .filter(|_| r >= ab_f32_min_rows() || w.alpha_w.is_none())
                    {
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
                            r,
                            n_v_heads,
                        )?;
                    } else {
                        if exec.has_q8_0_gemm_repacked_x2() {
                            // fused pair: x staged once for both decay projections
                            // (bit-exact per output vs the two separate calls)
                            exec.q8_0_gemm_repacked_x2(
                                w.alpha_w.as_ref().expect("Q8 alpha (x2 path)"),
                                w.beta_w.as_ref().expect("Q8 beta (x2 path)"),
                                &sc.d_xn,
                                &mut sc.d_a,
                                &mut sc.d_b,
                                r,
                            )?;
                        } else {
                            exec.q8_0_gemm_repacked(
                                w.alpha_w.as_ref().expect("Q8 alpha"),
                                None,
                                &sc.d_xn,
                                &mut sc.d_a,
                                r,
                            )?;
                            exec.q8_0_gemm_repacked(
                                w.beta_w.as_ref().expect("Q8 beta"),
                                None,
                                &sc.d_xn,
                                &mut sc.d_b,
                                r,
                            )?;
                        }
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
                    }
                    prefill_delta_recurrent(
                        &exec,
                        sc,
                        ds.recur[li]
                            .as_mut()
                            .expect("DeltaNet layer keeps recurrent state"),
                        0,
                        r,
                        n_v_heads,
                        state_size,
                        false,
                    )?;
                    // d_xn/d_yq untouched since in_qkv's prefill_quant: reuse
                    if let Some(l8) = lw8.and_then(|l| l.in_qkv.as_ref()) {
                        // fused in_qkv|gate_w plane: gate_w rows start at conv_dim
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        exec.f8_gemm_w8(
                            l8,
                            w.in_qkv.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_z,
                            w.gate_w.dims()[0],
                            w.gate_w.dims()[1],
                            r,
                        )?;
                    } else {
                        super::stub_guard(&w.gate_w, "multimodal.rs vision gate_w")?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.gate_w,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
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
                    if let Some(l8) = lw8.and_then(|l| l.out_w.as_ref()) {
                        exec.quantize_e4m3(
                            &sc.d_core,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * w.out_w.dims()[0],
                        )?;
                        exec.f8_gemm_w8(
                            l8,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_proj,
                            w.out_w.dims()[0],
                            w.out_w.dims()[1],
                            r,
                        )?;
                    } else {
                        super::stub_guard(&w.out_w, "multimodal.rs vision out_w")?;
                        // rotated-basis model: ssm_out reads the regrouped + rotated rows,
                        // landed in d_dattn (free once the gated norm has consumed it)
                        if let Some(rt) = rot {
                            let vd = n_v_heads * state_size;
                            rt.rotate_ssm_out(&exec, &sc.d_core, &mut sc.d_dattn, vd, r)?;
                        }
                        prefill_mm_any(
                            &exec,
                            &w.out_w,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            if rot.is_some() {
                                &sc.d_dattn
                            } else {
                                &sc.d_core
                            },
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
            }
            // residual add + post_norm + gate/up quantize in one pass (P6k);
            // xn skipped - the ffn quantize is its only consumer here
            let mut proj_is_b16 = false;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    // prefill-FFN f8 arm: the W8 prefill class extended to
                    // the FFN, because ~70% of prefill bytes were still
                    // running through int8-mmq. f8_gemm_w8 measures
                    // 1.27-1.85x best-q8 at M >= 512.
                    // Same e4m3 planes the decode lane built.
                    // FFN floor is f8_ffn_pf_min(), not w8_min (the PROJECTION
                    // floor, 64). The R2.3 split rewired prefix.rs and
                    // batch.rs and MISSED this vision arm, so every image
                    // prefill chunk of <= 64 rows fell to the Q8_0 planes --
                    // which the q8 reclaim stubs to 32 bytes. Silent garbage
                    // on exactly the requests that carry an image.
                    let f8f = bs_f8ffn_p.get(li).and_then(|o| o.as_ref()).filter(|_| {
                        r > super::f8_ffn_pf_min()
                            && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
                    });
                    let f8r = bs_f8row_p.get(li).and_then(|o| o.as_ref());
                    prefill_add_norm_quant_rot(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        false,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        f8r.is_some() || f8f.is_some(),
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                        rot,
                    )?;
                    if let Some(p) = f8r {
                        super::ops::ffn_f8row_rows(
                            &exec,
                            p,
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else if let Some([gu8, d8]) = f8f {
                        // fused plane, row-sliced: gate = rows [0,ff), up =
                        // rows [ff,2ff) - byte-identical to the old separate
                        // planes (same repack stream, offset math only)
                        let ffh = gu8.2 / 2;
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * gu8.1)?;
                        // bf16 epilogue pair when the pack ships it: halves
                        // the gate/up store traffic (the rival's cutlass
                        // writes bf16; ours wrote f32) and the fused quant
                        // reads bf16 - else the f32 chain below.
                        static O16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let o16 = *O16.get_or_init(|| {
                            std::env::var_os("PADDOCK_NO_F8W8_TMA").is_none()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        if o16 && exec.has_f8_o16() {
                            exec.f8_gemm_w8_o16(
                                &gu8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_gate,
                                gu8.1,
                                ffh,
                                r,
                            )?;
                            exec.f8_gemm_w8_o16(
                                &gu8.0,
                                ffh,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_up,
                                gu8.1,
                                ffh,
                                r,
                            )?;
                            exec.quantize_e4m3_swiglu_b16(
                                &sc.d_ffn_gate,
                                &sc.d_ffn_up,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                r * d8.1,
                            )?;
                        } else {
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
                            // fused swiglu+e4m3-quant: one pass instead of
                            // swiglu-write + quant-read (286 MB/layer-tick of f32
                            // round-trip at r=2048 - the bf16-activations gap
                            // vs the engines that write bf16 epilogues, closed at
                            // the seam that matters)
                            exec.quantize_e4m3_swiglu(
                                &sc.d_ffn_gate,
                                &sc.d_ffn_up,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                r * d8.1,
                            )?;
                        }
                        if o16 && exec.has_f8_o16() && exec.has_add_b16() {
                            // bf16 down out (halves the last 42 MB/layer-tick
                            // f32 store of the FFN) - the tail add reads bf16
                            exec.f8_gemm_w8_o16(
                                &d8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                d8.1,
                                d8.2,
                                r,
                            )?;
                            proj_is_b16 = true;
                        } else {
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
                        }
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            gate,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_gate,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            up,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_up,
                            r,
                        )?;
                        prefill_ffn_down_rot(
                            &exec,
                            down,
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
                            r,
                            rot,
                        )?;
                    }
                }
                Ffn::Nvf4Dense { gu, down } => {
                    // off the f32 xn (write_xn=true; int8 staging unused) -
                    // the chain takes the W4A4 arm above the row band
                    prefill_add_norm_quant_rot(
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
                        r,
                        eps,
                        rot,
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
                        r,
                    )?;
                }
                Ffn::Moe(w) => {
                    // MoE needs the f32 xn (router + shared expert)
                    prefill_add_norm_quant_rot(
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
                        r,
                        eps,
                        rot,
                    )?;
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
            if proj_is_b16 {
                exec.add_b16(&mut sc.d_x, &sc.d_proj, r * embd)?;
            } else {
                exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
            }
        }

        exec.rmsnorm_batch(&sc.d_x, &self.out_norm.buf, &mut sc.d_h, embd, eps, r)?;
        exec.copy_region(&sc.d_h, (r - 1) * embd, &mut sc.d_xn, 0, embd)?;
        // rotated-basis model: the head reads the rotated row; d_h stays in
        // the model's basis for whoever reads it after the pass
        rotate_opt(rot, &exec, &mut sc.d_xn, embd, 1)?;
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
            super::stub_guard(&self.output, "multimodal.rs vision prefill head")?;
            gemv_any(&exec, &self.output, &sc.d_xn, &mut sc.d_logits)?;
        }
        let logits = exec.to_host(&sc.d_logits)?;
        {
            let ds = self.decode.as_mut().expect("decode state ensured");
            ds.pos = t_len;
            ds.mrope_pos = final_mrope_pos;
        }
        Ok(logits)
    }

    /// Greedy decode from a multimodal prompt (text + one image + text). Prefill
    /// injects the vision embeddings; decode continues on the graph-resident path
    /// with the llama-position offset carried by `mrope_pos`.
    pub fn generate_greedy_mm(
        &mut self,
        before: &[u32],
        image: &crate::gpu_model::qwen35::vision::VisionOutput,
        after: &[u32],
        max_new: usize,
        stop: Option<u32>,
    ) -> Result<Vec<u32>, GpuModelError> {
        assert!(max_new > 0);
        self.reset();
        // wrap the single image as a chunk list; build_mm_layout reads the grid
        // from the encoded `image`, so the chunk's raw bytes are unused here.
        let chunks = vec![
            crate::service::MmChunk::Text(before.to_vec()),
            crate::service::MmChunk::Image {
                rgb: Vec::new(),
                w: 0,
                h: 0,
            },
            crate::service::MmChunk::Text(after.to_vec()),
        ];
        // a parity tool's own tower output: one copy into a picture, off the
        // serving path
        let mut embd = self.exec.alloc(image.embd.len())?;
        self.exec
            .copy_region(&image.embd, 0, &mut embd, 0, image.embd.len())?;
        let picture = std::sync::Arc::new(super::pictures::PictureEmbd {
            embd,
            nx: image.nx,
            ny: image.ny,
        });
        let last = self.prefill_multimodal(&chunks, std::slice::from_ref(&picture))?;
        let exec = self.exec.clone();
        let mut out = Vec::with_capacity(max_new);
        out.push(argmax(&last));
        if Some(out[0]) == stop || max_new == 1 {
            return Ok(out);
        }
        let p = self
            .decode
            .as_ref()
            .expect("prefill built the decode state")
            .pos;
        assert!(p + max_new <= self.max_ctx);
        {
            let ds = self
                .decode
                .as_mut()
                .expect("prefill built the decode state");
            let mp = ds.mrope_pos as u32;
            let e = |x: cudarc::driver::DriverError| crate::gpu::from_driver(x);
            exec.stream
                .memcpy_htod(&[out[0]], &mut ds.d_token)
                .map_err(e)?;
            exec.stream
                .memcpy_htod(&[p as u32], &mut ds.d_pos)
                .map_err(e)?;
            exec.stream
                .memcpy_htod(&[mp; 4], &mut ds.d_mrope)
                .map_err(e)?;
        }
        if self
            .decode
            .as_ref()
            .expect("prefill built the decode state")
            .graph_gen
            .is_none()
        {
            self.capture_graph_gen()?;
        }
        let target = max_new - 1;
        let mut produced = 0usize;
        while produced < target {
            let k = (target - produced).min(GEN_CHUNK);
            {
                let ds = self
                    .decode
                    .as_mut()
                    .expect("prefill built the decode state");
                let e = |x: cudarc::driver::DriverError| crate::gpu::from_driver(x);
                exec.stream
                    .memcpy_htod(&[0u32], &mut ds.d_step)
                    .map_err(e)?;
                let g = ds.graph_gen.as_ref().expect("gen graph captured above");
                for _ in 0..k {
                    g.0.launch()
                        .map_err(|x| GpuError::Driver(format!("gen launch: {x}")))?;
                }
            }
            let ids = exec.to_host_u32(
                &self
                    .decode
                    .as_ref()
                    .expect("prefill built the decode state")
                    .d_out,
            )?;
            for &id in ids.iter().take(k) {
                out.push(id);
                produced += 1;
                if Some(id) == stop {
                    self.decode
                        .as_mut()
                        .expect("prefill built the decode state")
                        .pos = p + produced;
                    return Ok(out);
                }
            }
        }
        self.decode
            .as_mut()
            .expect("prefill built the decode state")
            .pos = p + produced;
        Ok(out)
    }
}
