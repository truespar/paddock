//! The r==1 decode tick's phase-split timing probe (bench harness only -
//! tests/gpu_laguna_profile.rs; the serving path never calls it). Moved out of
//! batch.rs whole; it walks the same scratch the captured tick does.

use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::gemv_any;

use super::*;

impl GpuLaguna {
    /// Phase-split timing probe for the r==1 decode tick (eager, no graph):
    /// runs each op group of the layer walk in an isolated ×`reps` loop with
    /// sync fences and prints the per-tick ms split. Activation VALUES are
    /// garbage in the isolated groups - only shapes/timing matter. Bench
    /// harness only (tests/gpu_laguna_profile.rs); the serving path never
    /// calls this. Requires enable_batch + at least one prefilled token in
    /// slot 0 so position/KV state is sane.
    pub fn profile_batch_tick(&mut self, pos: u32, reps: usize) -> Result<(), GpuModelError> {
        // the probe times the GGUF build's quant projections only
        if self.layers.iter().any(|l| l.proj.quant().is_none()) {
            return Err(GpuModelError::Unsupported(
                "profile: a BF16-projection build".into(),
            ));
        }
        fn q(l: &LagunaLayer) -> &QProj {
            l.proj.quant().expect("guarded above")
        }
        let exec = self.exec.clone();
        let hp_eps = self.hp.eps;
        let (embd, n_kv, hd) = (self.hp.n_embd, self.hp.n_kv_heads, self.hp.head_dim);
        let kv_dim = n_kv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let sections = [self.hp.n_rot as u32 / 2, 0, 0, 0];
        let rope_full = self.hp.rope_full;
        let rope_swa = self.hp.rope_swa;
        let n_rot = self.hp.n_rot;
        let swa_window = self.hp.swa_window;
        let m = self.hp.moe;
        self.upload_rows(&[100], &[pos], &[0])?;

        fn timed(
            me: &mut GpuLaguna,
            name: &str,
            reps: usize,
            f: &mut dyn FnMut(&mut GpuLaguna) -> Result<(), GpuModelError>,
        ) -> Result<f64, GpuModelError> {
            let exec = me.exec.clone();
            exec.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                f(me)?;
            }
            exec.synchronize()?;
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            eprintln!("  {name:<32} {ms:8.3} ms/tick");
            Ok(ms)
        }

        eprintln!("laguna r=1 tick phase split ({reps} reps):");
        let total = timed(self, "FULL step_body", reps, &mut |me| me.step_body(1))?;
        // under a profiler, run only the real tick so the kernel sums aren't
        // polluted by the isolated sub-group loops
        if paddock_models::dev_var_os!("PADDOCK_PROBE_FULL_ONLY").is_some() {
            return Ok(());
        }
        let proj = timed(self, "qkvg GEMVs + norms + rope", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let nh = layer.n_heads;
                let sc = &mut bs.sc;
                exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, hp_eps, 1)?;
                gemv_any(&exec, &q(layer).wq, &sc.xn, &mut sc.q)?;
                gemv_any(&exec, &q(layer).wk, &sc.xn, &mut sc.k)?;
                gemv_any(&exec, &q(layer).wv, &sc.xn, &mut sc.v)?;
                if let Some(g) = &q(layer).g_proj {
                    gemv_any(&exec, g, &sc.xn, &mut sc.gate_h)?;
                }
                exec.rmsnorm_batch(&sc.q, &layer.q_norm.buf, &mut sc.qn, hd, hp_eps, nh)?;
                exec.rmsnorm_batch(&sc.k, &layer.k_norm.buf, &mut sc.kn, hd, hp_eps, n_kv)?;
                if layer.is_swa {
                    exec.rope_yarn_batch(&mut sc.qn, &sc.d_pos, nh, hd, rope_swa, 1)?;
                    exec.rope_yarn_batch(&mut sc.kn, &sc.d_pos, n_kv, hd, rope_swa, 1)?;
                } else if !layer.nope {
                    exec.mrope(
                        &mut sc.qn,
                        &sc.d_mrope,
                        1,
                        nh,
                        hd,
                        n_rot,
                        rope_full,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.kn,
                        &sc.d_mrope,
                        1,
                        n_kv,
                        hd,
                        n_rot,
                        rope_full,
                        sections,
                    )?;
                }
            }
            Ok(())
        })?;
        let attn = timed(self, "append + attend + wo", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            let bps = bs.bps;
            for (li, layer) in me.layers.iter().enumerate() {
                let nh = layer.n_heads;
                let (bt, window) = if layer.is_swa {
                    (&bs.swa_bt, swa_window)
                } else {
                    (&bs.d_bt, 0usize)
                };
                let kvs = &mut bs.kv[li];
                let sc = &mut bs.sc;
                exec.kv_append_batch_paged(
                    &sc.kn,
                    &mut kvs.k,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    kv_dim,
                    1,
                    me.kv_dtype,
                )?;
                exec.kv_append_batch_paged(
                    &sc.v,
                    &mut kvs.v,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    kv_dim,
                    1,
                    me.kv_dtype,
                )?;
                exec.attn_decode_batch_paged(
                    &sc.qn,
                    &kvs.k,
                    &kvs.v,
                    &sc.sinks,
                    &mut sc.attn,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    1,
                    scale,
                    me.kv_dtype,
                )?;
                if q(layer).g_proj.is_some() {
                    exec.mul_softplus_head(&mut sc.attn, &sc.gate_h, nh, hd, 1)?;
                }
                gemv_any(&exec, &q(layer).wo, &sc.attn, &mut sc.proj)?;
            }
            Ok(())
        })?;
        let moe = timed(self, "MoE routed (quant+route+experts)", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, embd)?;
                exec.matvec_f32_batch(&w.router_w, &sc.xn, &mut sc.moe_logits, 1)?;
                m.route(
                    &exec,
                    &sc.moe_logits,
                    &w.probs_bias,
                    &mut sc.moe_idx,
                    &mut sc.moe_w,
                    1,
                )?;
                match (&w.gate_exps, &w.up_exps) {
                    (ExpW::Kq(g), ExpW::Kq(u)) => {
                        let needs =
                            crate::gpu::kq_needs_sums(g.ty) || crate::gpu::kq_needs_sums(u.ty);
                        if needs {
                            exec.q8_sums_strided(&sc.xq, &mut sc.ssums, embd, 1)?;
                        }
                        exec.kquant_moe_gate_up(
                            g,
                            u,
                            &sc.moe_idx,
                            &sc.xq,
                            &sc.xs,
                            needs.then_some(&sc.ssums),
                            &mut sc.moe_fused,
                            m.n_active,
                            1,
                        )?;
                    }
                    _ => unreachable!("XS election is k-quant experts"),
                }
                exec.quantize_q8(
                    &sc.moe_fused,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    m.n_active * m.moe_ff,
                )?;
                match &w.down_exps {
                    ExpW::Kq(d) => {
                        let needs = crate::gpu::kq_needs_sums(d.ty);
                        if needs {
                            exec.q8_sums_strided(&sc.moe_fq, &mut sc.ssums, m.moe_ff, m.n_active)?;
                        }
                        exec.kquant_moe_down(
                            d,
                            &sc.moe_idx,
                            &sc.moe_w,
                            &sc.moe_fq,
                            &sc.moe_fs,
                            needs.then_some(&sc.ssums),
                            &mut sc.proj,
                            m.n_active,
                            1,
                        )?;
                    }
                    ExpW::Q8(_) => unreachable!("XS election is k-quant experts"),
                }
            }
            Ok(())
        })?;
        let shexp = timed(self, "shared expert (GEMV swiglu)", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                gemv_any(&exec, &w.shexp_gate, &sc.xn, &mut sc.sh_gate)?;
                gemv_any(&exec, &w.shexp_up, &sc.xn, &mut sc.sh_up)?;
                exec.swiglu(&mut sc.sh_gate, &sc.sh_up, m.shexp_ff)?;
                gemv_any(&exec, &w.shexp_down, &sc.sh_gate, &mut sc.sh_out)?;
            }
            Ok(())
        })?;
        let head = timed(self, "lm head GEMV", reps, &mut |me| me.head_rows(1))?;
        eprintln!(
            "  {:<32} {:8.3} ms/tick (groups {:.3})",
            "sum vs full",
            total,
            proj + attn + moe + shexp + head
        );

        // fine-grained sub-groups (overlapping with the groups above)
        timed(self, "  · qkvg GEMVs only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let sc = &mut bs.sc;
                gemv_any(&exec, &q(layer).wq, &sc.xn, &mut sc.q)?;
                gemv_any(&exec, &q(layer).wk, &sc.xn, &mut sc.k)?;
                gemv_any(&exec, &q(layer).wv, &sc.xn, &mut sc.v)?;
                if let Some(g) = &q(layer).g_proj {
                    gemv_any(&exec, g, &sc.xn, &mut sc.gate_h)?;
                }
            }
            Ok(())
        })?;
        timed(self, "  · qkg FUSED GEMV (+v)", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let sc = &mut bs.sc;
                if let Some(qkg) = &q(layer).qkg {
                    exec.kquant_gemv(qkg, &sc.xn, &mut sc.q)?;
                    gemv_any(&exec, &q(layer).wv, &sc.xn, &mut sc.v)?;
                }
            }
            Ok(())
        })?;
        timed(self, "  · shexp FUSED (gu+swiglu+down)", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                if let Some(gu) = &w.shexp_gateup {
                    exec.kquant_gemv(gu, &sc.xn, &mut sc.sh_gate)?;
                    exec.swiglu_fused(&sc.sh_gate, &mut sc.sh_up, m.shexp_ff, 1)?;
                    gemv_any(&exec, &w.shexp_down, &sc.sh_up, &mut sc.sh_out)?;
                }
            }
            Ok(())
        })?;
        timed(self, "  · norms (attn+qk) only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let nh = layer.n_heads;
                let sc = &mut bs.sc;
                exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, hp_eps, 1)?;
                exec.rmsnorm_batch(&sc.q, &layer.q_norm.buf, &mut sc.qn, hd, hp_eps, nh)?;
                exec.rmsnorm_batch(&sc.k, &layer.k_norm.buf, &mut sc.kn, hd, hp_eps, n_kv)?;
            }
            Ok(())
        })?;
        timed(self, "  · ropes only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let nh = layer.n_heads;
                let sc = &mut bs.sc;
                if layer.is_swa {
                    exec.rope_yarn_batch(&mut sc.qn, &sc.d_pos, nh, hd, rope_swa, 1)?;
                    exec.rope_yarn_batch(&mut sc.kn, &sc.d_pos, n_kv, hd, rope_swa, 1)?;
                } else if !layer.nope {
                    exec.mrope(
                        &mut sc.qn,
                        &sc.d_mrope,
                        1,
                        nh,
                        hd,
                        n_rot,
                        rope_full,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.kn,
                        &sc.d_mrope,
                        1,
                        n_kv,
                        hd,
                        n_rot,
                        rope_full,
                        sections,
                    )?;
                }
            }
            Ok(())
        })?;
        timed(self, "  · appends only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            let bps = bs.bps;
            for (li, layer) in me.layers.iter().enumerate() {
                let bt = if layer.is_swa { &bs.swa_bt } else { &bs.d_bt };
                let kvs = &mut bs.kv[li];
                let sc = &mut bs.sc;
                exec.kv_append_batch_paged(
                    &sc.kn,
                    &mut kvs.k,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    kv_dim,
                    1,
                    me.kv_dtype,
                )?;
                exec.kv_append_batch_paged(
                    &sc.v,
                    &mut kvs.v,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    kv_dim,
                    1,
                    me.kv_dtype,
                )?;
            }
            Ok(())
        })?;
        timed(self, "  · attn kernels only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            let bps = bs.bps;
            for (li, layer) in me.layers.iter().enumerate() {
                let nh = layer.n_heads;
                let (bt, window) = if layer.is_swa {
                    (&bs.swa_bt, swa_window)
                } else {
                    (&bs.d_bt, 0usize)
                };
                let kvs = &mut bs.kv[li];
                let sc = &mut bs.sc;
                exec.attn_decode_batch_paged(
                    &sc.qn,
                    &kvs.k,
                    &kvs.v,
                    &sc.sinks,
                    &mut sc.attn,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    1,
                    scale,
                    me.kv_dtype,
                )?;
            }
            Ok(())
        })?;
        timed(self, "  · wo GEMVs only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let sc = &mut bs.sc;
                gemv_any(&exec, &q(layer).wo, &sc.attn, &mut sc.proj)?;
            }
            Ok(())
        })?;
        timed(self, "  · moe expert kernels only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                if let (ExpW::Kq(g), ExpW::Kq(u)) = (&w.gate_exps, &w.up_exps) {
                    let needs = crate::gpu::kq_needs_sums(g.ty) || crate::gpu::kq_needs_sums(u.ty);
                    exec.kquant_moe_gate_up(
                        g,
                        u,
                        &sc.moe_idx,
                        &sc.xq,
                        &sc.xs,
                        needs.then_some(&sc.ssums),
                        &mut sc.moe_fused,
                        m.n_active,
                        1,
                    )?;
                }
                if let ExpW::Kq(d) = &w.down_exps {
                    let needs = crate::gpu::kq_needs_sums(d.ty);
                    exec.kquant_moe_down(
                        d,
                        &sc.moe_idx,
                        &sc.moe_w,
                        &sc.moe_fq,
                        &sc.moe_fs,
                        needs.then_some(&sc.ssums),
                        &mut sc.proj,
                        m.n_active,
                        1,
                    )?;
                }
            }
            Ok(())
        })?;
        timed(self, "  · moe route+quant only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, embd)?;
                exec.matvec_f32_batch(&w.router_w, &sc.xn, &mut sc.moe_logits, 1)?;
                m.route(
                    &exec,
                    &sc.moe_logits,
                    &w.probs_bias,
                    &mut sc.moe_idx,
                    &mut sc.moe_w,
                    1,
                )?;
                exec.q8_sums_strided(&sc.xq, &mut sc.ssums, embd, 1)?;
                exec.quantize_q8(
                    &sc.moe_fused,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    m.n_active * m.moe_ff,
                )?;
                exec.q8_sums_strided(&sc.moe_fq, &mut sc.ssums, m.moe_ff, m.n_active)?;
            }
            Ok(())
        })?;
        Ok(())
    }
}
