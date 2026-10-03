//! The backbone's pass over every request of a tick: embed, 32 layers, the
//! final norm. Row-local work (norms, projections, gates, the FFN) runs once
//! over the whole pass; the convolution, the DeltaNet scan and attention run
//! per request, each from its own first row - a request never sees another's
//! rows, and a request alone is the same arithmetic as the request packed.

use cudarc::driver::CudaSlice;

use super::{ClefPass, GpuClef, Mixer, Proj};
use crate::gpu::{DiarEpi, GpuExecutor};
use crate::gpu_model::gpt_oss::GpuModelError;

/// `y = epi(x . W^T)` (`Resid`: `y +=`, the residual stream; `SwiGlu`: the
/// gated MLP's interleaved gate/up rows, silu(gate) * up) over `rows` rows,
/// on the backbone GEMM (slot 733: the activation split two ways in bf16
/// against the exact bf16 weight; a GGUF's Q8_0 on 741, the same split
/// against the exact int8).
///
/// Elected by the gate, class by class: the vendor's own (BF16 activations)
/// drifted past the vendor's BF16 on one fixture's probabilities; the F32
/// class (three parts, slot 722) held every fixture within 6.1e-5 logits,
/// two parts within 5.7e-5 - the long fixtures' error was the iterated
/// rope's either way - for a quarter off the GEMM (31-34 TF/s against
/// 26-27). With the tabled rope: 3.7e-5 logits, 8.3e-6 probabilities,
/// ~1000x inside the vendor.
fn gemm(
    e: &GpuExecutor,
    w: &Proj,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
    epi: DiarEpi,
) -> Result<(), GpuModelError> {
    match w {
        Proj::Bf16(w) => e.clef_gemm(x, w, y, rows, epi)?,
        Proj::Q8(w) => e.clef_gemm_q8(x, w, None, y, rows, epi)?,
    }
    Ok(())
}

/// The mrope positions `[3][rows]` (t, h, w) of a pass, as the reference's
/// `get_rope_index` forms them per request: a text row takes the running
/// position on every axis and advances it by one; an image's tokens (a
/// rows x columns grid of merged windows, row-major) all start at the
/// running position s - t = s, h = s + row, w = s + column - and the
/// running position moves on by max(rows, columns).
fn positions(pass: &ClefPass<'_>) -> Result<Vec<u32>, GpuModelError> {
    let t = pass.ids.len();
    let mut mpos = vec![0u32; 3 * t];
    let mut images = pass.images.iter().peekable();
    for &(s, n) in pass.runs {
        let (mut i, mut cur) = (0usize, 0u32);
        while i < n {
            match images.peek() {
                Some(im) if im.row == s + i => {
                    let (gh, gw) = im.grid;
                    if gh == 0 || gw == 0 || i + gh * gw > n {
                        return Err(GpuModelError::Unsupported(
                            "Clef pass: an image outside its request".into(),
                        ));
                    }
                    for j in 0..gh * gw {
                        let r = s + i + j;
                        mpos[r] = cur;
                        mpos[t + r] = cur + (j / gw) as u32;
                        mpos[2 * t + r] = cur + (j % gw) as u32;
                    }
                    i += gh * gw;
                    cur += gh.max(gw) as u32;
                    images.next();
                }
                _ => {
                    for a in 0..3 {
                        mpos[a * t + s + i] = cur;
                    }
                    i += 1;
                    cur += 1;
                }
            }
        }
    }
    if images.next().is_some() {
        return Err(GpuModelError::Unsupported(
            "Clef pass: an image outside every request".into(),
        ));
    }
    Ok(mpos)
}

impl GpuClef {
    /// Run the backbone over `pass`; its final hidden rows (the reference's
    /// `last_hidden_state`, after the final norm) are left in the
    /// workspace for the head (see [`Self::hidden`]).
    pub fn backbone(&mut self, pass: &ClefPass<'_>) -> Result<(), GpuModelError> {
        let t = pass.ids.len();
        let mut next = 0usize;
        for &(s, n) in pass.runs {
            if s != next || n == 0 {
                return Err(GpuModelError::Unsupported(
                    "Clef pass: runs must cover the ids back to back".into(),
                ));
            }
            next = s + n;
        }
        if next != t || t == 0 || t > self.max_rows {
            return Err(GpuModelError::Unsupported(format!(
                "Clef pass: {t} rows over {} runs (at most {} rows a pass)",
                pass.runs.len(),
                self.max_rows
            )));
        }
        let e = self.exec.clone();
        let c = &self.cfg;
        let ws = &mut self.ws;
        let (h, eps) = (c.hidden, c.eps);
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let (nk, hv, sd) = (c.gdn_k_heads, c.gdn_v_heads, c.gdn_k_dim);
        let scale = 1.0 / (hd as f32).sqrt();
        // the sequential scan below the size the chunked one wins at (the
        // qwen35 lane's measured crossover)
        let chunk_min = if e.sm_count() >= 128 { 384 } else { 128 };
        let state_elems = hv * sd * sd;

        let d_mpos = e.to_device_u32(&positions(pass)?)?;
        let d_ids = e.to_device_u32(pass.ids)?;
        let masks = self.rope_masks;

        // BF16 rows or Q8_0 rows as stored, widened exactly either way
        e.embed_gather_plane(&self.embed, &d_ids, &mut ws.x, h, t, 1.0)?;
        // the images' merged rows (encoded into xn, back to back) over their
        // <|image_pad|> rows - the reference's masked_scatter
        let mut staged = 0usize;
        for im in pass.images {
            let n = im.grid.0 * im.grid.1;
            e.copy_region(&ws.xn, staged * h, &mut ws.x, im.row * h, n * h)?;
            staged += n;
        }
        for layer in &self.layers {
            e.rmsnorm_batch(&ws.x, &layer.in_norm, &mut ws.xn, h, eps, t)?;
            match &layer.mixer {
                Mixer::Gdn(g) => {
                    gemm(&e, &g.qkv, &ws.xn, &mut ws.wide, t, DiarEpi::Store)?;
                    gemm(&e, &g.z, &ws.xn, &mut ws.z, t, DiarEpi::Store)?;
                    gemm(&e, &g.ab, &ws.xn, &mut ws.ab, t, DiarEpi::Store)?;
                    e.delta_gate_ab(&ws.ab, &g.ssm_a, &g.dt_bias, &mut ws.g, &mut ws.beta, t, hv)?;
                    for &(s, n) in pass.runs {
                        // conv over the run alone (rows before it read as
                        // zero), SiLU, split, key heads broadcast, q/k
                        // L2-normed
                        e.causal_conv1d_silu_qkv_at(
                            &ws.wide, &g.conv, &mut ws.dq, &mut ws.dk, &mut ws.dv, s, s, n, nk, hv,
                            sd, c.gdn_conv,
                        )?;
                        e.zero_region(&mut ws.state, 0, state_elems)?;
                        if n >= chunk_min {
                            e.gated_delta_chunked_at(
                                &ws.dq,
                                &ws.dk,
                                &ws.dv,
                                &ws.g,
                                &ws.beta,
                                &mut ws.state,
                                0,
                                &mut ws.out,
                                s,
                                &mut ws.dnc_dw,
                                &mut ws.dnc_du,
                                &mut ws.dnc_aqk,
                                &mut ws.dnc_cg,
                                n,
                                hv,
                                sd,
                            )?;
                        } else {
                            e.gated_delta_recurrent_v2_at(
                                &ws.dq,
                                &ws.dk,
                                &ws.dv,
                                &ws.g,
                                &ws.beta,
                                &mut ws.state,
                                0,
                                &mut ws.out,
                                s,
                                n,
                                hv,
                                sd,
                            )?;
                        }
                    }
                    e.gated_rmsnorm(&ws.out, &ws.z, &g.norm, &mut ws.core, t * hv, sd, eps)?;
                    gemm(&e, &g.out, &ws.core, &mut ws.x, t, DiarEpi::Resid)?;
                }
                Mixer::Attn(a) => {
                    gemm(&e, &a.q, &ws.xn, &mut ws.wide, t, DiarEpi::Store)?;
                    e.split_qg(&ws.wide, &mut ws.dq, &mut ws.dk, t, nh, hd)?;
                    gemm(&e, &a.k, &ws.xn, &mut ws.k, t, DiarEpi::Store)?;
                    gemm(&e, &a.v, &ws.xn, &mut ws.v, t, DiarEpi::Store)?;
                    e.rmsnorm_batch(&ws.dq, &a.q_norm, &mut ws.dv, hd, eps, t * nh)?;
                    e.rmsnorm_batch(&ws.k, &a.k_norm, &mut ws.kn, hd, eps, t * nkv)?;
                    e.clef_rope(&mut ws.dv, nh, t, &d_mpos, &self.rope, masks)?;
                    e.clef_rope(&mut ws.kn, nkv, t, &d_mpos, &self.rope, masks)?;
                    for &(s, n) in pass.runs {
                        e.clef_attn_tc(&ws.dv, &ws.kn, &ws.v, &mut ws.out, s, n, (nh, nkv), scale)?;
                    }
                    e.mul_sigmoid(&mut ws.out, &ws.dk, t * nh * hd)?;
                    gemm(&e, &a.o, &ws.out, &mut ws.x, t, DiarEpi::Resid)?;
                }
            }
            e.rmsnorm_batch(&ws.x, &layer.post_norm, &mut ws.xn, h, eps, t)?;
            gemm(
                &e,
                &layer.gate_up,
                &ws.xn,
                &mut ws.ffn_g,
                t,
                DiarEpi::SwiGlu,
            )?;
            gemm(&e, &layer.down, &ws.ffn_g, &mut ws.x, t, DiarEpi::Resid)?;
        }
        e.rmsnorm_batch(&ws.x, &self.final_norm, &mut ws.xn, h, eps, t)?;
        Ok(())
    }

    /// The last pass's final hidden rows `[rows][hidden]` (F32, device).
    pub fn hidden(&self) -> &cudarc::driver::CudaSlice<f32> {
        &self.ws.xn
    }

    /// The first `rows` final hidden rows on the host (tests, diagnostics).
    pub fn hidden_to_host(&self, rows: usize) -> Result<Vec<f32>, GpuModelError> {
        Ok(self.exec.to_host_len(&self.ws.xn, rows * self.cfg.hidden)?)
    }
}
