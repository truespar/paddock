//! The laguna body's ends - the embedding gather, the final norm + LM head -
//! per weight class (GGUF Q8_0 / k-quant, or the safetensors builds' BF16),
//! and the BF16 projection bands the NVFP4 build's attention and shared
//! expert ride. Split out of
//! batch.rs whole; the batched scratch is the same.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{gemv_any, mmq_kq_pre, mmq_pre};

use super::*;

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

/// q|k|v off the fused BF16 `[embd, q + 2 kv]` plane into the three row
/// planes: the prefill pair (slot 757, over `x16` - the rows narrowed to
/// bf16 once) when the executor elects it, the fused multi-row GEMV (slot
/// 773) at 2..=8 rows, else one tile launch above r 1 (`bf16_qkv_gemm`, per
/// row identical to the segment GEMMs), the segment GEMVs at r 1. Every arm
/// is the same per-row class; only the decode ladder's K-split regroups.
/// `x16_ready`: `x16` already holds the rows as bf16 (the fused sandwich norm
/// wrote them, and no f32 `x`).
#[allow(clippy::too_many_arguments)]
pub(super) fn bf16_qkv(
    exec: &GpuExecutor,
    w: &QuantTensor,
    x: &CudaSlice<f32>,
    x16: Option<&mut CudaSlice<half::bf16>>,
    x16_ready: bool,
    q: &mut CudaSlice<f32>,
    k: &mut CudaSlice<f32>,
    v: &mut CudaSlice<f32>,
    q_dim: usize,
    kv_dim: usize,
    r: usize,
) -> Result<(), GpuError> {
    let pf = exec.bf16_pf_elect(q_dim + 2 * kv_dim, r);
    // x16_ready: the fused norm wrote the bf16 rows and NOT `x` - only the
    // prefill pair may read them
    if x16_ready && !pf {
        return Err(GpuError::Unsupported(
            "bf16_qkv: bf16 rows staged for a width the prefill pair declines".into(),
        ));
    }
    if let Some(x16) = x16
        && pf
    {
        if !x16_ready {
            exec.convert_f32_bf16(x, x16, r * w.dims[0])?;
        }
        return exec.bf16_qkv_gemm_pf(w, x16, q, k, v, q_dim, kv_dim, r);
    }
    // the decode band: one multi-row GEMV over the fused rows (slot 773) -
    // the tile's BN=32 tier stages 32 activation columns for 2..8 rows
    if exec.bf16_qkv_gemv_mr(w, x, q, k, v, q_dim, kv_dim, r)? {
        return Ok(());
    }
    if r > 1 && exec.has_bf16_qkv_gemm() {
        return exec.bf16_qkv_gemm(w, x, q, k, v, q_dim, kv_dim, r);
    }
    exec.bf16_gemm_rows(w, 0, q_dim, x, q, r)?;
    exec.bf16_gemm_rows(w, q_dim, kv_dim, x, k, r)?;
    exec.bf16_gemm_rows(w, q_dim + kv_dim, kv_dim, x, v, r)
}

/// [`bf16_rows`] for the shared expert's 512-wide planes: in the 2..=8-row
/// band the multi-row GEMV's out/8-CTA grid sits latency-bound on them (64
/// CTAs; GB10 at 4 rows 23-25 us a 2.6 MB plane), so they take the narrow
/// tile (`bf16_gemm_tile`, 6-8 us in the same bench).
pub(super) fn bf16_rows_narrow(
    exec: &GpuExecutor,
    w: &QuantTensor,
    x: &CudaSlice<f32>,
    x16: &mut CudaSlice<half::bf16>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuError> {
    if (2..=8).contains(&r) {
        return exec.bf16_gemm_tile(w, None, x, y, w.dims[1], r);
    }
    bf16_rows(exec, w, x, x16, y, r)
}

/// `y = W x` over `r` rows of a BF16 plane: the prefill pair (slot 756) on
/// `x16` when elected, else the ladder (`bf16_gemm`).
pub(super) fn bf16_rows(
    exec: &GpuExecutor,
    w: &QuantTensor,
    x: &CudaSlice<f32>,
    x16: &mut CudaSlice<half::bf16>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuError> {
    if exec.bf16_pf_elect(w.dims[1], r) {
        exec.convert_f32_bf16(x, x16, r * w.dims[0])?;
        return exec.bf16_gemm_pf(w, None, x16, y, r);
    }
    exec.bf16_gemm(w, None, x, y, r)
}

impl GpuLaguna {
    pub(crate) fn embed_rows(&mut self, r: usize) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        match &self.tok_embd {
            TokEmbd::Q8(t) => {
                self.exec
                    .embed_gather_batch_q8(t, &sc.d_toks, &mut sc.x, self.hp.n_embd, r)?
            }
            TokEmbd::Kq(t) => {
                self.exec
                    .kquant_gather(t, &sc.d_toks, &mut sc.x, self.hp.n_embd, r)?
            }
            TokEmbd::Bf16(t) => {
                self.exec
                    .embed_gather_bf16(t, &sc.d_toks, &mut sc.x, self.hp.n_embd, r, 1.0)?
            }
        }
        Ok(())
    }

    /// Final norm + LM head over the first `rows` residual rows into
    /// head_logits [rows, vocab].
    pub(super) fn head_rows(&mut self, rows: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = &self.hp;
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        exec.rmsnorm_batch(
            &sc.x,
            &self.output_norm.buf,
            &mut sc.xn,
            hp.n_embd,
            hp.eps,
            rows,
        )?;
        let lm = match &self.lm_head {
            Head::Bf16(w) => {
                exec.bf16_gemm(w, None, &sc.xn, &mut sc.head_logits, rows)?;
                return Ok(());
            }
            Head::Quant(q) => q,
        };
        if rows == 1 {
            // the r1 head rides the W4A8 GEMV too (Q6_K head at
            // out=vocab is the single biggest r1 gemv - 293 us, against the
            // W4A8 class's ~630 GB/s byte rate). Same latch as the
            // layer walk; exact-f32 GEMV pinned via PADDOCK_KQ_EXACT_GEMV.
            if exec.has_kquant_gemv_w4a8()
                && paddock_models::dev_var_os!("PADDOCK_KQ_EXACT_GEMV").is_none()
                && let QuantW::Kq(k) = lm
            {
                exec.quantize_q8_sums(&sc.xn, &mut sc.xq, &mut sc.xs, &mut sc.ssums, hp.n_embd)?;
                let needs = crate::gpu::kq_needs_sums(k.ty);
                exec.kquant_gemv_w4a8(
                    k,
                    &sc.xq,
                    &sc.xs,
                    needs.then_some(&sc.ssums),
                    &mut sc.head_logits,
                )?;
                return Ok(());
            }
            return gemv_any(&exec, lm, &sc.xn, &mut sc.head_logits);
        }
        exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, rows * hp.n_embd)?;
        // vocab-wide out exceeds the mma partial plane -> dp4a rungs
        match lm {
            QuantW::Kq(k) => mmq_kq_pre(
                &exec,
                k,
                &sc.xq,
                &sc.xs,
                &mut sc.ssums,
                &mut sc.part,
                &mut sc.head_logits,
                rows,
            )?,
            QuantW::Q8(q) => mmq_pre(
                &exec,
                q,
                &sc.xq,
                &sc.xs,
                &mut sc.part,
                &mut sc.head_logits,
                rows,
            )?,
        }
        Ok(())
    }

    /// Prefill tail: head over residual row `row` (of the last chunk),
    /// returning that one vocab row on the host.
    pub(super) fn head_row(&mut self, row: usize, _rows: usize) -> Result<Vec<f32>, GpuModelError> {
        let exec = self.exec.clone();
        let (n_embd, n_vocab) = (self.hp.n_embd, self.hp.n_vocab);
        // norm+head the whole tail up to `row` would waste vocab GEMM rows;
        // stage the single residual row at row 0 of a fresh pass instead
        // (bounced through proj - src and dst live in the same buffer)
        if row > 0 {
            let bs = self.batch.as_mut().expect("batch enabled");
            let sc = &mut bs.sc;
            let src =
                sc.x.try_slice(row * n_embd..(row + 1) * n_embd)
                    .ok_or_else(|| GpuError::Driver("x row slice".into()))?;
            let mut dst = sc
                .proj
                .try_slice_mut(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj row slice".into()))?;
            exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
            let mut xd =
                sc.x.try_slice_mut(0..n_embd)
                    .ok_or_else(|| GpuError::Driver("x dst slice".into()))?;
            let ps = sc
                .proj
                .try_slice(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj src slice".into()))?;
            exec.stream.memcpy_dtod(&ps, &mut xd).map_err(drv)?;
        }
        self.head_rows(1)?;
        let bs = self.batch.as_mut().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..n_vocab)
            .ok_or_else(|| GpuError::Driver("head row slice".into()))?;
        let out = exec.stream.clone_dtoh(&v).map_err(drv)?;
        Ok(out)
    }
}
