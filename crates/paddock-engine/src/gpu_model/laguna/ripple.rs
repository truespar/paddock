//! Ripple's attention band on the batched lane, out of the layer walk: q, k
//! and the output gate off the fused `[q | k | g]` BF16 plane (all three read
//! the normed rows), then v off the value-residual blend `lerp(e, x, t)` of
//! the layer-0 input `e` and those rows. The gate is the elementwise
//! `sigmoid(g)` over the attention output (`mul_sigmoid` in the walk); the
//! o projection is the plain BF16 rows.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor, QuantTensor};

use super::batch::BatchScratch;

/// The verify band: a spec round's rows (16 a slot) on the planes 4096 and
/// wider take the TMA weight arm (slot 973). GB10, the 10240 x 5120 q / gate
/// planes: the ladder's cp.async tiles 594 / 745 us at 16 / 64 rows, the TMA
/// arm 493 / 541 (the o plane 607 / 764 against 511 / 553) - three such
/// planes a layer are half a verify round's bytes. The k / v planes (1024
/// out) stay on the ladder, which holds them at the arm's rate.
fn vb_band(out_dim: usize, r: usize) -> bool {
    out_dim >= 4096 && (9..=VB_MAX_ROWS).contains(&r)
}

/// Past this many rows the prefill pair or the ladder's wide tiles take over.
const VB_MAX_ROWS: usize = 128;

/// `y = W x` over `r` rows of a BF16 plane segment: the verify band's TMA
/// arm where it serves, else the ladder.
pub(super) fn rows_vb(
    exec: &GpuExecutor,
    w: &QuantTensor,
    first_row: usize,
    out_dim: usize,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<bool, GpuError> {
    if !vb_band(out_dim, r) {
        return Ok(false);
    }
    exec.bf16_gemm_vb_rows(w, first_row, out_dim, x, y, r)
}

/// One segment of the fused plane over `r` rows: the prefill pair on the
/// bf16 rows when the executor elects it for the segment's width (the rows
/// narrowed once, on the first segment that wants them), the verify band's
/// TMA arm, else the ladder's segment GEMM over the f32 rows. All are the
/// bf16 x bf16, f32-accumulate class.
#[allow(clippy::too_many_arguments)]
fn seg(
    exec: &GpuExecutor,
    w: &QuantTensor,
    first_row: usize,
    out_dim: usize,
    x: &CudaSlice<f32>,
    x16: &mut CudaSlice<half::bf16>,
    narrowed: &mut bool,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuError> {
    if exec.bf16_pf_elect(out_dim, r) {
        if !std::mem::replace(narrowed, true) {
            exec.convert_f32_bf16(x, x16, r * w.dims[0])?;
        }
        return exec.bf16_gemm_pf_rows(w, first_row, out_dim, x16, y, r);
    }
    if rows_vb(exec, w, first_row, out_dim, x, y, r)? {
        return Ok(());
    }
    exec.bf16_gemm_rows(w, first_row, out_dim, x, y, r)
}

/// The o projection over `r` attention rows: the verify band's TMA arm, else
/// the shared BF16 rows path.
pub(super) fn o_rows(
    exec: &GpuExecutor,
    wo: &QuantTensor,
    sc: &mut BatchScratch,
    r: usize,
) -> Result<(), GpuError> {
    if !exec.bf16_pf_elect(wo.dims[1], r)
        && rows_vb(exec, wo, 0, wo.dims[1], &sc.attn, &mut sc.proj, r)?
    {
        return Ok(());
    }
    super::head::bf16_rows(exec, wo, &sc.attn, &mut sc.x16, &mut sc.proj, r)
}

/// q | k | gate into `sc.q`, `sc.k`, `sc.gfull` off `sc.xn`, then the blend
/// into `sc.vin` and v into `sc.v`. `t` = sigmoid(lambda) of the layer.
#[allow(clippy::too_many_arguments)]
pub(super) fn qkgv_rows(
    exec: &GpuExecutor,
    wqkg: &QuantTensor,
    wv: &QuantTensor,
    sc: &mut BatchScratch,
    t: f32,
    q_dim: usize,
    kv_dim: usize,
    r: usize,
) -> Result<(), GpuError> {
    let embd = wqkg.dims[0];
    let (xn, x16) = (&sc.xn, &mut sc.x16);
    let mut narrowed = false;
    let mut run = |first: usize, out: usize, y: &mut CudaSlice<f32>| {
        seg(exec, wqkg, first, out, xn, x16, &mut narrowed, y, r)
    };
    run(0, q_dim, &mut sc.q)?;
    run(q_dim, kv_dim, &mut sc.k)?;
    run(q_dim + kv_dim, q_dim, &mut sc.gfull)?;
    exec.ripple_valres_mix(&mut sc.vin, &sc.emb, &sc.xn, r * embd, t)?;
    super::head::bf16_rows(exec, wv, &sc.vin, &mut sc.x16, &mut sc.v, r)
}
