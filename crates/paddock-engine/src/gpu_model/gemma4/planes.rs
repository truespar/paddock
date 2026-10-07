//! The family's weight-plane classes and the k-quant rungs behind them.
//!
//! gemma4 held every body plane as repacked Q8_0 until the first 4-bit file
//! (unsloth's diffusiongemma-26B-A4B Q4_K_M, 2026-09-24) arrived. That file
//! is mixed, as the quantization strategy says a UD file is: Q4_K on the
//! attention projections, the shared gate/up and the fused expert gate/up,
//! Q6_K on `token_embd` and half the `attn_v` planes, and - on every row a
//! 256-block format cannot encode (the 2112-wide shared down, the 704-wide
//! expert down) - Q5_0 or Q8_0. So the seam is the TENSOR: `Plane` carries
//! the class the file shipped, and every consumer dispatches on it. The Q8
//! and bf16 arms are the ones the family always had; the `Kq` arm rides the
//! W4A8 streams the qwen families serve k-quant files on (`gpu/kquant.rs`):
//!
//! - r == 1: `quantize_q8_sums` once per shared input, then the fused W4A8
//!   GEMV (`kquant_gemv_w4a8`, or `_multi` for q|k|v and gate|up, which read
//!   the staged row once for all segments);
//! - 2..=5 rows: the multi-column GEMV where it wins (`nc_fits`);
//! - 3..=64 rows: the K-split int8 mma (`kquant_gemm_mma_ks`) when the
//!   partials plane holds it, else the batch-invariant dp4a z-tile;
//! - more: the mmq-layout activations + `kquant_gemm_w4a8_pipe2`, the
//!   128x128 int8-mma tile with the cp.async ring.
//!
//! The rungs are qwen35's (`qwen35/ops.rs::mmq_kq_pre` / `kq_mm_pre`), read
//! rather than shared: gemma4 stages into its own `pf_*` planes and its
//! prefill lanes already own the row split, so what this module adds is the
//! dispatch, not a second ladder.
//!
//! The flat 32-weight types (Q5_0 here) reach the same entry points through
//! the pack's i-quant dense lanes; the loader pads their rows to whole
//! super-blocks (2112 -> 2304) so the tile GEMM applies at prefill widths.

use cudarc::driver::CudaSlice;

use super::{ExpertPlanes, Hparams, Plane, Scratch};
use crate::gpu::{GluAct, GpuError, GpuExecutor, QuantTensor, RepackedKQ, RepackedQ8};

/// The int8 staging a k-quant W4A8 launch reads for ONE activation row: the
/// quantized row, its per-32 scales and its per-16 sums (the Q4_K/Q5_K mu
/// operand; unread for the other types).
pub(crate) struct KqStage<'a> {
    pub xq: &'a mut CudaSlice<i8>,
    pub xs: &'a mut CudaSlice<f32>,
    pub ssums: &'a mut CudaSlice<f32>,
}

/// The staging the r-row rungs read: the strided int8 pair + per-16 sums
/// (2..=64 rows, the `pf_xq`/`pf_xs` planes), the mmq tile + its per-32
/// sums (> 64 rows), and the K-split partials plane.
pub(crate) struct KqRows<'a> {
    pub xq: &'a mut CudaSlice<i8>,
    pub xs: &'a mut CudaSlice<f32>,
    pub ssums: &'a mut CudaSlice<f32>,
    pub yq: &'a mut CudaSlice<u8>,
    pub xsums: &'a mut CudaSlice<f32>,
    pub part: &'a mut CudaSlice<f32>,
}

/// The decode-row staging off a `Scratch` - disjoint fields, so it sits next
/// to `&sc.normed` / `&mut sc.q` in the same call without a borrow clash.
macro_rules! kq_stage {
    ($sc:expr) => {
        &mut $crate::gpu_model::gemma4::planes::KqStage {
            xq: &mut $sc.kq_xq,
            xs: &mut $sc.kq_xs,
            ssums: &mut $sc.kq_ssums,
        }
    };
}
pub(crate) use kq_stage;

/// The r-row staging off a `Scratch` (same borrow property).
macro_rules! kq_rows {
    ($sc:expr) => {
        &mut $crate::gpu_model::gemma4::planes::KqRows {
            xq: &mut $sc.pf_xq,
            xs: &mut $sc.pf_xs,
            ssums: &mut $sc.pf_ssums,
            yq: &mut $sc.pf_yq,
            xsums: &mut $sc.pf_xsums,
            part: &mut $sc.pf_skfix,
        }
    };
}
pub(crate) use kq_rows;

impl Plane {
    pub(crate) fn dims(&self) -> &[usize] {
        match self {
            Plane::Q8(w) => &w.dims,
            Plane::Bf16(w) => &w.dims,
            Plane::Kq(w) => &w.dims,
        }
    }

    /// The Q8 plane, or None when this tensor is not in that class. Arms that
    /// can only consume Q8 must route around on None rather than assume.
    pub(crate) fn q8(&self) -> Option<&RepackedQ8> {
        match self {
            Plane::Q8(w) => Some(w),
            Plane::Bf16(_) | Plane::Kq(_) => None,
        }
    }

    /// The k-quant plane, or None.
    pub(crate) fn kq(&self) -> Option<&RepackedKQ> {
        match self {
            Plane::Kq(w) => Some(w),
            Plane::Q8(_) | Plane::Bf16(_) => None,
        }
    }

    /// Has the Q8 plane been stubbed by the reclaim pass (bytes freed, dims
    /// kept)? A bf16 or k-quant plane is never stubbed - nothing else can
    /// serve it.
    pub(crate) fn is_stub(&self) -> bool {
        matches!(self, Plane::Q8(w) if w.data.len() == 48)
    }

    /// The Q8 data stream's length - what the stub tests read (32 bytes =
    /// an F8R/F8A stub, 48 = a reclaim stub). A bf16 or k-quant plane is
    /// never a stub and answers `usize::MAX`, so every `<= N` test stays
    /// false for it.
    pub(crate) fn q8_len(&self) -> usize {
        match self {
            Plane::Q8(w) => w.data.len(),
            Plane::Bf16(_) | Plane::Kq(_) => usize::MAX,
        }
    }

    /// Resident device bytes of the plane (both streams).
    pub(crate) fn bytes(&self) -> u64 {
        match self {
            Plane::Q8(w) => (w.data.len() + w.scale.len()) as u64,
            Plane::Bf16(w) => w.bytes.len() as u64,
            Plane::Kq(w) => (w.data.len() + w.scales.len()) as u64,
        }
    }

    /// `y = W x`, r == 1. Q8 and bf16 stay exact; a k-quant plane stages the
    /// row into `st` and takes the W4A8 GEMV.
    pub(crate) fn gemv(
        &self,
        exec: &GpuExecutor,
        st: &mut KqStage<'_>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        match self {
            Plane::Q8(w) => exec.q8_0_gemv_repacked(w, None, x, y),
            Plane::Bf16(w) => exec.bf16_gemv(w, None, x, y),
            Plane::Kq(w) => {
                kq_stage_x(exec, st, x, w.dims[0])?;
                kq_gemv(exec, w, st, y)
            }
        }
    }

    /// `y = W x` landing at output-row offset `off` - the fused `[q|k|v]`
    /// decode row's writer.
    pub(crate) fn gemv_at(
        &self,
        exec: &GpuExecutor,
        st: &mut KqStage<'_>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        off: usize,
    ) -> Result<(), GpuError> {
        match self {
            Plane::Q8(w) => exec.q8_0_gemv_repacked_at(w, x, y, off),
            Plane::Bf16(w) => exec.bf16_gemv_at(w, x, y, off),
            Plane::Kq(w) => {
                kq_stage_x(exec, st, x, w.dims[0])?;
                kq_gemv_at(exec, w, st, y, off)
            }
        }
    }

    /// [`Plane::gemv_at`] over a stage the caller already filled for this
    /// input: the k-quant arm reads `st` without re-quantizing, the others
    /// ignore it. The q|k|v concat row at r == 1 stages once and lands all
    /// three segments through this.
    pub(crate) fn gemv_at_pre(
        &self,
        exec: &GpuExecutor,
        st: &KqStage<'_>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        off: usize,
    ) -> Result<(), GpuError> {
        match self {
            Plane::Q8(w) => exec.q8_0_gemv_repacked_at(w, x, y, off),
            Plane::Bf16(w) => exec.bf16_gemv_at(w, x, y, off),
            Plane::Kq(w) => kq_gemv_at(exec, w, st, y, off),
        }
    }

    /// The exact-class GEMV for a family that never loads a k-quant plane
    /// (PaddleOCR-VL borrows this enum for its Q8_0 / bf16 file and stages no
    /// int8 row): Q8 and bf16 as [`Plane::gemv`], a k-quant plane refused
    /// by name rather than served through staging the caller does not own.
    pub(crate) fn gemv_exact(
        &self,
        exec: &GpuExecutor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        match self {
            Plane::Q8(w) => exec.q8_0_gemv_repacked(w, None, x, y),
            Plane::Bf16(w) => exec.bf16_gemv(w, None, x, y),
            Plane::Kq(_) => Err(GpuError::Unsupported(
                "k-quant plane on a lane without W4A8 staging".into(),
            )),
        }
    }

    /// The exact-class GEMM twin of [`Plane::gemv_exact`].
    pub(crate) fn gemm_exact(
        &self,
        exec: &GpuExecutor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        r: usize,
    ) -> Result<(), GpuError> {
        match self {
            Plane::Q8(w) => exec.q8_0_gemm_repacked(w, None, x, y, r),
            Plane::Bf16(w) => exec.bf16_gemm(w, None, x, y, r),
            Plane::Kq(_) => Err(GpuError::Unsupported(
                "k-quant plane on a lane without W4A8 staging".into(),
            )),
        }
    }

    /// `y = W x` over `r` activation rows (`x` `[r, in]`, `y` `[r, out]`).
    /// The Q8 arm is the plain repacked GEMM - the callers' tuned Q8 ladders
    /// sit at the call sites; this is the class-generic form.
    pub(crate) fn gemm(
        &self,
        exec: &GpuExecutor,
        rows: &mut KqRows<'_>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        r: usize,
    ) -> Result<(), GpuError> {
        match self {
            Plane::Q8(w) => exec.q8_0_gemm_repacked(w, None, x, y, r),
            Plane::Bf16(w) => exec.bf16_gemm(w, None, x, y, r),
            Plane::Kq(w) => kq_mm_rows(exec, w, rows, x, y, r),
        }
    }
}

/// Quantize one f32 row of `n` into the stage (int8 + per-32 scales + per-16
/// sums, one launch). Callers stage once per shared input and run every
/// plane that reads it off the same stage (the quantize-dedupe rule).
pub(crate) fn kq_stage_x(
    exec: &GpuExecutor,
    st: &mut KqStage<'_>,
    x: &CudaSlice<f32>,
    n: usize,
) -> Result<(), GpuError> {
    exec.quantize_q8_sums(x, st.xq, st.xs, st.ssums, n)
}

/// The staged W4A8 GEMV: `y = W x` off `st`.
pub(crate) fn kq_gemv(
    exec: &GpuExecutor,
    w: &RepackedKQ,
    st: &KqStage<'_>,
    y: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    let needs = crate::gpu::kq_needs_sums(w.ty);
    exec.kquant_gemv_w4a8(w, st.xq, st.xs, needs.then_some(&*st.ssums), y)
}

/// The staged W4A8 GEMV writing from `y[off]` - the whole plane at an
/// output offset (the k-quant family only: a flat 32-weight plane has no
/// row-offset form, and the loader never puts one on a q/k/v seat, whose
/// width is the 256-aligned n_embd).
pub(crate) fn kq_gemv_at(
    exec: &GpuExecutor,
    w: &RepackedKQ,
    st: &KqStage<'_>,
    y: &mut CudaSlice<f32>,
    off: usize,
) -> Result<(), GpuError> {
    let needs = crate::gpu::kq_needs_sums(w.ty);
    exec.kquant_gemv_w4a8_rows(
        w,
        0,
        w.dims[1],
        st.xq,
        st.xs,
        needs.then_some(&*st.ssums),
        y,
        off,
    )
}

/// One launch over 2-3 same-input k-quant planes off one stage (the decode
/// q|k|v and gate|up merges); segments may mix types (Q4_K q/k beside a
/// Q6_K v is exactly the Q4_K_M pairing).
pub(crate) fn kq_gemv_multi(
    exec: &GpuExecutor,
    segs: &mut [(&RepackedKQ, &mut CudaSlice<f32>)],
    st: &KqStage<'_>,
) -> Result<(), GpuError> {
    if exec.has_kquant_gemv_w4a8_multi() {
        return exec.kquant_gemv_w4a8_multi(segs, st.xq, st.xs, st.ssums);
    }
    for (w, y) in segs.iter_mut() {
        kq_gemv(exec, w, st, y)?;
    }
    Ok(())
}

/// The r-row W4A8 ladder over one k-quant plane: `y [r, out] = W x [r, in]`.
/// See the module note for the rungs. `r == 1` stages into the same planes
/// (the strided pair is the single row's layout too).
pub(crate) fn kq_mm_rows(
    exec: &GpuExecutor,
    w: &RepackedKQ,
    rows: &mut KqRows<'_>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuError> {
    let needs = crate::gpu::kq_needs_sums(w.ty);
    let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
    if r == 1 {
        exec.quantize_q8_sums(x, rows.xq, rows.xs, rows.ssums, in_dim)?;
        return exec.kquant_gemv_w4a8(w, rows.xq, rows.xs, needs.then_some(&*rows.ssums), y);
    }
    if r > 64 {
        // The tile walks whole super-blocks: the loader pads every k-quant
        // plane's in dim to 256 (the flat types' rows included), so a
        // width that is not is a loader defect, said so rather than served
        // through a per-token dp4a walk the strided planes cannot even hold
        // (they are 192 rows deep).
        if !in_dim.is_multiple_of(256) {
            return Err(GpuError::Unsupported(format!(
                "k-quant plane [{in_dim} x {out_dim}] at {r} rows: in dim not tile-aligned"
            )));
        }
        exec.quantize_q8_mmq(x, rows.yq, in_dim, r)?;
        if needs {
            exec.mmq_sums(rows.yq, rows.xsums, in_dim, r)?;
        }
        let xs = needs.then_some(&*rows.xsums);
        return exec.kquant_gemm_w4a8_tile(w, rows.yq, xs, y, r);
    }
    exec.quantize_q8(x, rows.xq, rows.xs, r * in_dim)?;
    if needs {
        exec.q8_sums_strided(rows.xq, rows.ssums, in_dim, r)?;
    }
    let ss = needs.then_some(&*rows.ssums);
    if exec.has_kquant_gemv_w4a8_nc() && GpuExecutor::kquant_gemv_w4a8_nc_fits(w, r) {
        return exec.kquant_gemv_w4a8_nc(w, rows.xq, rows.xs, ss, y, r);
    }
    // the ks fixup plane holds the K-split partials for every layer shape;
    // the vocab head does not fit it (8 x 262144 x 64 f32) and takes the
    // z-tile, which needs no partials
    if r >= 3 && exec.has_kquant_mma_ks() && rows.part.len() >= 8 * 64 * out_dim {
        exec.kquant_gemm_mma_ks(w, rows.xq, rows.xs, ss, rows.part, y, r)
    } else {
        exec.kquant_gemm_dp4a(w, rows.xq, rows.xs, ss, y, r)
    }
}

/// Where the family's embedding rows come from. A Q8_0 or bf16 file keeps
/// the raw table beside the repacked head (the row gathers read it, the
/// diffusion lane transposes it); a k-quant file keeps no raw copy - its
/// repacked head IS the table, and the gathers read that.
pub(crate) enum EmbdTable<'a> {
    Raw(&'a QuantTensor),
    Kq(&'a RepackedKQ),
}

impl<'a> EmbdTable<'a> {
    /// Off the model's fields (not a method on the model: the callers hold
    /// `&mut self.scratch` at the same time).
    pub(crate) fn of(raw: &'a Option<QuantTensor>, head: &'a Plane) -> Self {
        match (raw, head) {
            (Some(t), _) => EmbdTable::Raw(t),
            (None, Plane::Kq(w)) => EmbdTable::Kq(w),
            // the loader drops the raw table only when the head is k-quant
            (None, _) => {
                unreachable!("gemma4: raw embedding table dropped under a non-k-quant head")
            }
        }
    }

    /// Device-selected rows -> `out [n, embd]`, times `scale`.
    pub(crate) fn gather(
        &self,
        exec: &GpuExecutor,
        ids: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        match self {
            EmbdTable::Raw(t) => exec.embed_gather_plane(t, ids, out, embd, n, scale),
            EmbdTable::Kq(w) => {
                exec.kquant_gather(w, ids, out, embd, n)?;
                if scale != 1.0 {
                    exec.scale(out, scale, n * embd)?;
                }
                Ok(())
            }
        }
    }

    /// One host-side id -> `out [embd]`, unscaled. `id_dev` is the one-slot
    /// device buffer the k-quant gather reads the id from.
    pub(crate) fn row(
        &self,
        exec: &GpuExecutor,
        id: u32,
        id_dev: &mut CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
    ) -> Result<(), GpuError> {
        match self {
            EmbdTable::Raw(t) => exec.dequant_slice(t, id as usize * t.row_bytes(embd), out),
            EmbdTable::Kq(w) => {
                exec.stream
                    .memcpy_htod(&[id], id_dev)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                exec.kquant_gather(w, id_dev, out, embd, 1)
            }
        }
    }
}

impl ExpertPlanes {
    /// The Q8 expert trio, or None when the layer's experts are k-quant.
    pub(crate) fn q8(&self) -> Option<(&RepackedQ8, &RepackedQ8, &RepackedQ8)> {
        match self {
            ExpertPlanes::Q8 { gate, up, down } => Some((gate, up, down)),
            ExpertPlanes::Kq { .. } => None,
        }
    }

    pub(crate) fn kq(&self) -> Option<(&RepackedKQ, &RepackedKQ, &RepackedKQ)> {
        match self {
            ExpertPlanes::Kq { gate, up, down } => Some((gate, up, down)),
            ExpertPlanes::Q8 { .. } => None,
        }
    }
}

/// The routed experts on k-quant seats: `moe_xn [r, n_embd] = sum_slot w *
/// down_e(geglu(gate_e x, up_e x))` off the router's `moe_idx` / `moe_w`
/// and the int8 expert input in `moe_xq` / `moe_xs`. Two classes, the
/// boundary the shape sets (rows >= experts, as `qwen4exp` elects it):
///
/// - decode: the token-batched pair (one block per routed pair per output
///   row - fills the die from r = 1, re-reads a routed row per token);
/// - prefill: gate+up on the sorted TENSOR-CORE pair over a `moe_align`
///   BM=32 CSR (slot 660 - int8 mma straight off the raw k-quant strips,
///   its fused per-32 quantize writing the sorted rows), moved to pair-
///   major rows by the unsort (slot 601); the register-tiled or grouped
///   pair when the tensor-core pair cannot take the planes (mixed types,
///   an i-quant seat, an unaligned width); then the expert-grouped CSR
///   (`moe_align_bm` at the group the routing density wants) and the
///   grouped down over one full-width column chunk (the partials plane is
///   `[pairs, n_embd]` exactly), folded per token in ascending slot order.
///   Measured on the 4-slot batched tick (3 x 256-row canvases + a read,
///   pairs 6272): the tiled pair took the tick to 12.25 s where the Q8
///   sorted pair had it at 9.2; the tensor-core pair is the same class
///   the Q8 seats ride.
///
/// The gate+up epilogue is the family's GEGLU (slots 657-660). The down is
/// a flat 32-weight plane at the expert's own width (704 on the A4B), and
/// behind the tensor-core pair it rides the expert-major TENSOR-CORE down
/// (slot 603): one block per (64-row strip, expert) walks all of the
/// expert's pairs straight off the pair's sorted rows, so the unsort, the
/// grouped CSR and the grouped down drop out of the walk. A width that is
/// not a whole number of 128-weight stages (704 = 5.5) needs the pack's
/// tail marker (slot 668); without it, or without the tensor-core pair,
/// the grouped down over the CSR is the prefill down. Same numeric class
/// as the Q8 pair: exact int8 dots, f32 block scales, the Q4_K mu term on
/// per-16 activation sums.
pub(super) fn g4_moe_experts_kq(
    exec: &GpuExecutor,
    sc: &mut Scratch,
    hp: &Hparams,
    gate: &RepackedKQ,
    up: &RepackedKQ,
    down: &RepackedKQ,
    r: usize,
) -> Result<(), GpuError> {
    let (k, ff, embd, n_expert) = (hp.n_expert_used, hp.ff_exp, hp.n_embd, hp.n_expert);
    let pairs = r * k;
    let act = hp.glu_act();
    if act == GluAct::Gelu && !exec.has_kquant_moe_geglu() {
        return Err(GpuError::MissingOp("kquant MoE GEGLU pair (slots 657-660)"));
    }
    let needs = crate::gpu::kq_needs_sums;
    let ng = needs(gate.ty) || needs(up.ty);
    let nd = needs(down.ty);
    if ng {
        exec.q8_sums_strided(&sc.moe_xq, &mut sc.moe_ssums, embd, r)?;
    }
    let grp = (pairs >= n_expert
        && exec.has_kquant_moe_grp()
        && exec.has_kquant_moe_down_grp()
        && paddock_models::dev_var_os!("PADDOCK_NO_KQMOE_GRP").is_none())
    .then(|| GpuExecutor::kq_moe_group_for(pairs, n_expert));
    // a block per group with every expert's tail padded, and never more
    // blocks than routed pairs (an expert with no row needs no block)
    let blocks_for = |bm: usize| (pairs + n_expert * (bm - 1)).div_ceil(bm).min(pairs).max(1);
    // the tensor-core pair: one k-quant type for gate and up, whole
    // super-blocks in, a 32-aligned ff, and the unsort to feed the grouped
    // down from its sorted rows
    let mma = grp.is_some()
        && gate.ty == up.ty
        && !crate::gpu::kq_is_iq(gate.ty)
        && embd.is_multiple_of(256)
        && ff.is_multiple_of(32)
        && exec.has_kquant_moe_mma()
        && exec.has_moe_q8_rows_unsort()
        && paddock_models::dev_var_os!("PADDOCK_NO_KQMOE_MMA").is_none();
    // the expert-major tensor-core down behind it (slot 603): reads the
    // pair's sorted rows in place, so the unsort and the bm CSR are not
    // built when it runs. One scale per 32 weights = the flat seats (Q5_0 /
    // Q8_0 on the A4B's 704-wide rows); a width that is not whole 128-weight
    // stages needs the tail marker (slot 668)
    let dmma = mma
        && crate::gpu::kq_flat32(down.ty)
        && (ff.is_multiple_of(128) || exec.has_kquant_moe_down_mma_e_tail())
        && exec.has_kquant_moe_down_mma_e()
        && paddock_models::dev_var_os!("PADDOCK_NO_KQMOE_DMMA").is_none();
    let mb32 = blocks_for(32);
    match grp {
        Some(bm) => {
            let blocks = blocks_for(bm);
            if !dmma {
                exec.moe_align_bm(
                    &sc.moe_idx,
                    &mut sc.kq_srow,
                    &mut sc.kq_sslot,
                    &mut sc.kq_bexp,
                    r,
                    k,
                    n_expert,
                    bm,
                    blocks,
                )?;
            }
            if mma {
                exec.moe_align(
                    &sc.moe_idx,
                    &mut sc.moe_srow,
                    &mut sc.moe_sslot,
                    &mut sc.moe_bexp,
                    r,
                    k,
                    n_expert,
                    mb32,
                )?;
                exec.kquant_moe_gate_up_mma_act(
                    act,
                    gate,
                    up,
                    &sc.moe_srow,
                    &sc.moe_bexp,
                    &sc.moe_xq,
                    &sc.moe_xs,
                    ng.then_some(&sc.moe_ssums),
                    &mut sc.kq_sfq,
                    &mut sc.kq_sfs,
                    mb32,
                )?;
                if !dmma {
                    exec.moe_q8_rows_unsort(
                        &sc.kq_sfq,
                        &sc.kq_sfs,
                        &sc.moe_srow,
                        &sc.moe_sslot,
                        &sc.moe_bexp,
                        &mut sc.moe_fq,
                        &mut sc.moe_fs,
                        ff,
                        k,
                        mb32,
                    )?;
                }
            } else if bm == GpuExecutor::KQ_MOE_TILE_BM
                && embd.is_multiple_of(128)
                && exec.has_kquant_moe_gate_up_tile()
            {
                exec.kquant_moe_gate_up_tile_act(
                    act,
                    gate,
                    up,
                    &sc.kq_srow,
                    &sc.kq_sslot,
                    &sc.kq_bexp,
                    &sc.moe_xq,
                    &sc.moe_xs,
                    ng.then_some(&sc.moe_ssums),
                    &mut sc.moe_fused,
                    k,
                    r,
                    blocks,
                )?;
            } else {
                exec.kquant_moe_gate_up_grp_act(
                    act,
                    gate,
                    up,
                    &sc.kq_srow,
                    &sc.kq_sslot,
                    &sc.kq_bexp,
                    &sc.moe_xq,
                    &sc.moe_xs,
                    ng.then_some(&sc.moe_ssums),
                    &mut sc.moe_fused,
                    k,
                    r,
                    blocks,
                    bm,
                )?;
            }
        }
        None => exec.kquant_moe_gate_up_act(
            act,
            gate,
            up,
            &sc.moe_idx,
            &sc.moe_xq,
            &sc.moe_xs,
            ng.then_some(&sc.moe_ssums),
            &mut sc.moe_fused,
            k,
            r,
        )?,
    }
    // the tensor-core pair quantized its rows in registers; every other
    // form left f32 rows in moe_fused
    if !mma {
        exec.quantize_q8(&sc.moe_fused, &mut sc.moe_fq, &mut sc.moe_fs, pairs * ff)?;
    }
    // (the tensor-core down sums its per-16 bytes in-kernel off the sorted
    // rows; the pair-major sums plane is for the grouped and pair downs)
    if nd && !dmma {
        exec.q8_sums_strided(&sc.moe_fq, &mut sc.moe_ssums, ff, pairs)?;
    }
    if dmma {
        // one full-width chunk: the partials plane is [pairs, n_embd] exactly,
        // and a strip's block walks every pair of its expert once
        exec.kquant_moe_down_mma_e(
            down,
            &sc.moe_srow,
            &sc.moe_sslot,
            &sc.moe_bexp,
            &sc.moe_w,
            &sc.kq_sfq,
            &sc.kq_sfs,
            &mut sc.moe_emap,
            &mut sc.moe_part,
            0,
            embd,
            k,
            r,
            n_expert,
            mb32,
        )?;
        exec.moe_part_fold_at(&sc.moe_part, &mut sc.moe_xn, embd, 0, embd, k, r)?;
        return Ok(());
    }
    match grp {
        Some(bm) => {
            let blocks = blocks_for(bm);
            exec.kquant_moe_down_grp(
                down,
                &sc.kq_srow,
                &sc.kq_sslot,
                &sc.kq_bexp,
                &sc.moe_w,
                &sc.moe_fq,
                &sc.moe_fs,
                nd.then_some(&sc.moe_ssums),
                &mut sc.moe_part,
                0,
                embd,
                k,
                r,
                blocks,
                bm,
            )?;
            exec.moe_part_fold_at(&sc.moe_part, &mut sc.moe_xn, embd, 0, embd, k, r)?;
        }
        None => {
            // the column-tiled down keeps the die full once there are a few
            // tokens' worth of blocks; below that the plain kernel's
            // one-float blocks already outnumber the SMs
            if r >= 16 && exec.has_kquant_moe_down_cols() {
                exec.kquant_moe_down_cols(
                    down,
                    &sc.moe_idx,
                    &sc.moe_w,
                    &sc.moe_fq,
                    &sc.moe_fs,
                    nd.then_some(&sc.moe_ssums),
                    &mut sc.moe_xn,
                    k,
                    r,
                    4,
                )?;
            } else {
                exec.kquant_moe_down(
                    down,
                    &sc.moe_idx,
                    &sc.moe_w,
                    &sc.moe_fq,
                    &sc.moe_fs,
                    nd.then_some(&sc.moe_ssums),
                    &mut sc.moe_xn,
                    k,
                    r,
                )?;
            }
        }
    }
    Ok(())
}
