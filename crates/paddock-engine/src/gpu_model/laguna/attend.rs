//! The batched lane's prefill attention for one layer, out of the layer walk:
//! append the chunk's K/V and attend, per cache kind. An SWA layer steps its
//! ring in `SWA_SPAN` sub-spans (within one, nothing a row reads has been
//! recycled); a full layer appends the whole chunk in one pass and attends
//! per same-slot run. A fused tick's decode band rides the decode tick's own
//! launches (see `PfCuts::dec` and [`band`]).
//!
//! With Flint's prefill tile and no decode band, the f16 wide rung lands the
//! bf16 rows the o-projection's tile reads straight in the tile's scratch
//! (slot 925: the f32 output's bits put through the cast), and the
//! o-projection skips its cast. Its q may come in f16 too (slot 930), when
//! the layer's norm + rope wrote it so (slot 929, see `W16r::q16_plane`). A
//! decode band keeps both: its rows attend in f32 and land in the bf16 plane
//! through the cast the o-projection would have applied.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor, KvDtype, PfQ};

use super::batch::{BatchScratch, LayerKv, SWA_SPAN};
use super::nv::W16r;

/// One layer's attention geometry and class.
pub(super) struct PfAttend<'a> {
    /// the layer's learned sinks; None = the scratch's -inf row
    pub sinks: Option<&'a CudaSlice<f32>>,
    pub bt: &'a CudaSlice<u32>,
    pub bps: usize,
    pub nh: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub kv_dim: usize,
    /// the sliding window (0 on a full layer)
    pub window: usize,
    pub scale: f32,
    pub kv_dtype: KvDtype,
    /// the f16 WMMA-class prefill tile serves this layer's prefill spans
    pub wmma_pf: bool,
}

/// The f16 prefill tile over rows `[off, off + len)`: into `out16` when the
/// layer elected it (None again once the first span is refused - nothing has
/// landed there yet then), else the f32 plane. `q16`: this layer's q is in
/// f16 only (so the bf16 rung must take every span).
#[allow(clippy::too_many_arguments)]
fn tile(
    exec: &GpuExecutor,
    a: &PfAttend,
    sinks: &CudaSlice<f32>,
    q: &CudaSlice<f32>,
    q16: Option<&CudaSlice<half::f16>>,
    out: &mut CudaSlice<f32>,
    pos: &CudaSlice<u32>,
    slots: &CudaSlice<u32>,
    kvs: &LayerKv,
    out16: &mut Option<&mut CudaSlice<half::bf16>>,
    first: bool,
    off: usize,
    len: usize,
) -> Result<(), GpuError> {
    let (k, v) = (&kvs.k, &kvs.v);
    if let Some(o) = out16.as_deref_mut() {
        let (bt, bps, nh, n_kv, hd) = (a.bt, a.bps, a.nh, a.n_kv, a.hd);
        let qp = q16.map_or(PfQ::F32(q), PfQ::F16);
        if exec.attn_prefill_f16_paged_o16_at(
            qp, k, v, sinks, o, pos, slots, off, bt, bps, nh, n_kv, hd, a.kv_dim, a.window, len,
            a.scale, a.kv_dtype,
        )? {
            return Ok(());
        }
        if q16.is_some() {
            // the ready probe said yes before q was written in f16
            return Err(GpuError::Unsupported(
                "attn q16: the f16 wide rung refused a span".into(),
            ));
        }
        if !first {
            // a refusal past the first span would split the layer's rows
            // across two planes - the shape is the same for every span
            return Err(GpuError::Unsupported(
                "attn o16: the f16 wide rung refused a later span".into(),
            ));
        }
        *out16 = None;
    }
    exec.attn_prefill_f16_paged_at(
        q, k, v, sinks, out, pos, slots, off, a.bt, a.bps, a.nh, a.n_kv, a.hd, a.kv_dim, a.window,
        len, a.scale, a.kv_dtype,
    )
}

/// A fused tick's decode band, rows `[0, dec)`: the launches a decode tick of
/// the same rows makes (split partials + combine at the same split count, so
/// a band row attends exactly as it would on a decode tick), else the
/// decode-batch rows kernel. Takes the scratch fields it touches, so the
/// caller's other borrows (the sinks row) stay live.
#[allow(clippy::too_many_arguments)]
fn band(
    exec: &GpuExecutor,
    kvs: &LayerKv,
    a: &PfAttend,
    sinks: &CudaSlice<f32>,
    (qn, pos, slots): (&CudaSlice<f32>, &CudaSlice<u32>, &CudaSlice<u32>),
    (part_o, part_ml, out): (
        &mut CudaSlice<f32>,
        &mut CudaSlice<f32>,
        &mut CudaSlice<f32>,
    ),
    off: usize,
    len: usize,
) -> Result<(), GpuError> {
    let (bt, bps, nh, n_kv, hd, kv_dim) = (a.bt, a.bps, a.nh, a.n_kv, a.hd, a.kv_dim);
    let (window, scale, kv_dtype) = (a.window, a.scale, a.kv_dtype);
    let ns = super::batch::attn_splits(nh, n_kv, len, exec.sm_count(), window);
    let fused1 = ns == 1 && super::batch::attn_gqa_fused(nh, n_kv, len);
    let (k, v) = (&kvs.k, &kvs.v);
    if off == 0 && (ns > 1 || fused1) && exec.has_attn_partial_batch_paged() {
        exec.attn_partial_batch_paged(
            qn,
            k,
            v,
            part_o,
            part_ml,
            pos,
            Some(slots),
            bt,
            bps,
            nh,
            n_kv,
            hd,
            kv_dim,
            window,
            ns,
            len,
            scale,
            kv_dtype,
        )?;
        return exec.attn_combine_batch(part_o, part_ml, sinks, out, nh, hd, ns, len);
    }
    exec.attn_decode_batch_rows_paged(
        qn,
        k,
        v,
        sinks,
        out,
        pos,
        Some(slots),
        bt,
        bps,
        nh,
        n_kv,
        hd,
        kv_dim,
        window,
        off,
        len,
        scale,
        kv_dtype,
    )
}

/// Append + attend for one prefill layer: `swa` = the SWA ladder's sub-spans
/// (an SWA layer), else a full layer over `runs`; `dec` leading rows are a
/// fused tick's decode band. `ring` blocks a slot (the SWA ring's size).
#[allow(clippy::too_many_arguments)]
pub(super) fn attend_pf(
    exec: &GpuExecutor,
    sc: &mut BatchScratch,
    kvs: &mut LayerKv,
    a: &PfAttend,
    swa: Option<(&[(usize, usize)], usize)>,
    runs: &[(usize, usize)],
    dec: usize,
    r: usize,
    ws: Option<&mut W16r>,
) -> Result<(), GpuError> {
    let sinks = a.sinks.unwrap_or(&sc.sinks);
    let (bt, bps, nh, n_kv, hd, kv_dim) = (a.bt, a.bps, a.nh, a.n_kv, a.hd, a.kv_dim);
    let (window, scale, kv_dtype) = (a.window, a.scale, a.kv_dtype);
    // every span of the layer must land in one plane: bf16 on the f16 tile
    // (a decode band's f32 rows are cast in after the spans)
    let o16_ok = a.wmma_pf && a.kv_dtype == KvDtype::Fp16;
    let mut ws = ws;
    let (mut out16, q16) = match ws.as_deref_mut() {
        Some(w) => w.attn_planes(),
        None => (None, None),
    };
    if !o16_ok {
        if q16.is_some() {
            return Err(GpuError::Unsupported(
                "attn q16 without the bf16 rung".into(),
            ));
        }
        out16 = None;
    }
    let mut first = true;
    if let Some((spans, ring)) = swa {
        // span ladder: within one sub-span nothing a row reads has been
        // recycled (ring holds SWA_SPAN + window)
        debug_assert!(ring * 16 >= SWA_SPAN + window);
        for &(off, len) in spans {
            exec.kv_append_batch_paged_rows(
                &sc.kn,
                &mut kvs.k,
                &sc.d_pos,
                Some(&sc.d_slots),
                bt,
                bps,
                kv_dim,
                off,
                len,
                kv_dtype,
            )?;
            exec.kv_append_batch_paged_rows(
                &sc.v,
                &mut kvs.v,
                &sc.d_pos,
                Some(&sc.d_slots),
                bt,
                bps,
                kv_dim,
                off,
                len,
                kv_dtype,
            )?;
            // the fused tick's decode band (see PfCuts::dec)
            if off < dec {
                let ins = (&sc.qn, &sc.d_pos, &sc.d_slots);
                let outs = (&mut sc.attn_o, &mut sc.attn_ml, &mut sc.attn);
                band(exec, kvs, a, sinks, ins, outs, off, len)?;
            } else if a.wmma_pf {
                let (q, o, p, s) = (&sc.qn, &mut sc.attn, &sc.d_pos, &sc.d_slots);
                tile(
                    exec, a, sinks, q, q16, o, p, s, kvs, &mut out16, first, off, len,
                )?;
                first = false;
            } else if len > 24 && exec.has_attn_prefill_paged() {
                exec.attn_prefill_rows_paged(
                    &sc.qn,
                    &kvs.k,
                    &kvs.v,
                    sinks,
                    &mut sc.attn,
                    &sc.d_pos,
                    &sc.d_slots,
                    bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    off,
                    len,
                    scale,
                    kv_dtype,
                )?;
            } else {
                exec.attn_decode_batch_rows_paged(
                    &sc.qn,
                    &kvs.k,
                    &kvs.v,
                    sinks,
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
                    off,
                    len,
                    scale,
                    kv_dtype,
                )?;
            }
        }
    } else {
        // full layer, prefill: append the whole chunk in one pass (per-row
        // slot/pos - rows land in their own slots' pool blocks, no
        // aliasing), attend per same-slot run (causal bound =
        // positions[row]). Single-run chunks keep the whole-chunk calls -
        // bit-for-bit the serial path.
        exec.kv_append_batch_paged(
            &sc.kn,
            &mut kvs.k,
            &sc.d_pos,
            Some(&sc.d_slots),
            bt,
            bps,
            kv_dim,
            r,
            kv_dtype,
        )?;
        exec.kv_append_batch_paged(
            &sc.v,
            &mut kvs.v,
            &sc.d_pos,
            Some(&sc.d_slots),
            bt,
            bps,
            kv_dim,
            r,
            kv_dtype,
        )?;
        if runs.len() == 1 {
            if a.wmma_pf {
                // the row-offset entry at 0 is the whole-chunk call's launch
                let (q, o, p, s) = (&sc.qn, &mut sc.attn, &sc.d_pos, &sc.d_slots);
                tile(
                    exec, a, sinks, q, q16, o, p, s, kvs, &mut out16, first, 0, r,
                )?;
            } else if r > 24 && exec.has_attn_prefill_paged() {
                exec.attn_prefill_paged(
                    &sc.qn,
                    &kvs.k,
                    &kvs.v,
                    sinks,
                    &mut sc.attn,
                    &sc.d_pos,
                    &sc.d_slots,
                    bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    r,
                    scale,
                    kv_dtype,
                )?;
            } else {
                exec.attn_decode_batch_paged(
                    &sc.qn,
                    &kvs.k,
                    &kvs.v,
                    sinks,
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
                    r,
                    scale,
                    kv_dtype,
                )?;
            }
        } else {
            for &(off, len) in runs {
                if off < dec {
                    // fused decode band
                    let ins = (&sc.qn, &sc.d_pos, &sc.d_slots);
                    let outs = (&mut sc.attn_o, &mut sc.attn_ml, &mut sc.attn);
                    band(exec, kvs, a, sinks, ins, outs, off, len)?;
                } else if a.wmma_pf {
                    let (q, o, p, s) = (&sc.qn, &mut sc.attn, &sc.d_pos, &sc.d_slots);
                    tile(
                        exec, a, sinks, q, q16, o, p, s, kvs, &mut out16, first, off, len,
                    )?;
                    first = false;
                } else if len > 24 && exec.has_attn_prefill_paged() {
                    exec.attn_prefill_rows_paged(
                        &sc.qn,
                        &kvs.k,
                        &kvs.v,
                        sinks,
                        &mut sc.attn,
                        &sc.d_pos,
                        &sc.d_slots,
                        bt,
                        bps,
                        nh,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        off,
                        len,
                        scale,
                        kv_dtype,
                    )?;
                } else {
                    exec.attn_decode_batch_rows_paged(
                        &sc.qn,
                        &kvs.k,
                        &kvs.v,
                        sinks,
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
                        off,
                        len,
                        scale,
                        kv_dtype,
                    )?;
                }
            }
        }
    }
    if let Some(o) = out16.as_deref_mut()
        && dec > 0
    {
        // the band's rows lead both planes: the cast o_rows would apply
        exec.convert_f32_bf16(&sc.attn, o, dec * nh * hd)?;
    }
    let landed = out16.is_some();
    if landed && let Some(ws) = ws {
        ws.set_fresh();
    }
    Ok(())
}
