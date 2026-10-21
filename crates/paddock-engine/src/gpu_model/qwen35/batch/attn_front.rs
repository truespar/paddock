//! The decode tick's full-attention front: the one-GEMM q|gate|k|v projection
//! lands `[b][nq + nk2 + nv2]` in `d_gu_fused` on both decode lanes (the f8t
//! tile plane and the W8 rung), and [`qkg_front`] consumes it.

use super::*;

/// The fused plane's segments (q|gate interleaved, k, v) and the layer's
/// attention geometry.
pub(super) struct QkgGeo {
    pub nq: usize,
    pub nk2: usize,
    pub nv2: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_rot: usize,
}

/// Consume the fused plane. Where it applies, one pd_q36_qkg_nra_rows - the
/// decode twin of the prefill Phase-A consumer - reads the plane strided and
/// does split_qg + q/k rmsnorm + mrope + the paged k/v append in registers:
/// one launch for the nine (three row slices, split_qg, 2x rmsnorm, 2x mrope,
/// 2x append), bit-identical qn / gate planes and pools. Otherwise the plane is
/// split (row slices + split_qg) and the caller's norm / rope / append chain
/// follows. True when the one launch ran. The nra kernel is PDL-armed while
/// the chain it replaces is plain-launched - judge it end-to-end, per the
/// second PDL law. Kill: PADDOCK_NO_QNF_DEC.
#[allow(clippy::too_many_arguments)]
pub(super) fn qkg_front(
    exec: &GpuExecutor,
    bs: &mut BatchState,
    sc: &mut Scratch,
    w: &FullAttnWeights,
    li: usize,
    g: &QkgGeo,
    eps: f32,
    yarn: (f32, f32, f32, f32, f32, f32),
    sections: [u32; 4],
    b: usize,
    kv_dtype: KvDtype,
) -> Result<bool, GpuError> {
    static QNF_DEC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let qnf_dec = bs.paged
        && g.head_dim == 256
        && g.n_rot == 64
        && exec.has_q36_qkg_nra_rows()
        && *QNF_DEC.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_QNF_DEC").is_none());
    let (nq, nk2, nv2) = (g.nq, g.nk2, g.nv2);
    let nt = nq + nk2 + nv2;
    let fused = bs.d_gu_fused.as_ref().expect("qkv fused landing");
    if qnf_dec {
        exec.q36_qkg_nra_rows(
            fused,
            0,
            nt,
            nq,
            nq + nk2,
            &w.q_norm.buf,
            &w.k_norm.buf,
            &mut sc.d_qn,
            &mut sc.d_gate,
            bs.kv_k[li].as_mut().expect("full-attn layer KV"),
            bs.kv_v[li].as_mut().expect("full-attn layer KV"),
            &bs.d_pos,
            Some(&bs.d_slots),
            &bs.d_mrope,
            bs.d_block_tables.as_ref().expect("paged block tables"),
            bs.blocks_per_slot,
            g.n_heads,
            g.n_kv_heads,
            g.head_dim,
            g.n_rot,
            eps,
            yarn,
            sections,
            b,
            kv_dtype,
        )?;
        return Ok(true);
    }
    if exec.has_row_slice4() {
        exec.row_slice4(
            fused,
            nt,
            b,
            &mut [
                (&mut sc.d_qg, 0, nq),
                (&mut sc.d_k, nq, nk2),
                (&mut sc.d_v, nq + nk2, nv2),
            ],
        )?;
    } else {
        exec.row_slice(fused, &mut sc.d_qg, nt, 0, nq, b)?;
        exec.row_slice(fused, &mut sc.d_k, nt, nq, nk2, b)?;
        exec.row_slice(fused, &mut sc.d_v, nt, nq + nk2, nv2, b)?;
    }
    exec.split_qg(
        &sc.d_qg,
        &mut sc.d_q,
        &mut sc.d_gate,
        b,
        g.n_heads,
        g.head_dim,
    )?;
    Ok(false)
}
