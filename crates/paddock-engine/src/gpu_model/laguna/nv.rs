//! Flint's NVFP4 linears on the laguna body. One election for every dense
//! plane the GGUF ships (attention q/k/v/o, the dense lead FFNs, the shared
//! expert, the head), all W4A16 because the file marks every tensor
//! 16-bit-activation:
//!
//! - 1 row: the serial f32-activation GEMV - the class the serial lane runs;
//! - up to [`TC_FLOOR`] rows: its row-batched twin, bit-exact per row, so the
//!   decode band and short verify rounds stay on the serial lane's class;
//! - wider: the exact-dequant bf16 tensor-core GEMM (bf16 rows, f32 sums) -
//!   still 16-bit activations, so still the checkpoint's class, never the
//!   W4A4 block-scale pair.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor, Nvf4Plane};

use super::batch::BatchScratch;

/// Rows above which the tensor-core GEMM takes a plane. A numerics line, not
/// a speed one: at or below it every width reads the serial lane's bits.
pub(crate) const TC_FLOOR: usize = 8;

/// `y [rows, w.out_dim] = x [rows, w.in_dim] * W^T`, W4A16.
pub(crate) fn nv_rows(
    exec: &GpuExecutor,
    w: &Nvf4Plane,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuError> {
    if rows == 1 {
        return exec.nvf4_gemv(w, x, y, None);
    }
    // the same dev switch the other NVFP4 lanes read
    static TC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let tc = *TC.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_NVF4_TC")
            .map(|v| v != "0")
            .unwrap_or(true)
    });
    if tc && rows > TC_FLOOR && w.in_dim.is_multiple_of(16) && exec.has_nvf4_gemm_tc() {
        return exec.nvf4_gemm_tc(w, x, y, None, rows);
    }
    exec.nvf4_gemv_batch(w, x, y, None, rows)
}

/// The NVFP4 SwiGLU FFN over `rows` rows of `x` into `y`: gate and up into
/// the two scratch planes, SiLU(gate) * up in place, then down.
#[allow(clippy::too_many_arguments)]
pub(crate) fn nv_swiglu(
    exec: &GpuExecutor,
    gate: &Nvf4Plane,
    up: &Nvf4Plane,
    down: &Nvf4Plane,
    x: &CudaSlice<f32>,
    g: &mut CudaSlice<f32>,
    u: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuError> {
    // the decode band's gate|up in one launch (slot 927; rows bit-identical)
    let pair = (2..=TC_FLOOR).contains(&rows)
        && exec.nvf4_gemv_batch3(&mut [(gate, &mut *g), (up, &mut *u)], x, rows)?;
    if !pair {
        nv_rows(exec, gate, x, g, rows)?;
        nv_rows(exec, up, x, u, rows)?;
    }
    exec.swiglu(g, u, rows * gate.out_dim)?;
    nv_rows(exec, down, g, y, rows)
}

/// A layer's q / k norm + rope inputs for the fused q|k|v grid (slot 934).
pub(super) struct NrLayer<'a> {
    pub qw: &'a CudaSlice<f32>,
    pub kw: &'a CudaSlice<f32>,
    pub n_head: usize,
    pub n_kv: usize,
    /// a fused tick's decode band rows (their q stays f32 for the band's
    /// decode launches)
    pub dec: usize,
    pub eps: f32,
    pub rope: (f32, f32, f32, f32, f32, f32),
    pub qscale: f32,
}

/// Prefill mode on the W4A16 prefill tile: q|k|v with the q / k norm + rope
/// in the epilogues (slot 934) - q in the f16 attention's halves (the band's
/// rows f32 in `sc.qn`), k normed + roped in `sc.kn`, v in `sc.v` - and the
/// layer's q marked f16 for the attention. False, nothing changed, where the
/// pack or the shape declines; the caller then runs the separate passes.
pub(super) fn qkv_nr_rows(
    exec: &GpuExecutor,
    planes: [&Nvf4Plane; 3],
    sc: &mut BatchScratch,
    ws: Option<&mut W16r>,
    nr: &NrLayer,
    r: usize,
) -> Result<bool, GpuError> {
    let Some(ws) = ws.filter(|w| w.pf && w.q16.is_some()) else {
        return Ok(false);
    };
    if !exec.has_nvf4_pf_qkv_nr() {
        return Ok(false);
    }
    let k = planes[0].in_dim;
    if !std::mem::take(&mut ws.x16_fresh) {
        exec.convert_f32_bf16(&sc.xn, &mut ws.x16, r * k)?;
    }
    let Some(q16) = ws.q16.as_mut() else {
        return Ok(false);
    };
    let out = crate::gpu::PfQkvOut {
        q16,
        qn: &mut sc.qn,
        kn: &mut sc.kn,
        v: &mut sc.v,
    };
    let norm = crate::gpu::PfNorm {
        qw: nr.qw,
        kw: nr.kw,
        pos: &sc.d_pos,
        n_head: nr.n_head,
        n_kv: nr.n_kv,
        dec: nr.dec,
        eps: nr.eps,
        rope: nr.rope,
        qscale: nr.qscale,
    };
    let ran = exec.nvf4_pf_qkv_nr(planes, out, &norm, &ws.x16, r)?;
    if ran {
        ws.q16_live = true;
    } else {
        // the cast stays for the separate passes' tile
        ws.x16_fresh = true;
    }
    Ok(ran)
}

/// The batched lane's q, k and v planes over `r` rows of `sc.xn`; prefill
/// mode on the W4A16 prefill tile (one bf16 cast serves all three).
pub(super) fn qkv_rows(
    exec: &GpuExecutor,
    [wq, wk, wv]: [&Nvf4Plane; 3],
    sc: &mut BatchScratch,
    ws: Option<&mut W16r>,
    pf: bool,
    r: usize,
) -> Result<(), GpuError> {
    if let Some(ws) = ws.filter(|w| pf && w.pf) {
        let k = wq.in_dim;
        if !std::mem::take(&mut ws.x16_fresh) {
            exec.convert_f32_bf16(&sc.xn, &mut ws.x16, r * k)?;
        }
        // one grid (slot 931): k's and v's thin grids fill q's last wave
        let planes = &mut [(wq, &mut sc.q), (wk, &mut sc.k), (wv, &mut sc.v)];
        if exec.nvf4_pf_gemm3(planes, &ws.x16, r)? {
            return Ok(());
        }
        exec.nvf4_pf_gemm(wq, &ws.x16, k, 0, &mut sc.q, wq.out_dim, 0, r)?;
        exec.nvf4_pf_gemm(wk, &ws.x16, k, 0, &mut sc.k, wk.out_dim, 0, r)?;
        return exec.nvf4_pf_gemm(wv, &ws.x16, k, 0, &mut sc.v, wv.out_dim, 0, r);
    }
    // the decode band's q|k|v in one launch (slot 927; rows bit-identical)
    if (2..=TC_FLOOR).contains(&r) {
        let planes = &mut [(wq, &mut sc.q), (wk, &mut sc.k), (wv, &mut sc.v)];
        if exec.nvf4_gemv_batch3(planes, &sc.xn, r)? {
            return Ok(());
        }
    }
    nv_rows(exec, wq, &sc.xn, &mut sc.q, r)?;
    nv_rows(exec, wk, &sc.xn, &mut sc.k, r)?;
    nv_rows(exec, wv, &sc.xn, &mut sc.v, r)
}

/// The batched lane's attention output projection over `r` rows of
/// `sc.attn` into `sc.proj`; prefill mode on the prefill tile.
pub(super) fn o_rows(
    exec: &GpuExecutor,
    wo: &Nvf4Plane,
    sc: &mut BatchScratch,
    ws: Option<&mut W16r>,
    pf: bool,
    r: usize,
) -> Result<(), GpuError> {
    if let Some(ws) = ws.filter(|w| pf && w.pf) {
        // the attention may have landed its rows bf16 already (slot 925)
        if !std::mem::take(&mut ws.x16_fresh) {
            exec.convert_f32_bf16(&sc.attn, &mut ws.x16, r * wo.in_dim)?;
        }
        let (k, n) = (wo.in_dim, wo.out_dim);
        return exec.nvf4_pf_gemm(wo, &ws.x16, k, 0, &mut sc.proj, n, 0, r);
    }
    nv_rows(exec, wo, &sc.attn, &mut sc.proj, r)
}

/// The batched lane's dense lead FFN over `r` rows of `sc.xn` into `sc.proj`;
/// prefill mode on the prefill tile (SwiGLU rows land bf16 for the down).
pub(super) fn dense_rows(
    exec: &GpuExecutor,
    [gate, up, down]: [&Nvf4Plane; 3],
    sc: &mut BatchScratch,
    ws: Option<&mut W16r>,
    pf: bool,
    r: usize,
) -> Result<(), GpuError> {
    if let Some(ws) = ws.filter(|w| pf && w.pf) {
        let (k, ff) = (gate.in_dim, gate.out_dim);
        if !std::mem::take(&mut ws.x16_fresh) {
            exec.convert_f32_bf16(&sc.xn, &mut ws.x16, r * k)?;
        }
        exec.nvf4_pf_gu(gate, up, &ws.x16, k, &mut ws.act, ff, 0, r)?;
        return exec.nvf4_pf_gemm(down, &ws.act, ff, 0, &mut sc.proj, down.out_dim, 0, r);
    }
    nv_swiglu(
        exec,
        gate,
        up,
        down,
        &sc.xn,
        &mut sc.ffn_gate,
        &mut sc.ffn_up,
        &mut sc.proj,
        r,
    )
}

/// Scratch for the 16-bit-activation tensor-core classes (the W16 expert
/// pair, moe/nvf4_w16r, and the W4A16 prefill tile, gemm/nvf4_pf): the rows'
/// bf16 activations, the SwiGLU rows a down reads, and the per-slot partials
/// (k routed slots + the shared expert's) the fold sums in a fixed order.
pub(crate) struct W16r {
    x16: CudaSlice<half::bf16>,
    act: CudaSlice<half::bf16>,
    part: CudaSlice<f32>,
    /// the pack carries the prefill tile (slots 920-923)
    pf: bool,
    /// prefill mode: routed slots whose bf16 partials sit in `part` unfolded
    /// (the FFN tail folds them with the residual add and the next norm)
    pending: usize,
    /// prefill mode: a residual tail already wrote `x16` as the next
    /// consumer's rows (taken by that consumer; reset every walk)
    x16_fresh: bool,
    /// prefill mode: q as the f16 wide attention rung reads it (slot 929
    /// writes, slot 930 reads; None without the pair)
    q16: Option<CudaSlice<half::f16>>,
    /// this layer's norm + rope wrote `q16` instead of the f32 plane
    q16_live: bool,
}

impl W16r {
    /// Allocated beside the batch state for a model whose MoE is NVFP4 with
    /// an NVFP4 shared expert (Flint); None everywhere else, and None when
    /// the pack lacks the pair (the GEMVs then serve every width).
    pub(crate) fn new(
        exec: &GpuExecutor,
        hp: &super::Hparams,
        layers: &[super::LagunaLayer],
        cap: usize,
    ) -> Result<Option<Self>, GpuError> {
        let nv_shared = layers.iter().any(|l| {
            matches!(&l.ffn, super::Ffn::MoeNv(w) if matches!(w.shared, super::SharedNv::Nv { .. }))
        });
        if !nv_shared || !exec.has_nvf4_moe_w16r() {
            return Ok(None);
        }
        let m = &hp.moe;
        // widest bf16 row: the normed rows, the attention output, a dense
        // FFN's SwiGLU rows
        let q_max = hp.n_heads.iter().copied().max().unwrap_or(0) * hp.head_dim;
        let dense_ff = layers
            .iter()
            .filter_map(|l| l.ffn.dense_ff())
            .max()
            .unwrap_or(0);
        let act_w = (m.n_active * m.moe_ff + m.shexp_ff).max(dense_ff);
        let q16 = exec.has_nvf4_pf()
            && exec.has_qkv_norm_rope_batch_q16()
            && exec.has_attn_prefill_f16_paged_q16();
        Ok(Some(Self {
            x16: exec.stream_alloc_bf16(cap * hp.n_embd.max(q_max))?,
            act: exec.stream_alloc_bf16(cap * act_w)?,
            part: exec.alloc(cap * (m.n_active + 1) * hp.n_embd)?,
            pf: exec.has_nvf4_pf(),
            pending: 0,
            x16_fresh: false,
            q16: if q16 {
                Some(exec.alloc_f16(cap * q_max)?)
            } else {
                None
            },
            q16_live: false,
        }))
    }

    /// A walk starts with no deferred fold and no rows written ahead.
    pub(crate) fn reset(&mut self) {
        (self.pending, self.x16_fresh, self.q16_live) = (0, false, false);
    }

    /// Prefill mode: the f16 q plane a norm + rope may land this layer's q
    /// in (None without the pair or the prefill tile); [`Self::set_q16`]
    /// once it has, and the attention then reads only that.
    pub(super) fn q16_plane(&mut self) -> Option<&mut CudaSlice<half::f16>> {
        if self.pf { self.q16.as_mut() } else { None }
    }

    /// This layer's q sits in the f16 plane.
    pub(super) fn set_q16(&mut self) {
        self.q16_live = true;
    }

    /// The attention's planes: the bf16 rows it may land its output in (None
    /// without the prefill tile; [`Self::set_fresh`] once it has) and the f16
    /// q this layer's norm + rope wrote (taken: the next layer decides
    /// again).
    pub(super) fn attn_planes(
        &mut self,
    ) -> (
        Option<&mut CudaSlice<half::bf16>>,
        Option<&CudaSlice<half::f16>>,
    ) {
        let live = std::mem::take(&mut self.q16_live);
        let q16 = self.q16.as_ref().filter(|_| live);
        (self.pf.then_some(&mut self.x16), q16)
    }

    /// The bf16 rows [`Self::attn_planes`] handed out are the next consumer's.
    pub(super) fn set_fresh(&mut self) {
        self.x16_fresh = true;
    }

    /// The pack carries the W4A16 prefill tile (slots 920-923).
    pub(crate) fn pf_tile(&self) -> bool {
        self.pf
    }
}

/// The post-attention residual tail: `x += proj`, `xn = rmsnorm(x) * w`. In
/// prefill mode with the prefill tile it also lands the MoE's bf16 rows in
/// the same pass (slot 924, add_rmsnorm_batch's bits); otherwise the plain
/// fused add + norm.
#[allow(clippy::too_many_arguments)]
pub(super) fn attn_tail(
    exec: &GpuExecutor,
    sc: &mut BatchScratch,
    ws: Option<&mut W16r>,
    pf: bool,
    w: &CudaSlice<f32>,
    embd: usize,
    eps: f32,
    r: usize,
) -> Result<(), GpuError> {
    if let Some(ws) = ws.filter(|s| pf && s.pf) {
        let x16 = Some(&mut ws.x16);
        let (x, xn) = (&mut sc.x, Some(&mut sc.xn));
        ws.x16_fresh = exec.add_norm16(x, &sc.proj, None, w, xn, x16, embd, eps, r, false)?;
        if ws.x16_fresh {
            return Ok(());
        }
    }
    exec.add_rmsnorm_batch(&mut sc.x, &sc.proj, w, &mut sc.xn, embd, eps, r)
}

/// The post-FFN residual tail of a layer without a post-FFN norm: `x +=
/// proj` (plus the routed partials a prefill MoE left unfolded). With the
/// next layer's attention norm `next`, in prefill mode, the fold, the add
/// and that norm run as one pass landing only the bf16 rows the q|k|v
/// prefill tile reads (slot 924, the chain's bits) - true then, and the
/// caller skips the next layer's own norm.
#[allow(clippy::too_many_arguments)]
pub(super) fn ffn_tail(
    exec: &GpuExecutor,
    sc: &mut BatchScratch,
    ws: Option<&mut W16r>,
    pf: bool,
    next: Option<&CudaSlice<f32>>,
    embd: usize,
    eps: f32,
    r: usize,
) -> Result<bool, GpuError> {
    let Some(ws) = ws else {
        exec.add(&mut sc.x, &sc.proj, r * embd)?;
        return Ok(false);
    };
    let k = std::mem::take(&mut ws.pending);
    ws.x16_fresh = false;
    if let Some(w) = next.filter(|_| pf && ws.pf) {
        let part = (k > 0).then_some((&ws.part, k));
        let x16 = Some(&mut ws.x16);
        ws.x16_fresh =
            exec.add_norm16(&mut sc.x, &sc.proj, part, w, None, x16, embd, eps, r, true)?;
        if ws.x16_fresh {
            return Ok(true);
        }
    }
    if k > 0 {
        exec.moe_slot_combine_bf16(&ws.part, &mut sc.proj, embd, k, r)?;
    }
    exec.add(&mut sc.x, &sc.proj, r * embd)?;
    Ok(false)
}

/// Flint's MoE over `r` rows of `sc.xn` into `sc.proj` on the W16 pair: the
/// router as the GEMV path routes, bf16 rows, the 32-row sorting, routed +
/// shared gate|up and down on tensor cores, then the fixed-order fold of the
/// k + 1 slot partials (routed slots ascending, the shared expert last).
#[allow(clippy::too_many_arguments)]
pub(super) fn moe_w16r_rows(
    exec: &GpuExecutor,
    m: &super::MoeDims,
    w: &super::MoeNv,
    sc: &mut BatchScratch,
    ws: &mut W16r,
    embd: usize,
    r: usize,
) -> Result<(), GpuError> {
    let super::SharedNv::Nv {
        gate: sh_gate,
        up: sh_up,
        down: sh_down,
    } = &w.shared
    else {
        return Err(GpuError::Unsupported(
            "the W16 pair serves an NVFP4 shared expert".into(),
        ));
    };
    let k = m.n_active;
    if r <= 8 {
        exec.bf16_gemm(&w.router_b16, None, &sc.xn, &mut sc.moe_logits, r)?;
    } else {
        exec.matvec_f32_batch(&w.router_w, &sc.xn, &mut sc.moe_logits, r)?;
    }
    m.route(
        exec,
        &sc.moe_logits,
        &w.probs_bias,
        &mut sc.moe_idx,
        &mut sc.moe_w,
        r,
    )?;
    exec.convert_f32_bf16(&sc.xn, &mut ws.x16, r * embd)?;
    let nb = super::moe_nv::align_blocks(r * k, m.n_expert, crate::gpu::W16R_BLOCK);
    exec.moe_align(
        &sc.moe_idx,
        &mut sc.srow,
        &mut sc.sslot,
        &mut sc.bexp,
        r,
        k,
        m.n_expert,
        nb,
    )?;
    exec.nvf4_moe_gu_swiglu_w16r(
        &w.gate,
        &w.up,
        sh_gate,
        sh_up,
        &sc.srow,
        &sc.sslot,
        &sc.bexp,
        &ws.x16,
        &mut ws.act,
        k,
        nb,
        r,
    )?;
    exec.nvf4_moe_down_part_w16r(
        &w.down,
        sh_down,
        &sc.srow,
        &sc.sslot,
        &sc.bexp,
        &sc.moe_w,
        &ws.act,
        &mut ws.part,
        k,
        nb,
        r,
    )?;
    exec.moe_slot_combine_init_at(&ws.part, &mut sc.proj, 0, embd, k + 1, r)
}

/// The NVFP4 builds' MoE over `r` rows, per build and mode: `wide` (prefill
/// mode, wide verify rounds) takes Flint's W16 pair when its scratch exists
/// and Kolibri's W4A4 sorted pair otherwise; everything else rides the W4A16
/// GEMVs. Flint's W4A16 file never takes the W4A4 pair, whatever the pack.
#[allow(clippy::too_many_arguments)]
pub(super) fn moe_nv_any(
    exec: &GpuExecutor,
    hp: &super::Hparams,
    w: &super::MoeNv,
    sc: &mut BatchScratch,
    ws: Option<&mut W16r>,
    pf: bool,
    wide: bool,
    r: usize,
    x16_ready: bool,
) -> Result<(), crate::gpu_model::gpt_oss::GpuModelError> {
    let (m, embd) = (&hp.moe, hp.n_embd);
    match ws {
        Some(ws) if pf && ws.pf => Ok(moe_pf_rows(exec, m, w, sc, ws, embd, r)?),
        Some(ws) if wide => Ok(moe_w16r_rows(exec, m, w, sc, ws, embd, r)?),
        _ => {
            let sorted = wide && hp.flavor != super::Flavor::Flint;
            super::moe_nv::moe_nv_rows(exec, m, w, sc, embd, r, sorted, x16_ready)
        }
    }
}

/// The prefill router rides the BF16x3 GEMM (slot 722): a BF16 plane, the
/// pack carries it, PADDOCK_NO_ROUTER_X3 unset.
fn router_x3(exec: &GpuExecutor, w: &crate::gpu::QuantTensor) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_ROUTER_X3").is_some());
    !off && exec.has_diar_gemm() && w.ty == paddock_models::ggml_type::GgmlType::Bf16
}

/// Flint's MoE in prefill mode on the W4A16 prefill tile: the router as the
/// GEMV path routes, bf16 rows (unless the attention tail landed them),
/// moe_align_bm's 64-token blocks, the routed gate|up (slot 922) and the
/// shared expert's (921) into one SwiGLU plane, the shared down (920)
/// straight into `sc.proj`, the routed down's bf16 per-slot partials (923)
/// left for [`ffn_tail`]'s fixed-order fold.
#[allow(clippy::too_many_arguments)]
fn moe_pf_rows(
    exec: &GpuExecutor,
    m: &super::MoeDims,
    w: &super::MoeNv,
    sc: &mut BatchScratch,
    ws: &mut W16r,
    embd: usize,
    r: usize,
) -> Result<(), GpuError> {
    let super::SharedNv::Nv {
        gate: sh_gate,
        up: sh_up,
        down: sh_down,
    } = &w.shared
    else {
        return Err(GpuError::Unsupported(
            "the prefill tile's MoE serves an NVFP4 shared expert".into(),
        ));
    };
    let (k, ff) = (m.n_active, m.moe_ff);
    let aw = k * ff + m.shexp_ff;
    if r <= 8 {
        exec.bf16_gemm(&w.router_b16, None, &sc.xn, &mut sc.moe_logits, r)?;
    } else if router_x3(exec, &w.router_b16) {
        // the BF16 plane on the tensor cores with xn split into three bf16
        // parts: every product exact, f32 sums in the mma's fixed k order -
        // row-invariant at any chunk size (no split-K; the tile ladder never
        // moves a row's K walk). The exact-f32 tiles below ran the router at
        // ~2 TF/s, 36 ms of an 8K prefill.
        let (k, n) = (embd, m.n_expert);
        let (x, wb) = (
            (&sc.xn, 0),
            crate::gpu::DiarWeights::Bf16(&w.router_b16.bytes),
        );
        let store = crate::gpu::DiarEpi::Store;
        exec.diar_gemm(x, wb, None, &mut sc.moe_logits, (k, n, r), store, None)?;
    } else {
        // the exact-f32 tiles at every chunk size: one K walk a logit
        // whatever the rows (the matvec tile below 1024 rows reassociated it,
        // and ran a 0.8K prompt's router at ~2.5 TF/s)
        if !exec.gemm_f32_tiled(&w.router_w, &sc.xn, &mut sc.moe_logits, r)? {
            exec.matvec_f32_batch(&w.router_w, &sc.xn, &mut sc.moe_logits, r)?;
        }
    }
    m.route(
        exec,
        &sc.moe_logits,
        &w.probs_bias,
        &mut sc.moe_idx,
        &mut sc.moe_w,
        r,
    )?;
    if !std::mem::take(&mut ws.x16_fresh) {
        exec.convert_f32_bf16(&sc.xn, &mut ws.x16, r * embd)?;
    }
    let bm = crate::gpu::NVPF_MOE_BLOCK;
    let nb = super::moe_nv::align_blocks(r * k, m.n_expert, bm);
    exec.moe_align_bm(
        &sc.moe_idx,
        &mut sc.srow,
        &mut sc.sslot,
        &mut sc.bexp,
        r,
        k,
        m.n_expert,
        bm,
        nb,
    )?;
    // the shared expert's tiles ride the routed grids' last waves (slots
    // 932 / 933; each half its own launch's bits)
    let sh = exec.has_nvf4_pf_moe_sh();
    let fused = sh
        && exec.nvf4_pf_moe_gu_sh(
            &w.gate,
            &w.up,
            &sc.srow,
            &sc.sslot,
            &sc.bexp,
            &ws.x16,
            &mut ws.act,
            aw,
            nb,
            (sh_gate, sh_up),
            k * ff,
            r,
        )?;
    if !fused {
        exec.nvf4_pf_moe_gu(
            &w.gate,
            &w.up,
            &sc.srow,
            &sc.sslot,
            &sc.bexp,
            &ws.x16,
            &mut ws.act,
            aw,
            nb,
        )?;
        exec.nvf4_pf_gu(sh_gate, sh_up, &ws.x16, embd, &mut ws.act, aw, k * ff, r)?;
    }
    if sh
        && exec.nvf4_pf_moe_dn_sh(
            &w.down,
            &sc.srow,
            &sc.sslot,
            &sc.bexp,
            &sc.moe_w,
            &ws.act,
            aw,
            &mut ws.part,
            k,
            nb,
            (sh_down, &mut sc.proj),
            k * ff,
            r,
        )?
    {
        // the fold rides the FFN tail (`ffn_tail`), with the residual add
        ws.pending = k;
        return Ok(());
    }
    exec.nvf4_pf_gemm(sh_down, &ws.act, aw, k * ff, &mut sc.proj, embd, 0, r)?;
    exec.nvf4_pf_moe_dn(
        &w.down,
        &sc.srow,
        &sc.sslot,
        &sc.bexp,
        &sc.moe_w,
        &ws.act,
        aw,
        &mut ws.part,
        k,
        nb,
    )?;
    // the fold rides the FFN tail (`ffn_tail`), with the residual add
    ws.pending = k;
    Ok(())
}
