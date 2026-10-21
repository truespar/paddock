//! The Q8_0 prefill GEMM rungs of the gemma4 lane: the mmq ladder every Q8
//! projection takes above the decode band, the FFN's gate|up pair landed
//! straight into the down GEMM's input, and the sandwich norms landed
//! straight into the next GEMM's quantized activations.

use cudarc::driver::CudaSlice;

use crate::gpu::{GluAct, GpuError, GpuExecutor, RepackedQ8};

/// mmq GEMM ladder for the prefill lane (qwen35's `prefill_mm_pre` shape):
/// the deep-pipe kernel on %128 weights, plain split-tile mmq otherwise.
/// Caller has already quantized the activation tile into `yq`.
///
/// The pipe used to be gated at `r > 1024` - inherited from qwen35's
/// `mmq_hi_min_batch`, never measured below it on this lane. It is BIT-EQUAL
/// to the sync mmq (both fold per k32 in ascending order; only the staging
/// differs - memcmp-gated across 24 shape x batch cells, and the live server's
/// greedy text is byte-identical across this flip), and
/// the mcol prototype bench measured it faster at every row count 64..512 on
/// all three muse shapes: ffn_gate 128 rows 520 -> 331 us, 136 rows 828 -> 570,
/// ffn_down 136 rows 1254 -> 805, wq 202 -> 125. The sync kernel is
/// barrier-bound at 1 block/SM (its own header says so) and nothing about that
/// changes below 1024 rows, which is why the old gate was leaving a third of
/// the prefill on the floor. PADDOCK_G4_PF_SYNC=1 pins the old kernel for A/B.
pub(super) fn pf_mmq(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    yq: &CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuError> {
    // The A/B pin reproduces the old GATE, not "sync everywhere": this lane
    // already took the pipe above 1024 rows, so a pin that forced sync at
    // every r would measure pipe-vs-sync rather than new-vs-old and credit
    // the 1024-row flip with a gain at 2048..8192 tokens that was already
    // there. Keep the `r <= 1024` clause.
    if pf_sync_pin() && r <= 1024 {
        return exec.q8_0_gemm_mmq(w, yq, Some(skfix), y, r);
    }
    if w.dims[0].is_multiple_of(128) && exec.has_q8_0_gemm_mmq_pipe() {
        // stream-K tail split (qwen35's serving rung) - OPT-IN via
        // PADDOCK_G4_SK: it recovers last-wave slack on the narrow
        // projections (~1% throughput) but reproducibly costs c8 TTFT p50
        // around +180 ms - and TTFT is the standing target. skfix doubles
        // as the partials plane (sized 256*128*128, the sk contract).
        // (kept on its original r > 1024 band - the tail-split arm was only
        // ever measured there, and this change is not the place to widen it)
        if r > 1024
            && exec.has_q8_0_gemm_mmq_pipe_sk()
            && paddock_models::dev_var_os!("PADDOCK_G4_SK").is_some()
        {
            exec.q8_0_gemm_mmq_pipe_sk(w, yq, skfix, y, r)
        } else {
            exec.q8_0_gemm_mmq_pipe(w, None, yq, y, r)
        }
    } else {
        exec.q8_0_gemm_mmq(w, yq, Some(skfix), y, r)
    }
}

/// A/B pin for the prefill GEMM rung: `PADDOCK_G4_PF_SYNC=1` restores the
/// old ladder - sync mmq at or below 1024 rows, pipe above. Read once; this
/// sits inside the per-projection prefill path.
fn pf_sync_pin() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_G4_PF_SYNC")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// The FFN's gate and up mmq GEMMs with `glu(gate) * up` quantized in the
/// landing (pack slot 865): `yq` the FFN input in the mmq layout, `yo` the
/// down GEMM's input. Byte-identical to two [`pf_mmq`] pipe GEMMs +
/// `quantize_q8_mmq_glu` - same per-row values at any row count, so the
/// warm-vs-cold prefill invariant holds. Ok(false) = not this pack / shape
/// / activation (the caller keeps its chain). `PADDOCK_G4_NO_GU8=1` pins the
/// chain for A/B.
pub(super) fn pf_gu_q8(
    exec: &GpuExecutor,
    gate: &RepackedQ8,
    up: &RepackedQ8,
    yq: &CudaSlice<u8>,
    yo: &mut CudaSlice<u8>,
    r: usize,
    act: GluAct,
) -> Result<bool, GpuError> {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_GU8").is_some());
    let fits = gate.dims[1].is_multiple_of(128)
        && gate.dims[0].is_multiple_of(128)
        && up.dims == gate.dims
        && yo.len() >= gate.dims[1] / 128 * r.next_multiple_of(128) * 144;
    if off || pf_sync_pin() || !fits || !exec.has_q8_0_gemm_mmq_gu(act) {
        return Ok(false);
    }
    exec.q8_0_gemm_mmq_gu(gate, up, yq, yo, r, act)?;
    Ok(true)
}

/// The batched lane's Q8_0 gate|up arm above the decode band: quantize the
/// normed rows, then land the down GEMM's input directly when `fuse` allows
/// ([`pf_gu_q8`]), else run gate and up and fold the GLU into `pf_gate`.
/// True = the down's input already sits in `pf_yq_ff` ([`pf_down_q8`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn pf_gu_q8_arm(
    exec: &GpuExecutor,
    sc: &mut super::Scratch,
    gate: &RepackedQ8,
    up: &RepackedQ8,
    n_embd: usize,
    r: usize,
    act: GluAct,
    fuse: bool,
) -> Result<bool, GpuError> {
    exec.quantize_q8_mmq(&sc.pf_normed, &mut sc.pf_yq, n_embd, r)?;
    if fuse && pf_gu_q8(exec, gate, up, &sc.pf_yq, &mut sc.pf_yq_ff, r, act)? {
        return Ok(true);
    }
    pf_mmq(exec, gate, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_gate, r)?;
    pf_mmq(exec, up, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_up, r)?;
    exec.glu(&mut sc.pf_gate, &sc.pf_up, r * gate.dims[1], act)
        .map(|()| false)
}

/// The down GEMM over the gate|up arm's output: `landed` = its mmq input
/// is already in `pf_yq_ff`, else quantize the GLU'd `pf_gate` first.
pub(super) fn pf_down_q8(
    exec: &GpuExecutor,
    sc: &mut super::Scratch,
    down: &RepackedQ8,
    r: usize,
    landed: bool,
) -> Result<(), GpuError> {
    if !landed {
        exec.quantize_q8_mmq(&sc.pf_gate, &mut sc.pf_yq, down.dims[0], r)?;
    }
    let yq = if landed { &sc.pf_yq_ff } else { &sc.pf_yq };
    pf_mmq(exec, down, yq, &mut sc.pf_skfix, &mut sc.pf_proj, r)
}

/// Whether a layer's attention reads its input norm only through the Q8_0
/// mmq arm - one quantize of the normed rows feeds q / k / v and nothing
/// reads the f32 rows: no e4m3 attention planes, live Q8_0 q / k / v, and not
/// gated (muse's output gate reads them). The precondition for landing that
/// norm from the previous layer's sandwich ([`pf_sandwich_q8`]).
pub(super) fn attn_q8_only(lw: &super::LayerWeights) -> bool {
    let live = |p: &super::Plane| p.q8().is_some_and(|w| w.data.len() > 48);
    lw.fused_norm_ok()
        && lw.f8t_qkv.is_none()
        && lw.f8a_wq.is_none()
        && lw.f8a_wqkv.is_none()
        && lw.f8w_wq.is_none()
        && lw.f8_wq.is_none()
        && lw.qkv_ws.is_none()
        && live(&lw.wq)
        && live(&lw.wk)
        && lw.wv.as_ref().is_none_or(live)
}

/// A sandwich block's post-norm + residual, `x = (x + rmsnorm(proj) *
/// w_post) * s`, with the NEXT norm landed mmq-quantized in `pf_yq` (pack
/// slot 866) - the bytes rmsnorm_add_scale + rmsnorm_batch +
/// quantize_q8_mmq write, `pf_normed` left unwritten. Ok(false) = not this
/// pack / band (rows < 256, n % 128, the two norms' eps differ, or
/// `PADDOCK_G4_NO_NQ8=1`): the caller runs its chain.
#[allow(clippy::too_many_arguments)]
pub(super) fn pf_sandwich_q8(
    exec: &GpuExecutor,
    sc: &mut super::Scratch,
    w_post: &CudaSlice<f32>,
    w_next: &CudaSlice<f32>,
    hp: &super::Hparams,
    s: f32,
    r: usize,
) -> Result<bool, GpuError> {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_NQ8").is_some())
        || !hp.fused_two_norm_ok()
    {
        return Ok(false);
    }
    exec.rmsnorm_add_scale_norm_q8(
        &mut sc.pf_x,
        &sc.pf_proj,
        w_post,
        w_next,
        None,
        &mut sc.pf_yq,
        hp.n_embd,
        hp.eps,
        s,
        r,
    )
}
