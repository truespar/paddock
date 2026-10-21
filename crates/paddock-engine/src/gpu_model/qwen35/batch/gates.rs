//! The batched tick's route gates: env-read once switches and row ceilings that pick
//! the f8t tile plane, the fused GDN recurrence and the varlen chunked GDN.

/// f8t unified arm master switch (see the bs_f8t_attn_p note in
/// `unified_launch_core`). Kill: PADDOCK_NO_F8T_UNIFIED.
pub(super) fn f8t_unified_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_F8T_UNIFIED").is_none())
}

/// f8t CHUNK arm row ceiling: `prefill_batch_pass` rides the tile plane
/// while r <= this (65+ takes the launcher's tc5r 2-SM whole-prefill route,
/// which pads batch to 256 and TMA-reads the activation buffer to that pad -
/// d_f8t_q/d_f8t_rs are sized to next_multiple_of(256) of this value, see
/// the scratch alloc). Motivation: the split path's admission chunk passes
/// (r ~= 128 at the c32 board) ran q8_0/f8bs at ~18.6 ms/admission where the
/// tile plane's roofline says ~4 - the same mixed-tick mechanism the unified
/// arm already fixed at r <= 64. Default 256 = one full tc5r batch band.
/// PADDOCK_F8T_CHUNK_RMAX overrides; PADDOCK_NO_F8T_CHUNK kills (0).
/// tc5r@128 probe instrument: decode-side f8t arm batch bound.
/// Default 64 = shipped behavior exactly. PADDOCK_F8T_DEC_BMAX=128 lets the
/// b<=128 decode tick keep the f8t class (f8t_gemm routes 65..256 via tc5r,
/// one weight pass) instead of splitting into 2x64 halves - pair with
/// PADDOCK_NO_DEC_SPLIT=1 to probe.
pub(in crate::gpu_model::qwen35) fn f8t_dec_bmax() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PADDOCK_F8T_DEC_BMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    })
}

pub(in crate::gpu_model::qwen35) fn f8t_chunk_rmax() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        if paddock_models::dev_var_os!("PADDOCK_NO_F8T_CHUNK").is_some() {
            return 0;
        }
        paddock_models::dev_var!("PADDOCK_F8T_CHUNK_RMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

/// P70 fused decode recurrence kill (PADDOCK_NO_DN_V2F restores the
/// split_gqa_norm + recurrent_v2 chain for A/B).
pub(super) fn dn_v2f_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_DN_V2F").is_none())
}

/// The decode DeltaNet front in one launch above the v2f band (slot 875: conv
/// step + silu + split + q/k norms + GQA widening, bit-identical to conv_step +
/// split_gqa_norm). Kill: PADDOCK_NO_DN_QKVN.
pub(in crate::gpu_model::qwen35) fn dn_qkvn_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_DN_QKVN").is_none())
}

/// Row ceiling for the P70/P71-R2 fused GDN recurrence (`v2f` and the strided
/// `v2f_g` twin). 24 came from a probe curve of the RECURRENCE KERNEL alone
/// (2.01x at b=1, 1.5x at 8, 1.22x at 12, 1.02x at 24, 0.94x at 32) - the
/// fused form loses 6% at b=32 because the recurrence is state-bandwidth-bound
/// there and the in-block norm recompute is redundant.
///
/// But the kernel is not the band. At b=32 the fused form ELIMINATES
/// `deltanet_split_gqa_norm` (159.1 ms in a long-prompt census) and, in
/// the strided form, `row_slice2_gate` (204.8 ms) - against a 6% premium on a
/// 1465.6 ms recurrence (+93.5 ms). Band arithmetic says -270 ms of an
/// 11 287 ms die, ~2.4%. That is worth a measurement the kernel probe could
/// never see. `PADDOCK_DN_V2F_BMAX` overrides.
pub(super) fn dn_v2f_bmax() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_DN_V2F_BMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24)
    })
}

/// Varlen chunked-GDN route gate - the same env chain the unified tick's
/// `vl_route` static checks (GDN formulation band); kept in
/// sync by hand because that one is fn-local. Kill: PADDOCK_NO_DNC_VL.
pub(super) fn dnc_vl_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_NO_DNC_VL").is_none()
            && std::env::var("PADDOCK_DNC_RS").map(|v| v != "0").unwrap_or(false)
            && std::env::var("PADDOCK_DNC_S1MMA").map(|v| v != "0").unwrap_or(false)
            && paddock_models::dev_var_os!("PADDOCK_DNC_DWB16").is_none()
            // bf16 state stays falsified and excluded; f16 state rides the
            // ST walk_rs (PPL-gated +0.09%) so it does not disqualify VL.
            && paddock_models::dev_var_os!("PADDOCK_DN_STATE_BF16").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_DNC_MMA_V2").is_none()
            && paddock_models::dev_var_os!("PADDOCK_DNC_SCAN").is_none()
            && paddock_models::dev_var_os!("PADDOCK_DNC_FLA").is_none()
            && paddock_models::dev_var_os!("PADDOCK_DNC_SPLIT").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_DN").is_none()
    })
}

/// The varlen prefill attention takes the tcgen05 arm (slot 872, 128-row
/// tiles) for the G 6 geometry when the pack ships it and `PADDOCK_NO_PF_TC5`
/// is unset; the item builder and the launch both ask, so they agree.
pub(super) fn pf_tc5_vl(exec: &crate::gpu::GpuExecutor, g: usize) -> bool {
    g == 6 && exec.has_attn_prefill_tc5_vl()
}

/// PADDOCK_DN_CZ_STRIDED (opt-in): no split of the decode tick's fused DN in-proj plane - the
/// conv step and the gated norm read their halves of it strided (conv_step_slots_s /
/// gated_rmsnorm_s, bit-identical; 96 fewer row_slice launches a tick at depth 128). Measured
/// SLOWER on the B200 serve: strided 308.0 / 309.5 vs split 314.5 / 314.5 pages/min.
pub(super) fn dn_cz_strided_on() -> bool {
    static CZ: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CZ.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DN_CZ_STRIDED").is_some())
}

/// Whether a `rows`-row tick can ride the f8t lane on the mixer planes `p`: past 64 rows the f8t
/// GEMM takes the 2-SM tc5r route, which serves even 128-row tile counts only. qwen3.8-27b's
/// DeltaNet in-proj plane is 129 tiles (in_qkv + z + the b|a fold), so with sm_100's decode bound
/// of 128 (PADDOCK_F8T_DEC_BMAX) every 65..=128-row tick was refused and the serve dropped its
/// overlap decode pipe. The prefill pass applies the same rule (its tc5r_ok).
pub(super) fn f8t_rows_ok(rows: usize, p: &[crate::gpu::F8TilePlane; 2]) -> bool {
    rows <= 64 || p.iter().all(|t| (t.scale.len() >> 7) % 2 == 0)
}
