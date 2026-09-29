//! Single-token decode step (batch 1) - the correctness milestone.
//!
//! Op-for-op mirror of llama.cpp b10058 `llama_model_gemma4::graph` (dense
//! path; the 31B has no MoE/per-layer-embd branches). Anything that deviates
//! from that graph is a parity bug, not an optimization opportunity - batched
//! prefill/decode lanes come after greedy parity locks.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor, RepackedKQ, RepackedQ8};

use super::{EmbdTable, GpuGemma4, kq_rows, kq_stage, planes};

/// mmq GEMM ladder for the prefill lane (qwen35's `prefill_mm_pre` shape):
/// the deep-pipe kernel on %128 weights, plain split-tile mmq otherwise.
/// Caller has already quantized the activation tile into `yq`.
///
/// The pipe used to be gated at `r > 1024` - inherited from qwen35's
/// `mmq_hi_min_batch`, never measured below it on this lane. It is BIT-EQUAL
/// to the sync mmq (both fold per k32 in ascending order; only the staging
/// differs - memcmp-gated across 24 shape x batch cells, and the live server's
/// greedy text is byte-identical across this flip), and
/// bench/muse_mcol_proto.cu measures it faster at every row count 64..512 on
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

/// GEGLU -> e4m3 for the F8R down-GEMM input: fused single-pass kernel when
/// the pack ships it (gate/up read once, the f32 activation never lands -
/// saves a full r x n_ff write + re-read per FFN), else the geglu +
/// quantize_e4m3 sequence. Bit-identical values either way.
pub(super) fn g4_e4m3_glu(
    exec: &GpuExecutor,
    sc: &mut super::Scratch,
    n: usize,
    act: crate::gpu::GluAct,
) -> Result<(), GpuError> {
    if exec.has_quantize_e4m3_glu(act) {
        exec.quantize_e4m3_glu(
            &sc.pf_gate,
            &sc.pf_up,
            &mut sc.pf_e4q,
            &mut sc.pf_e4s,
            n,
            act,
        )
    } else {
        exec.glu(&mut sc.pf_gate, &sc.pf_up, n, act)?;
        exec.quantize_e4m3(&sc.pf_gate, &mut sc.pf_e4q, &mut sc.pf_e4s, n)
    }
}

/// Batched-prefill chunk rows. 2048 (llama's ubatch class) puts the mmq
/// GEMMs above the pipe rung's 1024-row floor; widest scratch row set is
/// ~1 GB - noise next to the 33 GB of weights.
// 8192: SGLang's B200 heuristic runs 16384-token prefill chunks -
// at our old 2048 cap every full-weight pass amortized over 8x fewer
// tokens and the BM=128 expert-sort PAD fraction sat at ~2x (255 blocks
// for 128 experts). The tick stays 2048 by DEFAULT (decode cadence);
// PADDOCK_G4_TICK_ROWS raises it per deployment up to this cap.
pub(crate) const PF_ROWS: usize = 8192;

/// Rows the prefill scratch is actually allocated for on this server - and
/// therefore the size every prefill lane chunks at. The two must never
/// diverge: the scratch planes are `[rows, dim]`, so a chunk wider than the
/// allocation is an out-of-bounds write, not a slow path.
///
/// PF_ROWS above is a CEILING, not a size. It was raised 2048 -> 8192 for
/// B200 muse waves, and allocating the full 8192 unconditionally costs every
/// server ~5.0 GiB on gemma-4-31B - even a 4096-context serve, which can
/// never present more than 4096 rows in one chunk. That unusable half of the
/// scratch is enough to push the batch pool to zero slots and drop the model
/// onto the serial engine.
///
/// Two things bound a chunk, and the wider wins:
///  - a single-sequence bulk prefill, at most `max_ctx` rows;
///  - one serving tick - the mixed prefill budget plus every live slot's
///    spec-verify rows. [`pf_rows_floor`] covers the budget plus 1024 rows of
///    verify, i.e. any width up to ~113 slots.
///
/// This is where a server STARTS. `enable_batch` steps it down the halving
/// ladder when the KV plan cannot seat the configured context beside it, so
/// `GpuGemma4::pf_rows` - not this - is the width every lane must read.
pub(crate) fn pf_rows(max_ctx: usize) -> usize {
    PF_ROWS.min(max_ctx.next_multiple_of(128).max(pf_rows_floor()))
}

/// The narrowest chunk this server may run: one serving tick - the mixed
/// prefill budget plus 1024 rows of spec verify.
///
/// Below it a tick overruns the planes, which is an out-of-bounds write and
/// not a slow path, so this is where `batch::set_pf_rows`'s ladder stops.
pub(crate) fn pf_rows_floor() -> usize {
    (super::batch::mixed_tick_rows() + 1024).next_multiple_of(128)
}

/// SWA append+attend sub-span rows. The WindowRing only has to absorb one
/// sub-span of appends (plus the window behind it) before older blocks may
/// alias - shrinking the ring from (PF_ROWS+window) to (span+picture overshoot
/// +window): 193 -> 115 blocks, 2.36 -> 1.41 GB/slot at window 1024. GEMMs
/// still run whole PF_ROWS chunks; only the SWA append/attend ladder steps.
///
/// A 512-row span buys the smallest ring, but it is blind to launch shape:
/// v3s runs 254 regs/thread = 1 CTA/SM, so a 512-row span is a 128-CTA
/// launch on a 188-SM die and a ~1.6k-row chunk becomes 4 SERIALIZED
/// under-wave launches per SWA layer (~1.07 ms where the CTA-work says
/// ~0.3). PADDOCK_G4_SWA_SPAN trades ring VRAM back for die-filling grids.
/// 2048 is the elected DEFAULT: it costs ~+8.8 GB at 32 slots and buys back
/// both throughput and TTFT; 512 remains the revert lever for VRAM-tight
/// deployments. Ring sizing and the span cutters all read [`swa_span`] so
/// the aliasing invariant holds by construction.
pub(crate) const SWA_SPAN_DEFAULT: usize = 2048;

/// Spans the engine may elect between, widest first. Every rung has been
/// measured (the A/B in [`SWA_SPAN_DEFAULT`]); `enable_batch` walks this
/// ladder down until the asked-for width fits, so a VRAM-tight box gets a
/// narrower span AND a real batch lane instead of the widest span and the
/// serial engine. Not a knob: the operator never picks, the fit does.
pub(crate) const SWA_SPAN_LADDER: [usize; 3] = [2048, 1024, 512];

/// Operator pin for the SWA sub-span (`PADDOCK_G4_SWA_SPAN`, rows, clamped
/// to [64, PF_ROWS]), read once. `Some` suppresses the fit-driven election
/// entirely - a development instrument for A/Bing the ladder itself.
pub(crate) fn swa_span_pin() -> Option<usize> {
    static SPAN: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *SPAN.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_G4_SWA_SPAN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(64, PF_ROWS))
    })
}

/// The span this process starts at: the pin if there is one, else the
/// widest rung. `enable_batch` may narrow it; `GpuGemma4::swa_span` is the
/// live value and must be the only source for both the ring sizing and the
/// span cuts - a mismatch aliases live window blocks.
pub(crate) fn swa_span_initial() -> usize {
    swa_span_pin().unwrap_or(SWA_SPAN_DEFAULT)
}

pub(crate) fn swa_spans(span: usize, runs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    for &(off, n) in runs {
        let mut o = off;
        while o < off + n {
            let len = span.min(off + n - o);
            spans.push((o, len));
            o += len;
        }
    }
    spans
}

impl GpuGemma4 {
    /// muse-glimmer's embedding preamble: an UNWEIGHTED RMSNorm over each
    /// gathered row, at the pre-norm epsilon, applied in place. No-op on
    /// gemma4, whose preamble is the sqrt(n_embd) scale already folded into
    /// the gather (see `Hparams::embd_scale`).
    ///
    /// Reference: `inpL = build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS,
    /// -1)` - null weight, null bias, il = -1 so the eps is `f_norm_rms_eps`
    /// (1e-5 here), not the 1e-8 the two post-norms use.
    ///
    /// Takes the fields rather than `&self`: every call site already holds
    /// `&mut self.scratch`, so a whole-`self` borrow would collide.
    pub(crate) fn embd_preamble(
        exec: &GpuExecutor,
        hp: &super::Hparams,
        ones: Option<&CudaSlice<f32>>,
        x: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let Some(w) = ones else { return Ok(()) };
        exec.rmsnorm_batch_inplace(x, w, hp.n_embd, hp.eps, rows)
    }

    /// muse-glimmer's attention output gate, applied in place to the
    /// head-concatenated attention output before `wo`:
    ///
    /// ```text
    ///   g   = W_gate @ attn_norm(x)          [rows, n_head*head_dim]
    ///   out = out * sigmoid(g)
    /// ```
    ///
    /// `W_gate` has the exact shape of `W_q` ([n_embd, n_head*head_dim]) and
    /// eats the same post-`attn_norm` state the Q/K/V projections do - Not the
    /// raw residual. Order matters: the sigmoid multiplies the concatenated
    /// heads, so it has to land before the out-projection AND before whatever
    /// activation quantize the chosen `wo` arm does.
    ///
    /// No-op (and not even a launch) on gemma4, which ships no gate planes.
    ///
    /// SOTA gap, deliberate: this is a separate GEMM re-reading the normed
    /// activations, which is what the reference graph does but not what this
    /// engine can do best. Because `attn_gate` is dimensionally identical to
    /// `attn_q` and consumes the same input, it belongs as a fourth segment of
    /// the fused q|k|v concat plane (`f8a_wqkv`/`f8t_qkv`) - one GEMM, one
    /// activation quantize, one plane read instead of two. That needs the
    /// offset-addressed consumers, `pf_q` sizing and the fused nra epilogue to
    /// all learn a 4-segment layout, so it lands as a measured perf rung after
    /// the correctness gate, not before it.
    ///
    /// What is not deferrable, and was: the gate had no fp8 arm at
    /// all, so every batched prefill row fell to `q8_0_gemm_repacked` - a
    /// GEMV-shaped kernel (one block per output column, 16-row tiles) running a
    /// prefill-shaped GEMM. On the binding workload that is 15.86 ms per layer
    /// versus 0.159 ms for `wo`, which has the identical 93.3 GFLOP - 5.9 vs
    /// 587 TFLOP/s. At 52 layers that one plane was ~831 ms of a ~1180 ms
    /// prefill. The gate now takes wq's tile plane, which is the same class
    /// change already applied to q/k/v/o and the FFN on this model, so the
    /// e4m3 quality gate covers one more plane rather than a new family.
    ///
    /// It rides the same Q8_0 weights, on the int8 tensor-core GEMMs `wo`
    /// already uses - `mma_ks` through 192 rows (the `pf_xq` sizing cap), the
    /// `mmq` tile ladder above. The weight class does not move at all, which is
    /// the point: an e4m3 plane for this tensor was built and measured first,
    /// and although it was slightly faster, it moved an answer on the
    /// correctness gate (the reference's value survived but the surrounding
    /// text drifted, which an unconditional criterion counts as a fail).
    /// Same-weights int8 keeps the gate green AND costs no VRAM, so it is
    /// strictly the better rung - the e4m3 plane is a measured, rejected
    /// alternative.
    ///
    /// What does change vs the old fallback is the activation side: f32 rows
    /// become int8 per-32, exactly as `wo`'s arms do. That is a far smaller
    /// move than e4m3 (8 mantissa bits per block vs 3), and it is the class
    /// every other Q8_0-lane projection on this model already runs.
    ///
    /// The quantize is the gate's own rather than the QKV block's: this runs
    /// after attention, and the callers in between reuse this scratch for the
    /// attention epilogue. Clobbering it back is safe in the other direction -
    /// every `wo` arm downstream re-quantizes `pf_attn` into it before reading.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_gate_apply(
        exec: &GpuExecutor,
        lw: &super::LayerWeights,
        normed: &CudaSlice<f32>,
        agate: &mut CudaSlice<f32>,
        attn: &mut CudaSlice<f32>,
        xq: &mut CudaSlice<i8>,
        xs: &mut CudaSlice<f32>,
        yq: &mut CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        e4q: &mut CudaSlice<i8>,
        e4rs: &mut CudaSlice<f32>,
        n_embd: usize,
        q_dim: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let Some(wgp) = &lw.attn_gate else {
            return Ok(());
        };
        debug_assert_eq!(wgp.dims()[0], n_embd, "attn_gate in-dim must match n_embd");
        debug_assert_eq!(
            wgp.dims()[1],
            q_dim,
            "attn_gate out-dim must match n_head*head_dim"
        );
        // the gate rides the Q8 rungs; no file ships it in another class
        // today (muse-glimmer's UD-Q8_K_XL keeps it Q8_0), and its int8
        // staging here carries no sums plane for a k-quant one
        let Some(wg) = wgp.q8() else {
            return Err(GpuError::Unsupported(
                "attention output gate outside the Q8 class".into(),
            ));
        };
        let r = rows;
        if let Some(p) = lw.f8t_attn_gate.as_ref() {
            // cc-10: ride the e4m3 tile GEMV like qkv/wo at every r.
            // The o-gate was the last per-layer projection on the crippled int8
            // path (dp4a at r==1, int8-mma/mmq at r>1) - ~52 us/layer, 17% of
            // the c1 decode tick, and at c32 PREFILL the wide-chunk o-gate was
            // still on Q8 int8 (the r<=64 cap this replaces). f8t_gemm serves
            // every r for wq/wo (the decode plane covers r==1 through prefill),
            // so the o-gate rides it too. r==1 +10% c1; the prefill band is the
            // c32/TTFT reach. `part` doubles as the skfix. Gated behind the
            // plane (PADDOCK_MUSE_OGATE_F8T); Q8 is the quality-safe fallback.
            exec.quantize_e4m3_row(normed, e4q, e4rs, n_embd, r)?;
            exec.f8t_gemm(p, e4q, e4rs, part, agate, n_embd, q_dim, r)?;
        } else if r == 1 {
            exec.q8_0_gemv_repacked(wg, None, normed, agate)?;
        } else if paddock_models::dev_var_os!("PADDOCK_G4_NO_AGATE_TC").is_some() {
            // A/B kill for this rung - the earlier route, which put every
            // batched prefill row on a GEMV-shaped kernel: 15.86 ms per layer
            // against wo's 0.159 ms for the identical 93.3 GFLOP (5.9 vs 587
            // TFLOP/s), ~831 ms of a ~1180 ms prefill across 52 layers
            exec.q8_0_gemm_repacked(wg, None, normed, agate, r)?;
        } else if r <= 192 {
            exec.quantize_q8(normed, xq, xs, r * n_embd)?;
            exec.q8_0_gemm_mma_ks(wg, xq, xs, part, agate, r)?;
        } else {
            exec.quantize_q8_mmq(normed, yq, n_embd, r)?;
            pf_mmq(exec, wg, yq, part, agate, r)?;
        }
        exec.mul_sigmoid(attn, agate, rows * q_dim)
    }

    /// Device-side twin of `Hparams::logit_epilogue` - same order (scale, then
    /// cap), used by the staged head that must not sync to the host.
    fn logit_epilogue_dev(
        exec: &GpuExecutor,
        logits: &mut CudaSlice<f32>,
        vocab: usize,
        hp: &super::Hparams,
    ) -> Result<(), GpuError> {
        super::logit_epilogue_dev(exec, logits, vocab, hp.logit_scale, hp.final_softcap)
    }

    /// One token at `self.pos` -> full-vocab logits (host, softcap applied).
    pub(crate) fn step(&mut self, token: u32) -> Result<Vec<f32>, GpuError> {
        // pool mode (multimodal exclusive decode runs here): back slot 0's
        // global KV for this position before the walk appends to it
        self.ensure_global_rows(&[0], &[self.pos as u32])?;
        let hp = &self.hp;
        let sc = &mut self.scratch;
        let exec = &self.exec;
        let pos = self.pos;

        exec.stream
            .memcpy_htod(&[pos as u32], &mut sc.pos)
            .map_err(|e| GpuError::Driver(e.to_string()))?;

        // ── embedding: widen the token's row out of the embedding plane, then
        // the arch's preamble - gemma4 scales by sqrt(n_embd) (ggml: inpL =
        // get_rows(tok_embd) * sqrtf(n_embd)), muse-glimmer RMS-normalizes
        // instead. f32 throughout either way.
        EmbdTable::of(&self.token_embd, &self.head).row(
            exec,
            token,
            &mut sc.embd_id,
            &mut sc.stream_tmp,
            hp.n_embd,
        )?;
        exec.stream
            .memset_zeros(&mut sc.x)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        exec.scale_add(&mut sc.x, &sc.stream_tmp, hp.embd_scale(), hp.n_embd)?;
        Self::embd_preamble(exec, hp, self.embd_ones.as_ref(), &mut sc.x, 1)?;

        for (li, lw) in self.layers.iter().enumerate() {
            let kvl = &mut self.kv[li];
            // per-layer twin registration before any append enqueue
            self.exec.vdim_set(kvl.vdim.as_ref())?;
            let hd = lw.head_dim;
            let n_kv = lw.n_kv_heads;
            let kv_dim = kvl.kv_dim;
            let rope = if lw.is_swa {
                hp.rope_swa
            } else {
                hp.rope_global
            };
            let factors = (!lw.is_swa).then_some(&self.rope_factors);
            let window = if lw.is_swa { hp.swa_window } else { 0 };
            // QK score scale. gemma4 folds its query scale into the q-norm
            // weights and scores UNSCALED (f_attention_scale = 1.0);
            // muse-glimmer passes kq_scale = 1/sqrt(head_dim) on top of its
            // own q-norm weights. Hparams::attn_scale carries the difference.
            let ascale = hp.attn_scale(hd);

            // ── attention half
            exec.rmsnorm(&sc.x, &lw.attn_norm, &mut sc.normed, hp.n_embd, hp.eps)?;

            // Q: project -> per-head learned RMS norm -> rope
            let q_dim = hp.n_head * hd;
            // k-quant layer (every projection Kq): the normed row is
            // quantized once and one W4A8 launch reads it for q, k and v -
            // the K and V matches below then have nothing to do
            let kq_qkv = lw.wq.kq().is_some()
                && lw.wk.kq().is_some()
                && lw.wv.as_ref().is_none_or(|v| v.kq().is_some());
            match (&lw.f8a_wqkv, &lw.f8a_wq) {
                _ if kq_qkv => {
                    let st = kq_stage!(sc);
                    planes::kq_stage_x(exec, st, &sc.normed, hp.n_embd)?;
                    let mut segs: Vec<(&RepackedKQ, &mut CudaSlice<f32>)> = vec![
                        (lw.wq.kq().expect("kq_qkv"), &mut sc.q),
                        (lw.wk.kq().expect("kq_qkv"), &mut sc.k),
                    ];
                    if let Some(kv) = lw.wv.as_ref().and_then(|v| v.kq()) {
                        segs.push((kv, &mut sc.v));
                    }
                    planes::kq_gemv_multi(exec, &mut segs, st)?;
                    if lw.wv.is_none() {
                        exec.copy_slice(&sc.k, 0, kv_dim, &mut sc.v)?;
                    }
                }
                // fused plane: row-offset sub-views (out-row-major) keep the
                // oracle lane's separate buffers/epilogue bit-identical
                (Some(w8), _) if super::batch::fp4_on() => {
                    exec.fp4_gemv_at_off(w8, 0, &sc.normed, &mut sc.q, 0, hp.n_embd, q_dim)?
                }
                (Some(w8), _) if w8.is_lin() => {
                    // lin layout: per-32 quantize once here - the K/V arms
                    // below reuse pf_e4q/pf_e4s (nothing between touches
                    // them) - then box-offset K-split GEMMs per segment
                    exec.quantize_e4m3(&sc.normed, &mut sc.pf_e4q, &mut sc.pf_e4s, hp.n_embd)?;
                    exec.f8_gemm_lin(
                        w8,
                        0,
                        hp.n_embd,
                        q_dim,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_skfix,
                        &mut sc.q,
                        1,
                    )?;
                }
                (Some(w8), _) => {
                    exec.f8_gemv_at_off(w8, 0, &sc.normed, &mut sc.q, 0, hp.n_embd, q_dim)?
                }
                (None, Some(w8)) => {
                    exec.f8_gemv_at(w8, &sc.normed, &mut sc.q, 0, hp.n_embd, q_dim)?
                }
                // Q8-reclaim lane: the original was stubbed, the f8w prefill
                // plane serves the serial gemv (same q8_0_to_f8w class as f8a)
                (None, None) if lw.wq.is_stub() => {
                    if let Some(w8) = &lw.f8w_wq {
                        exec.f8_gemv_at(w8, &sc.normed, &mut sc.q, 0, hp.n_embd, q_dim)?;
                    } else {
                        // unified planes: the fused f8t qkv plane's
                        // row-tile sub-view serves the serial single row
                        exec.quantize_e4m3_row(
                            &sc.normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            hp.n_embd,
                            1,
                        )?;
                        if let Some(qkv) = &lw.f8t_qkv {
                            exec.f8t_gemm_off(
                                qkv,
                                0,
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                &mut sc.pf_skfix,
                                &mut sc.q,
                                hp.n_embd,
                                q_dim,
                                1,
                            )?;
                        } else {
                            exec.f8t_gemm(
                                lw.f8t_wq
                                    .as_ref()
                                    .expect("reclaim requires an f8t attn plane"),
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                &mut sc.pf_skfix,
                                &mut sc.q,
                                hp.n_embd,
                                q_dim,
                                1,
                            )?;
                        }
                    }
                }
                _ => lw.wq.gemv(exec, kq_stage!(sc), &sc.normed, &mut sc.q)?,
            }
            exec.rmsnorm_batch(&sc.q, &lw.q_norm, &mut sc.qn, hd, hp.eps, hp.n_head)?;
            exec.rope_factors_batch(
                &mut sc.qn,
                &sc.pos,
                factors,
                hp.n_head,
                hd,
                rope,
                1,
                hp.rope_neox(),
            )?;

            // K projection feeds both K and V on the V-less global layers -
            // V branches off the RAW projection (before K's learned norm/rope)
            match (&lw.f8a_wqkv, &lw.f8a_wk) {
                // the k-quant arm above wrote k (and v) already
                _ if kq_qkv => {}
                (Some(w8), _) if super::batch::fp4_on() => {
                    exec.fp4_gemv_at_off(w8, q_dim, &sc.normed, &mut sc.k, 0, hp.n_embd, kv_dim)?
                }
                // pf_e4q/pf_e4s still hold the Q arm's quantized normed row
                (Some(w8), _) if w8.is_lin() => exec.f8_gemm_lin(
                    w8,
                    q_dim,
                    hp.n_embd,
                    kv_dim,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_skfix,
                    &mut sc.k,
                    1,
                )?,
                (Some(w8), _) => {
                    exec.f8_gemv_at_off(w8, q_dim, &sc.normed, &mut sc.k, 0, hp.n_embd, kv_dim)?
                }
                (None, Some(w8)) => {
                    exec.f8_gemv_at(w8, &sc.normed, &mut sc.k, 0, hp.n_embd, kv_dim)?
                }
                (None, None) if lw.wk.is_stub() => {
                    if let Some(w8) = &lw.f8w_wk {
                        exec.f8_gemv_at(w8, &sc.normed, &mut sc.k, 0, hp.n_embd, kv_dim)?;
                    } else {
                        exec.quantize_e4m3_row(
                            &sc.normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            hp.n_embd,
                            1,
                        )?;
                        if let Some(qkv) = &lw.f8t_qkv {
                            exec.f8t_gemm_off(
                                qkv,
                                q_dim / 128,
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                &mut sc.pf_skfix,
                                &mut sc.k,
                                hp.n_embd,
                                kv_dim,
                                1,
                            )?;
                        } else {
                            exec.f8t_gemm(
                                lw.f8t_wk
                                    .as_ref()
                                    .expect("reclaim requires an f8t attn plane"),
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                &mut sc.pf_skfix,
                                &mut sc.k,
                                hp.n_embd,
                                kv_dim,
                                1,
                            )?;
                        }
                    }
                }
                _ => lw.wk.gemv(exec, kq_stage!(sc), &sc.normed, &mut sc.k)?,
            }
            match (&lw.wv, &lw.f8a_wqkv, &lw.f8a_wv) {
                _ if kq_qkv => {}
                (Some(_), Some(w8), _) if super::batch::fp4_on() => exec.fp4_gemv_at_off(
                    w8,
                    q_dim + kv_dim,
                    &sc.normed,
                    &mut sc.v,
                    0,
                    hp.n_embd,
                    kv_dim,
                )?,
                (Some(_), Some(w8), _) if w8.is_lin() => exec.f8_gemm_lin(
                    w8,
                    q_dim + kv_dim,
                    hp.n_embd,
                    kv_dim,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_skfix,
                    &mut sc.v,
                    1,
                )?,
                (Some(_), Some(w8), _) => exec.f8_gemv_at_off(
                    w8,
                    q_dim + kv_dim,
                    &sc.normed,
                    &mut sc.v,
                    0,
                    hp.n_embd,
                    kv_dim,
                )?,
                (Some(_), None, Some(v8)) => {
                    exec.f8_gemv_at(v8, &sc.normed, &mut sc.v, 0, hp.n_embd, kv_dim)?
                }
                (Some(wv), None, None) if wv.is_stub() => {
                    if let Some(v8) = &lw.f8w_wv {
                        exec.f8_gemv_at(v8, &sc.normed, &mut sc.v, 0, hp.n_embd, kv_dim)?;
                    } else {
                        exec.quantize_e4m3_row(
                            &sc.normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            hp.n_embd,
                            1,
                        )?;
                        if let Some(qkv) = &lw.f8t_qkv {
                            exec.f8t_gemm_off(
                                qkv,
                                (q_dim + kv_dim) / 128,
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                &mut sc.pf_skfix,
                                &mut sc.v,
                                hp.n_embd,
                                kv_dim,
                                1,
                            )?;
                        } else {
                            exec.f8t_gemm(
                                lw.f8t_wv
                                    .as_ref()
                                    .expect("reclaim requires an f8t attn plane"),
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                &mut sc.pf_skfix,
                                &mut sc.v,
                                hp.n_embd,
                                kv_dim,
                                1,
                            )?;
                        }
                    }
                }
                (Some(wv), None, None) => wv.gemv(exec, kq_stage!(sc), &sc.normed, &mut sc.v)?,
                (None, _, _) => exec.copy_slice(&sc.k, 0, kv_dim, &mut sc.v)?,
            }
            // K: learned per-head norm + rope; V: WEIGHTLESS per-head norm,
            // no rope (both exactly as build_gemma4 orders them)
            exec.rmsnorm_batch(&sc.k, &lw.k_norm, &mut sc.kn, hd, hp.eps, n_kv)?;
            exec.rope_factors_batch(
                &mut sc.kn,
                &sc.pos,
                factors,
                n_kv,
                hd,
                rope,
                1,
                hp.rope_neox(),
            )?;
            // V: weightless per-head RMS norm on gemma4, straight copy on
            // muse-glimmer (see Hparams::v_norm). Same fork the fused
            // epilogue's `vnorm` makes - this is the unfused twin of it.
            if hp.v_norm() {
                exec.rmsnorm_batch(&sc.v, &sc.ones, &mut sc.vn, hd, hp.eps, n_kv)?;
            } else {
                exec.copy_slice(&sc.v, 0, n_kv * hd, &mut sc.vn)?;
            }

            // gemma4 scores are UNSCALED (f_attention_scale = 1.0); no sinks
            // (SWA layers ride the WindowRing; global layers the budget pool
            // when enable_batch built one - slot 0, the single-stream slot)
            let layer_bt = if lw.is_swa {
                self.paging.as_ref().map(|pg| (&pg.bt, pg.bps))
            } else {
                self.gpool.as_ref().map(|gp| (&gp.d_bt, gp.bps))
            };
            match layer_bt {
                Some((bt, bps)) => {
                    exec.kv_append_batch_paged(
                        &sc.kn, &mut kvl.k, &sc.pos, None, bt, bps, kv_dim, 1, kvl.dtype,
                    )?;
                    exec.kv_append_batch_paged(
                        &sc.vn, &mut kvl.v, &sc.pos, None, bt, bps, kv_dim, 1, kvl.dtype,
                    )?;
                    exec.attn_decode_batch_paged(
                        &sc.qn,
                        &kvl.k,
                        &kvl.v,
                        &sc.neg_inf_sinks,
                        &mut sc.attn,
                        &sc.pos,
                        None,
                        bt,
                        bps,
                        hp.n_head,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        1,
                        ascale,
                        kvl.dtype,
                    )?;
                }
                _ => {
                    exec.kv_append_batch(
                        &sc.kn,
                        &mut kvl.k,
                        &sc.pos,
                        None,
                        kv_dim,
                        self.max_ctx,
                        1,
                        kvl.dtype,
                    )?;
                    exec.kv_append_batch(
                        &sc.vn,
                        &mut kvl.v,
                        &sc.pos,
                        None,
                        kv_dim,
                        self.max_ctx,
                        1,
                        kvl.dtype,
                    )?;
                    exec.attn_decode_batch(
                        &sc.qn,
                        &kvl.k,
                        &kvl.v,
                        &sc.neg_inf_sinks,
                        &mut sc.attn,
                        &sc.pos,
                        None,
                        hp.n_head,
                        n_kv,
                        hd,
                        self.max_ctx,
                        kv_dim,
                        window,
                        1,
                        ascale,
                        kvl.dtype,
                    )?;
                }
            }

            // sigmoid output gate (muse-glimmer) - must land before any wo arm,
            // several of which quantize sc.attn on the way in. sc.normed still
            // holds this layer's attn_norm output; the FFN pre-norm below is
            // what eventually overwrites it.
            Self::attn_gate_apply(
                exec,
                lw,
                &sc.normed,
                &mut sc.agate,
                &mut sc.attn,
                &mut sc.pf_xq,
                &mut sc.pf_xs,
                &mut sc.pf_yq,
                &mut sc.pf_skfix,
                &mut sc.pf_e4q,
                &mut sc.pf_e4rs,
                hp.n_embd,
                hp.n_head * hd,
                1,
            )?;
            match &lw.f8a_wo {
                Some(w8) if super::batch::fp4_on() => exec.fp4_gemv_at_off(
                    w8,
                    0,
                    &sc.attn,
                    &mut sc.proj,
                    0,
                    hp.n_head * hd,
                    hp.n_embd,
                )?,
                Some(w8) if w8.is_lin() => {
                    exec.quantize_e4m3(&sc.attn, &mut sc.pf_e4q, &mut sc.pf_e4s, hp.n_head * hd)?;
                    exec.f8_gemm_lin(
                        w8,
                        0,
                        hp.n_head * hd,
                        hp.n_embd,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_skfix,
                        &mut sc.proj,
                        1,
                    )?;
                }
                Some(w8) => {
                    exec.f8_gemv_at(w8, &sc.attn, &mut sc.proj, 0, hp.n_head * hd, hp.n_embd)?
                }
                None if lw.wo.is_stub() => {
                    if let Some(w8) = &lw.f8w_wo {
                        exec.f8_gemv_at(w8, &sc.attn, &mut sc.proj, 0, hp.n_head * hd, hp.n_embd)?;
                    } else {
                        exec.quantize_e4m3_row(
                            &sc.attn,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            hp.n_head * hd,
                            1,
                        )?;
                        exec.f8t_gemm(
                            lw.f8t_wo.as_ref().expect("reclaim requires f8t_wo"),
                            &sc.pf_e4q,
                            &sc.pf_e4rs,
                            &mut sc.pf_skfix,
                            &mut sc.proj,
                            hp.n_head * hd,
                            hp.n_embd,
                            1,
                        )?;
                    }
                }
                None => lw.wo.gemv(exec, kq_stage!(sc), &sc.attn, &mut sc.proj)?,
            }
            // fused post-norm + residual - the same kernel (and reduction
            // order) as the batched walk and the prefill lane, so all three
            // lanes stay ULP-aligned (separate ops here flipped near-ties
            // in gemma4_batch_check when prefill fused)
            exec.rmsnorm_add_scale(
                &mut sc.x,
                &sc.proj,
                &lw.attn_post_norm,
                hp.n_embd,
                hp.post_norm_eps,
                1.0,
                1,
            )?;

            // ── FFN half: parallel GEGLU
            exec.rmsnorm(&sc.x, &lw.ffn_norm, &mut sc.normed, hp.n_embd, hp.eps)?;
            let n_ff = lw.ffn_gate.dims()[1];
            if let Some(f8_gu) = &lw.f8_gu
                && lw.ffn_gate.q8_len() <= 48
            {
                // F8R fused gate|up plane (verify-GEMM dedup): one gemv
                // lands the concatenated [gate|up] row, geglu_pair folds it
                // in place - same values as the split gemvs (same kernel,
                // concatenated weights; geglu_pair == geglu formula)
                if super::batch::fp4_on() {
                    exec.fp4_gemv_at_off(
                        f8_gu,
                        0,
                        &sc.normed,
                        &mut sc.pf_gate,
                        0,
                        hp.n_embd,
                        2 * n_ff,
                    )?;
                    exec.glu_pair(&mut sc.pf_gate, n_ff, 1, hp.glu_act())?;
                    exec.fp4_gemv_at_off(
                        lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                        0,
                        &sc.pf_gate,
                        &mut sc.proj,
                        0,
                        n_ff,
                        hp.n_embd,
                    )?;
                } else if f8_gu.is_lin() {
                    // lin trio (all-or-nothing with down): the batched mma_ks
                    // band's exact chain at r=1 - quantize, fused [gate|up]
                    // GEMM, fused geglu+quant, down GEMM. gu_il = the plane
                    // rows are interleaved -> pair-addressed geglu twin
                    exec.quantize_e4m3(&sc.normed, &mut sc.pf_e4q, &mut sc.pf_e4s, hp.n_embd)?;
                    exec.f8_gemm_lin(
                        f8_gu,
                        0,
                        hp.n_embd,
                        2 * n_ff,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_skfix,
                        &mut sc.pf_gate,
                        1,
                    )?;
                    if lw.gu_il {
                        exec.quantize_e4m3_glu2i(
                            &sc.pf_gate,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4s,
                            n_ff,
                            1,
                            hp.glu_act(),
                        )?;
                    } else {
                        exec.quantize_e4m3_glu2(
                            &sc.pf_gate,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4s,
                            n_ff,
                            1,
                            hp.glu_act(),
                        )?;
                    }
                    exec.f8_gemm_lin(
                        lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                        0,
                        n_ff,
                        hp.n_embd,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_skfix,
                        &mut sc.proj,
                        1,
                    )?;
                } else {
                    exec.f8_gemv_at(f8_gu, &sc.normed, &mut sc.pf_gate, 0, hp.n_embd, 2 * n_ff)?;
                    exec.glu_pair(&mut sc.pf_gate, n_ff, 1, hp.glu_act())?;
                    exec.f8_gemv_at(
                        lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                        &sc.pf_gate,
                        &mut sc.proj,
                        0,
                        n_ff,
                        hp.n_embd,
                    )?;
                }
            } else if let Some(gate8) = &lw.f8_gate
                && lw.ffn_gate.q8_len() <= 48
            {
                // F8R: e4m3 gemvs (f32 x, bandwidth-floor parity with q8)
                exec.f8_gemv_at(gate8, &sc.normed, &mut sc.gate, 0, hp.n_embd, n_ff)?;
                exec.f8_gemv_at(
                    lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                    &sc.normed,
                    &mut sc.up,
                    0,
                    hp.n_embd,
                    n_ff,
                )?;
                exec.glu(&mut sc.gate, &sc.up, n_ff, hp.glu_act())?;
                exec.f8_gemv_at(
                    lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                    &sc.gate,
                    &mut sc.proj,
                    0,
                    n_ff,
                    hp.n_embd,
                )?;
            } else if let Some(f8t_gu) = &lw.f8t_gu
                && lw.ffn_gate.q8_len() <= 48
            {
                // unified planes: the f8t fused chain at r=1 -
                // same [gate|up] -> geglu2-quant -> down as the batched arm
                exec.quantize_e4m3_row(&sc.normed, &mut sc.pf_e4q, &mut sc.pf_e4rs, hp.n_embd, 1)?;
                exec.f8t_gemm(
                    f8t_gu,
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_gate,
                    hp.n_embd,
                    2 * n_ff,
                    1,
                )?;
                exec.quantize_e4m3_glu2_row(
                    &sc.pf_gate,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    n_ff,
                    1,
                    hp.glu_act(),
                )?;
                exec.f8t_gemm(
                    lw.f8t_down.as_ref().expect("f8t FFN planes built as a set"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_skfix,
                    &mut sc.proj,
                    n_ff,
                    hp.n_embd,
                    1,
                )?;
            } else {
                debug_assert!(!lw.ffn_gate.is_stub(), "stubbed ffn without f8 plane");
                if let (Some(g), Some(u)) = (lw.ffn_gate.kq(), lw.ffn_up.kq()) {
                    // k-quant gate|up off one staged row, one launch
                    let st = kq_stage!(sc);
                    planes::kq_stage_x(exec, st, &sc.normed, hp.n_embd)?;
                    planes::kq_gemv_multi(exec, &mut [(g, &mut sc.gate), (u, &mut sc.up)], st)?;
                } else {
                    lw.ffn_gate
                        .gemv(exec, kq_stage!(sc), &sc.normed, &mut sc.gate)?;
                    lw.ffn_up
                        .gemv(exec, kq_stage!(sc), &sc.normed, &mut sc.up)?;
                }
                exec.glu(&mut sc.gate, &sc.up, n_ff, hp.glu_act())?;
                lw.ffn_down
                    .gemv(exec, kq_stage!(sc), &sc.gate, &mut sc.proj)?;
            }
            // fused post-norm + residual + layer_output_scale (see above);
            // 26B-A4B layers route through the hybrid two-branch tail
            if let Some(moe) = &lw.moe {
                super::batch::g4_moe_tail(
                    exec,
                    sc,
                    moe,
                    &lw.ffn_post_norm,
                    hp,
                    lw.out_scale,
                    1,
                    false,
                )?;
            } else {
                exec.rmsnorm_add_scale(
                    &mut sc.x,
                    &sc.proj,
                    &lw.ffn_post_norm,
                    hp.n_embd,
                    hp.post_norm_eps,
                    lw.out_scale,
                    1,
                )?;
            }
        }

        // ── final norm -> LM head -> logit scale -> softcap (host: already read back)
        // head rides the f8t tile plane where built (one logits class per
        // binary - the batched tick's arm), else the REPACKED Q8 plane (884us
        // vs the raw-plane gemv's 1693us on the same 1.5GB read)
        exec.rmsnorm(&sc.x, &self.output_norm, &mut sc.normed, hp.n_embd, hp.eps)?;
        if let Some(ht) = self.head_f8t.as_ref() {
            exec.quantize_e4m3_row(&sc.normed, &mut sc.pf_e4q, &mut sc.pf_e4rs, hp.n_embd, 1)?;
            exec.f8t_gemm(
                ht,
                &sc.pf_e4q,
                &sc.pf_e4rs,
                &mut sc.pf_skfix,
                &mut sc.logits,
                hp.n_embd,
                hp.n_vocab,
                1,
            )?;
        } else {
            self.head
                .gemm(exec, kq_rows!(sc), &sc.normed, &mut sc.logits, 1)?;
        }
        let mut logits = exec.to_host(&sc.logits)?;
        hp.logit_epilogue(&mut logits);

        self.pos = pos + 1;
        Ok(logits)
    }

    /// Whole-prompt batched prefill (single stream, slot 0): PF_ROWS-row
    /// chunks through the same graph as `step`, one weight read per chunk
    /// instead of one per token. Returns the last token's logits.
    ///
    /// Attention per geometry: SWA layers (head_dim 256) ride the tiled
    /// `attn_prefill`; the 10 global layers (head_dim 512 - outside the
    /// prefill tiles' 128/256 set) ride `attn_decode_batch` with all rows
    /// mapped to slot 0, which is per-row causal prefill attention (each row
    /// bounds its walk at its own position), just decode-class perf. Fine:
    /// 10 of 60 layers, and a dedicated 512 prefill tile is a later lever.
    pub(crate) fn prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GpuError> {
        let mut base = self.pos;
        // pool mode: slot 0 (the single-stream slot) - fresh sequences
        // return their old blocks first, continuations just grow
        if base == 0 {
            self.gpool_clear_slot(0);
        }
        self.ensure_global_rows(&[0], &[(base + tokens.len() - 1) as u32])?;
        for chunk in tokens.chunks(self.pf_rows) {
            self.prefill_chunk(chunk, base)?;
            base += chunk.len();
        }
        self.pos = base;
        self.logits_from_pf_row((tokens.len() - 1) % self.pf_rows)
    }

    /// Shared prefill tail: single-row norm -> tied head -> host softcap on
    /// row `last` of the prefill stream buffer.
    pub(crate) fn logits_from_pf_row(&mut self, last: usize) -> Result<Vec<f32>, GpuError> {
        let n_embd = self.hp.n_embd;
        let (sc, hp) = (&mut self.scratch, &self.hp);
        self.exec
            .copy_region(&sc.pf_x, last * n_embd, &mut sc.x, 0, n_embd)?;
        self.exec
            .rmsnorm(&sc.x, &self.output_norm, &mut sc.normed, n_embd, hp.eps)?;
        if let Some(ht) = self.head_f8t.as_ref() {
            self.exec
                .quantize_e4m3_row(&sc.normed, &mut sc.pf_e4q, &mut sc.pf_e4rs, n_embd, 1)?;
            self.exec.f8t_gemm(
                ht,
                &sc.pf_e4q,
                &sc.pf_e4rs,
                &mut sc.pf_skfix,
                &mut sc.logits,
                n_embd,
                self.hp.n_vocab,
                1,
            )?;
        } else {
            self.head
                .gemm(&self.exec, kq_rows!(sc), &sc.normed, &mut sc.logits, 1)?;
        }
        let mut logits = self.exec.to_host(&sc.logits)?;
        hp.logit_epilogue(&mut logits);
        Ok(logits)
    }

    /// Deferred twin of `logits_from_pf_row`: same copy/rmsnorm/gemv chain,
    /// but the head lands in pf_fin[out_idx] with no sync - callers read all
    /// staged rows back with `logits_finish_read_all` after their chunk loop.
    /// Softcap runs on device at stage time (the host tanh over
    /// 262k floats was ~1.5ms per FINISHER inside the mixed wait's [mix-wait]
    /// 18ms window; same numeric class as the device-sampled lane's softcap).
    pub(crate) fn logits_head_stage(
        &mut self,
        last: usize,
        out_idx: usize,
    ) -> Result<(), GpuError> {
        let n_embd = self.hp.n_embd;
        let (sc, hp) = (&mut self.scratch, &self.hp);
        self.exec
            .copy_region(&sc.pf_x, last * n_embd, &mut sc.x, 0, n_embd)?;
        self.exec
            .rmsnorm(&sc.x, &self.output_norm, &mut sc.normed, n_embd, hp.eps)?;
        if let Some(ht) = self.head_f8t.as_ref() {
            self.exec
                .quantize_e4m3_row(&sc.normed, &mut sc.pf_e4q, &mut sc.pf_e4rs, n_embd, 1)?;
            self.exec.f8t_gemm(
                ht,
                &sc.pf_e4q,
                &sc.pf_e4rs,
                &mut sc.pf_skfix,
                &mut sc.logits,
                n_embd,
                self.hp.n_vocab,
                1,
            )?;
        } else {
            self.head
                .gemm(&self.exec, kq_rows!(sc), &sc.normed, &mut sc.logits, 1)?;
        }
        let vocab = self.hp.n_vocab;
        Self::logit_epilogue_dev(&self.exec, &mut sc.logits, vocab, hp)?;
        self.exec
            .copy_region(&sc.logits, 0, &mut sc.pf_fin, out_idx * vocab, vocab)?;
        Ok(())
    }

    /// batched twin of `logits_head_stage` - N finishers in one
    /// chain (gather rows -> fused rmsnorm+quantize at r=N -> one m=N head
    /// GEMM straight into pf_fin -> one elementwise epilogue over N*vocab).
    /// The per-item path ran 32 m=1 wmma head launches (~6.6ms of the c32
    /// burst pass tail). f8t head only; callers fall back per-item else.
    pub(crate) fn logits_head_stage_batch(
        &mut self,
        items: &[(usize, usize)],
    ) -> Result<(), GpuError> {
        let n_embd = self.hp.n_embd;
        let vocab = self.hp.n_vocab;
        let n = items.len();
        {
            let sc = &mut self.scratch;
            for (i, &(last, _)) in items.iter().enumerate() {
                self.exec.copy_region(
                    &sc.pf_x,
                    last * n_embd,
                    &mut sc.pf_normed,
                    i * n_embd,
                    n_embd,
                )?;
            }
        }
        let (sc, hp) = (&mut self.scratch, &self.hp);
        if let Some(ht) = self.head_f8t.as_ref() {
            self.exec.rmsnorm_e4m3_row(
                &sc.pf_normed,
                &self.output_norm,
                &mut sc.pf_e4q,
                &mut sc.pf_e4rs,
                n_embd,
                hp.eps,
                n,
            )?;
            self.exec.f8t_gemm(
                ht,
                &sc.pf_e4q,
                &sc.pf_e4rs,
                &mut sc.pf_skfix,
                &mut sc.pf_fin,
                n_embd,
                vocab,
                n,
            )?;
        } else {
            // f8row/wmma head: norm rows in place then one m=N gemm
            for i in 0..n {
                self.exec
                    .copy_region(&sc.pf_normed, i * n_embd, &mut sc.x, 0, n_embd)?;
                self.exec
                    .rmsnorm(&sc.x, &self.output_norm, &mut sc.normed, n_embd, hp.eps)?;
                self.exec
                    .copy_region(&sc.normed, 0, &mut sc.pf_normed, i * n_embd, n_embd)?;
            }
            if let Some(hq) = self.head.q8() {
                // the repacked plane walk is ~650us per FINISHER at
                // m=N (grid=vocab, ~136 GB/s effective - 22.7ms of the 88ms
                // c32 admission wave on muse, where vocab % 128 blocks f8t).
                // The decode tick's own 2..=192 head rung (quantize_q8 +
                // mma_ks) runs the same plane at r=32 in ~430us; finisher
                // logits join the class every decode-step token already uses.
                self.exec
                    .quantize_q8(&sc.pf_normed, &mut sc.pf_xq, &mut sc.pf_xs, n * n_embd)?;
                self.exec.q8_0_gemm_mma_ks(
                    hq,
                    &sc.pf_xq,
                    &sc.pf_xs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_fin,
                    n,
                )?;
            } else {
                self.head
                    .gemm(&self.exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_fin, n)?;
            }
        }
        // rows land in item order: pf_fin[i*vocab..] == items[i].1's slot -
        // callers pass out_idx == i (the pure-prefill tail's convention)
        for (i, &(_, oi)) in items.iter().enumerate() {
            debug_assert_eq!(i, oi, "batched head needs identity out_idx");
        }
        Self::logit_epilogue_dev_len(&self.exec, &mut sc.pf_fin, n * vocab, hp)
    }

    /// Length-generic twin of logit_epilogue_dev (scale+softcap are
    /// elementwise - one launch over the whole staged prefix).
    fn logit_epilogue_dev_len(
        exec: &GpuExecutor,
        logits: &mut CudaSlice<f32>,
        len: usize,
        hp: &super::Hparams,
    ) -> Result<(), GpuError> {
        super::logit_epilogue_dev(exec, logits, len, hp.logit_scale, hp.final_softcap)
    }

    /// Batched finisher readback: one stream sync + one dtoh over
    /// the staged prefix of pf_fin, split per staged index - replaces N
    /// sequential logits_finish_read calls (each a sync + 1MB pageable dtoh;
    /// 8 finishers cost ~18ms serialized in the mixed wait, the largest
    /// single slice of the 128x128 boundary idle).
    pub(crate) fn logits_finish_read_all(
        &mut self,
        staged: &[bool],
    ) -> Result<Vec<Option<Vec<f32>>>, GpuError> {
        let vocab = self.hp.n_vocab;
        let hi = staged.iter().rposition(|&s| s).map_or(0, |i| i + 1);
        if hi == 0 {
            return Ok(vec![None; staged.len()]);
        }
        let view = self
            .scratch
            .pf_fin
            .try_slice(0..hi * vocab)
            .ok_or_else(|| GpuError::Driver("pf_fin slice".into()))?;
        self.exec
            .stream
            .synchronize()
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        let flat = self
            .exec
            .stream
            .clone_dtoh(&view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        Ok(staged
            .iter()
            .enumerate()
            .map(|(i, &s)| (s && i < hi).then(|| flat[i * vocab..(i + 1) * vocab].to_vec()))
            .collect())
    }

    /// One prefill chunk at positions [base, base+r) into the slot the
    /// caller has staged in `pf_slots` (slot 0 for the single-stream path;
    /// `forward_prefill` refills it per target slot). Does not touch
    /// `self.pos` - the single-stream path owns that bookkeeping.
    pub(crate) fn prefill_chunk(&mut self, toks: &[u32], base: usize) -> Result<(), GpuError> {
        let r = toks.len();
        let hp = &self.hp;
        let sc = &mut self.scratch;
        let exec = &self.exec;

        let positions: Vec<u32> = (0..r).map(|i| (base + i) as u32).collect();
        exec.stream
            .memcpy_htod(&positions, &mut sc.pf_pos)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        // text-only prefill: attention bounds == real positions
        exec.stream
            .memcpy_htod(&positions, &mut sc.pf_attn_pos)
            .map_err(|e| GpuError::Driver(e.to_string()))?;

        // rows: per-token dequant into pf_tmp, then one √n_embd scale
        let embd = EmbdTable::of(&self.token_embd, &self.head);
        for (i, &t) in toks.iter().enumerate() {
            embd.row(exec, t, &mut sc.embd_id, &mut sc.pf_row, hp.n_embd)?;
            exec.copy_region(&sc.pf_row, 0, &mut sc.pf_tmp, i * hp.n_embd, hp.n_embd)?;
        }
        exec.stream
            .memset_zeros(&mut sc.pf_x)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        exec.scale_add(&mut sc.pf_x, &sc.pf_tmp, hp.embd_scale(), r * hp.n_embd)?;
        Self::embd_preamble(exec, hp, self.embd_ones.as_ref(), &mut sc.pf_x, r)?;

        self.prefill_layers(r, &[(0, r)], &swa_spans(self.swa_span, &[(0, r)]), 0)
    }

    /// The shared prefill layer walk over `r` pre-assembled rows in `pf_x`
    /// (positions in `pf_pos`, attention BOUNDS in `pf_attn_pos` - they
    /// differ only on multimodal image spans, which attend non-causally
    /// through their span). `runs` are contiguous same-slot row spans
    /// (row_off, rows): the coalesced multi-prompt prefill dispatches
    /// attention per run because the tiled prefill kernels read slots[0];
    /// everything else in the walk is per-row (slots array) and shared.
    /// Single-prompt callers pass `[(0, r)]` - identical launches to before.
    ///
    /// `decode_rows`: the UNIFIED tick lays nd live decode rows at the FRONT
    /// of the stream (rows [0, nd), one per decoding slot) so they share the
    /// chunk's GEMMs (one weight walk for prefill AND decode) while their
    /// attention dispatches through the DECODE kernels natively (rows start
    /// at buffer offset 0 - no pointer surgery). `runs`/`spans` must exclude
    /// those rows. 0 = plain prefill (all existing callers).
    pub(crate) fn prefill_layers(
        &mut self,
        r: usize,
        runs: &[(usize, usize)],
        spans: &[(usize, usize)],
        decode_rows: usize,
    ) -> Result<(), GpuError> {
        // The one choke point every row-bearing pass goes through, so this is
        // where the scratch bound is enforced. `pf_rows` sizes the [rows, dim]
        // planes below and every caller chunks at it; a pass that arrived here
        // wider than the allocation would write past them - silent corruption,
        // not a slow path. Cheap enough to be unconditional.
        if r > self.pf_rows {
            return Err(GpuError::Driver(format!(
                "prefill pass of {r} rows exceeds the {}-row prefill scratch \
                 (ctx {}); a lane is chunking on something other than pf_rows",
                self.pf_rows, self.max_ctx,
            )));
        }
        // this walk overwrites pf_normed without recording rows - the MTP h
        // map would point at garbage; drop it (self-heals next sampled tick)
        for e in self.spec_rows.iter_mut() {
            *e = None;
        }
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!("[hwipe] r={r}");
        }
        //  batched-runs pf5 (PADDOCK_PF_RUNS=1): arm the per-pass
        // run table so the sliding paged attention launches once with
        // grid.z over spans instead of 32 underfilled per-span launches
        // (43ms of the 372ms c32 burst pass). Engine gate keeps the
        // per-span loop wherever another launcher arm could win the
        // election (a16 sliding, globals, dense fallbacks).
        let pf_runs_batched = crate::envset::env_on("PADDOCK_PF_RUNS")
            && decode_rows == 0
            && spans.len() > 1
            && spans.len() <= 64
            && self.paging.is_some()
            && self.gpool.is_some()
            && self.exec.kernels_pf_runs_available();
        if pf_runs_batched {
            debug_assert!(
                spans.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0),
                "spans must be contiguous"
            );
            let mut offs: Vec<u32> = spans.iter().map(|s| s.0 as u32).collect();
            let last = spans.last().expect("spans non-empty (pf_runs_batched)");
            offs.push((last.0 + last.1) as u32);
            let sc = &mut self.scratch;
            let mut v = sc
                .pf_runs
                .try_slice_mut(0..offs.len())
                .ok_or_else(|| GpuError::Driver("pf_runs slice".into()))?;
            self.exec
                .stream
                .memcpy_htod(&offs, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            let maxn = spans
                .iter()
                .map(|s| s.1)
                .max()
                .expect("spans non-empty (pf_runs_batched)") as u32;
            self.exec
                .pf_runs_register(Some((&sc.pf_runs, spans.len() as u32, maxn)))?;
        }
        // Sliding layers read each row's window from its TRUE position
        // (`pf_pos`) while the causal ceiling stays the row's BOUND
        // (`pf_attn_pos`). They coincide on text rows; on a bidirectional
        // span - an image span, the diffusion canvas - the bound sits at the
        // span end, and deriving the floor from it dropped up to span-1 of
        // the row's oldest keys. A pack without the win_pos entries keeps
        // the old floor (exact for every causal row).
        let wp_ok = self.exec.has_win_pos();
        // Cross-lane overlap for the eager chunk walk: per SWA layer,
        // decode-row attention (~270 us) and the k/v projection GEMM tail
        // waves would otherwise run serialized
        // after work they don't depend on - disjoint rows, disjoint KV
        // pages, disjoint outputs. A forked lane + two device events per
        // layer overlaps them into the main lane's tail waves. Same
        // kernels, same inputs - bit-identical by construction; the only
        // change is stream placement. This walk is never graph-captured
        // (host memcpys precede every call site), so the fork/event pair
        // is legal here. Kill: PADDOCK_G4_NO_PF_OVERLAP=1.
        //
        // Row floor: at small chunk r there is nothing to hide - v9q and the
        // k/v tails are tiny - but the ~120 cross-stream event edges per pass
        // still gap the GPU, so short prompts lose from the overlap. Only
        // genuinely wide chunks engage the lane. Tune:
        // PADDOCK_G4_PF_OVERLAP_MIN.
        //
        // Decode-rows floor, same reason from the other side: a wide chunk
        // carrying only ~7 decode rows has a trivial v9q and non-trivial
        // event edges, and loses too. The lane needs both a wide chunk and a
        // real decode complement.
        let ov_floor = super::batch::pf_overlap_min();
        let overlap_on = paddock_models::dev_var_os!("PADDOCK_G4_NO_PF_OVERLAP").is_none()
            && r >= ov_floor
            && decode_rows >= 16;
        if overlap_on && self.pf_side.is_none() {
            // one-time: fork_stream drains the parent, so everything the
            // serve has uploaded so far is visible to the new lane
            self.pf_side = Some(self.exec.fork_stream()?);
            tracing::info!("gemma4: prefill overlap lane forked");
        }
        // verify-fold rung A: the spec-mixed caller arms
        // spec_k1/spec_long for this pass when the front decode rows are
        // UNIFORM k1-deep verify chunks - the decode arms below then route
        // them through the spec attention kernels (per-layer width gate)
        // instead of k1 per-row window re-walks. Copied out here before
        // the field borrows.
        let deco_k1 = if decode_rows > 0 {
            self.spec_k1
                .filter(|&k1| k1 > 1 && decode_rows.is_multiple_of(k1))
        } else {
            None
        };
        let deco_long = self.spec_long;
        let n_slots_cap = self.n_slots;
        let hp = &self.hp;
        let sc = &mut self.scratch;
        let exec = &self.exec;
        let side = if overlap_on {
            self.pf_side.as_ref()
        } else {
            None
        };
        // DFlash feature taps  - the prefill twin of the decode
        // walk's. Same contract: pf_x at the top of a tapped layer is that
        // layer's input residual, which holds because muse never defers a
        // post-norm (every gated layer answers `fused_norm_ok() == false`).
        let mut dtap = self.dflash.as_mut().filter(|d| d.state.is_some());
        for (li, lw) in self.layers.iter().enumerate() {
            if let Some(df) = dtap.as_mut()
                && let Some(band) = df.target_layers.iter().position(|&t| t == li)
            {
                super::dflash::tap_band(exec, df, &sc.pf_x, band, hp.n_embd, r)?;
            }
            let kvl = &mut self.kv[li];
            // per-layer twin registration before any append enqueue
            self.exec.vdim_set(kvl.vdim.as_ref())?;
            let hd = lw.head_dim;
            let n_kv = lw.n_kv_heads;
            let kv_dim = kvl.kv_dim;
            let rope = if lw.is_swa {
                hp.rope_swa
            } else {
                hp.rope_global
            };
            let factors = (!lw.is_swa).then_some(&self.rope_factors);
            let window = if lw.is_swa { hp.swa_window } else { 0 };
            // QK score scale. gemma4 folds its query scale into the q-norm
            // weights and scores UNSCALED (f_attention_scale = 1.0);
            // muse-glimmer passes kq_scale = 1/sqrt(head_dim) on top of its
            // own q-norm weights. Hparams::attn_scale carries the difference.
            let ascale = hp.attn_scale(hd);

            // v11 norms rung: fused rmsnorm->e4m3 when the qkv prefill arm
            // (f8a on sm_120, f8w duplicates on cc-10) consumes only the
            // quantized form; the arm's own quantize is skipped below.
            // Bit-identical (same width policy as rmsnorm_batch). The f8w
            // extension is the fix - the v22 fusion was wired into
            // batch_step_body, which never runs the r>=65 band in this
            // workload; prefill_chunk is the path that counts.
            // f8t rowwise attn at prefill: the concat plane serves
            // q/k/v via row-tile sub-views (f8t_gemm_off) - the decode
            // band's class at every r. Kill: PADDOCK_G4_NO_F8TPF.
            let f8t_att = lw.f8t_qkv.is_some()
                && lw.f8a_wq.is_none()
                && lw.f8a_wqkv.is_none()
                && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8TPF").is_none();
            // (fused_norm_ok: a gated layer needs the f32 pf_normed these arms
            // skip writing - see LayerWeights::fused_norm_ok)
            let nqf_row_att = f8t_att
                && lw.fused_norm_ok()
                && exec.has_rmsnorm_e4m3_row()
                && super::batch::nqfuse_on();
            let nqf_attn = (lw.f8a_wq.is_some() || lw.f8a_wqkv.is_some() || lw.f8w_wq.is_some())
                && !f8t_att
                && lw.fused_norm_ok()
                && super::batch::nqfuse_on()
                && exec.has_rmsnorm_e4m3_batch();
            // pc engagement for the qkv chunk route (fix: the
            // producer emits ROW scales directly - no duplicate norm+quant
            // pass; the 28d gate measured the duplication eating the twin's
            // win). Must match the consumer arms below exactly.
            let pc_qkv = lw.qkv_ws.is_some()
                && r >= super::batch::pc_floor()
                && lw.f8a_wqkv.is_some()
                && !super::batch::fp4_on()
                && lw.fused_norm_ok()  // its producer is a norm-fusion arm too
                && exec.has_f8_gemm_w8_pc()
                && exec.has_rmsnorm_e4m3_row();
            if nqf_row_att || pc_qkv {
                exec.rmsnorm_e4m3_row(
                    &sc.pf_x,
                    &lw.attn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_embd,
                    hp.eps,
                    r,
                )?;
            } else if nqf_attn {
                exec.rmsnorm_e4m3_batch(
                    &sc.pf_x,
                    &lw.attn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4s,
                    hp.n_embd,
                    r,
                    hp.eps,
                )?;
            } else {
                exec.rmsnorm_batch(
                    &sc.pf_x,
                    &lw.attn_norm,
                    &mut sc.pf_normed,
                    hp.n_embd,
                    hp.eps,
                    r,
                )?;
            }

            // int8-mma GEMMs (one activation quantize feeds q/k/v) - llama's
            // own Q8 prefill numeric class. Always on in the prefill lane
            // (no batch floor): a prefix-cache resume re-prefills short
            // tails, and per-row-independent mmq keeps those rows BIT-EQUAL
            // to the same rows of a cold full-prompt prefill - the floor
            // made warm-vs-cold greedy flip on near-ties (found live).
            let mmq = true;
            // Rowwise-e4m3 prefill class (see the LayerWeights field note):
            // when the planes exist it takes every r in this lane - the
            // warm-vs-cold bit-equal invariant that forced mmq's no-floor
            // rule applies unchanged (per-row-independent quantize + GEMM),
            // so the class boundary must not depend on chunk size.
            let f8w_pf = lw.f8w_wq.is_some();
            let f8row_pf = lw.f8_wq.is_some();
            // F8A (sm_120 replace design) and f8w (cc-10 duplicates) share
            // the per-32 class + kernels; at most one set is built per box.
            let f8a_pf = lw.f8a_wq.is_some() || lw.f8a_wqkv.is_some();
            // fused-plane prefill rides row-offset sub-views: dense per-
            // projection outputs, zero epilogue changes (see f8_gemm_w8_off)
            let qkv8 = lw.f8a_wqkv.as_ref();
            // K and V get their own plane gates rather than reusing wq's.
            // A layer whose k/v ship bf16 (LayerWeights::kv_q8) has no fp8
            // twin for them while q/o still do, so the q-derived flag would
            // send the K arm into an `f8a_wk.unwrap()` on a None.
            let k_f8a = f8a_pf && (qkv8.is_some() || lw.f8a_wk.is_some());
            let k_f8w = f8w_pf && lw.f8w_wk.is_some();
            let k_f8row = f8row_pf && lw.f8_wk.is_some();
            let v_f8a = f8a_pf && (qkv8.is_some() || lw.f8a_wv.is_some());
            let v_f8w = f8w_pf && lw.f8w_wv.is_some();
            let v_f8row = f8row_pf && lw.f8_wv.is_some();
            // fused epilogue norms + rope (hoisted for the lane
            // gate below): five launches -> one, issued after the V GEMM
            let qk_fused = exec.has_qkv_norm_rope_batch()
                && paddock_models::dev_var_os!("PADDOCK_G4_NO_QKNF").is_none();
            // kv-epilogue fold: the K/V halves of the fused norm
            // pass move into the append - pd_kv_nra_rows reads the raw GEMM
            // planes and writes the caches directly, so the kn/vn planes
            // (and the V-less raw-k copy) never land; the v2 launch below
            // runs q-only. Paged arms only (the dense fallbacks keep the
            // plane chain). Kill: PADDOCK_G4_NO_KVF.
            let kvf = qk_fused
                && exec.has_kv_nra_rows()
                && super::batch::kvf_on()
                && if lw.is_swa {
                    self.paging.is_some()
                } else {
                    self.gpool.is_some()
                };
            // single fused qkv launch on the pc rowwise plane -
            // at admission-M the three segment GEMMs run 1.5-3-wave grids
            // at 1 CTA/SM, each paying its own ramp + straggler-wave chain
            // (q 571 / k,v 564 TF vs kt4a's 627 on wide grids); one
            // 128-tile-wide grid over the whole plane pays one tail.
            // Bit-exact vs the split launches (same mainloop per tile,
            // same absolute box/scale rows). Kill: PADDOCK_G4_NO_QKV1.
            let qkv_fused1 = pc_qkv
                && lw.wv.is_some()
                && exec.has_f8_gemm_w8_pc_qkv()
                && paddock_models::dev_var_os!("PADDOCK_G4_NO_QKV1").is_none();
            let n_ff = lw.ffn_gate.dims()[1];
            // f8 FFN lane (PADDOCK_G4_F8, r>1024 = the pipe class): TMA
            // block-scale W8A8 at 1.43x the q8 pipe. Lossy (e4m3 weights +
            // activations) - quality-gated; q8 lanes below stay bit-classic.
            // (This gate block lives up here, before the qkv section, since
            // the c16 stream gate below needs pc_gu.)
            let has_gu = lw.f8_gu.is_some() && exec.has_quantize_e4m3_glu2(hp.glu_act());
            let f8r = (has_gu || lw.f8_gate.is_some()) && lw.ffn_gate.q8_len() <= 32;
            // F8R e4m3-GEMM ladder: mma_ks twin 2..=31 (the TMA tile pays
            // ~2x there), TMA GEMM from 32; old packs without the twin keep
            // the r>=4 TMA cut. Non-F8R keeps the r>1024 pipe-class opt-in.
            // lin gu planes pull r==1 into the twin band too (the gemv arm
            // below reads f32 activations and can't take lin boxes; the
            // twin wrapper dispatches them onto pd_f8_gemm_lin).
            let gu_lin = has_gu && lw.f8_gu.as_ref().expect("f8_gu present (has_gu)").is_lin();
            let f8ks =
                f8r && ((2..=31).contains(&r) || (r == 1 && gu_lin)) && exec.has_f8_gemm_mma_ks();
            let f8 = (has_gu || lw.f8_gate.is_some())
                && (if f8r {
                    f8ks || r >= if exec.has_f8_gemm_mma_ks() { 32 } else { 4 }
                } else {
                    r > 1024
                });
            // v11 norms rung: fuse the FFN input norm+quantize when the f8
            // GEMM arm consumes only the e4m3 form (the gemv/f8row/mmq arms
            // below still read the f32 normed rows). : the cc-10 f8w
            // FFN arm (below the r>1024 f8 lane) is the same e4m3-only
            // consumer - fuse it too.
            let f8w_ffn = !f8 && !f8r && f8w_pf && lw.f8_gate.is_some();
            // f8t rowwise FFN at prefill: the decode band's class
            // extended to every r (class-uniform per the warm-vs-cold rule;
            // the launcher routes <=64 to tc5p/tc5q and >=65 to the 2-SM
            // tc5r). gu's concat output feeds geglu2_row unchanged; only
            // the ffn band switches - attn stays f8w until the concat-plane
            // sub-views land. Kill: PADDOCK_G4_NO_F8TPF.
            let f8t_pf2 = lw.f8t_gu.is_some()
                && lw.f8t_down.is_some()
                && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8TPF").is_none();
            let nqf_row_pf = f8t_pf2 && exec.has_rmsnorm_e4m3_row() && super::batch::nqfuse_on();
            let nqf_ffn = (f8 || f8w_ffn)
                && !f8t_pf2
                && super::batch::nqfuse_on()
                && exec.has_rmsnorm_e4m3_batch();
            // attn post-norm + residual, fused with the FFN pre-norm + per-32
            // quant on the nqf_ffn arm (the batch path's door-1 kernel -
            // bit-identical to the rmsnorm_add_scale -> rmsnorm_e4m3_batch
            // pair); other arms keep the two-kernel chain.
            // pc engagement for the fused-gu chunk route (fix:
            // producer emits ROW scales via the addnorm row twin - the 28c
            // lane paid a duplicate norm+quant pass here). Precedence terms
            // guarantee the fused-gu pc arm below is the consumer.
            let pc_gu = !f8t_pf2
                && f8
                && !f8ks
                && lw.fp4_gu.is_none()
                && !super::batch::fp4_on()
                && has_gu
                && lw.gu_il
                && super::batch::gu_fuse_on()
                && lw.gu_ws.is_some()
                && r >= super::batch::pc_floor()
                && hp.fused_two_norm_ok()  // its epilogue is an addnorm (2 norms, 1 eps)
                && exec.has_f8_gemm_lin_gu_pc(hp.glu_act())
                && exec.has_addnorm_e4m3_row();
            // Route-fired witness (the /55 doctrine, applied to the FFN
            // ladder): every arm below is bit-plausible, so a model silently
            // sitting two rungs down the ladder looks exactly like one on the
            // top rung - only slower. muse-glimmer did precisely that. Logged
            // once per process, and only for a real prefill chunk (r > 1), so
            // it names the arm the prefill band actually runs.
            {
                // Two bands, because the ladder is r-dependent: a short warmup
                // chunk and a real prompt can land on different arms, and the
                // one that owns prefill time is the wide one.
                static WITNESSED: [std::sync::atomic::AtomicBool; 2] = [
                    std::sync::atomic::AtomicBool::new(false),
                    std::sync::atomic::AtomicBool::new(false),
                ];
                let band = usize::from(r >= 512);
                if r > 1 && !WITNESSED[band].swap(true, std::sync::atomic::Ordering::Relaxed) {
                    let arm = if f8t_pf2 {
                        "f8t_pf2"
                    } else if f8 {
                        if pc_gu {
                            "f8/pc_gu"
                        } else if f8ks {
                            "f8/mma_ks"
                        } else {
                            "f8/w8"
                        }
                    } else if f8r {
                        "f8r/gemv"
                    } else if f8w_pf && lw.f8_gate.is_some() {
                        "f8w"
                    } else if f8row_pf && lw.f8r_gate.is_some() {
                        "f8row"
                    } else if mmq {
                        "mmq"
                    } else {
                        "q8_0_repacked"
                    };
                    tracing::info!(
                        rows = r,
                        arm,
                        has_gu,
                        f8,
                        f8r,
                        f8ks,
                        f8w_pf,
                        f8row_pf,
                        pc_gu,
                        "gemma4 prefill FFN arm elected"
                    );
                }
            }
            // chunk-band 16-bit streams. The o16 GEMM epilogues
            // write bf16 into pf_q/k/v and pf_proj (the rival's own stream
            // class - our f32 intermediates paid 2x the glue bytes) and the
            // bf16-in twins consume them; pf_x / pf_qn / pf_attn / the e4m3
            // planes are untouched. Numerics: activations round to bf16
            // before norms whose outputs the e4m3 quantizers crush anyway -
            // acceptance-gated (v8q/v9q precedent), not bit-parity. The gate
            // is all-or-nothing per layer: it requires every producer AND
            // consumer arm below (fused qkv, q-only norm+rope, kv fold, pc
            // wo - implied by pc_qkv's caps + pc_gu's r/fp4 terms - the
            // addnorm arm, and the pcd down). Kill: PADDOCK_G4_NO_CHUNK16.
            let c16 = qkv_fused1
                && qk_fused
                && kvf
                && pc_gu
                && r > 64  // the down chain's r<=64 mma_ks arm writes f32
                && lw.wo_ws.is_some()
                && lw.down_ws.is_some()
                && exec.has_f8_gemm_w8_pcd()
                && lw.moe.is_none()
                && exec.has_chunk16()
                && super::batch::chunk16_on();
            // Attention streams (a16): pf_qn's q plane and pf_attn
            // flip to f16 on this pass - nr stores half, the attention
            // family reads/writes half, the wo quantize reads half. Rides
            // only the c16 serve stack under the paged SWA ring + GLB pool
            // (the dense arms have no f16 forms) on the f8a route (there
            // pf_attn's only readers are the e4m3 quantizers). Kill:
            // PADDOCK_G4_NO_ATTN16.
            let a16 = c16
                && f8a_pf
                && lw.f16_attn_ok()  // the gate multiply is f32-in-place
                && self.paging.is_some()
                && self.gpool.is_some()
                && exec.has_attn16()
                && super::batch::attn16_on();
            // k/v ride the side lane on the fused-plane f8 arm: three
            // serial launches at ~4.8/2.4/2.4 waves pay three tail waves,
            // two lanes pack them (~9.5 total). Only the w8_off arm - it
            // touches no shared scratch (pf_skfix stays on the f8t arm;
            // the ktz K-split scratch can't fire at chunk-M nt). The
            // split-norm arm reads q between the k and v GEMMs, so the
            // fused epilogue is required.
            let kv_side = side.filter(|_| {
                !qkv_fused1 && f8a_pf && qkv8.is_some() && !super::batch::fp4_on() && qk_fused
            });
            if let Some(sx) = kv_side {
                sx.wait_event(&exec.record_event()?)?;
            }
            let kvx = kv_side.unwrap_or(exec);
            if f8t_att {
                if !nqf_row_att {
                    exec.quantize_e4m3_row(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        r,
                    )?;
                }
                //  b16 slice 2 (gate hoisted next to a16)
                {
                    exec.f8t_gemm_off(
                        lw.f8t_qkv.as_ref().expect("f8t_qkv checked (f8t_att)"),
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_q,
                        hp.n_embd,
                        hp.n_head * hd,
                        r,
                    )?;
                }
            } else if f8a_pf {
                // PC route: the producer above already emitted
                // row-scaled pf_e4q when pc_qkv - the three segment GEMMs
                // ride the scale-free kt4 twin. A refused route is an ERROR
                // (the per-32 fallback would read stale scales).
                let pc_take = |ok: bool| -> Result<(), crate::gpu::GpuError> {
                    if ok {
                        Ok(())
                    } else {
                        Err(crate::gpu::GpuError::Driver(
                            "pc qkv/wo route refused".into(),
                        ))
                    }
                };
                if !pc_qkv && !nqf_attn {
                    exec.quantize_e4m3(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        r * hp.n_embd,
                    )?;
                }
                match qkv8 {
                    // one launch writes all three projections; the k/v
                    // sections below no-op on qkv_fused1. c16: same launch,
                    // bf16 stores (the consumers below read the i16 twins).
                    Some(w8) if qkv_fused1 && c16 => pc_take(exec.f8_gemm_w8_pc_qkv_o16(
                        w8,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        lw.qkv_ws.as_ref().expect("qkv_ws checked (pc_qkv)"),
                        &mut sc.pf_q,
                        &mut sc.pf_k,
                        &mut sc.pf_v,
                        hp.n_embd,
                        hp.n_head * hd,
                        kv_dim,
                        r,
                    )?)?,
                    Some(w8) if qkv_fused1 => pc_take(exec.f8_gemm_w8_pc_qkv(
                        w8,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        lw.qkv_ws.as_ref().expect("qkv_ws checked (pc_qkv)"),
                        &mut sc.pf_q,
                        &mut sc.pf_k,
                        &mut sc.pf_v,
                        hp.n_embd,
                        hp.n_head * hd,
                        kv_dim,
                        r,
                    )?)?,
                    Some(w8) if pc_qkv => pc_take(exec.f8_gemm_w8_pc(
                        w8,
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        lw.qkv_ws.as_ref().expect("qkv_ws checked (pc_qkv)"),
                        0,
                        &mut sc.pf_q,
                        hp.n_embd,
                        hp.n_head * hd,
                        r,
                    )?)?,
                    Some(w8) if super::batch::fp4_on() => exec.mxfp4_gemm_bs_off(
                        w8,
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_q,
                        hp.n_embd,
                        hp.n_head * hd,
                        r,
                    )?,
                    Some(w8) => exec.f8_gemm_w8_off(
                        w8,
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_q,
                        hp.n_embd,
                        hp.n_head * hd,
                        r,
                    )?,
                    None => exec.f8_gemm_w8(
                        lw.f8a_wq
                            .as_ref()
                            .expect("f8a_pf without f8a_wqkv: f8a_wq present"),
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_q,
                        hp.n_embd,
                        hp.n_head * hd,
                        r,
                    )?,
                }
            } else if f8w_pf {
                // per-32 f8w planes through the tcgen05 block-scale route -
                // finer than rowwise AND faster (async-SF v2); same
                // row-independent quantize, so the warm-vs-cold rule holds
                if !nqf_attn {
                    exec.quantize_e4m3(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        r * hp.n_embd,
                    )?;
                }
                exec.f8_gemm_w8(
                    lw.f8w_wq.as_ref().expect("f8w_wq checked (f8w_pf)"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_q,
                    hp.n_embd,
                    hp.n_head * hd,
                    r,
                )?;
            } else if f8row_pf {
                exec.quantize_e4m3_row(
                    &sc.pf_normed,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_embd,
                    r,
                )?;
                exec.f8row_gemm(
                    lw.f8_wq.as_ref().expect("f8_wq checked (f8row_pf)"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_q,
                    hp.n_embd,
                    hp.n_head * hd,
                    r,
                )?;
            } else if let (true, Some(wq)) = (mmq, lw.wq.q8()) {
                exec.quantize_q8_mmq(&sc.pf_normed, &mut sc.pf_yq, hp.n_embd, r)?;
                pf_mmq(exec, wq, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_q, r)?;
            } else {
                // the class-generic form: the plain repacked GEMM for Q8, the
                // W4A8 ladder for a k-quant plane
                lw.wq
                    .gemm(exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_q, r)?;
            }
            // (norm+rope must run after the V GEMM - all three planes live;
            // fusing earlier read stale K/V and broke coherence on the
            // first cut); appends keep the sub-span walk (ring-shrink
            // contract)
            if !qk_fused {
                exec.rmsnorm_batch(
                    &sc.pf_q,
                    &lw.q_norm,
                    &mut sc.pf_qn,
                    hd,
                    hp.eps,
                    r * hp.n_head,
                )?;
                exec.rope_factors_batch(
                    &mut sc.pf_qn,
                    &sc.pf_pos,
                    factors,
                    hp.n_head,
                    hd,
                    rope,
                    r,
                    hp.rope_neox(),
                )?;
            }

            if f8t_att {
                {
                    exec.f8t_gemm_off(
                        lw.f8t_qkv.as_ref().expect("f8t_qkv checked (f8t_att)"),
                        hp.n_head * hd / 128,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_k,
                        hp.n_embd,
                        kv_dim,
                        r,
                    )?;
                }
            } else if k_f8a {
                match qkv8 {
                    // fused single launch already wrote pf_k
                    Some(_) if qkv_fused1 => {}
                    Some(w8) if pc_qkv => {
                        if !kvx.f8_gemm_w8_pc(
                            w8,
                            hp.n_head * hd,
                            &sc.pf_e4q,
                            &sc.pf_e4rs,
                            lw.qkv_ws.as_ref().expect("qkv_ws checked (pc_qkv)"),
                            hp.n_head * hd,
                            &mut sc.pf_k,
                            hp.n_embd,
                            kv_dim,
                            r,
                        )? {
                            return Err(crate::gpu::GpuError::Driver("pc k route refused".into()));
                        }
                    }
                    Some(w8) if super::batch::fp4_on() => exec.mxfp4_gemm_bs_off(
                        w8,
                        hp.n_head * hd,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_k,
                        hp.n_embd,
                        kv_dim,
                        r,
                    )?,
                    Some(w8) => kvx.f8_gemm_w8_off(
                        w8,
                        hp.n_head * hd,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_k,
                        hp.n_embd,
                        kv_dim,
                        r,
                    )?,
                    None => exec.f8_gemm_w8(
                        lw.f8a_wk
                            .as_ref()
                            .expect("k_f8a without f8a_wqkv: f8a_wk present"),
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_k,
                        hp.n_embd,
                        kv_dim,
                        r,
                    )?,
                }
            } else if k_f8w {
                exec.f8_gemm_w8(
                    lw.f8w_wk.as_ref().expect("f8w_wk checked (k_f8w)"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_k,
                    hp.n_embd,
                    kv_dim,
                    r,
                )?;
            } else if k_f8row {
                exec.f8row_gemm(
                    lw.f8_wk.as_ref().expect("f8_wk checked (k_f8row)"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_k,
                    hp.n_embd,
                    kv_dim,
                    r,
                )?;
            } else if let (true, Some(wk)) = (mmq, lw.wk.q8()) {
                pf_mmq(exec, wk, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_k, r)?;
            } else {
                // a bf16 or k-quant k plane has no mmq rung - its own
                // dispatch serves
                lw.wk
                    .gemm(exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_k, r)?;
            }
            match &lw.wv {
                // fused single launch already wrote pf_v
                Some(_) if qkv_fused1 => {}
                Some(_) if f8t_att => exec.f8t_gemm_off(
                    lw.f8t_qkv.as_ref().expect("f8t_qkv checked (f8t_att)"),
                    (hp.n_head * hd + kv_dim) / 128,
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_v,
                    hp.n_embd,
                    kv_dim,
                    r,
                )?,
                Some(_) if v_f8a && qkv8.is_some() && super::batch::fp4_on() => exec
                    .mxfp4_gemm_bs_off(
                        qkv8.expect("qkv8 checked in the guard"),
                        hp.n_head * hd + kv_dim,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_v,
                        hp.n_embd,
                        kv_dim,
                        r,
                    )?,
                Some(_) if v_f8a && pc_qkv => {
                    if !kvx.f8_gemm_w8_pc(
                        qkv8.expect("f8a_wqkv checked (pc_qkv)"),
                        hp.n_head * hd + kv_dim,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        lw.qkv_ws.as_ref().expect("qkv_ws checked (pc_qkv)"),
                        hp.n_head * hd + kv_dim,
                        &mut sc.pf_v,
                        hp.n_embd,
                        kv_dim,
                        r,
                    )? {
                        return Err(crate::gpu::GpuError::Driver("pc v route refused".into()));
                    }
                }
                Some(_) if v_f8a && qkv8.is_some() => kvx.f8_gemm_w8_off(
                    qkv8.expect("qkv8 checked in the guard"),
                    hp.n_head * hd + kv_dim,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_v,
                    hp.n_embd,
                    kv_dim,
                    r,
                )?,
                Some(_) if v_f8a => exec.f8_gemm_w8(
                    lw.f8a_wv
                        .as_ref()
                        .expect("v_f8a without f8a_wqkv: f8a_wv present"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_v,
                    hp.n_embd,
                    kv_dim,
                    r,
                )?,
                Some(_) if v_f8w => exec.f8_gemm_w8(
                    lw.f8w_wv.as_ref().expect("f8w_wv checked (v_f8w)"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_v,
                    hp.n_embd,
                    kv_dim,
                    r,
                )?,
                Some(_) if v_f8row => exec.f8row_gemm(
                    lw.f8_wv.as_ref().expect("f8_wv checked (v_f8row)"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_v,
                    hp.n_embd,
                    kv_dim,
                    r,
                )?,
                Some(wv) if mmq && wv.q8().is_some() => pf_mmq(
                    exec,
                    wv.q8().expect("checked"),
                    &sc.pf_yq,
                    &mut sc.pf_skfix,
                    &mut sc.pf_v,
                    r,
                )?,
                Some(wv) => wv.gemm(exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_v, r)?,
                None => {
                    // v = copy of k - dead under the kv fold (the fused
                    // append reads the raw k plane for both outputs)
                    if !kvf {
                        // the side lane may still be writing pf_k - join
                        // before reading it on the main lane
                        if let Some(sx) = kv_side {
                            exec.wait_event(&sx.record_event()?)?;
                        }
                        exec.copy_slice(&sc.pf_k, 0, r * kv_dim, &mut sc.pf_v)?
                    }
                }
            }
            if let Some(sx) = kv_side {
                // join: norm+rope on the main lane reads pf_k/pf_v
                exec.wait_event(&sx.record_event()?)?;
            }
            if qk_fused && a16 {
                // a16 twin: f16 q plane out (v3 register form, one rounding
                // at the store); input class = c16 (bf16 o16 epilogue)
                exec.qkv_norm_rope_batch_a16(
                    &sc.pf_q,
                    &sc.pf_k,
                    &sc.pf_v,
                    &lw.q_norm,
                    &lw.k_norm,
                    &mut sc.pf_qn,
                    &mut sc.pf_kn,
                    &mut sc.pf_vn,
                    &sc.pf_pos,
                    factors,
                    hp.n_head,
                    0,
                    hd,
                    hp.eps,
                    rope,
                    r,
                    true,
                    hp.rope_neox(),
                    hp.v_norm(),
                )?;
            } else if qk_fused && c16 {
                // c16 twin: pf_q holds bf16 from the o16 qkv epilogue
                exec.qkv_norm_rope_batch_i16(
                    &sc.pf_q,
                    &sc.pf_k,
                    &sc.pf_v,
                    &lw.q_norm,
                    &lw.k_norm,
                    &mut sc.pf_qn,
                    &mut sc.pf_kn,
                    &mut sc.pf_vn,
                    &sc.pf_pos,
                    factors,
                    hp.n_head,
                    0,
                    hd,
                    hp.eps,
                    rope,
                    r,
                    hp.rope_neox(),
                    hp.v_norm(),
                )?;
            } else if qk_fused {
                // kvf: q-only (n_kv=0 - the kernel never touches the k/v
                // slots); the fused appends below own K/V
                exec.qkv_norm_rope_batch(
                    &sc.pf_q,
                    &sc.pf_k,
                    &sc.pf_v,
                    &lw.q_norm,
                    &lw.k_norm,
                    &mut sc.pf_qn,
                    &mut sc.pf_kn,
                    &mut sc.pf_vn,
                    &sc.pf_pos,
                    factors,
                    hp.n_head,
                    if kvf { 0 } else { n_kv },
                    hd,
                    hp.eps,
                    rope,
                    r,
                    hp.rope_neox(),
                    hp.v_norm(),
                )?;
            } else {
                exec.rmsnorm_batch(&sc.pf_k, &lw.k_norm, &mut sc.pf_kn, hd, hp.eps, r * n_kv)?;
                exec.rope_factors_batch(
                    &mut sc.pf_kn,
                    &sc.pf_pos,
                    factors,
                    n_kv,
                    hd,
                    rope,
                    r,
                    hp.rope_neox(),
                )?;
                if hp.v_norm() {
                    exec.rmsnorm_batch(&sc.pf_v, &sc.ones, &mut sc.pf_vn, hd, hp.eps, r * n_kv)?;
                } else {
                    exec.copy_slice(&sc.pf_v, 0, r * n_kv * hd, &mut sc.pf_vn)?;
                }
            }

            // decode-row attention join event: set when the block below ran
            // on the side lane; the wo quantize (reads all pf_attn rows)
            // waits on it after the chunk-row attention
            let mut attn_join: Option<cudarc::driver::CudaEvent> = None;
            if lw.is_swa
                && let Some(pg) = &self.paging
            {
                // SWA under the WindowRing: append+attend advance one
                // sub-span at a time, so the ring only ever holds a span +
                // the window behind it (the ring-shrink contract - a whole-
                // chunk append would alias the window away mid-chunk).
                // WMMA f16 attention (bit-exact vs dense per pack contract);
                // spans never cross run (slot) boundaries.
                if decode_rows > 0 {
                    // unified tick: the leading decode rows append into their
                    // own slots' rings and attend via the DECODE kernels
                    // (one launch over all nd rows, split-K when starved -
                    // same dispatch as batch_step_body). Rides the side lane
                    // when forked: decode slots' pages are disjoint from the
                    // chunk slot's, pf_attn rows [0,nd) are its alone, so it
                    // overlaps the span walk's tail waves.
                    let att_side = side;
                    if let Some(sx) = att_side {
                        sx.wait_event(&exec.record_event()?)?;
                    }
                    let ax = att_side.unwrap_or(exec);
                    if kvf && c16 {
                        ax.kv_nra_rows_i16(
                            &sc.pf_k,
                            lw.wv.as_ref().map(|_| &sc.pf_v),
                            &lw.k_norm,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            factors,
                            &pg.bt,
                            pg.bps,
                            n_kv,
                            hd,
                            hp.eps,
                            rope,
                            0,
                            decode_rows,
                            kvl.dtype,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    } else if kvf {
                        ax.kv_nra_rows(
                            &sc.pf_k,
                            lw.wv.as_ref().map(|_| &sc.pf_v),
                            &lw.k_norm,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            factors,
                            &pg.bt,
                            pg.bps,
                            n_kv,
                            hd,
                            hp.eps,
                            rope,
                            0,
                            decode_rows,
                            kvl.dtype,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    } else {
                        ax.kv_append_batch_paged_rows(
                            &sc.pf_kn,
                            &mut kvl.k,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            &pg.bt,
                            pg.bps,
                            kv_dim,
                            0,
                            decode_rows,
                            kvl.dtype,
                        )?;
                        ax.kv_append_batch_paged_rows(
                            &sc.pf_vn,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            &pg.bt,
                            pg.bps,
                            kv_dim,
                            0,
                            decode_rows,
                            kvl.dtype,
                        )?;
                    }
                    let splits = super::batch::attn_splits(
                        hp.n_head,
                        decode_rows,
                        exec.sm_count(),
                        self.n_slots,
                    );
                    // verify-fold rung A: uniform k1-deep front rows go
                    // through the spec arm - One KV walk per chunk with
                    // per-row causal bounds - instead of the per-row decode
                    // kernels, which re-walk the window k1 times. On a mixed
                    // tick that re-walk is the largest slice of the
                    // decode-attn complex, where the same rows cost ~2/3 as
                    // much through krs. Same width gate as the pure verify
                    // arm, so narrow (c8-class) ticks keep the per-row route
                    // that is faster there.
                    // Kill: PADDOCK_G4_NO_MIXED_SPECFA.
                    let spec_arm = deco_k1
                        .filter(|_| {
                            super::batch::mixed_specfa_on() && exec.has_attn_spec_batch_paged()
                        })
                        .filter(|&k1| {
                            super::batch::spec_width_ok(
                                decode_rows / k1,
                                window,
                                n_kv,
                                hd,
                                kvl.dtype,
                            )
                        });
                    // tcgen05 decode attention - FINAL
                    // output (no partials/combine), opt-in PADDOCK_ATTN_TC5.
                    // Pure-decode SWA rows only: spec ticks keep the
                    // k1-folded walk (one KV walk per chunk - tc5 would
                    // re-walk per row), a16 planes are f16 (tc5 is f32
                    // in/out), and the pack entry re-gates shape/arch
                    // (rc -2 keeps the split route).
                    let tc5_done = spec_arm.is_none()
                        && !a16
                        && window > 0
                        && super::batch::attn_tc5_on()
                        && ax.has_attn_decode_tc5_paged()
                        && ax.attn_decode_tc5_paged(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            &pg.bt,
                            pg.bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            decode_rows,
                            ascale,
                            kvl.dtype,
                        )?;
                    if tc5_done {
                        // final rows already in pf_attn
                    } else if let Some(k1) = spec_arm {
                        let (ao, aml) = self
                            .attn_scratch
                            .as_mut()
                            .expect("unified tick requires enable_batch");
                        // sm_120 wide-band split election, mirrored from the
                        // pure verify arm (the fin coupling there is
                        // fin-specific - sp needs only the width, the
                        // long-KV band, and the die)
                        let wide_sp = if decode_rows / k1 >= 16
                            && deco_long
                            && super::batch::spec_sp_on()
                            && exec.compute_capability().0 >= 12
                        {
                            let cap =
                                (n_slots_cap * super::batch::MAX_ATTN_SPLITS) / decode_rows.max(1);
                            (if window > 0 { 2 } else { 4 }).min(cap)
                        } else {
                            1
                        };
                        let splits = if wide_sp > 1 { wide_sp } else { splits };
                        if a16 {
                            ax.attn_spec_batch_paged_a16(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                ao,
                                aml,
                                &sc.pf_pos,
                                Some(&sc.pf_slots),
                                &pg.bt,
                                pg.bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                splits,
                                decode_rows,
                                k1,
                                ascale,
                                kvl.dtype,
                            )?;
                            ax.attn_combine_batch_o16(
                                ao,
                                aml,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                hp.n_head,
                                hd,
                                splits,
                                decode_rows,
                            )?;
                        } else {
                            ax.attn_spec_batch_paged(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                ao,
                                aml,
                                &sc.pf_pos,
                                Some(&sc.pf_slots),
                                &pg.bt,
                                pg.bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                splits,
                                decode_rows,
                                k1,
                                ascale,
                                kvl.dtype,
                            )?;
                            ax.attn_combine_batch(
                                ao,
                                aml,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                hp.n_head,
                                hd,
                                splits,
                                decode_rows,
                            )?;
                        }
                    } else if splits > 1 {
                        let (ao, aml) = self
                            .attn_scratch
                            .as_mut()
                            .expect("unified tick requires enable_batch");
                        if a16 {
                            ax.attn_partial_batch_paged_a16(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                ao,
                                aml,
                                &sc.pf_pos,
                                Some(&sc.pf_slots),
                                &pg.bt,
                                pg.bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                splits,
                                decode_rows,
                                ascale,
                                kvl.dtype,
                            )?;
                            ax.attn_combine_batch_o16(
                                ao,
                                aml,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                hp.n_head,
                                hd,
                                splits,
                                decode_rows,
                            )?;
                        } else {
                            ax.attn_partial_batch_paged(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                ao,
                                aml,
                                &sc.pf_pos,
                                Some(&sc.pf_slots),
                                &pg.bt,
                                pg.bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                splits,
                                decode_rows,
                                ascale,
                                kvl.dtype,
                            )?;
                            ax.attn_combine_batch(
                                ao,
                                aml,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                hp.n_head,
                                hd,
                                splits,
                                decode_rows,
                            )?;
                        }
                    } else if a16 {
                        ax.attn_decode_batch_paged_a16(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            &pg.bt,
                            pg.bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            decode_rows,
                            ascale,
                            kvl.dtype,
                        )?;
                    } else {
                        ax.attn_decode_batch_paged(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            &pg.bt,
                            pg.bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            decode_rows,
                            ascale,
                            kvl.dtype,
                        )?;
                    }
                    if let Some(sx) = att_side {
                        attn_join = Some(sx.record_event()?);
                    }
                }
                // kv_nra is row-indexed (positions/slots per row)
                // so span boundaries are irrelevant to the append - under
                // the batched-attend arm, one launch covers every span's
                // rows (was 32 underfilled launches/layer, 10.9ms/pass).
                let batched_append = pf_runs_batched && !a16 && kvf;
                if batched_append {
                    if c16 {
                        exec.kv_nra_rows_i16(
                            &sc.pf_k,
                            lw.wv.as_ref().map(|_| &sc.pf_v),
                            &lw.k_norm,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            factors,
                            &pg.bt,
                            pg.bps,
                            n_kv,
                            hd,
                            hp.eps,
                            rope,
                            0,
                            r,
                            kvl.dtype,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    } else {
                        exec.kv_nra_rows(
                            &sc.pf_k,
                            lw.wv.as_ref().map(|_| &sc.pf_v),
                            &lw.k_norm,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            factors,
                            &pg.bt,
                            pg.bps,
                            n_kv,
                            hd,
                            hp.eps,
                            rope,
                            0,
                            r,
                            kvl.dtype,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    }
                }
                for &(off, n) in spans {
                    if batched_append {
                        let _ = (off, n);
                        break;
                    }
                    if kvf && c16 {
                        exec.kv_nra_rows_i16(
                            &sc.pf_k,
                            lw.wv.as_ref().map(|_| &sc.pf_v),
                            &lw.k_norm,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            factors,
                            &pg.bt,
                            pg.bps,
                            n_kv,
                            hd,
                            hp.eps,
                            rope,
                            off,
                            n,
                            kvl.dtype,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    } else if kvf {
                        exec.kv_nra_rows(
                            &sc.pf_k,
                            lw.wv.as_ref().map(|_| &sc.pf_v),
                            &lw.k_norm,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            factors,
                            &pg.bt,
                            pg.bps,
                            n_kv,
                            hd,
                            hp.eps,
                            rope,
                            off,
                            n,
                            kvl.dtype,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    } else {
                        exec.kv_append_batch_paged_rows(
                            &sc.pf_kn,
                            &mut kvl.k,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            &pg.bt,
                            pg.bps,
                            kv_dim,
                            off,
                            n,
                            kvl.dtype,
                        )?;
                        exec.kv_append_batch_paged_rows(
                            &sc.pf_vn,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            &pg.bt,
                            pg.bps,
                            kv_dim,
                            off,
                            n,
                            kvl.dtype,
                        )?;
                    }
                    if pf_runs_batched && !a16 {
                        // attends collapse into one armed launch below -
                        // appends stay per-span (different slots, no
                        // cross-span reads, so completing them all first
                        // is order-equivalent)
                    } else if a16 {
                        exec.attn_prefill_f16_rows_paged_a16(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_attn_pos,
                            wp_ok.then_some(&sc.pf_pos),
                            &sc.pf_slots,
                            &pg.bt,
                            pg.bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            off,
                            n,
                            ascale,
                            kvl.dtype,
                        )?;
                    } else {
                        exec.attn_prefill_f16_rows_paged(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_attn_pos,
                            wp_ok.then_some(&sc.pf_pos),
                            &sc.pf_slots,
                            &pg.bt,
                            pg.bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            off,
                            n,
                            ascale,
                            kvl.dtype,
                        )?;
                    }
                }
                if pf_runs_batched && !a16 {
                    exec.attn_prefill_f16_rows_paged(
                        &sc.pf_qn,
                        &kvl.k,
                        &kvl.v,
                        &sc.neg_inf_sinks,
                        &mut sc.pf_attn,
                        &sc.pf_attn_pos,
                        wp_ok.then_some(&sc.pf_pos),
                        &sc.pf_slots,
                        &pg.bt,
                        pg.bps,
                        hp.n_head,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        0,
                        r,
                        ascale,
                        kvl.dtype,
                    )?;
                }
            } else if lw.is_swa {
                debug_assert_eq!(decode_rows, 0, "unified tick requires paging");
                // dense SWA (no paging - full planes, no aliasing): whole-
                // chunk append, per-run attend. WMMA tile when %64 holds.
                exec.kv_append_batch(
                    &sc.pf_kn,
                    &mut kvl.k,
                    &sc.pf_pos,
                    Some(&sc.pf_slots),
                    kv_dim,
                    self.max_ctx,
                    r,
                    kvl.dtype,
                )?;
                exec.kv_append_batch(
                    &sc.pf_vn,
                    &mut kvl.v,
                    &sc.pf_pos,
                    Some(&sc.pf_slots),
                    kv_dim,
                    self.max_ctx,
                    r,
                    kvl.dtype,
                )?;
                for &(off, n) in runs {
                    if self.max_ctx.is_multiple_of(64) {
                        exec.attn_prefill_f16_rows(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_attn_pos,
                            wp_ok.then_some(&sc.pf_pos),
                            &sc.pf_slots,
                            hp.n_head,
                            n_kv,
                            hd,
                            self.max_ctx,
                            kv_dim,
                            window,
                            off,
                            n,
                            ascale,
                            kvl.dtype,
                        )?;
                    } else {
                        exec.attn_prefill_rows(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_attn_pos,
                            wp_ok.then_some(&sc.pf_pos),
                            &sc.pf_slots,
                            hp.n_head,
                            n_kv,
                            hd,
                            self.max_ctx,
                            kv_dim,
                            window,
                            off,
                            n,
                            ascale,
                            kvl.dtype,
                        )?;
                    }
                }
            } else {
                match self.gpool.as_ref().map(|gp| (&gp.d_bt, gp.bps)) {
                    Some((bt, bps)) if kvf && c16 => {
                        exec.kv_nra_rows_i16(
                            &sc.pf_k,
                            lw.wv.as_ref().map(|_| &sc.pf_v),
                            &lw.k_norm,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            factors,
                            bt,
                            bps,
                            n_kv,
                            hd,
                            hp.eps,
                            rope,
                            0,
                            r,
                            kvl.dtype,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    }
                    Some((bt, bps)) if kvf => {
                        // V-less global layers: vp None - the fused append
                        // takes v from the raw k plane (weightless norm)
                        exec.kv_nra_rows(
                            &sc.pf_k,
                            lw.wv.as_ref().map(|_| &sc.pf_v),
                            &lw.k_norm,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            factors,
                            bt,
                            bps,
                            n_kv,
                            hd,
                            hp.eps,
                            rope,
                            0,
                            r,
                            kvl.dtype,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    }
                    Some((bt, bps)) => {
                        exec.kv_append_batch_paged(
                            &sc.pf_kn,
                            &mut kvl.k,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            kvl.dtype,
                        )?;
                        exec.kv_append_batch_paged(
                            &sc.pf_vn,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            kvl.dtype,
                        )?;
                    }
                    _ => {
                        exec.kv_append_batch(
                            &sc.pf_kn,
                            &mut kvl.k,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            kv_dim,
                            self.max_ctx,
                            r,
                            kvl.dtype,
                        )?;
                        exec.kv_append_batch(
                            &sc.pf_vn,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            kv_dim,
                            self.max_ctx,
                            r,
                            kvl.dtype,
                        )?;
                    }
                }
                if decode_rows > 0 {
                    let (bt, bps) = self
                        .gpool
                        .as_ref()
                        .map(|gp| (&gp.d_bt, gp.bps))
                        .expect("unified tick requires the global pool");
                    // side lane (see the SWA arm): decode rows attend over
                    // pages the v3w runs below never touch; the fork event
                    // covers the whole-chunk appends above
                    let att_side = side;
                    if let Some(sx) = att_side {
                        sx.wait_event(&exec.record_event()?)?;
                    }
                    let ax = att_side.unwrap_or(exec);
                    let splits = super::batch::attn_splits(
                        hp.n_head,
                        decode_rows,
                        exec.sm_count(),
                        self.n_slots,
                    );
                    // verify-fold rung A: global-layer twin of the SWA arm
                    // above (window 0 - the volume gate uses the 2k span
                    // bound, the sp election picks 4)
                    let spec_arm = deco_k1
                        .filter(|_| {
                            super::batch::mixed_specfa_on() && exec.has_attn_spec_batch_paged()
                        })
                        .filter(|&k1| {
                            super::batch::spec_width_ok(decode_rows / k1, 0, n_kv, hd, kvl.dtype)
                        });
                    if let Some(k1) = spec_arm {
                        let (ao, aml) = self
                            .attn_scratch
                            .as_mut()
                            .expect("unified tick requires enable_batch");
                        let wide_sp = if decode_rows / k1 >= 16
                            && deco_long
                            && super::batch::spec_sp_on()
                            && exec.compute_capability().0 >= 12
                        {
                            let cap =
                                (n_slots_cap * super::batch::MAX_ATTN_SPLITS) / decode_rows.max(1);
                            4.min(cap)
                        } else {
                            1
                        };
                        let splits = if wide_sp > 1 { wide_sp } else { splits };
                        if a16 {
                            ax.attn_spec_batch_paged_a16(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                ao,
                                aml,
                                &sc.pf_pos,
                                Some(&sc.pf_slots),
                                bt,
                                bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                splits,
                                decode_rows,
                                k1,
                                ascale,
                                kvl.dtype,
                            )?;
                            ax.attn_combine_batch_o16(
                                ao,
                                aml,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                hp.n_head,
                                hd,
                                splits,
                                decode_rows,
                            )?;
                        } else {
                            ax.attn_spec_batch_paged(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                ao,
                                aml,
                                &sc.pf_pos,
                                Some(&sc.pf_slots),
                                bt,
                                bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                splits,
                                decode_rows,
                                k1,
                                ascale,
                                kvl.dtype,
                            )?;
                            ax.attn_combine_batch(
                                ao,
                                aml,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                hp.n_head,
                                hd,
                                splits,
                                decode_rows,
                            )?;
                        }
                    } else if splits > 1 {
                        let (ao, aml) = self
                            .attn_scratch
                            .as_mut()
                            .expect("unified tick requires enable_batch");
                        if a16 {
                            ax.attn_partial_batch_paged_a16(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                ao,
                                aml,
                                &sc.pf_pos,
                                Some(&sc.pf_slots),
                                bt,
                                bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                splits,
                                decode_rows,
                                ascale,
                                kvl.dtype,
                            )?;
                            ax.attn_combine_batch_o16(
                                ao,
                                aml,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                hp.n_head,
                                hd,
                                splits,
                                decode_rows,
                            )?;
                        } else {
                            ax.attn_partial_batch_paged(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                ao,
                                aml,
                                &sc.pf_pos,
                                Some(&sc.pf_slots),
                                bt,
                                bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                splits,
                                decode_rows,
                                ascale,
                                kvl.dtype,
                            )?;
                            ax.attn_combine_batch(
                                ao,
                                aml,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                hp.n_head,
                                hd,
                                splits,
                                decode_rows,
                            )?;
                        }
                    } else if a16 {
                        ax.attn_decode_batch_paged_a16(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            0,
                            decode_rows,
                            ascale,
                            kvl.dtype,
                        )?;
                    } else {
                        ax.attn_decode_batch_paged(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_pos,
                            Some(&sc.pf_slots),
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            0,
                            decode_rows,
                            ascale,
                            kvl.dtype,
                        )?;
                    }
                    if let Some(sx) = att_side {
                        attn_join = Some(sx.record_event()?);
                    }
                }
                // global layers, per slot-run: the paged WMMA f16 tile (NC 16
                // at HD=512) over the budget-pool table in pool mode; off the
                // pool, the dense scalar tiled prefill, whose HD=512 tile
                // fits the 99 KB opt-in window with 384 B to spare only
                // because none of it is static shared (see pd_apf_smem).
                // the armed run table is spans-based; globals
                // iterate runs, so the collapsed pf6g launch only engages
                // when the two coincide (no swa-ladder splits - always true
                // for prompts shorter than the window)
                let g_batched = pf_runs_batched && !a16 && spans == runs;
                for &(off, n) in runs {
                    if let Some(gp) = &self.gpool {
                        // WMMA f16 tile at HD=512 (NC=16) - the scalar tile ran
                        // 2.3 us/tile-step and grows quadratically with prompt
                        // length - ~13% of a live c8 wave, and dominant at
                        // long prompts. f16 class = the SWA layers' (and
                        // llama.cpp's) own prefill numerics.
                        if g_batched {
                            let _ = gp;
                        } else if a16 {
                            // global layers (window 0): no floor to derive,
                            // so no win_pos - the global arms stay untouched
                            exec.attn_prefill_f16_rows_paged_a16(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                &sc.pf_attn_pos,
                                None,
                                &sc.pf_slots,
                                &gp.d_bt,
                                gp.bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                off,
                                n,
                                ascale,
                                kvl.dtype,
                            )?;
                        } else {
                            exec.attn_prefill_f16_rows_paged(
                                &sc.pf_qn,
                                &kvl.k,
                                &kvl.v,
                                &sc.neg_inf_sinks,
                                &mut sc.pf_attn,
                                &sc.pf_attn_pos,
                                None,
                                &sc.pf_slots,
                                &gp.d_bt,
                                gp.bps,
                                hp.n_head,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                off,
                                n,
                                ascale,
                                kvl.dtype,
                            )?;
                        }
                    } else {
                        exec.attn_prefill_rows(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_attn_pos,
                            None,
                            &sc.pf_slots,
                            hp.n_head,
                            n_kv,
                            hd,
                            self.max_ctx,
                            kv_dim,
                            0,
                            off,
                            n,
                            ascale,
                            kvl.dtype,
                        )?;
                    }
                }
                if g_batched && let Some(gp) = &self.gpool {
                    exec.attn_prefill_f16_rows_paged(
                        &sc.pf_qn,
                        &kvl.k,
                        &kvl.v,
                        &sc.neg_inf_sinks,
                        &mut sc.pf_attn,
                        &sc.pf_attn_pos,
                        None,
                        &sc.pf_slots,
                        &gp.d_bt,
                        gp.bps,
                        hp.n_head,
                        n_kv,
                        hd,
                        kv_dim,
                        0,
                        0,
                        r,
                        ascale,
                        kvl.dtype,
                    )?;
                }
            }
            if let Some(ev) = attn_join.take() {
                // decode-row attention finished on the side lane; the wo
                // quantize below reads every pf_attn row
                exec.wait_event(&ev)?;
            }

            // sigmoid output gate (muse-glimmer) - after the side-lane join,
            // since it touches every pf_attn row, and before every wo arm.
            // pf_normed is guaranteed f32-materialized here: the fused-norm
            // arms above are gated off on gated layers (fused_norm_ok).
            Self::attn_gate_apply(
                exec,
                lw,
                &sc.pf_normed,
                &mut sc.pf_agate,
                &mut sc.pf_attn,
                &mut sc.pf_xq,
                &mut sc.pf_xs,
                &mut sc.pf_yq,
                &mut sc.pf_skfix,
                &mut sc.pf_e4q,
                &mut sc.pf_e4rs,
                hp.n_embd,
                hp.n_head * hd,
                r,
            )?;

            if f8a_pf {
                let pc_wo = lw.wo_ws.is_some()
                    && r >= super::batch::pc_floor()
                    && !super::batch::fp4_on()
                    && exec.has_f8_gemm_w8_pc();
                if pc_wo && c16 {
                    if a16 {
                        exec.quantize_e4m3_row_f16in(
                            &sc.pf_attn,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            hp.n_head * hd,
                            r,
                        )?;
                    } else {
                        exec.quantize_e4m3_row(
                            &sc.pf_attn,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            hp.n_head * hd,
                            r,
                        )?;
                    }
                    if !exec.f8_gemm_w8_pc_o16(
                        lw.f8a_wo.as_ref().expect("f8a attn planes built as a set"),
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        lw.wo_ws.as_ref().expect("wo_ws checked (pc_wo)"),
                        0,
                        &mut sc.pf_proj,
                        hp.n_head * hd,
                        hp.n_embd,
                        r,
                    )? {
                        return Err(crate::gpu::GpuError::Driver(
                            "pc wo o16 route refused".into(),
                        ));
                    }
                } else if pc_wo {
                    // (a16 requires c16, so this arm never sees an f16 plane)
                    exec.quantize_e4m3_row(
                        &sc.pf_attn,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_head * hd,
                        r,
                    )?;
                    if !exec.f8_gemm_w8_pc(
                        lw.f8a_wo.as_ref().expect("f8a attn planes built as a set"),
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        lw.wo_ws.as_ref().expect("wo_ws checked (pc_wo)"),
                        0,
                        &mut sc.pf_proj,
                        hp.n_head * hd,
                        hp.n_embd,
                        r,
                    )? {
                        return Err(crate::gpu::GpuError::Driver("pc wo route refused".into()));
                    }
                } else {
                    if a16 {
                        exec.quantize_e4m3_f16in(
                            &sc.pf_attn,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4s,
                            r * hp.n_head * hd,
                        )?;
                    } else {
                        exec.quantize_e4m3(
                            &sc.pf_attn,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4s,
                            r * hp.n_head * hd,
                        )?;
                    }
                    if super::batch::fp4_on() {
                        exec.mxfp4_gemm_bs(
                            lw.f8a_wo.as_ref().expect("f8a attn planes built as a set"),
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_proj,
                            hp.n_head * hd,
                            hp.n_embd,
                            r,
                        )?;
                    } else {
                        exec.f8_gemm_w8(
                            lw.f8a_wo.as_ref().expect("f8a attn planes built as a set"),
                            0,
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_proj,
                            hp.n_head * hd,
                            hp.n_embd,
                            r,
                        )?;
                    }
                }
            } else if lw.f8t_wo.is_some()
                && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8TPF").is_none()
            {
                // rowwise wo at prefill: the decode band's plane +
                // class, through the same tc5p/tc5r-routed launcher
                exec.quantize_e4m3_row(
                    &sc.pf_attn,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_head * hd,
                    r,
                )?;
                {
                    exec.f8t_gemm(
                        lw.f8t_wo.as_ref().expect("f8t_wo checked above"),
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_proj,
                        hp.n_head * hd,
                        hp.n_embd,
                        r,
                    )?;
                }
            } else if f8w_pf {
                exec.quantize_e4m3(
                    &sc.pf_attn,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4s,
                    r * hp.n_head * hd,
                )?;
                exec.f8_gemm_w8(
                    lw.f8w_wo.as_ref().expect("f8w attn planes built as a set"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_proj,
                    hp.n_head * hd,
                    hp.n_embd,
                    r,
                )?;
            } else if f8row_pf {
                exec.quantize_e4m3_row(
                    &sc.pf_attn,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_head * hd,
                    r,
                )?;
                exec.f8row_gemm(
                    lw.f8_wo.as_ref().expect("f8row attn planes built as a set"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_proj,
                    hp.n_head * hd,
                    hp.n_embd,
                    r,
                )?;
            } else if let (true, Some(wo)) = (mmq, lw.wo.q8()) {
                exec.quantize_q8_mmq(&sc.pf_attn, &mut sc.pf_yq, hp.n_head * hd, r)?;
                pf_mmq(exec, wo, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_proj, r)?;
            } else {
                lw.wo
                    .gemm(exec, kq_rows!(sc), &sc.pf_attn, &mut sc.pf_proj, r)?;
            }
            // (n_ff and the FFN-side gates are hoisted above the qkv
            // section - the c16 stream gate needs them early)
            if pc_gu && c16 {
                exec.addnorm_e4m3_row_p16(
                    &mut sc.pf_x,
                    &sc.pf_proj,
                    &lw.attn_post_norm,
                    &lw.ffn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_embd,
                    hp.eps,
                    1.0,
                    r,
                )?;
            } else if pc_gu {
                exec.addnorm_e4m3_row(
                    &mut sc.pf_x,
                    &sc.pf_proj,
                    &lw.attn_post_norm,
                    &lw.ffn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_embd,
                    hp.eps,
                    1.0,
                    r,
                )?;
            } else if nqf_ffn && hp.fused_two_norm_ok() && exec.has_addnorm_e4m3_b32() {
                exec.addnorm_e4m3_b32(
                    &mut sc.pf_x,
                    &sc.pf_proj,
                    &lw.attn_post_norm,
                    &lw.ffn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4s,
                    hp.n_embd,
                    hp.eps,
                    1.0,
                    r,
                )?;
            } else {
                // fused post-norm + residual (the decode walk's kernel, same
                // bit-exact math): one pass instead of rmsnorm + add
                {
                    exec.rmsnorm_add_scale(
                        &mut sc.pf_x,
                        &sc.pf_proj,
                        &lw.attn_post_norm,
                        hp.n_embd,
                        hp.post_norm_eps,
                        1.0,
                        r,
                    )?;
                }
                if nqf_row_pf {
                    exec.rmsnorm_e4m3_row(
                        &sc.pf_x,
                        &lw.ffn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        hp.eps,
                        r,
                    )?;
                } else if nqf_ffn {
                    exec.rmsnorm_e4m3_batch(
                        &sc.pf_x,
                        &lw.ffn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        hp.n_embd,
                        r,
                        hp.eps,
                    )?;
                } else {
                    exec.rmsnorm_batch(
                        &sc.pf_x,
                        &lw.ffn_norm,
                        &mut sc.pf_normed,
                        hp.n_embd,
                        hp.eps,
                        r,
                    )?;
                }
            }
            // set by the fused gu-epilogue arm below: ff activations landed
            // in pf_ffq/pf_ffs straight from the GEMM (no geglu launch)
            let mut gu_fused = false;
            // pf_gate came from the INTERLEAVED f8_gu plane (pair-addressed
            // geglu) - Not set by the fp4_gu quality-probe arm, whose plane
            // keeps the plain [gate|up] order
            let mut gu_pf_il = false;
            if f8t_pf2 {
                if !nqf_row_pf {
                    exec.quantize_e4m3_row(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        r,
                    )?;
                }
                {
                    exec.f8t_gemm(
                        lw.f8t_gu.as_ref().expect("f8t_gu checked (f8t_pf2)"),
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_gate,
                        hp.n_embd,
                        2 * n_ff,
                        r,
                    )?;
                    exec.quantize_e4m3_glu2_row(
                        &sc.pf_gate,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        n_ff,
                        r,
                        hp.glu_act(),
                    )?;
                }
                {
                    exec.f8t_gemm(
                        lw.f8t_down.as_ref().expect("f8t_down checked (f8t_pf2)"),
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_proj,
                        n_ff,
                        hp.n_embd,
                        r,
                    )?;
                }
            } else if f8 {
                if !nqf_ffn {
                    exec.quantize_e4m3(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        r * hp.n_embd,
                    )?;
                }
                if f8ks {
                    if has_gu && super::batch::fp4_on() {
                        gu_pf_il = lw.gu_il;
                        exec.fp4_gemm_mma_ks(
                            lw.f8_gu.as_ref().expect("f8_gu present (has_gu)"),
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_skfix,
                            &mut sc.pf_gate,
                            hp.n_embd,
                            2 * n_ff,
                            r,
                        )?;
                    } else if has_gu {
                        gu_pf_il = lw.gu_il;
                        exec.f8_gemm_mma_ks(
                            lw.f8_gu.as_ref().expect("f8_gu present (has_gu)"),
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_skfix,
                            &mut sc.pf_gate,
                            hp.n_embd,
                            2 * n_ff,
                            r,
                        )?;
                    } else {
                        exec.f8_gemm_mma_ks(
                            lw.f8_gate
                                .as_ref()
                                .expect("f8 lane without gu: f8_gate present"),
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_skfix,
                            &mut sc.pf_gate,
                            hp.n_embd,
                            n_ff,
                            r,
                        )?;
                        exec.f8_gemm_mma_ks(
                            lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_skfix,
                            &mut sc.pf_up,
                            hp.n_embd,
                            n_ff,
                            r,
                        )?;
                    }
                } else if let (Some(gu4), true) = (lw.fp4_gu.as_ref(), has_gu) {
                    // fp4 QUALITY probe: prefill gate|up on 4-bit weights
                    // (same concat layout, geglu2 epilogue unchanged)
                    exec.mxfp4_gemm_bs(
                        gu4,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_gate,
                        hp.n_embd,
                        2 * n_ff,
                        r,
                    )?;
                } else if has_gu && super::batch::fp4_on() {
                    gu_pf_il = lw.gu_il;
                    exec.mxfp4_gemm_bs(
                        lw.f8_gu.as_ref().expect("f8_gu present (has_gu)"),
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_gate,
                        hp.n_embd,
                        2 * n_ff,
                        r,
                    )?;
                } else if has_gu {
                    // fused gu epilogue (batch.rs twin): geglu+quant in the
                    // GEMM on the interleaved plane, landing pf_ffq/pf_ffs
                    gu_pf_il = lw.gu_il;
                    if lw.gu_il && super::batch::gu_fuse_on() {
                        // PC route: chunk shapes take the scale-free
                        // kt4a twin - re-quantize the same pf_x stream with the
                        // row quantizer (bit-consistent input values, per-token
                        // scales) and skip the per-32 operand machinery in the
                        // mainloop. Below the chunk floor (or pc plane absent)
                        // the per-32 fused kernel serves the identical plane.
                        gu_fused = if pc_gu {
                            // producer already emitted row-scaled pf_e4q
                            exec.f8_gemm_lin_gu_pc(
                                lw.f8_gu.as_ref().expect("f8_gu present (has_gu)"),
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                lw.gu_ws.as_ref().expect("gu_ws checked (pc_gu)"),
                                &mut sc.pf_ffq,
                                &mut sc.pf_ffs,
                                hp.n_embd,
                                2 * n_ff,
                                r,
                                hp.glu_act(),
                            )?
                        } else {
                            exec.f8_gemm_lin_gu(
                                lw.f8_gu.as_ref().expect("f8_gu present (has_gu)"),
                                &sc.pf_e4q,
                                &sc.pf_e4s,
                                &mut sc.pf_ffq,
                                &mut sc.pf_ffs,
                                hp.n_embd,
                                2 * n_ff,
                                r,
                                hp.glu_act(),
                            )?
                        };
                    }
                    if !gu_fused {
                        exec.f8_gemm_w8(
                            lw.f8_gu.as_ref().expect("f8_gu present (has_gu)"),
                            0,
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_gate,
                            hp.n_embd,
                            2 * n_ff,
                            r,
                        )?;
                    }
                } else {
                    exec.f8_gemm_w8(
                        lw.f8_gate
                            .as_ref()
                            .expect("f8 lane without gu: f8_gate present"),
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_gate,
                        hp.n_embd,
                        n_ff,
                        r,
                    )?;
                    exec.f8_gemm_w8(
                        lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_up,
                        hp.n_embd,
                        n_ff,
                        r,
                    )?;
                }
            } else if f8r {
                // F8R below the GEMM band: r==1 only when the fused plane is
                // live (gufuse requires the twin, which owns 2..31)
                if has_gu {
                    exec.f8_gemv_batch(
                        lw.f8_gu.as_ref().expect("f8_gu present (has_gu)"),
                        &sc.pf_normed,
                        &mut sc.pf_gate,
                        hp.n_embd,
                        2 * n_ff,
                        r,
                    )?;
                } else {
                    exec.f8_gemv_batch(
                        lw.f8_gate
                            .as_ref()
                            .expect("f8 lane without gu: f8_gate present"),
                        &sc.pf_normed,
                        &mut sc.pf_gate,
                        hp.n_embd,
                        n_ff,
                        r,
                    )?;
                    exec.f8_gemv_batch(
                        lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                        &sc.pf_normed,
                        &mut sc.pf_up,
                        hp.n_embd,
                        n_ff,
                        r,
                    )?;
                }
            } else if f8w_pf && lw.f8_gate.is_some() {
                if !nqf_ffn {
                    exec.quantize_e4m3(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        r * hp.n_embd,
                    )?;
                }
                exec.f8_gemm_w8(
                    lw.f8_gate.as_ref().expect("f8_gate checked above"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_gate,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
                exec.f8_gemm_w8(
                    lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_up,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
            } else if f8row_pf && let Some(f8r_gate) = &lw.f8r_gate {
                exec.quantize_e4m3_row(
                    &sc.pf_normed,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_embd,
                    r,
                )?;
                exec.f8row_gemm(
                    f8r_gate,
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_gate,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
                exec.f8row_gemm(
                    lw.f8r_up.as_ref().expect("f8row FFN planes built as a set"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_up,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
            } else if let (true, Some(g8), Some(u8_)) = (mmq, lw.ffn_gate.q8(), lw.ffn_up.q8()) {
                exec.quantize_q8_mmq(&sc.pf_normed, &mut sc.pf_yq, hp.n_embd, r)?;
                pf_mmq(exec, g8, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_gate, r)?;
                pf_mmq(exec, u8_, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_up, r)?;
            } else {
                lw.ffn_gate
                    .gemm(exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_gate, r)?;
                lw.ffn_up
                    .gemm(exec, kq_rows!(sc), &sc.pf_normed, &mut sc.pf_up, r)?;
            }
            if f8t_pf2 {
                // gu/geglu/down handled in the self-contained arm above
            } else if f8 {
                if has_gu {
                    if !gu_fused {
                        if gu_pf_il {
                            exec.quantize_e4m3_glu2i(
                                &sc.pf_gate,
                                &mut sc.pf_e4q,
                                &mut sc.pf_e4s,
                                n_ff,
                                r,
                                hp.glu_act(),
                            )?;
                        } else {
                            exec.quantize_e4m3_glu2(
                                &sc.pf_gate,
                                &mut sc.pf_e4q,
                                &mut sc.pf_e4s,
                                n_ff,
                                r,
                                hp.glu_act(),
                            )?;
                        }
                    }
                } else {
                    g4_e4m3_glu(exec, sc, r * n_ff, hp.glu_act())?;
                }
                // the fused arm landed the ff activations in pf_ffq/pf_ffs
                let (dq, ds) = if gu_fused {
                    (&sc.pf_ffq, &sc.pf_ffs)
                } else {
                    (&sc.pf_e4q, &sc.pf_e4s)
                };
                // down rides the twin through r<=64 (42 out-tiles underfill
                // the die on TMA - see the batch.rs note)
                if f8ks || (r <= 64 && exec.has_f8_gemm_mma_ks()) {
                    if super::batch::fp4_on() {
                        exec.fp4_gemm_mma_ks(
                            lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                            dq,
                            ds,
                            &mut sc.pf_skfix,
                            &mut sc.pf_proj,
                            n_ff,
                            hp.n_embd,
                            r,
                        )?;
                    } else {
                        exec.f8_gemm_mma_ks(
                            lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                            dq,
                            ds,
                            &mut sc.pf_skfix,
                            &mut sc.pf_proj,
                            n_ff,
                            hp.n_embd,
                            r,
                        )?;
                    }
                } else if super::batch::fp4_on() {
                    exec.mxfp4_gemm_bs(
                        lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                        dq,
                        ds,
                        &mut sc.pf_proj,
                        n_ff,
                        hp.n_embd,
                        r,
                    )?;
                } else if let Some(down_ws) = &lw.down_ws
                    && r >= super::batch::pc_floor()
                    && exec.has_f8_gemm_w8_pcd()
                    && c16
                {
                    if !exec.f8_gemm_w8_pcd_o16(
                        lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                        dq,
                        ds,
                        down_ws,
                        &mut sc.pf_proj,
                        n_ff,
                        hp.n_embd,
                        r,
                    )? {
                        return Err(crate::gpu::GpuError::Driver(
                            "pc down o16 route refused".into(),
                        ));
                    }
                } else if let Some(down_ws) = &lw.down_ws
                    && r >= super::batch::pc_floor()
                    && exec.has_f8_gemm_w8_pcd()
                {
                    // down twin: activations keep their per-32
                    // scales (ds = the fused gu epilogue's own output), only
                    // the weight-scale machinery leaves the loop
                    if !exec.f8_gemm_w8_pcd(
                        lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                        dq,
                        ds,
                        down_ws,
                        &mut sc.pf_proj,
                        n_ff,
                        hp.n_embd,
                        r,
                    )? {
                        return Err(crate::gpu::GpuError::Driver("pc down route refused".into()));
                    }
                } else {
                    exec.f8_gemm_w8(
                        lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                        0,
                        dq,
                        ds,
                        &mut sc.pf_proj,
                        n_ff,
                        hp.n_embd,
                        r,
                    )?;
                }
            } else if f8r {
                if has_gu {
                    // r==1 here (see the arm above): pair-fold in place,
                    // result dense in the row's first half
                    exec.glu_pair(&mut sc.pf_gate, n_ff, r, hp.glu_act())?;
                } else {
                    exec.glu(&mut sc.pf_gate, &sc.pf_up, r * n_ff, hp.glu_act())?;
                }
                exec.f8_gemv_batch(
                    lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                    &sc.pf_gate,
                    &mut sc.pf_proj,
                    n_ff,
                    hp.n_embd,
                    r,
                )?;
            } else if f8w_pf && let Some(f8_down) = &lw.f8_down {
                g4_e4m3_glu(exec, sc, r * n_ff, hp.glu_act())?;
                exec.f8_gemm_w8(
                    f8_down,
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_proj,
                    n_ff,
                    hp.n_embd,
                    r,
                )?;
            } else if f8row_pf && let Some(f8r_down) = &lw.f8r_down {
                exec.glu(&mut sc.pf_gate, &sc.pf_up, r * n_ff, hp.glu_act())?;
                exec.quantize_e4m3_row(&sc.pf_gate, &mut sc.pf_e4q, &mut sc.pf_e4rs, n_ff, r)?;
                exec.f8row_gemm(
                    f8r_down,
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_proj,
                    n_ff,
                    hp.n_embd,
                    r,
                )?;
            } else if let (true, Some(d8)) = (mmq, lw.ffn_down.q8()) {
                if exec.has_quantize_q8_mmq_glu(hp.glu_act()) {
                    // GEGLU fused into the down-GEMM's quantize (qwen35 P6j
                    // shape): gate/up read once, the f32 activation never
                    // lands - saves pd_geglu's full n_ff round trip per chunk.
                    // Values bit-identical to geglu -> quantize (same formula,
                    // same scale math), so the parity gates arbitrate as usual.
                    exec.quantize_q8_mmq_glu(
                        &sc.pf_gate,
                        &sc.pf_up,
                        &mut sc.pf_yq,
                        n_ff,
                        r,
                        hp.glu_act(),
                    )?;
                } else {
                    exec.glu(&mut sc.pf_gate, &sc.pf_up, r * n_ff, hp.glu_act())?;
                    exec.quantize_q8_mmq(&sc.pf_gate, &mut sc.pf_yq, n_ff, r)?;
                }
                pf_mmq(exec, d8, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_proj, r)?;
            } else {
                exec.glu(&mut sc.pf_gate, &sc.pf_up, r * n_ff, hp.glu_act())?;
                lw.ffn_down
                    .gemm(exec, kq_rows!(sc), &sc.pf_gate, &mut sc.pf_proj, r)?;
            }
            // fused post-norm + residual + layer_output_scale: replaces
            // rmsnorm + add + (swap/memset/scale_add) - the out_scale path
            // alone round-tripped r x n_embd twice per layer. Same element
            // order as the sequence (sum, then one multiply) - bit-exact.
            // 26B-A4B layers route through the hybrid two-branch tail.
            if let Some(moe) = &lw.moe {
                super::batch::g4_moe_tail(
                    exec,
                    sc,
                    moe,
                    &lw.ffn_post_norm,
                    hp,
                    lw.out_scale,
                    r,
                    true,
                )?;
            } else if c16 {
                // pf_proj holds bf16 (o16 down epilogue, or the b16 wide arm)
                exec.rmsnorm_add_scale_p16(
                    &mut sc.pf_x,
                    &sc.pf_proj,
                    &lw.ffn_post_norm,
                    hp.n_embd,
                    hp.post_norm_eps,
                    lw.out_scale,
                    r,
                )?;
            } else {
                exec.rmsnorm_add_scale(
                    &mut sc.pf_x,
                    &sc.pf_proj,
                    &lw.ffn_post_norm,
                    hp.n_embd,
                    hp.post_norm_eps,
                    lw.out_scale,
                    r,
                )?;
            }
        }

        if pf_runs_batched {
            self.exec.pf_runs_register(None)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prefill scratch is sized by `pf_rows` and every prefill lane chunks
    /// at it, so the invariants that keep those two honest are worth pinning:
    /// never above the ceiling (the allocation would grow past what any lane
    /// can present), and never below one serving tick (a tick that cannot fit
    /// the scratch is an out-of-bounds write, not a slow path).
    #[test]
    fn pf_rows_stays_between_one_tick_and_the_ceiling() {
        for ctx in [512usize, 2048, 4096, 8192, 32768, 262144] {
            let r = pf_rows(ctx);
            assert!(r <= PF_ROWS, "ctx {ctx}: {r} over the {PF_ROWS} ceiling");
            assert!(
                r >= super::super::batch::mixed_tick_rows(),
                "ctx {ctx}: {r} cannot hold one {}-row tick",
                super::super::batch::mixed_tick_rows()
            );
        }
    }

    /// `enable_batch`'s chunk ladder halves toward the floor when the KV plan
    /// cannot seat the configuration beside the scratch. It must reach the
    /// floor exactly and never step under it: the planes are `[pf_rows, dim]`,
    /// so a tick wider than the allocation is an out-of-bounds write.
    #[test]
    fn the_chunk_ladder_lands_on_one_tick_and_stops() {
        let floor = pf_rows_floor();
        for ctx in [4096usize, 16384, 262144] {
            let mut chunk = pf_rows(ctx);
            assert!(chunk >= floor, "ctx {ctx}: starts under the floor");
            let mut steps = 0;
            while chunk > floor {
                chunk = (chunk / 2).max(floor);
                assert!(chunk >= floor, "ctx {ctx}: stepped under one tick");
                steps += 1;
                assert!(steps < 32, "ctx {ctx}: ladder does not terminate");
            }
            assert_eq!(chunk, floor);
        }
    }

    /// A small-context server is the whole point of the sizing: at ctx 4096 a
    /// chunk can never be 8192 rows, and paying for them cost gemma-4-31B its
    /// batch lane on a 48 GB card.
    #[test]
    fn a_small_context_server_does_not_pay_for_the_ceiling() {
        assert!(pf_rows(4096) < PF_ROWS);
        assert_eq!(pf_rows(262144), PF_ROWS);
    }

    /// The span ladder is walked widest-first and its rungs must be strictly
    /// descending, or `find` would elect a narrower ring than one that fits.
    #[test]
    fn swa_span_ladder_descends_from_the_default() {
        assert_eq!(SWA_SPAN_LADDER[0], SWA_SPAN_DEFAULT);
        assert!(SWA_SPAN_LADDER.windows(2).all(|w| w[0] > w[1]));
        assert!(SWA_SPAN_LADDER.iter().all(|&s| (64..=PF_ROWS).contains(&s)));
    }
}
