//! Gemma 4 MTP drafter (`gemma4-assistant`) - EAGLE-class separate-model
//! speculative drafter, wired into the generic serving spec rounds.
//!
//! Reference: llama.cpp b9969 `src/models/gemma4-assistant.cpp` (study only).
//! Shape of the thing, all verified against that graph:
//!
//! - 4 layers, embd 1024, GEGLU ffn 8192; SWA pattern [T,T,T,F] mirroring the
//!   main model's two geometries (SWA hd256/32Q, global hd512/32Q).
//! - Q-only attention: the drafter has no K/V projections and no cache of its
//!   own - `shared_kv_layers=4` means every drafter layer reads a MAIN-model
//!   plane: SWA layers share main layer n-2 (a SWA plane), the global layer
//!   shares main layer n-1 (a global plane). Nothing is ever appended, so the
//!   whole warm/desync machinery the qwen35 drafter needed does not exist.
//! - Input per drafted row: concat(main token_embd[t]·√5376, h) -> pre_proj
//!   [10752->1024], where h is the main model's POST-OUTPUT-NORM hidden (the
//!   LM-head input feature) for the previous position - or the drafter's own
//!   `nextn.post_projection` output when chaining draft steps.
//! - Output head: the drafter's own tied 1024-dim token_embd (no softcap in
//!   the reference graph; greedy argmax is softcap-invariant anyway).
//!
//! Attention positions: rope uses the true draft position (p, p+1, ...), but
//! the attention bound is CLAMPED to the committed cache head (llama.cpp gets
//! this via valid-cell masking; our decode kernels derive kv_len/window from
//! the per-row position, so the drafter carries two position buffers - the
//! same split the multimodal splice uses).
//!
//! h freshness: `spec_rows` records (slot, pos, row) for every row of the
//! last sampled batch tick - `pf_normed` still holds those rows' final-norm
//! hiddens when the next spec round drafts (nothing runs in between). Ticks
//! that don't record (unified/mixed with prompt tails) leave the map stale;
//! drafting then declines for one round and the verify pass itself (a
//! sampled-rows tick) re-freshens it - self-healing, correctness never at
//! stake since every draft is verified by the main model.

use cudarc::driver::CudaSlice;

use super::GpuGemma4;
use super::load::LoadError;
use crate::gpu::{GpuError, GpuExecutor, RepackedQ8};
use paddock_models::mapped::MappedGguf;

/// PADDOCK_SPEC_STATS=N: report cumulative + window acceptance economics
/// every N verify rounds (unset/0 = off). Committed counts the guaranteed
/// sampled token (a+1 per slot), matching tokens/slot/round as derived
/// from kernel launch counts.
fn spec_stat_interval() -> u64 {
    static IV: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *IV.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_STATS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

#[derive(Default)]
pub(crate) struct SpecStats {
    rounds: u64,
    slot_rounds: u64,
    rows: u64,
    committed: u64,
    win: (u64, u64, u64, u64),
    /// Accept-depth histogram: depth[d] counts slot-rounds whose committed
    /// count (accepted drafts + 1, i.e. o[0]/a+1) was d. Exact per-depth
    /// conditional acceptance for the tree-verify arithmetic.
    depth: [u64; SPEC_K1_MAX + 2],
}

impl SpecStats {
    /// Per-slot accept depth (committed = accepted drafts + 1). Zero-cost
    /// when PADDOCK_SPEC_STATS is off.
    pub(crate) fn bump_depth(&mut self, d: usize) {
        if spec_stat_interval() == 0 {
            return;
        }
        self.depth[d.min(SPEC_K1_MAX + 1)] += 1;
    }

    pub(crate) fn bump(&mut self, n: usize, rows: usize, committed: usize) {
        let iv = spec_stat_interval();
        if iv == 0 {
            return;
        }
        self.rounds += 1;
        self.slot_rounds += n as u64;
        self.rows += rows as u64;
        self.committed += committed as u64;
        if !self.rounds.is_multiple_of(iv) {
            return;
        }
        let (wr, wsr, wrow, wc) = (
            self.rounds - self.win.0,
            self.slot_rounds - self.win.1,
            self.rows - self.win.2,
            self.committed - self.win.3,
        );
        self.win = (self.rounds, self.slot_rounds, self.rows, self.committed);
        let per = |c: u64, sr: u64| c as f64 / sr.max(1) as f64;
        let dh: Vec<String> = self
            .depth
            .iter()
            .enumerate()
            .skip(1)
            .filter(|e| *e.1 > 0)
            .map(|(d, &c)| format!("{d}:{:.3}", c as f64 / self.slot_rounds.max(1) as f64))
            .collect();
        tracing::info!(
            "[g4-spec-stats] rounds={} slots/rnd={:.1} acc/slot/rnd={:.3} rows/slot/rnd={:.3} | win({wr}) acc={:.3} rows={:.3} | depth {}",
            self.rounds,
            self.slot_rounds as f64 / self.rounds as f64,
            per(self.committed, self.slot_rounds),
            per(self.rows, self.slot_rounds),
            per(wc, wsr),
            per(wrow, wsr),
            dh.join(" "),
        );
    }
}

/// Drafter chain row ceiling - one chain row per live SLOT (not per verify
/// row), so max_batch bounds it.
pub(crate) const DR_MAX_ROWS: usize = 32;
/// Drafter chain depth ceiling (service k cap is 15 for this family).
pub(crate) const DR_MAX_K: usize = 16;
/// Verify rows per slot the tick-path buffers are sized for when the
/// drafter is attached: pending + up to SERVE_SPEC_MAX_K drafts.
pub(crate) const SPEC_K1_MAX: usize = 8;

/// Spec engagement cap. Unlike the qwen35 drafter, which loses under
/// concurrency, this one has no KV to warm - and gemma4's MTP head is
/// designed to pay at every batch width - so the default is uncapped
/// (bounded by max_batch); env still pins it down for A/Bs.
pub(crate) fn spec_live_max() -> usize {
    std::env::var("PADDOCK_G4_SPEC_LIVE_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX)
}

pub(crate) struct DrLayer {
    attn_norm: CudaSlice<f32>,
    wq: RepackedQ8,
    q_norm: CudaSlice<f32>,
    wo: RepackedQ8,
    attn_post_norm: CudaSlice<f32>,
    ffn_norm: CudaSlice<f32>,
    ffn_gate: RepackedQ8,
    ffn_up: RepackedQ8,
    ffn_down: RepackedQ8,
    ffn_post_norm: CudaSlice<f32>,
    out_scale: f32,
    is_swa: bool,
    head_dim: usize,
    /// main-model layer whose KV plane this layer attends (read-only)
    share_il: usize,
}

pub(crate) struct MtpDrafter {
    layers: Vec<DrLayer>,
    /// nextn.pre_projection [2*n_embd_main -> n_embd]
    pre_proj: RepackedQ8,
    /// nextn.post_projection [n_embd -> n_embd_main] (the chained h)
    post_proj: RepackedQ8,
    output_norm: CudaSlice<f32>,
    /// drafter's own tied embedding as the LM head [n_embd -> n_vocab]
    head: RepackedQ8,
    /// per-pair rope factors for the global layer (drafter's own copy)
    rope_factors: CudaSlice<f32>,
    n_embd: usize,      // 1024
    n_embd_main: usize, // 5376
    n_head: usize,      // 32
    n_vocab: usize,
    n_ff: usize, // 8192
    eps: f32,
    theta_swa: f32,
    theta_global: f32,
    swa_window: usize,

    // ── scratch, all DR_MAX_ROWS-row (small: the wide ones are logits/h) ──
    xh: CudaSlice<f32>,             // [R, 2*n_embd_main] concat(embd, h)
    x: CudaSlice<f32>,              // [R, n_embd] residual stream
    normed: CudaSlice<f32>,         // [R, n_embd]
    q: CudaSlice<f32>,              // [R, n_head*512]
    qn: CudaSlice<f32>,             // normed+roped q
    attn: CudaSlice<f32>,           // [R, n_head*512]
    o: CudaSlice<f32>,              // wo / ffn_down out [R, n_embd]
    gate: CudaSlice<f32>,           // [R, n_ff]
    up: CudaSlice<f32>,             // [R, n_ff]
    h: CudaSlice<f32>,              // [R, n_embd_main] chain h input
    emb: CudaSlice<f32>,            // [R, n_embd_main] main-embd gather
    logits: CudaSlice<f32>,         // [R, n_vocab]
    pos: CudaSlice<u32>,            // [R] true rope positions (per step)
    attn_pos: CudaSlice<u32>,       // [R] clamped attention bound (per round)
    slots: CudaSlice<u32>,          // [R] row -> KV slot
    tok: CudaSlice<u32>,            // [R] chain tokens (device - argmax feeds gather)
    hidx: CudaSlice<u32>,           // [R] pf_normed row indices for the h gather (stage C)
    par: CudaSlice<u32>,            // [R*4] all-greedy sample_rows params
    pub(crate) out: CudaSlice<u32>, // [DR_MAX_K * R] draft picks
    // async round: [pend|srcrow|ndr|clen|base] meta for the
    // device-side verify-token assembly (pd_spec_toks) - 5 lanes over the
    // service's live-slot count, one pageable upload per round
    pub(crate) asm_meta: CudaSlice<u32>,
    // rung B1: the device accept's per-slot output strip,
    // stride 4 + DR_MAX_K u32 per slot
    strip: CudaSlice<u32>,
    // gemma_qkv_nra with n_kv=0 never reads/writes its k/v arguments - these
    // dummies only satisfy the wrapper's reference types
    kv_dummy_f: CudaSlice<f32>,
    kv_dummy_c1: CudaSlice<u8>,
    kv_dummy_c2: CudaSlice<u8>,
    // Canonical rejection sampling (PADDOCK_SPEC_RS): sampled-draft
    // chain state + the fp16 q-store the verify resolve reads. None when the
    // arm is off - zero cost.
    pub(crate) rs: Option<RsBufs>,
}

/// Canonical-RS device state. The q-store is [DR_MAX_K,
/// DR_MAX_ROWS, n_vocab] fp16 (~268 MB at the 262k vocab) - allocated at
/// drafter load, before enable_batch sizes its budget, so the slot fit
/// absorbs it. One buffer, no double-banking: chain N writes it, verify N's
/// resolve reads it, chain N+1 is stream-ordered after both.
pub(crate) struct RsBufs {
    /// fp16 exp mass per (step, chain row): q is this rounded row
    pub(crate) qstore: CudaSlice<u16>,
    /// exact f32 sums of the stored rows; 0 marks a greedy (argmax) row
    pub(crate) qsum: CudaSlice<f32>,
    /// per-chain-row 1/T for the draft draw (0 = argmax)
    pub(crate) invt: CudaSlice<f32>,
    /// [k_use, rr] per-step draft-draw uniforms
    pub(crate) uplane: CudaSlice<f32>,
    /// device chain-step counter (reset per round, +1 inside the graph)
    pub(crate) step: CudaSlice<u32>,
    /// verify-resolve params, 8 u32 words per drafted row
    pub(crate) par: CudaSlice<u32>,
}

/// Drafted-row ceiling the RS resolve param buffer is sized for (the service
/// row budget is 128; 1024 leaves an order of magnitude of headroom).
pub(crate) const RS_PAR_ROWS: usize = 1024;

fn drv(e: cudarc::driver::DriverError) -> LoadError {
    LoadError::Tensor("mtp scratch".into(), e.to_string())
}

/// Rope-convention selector for chained draft steps (see the chain-loop
/// comment): 0 = advance +1 per draft step (the default), 1 = hold at p+1,
/// 2 = hold at p (the vLLM/SGLang last-written-slot convention).
/// Process-constant - the rr-keyed chain graphs bake one mode.
fn spec_rope_mode() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_ROPE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&m| m <= 2)
            .unwrap_or(0)
    })
}

/// PADDOCK_SPEC_RS=1 arms canonical rejection sampling - drafts
/// SAMPLED from the drafter softmax at the request temperature, verify
/// accepts with prob min(1, p/q) and recovers from the residual. Lossless
/// (same emitted distribution as the classic rule); the lever is expected
/// acceptance: sum min(p,q) vs p(argmax q) on broad distributions.
/// Process-constant - chain and decode graphs bake the election.
pub(crate) fn spec_rs_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_SPEC_RS").is_some())
}

/// Stage C kill switch: batched drafter stitches (xh interleave
/// + chain-h gather) instead of 224 single-row DtoD memcpys per wide round.
///   Movement is bit-identical; the win is host issue time.
fn spec_stitch_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_SPEC_STITCH").is_none())
}

impl MtpDrafter {
    pub(crate) fn load(
        exec: &GpuExecutor,
        map: &MappedGguf,
        // main-model facts to validate the share contract against
        main_n_embd: usize,
        main_n_vocab: usize,
        main_n_layer: usize,
        main_share_swa: (usize, usize), // (head_dim, n_kv) of layer n-2
        main_share_global: (usize, usize), // (head_dim, n_kv) of layer n-1
    ) -> Result<Self, LoadError> {
        use super::load::{key_bool_array, key_f32, key_u64};
        let arch = map
            .gguf()
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if arch != "gemma4-assistant" {
            return Err(LoadError::BadKey(format!(
                "mtp gguf arch {arch:?} (want gemma4-assistant)"
            )));
        }
        let n_layer = key_u64(map, "gemma4-assistant.block_count")? as usize;
        let n_embd = key_u64(map, "gemma4-assistant.embedding_length")? as usize;
        let n_embd_main = key_u64(map, "gemma4-assistant.embedding_length_out")? as usize;
        let n_ff = key_u64(map, "gemma4-assistant.feed_forward_length")? as usize;
        let n_head = key_u64(map, "gemma4-assistant.attention.head_count")? as usize;
        let hd_global = key_u64(map, "gemma4-assistant.attention.key_length")? as usize;
        let hd_swa = key_u64(map, "gemma4-assistant.attention.key_length_swa")? as usize;
        let swa_window = key_u64(map, "gemma4-assistant.attention.sliding_window")? as usize;
        let eps = key_f32(
            map,
            "gemma4-assistant.attention.layer_norm_rms_epsilon",
            None,
        )?;
        let base_global = key_f32(map, "gemma4-assistant.rope.freq_base", Some(1_000_000.0))?;
        let base_swa = key_f32(map, "gemma4-assistant.rope.freq_base_swa", Some(10_000.0))?;
        let is_swa = key_bool_array(map, "gemma4-assistant.attention.sliding_window_pattern")?;
        let shared = key_u64(map, "gemma4-assistant.attention.shared_kv_layers")? as usize;

        if n_embd_main != main_n_embd {
            return Err(LoadError::BadKey(format!(
                "mtp embedding_length_out {n_embd_main} != main n_embd {main_n_embd}"
            )));
        }
        if shared != n_layer || is_swa.len() != n_layer {
            return Err(LoadError::BadKey(
                "mtp shared_kv_layers must cover every layer".into(),
            ));
        }
        // the reference share map: SWA layers -> main n-2, global -> main n-1
        for (i, &swa) in is_swa.iter().enumerate() {
            let (hd, want) = if swa {
                (hd_swa, main_share_swa)
            } else {
                (hd_global, main_share_global)
            };
            if hd != want.0 {
                return Err(LoadError::BadKey(format!(
                    "mtp layer {i} head_dim {hd} != shared main plane's {}",
                    want.0
                )));
            }
        }

        let norm = |name: String| -> Result<CudaSlice<f32>, LoadError> {
            Ok(exec.upload(map, &name)?.buf)
        };
        let mut layers = Vec::with_capacity(n_layer);
        for (i, &swa) in is_swa.iter().enumerate() {
            let out_scale = match map.tensor_bytes(&format!("blk.{i}.layer_output_scale.weight")) {
                Ok((_, bytes)) => f32::from_le_bytes(bytes[..4].try_into().expect("f32 scalar")),
                Err(_) => 1.0,
            };
            layers.push(DrLayer {
                attn_norm: norm(format!("blk.{i}.attn_norm.weight"))?,
                wq: exec.repack_q8(map, &format!("blk.{i}.attn_q.weight"))?,
                q_norm: norm(format!("blk.{i}.attn_q_norm.weight"))?,
                wo: exec.repack_q8(map, &format!("blk.{i}.attn_output.weight"))?,
                attn_post_norm: norm(format!("blk.{i}.post_attention_norm.weight"))?,
                ffn_norm: norm(format!("blk.{i}.ffn_norm.weight"))?,
                ffn_gate: exec.repack_q8(map, &format!("blk.{i}.ffn_gate.weight"))?,
                ffn_up: exec.repack_q8(map, &format!("blk.{i}.ffn_up.weight"))?,
                ffn_down: exec.repack_q8(map, &format!("blk.{i}.ffn_down.weight"))?,
                ffn_post_norm: norm(format!("blk.{i}.post_ffw_norm.weight"))?,
                out_scale,
                is_swa: swa,
                head_dim: if swa { hd_swa } else { hd_global },
                share_il: if swa {
                    main_n_layer - 2
                } else {
                    main_n_layer - 1
                },
            });
        }

        let head = exec.repack_q8(map, "token_embd.weight")?;
        let n_vocab = head.dims[1];
        if n_vocab != main_n_vocab {
            return Err(LoadError::BadKey(format!(
                "mtp vocab {n_vocab} != main {main_n_vocab}"
            )));
        }

        let r = DR_MAX_ROWS;
        let s = &exec.stream;
        let hd_max = hd_global.max(hd_swa);
        // all-greedy sampler params, uploaded once (mode 1 at word 2)
        let mut par_host = vec![0u32; r * 4];
        for i in 0..r {
            par_host[i * 4 + 2] = 1;
        }
        let mut par = s.alloc_zeros::<u32>(r * 4).map_err(drv)?;
        s.memcpy_htod(&par_host, &mut par).map_err(drv)?;

        Ok(Self {
            pre_proj: exec.repack_q8(map, "nextn.pre_projection.weight")?,
            post_proj: exec.repack_q8(map, "nextn.post_projection.weight")?,
            output_norm: norm("output_norm.weight".into())?,
            head,
            rope_factors: exec.upload(map, "rope_freqs.weight")?.buf,
            layers,
            n_embd,
            n_embd_main,
            n_head,
            n_vocab,
            n_ff,
            eps,
            theta_swa: base_swa.powf(-2.0 / hd_swa as f32),
            theta_global: base_global.powf(-2.0 / hd_global as f32),
            swa_window,
            xh: s.alloc_zeros::<f32>(r * 2 * n_embd_main).map_err(drv)?,
            x: s.alloc_zeros::<f32>(r * n_embd).map_err(drv)?,
            normed: s.alloc_zeros::<f32>(r * n_embd).map_err(drv)?,
            q: s.alloc_zeros::<f32>(r * n_head * hd_max).map_err(drv)?,
            qn: s.alloc_zeros::<f32>(r * n_head * hd_max).map_err(drv)?,
            attn: s.alloc_zeros::<f32>(r * n_head * hd_max).map_err(drv)?,
            o: s.alloc_zeros::<f32>(r * n_embd).map_err(drv)?,
            gate: s.alloc_zeros::<f32>(r * n_ff).map_err(drv)?,
            up: s.alloc_zeros::<f32>(r * n_ff).map_err(drv)?,
            h: s.alloc_zeros::<f32>(r * n_embd_main).map_err(drv)?,
            emb: s.alloc_zeros::<f32>(r * n_embd_main).map_err(drv)?,
            logits: s.alloc_zeros::<f32>(r * n_vocab).map_err(drv)?,
            pos: s.alloc_zeros::<u32>(r).map_err(drv)?,
            attn_pos: s.alloc_zeros::<u32>(r).map_err(drv)?,
            slots: s.alloc_zeros::<u32>(r).map_err(drv)?,
            tok: s.alloc_zeros::<u32>(r).map_err(drv)?,
            hidx: s.alloc_zeros::<u32>(r).map_err(drv)?,
            par,
            out: s.alloc_zeros::<u32>(DR_MAX_K * r).map_err(drv)?,
            // 5 meta lanes over the service's max live width (spec_live_max
            // cap is 64-class; 128 is safe headroom, 2.5 KB)
            asm_meta: s.alloc_zeros::<u32>(5 * 128).map_err(drv)?,
            strip: s
                .alloc_zeros::<u32>((4 + DR_MAX_K) * 128 * 2)
                .map_err(drv)?,
            kv_dummy_f: s.alloc_zeros::<f32>(4).map_err(drv)?,
            kv_dummy_c1: s.alloc_zeros::<u8>(4).map_err(drv)?,
            kv_dummy_c2: s.alloc_zeros::<u8>(4).map_err(drv)?,
            // the RS arm needs both new kernels AND the device
            // step counter (the chain graph advances it via u32_addk)
            rs: (spec_rs_on() && exec.has_spec_rs() && exec.has_u32_addk())
                .then(|| -> Result<RsBufs, LoadError> {
                    tracing::info!(
                        "[g4-rs] canonical rejection sampling armed (q-store {} MB)",
                        DR_MAX_K * r * n_vocab * 2 / (1 << 20)
                    );
                    Ok(RsBufs {
                        qstore: s.alloc_zeros::<u16>(DR_MAX_K * r * n_vocab).map_err(drv)?,
                        qsum: s.alloc_zeros::<f32>(DR_MAX_K * r).map_err(drv)?,
                        invt: s.alloc_zeros::<f32>(r).map_err(drv)?,
                        uplane: s.alloc_zeros::<f32>(DR_MAX_K * r).map_err(drv)?,
                        step: s.alloc_zeros::<u32>(1).map_err(drv)?,
                        par: s.alloc_zeros::<u32>(RS_PAR_ROWS * 8).map_err(drv)?,
                    })
                })
                .transpose()?,
        })
    }
}

impl GpuGemma4 {
    /// Attach the separate-model MTP drafter (mtp-*.gguf). Requires the main
    /// model's last two layers to be the (SWA, global) pair the drafter's
    /// shared-KV contract expects.
    pub fn attach_mtp(&mut self, map: &MappedGguf) -> Result<(), LoadError> {
        if paddock_models::dev_var_os!("PADDOCK_G4_NO_MTP").is_some() {
            return Ok(());
        }
        // The f8t (cutlass-class) GEMM intercept is a WIDTH GATE on the spec
        // lane, not a blanket kill. Its 128x128 tile padding loses on a
        // narrow (~18-row) c4-spec verify but wins on a wide (~96-row)
        // c32-spec verify, so the intercept stays live with a 32-row spec
        // floor. 32 and not higher because a c32-spec serve has no f8t call
        // in 16..31 rows at all (decode 32, chunks 65..256, verify ~56..296),
        // so a 32 floor is exactly the wide arm while narrow verifies stay on
        // tc5r. Overrides for A/B: PADDOCK_SPEC_F8CUT=1 fully on (no spec
        // floor), =0 blanket kill.
        if paddock_models::dev_var_os!("PADDOCK_SPEC_F8CUT").is_some() {
            if paddock_models::dev_on!("PADDOCK_SPEC_F8CUT") {
                tracing::info!("spec lane: F8CUT dispatch left ON (PADDOCK_SPEC_F8CUT=1)");
            } else {
                tracing::info!("spec lane: F8CUT blanket kill (PADDOCK_SPEC_F8CUT=0)");
                self.exec.set_f8cut_spec_off();
            }
        } else {
            self.exec.set_f8cut_spec_minb(32);
            tracing::info!("spec lane: F8CUT width-gated at 32 rows");
        }
        // Wide-batch spec: the service's per-round row budget defaults to 32
        // (the qwen35/gpt-oss verify caps) and its device-plans live gate to
        // 4 (also a qwen35 shape). With this drafter attached the tick
        // buffers hold slots*(K+1) rows and spec pays at every width, so
        // raise both - at live 32 a 32-row budget zeroes the draft allowance
        // and spec never engages. Explicit env still overrides.
        // SAFETY: called once at model load, before serving threads spawn.
        // Rounds wider than ~128 rows regress on the M-col B-direct GEMM
        // rung: the direct-L2 B fragments stall the mma issue chain and the
        // verify tick blows up. The M-col kernel + knobs stay for the
        // smem-staged-B iteration (PADDOCK_SPEC_MAX_ROWS=160 to re-engage).
        // Narrow tier stays 32 rows (BN32 single pass).
        if std::env::var_os("PADDOCK_SPEC_MAX_ROWS").is_none() {
            // 128 is the row-budget optimum: a wide verify tick amortizes the
            // same ~30GB weight stream over (b/live)-1 drafts per slot, so
            // 64->128 lifts live-32 slots from k=1 to k=3. Past the top spec
            // waste wins again, hence a peak rather than a monotone curve.
            // Narrow tiers are b-independent (they pin 32 rows).
            crate::envset::set_env("PADDOCK_SPEC_MAX_ROWS", "128");
        }
        // Post-miss k floor 4: the classic drop-to-accepted rule makes a
        // ~78%/token drafter spend whole rounds re-climbing. A floor of 7
        // gives some of that back - keep some adaptivity. Budget-clamped
        // configs (c8 k3, c32 k1) are untouched: the floor mins with
        // slot.k_now.
        if std::env::var_os("PADDOCK_SPEC_K_MISS_FLOOR").is_none() {
            crate::envset::set_env("PADDOCK_SPEC_K_MISS_FLOOR", "4");
        }
        // Deep chains measure NEGATIVE on this drafter: 15 sequential drafter
        // steps cost more than the extra acceptance buys. Knobs stay for
        // A/Bs: PADDOCK_SPEC_MAX_K, PADDOCK_SPEC_DEEP_LIVE_MAX (deep tier
        // default-off via live<=0).
        if std::env::var_os("PADDOCK_QWEN35_SPEC_PLANS_LIVE_MAX").is_none() {
            crate::envset::set_env("PADDOCK_QWEN35_SPEC_PLANS_LIVE_MAX", "32");
        }
        let n = self.hp.n_layer;
        let (l_swa, l_glob) = (&self.layers[n - 2], &self.layers[n - 1]);
        if !l_swa.is_swa || l_glob.is_swa {
            return Err(LoadError::BadKey(
                "main layers n-2/n-1 are not the (SWA, global) pair the drafter shares".into(),
            ));
        }
        let m = MtpDrafter::load(
            &self.exec,
            map,
            self.hp.n_embd,
            self.hp.n_vocab,
            n,
            (l_swa.head_dim, l_swa.n_kv_heads),
            (l_glob.head_dim, l_glob.n_kv_heads),
        )?;
        self.mtp = Some(m);
        Ok(())
    }

    /// One drafter forward over `r` chained rows: token ids in `m.tok`
    /// (device), h in `m.h`, true rope positions in `m.pos`, attention bound
    /// in `m.attn_pos`. Leaves greedy picks in `m.tok` and the chained h in
    /// `m.h` - ready for the next step without any host round-trip.
    fn mtp_step(&mut self, r: usize) -> Result<(), GpuError> {
        let exec = self.exec.clone();
        let m = self.mtp.as_mut().expect("mtp attached");
        let sc = &mut self.scratch;
        let (n_embd, n_main, n_ff) = (m.n_embd, m.n_embd_main, m.n_ff);

        // xh rows: [ main_embd(t)·√n_main | h ]. Stage C: the
        // per-row copy_region pair issued 2r host memcpys per step (~5-8us
        // CPU each - 192/round of the wide round's 224 DtoD calls); the
        // stitch kernel is one launch, bit-identical movement.
        // PADDOCK_NO_SPEC_STITCH restores the loop.
        super::EmbdTable::of(&self.token_embd, &self.head).gather(
            &exec,
            &m.tok,
            &mut m.emb,
            n_main,
            r,
            (n_main as f32).sqrt(),
        )?;
        if spec_stitch_on() && exec.has_spec_xh_stitch() {
            exec.spec_xh_stitch(&m.emb, &m.h, &mut m.xh, r, n_main)?;
        } else {
            for i in 0..r {
                exec.copy_region(&m.emb, i * n_main, &mut m.xh, i * 2 * n_main, n_main)?;
                exec.copy_region(&m.h, i * n_main, &mut m.xh, i * 2 * n_main + n_main, n_main)?;
            }
        }
        exec.quantize_q8(&m.xh, &mut sc.pf_xq, &mut sc.pf_xs, r * 2 * n_main)?;
        exec.q8_0_gemm_mma_ks(
            &m.pre_proj,
            &sc.pf_xq,
            &sc.pf_xs,
            &mut sc.pf_skfix,
            &mut m.x,
            r,
        )?;

        for li in 0..m.layers.len() {
            let l = &m.layers[li];
            let hd = l.head_dim;
            let q_dim = m.n_head * hd;
            let (kvl, main_lw) = (&self.kv[l.share_il], &self.layers[l.share_il]);
            let (n_kv, kv_dim) = (main_lw.n_kv_heads, kvl.kv_dim);
            // theta comes off the DRAFTER's own rope bases; the rest of the
            // rope shape is the TARGET's, because a decoder-shaped drafter
            // block that ropes on a different convention (or ropes a layer
            // the target leaves NoPE) drafts from a different model and
            // quietly tanks acceptance. Same reasoning as the FFN activation.
            let theta = if l.is_swa {
                m.theta_swa
            } else {
                m.theta_global
            };
            let freq_scale = if l.is_swa {
                self.hp.rope_swa.1
            } else {
                self.hp.rope_global.1
            };
            let window = if l.is_swa { m.swa_window } else { 0 };
            // see Hparams::attn_scale - gemma4 1.0, muse-glimmer 1/sqrt(hd)
            let ascale = self.hp.attn_scale(hd);
            let layer_bt: Option<(&CudaSlice<u32>, usize)> = if l.is_swa {
                self.paging.as_ref().map(|pg| (&pg.bt, pg.bps))
            } else {
                self.gpool.as_ref().map(|gp| (&gp.d_bt, gp.bps))
            };

            exec.rmsnorm_batch(&m.x, &l.attn_norm, &mut m.normed, n_embd, m.eps, r)?;
            exec.quantize_q8(&m.normed, &mut sc.pf_xq, &mut sc.pf_xs, r * n_embd)?;
            exec.q8_0_gemm_mma_ks(&l.wq, &sc.pf_xq, &sc.pf_xs, &mut sc.pf_skfix, &mut m.q, r)?;

            // Q-only epilogue: gemma_qkv_nra with n_kv=0 degenerates to
            // per-head q RMS norm + rope (kinds 1/2 unreachable - the k/v/
            // cache arguments are never touched; dummies satisfy the types).
            {
                let factors = (!l.is_swa).then_some(&m.rope_factors);
                let kdf: *mut CudaSlice<f32> = &mut m.kv_dummy_f;
                // SAFETY: with n_kv=0 the kernel launches n_head q blocks
                // only; the aliased dummy pointers are never dereferenced.
                unsafe {
                    exec.gemma_qkv_nra(
                        (&mut m.q, 0),
                        (&mut *kdf, 0),
                        (&mut *kdf, 0),
                        &l.q_norm,
                        &l.q_norm, // wk_norm: unread for q blocks
                        &mut m.qn,
                        &mut m.kv_dummy_c1,
                        &mut m.kv_dummy_c2,
                        &m.pos,
                        None,
                        factors,
                        None,
                        0,
                        m.n_head,
                        0,
                        hd,
                        0,
                        r,
                        m.eps,
                        (theta, freq_scale, 0.0, 0.0, 0.0, 0.0),
                        crate::gpu::KvDtype::Fp16,
                        0,
                        self.hp.rope_neox(),
                        self.hp.v_norm(),
                    )?;
                }
            }

            // attention over the shared main plane, bound by m.attn_pos (the
            // committed head - rejected/future cache rows are never read)
            let splits = super::batch::attn_splits(m.n_head, r, exec.sm_count(), self.n_slots);
            if splits > 1 {
                let (ao, aml) = self
                    .attn_scratch
                    .as_mut()
                    .expect("enable_batch allocates attention scratch");
                match layer_bt {
                    Some((bt, bps)) => exec.attn_partial_batch_paged(
                        &m.qn,
                        &kvl.k,
                        &kvl.v,
                        ao,
                        aml,
                        &m.attn_pos,
                        Some(&m.slots),
                        bt,
                        bps,
                        m.n_head,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        splits,
                        r,
                        ascale,
                        kvl.dtype,
                    )?,
                    _ => exec.attn_partial_batch(
                        &m.qn,
                        &kvl.k,
                        &kvl.v,
                        ao,
                        aml,
                        &m.attn_pos,
                        Some(&m.slots),
                        m.n_head,
                        n_kv,
                        hd,
                        self.max_ctx,
                        kv_dim,
                        window,
                        splits,
                        r,
                        ascale,
                        kvl.dtype,
                    )?,
                }
                exec.attn_combine_batch(
                    ao,
                    aml,
                    &sc.neg_inf_sinks,
                    &mut m.attn,
                    m.n_head,
                    hd,
                    splits,
                    r,
                )?;
            } else {
                match layer_bt {
                    Some((bt, bps)) => exec.attn_decode_batch_paged(
                        &m.qn,
                        &kvl.k,
                        &kvl.v,
                        &sc.neg_inf_sinks,
                        &mut m.attn,
                        &m.attn_pos,
                        Some(&m.slots),
                        bt,
                        bps,
                        m.n_head,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        r,
                        ascale,
                        kvl.dtype,
                    )?,
                    _ => exec.attn_decode_batch(
                        &m.qn,
                        &kvl.k,
                        &kvl.v,
                        &sc.neg_inf_sinks,
                        &mut m.attn,
                        &m.attn_pos,
                        Some(&m.slots),
                        m.n_head,
                        n_kv,
                        hd,
                        self.max_ctx,
                        kv_dim,
                        window,
                        r,
                        ascale,
                        kvl.dtype,
                    )?,
                }
            }

            exec.quantize_q8(&m.attn, &mut sc.pf_xq, &mut sc.pf_xs, r * q_dim)?;
            exec.q8_0_gemm_mma_ks(&l.wo, &sc.pf_xq, &sc.pf_xs, &mut sc.pf_skfix, &mut m.o, r)?;
            exec.rmsnorm_add_scale(&mut m.x, &m.o, &l.attn_post_norm, n_embd, m.eps, 1.0, r)?;

            exec.rmsnorm_batch(&m.x, &l.ffn_norm, &mut m.normed, n_embd, m.eps, r)?;
            exec.quantize_q8(&m.normed, &mut sc.pf_xq, &mut sc.pf_xs, r * n_embd)?;
            exec.q8_0_gemm_mma_ks(
                &l.ffn_gate,
                &sc.pf_xq,
                &sc.pf_xs,
                &mut sc.pf_skfix,
                &mut m.gate,
                r,
            )?;
            exec.q8_0_gemm_mma_ks(
                &l.ffn_up,
                &sc.pf_xq,
                &sc.pf_xs,
                &mut sc.pf_skfix,
                &mut m.up,
                r,
            )?;
            // the drafter block is decoder-shaped, so it takes the TARGET's
            // FFN activation - a drafter folding the other nonlinearity would
            // propose from a different model and quietly tank acceptance
            exec.glu(&mut m.gate, &m.up, r * n_ff, self.hp.glu_act())?;
            exec.quantize_q8(&m.gate, &mut sc.pf_xq, &mut sc.pf_xs, r * n_ff)?;
            exec.q8_0_gemm_mma_ks(
                &l.ffn_down,
                &sc.pf_xq,
                &sc.pf_xs,
                &mut sc.pf_skfix,
                &mut m.o,
                r,
            )?;
            exec.rmsnorm_add_scale(
                &mut m.x,
                &m.o,
                &l.ffn_post_norm,
                n_embd,
                m.eps,
                l.out_scale,
                r,
            )?;
        }

        exec.rmsnorm_batch(&m.x, &m.output_norm, &mut m.normed, n_embd, m.eps, r)?;
        exec.quantize_q8(&m.normed, &mut sc.pf_xq, &mut sc.pf_xs, r * n_embd)?;
        exec.q8_0_gemm_mma_ks(
            &m.head,
            &sc.pf_xq,
            &sc.pf_xs,
            &mut sc.pf_skfix,
            &mut m.logits,
            r,
        )?;
        // next chain h (from the same normed activation quant - but the head
        // GEMM consumed pf_xq; requantize is cheaper than a second xq plane)
        exec.quantize_q8(&m.normed, &mut sc.pf_xq, &mut sc.pf_xs, r * n_embd)?;
        exec.q8_0_gemm_mma_ks(
            &m.post_proj,
            &sc.pf_xq,
            &sc.pf_xs,
            &mut sc.pf_skfix,
            &mut m.h,
            r,
        )?;
        // picks -> m.tok (device): the next step's gather reads them.
        // Classic: greedy argmax. RS: sampled draw from the
        // drafter softmax, materializing q into the step-indexed store -
        // the branch is env-constant, so every captured chain graph bakes
        // one sampler consistently.
        if m.rs.is_some() {
            let n_vocab = m.n_vocab;
            let MtpDrafter {
                logits, tok, rs, ..
            } = m;
            let rs = rs.as_mut().expect("checked");
            exec.draft_rs(
                logits,
                &rs.invt,
                &rs.uplane,
                &rs.step,
                &mut rs.qstore,
                &mut rs.qsum,
                tok,
                r,
                n_vocab,
                DR_MAX_ROWS,
            )?;
        } else {
            exec.sample_rows(&m.logits, &m.par, &mut m.tok, r, m.n_vocab)?;
        }
        Ok(())
    }

    /// Serving draft hook: chain the drafter `k` steps for the live pending
    /// tokens. Declines (None) when the h map is stale for any slot - the
    /// next verify/dense tick re-freshens it.
    /// Synchronous drafts (the pre-Phase-35 contract): chain + host readback
    /// in one call. Kept for the mixed-tick path and as the fallback when
    /// the async round can't arm.
    pub(crate) fn spec_draft_batch_impl(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GpuError> {
        if self.spec_draft_begin_impl(pendings, k)?.is_none() {
            return Ok(None);
        }
        self.spec_draft_fetch_impl()
    }

    /// Async spec round, phase 1: enqueue the drafter chain and
    /// return its effective depth without reading drafts back - the caller
    /// launches the verify tick immediately (device-side token assembly
    /// bridges the values) and fetches drafts after, so the chain->verify
    /// boundary is a fully queued stream sequence instead of a host stall
    /// (g4w9 gap anatomy: 487ms/25s at wide, 140ms at churn on exactly
    /// this boundary). Ok(Some(k_use)) arms `spec_async`; every armed call
    /// must be paired with spec_draft_fetch_impl before the next round.
    pub(crate) fn spec_draft_begin_impl(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<(usize, Vec<bool>)>, GpuError> {
        let dbg = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some();
        if self.mtp.is_none() || pendings.is_empty() || k == 0 {
            if dbg {
                tracing::info!(
                    "[draft-decl] early mtp={} n={} k={}",
                    self.mtp.is_some(),
                    pendings.len(),
                    k
                );
            }
            return Ok(None);
        }
        let r = pendings.len();
        if r > spec_live_max().min(DR_MAX_ROWS) {
            if dbg {
                tracing::info!("[draft-decl] rowcap r={r}");
            }
            return Ok(None);
        }
        let k_use = k.min(DR_MAX_K);
        // h rows come from the last sampled tick's row map. Per-sequence
        // independence (the SOTA ground vLLM has): a slot missing its row
        // does not decline the whole batch - it is skipped and gets no
        // drafts this round. All-or-nothing declining zeroes out wide-batch
        // drafting almost every round under admission/chunk churn.
        let mut h_rows = Vec::with_capacity(r);
        let mut keep: Vec<usize> = Vec::with_capacity(r);
        for (i, &(slot, _)) in pendings.iter().enumerate() {
            if let Some(pr) = self.spec_rows.get(slot).copied().flatten() {
                h_rows.push(pr);
                keep.push(i);
            } else if dbg {
                tracing::warn!("[draft-decl] cold slot={slot} (skipped, batch continues)");
            }
        }
        if h_rows.is_empty() {
            if dbg {
                tracing::info!("[draft-decl] all cold n={r}");
            }
            return Ok(None);
        }
        let rr = h_rows.len();
        let exec = self.exec.clone();
        {
            let m = self.mtp.as_mut().expect("mtp");
            let n_main = m.n_embd_main;
            // Stage C: rr host-known rows - one tiny index upload
            // + one gather launch instead of rr single-row DtoD memcpys.
            if spec_stitch_on() && exec.has_hrow_gather() {
                let rows_u32: Vec<u32> = h_rows.iter().map(|&(_, row)| row).collect();
                let e = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
                let mut v = m
                    .hidx
                    .try_slice_mut(0..rr)
                    .ok_or_else(|| GpuError::Driver("hidx".into()))?;
                exec.stream.memcpy_htod(&rows_u32, &mut v).map_err(e)?;
                exec.hrow_gather(&self.scratch.pf_normed, &m.hidx, &mut m.h, rr, n_main)?;
            } else {
                for (i, &(_, row)) in h_rows.iter().enumerate() {
                    exec.copy_region(
                        &self.scratch.pf_normed,
                        row as usize * n_main,
                        &mut m.h,
                        i * n_main,
                        n_main,
                    )?;
                }
            }
            let toks: Vec<u32> = keep.iter().map(|&i| pendings[i].1).collect();
            let slots: Vec<u32> = keep.iter().map(|&i| pendings[i].0 as u32).collect();
            let bounds: Vec<u32> = h_rows.iter().map(|&(p, _)| p).collect();
            let e = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
            let mut v = m
                .tok
                .try_slice_mut(0..rr)
                .ok_or_else(|| GpuError::Driver("tok".into()))?;
            exec.stream.memcpy_htod(&toks, &mut v).map_err(e)?;
            let mut v = m
                .slots
                .try_slice_mut(0..rr)
                .ok_or_else(|| GpuError::Driver("slots".into()))?;
            exec.stream.memcpy_htod(&slots, &mut v).map_err(e)?;
            let mut v = m
                .attn_pos
                .try_slice_mut(0..rr)
                .ok_or_else(|| GpuError::Driver("apos".into()))?;
            exec.stream.memcpy_htod(&bounds, &mut v).map_err(e)?;
        }
        let t_chain = std::time::Instant::now();
        // Chain-step rope positions: upload once (step-0 values); the
        // captured graph carries a device-side u32_addk(+1) at its tail so
        // steps replay back-to-back with no host copies between them. The
        // per-step pageable memcpy_htod was an implicit sync - it shows up
        // as a 5-20us GPU idle band per step. Positions per step are
        // identical to the host-copy path by construction.
        //
        // ROPE CONVENTION (PADDOCK_SPEC_ROPE): the other engines both
        // hold the drafter's rope position constant across chained steps -
        // vLLM's gemma4 proposer (constant_draft_positions = True, "all
        // draft steps predict from the same last target-model position",
        // v1/spec_decode/gemma4.py:44-47) and SGLang's frozen-KV MTP
        // worker ("Rope phase = last written target slot, not advanced per
        // draft step"). Ours advances +1 per step. The drafter is Q-only
        // (n_kv=0 - nothing appended), so the mode changes only the Q
        // rotation per step; every draft is still target-validated, so
        // output distribution is untouched - acceptance rate is the metric.
        //   0 = advance +1 per step (the default), 1 = hold at p+1,
        //   2 = hold at p (the last-written-slot convention).
        // Process-constant, so the rr-keyed chain graphs bake one mode.
        let rope_mode = spec_rope_mode();
        let dev_pos = exec.has_u32_addk();
        {
            let m = self.mtp.as_mut().expect("mtp");
            let off = if rope_mode == 2 { 0 } else { 1 };
            let rope_pos: Vec<u32> = h_rows.iter().map(|&(p, _)| p + off).collect();
            let mut v = m
                .pos
                .try_slice_mut(0..rr)
                .ok_or_else(|| GpuError::Driver("pos".into()))?;
            exec.stream
                .memcpy_htod(&rope_pos, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        // RS: per-round chain-draw state - inv_t per kept row, the
        // [k_use x rr] uniform plane, the device step-counter reset. The
        // captured chain graph reads all three. Missing draws (the service
        // stashed nothing this round) degrade every row to inv_t 0 = argmax
        // chain, whose greedy q-store marker the resolver handles - the
        // round is then just the classic scheme.
        let rs_active = self.mtp.as_ref().is_some_and(|m| m.rs.is_some());
        if rs_active {
            let draws = self.spec_rs_draws.take();
            let mut invt = vec![0.0f32; rr];
            let mut upl = vec![0.0f32; k_use * rr];
            if let Some(dr) = draws.as_ref() {
                for (i, &ki) in keep.iter().enumerate() {
                    let slot = pendings[ki].0;
                    if let Some(d) = dr.iter().find(|d| d.slot == slot) {
                        invt[i] = d.inv_t;
                        for (j, u) in upl.chunks_mut(rr).enumerate().take(k_use) {
                            u[i] = d.u.get(j).copied().unwrap_or(0.5);
                        }
                    }
                }
            }
            let m = self.mtp.as_mut().expect("mtp");
            let rs = m.rs.as_mut().expect("rs_active");
            let e = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
            let mut v = rs
                .invt
                .try_slice_mut(0..rr)
                .ok_or_else(|| GpuError::Driver("rs invt".into()))?;
            exec.stream.memcpy_htod(&invt, &mut v).map_err(e)?;
            let mut v = rs
                .uplane
                .try_slice_mut(0..k_use * rr)
                .ok_or_else(|| GpuError::Driver("rs uplane".into()))?;
            exec.stream.memcpy_htod(&upl, &mut v).map_err(e)?;
            exec.stream.memcpy_htod(&[0u32], &mut rs.step).map_err(e)?;
        }
        for j in 0..k_use {
            if !dev_pos && j > 0 && rope_mode == 0 {
                // fallback for packs without u32_addk: the old per-step upload
                let m = self.mtp.as_mut().expect("mtp");
                let rope_pos: Vec<u32> = h_rows.iter().map(|&(p, _)| p + 1 + j as u32).collect();
                let mut v = m
                    .pos
                    .try_slice_mut(0..r)
                    .ok_or_else(|| GpuError::Driver("pos".into()))?;
                exec.stream
                    .memcpy_htod(&rope_pos, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            // graph-captured chain step (~60 launches -> 1 replay; ~10ms of
            // host-launch per round measured at c32). The h/tok inputs are
            // device buffers updated before the replay.
            if let Some(g) = self.mtp_graphs.get(&rr) {
                g.0.launch()
                    .map_err(|e| GpuError::Driver(format!("mtp graph launch: {e}")))?;
            } else {
                use cudarc::driver::sys::CUstreamCaptureMode;
                exec.stream
                    .synchronize()
                    .map_err(|e| GpuError::Driver(format!("mtp pre-capture sync: {e}")))?;
                exec.stream
                    .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .map_err(|e| GpuError::Driver(format!("mtp begin_capture: {e}")))?;
                let rec = self.mtp_step(rr);
                // held-rope modes bake no advance into the chain graph
                let rec = if dev_pos && rope_mode == 0 {
                    rec.and_then(|_| {
                        let m = self.mtp.as_mut().expect("mtp");
                        exec.u32_addk(&mut m.pos, rr, 1)
                    })
                } else {
                    rec
                };
                // the chain graph advances the RS step counter
                // itself so replays need no host patching (rs alloc is
                // gated on has_u32_addk)
                let rec = if rs_active {
                    rec.and_then(|_| {
                        let m = self.mtp.as_mut().expect("mtp");
                        let rs = m.rs.as_mut().expect("rs_active");
                        exec.u32_addk(&mut rs.step, 1, 1)
                    })
                } else {
                    rec
                };
                let graph = crate::gpu::end_capture_no_flags(&exec.stream)
                    .map_err(|e| GpuError::Driver(format!("mtp end_capture: {e}")));
                rec?;
                let graph = graph?
                    .ok_or_else(|| GpuError::Driver("mtp capture produced no graph".into()))?;
                let g = super::SendGraph(graph);
                g.0.launch()
                    .map_err(|e| GpuError::Driver(format!("mtp first launch: {e}")))?;
                self.mtp_graphs.insert(rr, g);
            }
            let m = self.mtp.as_mut().expect("mtp");
            exec.copy_region(&m.tok, 0, &mut m.out, j * rr, rr)?;
        }
        let _ = t_chain;
        let chain_slot: Vec<u32> = keep.iter().map(|&i| pendings[i].0 as u32).collect();
        // record the chain layout for the RS verify resolve - the
        // async plan clears at fetch, but m.out/q-store keep this layout
        // until the next chain runs (stream-ordered after the resolve)
        if rs_active {
            self.spec_rs_chain = Some((chain_slot.clone(), rr, k_use));
        }
        // per-pendings kept flags: the service builds un-kept (chain-cold)
        // entries as length-1 chunks so async row counts - the graph keys -
        // match the synchronous path exactly
        let mut kept = vec![false; r];
        for &i in &keep {
            kept[i] = true;
        }
        self.spec_async = Some(super::SpecAsyncPlan {
            keep,
            chain_slot,
            r,
            rr,
            k_use,
            fetched: None,
        });
        Ok(Some((k_use, kept)))
    }

    /// Read the armed chain's drafts back into the plan cache (idempotent).
    /// Callers run this after the verify tick - its picks dtoh already
    /// drained the stream, so this costs the copy, not a boundary.
    pub(crate) fn spec_async_drafts(&mut self) -> Result<(), GpuError> {
        let Some(plan) = self.spec_async.as_mut() else {
            return Ok(());
        };
        if plan.fetched.is_some() {
            return Ok(());
        }
        let exec = self.exec.clone();
        let m = self.mtp.as_ref().expect("armed plan requires mtp");
        let view = m
            .out
            .try_slice(0..plan.k_use * plan.rr)
            .ok_or_else(|| GpuError::Driver("draft out slice".into()))?;
        let flat = exec
            .stream
            .clone_dtoh(&view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        let mut out_v: Vec<Vec<u32>> = vec![Vec::new(); plan.r];
        for (ci, &pi) in plan.keep.iter().enumerate() {
            out_v[pi] = (0..plan.k_use).map(|j| flat[j * plan.rr + ci]).collect();
        }
        plan.fetched = Some(out_v);
        Ok(())
    }

    /// Rung B2: one PIPELINED spec round - enqueue the whole
    /// chain -> assembly -> verify -> prep -> h-gather sequence with no
    /// host uploads (the previous round's pd_spec_prep wrote every device
    /// input) and record the round's strip event. Never syncs. The caller
    /// guarantees steady state: the pipe config was armed by a classic
    /// strip round at this exact shape (graphs captured, all slots kept,
    /// uniform k1) and `par` carries this round's sampler params.
    pub(crate) fn spec_pipe_round_impl(&mut self, par: &[u32]) -> Result<(), GpuError> {
        let exec = self.exec.clone();
        let cfg = self
            .spec_pipe
            .as_ref()
            .ok_or_else(|| GpuError::Driver("spec pipe not armed".into()))?;
        let (n, rr, k_use, gkey, stride, half) =
            (cfg.n, cfg.rr, cfg.k_use, cfg.gkey, cfg.stride, cfg.flip);
        let k1 = gkey.1;
        // sampler params for this round (stream-ordered behind the previous
        // round's kernels, consumed by this round's replay)
        {
            let (d_par, _) = self
                .samp
                .as_mut()
                .expect("enable_batch allocates sampler buffers");
            let mut v = d_par
                .try_slice_mut(0..par.len())
                .ok_or_else(|| GpuError::Driver("samp par slice".into()))?;
            exec.stream
                .memcpy_htod(par, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        // chain: k_use replays of the rr-keyed graph (tok/pos/attn written
        // by the previous round's prep; h by its gather)
        for j in 0..k_use {
            let g = self
                .mtp_graphs
                .get(&rr)
                .ok_or_else(|| GpuError::Driver("pipe: chain graph missing".into()))?;
            g.0.launch()
                .map_err(|e| GpuError::Driver(format!("pipe chain launch: {e}")))?;
            let m = self.mtp.as_mut().expect("pipe requires mtp");
            exec.copy_region(&m.tok, 0, &mut m.out, j * rr, rr)?;
        }
        // verify token assembly from the chain's output plane
        {
            let m = self.mtp.as_ref().expect("mtp");
            let d_tokens = self.d_tokens.as_mut().expect("pipe requires enable_batch");
            exec.spec_toks(&m.asm_meta, &m.out, d_tokens, n, k1, rr)?;
        }
        // the verify tick graph (positions/slots device-resident)
        let g = self
            .decode_graphs
            .get(&gkey)
            .ok_or_else(|| GpuError::Driver("pipe: verify graph missing".into()))?;
        g.0.launch()
            .map_err(|e| GpuError::Driver(format!("pipe verify launch: {e}")))?;
        // accept + next-round prep + h gather, then the strip event
        {
            let (_, d_out) = self.samp.as_ref().expect("sampler buffers");
            let sc_pos = &mut self.scratch.pf_pos;
            let m = self.mtp.as_mut().expect("mtp");
            let hold2 = spec_rope_mode() == 2;
            {
                let (out_ref, meta, strip, tok, pos, attn) = (
                    &m.out,
                    &mut m.asm_meta,
                    &mut m.strip,
                    &mut m.tok,
                    &mut m.pos,
                    &mut m.attn_pos,
                );
                exec.spec_prep(
                    d_out,
                    out_ref,
                    meta,
                    sc_pos,
                    strip,
                    half * 128 * stride,
                    tok,
                    pos,
                    attn,
                    n,
                    rr,
                    stride,
                    hold2,
                )?;
            }
            let n_main = m.n_embd_main;
            let (strip_r, meta_r, h) = (&m.strip, &m.asm_meta, &mut m.h);
            exec.spec_hgather(
                &self.scratch.pf_normed,
                strip_r,
                half * 128 * stride,
                meta_r,
                h,
                n,
                n_main,
                stride,
            )?;
        }
        let ev = exec.record_event()?;
        let cfg = self.spec_pipe.as_mut().expect("armed");
        cfg.events[half] = Some(ev);
        cfg.flip = half ^ 1;
        Ok(())
    }

    /// Arm the one-ahead pipeline on the last strip round's shape. False
    /// when the shape wasn't steady or the graphs aren't captured yet.
    pub(crate) fn spec_pipe_arm_impl(&mut self) -> bool {
        let Some(cfg) = self.spec_pipe_cfg else {
            return false;
        };
        if !self.exec.has_spec_pipe()
            || !self.mtp_graphs.contains_key(&cfg.rr)
            || !self.decode_graphs.contains_key(&cfg.gkey)
        {
            return false;
        }
        self.spec_pipe = Some(super::SpecPipe {
            n: cfg.n,
            rr: cfg.rr,
            k_use: cfg.k_use,
            gkey: cfg.gkey,
            slot_ids: self.spec_pipe_slots.0.clone(),
            stride: 4 + DR_MAX_K,
            flip: 0,
            events: [None, None],
        });
        // one-shot engagement witness (A/B protocol: an arm leg must PROVE
        // the pipe armed; grep "[spec-pipe] armed")
        static WITNESS: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WITNESS.get_or_init(|| {
            eprintln!(
                "[spec-pipe] armed: n={} rr={} k_use={} gkey={:?}",
                cfg.n, cfg.rr, cfg.k_use, cfg.gkey
            );
        });
        true
    }

    /// Read a pipelined round's strip (event-gated on the copy stream -
    /// syncs only up to that round, not the rounds queued behind it).
    pub(crate) fn spec_pipe_strip_impl(&mut self, half: usize) -> Result<Vec<u32>, GpuError> {
        let exec = self.exec.clone();
        let (n, stride) = {
            let cfg = self
                .spec_pipe
                .as_ref()
                .ok_or_else(|| GpuError::Driver("spec pipe not armed".into()))?;
            (cfg.n, cfg.stride)
        };
        let ev = self.spec_pipe.as_mut().expect("armed").events[half]
            .take()
            .ok_or_else(|| GpuError::Driver("pipe: no event for half".into()))?;
        let side = self
            .pipe_copy
            .get_or_insert_with(|| exec.fork_stream().expect("pipe copy stream fork"));
        side.wait_event(&ev)?;
        let m = self.mtp.as_ref().expect("mtp");
        let view = m
            .strip
            .try_slice(half * 128 * stride..half * 128 * stride + n * stride)
            .ok_or_else(|| GpuError::Driver("pipe strip slice".into()))?;
        side.stream
            .clone_dtoh(&view)
            .map_err(|e| GpuError::Driver(e.to_string()))
    }

    /// Drain the pipe: wait out everything queued and clear the config.
    /// Host mirrors (pos/pending/spec_rows) are the caller's to resync
    /// from the strips it consumed.
    pub(crate) fn spec_pipe_drain_impl(&mut self) -> Result<(), GpuError> {
        self.exec
            .stream
            .synchronize()
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        self.spec_pipe = None;
        Ok(())
    }

    /// Async spec round, phase 2: read the chain's drafts back and clear
    /// the armed plan. Called after the verify tick - the dtoh syncs a
    /// stream whose chain work finished long ago (verify ran after it), so
    /// this costs the copy, not a boundary stall. Indexed by the begin
    /// call's pendings order (skipped/cold entries stay empty).
    pub(crate) fn spec_draft_fetch_impl(&mut self) -> Result<Option<Vec<Vec<u32>>>, GpuError> {
        self.spec_async_drafts()?;
        let Some(plan) = self.spec_async.take() else {
            return Ok(None);
        };
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!(
                "[g4-chain] n={} rr={} k={} (fetched)",
                plan.r,
                plan.rr,
                plan.k_use
            );
        }
        Ok(plan.fetched)
    }

    /// Spec verify through the standard sampled decode tick: rows = per-slot
    /// [pending, drafts...] at consecutive positions. Positions are service-
    /// owned, so rejected rows need no rollback - their KV entries sit
    /// beyond the committed cursor and are overwritten by later ticks (ring
    /// and pool rows are position-keyed). `plans` is per chunk row, flat in
    /// request order (the service's layout).
    ///
    /// Rows stay RAGGED (no padding): padding every round to the max chunk
    /// pushes a c8-shaped ~30-row verify to 64 rows, which buys an extra
    /// whole weight pass through the mma_ks BN col-tile grid.
    /// The k1-deep attention arm needs uniform chunks, but the only rounds
    /// where it PAYS (≥16 chunks, KV past L2) have k_budget-clamped chunks
    /// that are naturally uniform - spec_k1 is set exactly then.
    pub(crate) fn forward_spec_rows_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::generator::RowSample],
    ) -> Result<Option<Vec<u32>>, GpuError> {
        if reqs.is_empty() || reqs.len() > spec_live_max().min(self.n_slots) {
            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!(
                    "[spec-rows] DECLINE reqs={} live_max={} slots={}",
                    reqs.len(),
                    spec_live_max(),
                    self.n_slots
                );
            }
            return Ok(None);
        }
        let n = reqs.len();
        let cap_rows = self
            .batch_logits
            .as_ref()
            .map(|b| b.len() / self.hp.n_vocab)
            .unwrap_or(0);
        let r0: usize = reqs.iter().map(|(_, _, c)| c.len()).sum();
        let dbg = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some();
        if r0 == 0 || r0 > cap_rows || r0 > super::forward::PF_ROWS {
            if dbg {
                tracing::info!(
                    "[spec-rows] DECLINE n={n} r0={r0} cap_rows={cap_rows} pf_rows={}",
                    super::forward::PF_ROWS
                );
            }
            return Ok(None);
        }
        // WIDE rounds (>64 raw rows) pad chunks up to the round max: inside
        // the M-col single-weight-pass class pad rows cost only attention/
        // head lanes, and uniformity turns on the k1-deep spec attention +
        // bounds the graph keys. NARROW rounds stay ragged (padding across
        // the 32/64-row GEMM boundaries was the c8 258->196 regressor).
        let raw: usize = reqs.iter().map(|(_, _, c)| c.len()).sum();
        let kmax = reqs.iter().map(|(_, _, c)| c.len()).max().unwrap_or(0);
        let pad_to = if raw > 64 && n * kmax <= cap_rows {
            kmax
        } else {
            0
        };
        let k1 = if pad_to > 0 { kmax } else { reqs[0].2.len() };
        let uniform = k1 > 1 && (pad_to > 0 || reqs.iter().all(|(_, _, c)| c.len() == k1));
        let rows_total = if pad_to > 0 { n * k1 } else { raw };
        let mut tokens = Vec::with_capacity(rows_total);
        let mut positions = Vec::with_capacity(rows_total);
        let mut slots = Vec::with_capacity(rows_total);
        let mut padded_plans = Vec::with_capacity(rows_total);
        let mut flat = 0usize;
        for (slot, start, chunk) in reqs {
            let clen = if pad_to > 0 { k1 } else { chunk.len() };
            if *slot >= self.n_slots || chunk.is_empty() || start + clen > self.max_ctx {
                if dbg {
                    tracing::info!(
                        "[spec-rows] DECLINE slot={slot} chunk={} start={start} clen={clen} max_ctx={}",
                        chunk.len(),
                        self.max_ctx
                    );
                }
                return Ok(None);
            }
            let last = *chunk.last().expect("non-empty chunk checked above");
            for i in 0..clen {
                tokens.push(chunk.get(i).copied().unwrap_or(last));
                positions.push((start + i) as u32);
                slots.push(*slot as u32);
                padded_plans.push(if i < chunk.len() {
                    plans[flat + i]
                } else {
                    crate::generator::RowSample::Device(crate::sampler::DevicePlan::Greedy)
                });
            }
            flat += chunk.len();
        }
        // Async round: an armed drafter plan means the service
        // built chunk VALUES as placeholders - assemble the real verify
        // tokens on device from the chain's output plane (pd_spec_toks) and
        // have batch_upload skip its host token copy. Positions/slots/plans
        // are value-independent and stay host-built. The pad rule matches
        // the host loop above exactly (repeat the last real token).
        if let Some(plan) = self.spec_async.as_ref() {
            let exec = self.exec.clone();
            if n > 128 {
                return Err(GpuError::Driver(
                    "async spec round: n > asm_meta cap".into(),
                ));
            }
            let mut meta = vec![0u32; 5 * n];
            let mut base = 0u32;
            let mut cmax = 0u32;
            for (i, (slot, _, chunk)) in reqs.iter().enumerate() {
                let clen = (if pad_to > 0 { k1 } else { chunk.len() }) as u32;
                // match by SLOT id - the verify reqs interleave cold slots
                // the chain never saw (n <= 64: linear scan)
                let ci = plan.chain_slot.iter().position(|&s| s == *slot as u32);
                let (srow, ndr) = match ci {
                    Some(ci) => (ci as u32, (chunk.len() - 1).min(plan.k_use) as u32),
                    // cold slot: no chain row - every row past pending pads
                    // with pending itself (ndr 0)
                    None => (0, 0),
                };
                meta[i] = chunk[0];
                meta[n + i] = srow;
                meta[2 * n + i] = ndr;
                meta[3 * n + i] = clen;
                meta[4 * n + i] = base;
                base += clen;
                cmax = cmax.max(clen);
            }
            debug_assert_eq!(base as usize, rows_total, "meta rows == verify rows");
            let rr = plan.rr;
            {
                let m = self.mtp.as_mut().expect("armed plan requires mtp");
                let mut v = m
                    .asm_meta
                    .try_slice_mut(0..5 * n)
                    .ok_or_else(|| GpuError::Driver("asm_meta slice".into()))?;
                exec.stream
                    .memcpy_htod(&meta, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            {
                let mtp = self.mtp.as_ref().expect("mtp");
                let d_tokens = self
                    .d_tokens
                    .as_mut()
                    .expect("spec verify requires enable_batch");
                exec.spec_toks(&mtp.asm_meta, &mtp.out, d_tokens, n, cmax as usize, rr)?;
            }
            self.toks_dev = true;
        }
        // canonical-RS rounds - collect resolve params for drafted
        // rows carrying RsVerify plans, keyed to the recorded chain layout
        // (srow via chain_slot, q-store step = draft index). Built before
        // the tick: rs rounds skip the tick's ids dtoh and re-read the
        // resolved plane after the resolve kernel runs.
        let mut rs_par: Vec<u32> = Vec::new();
        let rs_chain = if self.mtp.as_ref().is_some_and(|m| m.rs.is_some()) {
            self.spec_rs_chain.clone()
        } else {
            None
        };
        if let Some((cs, _crr, ck)) = rs_chain.as_ref() {
            let mut vbase = 0usize;
            for (slot, _start, chunk) in reqs {
                let clen = if pad_to > 0 { k1 } else { chunk.len() };
                if let Some(ci) = cs.iter().position(|&s| s == *slot as u32) {
                    let ndr = (chunk.len() - 1).min(*ck);
                    for i in 0..ndr.min(RS_PAR_ROWS - rs_par.len() / 8) {
                        if let crate::generator::RowSample::Device(
                            crate::sampler::DevicePlan::RsVerify { inv_t, u1, u2 },
                        ) = padded_plans[vbase + i]
                        {
                            rs_par.extend_from_slice(&[
                                (vbase + i) as u32,
                                i as u32,
                                ci as u32,
                                inv_t.to_bits(),
                                u1.to_bits(),
                                u2.to_bits(),
                                0,
                                0,
                            ]);
                        }
                    }
                }
                vbase += clen;
            }
        }
        let rs_round = !rs_par.is_empty();
        // k1-deep spec attention engages only for uniform wide rounds (the
        // batch.rs width gate re-checks chunks ≥ threshold per layer); the
        // graph key carries k1 so a 64-row 32×2 tick never replays as 8×8.
        let r = rows_total;
        self.spec_k1 = uniform.then_some(k1);
        // span band for the FIN route (door 3): position floor picks the
        // graph variant - long-KV ticks capture/replay the one-split
        // in-kernel-finalize walk, short ones keep splits+combine
        let pmax = positions.iter().max().copied().unwrap_or(0) as usize;
        self.spec_long = uniform && pmax >= super::batch::spec_fin_pos_floor();
        // shallow-walk band elects LCO (its own graph-key bit -
        // the elected kernel is frozen into the capture)
        let lco_floor = super::batch::spec_lco_pos_floor();
        self.spec_shallow = uniform && lco_floor > 0 && pmax < lco_floor;
        if self.spec_shallow {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::info!("[g4-lco] pos-election engaged (floor={lco_floor}, pmax={pmax})");
            });
        }
        let t_verify = std::time::Instant::now();
        // rung B1: strip rounds skip the sampled-ids dtoh - the
        // device accept below emits everything the round consumes.
        // PADDOCK_SPEC_STRIP_CHECK=1 keeps the ids readback and cross-checks
        // the device accept against the host walk per slot (divergence
        // triage - logs, never corrects).
        let strip_mode = self.want_strip && self.spec_async.is_some();
        let strip_check =
            strip_mode && paddock_models::dev_var_os!("PADDOCK_SPEC_STRIP_CHECK").is_some();
        if (strip_mode && !strip_check) || rs_round {
            self.ids_skip = true;
        }
        self.dflash_defer = self.dflash.is_some();
        let step = self.forward_batch_sampled_rows(&tokens, &positions, &slots, &padded_plans);
        self.ids_skip = false;
        self.dflash_defer = false;
        let verify_ms = t_verify.elapsed().as_secs_f64() * 1e3;
        let gkey_long = self.spec_long;
        let gkey_shallow = self.spec_shallow;
        self.spec_k1 = None;
        self.spec_long = false;
        self.spec_shallow = false;
        let step = step?;
        // run the RS resolve before any accept path reads the ids
        // plane - accepted rows get their draft token, rejects the residual
        // draw; the accept-while-match machinery downstream is unchanged.
        if rs_round {
            let (_, crr, _) = rs_chain.as_ref().expect("rs_round");
            let nrs = rs_par.len() / 8;
            let exec = self.exec.clone();
            {
                let m = self.mtp.as_mut().expect("rs chain requires mtp");
                let rs = m.rs.as_mut().expect("rs bufs");
                let mut v = rs
                    .par
                    .try_slice_mut(0..rs_par.len())
                    .ok_or_else(|| GpuError::Driver("rs par slice".into()))?;
                exec.stream
                    .memcpy_htod(&rs_par, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            {
                let logits = self.batch_logits.as_ref().expect("enable_batch");
                let m = self.mtp.as_ref().expect("mtp");
                let rs = m.rs.as_ref().expect("rs bufs");
                let (_, d_out) = self.samp.as_mut().expect("sampler buffers");
                exec.spec_rs_resolve(
                    logits,
                    &m.out,
                    &rs.qstore,
                    &rs.qsum,
                    &rs.par,
                    d_out,
                    nrs,
                    *crr,
                    self.hp.n_vocab,
                    DR_MAX_ROWS,
                )?;
            }
        }
        // RS rounds skipped the tick's ids dtoh; every host consumer below
        // reads the RESOLVED plane instead (strip-mode consumes on device)
        let rs_ids: Option<Vec<u32>> = if rs_round && (!strip_mode || strip_check) {
            let exec = self.exec.clone();
            let (_, d_out) = self.samp.as_ref().expect("sampler buffers");
            let view = d_out
                .try_slice(0..r)
                .ok_or_else(|| GpuError::Driver("samp out slice".into()))?;
            Some(
                exec.stream
                    .clone_dtoh(&view)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
            )
        } else {
            None
        };
        let ids: &[u32] = rs_ids.as_deref().unwrap_or(&step.ids);
        // DFlash: commit the accepted prefix of every request's rows into
        // the feature ring. `ids` is in PADDED row order when this round
        // padded, so compact it back to one pick per raw chunk row first -
        // the commit's accept walk indexes chunks, not rows.
        if self.dflash.is_some() && !ids.is_empty() {
            let picks: Vec<u32> = if pad_to > 0 {
                reqs.iter()
                    .enumerate()
                    .flat_map(|(b, (_, _, c))| (0..c.len()).map(move |i| (b * k1 + i, ())))
                    .map(|(row, ())| ids[row])
                    .collect()
            } else {
                ids.to_vec()
            };
            self.dflash_spec_commit(reqs, &picks)?;
        }
        if strip_mode {
            // device accept: the walk runs on-stream right after the tick;
            // one compact strip dtoh replaces picks + drafts + two CPU
            // replays. The h re-point comes straight from the strip.
            let exec = self.exec.clone();
            let stride = 4 + DR_MAX_K;
            let rr = self.spec_async.as_ref().expect("strip_mode").rr;
            let pipe_kernels = exec.has_spec_pipe();
            {
                let (_, d_out) = self.samp.as_ref().expect("sampler buffers");
                if pipe_kernels {
                    // prep is a superset of accept: it additionally writes
                    // round-N+1 device state (chain tok/rope/bound, meta
                    // pend, next verify positions). On non-steady rounds
                    // those writes land in buffers every CLASSIC round
                    // re-uploads anyway - harmless; on steady rounds they
                    // are exactly what pipe entry needs.
                    let sc_pos = &mut self.scratch.pf_pos;
                    let m = self.mtp.as_mut().expect("armed plan requires mtp");
                    let hold2 = spec_rope_mode() == 2;
                    let (out_ref, meta, strip, tok, pos_b, attn) = (
                        &m.out,
                        &mut m.asm_meta,
                        &mut m.strip,
                        &mut m.tok,
                        &mut m.pos,
                        &mut m.attn_pos,
                    );
                    exec.spec_prep(
                        d_out, out_ref, meta, sc_pos, strip, 0, tok, pos_b, attn, n, rr, stride,
                        hold2,
                    )?;
                    let n_main = m.n_embd_main;
                    let (strip_r, meta_r, h) = (&m.strip, &m.asm_meta, &mut m.h);
                    exec.spec_hgather(
                        &self.scratch.pf_normed,
                        strip_r,
                        0,
                        meta_r,
                        h,
                        n,
                        n_main,
                        stride,
                    )?;
                } else {
                    let sc_pos = &self.scratch.pf_pos;
                    let m = self.mtp.as_mut().expect("armed plan requires mtp");
                    let (out_ref, meta_ref, strip_mut) = (&m.out, &m.asm_meta, &mut m.strip);
                    exec.spec_accept(d_out, out_ref, meta_ref, sc_pos, strip_mut, n, rr, stride)?;
                }
            }
            let flat = {
                let m = self.mtp.as_ref().expect("mtp");
                let view = m
                    .strip
                    .try_slice(0..n * stride)
                    .ok_or_else(|| GpuError::Driver("strip slice".into()))?;
                exec.stream
                    .clone_dtoh(&view)
                    .map_err(|e| GpuError::Driver(e.to_string()))?
            };
            let mut accepted = 0usize;
            for (i, (slot, _start, _chunk)) in reqs.iter().enumerate() {
                let o = &flat[i * stride..(i + 1) * stride];
                accepted += o[0] as usize;
                self.spec_stats.bump_depth(o[0] as usize);
                self.spec_rows[*slot] = Some((o[1], o[2]));
            }
            self.spec_stats.bump(n, r, accepted);
            if strip_check {
                // host walk over the real drafts (fetched) vs the kernel's
                // strip - any disagreement is a strip bug, log it loudly
                self.spec_async_drafts()?;
                let plan = self.spec_async.as_ref().expect("strip_mode");
                let dr = plan.fetched.clone().unwrap_or_default();
                let cs = plan.chain_slot.clone();
                let kp = plan.keep.clone();
                let mut base = 0usize;
                for (i, (slot, start, chunk)) in reqs.iter().enumerate() {
                    let dref = cs
                        .iter()
                        .position(|&s| s == *slot as u32)
                        .and_then(|ci| dr.get(kp[ci]));
                    let mut a = 0usize;
                    while a + 1 < chunk.len() {
                        let dv = dref.and_then(|d| d.get(a).copied()).unwrap_or(chunk[0]);
                        if dv != ids[base + a] {
                            break;
                        }
                        a += 1;
                    }
                    let o = &flat[i * stride..(i + 1) * stride];
                    if o[0] as usize != a + 1
                        || o[1] != (*start + a) as u32
                        || o[2] != (base + a) as u32
                        || o[3] != ids[base + a]
                    {
                        tracing::warn!(
                            "[strip-div] slot={slot} host a+1={} pfin={} row={} pend={} vs strip {} {} {} {}",
                            a + 1,
                            *start + a,
                            base + a,
                            ids[base + a],
                            o[0],
                            o[1],
                            o[2],
                            o[3]
                        );
                    }
                    base += if pad_to > 0 { k1 } else { chunk.len() };
                }
            }
            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!(
                    "[g4-spec] n={n} rows={r} uniform_k1={:?} committed={accepted} ({:.2}/slot) verify_ms={verify_ms:.2} (strip)",
                    uniform.then_some(k1),
                    accepted as f64 / n as f64
                );
            }
            let _ = step;
            // rung B2: stash the pipe-candidate shape when the round was
            // steady (every slot kept, uniform k1)
            {
                let plan = self.spec_async.as_ref().expect("strip_mode");
                self.spec_pipe_cfg =
                    (pipe_kernels && uniform && plan.rr == n).then(|| super::SpecPipeCfg {
                        n,
                        rr: plan.rr,
                        k_use: plan.k_use,
                        gkey: (
                            r,
                            k1,
                            gkey_long,
                            gkey_shallow,
                            super::batch::kv_split_band(self.attn_pos_max),
                        ),
                    });
                if self.spec_pipe_cfg.is_some() {
                    self.spec_pipe_slots = super::SpecPipeSlots(plan.chain_slot.clone());
                }
            }
            self.spec_async = None;
            self.spec_strip = Some((flat, n, stride));
            return Ok(Some(Vec::new()));
        }
        // Fix the h map to the ACCEPTED-final row per slot (the choke-point
        // recording kept each slot's last chunk row): replay the service's
        // accept-while-match rule - same rule, same picks - so the next
        // round's chain starts from the right hidden.
        //
        // ASYNC ROUNDS: the reqs chunks carry PLACEHOLDER values
        // - replaying against them re-points h at the wrong row, the next
        // chain drafts from the wrong hidden, and acceptance collapses
        // (found live: async c8 -14%, verify attention degraded to per-row
        // v9q as rounds fragmented ragged). Fetch the real drafts here -
        // the picks dtoh above already drained the stream, so this costs
        // the copy, not a boundary - and replay against them exactly as
        // the service will (real drafts; placeholder pending stays for
        // rows past a cold chain's supply).
        let (async_drafts, plan_cs, plan_keep): (Option<Vec<Vec<u32>>>, Vec<u32>, Vec<usize>) =
            if self.spec_async.is_some() {
                self.spec_async_drafts()?;
                let p = self.spec_async.as_ref().expect("armed");
                (p.fetched.clone(), p.chain_slot.clone(), p.keep.clone())
            } else {
                (None, Vec::new(), Vec::new())
            };
        let mut accepted = 0usize;
        let mut base = 0usize;
        let mut picks = Vec::with_capacity(flat);
        for (slot, start, chunk) in reqs {
            // effective chunk value at j: the real draft where the chain
            // supplied one (matching the service's fill rule exactly);
            // placeholder pending stays for a cold chain's rows
            let dref: Option<&Vec<u32>> = async_drafts.as_ref().and_then(|dr| {
                plan_cs
                    .iter()
                    .position(|&s| s == *slot as u32)
                    .and_then(|ci| dr.get(plan_keep[ci]))
            });
            let eff = |j: usize| -> u32 {
                if j == 0 {
                    return chunk[0];
                }
                if let Some(d) = dref
                    && j - 1 < d.len()
                {
                    return d[j - 1];
                }
                chunk[j]
            };
            let mut a = 0usize;
            while a + 1 < chunk.len() && eff(a + 1) == ids[base + a] {
                a += 1;
            }
            accepted += a + 1;
            self.spec_stats.bump_depth(a + 1);
            self.spec_rows[*slot] = Some(((start + a) as u32, (base + a) as u32));
            picks.extend_from_slice(&ids[base..base + chunk.len()]);
            base += if pad_to > 0 { k1 } else { chunk.len() };
        }
        self.spec_stats.bump(n, r, accepted);
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!(
                "[g4-spec] n={n} rows={r} uniform_k1={:?} committed={accepted} ({:.2}/slot) verify_ms={verify_ms:.2}",
                uniform.then_some(k1),
                accepted as f64 / n as f64
            );
        }
        Ok(Some(picks))
    }

    /// h-map freshness check for the service's pre-round warm sweep. There is
    /// no draft-head KV to re-warm (the drafter shares the main cache); h
    /// self-heals on the next sampled tick, so a stale slot just declines.
    /// `want_pos` is the KV-space position of the slot's last committed row
    /// (== committed.len()-1 for text; offset by the image rows on mm slots
    /// - spec_rows records KV positions, so the comparison is space-correct
    ///   either way).
    pub(crate) fn spec_ensure_warm_impl(
        &self,
        slot: usize,
        committed: &[u32],
        want_pos: u32,
    ) -> bool {
        if self.mtp.is_none() || committed.is_empty() {
            return false;
        }
        let want = want_pos;
        let ok = matches!(self.spec_rows.get(slot).copied().flatten(), Some((p, _)) if p == want);
        if !ok && paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!(
                "[warm] slot={} want={} got={:?}",
                slot,
                want,
                self.spec_rows.get(slot).copied().flatten().map(|(p, _)| p)
            );
        }
        ok
    }
}
