//! Nemotron serial forward - batch-1 decode + token-by-token prefill (the
//! Generator bring-up trio, laguna's forward.rs is the template). One
//! residual per layer: `x = x + mixer(rms_norm(x))`; the mixer is mamba-2,
//! NoPE attention, or MoE per `layers_block_type`.
//!
//! Decode ticks ride a captured CUDA graph (- one submit replaces the
//! ~350-launch train; `PADDOCK_NO_NEMO_GRAPH=1` pins the eager path, and probe
//! runs are always eager since they dtoh mid-walk). On top of the graph sits
//! the serial depth-2 decode pipe (gpt-oss's shape at rows=1): device
//! sampling + on-device advance, so tick N+1 is already running while the host
//! reads tick N's id.

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

use crate::generator::{GenError, Generator, RowSample};
use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::sampler::DevicePlan;

use super::*;
use crate::gpu_model::qwen35::{gemv_any, prefill_mm_pre_any, prefill_quant};
use paddock_models::nemotron::NemotronBlock;

fn gen_err(e: GpuModelError) -> GenError {
    match e {
        GpuModelError::PoolExhausted => GenError::PoolExhausted,
        other => GenError::Backend(other.to_string()),
    }
}

/// Per-sequence decode state: per-layer KV on the 6 attention layers, f32
/// SSM state + conv window on the 23 mamba layers (the checkpoint's own
/// `mamba_ssm_cache_dtype: float32`).
///
/// f32 stays the elected class on this serial lane, and the reason is
/// measured rather than assumed: the decode scan runs 59.0 us/launch, which
/// is not what limits the tick, so an f16 arena here buys little. The f16
/// kernels exist (ABI 443-445, gated) and are deliberately unwired; the
/// batched lane elects f16 in `ssm_arena`.
pub(crate) struct DecodeState {
    pub kv_k: Vec<Option<CudaSlice<u8>>>,
    pub kv_v: Vec<Option<CudaSlice<u8>>>,
    /// [n_heads, head_dim, d_state] f32 per mamba layer
    pub ssm: Vec<Option<CudaSlice<f32>>>,
    /// [k-1, conv_dim] f32 pre-conv rows per mamba layer
    pub conv_win: Vec<Option<CudaSlice<f32>>>,
    pub pos: usize,
    pub d_token: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    /// constant [0] - slot 0
    pub d_slots: CudaSlice<u32>,
    /// captured decode tick. Survives `reset` - the buffers keep
    /// their allocations, so every baked address stays valid; dropped with
    /// the whole DecodeState on `set_max_ctx`.
    pub graph: Option<SendGraph>,
    /// pipe sampler-param ring, 2 slots x 4 u32 (`sample_rows` packing)
    pub d_pipe_par: CudaSlice<u32>,
    /// pipe sampled-id ring, 2 slots x 1 row
    pub d_pipe_out: CudaSlice<u32>,
}

/// Depth-2 serial pipe bookkeeping: which tick is being enqueued, plus the
/// readability event per ring slot (gpt-oss's PipeState at rows=1).
pub(crate) struct PipeState {
    pub tick: usize,
    pub ev: [Option<cudarc::driver::CudaEvent>; 2],
}

pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: the model lives on the single engine thread; the graph handle is
// never touched from two threads at once (same argument as every family).
unsafe impl Send for SendGraph {}

/// Bulk-prefill chunk width on dies of 128 SMs and up: the scan kernel walks
/// the chunk sequentially in one launch (state register-resident), so a wider
/// chunk amortizes the per-chunk launch train; scratch cost is ~60 MB at 512.
/// This is the width every big-die number was measured at.
const PREFILL_CHUNK_BIG: usize = 512;
/// Small dies (under 128 SMs) size the row scratch for 8192 rows - the
/// service's own tick budget - and take ticks by the WHOLE-PROMPT rule in
/// `plan_chunk` (`PREFILL_TICK_BASE` rows, or the first queued prompt's
/// remaining rows if that is more). Every mixed tick re-streams the touched
/// experts' weights (~0.7 GB per MoE layer once the tick is a few hundred
/// rows wide, 16 GB per pass over the 23 MoE layers), and on GB10's ~240
/// GB/s that pass is ~70 ms - so a 1k prompt chunked at 512 paid two passes
/// (280 ms TTFT vs vLLM's 202 in one pass), an 8x1k cohort sixteen, and a
/// 4k / 8k prompt at the 2048 cap two / four where vLLM's 8192-token step
/// pays one. Scratch is ~embd*4 B/row over a handful of planes (~2.4 GB at
/// 8192; the reserve is charged before the KV pool is sized).
const PREFILL_CHUNK_SMALL: usize = 8192;
/// The per-tick take on small dies when no single queued prompt needs more:
/// a cohort of short prompts ramps through 2048-row ticks (measured on the
/// 8x1k cohort: 2048 gives the best median TTFT, 4096 / 8192 make every
/// request wait one fat tick - 1252 vs 1921 / 2205 ms), while a single long
/// prompt takes one tick of its own size up to the scratch cap.
pub(crate) const PREFILL_TICK_BASE: usize = 2048;

/// Row-scratch cap (and the whole-prompt chunk loop's width), read once at
/// load: explicit `PADDOCK_PREFILL_CHUNK` (the gpt-oss lane's switch,
/// 256..=8192) wins, else the die class picks. Stored on the model
/// (`prefill_chunk`) so every consumer - scratch sizing, the admission
/// floor, chunk loops, the tick's clamp - reads the same value.
pub(crate) fn prefill_chunk_rows(sm_count: usize) -> usize {
    paddock_models::dev_var!("PADDOCK_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| (256..=8192).contains(&n))
        .unwrap_or(if sm_count < 128 {
            PREFILL_CHUNK_SMALL
        } else {
            PREFILL_CHUNK_BIG
        })
}

/// Chunk-wide scratch for the bulk serial prefill (rung): every
/// buffer is the serial `Scratch` twin at [prefill_chunk, dim]. The mamba
/// in/out projections ride the W8A8 f8row GEMM here (dynamic per-token
/// activation e4m3 - the checkpoint's own W8A8 class; decode stays W8A16
/// GEMV), hence the activation-quant image buffers.
pub(crate) struct PrefillScratch {
    pub cap: usize,
    pub d_tok: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    /// zeroed - every prefill row is slot 0 (serial lane)
    pub d_slots: CudaSlice<u32>,
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    pub d_zxbcdt: CudaSlice<f32>,
    pub d_conv: CudaSlice<f32>,
    pub d_y: CudaSlice<f32>,
    pub d_yn: CudaSlice<f32>,
    /// e4m3 activation image for the f8row GEMM, [cap, max(hidden, d_inner)]
    pub d_xq: CudaSlice<i8>,
    pub d_xrs: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_sinks: CudaSlice<f32>,
    pub d_logits_r: CudaSlice<f32>,
    pub d_idx: CudaSlice<u32>,
    pub d_w: CudaSlice<f32>,
    pub d_up: CudaSlice<f32>,
    /// zeroed - the shared expert is plane index 0 for every row
    pub d_sh_idx: CudaSlice<u32>,
    /// all-ones combine weights for the shared expert
    pub d_sh_w: CudaSlice<f32>,
    pub d_sh_up: CudaSlice<f32>,
    // sorted-tile MoE MMA lane (W4A4 prefill class). nb_r/nb_s are
    // the moe_align block capacities the buffers were sized for (routed:
    // pairs/32 + n_expert per-expert tail bound; shared: cap/32 + 1).
    pub nb_r: usize,
    pub nb_s: usize,
    pub d_xq4: CudaSlice<i8>,
    pub d_xs4: CudaSlice<u8>,
    pub d_srow: CudaSlice<u32>,
    pub d_sslot: CudaSlice<u32>,
    pub d_bexp: CudaSlice<u32>,
    pub d_srow_s: CudaSlice<u32>,
    pub d_sslot_s: CudaSlice<u32>,
    pub d_bexp_s: CudaSlice<u32>,
    pub d_fq: CudaSlice<u8>,
    pub d_fs: CudaSlice<u8>,
    pub d_fq_s: CudaSlice<u8>,
    pub d_fs_s: CudaSlice<u8>,
    pub d_part: CudaSlice<f32>,
    /// last-row staging for the final norm + lm_head tail
    pub d_last: CudaSlice<f32>,
    pub d_lastn: CudaSlice<f32>,
    pub d_logits: CudaSlice<f32>,
    /// GGUF-lane extras (None on the NVFP4 lane)
    pub q8: Option<PrefillQ8>,
}

/// Q8_0-lane prefill scratch: int8 activation images for the mmq
/// GEMM ladder (both layouts - strided <=64 rows, flat mmq above), the
/// kquant sums/fixup planes `prefill_mm_pre_any` requires (sized so a UD
/// k-quant DENSE plane works; the expert planes refuse non-Q8_0 at load),
/// and the sorted-MoE f32 fused planes + their quantized twins the
/// relu2-sorted -> down_sorted chain reads.
pub(crate) struct PrefillQ8 {
    pub xq: CudaSlice<i8>,
    pub xs: CudaSlice<f32>,
    pub yq: CudaSlice<u8>,
    pub skfix: CudaSlice<f32>,
    pub xsums: CudaSlice<f32>,
    pub ssums: CudaSlice<f32>,
    pub fu_r: CudaSlice<f32>,
    pub fq_r: CudaSlice<i8>,
    pub fs_r: CudaSlice<f32>,
    pub fu_s: CudaSlice<f32>,
    pub fq_s: CudaSlice<i8>,
    pub fs_s: CudaSlice<f32>,
}

pub(crate) struct Scratch {
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    // mamba lane
    pub d_zxbcdt: CudaSlice<f32>,
    pub d_conv: CudaSlice<f32>,
    pub d_y: CudaSlice<f32>,
    pub d_yn: CudaSlice<f32>,
    // attention lane
    pub d_q: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_sinks: CudaSlice<f32>,
    // moe lane
    pub d_logits_r: CudaSlice<f32>,
    pub d_idx: CudaSlice<u32>,
    pub d_w: CudaSlice<f32>,
    pub d_up: CudaSlice<f32>,
    pub d_sh_idx: CudaSlice<u32>,
    pub d_sh_w: CudaSlice<f32>,
    pub d_sh_up: CudaSlice<f32>,
    // decode rung: fused-MoE activations [k*moe_ff | shared_ff] and the
    // (n_active+1)-slot pre-weighted partial planes the slot fold consumes
    pub d_act: CudaSlice<f32>,
    pub d_part7: CudaSlice<f32>,
    pub d_logits: CudaSlice<f32>,
    /// GGUF-lane extras (None on the NVFP4 lane)
    pub q8: Option<ScratchQ8>,
}

/// Q8_0-lane decode scratch: the int8 image of xn for the token-batched
/// relu2 kernel, quantized fused planes for the two down passes, and the
/// shared expert's output row (the token-batched down writes, never
/// accumulates - routed and shared fold with one add).
pub(crate) struct ScratchQ8 {
    pub xq: CudaSlice<i8>,
    pub xs: CudaSlice<f32>,
    pub fq_r: CudaSlice<i8>,
    pub fs_r: CudaSlice<f32>,
    pub fq_s: CudaSlice<i8>,
    pub fs_s: CudaSlice<f32>,
    pub shproj: CudaSlice<f32>,
}

impl GpuNemotron {
    fn ensure_decode(&mut self) -> Result<(), GpuModelError> {
        if self.decode.is_some() && self.scratch.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let hp = &self.hp;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * hp.conv_dim();

        let n = hp.n_layer;
        let mut kv_k = Vec::with_capacity(n);
        let mut kv_v = Vec::with_capacity(n);
        let mut ssm = Vec::with_capacity(n);
        let mut conv_win = Vec::with_capacity(n);
        for li in 0..n {
            match hp.blocks[li] {
                NemotronBlock::Attention => {
                    kv_k.push(Some(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?));
                    kv_v.push(Some(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?));
                    ssm.push(None);
                    conv_win.push(None);
                }
                NemotronBlock::Mamba => {
                    kv_k.push(None);
                    kv_v.push(None);
                    // alloc() zeroes; reset() re-zeroes - a fresh sequence
                    // must start from S = 0 and an all-zero conv window
                    ssm.push(Some(e.alloc(state_elems)?));
                    conv_win.push(Some(e.alloc(win_elems)?));
                }
                NemotronBlock::Moe => {
                    kv_k.push(None);
                    kv_v.push(None);
                    ssm.push(None);
                    conv_win.push(None);
                }
            }
        }
        self.decode = Some(DecodeState {
            kv_k,
            kv_v,
            ssm,
            conv_win,
            pos: 0,
            d_token: e.alloc_u32(1)?,
            d_pos: e.alloc_u32(1)?,
            d_slots: e.alloc_u32(1)?, // zeroed -> slot 0
            graph: None,
            d_pipe_par: e.alloc_u32(2 * 4)?,
            d_pipe_out: e.alloc_u32(2)?,
        });

        let q_dim = hp.n_heads * hp.head_dim;
        // shared-expert constants: idx = [0] (zeroed alloc), weight = [1.0]
        let d_sh_idx = e.alloc_u32(1)?;
        let d_sh_w = e.to_device(&[1.0f32])?;
        self.scratch = Some(Scratch {
            d_x: e.alloc(hp.hidden)?,
            d_xn: e.alloc(hp.hidden)?,
            d_proj: e.alloc(hp.hidden)?,
            d_zxbcdt: e.alloc(hp.in_proj_rows())?,
            d_conv: e.alloc(hp.conv_dim())?,
            d_y: e.alloc(hp.d_inner())?,
            d_yn: e.alloc(hp.d_inner())?,
            d_q: e.alloc(q_dim)?,
            d_k: e.alloc(kv_dim)?,
            d_v: e.alloc(kv_dim)?,
            d_attn: e.alloc(q_dim)?,
            d_sinks: e.alloc_no_sinks(hp.n_heads)?,
            d_logits_r: e.alloc(hp.n_expert)?,
            d_idx: e.alloc_u32(hp.n_active)?,
            d_w: e.alloc(hp.n_active)?,
            d_up: e.alloc(hp.n_active * hp.moe_ff)?,
            d_sh_idx,
            d_sh_w,
            d_sh_up: e.alloc(hp.shared_ff)?,
            d_act: e.alloc(hp.n_active * hp.moe_ff + hp.shared_ff)?,
            d_part7: e.alloc((hp.n_active + 1) * hp.hidden)?,
            d_logits: e.alloc(hp.vocab)?,
            q8: if self.is_gguf() {
                Some(ScratchQ8 {
                    xq: e.alloc_i8(hp.hidden)?,
                    xs: e.alloc(hp.hidden / 32)?,
                    fq_r: e.alloc_i8(hp.n_active * hp.moe_ff)?,
                    fs_r: e.alloc(hp.n_active * hp.moe_ff / 32)?,
                    fq_s: e.alloc_i8(hp.shared_ff)?,
                    fs_s: e.alloc(hp.shared_ff / 32)?,
                    shproj: e.alloc(hp.hidden)?,
                })
            } else {
                None
            },
        });
        Ok(())
    }

    fn ensure_prefill(&mut self) -> Result<(), GpuModelError> {
        if self.prefill.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let hp = &self.hp;
        let c = self.prefill_chunk;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let q_dim = hp.n_heads * hp.head_dim;
        let qmax = hp.hidden.max(hp.d_inner());
        let d_sh_w = e.to_device(&vec![1.0f32; c])?;
        //  capacities: routed blocks are bounded by pairs/32 plus one
        // padded tail block per expert; the shared identity layout by
        // ceil(cap/32) (+1 covers a non-multiple tail chunk).
        let nb_r = c * hp.n_active / 32 + hp.n_expert;
        let nb_s = c / 32 + 1;
        self.prefill = Some(PrefillScratch {
            cap: c,
            d_tok: e.alloc_u32(c)?,
            d_pos: e.alloc_u32(c)?,
            d_slots: e.alloc_u32(c)?, // zeroed -> slot 0
            d_x: e.alloc(c * hp.hidden)?,
            d_xn: e.alloc(c * hp.hidden)?,
            d_proj: e.alloc(c * hp.hidden)?,
            d_zxbcdt: e.alloc(c * hp.in_proj_rows())?,
            d_conv: e.alloc(c * hp.conv_dim())?,
            d_y: e.alloc(c * hp.d_inner())?,
            d_yn: e.alloc(c * hp.d_inner())?,
            d_xq: e.alloc_i8(c * qmax)?,
            d_xrs: e.alloc(c)?,
            d_q: e.alloc(c * q_dim)?,
            d_k: e.alloc(c * kv_dim)?,
            d_v: e.alloc(c * kv_dim)?,
            d_attn: e.alloc(c * q_dim)?,
            d_sinks: e.alloc_no_sinks(hp.n_heads)?,
            d_logits_r: e.alloc(c * hp.n_expert)?,
            d_idx: e.alloc_u32(c * hp.n_active)?,
            d_w: e.alloc(c * hp.n_active)?,
            d_up: e.alloc(c * hp.n_active * hp.moe_ff)?,
            d_sh_idx: e.alloc_u32(c)?, // zeroed -> plane index 0
            d_sh_w,
            d_sh_up: e.alloc(c * hp.shared_ff)?,
            nb_r,
            nb_s,
            d_xq4: e.alloc_i8(c * hp.hidden / 2)?,
            d_xs4: e.alloc_u8(c * hp.hidden / 16)?,
            d_srow: e.alloc_u32(nb_r * 32)?,
            d_sslot: e.alloc_u32(nb_r * 32)?,
            d_bexp: e.alloc_u32(nb_r)?,
            d_srow_s: e.alloc_u32(nb_s * 32)?,
            d_sslot_s: e.alloc_u32(nb_s * 32)?,
            d_bexp_s: e.alloc_u32(nb_s)?,
            d_fq: e.alloc_u8(nb_r * 32 * hp.moe_ff / 2)?,
            d_fs: e.alloc_u8(nb_r * 32 * hp.moe_ff / 16)?,
            d_fq_s: e.alloc_u8(nb_s * 32 * hp.shared_ff / 2)?,
            d_fs_s: e.alloc_u8(nb_s * 32 * hp.shared_ff / 16)?,
            d_part: e.alloc(c * (hp.n_active + 1) * hp.hidden)?,
            d_last: e.alloc(hp.hidden)?,
            d_lastn: e.alloc(hp.hidden)?,
            d_logits: e.alloc(hp.vocab)?,
            q8: if self.is_gguf() {
                // qmax covers every GEMM input width (hidden, d_inner, q_dim);
                // yq/xsums use granite's mmq-layout sizing, skfix the fixed
                // splitk fixup plane
                Some(PrefillQ8 {
                    xq: e.alloc_i8(c * qmax)?,
                    xs: e.alloc(c * qmax / 32)?,
                    yq: e.alloc_u8(qmax.div_ceil(128) * c.next_multiple_of(128) * 144)?,
                    skfix: e.alloc(256 * 128 * 128 + 256)?,
                    xsums: e.alloc(qmax.div_ceil(128) * c.next_multiple_of(128) * 4)?,
                    ssums: e.alloc(c * qmax / 16)?,
                    fu_r: e.alloc(nb_r * 32 * hp.moe_ff)?,
                    fq_r: e.alloc_i8(nb_r * 32 * hp.moe_ff)?,
                    fs_r: e.alloc(nb_r * 32 * hp.moe_ff / 32)?,
                    fu_s: e.alloc(nb_s * 32 * hp.shared_ff)?,
                    fq_s: e.alloc_i8(nb_s * 32 * hp.shared_ff)?,
                    fs_s: e.alloc(nb_s * 32 * hp.shared_ff / 32)?,
                })
            } else {
                None
            },
        });
        Ok(())
    }

    /// Whole-prompt bulk prefill for the serial lane: chunk the tokens and
    /// run each span through batched kernels. State advance is exactly T
    /// serial steps' (conv window/scan state carried in place, KV appended
    /// at per-row positions); only the last chunk pays the lm_head.
    pub(crate) fn prefill_stream_impl(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        self.ensure_decode()?;
        self.ensure_prefill()?;
        let n = tokens.len();
        let mut done = 0usize;
        let mut logits = None;
        for chunk in tokens.chunks(self.prefill_chunk) {
            done += chunk.len();
            logits = self.prefill_chunk(chunk, done == n)?;
        }
        Ok(logits.expect("non-empty prompt"))
    }

    /// One chunk (T <= prefill_chunk tokens) through the whole stack.
    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, GpuModelError> {
        let t = tokens.len();
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let (embd, eps) = (hp.hidden, hp.eps);
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let q_dim = hp.n_heads * hp.head_dim;
        let d_inner = hp.d_inner();
        let conv_dim = hp.conv_dim();
        let in_rows = hp.in_proj_rows();
        let scale = 1.0 / (hp.head_dim as f32).sqrt();
        let kv_dtype = self.kv_dtype;
        //  dispatch: sorted-tile MMA MoE when the pack carries it
        // (cc12-gated slots 407-408). PADDOCK_NO_NVF4_MOE_BS=1 is the A/B
        // kill switch (measurement precedent, not a serving regime knob).
        let moe_bs = self.exec.has_nvf4_moe_bs()
            && paddock_models::dev_var_os!("PADDOCK_NO_NVF4_MOE_BS").is_none();
        let px = self.prefill.as_mut().expect("prefill scratch");
        let ds = self.decode.as_mut().expect("decode");
        debug_assert!(t >= 1 && t <= px.cap);

        if ds.pos + t > self.max_ctx {
            return Err(GpuModelError::ContextExceeded {
                got: ds.pos + t,
                max: self.max_ctx,
            });
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let positions: Vec<u32> = (ds.pos as u32..(ds.pos + t) as u32).collect();
        {
            let mut tv = px.d_tok.try_slice_mut(0..t).expect("tok view");
            exec.stream.memcpy_htod(tokens, &mut tv).map_err(drv)?;
            let mut pv = px.d_pos.try_slice_mut(0..t).expect("pos view");
            exec.stream.memcpy_htod(&positions, &mut pv).map_err(drv)?;
        }

        match &self.tok_embd {
            TokEmbd::F32(tab) => exec.embed_gather_batch(tab, &px.d_tok, &mut px.d_x, embd, t)?,
            // bit-identical rows: bf16 widens exactly, x1.0 scale is exact
            TokEmbd::Bf16(tab) => {
                exec.embed_gather_bf16(tab, &px.d_tok, &mut px.d_x, embd, t, 1.0)?
            }
            TokEmbd::Q8(tab) => exec.embed_gather_batch_q8(tab, &px.d_tok, &mut px.d_x, embd, t)?,
        }

        // Glue rung: the checkpoint's pattern is strictly
        // (mamba|attention) -> moe, so every MoE layer's prologue is the same
        // three latency-bound launches - add(x += proj), rmsnorm, quantize_nvf4
        // - 23 times per decode tick. One fused row-per-CTA kernel does all
        // three, so the previous layer's trailing add is hoisted into it and
        // this layer skips the two it already ran. Bit-exact, so the only
        // thing that moves is the launch count.
        let mut fused_pro = false;
        for (li, layer) in self.layers.iter().enumerate() {
            let pro_done = std::mem::take(&mut fused_pro);
            if !pro_done {
                exec.rmsnorm_batch(&px.d_x, &layer.norm.buf, &mut px.d_xn, embd, eps, t)?;
            }
            match &layer.mixer {
                Mixer::Mamba(w) => {
                    match &w.in_proj {
                        // W8A8 f8row GEMM (dynamic per-token e4m3 activations
                        // - the NVFP4 checkpoint's own W8A8 class)
                        LinW::F8(p) => {
                            exec.quantize_e4m3_row(&px.d_xn, &mut px.d_xq, &mut px.d_xrs, embd, t)?;
                            exec.f8row_gemm(
                                p,
                                &px.d_xq,
                                &px.d_xrs,
                                &mut px.d_zxbcdt,
                                embd,
                                in_rows,
                                t,
                            )?;
                        }
                        // GGUF lane: int8 mmq ladder - llama.cpp's own W8A8
                        // prefill class for Q8_0
                        LinW::Qw(q) => {
                            let s8 = px.q8.as_mut().expect("q8 prefill scratch");
                            prefill_quant(
                                &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &px.d_xn, embd, t,
                            )?;
                            prefill_mm_pre_any(
                                &exec,
                                q,
                                &s8.xq,
                                &s8.xs,
                                &s8.yq,
                                &mut s8.xsums,
                                &mut s8.ssums,
                                &mut s8.skfix,
                                &mut px.d_zxbcdt,
                                t,
                            )?;
                        }
                    }
                    exec.mamba_conv_seq(
                        ds.conv_win[li].as_mut().expect("win"),
                        &px.d_zxbcdt,
                        d_inner,
                        in_rows,
                        &w.conv_w,
                        &w.conv_b,
                        &mut px.d_conv,
                        conv_dim,
                        hp.d_conv,
                        t,
                    )?;
                    exec.mamba2_scan_seq(
                        ds.ssm[li].as_mut().expect("ssm"),
                        &px.d_conv,
                        &px.d_zxbcdt,
                        d_inner + conv_dim,
                        in_rows,
                        &w.a,
                        &w.d,
                        &w.dt_bias,
                        &mut px.d_y,
                        t,
                        hp.mamba_heads,
                        hp.mamba_head_dim,
                        hp.d_state,
                        hp.n_groups,
                    )?;
                    exec.mamba_rmsnorm_gated_g(
                        &px.d_y,
                        &px.d_zxbcdt,
                        0,
                        in_rows,
                        &w.norm_w,
                        &mut px.d_yn,
                        t,
                        d_inner,
                        hp.n_groups,
                        eps,
                    )?;
                    match &w.out_proj {
                        LinW::F8(p) => {
                            exec.quantize_e4m3_row(
                                &px.d_yn,
                                &mut px.d_xq,
                                &mut px.d_xrs,
                                d_inner,
                                t,
                            )?;
                            exec.f8row_gemm(
                                p,
                                &px.d_xq,
                                &px.d_xrs,
                                &mut px.d_proj,
                                d_inner,
                                embd,
                                t,
                            )?;
                        }
                        LinW::Qw(q) => {
                            let s8 = px.q8.as_mut().expect("q8 prefill scratch");
                            prefill_quant(
                                &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &px.d_yn, d_inner, t,
                            )?;
                            prefill_mm_pre_any(
                                &exec,
                                q,
                                &s8.xq,
                                &s8.xs,
                                &s8.yq,
                                &mut s8.xsums,
                                &mut s8.ssums,
                                &mut s8.skfix,
                                &mut px.d_proj,
                                t,
                            )?;
                        }
                    }
                }
                Mixer::Attn(w) => {
                    match w {
                        AttnWeights::F32 { wq, wk, wv, .. } => {
                            exec.gemm_f32(&wq.buf, embd, q_dim, &px.d_xn, &mut px.d_q, t)?;
                            exec.gemm_f32(&wk.buf, embd, kv_dim, &px.d_xn, &mut px.d_k, t)?;
                            exec.gemm_f32(&wv.buf, embd, kv_dim, &px.d_xn, &mut px.d_v, t)?;
                        }
                        AttnWeights::Qw { wq, wk, wv, .. } => {
                            let s8 = px.q8.as_mut().expect("q8 prefill scratch");
                            prefill_quant(
                                &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &px.d_xn, embd, t,
                            )?;
                            prefill_mm_pre_any(
                                &exec,
                                wq,
                                &s8.xq,
                                &s8.xs,
                                &s8.yq,
                                &mut s8.xsums,
                                &mut s8.ssums,
                                &mut s8.skfix,
                                &mut px.d_q,
                                t,
                            )?;
                            prefill_mm_pre_any(
                                &exec,
                                wk,
                                &s8.xq,
                                &s8.xs,
                                &s8.yq,
                                &mut s8.xsums,
                                &mut s8.ssums,
                                &mut s8.skfix,
                                &mut px.d_k,
                                t,
                            )?;
                            prefill_mm_pre_any(
                                &exec,
                                wv,
                                &s8.xq,
                                &s8.xs,
                                &s8.yq,
                                &mut s8.xsums,
                                &mut s8.ssums,
                                &mut s8.skfix,
                                &mut px.d_v,
                                t,
                            )?;
                        }
                    }
                    exec.kv_append_batch(
                        &px.d_k,
                        ds.kv_k[li].as_mut().expect("kv_k"),
                        &px.d_pos,
                        Some(&px.d_slots),
                        kv_dim,
                        self.max_ctx,
                        t,
                        kv_dtype,
                    )?;
                    exec.kv_append_batch(
                        &px.d_v,
                        ds.kv_v[li].as_mut().expect("kv_v"),
                        &px.d_pos,
                        Some(&px.d_slots),
                        kv_dim,
                        self.max_ctx,
                        t,
                        kv_dtype,
                    )?;
                    // the tiled scalar prefill (pd_attn_prefill) is the hd128
                    // arm - attn_prefill_f16 only carries 64/256/512
                    exec.attn_prefill(
                        &px.d_q,
                        ds.kv_k[li].as_ref().expect("kv_k"),
                        ds.kv_v[li].as_ref().expect("kv_v"),
                        &px.d_sinks,
                        &mut px.d_attn,
                        &px.d_pos,
                        &px.d_slots,
                        hp.n_heads,
                        hp.n_kv_heads,
                        hp.head_dim,
                        self.max_ctx,
                        kv_dim,
                        0,
                        t,
                        scale,
                        kv_dtype,
                    )?;
                    match w {
                        AttnWeights::F32 { wo, .. } => {
                            exec.gemm_f32(&wo.buf, q_dim, embd, &px.d_attn, &mut px.d_proj, t)?;
                        }
                        AttnWeights::Qw { wo, .. } => {
                            let s8 = px.q8.as_mut().expect("q8 prefill scratch");
                            prefill_quant(
                                &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &px.d_attn, q_dim, t,
                            )?;
                            prefill_mm_pre_any(
                                &exec,
                                wo,
                                &s8.xq,
                                &s8.xs,
                                &s8.yq,
                                &mut s8.xsums,
                                &mut s8.ssums,
                                &mut s8.skfix,
                                &mut px.d_proj,
                                t,
                            )?;
                        }
                    }
                }
                Mixer::Moe(w) => {
                    exec.matvec_f32_batch(&w.router, &px.d_xn, &mut px.d_logits_r, t)?;
                    exec.moe_topk_sigmoid_batch(
                        &px.d_logits_r,
                        &w.bias.buf,
                        hp.routed_scale,
                        hp.n_expert,
                        hp.n_active,
                        &mut px.d_idx,
                        &mut px.d_w,
                        t,
                    )?;
                    match &w.planes {
                        MoePlanes::Nvf4 {
                            up,
                            down,
                            sh_up,
                            sh_down,
                        } => {
                            let moe_tiled = up.layout == crate::gpu::Nvf4MoeLayout::Tiled64;
                            if moe_bs {
                                // sorted-tile mxf4nvf4 MMA class (W4A4 acts).
                                // Routed picks + the shared expert both scatter
                                // pre-weighted partials into d_part ([t, k+1, embd]:
                                // slots 0..k routed, slot k shared), folded once in
                                // fixed slot order straight into the residual - this
                                // arm replaces the trailing add().
                                let np = hp.n_active + 1;
                                if !pro_done {
                                    exec.quantize_nvf4(
                                        &px.d_xn,
                                        &mut px.d_xq4,
                                        &mut px.d_xs4,
                                        t * embd,
                                    )?;
                                }
                                exec.moe_align(
                                    &px.d_idx,
                                    &mut px.d_srow,
                                    &mut px.d_sslot,
                                    &mut px.d_bexp,
                                    t,
                                    hp.n_active,
                                    hp.n_expert,
                                    px.nb_r,
                                )?;
                                if moe_tiled {
                                    exec.nvf4_moe_up_relu2_st(
                                        up,
                                        &px.d_srow,
                                        &px.d_bexp,
                                        &px.d_xq4,
                                        &px.d_xs4,
                                        &mut px.d_fq,
                                        &mut px.d_fs,
                                        px.nb_r,
                                        32,
                                    )?;
                                    exec.nvf4_moe_down_st(
                                        down,
                                        &px.d_srow,
                                        &px.d_sslot,
                                        &px.d_bexp,
                                        Some(&px.d_w),
                                        &px.d_fq,
                                        &px.d_fs,
                                        &mut px.d_part,
                                        hp.n_active,
                                        np,
                                        0,
                                        px.nb_r,
                                        32,
                                    )?;
                                } else {
                                    exec.nvf4_moe_up_relu2_bs(
                                        up,
                                        &px.d_srow,
                                        &px.d_bexp,
                                        &px.d_xq4,
                                        &px.d_xs4,
                                        &mut px.d_fq,
                                        &mut px.d_fs,
                                        px.nb_r,
                                    )?;
                                    exec.nvf4_moe_down_bs(
                                        down,
                                        &px.d_srow,
                                        &px.d_sslot,
                                        &px.d_bexp,
                                        Some(&px.d_w),
                                        &px.d_fq,
                                        &px.d_fs,
                                        &mut px.d_part,
                                        hp.n_active,
                                        np,
                                        0,
                                        px.nb_r,
                                    )?;
                                }
                                // shared expert: identity layout over the chunk's t
                                // tokens (idx = zeros, 1 "expert"), weight 1.0
                                exec.moe_align(
                                    &px.d_sh_idx,
                                    &mut px.d_srow_s,
                                    &mut px.d_sslot_s,
                                    &mut px.d_bexp_s,
                                    t,
                                    1,
                                    1,
                                    px.nb_s,
                                )?;
                                if moe_tiled {
                                    exec.nvf4_moe_up_relu2_st(
                                        sh_up,
                                        &px.d_srow_s,
                                        &px.d_bexp_s,
                                        &px.d_xq4,
                                        &px.d_xs4,
                                        &mut px.d_fq_s,
                                        &mut px.d_fs_s,
                                        px.nb_s,
                                        32,
                                    )?;
                                    exec.nvf4_moe_down_st(
                                        sh_down,
                                        &px.d_srow_s,
                                        &px.d_sslot_s,
                                        &px.d_bexp_s,
                                        None,
                                        &px.d_fq_s,
                                        &px.d_fs_s,
                                        &mut px.d_part,
                                        1,
                                        np,
                                        hp.n_active,
                                        px.nb_s,
                                        32,
                                    )?;
                                } else {
                                    exec.nvf4_moe_up_relu2_bs(
                                        sh_up,
                                        &px.d_srow_s,
                                        &px.d_bexp_s,
                                        &px.d_xq4,
                                        &px.d_xs4,
                                        &mut px.d_fq_s,
                                        &mut px.d_fs_s,
                                        px.nb_s,
                                    )?;
                                    exec.nvf4_moe_down_bs(
                                        sh_down,
                                        &px.d_srow_s,
                                        &px.d_sslot_s,
                                        &px.d_bexp_s,
                                        None,
                                        &px.d_fq_s,
                                        &px.d_fs_s,
                                        &mut px.d_part,
                                        1,
                                        np,
                                        hp.n_active,
                                        px.nb_s,
                                    )?;
                                }
                                exec.moe_slot_combine(&px.d_part, &mut px.d_x, embd, np, t)?;
                                continue;
                            }
                            exec.nvf4_moe_up_relu2(
                                up,
                                &px.d_idx,
                                &px.d_xn,
                                &mut px.d_up,
                                hp.n_active,
                                t,
                            )?;
                            exec.nvf4_moe_down_acc(
                                down,
                                &px.d_idx,
                                &px.d_w,
                                &px.d_up,
                                &mut px.d_proj,
                                None,
                                hp.n_active,
                                t,
                                false,
                            )?;
                            exec.nvf4_moe_up_relu2(
                                sh_up,
                                &px.d_sh_idx,
                                &px.d_xn,
                                &mut px.d_sh_up,
                                1,
                                t,
                            )?;
                            exec.nvf4_moe_down_acc(
                                sh_down,
                                &px.d_sh_idx,
                                &px.d_sh_w,
                                &px.d_sh_up,
                                &mut px.d_proj,
                                None,
                                1,
                                t,
                                true,
                            )?;
                        }
                        MoePlanes::Q8 {
                            up,
                            down,
                            sh_up,
                            sh_down,
                        } => {
                            // GGUF lane: sorted dp4a class - moe_align
                            // tiles + the relu2-sorted up, quantized fused rows
                            // into the (tail-guarded) sorted down, folded per-slot
                            // straight into the residual. Routed and shared fold
                            // as separate planes (6 slots then 1 - d_proj doubles
                            // as the shared [t, 1, embd] partial), replacing the
                            // trailing add() like the nvf4 bs arm.
                            let s8 = px.q8.as_mut().expect("q8 prefill scratch");
                            exec.quantize_q8(&px.d_xn, &mut s8.xq, &mut s8.xs, t * embd)?;
                            exec.moe_align(
                                &px.d_idx,
                                &mut px.d_srow,
                                &mut px.d_sslot,
                                &mut px.d_bexp,
                                t,
                                hp.n_active,
                                hp.n_expert,
                                px.nb_r,
                            )?;
                            exec.q8_0_moe_up_relu2_sorted(
                                up,
                                &px.d_srow,
                                &px.d_bexp,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.fu_r,
                                px.nb_r,
                            )?;
                            exec.quantize_q8(
                                &s8.fu_r,
                                &mut s8.fq_r,
                                &mut s8.fs_r,
                                px.nb_r * 32 * hp.moe_ff,
                            )?;
                            exec.q8_0_moe_down_sorted(
                                down,
                                &px.d_srow,
                                &px.d_sslot,
                                &px.d_bexp,
                                &px.d_w,
                                &s8.fq_r,
                                &s8.fs_r,
                                &mut px.d_part,
                                hp.n_active,
                                px.nb_r,
                            )?;
                            exec.moe_slot_combine(&px.d_part, &mut px.d_x, embd, hp.n_active, t)?;
                            exec.moe_align(
                                &px.d_sh_idx,
                                &mut px.d_srow_s,
                                &mut px.d_sslot_s,
                                &mut px.d_bexp_s,
                                t,
                                1,
                                1,
                                px.nb_s,
                            )?;
                            exec.q8_0_moe_up_relu2_sorted(
                                sh_up,
                                &px.d_srow_s,
                                &px.d_bexp_s,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.fu_s,
                                px.nb_s,
                            )?;
                            exec.quantize_q8(
                                &s8.fu_s,
                                &mut s8.fq_s,
                                &mut s8.fs_s,
                                px.nb_s * 32 * hp.shared_ff,
                            )?;
                            exec.q8_0_moe_down_sorted(
                                sh_down,
                                &px.d_srow_s,
                                &px.d_sslot_s,
                                &px.d_bexp_s,
                                &px.d_sh_w,
                                &s8.fq_s,
                                &s8.fs_s,
                                &mut px.d_proj,
                                1,
                                px.nb_s,
                            )?;
                            exec.moe_slot_combine(&px.d_proj, &mut px.d_x, embd, 1, t)?;
                            continue;
                        }
                    }
                }
            }
            // Hoist this add into the next layer's prologue when that layer is
            // an nvf4 MoE on the bs arm - the only shape the fused kernel
            // serves, and the one every non-MoE layer here is followed by.
            let next_bs_moe = moe_bs
                && paddock_models::dev_var_os!("PADDOCK_NO_GLUE_FUSE").is_none()
                && matches!(
                    self.layers.get(li + 1).map(|l| &l.mixer),
                    Some(Mixer::Moe(w)) if matches!(w.planes, MoePlanes::Nvf4 { .. })
                );
            if next_bs_moe && exec.has_add_rmsnorm_quant_nvf4() {
                let next = &self.layers[li + 1];
                exec.add_rmsnorm_quant_nvf4_batch(
                    &mut px.d_x,
                    Some(&px.d_proj),
                    &next.norm.buf,
                    &mut px.d_xn,
                    &mut px.d_xq4,
                    &mut px.d_xs4,
                    embd,
                    eps,
                    t,
                )?;
                fused_pro = true;
            } else {
                exec.add(&mut px.d_x, &px.d_proj, t * embd)?;
            }
        }
        ds.pos += t;

        if !want_logits {
            return Ok(None);
        }
        {
            let src = px
                .d_x
                .try_slice((t - 1) * embd..t * embd)
                .expect("last row");
            exec.stream.memcpy_dtod(&src, &mut px.d_last).map_err(drv)?;
        }
        exec.rmsnorm_batch(
            &px.d_last,
            &self.final_norm.buf,
            &mut px.d_lastn,
            embd,
            eps,
            1,
        )?;
        match &self.lm_head {
            HeadW::Nvf4(h) => exec.nvf4_gemv(h, &px.d_lastn, &mut px.d_logits, None)?,
            HeadW::Qw(q) => gemv_any(&exec, q, &px.d_lastn, &mut px.d_logits)?,
        }
        Ok(Some(exec.to_host(&px.d_logits)?))
    }

    /// One token through the whole stack; returns the full logits row.
    pub(crate) fn forward_one(&mut self, token: u32) -> Result<Vec<f32>, GpuModelError> {
        self.step(token, None)
    }

    /// `forward_one` with a dtoh snapshot of the residual stream after every
    /// layer - the oracle gate's stage-sum probe. Same code path as
    /// serving (the probe is an Option, not a fork), so the gate can never
    /// drift from what forward_one actually runs.
    #[doc(hidden)]
    pub fn forward_probed(
        &mut self,
        token: u32,
        stages: &mut Vec<Vec<f32>>,
    ) -> Result<Vec<f32>, GpuModelError> {
        self.step(token, Some(stages))
    }

    fn step(
        &mut self,
        token: u32,
        probe: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<f32>, GpuModelError> {
        self.ensure_decode()?;
        let exec = self.exec.clone();
        {
            let ds = self.decode.as_mut().expect("decode");
            if ds.pos >= self.max_ctx {
                return Err(GpuModelError::ContextExceeded {
                    got: ds.pos + 1,
                    max: self.max_ctx,
                });
            }
            let pos = ds.pos as u32;
            let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
            exec.stream
                .memcpy_htod(&[token], &mut ds.d_token)
                .map_err(drv)?;
            exec.stream
                .memcpy_htod(&[pos], &mut ds.d_pos)
                .map_err(drv)?;
        }
        // Probe runs dtoh mid-walk (uncapturable) and must not replay a graph
        // that skips their hooks; the env pin is the same-binary A/B kill.
        if probe.is_some() || paddock_models::dev_var_os!("PADDOCK_NO_NEMO_GRAPH").is_some() {
            self.record_step(probe)?;
        } else {
            if self.decode.as_ref().expect("decode").graph.is_none() {
                self.capture_step_graph()?;
            }
            self.decode
                .as_ref()
                .expect("decode")
                .graph
                .as_ref()
                .expect("step graph")
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("nemotron step graph launch: {e}")))?;
        }
        self.decode.as_mut().expect("decode").pos += 1;
        let logits = exec.to_host(&self.scratch.as_ref().expect("scratch").d_logits)?;
        Ok(logits)
    }

    /// Capture `record_step` into a replayable graph (qwen35's shape). The
    /// capture only records - the caller still launches the graph for this
    /// first tick. One capture is valid at every position: the kernels read
    /// pos from `d_pos` on device and every grid is position-independent.
    fn capture_step_graph(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        let rec = self.record_step(None);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        rec?; // surface a record failure only after capture is cleanly ended
        let graph = graph?.ok_or_else(|| GpuError::Driver("capture produced no graph".into()))?;
        self.decode.as_mut().expect("decode").graph = Some(SendGraph(graph));
        Ok(())
    }

    /// One decode tick's launch train - embed gather through lm_head, logits
    /// left in `sc.d_logits`. Reads `d_token`/`d_pos` from device, allocates
    /// and syncs nothing (probe runs excepted), so it records cleanly into a
    /// graph. Shared verbatim by the eager path, the capture, and (via the
    /// graph) the decode pipe.
    fn record_step(&mut self, mut probe: Option<&mut Vec<Vec<f32>>>) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let (embd, eps) = (hp.hidden, hp.eps);
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let d_inner = hp.d_inner();
        let conv_dim = hp.conv_dim();
        let scale = 1.0 / (hp.head_dim as f32).sqrt();
        let kv_dtype = self.kv_dtype;
        // decode rung: fused wave-dense MoE chain when the pack carries it
        // (PADDOCK_NO_NVF4_MOE_MT=1 is the same-binary A/B kill switch)
        let moe_mt = self.exec.has_nvf4_moe_mt()
            && paddock_models::dev_var_os!("PADDOCK_NO_NVF4_MOE_MT").is_none();
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");

        match &self.tok_embd {
            TokEmbd::F32(tab) => exec.embed_gather(tab, &ds.d_token, &mut sc.d_x, embd)?,
            TokEmbd::Bf16(tab) => {
                exec.embed_gather_bf16(tab, &ds.d_token, &mut sc.d_x, embd, 1, 1.0)?
            }
            TokEmbd::Q8(tab) => {
                exec.embed_gather_batch_q8(tab, &ds.d_token, &mut sc.d_x, embd, 1)?
            }
        }

        for (li, layer) in self.layers.iter().enumerate() {
            exec.rmsnorm_batch(&sc.d_x, &layer.norm.buf, &mut sc.d_xn, embd, eps, 1)?;
            match &layer.mixer {
                Mixer::Mamba(w) => {
                    // in_proj -> [z | x B C | dt] one fused row
                    match &w.in_proj {
                        LinW::F8(p) => {
                            exec.f8r_gemv(p, &sc.d_xn, &mut sc.d_zxbcdt, embd, hp.in_proj_rows())?
                        }
                        LinW::Qw(q) => gemv_any(&exec, q, &sc.d_xn, &mut sc.d_zxbcdt)?,
                    }
                    // conv over the x|B|C span (offset d_inner), bias + silu
                    exec.mamba_conv_step(
                        ds.conv_win[li].as_mut().expect("win"),
                        &sc.d_zxbcdt,
                        d_inner,
                        &w.conv_w,
                        &w.conv_b,
                        &mut sc.d_conv,
                        conv_dim,
                        hp.d_conv,
                    )?;
                    // scan: dt rides the fused row's tail (offset d_inner + conv_dim)
                    exec.mamba2_scan_seq(
                        ds.ssm[li].as_mut().expect("ssm"),
                        &sc.d_conv,
                        &sc.d_zxbcdt,
                        d_inner + conv_dim,
                        hp.in_proj_rows(),
                        &w.a,
                        &w.d,
                        &w.dt_bias,
                        &mut sc.d_y,
                        1,
                        hp.mamba_heads,
                        hp.mamba_head_dim,
                        hp.d_state,
                        hp.n_groups,
                    )?;
                    // gated grouped norm: z is the fused row's head
                    exec.mamba_rmsnorm_gated_g(
                        &sc.d_y,
                        &sc.d_zxbcdt,
                        0,
                        hp.in_proj_rows(),
                        &w.norm_w,
                        &mut sc.d_yn,
                        1,
                        d_inner,
                        hp.n_groups,
                        eps,
                    )?;
                    match &w.out_proj {
                        LinW::F8(p) => exec.f8r_gemv(p, &sc.d_yn, &mut sc.d_proj, d_inner, embd)?,
                        LinW::Qw(q) => gemv_any(&exec, q, &sc.d_yn, &mut sc.d_proj)?,
                    }
                }
                Mixer::Attn(w) => {
                    // NoPE: no rotary anywhere - q/k go to the cache as
                    // projected (kq_scale is the only scaling)
                    match w {
                        // bf16 twins when present: same products (the widen is
                        // exact), warp-local summation regrouped - the
                        // sanctioned reorder class; the reference battery gates it
                        AttnWeights::F32 {
                            wq, wk, wv, bf16, ..
                        } => {
                            if let Some(b) = bf16 {
                                // per-projection segments of the fused plane
                                // (same bytes, same products as the old
                                // separate planes)
                                exec.bf16_gemv_rows(&b.wqkv, 0, b.q_dim, &sc.d_xn, &mut sc.d_q)?;
                                exec.bf16_gemv_rows(
                                    &b.wqkv,
                                    b.q_dim,
                                    b.kv_dim,
                                    &sc.d_xn,
                                    &mut sc.d_k,
                                )?;
                                exec.bf16_gemv_rows(
                                    &b.wqkv,
                                    b.q_dim + b.kv_dim,
                                    b.kv_dim,
                                    &sc.d_xn,
                                    &mut sc.d_v,
                                )?;
                            } else {
                                exec.matvec_f32_batch(wq, &sc.d_xn, &mut sc.d_q, 1)?;
                                exec.matvec_f32_batch(wk, &sc.d_xn, &mut sc.d_k, 1)?;
                                exec.matvec_f32_batch(wv, &sc.d_xn, &mut sc.d_v, 1)?;
                            }
                        }
                        AttnWeights::Qw { wq, wk, wv, .. } => {
                            gemv_any(&exec, wq, &sc.d_xn, &mut sc.d_q)?;
                            gemv_any(&exec, wk, &sc.d_xn, &mut sc.d_k)?;
                            gemv_any(&exec, wv, &sc.d_xn, &mut sc.d_v)?;
                        }
                    }
                    exec.kv_append_batch(
                        &sc.d_k,
                        ds.kv_k[li].as_mut().expect("kv_k"),
                        &ds.d_pos,
                        Some(&ds.d_slots),
                        kv_dim,
                        self.max_ctx,
                        1,
                        kv_dtype,
                    )?;
                    exec.kv_append_batch(
                        &sc.d_v,
                        ds.kv_v[li].as_mut().expect("kv_v"),
                        &ds.d_pos,
                        Some(&ds.d_slots),
                        kv_dim,
                        self.max_ctx,
                        1,
                        kv_dtype,
                    )?;
                    exec.attn_decode_batch(
                        &sc.d_q,
                        ds.kv_k[li].as_ref().expect("kv_k"),
                        ds.kv_v[li].as_ref().expect("kv_v"),
                        &sc.d_sinks,
                        &mut sc.d_attn,
                        &ds.d_pos,
                        Some(&ds.d_slots),
                        hp.n_heads,
                        hp.n_kv_heads,
                        hp.head_dim,
                        self.max_ctx,
                        kv_dim,
                        0,
                        1,
                        scale,
                        kv_dtype,
                    )?;
                    match w {
                        AttnWeights::F32 { wo, bf16, .. } => {
                            if let Some(b) = bf16 {
                                exec.bf16_gemv(&b.wo, None, &sc.d_attn, &mut sc.d_proj)?;
                            } else {
                                exec.matvec_f32_batch(wo, &sc.d_attn, &mut sc.d_proj, 1)?;
                            }
                        }
                        AttnWeights::Qw { wo, .. } => {
                            gemv_any(&exec, wo, &sc.d_attn, &mut sc.d_proj)?;
                        }
                    }
                }
                Mixer::Moe(w) => {
                    // fp32 sigmoid router, selection-biased / combine-unbiased,
                    // renorm + x2.5 folded into the weights (routed only)
                    exec.matvec_f32_batch(&w.router, &sc.d_xn, &mut sc.d_logits_r, 1)?;
                    exec.moe_topk_sigmoid_batch(
                        &sc.d_logits_r,
                        &w.bias.buf,
                        hp.routed_scale,
                        hp.n_expert,
                        hp.n_active,
                        &mut sc.d_idx,
                        &mut sc.d_w,
                        1,
                    )?;
                    match &w.planes {
                        MoePlanes::Nvf4 {
                            up,
                            down,
                            sh_up,
                            sh_down,
                        } => {
                            if moe_mt && probe.is_none() {
                                // decode rung: two wave-dense fused launches, then the
                                // deterministic ascending-slot fold straight into the
                                // residual (shared expert = slot n_active, weight 1.0).
                                // The probe path below materializes the mixer output
                                // off d_proj for the oracle hook (mtt + zeroed combine
                                // under the TILED layout, the 4-launch pair on Row).
                                if up.layout == crate::gpu::Nvf4MoeLayout::Tiled64 {
                                    exec.nvf4_moe_up_relu2_mtt(
                                        up,
                                        sh_up,
                                        &sc.d_idx,
                                        &sc.d_xn,
                                        &mut sc.d_act,
                                        hp.n_active,
                                    )?;
                                    exec.nvf4_moe_down_part_tt(
                                        down,
                                        sh_down,
                                        &sc.d_idx,
                                        &sc.d_w,
                                        &sc.d_act,
                                        &mut sc.d_part7,
                                        hp.n_active,
                                    )?;
                                } else {
                                    exec.nvf4_moe_up_relu2_mt(
                                        up,
                                        sh_up,
                                        &sc.d_idx,
                                        &sc.d_xn,
                                        &mut sc.d_act,
                                        hp.n_active,
                                    )?;
                                    exec.nvf4_moe_down_part(
                                        down,
                                        sh_down,
                                        &sc.d_idx,
                                        &sc.d_w,
                                        &sc.d_act,
                                        &mut sc.d_part7,
                                        hp.n_active,
                                    )?;
                                }
                                exec.moe_slot_combine(
                                    &sc.d_part7,
                                    &mut sc.d_x,
                                    embd,
                                    hp.n_active + 1,
                                    1,
                                )?;
                                continue;
                            }
                            if up.layout == crate::gpu::Nvf4MoeLayout::Tiled64 {
                                // probe path under the TILED layout: the probe is not
                                // a fork (the oracle must see the SERVING kernels), so
                                // run the mtt pair and materialize the mixer output in
                                // d_proj via a zeroed combine (slot_combine adds).
                                exec.nvf4_moe_up_relu2_mtt(
                                    up,
                                    sh_up,
                                    &sc.d_idx,
                                    &sc.d_xn,
                                    &mut sc.d_act,
                                    hp.n_active,
                                )?;
                                exec.nvf4_moe_down_part_tt(
                                    down,
                                    sh_down,
                                    &sc.d_idx,
                                    &sc.d_w,
                                    &sc.d_act,
                                    &mut sc.d_part7,
                                    hp.n_active,
                                )?;
                                exec.zero_region(&mut sc.d_proj, 0, embd)?;
                                exec.moe_slot_combine(
                                    &sc.d_part7,
                                    &mut sc.d_proj,
                                    embd,
                                    hp.n_active + 1,
                                    1,
                                )?;
                            } else {
                                exec.nvf4_moe_up_relu2(
                                    up,
                                    &sc.d_idx,
                                    &sc.d_xn,
                                    &mut sc.d_up,
                                    hp.n_active,
                                    1,
                                )?;
                                exec.nvf4_moe_down_acc(
                                    down,
                                    &sc.d_idx,
                                    &sc.d_w,
                                    &sc.d_up,
                                    &mut sc.d_proj,
                                    None,
                                    hp.n_active,
                                    1,
                                    false,
                                )?;
                                // shared expert: same kernels at k=1, unscaled, accumulated
                                exec.nvf4_moe_up_relu2(
                                    sh_up,
                                    &sc.d_sh_idx,
                                    &sc.d_xn,
                                    &mut sc.d_sh_up,
                                    1,
                                    1,
                                )?;
                                exec.nvf4_moe_down_acc(
                                    sh_down,
                                    &sc.d_sh_idx,
                                    &sc.d_sh_w,
                                    &sc.d_sh_up,
                                    &mut sc.d_proj,
                                    None,
                                    1,
                                    1,
                                    true,
                                )?;
                            }
                        }
                        MoePlanes::Q8 {
                            up,
                            down,
                            sh_up,
                            sh_down,
                        } => {
                            // GGUF lane: token-batched dp4a relu2 pair (W8A8, same
                            // int8 class as llama.cpp's mmvq decode). The down
                            // kernel writes, never accumulates - shared lands in
                            // its own row and folds with one add, so d_proj still
                            // carries the full mixer output for the probe hook.
                            let s8 = sc.q8.as_mut().expect("q8 scratch");
                            exec.quantize_q8(&sc.d_xn, &mut s8.xq, &mut s8.xs, embd)?;
                            exec.q8_0_moe_up_relu2(
                                up,
                                &sc.d_idx,
                                &s8.xq,
                                &s8.xs,
                                &mut sc.d_up,
                                hp.n_active,
                                1,
                            )?;
                            exec.quantize_q8(
                                &sc.d_up,
                                &mut s8.fq_r,
                                &mut s8.fs_r,
                                hp.n_active * hp.moe_ff,
                            )?;
                            exec.q8_0_moe_down(
                                down,
                                &sc.d_idx,
                                &sc.d_w,
                                &s8.fq_r,
                                &s8.fs_r,
                                &mut sc.d_proj,
                                hp.n_active,
                                1,
                            )?;
                            exec.q8_0_moe_up_relu2(
                                sh_up,
                                &sc.d_sh_idx,
                                &s8.xq,
                                &s8.xs,
                                &mut sc.d_sh_up,
                                1,
                                1,
                            )?;
                            exec.quantize_q8(
                                &sc.d_sh_up,
                                &mut s8.fq_s,
                                &mut s8.fs_s,
                                hp.shared_ff,
                            )?;
                            exec.q8_0_moe_down(
                                sh_down,
                                &sc.d_sh_idx,
                                &sc.d_sh_w,
                                &s8.fq_s,
                                &s8.fs_s,
                                &mut s8.shproj,
                                1,
                                1,
                            )?;
                            exec.add(&mut sc.d_proj, &s8.shproj, embd)?;
                        }
                    }
                }
            }
            // probe before the residual add: the reference oracle hooks
            // vLLM's layer modules, whose forward returns (mixer_out, residual)
            // - output[0] is the mixer output, the residual rides separately
            if let Some(stages) = probe.as_deref_mut() {
                stages.push(exec.to_host(&sc.d_proj)?);
            }
            exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
        }

        exec.rmsnorm_batch(&sc.d_x, &self.final_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
        match &self.lm_head {
            HeadW::Nvf4(h) => exec.nvf4_gemv(h, &sc.d_xn, &mut sc.d_logits, None)?,
            HeadW::Qw(q) => gemv_any(&exec, q, &sc.d_xn, &mut sc.d_logits)?,
        }
        Ok(())
    }

    /// Enqueue pipe tick `pipe.tick`: params into its ring slot, the on-device
    /// advance (skipped for tick 0 - inputs were just uploaded), the step
    /// graph, the row sampler, and the readability event. Mirrors gpt-oss's
    /// `pipe_launch_tick` at rows = 1.
    fn pipe_launch_tick(&mut self, plan: &RowSample, advance: bool) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.hp.vocab;
        let tick = self.pipe.as_ref().expect("pipe active").tick;
        let ring = tick % 2;
        // pack the single row's sampler params ({inv_t, u, mode, pad})
        let mut par = [0u32; 4];
        match plan {
            RowSample::Hole | RowSample::Host => {}
            RowSample::Device(DevicePlan::Greedy) => par[2] = 1,
            RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                par[0] = inv_t.to_bits();
                par[1] = u.to_bits();
                par[2] = 2;
            }
            // RS plans never reach the serial pipe (spec-only)
            // P65 TruncCat is qwen35-only (supports_host_head); skip-safe
            RowSample::Device(DevicePlan::TruncCat { .. }) => {}
            RowSample::Device(DevicePlan::RsVerify { .. })
            | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        {
            let ds = self.decode.as_mut().expect("decode");
            // the tick consumes position ds.pos - refuse to run off the window
            if ds.pos >= self.max_ctx {
                return Err(GpuModelError::ContextExceeded {
                    got: ds.pos + 1,
                    max: self.max_ctx,
                });
            }
            let mut v = ds.d_pipe_par.slice_mut(ring * 4..ring * 4 + 4);
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if advance {
                // previous tick's out slot becomes this tick's input token
                let prev = (tick + 1) % 2;
                let (out, tok, pos) = (&ds.d_pipe_out, &mut ds.d_token, &mut ds.d_pos);
                exec.pipe_advance(out, prev, tok, pos, 1)?;
            }
        }
        self.decode
            .as_ref()
            .expect("decode")
            .graph
            .as_ref()
            .expect("step graph")
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("pipe step graph launch: {e}")))?;
        {
            let sc = self.scratch.as_ref().expect("scratch");
            let ds = self.decode.as_mut().expect("decode");
            let (par_buf, out) = (&ds.d_pipe_par, &mut ds.d_pipe_out);
            exec.sample_rows_at(&sc.d_logits, par_buf, ring * 4, out, ring, 1, vocab)?;
        }
        // host mirror of the on-device position advance, so the ctx guard
        // above and any post-drain bookkeeping see the true position
        self.decode.as_mut().expect("decode").pos += 1;
        let ev = exec.record_event()?;
        self.pipe.as_mut().expect("pipe active").ev[ring] = Some(ev);
        Ok(())
    }

    /// Kill an in-flight pipe (error paths + reset): quiesce the stream so
    /// nothing still reads the rings, then drop the state.
    pub(super) fn pipe_abort(&mut self) {
        if self.pipe.take().is_some() {
            let _ = self.exec.synchronize();
        }
    }
}

impl GpuNemotron {
    pub fn set_max_ctx(&mut self, max_ctx: usize) {
        self.pipe_abort(); // in-flight ticks still read the buffers we drop
        self.max_ctx = max_ctx;
        self.decode = None;
    }
}

impl Generator for GpuNemotron {
    fn release_idle_memory(&mut self) {
        self.exec.trim_mem_pool();
    }

    fn tier_pump(&mut self) {
        self.tier_pump_impl();
    }
    fn reply_pin(&mut self, slot: usize) {
        GpuNemotron::reply_pin(self, slot);
    }
    fn anchor_at(&mut self, _slot: usize, tokens: &[u32], upto: usize) {
        GpuNemotron::anchor_at(self, tokens, upto);
    }
    fn tier_prefix_loading(&mut self, slot: usize, tokens: &[u32]) -> bool {
        self.tier_consult_impl(slot, tokens)
    }
    fn tier_observe_prefill(&mut self, tokens: u32, wall_us: f64) {
        if let Some(t) = self.batch.as_mut().and_then(|b| b.tier.as_mut()) {
            t.cost.observe_prefill(tokens, wall_us);
        }
    }
    fn tier_stats(&self) -> Option<crate::kv_tier::TierStats> {
        self.tier_stats_impl()
    }
    fn tier_report(&self) -> Option<crate::kv_tier::TierReport> {
        self.batch
            .as_ref()?
            .tier
            .as_ref()
            .map(crate::kv_tier::PoolTier::report)
    }
    fn reset(&mut self) {
        // a dying request may leave pipe ticks in flight - quiesce before
        // zeroing the state those ticks still read/write
        self.pipe_abort();
        // KV reads are position-bounded; SSM state + conv window are not -
        // they must return to zero for a fresh sequence.
        let exec = self.exec.clone();
        if let Some(ds) = self.decode.as_mut() {
            ds.pos = 0;
            for s in ds.ssm.iter_mut().flatten() {
                let n = s.len();
                let _ = exec.zero_region(s, 0, n);
            }
            for w in ds.conv_win.iter_mut().flatten() {
                let n = w.len();
                let _ = exec.zero_region(w, 0, n);
            }
        }
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.forward_one(token).map_err(gen_err)
    }

    fn forward_prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        // Batched lane: slot 0 is this stream's own (paged pool + arenas).
        if self.batch.is_some() {
            return self.forward_prefill_impl(0, tokens).map_err(gen_err);
        }
        // Bulk chunked prefill: batched kernels over
        // token spans instead of one full forward per prompt token. Serial
        // fallback for tiny prompts, packs missing the kernel set, or a
        // non-f16 KV dtype (the f16 prefill-attention arm is the only one
        // this serial lane wires). Since the fp8 default (2026-10-04) that
        // means an fp8 serial stream walks token by token - acceptable
        // because serving never lands here: the service enables the batch
        // lane at every width, max_batch 1 included, and the batch lane
        // prefills fp8 through its own G=16 tile. This lane is the reference
        // walk the tests drive, and they pin f16.
        // Per-lane kernel set: the GGUF class prefills through the int8 mmq
        // ladder and never touches the fp8 pair, so asking it for f8row_gemm
        // used to send Q8_0 nemotron down the token-at-a-time fallback below
        // on anything under sm_89 - a 128-token prompt cost 1028 ms to first
        // token instead of a couple hundred. See has_nemotron_prefill_gguf.
        let prefill_ok = if self.is_gguf() {
            self.exec.has_nemotron_prefill_gguf()
        } else {
            self.exec.has_nemotron_prefill_f8()
        };
        if tokens.len() >= 8 && prefill_ok && self.kv_dtype == crate::gpu::KvDtype::Fp16 {
            return self.prefill_stream_impl(tokens).map_err(gen_err);
        }
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.forward(t)?;
        }
        Ok(logits)
    }

    // ── the batched serving lane (batch.rs, stages B+C) ─────────────

    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        self.enable_batch_impl(max_batch).map_err(gen_err)
    }

    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        self.batch_step(tokens, positions).map_err(gen_err)?;
        self.read_batch_logits(tokens.len()).map_err(gen_err)
    }

    fn supports_device_sampling(&self) -> bool {
        self.supports_device_sampling_impl()
    }

    fn supports_device_trunc(&self) -> bool {
        self.device_trunc_supported()
    }

    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<crate::generator::SampledStep, GenError> {
        self.forward_batch_sampled_impl(tokens, positions, plans)
            .map_err(gen_err)
    }

    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.forward_prefill_impl(slot, tokens).map_err(gen_err)
    }

    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        self.forward_prefill_batch_impl(items).map_err(gen_err)
    }

    fn supports_chunked_prefill(&self) -> bool {
        self.batch.is_some()
    }

    // the mixed tick's FIFO over the queue, row-exact from each cursor
    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.chunked
            .iter()
            .map(|c| (c.slot, c.cursor, c.tokens.len() - c.cursor))
            .collect()
    }

    // plan_chunk's cap under the pass's row capacity (the decode rows share
    // it); the prompt-aligned width rule only ever narrows a tick
    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        self.batch.as_ref().map_or(0, |bs| {
            self.prefill_chunk.min(bs.cap.saturating_sub(decode_rows))
        })
    }

    fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GenError> {
        self.prefill_begin_impl(slot, tokens).map_err(gen_err)
    }

    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.prefill_abort_impl(slot)
    }

    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        self.forward_mixed_impl(decodes, budget).map_err(gen_err)
    }

    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GenError,
    > {
        self.forward_mixed_sampled_impl(decodes, budget, plans, fin_plans)
            .map_err(gen_err)
    }

    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        self.forward_spec_batch_impl(reqs).map_err(gen_err)
    }

    fn spec_capable(&self) -> bool {
        // a drafter must be present (DFlash sideload or the in-file MTP)
        // AND the verify kernel set; the service's policy resolution
        // handles --spec/--no-spec
        (self.dflash.is_some() || self.mtp.is_some()) && self.spec_verify_ready()
    }

    fn forward_spec_batch_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GenError> {
        self.forward_spec_batch_plans_impl(reqs, plans)
            .map_err(gen_err)
    }

    fn forward_spec_verify(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<f32>>, GenError> {
        self.forward_spec_verify_impl(reqs).map_err(gen_err)
    }

    fn spec_commit(&mut self, committed: &[u32]) -> Result<(), GenError> {
        self.spec_commit_impl(committed).map_err(gen_err)
    }

    fn spec_block_width(&self) -> Option<usize> {
        self.dflash
            .as_ref()
            .map(|_| crate::gpu_model::nemotron::dflash::BLOCK)
    }

    fn spec_fixed_draft_depth(&self) -> Option<usize> {
        // the elected verify depth over the trained block (dflash.rs): the
        // chain ramp's per-slot k resized the block every round instead
        self.dflash.as_ref().map(|d| d.verify_depth())
    }

    fn spec_live_cap(&self) -> usize {
        // every live slot's verify rows must fit the round's snapshot planes
        // at depth 1 or more; the per-live budget below narrows the depth
        match &self.dflash {
            Some(d) => (1..=crate::gpu_model::nemotron::spec::SPEC_ROWS_NEMO)
                .take_while(|&live| d.depth_for_live(live) > 0)
                .last()
                .unwrap_or(0),
            None => usize::MAX,
        }
    }

    fn spec_batch_draft_budget(&self, live: usize) -> Option<usize> {
        self.dflash.as_ref().map(|d| d.depth_for_live(live))
    }

    fn spec_sampled_live_cap(&self) -> Option<usize> {
        // the block drafter's rounds pay for themselves at every width the
        // per-live depth budget speculates at - tool-carrying sampled traffic
        // at 6 / 8 live: 191 -> 229 and 215 -> 271 tok/s against no drafter
        // (GB10 2026-09-26), where the service's generic 4-live cap had it off
        self.dflash.as_ref().map(|_| self.spec_live_cap())
    }

    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        _committed: &[u32],
        want_pos: u32,
    ) -> Result<bool, GenError> {
        // Neither drafter has a re-warm path: state flows from every batched
        // walk, so cold means genuinely un-walked rows (laguna's stance).
        // Warm = coverage through the last committed row (want_pos), i.e.
        // [0, want_pos + 1).
        if self.dflash.is_some() {
            Ok(self.dflash_warm(slot, want_pos as usize + 1))
        } else {
            Ok(self.mtp_warm(slot, want_pos as usize + 1))
        }
    }

    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        if self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .is_some()
        {
            use crate::gpu_model::nemotron::dflash::{BLOCK, MAX_DRAFT};
            let k = k.min(MAX_DRAFT);
            // warm slots draft together in one pass; features cover committed
            // rows [0, end), the pending token sits at `end`
            let mut reqs = Vec::with_capacity(pendings.len());
            let mut at = Vec::with_capacity(pendings.len());
            for (i, &(slot, tok)) in pendings.iter().enumerate() {
                let end = {
                    let st = self
                        .dflash
                        .as_ref()
                        .expect("dflash")
                        .state
                        .as_ref()
                        .expect("dflash state");
                    st.feat[slot].1 as usize
                };
                if end > 0 && self.dflash_warm(slot, end) {
                    reqs.push((slot, end, tok));
                    at.push(i);
                }
            }
            let mut out = vec![Vec::new(); pendings.len()];
            // a verify round takes SPEC_ROWS rows, one per pending slot:
            // drafts past that would only make the round decline
            let kr = if reqs.is_empty() {
                0
            } else {
                k.min(
                    crate::gpu_model::nemotron::spec::SPEC_ROWS_NEMO.saturating_sub(pendings.len())
                        / reqs.len(),
                )
            };
            if kr > 0 {
                // DFlash runs its trained block whatever the round verifies
                // (see dflash::BLOCK) and the verify takes the leading kr;
                // DSpark's block is causal, so its first kr rows ARE the
                // trained block's first kr - drafting more buys nothing
                let causal = self.dflash.as_ref().is_some_and(|d| d.causal);
                let kg = if causal {
                    kr
                } else {
                    k.clamp(BLOCK - 1, MAX_DRAFT)
                };
                let d = self.dflash_draft_batch(&reqs, kg, kr).map_err(gen_err)?;
                for (i, d) in at.into_iter().zip(d) {
                    out[i] = d;
                }
            }
            return Ok(Some(out));
        }
        if self.mtp.as_ref().and_then(|m| m.state.as_ref()).is_some() {
            let k = k.min(crate::gpu_model::nemotron::mtp::MTP_MAX_DRAFT);
            let mut out = Vec::with_capacity(pendings.len());
            for &(slot, tok) in pendings {
                let end = {
                    let st = self
                        .mtp
                        .as_ref()
                        .expect("mtp weights")
                        .state
                        .as_ref()
                        .expect("mtp state");
                    st.feat[slot].1 as usize
                };
                if !self.mtp_warm(slot, end) || end == 0 {
                    out.push(Vec::new());
                    continue;
                }
                out.push(self.mtp_draft(slot, end, tok, k).map_err(gen_err)?);
            }
            return Ok(Some(out));
        }
        Ok(None)
    }

    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        self.release_inactive_slots_impl(occupied);
    }

    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.last_reused.get_mut(slot).map_or(0, std::mem::take)
    }

    fn pool_free_blocks(&self) -> Option<usize> {
        self.pool_free_blocks_impl()
    }

    fn kv_mem_bytes(&self) -> Option<u64> {
        self.kv_mem_bytes_impl()
    }

    /// Depth-2 decode pipe. Batch mode routes to the stage-E pipe-under-pool
    /// (batch.rs - per-r graph replay + device sampling + on-device advance);
    /// serial mode keeps the rung-5b serial pipe (which replays the serial
    /// DecodeState - enable_batch tears that down, so the two never mix).
    fn supports_decode_pipe(&self) -> bool {
        if self.batch.is_some() {
            return self.supports_decode_pipe_batch();
        }
        self.exec.has_sample_rows()
            && self.exec.has_pipe_advance()
            && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_NEMO_GRAPH").is_none()
    }

    fn decode_pipe_begin_slots(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), GenError> {
        self.decode_pipe_begin_b(tokens, positions, Some(slots), plans)
            .map_err(gen_err)
    }

    fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), GenError> {
        if self.batch.is_some() {
            return self
                .decode_pipe_begin_b(tokens, positions, None, plans)
                .map_err(gen_err);
        }
        if tokens.len() != 1 || positions.len() != 1 || plans.len() != 1 {
            return Err(GenError::Backend(
                "nemotron pipe is serial (rows = 1)".into(),
            ));
        }
        if !self.supports_decode_pipe() {
            return Err(GenError::Backend("decode pipe unsupported".into()));
        }
        assert!(self.pipe.is_none(), "decode pipe already active");
        let r = (|| -> Result<(), GpuModelError> {
            self.ensure_decode()?;
            let exec = self.exec.clone();
            {
                let ds = self.decode.as_mut().expect("decode");
                // the serve loop's position must agree with the sequence state
                // the prefill left behind - a mismatch means KV corruption
                if positions[0] as usize != ds.pos {
                    return Err(GpuModelError::Unsupported(format!(
                        "pipe begin at position {} but sequence is at {}",
                        positions[0], ds.pos
                    )));
                }
                let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
                exec.stream
                    .memcpy_htod(&[tokens[0]], &mut ds.d_token)
                    .map_err(drv)?;
                exec.stream
                    .memcpy_htod(&[positions[0]], &mut ds.d_pos)
                    .map_err(drv)?;
            }
            if self.decode.as_ref().expect("decode").graph.is_none() {
                self.capture_step_graph()?;
            }
            // once per serve: the bench-stamp trail must show the pipe engaged
            static PIPE_LOGGED: std::sync::Once = std::sync::Once::new();
            PIPE_LOGGED.call_once(|| {
                tracing::info!("nemotron: serial decode pipe ON (graph ticks + device sampling)");
            });
            self.pipe = Some(PipeState {
                tick: 0,
                ev: [None, None],
            });
            self.pipe_launch_tick(&plans[0], false)
        })();
        if let Err(e) = r {
            self.pipe_abort();
            return Err(gen_err(e));
        }
        Ok(())
    }

    fn decode_pipe_next(&mut self, plans: &[RowSample]) -> Result<Vec<u32>, GenError> {
        if self.batch.is_some() {
            return self.decode_pipe_next_b(plans).map_err(gen_err);
        }
        assert_eq!(plans.len(), 1, "nemotron pipe is serial (rows = 1)");
        let j = self
            .pipe
            .as_ref()
            .ok_or_else(|| GenError::Backend("decode_pipe_next without begin".into()))?
            .tick;
        self.pipe.as_mut().expect("pipe").tick = j + 1;
        if let Err(e) = self.pipe_launch_tick(&plans[0], true) {
            self.pipe_abort();
            return Err(gen_err(e));
        }
        let ring = j % 2;
        let r = {
            let ds = self.decode.as_ref().expect("decode");
            let ev = self.pipe.as_ref().expect("pipe").ev[ring]
                .as_ref()
                .expect("tick event");
            self.exec.to_host_u32_after(ev, &ds.d_pipe_out, ring, 1)
        };
        match r {
            Ok(ids) => Ok(ids),
            Err(e) => {
                self.pipe_abort();
                Err(gen_err(e.into()))
            }
        }
    }

    fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GenError> {
        if self.batch.is_some() {
            return self.decode_pipe_drain_b().map_err(gen_err);
        }
        let st = self
            .pipe
            .take()
            .ok_or_else(|| GenError::Backend("decode_pipe_drain without begin".into()))?;
        let ring = st.tick % 2;
        let ev = st.ev[ring].as_ref().expect("tick event");
        let ds = self.decode.as_ref().expect("decode");
        match self.exec.to_host_u32_after(ev, &ds.d_pipe_out, ring, 1) {
            Ok(ids) => Ok(ids),
            Err(e) => {
                // pipe state is already gone - quiesce so nothing reads the rings
                let _ = self.exec.synchronize();
                Err(gen_err(e.into()))
            }
        }
    }

    fn vocab(&self) -> usize {
        self.hp.vocab
    }

    fn max_context(&self) -> usize {
        self.max_ctx
    }

    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weights_bytes)
    }

    fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }
}
