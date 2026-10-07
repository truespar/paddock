//! DeepSeek-OCR decoder weights - loader-first bring-up against the real
//! `Unlimited-OCR-Q8_0.gguf`, the whisper/laguna discipline.
//!
//! The decoder is a 12-layer DeepSeek-V2 MoE that is not MLA (plain Llama MHA,
//! 10 q / 10 kv, head_dim 128 - see [`super::Hparams`] for the three metadata
//! traps). The FFN split mirrors laguna exactly - one dense lead layer, then
//! routed experts plus an always-on ungated shared expert - which is why this
//! file reads like laguna's loader with the quirks deleted: no per-head QK
//! norms, no attention gate, no YaRN, no per-layer head counts, no router
//! bias plane (greedy top-k has no `exp_probs_b`; that is the V3/noaux class).
//!
//! Router math this loader is feeding (pinned from the reference's
//! `MoEGate`): logits in f32 (the
//! file ships `ffn_gate_inp` as F32 for exactly that reason), softmax over 64,
//! greedy top-6 unsorted, weights are the chosen softmax probs with no top-k
//! renormalization and scale 1.0. Laguna's sigmoid/bias/renorm path must not
//! be reused - same shape, different math, silent when wrong.

use std::sync::Arc;

use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;

use super::Hparams;
use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype, QuantTensor, QuantW, RepackedQ8};
use crate::gpu_model::gpt_oss::GpuModelError;

/// Routed experts + the always-on shared expert of one MoE layer.
//
// Resident and shape-gated; the forward graph (the next step) is
// the consumer. Same allowance the tower structs carried before encode.rs.
#[allow(dead_code)]
pub(crate) struct DsMoe {
    /// `ffn_gate_inp` [embd, n_expert] F32 - the router matvec plane. F32 in
    /// the FILE, matching the reference's `F.linear(x.float(), W.float())`.
    pub router_w: DeviceTensor,
    /// `ffn_{gate,up}_exps` [embd, moe_ff, n_expert] Q8_0, repacked; expert
    /// row (e, o) sits at e*moe_ff + o. Same seat qwen35moe/laguna serve.
    pub gate_exps: RepackedQ8,
    pub up_exps: RepackedQ8,
    /// `ffn_down_exps` [moe_ff, embd, n_expert] Q8_0.
    pub down_exps: RepackedQ8,
    /// `ffn_{gate,up,down}_shexp` - plain SwiGLU FFN of width
    /// `n_expert_shared * n_ff_exp` (1792), the two shared experts pre-merged
    /// by the converter. Ungated, always added.
    pub shexp_gate: QuantW,
    pub shexp_up: QuantW,
    pub shexp_down: QuantW,
}

/// Per-layer FFN: layer 0 is dense (6848), layers 1..12 route.
// the MoE bundle dwarfs the dense triple by design; one per layer, matched in
// forward.rs and batch.rs, so boxing would only add a hop on the hot path
#[allow(dead_code, clippy::large_enum_variant)]
pub(crate) enum DsFfn {
    Dense {
        gate: QuantW,
        up: QuantW,
        down: QuantW,
    },
    Moe(DsMoe),
}

/// One decoder block. Everything unbiased; norms are RMS eps 1e-6.
#[allow(dead_code)]
pub(crate) struct DsLayer {
    pub attn_norm: DeviceTensor,
    /// `attn_{q,k,v}` [1280, 1280] - G=1, kv_dim == q_dim.
    pub wq: QuantW,
    pub wk: QuantW,
    pub wv: QuantW,
    pub wo: QuantW,
    pub ffn_norm: DeviceTensor,
    pub ffn: DsFfn,
}

/// The decoder, weights-resident. KV pool / batch state / the DeepEncoder
/// attachment arrive with the forward graph - this struct is the loader
/// milestone, and the load gate pins its geometry and residency.
#[allow(dead_code)]
pub struct GpuDeepseekOcr {
    pub(crate) exec: Arc<GpuExecutor>,
    pub hp: Hparams,
    /// Q8_0-resident token embeddings, gathered per token (`embed_gather_q8`).
    pub(crate) tok_embd: QuantTensor,
    pub(crate) layers: Vec<DsLayer>,
    pub(crate) output_norm: DeviceTensor,
    /// `output.weight` - untied lm_head, vocab 129280.
    pub(crate) lm_head: QuantW,
    pub max_ctx: usize,
    pub weights_bytes: u64,
    pub(crate) kv_dtype: KvDtype,
    pub(crate) decode: Option<super::forward::DecodeState>,
    pub(crate) scratch: Option<super::forward::Scratch>,
    pub(crate) batch: Option<super::batch::BatchState>,
    /// The DeepEncoder tower, attached from the mmproj GGUF
    /// ([`GpuDeepseekOcr::attach_vision`]) - None serves text-only.
    pub(crate) vision: Option<super::vision::DeepEncoder>,
    /// Views the tower has encoded since load - telemetry, and the gate's
    /// witness that a radix-resumed request skipped the tower work
    /// (the battery page runs 7 views cold, 1 on a block-aligned resume).
    pub mm_tower_views: u64,
    /// Persistent splice staging for `assemble_image_plane`, allocated by
    /// `attach_vision` at the tile budget's worst case: the per-image
    /// [crops|global|newline|separator] staging rows and the block-walk
    /// gather index. Slabs, not per-request allocs - a mid-serve cudaMalloc
    /// is a device-wide sync (see `encode::TowerWs`).
    pub(crate) mm_src: Option<cudarc::driver::CudaSlice<f32>>,
    pub(crate) mm_idx: Option<cudarc::driver::CudaSlice<u32>>,
    /// The stall-free chunked prefill queue: row plans advancing
    /// inside mixed ticks, image feature planes riding along for the splice.
    pub(crate) chunked: Vec<super::chunked::ChunkedPrefill>,
    /// Image prompts mid-encode under the ENCODER BUDGET - one tower call per
    /// tick; front entry only (per-slot, granite's WaveEncode made flat).
    pub(crate) enc: std::collections::VecDeque<super::chunked::OcrEnc>,
}

impl GpuDeepseekOcr {
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        exec.vram_load_gate(map.total_len(), "deepseek-ocr")
            .map_err(GpuModelError::WontFit)?;
        // Single-stream engine: cudarc's cross-stream event tracking is pure
        // overhead and blocks CUDA-graph capture. Must precede all allocs.
        exec.disable_event_tracking();

        let hp = Hparams::from_gguf(map)
            .map_err(|e| GpuModelError::Unsupported(format!("deepseek-ocr hparams: {e}")))?;

        let vfree = || {
            cudarc::driver::result::mem_get_info()
                .map(|(f, _)| f as u64)
                .unwrap_or(0)
        };
        let gb = |used: u64| used as f64 / 1e9;
        let v_start = vfree();

        // Token embeddings, Q8_0-resident (the only quant this family's
        // official conversion ships; a different file is a different election
        // and should fail loudly here, not degrade).
        let tok_embd = exec.upload_raw(map, "token_embd.weight")?;
        if tok_embd.ty != GgmlType::Q8_0 {
            return Err(GpuModelError::Unsupported(format!(
                "deepseek-ocr: token_embd quant {:?} - only the Q8_0 election is served",
                tok_embd.ty
            )));
        }
        if tok_embd.dims[1] != hp.n_vocab {
            return Err(GpuModelError::Unsupported(format!(
                "deepseek-ocr: token_embd vocab {} != header {}",
                tok_embd.dims[1], hp.n_vocab
            )));
        }
        let v_embd = vfree();
        tracing::info!(
            "deepseek-ocr VRAM  token embeddings          {:>6.2} GB",
            gb(v_start.saturating_sub(v_embd))
        );

        let mut layers = Vec::with_capacity(hp.n_layer);
        let (mut attn_bytes, mut expert_bytes, mut dense_bytes) = (0u64, 0u64, 0u64);
        for i in 0..hp.n_layer {
            let dt = |name: &str| exec.upload(map, &format!("blk.{i}.{name}"));
            let qt = |name: &str| exec.load_quantw(map, &format!("blk.{i}.{name}"));

            let v0 = vfree();
            let ffn = if !hp.is_moe_layer(i) {
                let l = DsFfn::Dense {
                    gate: qt("ffn_gate.weight")?,
                    up: qt("ffn_up.weight")?,
                    down: qt("ffn_down.weight")?,
                };
                dense_bytes += v0.saturating_sub(vfree());
                l
            } else {
                let gate_exps = exec.repack_q8(map, &format!("blk.{i}.ffn_gate_exps.weight"))?;
                let up_exps = exec.repack_q8(map, &format!("blk.{i}.ffn_up_exps.weight"))?;
                let down_exps = exec.repack_q8(map, &format!("blk.{i}.ffn_down_exps.weight"))?;
                expert_bytes += v0.saturating_sub(vfree());

                let v1 = vfree();
                let moe = DsMoe {
                    router_w: dt("ffn_gate_inp.weight")?,
                    gate_exps,
                    up_exps,
                    down_exps,
                    shexp_gate: qt("ffn_gate_shexp.weight")?,
                    shexp_up: qt("ffn_up_shexp.weight")?,
                    shexp_down: qt("ffn_down_shexp.weight")?,
                };
                // The router plane's width is the expert count; a mismatch
                // means the header lied about the routing geometry.
                if moe.router_w.dims != [hp.n_embd, hp.n_expert] {
                    return Err(GpuModelError::Unsupported(format!(
                        "deepseek-ocr blk.{i}: router {:?} != [{}, {}]",
                        moe.router_w.dims, hp.n_embd, hp.n_expert
                    )));
                }
                // Shared experts arrive pre-merged: the plane must carry the
                // merged width, or the graph runs the wrong dense MLP.
                if moe.shexp_gate.dims()[1] != hp.shexp_ff() {
                    return Err(GpuModelError::Unsupported(format!(
                        "deepseek-ocr blk.{i}: shexp width {} != merged {}",
                        moe.shexp_gate.dims()[1],
                        hp.shexp_ff()
                    )));
                }
                dense_bytes += v1.saturating_sub(vfree());
                DsFfn::Moe(moe)
            };

            let va = vfree();
            let layer = DsLayer {
                attn_norm: dt("attn_norm.weight")?,
                wq: qt("attn_q.weight")?,
                wk: qt("attn_k.weight")?,
                wv: qt("attn_v.weight")?,
                wo: qt("attn_output.weight")?,
                ffn_norm: dt("ffn_norm.weight")?,
                ffn,
            };
            attn_bytes += va.saturating_sub(vfree());

            // G=1 means every attention plane is square [embd, embd]; a file
            // where k/v are narrower is a GQA sibling this graph does not
            // serve, and it must not load as if it did.
            for (name, w) in [("q", &layer.wq), ("k", &layer.wk), ("v", &layer.wv)] {
                if w.dims()
                    != [
                        hp.n_embd,
                        hp.n_head_kv * hp.head_dim.max(hp.n_embd / hp.n_head),
                    ]
                    && w.dims() != [hp.n_embd, hp.n_embd]
                {
                    return Err(GpuModelError::Unsupported(format!(
                        "deepseek-ocr blk.{i}: attn_{name} {:?} is not the G=1 square plane",
                        w.dims()
                    )));
                }
            }
            layers.push(layer);
        }
        tracing::info!(
            "deepseek-ocr VRAM  attention (q/k/v/o)       {:>6.2} GB",
            gb(attn_bytes)
        );
        tracing::info!(
            "deepseek-ocr VRAM  routed experts ({}e × {})  {:>6.2} GB",
            hp.n_expert,
            hp.n_layer - hp.first_k_dense,
            gb(expert_bytes)
        );
        tracing::info!(
            "deepseek-ocr VRAM  shexp + dense + routers   {:>6.2} GB",
            gb(dense_bytes)
        );

        let vh = vfree();
        let output_norm = exec.upload(map, "output_norm.weight")?;
        let lm_head = exec.load_quantw(map, "output.weight")?;
        if lm_head.dims() != [hp.n_embd, hp.n_vocab] {
            return Err(GpuModelError::Unsupported(format!(
                "deepseek-ocr: lm_head {:?} != [{}, {}]",
                lm_head.dims(),
                hp.n_embd,
                hp.n_vocab
            )));
        }
        tracing::info!(
            "deepseek-ocr VRAM  lm_head + final norm      {:>6.2} GB",
            gb(vh.saturating_sub(vfree()))
        );

        // Published resident-weight line - the pool's used counter, not the
        // vfree() bracket the per-section logs use (that one counts
        // context/modules/cuBLAS as weights and does not reproduce).
        let weights_bytes = exec
            .settled_mem_used()
            .unwrap_or_else(|| v_start.saturating_sub(vfree()));
        tracing::info!(
            "deepseek-ocr VRAM  = decoder resident total  {:>6.2} GB  \
             ({} layers, {} experts top-{}, R-SWA {:?})",
            gb(weights_bytes),
            hp.n_layer,
            hp.n_expert,
            hp.n_expert_used,
            hp.rswa_window,
        );

        Ok(Self {
            exec,
            hp,
            tok_embd,
            layers,
            output_norm,
            lm_head,
            max_ctx: max_ctx.min(hp.n_ctx_train),
            weights_bytes,
            // KV stays f16 - the exception to the KV8 default (2026-10-04):
            // the model is G=1 and the pack has no G=1 fp8 prefill tile, so an
            // fp8 cache falls to the tiled/decode-class walk (batch.rs wmma_pf),
            // correct but slower, with no fp8 quality evidence.
            kv_dtype: KvDtype::Fp16,
            decode: None,
            scratch: None,
            batch: None,
            vision: None,
            mm_tower_views: 0,
            mm_src: None,
            mm_idx: None,
            chunked: Vec::new(),
            enc: std::collections::VecDeque::new(),
        })
    }

    pub fn set_kv_dtype(&mut self, dtype: KvDtype) {
        self.kv_dtype = dtype;
    }

    /// KV rows one sequence pins, forever: `prefill + min(new, ring)` - the
    /// family's whole point, and what the fit estimate must price instead of
    /// `prefill + max_new`.
    pub fn kv_rows(&self, prefill: usize, max_new: usize) -> usize {
        self.hp.kv_rows(prefill, max_new)
    }
}
