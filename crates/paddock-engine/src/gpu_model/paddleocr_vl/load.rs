//! PaddleOCR-VL decoder weights - loader-first bring-up against the official
//! `PaddleOCR-VL-1.6-GGUF.gguf`, the deepseek-ocr/whisper
//! discipline: every plane resident with an honest VRAM ledger before any
//! forward exists.
//!
//! Ingest election (recorded here because the quant-election memory demands
//! it happen before code): the official file ships every matmul plane as
//! BF16 and the oracle byte-verified all of them against the reference
//! safetensors (manifest_dec.json `gguf_same_weights`, 8/8 True) - so the
//! planes are served VERBATIM through the gemma4 `Plane` seam
//! (`bf16_gemv`/`bf16_gemm`), never narrowed to f16 and never requantized.
//! That keeps the parity chain on literally the reference's weights end to
//! end. At 0.3B the decoder is ~0.9 GB resident; a Q8_0 bandwidth lane is a
//! later decision, and the per-tensor `plane()` dispatcher below already
//! eats such a file without code change (the seam is the TENSOR, per
//! gemma4/load.rs's doctrine note).

use std::sync::Arc;

use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;

use super::vision::VisionModel;
use super::{Hparams, forward};
use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype, QuantTensor};
use crate::gpu_model::gemma4::Plane;
use crate::gpu_model::gpt_oss::GpuModelError;

/// One ERNIE decoder layer: two RMSNorm vectors and seven matmul planes.
/// No biases anywhere (config `use_bias: false`), no q/k norms, no gates -
/// the plain Llama shape with the decoupled head_dim being the only twist.
pub(crate) struct DecLayer {
    pub attn_norm: DeviceTensor,
    pub ffn_norm: DeviceTensor,
    pub wq: Plane,
    pub wk: Plane,
    pub wv: Plane,
    pub wo: Plane,
    pub gate: Plane,
    pub up: Plane,
    pub down: Plane,
}

pub struct GpuPaddleOcrVl {
    pub(crate) exec: Arc<GpuExecutor>,
    pub hp: Hparams,
    pub max_ctx: usize,
    pub(crate) kv_dtype: KvDtype,
    /// BF16 verbatim; `embed_gather_plane` picks its kernel off the class.
    pub(crate) tok_embd: QuantTensor,
    pub(crate) layers: Vec<DecLayer>,
    pub(crate) output_norm: DeviceTensor,
    /// UNTIED (the 1.6 config flipped tie_word_embeddings off) - the file
    /// always carries a real `output.weight`, so no tied fallback branch.
    pub(crate) lm_head: Plane,
    pub vision: Option<VisionModel>,
    pub(crate) decode: Option<forward::DecodeState>,
    pub(crate) scratch: Option<forward::Scratch>,
    /// The batched serving lane (paged pool + radix + scratch).
    /// None until `enable_batch`; the serial spine stays the parity lane.
    pub(crate) batch: Option<super::batch::BatchState>,
    /// Stall-free chunked-prefill queue (mixed ticks; chunked.rs).
    pub(crate) chunked: Vec<super::chunked::ChunkedPrefill>,
    /// Image prompts mid-encode under the encoder budget (chunked.rs).
    pub(crate) enc: std::collections::VecDeque<super::chunked::PoEnc>,
    pub weights_bytes: u64,
}

impl GpuPaddleOcrVl {
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        exec.vram_load_gate(map.total_len(), "paddleocr-vl")
            .map_err(GpuModelError::WontFit)?;
        // Single-stream engine: cudarc's cross-stream event tracking is pure
        // overhead and blocks CUDA-graph capture. Must precede all allocs.
        exec.disable_event_tracking();

        let hp = Hparams::from_gguf(map)
            .map_err(|e| GpuModelError::Unsupported(format!("paddleocr-vl hparams: {e}")))?;

        let vfree = || {
            cudarc::driver::result::mem_get_info()
                .map(|(f, _)| f as u64)
                .unwrap_or(0)
        };
        let v_start = vfree();

        // Per-TENSOR class dispatch (gemma4's rule): today's official file is
        // all-BF16, but a mixed or Q8_0 file resolves per plane, not per model.
        let plane = |name: &str| -> Result<Plane, crate::gpu::GpuError> {
            let (info, _) = map
                .tensor_bytes(name)
                .map_err(|e| crate::gpu::GpuError::Driver(e.to_string()))?;
            if info.ggml_type != GgmlType::Bf16 {
                return Ok(Plane::Q8(exec.repack_q8(map, name)?));
            }
            if !exec.has_bf16_dense() {
                return Err(crate::gpu::GpuError::MissingOp("bf16 dense plane lane"));
            }
            Ok(Plane::Bf16(exec.upload_raw(map, name)?))
        };
        // A plane whose dims don't match the declared geometry is a foreign
        // or reshuffled file - fail at the door with the name and both shapes.
        let expect = |p: Plane, name: &str, dims: [usize; 2]| -> Result<Plane, GpuModelError> {
            if p.dims() != dims {
                return Err(GpuModelError::Unsupported(format!(
                    "paddleocr-vl: {name} dims {:?}, wanted {dims:?}",
                    p.dims()
                )));
            }
            Ok(p)
        };

        let tok_embd = exec.upload_raw(map, "token_embd.weight")?;
        if tok_embd.dims != [hp.n_embd, hp.n_vocab] {
            return Err(GpuModelError::Unsupported(format!(
                "paddleocr-vl: token_embd dims {:?}, wanted [{}, {}]",
                tok_embd.dims, hp.n_embd, hp.n_vocab
            )));
        }

        let q_dim = hp.n_head * hp.head_dim; // 2048 - WIDER than the hidden 1024
        let kv_dim = hp.n_kv_heads * hp.head_dim; // 256
        let mut layers = Vec::with_capacity(hp.n_layer);
        for i in 0..hp.n_layer {
            let t = |name: &str| format!("blk.{i}.{name}");
            layers.push(DecLayer {
                attn_norm: exec.upload(map, &t("attn_norm.weight"))?,
                ffn_norm: exec.upload(map, &t("ffn_norm.weight"))?,
                wq: expect(plane(&t("attn_q.weight"))?, "attn_q", [hp.n_embd, q_dim])?,
                wk: expect(plane(&t("attn_k.weight"))?, "attn_k", [hp.n_embd, kv_dim])?,
                wv: expect(plane(&t("attn_v.weight"))?, "attn_v", [hp.n_embd, kv_dim])?,
                wo: expect(
                    plane(&t("attn_output.weight"))?,
                    "attn_output",
                    [q_dim, hp.n_embd],
                )?,
                gate: expect(
                    plane(&t("ffn_gate.weight"))?,
                    "ffn_gate",
                    [hp.n_embd, hp.n_ff],
                )?,
                up: expect(plane(&t("ffn_up.weight"))?, "ffn_up", [hp.n_embd, hp.n_ff])?,
                down: expect(
                    plane(&t("ffn_down.weight"))?,
                    "ffn_down",
                    [hp.n_ff, hp.n_embd],
                )?,
            });
        }

        let output_norm = exec.upload(map, "output_norm.weight")?;
        let lm_head = expect(plane("output.weight")?, "output", [hp.n_embd, hp.n_vocab])?;

        // Published resident-weight line - the pool's used counter, not the
        // vfree() bracket (that one counts context/modules/cuBLAS
        // as weights and does not reproduce between loads).
        let weights_bytes = exec
            .settled_mem_used()
            .unwrap_or_else(|| v_start.saturating_sub(vfree()));
        tracing::info!(
            "paddleocr-vl decoder resident: {:.2} GB ({} layers, vocab {})",
            weights_bytes as f64 / 1e9,
            hp.n_layer,
            hp.n_vocab
        );

        Ok(Self {
            exec,
            hp,
            max_ctx,
            // KV stays f16 - the exception to the KV8 default (2026-10-04):
            // this family's WMMA prefill gate is f16-only (batch.rs wmma_pf),
            // so an fp8 cache walks the scalar tile, correct but slower, and
            // there is no fp8 quality evidence for it yet. The pack's G=8 fp8
            // tile could take it - widen the gate and measure before flipping.
            kv_dtype: KvDtype::Fp16,
            tok_embd,
            layers,
            output_norm,
            lm_head,
            vision: None,
            decode: None,
            scratch: None,
            batch: None,
            chunked: Vec::new(),
            enc: std::collections::VecDeque::new(),
            weights_bytes,
        })
    }

    /// Load the vision tower + projector from the companion mmproj GGUF
    /// (the earlier `VisionModel`), refusing a pair whose widths disagree.
    pub fn attach_vision(&mut self, map: &MappedGguf) -> Result<(), GpuModelError> {
        let vm = VisionModel::load(Arc::clone(&self.exec), map)?;
        if vm.llm_embd() != self.hp.n_embd {
            return Err(GpuModelError::Unsupported(format!(
                "paddleocr-vl: projector emits {} per token, decoder hidden is {}",
                vm.llm_embd(),
                self.hp.n_embd
            )));
        }
        self.weights_bytes += vm.weight_bytes() as u64;
        self.vision = Some(vm);
        // the tower's workspace, once, for the widest pass serving can form:
        // TOWER_PASS_ROWS, or one maximum-size picture (four patch rows per
        // token at the published ceiling) alone - in the ledger before any
        // plan reads its grant, never grown after
        let max_picture = self
            .vision_budget_impl()
            .map_or(0, |b| 4 * b.max_tokens as usize);
        let rows = super::vision::TOWER_PASS_ROWS.max(max_picture);
        self.vision
            .as_mut()
            .expect("attached above")
            .size_workspace(rows)
            .map_err(GpuModelError::from)?;
        Ok(())
    }

    /// Select the KV cache element type (default [`KvDtype::Fp16`],
    /// greedy-exact). Drops decode AND batch state so the caches re-allocate
    /// at the new element size; call before serving.
    pub fn set_kv_dtype(&mut self, dtype: KvDtype) {
        self.kv_dtype = dtype;
        self.decode = None;
        self.scratch = None;
        self.batch = None;
    }
}
