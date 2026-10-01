//! Apple GPU resources and command submission. Shared storage does not remove
//! synchronization: host access is allowed only before submit or after completion.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::*;
use paddock_engine::backend::{Backend, BackendInfo};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

mod residency;
use residency::ResidentSet;

pub(crate) const SHADER_SOURCE: &str = concat!(
    include_str!("../../../packs/metal/granite.metal"),
    "\n",
    include_str!("../../../packs/metal/linear.metal"),
    "\n",
    include_str!("../../../packs/metal/attention.metal"),
    "\n",
    include_str!("../../../packs/metal/granite_attention64.metal"),
    "\n",
    include_str!("../../../packs/metal/granite_prefill.metal"),
    "\n",
    include_str!("../../../packs/metal/kquant.metal"),
    include_str!("../../../packs/metal/qwen_projection.metal"),
    "\n",
    include_str!("../../../packs/metal/mlx_affine.metal"),
    include_str!("../../../packs/metal/splash.metal"),
    "\n",
    include_str!("../../../packs/metal/qwen4exp_affine.metal"),
    "\n",
    include_str!("../../../packs/metal/deltanet.metal"),
    "\n",
    include_str!("../../../packs/metal/qwen_attention.metal"),
    "\n",
    include_str!("../../../packs/metal/mlx_qwen.metal"),
    include_str!("../../../packs/metal/bonsai.metal"),
    include_str!("../../../packs/metal/ptq1.metal"),
    include_str!("../../../packs/metal/ternary_add.metal"),
    include_str!("../../../packs/metal/splash_attention.metal"),
    "\n",
    include_str!("../../../packs/metal/qwen4exp_mlx.metal"),
    "\n",
    include_str!("../../../packs/metal/spec.metal"),
    "\n",
    include_str!("../../../packs/metal/dflash.metal"),
    include_str!("../../../packs/metal/splash_draft.metal"),
    include_str!("../../../packs/metal/splash_draft_attention.metal"),
    "\n",
    include_str!("../../../packs/metal/vision.metal"),
    "\n",
    include_str!("../../../packs/metal/gemma4.metal"),
    "\n",
    include_str!("../../../packs/metal/gemma4_vision.metal"),
    "\n",
    include_str!("../../../packs/metal/muse.metal"),
    "\n",
    include_str!("../../../packs/metal/muse_vision.metal"),
    "\n",
    include_str!("../../../packs/metal/gemma_mlx.metal"),
    include_str!("../../../packs/metal/llama_mlx.metal"),
    "\n",
    include_str!("../../../packs/metal/gemma_mlx_vision.metal"),
    "\n",
    include_str!("../../../packs/metal/granite_vision.metal"),
    "\n",
    include_str!("../../../packs/metal/qwen3_encoder.metal"),
    "\n",
    include_str!("../../../packs/metal/moe.metal"),
    "\n",
    include_str!("../../../packs/metal/gpt_oss.metal"),
    "\n",
    include_str!("../../../packs/metal/qwen_moe.metal"),
    include_str!("../../../packs/metal/gemma_moe.metal"),
    include_str!("../../../packs/metal/diffusion_gemma.metal"),
    include_str!("../../../packs/metal/laguna.metal"),
    include_str!("../../../packs/metal/nemotron.metal"),
    include_str!("../../../packs/metal/paddleocr.metal"),
    include_str!("../../../packs/metal/unlimited_ocr.metal"),
    include_str!("../../../packs/metal/unlimited_vision.metal"),
    include_str!("../../../packs/metal/qwen3_asr.metal"),
    "\n",
    include_str!("../../../packs/metal/qwen3_aligner.metal"),
    include_str!("../../../packs/metal/granite_speech.metal"),
    include_str!("../../../packs/metal/whisper.metal"),
    include_str!("../../../packs/metal/iquant_tables.metal"),
    include_str!("../../../packs/metal/iquant.metal"),
    include_str!("../../../packs/metal/qwen4exp_moe.metal"),
    include_str!("../../../packs/metal/qwen4exp.metal"),
    "\n",
    include_str!("../../../packs/metal/qwen4exp_qsa.metal"),
    include_str!("../../../packs/metal/qwen_image.metal"),
    include_str!("../../../packs/metal/laya.metal"),
    include_str!("../../../packs/metal/kumo.metal"),
);

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("Metal: {0}")]
    Device(String),
    #[error("Metal memory budget exceeded: {0}")]
    Memory(String),
    #[error("Metal model: {0}")]
    Model(String),
}

pub(crate) type Result<T> = std::result::Result<T, MetalError>;
type Obj<T> = Retained<ProtocolObject<T>>;

// A whole Flash Next batch can include 64 slot resets plus 48 layer walks.
// The old 2,048-dispatch profile ceiling could panic on a valid full walk.
// M5 limits each timestamp buffer to 32 KiB (4,096 eight-byte samples).
// Page the counters instead of enlarging one buffer past that device limit.
// Two samples per dispatch; normal command buffers allocate no counter pages.
const PROFILE_SAMPLES: usize = 4096;
const PROFILE_COUNTER_PAGES: usize = 4;

#[cfg(test)]
thread_local! {
    pub(crate) static ISOLATE_ENCODERS_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static LIMIT_PIPELINE_THREADS_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) struct Buffer {
    pub raw: Obj<dyn MTLBuffer>,
    ledger: Arc<AtomicU64>,
    residency: Option<Arc<ResidentSet>>,
    resident_group: usize,
}
// SAFETY: Metal resource ownership may move between host threads. Contents are
// accessed only through unsafe methods whose caller must synchronize GPU use;
// this wrapper is intentionally not Sync and never exposes a safe mutable view.
unsafe impl Send for Buffer {}
impl Buffer {
    pub fn len(&self) -> usize {
        self.raw.length()
    }
    // Callers own all submission for this buffer and wait before accessing it.
    pub unsafe fn write_u32(&self, data: &[u32]) {
        assert!(std::mem::size_of_val(data) <= self.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.raw.contents().as_ptr().cast(),
                data.len(),
            );
        }
    }
    pub unsafe fn read_f32(&self, offset: usize, count: usize) -> Vec<f32> {
        assert!((offset + count) * 4 <= self.len());
        unsafe {
            std::slice::from_raw_parts(
                self.raw.contents().as_ptr().cast::<f32>().add(offset),
                count,
            )
            .to_vec()
        }
    }
    pub unsafe fn read_u32(&self, count: usize) -> Vec<u32> {
        assert!(count <= self.len() / 4);
        unsafe {
            std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<u32>(), count).to_vec()
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(set) = &self.residency {
            set.remove(&self.raw, self.resident_group);
        }
        self.ledger.fetch_sub(self.len() as u64, Ordering::Relaxed);
    }
}

/// One physical GPU and its allocation ledger. Each model owns its execution
/// queue; the shared serving scheduler controls when work is submitted.
pub struct MetalDevice {
    raw: Obj<dyn MTLDevice>,
    queue: Obj<dyn MTLCommandQueue>,
    kernels: HashMap<&'static str, Obj<dyn MTLComputePipelineState>>,
    #[cfg(test)]
    limited_kernels: HashMap<&'static str, Obj<dyn MTLComputePipelineState>>,
    ledger: Arc<AtomicU64>,
    budget: u64,
    healthy: AtomicBool,
    profile: bool,
    tensor_accelerated: bool,
    residency: Option<Arc<ResidentSet>>,
}

impl MetalDevice {
    /// Driver/compiler changes can change recurrent arithmetic even when the
    /// checkpoint layout is unchanged. Never reuse those states across them.
    pub(crate) fn checkpoint_platform(&self) -> String {
        format!(
            "{}:{}",
            self.raw.name(),
            objc2_foundation::NSProcessInfo::processInfo().operatingSystemVersionString()
        )
    }

    /// Gather/scatter immutable checkpoint spans with the blit engine. All
    /// ranges are validated before encoding; completion fences host consumers.
    pub(crate) fn copy_regions(
        &self,
        copies: &[(&Buffer, usize, &Buffer, usize, usize)],
    ) -> Result<()> {
        if !self.healthy() {
            return Err(MetalError::Device("GPU is unhealthy".into()));
        }
        for &(src, from, dst, to, len) in copies {
            if from.checked_add(len).is_none_or(|end| end > src.len())
                || to.checked_add(len).is_none_or(|end| end > dst.len())
            {
                return Err(MetalError::Model(
                    "checkpoint copy range exceeds buffer".into(),
                ));
            }
        }
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| MetalError::Device("checkpoint command buffer".into()))?;
        let enc = cmd
            .blitCommandEncoder()
            .ok_or_else(|| MetalError::Device("checkpoint blit encoder".into()))?;
        for &(src, from, dst, to, len) in copies {
            if len != 0 {
                // SAFETY: both owned buffers outlive command completion; every
                // source/destination range was checked before encoding.
                unsafe {
                    enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &src.raw, from, &dst.raw, to, len,
                    );
                }
            }
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        crate::telemetry::completed(&cmd);
        if let Some(e) = cmd.error() {
            self.healthy.store(false, Ordering::Release);
            return Err(MetalError::Device(e.to_string()));
        }
        Ok(())
    }
    /// Read immutable weight planes directly into their final GPU allocation.
    /// The callback is I/O only and must fill every byte before submission.
    pub(crate) fn upload_with(
        &self,
        size: usize,
        read: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<Buffer> {
        let buffer = self.alloc(size)?;
        // SAFETY: this fresh shared allocation has no submitted users; the
        // callback borrows the entire bounded byte range exclusively.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(buffer.raw.contents().as_ptr().cast::<u8>(), size)
        };
        read(bytes)?;
        Ok(buffer)
    }
    /// Assemble storage planes with one allocation and byte copies only.
    pub(crate) fn upload_parts(&self, parts: &[&[u8]]) -> Result<Buffer> {
        let size = parts
            .iter()
            .try_fold(0usize, |n, p| n.checked_add(p.len()))
            .ok_or_else(|| MetalError::Memory("upload size overflow".into()))?;
        let buffer = self.alloc(size)?;
        let mut offset = 0;
        for part in parts {
            // SAFETY: the fresh allocation is exclusively owned and unsubmitted;
            // the checked total contains every disjoint destination range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    part.as_ptr(),
                    buffer.raw.contents().as_ptr().cast::<u8>().add(offset),
                    part.len(),
                );
            }
            offset += part.len();
        }
        Ok(buffer)
    }

    pub fn new(budget: Option<u64>) -> Result<Self> {
        objc2::rc::autoreleasepool(|_| Self::new_inner(budget, None))
    }

    /// Fixed-resident models can use an exact audited reservation instead of
    /// the generic 90% estimate. This never raises an explicit user grant or
    /// an OS limit, and leaves 2 GiB below Apple's recommendation for driver,
    /// compiler and transient allocations outside the native buffer ledger.
    pub(crate) fn new_planned(budget: Option<u64>, required: u64) -> Result<Self> {
        objc2::rc::autoreleasepool(|_| Self::new_inner(budget, Some(required)))
    }

    fn planned_budget(recommended: u64, budget: Option<u64>, required: u64) -> Result<u64> {
        let ceiling = recommended.saturating_sub(2 << 30);
        let grant = budget
            .unwrap_or_else(|| (recommended * 9 / 10).max(required))
            .min(ceiling);
        if required == 0 || required > grant {
            return Err(MetalError::Memory(format!(
                "fixed-resident plan needs {required} bytes; grant {grant}, Apple recommendation {recommended}, 2 GiB driver/transient reserve; no memory-limit override or offload fallback"
            )));
        }
        Ok(grant)
    }

    fn new_inner(budget: Option<u64>, required: Option<u64>) -> Result<Self> {
        let raw = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| MetalError::Device("no Apple GPU".into()))?;
        // MPP TensorOps also executes on the shader cores of every Metal 4
        // Apple Silicon GPU, Apple7 (M1) up. Apple10 adds neural acceleration,
        // not a different tensor API contract. Keep the unified-memory
        // requirement and the existing allocation limits. Apple7 is the floor
        // because it is the oldest family measured: M1 Max, 2026-09-20, whole
        // pack compiles + Granite same-weights greedy parity.
        if !raw.hasUnifiedMemory() || !raw.supportsFamily(MTLGPUFamily::Apple7) {
            return Err(MetalError::Device(
                "this Metal backend requires an Apple Silicon GPU with unified memory (Apple7 or newer: M1 and later)"
                    .into(),
            ));
        }
        if !raw.supportsFamily(MTLGPUFamily::Apple10) {
            // Pre-Apple10 parts all compile the PADDOCK_APPLE9 shader variant;
            // the name is where that path started, not the only family on it.
            tracing::warn!(
                "experimental pre-Apple10 Metal path (M1-M4): TensorOps runs on GPU shader cores; M5 performance qualifications do not apply"
            );
        }
        let recommended = raw.recommendedMaxWorkingSetSize();
        let budget = if let Some(required) = required {
            Self::planned_budget(recommended, budget, required)?
        } else {
            budget.unwrap_or(recommended * 9 / 10).min(recommended)
        };
        if budget == 0 {
            return Err(MetalError::Memory("zero budget".into()));
        }
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| MetalError::Device("cannot create command queue".into()))?;
        crate::telemetry::activate();
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
        // Permit contraction/reassociation, but preserve the infinities used
        // by masked softmax and empty split states. Same-weights parity gates
        // the resulting arithmetic; Fast mode would erase those invariants.
        opts.setMathMode(MTLMathMode::Relaxed);
        let lib = raw
            .newLibraryWithSource_options_error(
                &NSString::from_str(&format!(
                    "{}{}{}",
                    if raw.supportsFamily(MTLGPUFamily::Apple10) {
                        ""
                    } else {
                        "#define PADDOCK_APPLE9 1\n"
                    },
                    if cfg!(test) {
                        "#define PADDOCK_KERNEL_DIAGNOSTICS 1\n"
                    } else {
                        ""
                    },
                    SHADER_SOURCE
                )),
                Some(&opts),
            )
            .map_err(|e| MetalError::Device(e.to_string()))?;
        let mut kernels = HashMap::new();
        #[cfg(test)]
        let mut limited_kernels = HashMap::new();
        for name in [
            "kumo_kv",
            "kumo_mm",
            "kumo_mlp_out",
            "kumo_norm",
            "kumo_add_norm",
            "kumo_heads",
            "kumo_scale",
            "kumo_attention",
            "kumo_add",
            "kumo_fourier",
            "kumo_cell_weights",
            "kumo_cell_mm",
            "kumo_cell_bias",
            "kumo_copy_repeat",
            "kumo_cache_heads",
            "kumo_attention_tile32",
            "kumo_attention_tile64",
            "kumo_single_value",
            "kumo_rows",
            "kumo_unrows",
            "kumo_labels",
            "laya_mm",
            "laya_geglu",
            "laya_embed",
            "laya_norm",
            "laya_head_entry",
            "laya_qkv",
            "laya_attention12",
            "laya_attention16",
            "laya_gather",
            "laya_score",
            "laya_act",
            "qi_activation",
            "qi_vision_patches",
            "qi_vision_affine",
            "qi_vision_position",
            "qi_text_splice",
            "qi_text_mrope",
            "qi_round_bf",
            "qi_attention",
            "qi_attention_deep",
            "qi_mlx_residual",
            "qi_mlx_swiglu",
            "qi_mlx_gate_up",
            "qi_mlx_gated_add",
            "qi_mlx_head_rope",
            "qi_mlx_text_rope",
            "qi_text_attention",
            "qi_norm",
            "qi_gated_add",
            "qi_head_rope",
            "qi_store_half",
            "qi_prefix_copy",
            "qi_affine",
            "qi_bias",
            "qi_guidance",
            "qi_noise",
            "qi_vae_norm",
            "qi_im2row",
            "qi_dupup",
            "qi_avgdown",
            "qi_pixels",
            "qi_split3",
            "qi_transpose",
            "qi_softmax",
            "qi_f32_mm",
            "qi_swiglu",
            "qasr_decode",
            "gs_inject",
            "gs_fmm",
            "gs_attention",
            "gs_attention_check",
            "gs_qattention",
            "gs_split",
            "gs_residual",
            "gs_silu",
            "gs_glu",
            "gs_depthwise",
            "gs_softmax",
            "gs_windows",
            "gs_queries",
            "qalign_widen",
            "qalign_sinusoid",
            "qalign_inject",
            "qalign_argmax",
            "qalign_round",
            "qalign_project",
            "qalign_rms",
            "qalign_residual",
            "qalign_swiglu",
            "qalign_gelu",
            "qalign_gelu_bf",
            "qalign_conv_rows",
            "qalign_flatten",
            "qalign_ln",
            "qalign_position",
            "qalign_heads",
            "qalign_head_rope",
            "qalign_audio_attention",
            "qalign_text_attention",
            "qasr_extract",
            "qasr_conv_rows",
            "qasr_flatten",
            "qasr_half_mm",
            "wh_widen",
            "wh_mv1",
            "wh_mv2",
            "wh_mv4",
            "wh_mv8",
            "wh_mv16",
            "wh_mm",
            "wh_conv_rows",
            "wh_position",
            "wh_embed",
            "wh_split",
            "wh_attention",
            "wh_append",
            "wh_cross_store",
            "wh_decode",
            "wh_merge",
            "wh_align_probs",
            "wh_rules",
            "wh_pick",
            "qasr_position",
            "qasr_heads",
            "qasr_attention",
            "qasr_attention_check",
            "uocr_dense",
            "uocr_route",
            "uocr_gu_decode",
            "uocr_gu_grouped",
            "uocr_down_grouped",
            "uocr_fold",
            "uocr_rope",
            "uocr_store",
            "uocr_decode",
            "uov_patches",
            "uov_mm",
            "uov_mm_f32",
            "uov_resize_pos",
            "uov_add_pos",
            "uov_partition",
            "uov_unpartition",
            "uov_heads",
            "uov_relative",
            "uov_sam_attention",
            "uov_sam_check",
            "uov_clip_attention",
            "uov_activation",
            "uov_conv_rows",
            "uov_clip_embed",
            "uov_concat",
            "uov_assemble",
            "uov_separators",
            "uov_copy",
            "pocr_rope",
            "pocr_store",
            "pocr_patches",
            "pocr_patch_mm",
            "pocr_position",
            "pocr_heads",
            "pocr_gelu",
            "pocr_extract",
            "pocr_attention",
            "nemo_state_copy",
            "nemo_conv",
            "nemo_conv_commit",
            "nemo_scan",
            "nemo_dt_check",
            "nemo_ssd_prepare",
            "nemo_ssd_matrix",
            "nemo_ssd_delta",
            "nemo_ssd_states",
            "nemo_ssd_output",
            "nemo_gated_norm",
            "nemo_store",
            "nemo_attention_decode",
            "nemo_route",
            "nemo_up_decode",
            "nemo_down_decode",
            "nemo_up_grouped",
            "nemo_down_grouped",
            "nemo_relu2",
            "nemo_fold",
            "laguna_qnorm_rope",
            "laguna_dense_f32",
            "q4x_norm",
            "q4x_status",
            "q4x_select_rows",
            "q4x_hc_init",
            "q4x_scale_silu",
            "q4x_hc_mix",
            "q4x_hc_combine",
            "q4x_f32_mv",
            "q4x_inject_parallel",
            "q4x_q8_mm",
            "q4x_q8_mm64",
            "q4x_hc_down_split",
            "q4x_hc_down_reduce",
            "q4x_ple_gate",
            "q4x_ple_hash",
            "q4x_ple_conv",
            "q4x_ple_commit",
            "q4x_ple_tokens_commit",
            "q4x_ple_reset",
            "q4x_dn_gated_norm",
            "q4x_dn_reset",
            "q4s_bf16_mm",
            "q4s_norm_rope",
            "q4s_store",
            "q4s_pool",
            "q4s_ring_commit",
            "q4s_score",
            "q4s_select",
            "q4s_attention",
            "q4s_join_gate",
            "q4s_copy",
            "iq_mv20_1",
            "iq_mv20_4",
            "iq_mv20_8",
            "iq_mv21_1",
            "iq_mv21_4",
            "iq_mv21_8",
            "iq_mv22_1",
            "iq_mv22_4",
            "iq_mv22_8",
            "iq_mm20",
            "iq_mm21",
            "iq_mm22",
            "iq_prepared20",
            "iq_prepared21",
            "iq_prepared22",
            "iq_expert_mv20",
            "iq_expert_mv21",
            "iq_expert_mv22",
            "iq_expert_mm20",
            "iq_expert_mm21",
            "iq_expert_mm22",
            "iq_tiles512",
            "iq_gather",
            "iq_route512",
            "q4m_router_mm",
            "q4m_gu_mv21",
            "q4m_gu_mv22",
            "q4m_gu_mm21",
            "q4m_gu_mm22",
            "q4m_down_grouped",
            "q4m_silu",
            "q4m_fold",
            "laguna_store",
            "laguna_gate",
            "laguna_decode6",
            "laguna_decode8",
            "laguna_decode9",
            "laguna_prefill",
            "laguna_route",
            "laguna_gu_decode",
            "laguna_down_decode",
            "laguna_gu_grouped16",
            "laguna_gu_grouped32",
            "laguna_down_grouped16",
            "laguna_down_grouped32",
            "laguna_fold",
            "laguna_route_top10",
            "laguna_gu_decode_top10",
            "laguna_gu_grouped16_top10",
            "laguna_gu_grouped32_top10",
            "laguna_down_grouped16_top10",
            "laguna_down_grouped32_top10",
            "laguna_fold_top10",
            "qmoe_route",
            "qmoe_gu_decode",
            "qmoe_down_decode",
            "qmoe_gu_grouped16",
            "qmoe_gu_grouped32",
            "qmoe_down_grouped16",
            "qmoe_down_grouped32",
            "qmoe_gu_strict16",
            "qmoe_gu_strict32",
            "qmoe_down_strict16",
            "qmoe_down_strict32",
            "qmoe_swiglu",
            "qmoe_fold",
            "gmoe_head",
            "gmoe_route",
            "gmoe_gu_decode",
            "gmoe_gu_strict16",
            "gmoe_gu_strict32",
            "gmoe_geglu",
            "gmoe_fold",
            "gmoe_branches",
            "gmoe_prefill256",
            "gmoe_prefill512",
            "gmoe_image_prefill256",
            "gmoe_image_prefill512",
            "qwen_attention_decode_gqa8",
            "moe_route",
            "moe_align",
            "moe_tiles",
            "moe_gu_decode",
            "moe_down_decode",
            "moe_gu_grouped",
            "moe_down_grouped",
            "moe_gu_grouped32",
            "moe_down_grouped32",
            "moe_swiglu",
            "moe_fold",
            "oss_rope_store",
            "oss_bias_residual",
            "oss_attention_prefill",
            "oss_attention_decode",
            "oss_attention_check",
            "qwen3_head_rope",
            "qwen3_attention",
            "qwen3_pool",
            "qwen3_score",
            "mlx_small",
            "mlx_rms",
            "mlx_residual_rms",
            "mlx_rms_selected",
            "mlx_residual",
            "mlx_swiglu",
            "mlx_dn_conv",
            "mlx_dn_qk_norm",
            "mlx_dn_gates",
            "mlx_dn_recurrent",
            "mlx_dn_recurrent_quad",
            "mlx_dn_recurrent_packed",
            "mlx_dn_verify",
            "mlx_dn_verify_commit",
            "mlx_dn_gated_norm",
            "mlx_qnorm_rope",
            "mlx_knorm_store",
            "mlx_attn_gate",
            "mlx_attention_query",
            "mlx_attention_decode",
            "mlx_attention_verify",
            "mlx_attention_stable",
            "mlx_attention_prefill",
            "mlx_attention_prefill_direct",
            "mlx_attention_prefill_indexed",
            "mlx_attention_prefill_gqa",
            "mlx_attention_query_grouped",
            "mlx_attention_prefill_split",
            "splash_attention_prefill",
            "splash_attention_prefill64",
            "splash_attention_prefill_grouped",
            "splash_attention_query_grouped",
            "splash_attention_query_decode",
            "splash_attention_decode_grouped",
            "splash_attention_decode_join",
            "mlx_embed",
            "bonsai_embed",
            "ptq1_unpack",
            "ptq1_gate",
            "ptq1_embed",
            "ptq1_rotate_grouped",
            "ptq1_vectors1",
            "ptq1_vectors2",
            "ptq1_vectors3",
            "ptq1_vectors4",
            "ptq1_mm16",
            "ptq1_mm32",
            "ptq1_mm64",
            "bonsai_add_1",
            "bonsai_add_2",
            "bonsai_add_3",
            "bonsai_add_4",
            "bonsai_add_8",
            "ptq1_add_1",
            "ptq1_add_2",
            "ptq1_add_3",
            "ptq1_add_4",
            "ptq1_add_8",
            "bonsai_add_full_1",
            "bonsai_add_full_2",
            "bonsai_add_full_3",
            "bonsai_add_full_4",
            "ptq1_add_full_1",
            "ptq1_add_full_2",
            "ptq1_add_full_3",
            "ptq1_add_full_4",
            "ptq1_unroll_1",
            "ptq1_unroll_2",
            "ptq1_unroll_3",
            "ptq1_unroll_4",
            "ptq1_swar_1",
            "ptq1_triplet_check",
            "bonsai_rotate",
            "bonsai_a",
            "bonsai_mv",
            "bonsai_vectors1",
            "bonsai_vectors4",
            "bonsai_full1",
            "bonsai_full2",
            "bonsai_full3",
            "bonsai_full4",
            "bonsai_multi1",
            "bonsai_multi2",
            "bonsai_multi3",
            "bonsai_multi4",
            "bonsai_mm32",
            "bonsai_mm64",
            "bonsai_tile32x32x64",
            "bonsai_tile64x32x64",
            "bonsai_prefill32",
            "bonsai_prefill64",
            "bonsai_prefill16",
            "bonsai_rms",
            "bonsai_residual_rms",
            "bonsai_rms_selected",
            "bonsai_dn_qk_norm",
            "bonsai_dn_gated_norm",
            "bonsai_dn_recurrent",
            "bonsai_attention_decode",
            "bonsai_knorm_store",
            "bonsai_attention_prefill",
            "q4a_gather",
            "q4a_small",
            "q4a_mv",
            "q4a_mv4",
            "q4a_mv4_fast",
            "q4a_mv4_narrow",
            "q4a_mv4_fast2",
            "q4a_expert_gate_up_vector",
            "q4a_expert_gate_up_pair2",
            #[cfg(test)]
            "q4a_expert_gate_up_pair4",
            #[cfg(test)]
            "q4a_expert_gate_up_ordered",
            "q4a_expert_gate_up_ordered_pair2",
            "q4a_expert_gate_up_ordered_pair4",
            #[cfg(test)]
            "q4a_expert_down_vector4",
            #[cfg(test)]
            "q4a_expert_down_vector8",
            #[cfg(test)]
            "q4a_expert_down_vector16",
            "q4a_hc_down_vector",
            "q4a_hc_up_mix_vector",
            "q4a_mv8",
            "q4a_mv8_fast",
            "q4a_wide",
            "q4a_wide4_packed",
            "q4a_wide8_packed",
            "q4a_wide4_rows2",
            "q4a_wide4_rows4",
            "q4a_wide8_rows2",
            "q4a_wide8_rows4",
            "q4a_mm",
            "q4a_mm4_packed",
            "q4a_input",
            "q4a_mm4_device128",
            "q4a_mm4_device128_group32",
            "q4a_mm8_device128",
            "q4a_mm4_device64",
            "q4a_mm4_device64_group32",
            "q4a_mm4_device128_pad8",
            "q4a_mm4_device64_pad8",
            "q4a_mm4_device128_wide",
            "q4a_mm4_pair",
            "q4a_mm4_pair_wide",
            #[cfg(test)]
            "q4a_weight_pair_slab",
            #[cfg(test)]
            "q4a_mm4_pair_slab",
            "q4a_weight_slab",
            "q4a_mm4_slab",
            #[cfg(test)]
            "q4a_mm4_device128_pad16",
            #[cfg(test)]
            "q4a_mm4_device64_pad16",
            "q4a_mm8_device64",
            "q4a_mm8_packed",
            "q4a_mm4_reuse",
            "q4a_mm8_reuse",
            "q4a_mm_split",
            "q4a_mm4_split_device128_group32",
            "q4a_mm4_split_device64_group32",
            "q4a_mm4_split_device32_group32",
            "q4a_mm4_split_device128_pad8",
            "q4a_mm4_split_device64_pad8",
            "q4a_mm4_split_device32_pad8",
            "q4a_mm8_split_device128",
            "q4a_mm8_split_device64",
            "q4a_mm8_split_device128_pad8",
            "q4a_mm8_split_device64_pad8",
            "q4a_mm_join",
            "q4b_margin",
            "q4a_expert_mv",
            "q4a_expert_order",
            "q4a_expert_ordered",
            "q4a_expert4",
            "q4a_expert4_fast",
            "q4a_expert4_ordered",
            "q4a_expert4_fast_ordered",
            "q4a_expert4_pair",
            "q4a_expert4_fast_pair",
            "q4a_expert_mm",
            "q4a_expert_mm_wide",
            "q4a_expert_mm_tail",
            "q4a_expert_mm_group32",
            "q4a_expert_mm_group32_pad",
            "q4a_expert_mm_group32_packed",
            #[cfg(test)]
            "q4a_expert_mm_k128_n32",
            #[cfg(test)]
            "q4a_expert_mm_k128_n64",
            #[cfg(test)]
            "q4a_expert_mm_load4",
            #[cfg(test)]
            "q4a_expert_mm_load8",
            #[cfg(test)]
            "q4a_expert_mm_sg1",
            #[cfg(test)]
            "q4a_expert_mm_sg2",
            #[cfg(test)]
            "q4a_expert_mm_pad4",
            #[cfg(test)]
            "q4a_expert_mm_pad16",
            #[cfg(test)]
            "q4a_expert_mm_register",
            "q4b_hc_combine_norm",
            #[cfg(test)]
            "q4a_expert_mm_masked",
            #[cfg(test)]
            "q4a_expert_tiles64",
            #[cfg(test)]
            "q4a_expert_mm_rows64",
            "q4a_expert_offsets",
            #[cfg(test)]
            "q4a_expert_plan64",
            "q4a_expert_pack",
            #[cfg(test)]
            "q4a_expert_pack64",
            "q4a_expert_mm_direct",
            #[cfg(test)]
            "q4a_expert_mm_direct64",
            #[cfg(test)]
            "q4a_expert_gate_up_packed",
            #[cfg(test)]
            "q4a_expert_gate_up_dispatch",
            #[cfg(test)]
            "q4a_expert_swiglu_masked",
            "q4a_expert_vector_masked",
            "q4b_norm",
            "q4b_trace_pointwise",
            "q4b_trace_qk",
            "q4b_norm_rope",
            "q4b_index_split",
            "q4b_store",
            "q4b_cache_finite",
            "q4b_pool",
            "q4b_attention",
            "q4b_attention_contract",
            "q4b_attention_compact",
            "q4b_attention_local",
            "q4b_attention_local_pad",
            #[cfg(test)]
            "q4b_attention_local_wide",
            "q4b_attention_local_gather",
            "q4b_attention_local_vpad1",
            "q4b_attention_direct_runs",
            #[cfg(test)]
            "q4b_attention_direct_keys",
            #[cfg(test)]
            "q4b_attention_direct_values",
            #[cfg(test)]
            "q4b_attention_pair_pack",
            #[cfg(test)]
            "q4b_attention_pairs",
            "q4b_join_gate_compact",
            "q4a_ple_staged",
            "q4b_join_gate",
            "q4b_route",
            "q4b_fold",
            "q4b_dn_qk_norm",
            "q4b_dn_gated_norm",
            "q4b_scale_silu",
            "q4b_hc_mix",
            "q4b_hc_combine",
            "q4b_injection",
            "q4a_ple_gather",
            "q4b_ple_gate",
            "q4b_ple_conv",
            "mlx_input",
            "mlx_affine1",
            "splash_input",
            "splash_image_copy",
            "splash_affine8_compact",
            "splash_affine32_compact",
            "splash_affine64_compact",
            "splash_affine8_compactpipe",
            "splash_affine16_compact4",
            "splash_affine24_compact4",
            "splash_affine32_compact4",
            "splash_affine32_bf16",
            "splash_pair32_bf16",
            "splash_affine32_bf16_tail",
            "splash_pair32_bf16_tail",
            "splash_widen",
            "splash_gateup_bf16",
            "splash_pair8",
            "splash_pair16",
            "splash_pair24",
            "splash_pair32",
            "splash_pair64",
            "splash_gateup_reduce",
            "splash_affine_vector",
            "splash_reduce",
            "gmlx_centered_weights",
            "dg_project",
            "dg_project_f32",
            "dg_project_q4",
            "dg_project_q5",
            "dg_project_q6",
            "dg_project_q8",
            "dg_project_a4",
            "dg_project_a8",
            "dg_attention256",
            "dg_attention512",
            "dg_gguf_attention256",
            "dg_gguf_attention512",
            "dg_affine_vector",
            "dg_expert_vector",
            "dg_embed",
            "dg_soft_embed",
            "dg_soft_embed_q6",
            "dg_soft_embed_q8",
            "dg_soft_embed_a8",
            "dg_soft_fold",
            "dg_add_norm",
            "dg_zero",
            "dg_experts",
            "dg_experts_q4",
            "dg_experts_q5",
            "dg_experts_q8",
            "dg_experts_a4",
            "dg_experts_a8",
            "dg_sample",
            "dg_accept",
            "dg_labels",
            "dg_moe_head",
            "dg_moe_route",
            "dg_moe_geglu",
            "dg_moe_fold",
            "dg_moe_branches",
            "gmlx_round",
            "gmlx_embed",
            "llama_mlx_rope",
            "llama_mlx_prefill",
            "llama_mlx_decode",
            "llama_prefill_direct",
            "llama_decode",
            "gmlx_norm",
            "gmlx_sandwich",
            "gmlx_qnorm",
            "gmlx_store",
            "mmlx_qnorm",
            "mmlx_store",
            "gmlx_decode128",
            "gmlx_decode256",
            "gmlx_decode512",
            "gmlx_prefill128",
            "gmlx_prefill256",
            "gmlx_prefill512",
            "gmlx_image_prefill128",
            "gmlx_image_prefill256",
            "gmlx_image_prefill512",
            "gmlx_gate",
            "gmlx_geglu",
            "gmlx_softcap",
            "gmlx_bmm",
            "gmlx_vmm64",
            "gmlx_erfgelu",
            "gmlx_layer_norm",
            "gmlx_gv_patches",
            "gmlx_gv_position",
            "gmlx_gv_qkv",
            "gmlx_gv_pool",
            "gmlx_gv_attention",
            "gmlx_mv_patches",
            "gmlx_mv_position",
            "gmlx_mv_qkv",
            "gmlx_mv_attention",
            "mlx_affine_single",
            "mlx_affine_verify_single4",
            "mlx_affine_verify_single2",
            "mlx_affine_verify_single3",
            "mlx_affine_verify_narrow3",
            "mlx_affine_verify_narrow2",
            "mlx_affine_verify_narrow4",
            "mlx_affine_verify_half2",
            "mlx_affine_verify_half3",
            "mlx_affine_bias",
            "mlx_affine2",
            "mlx_affine3",
            "mlx_affine4",
            "mlx_affine5",
            "mlx_affine_stream2",
            "mlx_affine_stream3",
            "mlx_affine_stream4",
            "mlx_affine_stream5",
            "mlx_input_compact",
            "mlx_affine_stable1",
            "mlx_affine_stable2",
            "mlx_affine_stable3",
            "mlx_affine_stable4",
            "mlx_affine_stable5",
            "mlx_affine_compact2",
            "mlx_affine_compact3",
            "mlx_affine_compact4",
            "mlx_affine_compact5",
            "mlx_affine_tile32",
            "mlx_affine_tile64",
            "mlx_affine_prefill64",
            "mlx_affine_prefill_load32",
            "mlx_affine_prefill_load32_m32",
            "mlx_affine_prefill_rows256",
            "mlx_affine_prefill_store256",
            "mlx_affine_prefill_compact256",
            "mlx_affine_prefill_wide128",
            "mlx_affine_prefill_deep128",
            "mlx_affine_prefill_compact128",
            "mlx_swiglu_compact",
            "mlx_affine_parts",
            "mlx_affine_parts_padded",
            "mlx_affine_join",
            "grv_patches",
            "grv_position",
            "grv_half",
            "grv_heads",
            "grv_qattention",
            "grv_qattention_check",
            "grv_window",
            "grv_unwindow",
            "grv_pack",
            "grv_add",
            "mv_coeff",
            "mv_patches",
            "mv_position",
            "mv_permute",
            "mv_qkv",
            "mv_gelu",
            "mv_gelu_series_check",
            "mv_bmm64",
            "mv_attention",
            "mv_attention_check",
            "mv_shuffle",
            "muse_qnorm",
            "muse_q8_f32_16",
            "muse_q8_expand",
            "mv_bmm128",
            "muse_df_ktile128",
            "muse_df_ktile96",
            "muse_df_ktile64",
            "muse_q8_f32_32",
            "muse_q8_f32_64",
            "muse_kv_store",
            "muse_embedding_norm",
            "muse_sandwich",
            "muse_gate",
            "muse_swiglu",
            "muse_softcap",
            "muse_decode",
            "muse_decode_vector",
            "muse_decode_register",
            "muse_merge",
            "muse_prefill",
            "embed",
            "gv_coeff",
            "gv_resize_h",
            "gv_patches",
            "gv_patch_project",
            "gv_position",
            "gv_rms",
            "gv_qkv",
            "gv_pool",
            "gv_geglu_quick",
            "gv_attention",
            "gemma_image_prefill256",
            "gemma_image_prefill512",
            "gemma_qnorm",
            "gemma_kv_store",
            "gemma_sandwich",
            "gemma_geglu",
            "gemma_softcap",
            "gemma_decode256",
            "gemma_decode512",
            "gemma_merge",
            "gemma_merge_shared",
            "gemma_draft_advance",
            "gemma_verify8",
            "gemma_full5",
            "gemma_full6",
            "gemma_full7",
            "gemma_full8",
            "gemma_verify16",
            "gemma_f32_8",
            "gemma_f32_16",
            "gemma_f32_32",
            "gemma_staged_f32_32",
            "gemma_pair2",
            "gemma_pair3",
            "gemma_pair4",
            "gemma_multi_pair3",
            "gemma_multi_pair4",
            "gemma_verify_attn256",
            "gemma_prefill256",
            "gemma_prefill512",
            "vis_cast",
            "vis_finite",
            "vis_attention_check",
            "vis_inject",
            "vis_mm32",
            "vis_mm64",
            "vis_hmm32",
            "vis_hmm64",
            "vis_bmm32",
            "vis_bmm64",
            "vis_bmm_fast32",
            "vis_bmm_fast64",
            "vis_patches",
            "vis_position",
            "vis_ln",
            "vis_qkv",
            "vis_attention",
            "dn_verify",
            "dn_verify_commit",
            "dn_verify_conv_commit",
            "spec_copy",
            "spec_copy_words",
            "spec_argmax",
            "mtp_hidden",
            "mtp_concat",
            "mtp_publish",
            "mtp_advance",
            "df_tap",
            "splash_df_rms",
            "splash_df_qnorm",
            "splash_df_kstore",
            "splash_df_query",
            "splash_df_attention_grouped",
            "splash_df_attention_join",
            "df_conv",
            "df_qnorm",
            "df_kstore",
            "df_attention",
            "df_attention_muse",
            "df_top16",
            "df_top16_merge",
            "df_select",
            "dn_conv",
            "dn_qk_norm",
            "dn_gates",
            "dn_conv_commit",
            "dn_recurrent",
            "dn_chunk_dots",
            "dn_chunk_prepare",
            "dn_chunk_walk",
            "dn_chunk_dots_strict",
            "dn_chunk_prepare_strict",
            "dn_chunk_walk_strict",
            "dn_gated_norm",
            "dn_checkpoint",
            "qwen_qnorm_rope",
            "qwen_knorm_store",
            "qwen_attn_gate",
            "qwen_attention_decode",
            "qwen_attention_decode_gqa4",
            "qwen_attention_merge",
            "qwen_attention_prefill",
            "qwen_attention_prefill_split",
            "qwen_attention_prefill_strict",
            "qwen_attention_prefill_split_strict",
            "qwen_attention_prefill_join",
            "rms",
            "linear",
            "linear_q8",
            "linear_kquant1",
            "linear_kquant2",
            "linear_kquant3",
            "linear_kquant4",
            "linear_kquant_tail4",
            "qwen_pair1",
            "qwen_pair2",
            "qwen_pair3",
            "qwen_pair4",
            "qwen_multi_pair1",
            "qwen_multi_pair2",
            "qwen_multi_pair3",
            "qwen_multi_pair4",
            "linear_multi_kquant1",
            "linear_multi_kquant2",
            "linear_multi_kquant3",
            "linear_multi_kquant4",
            "linear_ktile32",
            "linear_ktile_split8",
            "linear_ktile_split32",
            "linear_ktile_split64",
            "linear_ktile_join",
            "linear_ktile8",
            "linear_ktile64",
            "linear_ktile96",
            "linear_ktile128",
            "linear_kexpand",
            "linear_kexpanded128",
            "linear_multi_ktile32",
            "linear_multi_ktile8",
            "linear_multi_ktile64",
            "linear_multi_ktile96",
            "linear_multi_ktile128",
            "linear_q8_r1",
            "linear_q8_r4",
            "linear_q8_h4",
            "linear_q8_h2",
            "linear_q8_h3",
            "linear_q8_full4",
            "linear_multi_q8_r1",
            "linear_multi_q8_r4",
            "linear_multi_q8_r2",
            "linear_multi_q8_r3",
            "linear_q8_r8",
            "linear_q8_r16",
            "linear_quant_tile32",
            "linear_quant_tile64",
            "linear_quant_tile128",
            "linear_quant_full32",
            "linear_quant_full64",
            "linear_quant_full128",
            "linear_multi_quant32",
            "linear_multi_quant64",
            "linear_multi_quant128",
            "linear_input",
            "linear_input_padded",
            "rms_selected",
            "residual_rms",
            "linear_mpp",
            "linear_prepare",
            "rope_store",
            "attention",
            "attention_merge",
            "attention_gqa4",
            "attention_gqa_merge",
            "attention_gqa5_64",
            "attention_gqa_merge64",
            "attention_check64",
            "attention_mpp",
            "attention_prefill",
            "attention_prefill_batched",
            "attention_prefill_batched64",
            "granite_attention_prefill128",
            "attention_query",
            "residual",
            "swiglu",
        ] {
            if matches!(
                name,
                "mlx_affine_prefill_store256"
                    | "mlx_affine_prefill_compact256"
                    | "mlx_affine_prefill_wide128"
                    | "mlx_affine_prefill_deep128"
                    | "mlx_affine_prefill_compact128"
                    | "mlx_dn_recurrent_packed"
                    | "mlx_attention_prefill_gqa"
            ) && !raw.supportsFamily(MTLGPUFamily::Apple10)
            {
                continue;
            }
            let fun = lib
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| MetalError::Device(format!("missing kernel {name}")))?;
            let pipeline = raw
                .newComputePipelineStateWithFunction_error(&fun)
                .map_err(|e| MetalError::Device(format!("{name}: {e}")))?;
            #[cfg(test)]
            if LIMIT_PIPELINE_THREADS_FOR_TEST.with(|v| v.get())
                && matches!(
                    name,
                    "q4a_mm4_device128_pad8"
                        | "q4a_mm4_device128_wide"
                        | "q4a_mm4_pair_wide"
                        | "q4a_mm4_slab"
                        | "q4a_mm8_device128"
                        | "q4a_mm4_split_device128_pad8"
                        | "q4a_mm8_split_device128_pad8"
                        | "q4a_expert_mm_group32_packed"
                        | "q4a_expert_mm_direct"
                )
            {
                let desc = MTLComputePipelineDescriptor::new();
                desc.setComputeFunction(Some(&fun));
                desc.setMaxTotalThreadsPerThreadgroup(128);
                let limited = raw
                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                        &desc,
                        MTLPipelineOption::None,
                        None,
                    )
                    .map_err(|e| MetalError::Device(format!("limited {name}: {e}")))?;
                eprintln!(
                    "FLASH_PIPELINE_LIMIT kernel={name} default_threads={} limited_threads={} default_shared={} limited_shared={}",
                    pipeline.maxTotalThreadsPerThreadgroup(),
                    limited.maxTotalThreadsPerThreadgroup(),
                    pipeline.staticThreadgroupMemoryLength(),
                    limited.staticThreadgroupMemoryLength(),
                );
                limited_kernels.insert(name, limited);
            }
            kernels.insert(name, pipeline);
        }
        let profile = paddock_models::dev_var_os!("PADDOCK_METAL_PROFILE").is_some();
        let residency = if required.is_some() {
            Some(Arc::new(ResidentSet::new(&raw, &queue)?))
        } else {
            None
        };
        tracing::info!(device = %raw.name(), budget_gib = budget as f64 / (1u64 << 30) as f64,
            "native Metal backend initialized");
        Ok(Self {
            tensor_accelerated: raw.supportsFamily(MTLGPUFamily::Apple10),
            raw,
            queue,
            kernels,
            #[cfg(test)]
            limited_kernels,
            ledger: Arc::new(AtomicU64::new(0)),
            budget,
            healthy: AtomicBool::new(true),
            profile,
            residency,
        })
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.ledger.load(Ordering::Relaxed)
    }

    pub(crate) fn tensor_accelerated(&self) -> bool {
        self.tensor_accelerated
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget
    }

    pub(crate) fn alloc(&self, size: usize) -> Result<Buffer> {
        if size == 0 || size as u64 > self.raw.maxBufferLength() as u64 {
            return Err(MetalError::Memory(format!("invalid buffer size {size}")));
        }
        self.ledger
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(size as u64)
                    .filter(|&next| next <= self.budget)
            })
            .map_err(|used| {
                MetalError::Memory(format!(
                    "{used} live + {size} requested > {} budget",
                    self.budget
                ))
            })?;
        let Some(raw) = self
            .raw
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
        else {
            self.ledger.fetch_sub(size as u64, Ordering::Relaxed);
            return Err(MetalError::Memory(format!(
                "allocation of {size} bytes failed"
            )));
        };
        let resident_group = if let Some(set) = &self.residency {
            match set.add(&raw) {
                Ok(group) => group,
                Err(error) => {
                    self.ledger.fetch_sub(size as u64, Ordering::Relaxed);
                    return Err(error);
                }
            }
        } else {
            0
        };
        Ok(Buffer {
            raw,
            ledger: self.ledger.clone(),
            residency: self.residency.clone(),
            resident_group,
        })
    }

    pub(crate) fn upload(&self, bytes: &[u8]) -> Result<Buffer> {
        let buffer = self.alloc(bytes.len())?;
        // SAFETY: the freshly allocated shared buffer has no in-flight users.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.raw.contents().as_ptr().cast(),
                bytes.len(),
            );
        }
        Ok(buffer)
    }

    pub(crate) fn begin(&self) -> Result<Commands<'_>> {
        if !self.healthy() {
            return Err(MetalError::Device(
                "previous GPU submission failed; reload the model".into(),
            ));
        }
        let counters = if self.profile {
            if !self
                .raw
                .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
            {
                return Err(MetalError::Device("stage counters unsupported".into()));
            }
            let sets = self
                .raw
                .counterSets()
                .ok_or_else(|| MetalError::Device("no counter sets".into()))?;
            let set = sets
                .iter()
                .find(|s| s.name().to_string().eq_ignore_ascii_case("timestamp"))
                .ok_or_else(|| {
                    MetalError::Device(format!(
                        "no timestamp counter set: {:?}",
                        sets.iter()
                            .map(|s| s.name().to_string())
                            .collect::<Vec<_>>()
                    ))
                })?;
            let desc = MTLCounterSampleBufferDescriptor::new();
            desc.setCounterSet(Some(&set));
            desc.setStorageMode(MTLStorageMode::Shared);
            unsafe {
                desc.setSampleCount(PROFILE_SAMPLES);
            }
            (0..PROFILE_COUNTER_PAGES)
                .map(|_| {
                    self.raw
                        .newCounterSampleBufferWithDescriptor_error(&desc)
                        .map_err(|e| MetalError::Device(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or_else(|| MetalError::Device("cannot create command buffer".into()))?;
        #[cfg(test)]
        let isolate = ISOLATE_ENCODERS_FOR_TEST.with(|v| v.get());
        #[cfg(not(test))]
        let isolate = false;
        let enc = if self.profile || isolate {
            None
        } else {
            Some(
                // Dispatch ordering is part of every backend's graph contract.
                // Concurrent encoding requires explicit dependency tracking;
                // make the serial election explicit instead of relying on a default.
                cmd.computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                    .ok_or_else(|| MetalError::Device("cannot create compute encoder".into()))?,
            )
        };
        Ok(Commands {
            device: self,
            cmd,
            enc,
            profile_names: std::cell::RefCell::new(Vec::new()),
            projection_workspace: None,
            independent_rows: false,
            projection_rows: None,
            packed_spans: None,
            affine_prefill_rows: None,
            counters,
        })
    }
}

impl Backend for MetalDevice {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "metal",
            device: self.raw.name().to_string(),
            memory_total: self.raw.recommendedMaxWorkingSetSize(),
        }
    }
    fn healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }
}

pub(crate) struct Commands<'a> {
    device: &'a MetalDevice,
    cmd: Obj<dyn MTLCommandBuffer>,
    enc: Option<Obj<dyn MTLComputeCommandEncoder>>,
    profile_names: std::cell::RefCell<Vec<String>>,
    counters: Vec<Obj<dyn MTLCounterSampleBuffer>>,
    projection_workspace: Option<&'a Buffer>,
    independent_rows: bool,
    projection_rows: Option<&'a [(usize, usize, usize)]>,
    packed_spans: Option<&'a [(usize, usize, crate::splash::Phase)]>,
    affine_prefill_rows: Option<usize>,
}

impl<'a> Commands<'a> {
    /// Fix affine prefill's BF16 partial tree independently of the physical
    /// admission slice. Only explicitly qualified model graphs opt in.
    pub(crate) fn with_affine_prefill_rows(mut self, rows: usize) -> Self {
        assert!(rows > 0);
        self.affine_prefill_rows = Some(rows);
        self
    }
    pub(crate) fn affine_prefill_rows(&self) -> Option<usize> {
        self.affine_prefill_rows
    }
    pub(crate) fn tensor_accelerated(&self) -> bool {
        self.device.tensor_accelerated()
    }
    /// Explicit packed-model (start, count, phase) roles. Cached prompt
    /// suffixes remain prefill; adding a decode rider cannot change their math.
    pub(crate) fn with_packed_spans(
        mut self,
        spans: &'a [(usize, usize, crate::splash::Phase)],
    ) -> Self {
        self.packed_spans = Some(spans);
        self
    }
    pub(crate) fn packed_spans(&self) -> Option<&[(usize, usize, crate::splash::Phase)]> {
        self.packed_spans
    }
    /// Model-owned, ledger-accounted storage reused between projections on
    /// this ordered encoder. No dispatch-time allocation or host readback.
    pub(crate) fn with_projection_workspace(mut self, workspace: &'a Buffer) -> Self {
        self.projection_workspace = Some(workspace);
        self
    }

    pub(crate) fn projection_workspace(&self) -> Option<&Buffer> {
        self.projection_workspace
    }

    /// Some quantized graphs require singleton-equivalent contractions for
    /// independent decode sequences. This is a numerical contract, not a
    /// customer tuning switch; projection implementations opt in explicitly.
    pub(crate) fn with_independent_rows(mut self, independent: bool) -> Self {
        self.independent_rows = independent;
        self
    }

    pub(crate) fn independent_rows(&self) -> bool {
        self.independent_rows
    }

    /// Contiguous (start, count, logical_count) spans for an opt-in affine
    /// graph. Admission may slice a logical prompt chunk without changing
    /// its contraction. Decode is always logical_count=1.
    pub(crate) fn with_projection_rows(mut self, rows: &'a [(usize, usize, usize)]) -> Self {
        self.projection_rows = Some(rows);
        self
    }

    pub(crate) fn projection_rows(&self) -> Option<&[(usize, usize, usize)]> {
        self.projection_rows
    }

    #[cfg(test)]
    pub(crate) fn trace_row(self, label: &str, buffer: &Buffer, count: usize) -> Result<Self> {
        use std::hash::{Hash, Hasher};
        let (device, workspace, independent, spans, packed, affine_rows) = (
            self.device,
            self.projection_workspace,
            self.independent_rows,
            self.projection_rows,
            self.packed_spans,
            self.affine_prefill_rows,
        );
        self.finish()?;
        // Test-only serialization of completed GPU values, never an oracle
        // or host inference path. Fencing makes this unsuitable for timing.
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        for value in unsafe { buffer.read_f32(0, count) } {
            value.to_bits().hash(&mut hash);
        }
        eprintln!("MLX_ROW_TRACE {label} {:016x}", hash.finish());
        let mut next = device.begin()?;
        next.projection_workspace = workspace;
        next.independent_rows = independent;
        next.projection_rows = spans;
        next.packed_spans = packed;
        next.affine_prefill_rows = affine_rows;
        Ok(next)
    }
}

/// A retained, submitted command. Dropping a cancelled result still fences its
/// resources before their allocation-ledger entries can be released.
pub(crate) struct Completion(Obj<dyn MTLCommandBuffer>, AtomicBool);
impl Completion {
    fn record(&self) {
        if !self.1.swap(true, Ordering::Relaxed) {
            crate::telemetry::completed(&self.0);
        }
    }
    pub fn ready(&self) -> bool {
        matches!(
            self.0.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        )
    }
    pub fn wait(&self) -> Result<()> {
        self.0.waitUntilCompleted();
        self.record();
        self.0
            .error()
            .map_or(Ok(()), |e| Err(MetalError::Device(e.to_string())))
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        self.0.waitUntilCompleted();
        self.record();
    }
}
impl Commands<'_> {
    pub(crate) fn submit(self) -> Result<Completion> {
        if !self.counters.is_empty() {
            return Err(MetalError::Device(
                "async encoder dispatch profiling is not supported".into(),
            ));
        }
        if let Some(enc) = &self.enc {
            enc.endEncoding();
        }
        self.cmd.commit();
        Ok(Completion(self.cmd, AtomicBool::new(false)))
    }
    /// Parameters are fixed-width words, with floats passed as IEEE bit patterns.
    /// All buffers are retained by the default command-buffer resource policy.
    pub fn dispatch(
        &self,
        name: &str,
        buffers: &[&Buffer],
        params: &[u32],
        groups: [usize; 3],
        threads: usize,
    ) {
        self.dispatch_inner::<false>(name, buffers, &[], params, groups, threads);
    }

    /// Byte offsets select bounded views without copying activations or
    /// allocating alias buffers. Kernel callers still validate view extents.
    pub(crate) fn dispatch_at(
        &self,
        name: &str,
        buffers: &[&Buffer],
        offsets: &[usize],
        params: &[u32],
        groups: [usize; 3],
        threads: usize,
    ) {
        assert_eq!(buffers.len(), offsets.len());
        assert!(
            buffers
                .iter()
                .zip(offsets)
                .all(|(buffer, offset)| *offset < buffer.len() && offset.is_multiple_of(4))
        );
        self.dispatch_inner::<true>(name, buffers, offsets, params, groups, threads);
    }

    fn dispatch_inner<const OFFSET: bool>(
        &self,
        name: &str,
        buffers: &[&Buffer],
        offsets: &[usize],
        params: &[u32],
        groups: [usize; 3],
        threads: usize,
    ) {
        let pipeline = &self.device.kernels[name];
        #[cfg(test)]
        let pipeline = if LIMIT_PIPELINE_THREADS_FOR_TEST.with(|v| v.get()) {
            self.device.limited_kernels.get(name).unwrap_or(pipeline)
        } else {
            pipeline
        };
        let sample = self.profile_names.borrow().len() * 2;
        let measured = if !self.counters.is_empty() {
            let counters = self
                .counters
                .get(sample / PROFILE_SAMPLES)
                .expect("dispatch profile counter pages exhausted");
            let sample = sample % PROFILE_SAMPLES;
            // Apple GPUs expose stage counters, not dispatch counters. Profiling
            // uses one encoder per dispatch; never compare its wall time to the
            // production single-encoder path.
            let desc = MTLComputePassDescriptor::new();
            desc.setDispatchType(MTLDispatchType::Serial);
            let attachment = unsafe { desc.sampleBufferAttachments().objectAtIndexedSubscript(0) };
            attachment.setSampleBuffer(Some(counters));
            unsafe {
                attachment.setStartOfEncoderSampleIndex(sample);
                attachment.setEndOfEncoderSampleIndex(sample + 1);
            }
            // Shape attribution is control metadata only. Keep the generic
            // kernel name for every unrelated backend/profiler consumer.
            let label = if name.starts_with("dg_project")
                || name.starts_with("dg_experts")
                || name.starts_with("q4a_mv")
                || name.starts_with("q4a_mm")
                || name.starts_with("splash_affine")
                || name.starts_with("bonsai_vectors")
                || name.starts_with("bonsai_full")
                || name.starts_with("bonsai_mm")
                || name.starts_with("bonsai_tile")
                || name.starts_with("bonsai_prefill")
                || name.starts_with("ptq1_vectors")
                || name.starts_with("ptq1_mm")
                || name.starts_with("ptq1_add_")
                || name.starts_with("ptq1_swar_")
                || name.starts_with("ptq1_unroll_")
                || name.starts_with("bonsai_add_")
                || (name.starts_with("q4a_expert")
                    && !matches!(
                        name,
                        "q4a_expert_order" | "q4a_expert_offsets" | "q4a_expert_plan64"
                    )) {
                format!("{name}[k={},n={},m={}]", params[0], params[1], params[2])
            } else if name.starts_with("mlx_affine_prefill")
                || name.starts_with("mlx_affine_tile")
                || name.starts_with("bonsai_multi")
            {
                format!(
                    "{name}[k={},n={}+{}+{},m={}]",
                    params[0], params[1], params[2], params[3], params[4]
                )
            } else {
                name.to_owned()
            };
            self.profile_names.borrow_mut().push(label);
            Some(
                self.cmd
                    .computeCommandEncoderWithDescriptor(&desc)
                    .expect("profile encoder"),
            )
        } else if self.enc.is_none() {
            // Counter-free test comparator for encoder-granularity costs.
            // Keep the same command buffer, serial ordering and final fence.
            Some(
                self.cmd
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                    .expect("isolated compute encoder"),
            )
        } else {
            None
        };
        let enc = measured
            .as_ref()
            .or(self.enc.as_ref())
            .expect("compute encoder");
        assert!(threads <= pipeline.maxTotalThreadsPerThreadgroup());
        enc.setComputePipelineState(pipeline);
        // SAFETY: call sites validate shapes and storage before encoding; parameter
        // blocks are copied by Metal, and buffers outlive command completion.
        unsafe {
            for (i, buffer) in buffers.iter().enumerate() {
                enc.setBuffer_offset_atIndex(
                    Some(&buffer.raw),
                    if OFFSET { offsets[i] } else { 0 },
                    i,
                );
            }
            enc.setBytes_length_atIndex(
                NonNull::new(params.as_ptr().cast_mut().cast())
                    .expect("slice pointers are non-null"),
                std::mem::size_of_val(params),
                buffers.len(),
            );
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: groups[0],
                    height: groups[1],
                    depth: groups[2],
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
        }
        // Apple's memoryBarrier APIs are ignored on serial compute encoders.
        // Serial dispatch ordering provides the dependency contract above;
        // do not pay an Objective-C call for a no-op after every dispatch.
        if measured.is_some() {
            enc.endEncoding();
        }
    }
    pub fn finish(self) -> Result<f64> {
        self.finish_with_host_work(|_| ())
    }

    /// Submit before bounded, CPU-only work, then take the same completion,
    /// error, telemetry and profiling path as finish. The closure must not
    /// access any buffer referenced by this command. No detached work remains.
    pub(crate) fn finish_with_host_work(self, work: impl FnOnce(&dyn Fn() -> bool)) -> Result<f64> {
        if let Some(enc) = &self.enc {
            enc.endEncoding();
        }
        self.cmd.commit();
        {
            struct Fence<'a>(&'a ProtocolObject<dyn MTLCommandBuffer>);
            impl Drop for Fence<'_> {
                fn drop(&mut self) {
                    self.0.waitUntilCompleted();
                }
            }
            // Even a host-side panic must not unwind past in-flight resources.
            let _fence = Fence(&self.cmd);
            work(&|| {
                matches!(
                    self.cmd.status(),
                    MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                )
            });
        }
        crate::telemetry::completed(&self.cmd);
        if let Some(e) = self.cmd.error() {
            self.device.healthy.store(false, Ordering::Relaxed);
            return Err(MetalError::Device(e.to_string()));
        }
        if !self.counters.is_empty() {
            let names = self.profile_names.borrow();
            let mut timestamps = Vec::with_capacity(names.len() * 2);
            for (page, counters) in self.counters.iter().enumerate() {
                let count = (names.len() * 2)
                    .saturating_sub(page * PROFILE_SAMPLES)
                    .min(PROFILE_SAMPLES);
                if count == 0 {
                    break;
                }
                let data = unsafe { counters.resolveCounterRange(NSRange::new(0, count)) }
                    .ok_or_else(|| MetalError::Device("counter resolution failed".into()))?;
                let bytes = unsafe { data.as_bytes_unchecked() };
                if bytes.len() != count * 8 {
                    return Err(MetalError::Device(
                        "incomplete timestamp counter page".into(),
                    ));
                }
                timestamps.extend(
                    bytes.chunks_exact(8).map(|b| {
                        u64::from_ne_bytes(b.try_into().expect("eight-byte counter chunk"))
                    }),
                );
            }
            let mut totals = std::collections::BTreeMap::<&str, (u64, usize)>::new();
            for (i, name) in names.iter().enumerate() {
                let entry = totals.entry(name).or_default();
                entry.0 += timestamps[2 * i + 1].saturating_sub(timestamps[2 * i]);
                entry.1 += 1;
            }
            eprintln!("Metal dispatch counters (nanoseconds, count; instrumented): {totals:?}");
        }
        Ok(self.cmd.GPUEndTime() - self.cmd.GPUStartTime())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_host_work_fences_gpu_writes_even_on_unwind() {
        let d = MetalDevice::new(Some(16 << 20)).unwrap();
        let out = d.alloc(4096).unwrap();
        for panic_host in [false, true] {
            unsafe { out.write_u32(&vec![u32::MAX; 1024]) };
            let called = std::cell::Cell::new(false);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let cmd = d.begin().unwrap();
                cmd.dispatch("dg_zero", &[&out], &[1024], [4, 1, 1], 256);
                cmd.finish_with_host_work(|_| {
                    called.set(true);
                    assert!(!panic_host, "injected bounded host work failure");
                })
                .unwrap();
            }));
            assert_eq!(result.is_err(), panic_host);
            assert!(called.get());
            assert_eq!(unsafe { out.read_u32(1024) }, vec![0; 1024]);
        }
    }

    #[test]
    fn telemetry_async_completion_records_once_even_after_repeated_wait_and_drop() {
        // Counters deliberately include every queue in a runner. Other GPU
        // unit tests run concurrently, so an exact +1 assertion belongs in
        // its own process, not behind a lock that those tests do not share.
        // Preserve the exact assertion instead of tolerating duplicate counts.
        const CHILD: &str = "PADDOCK_TEST_COMPLETION_TELEMETRY_CHILD";
        if std::env::var(CHILD).as_deref() != Ok("1") {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "device::tests::telemetry_async_completion_records_once_even_after_repeated_wait_and_drop",
                    "--exact",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .env_remove("PADDOCK_METAL_TELEMETRY")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated telemetry check failed: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let raw = MTLCreateSystemDefaultDevice().expect("Apple GPU");
        let queue = raw.newCommandQueue().unwrap();
        let buffer = raw
            .newBufferWithLength_options(1 << 20, MTLResourceOptions::StorageModeShared)
            .unwrap();
        crate::telemetry::activate();
        let before = crate::telemetry::telemetry_snapshot().unwrap();
        let cmd = queue.commandBuffer().unwrap();
        let enc = cmd.blitCommandEncoder().unwrap();
        enc.fillBuffer_range_value(&buffer, NSRange::new(0, 1 << 20), 7);
        enc.endEncoding();
        cmd.commit();
        let completion = Completion(cmd, AtomicBool::new(false));
        completion.wait().unwrap();
        completion.wait().unwrap();
        drop(completion);
        let after = crate::telemetry::telemetry_snapshot().unwrap();
        assert_eq!(
            after["completed_commands"].as_u64().unwrap(),
            before["completed_commands"].as_u64().unwrap() + 1
        );
        assert!(
            after["gpu_seconds_total"].as_f64().unwrap()
                > before["gpu_seconds_total"].as_f64().unwrap()
        );
        assert!(after["allocated_bytes"].as_u64().unwrap() >= 1 << 20);
        assert!(after["last_command_ms"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn bf16_gpu_rounding_preserves_ties_range_and_special_values() {
        let device = MetalDevice::new(Some(32 << 20)).unwrap();
        // Literal IEEE fixtures, not a CPU inference/reference implementation.
        // Exercise the actual MLX operation boundary, including the Apple9
        // compiler workaround. Both tie directions and signs must survive.
        let cases: [(u32, u32); 14] = [
            (0x00000000, 0x00000000),
            (0x80000000, 0x80000000),
            (0x3f807fff, 0x3f800000),
            (0x3f808000, 0x3f800000),
            (0x3f808001, 0x3f810000),
            (0x3f818000, 0x3f820000),
            (0xbf808000, 0xbf800000),
            (0xbf818000, 0xbf820000),
            (0x47800000, 0x47800000),
            (0xc7800000, 0xc7800000),
            (0x7f7fffff, 0x7f800000),
            (0xff7fffff, 0xff800000),
            (0x7f800000, 0x7f800000),
            (0xff800000, 0xff800000),
        ];
        let input: Vec<u32> = cases
            .iter()
            .map(|&(v, _)| v)
            .chain([0x7fc00001, 0xffc00001])
            .collect();
        let x = device
            .upload(
                &input
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let zero = device
            .upload(
                &input
                    .iter()
                    .flat_map(|v| (v & 0x80000000).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "mlx_residual",
            &[&x, &zero],
            &[input.len() as u32],
            [1, 1, 1],
            32,
        );
        cmd.finish().unwrap();
        let actual = unsafe { x.read_u32(input.len()) };
        for (i, &(value, expected)) in cases.iter().enumerate() {
            assert_eq!(actual[i], expected, "BF16 rounding of {value:08x}");
        }
        assert!(
            actual[cases.len()..]
                .iter()
                .all(|&v| f32::from_bits(v).is_nan())
        );
    }

    #[test]
    fn fixed_resident_budget_preserves_explicit_limits_and_headroom() {
        let recommended = 115_448_725_504;
        let required = 112_000_000_000;
        assert_eq!(
            MetalDevice::planned_budget(recommended, None, required).unwrap(),
            required
        );
        assert!(MetalDevice::planned_budget(recommended, Some(100_000_000_000), required).is_err());
        assert!(MetalDevice::planned_budget(recommended, None, recommended - (1 << 30)).is_err());
        assert!(MetalDevice::planned_budget(1 << 30, None, 1).is_err());
        assert!(MetalDevice::planned_budget(recommended, None, 0).is_err());
        assert_eq!(
            MetalDevice::planned_budget(recommended, Some(u64::MAX), required).unwrap(),
            recommended - (2 << 30)
        );
    }

    #[test]
    fn planned_residency_tracks_buffer_lifetimes() {
        let d = MetalDevice::new_planned(Some(32 << 20), 16 << 20).unwrap();
        let set = d.residency.as_ref().unwrap().clone();
        assert_eq!(set.allocations(), 0);
        let first = d.alloc(4096).unwrap();
        let second = d.alloc(1 << 20).unwrap();
        assert_eq!(set.allocations(), 2);
        drop(first);
        assert_eq!(set.allocations(), 1);
        // A buffer may outlive its device wrapper; the request must survive
        // until that buffer is released, without retaining the GPU resource.
        drop(d);
        assert_eq!(set.allocations(), 1);
        drop(second);
        assert_eq!(set.allocations(), 0);
    }

    #[test]
    fn flash_next_profile_pages_cross_the_metal_counter_buffer_limit() {
        let mut d = MetalDevice::new(Some(32 << 20)).unwrap();
        d.profile = true;
        let flags = d.upload(&0u32.to_le_bytes()).unwrap();
        let bad = d.upload(&0u32.to_le_bytes()).unwrap();
        let cmd = d.begin().unwrap();
        assert_eq!(cmd.counters.len(), PROFILE_COUNTER_PAGES);
        // Exercise every page and its final pair, not just the first 2,048
        // dispatches. No model math or user-global environment changes.
        for _ in 0..PROFILE_SAMPLES * PROFILE_COUNTER_PAGES / 2 {
            cmd.dispatch("q4x_status", &[&flags, &bad], &[1, 0, 2], [1, 1, 1], 256);
        }
        assert!(cmd.finish().unwrap().is_finite());
        assert_eq!(unsafe { bad.read_u32(1) }, [0]);
    }

    #[test]
    fn gpt_oss_pipelines_fit_threadgroup_memory() {
        let device = MetalDevice::new(Some(32 << 20)).expect("M5 GPU required");
        let limit = device.raw.maxThreadgroupMemoryLength();
        for (name, pipeline) in &device.kernels {
            if name.starts_with("moe_")
                || name.starts_with("oss_")
                || name.starts_with("qmoe_")
                || name.starts_with("gmoe_")
                || name.starts_with("laguna_")
                || name.starts_with("nemo_")
            {
                // TensorOps/compiler staging can double the source-declared
                // arrays. Check compiled requirements, not a hand calculation.
                let used = pipeline.staticThreadgroupMemoryLength();
                eprintln!("{name}: {used}/{limit} threadgroup bytes");
                assert!(used <= limit, "{name}: {used} bytes exceeds {limit}");
            }
        }
    }

    #[test]
    fn mlx_packed_kernels_fit_threadgroup_memory() {
        let device = MetalDevice::new(Some(32 << 20)).unwrap();
        if !device.tensor_accelerated() {
            return;
        }
        for name in ["mlx_attention_prefill_gqa", "mlx_dn_recurrent_packed"] {
            let pipeline = &device.kernels[name];
            assert!(
                pipeline.staticThreadgroupMemoryLength() <= device.raw.maxThreadgroupMemoryLength(),
                "{name}"
            );
        }
    }

    #[test]
    fn gemma_attention_pipelines_fit_threadgroup_memory() {
        let device = MetalDevice::new(Some(32 << 20)).expect("M5 GPU required");
        let limit = device.raw.maxThreadgroupMemoryLength();
        for (name, pipeline) in &device.kernels {
            if name.starts_with("gemma_prefill")
                || name.starts_with("gemma_image_prefill")
                || name.starts_with("gemma_verify_attn")
            {
                let used = pipeline.staticThreadgroupMemoryLength();
                eprintln!("{name}: {used}/{limit} threadgroup bytes");
                assert!(used <= limit, "{name}: {used} exceeds {limit}");
            }
        }
    }

    #[test]
    fn unlimited_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        let limit = d.raw.maxThreadgroupMemoryLength();
        for (name, p) in &d.kernels {
            if name.starts_with("uocr_") || name.starts_with("uov_") {
                let used = p.staticThreadgroupMemoryLength();
                eprintln!("{name}: {used}/{limit} threadgroup bytes");
                assert!(used <= limit, "{name}: {used} exceeds {limit}");
            }
        }
    }

    #[test]
    fn mlx_wide_staging_fits_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        if !d.tensor_accelerated() {
            return;
        }
        for name in [
            "mlx_affine_prefill_wide128",
            "mlx_affine_prefill_deep128",
            "mlx_affine_prefill_compact128",
        ] {
            eprintln!(
                "{name}: {}/{} threadgroup bytes",
                d.kernels[name].staticThreadgroupMemoryLength(),
                d.raw.maxThreadgroupMemoryLength()
            );
            assert!(
                d.kernels[name].staticThreadgroupMemoryLength()
                    <= d.raw.maxThreadgroupMemoryLength(),
                "{name}"
            );
        }
    }

    #[test]
    fn qasr_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        for name in [
            "qasr_attention",
            "qasr_decode",
            "qasr_half_mm",
            "vis_bmm32",
            "laguna_dense_f32",
            "laguna_prefill",
        ] {
            let used = d.kernels[name].staticThreadgroupMemoryLength();
            eprintln!(
                "{name}: {used}/{} bytes",
                d.raw.maxThreadgroupMemoryLength()
            );
            assert!(used <= d.raw.maxThreadgroupMemoryLength());
        }
    }

    #[test]
    fn whisper_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        let limit = d.raw.maxThreadgroupMemoryLength();
        for (name, pipeline) in &d.kernels {
            if name.starts_with("wh_") {
                let used = pipeline.staticThreadgroupMemoryLength();
                eprintln!("{name}: {used}/{limit} threadgroup bytes");
                assert!(used <= limit, "{name}: {used} exceeds {limit}");
            }
        }
    }

    #[test]
    fn iquant_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        let limit = d.raw.maxThreadgroupMemoryLength();
        eprintln!(
            "recommended working set={} max buffer={}",
            d.raw.recommendedMaxWorkingSetSize(),
            d.raw.maxBufferLength()
        );
        let mut count = 0;
        for (name, pipeline) in &d.kernels {
            if name.starts_with("iq_") {
                let used = pipeline.staticThreadgroupMemoryLength();
                eprintln!("{name}: {used}/{limit} threadgroup bytes");
                assert!(used <= limit, "{name}: {used} exceeds {limit}");
                count += 1;
            }
        }
        assert_eq!(count, 24);
    }

    #[test]
    fn flash_next_residual_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        let mut count = 0;
        for (name, pipeline) in &d.kernels {
            if name.starts_with("q4x_") {
                let used = pipeline.staticThreadgroupMemoryLength();
                assert!(used <= d.raw.maxThreadgroupMemoryLength(), "{name}: {used}");
                count += 1;
            }
        }
        assert_eq!(count, 21);
        for name in [
            "dn_conv",
            "dn_qk_norm",
            "dn_gates",
            "dn_recurrent",
            "dn_chunk_dots_strict",
            "dn_chunk_prepare_strict",
            "dn_chunk_walk_strict",
            "dn_conv_commit",
            "dn_checkpoint",
            "laguna_dense_f32",
            "gemma_full5",
            "gemma_full6",
            "gemma_full7",
            "gemma_full8",
        ] {
            let used = d.kernels[name].staticThreadgroupMemoryLength();
            assert!(used <= d.raw.maxThreadgroupMemoryLength(), "{name}: {used}");
        }
    }

    #[test]
    fn flash_next_qsa_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        let mut count = 0;
        for (name, pipeline) in &d.kernels {
            if name.starts_with("q4s_") {
                let used = pipeline.staticThreadgroupMemoryLength();
                eprintln!(
                    "{name}: {used}/{} threadgroup bytes",
                    d.raw.maxThreadgroupMemoryLength()
                );
                assert!(used <= d.raw.maxThreadgroupMemoryLength(), "{name}: {used}");
                count += 1;
            }
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn flash_next_moe_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        let mut count = 0;
        for (name, p) in &d.kernels {
            if name.starts_with("q4m_") {
                let used = p.staticThreadgroupMemoryLength();
                eprintln!(
                    "{name}: {used}/{} threadgroup bytes",
                    d.raw.maxThreadgroupMemoryLength()
                );
                assert!(used <= d.raw.maxThreadgroupMemoryLength(), "{name}: {used}");
                count += 1;
            }
        }
        assert_eq!(count, 8);
    }

    #[test]
    fn qalign_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        for name in [
            "qalign_text_attention",
            "qalign_audio_attention",
            "qalign_project",
            "qalign_argmax",
        ] {
            let used = d.kernels[name].staticThreadgroupMemoryLength();
            eprintln!("aligner {name} threadgroup bytes {used}");
            assert!(used <= d.raw.maxThreadgroupMemoryLength());
        }
    }

    #[test]
    fn gs_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        for name in [
            "gs_attention",
            "gs_fmm",
            "gs_qattention",
            "gs_depthwise",
            "qasr_half_mm",
        ] {
            let used = d.kernels[name].staticThreadgroupMemoryLength();
            eprintln!(
                "Granite Speech {name}: {used}/{}",
                d.raw.maxThreadgroupMemoryLength()
            );
            assert!(used <= d.raw.maxThreadgroupMemoryLength());
        }
    }

    #[test]
    fn paddleocr_pipelines_fit_threadgroup_memory() {
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        for name in [
            "pocr_attention",
            "pocr_patch_mm",
            "vis_bmm32",
            "laguna_prefill",
            "laguna_decode8",
        ] {
            let used = d.kernels[name].staticThreadgroupMemoryLength();
            eprintln!(
                "{name}: {used}/{} bytes",
                d.raw.maxThreadgroupMemoryLength()
            );
            assert!(used <= d.raw.maxThreadgroupMemoryLength());
        }
    }

    #[test]
    fn native_mlx_pipelines_fit_threadgroup_memory() {
        let device = MetalDevice::new(Some(32 << 20)).expect("M5 GPU required");
        let limit = device.raw.maxThreadgroupMemoryLength();
        for (name, pipeline) in &device.kernels {
            if name.starts_with("mlx_") {
                let used = pipeline.staticThreadgroupMemoryLength();
                assert!(used <= limit, "{name}: {used} bytes exceeds {limit}");
            }
        }
    }

    #[test]
    fn gguf_qwen_prefill_pipelines_fit_threadgroup_memory() {
        let device = MetalDevice::new(Some(32 << 20)).expect("M5 GPU required");
        let limit = device.raw.maxThreadgroupMemoryLength();
        for name in [
            "dn_chunk_dots",
            "dn_chunk_prepare",
            "dn_chunk_walk",
            "dn_chunk_dots_strict",
            "dn_chunk_prepare_strict",
            "dn_chunk_walk_strict",
            "dn_recurrent",
            "qwen_attention_prefill",
            "qwen_attention_prefill_split",
            "qwen_attention_prefill_strict",
            "qwen_attention_prefill_split_strict",
        ] {
            let used = device.kernels[name].staticThreadgroupMemoryLength();
            eprintln!("{name}: {used}/{limit} threadgroup bytes");
            assert!(used <= limit, "{name}: {used} bytes exceeds {limit}");
        }
        for name in [
            "qwen_attention_decode",
            "qwen_attention_decode_gqa4",
            "qwen_attention_decode_gqa8",
        ] {
            let used = device.kernels[name].staticThreadgroupMemoryLength();
            assert!(used <= limit, "{name}: {used} bytes exceeds {limit}");
        }
    }

    // Both sides execute on the GPU. Dyadic fixtures make FP16 tile staging
    // exact so this checks layouts, ragged bounds and K accumulation, not a
    // CPU model or a loose tolerance masking an indexing error.
    #[test]
    fn tensorops_matches_vector_q8_on_ragged_tiles() {
        let device = MetalDevice::new(Some(32 << 20)).expect("M5 GPU required");
        for (m, n, k) in [
            (2usize, 35usize, 96usize),
            (3, 35, 96),
            (4, 35, 96),
            (17, 35, 96),
            (32, 64, 128),
            (64, 128, 256),
            (128, 128, 256),
            (129, 67, 160),
        ] {
            let mut weights = Vec::new();
            for block in 0..n * k / 32 {
                weights.extend_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
                weights.extend((0..32).map(|i| (((block * 13 + i * 7) % 31) as i8 - 15) as u8));
            }
            let xs: Vec<u8> = (0..m * k)
                .flat_map(|i| (((i * 11 % 29) as f32 - 14.0) / 16.0).to_le_bytes())
                .collect();
            let w = device.upload(&weights).unwrap();
            let x = device.upload(&xs).unwrap();
            let a = device.alloc(m * n * 4).unwrap();
            let b = device
                .upload(
                    &(0..(m + 128) * n)
                        .flat_map(|_| f32::NAN.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let wf = device.alloc(k * n * 2).unwrap();
            // Poison guard storage: a static tensor slice must never turn a
            // ragged K/M tail into real reads, even if adjacent bytes happen
            // to be zero in another allocation. 0*NaN must not reach outputs.
            let poison: Vec<u8> = (0..k * m + 32768)
                .flat_map(|_| half::f16::NAN.to_le_bytes())
                .collect();
            let xf = device.upload(&poison).unwrap();
            let cmd = device.begin().unwrap();
            let p = [k as u32, n as u32, m as u32, 8, 1.0f32.to_bits()];
            cmd.dispatch("linear", &[&w, &x, &a], &p, [n.div_ceil(4), m, 1], 128);
            cmd.dispatch(
                "linear_prepare",
                &[&w, &x, &wf, &xf],
                &p,
                [(k * n.max(m)).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "linear_mpp",
                &[&wf, &xf, &b],
                &p,
                [n.div_ceil(64), m.div_ceil(32), 1],
                128,
            );
            cmd.finish().unwrap();
            // SAFETY: the GPU submission has completed.
            let (vector, tiled) = unsafe { (a.read_f32(0, m * n), b.read_f32(0, m * n)) };
            let error = vector
                .iter()
                .zip(tiled)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 1e-4, "{m}x{n}x{k}: max error {error}");
            let padded = k.div_ceil(128) * 128 * m.div_ceil(128) * 128;
            let xp = device
                .upload(
                    &(0..padded + 32768)
                        .flat_map(|_| half::f16::NAN.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                "linear_input_padded",
                &[&x, &xp],
                &p,
                [padded.div_ceil(256), 1, 1],
                256,
            );
            cmd.finish().unwrap();
            for (tile, kernel) in [
                (32, "linear_quant_tile32"),
                (64, "linear_quant_tile64"),
                (128, "linear_quant_tile128"),
            ] {
                let cmd = device.begin().unwrap();
                cmd.dispatch(
                    kernel,
                    &[&w, &xp, &b],
                    &p,
                    [n.div_ceil(64), m.div_ceil(tile), 1],
                    128,
                );
                cmd.finish().unwrap();
                assert_eq!(
                    vector,
                    unsafe { b.read_f32(0, m * n) },
                    "quantized tiled MPP {tile}"
                );
                for count in [2, 3] {
                    let c = device.alloc(b.len()).unwrap();
                    let d = device.alloc(b.len()).unwrap();
                    let cmd = device.begin().unwrap();
                    cmd.dispatch(
                        &format!("linear_multi_quant{tile}"),
                        &[&w, &w, &w, &xp, &b, &c, &d],
                        &[
                            k as u32,
                            n as u32,
                            n as u32,
                            if count == 3 { n as u32 } else { 0 },
                            m as u32,
                            8,
                            8,
                            8,
                        ],
                        [n.div_ceil(64) * count, m.div_ceil(tile), 1],
                        128,
                    );
                    cmd.finish().unwrap();
                    for out in [&b, &c].into_iter().chain((count == 3).then_some(&d)) {
                        assert_eq!(
                            vector,
                            unsafe { out.read_f32(0, m * n) },
                            "multi quant {tile}/{count}"
                        );
                    }
                }
                if m.is_multiple_of(tile) && n.is_multiple_of(64) && k.is_multiple_of(128) {
                    let cmd = device.begin().unwrap();
                    cmd.dispatch(
                        &format!("linear_quant_full{tile}"),
                        &[&w, &xp, &b],
                        &p,
                        [n / 64, m / tile, 1],
                        128,
                    );
                    cmd.finish().unwrap();
                    assert_eq!(
                        vector,
                        unsafe { b.read_f32(0, m * n) },
                        "full Q8 tile {tile}"
                    );
                }
                assert!(
                    unsafe { b.read_f32(m * n, 128 * n) }
                        .iter()
                        .all(|v| v.is_nan()),
                    "projection output tail"
                );
            }
            if (2..=4).contains(&m) {
                let cmd = device.begin().unwrap();
                cmd.dispatch(
                    &format!("linear_q8_h{m}"),
                    &[&w, &xf, &b],
                    &p,
                    [n.div_ceil(4), 1, 1],
                    128,
                );
                cmd.finish().unwrap();
                assert_eq!(
                    vector,
                    unsafe { b.read_f32(0, m * n) },
                    "full four-row F16-input projection"
                );
                for count in [2, 3] {
                    let c = device.alloc(b.len()).unwrap();
                    let d = device.alloc(b.len()).unwrap();
                    let cmd = device.begin().unwrap();
                    cmd.dispatch(
                        &format!("linear_multi_q8_r{m}"),
                        &[&w, &w, &w, &xf, &b, &c, &d],
                        &[
                            k as u32,
                            n as u32,
                            n as u32,
                            if count == 3 { n as u32 } else { 0 },
                            m as u32,
                        ],
                        [(n * count).div_ceil(4), 1, 1],
                        128,
                    );
                    cmd.finish().unwrap();
                    assert_eq!(
                        vector,
                        unsafe { b.read_f32(0, m * n) },
                        "multi projection 0"
                    );
                    assert_eq!(
                        vector,
                        unsafe { c.read_f32(0, m * n) },
                        "multi projection 1"
                    );
                    if count == 3 {
                        assert_eq!(
                            vector,
                            unsafe { d.read_f32(0, m * n) },
                            "multi projection 2"
                        );
                    }
                }
            }
            let cmd = device.begin().unwrap();
            cmd.dispatch("linear_q8", &[&w, &x, &b], &p, [n.div_ceil(4), m, 1], 128);
            cmd.finish().unwrap();
            let packed = unsafe { b.read_f32(0, m * n) };
            assert_eq!(vector, packed, "packed Q8 vector layout");
            for (tile, kernel) in [
                (1, "linear_q8_r1"),
                (4, "linear_q8_r4"),
                (8, "linear_q8_r8"),
                (16, "linear_q8_r16"),
            ] {
                let cmd = device.begin().unwrap();
                cmd.dispatch(
                    kernel,
                    &[&w, &x, &b],
                    &p,
                    [n.div_ceil(4), m.div_ceil(tile), 1],
                    128,
                );
                cmd.finish().unwrap();
                assert_eq!(
                    vector,
                    unsafe { b.read_f32(0, m * n) },
                    "shared-weight tile {tile}"
                );
            }
        }
        assert_eq!(device.allocated_bytes(), 0);
    }

    #[test]
    fn split_attention_matches_unsplit_with_ragged_shared_pages() {
        let device = MetalDevice::new(Some(8 << 20)).expect("M5 GPU required");
        let heads = 8usize;
        let kv_heads = 2usize;
        let stride = 19usize;
        let rows = 2usize;
        let q: Vec<u8> = (0..rows * heads * 128)
            .flat_map(|i| (((i * 17 % 31) as f32 - 15.0) / 16.0).to_le_bytes())
            .collect();
        let keys: Vec<u8> = (0..stride * 16 * kv_heads * 128)
            .flat_map(|i| half::f16::from_f32(((i * 7 % 43) as f32 - 21.0) / 16.0).to_le_bytes())
            .collect();
        let values: Vec<u8> = (0..stride * 16 * kv_heads * 128)
            .flat_map(|i| half::f16::from_f32(((i * 11 % 47) as f32 - 23.0) / 32.0).to_le_bytes())
            .collect();
        // Two logical slots share permuted physical pages. One has only two
        // tokens, so most of its split groups are empty even at high splits.
        let table: Vec<u8> = (0..rows * stride)
            .flat_map(|i| ((i * 7 % stride) as u32).to_le_bytes())
            .collect();
        let meta: Vec<u8> = [0u32, 1, 1, 289]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let q = device.upload(&q).unwrap();
        let k = device.upload(&keys).unwrap();
        let v = device.upload(&values).unwrap();
        let pages = device.upload(&table).unwrap();
        let meta = device.upload(&meta).unwrap();
        let a = device.alloc(rows * heads * 128 * 4).unwrap();
        let b = device
            .upload(
                &(0..(rows + 32) * heads * 128)
                    .flat_map(|_| f32::NAN.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let parts = device.alloc(rows * heads * 32 * 130 * 4).unwrap();
        let mut p = [
            heads as u32,
            kv_heads as u32,
            128,
            0,
            stride as u32,
            (1.0f32 / 128.0).to_bits(),
            1,
        ];
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "attention",
            &[&q, &k, &v, &meta, &pages, &a],
            &p,
            [heads, rows, 1],
            32,
        );
        cmd.finish().unwrap();
        let reference = unsafe { a.read_f32(0, rows * heads * 128) };
        let row_map = device
            .upload(
                &[1u32, 0]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for splits in [1usize, 2, 7, 32] {
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                "attention_gqa4",
                &[
                    &q,
                    &k,
                    &v,
                    &meta,
                    &pages,
                    &row_map,
                    if splits == 1 { &b } else { &parts },
                ],
                &[
                    heads as u32,
                    kv_heads as u32,
                    stride as u32,
                    p[5],
                    rows as u32,
                    splits as u32,
                ],
                [kv_heads, rows, splits],
                128,
            );
            if splits > 1 {
                cmd.dispatch(
                    "attention_gqa_merge",
                    &[&parts, &b, &row_map],
                    &[splits as u32, heads as u32],
                    [rows * heads, 1, 1],
                    32,
                );
            }
            cmd.finish().unwrap();
            let actual = unsafe { b.read_f32(0, reference.len()) };
            assert!(actual.iter().all(|v| v.is_finite()));
            let error = reference
                .iter()
                .zip(actual)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 1e-5, "tiled GQA, {splits} splits: {error}");
        }
        for splits in [2, 7, 32] {
            p[6] = splits as u32;
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                "attention",
                &[&q, &k, &v, &meta, &pages, &parts],
                &p,
                [heads, rows, splits],
                32,
            );
            cmd.dispatch(
                "attention_merge",
                &[&parts, &b],
                &[splits as u32, 0],
                [rows * heads, 1, 1],
                32,
            );
            cmd.finish().unwrap();
            let actual = unsafe { b.read_f32(0, reference.len()) };
            assert!(actual.iter().all(|v| v.is_finite()));
            let error = reference
                .iter()
                .zip(actual)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 1e-5, "{splits} attention splits: {error}");
        }

        // Tensor attention: nonzero first row, a ragged second tile, causal
        // positions inside existing pages, and the same permuted GQA cache.
        let rows = 40usize;
        let qs: Vec<u8> = (0..rows * heads * 128)
            .flat_map(|i| (((i * 17 % 31) as f32 - 15.0) / 16.0).to_le_bytes())
            .collect();
        let metadata: Vec<u8> = (0..rows)
            .flat_map(|i| [0u32, 100 + i as u32])
            .flat_map(u32::to_le_bytes)
            .collect();
        let q = device.upload(&qs).unwrap();
        let meta = device.upload(&metadata).unwrap();
        let a = device.alloc(rows * heads * 128 * 4).unwrap();
        let b = device
            .upload(
                &(0..(rows + 32) * heads * 128)
                    .flat_map(|_| f32::NAN.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        p[3] = 0;
        p[6] = 1;
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "attention",
            &[&q, &k, &v, &meta, &pages, &a],
            &p,
            [heads, rows, 1],
            32,
        );
        cmd.dispatch(
            "attention_mpp",
            &[&q, &k, &v, &meta, &pages, &b],
            &[heads as u32, kv_heads as u32, stride as u32, p[5], 3, 37],
            [heads, 3, 1],
            32,
        );
        cmd.finish().unwrap();
        let reference = unsafe { a.read_f32(3 * heads * 128, 37 * heads * 128) };
        let actual = unsafe { b.read_f32(3 * heads * 128, reference.len()) };
        assert!(actual.iter().all(|v| v.is_finite()));
        let error = reference
            .iter()
            .zip(actual)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(error < 1e-3, "tiled attention error: {error}");
        let qh = device.alloc((rows + 32) * heads * 128 * 2).unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "attention_query",
            &[&q, &qh],
            &[heads as u32 * 128, 0, rows as u32],
            [((rows + 32) * heads * 128).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "attention_prefill",
            &[&qh, &k, &v, &meta, &pages, &b],
            &[heads as u32, kv_heads as u32, stride as u32, p[5], 3, 37],
            [heads, 2, 1],
            128,
        );
        cmd.finish().unwrap();
        let actual = unsafe { b.read_f32(3 * heads * 128, reference.len()) };
        assert!(actual.iter().all(|v| v.is_finite()));
        let error = reference
            .iter()
            .zip(actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(error < 1e-3, "register-held tiled attention error: {error}");
        // Reverse dispatch order and retain the nonzero start/ragged tail.
        // The real-model cohort test adds multiple distinct logical slots.
        let tiles = device
            .upload(
                &[35u32, 5, 3, 32]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "attention_prefill_batched",
            &[&qh, &k, &v, &meta, &pages, &b, &tiles],
            &[heads as u32, kv_heads as u32, stride as u32, p[5]],
            [heads, 2, 1],
            128,
        );
        cmd.finish().unwrap();
        let actual = unsafe { b.read_f32(3 * heads * 128, reference.len()) };
        assert!(actual.iter().all(|v| v.is_finite()));
        let error = reference
            .iter()
            .zip(actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(error < 1e-3, "batched tiled attention error: {error}");
        assert!(
            unsafe { b.read_f32(0, 3 * heads * 128) }
                .iter()
                .all(|v| v.is_nan())
        );
        assert!(
            unsafe { b.read_f32(rows * heads * 128, 32 * heads * 128) }
                .iter()
                .all(|v| v.is_nan()),
            "attention output tail"
        );
    }

    #[test]
    fn memory_grant_is_enforced_and_released() {
        let device = MetalDevice::new(Some(4096)).expect("M5 GPU required");
        assert!(device.alloc(0).is_err());
        let a = device.alloc(4096).unwrap();
        assert!(device.alloc(1).is_err());
        drop(a);
        assert_eq!(device.allocated_bytes(), 0);
        assert!(device.alloc(4096).is_ok());
    }

    #[test]
    fn selected_norm_preserves_order_and_only_writes_selected_rows() {
        let device = MetalDevice::new(Some(1 << 20)).unwrap();
        let n = 96usize;
        let values: Vec<u8> = (0..7 * n)
            .flat_map(|i| ((i % 37) as f32 / 16.0).to_le_bytes())
            .collect();
        let weights: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let x = device.upload(&values).unwrap();
        let w = device.upload(&weights).unwrap();
        let rows = device
            .upload(
                &[5u32, 2, 0]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let a = device.alloc(7 * n * 4).unwrap();
        let b = device.alloc(3 * n * 4).unwrap();
        let p = [n as u32, 0, 1e-5f32.to_bits()];
        let cmd = device.begin().unwrap();
        cmd.dispatch("rms", &[&x, &w, &a], &p, [7, 1, 1], 256);
        cmd.dispatch("rms_selected", &[&x, &w, &rows, &b], &p, [3, 1, 1], 256);
        cmd.finish().unwrap();
        for (i, row) in [5, 2, 0].into_iter().enumerate() {
            assert_eq!(unsafe { a.read_f32(row * n, n) }, unsafe {
                b.read_f32(i * n, n)
            });
        }
        let x2 = device.upload(&values).unwrap();
        let delta = device.upload(&values).unwrap();
        let c = device.alloc(7 * n * 4).unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "residual",
            &[&x, &delta],
            &[(7 * n) as u32, 0.25f32.to_bits()],
            [(7 * n).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch("rms", &[&x, &w, &a], &p, [7, 1, 1], 256);
        cmd.dispatch(
            "residual_rms",
            &[&x2, &delta, &w, &c],
            &[n as u32, 0, p[2], 0.25f32.to_bits()],
            [7, 1, 1],
            256,
        );
        cmd.finish().unwrap();
        assert_eq!(
            unsafe { a.read_f32(0, 7 * n) },
            unsafe { c.read_f32(0, 7 * n) },
            "fused normalization"
        );
        assert_eq!(
            unsafe { x.read_f32(0, 7 * n) },
            unsafe { x2.read_f32(0, 7 * n) },
            "stored residual"
        );
    }
}
