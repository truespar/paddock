//! Kolibri 1's NVFP4 safetensors build (compressed-tensors
//! `nvfp4-pack-quantized`): the routed experts NVFP4 off the checkpoint's own
//! nibbles, everything else BF16 as shipped - attention (q|k|v fused at load,
//! o), the shared expert, embeddings and head as device-resident bf16 planes;
//! norms, router and selection bias widened to f32 (exact). The geometry and
//! rope are the GGUF Kolibri flavor's, read off config.json instead of the
//! header, and the same body serves it: only the projection / head / MoE
//! arms differ (`Proj::Bf16`, `Head::Bf16`, `Ffn::MoeNv`).
//!
//! Tensor map (the HF names; `post_attention_layernorm` is the PRE-FFN norm
//! here, the sandwich pair is `post_{attn,ffn}_norm` - the converter's
//! mapping): `model.layers.N.{input_layernorm, self_attn.{q,k,v,o}_proj,
//! self_attn.{q,k}_norm, post_attn_norm, post_attention_layernorm,
//! mlp.gate, moe.router.expert_bias, mlp.experts.E.{gate,up,down}_proj,
//! mlp.shared_experts.{gate,up,down}_proj, post_ffn_norm}`, plus
//! `model.{embed_tokens, norm}` and `lm_head`.

use std::path::Path;
use std::sync::Arc;

use paddock_kernels::reference::ops::YarnRope;
use paddock_models::kolibri::{self, KolibriConfig};
use paddock_models::safetensors::ShardedSafetensors;

use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::{bf16_concat_plane, bf16_plane, f32_tensor, nvf4_expert_stack};

use super::*;

impl GpuLaguna {
    /// Load the Kolibri NVFP4 checkpoint directory. `max_ctx` bounds the
    /// serial lane's KV, as on the GGUF lane.
    pub fn load_kolibri_nvfp4(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        let fam = Flavor::Kolibri.name();
        KolibriConfig::read_nvfp4(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("{fam} config: {e}")))?;
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("{fam} shards: {e}")))?;
        // the W4A16 decode GEMVs are the floor every NVFP4 build needs; the
        // block-scaled prefill pair is elected per tick when present
        if !exec.has_nvf4_ckpt() || !exec.has_q4x_moe_gu_swiglu() {
            let (maj, min) = exec.compute_capability();
            return Err(GpuModelError::Unsupported(format!(
                "{fam} NVFP4: sm_{maj}{min} / this kernel pack has no NVFP4 SwiGLU MoE lane - \
                 serve the GGUF build instead"
            )));
        }
        if !exec.has_moe_topk_logit_sigmoid() {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: the kernel pack has no sigmoid_logit_add router (slot 748) - update the pack"
            )));
        }
        // the checkpoint's own byte total off the tensor directory (shard
        // headers stay out of it) - the number that has to fit
        let ckpt_bytes: u64 = st
            .names()
            .filter_map(|n| st.bytes(n).map(|(_, b)| b.len() as u64))
            .sum();
        exec.vram_load_gate(ckpt_bytes, fam)
            .map_err(GpuModelError::WontFit)?;
        exec.disable_event_tracking();

        let (n_layer, n_embd, head_dim) = (kolibri::LAYERS, kolibri::WIDTH, kolibri::HEAD_DIM);
        let (n_kv_heads, n_vocab) = (kolibri::KV_HEADS, kolibri::VOCAB);
        let q_dim = kolibri::HEADS * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let ctx_train = kolibri::MAX_CONTEXT;
        // config.json carries no routed_scaling_factor and norm_topk_prob is
        // false (read_nvfp4 checked it): the weights are the bare sigmoids
        let moe = MoeDims {
            n_expert: kolibri::EXPERTS,
            n_active: kolibri::ACTIVE,
            moe_ff: kolibri::FF,
            shexp_ff: kolibri::FF,
            routed_scale: 1.0,
            router: Router::LogitSigmoid,
        };
        // the GGUF flavor's rope: SWA layers full-rotary at theta 10k, full
        // layers NoPE (rope_full is never read there, built for the shape)
        let rope_swa =
            YarnRope::new(head_dim, 10_000.0, 1.0, ctx_train, 0.0, 1.0, 32.0, 1.0).kernel_params();
        let rope_full =
            YarnRope::new(head_dim, 10_000.0, 1.0, ctx_train, 1.0, 1.0, 32.0, 1.0).kernel_params();

        let f32_dt = |name: &str, dims: Vec<usize>| -> Result<DeviceTensor, GpuModelError> {
            let n: usize = dims.iter().product();
            Ok(DeviceTensor {
                buf: exec.to_device(&f32_tensor(&st, name, n)?)?,
                dims,
            })
        };
        let vfree = || {
            cudarc::driver::result::mem_get_info()
                .map(|(f, _)| f as u64)
                .unwrap_or(0)
        };
        let gb = |used: u64| used as f64 / 1e9;
        let v_start = vfree();

        let tok_embd = TokEmbd::Bf16(bf16_plane(
            &exec,
            &st,
            "model.embed_tokens.weight",
            n_vocab,
            n_embd,
        )?);
        let mut layers = Vec::with_capacity(n_layer);
        let (mut attn_bytes, mut expert_bytes, mut shared_bytes) = (0u64, 0u64, 0u64);
        for i in 0..n_layer {
            let p = format!("model.layers.{i}");
            let va = vfree();
            let proj = Proj::Bf16 {
                wqkv: bf16_concat_plane(
                    &exec,
                    &st,
                    &[
                        (&format!("{p}.self_attn.q_proj.weight"), q_dim),
                        (&format!("{p}.self_attn.k_proj.weight"), kv_dim),
                        (&format!("{p}.self_attn.v_proj.weight"), kv_dim),
                    ],
                    n_embd,
                )?,
                wo: bf16_plane(
                    &exec,
                    &st,
                    &format!("{p}.self_attn.o_proj.weight"),
                    n_embd,
                    q_dim,
                )?,
            };
            attn_bytes += va.saturating_sub(vfree());
            let ve = vfree();
            let ff = moe.moe_ff;
            let expert = |role: &str, rows: usize, in_dim: usize| {
                nvf4_expert_stack(
                    &exec,
                    &st,
                    |e| format!("{p}.mlp.experts.{e}.{role}"),
                    moe.n_expert,
                    rows,
                    in_dim,
                )
            };
            let (gate, up, down) = (
                expert("gate_proj", ff, n_embd)?,
                expert("up_proj", ff, n_embd)?,
                expert("down_proj", n_embd, ff)?,
            );
            expert_bytes += ve.saturating_sub(vfree());
            let vs = vfree();
            let sh = |role: &str, n: usize, k: usize| {
                bf16_plane(
                    &exec,
                    &st,
                    &format!("{p}.mlp.shared_experts.{role}.weight"),
                    n,
                    k,
                )
            };
            let ffn = Ffn::MoeNv(MoeNv {
                router_w: f32_dt(&format!("{p}.mlp.gate.weight"), vec![n_embd, moe.n_expert])?,
                router_b16: bf16_plane(
                    &exec,
                    &st,
                    &format!("{p}.mlp.gate.weight"),
                    moe.n_expert,
                    n_embd,
                )?,
                probs_bias: f32_dt(&format!("{p}.moe.router.expert_bias"), vec![moe.n_expert])?,
                gate,
                up,
                down,
                sh_gate: sh("gate_proj", ff, n_embd)?,
                sh_up: sh("up_proj", ff, n_embd)?,
                sh_down: sh("down_proj", n_embd, ff)?,
            });
            shared_bytes += vs.saturating_sub(vfree());
            let swa = KolibriConfig::sliding(i);
            let norm = |name: &str, n: usize| f32_dt(&format!("{p}.{name}.weight"), vec![n]);
            layers.push(LagunaLayer {
                attn_norm: norm("input_layernorm", n_embd)?,
                proj,
                q_norm: norm("self_attn.q_norm", head_dim)?,
                k_norm: norm("self_attn.k_norm", head_dim)?,
                ffn_norm: norm("post_attention_layernorm", n_embd)?,
                post_attn_norm: Some(norm("post_attn_norm", n_embd)?),
                post_ffn_norm: Some(norm("post_ffn_norm", n_embd)?),
                ffn,
                is_swa: swa,
                nope: !swa,
                n_heads: kolibri::HEADS,
            });
        }
        let output_norm = f32_dt("model.norm.weight", vec![n_embd])?;
        let lm_head = Head::Bf16(bf16_plane(&exec, &st, "lm_head.weight", n_vocab, n_embd)?);
        tracing::info!(
            "{fam} VRAM  attention BF16 (q|k|v fused, o)        {:>7.2} GB",
            gb(attn_bytes)
        );
        tracing::info!(
            "{fam} VRAM  routed experts NVFP4 ({}e x {n_layer} layers) {:>7.2} GB",
            moe.n_expert,
            gb(expert_bytes)
        );
        tracing::info!(
            "{fam} VRAM  shared experts BF16 + routers      {:>7.2} GB",
            gb(shared_bytes)
        );
        let weights_bytes = exec
            .settled_mem_used()
            .unwrap_or_else(|| v_start.saturating_sub(vfree()));
        tracing::info!(
            "{fam} VRAM  = model resident total                {:>7.2} GB  \
             (NVFP4 experts, BF16 rest; {n_layer} layers, {} experts top-{})",
            gb(weights_bytes),
            moe.n_expert,
            moe.n_active,
        );
        let content_id = (
            crate::kv_tier::fingerprint::weights_safetensors(&st),
            crate::kv_tier::fingerprint::tokenizer_dir(dir),
        );
        Ok(Self {
            exec,
            hp: Hparams {
                flavor: Flavor::Kolibri,
                n_layer,
                n_embd,
                n_heads: vec![kolibri::HEADS; n_layer],
                n_kv_heads,
                head_dim,
                n_vocab,
                eps: 1e-6,
                swa_window: kolibri::WINDOW,
                n_rot: head_dim,
                rope_full,
                rope_swa,
                moe,
            },
            tok_embd,
            layers,
            output_norm,
            lm_head,
            max_ctx: max_ctx.min(ctx_train),
            weights_bytes,
            content_id,
            // KV8 default, as on the GGUF lane
            kv_dtype: KvDtype::Fp8E4m3,
            decode: None,
            scratch: None,
            batch: None,
            pipe: None,
            dflash: None,
            chunked: Vec::new(),
        })
    }
}
