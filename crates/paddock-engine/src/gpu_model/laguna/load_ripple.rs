//! Ricursive's Ripple-65B NVFP4 build (ModelOpt `NVFP4`): the MLP linears
//! NVFP4 off the checkpoint's own nibbles and scales (W4A16 here - the
//! engine's 16-bit-activation class, the static input scales unread),
//! everything else BF16 as shipped - attention (q|k|g fused at load, v on
//! its own because it reads the value-residual blend, o), embeddings and
//! head as device-resident bf16 planes; norms widened to f32 (exact).
//!
//! Tensor map: `model.layers.N.{input_layernorm, self_attn.{q,k,v,o,g}_proj,
//! self_attn.{q,k}_norm, self_attn.value_res_lambda,
//! post_attention_layernorm, pre_feedforward_layernorm,
//! mlp.{gate,up,down}_proj.{weight, weight_scale, weight_scale_2,
//! input_scale}, post_feedforward_layernorm}`, plus
//! `model.{embed_tokens, norm}` and `lm_head`. `post_attention_layernorm` is
//! the post-attention sandwich norm here (PeriNorm), the pre-FFN norm is
//! `pre_feedforward_layernorm`.

use std::path::Path;
use std::sync::Arc;

use paddock_kernels::reference::ops::YarnRope;
use paddock_models::ripple::{self, RippleConfig};
use paddock_models::safetensors::ShardedSafetensors;

use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::{bf16_concat_plane, bf16_plane, bf16_to_f32, f32_tensor};

use super::*;

/// The positions the card and the BF16 release's tokenizer config declare
/// (1Mi; the NVFP4 build's tokenizer config carries its Spark bundle's 128K
/// serving context instead); the config's own `max_position_embeddings`
/// (2Mi) is past what the preview was trained and evaluated at.
const RIPPLE_CONTEXT: usize = 1 << 20;

impl GpuLaguna {
    /// Load the Ripple NVFP4 checkpoint directory. `max_ctx` bounds the
    /// serial lane's KV.
    pub fn load_ripple_nvfp4(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        let fam = Flavor::Ripple.name();
        let cfg = RippleConfig::read_nvfp4(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("{fam} config: {e}")))?;
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("{fam} shards: {e}")))?;
        if !exec.has_nvf4_ckpt() {
            let (maj, min) = exec.compute_capability();
            return Err(GpuModelError::Unsupported(format!(
                "{fam} NVFP4: sm_{maj}{min} / this kernel pack has no NVFP4 checkpoint lane"
            )));
        }
        if !exec.has_ripple_ops() {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: the kernel pack has no clamped SwiGLU / value-residual blend (slots \
                 968-969) - update the pack"
            )));
        }
        let ckpt_bytes: u64 = st
            .names()
            .filter_map(|n| st.bytes(n).map(|(_, b)| b.len() as u64))
            .sum();
        exec.vram_load_gate(ckpt_bytes, fam)
            .map_err(GpuModelError::WontFit)?;
        exec.disable_event_tracking();

        let (n_layer, n_embd, head_dim) = (ripple::LAYERS, ripple::WIDTH, ripple::HEAD_DIM);
        let (n_kv_heads, n_vocab, ff) = (ripple::KV_HEADS, ripple::VOCAB, ripple::FF);
        let q_dim = ripple::HEADS * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        // SWA layers: full-rotary rope at theta 10k (rotate_half); the full
        // layers are NoPE (rope_full is built for the shape, never read)
        let rope_swa = YarnRope::new(
            head_dim,
            ripple::ROPE_THETA,
            1.0,
            RIPPLE_CONTEXT,
            0.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params();
        let rope_full = rope_swa;

        let f32_dt = |name: &str, dims: Vec<usize>| -> Result<DeviceTensor, GpuModelError> {
            let n: usize = dims.iter().product();
            Ok(DeviceTensor {
                buf: exec.to_device(&f32_tensor(&st, name, n)?)?,
                dims,
            })
        };
        let nv = |name: &str| -> Result<crate::gpu::Nvf4Plane, GpuModelError> {
            let v = paddock_models::modelopt::nvfp4_view(&st, name)
                .map_err(|e| GpuModelError::Unsupported(format!("{fam} {name}: {e}")))?;
            Ok(exec.nvf4_upload(v.packed, v.scales, v.scale2, v.n, v.k)?)
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
        let (mut attn_bytes, mut mlp_bytes) = (0u64, 0u64);
        for i in 0..n_layer {
            let p = format!("model.layers.{i}");
            let va = vfree();
            let proj = Proj::Ripple {
                wqkg: bf16_concat_plane(
                    &exec,
                    &st,
                    &[
                        (&format!("{p}.self_attn.q_proj.weight"), q_dim),
                        (&format!("{p}.self_attn.k_proj.weight"), kv_dim),
                        (&format!("{p}.self_attn.g_proj.weight"), q_dim),
                    ],
                    n_embd,
                )?,
                wv: bf16_plane(
                    &exec,
                    &st,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
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
            let vm = vfree();
            let mlp = |role: &str| nv(&format!("{p}.mlp.{role}"));
            let ffn = Ffn::DenseNv {
                gate: mlp("gate_proj")?,
                up: mlp("up_proj")?,
                down: mlp("down_proj")?,
            };
            if let Ffn::DenseNv { gate, down, .. } = &ffn
                && (gate.out_dim != ff || gate.in_dim != n_embd || down.out_dim != n_embd)
            {
                return Err(GpuModelError::Unsupported(format!(
                    "{fam} layer {i}: MLP planes are not {n_embd} -> {ff} -> {n_embd}"
                )));
            }
            mlp_bytes += vm.saturating_sub(vfree());
            // the blend weight, sigmoid(lambda): one bf16 scalar a layer
            let lambda = st
                .bytes(&format!("{p}.self_attn.value_res_lambda"))
                .map(|(_, b)| bf16_to_f32(b))
                .filter(|v| v.len() == 1)
                .ok_or_else(|| {
                    GpuModelError::Unsupported(format!(
                        "{fam} layer {i}: value_res_lambda missing or not a scalar"
                    ))
                })?[0];
            // the MLP's static activation scales; gate and up read one input
            // and the checkpoint gives them one scale
            let act = |role: &str| -> Result<f32, GpuModelError> {
                let name = format!("{p}.mlp.{role}.input_scale");
                st.bytes(&name)
                    .filter(|(_, b)| b.len() == 4)
                    .map(|(_, b)| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .ok_or_else(|| {
                        GpuModelError::Unsupported(format!("{fam} {name}: not a positive f32"))
                    })
            };
            if act("gate_proj")? != act("up_proj")? {
                return Err(GpuModelError::Unsupported(format!(
                    "{fam} layer {i}: gate and up carry different input scales"
                )));
            }
            let swa = RippleConfig::sliding(i);
            let norm = |name: &str, n: usize| f32_dt(&format!("{p}.{name}.weight"), vec![n]);
            layers.push(LagunaLayer {
                attn_norm: norm("input_layernorm", n_embd)?,
                proj,
                q_norm: norm("self_attn.q_norm", head_dim)?,
                k_norm: norm("self_attn.k_norm", head_dim)?,
                ffn_norm: norm("pre_feedforward_layernorm", n_embd)?,
                post_attn_norm: Some(norm("post_attention_layernorm", n_embd)?),
                post_ffn_norm: Some(norm("post_feedforward_layernorm", n_embd)?),
                sinks: None,
                ffn,
                is_swa: swa,
                nope: !swa,
                n_heads: ripple::HEADS,
                valres: Some(1.0 / (1.0 + (-lambda).exp())),
                ffn_act: Some([act("gate_proj")?, act("down_proj")?]),
            });
        }
        let output_norm = f32_dt("model.norm.weight", vec![n_embd])?;
        let lm_head = Head::Bf16(bf16_plane(&exec, &st, "lm_head.weight", n_vocab, n_embd)?);
        tracing::info!(
            "{fam} VRAM  attention BF16 (q|k|g fused, v, o)     {:>7.2} GB",
            gb(attn_bytes)
        );
        tracing::info!(
            "{fam} VRAM  MLP NVFP4 ({n_layer} layers)                {:>7.2} GB",
            gb(mlp_bytes)
        );
        let weights_bytes = exec
            .settled_mem_used()
            .unwrap_or_else(|| v_start.saturating_sub(vfree()));
        tracing::info!(
            "{fam} VRAM  = model resident total                {:>7.2} GB  \
             (NVFP4 MLP, BF16 rest; {n_layer} layers)",
            gb(weights_bytes),
        );
        let content_id = (
            crate::kv_tier::fingerprint::weights_safetensors(&st),
            crate::kv_tier::fingerprint::tokenizer_dir(dir),
        );
        Ok(Self {
            exec,
            hp: Hparams {
                flavor: Flavor::Ripple,
                n_layer,
                n_embd,
                n_heads: vec![ripple::HEADS; n_layer],
                n_kv_heads,
                head_dim,
                n_vocab,
                eps: ripple::EPS,
                swa_window: ripple::WINDOW,
                n_rot: head_dim,
                rope_full,
                rope_swa,
                // dense: no router, no experts - the MoE dims are inert
                moe: MoeDims {
                    n_expert: 0,
                    n_active: 0,
                    moe_ff: 0,
                    shexp_ff: 0,
                    routed_scale: 1.0,
                    router: Router::SigmoidNorm,
                },
                embed_scale: cfg.embed_scale,
                ffn_clamp: Some(ripple::FFN_CLAMP),
                // the reference serves the MLP W4A4 off the checkpoint's own
                // static input scales; the prefill rides that class (3.6x
                // the W4A16 tile at a 1.2K chunk on GB10), decode stays W4A16.
                // PADDOCK_RIPPLE_PF_W4A16=1 pins the tile for A/B.
                pf_w4a4: paddock_models::dev_var_os!("PADDOCK_RIPPLE_PF_W4A16").is_none(),
            },
            tok_embd,
            layers,
            output_norm,
            lm_head,
            max_ctx: max_ctx.min(RIPPLE_CONTEXT).min(cfg.max_positions),
            weights_bytes,
            content_id,
            // KV8 default, as on every laguna lane
            kv_dtype: KvDtype::Fp8E4m3,
            decode: None,
            scratch: None,
            batch: None,
            pipe: None,
            dflash: None,
            dflash2: None,
            chunked: Vec::new(),
            w16r: None,
        })
    }
}
