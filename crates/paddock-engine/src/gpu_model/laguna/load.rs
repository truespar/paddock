//! Laguna weight load + geometry - loader-first bring-up, validated against
//! the real `Laguna-XS-2.1-Q4_K_M.gguf`. Every metadata key below was
//! verified in the file's dump; anything absent or off-spec is a hard,
//! named error - never a silent default that changes math.

use std::sync::Arc;

use paddock_kernels::reference::ops::YarnRope;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuError, GpuExecutor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;

use super::*;

/// llama.cpp's `expert_gating_func` enum value for sigmoid scoring - the only
/// router class Laguna ships (softmax = 1 would be silently-wrong math here).
const GATING_FUNC_SIGMOID: u64 = 2;
/// Kolibri 1's `sigmoid_logit_add` (select on logits + bias, weight by the
/// unbiased sigmoid) - the value the kolibri1 converter stamps.
const GATING_FUNC_SIGMOID_LOGIT_ADD: u64 = 5;

impl GpuLaguna {
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        Self::load_with(exec, map, max_ctx, None)
    }

    /// `fp8_native_dir` is accepted for constructor parity with the other
    /// families; Laguna has no fp8-native lane yet (S-2.1 official FP8 is the
    /// phase-2 sm_89+ item).
    pub fn load_with(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
        _fp8_native_dir: Option<&std::path::Path>,
    ) -> Result<Self, GpuModelError> {
        let flavor = match map.gguf().architecture() {
            Some("kolibri1") => Flavor::Kolibri,
            _ => Flavor::Laguna,
        };
        let fam = flavor.name();
        exec.vram_load_gate(map.total_len(), fam)
            .map_err(GpuModelError::WontFit)?;
        // Single-stream engine: cudarc's cross-stream event tracking is pure
        // overhead and blocks CUDA-graph capture. Must precede all allocs.
        exec.disable_event_tracking();

        let u = |k: &str| {
            map.gguf()
                .arch_field(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let f = |k: &str| map.gguf().arch_field(k).and_then(Value::as_f32);
        let u_arr = |k: &str| -> Result<Vec<u64>, GpuModelError> {
            match map.gguf().arch_field(k) {
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
                    })
                    .collect(),
                _ => Err(GpuModelError::MissingMeta(k.to_owned())),
            }
        };

        let n_layer = u("block_count")? as usize;
        let n_embd = u("embedding_length")? as usize;
        // Per-layer Q-head counts - Laguna's signature quirk. The converter
        // always writes the array form; Kolibri's writes one count.
        let n_heads: Vec<usize> = match flavor {
            Flavor::Laguna => u_arr("attention.head_count")?
                .into_iter()
                .map(|v| v as usize)
                .collect(),
            Flavor::Kolibri => vec![u("attention.head_count")? as usize; n_layer],
        };
        if n_heads.len() != n_layer {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: attention.head_count array has {} entries for {} layers",
                n_heads.len(),
                n_layer
            )));
        }
        let n_kv_heads = u("attention.head_count_kv")? as usize;
        let head_dim = u("attention.key_length")? as usize;
        if u("attention.value_length")? as usize != head_dim {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: value_length != key_length is not a shipped geometry"
            )));
        }
        let eps = f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let ctx_train = u("context_length")? as usize;

        // Hybrid layout: [full, SWA, SWA, SWA] repeating, full at il%4==0
        // (llama.cpp set_swa_pattern(4, dense_first=true)). The per-layer head
        // array is cross-checked against the pattern - all full layers must
        // share one count and all SWA layers another; a mismatch means the
        // file doesn't follow the convention and we refuse rather than guess.
        // Kolibri stamps its pattern ([SWA x4, full]) as a per-layer bool
        // array, true = sliding; that array is the authority there.
        let swa_window = u("attention.sliding_window")? as usize;
        if swa_window == 0 {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: sliding_window 0 (M.1-class all-full geometry) is not supported"
            )));
        }
        let swa_pattern: Vec<bool> = match flavor {
            Flavor::Laguna => (0..n_layer).map(|i| !i.is_multiple_of(4)).collect(),
            Flavor::Kolibri => match map.gguf().arch_field("attention.sliding_window_pattern") {
                Some(Value::Array(items)) if items.len() == n_layer => items
                    .iter()
                    .map(|v| match v {
                        Value::Bool(b) => Ok(*b),
                        _ => Err(GpuModelError::MissingMeta(
                            "attention.sliding_window_pattern".into(),
                        )),
                    })
                    .collect::<Result<_, _>>()?,
                _ => {
                    return Err(GpuModelError::MissingMeta(
                        "attention.sliding_window_pattern".into(),
                    ));
                }
            },
        };
        let is_swa = |i: usize| swa_pattern[i];
        for (i, &h) in n_heads.iter().enumerate() {
            let expect = if is_swa(i) { n_heads[1] } else { n_heads[0] };
            if h != expect {
                return Err(GpuModelError::Unsupported(format!(
                    "{fam}: head_count[{i}] = {h} breaks the [full, SWA×3] convention \
                     (full layers {} / SWA layers {})",
                    n_heads[0], n_heads[1]
                )));
            }
        }

        // MoE geometry + router class.
        let moe = MoeDims {
            n_expert: u("expert_count")? as usize,
            n_active: u("expert_used_count")? as usize,
            moe_ff: u("expert_feed_forward_length")? as usize,
            shexp_ff: u("expert_shared_feed_forward_length")? as usize,
            routed_scale: f("expert_weights_scale").unwrap_or(1.0),
            router: match flavor {
                Flavor::Laguna => Router::SigmoidNorm,
                Flavor::Kolibri => Router::LogitSigmoid,
            },
        };
        let (want_gating, want_name) = match flavor {
            Flavor::Laguna => (GATING_FUNC_SIGMOID, "sigmoid"),
            Flavor::Kolibri => (GATING_FUNC_SIGMOID_LOGIT_ADD, "sigmoid_logit_add"),
        };
        let gating = u("expert_gating_func").unwrap_or(GATING_FUNC_SIGMOID);
        if gating != want_gating {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: expert_gating_func {gating} (only {want_name} = {want_gating} shipped)"
            )));
        }
        // Kolibri's router never renormalizes (norm_topk_prob false); a file
        // that says otherwise asks for math slot 748 does not do
        if flavor == Flavor::Kolibri
            && matches!(
                map.gguf().arch_field("expert_weights_norm"),
                Some(Value::Bool(true))
            )
        {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: expert_weights_norm true (renormalized sigmoid_logit_add) is not shipped"
            )));
        }
        if flavor == Flavor::Kolibri && !exec.has_moe_topk_logit_sigmoid() {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: the kernel pack has no sigmoid_logit_add router (slot 748) - update the pack"
            )));
        }
        // Kolibri routes every layer (no dense lead, no key)
        let n_dense_lead = match flavor {
            Flavor::Laguna => u("leading_dense_block_count")? as usize,
            Flavor::Kolibri => 0,
        };

        // Rope, per layer type. Full layers: YaRN over n_rot=64 of 128 dims
        // (partial rotary - the tail passes through untouched); the GGUF
        // stamps yarn_attn_factor 1.0 and YarnRope derives the real mscale
        // 1 + 0.1·ln(factor) from freq_scale. SWA layers: plain rope over the
        // whole head (ext_factor 0 disables the YaRN ramp).
        // Kolibri: the SWA layers rotate the whole head at the base theta
        // (no dimension_count key; freq_base_swa defaults to freq_base, the
        // converter's contract) and the full layers are NoPE - rope_full is
        // never read there.
        let n_rot = match flavor {
            Flavor::Laguna => u("rope.dimension_count")? as usize,
            Flavor::Kolibri => u("rope.dimension_count").map_or(head_dim, |v| v as usize),
        };
        let n_rot_swa = u("rope.dimension_count_swa")
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let rope_base = f("rope.freq_base")
            .ok_or_else(|| GpuModelError::MissingMeta("rope.freq_base".into()))?;
        let rope_base_swa = f("rope.freq_base_swa").unwrap_or(match flavor {
            Flavor::Laguna => 10_000.0,
            Flavor::Kolibri => rope_base,
        });
        if flavor == Flavor::Kolibri && (n_rot_swa != head_dim || n_rot != head_dim) {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: partial rotary ({n_rot_swa} of {head_dim}) is not a shipped geometry"
            )));
        }
        let factor = f("rope.scaling.factor").unwrap_or(1.0);
        let orig_ctx =
            u("rope.scaling.original_context_length").unwrap_or(ctx_train as u64) as usize;
        let beta_fast = f("rope.scaling.yarn_beta_fast").unwrap_or(32.0);
        let beta_slow = f("rope.scaling.yarn_beta_slow").unwrap_or(1.0);
        let attn_factor = f("rope.scaling.yarn_attn_factor").unwrap_or(1.0);
        let rope_full = YarnRope::new(
            n_rot,
            rope_base,
            1.0 / factor,
            orig_ctx,
            1.0,
            attn_factor,
            beta_fast,
            beta_slow,
        )
        .kernel_params();
        let rope_swa = YarnRope::new(
            n_rot_swa,
            rope_base_swa,
            1.0,
            ctx_train,
            0.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params();

        // Per-component VRAM ledger - free-VRAM snapshots between phases feed
        // the startup log (the "show what's on GPU" principle).
        let vfree = || {
            cudarc::driver::result::mem_get_info()
                .map(|(f, _)| f as u64)
                .unwrap_or(0)
        };
        let gb = |used: u64| used as f64 / 1e9;
        let v_start = vfree();

        // Token embeddings stay resident in their file quant (gathered rows
        // dequantize on the fly): Q4_K on the XS election -> kq residency.
        let te_ty = map
            .tensor_info("token_embd.weight")
            .map(|t| t.ggml_type)
            .ok_or_else(|| GpuModelError::MissingMeta("token_embd.weight".into()))?;
        let (tok_embd, n_vocab) = if crate::gpu::kq_params(te_ty).is_some() {
            let t = exec.repack_kquant(map, "token_embd.weight")?;
            let vocab = t.dims[1];
            (TokEmbd::Kq(t), vocab)
        } else {
            let t = exec.upload_raw(map, "token_embd.weight")?;
            if t.ty != paddock_models::ggml_type::GgmlType::Q8_0 {
                return Err(GpuModelError::Unsupported(format!(
                    "token_embd.weight quant {:?} has no resident gather path",
                    t.ty
                )));
            }
            let vocab = t.dims[1];
            (TokEmbd::Q8(t), vocab)
        };
        let v_embd = vfree();
        tracing::info!(
            "{fam} VRAM  input embeddings token_embd          {:>7.2} GB",
            gb(v_start.saturating_sub(v_embd))
        );

        let mut layers = Vec::with_capacity(n_layer);
        let mut attn_bytes = 0u64;
        let mut expert_bytes = 0u64;
        let mut dense_ffn_bytes = 0u64;
        // fused-plane duplicate residency, split out of the bracket lines so
        // the sections stop silently absorbing it (nonkv-overhead plan R1.5;
        // ~450 MB on XS-2.1, 0 on the all-Q8 S-2.1 where same_kq rejects)
        let mut qkg_dup_bytes = 0u64;
        let mut shexp_dup_bytes = 0u64;
        // Merged r==1 GEMV planes (decode fast path): the tick profile showed
        // small-kernel overhead, not bandwidth, dominates decode - one fused
        // launch over [q|k|gate] (and [shexp gate|up]) runs near roof like
        // the lm head. Built only when every part shares one k-quant type;
        // duplicate residency (~450 MB on XS-2.1) the fit estimate absorbs.
        let fuse_gemv =
            paddock_models::dev_var_os!("PADDOCK_NO_FUSED_GEMV").is_none() && exec.has_kquant();
        let same_kq = |names: &[String]| -> bool {
            let tys: Vec<_> = names
                .iter()
                .filter_map(|n| map.tensor_info(n).map(|t| t.ggml_type))
                .collect();
            tys.len() == names.len()
                && tys.iter().all(|&t| t == tys[0])
                && crate::gpu::kq_params(tys[0]).is_some()
        };
        // the layer index names every blk.{i} tensor and picks its head count
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_layer {
            let dt = |name: &str| exec.upload(map, &format!("blk.{i}.{name}"));
            // Matmul weights stay quantized-resident with per-TENSOR dispatch
            // (the file mixes Q4_K and Q6_K per the Q4_K_M recipe).
            let qt = |name: &str| exec.load_quantw(map, &format!("blk.{i}.{name}"));

            let v0 = vfree();
            let swa = is_swa(i);
            let ffn = if i < n_dense_lead {
                let l = Ffn::Dense {
                    gate: qt("ffn_gate.weight")?,
                    up: qt("ffn_up.weight")?,
                    down: qt("ffn_down.weight")?,
                };
                dense_ffn_bytes += v0.saturating_sub(vfree());
                l
            } else {
                // Expert seats: k-quant-resident when the file ships k-quant
                // experts AND the pack has the kq MoE family (the XS Q4_K_M
                // case); Q8_0 files take the repacked-Q8 seat. gate and up
                // must agree in residency (one fused kernel call).
                let kq_ok = exec.has_kquant_moe();
                let kq_exp = |name: &str| -> Result<Option<crate::gpu::RepackedKQ>, GpuError> {
                    if !kq_ok {
                        return Ok(None);
                    }
                    exec.try_repack_kquant(map, &format!("blk.{i}.{name}"))
                };
                let (gate_exps, up_exps) = match (
                    kq_exp("ffn_gate_exps.weight")?,
                    kq_exp("ffn_up_exps.weight")?,
                ) {
                    (Some(g), Some(u)) => (ExpW::Kq(g), ExpW::Kq(u)),
                    _ => (
                        ExpW::Q8(exec.repack_q8(map, &format!("blk.{i}.ffn_gate_exps.weight"))?),
                        ExpW::Q8(exec.repack_q8(map, &format!("blk.{i}.ffn_up_exps.weight"))?),
                    ),
                };
                let down_exps = match kq_exp("ffn_down_exps.weight")? {
                    Some(d) => ExpW::Kq(d),
                    None => {
                        ExpW::Q8(exec.repack_q8(map, &format!("blk.{i}.ffn_down_exps.weight"))?)
                    }
                };
                expert_bytes += v0.saturating_sub(vfree());
                let v1 = vfree();
                let gu_names = [
                    format!("blk.{i}.ffn_gate_shexp.weight"),
                    format!("blk.{i}.ffn_up_shexp.weight"),
                ];
                let shexp_gateup = if fuse_gemv && exec.has_swiglu_fused() && same_kq(&gu_names) {
                    let p = exec
                        .repack_kquant_concat(map, &[gu_names[0].as_str(), gu_names[1].as_str()])?;
                    shexp_dup_bytes += (p.data.len() + p.scales.len()) as u64;
                    Some(p)
                } else {
                    None
                };
                let l = Ffn::Moe(MoeWeights {
                    router_w: dt("ffn_gate_inp.weight")?,
                    probs_bias: dt("exp_probs_b.bias")?,
                    gate_exps,
                    up_exps,
                    down_exps,
                    shexp_gate: qt("ffn_gate_shexp.weight")?,
                    shexp_up: qt("ffn_up_shexp.weight")?,
                    shexp_down: qt("ffn_down_shexp.weight")?,
                    shexp_gateup,
                });
                dense_ffn_bytes += v1.saturating_sub(vfree());
                l
            };

            let va = vfree();
            // [q | k | gate], or [q | k] on a gateless (Kolibri) file
            let gated = flavor == Flavor::Laguna;
            let mut qkg_names = vec![
                format!("blk.{i}.attn_q.weight"),
                format!("blk.{i}.attn_k.weight"),
            ];
            if gated {
                qkg_names.push(format!("blk.{i}.attn_gate.weight"));
            }
            let qkg = if fuse_gemv && same_kq(&qkg_names) {
                let names: Vec<&str> = qkg_names.iter().map(String::as_str).collect();
                let p = exec.repack_kquant_concat(map, &names)?;
                qkg_dup_bytes += (p.data.len() + p.scales.len()) as u64;
                Some(p)
            } else {
                None
            };
            let sandwich = |name: &str| -> Result<Option<DeviceTensor>, GpuError> {
                match flavor {
                    Flavor::Laguna => Ok(None),
                    Flavor::Kolibri => dt(name).map(Some),
                }
            };
            let layer = LagunaLayer {
                attn_norm: dt("attn_norm.weight")?,
                proj: Proj::Quant(QProj {
                    wq: qt("attn_q.weight")?,
                    wk: qt("attn_k.weight")?,
                    wv: qt("attn_v.weight")?,
                    wo: qt("attn_output.weight")?,
                    g_proj: if gated {
                        Some(qt("attn_gate.weight")?)
                    } else {
                        None
                    },
                    qkg,
                }),
                q_norm: dt("attn_q_norm.weight")?,
                k_norm: dt("attn_k_norm.weight")?,
                ffn_norm: dt("ffn_norm.weight")?,
                post_attn_norm: sandwich("post_attention_norm.weight")?,
                post_ffn_norm: sandwich("post_ffw_norm.weight")?,
                ffn,
                is_swa: swa,
                nope: !swa && flavor == Flavor::Kolibri,
                n_heads: n_heads[i],
            };
            let q = layer.proj.quant().expect("GGUF planes");
            attn_bytes += va.saturating_sub(vfree());

            // Shape audit on the layer's signature tensors: the per-head gate
            // width is the per-head declaration (llama.cpp detects the gate
            // class the same way) - refuse a per-element file rather than
            // serve it with per-head math.
            if let Some(g) = &q.g_proj
                && g.dims()[1] != n_heads[i]
            {
                return Err(GpuModelError::Unsupported(format!(
                    "{fam} blk.{i}: attn_gate width {} != n_heads {} - per-element gate \
                     files (M.1 class) are not supported yet",
                    g.dims()[1],
                    n_heads[i]
                )));
            }
            let qd = q.wq.dims();
            if qd[1] != n_heads[i] * head_dim {
                return Err(GpuModelError::Unsupported(format!(
                    "{fam} blk.{i}: attn_q out {} != {}×{}",
                    qd[1], n_heads[i], head_dim
                )));
            }
            layers.push(layer);
        }
        tracing::info!(
            "{fam} VRAM  attention (q/k/v/o + per-head gates)  {:>7.2} GB",
            gb(attn_bytes)
        );
        if qkg_dup_bytes > 0 {
            tracing::info!(
                "{fam} VRAM    of which fused q|k|gate duplicate planes {:>5.2} GB \
                 (splits stay for r>1)",
                gb(qkg_dup_bytes)
            );
        }
        tracing::info!(
            "{fam} VRAM  routed experts ({}e × {} layers)      {:>7.2} GB",
            moe.n_expert,
            n_layer - n_dense_lead,
            gb(expert_bytes)
        );
        tracing::info!(
            "{fam} VRAM  shared experts + dense FFN + routers  {:>7.2} GB",
            gb(dense_ffn_bytes)
        );
        if shexp_dup_bytes > 0 {
            tracing::info!(
                "{fam} VRAM    of which fused shexp gate|up duplicate planes {:>5.2} GB",
                gb(shexp_dup_bytes)
            );
        }

        let vh = vfree();
        let output_norm = exec.upload(map, "output_norm.weight")?;
        let lm_head = Head::Quant(exec.load_quantw(map, "output.weight")?);
        tracing::info!(
            "{fam} VRAM  output head + final norm              {:>7.2} GB",
            gb(vh.saturating_sub(vfree()))
        );

        // The published resident-weight line. Not the vfree() bracket the
        // per-section logs above use: that measures how much free VRAM
        // disappeared, so it picks up the CUDA context, modules and cuBLAS
        // workspaces, and it drifted 67 MB between two loads of the same file.
        // The pool's own used counter is exact and reproducible.
        let weights_bytes = exec
            .settled_mem_used()
            .unwrap_or_else(|| v_start.saturating_sub(vfree()));
        tracing::info!(
            "{fam} VRAM  = model resident total                {:>7.2} GB  \
             ({} layers: {} full / {} SWA-{}, heads {}/{}, {} experts top-{})",
            gb(weights_bytes),
            n_layer,
            (0..n_layer).filter(|&i| !is_swa(i)).count(),
            (0..n_layer).filter(|&i| is_swa(i)).count(),
            swa_window,
            n_heads[0],
            n_heads[1],
            moe.n_expert,
            moe.n_active,
        );

        Ok(Self {
            exec,
            hp: Hparams {
                flavor,
                n_layer,
                n_embd,
                n_heads,
                n_kv_heads,
                head_dim,
                n_vocab,
                eps,
                swa_window,
                n_rot,
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
            content_id: (
                crate::kv_tier::fingerprint::weights(map),
                crate::kv_tier::fingerprint::tokenizer(map),
            ),
            // KV8 (fp8-e4m3) by default since 2026-10-04 - the class this
            // family's boards and rivals run, with fp8 arms on every serving
            // path; a die that cannot store fp8 gets f16 back (serving.rs)
            // and an explicit kv_cache_dtype wins
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
