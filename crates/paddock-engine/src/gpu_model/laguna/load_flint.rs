//! Flint's GGUF (`flint` arch, the NVFP4 build) onto the laguna body.
//! Verified against `flint-nvfp4.gguf` (RecursiveSI/flint-nvfp4-gguf) and
//! llama.cpp's `src/models/flint.cpp`: every projection ships as ggml NVFP4
//! with its per-tensor F32 factor in a sibling `.scale` tensor ([1], or one
//! per expert on the stacked planes); the router and token embeddings ship
//! BF16; norms and sinks F32. Every metadata key read here is in the file;
//! anything off-spec is a named refusal, never a default that changes math.

use std::sync::Arc;

use paddock_kernels::reference::ops::YarnRope;
use paddock_models::ggml_type::GgmlType;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuExecutor, KvDtype, Nvf4MoePlane, Nvf4Plane};
use crate::gpu_model::gpt_oss::GpuModelError;

use super::*;

/// llama.cpp's `expert_gating_func` for sigmoid scoring.
const GATING_FUNC_SIGMOID: u64 = 2;

/// A GGUF NVFP4 tensor's dims (ggml order: in first) and its per-tensor
/// factors, read from the sibling `.scale` tensor.
fn nv_parts<'a>(
    map: &'a MappedGguf,
    name: &str,
) -> Result<(Vec<usize>, &'a [u8], Vec<f32>), GpuModelError> {
    let (info, bytes) = map
        .tensor_bytes(name)
        .map_err(|e| GpuModelError::Unsupported(format!("flint: {name}: {e}")))?;
    if info.ggml_type != GgmlType::Nvfp4 {
        return Err(GpuModelError::Unsupported(format!(
            "flint: {name} is {:?}, this build serves NVFP4 projections only",
            info.ggml_type
        )));
    }
    let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    let scale_name = name
        .strip_suffix(".weight")
        .map(|b| format!("{b}.scale"))
        .ok_or_else(|| GpuModelError::MissingMeta(format!("{name}: not a .weight tensor")))?;
    let (sinfo, sbytes) = map
        .tensor_bytes(&scale_name)
        .map_err(|e| GpuModelError::Unsupported(format!("flint: {scale_name}: {e}")))?;
    if sinfo.ggml_type != GgmlType::F32 {
        return Err(GpuModelError::Unsupported(format!(
            "flint: {scale_name} is {:?}, expected F32",
            sinfo.ggml_type
        )));
    }
    let scale2: Vec<f32> = sbytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if scale2.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err(GpuModelError::Unsupported(format!(
            "flint: {scale_name} holds a non-positive or non-finite factor"
        )));
    }
    Ok((dims, bytes, scale2))
}

/// The ggml blocks of `rows` x `in_dim` in the house row layout, or a named
/// refusal; logs any NaN scale bytes it zeroed (ggml reads them as 0).
fn repack(
    name: &str,
    bytes: &[u8],
    rows: usize,
    in_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>), GpuModelError> {
    let (codes, scales, stats) = paddock_models::gguf_nvfp4::repack_rows(bytes, rows, in_dim)
        .map_err(|e| GpuModelError::Unsupported(format!("flint: {name}: {e}")))?;
    if stats.nan_scales > 0 {
        tracing::warn!(
            tensor = name,
            count = stats.nan_scales,
            "NVFP4 scale bytes 0x7F read as zero (the format's NaN; ggml decodes it as 0)"
        );
    }
    Ok((codes, scales))
}

/// One dense NVFP4 plane `[out, in]`.
fn nv_plane(exec: &GpuExecutor, map: &MappedGguf, name: &str) -> Result<Nvf4Plane, GpuModelError> {
    let (dims, bytes, scale2) = nv_parts(map, name)?;
    let [in_dim, out_dim] = dims[..] else {
        return Err(GpuModelError::Unsupported(format!(
            "flint: {name} has {} dims, a dense plane has 2",
            dims.len()
        )));
    };
    if scale2.len() != 1 {
        return Err(GpuModelError::Unsupported(format!(
            "flint: {name} carries {} factors, a dense plane has 1",
            scale2.len()
        )));
    }
    let (codes, scales) = repack(name, bytes, out_dim, in_dim)?;
    Ok(exec.nvf4_upload(&codes, &scales, scale2[0], out_dim, in_dim)?)
}

/// One stacked expert plane `[n_expert][ff, in]` with a factor per expert.
fn nv_experts(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    n_expert: usize,
) -> Result<Nvf4MoePlane, GpuModelError> {
    let (dims, bytes, scale2) = nv_parts(map, name)?;
    let [in_dim, ff, ne] = dims[..] else {
        return Err(GpuModelError::Unsupported(format!(
            "flint: {name} has {} dims, an expert stack has 3",
            dims.len()
        )));
    };
    if ne != n_expert || scale2.len() != n_expert {
        return Err(GpuModelError::Unsupported(format!(
            "flint: {name} stacks {ne} experts with {} factors, the header says {n_expert}",
            scale2.len()
        )));
    }
    let (codes, scales) = repack(name, bytes, ne * ff, in_dim)?;
    Ok(exec.nvf4_moe_upload(&codes, &scales, &scale2, ne, ff, in_dim)?)
}

/// A per-layer bool array from the header, true = the property holds.
fn bool_arr(map: &MappedGguf, key: &str, n: usize) -> Result<Vec<bool>, GpuModelError> {
    match map.gguf().arch_field(key) {
        Some(Value::Array(items)) if items.len() == n => items
            .iter()
            .map(|v| match v {
                Value::Bool(b) => Ok(*b),
                _ => Err(GpuModelError::MissingMeta(key.into())),
            })
            .collect(),
        _ => Err(GpuModelError::MissingMeta(key.into())),
    }
}

impl GpuLaguna {
    /// Load a `flint` GGUF. `max_ctx` bounds the serial lane's KV.
    pub(super) fn load_flint(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        let fam = Flavor::Flint.name();
        // the W4A16 GEMV floor every width below the tensor-core band rides
        if !exec.has_nvf4_ckpt() || !exec.has_q4x_moe_gu_swiglu() || !exec.has_nvf4_gemv_batch() {
            let (maj, min) = exec.compute_capability();
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: sm_{maj}{min} / this kernel pack has no NVFP4 W4A16 lane"
            )));
        }
        exec.vram_load_gate(map.total_len(), fam)
            .map_err(GpuModelError::WontFit)?;
        // single-stream engine (see load_with)
        exec.disable_event_tracking();

        let u = |k: &str| {
            map.gguf()
                .arch_field(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let f = |k: &str| map.gguf().arch_field(k).and_then(Value::as_f32);
        let n_layer = u("block_count")? as usize;
        let n_embd = u("embedding_length")? as usize;
        let n_heads = u("attention.head_count")? as usize;
        let n_kv_heads = u("attention.head_count_kv")? as usize;
        let head_dim = u("attention.key_length")? as usize;
        if u("attention.value_length")? as usize != head_dim {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: value_length != key_length is not a shipped geometry"
            )));
        }
        let eps = f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let ctx_train = u("context_length")? as usize;
        let swa_window = u("attention.sliding_window")? as usize;
        // true = sliding; the rope pattern must be the same array - RoPE on
        // the SWA layers, none on the full ones (flint.cpp has_rope(il))
        let swa = bool_arr(map, "attention.sliding_window_pattern", n_layer)?;
        let rope = bool_arr(map, "attention.rope_pattern", n_layer)?;
        if swa != rope || swa_window == 0 {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: rope on a layer other than the sliding ones (or no window) is not a \
                 shipped geometry"
            )));
        }

        let gating = u("expert_gating_func")?;
        if gating != GATING_FUNC_SIGMOID {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: expert_gating_func {gating} (only sigmoid = 2 shipped)"
            )));
        }
        // the sigmoid router renormalizes always; a file that says it
        // does not asks for math this lane does not do
        if !matches!(
            map.gguf().arch_field("expert_weights_norm"),
            Some(Value::Bool(true))
        ) {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: expert_weights_norm false is not shipped"
            )));
        }
        let moe = MoeDims {
            n_expert: u("expert_count")? as usize,
            n_active: u("expert_used_count")? as usize,
            moe_ff: u("expert_feed_forward_length")? as usize,
            shexp_ff: u("expert_shared_feed_forward_length")? as usize,
            routed_scale: f("expert_weights_scale").unwrap_or(1.0),
            router: Router::SigmoidNorm,
        };
        let n_dense_lead = u("leading_dense_block_count")? as usize;
        let rope_base = f("rope.freq_base")
            .ok_or_else(|| GpuModelError::MissingMeta("rope.freq_base".into()))?;
        // SWA layers rotate the whole head (NEOX, no scaling); the full
        // layers are NoPE, so rope_full is built for the shape and never read
        let rope_swa =
            YarnRope::new(head_dim, rope_base, 1.0, ctx_train, 0.0, 1.0, 32.0, 1.0).kernel_params();
        let rope_full = rope_swa;

        let vfree = || {
            cudarc::driver::result::mem_get_info()
                .map(|(f, _)| f as u64)
                .unwrap_or(0)
        };
        let gb = |used: u64| used as f64 / 1e9;
        let v_start = vfree();

        let te = exec.upload_raw(map, "token_embd.weight")?;
        if te.ty != GgmlType::Bf16 {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: token_embd.weight is {:?}, this build gathers BF16",
                te.ty
            )));
        }
        let n_vocab = te.dims[1];
        let tok_embd = TokEmbd::Bf16(te);
        tracing::info!(
            "{fam} VRAM  input embeddings BF16                  {:>7.2} GB",
            gb(v_start.saturating_sub(vfree()))
        );

        let mut layers = Vec::with_capacity(n_layer);
        let (mut attn_bytes, mut expert_bytes, mut ffn_bytes) = (0u64, 0u64, 0u64);
        for (i, &is_swa) in swa.iter().enumerate() {
            let dt = |name: &str| exec.upload(map, &format!("blk.{i}.{name}"));
            let nv = |name: &str| nv_plane(&exec, map, &format!("blk.{i}.{name}"));
            let va = vfree();
            let proj = Proj::Nv {
                wq: nv("attn_q.weight")?,
                wk: nv("attn_k.weight")?,
                wv: nv("attn_v.weight")?,
                wo: nv("attn_output.weight")?,
            };
            let Proj::Nv { wq, wk, .. } = &proj else {
                unreachable!("built above")
            };
            if wq.out_dim != n_heads * head_dim || wk.out_dim != n_kv_heads * head_dim {
                return Err(GpuModelError::Unsupported(format!(
                    "{fam} blk.{i}: q/k widths {} / {} != {n_heads} / {n_kv_heads} heads x {head_dim}",
                    wq.out_dim, wk.out_dim
                )));
            }
            let sinks = dt("attn_sinks.weight")?;
            if sinks.element_count() != n_heads {
                return Err(GpuModelError::Unsupported(format!(
                    "{fam} blk.{i}: {} sinks for {n_heads} heads",
                    sinks.element_count()
                )));
            }
            attn_bytes += va.saturating_sub(vfree());
            let vf = vfree();
            let ffn = if i < n_dense_lead {
                let l = Ffn::DenseNv {
                    gate: nv("ffn_gate.weight")?,
                    up: nv("ffn_up.weight")?,
                    down: nv("ffn_down.weight")?,
                };
                ffn_bytes += vf.saturating_sub(vfree());
                l
            } else {
                let ex =
                    |name: &str| nv_experts(&exec, map, &format!("blk.{i}.{name}"), moe.n_expert);
                let (gate, up, down) = (
                    ex("ffn_gate_exps.weight")?,
                    ex("ffn_up_exps.weight")?,
                    ex("ffn_down_exps.weight")?,
                );
                expert_bytes += vf.saturating_sub(vfree());
                let vs = vfree();
                let router = format!("blk.{i}.ffn_gate_inp.weight");
                let l = Ffn::MoeNv(MoeNv {
                    // the BF16 router widened exactly (prefill widths' f32
                    // GEMM) and as shipped (the decode widths' GEMVs)
                    router_w: exec.upload(map, &router)?,
                    router_b16: exec.upload_raw(map, &router)?,
                    // no selection bias ships: zeros select on the scores
                    probs_bias: DeviceTensor {
                        buf: exec.to_device(&vec![0.0f32; moe.n_expert])?,
                        dims: vec![moe.n_expert],
                    },
                    gate,
                    up,
                    down,
                    shared: SharedNv::Nv {
                        gate: nv("ffn_gate_shexp.weight")?,
                        up: nv("ffn_up_shexp.weight")?,
                        down: nv("ffn_down_shexp.weight")?,
                    },
                });
                ffn_bytes += vs.saturating_sub(vfree());
                l
            };
            layers.push(LagunaLayer {
                attn_norm: dt("attn_norm.weight")?,
                proj,
                q_norm: dt("attn_q_norm.weight")?,
                k_norm: dt("attn_k_norm.weight")?,
                ffn_norm: dt("ffn_norm.weight")?,
                post_attn_norm: None,
                post_ffn_norm: None,
                sinks: Some(sinks),
                ffn,
                is_swa,
                nope: !is_swa,
                n_heads,
            });
        }
        let vh = vfree();
        let output_norm = exec.upload(map, "output_norm.weight")?;
        let lm_head = Head::Nv(nv_plane(&exec, map, "output.weight")?);
        tracing::info!(
            "{fam} VRAM  attention NVFP4 (q/k/v/o) + sinks      {:>7.2} GB",
            gb(attn_bytes)
        );
        tracing::info!(
            "{fam} VRAM  routed experts NVFP4 ({}e x {} layers) {:>7.2} GB",
            moe.n_expert,
            n_layer - n_dense_lead,
            gb(expert_bytes)
        );
        tracing::info!(
            "{fam} VRAM  shared + dense FFN NVFP4, routers BF16 {:>7.2} GB",
            gb(ffn_bytes)
        );
        tracing::info!(
            "{fam} VRAM  output head NVFP4 + final norm         {:>7.2} GB",
            gb(vh.saturating_sub(vfree()))
        );
        let weights_bytes = exec
            .settled_mem_used()
            .unwrap_or_else(|| v_start.saturating_sub(vfree()));
        tracing::info!(
            "{fam} VRAM  = model resident total                {:>7.2} GB  \
             ({n_layer} layers: {} full / {} SWA-{swa_window}, {n_heads} heads over {n_kv_heads}, \
             {} experts top-{}, W4A16)",
            gb(weights_bytes),
            swa.iter().filter(|s| !**s).count(),
            swa.iter().filter(|s| **s).count(),
            moe.n_expert,
            moe.n_active,
        );

        Ok(Self {
            exec,
            hp: Hparams {
                flavor: Flavor::Flint,
                n_layer,
                n_embd,
                n_heads: vec![n_heads; n_layer],
                n_kv_heads,
                head_dim,
                n_vocab,
                eps,
                swa_window,
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
            content_id: (
                crate::kv_tier::fingerprint::weights(map),
                crate::kv_tier::fingerprint::tokenizer(map),
            ),
            // Recursive recommends f16 KV, and at 14 full layers x 4 KV
            // heads the cache is small enough that fp8 buys little
            kv_dtype: KvDtype::Fp16,
            decode: None,
            scratch: None,
            batch: None,
            pipe: None,
            dflash: None,
            chunked: Vec::new(),
            w16r: None,
        })
    }
}
