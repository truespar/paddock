//! Loading the official Inkling 2 Mini release into the lane's planes:
//! every BF16 tensor from its own bytes (the query input's q | relative
//! states | sinks fused into one plane), the experts' NVFP4 / MXFP8 bytes as
//! shipped (gate and up still interleaved), the f32 queries and biases.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::inkling2 as ik;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::{Controller, DYN, GpuInkling2, KV, Layer, Scratch, kv};
use crate::gpu::{GpuExecutor, Inkling2Experts, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::{bf16_concat_plane, bf16_plane, f32_tensor};

fn unsupported(s: String) -> GpuModelError {
    GpuModelError::Unsupported(s)
}

/// A tensor's raw bytes after checking its dtype and shape.
fn raw<'a>(
    st: &'a ShardedSafetensors,
    name: &str,
    dtype: &[StDtype],
    shape: &[usize],
) -> Result<&'a [u8], GpuModelError> {
    let (t, bytes) = st
        .bytes(name)
        .ok_or_else(|| unsupported(format!("{name}: tensor missing")))?;
    if !dtype.contains(&t.dtype) || t.shape != shape {
        return Err(unsupported(format!(
            "{name}: {:?} {:?}, expected one of {dtype:?} {shape:?}",
            t.dtype, t.shape
        )));
    }
    Ok(bytes)
}

fn dev_u8(exec: &GpuExecutor, b: &[u8]) -> Result<CudaSlice<u8>, GpuModelError> {
    exec.to_device_u8(b).map_err(GpuModelError::from)
}

fn dev_f32(exec: &GpuExecutor, v: &[f32]) -> Result<CudaSlice<f32>, GpuModelError> {
    exec.to_device(v).map_err(GpuModelError::from)
}

fn bf16_bytes_shaped(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    shape: &[usize],
) -> Result<CudaSlice<u8>, GpuModelError> {
    dev_u8(exec, raw(st, name, &[StDtype::Bf16], shape)?)
}

/// A layer's routed experts as shipped: w13 `[E][2 ff][..]` (gate and up
/// rows interleaved), w2 `[E][hidden][..]`, the NVFP4 per-expert factor
/// `amax / outer_scale_denominator`.
fn experts(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    i: usize,
) -> Result<Inkling2Experts, GpuModelError> {
    let (e, h, ff) = (ik::EXPERTS, ik::WIDTH, ik::FF);
    let p = format!("model.llm.layers.{i}.mlp.experts");
    let fp8 = [StDtype::F8E4m3, StDtype::U8];
    if i < ik::MXFP8_FROM {
        let w13 = raw(st, &format!("{p}.w13_weight"), &fp8, &[e, 2 * ff, h / 2])?;
        let s13 = raw(
            st,
            &format!("{p}.w13_weight_scale"),
            &fp8,
            &[e, 2 * ff, h / 16],
        )?;
        let w2 = raw(st, &format!("{p}.w2_weight"), &fp8, &[e, h, ff / 2])?;
        let s2 = raw(st, &format!("{p}.w2_weight_scale"), &fp8, &[e, h, ff / 16])?;
        let fac = |w: &str| -> Result<Vec<f32>, GpuModelError> {
            let amax = f32_tensor(st, &format!("{p}.{w}_weight_amax"), e)?;
            let den = f32_tensor(st, &format!("{p}.{w}_weight_outer_scale_denominator"), 1)?[0];
            Ok(amax.iter().map(|a| a / den).collect())
        };
        Ok(Inkling2Experts {
            fmt: 0,
            w13: dev_u8(exec, w13)?,
            s13: dev_u8(exec, s13)?,
            f13: Some(dev_f32(exec, &fac("w13")?)?),
            w2: dev_u8(exec, w2)?,
            s2: dev_u8(exec, s2)?,
            f2: Some(dev_f32(exec, &fac("w2")?)?),
            n_expert: e,
            hidden: h,
            ff,
        })
    } else {
        let w13 = raw(st, &format!("{p}.w13_weight"), &fp8, &[e, 2 * ff, h])?;
        let s13 = raw(
            st,
            &format!("{p}.w13_weight_scale"),
            &[StDtype::U8],
            &[e, 2 * ff, h / 32],
        )?;
        let w2 = raw(st, &format!("{p}.w2_weight"), &fp8, &[e, h, ff])?;
        let s2 = raw(
            st,
            &format!("{p}.w2_weight_scale"),
            &[StDtype::U8],
            &[e, h, ff / 32],
        )?;
        Ok(Inkling2Experts {
            fmt: 1,
            w13: dev_u8(exec, w13)?,
            s13: dev_u8(exec, s13)?,
            f13: None,
            w2: dev_u8(exec, w2)?,
            s2: dev_u8(exec, s2)?,
            f2: None,
            n_expert: e,
            hidden: h,
            ff,
        })
    }
}

fn layer(exec: &GpuExecutor, st: &ShardedSafetensors, i: usize) -> Result<Layer, GpuModelError> {
    let (w, heads, hd) = (ik::WIDTH, ik::HEADS, ik::HEAD_DIM);
    let full = ik::full_attention(i);
    let p = format!("model.llm.layers.{i}");
    let a = format!("{p}.attn");
    let qrs = bf16_concat_plane(
        exec,
        st,
        &[
            (&format!("{a}.wq_du.weight"), heads * hd),
            (&format!("{a}.wr_du.weight"), heads * ik::D_REL),
            (&format!("{a}.attn_sink_proj.weight"), heads),
        ],
        w,
    )?;
    let gate = bf16_plane(exec, st, &format!("{a}.gate_attn.weight"), heads * hd, w)?;
    let kv = |n: &str| -> Result<QuantTensor, GpuModelError> {
        let mut parts = vec![(format!("{a}.w{n}_dv.weight"), KV)];
        if full {
            parts.push((format!("{a}.{n}_sconv.weight_proj"), DYN));
        }
        let refs: Vec<(&str, usize)> = parts.iter().map(|(s, r)| (s.as_str(), *r)).collect();
        bf16_concat_plane(exec, st, &refs, w)
    };
    let k = kv("k")?;
    let v = kv("v")?;
    let o = bf16_plane(exec, st, &format!("{a}.wo_ud.weight"), w, heads * hd)?;
    let q_scale = bf16_bytes_shaped(exec, st, &format!("{a}.q_l2_clip.scale"), &[heads, hd])?;
    let rel_scale = bf16_bytes_shaped(
        exec,
        st,
        &format!("{a}.rel_l2_clip.scale"),
        &[heads, ik::D_REL],
    )?;
    let bank = raw(
        st,
        &format!("{a}.rel_logits_proj.proj"),
        &[StDtype::Bf16],
        &[ik::D_REL, ik::REL_EXTENT],
    )?;
    let rel_bank = dev_u8(exec, bank)?;
    // the prompt-row attention reads a distance's 16 weights together
    let mut bank_t = vec![0u8; bank.len()];
    for i in 0..ik::D_REL {
        for d in 0..ik::REL_EXTENT {
            let (src, dst) = (2 * (i * ik::REL_EXTENT + d), 2 * (d * ik::D_REL + i));
            bank_t[dst..dst + 2].copy_from_slice(&bank[src..src + 2]);
        }
    }
    let rel_bank_t = dev_u8(exec, &bank_t)?;
    let conv = |n: &str| -> Result<CudaSlice<u8>, GpuModelError> {
        if full {
            bf16_bytes_shaped(
                exec,
                st,
                &format!("{a}.{n}_sconv.static_w"),
                &[ik::SCONV_K, KV],
            )
        } else {
            bf16_bytes_shaped(
                exec,
                st,
                &format!("{a}.{n}_sconv.weight"),
                &[KV, 1, ik::SCONV_K],
            )
        }
    };
    let k_conv = conv("k")?;
    let v_conv = conv("v")?;
    let attn_conv = bf16_bytes_shaped(
        exec,
        st,
        &format!("{p}.attn_sconv.weight"),
        &[w, 1, ik::SCONV_K],
    )?;
    let mlp_conv = bf16_bytes_shaped(
        exec,
        st,
        &format!("{p}.mlp_sconv.weight"),
        &[w, 1, ik::SCONV_K],
    )?;
    let norm = |n: &str| -> Result<CudaSlice<f32>, GpuModelError> {
        dev_f32(exec, &f32_tensor(st, &format!("{p}.{n}.weight"), w)?)
    };
    let in_norm = if full {
        vec![
            norm("q_input_norm")?,
            norm("k_input_norm")?,
            norm("v_input_norm")?,
        ]
    } else {
        vec![norm("attn_norm")?]
    };
    let mlp_norm = norm("mlp_norm")?;
    let attn_q = if i == 0 {
        None
    } else {
        let nq = if full { 3 } else { 1 };
        Some(dev_f32(
            exec,
            &f32_tensor(
                st,
                &format!("model.llm.attn_res.attn_queries.{}", i - 1),
                nq * w,
            )?,
        )?)
    };
    let mlp_q = dev_f32(
        exec,
        &f32_tensor(st, &format!("model.llm.attn_res.mlp_queries.{i}"), w)?,
    )?;
    let router = bf16_plane(exec, st, &format!("{p}.mlp.gate.weight"), ik::EXPERTS, w)?;
    let router_bias = dev_f32(
        exec,
        &f32_tensor(st, &format!("{p}.mlp.gate.bias"), ik::EXPERTS)?,
    )?;
    let ctrl = if ik::looped(i) {
        let c = |n: &str| -> Result<Controller, GpuModelError> {
            let pre = format!("{p}.controllers.{n}");
            Ok(Controller {
                down: bf16_plane(
                    exec,
                    st,
                    &format!("{pre}.down_weight"),
                    ik::GATE_RANK,
                    2 * w,
                )?,
                forget: bf16_bytes_shaped(
                    exec,
                    st,
                    &format!("{pre}.forget_weight"),
                    &[w, ik::GATE_RANK],
                )?,
                gain: bf16_bytes_shaped(
                    exec,
                    st,
                    &format!("{pre}.gain_weight"),
                    &[w, ik::GATE_RANK],
                )?,
            })
        };
        // the checkpoint's `attn` controller gates the query input
        Some([c("attn")?, c("k")?, c("v")?, c("mlp")?])
    } else {
        None
    };
    Ok(Layer {
        full,
        qrs,
        gate,
        k,
        v,
        o,
        q_scale,
        rel_scale,
        rel_bank,
        rel_bank_t,
        k_conv,
        v_conv,
        attn_conv,
        mlp_conv,
        in_norm,
        mlp_norm,
        attn_q,
        mlp_q,
        router,
        router_bias,
        experts: experts(exec, st, i)?,
        ctrl,
    })
}

impl GpuInkling2 {
    /// Load the official release from `dir` (its picture tower too unless
    /// `PADDOCK_NO_VISION`): sequences of up to
    /// `max_ctx` positions, prompt chunks of up to `chunk` rows, K / V rows
    /// in class `kv_dtype` (0 f16, 1 fp8 e4m3, 2 f32). One slot until
    /// `enable_batch` seats more.
    pub fn load(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_ctx: usize,
        chunk: usize,
        kv_dtype: u32,
    ) -> Result<Self, GpuModelError> {
        let cfg = ik::Inkling2Config::read(dir).map_err(unsupported)?;
        if !exec.has_inkling2() || !exec.has_bf16_dense() || !exec.has_embed_gather_bf16() {
            return Err(GpuModelError::Unsupported(
                "Inkling 2 needs pack slots 977-990 and the bf16 lane".into(),
            ));
        }
        let st = ShardedSafetensors::open_dir(dir).map_err(|e| unsupported(e.to_string()))?;
        let w = ik::WIDTH;
        let e = &*exec;
        let held0 = e.process_mem_used().unwrap_or(0);
        let embed = QuantTensor {
            bytes: dev_u8(
                e,
                raw(
                    &st,
                    "model.llm.embed.weight",
                    &[StDtype::Bf16],
                    &[ik::VOCAB, w],
                )?,
            )?,
            ty: paddock_models::ggml_type::GgmlType::Bf16,
            dims: vec![w, ik::VOCAB],
        };
        let unembed = bf16_plane(e, &st, "model.llm.unembed.weight", ik::VOCAB, w)?;
        let embed_norm = dev_f32(e, &f32_tensor(&st, "model.llm.embed_norm.weight", w)?)?;
        let norm = dev_f32(e, &f32_tensor(&st, "model.llm.norm.weight", w)?)?;
        let final_q = dev_f32(e, &f32_tensor(&st, "model.llm.attn_res.output_query", w)?)?;
        // the checkpoint's two [2 * loop layers, width] query tensors: (q,
        // mlp) and (k, v) of each loop layer -> [q, k, v, mlp] a layer
        let core = ik::LOOP_END - ik::LOOP_START;
        let qm = f32_tensor(
            &st,
            "model.llm.middle_cycle_cross_pass_queries",
            2 * core * w,
        )?;
        let kvq = f32_tensor(
            &st,
            "model.llm.middle_cycle_cross_pass_kv_queries",
            2 * core * w,
        )?;
        let msg_q = (0..core)
            .map(|l| {
                let row = |t: &[f32], j: usize| t[(2 * l + j) * w..(2 * l + j + 1) * w].to_vec();
                let v: Vec<f32> = [row(&qm, 0), row(&kvq, 0), row(&kvq, 1), row(&qm, 1)].concat();
                dev_f32(e, &v)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layers = (0..ik::LAYERS)
            .map(|i| layer(e, &st, i))
            .collect::<Result<Vec<_>, _>>()?;
        e.synchronize()?;
        let target_bytes = e.process_mem_used().unwrap_or(0).saturating_sub(held0);
        let mut draft = super::draft::Draft::load(e, dir, &unembed)?;
        let on = draft.is_some();
        // the draft's planes count as weights (they are resident whenever it
        // is attached), its round scratch below does not
        e.synchronize()?;
        let weights_bytes = e.process_mem_used().unwrap_or(0).saturating_sub(held0);
        tracing::info!(
            target_bytes,
            draft_bytes = weights_bytes.saturating_sub(target_bytes),
            "inkling2 weights: {:.2} GiB ({:.2} GiB of it the DSpark draft)",
            weights_bytes as f64 / (1u64 << 30) as f64,
            weights_bytes.saturating_sub(target_bytes) as f64 / (1u64 << 30) as f64
        );
        if let Some(d) = draft.as_mut() {
            d.alloc_scratch(e, 1, chunk)?;
        }
        // the picture tower and its pass workspace, resident before the KV
        // plan reads what is free; `vision = false` (PADDOCK_NO_VISION)
        // leaves both unloaded
        let processor = std::fs::read(dir.join("processor_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        let has_tower = ik::vision::check(cfg.value(), processor.as_ref()).map_err(unsupported)?;
        let vision = if !has_tower {
            None
        } else if std::env::var_os("PADDOCK_NO_VISION").is_some() {
            tracing::info!("inkling2: PADDOCK_NO_VISION - the picture tower left unloaded");
            None
        } else {
            let before = e.process_mem_used().unwrap_or(0);
            let t = super::vision::Tower::load(exec.clone(), &st)?;
            e.synchronize()?;
            let held = e.process_mem_used().unwrap_or(0).saturating_sub(before);
            if let Some(t) = &t {
                tracing::info!(
                    held,
                    ws_bytes = t.ws_bytes,
                    "inkling2 picture tower: {:.2} GiB held, {:.2} GiB of it the pass \
                     workspace ({} rows)",
                    held as f64 / (1u64 << 30) as f64,
                    t.ws_bytes as f64 / (1u64 << 30) as f64,
                    super::vision::PASS_ROWS
                );
            }
            t
        };
        let kv = kv::Kv::alloc(e, 1, max_ctx, chunk, kv_dtype, max_ctx.div_ceil(16), on)?;
        let scratch = Scratch::alloc(e, chunk, on)?;
        Ok(Self {
            exec,
            embed,
            embed_norm,
            norm,
            final_q,
            msg_q,
            unembed,
            layers,
            kv,
            draft,
            last_verify: Vec::new(),
            draft_margins: None,
            queue: Default::default(),
            vision,
            prefix: None,
            overlay: Vec::new(),
            graphs: Default::default(),
            graph_seen: Default::default(),
            scratch,
            max_ctx,
            chunk,
            weights_bytes,
            trace_rows: 0,
            trace: Vec::new(),
            stages: Vec::new(),
        })
    }
}
