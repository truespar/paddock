//! Nemotron NVFP4 checkpoint loader - safetensors dir straight to device
//! planes, no GGUF. Every residency is byte-exact to the checkpoint:
//! NVFP4 triples ride packed (per-expert scale2 arrays for the MoE planes),
//! mamba FP8 planes keep their e4m3 bytes with the per-tensor scale
//! broadcast, embeddings and the attention decode twins keep their bf16
//! bytes, and every other bf16 tensor widens to f32 (exact).

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;

use crate::gpu::{DeviceTensor, GpuError, GpuExecutor, KvDtype, Nvf4MoePlane, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use paddock_models::ggml_type::GgmlType;
use paddock_models::modelopt::{ModeloptQuantMap, fp8_view, nvfp4_view};
use paddock_models::nemotron::{NemotronBlock, NemotronConfig};
use paddock_models::safetensors::ShardedSafetensors;

use super::*;

use crate::gpu_model::st_load::{bf16_bytes, f32_tensor};

/// Small-die (under 128 SMs) serve defaults for this family, installed as
/// process env at load exactly like the qwen35 default stack: the pack and
/// service launchers latch getenv, a paddock process serves one model, and
/// an explicit env value always wins (only unset vars are filled).
///
/// `PADDOCK_PIPE_MIN_LIVE=9`: the side-stream decode paths (the overlap
/// scheduler and the classic decode pipe) engage only from 9 requests in
/// flight. On GB10 (48 SMs, LPDDR5X) a decode tick is weight streaming and
/// the two lanes time-slice the same ~240 GB/s; measured on the nemotron
/// NVFP4 lane at 8 slots (2026-09-11): 1024x1024 c1 TTFT
/// 265 -> 240 ms, 128x128 c1 98 -> 90, c8 decode +2..3%, same finding as
/// the qwen35 lane's 2026-09-10 audit. The 188-SM die keeps 0 (both paths
/// measured to win there).
fn small_die_defaults(exec: &GpuExecutor) {
    if exec.sm_count() >= 128 {
        return;
    }
    // The batched decode pipe's live-slot floor. 9 (= never at 8 slots) was
    // measured before the whole-prompt/prompt-aligned ticks and the reply
    // checkpoint reshaped the c8 tick; re-measured after them (the pipe
    // floor), 8 buys 1024x1024 c8 +2.5% and the
    // agentic c8 turn -45 ms TTFT with every other cell unchanged.
    if std::env::var_os("PADDOCK_PIPE_MIN_LIVE").is_none() {
        crate::envset::set_env("PADDOCK_PIPE_MIN_LIVE", "8");
    }
    // Speculation is NOT defaulted off here any more (2026-10-04). It used to
    // be, measured when a spec-capable serve meant the n-gram drafter: at 37%
    // acceptance on agentic prose the c1 tick was 17.0 ms/token against the
    // pipe's 11.6. Since then the family is spec-capable only with a MODEL
    // drafter attached (`spec_capable`: DSpark/DFlash sideload or the in-file
    // MTP), and those win on this die - DSpark on GB10: tool-carrying sampled
    // 126.5 tok/s, code 129.7, against ~83 without speculation; a Claude Code
    // replay to 248K decodes at 105.6 tok/s. The blanket kill set here at
    // load, before any drafter attached, turned all of that off for every
    // serve that did not say `spec` explicitly.
}

impl GpuNemotron {
    /// Load the checkpoint directory onto the device. `max_ctx` bounds the
    /// serial lane's KV allocation (attention layers only - 6 of 52).
    pub fn load_dir(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        let hp = NemotronConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("nemotron config: {e}")))?;
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("nemotron shards: {e}")))?;
        let quant = ModeloptQuantMap::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("nemotron quant map: {e}")))?;
        if quant.is_empty() {
            return Err(GpuModelError::Unsupported(
                "nemotron: checkpoint has no modelopt quantization_config".into(),
            ));
        }

        let up = |exec: &GpuExecutor, v: Vec<f32>| -> Result<CudaSlice<f32>, GpuError> {
            exec.to_device(&v)
        };
        let dt =
            |exec: &GpuExecutor, v: Vec<f32>, dims: Vec<usize>| -> Result<DeviceTensor, GpuError> {
                Ok(DeviceTensor {
                    buf: exec.to_device(&v)?,
                    dims,
                })
            };

        let mut weights_bytes: u64 = 0;
        let mut count = |b: usize| weights_bytes += b as u64;
        // itemized duplicate residency (nonkv-overhead plan R1.5): both are
        // inside weights_bytes already - these split them out for the log
        let mut bf16_twin_bytes: u64 = 0;
        let mut moe_foldin_bytes: u64 = 0;

        // Same-binary kill switch for the bf16-resident lane (decode attn
        // GEMV twins + the embedding table); the widened-f32 residency is the
        // measured baseline leg.
        let bf16_lane = paddock_models::dev_var_os!("PADDOCK_NO_NEMO_BF16").is_none();

        // embeddings [vocab, hidden]: checkpoint bf16 bytes when the pack can
        // gather them (bit-identical rows - the widen happens in-kernel),
        // else widened to f32
        let tok_embd = if bf16_lane && exec.has_embed_gather_bf16() {
            let raw = bf16_bytes(&st, "backbone.embeddings.weight", hp.vocab * hp.hidden)?;
            count(hp.vocab * hp.hidden * 2);
            TokEmbd::Bf16(QuantTensor {
                bytes: exec.to_device_u8(raw).map_err(GpuModelError::from)?,
                ty: GgmlType::Bf16,
                dims: vec![hp.hidden, hp.vocab],
            })
        } else {
            let embd_host = f32_tensor(&st, "backbone.embeddings.weight", hp.vocab * hp.hidden)?;
            count(hp.vocab * hp.hidden * 4);
            TokEmbd::F32(up(&exec, embd_host)?)
        };

        let mut layers = Vec::with_capacity(hp.n_layer);
        // reusable concat buffers for the expert planes
        let mut cat_p: Vec<u8> = Vec::new();
        let mut cat_s: Vec<u8> = Vec::new();

        for li in 0..hp.n_layer {
            let pfx = format!("backbone.layers.{li}");
            let norm = dt(
                &exec,
                f32_tensor(&st, &format!("{pfx}.norm.weight"), hp.hidden)?,
                vec![hp.hidden],
            )?;
            count(hp.hidden * 4);

            let mixer = match hp.blocks[li] {
                NemotronBlock::Mamba => {
                    let ip = fp8_view(&st, &format!("{pfx}.mixer.in_proj"))
                        .map_err(|e| GpuModelError::Unsupported(format!("{pfx} in_proj: {e}")))?;
                    if (ip.n, ip.k) != (hp.in_proj_rows(), hp.hidden) {
                        return Err(GpuModelError::Unsupported(format!(
                            "{pfx} in_proj is [{}, {}]",
                            ip.n, ip.k
                        )));
                    }
                    let in_proj = exec.fp8_ckpt_to_f8row(ip.weight, ip.weight_scale, ip.k, ip.n)?;
                    count(ip.n * ip.k + ip.n * 4);
                    let op = fp8_view(&st, &format!("{pfx}.mixer.out_proj"))
                        .map_err(|e| GpuModelError::Unsupported(format!("{pfx} out_proj: {e}")))?;
                    if (op.n, op.k) != (hp.hidden, hp.d_inner()) {
                        return Err(GpuModelError::Unsupported(format!(
                            "{pfx} out_proj is [{}, {}]",
                            op.n, op.k
                        )));
                    }
                    let out_proj =
                        exec.fp8_ckpt_to_f8row(op.weight, op.weight_scale, op.k, op.n)?;
                    count(op.n * op.k + op.n * 4);

                    // conv weight ships [conv_dim, 1, k]; squeezed row-major
                    // that is exactly the kernels' [c, k] layout
                    let conv_w = up(
                        &exec,
                        f32_tensor(
                            &st,
                            &format!("{pfx}.mixer.conv1d.weight"),
                            hp.conv_dim() * hp.d_conv,
                        )?,
                    )?;
                    let conv_b = up(
                        &exec,
                        f32_tensor(&st, &format!("{pfx}.mixer.conv1d.bias"), hp.conv_dim())?,
                    )?;
                    count(hp.conv_dim() * (hp.d_conv + 1) * 4);

                    // A = -exp(A_log), the same load-time transform the
                    // reference applies (mamba_mixer2 a_weight_loader)
                    let a_log = f32_tensor(&st, &format!("{pfx}.mixer.A_log"), hp.mamba_heads)?;
                    let a = up(&exec, a_log.iter().map(|v| -v.exp()).collect())?;
                    let d = up(
                        &exec,
                        f32_tensor(&st, &format!("{pfx}.mixer.D"), hp.mamba_heads)?,
                    )?;
                    let dt_bias = up(
                        &exec,
                        f32_tensor(&st, &format!("{pfx}.mixer.dt_bias"), hp.mamba_heads)?,
                    )?;
                    let norm_w = up(
                        &exec,
                        f32_tensor(&st, &format!("{pfx}.mixer.norm.weight"), hp.d_inner())?,
                    )?;
                    count((3 * hp.mamba_heads + hp.d_inner()) * 4);

                    Mixer::Mamba(MambaWeights {
                        in_proj: LinW::F8(in_proj),
                        out_proj: LinW::F8(out_proj),
                        conv_w,
                        conv_b,
                        a,
                        d,
                        dt_bias,
                        norm_w,
                    })
                }
                NemotronBlock::Attention => {
                    let q_dim = hp.n_heads * hp.head_dim;
                    let kv_dim = hp.n_kv_heads * hp.head_dim;
                    let plane = |name: &str,
                                 out: usize,
                                 inn: usize|
                     -> Result<DeviceTensor, GpuModelError> {
                        let v = f32_tensor(&st, &format!("{pfx}.mixer.{name}.weight"), out * inn)?;
                        // row-major [out, in]; DeviceTensor dims are [in, out]
                        Ok(DeviceTensor {
                            buf: exec.to_device(&v).map_err(GpuModelError::from)?,
                            dims: vec![inn, out],
                        })
                    };
                    let wq = plane("q_proj", q_dim, hp.hidden)?;
                    let wk = plane("k_proj", kv_dim, hp.hidden)?;
                    let wv = plane("v_proj", kv_dim, hp.hidden)?;
                    let wo = plane("o_proj", hp.hidden, q_dim)?;
                    // q + o are both q_dim*hidden-sized, k + v kv_dim*hidden
                    // (an earlier form of this line triple-counted the o-class
                    // plane and over-reported ~264 MB across the 6 layers)
                    count((q_dim * hp.hidden + kv_dim * hp.hidden) * 2 * 4);
                    // checkpoint-byte twins for the decode GEMVs: half the
                    // per-tick DRAM on the same products (bf16 widens
                    // exactly); the f32 planes above keep prefill
                    // byte-identical
                    let bf16 = if bf16_lane && exec.has_bf16_dense() {
                        // q|k|v as one concatenated plane (row-major [out, in]
                        // makes the row concat a byte concat) - the batched
                        // tick's fused launch and the serial row's segment
                        // GEMVs both read it; see AttnBf16
                        let qraw = bf16_bytes(
                            &st,
                            &format!("{pfx}.mixer.q_proj.weight"),
                            q_dim * hp.hidden,
                        )?;
                        let kraw = bf16_bytes(
                            &st,
                            &format!("{pfx}.mixer.k_proj.weight"),
                            kv_dim * hp.hidden,
                        )?;
                        let vraw = bf16_bytes(
                            &st,
                            &format!("{pfx}.mixer.v_proj.weight"),
                            kv_dim * hp.hidden,
                        )?;
                        let mut fused = Vec::with_capacity(qraw.len() + kraw.len() + vraw.len());
                        fused.extend_from_slice(qraw);
                        fused.extend_from_slice(kraw);
                        fused.extend_from_slice(vraw);
                        let wqkv = QuantTensor {
                            bytes: exec.to_device_u8(&fused).map_err(GpuModelError::from)?,
                            ty: GgmlType::Bf16,
                            dims: vec![hp.hidden, q_dim + 2 * kv_dim],
                        };
                        let oraw = bf16_bytes(
                            &st,
                            &format!("{pfx}.mixer.o_proj.weight"),
                            hp.hidden * q_dim,
                        )?;
                        let wo = QuantTensor {
                            bytes: exec.to_device_u8(oraw).map_err(GpuModelError::from)?,
                            ty: GgmlType::Bf16,
                            dims: vec![q_dim, hp.hidden],
                        };
                        count((q_dim * hp.hidden + kv_dim * hp.hidden) * 2 * 2);
                        bf16_twin_bytes +=
                            ((q_dim * hp.hidden + kv_dim * hp.hidden) * 2 * 2) as u64;
                        Some(AttnBf16 {
                            wqkv,
                            q_dim,
                            kv_dim,
                            wo,
                        })
                    } else {
                        None
                    };
                    Mixer::Attn(AttnWeights::F32 {
                        wq,
                        wk,
                        wv,
                        wo,
                        bf16,
                    })
                }
                NemotronBlock::Moe => {
                    let router = dt(
                        &exec,
                        f32_tensor(
                            &st,
                            &format!("{pfx}.mixer.gate.weight"),
                            hp.n_expert * hp.hidden,
                        )?,
                        vec![hp.hidden, hp.n_expert],
                    )?;
                    let bias = dt(
                        &exec,
                        f32_tensor(
                            &st,
                            &format!("{pfx}.mixer.gate.e_score_correction_bias"),
                            hp.n_expert,
                        )?,
                        vec![hp.n_expert],
                    )?;
                    count(hp.n_expert * (hp.hidden + 1) * 4);

                    // Shared-expert fold-: when shared_ff is a
                    // clean multiple of moe_ff, the shared expert is also
                    // registered as `ns` pseudo-experts appended to the
                    // routed planes, so the batched path serves it inside
                    // the one sorted-tile launch (the separate 1-block
                    // shared pass measured 10-12% of the stream roof -
                    // bench/nv4moe_bench.cu). up: a row split - rows are
                    // independent through relu2 + the per-16-along-ff
                    // requant (moe_ff % 16 == 0 keeps fq/fs blocks intact,
                    // so shared fq bytes are identical to the 1-expert
                    // pass). down: a K split - each pseudo-expert takes a
                    // moe_ff-wide column slice and its output lands in its
                    // own combine slot; the fixed-order slot_combine fold
                    // is the sanctioned f32 regroup for the K sum.
                    let ns_sh = if hp.shared_ff % hp.moe_ff == 0 && hp.moe_ff % 32 == 0 {
                        hp.shared_ff / hp.moe_ff
                    } else {
                        0
                    };
                    // TILED-layout election: repack every
                    // MoE plane to the piece-major 64x64 tile order and serve
                    // the _st/_stw/_mtt consumer family (contiguous stages ->
                    // 512 B-class loads; bench: pair 79-83% -> 92-94% of the
                    // stream roof). Requires the full consumer set in the
                    // pack (cc12), 64-aligned dims on every plane, and the
                    // bs arm alive (the row-major kill switch reverts the
                    // whole family coherently - a tiled plane with any
                    // row-major reader left live would be silent garbage).
                    let moe_tiled = exec.has_nvf4_moe_st()
                        && hp.moe_ff % 64 == 0
                        && hp.hidden % 64 == 0
                        && hp.shared_ff % 64 == 0
                        && paddock_models::dev_var_os!("PADDOCK_NO_NVF4_MOE_ST").is_none()
                        && paddock_models::dev_var_os!("PADDOCK_NO_NVF4_MOE_BS").is_none();
                    let mut moe_plane =
                        |role: &str,
                         rows: usize,
                         in_dim: usize,
                         sh_row_split: bool|
                         -> Result<Nvf4MoePlane, GpuModelError> {
                            cat_p.clear();
                            cat_s.clear();
                            let mut s2 = Vec::with_capacity(hp.n_expert + ns_sh);
                            for e in 0..hp.n_expert {
                                let v = nvfp4_view(&st, &format!("{pfx}.mixer.experts.{e}.{role}"))
                                    .map_err(|err| {
                                        GpuModelError::Unsupported(format!(
                                            "{pfx} expert {e} {role}: {err}"
                                        ))
                                    })?;
                                if (v.n, v.k) != (rows, in_dim) {
                                    return Err(GpuModelError::Unsupported(format!(
                                        "{pfx} expert {e} {role} is [{}, {}], expected [{rows}, {in_dim}]",
                                        v.n, v.k
                                    )));
                                }
                                cat_p.extend_from_slice(v.packed);
                                cat_s.extend_from_slice(v.scales);
                                s2.push(v.scale2);
                            }
                            let mut ne = hp.n_expert;
                            if ns_sh > 0 {
                                let v =
                                    nvfp4_view(&st, &format!("{pfx}.mixer.shared_experts.{role}"))
                                        .map_err(|e| {
                                            GpuModelError::Unsupported(format!(
                                                "{pfx} shared {role}: {e}"
                                            ))
                                        })?;
                                let (want_n, want_k) = if sh_row_split {
                                    (ns_sh * rows, in_dim)
                                } else {
                                    (rows, ns_sh * in_dim)
                                };
                                if (v.n, v.k) != (want_n, want_k) {
                                    return Err(GpuModelError::Unsupported(format!(
                                        "{pfx} shared {role} is [{}, {}], expected [{want_n}, {want_k}]",
                                        v.n, v.k
                                    )));
                                }
                                if sh_row_split {
                                    // pseudo-experts are consecutive row spans -
                                    // already contiguous in the checkpoint plane
                                    cat_p.extend_from_slice(v.packed);
                                    cat_s.extend_from_slice(v.scales);
                                } else {
                                    // K split: pseudo-expert h takes columns
                                    // [h*in_dim, (h+1)*in_dim) of every row
                                    let rb = in_dim / 2;
                                    let sb = in_dim / 16;
                                    let vrb = v.k / 2;
                                    let vsb = v.k / 16;
                                    for h in 0..ns_sh {
                                        for r in 0..rows {
                                            cat_p.extend_from_slice(
                                                &v.packed[r * vrb + h * rb..r * vrb + (h + 1) * rb],
                                            );
                                            cat_s.extend_from_slice(
                                                &v.scales[r * vsb + h * sb..r * vsb + (h + 1) * sb],
                                            );
                                        }
                                    }
                                }
                                for _ in 0..ns_sh {
                                    s2.push(v.scale2);
                                }
                                ne += ns_sh;
                            }
                            if moe_tiled {
                                exec.nvf4_moe_upload_tiled(&cat_p, &cat_s, &s2, ne, rows, in_dim)
                            } else {
                                exec.nvf4_moe_upload(&cat_p, &cat_s, &s2, ne, rows, in_dim)
                            }
                            .map_err(GpuModelError::from)
                        };
                    let up_pl = moe_plane("up_proj", hp.moe_ff, hp.hidden, true)?;
                    let down_pl = moe_plane("down_proj", hp.hidden, hp.moe_ff, false)?;
                    count(hp.n_expert * (hp.moe_ff * hp.hidden + hp.moe_ff * hp.hidden / 8));
                    if ns_sh > 0 {
                        // the fold-in's appended shared copies (the shared
                        // planes below stay resident too, for the r=1 path)
                        count(hp.shared_ff * hp.hidden + hp.shared_ff * hp.hidden / 8);
                        moe_foldin_bytes +=
                            (hp.shared_ff * hp.hidden + hp.shared_ff * hp.hidden / 8) as u64;
                    }

                    let sh_plane = |role: &str,
                                    rows: usize,
                                    in_dim: usize|
                     -> Result<Nvf4MoePlane, GpuModelError> {
                        let v = nvfp4_view(&st, &format!("{pfx}.mixer.shared_experts.{role}"))
                            .map_err(|e| {
                                GpuModelError::Unsupported(format!("{pfx} shared {role}: {e}"))
                            })?;
                        if (v.n, v.k) != (rows, in_dim) {
                            return Err(GpuModelError::Unsupported(format!(
                                "{pfx} shared {role} is [{}, {}], expected [{rows}, {in_dim}]",
                                v.n, v.k
                            )));
                        }
                        if moe_tiled {
                            exec.nvf4_moe_upload_tiled(
                                v.packed,
                                v.scales,
                                &[v.scale2],
                                1,
                                rows,
                                in_dim,
                            )
                        } else {
                            exec.nvf4_moe_upload(v.packed, v.scales, &[v.scale2], 1, rows, in_dim)
                        }
                        .map_err(GpuModelError::from)
                    };
                    let sh_up = sh_plane("up_proj", hp.shared_ff, hp.hidden)?;
                    let sh_down = sh_plane("down_proj", hp.hidden, hp.shared_ff)?;
                    count(hp.shared_ff * hp.hidden + hp.shared_ff * hp.hidden / 8);

                    Mixer::Moe(MoeWeights {
                        router,
                        bias,
                        planes: MoePlanes::Nvf4 {
                            up: up_pl,
                            down: down_pl,
                            sh_up,
                            sh_down,
                        },
                    })
                }
            };
            layers.push(NemotronLayer { norm, mixer });
        }

        let final_norm = dt(
            &exec,
            f32_tensor(&st, "backbone.norm_f.weight", hp.hidden)?,
            vec![hp.hidden],
        )?;
        let lh = nvfp4_view(&st, "lm_head")
            .map_err(|e| GpuModelError::Unsupported(format!("lm_head: {e}")))?;
        if (lh.n, lh.k) != (hp.vocab, hp.hidden) {
            return Err(GpuModelError::Unsupported(format!(
                "lm_head is [{}, {}]",
                lh.n, lh.k
            )));
        }
        // lm_head residency election: FRAGMENT
        // order first - the tile-major blocks additionally permuted into
        // mma-fragment order, which puts the tensor-core head at marlin
        // parity on the kernel (probe 159.2 us b32 / 144.2 b8) - then
        // TILE-MAJOR, then row-major. Every twin set is bit-exact per class
        // vs the row-major family, and the head is the only non-row plane -
        // every other NVFP4 residency stays row-major. Kill switches:
        // PADDOCK_NO_NVF4_TF drops to the tiled lane, PADDOCK_NO_NVF4_TM to
        // row-major (so the whole prior election tree stays reachable).
        let k_ok = lh.k % 128 == 0;
        let tf = exec.has_nvf4_tf()
            && k_ok
            && paddock_models::dev_var_os!("PADDOCK_NO_NVF4_TF").is_none();
        let tm = exec.has_nvf4_tm()
            && k_ok
            && paddock_models::dev_var_os!("PADDOCK_NO_NVF4_TM").is_none();
        let lm_head = HeadW::Nvf4(if tf {
            exec.nvf4_upload_frag(lh.packed, lh.scales, lh.scale2, lh.n, lh.k)?
        } else if tm {
            exec.nvf4_upload_tiled(lh.packed, lh.scales, lh.scale2, lh.n, lh.k)?
        } else {
            exec.nvf4_upload(lh.packed, lh.scales, lh.scale2, lh.n, lh.k)?
        });
        count(lh.n * lh.k / 2 + lh.n * lh.k / 16 + hp.hidden * 4);

        let ssm_dtype = super::ssm_arena::ssm_dtype_from_env();
        // label-must-reflect-configuration: stamp the elected MoE plane layout
        let moe_layout = layers.iter().find_map(|l| match &l.mixer {
            Mixer::Moe(m) => match &m.planes {
                MoePlanes::Nvf4 { up, .. } => Some(up.layout),
                _ => None,
            },
            _ => None,
        });
        tracing::info!(
            layers = hp.n_layer,
            weights_gib = weights_bytes as f64 / (1u64 << 30) as f64,
            bf16_attn_twins_gib = bf16_twin_bytes as f64 / (1u64 << 30) as f64,
            moe_shared_foldin_gib = moe_foldin_bytes as f64 / (1u64 << 30) as f64,
            ssm_state = ?ssm_dtype,
            moe_layout = ?moe_layout,
            "nemotron NVFP4 checkpoint loaded (twin/fold-in fields are \
             duplicate residency, included in weights_gib)"
        );

        let prefill_chunk = super::forward::prefill_chunk_rows(exec.sm_count());
        small_die_defaults(&exec);
        Ok(Self {
            exec,
            hp,
            layers,
            tok_embd,
            final_norm,
            lm_head,
            kv_dtype: default_kv_dtype(),
            ssm_dtype,
            prefill_chunk,
            max_ctx,
            weights_bytes,
            content_id: (
                crate::kv_tier::fingerprint::weights_safetensors(&st),
                crate::kv_tier::fingerprint::tokenizer_dir(dir),
            ),
            decode: None,
            scratch: None,
            prefill: None,
            pipe: None,
            batch: None,
            chunked: Vec::new(),
            last_reused: Vec::new(),
            pipe_b: None,
            dflash: None,
            // the NVFP4 checkpoint's mtp.* planes are bf16 - a residency
            // class the expert enums don't carry; the NVFP4 lane's drafter
            // is DFlash (attach via --mtp)
            mtp: None,
        })
    }

    /// Load the unsloth Q8_0 GGUF (the second lane on the same
    /// graph). Residencies: Q8_0 planes repack to the split int8+f16-scale
    /// serving layout (attn q/k/v/o, ssm in/out, all expert planes, lm_head);
    /// the embedding table stays raw Q8_0 for the resident gather; every F32
    /// tensor uploads as-is. Two conventions verified against the NVFP4
    /// checkpoint byte-for-byte (layout probe): `ssm_a` ships
    /// already transformed to -exp(A_log), and `ssm_norm`/`ssm_conv1d`
    /// flatten to exactly the device layouts the kernels consume - so this
    /// loader applies no transforms at all.
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &paddock_models::mapped::MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        let hp = NemotronConfig::from_gguf(map.gguf())
            .map_err(|e| GpuModelError::Unsupported(format!("nemotron gguf: {e}")))?;
        exec.vram_load_gate(map.total_len(), "nemotron")
            .map_err(GpuModelError::WontFit)?;

        let mut weights_bytes: u64 = 0;

        // embedding table: raw Q8_0, gathered in-kernel (granite's path)
        let te = exec.upload_raw(map, "token_embd.weight")?;
        if te.ty != GgmlType::Q8_0 {
            return Err(GpuModelError::Unsupported(format!(
                "token_embd.weight quant {:?} has no resident gather path",
                te.ty
            )));
        }
        if te.dims != [hp.hidden, hp.vocab] {
            return Err(GpuModelError::Unsupported(format!(
                "token_embd.weight is {:?}, expected [{}, {}]",
                te.dims, hp.hidden, hp.vocab
            )));
        }
        weights_bytes += te.bytes.len() as u64;
        let tok_embd = TokEmbd::Q8(te);

        let mut layers = Vec::with_capacity(hp.n_layer);
        for li in 0..hp.n_layer {
            let dt = |name: &str| exec.upload(map, &format!("blk.{li}.{name}"));
            let qw = |name: &str| exec.load_quantw(map, &format!("blk.{li}.{name}"));
            let q8 = |name: &str| exec.repack_q8(map, &format!("blk.{li}.{name}"));

            let norm = dt("attn_norm.weight")?;
            weights_bytes += norm.buf.len() as u64 * 4;

            let mixer = match hp.blocks[li] {
                NemotronBlock::Mamba => {
                    let in_proj = qw("ssm_in.weight")?;
                    if in_proj.dims() != [hp.hidden, hp.in_proj_rows()] {
                        return Err(GpuModelError::Unsupported(format!(
                            "blk.{li}.ssm_in is {:?}, expected [{}, {}]",
                            in_proj.dims(),
                            hp.hidden,
                            hp.in_proj_rows()
                        )));
                    }
                    let out_proj = qw("ssm_out.weight")?;
                    if out_proj.dims() != [hp.d_inner(), hp.hidden] {
                        return Err(GpuModelError::Unsupported(format!(
                            "blk.{li}.ssm_out is {:?}, expected [{}, {}]",
                            out_proj.dims(),
                            hp.d_inner(),
                            hp.hidden
                        )));
                    }
                    let conv_w = dt("ssm_conv1d.weight")?;
                    let conv_b = dt("ssm_conv1d.bias")?;
                    if conv_w.buf.len() != hp.conv_dim() * hp.d_conv
                        || conv_b.buf.len() != hp.conv_dim()
                    {
                        return Err(GpuModelError::Unsupported(format!(
                            "blk.{li}.ssm_conv1d is {} + {} elems, expected {}x{} + {}",
                            conv_w.buf.len(),
                            conv_b.buf.len(),
                            hp.conv_dim(),
                            hp.d_conv,
                            hp.conv_dim()
                        )));
                    }
                    // already -exp(A_log) in-file - upload untransformed
                    let a = dt("ssm_a")?;
                    let d = dt("ssm_d")?;
                    let dt_bias = dt("ssm_dt.bias")?;
                    if a.buf.len() != hp.mamba_heads
                        || d.buf.len() != hp.mamba_heads
                        || dt_bias.buf.len() != hp.mamba_heads
                    {
                        return Err(GpuModelError::Unsupported(format!(
                            "blk.{li}: ssm_a/ssm_d/ssm_dt.bias sized {}/{}/{}, expected {}",
                            a.buf.len(),
                            d.buf.len(),
                            dt_bias.buf.len(),
                            hp.mamba_heads
                        )));
                    }
                    // grouped [d_inner/G, G] flattens to the natural [d_inner]
                    // channel order (groups are channel-contiguous - probed)
                    let norm_w = dt("ssm_norm.weight")?;
                    if norm_w.buf.len() != hp.d_inner() {
                        return Err(GpuModelError::Unsupported(format!(
                            "blk.{li}.ssm_norm is {} elems, expected {}",
                            norm_w.buf.len(),
                            hp.d_inner()
                        )));
                    }
                    weights_bytes += in_proj.bytes()
                        + out_proj.bytes()
                        + (conv_w.buf.len() + conv_b.buf.len() + 3 * hp.mamba_heads + hp.d_inner())
                            as u64
                            * 4;
                    Mixer::Mamba(MambaWeights {
                        in_proj: LinW::Qw(in_proj),
                        out_proj: LinW::Qw(out_proj),
                        conv_w: conv_w.buf,
                        conv_b: conv_b.buf,
                        a: a.buf,
                        d: d.buf,
                        dt_bias: dt_bias.buf,
                        norm_w: norm_w.buf,
                    })
                }
                NemotronBlock::Attention => {
                    let q_dim = hp.n_heads * hp.head_dim;
                    let kv_dim = hp.n_kv_heads * hp.head_dim;
                    let wq = qw("attn_q.weight")?;
                    let wk = qw("attn_k.weight")?;
                    let wv = qw("attn_v.weight")?;
                    let wo = qw("attn_output.weight")?;
                    if wq.dims() != [hp.hidden, q_dim]
                        || wk.dims() != [hp.hidden, kv_dim]
                        || wv.dims() != [hp.hidden, kv_dim]
                        || wo.dims() != [q_dim, hp.hidden]
                    {
                        return Err(GpuModelError::Unsupported(format!(
                            "blk.{li}: attn planes {:?}/{:?}/{:?}/{:?} disagree with heads {}x{} kv {}",
                            wq.dims(),
                            wk.dims(),
                            wv.dims(),
                            wo.dims(),
                            hp.n_heads,
                            hp.head_dim,
                            hp.n_kv_heads
                        )));
                    }
                    weights_bytes += wq.bytes() + wk.bytes() + wv.bytes() + wo.bytes();
                    Mixer::Attn(AttnWeights::Qw { wq, wk, wv, wo })
                }
                NemotronBlock::Moe => {
                    let router = dt("ffn_gate_inp.weight")?;
                    let bias = dt("exp_probs_b.bias")?;
                    if router.buf.len() != hp.n_expert * hp.hidden || bias.buf.len() != hp.n_expert
                    {
                        return Err(GpuModelError::Unsupported(format!(
                            "blk.{li}: router {} + bias {} elems, expected {}x{} + {}",
                            router.buf.len(),
                            bias.buf.len(),
                            hp.n_expert,
                            hp.hidden,
                            hp.n_expert
                        )));
                    }
                    weights_bytes += (router.buf.len() + bias.buf.len()) as u64 * 4;

                    let up = q8("ffn_up_exps.weight")?;
                    let down = q8("ffn_down_exps.weight")?;
                    let sh_up = q8("ffn_up_shexp.weight")?;
                    let sh_down = q8("ffn_down_shexp.weight")?;
                    if up.dims != [hp.hidden, hp.moe_ff, hp.n_expert]
                        || down.dims != [hp.moe_ff, hp.hidden, hp.n_expert]
                        || sh_up.dims != [hp.hidden, hp.shared_ff]
                        || sh_down.dims != [hp.shared_ff, hp.hidden]
                    {
                        return Err(GpuModelError::Unsupported(format!(
                            "blk.{li}: expert planes {:?}/{:?}/{:?}/{:?} disagree with ff {} shared {} experts {}",
                            up.dims,
                            down.dims,
                            sh_up.dims,
                            sh_down.dims,
                            hp.moe_ff,
                            hp.shared_ff,
                            hp.n_expert
                        )));
                    }
                    weights_bytes += (up.data.len()
                        + up.scale.len()
                        + down.data.len()
                        + down.scale.len()
                        + sh_up.data.len()
                        + sh_up.scale.len()
                        + sh_down.data.len()
                        + sh_down.scale.len()) as u64;
                    Mixer::Moe(MoeWeights {
                        router,
                        bias,
                        planes: MoePlanes::Q8 {
                            up,
                            down,
                            sh_up,
                            sh_down,
                        },
                    })
                }
            };
            layers.push(NemotronLayer { norm, mixer });
        }

        let final_norm = exec.upload(map, "output_norm.weight")?;
        weights_bytes += final_norm.buf.len() as u64 * 4;
        // untied head is present in this file; branch on PRESENCE like
        // granite so a tied export falls back to the embedding plane loudly
        let lm_head = if map.tensor_info("output.weight").is_some() {
            exec.load_quantw(map, "output.weight")?
        } else {
            tracing::info!("nemotron gguf: no output.weight - tied head, using token_embd");
            exec.load_quantw(map, "token_embd.weight")?
        };
        weights_bytes += lm_head.bytes();
        let lm_head = HeadW::Qw(lm_head);

        // In-file MTP block (C3): the trailing nextn block -
        // blk.52's eh_proj glue + a COMBINED attn+MoE transformer block
        // (unlike the one-mixer trunk layers) + shared_head_norm, all in
        // this file's Q8_0/F32 classes. Skipped under PADDOCK_NO_SPEC
        // (qwen35's precedent - spec "off" loads no drafter, reclaiming
        // ~1.4 GiB) and when the export stripped the tensors.
        let nl = hp.n_layer; // the nextn block index (block_count - 1)
        let mtp = if map
            .tensor_info(&format!("blk.{nl}.nextn.eh_proj.weight"))
            .is_none()
        {
            tracing::info!("nemotron gguf: no in-file nextn tensors - MTP drafter unavailable");
            None
        } else if std::env::var_os("PADDOCK_NO_SPEC").is_some() {
            tracing::info!("nemotron gguf: PADDOCK_NO_SPEC - nextn/MTP block left unloaded");
            None
        } else {
            let dt = |name: &str| exec.upload(map, &format!("blk.{nl}.{name}"));
            let qw = |name: &str| exec.load_quantw(map, &format!("blk.{nl}.{name}"));
            let q8 = |name: &str| exec.repack_q8(map, &format!("blk.{nl}.{name}"));
            let q_dim = hp.n_heads * hp.head_dim;
            let kv_dim = hp.n_kv_heads * hp.head_dim;

            let eh_proj = qw("nextn.eh_proj.weight")?;
            let enorm = dt("nextn.enorm.weight")?;
            let hnorm = dt("nextn.hnorm.weight")?;
            let head_norm = dt("nextn.shared_head_norm.weight")?;
            let attn_norm = dt("attn_norm.weight")?;
            let post_norm = dt("post_attention_norm.weight")?;
            let wq = qw("attn_q.weight")?;
            let wk = qw("attn_k.weight")?;
            let wv = qw("attn_v.weight")?;
            let wo = qw("attn_output.weight")?;
            let router = dt("ffn_gate_inp.weight")?;
            let bias = dt("exp_probs_b.bias")?;
            let up = q8("ffn_up_exps.weight")?;
            let down = q8("ffn_down_exps.weight")?;
            let sh_up = q8("ffn_up_shexp.weight")?;
            let sh_down = q8("ffn_down_shexp.weight")?;
            if eh_proj.dims() != [2 * hp.hidden, hp.hidden]
                || enorm.buf.len() != hp.hidden
                || hnorm.buf.len() != hp.hidden
                || head_norm.buf.len() != hp.hidden
                || attn_norm.buf.len() != hp.hidden
                || post_norm.buf.len() != hp.hidden
                || wq.dims() != [hp.hidden, q_dim]
                || wk.dims() != [hp.hidden, kv_dim]
                || wv.dims() != [hp.hidden, kv_dim]
                || wo.dims() != [q_dim, hp.hidden]
                || router.buf.len() != hp.n_expert * hp.hidden
                || bias.buf.len() != hp.n_expert
                || up.dims != [hp.hidden, hp.moe_ff, hp.n_expert]
                || down.dims != [hp.moe_ff, hp.hidden, hp.n_expert]
                || sh_up.dims != [hp.hidden, hp.shared_ff]
                || sh_down.dims != [hp.shared_ff, hp.hidden]
            {
                return Err(GpuModelError::Unsupported(format!(
                    "blk.{nl} nextn geometry disagrees with the trunk (eh_proj {:?})",
                    eh_proj.dims()
                )));
            }
            weights_bytes += eh_proj.bytes()
                + wq.bytes()
                + wk.bytes()
                + wv.bytes()
                + wo.bytes()
                + (up.data.len()
                    + up.scale.len()
                    + down.data.len()
                    + down.scale.len()
                    + sh_up.data.len()
                    + sh_up.scale.len()
                    + sh_down.data.len()
                    + sh_down.scale.len()) as u64
                + (6 * hp.hidden + hp.n_expert * hp.hidden + hp.n_expert) as u64 * 4;
            tracing::info!("nemotron gguf: in-file nextn/MTP block loaded (blk.{nl})");
            Some(super::mtp::MtpDrafter {
                w: super::mtp::MtpWeights {
                    enorm,
                    hnorm,
                    eh_proj,
                    attn_norm,
                    attn: AttnWeights::Qw { wq, wk, wv, wo },
                    post_norm,
                    moe: MoeWeights {
                        router,
                        bias,
                        planes: MoePlanes::Q8 {
                            up,
                            down,
                            sh_up,
                            sh_down,
                        },
                    },
                    head_norm,
                },
                state: None,
            })
        };

        exec.trim_mem_pool();
        let ssm_dtype = super::ssm_arena::ssm_dtype_from_env();
        tracing::info!(
            layers = hp.n_layer,
            weights_gib = weights_bytes as f64 / (1u64 << 30) as f64,
            ssm_state = ?ssm_dtype,
            "nemotron Q8_0 gguf loaded"
        );

        let prefill_chunk = super::forward::prefill_chunk_rows(exec.sm_count());
        small_die_defaults(&exec);
        Ok(Self {
            exec,
            hp,
            layers,
            tok_embd,
            final_norm,
            lm_head,
            kv_dtype: default_kv_dtype(),
            ssm_dtype,
            prefill_chunk,
            max_ctx,
            weights_bytes,
            content_id: (
                crate::kv_tier::fingerprint::weights(map),
                crate::kv_tier::fingerprint::tokenizer(map),
            ),
            decode: None,
            scratch: None,
            prefill: None,
            pipe: None,
            batch: None,
            chunked: Vec::new(),
            last_reused: Vec::new(),
            pipe_b: None,
            dflash: None,
            mtp,
        })
    }

    /// The GGUF-quant lane (Q8_0 dp4a class) - picks the Q8 forward arms and,
    /// until stage B, refuses `enable_batch` with a real error.
    pub(crate) fn is_gguf(&self) -> bool {
        matches!(self.lm_head, HeadW::Qw(_))
    }

    /// Select the attention KV cache element type (fp8-e4m3 is the default
    /// and the checkpoint's own KV spec - see `default_kv_dtype`; f16 is the
    /// greedy-exact class the reference tests pin).
    /// Drops decode state AND the batch lane so every cache re-allocates at
    /// the new element size - a live pool sized for the old dtype would be
    /// silently mis-strided by the paged kernels.
    pub fn set_kv_dtype(&mut self, dtype: KvDtype) {
        self.pipe_abort();
        self.pipe_b_abort();
        self.kv_dtype = dtype;
        self.decode = None;
        self.scratch = None;
        self.batch = None;
        self.chunked.clear();
    }
}

/// The family's KV default: fp8-e4m3 (KV8) on both lanes. It is the NVFP4
/// checkpoint's own spec (`kv_cache_quant_algo: FP8`), the class every board
/// and the vLLM rival run, and the batched lane's attention ladder has fp8
/// arms at every shape it elects (the G=16 prefill tile, the head-packed
/// decode partial, the verify rows). f16 was the default until 2026-10-04 -
/// the bring-up's greedy-exact class - so an `auto` serve paid twice the KV
/// bytes: on GB10 a Claude Code replay to 248K ran 193 -> 148 s at fp8
/// (NVFP4 + DSpark, TTFT 146 -> 98 s) and 253 -> 216 s on the Q8_0 lane.
/// Quality at fp8: top-1 equal with greedy 8/8 against f16 on the batch and
/// GGUF gates, but NOT free on long-context recall - six attention layers
/// carry every lookup, and e4m3's 3-bit mantissa shows. A paired probe (the
/// same 100 greedy route lookups in a ~32K ledger of look-alike records) read
/// fp8 75/100 against f16 82/100 (8 questions only f16 got, 1 only fp8), and
/// a recorded Claude Code post-compaction lookup resolved 2/40 sampled at fp8
/// against 9/40 at f16 (greedy wrong at both). KV8 stays the default
/// regardless (decided 2026-10-05): the speed above and the checkpoint's own
/// spec; per-head dynamic INT4/INT8 with inline scales is the KV doctrine's
/// next rung. A runner on a die that cannot store fp8 KV overrides this back
/// to f16 (serving.rs), and an explicit kv_cache_dtype wins.
pub(crate) fn default_kv_dtype() -> KvDtype {
    KvDtype::Fp8E4m3
}
