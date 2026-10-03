//! Clef from a GGUF: ggml-org's conversions of the release
//! (`ggml-org/Clef-Flash-GGUF` of `Cloudflare/clef-flash`, `ggml-org/Clef-GGUF`
//! of `Cloudflare/clef`), llama.cpp's `clef` schema (`conversion/clef.py` over
//! its Qwen3.5 text converter). The converter already applied every
//! transform `load.rs` applies to the shards - value heads tiled, `A_log` as
//! `-exp(A_log)`, conv1d squeezed, `+1` in every RMSNorm but the gated one -
//! so the planes come straight off the file: Q8_0 projections repacked for
//! slot 741 (the same 8.5 bits a weight), the embeddings Q8_0 rows as stored,
//! norms and the DeltaNet's small tensors F32 as stored. A BF16 GGUF loads
//! too (its projections on 733, as the shards'); any other type is refused,
//! never converted.
//!
//! The GGUF carries no vision tower. A companion beside it holds the official
//! checkpoint's `model.visual.*` tensors byte for byte with that checkpoint's
//! configs in its metadata (`paddock_models::clef::COMPANION_*`); without one
//! the model reads text and refuses images.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::clef::{ClefBlock, ClefConfig};
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;
use paddock_models::safetensors::{SafetensorsFile, ShardedSafetensors};

use super::{Attn, Gdn, GpuClef, Layer, MAX_ROWS, Mixer, Proj, Workspace};
use crate::gpu::{ClefQ8, GpuExecutor, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;

fn bad(m: impl Into<String>) -> GpuModelError {
    GpuModelError::Unsupported(format!("Clef GGUF: {}", m.into()))
}

/// The open GGUF and its tensor reads.
pub(super) struct Src<'a> {
    pub(super) g: &'a MappedGguf,
}

impl<'a> Src<'a> {
    /// A tensor's type and bytes, its dims checked against `ne` (`ne[0]`
    /// first, as stored).
    pub(super) fn raw(
        &self,
        name: &str,
        ne: &[usize],
    ) -> Result<(GgmlType, &'a [u8]), GpuModelError> {
        let (info, bytes) = self.g.tensor_bytes(name).map_err(|e| bad(e.to_string()))?;
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        // a vector may be stored [n] or [n, 1]
        let trimmed = |d: &[usize]| {
            let mut d = d.to_vec();
            while d.len() > 1 && d.last() == Some(&1) {
                d.pop();
            }
            d
        };
        if trimmed(&dims) != trimmed(ne) {
            return Err(bad(format!("{name}: dims {dims:?}, expected {ne:?}")));
        }
        Ok((info.ggml_type, bytes))
    }

    /// A small tensor's values widened exactly to F32: F32, BF16, F16 or
    /// Q8_0 as stored.
    pub(super) fn f32s(&self, name: &str, ne: &[usize]) -> Result<Vec<f32>, GpuModelError> {
        let (ty, b) = self.raw(name, ne)?;
        Ok(match ty {
            GgmlType::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            GgmlType::Bf16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect(),
            GgmlType::F16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect(),
            GgmlType::Q8_0 => b
                .as_chunks::<34>()
                .0
                .iter()
                .flat_map(|blk| {
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    blk[2..].iter().map(move |&q| q as i8 as f32 * d)
                })
                .collect(),
            ty => return Err(bad(format!("{name}: stored as {ty:?}"))),
        })
    }

    /// Weight rows `[n][k]` gathered from `names` - stacked one tensor after
    /// another, or interleaved row by row (`interleave`: the MLP's gate and
    /// up) - as stored: the shared type and the row bytes.
    pub(super) fn rows(
        &self,
        names: &[&str],
        k: usize,
        ns: &[usize],
        interleave: bool,
    ) -> Result<(GgmlType, Vec<u8>), GpuModelError> {
        let mut parts = Vec::with_capacity(names.len());
        for (name, &n) in names.iter().zip(ns) {
            parts.push(self.raw(name, &[k, n])?);
        }
        let ty = parts[0].0;
        if parts.iter().any(|p| p.0 != ty) {
            return Err(bad(format!("{names:?}: mixed types")));
        }
        if !matches!(ty, GgmlType::Q8_0 | GgmlType::Bf16) {
            return Err(bad(format!(
                "{}: stored as {ty:?} - the lane reads Q8_0 and BF16 projections",
                names[0]
            )));
        }
        let row = ty.byte_size(k as u64).ok_or_else(|| bad("row size"))? as usize;
        let mut out = Vec::with_capacity(parts.iter().map(|p| p.1.len()).sum());
        if interleave {
            let n = ns[0];
            if ns.iter().any(|&m| m != n) {
                return Err(bad("interleaved planes of different heights"));
            }
            for r in 0..n {
                for p in &parts {
                    out.extend_from_slice(&p.1[r * row..(r + 1) * row]);
                }
            }
        } else {
            for p in &parts {
                out.extend_from_slice(p.1);
            }
        }
        Ok((ty, out))
    }
}

impl GpuClef {
    /// Load a Clef GGUF (`path`), with the vision companion when one is
    /// given. Every byte the load places is counted against the budget
    /// before the first one.
    pub fn load_gguf(
        exec: Arc<GpuExecutor>,
        path: &Path,
        companion: Option<&Path>,
    ) -> Result<Self, GpuModelError> {
        if !exec.has_clef_gguf() {
            return Err(bad(
                "this kernel pack predates the Clef GGUF lane (slots 741-742) - rebuild or \
                 update the pack",
            ));
        }
        let g = MappedGguf::open(path).map_err(|e| bad(e.to_string()))?;
        let companion = match companion {
            Some(p) => {
                Some(SafetensorsFile::open(p).map_err(|e| bad(format!("{}: {e}", p.display())))?)
            }
            None => None,
        };
        let cfg = ClefConfig::from_gguf(g.gguf(), companion.as_ref().map(|c| &c.metadata))
            .map_err(bad)?;
        let src = Src { g: &g };
        let (h, f) = (cfg.hidden, cfg.ffn);
        let (hv, kd, vd) = (cfg.gdn_v_heads, cfg.gdn_k_dim, cfg.gdn_v_dim);
        if kd != 128 || vd != 128 {
            return Err(bad(format!(
                "DeltaNet heads of {kd}/{vd} - the chunked scan reads 128"
            )));
        }
        if cfg.n_rot != 64 || cfg.head_dim != 256 {
            return Err(bad(format!(
                "rope over {} of {} dims - the backbone kernels rotate 64 of 256",
                cfg.n_rot, cfg.head_dim
            )));
        }
        let rows = MAX_ROWS;
        let vision = cfg.vision.is_some() && exec.has_clef_vision();
        if cfg.vision.is_some() && !vision {
            eprintln!(
                "[clef] this kernel pack predates Clef's image lane (slots 735-740): images are \
                 refused until it is updated"
            );
        }
        let companion = companion
            .filter(|_| vision)
            .map(ShardedSafetensors::from_file);

        // the gate: the file's tensors as stored (the backbone's Q8_0 repack
        // is the same size; the head's F32 block scales add 2 bytes a block)
        let file_bytes: u64 = g.tensor_infos().filter_map(|t| t.byte_size()).sum();
        let head_bytes: u64 = g
            .tensor_infos()
            .filter(|t| t.name.starts_with("dec") || t.name.starts_with("decision."))
            .filter_map(|t| t.byte_size())
            .sum();
        let tower_bytes: u64 = companion.as_ref().map_or(0, |c| {
            c.names()
                .filter_map(|n| c.bytes(n).map(|(_, b)| b.len() as u64))
                .sum()
        });
        // The tower counts ONCE: its planes upload as the companion stores
        // them, BF16 (vision.rs), and its activations are in the Workspace
        // term below. Charged twice, the gate asked 0.85 GiB the load never
        // places - with the 1 GiB floor on top that refused Clef Flash on a
        // 48 GB card inside the 16.3 GiB the manager had (correctly) granted:
        // the runner's own ledger read 9.85 GiB of weights + 4.75 GiB of
        // workspace once loaded.
        exec.vram_load_gate(
            file_bytes
                + head_bytes / 16
                + tower_bytes
                + Workspace::bytes(&cfg, rows, vision)
                + 4 * super::head::HeadWs::floats(&cfg.head, cfg.hidden) as u64
                + (64 << 20),
            "Clef",
        )
        .map_err(GpuModelError::WontFit)?;
        exec.disable_event_tracking();
        // batch invariance, as the shards' load (`load.rs`)
        if std::env::var_os("PADDOCK_NO_BF16_KSPLIT").is_none() {
            crate::envset::set_env("PADDOCK_NO_BF16_KSPLIT", "1");
        }

        let weight_bytes = std::cell::Cell::new(0u64);
        let up_f32 = |v: &[f32]| -> Result<CudaSlice<f32>, GpuModelError> {
            weight_bytes.set(weight_bytes.get() + 4 * v.len() as u64);
            Ok(exec.to_device(v)?)
        };
        let vec = |name: &str, n: usize| up_f32(&src.f32s(name, &[n])?);
        // a projection `k -> n` from its row planes
        let proj =
            |(ty, raw): (GgmlType, Vec<u8>), n: usize, k: usize| -> Result<Proj, GpuModelError> {
                Ok(match ty {
                    GgmlType::Q8_0 => {
                        let (q, scale) = ClefQ8::repack(&raw, n, k);
                        weight_bytes.set(weight_bytes.get() + (q.len() + scale.len()) as u64);
                        Proj::Q8(ClefQ8 {
                            q: exec.to_device_u8(&q)?,
                            scale: exec.to_device_u8(&scale)?,
                            k,
                            n,
                        })
                    }
                    _ => {
                        weight_bytes.set(weight_bytes.get() + raw.len() as u64);
                        Proj::Bf16(QuantTensor {
                            bytes: exec.to_device_u8(&raw)?,
                            ty: GgmlType::Bf16,
                            dims: vec![k, n],
                        })
                    }
                })
            };
        let one = |name: &str, k: usize, n: usize| proj(src.rows(&[name], k, &[n], false)?, n, k);
        // an embedding table as stored (the gather widens BF16 or Q8_0 rows)
        let table = |name: &str| -> Result<QuantTensor, GpuModelError> {
            let (ty, raw) = src.raw(name, &[h, cfg.vocab])?;
            if !matches!(ty, GgmlType::Q8_0 | GgmlType::Bf16) {
                return Err(bad(format!(
                    "{name}: stored as {ty:?}, expected Q8_0 or BF16"
                )));
            }
            weight_bytes.set(weight_bytes.get() + raw.len() as u64);
            Ok(QuantTensor {
                bytes: exec.to_device_u8(raw)?,
                ty,
                dims: vec![h, cfg.vocab],
            })
        };

        let qkv_rows = cfg.gdn_qkv_rows();
        let mut layers = Vec::with_capacity(cfg.blocks.len());
        for (i, block) in cfg.blocks.iter().enumerate() {
            let p = |n: &str| format!("blk.{i}.{n}");
            let mixer = match block {
                ClefBlock::Gdn => {
                    let conv = src.f32s(&p("ssm_conv1d.weight"), &[cfg.gdn_conv, qkv_rows])?;
                    Mixer::Gdn(Gdn {
                        qkv: one(&p("attn_qkv.weight"), h, qkv_rows)?,
                        z: one(&p("attn_gate.weight"), h, hv * vd)?,
                        ab: proj(
                            src.rows(
                                &[&p("ssm_alpha.weight"), &p("ssm_beta.weight")],
                                h,
                                &[hv, hv],
                                false,
                            )?,
                            2 * hv,
                            h,
                        )?,
                        conv: up_f32(&conv)?,
                        ssm_a: vec(&p("ssm_a"), hv)?,
                        dt_bias: vec(&p("ssm_dt.bias"), hv)?,
                        norm: vec(&p("ssm_norm.weight"), vd)?,
                        out: one(&p("ssm_out.weight"), hv * vd, h)?,
                    })
                }
                ClefBlock::Attention => {
                    let (qr, kvw, qw) = (cfg.attn_q_rows(), cfg.kv_width(), cfg.q_width());
                    Mixer::Attn(Attn {
                        q: one(&p("attn_q.weight"), h, qr)?,
                        k: one(&p("attn_k.weight"), h, kvw)?,
                        v: one(&p("attn_v.weight"), h, kvw)?,
                        q_norm: vec(&p("attn_q_norm.weight"), cfg.head_dim)?,
                        k_norm: vec(&p("attn_k_norm.weight"), cfg.head_dim)?,
                        o: one(&p("attn_output.weight"), qw, h)?,
                    })
                }
            };
            layers.push(Layer {
                in_norm: vec(&p("attn_norm.weight"), h)?,
                post_norm: vec(&p("post_attention_norm.weight"), h)?,
                mixer,
                gate_up: proj(
                    src.rows(
                        &[&p("ffn_gate.weight"), &p("ffn_up.weight")],
                        h,
                        &[f, f],
                        true,
                    )?,
                    2 * f,
                    h,
                )?,
                down: one(&p("ffn_down.weight"), f, h)?,
            });
        }
        let embed = table("token_embd.weight")?;
        let lm_head = table("output.weight")?;
        let final_norm = vec("output_norm.weight", h)?;
        let rope = up_f32(&super::load::rope_table(cfg.rope_theta, cfg.n_rot, rows))?;
        let rope_masks = super::load::rope_masks(cfg.mrope_sections, cfg.n_rot / 2);
        let head = super::head::HeadW::load_gguf(&exec, &src, &cfg.head, h)?;
        let vision = match (cfg.vision.as_ref().filter(|_| vision), companion.as_ref()) {
            (Some((v, _)), Some(st)) => Some(super::vision::Vision::load(&exec, st, v)?),
            _ => None,
        };
        let ws = Workspace::new(&exec, &cfg, rows, vision.is_some())?;
        let head_ws = super::head::HeadWs::new(&exec, &cfg.head, h)?;
        Ok(Self {
            weight_bytes: weight_bytes.get() + head.bytes + vision.as_ref().map_or(0, |v| v.bytes),
            vision,
            exec,
            cfg,
            embed,
            lm_head,
            final_norm,
            rope,
            rope_masks,
            layers,
            head,
            ws,
            head_ws,
            max_rows: rows,
        })
    }
}
