//! Checkpoint -> device, straight off the release's BF16 safetensors.
//!
//! - GEMM planes widen bf16 -> f16 BIT-EXACTLY (`bf16_to_f16_exact`: every
//!   weight of both releases is inside f16's normal range or flushes from
//!   below 2^-14, where it is under the f32 accumulator's own rounding), so
//!   the f16 tensor-core lane multiplies the checkpoint's own numbers;
//! - q | k | v stack into one plane (one GEMM lands the fused rows the rope
//!   and attention read in place); gate | up interleave in 16-row blocks
//!   (gate rows 0-7, up rows 8-15 of each block - the gated-SiLU landing's
//!   layout, slot 895);
//! - the projector's merger has its input columns re-laid from `F.unfold`'s
//!   `c * 4 + k` to `k * 1024 + c` (k = dy * 2 + dx), so a merged token's
//!   input is its four patch rows back to back - one gather, no transpose;
//! - the patch conv `[1024][3][14][14]` is already the GEMM's `[out][588]`
//!   rows, padded to 592 zero columns;
//! - norms and the score row widen to f32 (exact); the embedding stays bf16
//!   on the device (the gather widens a row exactly - 1 GB, not 2 or 4).

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::nemotron_embed::{Kind, NemotronEmbedConfig};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::vision::{Tower, VisionLayer};
use super::{GpuNemotronEmbed, TextLayer};
use crate::gpu::{GpuExecutor, HalfTensor, NE_PATCH_K};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::{bf16_to_f16_exact, f32_tensor};

/// Reads and uploads with a running byte count.
pub(super) struct Up<'a> {
    pub exec: &'a GpuExecutor,
    pub st: &'a ShardedSafetensors,
    pub bytes: u64,
}

impl Up<'_> {
    /// A bf16 tensor's words, its shape checked.
    pub fn words(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>, GpuModelError> {
        let (t, b) = self
            .st
            .bytes(name)
            .ok_or_else(|| GpuModelError::MissingMeta(format!("nemotron-embed tensor {name}")))?;
        if t.dtype != StDtype::Bf16 || t.shape != shape {
            return Err(GpuModelError::Unsupported(format!(
                "nemotron-embed {name}: {:?} {:?} (want BF16 {shape:?})",
                t.dtype, t.shape
            )));
        }
        Ok(b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect())
    }

    /// bf16 words laid `[out][in]` -> an f16 GEMM plane, widened exactly
    /// (eight threads: the 8B body's 7.4G weights are most of a load).
    pub fn plane(
        &mut self,
        words: &[u16],
        in_dim: usize,
        out_dim: usize,
        what: &str,
    ) -> Result<HalfTensor, GpuModelError> {
        debug_assert_eq!(words.len(), in_dim * out_dim);
        if !in_dim.is_multiple_of(8) {
            return Err(GpuModelError::Unsupported(format!(
                "nemotron-embed {what}: GEMM input width {in_dim} is not a multiple of 8"
            )));
        }
        let chunk = words.len().div_ceil(8).next_multiple_of(8);
        let parts: Vec<Result<Vec<f16>, GpuModelError>> = std::thread::scope(|s| {
            let hs: Vec<_> = words
                .chunks(chunk)
                .map(|c| {
                    s.spawn(move || {
                        let raw: Vec<u8> = c.iter().flat_map(|x| x.to_le_bytes()).collect();
                        bf16_to_f16_exact(&raw, what)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("bf16 -> f16 thread"))
                .collect()
        });
        let mut h = Vec::with_capacity(words.len());
        for p in parts {
            h.extend(p?);
        }
        self.bytes += (h.len() * 2) as u64;
        Ok(HalfTensor {
            buf: self.exec.f16_to_device(&h)?,
            dims: vec![in_dim, out_dim],
        })
    }

    /// `nn.Linear` weight `[out][in]` as a plane.
    pub fn linear(
        &mut self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<HalfTensor, GpuModelError> {
        let w = self.words(name, &[out_dim, in_dim])?;
        self.plane(&w, in_dim, out_dim, name)
    }

    /// Several `[out_i][in]` weights stacked into one `[sum out][in]` plane.
    pub fn stacked(
        &mut self,
        names: &[(&str, usize)],
        in_dim: usize,
    ) -> Result<HalfTensor, GpuModelError> {
        let mut w = Vec::new();
        for (n, o) in names {
            w.extend(self.words(n, &[*o, in_dim])?);
        }
        let out: usize = names.iter().map(|p| p.1).sum();
        self.plane(&w, in_dim, out, names[0].0)
    }

    /// gate `[f][in]` and up `[f][in]` interleaved in 16-row blocks: block b
    /// is gate rows 8b..8b+8 then up rows 8b..8b+8 (`[2 f][in]`).
    pub fn gate_up(
        &mut self,
        gate: &str,
        up: &str,
        f: usize,
        in_dim: usize,
    ) -> Result<HalfTensor, GpuModelError> {
        if !f.is_multiple_of(8) {
            return Err(GpuModelError::Unsupported(format!(
                "nemotron-embed {gate}: FFN width {f} is not a multiple of 8"
            )));
        }
        let g = self.words(gate, &[f, in_dim])?;
        let u = self.words(up, &[f, in_dim])?;
        let mut w = vec![0u16; 2 * f * in_dim];
        for b in 0..f / 8 {
            for r in 0..8 {
                let src = (8 * b + r) * in_dim;
                let dg = (16 * b + r) * in_dim;
                let du = (16 * b + 8 + r) * in_dim;
                w[dg..dg + in_dim].copy_from_slice(&g[src..src + in_dim]);
                w[du..du + in_dim].copy_from_slice(&u[src..src + in_dim]);
            }
        }
        self.plane(&w, in_dim, 2 * f, gate)
    }

    /// A norm weight (or any small vector) widened to f32.
    pub fn vec(&mut self, name: &str, n: usize) -> Result<CudaSlice<f32>, GpuModelError> {
        let v = f32_tensor(self.st, name, n)?;
        self.bytes += (n * 4) as u64;
        Ok(self.exec.to_device(&v)?)
    }
}

impl GpuNemotronEmbed {
    /// Load a release directory (Embed or Rerank). `context` caps a
    /// sequence's tokens (at most the checkpoint's `max_seq_length`);
    /// `pictures` loads the Pixtral tower and projector.
    pub fn load_dir(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        context: usize,
        pictures: bool,
    ) -> Result<Self, GpuModelError> {
        let cfg = NemotronEmbedConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("nemotron-embed config: {e}")))?;
        if !exec.has_nemotron_embed() || !exec.has_f16_gemm_ring() || !exec.has_f16_gemm_h_gelu() {
            return Err(GpuModelError::Unsupported(
                "Nemotron 3.5 Embed / Rerank needs the pack's slots 894-902 - rebuild or update \
                 the pack"
                    .into(),
            ));
        }
        if pictures && !exec.has_clef_resample() {
            return Err(GpuModelError::Unsupported(
                "Nemotron 3.5 Embed / Rerank pictures need the uint8 resize (slots 735/736) - \
                 rebuild or update the pack"
                    .into(),
            ));
        }
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("nemotron-embed safetensors: {e}")))?;
        let file_bytes: u64 = std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        let context = context.clamp(1, cfg.max_ctx);
        // f16 planes are the file's bytes; a full-context pass holds ~72 KB a
        // row of activations
        exec.vram_load_gate(file_bytes + context as u64 * 74_000, "nemotron-embed")
            .map_err(GpuModelError::WontFit)?;
        exec.disable_event_tracking();

        let p = cfg.prefix;
        let (d, nh, nkv, hd, ffn) = (
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.ffn,
        );
        let mut up = Up {
            exec: &exec,
            st: &st,
            bytes: 0,
        };
        // the embedding: bf16 bytes resident as shipped
        let emb_name = format!("{p}language_model.embed_tokens.weight");
        let (t, raw) = st.bytes(&emb_name).ok_or_else(|| {
            GpuModelError::MissingMeta(format!("nemotron-embed tensor {emb_name}"))
        })?;
        if t.dtype != StDtype::Bf16 || t.shape != [cfg.vocab, d] {
            return Err(GpuModelError::Unsupported(format!(
                "nemotron-embed {emb_name}: {:?} {:?}",
                t.dtype, t.shape
            )));
        }
        let emb = exec.to_device_u8(raw)?;
        up.bytes += raw.len() as u64;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let l = |s: &str| format!("{p}language_model.layers.{i}.{s}");
            layers.push(TextLayer {
                ln1: up.vec(&l("input_layernorm.weight"), d)?,
                qkv: up.stacked(
                    &[
                        (&l("self_attn.q_proj.weight"), nh * hd),
                        (&l("self_attn.k_proj.weight"), nkv * hd),
                        (&l("self_attn.v_proj.weight"), nkv * hd),
                    ],
                    d,
                )?,
                o: up.linear(&l("self_attn.o_proj.weight"), d, nh * hd)?,
                ln2: up.vec(&l("post_attention_layernorm.weight"), d)?,
                gu: up.gate_up(&l("mlp.gate_proj.weight"), &l("mlp.up_proj.weight"), ffn, d)?,
                down: up.linear(&l("mlp.down_proj.weight"), d, ffn)?,
            });
        }
        let norm = up.vec(&format!("{p}language_model.norm.weight"), d)?;
        let score = match cfg.kind {
            // the sequence-classification head sits beside the wrapped body
            Kind::Rerank => Some(up.vec("language_model.score.weight", d)?),
            Kind::Embed => None,
        };
        let tower = if pictures {
            Some(load_tower(&mut up, &cfg)?)
        } else {
            None
        };
        let (cos, sin) = cfg.text_rope_tables(context);
        let rope_cos = exec.to_device(&cos)?;
        let rope_sin = exec.to_device(&sin)?;
        let weights_bytes = up.bytes;
        tracing::info!(
            kind = ?cfg.kind,
            context,
            pictures,
            "nemotron-embed: {:.1} GB of f16 planes",
            weights_bytes as f64 / 1e9
        );
        Ok(Self {
            exec,
            cfg,
            emb,
            layers,
            norm,
            score,
            tower,
            rope_cos,
            rope_sin,
            context,
            scratch: None,
            out: [None, None],
            flip: 0,
            weights_bytes,
        })
    }
}

fn load_tower(up: &mut Up<'_>, cfg: &NemotronEmbedConfig) -> Result<Tower, GpuModelError> {
    let p = cfg.prefix;
    let (vd, vf) = (cfg.v_dim, cfg.v_ffn);
    let t = |s: &str| format!("{p}vision_tower.{s}");
    // the patch conv [1024][3][14][14] is [out][588]; four zero columns
    let conv = up.words(&t("patch_conv.weight"), &[vd, 3, cfg.patch, cfg.patch])?;
    let k0 = 3 * cfg.patch * cfg.patch;
    if k0 > NE_PATCH_K {
        return Err(GpuModelError::Unsupported(format!(
            "nemotron-embed: a {}-pixel patch is past the stem's {NE_PATCH_K} columns",
            cfg.patch
        )));
    }
    let mut w = vec![0u16; vd * NE_PATCH_K];
    for o in 0..vd {
        w[o * NE_PATCH_K..o * NE_PATCH_K + k0].copy_from_slice(&conv[o * k0..(o + 1) * k0]);
    }
    let patch = up.plane(&w, NE_PATCH_K, vd, "vision_tower.patch_conv")?;
    let ln_pre = up.vec(&t("ln_pre.weight"), vd)?;
    let mut layers = Vec::with_capacity(cfg.v_layers);
    for i in 0..cfg.v_layers {
        let l = |s: &str| t(&format!("transformer.layers.{i}.{s}"));
        layers.push(VisionLayer {
            ln1: up.vec(&l("attention_norm.weight"), vd)?,
            qkv: up.stacked(
                &[
                    (&l("attention.q_proj.weight"), vd),
                    (&l("attention.k_proj.weight"), vd),
                    (&l("attention.v_proj.weight"), vd),
                ],
                vd,
            )?,
            o: up.linear(&l("attention.o_proj.weight"), vd, vd)?,
            ln2: up.vec(&l("ffn_norm.weight"), vd)?,
            gu: up.gate_up(
                &l("feed_forward.gate_proj.weight"),
                &l("feed_forward.up_proj.weight"),
                vf,
                vd,
            )?,
            down: up.linear(&l("feed_forward.down_proj.weight"), vd, vf)?,
        });
    }
    let m = |s: &str| format!("{p}multi_modal_projector.{s}");
    let proj_norm = up.vec(&m("norm.weight"), vd)?;
    // the merger: columns from unfold's c * 4 + k to k * vd + c
    let n4 = cfg.merge * cfg.merge;
    let mw = up.words(&m("patch_merger.merging_layer.weight"), &[vd, vd * n4])?;
    let mut rl = vec![0u16; mw.len()];
    for o in 0..vd {
        for c in 0..vd {
            for k in 0..n4 {
                rl[o * vd * n4 + k * vd + c] = mw[o * vd * n4 + c * n4 + k];
            }
        }
    }
    let merge = up.plane(&rl, vd * n4, vd, "multi_modal_projector.patch_merger")?;
    let l1 = up.linear(&m("linear_1.weight"), cfg.d_model, vd)?;
    let l2 = up.linear(&m("linear_2.weight"), cfg.d_model, cfg.d_model)?;
    Tower::new(
        up.exec, cfg, patch, ln_pre, layers, proj_norm, merge, l1, l2,
    )
}
