//! The backbone's tensors off the safetensors shards, through the transforms
//! llama.cpp's converter applies to a Qwen3.5 checkpoint
//! (`conversion/qwen.py`, `Qwen3NextModel` + `_LinearAttentionVReorderBase`):
//! every transform is a permutation, a sign-exp of `A_log` or a `+1` on a
//! norm weight, so the planes the kernels read are the checkpoint's values.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::clef::{ClefBlock, ClefConfig};
use paddock_models::ggml_type::GgmlType;
use paddock_models::safetensors::ShardedSafetensors;

use super::{Attn, Gdn, GpuClef, Layer, MAX_ROWS, Mixer, Proj, Workspace};
use crate::gpu::{GpuExecutor, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::{bf16_bytes, f32_tensor};

/// `perm[i]` = the grouped index the tiled index `i` reads: the checkpoint
/// stores value heads grouped by key head (`[g0: v0 v1, g1: v0 v1, ...]`),
/// the kernels broadcast key heads tiled (`[v0: g0 g1 ..., v1: g0 g1 ...]`).
fn tiled_perm(nk: usize, vpk: usize, hd: usize) -> Vec<usize> {
    (0..nk * vpk * hd)
        .map(|i| {
            let (j, rem) = (i / (nk * hd), i % (nk * hd));
            let (g, d) = (rem / hd, rem % hd);
            (g * vpk + j) * hd + d
        })
        .collect()
}

/// Rows `[first, first + perm.len())` of a row-major plane put in `perm`
/// order; rows outside the range pass through. `row` is a row's byte length.
fn permute_rows(raw: &[u8], row: usize, first: usize, perm: &[usize]) -> Vec<u8> {
    let mut out = raw.to_vec();
    for (i, &p) in perm.iter().enumerate() {
        let (dst, src) = ((first + i) * row, (first + p) * row);
        out[dst..dst + row].copy_from_slice(&raw[src..src + row]);
    }
    out
}

/// Every row's elements in `perm` order (`elem` bytes each) - a column
/// permutation of a row-major plane.
fn permute_cols(raw: &[u8], cols: usize, elem: usize, perm: &[usize]) -> Vec<u8> {
    let row = cols * elem;
    let mut out = vec![0u8; raw.len()];
    for (r, src_row) in raw.chunks_exact(row).enumerate() {
        let dst_row = &mut out[r * row..(r + 1) * row];
        for (i, &p) in perm.iter().enumerate() {
            dst_row[i * elem..(i + 1) * elem].copy_from_slice(&src_row[p * elem..(p + 1) * elem]);
        }
    }
    out
}

fn permute_f32(v: &[f32], first: usize, perm: &[usize], width: usize) -> Vec<f32> {
    let mut out = v.to_vec();
    for (i, &p) in perm.iter().enumerate() {
        let (dst, src) = ((first + i) * width, (first + p) * width);
        out[dst..dst + width].copy_from_slice(&v[src..src + width]);
    }
    out
}

/// The rope's `(cos, sin)` for every position below `positions` and every
/// rotated pair, `[positions][pairs][2]`, as the reference forms them:
/// `inv_freq[k] = 1 / theta^(2k / n_rot)` in F32 (the power correctly
/// rounded - Transformers' own F32 formation, bit for bit on all 32 of
/// Clef's), the angle one F32 product `position * inv_freq[k]`, then cos and
/// sin of that F32 angle evaluated in F64 and rounded once.
pub(super) fn rope_table(theta: f32, n_rot: usize, positions: usize) -> Vec<f32> {
    let pairs = n_rot / 2;
    let inv: Vec<f32> = (0..pairs)
        .map(|k| {
            let e = (2 * k) as f32 / n_rot as f32;
            1.0f32 / f64::from(theta).powf(f64::from(e)) as f32
        })
        .collect();
    let mut t = Vec::with_capacity(positions * pairs * 2);
    for p in 0..positions {
        for &f in &inv {
            let angle = f64::from(p as f32 * f);
            t.push(angle.cos() as f32);
            t.push(angle.sin() as f32);
        }
    }
    t
}

/// Interleaved mrope: pair k reads axis h when `k % 3 == 1` and `k < 3 * h`
/// sections, w when `k % 3 == 2` and `k < 3 * w`, t otherwise
/// (`apply_interleaved_mrope`).
pub(super) fn rope_masks([_, sh, sw]: [u32; 3], pairs: usize) -> (u32, u32) {
    let (mut h, mut w) = (0u32, 0u32);
    for k in 0..pairs as u32 {
        if k % 3 == 1 && k < 3 * sh {
            h |= 1 << k;
        } else if k % 3 == 2 && k < 3 * sw {
            w |= 1 << k;
        }
    }
    (h, w)
}

impl GpuClef {
    /// Load the backbone's text tower from a Clef checkpoint directory. The
    /// joint schema head is loaded with it (`super::head`).
    pub fn load(exec: Arc<GpuExecutor>, dir: &Path) -> Result<Self, GpuModelError> {
        let bad = |e: String| GpuModelError::Unsupported(format!("Clef: {e}"));
        if !exec.has_bf16_dense() || !exec.has_clef() {
            return Err(bad(
                "this kernel pack predates the Clef lane (BF16 dense, slots 722 and 724-731) - \
                 rebuild or update the pack"
                    .into(),
            ));
        }
        let cfg = ClefConfig::read(dir).map_err(bad)?;
        let st = ShardedSafetensors::open_dir(dir).map_err(|e| bad(e.to_string()))?;
        let (h, f) = (cfg.hidden, cfg.ffn);
        let (nk, hv, kd, vd) = (
            cfg.gdn_k_heads,
            cfg.gdn_v_heads,
            cfg.gdn_k_dim,
            cfg.gdn_v_dim,
        );
        if kd != 128 || vd != 128 {
            return Err(bad(format!(
                "DeltaNet heads of {kd}/{vd} - the chunked scan reads 128"
            )));
        }
        let vpk = hv / nk;
        let rows = MAX_ROWS;
        // images: a checkpoint with a vision tower on a pack with the image
        // lane (an older pack serves the text lane and refuses images)
        let vision = cfg.vision.is_some() && exec.has_clef_vision();
        if cfg.vision.is_some() && !vision {
            eprintln!(
                "[clef] this kernel pack predates Clef's image lane (slots 735-740): \
                 images are refused until it is updated"
            );
        }

        // the gate: every byte this load places, before the first one
        let text_bytes: u64 = st
            .names()
            .filter(|n| {
                n.starts_with("model.language_model.")
                    || *n == "lm_head.weight"
                    || (vision && n.starts_with("model.visual."))
            })
            .filter_map(|n| st.bytes(n).map(|(_, b)| b.len() as u64))
            .sum();
        let head_bytes = std::fs::metadata(dir.join(paddock_models::clef::HEAD_WEIGHTS))
            .map(|m| m.len())
            .unwrap_or(0);
        exec.vram_load_gate(
            text_bytes
                + 2 * head_bytes
                + Workspace::bytes(&cfg, rows, vision)
                + 4 * super::head::HeadWs::floats(&cfg.head, cfg.hidden) as u64
                + (64 << 20),
            "Clef",
        )
        .map_err(GpuModelError::WontFit)?;
        exec.disable_event_tracking();
        // Batch invariance: a request's answer must not depend on what else
        // shared its pass. The BF16 GEMM elects a K-split when its grid is
        // small (bf16_dense.cuh `pd_bf16ks_nz`) - the k/v projections of one
        // short request take it, the same rows in a fuller pass do not - and
        // a K-split sums in a different order. Every other config moves tile
        // ownership, never the k sequence, so with the split off a row's
        // bits are its own. Elected for the process (a runner serves one
        // model); the pack reads it once, at the first GEMM.
        if std::env::var_os("PADDOCK_NO_BF16_KSPLIT").is_none() {
            crate::envset::set_env("PADDOCK_NO_BF16_KSPLIT", "1");
        }

        let weight_bytes = std::cell::Cell::new(0u64);
        let up_f32 = |v: &[f32]| -> Result<CudaSlice<f32>, GpuModelError> {
            weight_bytes.set(weight_bytes.get() + 4 * v.len() as u64);
            Ok(exec.to_device(v)?)
        };
        let table = |raw: &[u8], n: usize, k: usize| -> Result<QuantTensor, GpuModelError> {
            weight_bytes.set(weight_bytes.get() + raw.len() as u64);
            Ok(QuantTensor {
                bytes: exec.to_device_u8(raw)?,
                ty: GgmlType::Bf16,
                dims: vec![k, n],
            })
        };
        let plane =
            |raw: &[u8], n: usize, k: usize| Ok::<_, GpuModelError>(Proj::Bf16(table(raw, n, k)?));
        let lm = |n: &str| format!("model.language_model.{n}");
        let norm_1p = |name: &str, n: usize| -> Result<CudaSlice<f32>, GpuModelError> {
            let v: Vec<f32> = f32_tensor(&st, name, n)?
                .into_iter()
                .map(|w| w + 1.0)
                .collect();
            up_f32(&v)
        };

        let vperm = tiled_perm(nk, vpk, vd);
        let hperm = tiled_perm(nk, vpk, 1);
        let qk_rows = 2 * nk * kd;

        let mut layers = Vec::with_capacity(cfg.blocks.len());
        for (i, block) in cfg.blocks.iter().enumerate() {
            let p = |n: &str| lm(&format!("layers.{i}.{n}"));
            let mixer = match block {
                ClefBlock::Gdn => {
                    let qkv_rows = cfg.gdn_qkv_rows();
                    let qkv = bf16_bytes(&st, &p("linear_attn.in_proj_qkv.weight"), qkv_rows * h)?;
                    let qkv = permute_rows(qkv, 2 * h, qk_rows, &vperm);
                    let z = bf16_bytes(&st, &p("linear_attn.in_proj_z.weight"), hv * vd * h)?;
                    let z = permute_rows(z, 2 * h, 0, &vperm);
                    let a = bf16_bytes(&st, &p("linear_attn.in_proj_a.weight"), hv * h)?;
                    let b = bf16_bytes(&st, &p("linear_attn.in_proj_b.weight"), hv * h)?;
                    let mut ab = permute_rows(a, 2 * h, 0, &hperm);
                    ab.extend(permute_rows(b, 2 * h, 0, &hperm));
                    // [qkv_rows, 1, conv] squeezed; the v channels tiled
                    let conv = f32_tensor(
                        &st,
                        &p("linear_attn.conv1d.weight"),
                        qkv_rows * cfg.gdn_conv,
                    )?;
                    let conv = permute_f32(&conv, qk_rows, &vperm, cfg.gdn_conv);
                    let a_log = f32_tensor(&st, &p("linear_attn.A_log"), hv)?;
                    let ssm_a: Vec<f32> = a_log.iter().map(|x| -x.exp()).collect();
                    let ssm_a = permute_f32(&ssm_a, 0, &hperm, 1);
                    let dt_bias = f32_tensor(&st, &p("linear_attn.dt_bias"), hv)?;
                    let dt_bias = permute_f32(&dt_bias, 0, &hperm, 1);
                    let out = bf16_bytes(&st, &p("linear_attn.out_proj.weight"), h * hv * vd)?;
                    let out = permute_cols(out, hv * vd, 2, &vperm);
                    Mixer::Gdn(Gdn {
                        qkv: plane(&qkv, qkv_rows, h)?,
                        z: plane(&z, hv * vd, h)?,
                        ab: plane(&ab, 2 * hv, h)?,
                        conv: up_f32(&conv)?,
                        ssm_a: up_f32(&ssm_a)?,
                        dt_bias: up_f32(&dt_bias)?,
                        norm: up_f32(&f32_tensor(&st, &p("linear_attn.norm.weight"), vd)?)?,
                        out: plane(&out, h, hv * vd)?,
                    })
                }
                ClefBlock::Attention => {
                    let (qr, kvw, qw) = (cfg.attn_q_rows(), cfg.kv_width(), cfg.q_width());
                    Mixer::Attn(Attn {
                        q: plane(
                            bf16_bytes(&st, &p("self_attn.q_proj.weight"), qr * h)?,
                            qr,
                            h,
                        )?,
                        k: plane(
                            bf16_bytes(&st, &p("self_attn.k_proj.weight"), kvw * h)?,
                            kvw,
                            h,
                        )?,
                        v: plane(
                            bf16_bytes(&st, &p("self_attn.v_proj.weight"), kvw * h)?,
                            kvw,
                            h,
                        )?,
                        q_norm: norm_1p(&p("self_attn.q_norm.weight"), cfg.head_dim)?,
                        k_norm: norm_1p(&p("self_attn.k_norm.weight"), cfg.head_dim)?,
                        o: plane(
                            bf16_bytes(&st, &p("self_attn.o_proj.weight"), h * qw)?,
                            h,
                            qw,
                        )?,
                    })
                }
            };
            layers.push(Layer {
                in_norm: norm_1p(&p("input_layernorm.weight"), h)?,
                post_norm: norm_1p(&p("post_attention_layernorm.weight"), h)?,
                mixer,
                gate_up: {
                    let g = bf16_bytes(&st, &p("mlp.gate_proj.weight"), f * h)?;
                    let u = bf16_bytes(&st, &p("mlp.up_proj.weight"), f * h)?;
                    let row = 2 * h;
                    let mut gu = Vec::with_capacity(2 * f * row);
                    for r in 0..f {
                        gu.extend_from_slice(&g[r * row..(r + 1) * row]);
                        gu.extend_from_slice(&u[r * row..(r + 1) * row]);
                    }
                    plane(&gu, 2 * f, h)?
                },
                down: plane(bf16_bytes(&st, &p("mlp.down_proj.weight"), h * f)?, h, f)?,
            });
        }
        let embed = table(
            bf16_bytes(&st, &lm("embed_tokens.weight"), cfg.vocab * h)?,
            cfg.vocab,
            h,
        )?;
        let lm_head = table(
            bf16_bytes(&st, "lm_head.weight", cfg.vocab * h)?,
            cfg.vocab,
            h,
        )?;
        let final_norm = norm_1p(&lm("norm.weight"), h)?;
        if cfg.n_rot != 64 || cfg.head_dim != 256 {
            return Err(bad(format!(
                "rope over {} of {} dims - the backbone kernels rotate 64 of 256",
                cfg.n_rot, cfg.head_dim
            )));
        }
        let rope = up_f32(&rope_table(cfg.rope_theta, cfg.n_rot, rows))?;
        let rope_masks = rope_masks(cfg.mrope_sections, cfg.n_rot / 2);
        let head = super::head::HeadW::load(&exec, dir, &cfg.head, h)?;
        let vision = match cfg.vision.as_ref().filter(|_| vision) {
            Some((v, _)) => Some(super::vision::Vision::load(&exec, &st, v)?),
            None => None,
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

impl Workspace {
    /// Per row, the widths of the three planes the vision tower shares
    /// (`wide`, `dq`, `ffn_g`): each the larger of the backbone's need and
    /// four patches' (a pass's patches are at most four times its rows).
    fn shared(cfg: &ClefConfig, vision: bool) -> (usize, usize, usize) {
        let wide = cfg.gdn_qkv_rows().max(cfg.attn_q_rows());
        let wv = cfg.gdn_v_width().max(cfg.q_width());
        match cfg.vision.as_ref().filter(|_| vision) {
            Some((v, _)) => {
                let (x, xn, w) = super::vision::plane_widths(v);
                (wide.max(4 * x), wv.max(4 * xn), cfg.ffn.max(4 * w))
            }
            None => (wide, wv, cfg.ffn),
        }
    }

    fn sizes(cfg: &ClefConfig, rows: usize, vision: bool) -> (usize, usize, usize) {
        let h = cfg.hidden;
        let vw = cfg.gdn_v_width();
        let (wide, dq, ffn) = Self::shared(cfg, vision);
        let per_row = 2 * h
            + wide
            + dq
            + 4 * vw.max(cfg.q_width())
            + cfg.q_width()
            + 2 * cfg.gdn_v_heads * 2
            + 3 * cfg.kv_width()
            + ffn;
        let nc = rows.div_ceil(64);
        let dn = 2 * nc * cfg.gdn_v_heads * 64 * cfg.gdn_v_dim
            + nc * cfg.gdn_v_heads * 64 * 64
            + cfg.gdn_v_heads * cfg.gdn_k_dim * cfg.gdn_v_dim;
        (per_row, dn, nc * cfg.gdn_v_heads * 64)
    }

    pub(super) fn bytes(cfg: &ClefConfig, rows: usize, vision: bool) -> u64 {
        let (per_row, dn, cg) = Self::sizes(cfg, rows, vision);
        (4 * (rows * per_row + dn) + 8 * cg) as u64
    }

    pub(super) fn new(
        e: &GpuExecutor,
        cfg: &ClefConfig,
        rows: usize,
        vision: bool,
    ) -> Result<Self, GpuModelError> {
        let p = |n: usize| e.alloc(rows * n);
        let (h, vw, qw, kvw) = (cfg.hidden, cfg.gdn_v_width(), cfg.q_width(), cfg.kv_width());
        let wv = vw.max(qw);
        let nc = rows.div_ceil(64);
        let hv = cfg.gdn_v_heads;
        let (wide, dq, ffn) = Self::shared(cfg, vision);
        Ok(Self {
            x: p(h)?,
            xn: p(h)?,
            wide: p(wide)?,
            dq: p(dq)?,
            dk: p(wv)?,
            dv: p(wv)?,
            z: p(wv)?,
            ab: p(2 * hv)?,
            g: p(hv)?,
            beta: p(hv)?,
            core: p(wv)?,
            out: p(qw)?,
            k: p(kvw)?,
            kn: p(kvw)?,
            v: p(kvw)?,
            ffn_g: p(ffn)?,
            state: e.alloc(hv * cfg.gdn_k_dim * cfg.gdn_v_dim)?,
            dnc_dw: e.alloc(nc * hv * 64 * cfg.gdn_v_dim)?,
            dnc_du: e.alloc(nc * hv * 64 * cfg.gdn_v_dim)?,
            dnc_aqk: e.alloc(nc * hv * 64 * 64)?,
            dnc_cg: e.alloc_f64(nc * hv * 64)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The converter's `_reorder_v_heads`: reshape `[nk, vpk, hd]`, swap the
    /// first two axes.
    #[test]
    fn rope_frequencies_and_axes() {
        // the formation Transformers uses for Clef's 32 frequencies (f32
        // pow, f32 reciprocal) - pinned values from its module
        let t = rope_table(1e7, 64, 2);
        assert_eq!(t[2], 1.0); // position 0: cos 1
        // position 1, pair 1: index (1 * 32 + 1) * 2
        assert_eq!(t[66], (f64::from(0.6042964f32)).cos() as f32);
        let (h, w) = rope_masks([11, 11, 10], 32);
        assert_eq!(h.count_ones(), 11);
        assert_eq!(w.count_ones(), 10);
        assert_eq!(h & w, 0);
        assert_eq!(h & 1, 0); // pair 0 reads t
        assert!(h & 2 != 0 && w & 4 != 0);
    }

    #[test]
    fn tiled_order_is_the_converters() {
        let (nk, vpk, hd) = (3, 2, 2);
        let perm = tiled_perm(nk, vpk, hd);
        // grouped index of (g, j, d) = (g*vpk + j)*hd + d; tiled position
        // of the same element = (j*nk + g)*hd + d
        for g in 0..nk {
            for j in 0..vpk {
                for d in 0..hd {
                    assert_eq!(perm[(j * nk + g) * hd + d], (g * vpk + j) * hd + d);
                }
            }
        }
        let rows: Vec<u8> = (0..12u8).collect();
        let out = permute_rows(&rows, 1, 0, &perm);
        assert_eq!(out, [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]);
        let cols = permute_cols(&rows, 12, 1, &perm);
        assert_eq!(cols, out);
    }
}
