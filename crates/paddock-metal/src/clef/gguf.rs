//! Native ggml-org Clef Q8_0 loading. Preserve the converter's tiled value-head
//! order, pre-offset RMS norms and already-transformed gates/scales exactly.
use super::*;
use paddock_models::{ggml_type::GgmlType, mapped::MappedGguf};

pub(super) struct Source<'a>(pub &'a MappedGguf);
impl Source<'_> {
    fn raw(&self, name: &str, shape: &[usize]) -> Result<(GgmlType, &[u8])> {
        let (t, b) = self
            .0
            .tensor_bytes(name)
            .map_err(|e| error(e.to_string()))?;
        let trim = |s: &[usize]| {
            let mut s = s.to_vec();
            while s.len() > 1 && s.last() == Some(&1) {
                s.pop();
            }
            s
        };
        let dims = t.dims.iter().map(|&d| d as usize).collect::<Vec<_>>();
        if trim(&dims) != trim(shape) {
            return Err(error(format!("{name}: expected {shape:?}, got {dims:?}")));
        }
        Ok((t.ggml_type, b))
    }
    pub fn values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (ty, b) = self.raw(name, shape)?;
        let v = match ty {
            GgmlType::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
            GgmlType::Bf16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                .collect(),
            GgmlType::F16 => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_le_bytes(*b).to_f32())
                .collect(),
            GgmlType::Q8_0 => b
                .as_chunks::<34>()
                .0
                .iter()
                .flat_map(|b| {
                    let scale = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                    b[2..].iter().map(move |&q| scale * q as i8 as f32)
                })
                .collect(),
            _ => return Err(error(format!("{name}: unsupported small tensor {ty:?}"))),
        };
        if v.iter().any(|v| !v.is_finite()) {
            return Err(error(format!("non-finite {name}")));
        }
        Ok(v)
    }
    pub fn vector(&self, d: &MetalDevice, name: &str, n: usize) -> Result<Buffer> {
        upload(d, &self.values(name, &[n])?)
    }
    pub fn norm(
        &self,
        d: &MetalDevice,
        name: &str,
        n: usize,
        eps: f32,
        bias: bool,
    ) -> Result<Norm> {
        Ok(Norm {
            weight: self.vector(d, &format!("{name}.weight"), n)?,
            bias: if bias {
                Some(self.vector(d, &format!("{name}.bias"), n)?)
            } else {
                None
            },
            width: n,
            eps,
        })
    }
    pub fn plane(
        &self,
        d: &MetalDevice,
        names: &[&str],
        k: usize,
        n: usize,
        interleave: bool,
    ) -> Result<quant::Plane> {
        let mut parts = Vec::new();
        let mut kind = None;
        for name in names {
            let (ty, b) = self.raw(name, &[k, n])?;
            if !matches!(ty, GgmlType::Q8_0 | GgmlType::Bf16) || kind.is_some_and(|old| old != ty) {
                return Err(error(format!(
                    "{name}: expected homogeneous Q8_0 or BF16 planes"
                )));
            }
            kind = Some(ty);
            parts.push(b);
        }
        let ty = kind.ok_or_else(|| error("empty plane"))?;
        let row = ty
            .byte_size(k as u64)
            .ok_or_else(|| error("invalid quantized row"))? as usize;
        let data = if !interleave {
            d.upload_parts(&parts)?
        } else {
            d.upload_with(n * row * parts.len(), |out| {
                for r in 0..n {
                    for (i, part) in parts.iter().enumerate() {
                        out[(r * parts.len() + i) * row..(r * parts.len() + i + 1) * row]
                            .copy_from_slice(&part[r * row..(r + 1) * row]);
                    }
                }
                Ok(())
            })?
        };
        Ok(quant::Plane {
            data,
            kind: u32::from(ty == GgmlType::Q8_0),
            affine: None,
        })
    }
    pub fn linear(
        &self,
        d: &MetalDevice,
        bases: &[&str],
        k: usize,
        n: usize,
        bias: bool,
    ) -> Result<Linear> {
        let names = bases
            .iter()
            .map(|b| format!("{b}.weight"))
            .collect::<Vec<_>>();
        let names = names.iter().map(String::as_str).collect::<Vec<_>>();
        let weight = self.plane(d, &names, k, n, false)?;
        let bias = if bias {
            let mut v = Vec::new();
            for b in bases {
                v.extend(self.values(&format!("{b}.bias"), &[n])?);
            }
            Some(upload(d, &v)?)
        } else {
            None
        };
        Ok(Linear {
            weight,
            bias,
            k,
            n: n * bases.len(),
        })
    }
}
pub(super) fn load(path: &Path, budget: Option<u64>) -> Result<Clef> {
    load_with_companion(path, None, budget)
}
pub(super) fn load_with_companion(
    path: &Path,
    companion: Option<&Path>,
    budget: Option<u64>,
) -> Result<Clef> {
    let g = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
    let companion = companion
        .map(|p| {
            paddock_models::safetensors::SafetensorsFile::open(p).map_err(|e| error(e.to_string()))
        })
        .transpose()?;
    let config =
        ClefConfig::from_gguf(g.gguf(), companion.as_ref().map(|f| &f.metadata)).map_err(error)?;
    load::validate(&config)?;
    let file_bytes: u64 = g.tensor_infos().filter_map(|t| t.byte_size()).sum();
    let fixed_workspace = workspace::Workspace::bytes(&config);
    let workspace_bytes = fixed_workspace
        + config
            .vision
            .as_ref()
            .map_or(0, |_| vision::workspace_bytes(config.hidden));
    let vision_bytes = companion.as_ref().map_or(0, |f| {
        f.tensors()
            .values()
            .map(|t| (t.end - t.begin) as u64)
            .sum::<u64>()
    });
    // Small norms/lexical scorer widening + rotary table, no full-plane uplift.
    let device = MetalDevice::new_planned(
        budget,
        file_bytes + vision_bytes + workspace_bytes + (32 << 20),
    )?;
    let s = Source(&g);
    let c = &config;
    let h = c.hidden;
    let mut layers = Vec::new();
    for (i, block) in c.blocks.iter().enumerate() {
        let p = |n: &str| format!("blk.{i}.{n}");
        let one = |name: &str, k, n| s.linear(&device, &[&p(name)], k, n, false);
        let vec = |name: &str, n| s.vector(&device, &p(name), n);
        let mixer = match block {
            ClefBlock::Gdn => Mixer::Delta(Delta {
                qkv: one("attn_qkv", h, c.gdn_qkv_rows())?,
                z: one("attn_gate", h, c.gdn_v_width())?,
                ab: s.linear(
                    &device,
                    &[&p("ssm_alpha"), &p("ssm_beta")],
                    h,
                    c.gdn_v_heads,
                    false,
                )?,
                out: one("ssm_out", c.gdn_v_width(), h)?,
                conv: upload(
                    &device,
                    &s.values(&p("ssm_conv1d.weight"), &[4, c.gdn_qkv_rows()])?,
                )?,
                a: vec("ssm_a", c.gdn_v_heads)?,
                dt: vec("ssm_dt.bias", c.gdn_v_heads)?,
                norm: vec("ssm_norm.weight", 128)?,
            }),
            ClefBlock::Attention => Mixer::Attention(Attention {
                q: one("attn_q", h, c.attn_q_rows())?,
                k: one("attn_k", h, c.kv_width())?,
                v: one("attn_v", h, c.kv_width())?,
                out: one("attn_output", c.q_width(), h)?,
                qnorm: vec("attn_q_norm.weight", 256)?,
                knorm: vec("attn_k_norm.weight", 256)?,
            }),
        };
        layers.push(Layer {
            norm: s.norm(&device, &p("attn_norm"), h, c.eps, false)?,
            post: s.norm(&device, &p("post_attention_norm"), h, c.eps, false)?,
            mixer,
            gate_up: Linear {
                weight: s.plane(
                    &device,
                    &[&p("ffn_gate.weight"), &p("ffn_up.weight")],
                    h,
                    c.ffn,
                    true,
                )?,
                bias: None,
                k: h,
                n: 2 * c.ffn,
            },
            down: one("ffn_down", c.ffn, h)?,
        });
    }
    let embed = s.plane(&device, &["token_embd.weight"], h, c.vocab, false)?;
    let lexical = s.plane(&device, &["output.weight"], h, c.vocab, false)?;
    let norm = s.norm(&device, "output_norm", h, c.eps, false)?;
    let head = head::Head::load_gguf(&device, &s, c)?;
    let mut angles = Vec::with_capacity(MAX_ROWS * c.n_rot);
    for pos in 0..MAX_ROWS {
        for pair in 0..c.n_rot / 2 {
            let freq = 1.0 / c.rope_theta.powf((2 * pair) as f32 / c.n_rot as f32);
            let angle = pos as f32 * freq;
            angles.extend([angle.cos(), angle.sin()]);
        }
    }
    let rope = upload(&device, &angles)?;
    let vision = match (companion, config.vision.as_ref()) {
        (Some(f), Some((v, _))) => Some(vision::Vision::load(
            &device,
            &paddock_models::safetensors::ShardedSafetensors::from_file(f),
            v,
            false,
        )?),
        _ => None,
    };
    let weight_bytes = device.allocated_bytes();
    let ws = workspace::Workspace::new(&device, c)?;
    debug_assert_eq!(device.allocated_bytes() - weight_bytes, fixed_workspace);
    tracing::info!(
        weight_bytes,
        workspace_bytes,
        "Clef Q8_0 native Metal decision graph loaded"
    );
    Ok(Clef {
        device,
        config,
        embed,
        lexical,
        tiled_heads: true,
        rms_qk: false,
        rope,
        layers,
        norm,
        head,
        ws,
        weight_bytes,
        workspace_bytes,
        vision,
        #[cfg(test)]
        trace: None,
    })
}
