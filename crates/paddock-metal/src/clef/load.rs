use super::*;
use paddock_models::safetensors::{SafetensorsFile, ShardedSafetensors, StDtype};

pub(super) struct Source {
    backbone: ShardedSafetensors,
    head: SafetensorsFile,
    mlx: bool,
}
impl Source {
    fn entry(&self, name: &str) -> Result<(&paddock_models::safetensors::StTensor, &[u8])> {
        let mapped;
        let name = if self.mlx && name.starts_with("model.language_model.") {
            mapped = name.replacen("model.language_model.", "language_model.model.", 1);
            mapped.as_str()
        } else if self.mlx && name.starts_with("lm_head.") {
            mapped = format!("language_model.{name}");
            mapped.as_str()
        } else {
            name
        };
        let entry = if name.starts_with("model.")
            || name.starts_with("language_model.")
            || name == "lm_head.weight"
        {
            self.backbone.bytes(name)
        } else {
            self.head.bytes(name)
        };
        entry.ok_or_else(|| error(format!("missing tensor {name}")))
    }
    pub fn raw(&self, name: &str, shape: &[usize]) -> Result<&[u8]> {
        let (t, b) = self.entry(name)?;
        if t.dtype != StDtype::Bf16 || t.shape != shape {
            return Err(error(format!(
                "{name}: expected BF16 {shape:?}, got {:?} {:?}",
                t.dtype, t.shape
            )));
        }
        Ok(b)
    }
    pub fn values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (t, b) = self.entry(name)?;
        let conv = self.mlx
            && name.ends_with("conv1d.weight")
            && shape.len() == 3
            && t.shape == [shape[0], shape[2], shape[1]];
        if t.shape != shape && !conv {
            return Err(error(format!(
                "{name}: shape {:?}, expected {shape:?}",
                t.shape
            )));
        }
        let values = match t.dtype {
            StDtype::Bf16 => b
                .chunks_exact(2)
                .map(|v| f32::from_bits((u16::from_le_bytes([v[0], v[1]]) as u32) << 16))
                .collect::<Vec<_>>(),
            StDtype::F32 => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect(),
            _ => {
                return Err(error(format!(
                    "{name}: unsupported vector dtype {:?}",
                    t.dtype
                )));
            }
        };
        if values.iter().any(|x| !x.is_finite()) {
            return Err(error(format!("non-finite {name}")));
        }
        Ok(values)
    }
    pub fn vector(&self, d: &MetalDevice, name: &str, n: usize, add: f32) -> Result<Buffer> {
        let values = self.values(name, &[n])?;
        let add = if self.mlx { 0. } else { add };
        upload(d, &values.iter().map(|x| x + add).collect::<Vec<_>>())
    }
    pub fn linear(
        &self,
        d: &MetalDevice,
        name: &str,
        n: usize,
        k: usize,
        bias: bool,
    ) -> Result<Linear> {
        if self.mlx && (name.starts_with("model.") || name == "lm_head") {
            if bias {
                return Err(error("quantized backbone bias is unsupported"));
            }
            return Ok(Linear {
                weight: self.planes(d, &[&format!("{name}.weight")], n, k, false)?,
                bias: None,
                k,
                n,
            });
        }
        self.rows(d, name, 0..n, n, k, bias)
    }
    pub fn rows(
        &self,
        d: &MetalDevice,
        name: &str,
        rows: std::ops::Range<usize>,
        n: usize,
        k: usize,
        bias: bool,
    ) -> Result<Linear> {
        let (wn, bn) = match name.strip_suffix(".in_proj") {
            Some(base) => (
                format!("{base}.in_proj_weight"),
                format!("{base}.in_proj_bias"),
            ),
            None => (format!("{name}.weight"), format!("{name}.bias")),
        };
        let weight = d.upload(&self.raw(&wn, &[n, k])?[rows.start * k * 2..rows.end * k * 2])?;
        let b = if bias {
            Some(upload(d, &self.values(&bn, &[n])?[rows.clone()])?)
        } else {
            None
        };
        Ok(Linear {
            weight: weight.into(),
            bias: b,
            k,
            n: rows.len(),
        })
    }
    pub fn norm(
        &self,
        d: &MetalDevice,
        name: &str,
        n: usize,
        eps: f32,
        layer_norm: bool,
    ) -> Result<Norm> {
        Ok(Norm {
            weight: self.vector(
                d,
                &format!("{name}.weight"),
                n,
                if layer_norm { 0. } else { 1. },
            )?,
            bias: if layer_norm {
                Some(self.vector(d, &format!("{name}.bias"), n, 0.)?)
            } else {
                None
            },
            width: n,
            eps,
        })
    }
    fn planes(
        &self,
        d: &MetalDevice,
        names: &[&str],
        n: usize,
        k: usize,
        interleave: bool,
    ) -> Result<quant::Plane> {
        let quant = self.entry(names[0])?.0.dtype == StDtype::U32;
        if quant && (!self.mlx || !k.is_multiple_of(64)) {
            return Err(error("invalid affine-8 plane"));
        }
        let gather = |suffix: &str, width: usize, dtype: StDtype| -> Result<Buffer> {
            let mut parts = Vec::new();
            for name in names {
                let key = if suffix.is_empty() {
                    name.to_string()
                } else {
                    format!(
                        "{}.{suffix}",
                        name.strip_suffix(".weight")
                            .ok_or_else(|| error("quantized tensor has no weight suffix"))?
                    )
                };
                let (t, b) = self.entry(&key)?;
                let expected = if dtype == StDtype::U32 {
                    vec![n, k / 4]
                } else {
                    vec![n, width / 2]
                };
                if t.dtype != dtype || t.shape != expected {
                    return Err(error(format!(
                        "{key}: expected {dtype:?} {expected:?}, got {:?} {:?}",
                        t.dtype, t.shape
                    )));
                }
                parts.push(b);
            }
            if !interleave {
                return d.upload_parts(&parts);
            }
            d.upload_with(n * width * parts.len(), |out| {
                for row in 0..n {
                    for (i, part) in parts.iter().enumerate() {
                        out[(row * parts.len() + i) * width..(row * parts.len() + i + 1) * width]
                            .copy_from_slice(&part[row * width..(row + 1) * width]);
                    }
                }
                Ok(())
            })
        };
        if !quant {
            return Ok(gather("", k * 2, StDtype::Bf16)?.into());
        }
        Ok(quant::Plane {
            data: gather("", k, StDtype::U32)?,
            kind: 2,
            affine: Some((
                gather("scales", k / 64 * 2, StDtype::Bf16)?,
                gather("biases", k / 64 * 2, StDtype::Bf16)?,
            )),
        })
    }
}

pub(super) fn validate(c: &ClefConfig) -> Result<()> {
    // Kernel geometry is a serving contract, not a best-effort reinterpretation.
    let family = (c.hidden, c.ffn, c.blocks.len(), c.n_heads, c.gdn_v_heads);
    if !matches!(
        family,
        (4096, 12288, 32, 16, 32) | (5120, 17408, 64, 24, 48)
    ) || c.vocab != 248320
        || (c.n_kv_heads, c.head_dim, c.n_rot) != (4, 256, 64)
        || (c.gdn_k_heads, c.gdn_k_dim, c.gdn_v_dim, c.gdn_conv) != (16, 128, 128, 4)
        || (
            c.head.width,
            c.head.heads,
            c.head.routing_layers,
            c.head.layers,
            c.head.feedforward,
        ) != (1024, 16, 2, 4, 4096)
        || !c.eps.is_finite()
        || c.eps <= 0.
        || !c.rope_theta.is_finite()
        || c.rope_theta <= 0.
    {
        return Err(error(
            "unsupported Clef checkpoint geometry; expected Clef Flash 9B or Clef 27B",
        ));
    }
    Ok(())
}
pub(super) fn load(dir: &Path, budget: Option<u64>) -> Result<Clef> {
    let config = ClefConfig::read(dir).map_err(error)?;
    validate(&config)?;
    let json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("config.json")).map_err(|e| error(e.to_string()))?,
    )
    .map_err(|e| error(e.to_string()))?;
    let mlx = json.get("quantization").is_some();
    if mlx
        && (json["quantization"]["bits"] != 8
            || json["quantization"]["group_size"] != 64
            || json["quantization"]["mode"] != "affine")
    {
        return Err(error("only MLX affine-8/group-64 is supported for Clef"));
    }
    let s = Source {
        mlx,
        backbone: ShardedSafetensors::open_dir(dir).map_err(|e| error(e.to_string()))?,
        head: SafetensorsFile::open(&dir.join(paddock_models::clef::HEAD_WEIGHTS))
            .map_err(|e| error(e.to_string()))?,
    };
    let backbone_bytes = s
        .backbone
        .names()
        .filter(|n| {
            n.starts_with("model.language_model.")
                || n.starts_with("language_model.")
                || n.as_str() == "lm_head.weight"
        })
        .map(|n| s.backbone.bytes(n).map_or(0, |(_, b)| b.len() as u64))
        .sum::<u64>();
    let head_bytes = s
        .head
        .tensors()
        .values()
        .map(|t| (t.end - t.begin) as u64)
        .sum::<u64>();
    let fixed_workspace = workspace::Workspace::bytes(&config);
    let vision_bytes = s
        .backbone
        .names()
        .filter(|n| n.starts_with("model.visual.") || n.starts_with("vision_tower."))
        .map(|n| s.backbone.bytes(n).map_or(0, |(_, b)| b.len() as u64))
        .sum::<u64>();
    let workspace_bytes = fixed_workspace
        + config
            .vision
            .as_ref()
            .map_or(0, |_| vision::workspace_bytes(config.hidden));
    // Head/norm widening and rotary tables are included before allocating any
    // weights. Large matrices stay in their checkpoint format, not widened.
    let device = MetalDevice::new_planned(
        budget,
        backbone_bytes + vision_bytes + head_bytes * 2 + workspace_bytes + (32 << 20),
    )?;
    let c = &config;
    let embed = s.planes(
        &device,
        &["model.language_model.embed_tokens.weight"],
        c.vocab,
        c.hidden,
        false,
    )?;
    let lexical = s.planes(&device, &["lm_head.weight"], c.vocab, c.hidden, false)?;
    let mut layers = Vec::with_capacity(c.blocks.len());
    for (i, kind) in c.blocks.iter().enumerate() {
        let p = format!("model.language_model.layers.{i}");
        let mixer = match kind {
            ClefBlock::Gdn => {
                let p = format!("{p}.linear_attn");
                let ab = s.planes(
                    &device,
                    &[
                        &format!("{p}.in_proj_a.weight"),
                        &format!("{p}.in_proj_b.weight"),
                    ],
                    c.gdn_v_heads,
                    c.hidden,
                    false,
                )?;
                let a = s
                    .values(&format!("{p}.A_log"), &[c.gdn_v_heads])?
                    .iter()
                    .map(|v| -v.exp())
                    .collect::<Vec<_>>();
                if a.iter().any(|v| !v.is_finite()) {
                    return Err(error(format!("overflow in {p}.A_log")));
                }
                let conv = upload(
                    &device,
                    &s.values(&format!("{p}.conv1d.weight"), &[c.gdn_qkv_rows(), 1, 4])?,
                )?;
                Mixer::Delta(Delta {
                    qkv: s.linear(
                        &device,
                        &format!("{p}.in_proj_qkv"),
                        c.gdn_qkv_rows(),
                        c.hidden,
                        false,
                    )?,
                    z: s.linear(
                        &device,
                        &format!("{p}.in_proj_z"),
                        c.gdn_v_width(),
                        c.hidden,
                        false,
                    )?,
                    ab: Linear {
                        weight: ab,
                        bias: None,
                        k: c.hidden,
                        n: 2 * c.gdn_v_heads,
                    },
                    out: s.linear(
                        &device,
                        &format!("{p}.out_proj"),
                        c.hidden,
                        c.gdn_v_width(),
                        false,
                    )?,
                    conv,
                    a: upload(&device, &a)?,
                    dt: s.vector(&device, &format!("{p}.dt_bias"), c.gdn_v_heads, 0.)?,
                    norm: s.vector(&device, &format!("{p}.norm.weight"), c.gdn_v_dim, 0.)?,
                })
            }
            ClefBlock::Attention => {
                let p = format!("{p}.self_attn");
                Mixer::Attention(Attention {
                    q: s.linear(
                        &device,
                        &format!("{p}.q_proj"),
                        c.attn_q_rows(),
                        c.hidden,
                        false,
                    )?,
                    k: s.linear(
                        &device,
                        &format!("{p}.k_proj"),
                        c.kv_width(),
                        c.hidden,
                        false,
                    )?,
                    v: s.linear(
                        &device,
                        &format!("{p}.v_proj"),
                        c.kv_width(),
                        c.hidden,
                        false,
                    )?,
                    out: s.linear(
                        &device,
                        &format!("{p}.o_proj"),
                        c.hidden,
                        c.q_width(),
                        false,
                    )?,
                    qnorm: s.vector(&device, &format!("{p}.q_norm.weight"), c.head_dim, 1.)?,
                    knorm: s.vector(&device, &format!("{p}.k_norm.weight"), c.head_dim, 1.)?,
                })
            }
        };
        let gu = s.planes(
            &device,
            &[
                &format!("{p}.mlp.gate_proj.weight"),
                &format!("{p}.mlp.up_proj.weight"),
            ],
            c.ffn,
            c.hidden,
            true,
        )?;
        layers.push(Layer {
            norm: s.norm(
                &device,
                &format!("{p}.input_layernorm"),
                c.hidden,
                c.eps,
                false,
            )?,
            post: s.norm(
                &device,
                &format!("{p}.post_attention_layernorm"),
                c.hidden,
                c.eps,
                false,
            )?,
            mixer,
            gate_up: Linear {
                weight: gu,
                bias: None,
                k: c.hidden,
                n: c.ffn * 2,
            },
            down: s.linear(
                &device,
                &format!("{p}.mlp.down_proj"),
                c.hidden,
                c.ffn,
                false,
            )?,
        });
    }
    let norm = s.norm(&device, "model.language_model.norm", c.hidden, c.eps, false)?;
    let head = head::Head::load(&device, &s, c)?;
    let mut angles = Vec::with_capacity(MAX_ROWS * c.n_rot);
    for pos in 0..MAX_ROWS {
        for pair in 0..c.n_rot / 2 {
            let freq = 1.0 / c.rope_theta.powf((2 * pair) as f32 / c.n_rot as f32);
            let angle = pos as f32 * freq;
            angles.extend([angle.cos(), angle.sin()]);
        }
    }
    let rope = upload(&device, &angles)?;
    let vision = config
        .vision
        .as_ref()
        .map(|(v, _)| vision::Vision::load(&device, &s.backbone, v, mlx))
        .transpose()?;
    let weight_bytes = device.allocated_bytes();
    let ws = workspace::Workspace::new(&device, c)?;
    debug_assert_eq!(device.allocated_bytes() - weight_bytes, fixed_workspace);
    tracing::info!(
        weight_bytes,
        workspace_bytes,
        "Clef native Metal decision graph loaded"
    );
    Ok(Clef {
        device,
        config,
        embed,
        lexical,
        tiled_heads: false,
        rms_qk: mlx,
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
