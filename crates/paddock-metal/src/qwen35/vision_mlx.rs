//! Sanitized MLX-VLM tower storage shared by Bonsai and small Qwen OCR.
//! Preserve matrix dtypes; expand only norms/biases and transpose patch axes.
use super::*;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("MLX vision: {}", s.into()))
}

pub(super) fn validate_processor(path: &Path, budget: (u64, u64)) -> Result<()> {
    let path = path.join("processor_config.json");
    if std::fs::metadata(&path)
        .map_err(|e| error(e.to_string()))?
        .len()
        > 1 << 20
    {
        return Err(error("processor metadata exceeds 1 MiB"));
    }
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| error(e.to_string()))?)
            .map_err(|e| error(e.to_string()))?;
    validate_image_processor(&v["image_processor"], budget)
}

fn validate_image_processor(p: &serde_json::Value, (min, max): (u64, u64)) -> Result<()> {
    for (key, value) in [
        ("do_convert_rgb", serde_json::json!(true)),
        ("do_normalize", serde_json::json!(true)),
        ("do_rescale", serde_json::json!(true)),
        ("rescale_factor", serde_json::json!(1.0 / 255.0)),
        ("image_mean", serde_json::json!([0.5, 0.5, 0.5])),
        ("image_std", serde_json::json!([0.5, 0.5, 0.5])),
        ("patch_size", serde_json::json!(16)),
        ("temporal_patch_size", serde_json::json!(2)),
        ("merge_size", serde_json::json!(2)),
        ("min_pixels", serde_json::json!(min)),
        ("max_pixels", serde_json::json!(max)),
    ] {
        if p[key] != value {
            return Err(error(format!(
                "unsupported/conflicting image processor {key}"
            )));
        }
    }
    Ok(())
}

fn bytes<'a>(
    source: &'a ShardedSafetensors,
    name: &str,
    dtype: StDtype,
    shape: &[usize],
) -> Result<&'a [u8]> {
    let (info, data) = source
        .bytes(name)
        .ok_or_else(|| error(format!("missing {name}")))?;
    if info.dtype != dtype || info.shape != shape {
        return Err(error(format!(
            "{name}: expected {dtype:?} {shape:?}, got {:?} {:?}",
            info.dtype, info.shape
        )));
    }
    Ok(data)
}

pub(super) fn vision_budget(path: &Path) -> Result<(u64, u64)> {
    let path = path.join("preprocessor_config.json");
    if std::fs::metadata(&path)
        .map_err(|e| error(e.to_string()))?
        .len()
        > 1 << 20
    {
        return Err(error("image processor metadata exceeds 1 MiB"));
    }
    let p: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| error(e.to_string()))?)
            .map_err(|e| error(e.to_string()))?;
    for (key, value) in [
        ("patch_size", serde_json::json!(16)),
        ("temporal_patch_size", serde_json::json!(2)),
        ("merge_size", serde_json::json!(2)),
        ("image_mean", serde_json::json!([0.5, 0.5, 0.5])),
        ("image_std", serde_json::json!([0.5, 0.5, 0.5])),
    ] {
        if p[key] != value {
            return Err(error(format!("unsupported image processor {key}")));
        }
    }
    let min = p["size"]["shortest_edge"]
        .as_u64()
        .filter(|v| *v >= 1024 && v.is_multiple_of(1024))
        .ok_or_else(|| error("invalid image minimum area"))?;
    let max = p["size"]["longest_edge"]
        .as_u64()
        .filter(|v| *v >= min && *v <= 16777216 && v.is_multiple_of(1024))
        .ok_or_else(|| error("invalid image maximum area"))?;
    Ok((min, max))
}

pub(super) fn vision_weight(
    device: &MetalDevice,
    source: &ShardedSafetensors,
    name: &str,
    dims: &[usize],
    half: bool,
) -> Result<Weight> {
    vision_weight_typed(device, source, name, dims, half, StDtype::F16)
}

pub(super) fn patch_weight(
    device: &MetalDevice,
    source: &ShardedSafetensors,
    width: usize,
) -> Result<Weight> {
    let data = bytes(
        source,
        "vision_tower.patch_embed.proj.weight",
        StDtype::Bf16,
        &[width, 2, 16, 16, 3],
    )?;
    if data
        .chunks_exact(2)
        .any(|b| !half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite())
    {
        return Err(error("non-finite MLX patch weight"));
    }
    Ok(Weight {
        buffer: device.upload(data)?,
        ty: 30,
        k: 1536,
        n: width,
    })
}

/// Shared sanitized MLX-VLM tower layout. Matrix planes retain their native
/// 16-bit dtype; patch operands use the encoder's F16 patchification contract.
pub(super) fn vision_weight_typed(
    device: &MetalDevice,
    source: &ShardedSafetensors,
    name: &str,
    dims: &[usize],
    half: bool,
    dtype: StDtype,
) -> Result<Weight> {
    let hf = match name {
        "v.patch_embd.weight" | "v.patch_embd.weight.1" => {
            "vision_tower.patch_embed.proj.weight".into()
        }
        "v.patch_embd.bias" => "vision_tower.patch_embed.proj.bias".into(),
        "v.position_embd.weight" => "vision_tower.pos_embed.weight".into(),
        "v.post_ln.weight" => "vision_tower.merger.norm.weight".into(),
        "v.post_ln.bias" => "vision_tower.merger.norm.bias".into(),
        n if n.starts_with("mm.") => n
            .replacen("mm.0.", "vision_tower.merger.linear_fc1.", 1)
            .replacen("mm.2.", "vision_tower.merger.linear_fc2.", 1),
        n => n
            .replacen("v.blk.", "vision_tower.blocks.", 1)
            .replace("ln1.", "norm1.")
            .replace("ln2.", "norm2.")
            .replace("attn_qkv.", "attn.qkv.")
            .replace("attn_out.", "attn.proj.")
            .replace("ffn_up.", "mlp.linear_fc1.")
            .replace("ffn_down.", "mlp.linear_fc2."),
    };
    let patch = name == "v.patch_embd.weight" || name == "v.patch_embd.weight.1";
    let shape = if patch {
        vec![dims[3], 2, 16, 16, 3]
    } else {
        dims.iter().copied().rev().collect()
    };
    let data = bytes(source, &hf, dtype, &shape)?;
    let bf16 = dtype == StDtype::Bf16;
    let count = dims.iter().product::<usize>();
    let buffer = device.upload_with(count * if half { 2 } else { 4 }, |out| {
        for i in 0..count {
            let source_index = if patch {
                let row = i / 768;
                let col = i % 768;
                ((row * 2 + usize::from(name.ends_with(".1"))) * 256 + col % 256) * 3 + col / 256
            } else {
                i
            };
            let bits = u16::from_le_bytes([data[source_index * 2], data[source_index * 2 + 1]]);
            let value = if bf16 {
                half::bf16::from_bits(bits).to_f32()
            } else {
                half::f16::from_bits(bits).to_f32()
            };
            if !value.is_finite() {
                return Err(error(format!("non-finite {hf}")));
            }
            if half {
                let bits = if patch && bf16 {
                    let value = half::f16::from_f32(value);
                    if !value.is_finite() {
                        return Err(error(format!("F16 patch overflow {hf}")));
                    }
                    value.to_bits()
                } else {
                    bits
                };
                out[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            } else {
                out[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        Ok(())
    })?;
    Ok(Weight {
        buffer,
        ty: if half && bf16 && !patch {
            30
        } else {
            u32::from(half)
        },
        k: dims[0],
        n: *dims.get(1).unwrap_or(&1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn processor_contract_rejects_conflicting_normalization_and_pixel_budgets() {
        let p = serde_json::json!({"do_convert_rgb":true,"do_normalize":true,"do_rescale":true,
            "rescale_factor":1.0/255.0,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
            "patch_size":16,"temporal_patch_size":2,"merge_size":2,"min_pixels":65536,"max_pixels":16777216});
        assert!(validate_image_processor(&p, (65536, 16777216)).is_ok());
        assert!(validate_image_processor(&p, (65536, 4194304)).is_err());
        for key in ["do_convert_rgb", "do_normalize", "do_rescale"] {
            let mut wrong = p.clone();
            wrong[key] = false.into();
            assert!(validate_image_processor(&wrong, (65536, 16777216)).is_err());
        }
    }
}
