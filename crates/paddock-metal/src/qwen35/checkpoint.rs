//! Storage adapters share allocation, scheduler and cache ownership. The MLX
//! adapter validates sanitized layout and never requantizes checkpoint values.
use super::*;
use paddock_models::{
    mapped::MappedGguf,
    safetensors::{ShardedSafetensors, StDtype, qwen35_hf_name},
};

pub(super) enum Source {
    Gguf(MappedGguf),
    Ternary(MappedGguf, paddock_models::hadamard::HadamardSpec),
    Mlx(ShardedSafetensors, bool),
    Bonsai(ShardedSafetensors, paddock_models::bonsai::BonsaiConfig),
    Splash(paddock_models::splash::Target),
}

impl Source {
    pub(super) fn load_head(&self, device: &MetalDevice, dims: &[usize]) -> Result<Option<Weight>> {
        // Tied GGUF/MLX reuse the embedding allocation itself. Untied storage
        // adapters still require their independently validated output head.
        if matches!(self, Self::Gguf(m) if m.tensor_info("output.weight").is_none()) {
            return Ok(None);
        }
        if let Self::Mlx(source, true) = self {
            if source
                .names()
                .any(|n| n.starts_with("language_model.lm_head."))
            {
                return Err(MetalError::Model(
                    "tied MLX checkpoint carries an independent output head".into(),
                ));
            }
            return Ok(None);
        }
        self.load(device, "output.weight", dims).map(Some)
    }

    pub(super) fn total_len(&self) -> u64 {
        match self {
            Self::Gguf(m) => m.total_len(),
            Self::Ternary(m, _) => m.total_len(),
            Self::Mlx(m, _) => m.total_len(),
            Self::Bonsai(m, _) => m.total_len(),
            Self::Splash(m) => m.total_len(),
        }
    }

    pub(super) fn load(&self, device: &MetalDevice, name: &str, dims: &[usize]) -> Result<Weight> {
        if let Self::Ternary(map, _) = self {
            return ternary::load(device, map, name, dims);
        }
        if let Self::Bonsai(source, _) = self {
            return bonsai::load_weight(device, source, name, dims);
        }
        if let Self::Splash(source) = self {
            let tensor = source
                .tensor(name)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            return splash_weight(device, tensor, name, dims);
        }
        let Self::Mlx(source, _) = self else {
            let Self::Gguf(map) = self else {
                unreachable!()
            };
            return Weight::load(device, map, name, dims);
        };
        let hf = match name {
            "token_embd.weight" => "language_model.model.embed_tokens.weight".into(),
            "output_norm.weight" => "language_model.model.norm.weight".into(),
            "output.weight" => "language_model.lm_head.weight".into(),
            _ => qwen35_hf_name(name)
                .and_then(|n| {
                    n.strip_prefix("model.language_model.")
                        .map(|n| format!("language_model.model.{n}"))
                })
                .ok_or_else(|| MetalError::Model(format!("no MLX mapping for {name}")))?,
        };
        let (info, data) = source
            .bytes(&hf)
            .ok_or_else(|| MetalError::Model(format!("missing {hf}")))?;
        if dims.len() == 2 && !name.ends_with("ssm_conv1d.weight") {
            return crate::affine::load(device, source, &hf, dims[0], dims[1]);
        }
        let expected = if name.ends_with("ssm_conv1d.weight") {
            vec![dims[1], dims[0], 1]
        } else {
            dims.iter().copied().rev().collect()
        };
        if info.dtype != StDtype::Bf16 || info.shape != expected {
            return Err(MetalError::Model(format!(
                "{hf}: expected sanitized BF16 {expected:?}, got {:?} {:?}",
                info.dtype, info.shape
            )));
        }
        let input = device.upload_parts(&[data])?;
        let count = data.len() / 2;
        let buffer = device.alloc(count * 4)?;
        let cmd = device.begin()?;
        cmd.dispatch(
            "mlx_small",
            &[&input, &buffer],
            &[count as u32, u32::from(name.ends_with("ssm_a"))],
            [count.div_ceil(256), 1, 1],
            256,
        );
        cmd.finish()?;
        Ok(Weight {
            buffer,
            ty: 0,
            k: dims[0],
            n: *dims.get(1).unwrap_or(&1),
        })
    }

    pub(super) fn post_norm(&self, layer: usize) -> &'static str {
        match self {
            Self::Gguf(m) | Self::Ternary(m, _)
                if m.tensor_info(&format!("blk.{layer}.attn_post_norm.weight"))
                    .is_some() =>
            {
                "attn_post_norm.weight"
            }
            _ => "post_attention_norm.weight",
        }
    }
}

pub(super) fn splash_weight(
    device: &MetalDevice,
    tensor: &paddock_models::splash::Tensor,
    name: &str,
    dims: &[usize],
) -> Result<Weight> {
    if tensor.dims() != dims {
        return Err(MetalError::Model(format!(
            "Splash {name}: expected {dims:?}, got {:?}",
            tensor.dims()
        )));
    }
    if matches!(
        name,
        "selector_predecessor.weight" | "selector_successor.weight"
    ) {
        let bytes = tensor
            .bf16_bytes()
            .ok_or_else(|| MetalError::Model("Splash selector must remain BF16".into()))?;
        return Ok(Weight {
            buffer: device.upload(bytes)?,
            ty: 30,
            k: dims[0],
            n: dims[1],
        });
    }
    let packed = tensor.affine() && name != "token_embd.weight";
    let buffer = device.upload_with(
        if packed {
            tensor.tiled_bytes()
        } else {
            tensor.output_bytes()
        },
        |out| {
            (if packed {
                tensor.copy_tiled(out)
            } else {
                tensor.copy_native(out)
            })
            .map_err(|e| MetalError::Model(e.to_string()))
        },
    )?;
    Ok(Weight {
        buffer,
        ty: if packed {
            crate::splash::PACKED4
        } else if tensor.affine() {
            crate::affine::AFFINE4
        } else {
            0
        },
        k: dims[0],
        n: *dims.get(1).unwrap_or(&1),
    })
}
