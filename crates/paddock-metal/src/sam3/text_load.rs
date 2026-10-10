//! Text weights use the same checked reader as vision. No external runtime,
//! no pooled CLIP projection, no model arithmetic on the CPU.
use super::text::TextWorkspace;
use super::*;
use paddock_models::safetensors::SafetensorsFile;
use paddock_models::sam3::Sam3TextConfig;

const TEXT: &str = "detector_model.text_encoder.text_model";

pub(super) fn validate_geometry(c: &Sam3TextConfig) -> Result<()> {
    if (
        c.vocab,
        c.context,
        c.hidden,
        c.n_layer,
        c.n_heads,
        c.intermediate,
        c.d_model,
    ) != (49408, 32, 1024, 24, 16, 4096, 256)
    {
        return Err(error(
            "SAM 3 Metal text requires the official 24-layer, 1024-wide CLIP geometry",
        ));
    }
    Ok(())
}

impl Sam3Text {
    pub(super) fn planned_weight_bytes() -> u64 {
        let (d, ff, out) = (1024u64, 4096u64, 256u64);
        let block = 2 * (4 * d * d + 2 * d * ff) + 4 * (9 * d + ff);
        49408 * d * 4 + 32 * d * 4 + 24 * block + 2 * d * 4 + out * d * 2 + out * 4
    }
    pub fn load(dir: &Path, max_prompts: usize, budget: Option<u64>) -> Result<Self> {
        let required = Self::resident_bytes_required(max_prompts)?;
        let cfg = Sam3TextConfig::read(dir).map_err(|e| error(e.to_string()))?;
        validate_geometry(&cfg)?;
        let st = SafetensorsFile::open(&dir.join("model.safetensors"))
            .map_err(|e| error(e.to_string()))?;
        let device = MetalDevice::new_planned(budget, required)?;
        let r = load::Reader {
            d: &device,
            st: &st,
        };
        let token = r.f32_buffer(
            &format!("{TEXT}.embeddings.token_embedding.weight"),
            &[49408, 1024],
        )?;
        let position = r.f32_buffer(
            &format!("{TEXT}.embeddings.position_embedding.weight"),
            &[32, 1024],
        )?;
        let mut blocks = Vec::with_capacity(24);
        for i in 0..24 {
            let name = |s: &str| format!("{TEXT}.encoder.layers.{i}.{s}");
            let mut weights = Vec::with_capacity(3072 * 1024);
            let mut biases = Vec::with_capacity(3072);
            for axis in ["q", "k", "v"] {
                // CLIP has no RoPE: keep checkpoint row order in all heads.
                weights.extend(r.values(
                    &name(&format!("self_attn.{axis}_proj.weight")),
                    &[1024, 1024],
                )?);
                biases.extend(r.values(&name(&format!("self_attn.{axis}_proj.bias")), &[1024])?);
            }
            blocks.push(Block {
                n1: r.norm(&name("layer_norm1"), 1024)?,
                n2: r.norm(&name("layer_norm2"), 1024)?,
                qkv: Conv {
                    w: r.matrix_values(&weights, 1024, 3072)?,
                    b: upload(&device, &biases)?,
                },
                out: r.conv(&name("self_attn.out_proj"), &[1024, 1024])?,
                up: r.conv(&name("mlp.fc1"), &[4096, 1024])?,
                down: r.conv(&name("mlp.fc2"), &[1024, 4096])?,
            });
        }
        let final_norm = r.norm(&format!("{TEXT}.final_layer_norm"), 1024)?;
        let resizer = r.conv("detector_model.text_projection", &[256, 1024])?;
        let weight_bytes = device.allocated_bytes();
        if weight_bytes != Self::planned_weight_bytes() {
            return Err(error(format!(
                "SAM 3 text weight reservation drift: {weight_bytes} vs {}",
                Self::planned_weight_bytes()
            )));
        }
        let ws = TextWorkspace::new(&device, max_prompts)?;
        if device.allocated_bytes() != required {
            return Err(error("SAM 3 text workspace reservation drift"));
        }
        Ok(Self {
            device,
            cfg,
            token,
            position,
            blocks,
            final_norm,
            resizer,
            ws,
            weight_bytes,
            encoded_prompts: 0,
        })
    }
}
