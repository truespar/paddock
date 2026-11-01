//! Phonon 31M English checkpoint contract. Tensor data only, no runtime adapter.
use crate::safetensors::{SafetensorsFile, StDtype};
use std::{collections::BTreeMap, path::Path};

pub const SAMPLE_RATE: usize = 24_000;
pub const FRAME_SAMPLES: usize = 1920;
pub const MAX_TEXT_TOKENS: usize = 512;
pub const MAX_FRAMES: usize = 750;
pub const MAX_INPUT_BYTES: usize = 4096;
pub const FLOW_CACHE: usize = 1536;
pub const CODEC_CACHE: usize = 512;

pub fn is_model(path: &Path) -> bool {
    std::fs::read(path.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["model_type"] == "gradium_phonon_tts")
}

/// Every inference tensor is shape-checked and finite before GPU allocation.
/// F32 expansion of BF16 is lossless; no weight is downcast to F16.
pub struct Checkpoint {
    pub tensors: BTreeMap<String, Vec<f32>>,
    pub voice: Vec<f32>,
    pub voice_frames: usize,
}
impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(path.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        for (key, expected) in [
            ("hidden_size", 512),
            ("intermediate_size", 2048),
            ("num_hidden_layers", 6),
            ("num_attention_heads", 8),
            ("num_key_value_heads", 8),
            ("head_dim", 64),
            ("latent_dim", 32),
            ("audio_prompt_dim", 512),
            ("vocab_size", 4001),
            ("num_speakers", 1),
        ] {
            if v[key].as_u64() != Some(expected) {
                return Err(format!("Phonon: unsupported {key}: {}", v[key]));
            }
        }
        if v["model_type"] != "gradium_phonon_tts"
            || v["flow_head_config"]["flow_type"] != "lsd"
            || v["flow_head_config"]["num_hidden_layers"] != 6
            || v["hidden_act"] != "gelu"
            || v["mimi"]["ratios"] != serde_json::json!([6, 5, 4])
            || v["mimi"]["sample_rate"] != SAMPLE_RATE
            || v["duration_delta"] != 0.0
        {
            return Err(
                "Phonon: only the released English 31M LSD/Mimi architecture is supported".into(),
            );
        }
        for (pointer, expected) in [
            ("/layer_norm_eps", serde_json::json!(1e-5)),
            (
                "/rope_parameters",
                serde_json::json!({"rope_theta":10000.0,"rope_type":"default"}),
            ),
            ("/duration_delta_embedding_dim", serde_json::json!(128)),
            ("/num_speakers_embedding_dim", serde_json::json!(16)),
            ("/flow_head_config/hidden_size", serde_json::json!(512)),
            ("/flow_head_config/latent_dim", serde_json::json!(32)),
            (
                "/flow_head_config/frequency_embedding_size",
                serde_json::json!(256),
            ),
            ("/flow_head_config/hidden_act", serde_json::json!("silu")),
            ("/flow_head_config/layer_norm_eps", serde_json::json!(1e-6)),
            ("/flow_head_config/rms_norm_eps", serde_json::json!(1e-5)),
            ("/mimi/frame_rate", serde_json::json!(12.5)),
            ("/mimi/transformer_context", serde_json::json!(250)),
            ("/mimi/transformer_num_heads", serde_json::json!(8)),
            ("/mimi/transformer_num_layers", serde_json::json!(2)),
            ("/mimi/transformer_max_period", serde_json::json!(10000.0)),
            ("/mimi/n_residual_layers", serde_json::json!(1)),
            ("/mimi/gelu", serde_json::json!("erf")),
        ] {
            if v.pointer(pointer) != Some(&expected) {
                return Err(format!(
                    "Phonon: unsupported {pointer}; expected {expected}"
                ));
            }
        }
        // Defaults and stopping policy are part of the checkpoint contract.
        // Reject different exported policies rather than silently serving them
        // with this checkpoint's cached one-step time embeddings and EOS tail.
        for (file, fields) in [
            (
                "generation_config.json",
                vec![
                    ("num_flow_steps", serde_json::json!(1)),
                    ("eos_threshold", serde_json::json!(4.0)),
                    ("min_frames_before_eos", serde_json::json!(0)),
                    ("temperature", serde_json::json!(0.3)),
                ],
            ),
            (
                "processor_config.json",
                vec![
                    ("frames_after_eos", serde_json::json!(1)),
                    ("frames_after_eos_short_text", serde_json::json!(3)),
                    ("short_text_max_words", serde_json::json!(4)),
                    ("fade_in_duration", serde_json::json!(0.0)),
                ],
            ),
        ] {
            let policy: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path.join(file)).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            for (key, expected) in fields {
                let same = if expected.is_number() {
                    policy[key].as_f64() == expected.as_f64()
                } else {
                    policy[key] == expected
                };
                if !same {
                    return Err(format!(
                        "Phonon: unsupported {file} {key}; expected {expected}"
                    ));
                }
            }
        }
        let source =
            SafetensorsFile::open(&path.join("model.safetensors")).map_err(|e| e.to_string())?;
        let mut tensors = BTreeMap::new();
        for (name, shape) in schema() {
            tensors.insert(name.clone(), values(&source, &name, &shape)?);
        }
        let voice_file = SafetensorsFile::open(&path.join("default-voice.safetensors"))
            .map_err(|e| e.to_string())?;
        let voice_shape = &voice_file
            .tensors()
            .get("speaker_wavs")
            .ok_or("Phonon: missing speaker_wavs")?
            .shape;
        if voice_shape.len() != 3
            || voice_shape[0] != 1
            || voice_shape[1] != 512
            || !(1..=250).contains(&voice_shape[2])
        {
            return Err(format!("Phonon: invalid voice shape {voice_shape:?}"));
        }
        let voice_frames = voice_shape[2];
        let channel_major = values(&voice_file, "speaker_wavs", voice_shape)?;
        let mut voice = vec![0.; channel_major.len()];
        for t in 0..voice_frames {
            for c in 0..512 {
                voice[t * 512 + c] = channel_major[c * voice_frames + t];
            }
        }
        Ok(Self {
            tensors,
            voice,
            voice_frames,
        })
    }
}

fn values(f: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let (t, b) = f
        .bytes(name)
        .ok_or_else(|| format!("Phonon: missing {name}"))?;
    if t.shape != shape {
        return Err(format!(
            "Phonon: {name} shape {:?}, expected {shape:?}",
            t.shape
        ));
    }
    let values: Vec<f32> = match t.dtype {
        StDtype::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        StDtype::Bf16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect(),
        _ => {
            return Err(format!(
                "Phonon: {name} unsupported stored type {:?}",
                t.dtype
            ));
        }
    };
    if values.iter().any(|x| !x.is_finite()) {
        return Err(format!("Phonon: non-finite tensor {name}"));
    }
    Ok(values)
}

fn schema() -> Vec<(String, Vec<usize>)> {
    let mut s = Vec::new();
    let mut add = |n: String, d: &[usize]| s.push((n, d.to_vec()));
    for (name, shape) in [
        ("bos_emb", vec![32]),
        ("emb_mean", vec![32]),
        ("emb_std", vec![32]),
        ("input_linear.weight", vec![512, 32]),
        ("out_norm.weight", vec![512]),
        ("out_norm.bias", vec![512]),
        ("out_eos.weight", vec![1, 512]),
        ("out_eos.bias", vec![1]),
        (
            "condition_provider.conditioners.num_speakers.embed.weight",
            vec![32, 16],
        ),
        (
            "condition_provider.conditioners.num_speakers.output_proj.weight",
            vec![512, 16],
        ),
        (
            "condition_provider.conditioners.duration_delta.output_proj.weight",
            vec![512, 128],
        ),
        (
            "condition_provider.conditioners.speaker_wavs.output_proj.weight",
            vec![512, 512],
        ),
        (
            "condition_provider.conditioners.transcript_in_segment.embed.weight",
            vec![4001, 512],
        ),
    ] {
        add(format!("flow_lm.{name}"), &shape);
    }
    for (prefix, count, scale) in [
        ("flow_lm.transformer", 6, false),
        ("mimi.decoder_transformer.transformer", 2, true),
    ] {
        for i in 0..count {
            let p = format!("{prefix}.layers.{i}");
            for n in ["norm1", "norm2"] {
                for suffix in ["weight", "bias"] {
                    add(format!("{p}.{n}.{suffix}"), &[512]);
                }
            }
            for (n, shape) in [
                ("self_attn.in_proj", [1536, 512]),
                ("self_attn.out_proj", [512, 512]),
                ("linear1", [2048, 512]),
                ("linear2", [512, 2048]),
            ] {
                add(format!("{p}.{n}.weight"), &shape);
            }
            if scale {
                for n in [1, 2] {
                    add(format!("{p}.layer_scale_{n}.scale"), &[512]);
                }
            }
        }
    }
    let p = "flow_lm.flow_net";
    for (n, out, k) in [
        ("input_proj", 512, 32),
        ("cond_embed", 512, 512),
        ("final_layer.adaLN_modulation.1", 1024, 512),
        ("final_layer.linear", 32, 512),
    ] {
        add(format!("{p}.{n}.weight"), &[out, k]);
        add(format!("{p}.{n}.bias"), &[out]);
    }
    for i in 0..2 {
        let p = format!("{p}.time_embed.{i}");
        add(format!("{p}.freqs"), &[128]);
        add(format!("{p}.mlp.3.alpha"), &[512]);
        for (n, k) in [(0, 256), (2, 512)] {
            add(format!("{p}.mlp.{n}.weight"), &[512, k]);
            add(format!("{p}.mlp.{n}.bias"), &[512]);
        }
    }
    for i in 0..6 {
        let p = format!("{p}.res_blocks.{i}");
        for n in ["weight", "bias"] {
            add(format!("{p}.in_ln.{n}"), &[512]);
        }
        for (n, out) in [("adaLN_modulation.1", 1536), ("mlp.0", 512), ("mlp.2", 512)] {
            add(format!("{p}.{n}.weight"), &[out, 512]);
            add(format!("{p}.{n}.bias"), &[out]);
        }
    }
    add("mimi.quantizer.output_proj.weight".into(), &[512, 32, 1]);
    add("mimi.upsample.convtr.convtr.weight".into(), &[512, 1, 32]);
    for (i, out, k, width, transpose) in [
        (0, 512, 512, 7, false),
        (2, 256, 512, 12, true),
        (5, 128, 256, 10, true),
        (8, 64, 128, 8, true),
        (11, 1, 64, 3, false),
    ] {
        let p = format!(
            "mimi.decoder.model.{i}.{}",
            if transpose { "convtr" } else { "conv" }
        );
        add(
            format!("{p}.weight"),
            &if transpose {
                [k, out, width]
            } else {
                [out, k, width]
            },
        );
        add(format!("{p}.bias"), &[out]);
    }
    for (i, dim) in [(3, 256), (6, 128), (9, 64)] {
        for (j, out, k, width) in [(1, dim / 2, dim, 3), (3, dim, dim / 2, 1)] {
            let p = format!("mimi.decoder.model.{i}.block.{j}.conv");
            add(format!("{p}.weight"), &[out, k, width]);
            add(format!("{p}.bias"), &[out]);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_context_and_unique_contract() {
        const { assert!(250 + MAX_TEXT_TOKENS + MAX_FRAMES < FLOW_CACHE) };
        let s = schema();
        let names: std::collections::BTreeSet<_> = s.iter().map(|v| &v.0).collect();
        assert_eq!(names.len(), s.len());
        assert!(s.len() > 150);
    }
}
