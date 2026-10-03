//! Nemotron 3 Diarization's two published checkpoints - the MLX BF16
//! directory and NVIDIA's GGUF Q8_0 file - behind one reader that validates
//! the whole architecture contract before anything is allocated. Tensors are
//! addressed by their MLX names; the GGUF's own names, its quantized
//! projections and its [out][c][t] convolution are translated here, so a
//! backend sees one inventory whatever the container.
//!
//! The older four-speaker Conformer Sortformer is rejected, not approximated.
use crate::mapped::MappedGguf;
use crate::safetensors::{ShardedSafetensors, StDtype};
use half::{bf16, f16};
use serde_json::{Value, json};
use std::path::Path;

/// GGML type ids as the GGUF stores them; BF16 is the MLX directory's.
pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q8_0: u32 = 8;
pub const BF16: u32 = 30;

/// Encoder layers of the v3 architecture.
pub const LAYERS: usize = 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// mlx-community's BF16 safetensors directory (the trained values exactly:
    /// NVIDIA's F32 export is these BF16 values widened)
    MlxBf16,
    /// NVIDIA's GGUF: Q8_0 projections, F32 norms/biases/head, F16 convolution
    GgufQ8,
}

enum Source {
    Mlx(ShardedSafetensors),
    Gguf(MappedGguf),
}

pub struct Checkpoint {
    source: Source,
}

/// One tensor as stored: its GGML type id and raw little-endian bytes.
pub struct Raw<'a> {
    pub ty: u32,
    pub bytes: &'a [u8],
}

/// The checked inventory, MLX names: (name, shape, is a projection weight a
/// backend may keep in its stored type rather than F32).
pub fn inventory() -> Vec<(String, Vec<usize>, bool)> {
    let mut out = Vec::new();
    let linear =
        |out: &mut Vec<(String, Vec<usize>, bool)>, n: &str, k: usize, o: usize, bias: bool| {
            out.push((format!("{n}.weight"), vec![o, k], true));
            if bias {
                out.push((format!("{n}.bias"), vec![o], false));
            }
        };
    linear(&mut out, "encoder.pre_encode.proj", 1024, 512, false);
    for i in 0..LAYERS {
        let p = format!("encoder.layers.{i}");
        for n in ["norm1", "norm2"] {
            for t in ["weight", "bias"] {
                out.push((format!("{p}.{n}.{t}"), vec![512], false));
            }
        }
        linear(&mut out, &format!("{p}.attn.w_qkv"), 512, 1536, false);
        linear(&mut out, &format!("{p}.attn.out_proj"), 512, 512, true);
        linear(&mut out, &format!("{p}.ffn.linear1"), 512, 2048, true);
        linear(&mut out, &format!("{p}.ffn.linear2"), 2048, 512, true);
    }
    for n in ["encoder.embed_norm", "encoder.final_norm"] {
        for t in ["weight", "bias"] {
            out.push((format!("{n}.{t}"), vec![512], false));
        }
    }
    let s = "sortformer_modules";
    linear(&mut out, &format!("{s}.encoder_proj"), 512, 192, true);
    out.push((
        format!("{s}.subpixel_upsample.weight"),
        vec![1536, 3, 192],
        true,
    ));
    out.push((format!("{s}.subpixel_upsample.bias"), vec![1536], false));
    linear(
        &mut out,
        &format!("{s}.first_hidden_to_hidden"),
        192,
        192,
        true,
    );
    linear(
        &mut out,
        &format!("{s}.single_hidden_to_spks"),
        192,
        8,
        true,
    );
    out.push((format!("{s}.learnable_sil_emb"), vec![512], false));
    out
}

fn mlx_config(dir: &Path) -> Result<(), String> {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    for (pointer, expected) in [
        ("/model_type", json!("nemotron_diarization")),
        ("/num_speakers", json!(8)),
        ("/output_subsampling_factor", json!(1)),
        ("/encoder_config/d_model", json!(512)),
        ("/encoder_config/n_layers", json!(31)),
        ("/encoder_config/n_heads", json!(8)),
        ("/encoder_config/feat_in", json!(128)),
        ("/encoder_config/subsampling_factor", json!(8)),
        ("/encoder_config/ff_expansion", json!(4.0)),
        ("/encoder_config/qk_norm", json!(false)),
        ("/encoder_config/qkv_bias", json!(false)),
        ("/encoder_config/xscaling", json!(false)),
        ("/encoder_config/pre_block_norm", json!(true)),
        ("/encoder_config/rope_base", json!(10000.0)),
        ("/encoder_config/rotary_fraction", json!(1.0)),
        ("/modules_config/tf_d_model", json!(192)),
        ("/modules_config/use_aosc", json!(true)),
        ("/modules_config/fc_d_model", json!(512)),
        ("/modules_config/num_speakers", json!(8)),
        ("/modules_config/subsampling_factor", json!(8)),
        ("/modules_config/use_learnable_sil_emb", json!(true)),
        ("/modules_config/spkcache_sil_frames_per_spk", json!(1)),
        ("/modules_config/pred_score_threshold", json!(0.25)),
        ("/modules_config/scores_boost_latest", json!(0.05)),
        ("/modules_config/strong_boost_rate", json!(0.75)),
        ("/modules_config/weak_boost_rate", json!(1.5)),
        ("/modules_config/min_pos_scores_rate", json!(0.5)),
        ("/processor_config/feature_size", json!(128)),
        ("/processor_config/sampling_rate", json!(16000)),
        ("/processor_config/hop_length", json!(160)),
        ("/processor_config/n_fft", json!(512)),
        ("/processor_config/win_length", json!(400)),
        ("/processor_config/preemphasis", json!(0.97)),
        ("/processor_config/pad_to", json!(16)),
    ] {
        if v.pointer(pointer) != Some(&expected) {
            return Err(format!("unsupported {pointer}"));
        }
    }
    Ok(())
}

fn gguf_metadata(g: &MappedGguf) -> Result<(), String> {
    use crate::gguf::Value as G;
    let m = &g.gguf().metadata;
    for (key, expected) in [
        ("general.architecture", "sortformer"),
        ("sortformer.version", "v3"),
        ("sortformer.encoder.type", "transformer_rope"),
        ("sortformer.encoder.subsampling_type", "feature_stacking"),
        ("sortformer.preprocessor.normalize", "NA"),
    ] {
        if m.get(key).and_then(|v| v.as_str()) != Some(expected) {
            return Err(format!("unsupported {key}"));
        }
    }
    for (key, expected) in [
        ("encoder.d_model", 512),
        ("encoder.n_layers", 31),
        ("encoder.n_heads", 8),
        ("encoder.d_ff", 2048),
        ("encoder.feat_in", 128),
        ("encoder.subsampling_factor", 8),
        ("num_speakers", 8),
        ("output_subsampling_factor", 1),
        ("preprocessor.sample_rate", 16000),
        ("preprocessor.n_fft", 512),
        ("preprocessor.features", 128),
        ("transformer.hidden_size", 192),
        ("upsample_factor", 8),
        ("scoring.spkcache_sil_frames_per_spk", 1),
    ] {
        if m.get(&format!("sortformer.{key}")).and_then(|v| v.as_u64()) != Some(expected) {
            return Err(format!("unsupported {key}"));
        }
    }
    for (key, expected) in [
        ("encoder.qkv_bias", false),
        ("encoder.qk_norm", false),
        ("encoder.xscaling", false),
        ("encoder.pre_block_norm", true),
        ("learnable_silence", true),
        ("high_resolution", true),
    ] {
        if m.get(&format!("sortformer.{key}")) != Some(&G::Bool(expected)) {
            return Err(format!("unsupported {key}"));
        }
    }
    for (key, expected) in [
        ("encoder.rope_base", 10000f32),
        ("encoder.rotary_fraction", 1.),
        ("preprocessor.window_size", 0.025),
        ("preprocessor.window_stride", 0.01),
        ("preprocessor.preemph", 0.97),
        ("preprocessor.log_zero_guard", 2f32.powi(-24)),
        ("scoring.pred_score_threshold", 0.25),
        ("scoring.scores_boost_latest", 0.05),
        ("scoring.strong_boost_rate", 0.75),
        ("scoring.weak_boost_rate", 1.5),
        ("scoring.min_pos_scores_rate", 0.5),
    ] {
        if m.get(&format!("sortformer.{key}")).and_then(|v| v.as_f32()) != Some(expected) {
            return Err(format!("unsupported {key}"));
        }
    }
    Ok(())
}

/// The GGUF's name for an MLX-named tensor.
fn gguf_name(name: &str) -> String {
    let renamed = name
        .replace(".ffn.linear1.", ".ffn.net.0.")
        .replace(".ffn.linear2.", ".ffn.net.3.");
    match renamed.strip_prefix("sortformer_modules.") {
        Some(s)
            if s.starts_with("encoder_proj.")
                || s.starts_with("subpixel_upsample.")
                || s == "learnable_sil_emb" =>
        {
            s.to_owned()
        }
        Some(s) => format!("head.{s}"),
        None => renamed,
    }
}

impl Checkpoint {
    /// A directory is the MLX export, a file the GGUF. Configuration and
    /// metadata are checked here; [`Self::validate`] checks every tensor.
    pub fn open(path: &Path) -> Result<Self, String> {
        let source = if path.is_dir() {
            mlx_config(path)?;
            Source::Mlx(ShardedSafetensors::open_dir(path).map_err(|e| e.to_string())?)
        } else {
            let g = MappedGguf::open(path).map_err(|e| e.to_string())?;
            gguf_metadata(&g)?;
            Source::Gguf(g)
        };
        Ok(Self { source })
    }

    pub fn format(&self) -> Format {
        match self.source {
            Source::Mlx(_) => Format::MlxBf16,
            Source::Gguf(_) => Format::GgufQ8,
        }
    }

    /// The stored tensor behind an MLX name and shape. MLX weights are BF16
    /// (the preprocessor buffers F32); GGUF ones F32, F16 or Q8_0. A GGUF's
    /// 3-D convolution is checked against its own [out][c][t] layout.
    pub fn raw(&self, name: &str, shape: &[usize]) -> Result<Raw<'_>, String> {
        match &self.source {
            Source::Mlx(s) => {
                let (info, bytes) = s.bytes(name).ok_or_else(|| format!("missing {name}"))?;
                let ty = match info.dtype {
                    StDtype::Bf16 => BF16,
                    StDtype::F32 => F32,
                    _ => return Err(format!("{name}: expected BF16 or F32")),
                };
                if !name.starts_with("preprocessor.") && ty != BF16 {
                    return Err(format!("{name}: the MLX export is BF16"));
                }
                if info.shape != shape
                    || bytes.len() != shape.iter().product::<usize>() * info.dtype.bytes()
                {
                    return Err(format!("{name}: shape mismatch"));
                }
                Ok(Raw { ty, bytes })
            }
            Source::Gguf(g) => {
                let renamed = gguf_name(name);
                let (info, bytes) = g.tensor_bytes(&renamed).map_err(|e| e.to_string())?;
                let expected: Vec<u64> = if shape.len() == 3 {
                    vec![shape[1] as u64, shape[2] as u64, shape[0] as u64]
                } else {
                    shape.iter().rev().map(|&n| n as u64).collect()
                };
                let n: usize = shape.iter().product();
                let len = match info.raw_type {
                    F32 => n * 4,
                    F16 | BF16 => n * 2,
                    Q8_0 if shape.last().is_some_and(|k| k % 32 == 0) => n / 32 * 34,
                    _ => usize::MAX,
                };
                if info.dims != expected || bytes.len() != len {
                    return Err(format!(
                        "{renamed}: unsupported shape/type {:?}/{}",
                        info.dims, info.raw_type
                    ));
                }
                Ok(Raw {
                    ty: info.raw_type,
                    bytes,
                })
            }
        }
    }

    /// F32 values of a tensor, MLX layout: BF16 and F16 widen exactly, Q8_0
    /// dequantizes exactly (an f16 scale times an int8 fits F32's mantissa),
    /// and a GGUF convolution is reordered to [out][t][c].
    pub fn values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let raw = self.raw(name, shape)?;
        let mut v = decode(&raw)?;
        if shape.len() == 3 && matches!(self.source, Source::Gguf(_)) {
            let (o, t, c) = (shape[0], shape[1], shape[2]);
            let src = v;
            v = vec![0.; src.len()];
            for oi in 0..o {
                for ti in 0..t {
                    for ci in 0..c {
                        v[(oi * t + ti) * c + ci] = src[(oi * c + ci) * t + ti];
                    }
                }
            }
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(format!("{name}: nonfinite weight"));
        }
        Ok(v)
    }

    /// Every inventory tensor present with its shape, type and size; the
    /// bytes a backend holding small tensors in F32 and projections as
    /// stored would need.
    pub fn validate(&self) -> Result<u64, String> {
        let mut bytes = 0u64;
        for (name, shape, projection) in inventory() {
            let raw = self.raw(&name, &shape)?;
            let allowed = match self.format() {
                Format::MlxBf16 => raw.ty == BF16,
                Format::GgufQ8 => {
                    matches!(raw.ty, F32 | F16 | BF16) || (projection && raw.ty == Q8_0)
                }
            };
            if !allowed {
                return Err(format!("{name}: unexpected stored type {}", raw.ty));
            }
            bytes += if projection {
                raw.bytes.len() as u64
            } else {
                4 * shape.iter().product::<usize>() as u64
            };
        }
        self.window()?;
        self.filterbank()?;
        Ok(bytes)
    }

    /// The analysis window, 400 taps: the MLX export stores its own (the
    /// trained checkpoint's buffer at BF16 resolution); the GGUF carries none
    /// and means the symmetric Hann window, F64 evaluated and rounded once.
    pub fn window(&self) -> Result<Vec<f32>, String> {
        match self.format() {
            Format::MlxBf16 => self.values("preprocessor.window", &[400]),
            Format::GgufQ8 => Ok((0..400)
                .map(|i| {
                    (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / 399.0).cos())) as f32
                })
                .collect()),
        }
    }

    /// The mel filterbank, [128][257].
    pub fn filterbank(&self) -> Result<Vec<f32>, String> {
        match self.format() {
            Format::MlxBf16 => self.values("preprocessor.fb", &[1, 128, 257]),
            Format::GgufQ8 => self.values("preprocessor.fb", &[128, 257]),
        }
    }

    /// The learned silence embedding a compressed speaker cache pads with.
    pub fn silence(&self) -> Result<Vec<f32>, String> {
        self.values("sortformer_modules.learnable_sil_emb", &[512])
    }
}

/// F32 values of a stored tensor (row-major as stored).
pub fn decode(raw: &Raw<'_>) -> Result<Vec<f32>, String> {
    let b = raw.bytes;
    Ok(match raw.ty {
        F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        BF16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
        Q8_0 => b
            .as_chunks::<34>()
            .0
            .iter()
            .flat_map(|blk| {
                let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                blk[2..].iter().map(move |&q| d * f32::from(q as i8))
            })
            .collect(),
        t => return Err(format!("unsupported stored type {t}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_dequant_is_exact_and_names_translate() {
        // d = 0.5 (f16 0x3800), q = -128, 1, 127, ...
        let mut blk = vec![0x00u8, 0x38];
        blk.extend((0..32).map(|i| [128u8, 1, 127][i % 3]));
        let v = decode(&Raw {
            ty: Q8_0,
            bytes: &blk,
        })
        .unwrap();
        assert_eq!(&v[..3], &[-64.0, 0.5, 63.5]);
        assert_eq!(
            gguf_name("encoder.layers.3.ffn.linear2.bias"),
            "encoder.layers.3.ffn.net.3.bias"
        );
        assert_eq!(
            gguf_name("sortformer_modules.first_hidden_to_hidden.weight"),
            "head.first_hidden_to_hidden.weight"
        );
        assert_eq!(
            gguf_name("sortformer_modules.learnable_sil_emb"),
            "learnable_sil_emb"
        );
        let inv = inventory();
        assert_eq!(
            inv.iter().filter(|(_, _, p)| *p).count(),
            1 + 4 * LAYERS + 4
        );
    }
}
