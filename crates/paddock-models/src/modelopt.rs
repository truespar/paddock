//! modelopt/llm-compressor checkpoint quantization maps + NVFP4/FP8 views.
//!
//! NVIDIA ModelOpt exports (quant_method "modelopt" in config.json) describe
//! per-module quantization in `quantization_config.quantized_layers`; modules
//! not listed (and everything in `ignore`) stay at the checkpoint dtype.
//! First consumer: NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, MIXED_PRECISION
//! with exactly two recipes (verified against the shipped checkpoint):
//!
//! - `W4A16_NVFP4` (group_size 16) - the fp4 triple, per quantized module:
//!   `<m>.weight`         U8      [N, K/2]  two e2m1 nibbles per byte along K,
//!   LOW nibble = even element (vLLM
//!   break_fp4_bytes convention)
//!   `<m>.weight_scale`   F8_E4M3 [N, K/16] per-16-block scale
//!   `<m>.weight_scale_2` F32     []        per-tensor global scale
//!   dequant: w[n,k] = e2m1 * e4m3_scale * scale_2 (multiplied, in that order)
//! - `FP8` - static W8A8, per quantized module:
//!   `<m>.weight`         F8_E4M3 [N, K]
//!   `<m>.weight_scale`   F32     []        per-tensor weight scale
//!   `<m>.input_scale`    F32     [1]       static activation scale
//!
//! llm-compressor exports (quant_method "compressed-tensors", e.g.
//! `unsloth/Qwen3.8-27B-NVFP4`) carry the same payloads
//! under different names, and [`nvfp4_view`] normalizes them into one view:
//!
//! - NVFP4 (`tensor_group`, group 16):
//!   `<m>.weight_packed`        U8      [N, K/2]  same nibble packing
//!   `<m>.weight_scale`         F8_E4M3 [N, K/16] same per-16 block scale
//!   `<m>.weight_global_scale`  F32     [1]       RECIPROCAL of modelopt's
//!   scale_2: llm-compressor scales the group scales up by `global` before
//!   fp8-encoding them (observed ~1e4 - raw scales would drown in e4m3
//!   subnormals), so dequant divides: scale2 = 1 / weight_global_scale.
//!   `<m>.input_global_scale` (W4A4 activation scale) is ignored - we serve
//!   the W4A16 class, weights bit-exact to the checkpoint.
//! - FP8 (`channel` strategy): `<m>.weight` F8_E4M3 [N, K] +
//!   `<m>.weight_scale` BF16|F32 [N, 1] per-output-row - [`fp8_channel_view`].
//!   No input_scale (activations are dynamic per-token).
//!
//! The reference dequants here are the correctness oracle for the CUDA lane -
//! host-only, exact f32, never a serving path.

use std::collections::HashMap;
use std::path::Path;

use crate::safetensors::{ShardedSafetensors, StDtype, StError};

/// One module's quantization recipe. Modules absent from the map are served
/// at the checkpoint dtype (bf16/f32) - represented by lookup returning None,
/// not by a variant, so a typo'd module name can never silently claim a recipe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantAlgo {
    /// Static-scale FP8 W8A8 (weight_scale + input_scale, both per-tensor).
    Fp8,
    /// NVFP4 weights, 16-bit activations (weight/weight_scale/weight_scale_2).
    Nvfp4 { group: usize },
}

/// Parsed `quantization_config` of a modelopt export.
#[derive(Debug)]
pub struct ModeloptQuantMap {
    /// module path (e.g. "backbone.layers.1.mixer.experts.0.up_proj") -> recipe
    algos: HashMap<String, QuantAlgo>,
    /// kv_cache_scheme present with 8-bit float type.
    pub kv_cache_fp8: bool,
}

impl ModeloptQuantMap {
    /// Read `<dir>/config.json` and parse its `quantization_config`. Every
    /// field the engine will act on is validated present so config drift
    /// fails at load, never as silently-unquantized planes.
    pub fn read(dir: &Path) -> Result<Self, StError> {
        let raw = std::fs::read(dir.join("config.json"))?;
        let cfg: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| StError::Header(e.to_string()))?;
        let qc = cfg
            .get("quantization_config")
            .ok_or_else(|| StError::Header("config.json: no quantization_config".into()))?;
        let method = qc
            .get("quant_method")
            .and_then(|m| m.as_str())
            .unwrap_or("");
        if method != "modelopt" {
            return Err(StError::Header(format!(
                "quantization_config.quant_method is {method:?}, expected \"modelopt\""
            )));
        }
        let layers = qc
            .get("quantized_layers")
            .and_then(|l| l.as_object())
            .ok_or_else(|| StError::Header("quantization_config: no quantized_layers".into()))?;
        let mut algos = HashMap::with_capacity(layers.len());
        for (module, spec) in layers {
            let algo = spec
                .get("quant_algo")
                .and_then(|a| a.as_str())
                .unwrap_or("");
            let parsed = match algo {
                "FP8" => QuantAlgo::Fp8,
                "W4A16_NVFP4" => {
                    let group =
                        spec.get("group_size")
                            .and_then(|g| g.as_u64())
                            .ok_or_else(|| {
                                StError::Header(format!("{module}: W4A16_NVFP4 without group_size"))
                            })? as usize;
                    QuantAlgo::Nvfp4 { group }
                }
                other => {
                    return Err(StError::Header(format!(
                        "{module}: unsupported quant_algo {other:?}"
                    )));
                }
            };
            algos.insert(module.clone(), parsed);
        }
        let kv_cache_fp8 = qc
            .get("kv_cache_scheme")
            .and_then(|s| s.get("num_bits"))
            .and_then(|b| b.as_u64())
            == Some(8);
        Ok(Self {
            algos,
            kv_cache_fp8,
        })
    }

    /// Recipe for a module path, None = unquantized (checkpoint dtype).
    pub fn algo_for(&self, module: &str) -> Option<QuantAlgo> {
        self.algos.get(module).copied()
    }

    pub fn len(&self) -> usize {
        self.algos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.algos.is_empty()
    }

    pub fn modules(&self) -> impl Iterator<Item = (&String, QuantAlgo)> {
        self.algos.iter().map(|(m, a)| (m, *a))
    }
}

/// Borrowed view of one NVFP4 module's tensor triple, geometry-validated.
pub struct Nvfp4View<'a> {
    /// [n, k/2] packed e2m1 pairs, low nibble = even element.
    pub packed: &'a [u8],
    /// [n, k/16] e4m3 block scales.
    pub scales: &'a [u8],
    /// per-tensor f32 global scale.
    pub scale2: f32,
    pub n: usize,
    pub k: usize,
}

/// Assemble the fp4 triple from either export dialect: modelopt
/// `<prefix>.weight{,_scale,_scale_2}` or llm-compressor
/// `<prefix>.weight_packed` + `weight_scale` + `weight_global_scale`
/// (scale2 = 1/global - see the module header). Both normalize to the same
/// view, so every downstream consumer (upload, kernels, oracle) is dialect-
/// blind.
pub fn nvfp4_view<'a>(st: &'a ShardedSafetensors, prefix: &str) -> Result<Nvfp4View<'a>, StError> {
    let bad = |m: String| StError::Header(m);
    // Packed nibbles: modelopt reuses ".weight", llm-compressor names it
    // ".weight_packed" (its ".weight"-named tensors are always fp8/bf16, so
    // presence of "_packed" is an unambiguous dialect marker).
    let (wt, wb, compressed) = match st.bytes(&format!("{prefix}.weight_packed")) {
        Some((t, b)) => (t, b, true),
        None => {
            let (t, b) = st
                .bytes(&format!("{prefix}.weight"))
                .ok_or_else(|| bad(format!("{prefix}.weight[_packed]: missing")))?;
            (t, b, false)
        }
    };
    if wt.dtype != StDtype::U8 || wt.shape.len() != 2 {
        return Err(bad(format!(
            "{prefix}.weight{}: want U8 2-D, got {:?} {:?}",
            if compressed { "_packed" } else { "" },
            wt.dtype,
            wt.shape
        )));
    }
    let (n, kh) = (wt.shape[0], wt.shape[1]);
    let (st_t, sb) = st
        .bytes(&format!("{prefix}.weight_scale"))
        .ok_or_else(|| bad(format!("{prefix}.weight_scale: missing")))?;
    if st_t.dtype != StDtype::F8E4m3 || st_t.shape != [n, kh / 8] {
        return Err(bad(format!(
            "{prefix}.weight_scale: want F8_E4M3 [{n}, {}], got {:?} {:?}",
            kh / 8,
            st_t.dtype,
            st_t.shape
        )));
    }
    let s2name = if compressed {
        "weight_global_scale"
    } else {
        "weight_scale_2"
    };
    let (s2t, s2b) = st
        .bytes(&format!("{prefix}.{s2name}"))
        .ok_or_else(|| bad(format!("{prefix}.{s2name}: missing")))?;
    if s2t.dtype != StDtype::F32 || s2b.len() != 4 {
        return Err(bad(format!(
            "{prefix}.{s2name}: want scalar F32, got {:?}",
            s2t.dtype
        )));
    }
    let raw = f32::from_le_bytes(s2b.try_into().expect("length checked above"));
    if compressed && !(raw.is_finite() && raw > 0.0) {
        return Err(bad(format!(
            "{prefix}.{s2name}: not a positive finite scale ({raw:e})"
        )));
    }
    let scale2 = if compressed { 1.0 / raw } else { raw };
    Ok(Nvfp4View {
        packed: wb,
        scales: sb,
        scale2,
        n,
        k: kh * 2,
    })
}

/// Borrowed view of one static-FP8 module (weight + two per-tensor scales).
pub struct Fp8View<'a> {
    /// [n, k] e4m3 weight bytes.
    pub weight: &'a [u8],
    pub weight_scale: f32,
    pub input_scale: f32,
    pub n: usize,
    pub k: usize,
}

/// Assemble `<prefix>.weight` + `weight_scale` + `input_scale`.
pub fn fp8_view<'a>(st: &'a ShardedSafetensors, prefix: &str) -> Result<Fp8View<'a>, StError> {
    let bad = |m: String| StError::Header(m);
    let (wt, wb) = st
        .bytes(&format!("{prefix}.weight"))
        .ok_or_else(|| bad(format!("{prefix}.weight: missing")))?;
    if wt.dtype != StDtype::F8E4m3 || wt.shape.len() != 2 {
        return Err(bad(format!(
            "{prefix}.weight: want F8_E4M3 2-D, got {:?} {:?}",
            wt.dtype, wt.shape
        )));
    }
    let scalar = |suffix: &str| -> Result<f32, StError> {
        let (t, b) = st
            .bytes(&format!("{prefix}.{suffix}"))
            .ok_or_else(|| bad(format!("{prefix}.{suffix}: missing")))?;
        if t.dtype != StDtype::F32 || b.len() != 4 {
            return Err(bad(format!(
                "{prefix}.{suffix}: want scalar F32, got {:?}",
                t.dtype
            )));
        }
        Ok(f32::from_le_bytes(
            b.try_into().expect("length checked above"),
        ))
    };
    Ok(Fp8View {
        weight: wb,
        weight_scale: scalar("weight_scale")?,
        input_scale: scalar("input_scale")?,
        n: wt.shape[0],
        k: wt.shape[1],
    })
}

/// Owned view of one channel-scaled FP8 module (llm-compressor `channel`
/// strategy): e4m3 weight bytes + one f32 scale per output row. Owned scales
/// because the checkpoint stores them BF16 and the consumers want f32.
pub struct Fp8ChannelView<'a> {
    /// [n, k] e4m3 weight bytes.
    pub weight: &'a [u8],
    /// [n] per-output-row scales, decoded to f32.
    pub scales: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

/// Assemble `<prefix>.weight` (e4m3 [n, k]) + `<prefix>.weight_scale`
/// (BF16 or F32, [n, 1] or [n]) into a channel-scaled fp8 view.
pub fn fp8_channel_view<'a>(
    st: &'a ShardedSafetensors,
    prefix: &str,
) -> Result<Fp8ChannelView<'a>, StError> {
    let bad = |m: String| StError::Header(m);
    let (wt, wb) = st
        .bytes(&format!("{prefix}.weight"))
        .ok_or_else(|| bad(format!("{prefix}.weight: missing")))?;
    if wt.dtype != StDtype::F8E4m3 || wt.shape.len() != 2 {
        return Err(bad(format!(
            "{prefix}.weight: want F8_E4M3 2-D, got {:?} {:?}",
            wt.dtype, wt.shape
        )));
    }
    let (n, k) = (wt.shape[0], wt.shape[1]);
    let (ts, sb) = st
        .bytes(&format!("{prefix}.weight_scale"))
        .ok_or_else(|| bad(format!("{prefix}.weight_scale: missing")))?;
    let rows: usize = ts.shape.iter().product();
    if rows != n {
        return Err(bad(format!(
            "{prefix}.weight_scale: want {n} channel scales, got {:?} {:?}",
            ts.dtype, ts.shape
        )));
    }
    let scales: Vec<f32> = match ts.dtype {
        StDtype::Bf16 => sb
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        StDtype::F32 => sb
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        other => {
            return Err(bad(format!(
                "{prefix}.weight_scale: want BF16/F32, got {other:?}"
            )));
        }
    };
    Ok(Fp8ChannelView {
        weight: wb,
        scales,
        n,
        k,
    })
}

impl Fp8ChannelView<'_> {
    /// Reference dequant of one output row, exact f32: e4m3 * row_scale.
    pub fn dequant_row_f32(&self, row: usize) -> Vec<f32> {
        assert!(row < self.n);
        let s = self.scales[row];
        self.weight[row * self.k..(row + 1) * self.k]
            .iter()
            .map(|&b| e4m3_to_f32(b) * s)
            .collect()
    }
}

/// One f32 scale per output row of an fp8 plane, from `<base>.weight_scale`
/// in either export's layout: llm-compressor's channel strategy (BF16, one
/// per row, `[rows]` or `[rows, 1]`) or ModelOpt's per-tensor scale (one F32,
/// `[]` or `[1]`), broadcast - exact either way, the plane's values are
/// `e4m3 * scale` per row. `None` for any other shape (a block grid has its
/// own `_scale_inv` path).
pub fn fp8_row_scales(st: &ShardedSafetensors, base: &str, rows: usize) -> Option<Vec<f32>> {
    let (t, b) = st.bytes(&format!("{base}.weight_scale"))?;
    row_scales_from(t.dtype, &t.shape, b, rows)
}

fn row_scales_from(dtype: StDtype, shape: &[usize], b: &[u8], rows: usize) -> Option<Vec<f32>> {
    let n: usize = shape.iter().product();
    match dtype {
        StDtype::Bf16 if n == rows && b.len() == rows * 2 => Some(
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect(),
        ),
        StDtype::F32 if n == 1 && b.len() == 4 => {
            Some(vec![f32::from_le_bytes([b[0], b[1], b[2], b[3]]); rows])
        }
        _ => None,
    }
}

/// e2m1 magnitude table (3-bit code), sign in bit 3.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decode one 4-bit e2m1 code (code 0x8 is -0.0, matching the hardware).
pub fn e2m1_to_f32(code: u8) -> f32 {
    let mag = E2M1[(code & 0x7) as usize];
    if code & 0x8 != 0 { -mag } else { mag }
}

/// Decode e4m3fn (bias 7, no inf, 0x7f/0xff = NaN).
pub fn e4m3_to_f32(b: u8) -> f32 {
    if b & 0x7f == 0x7f {
        return f32::NAN;
    }
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (b >> 3) & 0xf;
    let mant = (b & 0x7) as f32;
    if exp == 0 {
        sign * (mant / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + mant / 8.0) * 2f32.powi(exp as i32 - 7)
    }
}

impl Nvfp4View<'_> {
    /// Reference dequant of one output row, exact f32, op order
    /// (e2m1 * e4m3) * scale_2 - the order the oracle bits were pinned with.
    pub fn dequant_row_f32(&self, row: usize) -> Vec<f32> {
        assert!(row < self.n);
        let kh = self.k / 2;
        let packed = &self.packed[row * kh..(row + 1) * kh];
        let scales = &self.scales[row * (self.k / 16)..(row + 1) * (self.k / 16)];
        let mut out = Vec::with_capacity(self.k);
        for (j, &byte) in packed.iter().enumerate() {
            let s = e4m3_to_f32(scales[(2 * j) / 16]);
            out.push((e2m1_to_f32(byte & 0x0f) * s) * self.scale2);
            out.push((e2m1_to_f32(byte >> 4) * s) * self.scale2);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn e2m1_covers_all_sixteen_codes() {
        let want = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // positive
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // sign bit set
        ];
        for (code, &w) in want.iter().enumerate() {
            let got = e2m1_to_f32(code as u8);
            assert_eq!(got.to_bits(), w.to_bits(), "code {code:#x}");
        }
    }

    #[test]
    fn fp8_row_scales_read_both_exports() {
        // llm-compressor channel: one bf16 per row (1.0, 2.0)
        let bf = [0x80u8, 0x3f, 0x00, 0x40];
        assert_eq!(
            row_scales_from(StDtype::Bf16, &[2, 1], &bf, 2),
            Some(vec![1.0, 2.0])
        );
        // ModelOpt per-tensor: one f32, broadcast, bit-exact
        let s = 0.003_141_592_7f32;
        let got = row_scales_from(StDtype::F32, &[], &s.to_le_bytes(), 3).unwrap();
        assert!(got.iter().all(|v| v.to_bits() == s.to_bits()) && got.len() == 3);
        // anything else is not a row scale
        assert_eq!(row_scales_from(StDtype::Bf16, &[3, 1], &bf, 2), None);
        assert_eq!(row_scales_from(StDtype::F32, &[2], &[0u8; 8], 2), None);
    }

    #[test]
    fn e4m3_decode_spots() {
        assert_eq!(e4m3_to_f32(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(e4m3_to_f32(0x80).to_bits(), (-0.0f32).to_bits());
        assert_eq!(e4m3_to_f32(0x38), 1.0); // exp 7, mant 0
        assert_eq!(e4m3_to_f32(0x39), 1.125);
        assert_eq!(e4m3_to_f32(0x7e), 448.0); // largest finite
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal
        assert_eq!(e4m3_to_f32(0xc0), -2.0);
        assert!(e4m3_to_f32(0x7f).is_nan());
        assert!(e4m3_to_f32(0xff).is_nan());
    }

    fn nemotron_dir() -> Option<PathBuf> {
        let dir = std::env::var("NEMOTRON_NVFP4_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from("/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
            });
        if dir.join("model.safetensors.index.json").exists() {
            Some(dir)
        } else {
            eprintln!(
                "skip: Nemotron NVFP4 checkpoint not present at {}",
                dir.display()
            );
            None
        }
    }

    #[test]
    fn nemotron_quant_map_parses_and_counts() {
        let Some(dir) = nemotron_dir() else { return };
        let map = ModeloptQuantMap::read(&dir).unwrap();
        // 23 moe layers x 128 experts x {up,down} + 23 x shared {up,down}
        // + lm_head = 5935 nvfp4; 23 mamba layers x {in,out}_proj = 46 fp8.
        let (mut nv, mut f8) = (0usize, 0usize);
        for (_, a) in map.modules() {
            match a {
                QuantAlgo::Nvfp4 { group } => {
                    assert_eq!(group, 16);
                    nv += 1;
                }
                QuantAlgo::Fp8 => f8 += 1,
            }
        }
        assert_eq!((nv, f8), (5935, 46));
        assert!(map.kv_cache_fp8);
        assert_eq!(
            map.algo_for("backbone.layers.1.mixer.experts.0.up_proj"),
            Some(QuantAlgo::Nvfp4 { group: 16 })
        );
        assert_eq!(
            map.algo_for("backbone.layers.0.mixer.in_proj"),
            Some(QuantAlgo::Fp8)
        );
        // attention stays bf16 (not in the map), and the MTP block is entirely
        // unquantized even though its moe layer shape-matches the backbone's.
        assert_eq!(map.algo_for("backbone.layers.12.mixer.q_proj"), None);
        assert_eq!(map.algo_for("mtp.layers.1.mixer.experts.0.up_proj"), None);
    }

    /// Oracle spot values generated by an independent Python implementation
    /// (numpy, vLLM nibble/scale conventions) against the real checkpoint -
    /// exact f32 bit patterns, so any drift in decode tables, nibble order,
    /// scale addressing, or op order fails loudly.
    #[test]
    fn nemotron_nvfp4_reference_matches_python_oracle() {
        let Some(dir) = nemotron_dir() else { return };
        let st = ShardedSafetensors::open_dir(&dir).unwrap();
        struct Case {
            prefix: &'static str,
            shape: (usize, usize),
            scale2_bits: u32,
            spots: [(usize, usize, u32); 5],
        }
        let cases = [
            Case {
                prefix: "backbone.layers.1.mixer.experts.0.up_proj",
                shape: (1856, 2688),
                scale2_bits: 0x38b55555,
                spots: [
                    (165, 2307, 0x3cb55555),
                    (1436, 231, 0xbbb55555),
                    (1214, 1874, 0xbcbb0000),
                    (814, 541, 0x00000000),
                    (803, 253, 0xbc5d0000),
                ],
            },
            Case {
                prefix: "backbone.layers.51.mixer.experts.127.down_proj",
                shape: (2688, 1856),
                scale2_bits: 0x39275555,
                spots: [
                    (239, 1593, 0xbc126aaa),
                    (2080, 159, 0xbcbc4000),
                    (1759, 1294, 0xbcfb0000),
                    (1179, 373, 0xbc512aaa),
                    (1163, 174, 0x3c07f555),
                ],
            },
            Case {
                prefix: "backbone.layers.1.mixer.shared_experts.up_proj",
                shape: (3712, 2688),
                scale2_bits: 0x39640000,
                spots: [
                    (331, 2307, 0xbbe40000),
                    (2872, 231, 0xbb640000),
                    (2429, 1874, 0xbcd5c000),
                    (1629, 541, 0x00000000),
                    (1607, 253, 0xbceb2000),
                ],
            },
            Case {
                prefix: "lm_head",
                shape: (131072, 2688),
                scale2_bits: 0x39455555,
                spots: [
                    (11698, 2307, 0xbb940000),
                    (101443, 231, 0xbb5e0000),
                    (85795, 1874, 0x3c390000),
                    (57524, 541, 0xbc0ac000),
                    (56756, 253, 0x3cde0000),
                ],
            },
        ];
        for c in cases {
            let v = nvfp4_view(&st, c.prefix).unwrap();
            assert_eq!((v.n, v.k), c.shape, "{}", c.prefix);
            assert_eq!(v.scale2.to_bits(), c.scale2_bits, "{} scale_2", c.prefix);
            for (row, col, bits) in c.spots {
                let got = v.dequant_row_f32(row)[col];
                assert_eq!(
                    got.to_bits(),
                    bits,
                    "{} [{row}, {col}]: got {got:e}, want bits {bits:#010x}",
                    c.prefix
                );
            }
        }
    }

    fn qwen38_nvfp4_dir() -> Option<PathBuf> {
        let dir = PathBuf::from("/models/Qwen3.8-27B-NVFP4");
        if dir.join("model.safetensors.index.json").exists() {
            Some(dir)
        } else {
            eprintln!(
                "skip: Qwen3.8 NVFP4 checkpoint not present at {}",
                dir.display()
            );
            None
        }
    }

    /// Dequant one row of an official-FP8 (DeepSeek-style) plane: e4m3 weight
    /// + BF16 `weight_scale_inv` grid, one scale per 128x128 block. The
    ///   independent same-parent reference for the llm-compressor tests below.
    fn fp8_block_row(st: &ShardedSafetensors, name: &str, row: usize) -> Vec<f32> {
        let (wt, wb) = st.bytes(name).unwrap();
        assert_eq!(wt.dtype, StDtype::F8E4m3, "{name}");
        let (rows, cols) = (wt.shape[0], wt.shape[1]);
        assert!(row < rows);
        let (ts, sb) = st.bytes(&format!("{name}_scale_inv")).unwrap();
        assert_eq!(ts.dtype, StDtype::Bf16, "{name}_scale_inv");
        let scols = ts.shape[1];
        let srow = &sb[(row / 128) * scols * 2..];
        (0..cols)
            .map(|c| {
                let s16 = u16::from_le_bytes([srow[(c / 128) * 2], srow[(c / 128) * 2 + 1]]);
                e4m3_to_f32(wb[row * cols + c]) * f32::from_bits((s16 as u32) << 16)
            })
            .collect()
    }

    /// Relative RMS distance between two same-parent quantizations. Small
    /// (both are ~few-% encodings of one bf16 tensor); a flipped
    /// weight_global_scale direction would blow it up by ~global^2 (~1e8).
    fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let (mut d2, mut n2) = (0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            d2 += ((x - y) as f64).powi(2);
            n2 += (y as f64).powi(2);
        }
        assert!(n2 > 0.0, "reference row is all-zero");
        (d2 / n2).sqrt()
    }

    /// llm-compressor NVFP4 normalizes into the same Nvfp4View: names map
    /// (weight_packed / weight_global_scale), scale2 is the reciprocal, and
    /// the dequant lands within cross-quantization distance of the official
    /// FP8 checkpoint's identical plane (same bf16 parent).
    #[test]
    fn qwen38_llm_compressor_nvfp4_view_and_cross_checkpoint_oracle() {
        let Some(dir) = qwen38_nvfp4_dir() else {
            return;
        };
        let st = ShardedSafetensors::open_dir(&dir).unwrap();
        let prefix = "model.language_model.layers.10.mlp.gate_proj";
        let v = nvfp4_view(&st, prefix).unwrap();
        assert_eq!((v.n, v.k), (17408, 5120), "{prefix}");
        // checkpoint stores weight_global_scale = 10304.0 for this plane
        assert_eq!(
            v.scale2.to_bits(),
            (1.0f32 / 10304.0).to_bits(),
            "{prefix} scale2"
        );

        let fp8_dir = PathBuf::from("/models/Qwen3.8-27B-FP8");
        if !fp8_dir.join("model.safetensors.index.json").exists() {
            eprintln!("skip cross-check: official FP8 checkpoint not present");
            return;
        }
        let fp8 = ShardedSafetensors::open_dir(&fp8_dir).unwrap();
        for row in [0usize, 517, 9000, 17407] {
            let got = v.dequant_row_f32(row);
            let want = fp8_block_row(&fp8, &format!("{prefix}.weight"), row);
            let d = rel_rms(&got, &want);
            assert!(
                d < 0.25,
                "{prefix} row {row}: rel RMS {d:.4} vs FP8 checkpoint"
            );
        }
    }

    /// Channel-scaled fp8 planes (attention + the last-8-layer MLPs) decode
    /// against the same cross-checkpoint reference.
    #[test]
    fn qwen38_fp8_channel_view_cross_checkpoint_oracle() {
        let Some(dir) = qwen38_nvfp4_dir() else {
            return;
        };
        let st = ShardedSafetensors::open_dir(&dir).unwrap();
        let fp8_dir = PathBuf::from("/models/Qwen3.8-27B-FP8");
        if !fp8_dir.join("model.safetensors.index.json").exists() {
            eprintln!("skip cross-check: official FP8 checkpoint not present");
            return;
        }
        let fp8 = ShardedSafetensors::open_dir(&fp8_dir).unwrap();
        for (prefix, n, k) in [
            (
                "model.language_model.layers.56.mlp.gate_proj",
                17408usize,
                5120usize,
            ),
            (
                "model.language_model.layers.3.self_attn.q_proj",
                12288,
                5120,
            ),
        ] {
            let v = fp8_channel_view(&st, prefix).unwrap();
            assert_eq!((v.n, v.k), (n, k), "{prefix}");
            assert!(
                v.scales.iter().all(|s| s.is_finite() && *s > 0.0),
                "{prefix} scales"
            );
            for row in [0usize, n / 2, n - 1] {
                let got = v.dequant_row_f32(row);
                let want = fp8_block_row(&fp8, &format!("{prefix}.weight"), row);
                let d = rel_rms(&got, &want);
                assert!(
                    d < 0.25,
                    "{prefix} row {row}: rel RMS {d:.4} vs FP8 checkpoint"
                );
            }
        }
    }

    #[test]
    fn nemotron_fp8_view_reads_mamba_plane() {
        let Some(dir) = nemotron_dir() else { return };
        let st = ShardedSafetensors::open_dir(&dir).unwrap();
        let v = fp8_view(&st, "backbone.layers.0.mixer.in_proj").unwrap();
        assert_eq!((v.n, v.k), (10304, 2688));
        assert_eq!(v.weight.len(), 10304 * 2688);
        assert!(v.weight_scale.is_finite() && v.weight_scale > 0.0);
        assert!(v.input_scale.is_finite() && v.input_scale > 0.0);
    }
}
