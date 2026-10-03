//! Minimal safetensors reader - the fp8-native ingestion lane's foundation
//! (quant strategy: "stop double-quantizing through Q8_0 when the checkpoint
//! is fp8-native"). Zero-copy: mmap the file, parse the JSON header into a
//! tensor map, hand out byte slices. No external safetensors crate (the
//! format is a length-prefixed JSON header + raw little-endian tensor data;
//! dependency minimalism is a repo principle).
//!
//! Format (spec v0.4): u64 LE header length, then a JSON object mapping
//! tensor name -> {"dtype": "F8_E4M3"|"BF16"|..., "shape": [..],
//! "data_offsets": [begin, end]} (offsets relative to the byte after the
//! header), plus an optional "__metadata__" string map.

use std::collections::HashMap;
use std::path::Path;

/// Tensor dtypes the ingestion lane cares about (the fp8 checkpoints ship
/// F8_E4M3 weights with BF16 norms/scales; F32/F16 appear in older exports).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StDtype {
    F8E4m3,
    Bf16,
    F16,
    F32,
    /// int64 buffers - qwen4_exp ships its PLE n-gram hash constants
    /// (layer_multipliers, per-head vocab sizes/offsets) as I64.
    I64,
    /// Opaque byte tensors - modelopt NVFP4 exports pack two e2m1 nibbles per
    /// byte under dtype "U8" (the [N, K/2] `weight` of the fp4 triple).
    U8,
    /// MLX affine quantization packs eight 4-bit codes in each little-endian word.
    U32,
    /// Anything else - carried so callers can report it, never silently skipped.
    Other,
}

impl StDtype {
    fn parse(s: &str) -> StDtype {
        match s {
            "F8_E4M3" => StDtype::F8E4m3,
            "BF16" => StDtype::Bf16,
            "F16" => StDtype::F16,
            "F32" => StDtype::F32,
            "U8" => StDtype::U8,
            "U32" => StDtype::U32,
            "I64" => StDtype::I64,
            _ => StDtype::Other,
        }
    }

    /// Bytes per element (Other reports 0 - callers must reject, not guess).
    pub fn bytes(self) -> usize {
        match self {
            StDtype::F8E4m3 | StDtype::U8 => 1,
            StDtype::Bf16 | StDtype::F16 => 2,
            StDtype::F32 | StDtype::U32 => 4,
            StDtype::I64 => 8,
            StDtype::Other => 0,
        }
    }
}

/// One tensor's location in the mapped file.
#[derive(Clone, Debug)]
pub struct StTensor {
    pub dtype: StDtype,
    pub shape: Vec<usize>,
    /// Byte range relative to the data section (header end).
    pub begin: usize,
    pub end: usize,
}

/// A memory-mapped safetensors file with its parsed tensor map.
pub struct SafetensorsFile {
    map: memmap2::Mmap,
    #[cfg(unix)]
    file: std::sync::Arc<std::fs::File>,
    data_off: usize,
    tensors: HashMap<String, StTensor>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, thiserror::Error)]
pub enum StError {
    #[error("safetensors io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors header: {0}")]
    Header(String),
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self, StError> {
        let f = std::fs::File::open(path)?;
        // SAFETY: read-only mapping of a file we just opened; safetensors
        // consumers treat the contents as untrusted bytes (offsets validated
        // below before any slice is handed out).
        let map = unsafe { memmap2::Mmap::map(&f)? };
        if map.len() < 8 {
            return Err(StError::Header(
                "file shorter than the length prefix".into(),
            ));
        }
        let hlen = u64::from_le_bytes(map[0..8].try_into().expect("length checked above")) as usize;
        let data_off = 8usize
            .checked_add(hlen)
            .filter(|&o| o <= map.len())
            .ok_or_else(|| StError::Header(format!("header length {hlen} exceeds file")))?;
        let header: serde_json::Value = serde_json::from_slice(&map[8..data_off])
            .map_err(|e| StError::Header(e.to_string()))?;
        let obj = header
            .as_object()
            .ok_or_else(|| StError::Header("header is not a JSON object".into()))?;
        let data_len = map.len() - data_off;
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (k, mv) in m {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(k.clone(), s.to_owned());
                        }
                    }
                }
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .map(StDtype::parse)
                .ok_or_else(|| StError::Header(format!("{name}: missing dtype")))?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .and_then(|a| {
                    a.iter()
                        .map(|x| x.as_u64().and_then(|v| usize::try_from(v).ok()))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| StError::Header(format!("{name}: missing shape")))?;
            let offs = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .filter(|a| a.len() == 2)
                .ok_or_else(|| StError::Header(format!("{name}: missing data_offsets")))?;
            let (begin, end) = (
                offs[0].as_u64().unwrap_or(u64::MAX) as usize,
                offs[1].as_u64().unwrap_or(u64::MAX) as usize,
            );
            if begin > end || end > data_len {
                return Err(StError::Header(format!(
                    "{name}: offsets [{begin}, {end}) exceed data section ({data_len})"
                )));
            }
            // element count × dtype size must match the byte span (dtype
            // Other is exempt - bytes() is 0 and callers reject on use)
            let n = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| StError::Header(format!("{name}: element count overflow")))?;
            if dtype.bytes() != 0 && n.checked_mul(dtype.bytes()) != Some(end - begin) {
                return Err(StError::Header(format!(
                    "{name}: shape {shape:?} x {dtype:?} != {} bytes",
                    end - begin
                )));
            }
            tensors.insert(
                name.clone(),
                StTensor {
                    dtype,
                    shape,
                    begin,
                    end,
                },
            );
        }
        Ok(Self {
            map,
            #[cfg(unix)]
            file: std::sync::Arc::new(f),
            data_off,
            tensors,
            metadata,
        })
    }

    pub fn tensors(&self) -> &HashMap<String, StTensor> {
        &self.tensors
    }

    /// Raw bytes of `name`, or None if absent.
    pub fn bytes(&self, name: &str) -> Option<(&StTensor, &[u8])> {
        let t = self.tensors.get(name)?;
        Some((t, &self.map[self.data_off + t.begin..self.data_off + t.end]))
    }

    /// Positional file I/O directly into a caller-owned allocation. Unlike
    /// copying a touched mmap, this does not retain a second process-resident
    /// view of large immutable weights. No shared seek cursor or model math.
    #[cfg(unix)]
    fn read_into(&self, name: &str, out: &mut [u8]) -> Result<(), StError> {
        use std::os::unix::fs::FileExt;
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| StError::Header(format!("missing tensor {name}")))?;
        if out.len() != t.end - t.begin {
            return Err(StError::Header(format!("read size differs for {name}")));
        }
        self.file
            .read_exact_at(out, (self.data_off + t.begin) as u64)?;
        Ok(())
    }

    #[cfg(unix)]
    fn reader(&self, name: &str) -> Result<TensorFileReader, StError> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| StError::Header(format!("missing tensor {name}")))?;
        Ok(TensorFileReader {
            file: self.file.clone(),
            begin: (self.data_off + t.begin) as u64,
            length: t.end - t.begin,
        })
    }
}

/// Bounded positional access to one validated tensor. Retains the opened file,
/// not its mmap, and shares the descriptor rather than cloning it per plane.
/// Replacing a checkpoint path cannot redirect an already opened reader.
#[cfg(unix)]
pub struct TensorFileReader {
    file: std::sync::Arc<std::fs::File>,
    begin: u64,
    length: usize,
}
#[cfg(unix)]
impl TensorFileReader {
    pub fn read_at(&self, offset: usize, out: &mut [u8]) -> Result<(), StError> {
        use std::os::unix::fs::FileExt;
        if !offset
            .checked_add(out.len())
            .is_some_and(|end| end <= self.length)
        {
            return Err(StError::Header(
                "tensor read outside validated extent".into(),
            ));
        }
        self.file.read_exact_at(out, self.begin + offset as u64)?;
        Ok(())
    }
}

/// A sharded checkpoint directory (model.safetensors.index.json + shards).
/// Single-file checkpoints (model.safetensors, no index) also load.
pub struct ShardedSafetensors {
    shards: Vec<SafetensorsFile>,
    /// tensor name -> shard index
    index: HashMap<String, usize>,
}

impl ShardedSafetensors {
    /// Fault a tensor's bytes into the page cache and BLOCK until they are
    /// there. For a table the serve will random-read forever after - the PLE
    /// n-gram table is 26.8 GiB of this checkpoint and every token gathers 16
    /// rows out of it - the alternative is paying those faults one disk seek
    /// at a time on the critical path, which is a TTFT problem, not a
    /// throughput one (the repo's own `ple-table-page-cache-trap`).
    ///
    /// `advise(WillNeed)` only starts readahead, so the touch loop is the part
    /// that makes residency deterministic: one byte a page, which costs a
    /// fault the first time and nothing after. Returns bytes touched.
    ///
    /// Every platform, not only Unix: the touch loop is plain reads of a mapped
    /// file, and this model is served on Windows as well. Only the advisory
    /// hint is Unix (memmap2 has no Windows `advise`), and without it the loop
    /// does the same work a little less cleverly.
    pub fn warm_tensor(&self, name: &str) -> Result<usize, StError> {
        let Some((_, bytes)) = self.bytes(name) else {
            return Err(StError::Header(format!("missing tensor {name}")));
        };
        // advisory only - best effort, and a kernel that declines it just
        // means the touch loop below does the work itself
        #[cfg(unix)]
        {
            use memmap2::Advice;
            let shard = self
                .index
                .get(name)
                .ok_or_else(|| StError::Header(format!("missing tensor {name}")))?;
            let file = &self.shards[*shard];
            // offset of this tensor inside the mapping
            let base = file.map.as_ptr() as usize;
            let off = bytes.as_ptr() as usize - base;
            let _ = file.map.advise_range(Advice::WillNeed, off, bytes.len());
        }
        let mut sink = 0u64;
        let page = 4096usize;
        let mut i = 0usize;
        while i < bytes.len() {
            // volatile so the loop cannot be optimized away into nothing
            sink = sink.wrapping_add(unsafe { std::ptr::read_volatile(&bytes[i]) } as u64);
            i += page;
        }
        std::hint::black_box(sink);
        Ok(bytes.len())
    }

    #[cfg(unix)]
    pub fn read_into(&self, name: &str, out: &mut [u8]) -> Result<(), StError> {
        let shard = self
            .index
            .get(name)
            .ok_or_else(|| StError::Header(format!("missing tensor {name}")))?;
        self.shards[*shard].read_into(name, out)
    }
    #[cfg(unix)]
    pub fn reader(&self, name: &str) -> Result<TensorFileReader, StError> {
        let shard = self
            .index
            .get(name)
            .ok_or_else(|| StError::Header(format!("missing tensor {name}")))?;
        self.shards[*shard].reader(name)
    }
    pub fn open_dir(dir: &Path) -> Result<Self, StError> {
        let idx_path = dir.join("model.safetensors.index.json");
        if !idx_path.exists() {
            let single = dir.join("model.safetensors");
            let f = SafetensorsFile::open(&single)?;
            let index = f.tensors().keys().map(|k| (k.clone(), 0usize)).collect();
            return Ok(Self {
                shards: vec![f],
                index,
            });
        }
        let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx_path)?)
            .map_err(|e| StError::Header(e.to_string()))?;
        let wm = idx
            .get("weight_map")
            .and_then(|w| w.as_object())
            .ok_or_else(|| StError::Header("index.json: missing weight_map".into()))?;
        let mut shard_ids: Vec<String> = wm
            .values()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        shard_ids.sort();
        shard_ids.dedup();
        let mut shards = Vec::with_capacity(shard_ids.len());
        let mut pos = HashMap::new();
        for (i, sid) in shard_ids.iter().enumerate() {
            // An index names sibling shards, never an arbitrary host path.
            if Path::new(sid).components().count() != 1
                || sid.contains(['/', '\\'])
                || !matches!(
                    Path::new(sid).components().next(),
                    Some(std::path::Component::Normal(_))
                )
            {
                return Err(StError::Header(format!("invalid shard filename {sid:?}")));
            }
            pos.insert(sid.clone(), i);
            shards.push(SafetensorsFile::open(&dir.join(sid))?);
        }
        let mut index = HashMap::new();
        for (name, sid) in wm {
            let sid = sid
                .as_str()
                .ok_or_else(|| StError::Header("weight_map value".into()))?;
            index.insert(name.clone(), pos[sid]);
        }
        Ok(Self { shards, index })
    }

    /// One file as a single-shard set (a companion file beside a GGUF, say).
    pub fn from_file(f: SafetensorsFile) -> Self {
        let index = f.tensors().keys().map(|k| (k.clone(), 0usize)).collect();
        Self {
            shards: vec![f],
            index,
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    /// Mapped storage, including tensor headers, for conservative load budgeting.
    pub fn total_len(&self) -> u64 {
        self.shards.iter().map(|s| s.map.len() as u64).sum()
    }

    pub fn bytes(&self, name: &str) -> Option<(&StTensor, &[u8])> {
        self.shards[*self.index.get(name)?].bytes(name)
    }

    /// Hint how `ranges` (byte offset, length inside tensor `name`) are about
    /// to be read - see [`crate::mapped::MapAccess`]. Best effort; `None`
    /// only when the tensor is absent.
    pub fn advise_tensor(
        &self,
        name: &str,
        access: crate::mapped::MapAccess,
        ranges: &[(usize, usize)],
    ) -> Option<()> {
        let file = &self.shards[*self.index.get(name)?];
        let (_, bytes) = file.bytes(name)?;
        crate::mapped::advise_ranges(&file.map, bytes, access, ranges);
        Some(())
    }
}

/// DFlash drafter checkpoint config - the config.json beside a
/// Laguna-*-DFlash model.safetensors (poolside's block-diffusion speculator,
/// arXiv 2602.06036). Parsed manually off serde_json::Value like the header
/// above; every field the engine consumes is validated present so a config
/// drift fails loudly at load, never as silent garbage geometry.
#[derive(Clone, Debug)]
pub struct DflashConfig {
    pub n_layer: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub window: usize,
    pub max_pos: usize,
    pub eps: f32,
    pub rope_theta: f32,
    /// dflash_config.block_size - rows per draft block (committed + masks).
    pub block: usize,
    /// dflash_config.mask_token_id - the noise-row token.
    pub mask_token: u32,
    /// dflash_config.target_layer_ids - 0-indexed target layers whose
    /// post-block residuals feed the fusion fc (concat order = this order).
    pub target_layer_ids: Vec<usize>,
    /// dflash_config.causal - the poolside variant drafts causally; the
    /// engine only implements that flavor (bidirectional = z-lab's qwen3 arm).
    pub causal: bool,
}

impl DflashConfig {
    pub fn read(path: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| StError::Header(e.to_string()))?;
        let miss = |k: &str| StError::Header(format!("dflash config.json: missing {k}"));
        let u = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| miss(k))
        };
        let f = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| miss(k))
        };
        let dc = v
            .get("dflash_config")
            .ok_or_else(|| miss("dflash_config"))?;
        let du = |k: &str| {
            dc.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| {
                    StError::Header(format!("dflash config.json: missing dflash_config.{k}"))
                })
        };
        let ids = dc
            .get("target_layer_ids")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect::<Vec<_>>()
            })
            .ok_or_else(|| miss("dflash_config.target_layer_ids"))?;
        Ok(Self {
            n_layer: u("num_hidden_layers")?,
            hidden: u("hidden_size")?,
            n_heads: u("num_attention_heads")?,
            n_kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            intermediate: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            window: u("sliding_window")?,
            max_pos: u("max_position_embeddings")?,
            eps: f("rms_norm_eps")?,
            rope_theta: f("rope_theta")?,
            block: du("block_size")?,
            mask_token: du("mask_token_id")? as u32,
            target_layer_ids: ids,
            causal: dc.get("causal").and_then(|x| x.as_bool()).unwrap_or(false),
        })
    }
}

/// HF `config.json` of the Qwen3-ForcedAligner checkpoint
/// (`architectures = ["Qwen3ASRForTokenClassification"]`): the
/// Qwen3-ASR audio tower + a stock Qwen3 dense text stack + a bias-free
/// `score` head classifying each `<timestamp>` position into one of
/// `n_labels` time bins of `segment_ms` milliseconds. Same parsing stance as
/// [`DflashConfig`]: every consumed field validated present, loud on miss.
pub struct AlignerConfig {
    /// Exact ordinary causal Qwen3/audio semantics. Native backends must not
    /// silently ignore a scaled rotary, biased classifier or changed frontend.
    pub vanilla_graph: bool,
    /// Exact integrated audio frontend declaration; old CUDA readers do not
    /// require it, but a new native loader must fail closed on missing/drifted
    /// preprocessing instead of serving with guessed defaults.
    pub vanilla_frontend: bool,
    // text stack (config.text_config)
    pub n_layer: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub max_pos: usize,
    pub eps: f32,
    pub rope_theta: f32,
    // audio tower (config.audio_config)
    pub a_layers: usize,
    pub a_dmodel: usize,
    pub a_heads: usize,
    pub a_ffn: usize,
    pub a_out_dim: usize,
    pub a_mels: usize,
    /// conv stem channel width (`downsample_hidden_size`)
    pub a_ch: usize,
    /// sinusoidal position rows (`max_position_embeddings`, 13 - per-chunk)
    pub a_max_pos: usize,
    // aligner head
    pub audio_token_id: u32,
    pub timestamp_token_id: u32,
    /// milliseconds per time-bin class (`timestamp_segment_time`)
    pub segment_ms: f32,
    /// classification width (the `id2label` table's size; `score` rows)
    pub n_labels: usize,
}

impl AlignerConfig {
    pub fn read(path: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| StError::Header(e.to_string()))?;
        let arch = v
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|a| a.as_str())
            .unwrap_or_default();
        if arch != "Qwen3ASRForTokenClassification" {
            return Err(StError::Header(format!(
                "aligner config.json: architectures[0] '{arch}' (want Qwen3ASRForTokenClassification)"
            )));
        }
        let sub = |k: &str| {
            v.get(k)
                .ok_or_else(|| StError::Header(format!("aligner config.json: missing {k}")))
        };
        let getu = |o: &serde_json::Value, scope: &str, k: &str| {
            o.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| StError::Header(format!("aligner config.json: missing {scope}{k}")))
        };
        let getf = |o: &serde_json::Value, scope: &str, k: &str| {
            o.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| StError::Header(format!("aligner config.json: missing {scope}{k}")))
        };
        let t = sub("text_config")?;
        let a = sub("audio_config")?;
        let n_labels = v
            .get("id2label")
            .and_then(|m| m.as_object())
            .map(|m| m.len())
            .ok_or_else(|| StError::Header("aligner config.json: missing id2label".into()))?;
        let processor = path
            .parent()
            .and_then(|dir| std::fs::read(dir.join("processor_config.json")).ok())
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        let vanilla_frontend = processor.as_ref().is_some_and(|p| {
            p.get("processor_class").and_then(|x| x.as_str()) == Some("Qwen3ASRProcessor")
                && p.get("timestamp_segment_time").and_then(|x| x.as_u64()) == Some(80)
                && p.get("feature_extractor").is_some_and(|f| {
                    [
                        ("feature_size", 128),
                        ("hop_length", 160),
                        ("min_length", 8000),
                        ("n_fft", 400),
                        ("n_window", 50),
                        ("sampling_rate", 16000),
                        ("chunk_length", 30),
                    ]
                    .iter()
                    .all(|&(k, n)| f.get(k).and_then(|x| x.as_u64()) == Some(n))
                        && f.get("dither").and_then(|x| x.as_f64()) == Some(0.)
                        && f.get("padding_value").and_then(|x| x.as_f64()) == Some(0.)
                        && f.get("padding_side").and_then(|x| x.as_str()) == Some("right")
                        && f.get("return_attention_mask").and_then(|x| x.as_bool()) == Some(true)
                })
        });
        Ok(Self {
            vanilla_frontend,
            vanilla_graph: t.get("hidden_act").and_then(|x| x.as_str()) == Some("silu")
                && t.get("attention_bias").and_then(|x| x.as_bool()) == Some(false)
                && t.get("use_sliding_window").and_then(|x| x.as_bool()) == Some(false)
                && t.get("sliding_window").is_none_or(|x| x.is_null())
                && t.get("layer_types")
                    .and_then(|x| x.as_array())
                    .is_some_and(|ls| {
                        ls.len()
                            == t.get("num_hidden_layers")
                                .and_then(|x| x.as_u64())
                                .unwrap_or(0) as usize
                            && ls.iter().all(|x| x.as_str() == Some("full_attention"))
                    })
                && t.get("rope_parameters").is_some_and(|r| {
                    r.get("rope_type").and_then(|x| x.as_str()) == Some("default")
                        && r.as_object().is_some_and(|o| o.len() == 2)
                })
                && a.get("activation_function").and_then(|x| x.as_str()) == Some("gelu")
                && a.get("scale_embedding").and_then(|x| x.as_bool()) == Some(false)
                && a.get("n_window").and_then(|x| x.as_u64()) == Some(50)
                && a.get("n_window_infer").and_then(|x| x.as_u64()) == Some(800)
                && v.get("token_classification_bias").and_then(|x| x.as_bool()) == Some(false),
            n_layer: getu(t, "text_config.", "num_hidden_layers")?,
            hidden: getu(t, "text_config.", "hidden_size")?,
            n_heads: getu(t, "text_config.", "num_attention_heads")?,
            n_kv_heads: getu(t, "text_config.", "num_key_value_heads")?,
            head_dim: getu(t, "text_config.", "head_dim")?,
            intermediate: getu(t, "text_config.", "intermediate_size")?,
            vocab: getu(t, "text_config.", "vocab_size")?,
            max_pos: getu(t, "text_config.", "max_position_embeddings")?,
            eps: getf(t, "text_config.", "rms_norm_eps")?,
            rope_theta: t
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| {
                    StError::Header(
                        "aligner config.json: missing text_config.rope_parameters.rope_theta"
                            .into(),
                    )
                })?,
            a_layers: getu(a, "audio_config.", "encoder_layers")?,
            a_dmodel: getu(a, "audio_config.", "d_model")?,
            a_heads: getu(a, "audio_config.", "encoder_attention_heads")?,
            a_ffn: getu(a, "audio_config.", "encoder_ffn_dim")?,
            a_out_dim: getu(a, "audio_config.", "output_dim")?,
            a_mels: getu(a, "audio_config.", "num_mel_bins")?,
            a_ch: getu(a, "audio_config.", "downsample_hidden_size")?,
            a_max_pos: getu(a, "audio_config.", "max_position_embeddings")?,
            audio_token_id: getu(&v, "", "audio_token_id")? as u32,
            timestamp_token_id: getu(&v, "", "timestamp_token_id")? as u32,
            segment_ms: getf(&v, "", "timestamp_segment_time")?,
            n_labels,
        })
    }
}

/// HF checkpoint tensor name for a paddock/GGUF-style qwen3.5/3.6 tensor name
/// (`blk.{i}.<gguf>` -> `model.language_model.layers.{i}.<hf>`), so the fp8
/// ingestion lane can drive the existing loader name-by-name. Returns None
/// for names with no HF counterpart.
pub fn qwen35_hf_name(gguf: &str) -> Option<String> {
    // the head is the one non-blk plane the fp8 lane can source: the NVFP4
    // export ships lm_head.weight as an fp8 channel-strategy island
    if gguf == "output.weight" {
        return Some("lm_head.weight".to_string());
    }
    let rest = gguf.strip_prefix("blk.")?;
    let (i, name) = rest.split_once('.')?;
    let hf = match name {
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_a" => "linear_attn.A_log",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_norm.weight" => "input_layernorm.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        _ => return None,
    };
    Some(format!("model.language_model.layers.{i}.{hf}"))
}

/// HF checkpoint tensor name for a paddock/GGUF-style gemma4 tensor name
/// (`blk.{i}.<gguf>` -> `model.language_model.layers.{i}.<hf>`), so the fp8
/// ingestion lane can source the serving planes from google/gemma-4-31B-it
/// safetensors. Planes only - norms/embeddings stay GGUF-sourced. Note the
/// checkpoint has no v_proj on every 6th layer (5, 11, ..., 59) - the loader's
/// `Option<wv>` already models those; the lookup just misses there.
pub fn gemma4_hf_name(gguf: &str) -> Option<String> {
    let rest = gguf.strip_prefix("blk.")?;
    let (i, name) = rest.split_once('.')?;
    let hf = match name {
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.language_model.layers.{i}.{hf}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn unique_path() -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "paddock-st-{}-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
    }

    fn write_header(header: serde_json::Value, data: &[u8]) -> std::path::PathBuf {
        let path = unique_path();
        let header = header.to_string();
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(data).unwrap();
        path
    }

    fn write_st(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                (*name).to_owned(),
                serde_json::json!({"dtype": dtype, "shape": shape,
                                   "data_offsets": [begin, data.len()]}),
            );
        }
        write_header(serde_json::Value::Object(header), &data)
    }

    #[test]
    fn parses_packed_u32_and_rejects_malformed_dimensions() {
        let path = write_st(&[("w", "U32", vec![2, 4], vec![0; 32])]);
        let file = SafetensorsFile::open(&path).unwrap();
        assert_eq!(file.bytes("w").unwrap().0.dtype, StDtype::U32);
        assert_eq!(file.bytes("w").unwrap().1.len(), 32);
        drop(file);
        std::fs::remove_file(path).unwrap();
        for shape in [
            serde_json::json!([1, "bad"]),
            serde_json::json!([-1]),
            serde_json::json!([u64::MAX, 2]),
            serde_json::json!([u64::MAX]),
        ] {
            let path = write_header(
                serde_json::json!({"w":{"dtype":"U32","shape":shape,"data_offsets":[0,4]}}),
                &[0; 4],
            );
            assert!(SafetensorsFile::open(&path).is_err());
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    #[cfg(unix)]
    fn positional_weight_io_preserves_bytes_and_has_no_shared_cursor() {
        let path = write_st(&[
            ("first", "U32", vec![4], (0..16).collect()),
            ("second", "BF16", vec![3], vec![7, 8, 9, 10, 11, 12]),
        ]);
        let file = SafetensorsFile::open(&path).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let file = &file;
                scope.spawn(move || {
                    for name in ["second", "first", "second"] {
                        let expected = file.bytes(name).unwrap().1;
                        let mut out = vec![0; expected.len()];
                        file.read_into(name, &mut out).unwrap();
                        assert_eq!(out, expected);
                    }
                });
            }
        });
        let mut short = [42; 2];
        assert!(file.read_into("first", &mut short).is_err());
        assert!(file.read_into("missing", &mut short).is_err());
        assert_eq!(short, [42; 2]);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn tensor_reader_retains_file_not_mapping_and_checks_every_extent() {
        let path = write_st(&[("plane", "U32", vec![4], (0..16).collect())]);
        let mapped = SafetensorsFile::open(&path).unwrap();
        let reader = mapped.reader("plane").unwrap();
        assert!(mapped.reader("absent").is_err());
        drop(mapped);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let reader = &reader;
                scope.spawn(move || {
                    let mut out = [42; 4];
                    reader.read_at(5, &mut out).unwrap();
                    assert_eq!(out, [5, 6, 7, 8]);
                    assert!(reader.read_at(13, &mut out).is_err());
                    assert!(reader.read_at(usize::MAX, &mut out).is_err());
                    assert_eq!(out, [5, 6, 7, 8]);
                    reader.read_at(16, &mut []).unwrap();
                });
            }
        });
        // No mmap remains: a truncated backing file becomes a recoverable
        // I/O error, not an out-of-mapping access or stale-buffer success.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        assert!(reader.read_at(0, &mut [0; 4]).is_err());
        drop(reader);
        std::fs::remove_file(path).unwrap();
    }

    /// No cfg on this one deliberately: the method went in Unix-only, the
    /// engine called it unconditionally, and the Windows build broke. It has to
    /// exist and do its work wherever the workspace compiles.
    #[test]
    fn warming_a_tensor_touches_all_of_it_on_every_platform() {
        let dir = unique_path().with_extension("warm");
        std::fs::create_dir(&dir).unwrap();
        // three pages and a bit, so the touch loop takes more than one step
        let table: Vec<u8> = (0..4096 * 3 + 17).map(|i| (i % 251) as u8).collect();
        let single = write_st(&[
            ("small", "U8", vec![4], vec![1, 2, 3, 4]),
            ("table", "U8", vec![table.len()], table.clone()),
        ]);
        std::fs::rename(&single, dir.join("model.safetensors")).unwrap();
        let st = ShardedSafetensors::open_dir(&dir).unwrap();
        assert_eq!(st.warm_tensor("table").unwrap(), table.len());
        assert_eq!(st.warm_tensor("small").unwrap(), 4);
        // warming reads, it does not change what the mapping holds
        assert_eq!(st.bytes("table").unwrap().1, &table[..]);
        assert!(st.warm_tensor("missing").is_err());
        drop(st);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_non_sibling_shard_paths_before_opening() {
        let dir = unique_path().with_extension("shards");
        std::fs::create_dir(&dir).unwrap();
        let index = dir.join("model.safetensors.index.json");
        for name in [
            "../w.safetensors",
            "/tmp/w.safetensors",
            "a/b.safetensors",
            "a\\b.safetensors",
            ".",
            "..",
        ] {
            std::fs::write(
                &index,
                serde_json::json!({"weight_map":{"w":name}}).to_string(),
            )
            .unwrap();
            assert!(matches!(
                ShardedSafetensors::open_dir(&dir),
                Err(StError::Header(_))
            ));
        }
        std::fs::remove_file(index).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn parses_fp8_and_bf16_tensors() {
        let path = write_st(&[
            (
                "model.layers.0.mlp.gate_proj.weight",
                "F8_E4M3",
                vec![4, 8],
                (0..32).collect(),
            ),
            ("model.norm.weight", "BF16", vec![4], vec![0u8; 8]),
        ]);
        let st = SafetensorsFile::open(&path).expect("open");
        let (t, b) = st
            .bytes("model.layers.0.mlp.gate_proj.weight")
            .expect("tensor");
        assert_eq!(t.dtype, StDtype::F8E4m3);
        assert_eq!(t.shape, vec![4, 8]);
        assert_eq!(b.len(), 32);
        assert_eq!(b[31], 31);
        let (t2, b2) = st.bytes("model.norm.weight").expect("norm");
        assert_eq!(t2.dtype, StDtype::Bf16);
        assert_eq!(b2.len(), 8);
    }

    #[test]
    fn rejects_bad_offsets_and_shape_mismatch() {
        // shape says 8 elements of F32 (32 bytes) but span is 4 bytes
        let path = write_st(&[("w", "F32", vec![8], vec![0u8; 4])]);
        assert!(SafetensorsFile::open(&path).is_err());
    }

    /// Live test against a real sharded checkpoint (env-gated like the
    /// vision smoke test): QWEN36_HF_SNAPSHOT=<snapshot dir>.
    #[test]
    fn maps_qwen36_checkpoint_when_present() {
        let Some(dir) = std::env::var_os("QWEN36_HF_SNAPSHOT") else {
            return;
        };
        let st = ShardedSafetensors::open_dir(Path::new(&dir)).expect("open");
        for gguf in [
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ssm_out.weight",
        ] {
            let hf = qwen35_hf_name(gguf).expect("mapped");
            let (t, b) = st.bytes(&hf).unwrap_or_else(|| panic!("missing {hf}"));
            assert!(
                t.dtype == StDtype::Bf16 || t.dtype == StDtype::F8E4m3,
                "{hf}: {:?}",
                t.dtype
            );
            assert!(!b.is_empty());
        }
    }
}
