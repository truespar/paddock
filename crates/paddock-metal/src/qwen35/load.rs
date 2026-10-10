use super::*;
use paddock_models::{gguf::Value, mapped::MappedGguf};

impl Qwen35 {
    /// Load elected dense 0.8B/4B/9B/27B or MoE 35B geometry. In-file MTP weights are not
    /// executed with speculation off. Unsupported architectures fail loudly.
    pub fn load(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        Self::load_impl(
            path,
            context,
            max_batch,
            budget,
            crate::kv_offload_config().is_some(),
        )
    }

    pub fn load_for_offload(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        Self::load_impl(path, context, max_batch, budget, true)
    }

    fn load_impl(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
        tiered: bool,
    ) -> Result<Self> {
        let source_versions = crate::offload::versions(path)?;
        if path.join("manifest.json").is_file() {
            let source = paddock_models::splash::Target::open(path)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            return Self::from_source(
                checkpoint::Source::Splash(source),
                context,
                max_batch,
                budget,
                tiered,
                source_versions,
                (Geometry::DENSE_27B, 248320, 64, 262144, 1e-6, 10_000_000.),
            );
        }
        if path.is_dir() {
            if path.join("hadamard.json").is_file() {
                let cfg = paddock_models::bonsai::BonsaiConfig::read(path)
                    .map_err(|e| MetalError::Model(e.to_string()))?;
                let geometry = (
                    Geometry::DENSE_27B,
                    248320,
                    64,
                    cfg.text.context,
                    cfg.text.eps,
                    cfg.text.rope,
                );
                let source = paddock_models::safetensors::ShardedSafetensors::open_dir(path)
                    .map_err(|e| MetalError::Model(e.to_string()))?;
                bonsai::validate_tensors(&source, &cfg)?;
                return Self::from_source(
                    checkpoint::Source::Bonsai(source, cfg),
                    context,
                    max_batch,
                    budget,
                    tiered,
                    source_versions,
                    geometry,
                );
            }
            let cfg = paddock_models::mlx::QwenConfig::read(path)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            let source = paddock_models::safetensors::ShardedSafetensors::open_dir(path)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            if source
                .names()
                .any(|n| n.starts_with("mtp.") || n.contains(".mtp."))
            {
                return Err(MetalError::Model(
                    "native MLX requires the sanitized checkpoint without MTP tensors".into(),
                ));
            }
            return Self::from_source(
                checkpoint::Source::Mlx(source, cfg.tied),
                context,
                max_batch,
                budget,
                tiered,
                source_versions,
                (
                    Geometry {
                        width: cfg.width,
                        ff: cfg.ff,
                        layers: cfg.layers,
                        heads: cfg.heads,
                        kv_heads: cfg.kv_heads,
                        value_heads: cfg.value_heads,
                    }
                    .validate()?,
                    248320,
                    64,
                    cfg.context,
                    cfg.eps,
                    cfg.rope,
                ),
            );
        }
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        let is_moe = map.gguf().architecture() == Some("qwen35moe");
        if !matches!(map.gguf().architecture(), Some("qwen35" | "qwen35moe")) {
            return Err(MetalError::Model(
                "Metal Qwen requires elected qwen35/qwen35moe GGUF".into(),
            ));
        }
        let u = |key: &str| -> Result<usize> {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0 && n <= u32::MAX as usize)
                .ok_or_else(|| MetalError::Model(format!("invalid/missing qwen35.{key}")))
        };
        let f = |key: &str| -> Result<f32> {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_f32)
                .filter(|n| n.is_finite() && *n > 0.0)
                .ok_or_else(|| MetalError::Model(format!("invalid/missing qwen35.{key}")))
        };
        let width = u("embedding_length")?;
        let ff = u(if is_moe {
            "expert_shared_feed_forward_length"
        } else {
            "feed_forward_length"
        })?;
        let nextn = match map.gguf().arch_field("nextn_predict_layers") {
            None => 0,
            Some(v) => v.as_u64().filter(|&n| n <= 1).ok_or_else(|| {
                MetalError::Model("Metal Qwen requires zero or one nextn block".into())
            })?,
        };
        let count = u("block_count")?
            .checked_sub(nextn as usize)
            .ok_or_else(|| MetalError::Model("invalid MTP layer count".into()))?;
        let trained = u("context_length")?;
        let rotary = u("rope.dimension_count")?;
        let sections = match map.gguf().arch_field("rope.dimension_sections") {
            Some(Value::Array(values)) => {
                values.iter().map(Value::as_u64).collect::<Option<Vec<_>>>()
            }
            _ => None,
        };
        let eps = f("attention.layer_norm_rms_epsilon")?;
        let rope = f("rope.freq_base")?;
        let geometry = Geometry {
            width,
            ff,
            layers: count,
            heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            value_heads: u("ssm.time_step_rank")?,
        }
        .validate()?;
        if is_moe != geometry.moe() {
            return Err(MetalError::Model(
                "Qwen architecture/geometry mismatch".into(),
            ));
        }
        if is_moe {
            moe::validate(&map, count)?;
        }
        if u("attention.key_length")? != 256
            || u("attention.value_length")? != 256
            || rotary != 64
            || sections.as_deref() != Some(&[11, 11, 10, 0])
            || u("full_attention_interval")? != 4
            || u("ssm.state_size")? != 128
            || u("ssm.group_count")? != KEY_HEADS
            || u("ssm.conv_kernel")? != 4
        {
            return Err(MetalError::Model(
                "Metal Qwen requires head-256 attention, head-128 DeltaNet and unscaled interleaved rotary".into(),
            ));
        }
        if context == 0 || context > trained || max_batch == 0 || max_batch > CHUNK {
            return Err(MetalError::Model(format!(
                "context must be 1..={trained}, batch 1..={CHUNK}"
            )));
        }
        if map.gguf().arch_field("rope.scaling.type").is_some() {
            return Err(MetalError::Model(
                "scaled Qwen rotary is not implemented on Metal".into(),
            ));
        }
        let vocab = map
            .tensor_info("token_embd.weight")
            .and_then(|t| t.dims.get(1))
            .copied()
            .filter(|&n| n > 0 && n <= u32::MAX as u64 / CHUNK as u64)
            .ok_or_else(|| MetalError::Model("invalid embedding table".into()))?
            as usize;
        let rotation = ternary::validate(map.gguf(), geometry, nextn)?;
        let source = if let Some(rotation) = rotation {
            checkpoint::Source::Ternary(map, rotation)
        } else {
            checkpoint::Source::Gguf(map)
        };
        Self::from_source(
            source,
            context,
            max_batch,
            budget,
            tiered,
            source_versions,
            (geometry, vocab, rotary, trained, eps, rope),
        )
    }

    fn from_source(
        map: checkpoint::Source,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
        tiered: bool,
        source_versions: Vec<crate::offload::FileVersion>,
        (g, vocab, rotary, trained, eps, rope): (Geometry, usize, usize, usize, f32, f32),
    ) -> Result<Self> {
        let (width, ff, count) = (g.width, g.ff, g.layers);
        // Conservative implementation ceilings for the new expert graph;
        // checkpoint training length is not runtime qualification.
        let trained = if g.moe() { trained.min(32768) } else { trained };
        let batch_limit = if matches!(
            map,
            checkpoint::Source::Splash(_)
                | checkpoint::Source::Bonsai(_, _)
                | checkpoint::Source::Ternary(_, _)
        ) {
            4
        } else if g.moe() {
            64
        } else {
            CHUNK
        };
        if context == 0 || context > trained || max_batch == 0 || max_batch > batch_limit {
            return Err(MetalError::Model(format!(
                "context must be 1..={trained}, batch 1..={batch_limit}"
            )));
        }
        let mlx = matches!(
            map,
            checkpoint::Source::Mlx(_, _)
                | checkpoint::Source::Splash(_)
                | checkpoint::Source::Bonsai(_, _)
        );
        let is_bonsai = matches!(map, checkpoint::Source::Bonsai(_, _));
        let is_ternary = matches!(map, checkpoint::Source::Ternary(_, _));
        let splash = matches!(map, checkpoint::Source::Splash(_));
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let state_slots = max_batch * 3;
        let blocks = page_stride
            // One bounded async restore may outlive its requesting slot. Keep
            // one context of staging headroom so cancellation + slot reuse can
            // never prevent all live slots from growing to their context limit.
            .checked_mul(if tiered { max_batch + 1 } else { state_slots })
            .and_then(|n| n.checked_add(if tiered { max_batch * 2 } else { 0 }))
            .filter(|&n| n <= u32::MAX as usize / BLOCK_TOKENS)
            .ok_or_else(|| MetalError::Memory("Qwen page count overflow".into()))?;
        let kv_layer = blocks
            .checked_mul(BLOCK_TOKENS * g.kv_heads * 256 * if is_bonsai { 4 } else { 2 })
            .ok_or_else(|| MetalError::Memory("Qwen KV overflow".into()))?;
        let recurrent_bytes =
            (g.linear_layers() * state_slots * (g.state() + g.conv() * 3) * 4) as u64;
        let kv_bytes = kv_layer as u64 * 2 * g.full_layers() as u64 + recurrent_bytes;
        let max_chunks = CHUNK.div_ceil(32) + max_batch * 3;
        let device = MetalDevice::new(budget)?;
        // Conservative grant includes the entire GGUF (even unused MTP),
        // fixed recurrent checkpoints, physical KV and all workspaces.
        let scratch_bound = (CHUNK
            * (width * 5
                + ff * 2
                + g.heads * 256 * 5
                + g.kv_heads * 256 * 2
                + g.conv() * 2
                + g.value_heads * 132
                + g.heads * MAX_SPLITS * 258)
            * 4
            + if splash {
                crate::splash::workspace_bytes(CHUNK, g.projection_width(), ff)
            } else {
                CHUNK * g.projection_width() * 2
            }
            + (max_chunks * g.value_heads * 17408 * g.prepared_element_bytes()).max(if splash {
                crate::splash::attention_scratch_bytes(max_chunks, g.heads)
            } else {
                0
            })
            + max_batch * (page_stride + vocab) * 4
            + CHUNK * 128) as u64;
        let needed = map
            .total_len()
            .saturating_add(if is_bonsai || is_ternary {
                (CHUNK * ff * 4 + (5120 + 6144 + 17408) * 4) as u64
            } else {
                0
            })
            // Two 48-row scalar-gate projections per linear layer need a
            // full 256-row storage tile when split from the packed input.
            .saturating_add(if splash {
                (48 * 2 * 5120 * (256 - 48) / 16 * 9) as u64
            } else {
                0
            })
            // BF16 small planes are expanded on GPU. Bound that additional
            // storage (norms, convolution, scalar gates), not dense weights.
            .saturating_add(if mlx {
                (count * (width * 2 + g.conv() * 4 + 1024) * 4) as u64
            } else {
                0
            })
            .saturating_add(kv_bytes)
            .saturating_add(if g.moe() {
                moe::Workspace::bytes(CHUNK)
            } else {
                0
            })
            .saturating_add(scratch_bound);
        if needed > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "Qwen weights + KV/state + scratch need {:.2} GiB; grant {:.2} GiB",
                needed as f64 / (1u64 << 30) as f64,
                device.budget_bytes() as f64 / (1u64 << 30) as f64
            )));
        }
        let embedding = map.load(&device, "token_embd.weight", &[width, vocab])?;
        let output_norm = map.load(&device, "output_norm.weight", &[width])?;
        let head = map.load_head(&device, &[width, vocab])?;
        let mut layers = Vec::new();
        let mut index = 0;
        for i in 0..count {
            let w =
                |name: &str, dims: &[usize]| map.load(&device, &format!("blk.{i}.{name}"), dims);
            let f32w = |name: &str, dims: &[usize]| -> Result<Weight> {
                let t = w(name, dims)?;
                if t.ty != 0 {
                    return Err(MetalError::Model(format!(
                        "blk.{i}.{name}: expected F32 small tensor"
                    )));
                }
                Ok(t)
            };
            let mixer = if (i + 1) % 4 == 0 {
                Mixer::Full(FullAttention {
                    q: w("attn_q.weight", &[width, g.heads * 512])?,
                    k: w("attn_k.weight", &[width, g.kv_heads * 256])?,
                    v: w("attn_v.weight", &[width, g.kv_heads * 256])?,
                    o: w("attn_output.weight", &[g.heads * 256, width])?,
                    q_norm: f32w("attn_q_norm.weight", &[256])?,
                    k_norm: f32w("attn_k_norm.weight", &[256])?,
                    keys: device.alloc(kv_layer)?,
                    values: device.alloc(kv_layer)?,
                })
            } else {
                let layer = DeltaNet {
                    qkv: w("attn_qkv.weight", &[width, g.conv()])?,
                    z: w("attn_gate.weight", &[width, g.value_heads * 128])?,
                    alpha: w("ssm_alpha.weight", &[width, g.value_heads])?,
                    beta: w("ssm_beta.weight", &[width, g.value_heads])?,
                    out: w("ssm_out.weight", &[g.value_heads * 128, width])?,
                    conv: f32w("ssm_conv1d.weight", &[4, g.conv()])?,
                    a: f32w("ssm_a", &[g.value_heads])?,
                    dt: f32w("ssm_dt.bias", &[g.value_heads])?,
                    norm: f32w("ssm_norm.weight", &[128])?,
                    index,
                };
                index += 1;
                Mixer::Linear(layer)
            };
            let post_norm_name = map.post_norm(i);
            layers.push(Layer {
                norm: w("attn_norm.weight", &[width])?,
                post_norm: w(post_norm_name, &[width])?,
                mixer,
                gate: w(
                    if g.moe() {
                        "ffn_gate_shexp.weight"
                    } else {
                        "ffn_gate.weight"
                    },
                    &[width, ff],
                )?,
                up: w(
                    if g.moe() {
                        "ffn_up_shexp.weight"
                    } else {
                        "ffn_up.weight"
                    },
                    &[width, ff],
                )?,
                down: w(
                    if g.moe() {
                        "ffn_down_shexp.weight"
                    } else {
                        "ffn_down.weight"
                    },
                    &[ff, width],
                )?,
                moe: if g.moe() {
                    Some(moe::Experts::load(&device, &map, i)?)
                } else {
                    None
                },
            });
        }
        let weight_bytes = device.allocated_bytes() - kv_layer as u64 * 2 * g.full_layers() as u64;
        let bonsai = if let checkpoint::Source::Bonsai(_, ref cfg) = map {
            Some(bonsai::Bonsai::new(&device, cfg)?)
        } else {
            None
        };
        let ternary = if let checkpoint::Source::Ternary(_, ref spec) = map {
            Some(ternary::Ternary::new(&device, spec)?)
        } else {
            None
        };
        let state = device.alloc(g.linear_layers() * state_slots * g.state() * 4)?;
        let conv = device.alloc(g.linear_layers() * state_slots * g.conv() * 3 * 4)?;
        let a = |n: usize| device.alloc(CHUNK * n * 4);
        let scratch = Scratch {
            gemm: device.alloc(if splash {
                crate::splash::workspace_bytes(CHUNK, g.projection_width(), ff)
            } else {
                CHUNK * g.projection_width() * 2
            })?,
            ids: a(1)?,
            outputs: a(1)?,
            meta: a(2)?,
            mrope: a(4)?,
            limits: a(1)?,
            pages: device.alloc(max_batch * page_stride * 4)?,
            spans: a(4)?,
            chunks: device.alloc(max_chunks * 16)?,
            bounds: a(2)?,
            attn_tiles: a(2)?,
            decode_rows: a(1)?,
            long_decode_tiles: a(2)?,
            checkpoint_rows: a(1)?,
            checkpoint_spans: a(4)?,
            x: a(width)?,
            norm: a(width)?,
            delta: a(width)?,
            gate: a(ff)?,
            up: a(ff)?,
            logits: device.alloc(max_batch * vocab * 4)?,
            qraw: a(g.heads * 512)?,
            q: a(g.heads * 256)?,
            k: a(g.kv_heads * 256)?,
            v: a(g.kv_heads * 256)?,
            attn: a(g.heads * 256)?,
            attn_parts: a(g.heads * MAX_SPLITS * 258)?,
            qkv: a(g.conv())?,
            convolved: a(g.conv())?,
            z: a(g.value_heads * 128)?,
            alpha: a(g.value_heads)?,
            beta: a(g.value_heads)?,
            gates: a(g.value_heads * 2)?,
            prepared: device.alloc(
                (max_chunks * g.value_heads * 17408 * g.prepared_element_bytes()).max(if splash {
                    crate::splash::attention_scratch_bytes(max_chunks, g.heads)
                } else {
                    0
                }),
            )?,
        };
        let moe_scratch = if g.moe() {
            Some(moe::Workspace::new(&device, CHUNK)?)
        } else {
            None
        };
        tracing::info!(
            weight_bytes,
            kv_bytes,
            allocated = device.allocated_bytes(),
            "Qwen hybrid weights loaded on Metal"
        );
        Ok(Self {
            cold: None,
            source_versions,
            mlx,
            splash,
            bonsai,
            ternary,
            geometry: g,
            device,
            embedding,
            output_norm,
            head,
            layers,
            scratch,
            moe_scratch,
            state,
            conv,
            slots: (0..max_batch).map(|_| Slot::default()).collect(),
            pending: VecDeque::new(),
            cache: (0..max_batch * 2).map(|_| Checkpoint::default()).collect(),
            clock: 0,
            pool: KvPool::with_blocks(blocks as u32),
            width,
            ff,
            vocab,
            context,
            page_stride,
            state_slots,
            eps,
            rope,
            rotary,
            weight_bytes,
            kv_bytes,
            spec: None,
            verifying: false,
            greedy_output: false,
            mtp: None,
            dflash: None,
            lookup: lookup::Lookup::default(),
            vision: None,
            encoding: VecDeque::new(),
            image_cache: Vec::new(),
            image_cache_reused: 0,
            row_capacity: CHUNK,
            last_gpu_seconds: 0.0,
            admission_cost: serving::AdmissionCost::default(),
            #[cfg(test)]
            diagnostic_serial_prefill: false,
        })
    }
}
