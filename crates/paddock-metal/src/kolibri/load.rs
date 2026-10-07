use super::*;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};
use std::{collections::HashSet, path::Path};

struct Source {
    map: ShardedSafetensors,
    config: KolibriConfig,
}
impl Source {
    fn parts(&self, base: &str, shape: &[usize], bits: usize) -> Result<Vec<(String, usize)>> {
        self.config
            .quantization(base, bits)
            .map_err(MetalError::Model)?;
        let mut result = Vec::new();
        for (suffix, dtype, divisor) in [
            ("weight", StDtype::U32, 32 / bits),
            ("scales", StDtype::Bf16, 64),
            ("biases", StDtype::Bf16, 64),
        ] {
            let mut dims = shape.to_vec();
            *dims.last_mut().expect("internal nonempty matrix shape") /= divisor;
            let name = format!("{base}.{suffix}");
            let len = self.check(&name, &dims, dtype)?;
            result.push((name, len));
        }
        Ok(result)
    }
    fn check(&self, name: &str, shape: &[usize], dtype: StDtype) -> Result<usize> {
        let (t, b) = self
            .map
            .bytes(name)
            .ok_or_else(|| MetalError::Model(format!("missing {name}")))?;
        if t.shape != shape || t.dtype != dtype {
            return Err(MetalError::Model(format!(
                "{name}: expected {dtype:?} {shape:?}, got {:?} {:?}",
                t.dtype, t.shape
            )));
        }
        Ok(b.len())
    }
    fn matrix(&self, d: &MetalDevice, name: &str, shape: &[usize], bits: usize) -> Result<Matrix> {
        let parts = self.parts(name, shape, bits)?;
        let buffer = d.upload_with(parts.iter().map(|(_, n)| n).sum(), |out| {
            let mut offset = 0;
            for (name, len) in &parts {
                self.map
                    .read_into(name, &mut out[offset..offset + len])
                    .map_err(|e| MetalError::Model(e.to_string()))?;
                offset += len;
            }
            Ok(())
        })?;
        Ok(Matrix {
            buffer,
            k: *shape.last().expect("checked shape"),
            n: shape[..shape.len() - 1].iter().product(),
            bits,
        })
    }
    fn small(
        &self,
        d: &MetalDevice,
        name: &str,
        shape: &[usize],
        dtype: StDtype,
    ) -> Result<Buffer> {
        let len = self.check(name, shape, dtype)?;
        let source = d.upload_with(len, |out| {
            self.map
                .read_into(name, out)
                .map_err(|e| MetalError::Model(e.to_string()))
        })?;
        if dtype == StDtype::F32 {
            return Ok(source);
        }
        let n = shape.iter().product::<usize>();
        let out = d.alloc(n * 4)?;
        let cmd = d.begin()?;
        cmd.dispatch(
            "q4a_small",
            &[&source, &out],
            &[n as u32, 0],
            [n.div_ceil(256), 1, 1],
            256,
        );
        cmd.finish()?;
        Ok(out)
    }
}

impl Kolibri {
    pub fn load(
        path: &Path,
        context: usize,
        max_batch: usize,
        budget: Option<u64>,
    ) -> Result<Self> {
        if context == 0 || context > MAX_CONTEXT || max_batch == 0 || max_batch > 64 {
            return Err(MetalError::Model(
                "Kolibri context must be 1..=262144, batch 1..=64".into(),
            ));
        }
        let source = Source {
            config: KolibriConfig::read(path).map_err(MetalError::Model)?,
            map: ShardedSafetensors::open_dir(path)
                .map_err(|e| MetalError::Model(e.to_string()))?,
        };
        // Inventory the ENTIRE graph before GPU allocation. No missing tensor,
        // quantization override or unexpected tensor is silently ignored.
        let mut matrices = vec![
            ("model.embed_tokens".to_owned(), vec![VOCAB, WIDTH], 8),
            ("lm_head".to_owned(), vec![VOCAB, WIDTH], 8),
        ];
        let mut small = vec![("model.norm.weight".to_owned(), vec![WIDTH], StDtype::Bf16)];
        for i in 0..LAYERS {
            let root = format!("model.layers.{i}");
            for norm in [
                "input_layernorm",
                "post_attn_norm",
                "post_attention_layernorm",
                "post_ffn_norm",
            ] {
                small.push((format!("{root}.{norm}.weight"), vec![WIDTH], StDtype::Bf16));
            }
            for norm in ["q_norm", "k_norm"] {
                small.push((
                    format!("{root}.self_attn.{norm}.weight"),
                    vec![HEAD_DIM],
                    StDtype::Bf16,
                ));
            }
            small.push((
                format!("{root}.mlp.gate.weight"),
                vec![EXPERTS, WIDTH],
                StDtype::Bf16,
            ));
            small.push((
                format!("{root}.mlp.expert_bias"),
                vec![EXPERTS],
                StDtype::F32,
            ));
            for (name, n, k) in [
                ("self_attn.q_proj", HEADS * HEAD_DIM, WIDTH),
                ("self_attn.k_proj", KVWIDTH, WIDTH),
                ("self_attn.v_proj", KVWIDTH, WIDTH),
                ("self_attn.o_proj", WIDTH, HEADS * HEAD_DIM),
                ("mlp.shared_experts.gate_proj", FF, WIDTH),
                ("mlp.shared_experts.up_proj", FF, WIDTH),
                ("mlp.shared_experts.down_proj", WIDTH, FF),
            ] {
                matrices.push((format!("{root}.{name}"), vec![n, k], 8));
            }
            for (name, n, k) in [
                ("gate_proj", FF, WIDTH),
                ("up_proj", FF, WIDTH),
                ("down_proj", WIDTH, FF),
            ] {
                matrices.push((
                    format!("{root}.mlp.switch_mlp.{name}"),
                    vec![EXPERTS, n, k],
                    4,
                ));
            }
        }
        let mut expected = HashSet::new();
        let mut weight_bytes = 0u64;
        for (name, shape, bits) in &matrices {
            for (key, size) in source.parts(name, shape, *bits)? {
                weight_bytes += size as u64;
                expected.insert(key);
            }
        }
        for (name, shape, dtype) in &small {
            source.check(name, shape, *dtype)?;
            weight_bytes += shape.iter().product::<usize>() as u64 * 4;
            expected.insert(name.clone());
        }
        if source.map.names().any(|n| !expected.contains(n)) {
            return Err(MetalError::Model(
                "Kolibri checkpoint contains unexpected tensors".into(),
            ));
        }
        let page_stride = context.div_ceil(BLOCK_TOKENS);
        let blocks = (max_batch + 1) * page_stride;
        let kv_layer = blocks * BLOCK_TOKENS * KVWIDTH * 2;
        let kv_bytes = (kv_layer * 2 * LAYERS) as u64;
        let tile_cap = (CHUNK * ACTIVE).div_ceil(16) + EXPERTS;
        let sizes = [
            CHUNK * 4,
            CHUNK * 8,
            max_batch * page_stride * 4,
            max_batch * 4,
            CHUNK * 4,
            CHUNK * 8,
            CHUNK * WIDTH * 4,
            CHUNK * WIDTH * 4,
            CHUNK * HEADS * HEAD_DIM * 4,
            CHUNK * KVWIDTH * 4,
            CHUNK * KVWIDTH * 4,
            CHUNK * HEADS * HEAD_DIM * 4,
            CHUNK * HEADS * SPLITS * (HEAD_DIM + 2) * 4,
            CHUNK * WIDTH * 4,
            CHUNK * WIDTH * 4,
            CHUNK * FF * 4,
            CHUNK * FF * 4,
            CHUNK * EXPERTS * 4,
            CHUNK * ACTIVE * 4,
            CHUNK * ACTIVE * 4,
            EXPERTS * CHUNK * ACTIVE * 4,
            EXPERTS * 4,
            (1 + 2 * tile_cap) * 4,
            CHUNK * ACTIVE * FF * 4,
            CHUNK * ACTIVE * FF * 4,
            CHUNK * ACTIVE * WIDTH * 4,
            max_batch * VOCAB * 4,
        ];
        let device = MetalDevice::new(budget)?;
        // Include the largest transient BF16-to-F32 small tensor conversion.
        let required = weight_bytes
            + kv_bytes
            + sizes.iter().sum::<usize>() as u64
            + (EXPERTS * WIDTH * 2) as u64;
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "Kolibri weights/paged KV/scratch require {required} bytes; grant {}",
                device.budget_bytes()
            )));
        }
        let embedding = source.matrix(&device, "model.embed_tokens", &[VOCAB, WIDTH], 8)?;
        let head = source.matrix(&device, "lm_head", &[VOCAB, WIDTH], 8)?;
        let output_norm = source.small(&device, "model.norm.weight", &[WIDTH], StDtype::Bf16)?;
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            let root = format!("model.layers.{i}");
            let s = |name: &str, shape: &[usize], dtype| {
                source.small(&device, &format!("{root}.{name}"), shape, dtype)
            };
            let w =
                |name: &str, n, k| source.matrix(&device, &format!("{root}.{name}"), &[n, k], 8);
            let e = |name: &str, n, k| {
                source.matrix(
                    &device,
                    &format!("{root}.mlp.switch_mlp.{name}"),
                    &[EXPERTS, n, k],
                    4,
                )
            };
            layers.push(Layer {
                norm: s("input_layernorm.weight", &[WIDTH], StDtype::Bf16)?,
                post_attn: s("post_attn_norm.weight", &[WIDTH], StDtype::Bf16)?,
                pre_ffn: s("post_attention_layernorm.weight", &[WIDTH], StDtype::Bf16)?,
                post_ffn: s("post_ffn_norm.weight", &[WIDTH], StDtype::Bf16)?,
                qnorm: s("self_attn.q_norm.weight", &[HEAD_DIM], StDtype::Bf16)?,
                knorm: s("self_attn.k_norm.weight", &[HEAD_DIM], StDtype::Bf16)?,
                q: w("self_attn.q_proj", HEADS * HEAD_DIM, WIDTH)?,
                k: w("self_attn.k_proj", KVWIDTH, WIDTH)?,
                v: w("self_attn.v_proj", KVWIDTH, WIDTH)?,
                o: w("self_attn.o_proj", WIDTH, HEADS * HEAD_DIM)?,
                router: s("mlp.gate.weight", &[EXPERTS, WIDTH], StDtype::Bf16)?,
                bias: s("mlp.expert_bias", &[EXPERTS], StDtype::F32)?,
                gate: e("gate_proj", FF, WIDTH)?,
                up: e("up_proj", FF, WIDTH)?,
                down: e("down_proj", WIDTH, FF)?,
                shared_gate: w("mlp.shared_experts.gate_proj", FF, WIDTH)?,
                shared_up: w("mlp.shared_experts.up_proj", FF, WIDTH)?,
                shared_down: w("mlp.shared_experts.down_proj", WIDTH, FF)?,
                keys: device.alloc(kv_layer)?,
                values: device.alloc(kv_layer)?,
            });
        }
        drop(source);
        let mut sizes = sizes.into_iter();
        let mut a = || device.alloc(sizes.next().expect("scratch layout"));
        let scratch = Scratch {
            ids: a()?,
            meta: a()?,
            pages: a()?,
            output_rows: a()?,
            decode_rows: a()?,
            attention_tiles: a()?,
            x: a()?,
            norm: a()?,
            q: a()?,
            k: a()?,
            v: a()?,
            attn: a()?,
            parts: a()?,
            delta: a()?,
            normalized: a()?,
            fg: a()?,
            fu: a()?,
            router: a()?,
            picks: a()?,
            probabilities: a()?,
            lists: a()?,
            counts: a()?,
            tiles: a()?,
            gate: a()?,
            up: a()?,
            expert_out: a()?,
            logits: a()?,
        };
        assert!(sizes.next().is_none());
        tracing::info!(
            weight_bytes,
            kv_bytes,
            allocated = device.allocated_bytes(),
            context,
            max_batch,
            "Kolibri native Metal mixed affine4/8, paged BF16 KV"
        );
        Ok(Self {
            device,
            embedding,
            output_norm,
            head,
            layers,
            scratch,
            slots: (0..max_batch).map(|_| Slot::default()).collect(),
            pending: VecDeque::new(),
            pool: KvPool::with_blocks(blocks as u32),
            radix: PagedRadix::new(),
            context,
            page_stride,
            weight_bytes,
            kv_bytes,
            last_gpu_seconds: 0.,
        })
    }
}
