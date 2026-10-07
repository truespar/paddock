//! Granite weight load + geometry. Every metadata key below was verified in
//! a real `granite-4.1-8b-Q8_0.gguf` dump; anything absent or
//! off-spec is a hard, named error - never a silent default that changes math.
//! The four Granite scalars in particular are required: defaulting a missing
//! multiplier to 1.0 would produce fluent, wrong output instead of a failure.

use std::sync::Arc;

use paddock_kernels::reference::ops::YarnRope;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuExecutor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;

use super::*;

impl GpuGranite {
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        Self::load_with(exec, map, max_ctx, None)
    }

    /// `fp8_native_dir` is accepted for constructor parity with the other
    /// families; granite has no fp8-native lane (ibm-granite ships an
    /// `-fp8` safetensors repo, which is a later item).
    pub fn load_with(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
        _fp8_native_dir: Option<&std::path::Path>,
    ) -> Result<Self, GpuModelError> {
        exec.vram_load_gate(map.total_len(), "granite")
            .map_err(GpuModelError::WontFit)?;
        // Single-stream engine: cudarc's cross-stream event tracking is pure
        // overhead and blocks CUDA-graph capture. Must precede all allocs.
        exec.disable_event_tracking();

        let u = |k: &str| {
            map.gguf()
                .arch_field(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let f_req = |k: &str| {
            map.gguf()
                .arch_field(k)
                .and_then(Value::as_f32)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let f_opt = |k: &str| map.gguf().arch_field(k).and_then(Value::as_f32);

        let n_layer = u("block_count")? as usize;
        let n_embd = u("embedding_length")? as usize;
        let n_heads = u("attention.head_count")? as usize;
        let n_kv_heads = u("attention.head_count_kv")? as usize;
        let n_ff = u("feed_forward_length")? as usize;
        let ctx_train = u("context_length")? as usize;
        let eps = f_req("attention.layer_norm_rms_epsilon")?;

        // Granite does not stamp attention.key_length/value_length (laguna and
        // qwen35 both read those), so head_dim is derived. rope.dimension_count
        // is the independent witness: granite is full-rotary, so they must
        // agree - if they don't, the file is not the geometry we think it is.
        if n_heads == 0 || !n_embd.is_multiple_of(n_heads) {
            return Err(GpuModelError::Unsupported(format!(
                "granite: embedding_length {n_embd} not divisible by head_count {n_heads}"
            )));
        }
        let head_dim = n_embd / n_heads;
        let n_rot = u("rope.dimension_count")? as usize;
        if n_rot != head_dim {
            return Err(GpuModelError::Unsupported(format!(
                "granite: rope.dimension_count {n_rot} != head_dim {head_dim} - partial \
                 rotary is not a shipped granite geometry"
            )));
        }
        if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
            return Err(GpuModelError::Unsupported(format!(
                "granite: head_count {n_heads} not a multiple of head_count_kv {n_kv_heads}"
            )));
        }

        // The four Granite scalars. attention.scale replaces 1/sqrt(head_dim)
        // as the KQ scale - on the 8b it is 0.0078125 = 1/128 = 1/head_dim,
        // not 1/sqrt(128) ≈ 0.0884. On a granite file all four are required
        // rather than defaulted: a missing multiplier silently degrades output.
        //
        // A plain `llama` file (MiniCPM5-2B is the first one served) is the
        // same stack with every multiplier at identity - that is literally how
        // llama.cpp defines granite, as llama plus the four scalars - so the
        // arch string decides which rule applies. The llama arm is just as
        // strict in its own direction: a llama file that stamps any of the
        // granite keys is not a geometry we know, and refuses rather than
        // having the value dropped on the floor.
        let arch = map.gguf().architecture().unwrap_or("granite");
        let (embedding_scale, residual_scale, logit_scale, attention_scale) = if arch == "llama" {
            for k in [
                "embedding_scale",
                "residual_scale",
                "logit_scale",
                "attention.scale",
            ] {
                if map.gguf().arch_field(k).is_some() {
                    return Err(GpuModelError::Unsupported(format!(
                        "llama: file stamps llama.{k}, which no llama graph applies - refusing \
                         rather than ignoring a scalar the author thought mattered"
                    )));
                }
            }
            (1.0, 1.0, 1.0, 1.0 / (head_dim as f32).sqrt())
        } else {
            (
                f_req("embedding_scale")?,
                f_req("residual_scale")?,
                f_req("logit_scale")?,
                f_req("attention.scale")?,
            )
        };
        if logit_scale == 0.0 {
            return Err(GpuModelError::Unsupported(
                "granite: logit_scale 0 would divide the logits by zero".into(),
            ));
        }
        // Biases are a llama-arch option (Qwen1.5/2-era exports carry
        // attn_q/k/v.bias under this arch string) that the granite graph has no
        // seat for. Refuse on presence: loading the weights and skipping the
        // bias is fluent, wrong text.
        if map.tensor_info("blk.0.attn_q.bias").is_some()
            || map.tensor_info("blk.0.attn_k.bias").is_some()
            || map.tensor_info("blk.0.attn_v.bias").is_some()
        {
            return Err(GpuModelError::Unsupported(format!(
                "{arch}: attention biases present - this graph serves the bias-free llama stack \
                 only"
            )));
        }

        // DeepStack: one entry per LLM layer, holding which vision stream to
        // add into the image rows before that layer runs, or -1 for none. Read
        // from the file rather than derived - on granite-vision-4.1-4b it comes
        // out as layer 3->1, 6->2, 9->3, 12->4, 15->5, 18->6, 21->7, which happens to
        // be 3·k but is a property of this checkpoint, not of the architecture.
        //
        // Stream 0 is absent from the mapping deliberately: it is not injected at
        // a layer, it is the image's input embedding (see `embed_rows`). That
        // is the same thing as "inject at layer 0 into zeroed slots", which is
        // how upstream's modeling.py words it.
        //
        // Absent on text-only granite checkpoints, which is the normal case.
        let deepstack: Vec<i32> = match map.gguf().arch_field("deepstack_mapping") {
            Some(paddock_models::gguf::Value::Array(items)) => {
                let v: Vec<i32> = items
                    .iter()
                    .map(|x| x.as_i64().map(|i| i as i32))
                    .collect::<Option<_>>()
                    .ok_or_else(|| {
                        GpuModelError::Unsupported(
                            "granite: deepstack_mapping holds a non-integer entry".into(),
                        )
                    })?;
                if v.len() != n_layer {
                    return Err(GpuModelError::Unsupported(format!(
                        "granite: deepstack_mapping has {} entries for {n_layer} layers - it is \
                         indexed BY layer, so a length mismatch means the wrong convention",
                        v.len()
                    )));
                }
                v
            }
            Some(_) => {
                return Err(GpuModelError::Unsupported(
                    "granite: deepstack_mapping is present but not an array".into(),
                ));
            }
            None => vec![-1; n_layer],
        };

        // The multimodal placeholder ids: the tokens the chat template renders
        // once per picture / per clip, and the tokens the mm prefill lane fills
        // those rows with before the tower's features overwrite them. Read,
        // never hard-coded - the numbers (100266 `<image>` on
        // granite-vision-4.1-4b, matching config.json's `image_token_index`;
        // 100352 `<|audio|>` on granite-speech-4.1-2b, matching its
        // processor_config.json) are properties of the vocab file.
        //
        // Both resolved in one pass, and unconditionally: `deepstack_mapping`
        // marks a vision checkpoint, but a granite-SPEECH text model is
        // header-identical to a text-only one (all -1) - its companion mmproj
        // is the only thing that says "audio", and that file is not open yet.
        // So there is nothing to gate on; one walk over the vocab resolves
        // what is there and leaves the rest None.
        let (img_pad_id, audio_pad_id) = match map.gguf().metadata.get("tokenizer.ggml.tokens") {
            Some(paddock_models::gguf::Value::Array(toks)) => {
                let (mut img, mut aud) = (None, None);
                for (i, t) in toks.iter().enumerate() {
                    match t.as_str() {
                        Some("<image>") => img = Some(i as u32),
                        Some("<|audio|>") => aud = Some(i as u32),
                        _ => {}
                    }
                }
                (img, aud)
            }
            _ => (None, None),
        };

        // Plain rope: ext_factor 0 collapses YarnRope's ramp, freq_scale 1.
        // Granite ships no rope-scaling keys; if a future file does, it must
        // be handled explicitly rather than silently ignored here.
        // `rope_freqs.weight` is the llama-arch spelling of the same thing:
        // Llama 3.1+ ships its per-frequency factors as a tensor (the
        // converter stamps no scaling key for them), and a graph that rotates
        // without them is right for short prompts and silently wrong past the
        // original window.
        if map.gguf().arch_field("rope.scaling.type").is_some()
            || f_opt("rope.scaling.factor").is_some_and(|v| v != 1.0)
            || map.tensor_info("rope_freqs.weight").is_some()
        {
            return Err(GpuModelError::Unsupported(format!(
                "{arch}: rope scaling is stamped but unhandled - refusing to ignore it"
            )));
        }
        let rope_base = f_req("rope.freq_base")?;
        let rope =
            YarnRope::new(n_rot, rope_base, 1.0, ctx_train, 0.0, 1.0, 32.0, 1.0).kernel_params();

        // With no rope scaling there is nothing extending granite past its
        // trained window: positions beyond it are raw extrapolation, which
        // degrades output without erroring. Refuse rather than serve quietly
        // wrong long-context results (no-silent-failures).
        if max_ctx > ctx_train {
            return Err(GpuModelError::Unsupported(format!(
                "granite: requested context {max_ctx} exceeds the trained {ctx_train} and the \
                 file stamps no rope scaling - serving past it would extrapolate silently"
            )));
        }

        // Per-component VRAM ledger - every line sums RESIDENT BUFFER LENGTHS
        // (the "show what's on GPU" principle needs the number to be true, and
        // free-VRAM deltas are not: see the total below).
        let gb = |used: u64| used as f64 / 1e9;

        let te_ty = map
            .tensor_info("token_embd.weight")
            .map(|t| t.ggml_type)
            .ok_or_else(|| GpuModelError::MissingMeta("token_embd.weight".into()))?;
        let (tok_embd, n_vocab) = if crate::gpu::kq_params(te_ty).is_some() {
            let t = exec.repack_kquant(map, "token_embd.weight")?;
            let vocab = t.dims[1];
            (TokEmbd::Kq(t), vocab)
        } else {
            let t = exec.upload_raw(map, "token_embd.weight")?;
            if t.ty != paddock_models::ggml_type::GgmlType::Q8_0 {
                return Err(GpuModelError::Unsupported(format!(
                    "token_embd.weight quant {:?} has no resident gather path",
                    t.ty
                )));
            }
            let vocab = t.dims[1];
            (TokEmbd::Q8(t), vocab)
        };
        tracing::info!(
            "granite VRAM  input embeddings token_embd          {:>7.2} GB",
            gb(tok_embd.resident_bytes() as u64)
        );

        let mut layers = Vec::with_capacity(n_layer);
        let mut attn_bytes = 0u64;
        let mut ffn_bytes = 0u64;
        for i in 0..n_layer {
            let dt = |name: &str| exec.upload(map, &format!("blk.{i}.{name}"));
            let qt = |name: &str| exec.load_quantw(map, &format!("blk.{i}.{name}"));

            // Per-group bytes are summed from the RESIDENT BUFFERS, not from
            // free-VRAM deltas: allocations come from a stream-ordered pool,
            // so mem_get_info differences track pool growth and attribute
            // whole blocks to whichever tensor triggered them. Bracketing the
            // groups that way reported 0.23 GB of attention against 8.46 GB of
            // FFN on the 8b, when the shapes say ~1.8 / ~6.7 - a correct total
            // hiding a meaningless split. The pool total below is still taken
            // from mem_get_info, which is the one thing it does measure.
            let attn_norm = dt("attn_norm.weight")?;
            let wq = qt("attn_q.weight")?;
            let wk = qt("attn_k.weight")?;
            let wv = qt("attn_v.weight")?;
            let wo = qt("attn_output.weight")?;
            attn_bytes +=
                attn_norm.buf.len() as u64 * 4 + wq.bytes() + wk.bytes() + wv.bytes() + wo.bytes();

            let ffn_norm = dt("ffn_norm.weight")?;
            let gate = qt("ffn_gate.weight")?;
            let up = qt("ffn_up.weight")?;
            let down = qt("ffn_down.weight")?;
            ffn_bytes += ffn_norm.buf.len() as u64 * 4 + gate.bytes() + up.bytes() + down.bytes();

            // The GGUF lane is uniformly the quant class; the checkpoint
            // classes only arrive through `load_dir`.
            let layer = GraniteLayer {
                qkv_f8: None,
                qkv_nv4: None,
                attn_norm,
                wq: GraniteW::Quant(wq),
                wk: GraniteW::Quant(wk),
                wv: GraniteW::Quant(wv),
                wo: GraniteW::Quant(wo),
                ffn_norm,
                // The GGUF lane never merges gate|up: the merge is exact only
                // because nvfp4's two per-tensor scales happen to match, and a
                // k-quant plane has no such scalar to compare.
                gate: Some(GraniteW::Quant(gate)),
                up: Some(GraniteW::Quant(up)),
                gate_up: None,
                down: GraniteW::Quant(down),
            };

            // Shape audit on the signature tensors - refuse a file whose
            // projections disagree with the header rather than serve it.
            let qd = layer.wq.dims();
            if qd[0] != n_embd || qd[1] != n_heads * head_dim {
                return Err(GpuModelError::Unsupported(format!(
                    "granite blk.{i}: attn_q {:?} != [{n_embd}, {}]",
                    qd,
                    n_heads * head_dim
                )));
            }
            let kd = layer.wk.dims();
            if kd[1] != n_kv_heads * head_dim {
                return Err(GpuModelError::Unsupported(format!(
                    "granite blk.{i}: attn_k out {} != {}×{head_dim}",
                    kd[1], n_kv_heads
                )));
            }
            let gd = layer
                .gate
                .as_ref()
                .expect("GGUF lane is always split")
                .dims();
            if gd[1] != n_ff {
                return Err(GpuModelError::Unsupported(format!(
                    "granite blk.{i}: ffn_gate out {} != feed_forward_length {n_ff}",
                    gd[1]
                )));
            }
            layers.push(layer);
        }
        tracing::info!(
            "granite VRAM  attention (q/k/v/o)                  {:>7.2} GB",
            gb(attn_bytes)
        );
        tracing::info!(
            "granite VRAM  dense FFN (gate/up/down)             {:>7.2} GB",
            gb(ffn_bytes)
        );

        let output_norm = exec.upload(map, "output_norm.weight")?;
        // Granite 4.1 ships a real `output.weight` despite tie_word_embeddings
        // in the HF config; tied exports omit it and llama.cpp duplicates
        // token_embd instead. Branch on PRESENCE, never on "the load failed" -
        // catching the error would turn an unsupported-quant head into a
        // silent fall back to the embedding plane, i.e. a different model that
        // still generates text.
        let lm_head = if map.tensor_info("output.weight").is_some() {
            GraniteW::Quant(exec.load_quantw(map, "output.weight")?)
        } else {
            tracing::info!("granite: no output.weight - tied head, using token_embd");
            GraniteW::Quant(exec.load_quantw(map, "token_embd.weight")?)
        };
        let head_bytes = output_norm.buf.len() as u64 * 4 + lm_head.bytes();
        tracing::info!(
            "granite VRAM  output head + final norm             {:>7.2} GB",
            gb(head_bytes)
        );

        // Every group above is summed from RESIDENT BUFFER LENGTHS, so the
        // total is their sum - not a free-VRAM delta across the whole load.
        // Measured on the 30b Q4_K_M: the delta read 20.94 GB where
        // the buffers hold 18.71, because each repack's staging block stays in
        // the stream-ordered pool until a trim and the bracket counts it twice.
        // That 2.2 GB of phantom is what `weights_mem_bytes` was reporting to
        // will-it-fit and the Studio, and it is exactly the kind of number the
        // no-silent-failures principle says must be honest. Trim afterwards so
        // any later free-VRAM read (pool sizing) sees the holes returned -
        // qwen35 and gemma4 already end their loads this way; granite did not,
        // which is why only granite's ledger drifted.
        let weights_bytes = tok_embd.resident_bytes() as u64 + attn_bytes + ffn_bytes + head_bytes;
        exec.trim_mem_pool();
        tracing::info!(
            "granite VRAM  = model resident total               {:>7.2} GB  \
             ({n_layer} layers, heads {n_heads}/{n_kv_heads}×{head_dim}, ff {n_ff}, \
             rope θ {rope_base:.0}, scales e{embedding_scale} r{residual_scale} \
             l{logit_scale} a{attention_scale})",
            gb(weights_bytes),
        );

        Ok(Self {
            exec,
            hp: Hparams {
                n_layer,
                n_embd,
                n_heads,
                n_kv_heads,
                head_dim,
                n_ff,
                n_vocab,
                eps,
                rope,
                embedding_scale,
                residual_scale,
                logit_scale,
                attention_scale,
                deepstack,
            },
            tok_embd,
            layers,
            output_norm,
            lm_head,
            max_ctx,
            weights_bytes,
            content_id: (
                crate::kv_tier::fingerprint::weights(map),
                crate::kv_tier::fingerprint::tokenizer(map),
            ),
            // KV8 (fp8-e4m3) by default since 2026-10-04 - the class this
            // family's boards and rivals run, with fp8 arms on every serving
            // path; a die that cannot store fp8 gets f16 back (serving.rs)
            // and an explicit kv_cache_dtype wins
            kv_dtype: KvDtype::Fp8E4m3,
            decode: None,
            scratch: None,
            batch: None,
            chunked: Vec::new(),
            last_reused: Vec::new(),
            seal_hist: Vec::new(),
            seal_ok: Vec::new(),
            vision: None,
            audio: None,
            img_pad_id,
            audio_pad_id,
            media: Default::default(),
            img_cache: Vec::new(),
            img_cache_bytes: 0,
            img_cache_clock: 0,
            img_cache_reused: 0,
            pipe: None,
            enc: std::collections::VecDeque::new(),
        })
    }
}
