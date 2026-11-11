//! Whisper loader - every plane resident at the file's own f16, hparams and
//! the decode contract from our converter's metadata.
//! Our whisper converter is the only writer of this schema,
//! so a missing key means a wrong or stale file - never a tolerable default.
//! The one deliberate default is the LayerNorm epsilon: the HF config
//! declares none (torch nn.LayerNorm's 1e-5 is the trained value).

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuExecutor, HalfTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::vision::host_f32;

use super::{
    AttnP, DecLayer, DecodeTokens, EncLayer, GpuWhisper, Hparams, LayerNormP, MelSpec, MlpP,
    SelfAttnP,
};

fn dt(exec: &GpuExecutor, map: &MappedGguf, name: &str) -> Result<HalfTensor, GpuModelError> {
    Ok(exec.upload_f16(map, name)?)
}

/// Bias/LN vectors ride to device as f32 - the file stores f16, and widening
/// a [1280]-class vector is noise next to the f32-accumulate graph that
/// consumes it (the audio-tower chassis this family inherits).
fn vec1(exec: &GpuExecutor, map: &MappedGguf, name: &str) -> Result<CudaSlice<f32>, GpuModelError> {
    let (host, _) = host_f32(map, name)?;
    Ok(exec.to_device(&host)?)
}

/// Load a conv1d plane with its k-axis permuted from the file's
/// (out, in_channel, kpos) order to (out, kpos, in_channel) - i.e. GGML dims
/// [3, cin, cout] become a plain [3*cin, cout] GEMM plane whose k index is
/// `kpos*cin + c`.
///
/// That is the order the im2col side wants: activations are frame-major, so
/// one tap position is a CONTIGUOUS cin-run of the input, and the gather (or
/// the host copy, for conv1) emits three such runs per output row with no
/// per-element striding. Same trick the qwen3_asr tower plays on its conv
/// stem; doing it once at load costs nothing per request.
fn conv1d(exec: &GpuExecutor, map: &MappedGguf, name: &str) -> Result<HalfTensor, GpuModelError> {
    let (w, dims) = host_f32(map, name)?;
    if dims.len() != 3 || dims[0] != 3 {
        return Err(GpuModelError::Unsupported(format!(
            "whisper: {name} dims {dims:?} - expected a [3, cin, cout] k3 conv plane"
        )));
    }
    let (cin, cout) = (dims[1], dims[2]);
    let k = 3 * cin;
    let mut out = vec![0f32; k * cout];
    for o in 0..cout {
        for c in 0..cin {
            for kp in 0..3 {
                out[o * k + kp * cin + c] = w[o * k + c * 3 + kp];
            }
        }
    }
    Ok(HalfTensor {
        buf: exec.to_device_f16(&out, name)?,
        dims: vec![k, cout],
    })
}

fn ln(exec: &GpuExecutor, map: &MappedGguf, prefix: &str) -> Result<LayerNormP, GpuModelError> {
    Ok(LayerNormP {
        w: vec1(exec, map, &format!("{prefix}.weight"))?,
        b: vec1(exec, map, &format!("{prefix}.bias"))?,
    })
}

/// A cross-attention block's per-layer planes: `proj` names the projection
/// group ("decoder.layers.0.encoder_attn"), `ln_name` its pre-LN. Only q and
/// o load here - the cross wk/wv/bv are consumed by [`cross_kv_all`] into
/// the model-level layer-batched planes.
fn attn(
    exec: &GpuExecutor,
    map: &MappedGguf,
    proj: &str,
    ln_name: &str,
) -> Result<AttnP, GpuModelError> {
    Ok(AttnP {
        ln: ln(exec, map, ln_name)?,
        wq: dt(exec, map, &format!("{proj}.q_proj.weight"))?,
        bq: vec1(exec, map, &format!("{proj}.q_proj.bias"))?,
        wo: dt(exec, map, &format!("{proj}.out_proj.weight"))?,
        bo: vec1(exec, map, &format!("{proj}.out_proj.bias"))?,
    })
}

/// Every decoder layer's cross-attention K and V projections, concatenated
/// on the out axis into two `[d, n_layer*d]` planes plus the matching
/// `[n_layer*d]` v-bias. All 2*n_layer of these GEMMs read the
/// same encoder states at admission, and separately each is too small to
/// fill the machine - batched, the whole set runs as two GEMM calls
/// (806us -> 270 per window measured). Layer li owns rows
/// `[li*d, (li+1)*d)`. Source planes are dropped as they are copied - same
/// footprint rule as [`self_attn`]. k_proj has no bias, by architecture.
fn cross_kv_all(
    exec: &GpuExecutor,
    map: &MappedGguf,
    n_layer: usize,
    d: usize,
) -> Result<(HalfTensor, HalfTensor, CudaSlice<f32>), GpuModelError> {
    let n = d * d;
    let mut kall = exec.alloc_f16(n_layer * n)?;
    let mut vall = exec.alloc_f16(n_layer * n)?;
    let mut bv_host = vec![0.0f32; n_layer * d];
    for i in 0..n_layer {
        let proj = format!("decoder.layers.{i}.encoder_attn");
        let wk = dt(exec, map, &format!("{proj}.k_proj.weight"))?;
        let wv = dt(exec, map, &format!("{proj}.v_proj.weight"))?;
        expect_dims(&format!("{proj}.k_proj.weight"), &wk.dims, &[d, d])?;
        expect_dims(&format!("{proj}.v_proj.weight"), &wv.dims, &[d, d])?;
        exec.copy_region(&wk.buf, 0, &mut kall, i * n, n)?;
        exec.copy_region(&wv.buf, 0, &mut vall, i * n, n)?;
        let (bv, bv_dims) = host_f32(map, &format!("{proj}.v_proj.bias"))?;
        expect_dims(&format!("{proj}.v_proj.bias"), &bv_dims, &[d])?;
        bv_host[i * d..(i + 1) * d].copy_from_slice(&bv);
    }
    Ok((
        HalfTensor {
            buf: kall,
            dims: vec![d, n_layer * d],
        },
        HalfTensor {
            buf: vall,
            dims: vec![d, n_layer * d],
        },
        exec.to_device(&bv_host)?,
    ))
}

/// The decoder's self-attention, with q|k|v merged into one out-axis plane
/// (see [`SelfAttnP`]). GGUF weight planes are row-major over out_dim - one
/// output row is a contiguous in_dim run - so the concatenation is three
/// device copies and the result is exactly what a `[d, 3d]` GEMM expects.
/// The three source planes are dropped at the end of this function: keeping
/// them would cost another 315 MB on a 3 GB model for nothing.
fn self_attn(
    exec: &GpuExecutor,
    map: &MappedGguf,
    proj: &str,
    ln_name: &str,
) -> Result<SelfAttnP, GpuModelError> {
    let wq = dt(exec, map, &format!("{proj}.q_proj.weight"))?;
    let wk = dt(exec, map, &format!("{proj}.k_proj.weight"))?;
    let wv = dt(exec, map, &format!("{proj}.v_proj.weight"))?;
    let (d_in, d_out) = (wq.dims[0], wq.dims[1]);
    if wk.dims != wq.dims || wv.dims != wq.dims {
        return Err(GpuModelError::Unsupported(format!(
            "whisper: {proj} q/k/v dims disagree ({:?} / {:?} / {:?}) - cannot merge",
            wq.dims, wk.dims, wv.dims
        )));
    }
    let n = d_in * d_out;
    let mut buf = exec.alloc_f16(3 * n)?;
    for (i, src) in [&wq, &wk, &wv].iter().enumerate() {
        exec.copy_region(&src.buf, 0, &mut buf, i * n, n)?;
    }
    Ok(SelfAttnP {
        ln: ln(exec, map, ln_name)?,
        wqkv: HalfTensor {
            buf,
            dims: vec![d_in, 3 * d_out],
        },
        bq: vec1(exec, map, &format!("{proj}.q_proj.bias"))?,
        bv: vec1(exec, map, &format!("{proj}.v_proj.bias"))?,
        wo: dt(exec, map, &format!("{proj}.out_proj.weight"))?,
        bo: vec1(exec, map, &format!("{proj}.out_proj.bias"))?,
    })
}

fn mlp(exec: &GpuExecutor, map: &MappedGguf, layer: &str) -> Result<MlpP, GpuModelError> {
    Ok(MlpP {
        ln: ln(exec, map, &format!("{layer}.final_layer_norm"))?,
        fc1_w: dt(exec, map, &format!("{layer}.fc1.weight"))?,
        fc1_b: vec1(exec, map, &format!("{layer}.fc1.bias"))?,
        fc2_w: dt(exec, map, &format!("{layer}.fc2.weight"))?,
        fc2_b: vec1(exec, map, &format!("{layer}.fc2.bias"))?,
    })
}

/// Parse the checkpoint's `lang_to_id` map - `{"<|af|>": 50327, ...}`, the
/// flat string->int object our converter copies out of generation_config.json
/// - into (bare code, token id) pairs.
///
/// Hand-parsed rather than pulling a JSON crate into the engine: this is our
/// own writer's output, one flat object with no nesting, escapes or floats,
/// and the engine's dependency minimalism is a stated principle. Anything
/// that does not match that exact shape is a wrong or hand-edited file and
/// says so loudly instead of silently yielding a short language list.
fn parse_lang_map(json: &str) -> Result<Vec<(String, u32)>, GpuModelError> {
    let body = json
        .trim()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| {
            GpuModelError::Unsupported("whisper: lang_to_id_json is not a JSON object".into())
        })?;
    let mut out = Vec::new();
    for entry in body.split(',').filter(|e| !e.trim().is_empty()) {
        let (key, val) = entry.split_once(':').ok_or_else(|| {
            GpuModelError::Unsupported(format!("whisper: lang_to_id entry {entry:?} has no value"))
        })?;
        // keys are whisper's language TOKENS ("<|sv|>"); callers speak bare
        // codes, so strip the marker here rather than at every call site
        let code = key
            .trim()
            .trim_matches('"')
            .trim_start_matches("<|")
            .trim_end_matches("|>")
            .to_owned();
        let id: u32 = val.trim().parse().map_err(|_| {
            GpuModelError::Unsupported(format!(
                "whisper: lang_to_id {code} value {val:?} is not an id"
            ))
        })?;
        out.push((code, id));
    }
    if out.is_empty() {
        return Err(GpuModelError::Unsupported(
            "whisper: lang_to_id map is empty - language forcing and detection would both fail"
                .into(),
        ));
    }
    Ok(out)
}

/// Loud geometry check - a plane whose dims disagree with the declared
/// hparams is a corrupt or foreign conversion, caught at load, not mid-graph.
fn expect_dims(name: &str, got: &[usize], want: &[usize]) -> Result<(), GpuModelError> {
    if got != want {
        return Err(GpuModelError::Unsupported(format!(
            "whisper: {name} dims {got:?}, expected {want:?} - not a conversion this loader knows"
        )));
    }
    Ok(())
}

impl GpuWhisper {
    /// Conservative serving envelope before any weight upload. Includes both
    /// KV pools, encoder batch workspaces, 32-row alignment scratch and selected
    /// attention heads. The executor adds driver/allocator slack separately.
    pub fn residency_bytes(map: &MappedGguf, max_ctx: usize, slots: usize) -> Result<u64, String> {
        let u = |key: &str| {
            map.gguf()
                .arch_field(key)
                .and_then(Value::as_u64)
                .filter(|&n| n > 0 && n <= 1_000_000)
                .map(u128::from)
                .ok_or_else(|| format!("invalid Whisper memory-plan field {key}"))
        };
        if !(1..=1024).contains(&slots) {
            return Err("Whisper slots must be 1..1024".into());
        }
        let d = u("d_model")?;
        let n = u("max_source_positions")?;
        let ctx = u("max_target_positions")?.min(max_ctx as u128);
        let layers = u("decoder.layer_count")?;
        let heads = u("decoder.head_count")?;
        let vocab = u("vocab_size")?;
        let bins = u("mel.bins")?;
        let ff = u("encoder.ffn_length")?.max(u("decoder.ffn_length")?);
        let b = super::encoder::enc_batch_env() as u128;
        let cap = slots as u128;
        // FP16 is an upper bound even when explicitly using FP8 KV.
        let kv = layers * cap * (n + ctx) * d * 4;
        let encoder = (b * 2 * n + 1) * bins * 4
            + b * 9 * n * 4
            + b * (6 * n * bins).max(3 * n * d) * 4
            + (b * 2 * n + 1) * d * 4
            + b * (6 * n * bins).max(3 * n * d).max(n * ff) * 2
            + b * (11 * n * d * 4 + n * ff * 4 + n * d * 2);
        let decoder =
            cap.max(32) * (26 * d + 6 * ff + 4 * vocab + heads * 32 * (d / heads + 2) * 4 + 52);
        let cross_landing = b * n * d * (2 + layers * 4);
        let selection = super::align::heads_for(
            layers as usize,
            heads as usize,
            bins as usize,
            vocab as usize,
        )
        .unwrap_or_else(|| super::align::fallback_heads(layers as usize, heads as usize));
        let alignment = selection.heads.len() as u128 * ((ctx + 1) * n * 4 + 4);
        // The F32 positional and 1-D norm/bias planes are wider than the file.
        let mut widening = 0u128;
        let mut staging = 0u128;
        for tensor in &map.gguf().tensors {
            let (_, bytes) = map.tensor_bytes(&tensor.name).map_err(|e| e.to_string())?;
            staging = staging.max(bytes.len() as u128);
            if tensor.dims.len() == 1 || tensor.name.ends_with("embed_positions.weight") {
                widening += bytes.len() as u128 * 2;
            }
        }
        (map.total_len() as u128
            + widening
            + staging
            + kv
            + encoder
            + decoder
            + cross_landing
            + alignment)
            .try_into()
            .map_err(|_| "Whisper memory plan overflow".into())
    }

    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        exec.vram_load_gate(map.total_len(), "whisper")
            .map_err(GpuModelError::WontFit)?;
        // Single-stream engine, same stance as every family: cudarc's
        // cross-stream event tracking is pure overhead and blocks CUDA-graph
        // capture. Must precede all allocs.
        exec.disable_event_tracking();

        let u = |k: &str| {
            map.gguf()
                .arch_field(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| GpuModelError::MissingMeta(format!("whisper.{k}")))
        };

        let d_model = u("d_model")?;
        let n_enc_layer = u("encoder.layer_count")?;
        let n_enc_heads = u("encoder.head_count")?;
        let enc_ffn = u("encoder.ffn_length")?;
        let n_dec_layer = u("decoder.layer_count")?;
        let n_dec_heads = u("decoder.head_count")?;
        let dec_ffn = u("decoder.ffn_length")?;
        let n_vocab = u("vocab_size")?;
        let n_audio_ctx = u("max_source_positions")?;
        let n_text_ctx_trained = u("max_target_positions")?;
        if n_enc_heads == 0
            || n_dec_heads == 0
            || d_model % n_enc_heads != 0
            || d_model % n_dec_heads != 0
        {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: d_model {d_model} not divisible by heads (enc {n_enc_heads}, dec {n_dec_heads})"
            )));
        }
        let head_dim = d_model / n_enc_heads;
        if d_model / n_dec_heads != head_dim {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: encoder head_dim {head_dim} != decoder head_dim {} - one attention \
                 geometry per family",
                d_model / n_dec_heads
            )));
        }

        // The decoder's position table is LEARNED and finite (448 trained
        // rows) - unlike an LLM's rope ctx there is nothing a bigger request
        // could mean, so a larger --max-ctx caps at the trained table rather
        // than erroring (the default serve ctx would otherwise never load a
        // whisper). A SMALLER --max-ctx narrows the served table and is worth
        // saying out loud.
        let n_text_ctx = n_text_ctx_trained.min(max_ctx.max(1));
        if n_text_ctx < n_text_ctx_trained {
            tracing::info!(
                requested = max_ctx,
                trained = n_text_ctx_trained,
                "whisper: decoder context narrowed below the trained position table"
            );
        }

        let mel = MelSpec {
            bins: u("mel.bins")?,
            n_fft: u("mel.n_fft")?,
            hop: u("mel.hop_length")?,
            chunk_s: u("mel.chunk_length_s")?,
            sr: u("mel.sampling_rate")?,
        };
        let tok = |k: &str| {
            u(k).and_then(|v| {
                u32::try_from(v).map_err(|_| GpuModelError::MissingMeta(format!("whisper.{k}")))
            })
        };
        // `<|nospeech|>` and `<|0.00|>` sit either side of `<|notimestamps|>`
        // in every whisper vocabulary - the special-token block is ordered
        // `... <|nospeech|> <|notimestamps|> <|0.00|> <|0.02|> ... <|30.00|>`,
        // which is also how whisper.cpp resolves them (`token_beg =
        // token_not + 1`). The converter stamps both explicitly now; the
        // neighbour derivation is the fallback for files written before it
        // did, and it is a layout, not a guess.
        let no_timestamps = tok("token.no_timestamps")?;
        let tokens = DecodeTokens {
            sot: tok("token.sot")?,
            eot: tok("token.eot")?,
            no_timestamps,
            transcribe: tok("token.transcribe")?,
            translate: tok("token.translate").ok(),
            nospeech: tok("token.nospeech").unwrap_or(no_timestamps.saturating_sub(1)),
            timestamp_begin: tok("token.timestamp_begin").unwrap_or(no_timestamps + 1),
            // same block, same rule: `... <|startofprev|> <|nospeech|>
            // <|notimestamps|> ...`, so two below `<|notimestamps|>`. Stamped
            // explicitly by the converter now; the neighbour derivation is
            // the fallback for files written before it was.
            sot_prev: tok("token.sot_prev").unwrap_or(no_timestamps.saturating_sub(2)),
            // `<|startoflm|>` opens the same block: `<|startoflm|>
            // <|startofprev|> <|nospeech|> <|notimestamps|>`, three below.
            sot_lm: tok("token.sot_lm").unwrap_or(no_timestamps.saturating_sub(3)),
        };
        // A timestamp id past the head's width would index off the end of a
        // logits row, so check it here rather than at the first transcript.
        if tokens.timestamp_begin as usize >= n_vocab {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: timestamp_begin {} is outside the {n_vocab}-token vocabulary",
                tokens.timestamp_begin
            )));
        }
        let langs = parse_lang_map(
            map.gguf()
                .arch_field("lang_to_id_json")
                .and_then(Value::as_str)
                .ok_or_else(|| GpuModelError::MissingMeta("whisper.lang_to_id_json".into()))?,
        )?;

        // ---- encoder ----
        let conv1_w = conv1d(&exec, map, "encoder.conv1.weight")?;
        expect_dims(
            "encoder.conv1.weight",
            &conv1_w.dims,
            &[3 * mel.bins, d_model],
        )?;
        let conv1_b = vec1(&exec, map, "encoder.conv1.bias")?;
        let conv2_w = conv1d(&exec, map, "encoder.conv2.weight")?;
        expect_dims(
            "encoder.conv2.weight",
            &conv2_w.dims,
            &[3 * d_model, d_model],
        )?;
        let conv2_b = vec1(&exec, map, "encoder.conv2.bias")?;
        let (pos_host, pos_dims) = host_f32(map, "encoder.embed_positions.weight")?;
        expect_dims(
            "encoder.embed_positions.weight",
            &pos_dims,
            &[d_model, n_audio_ctx],
        )?;
        let enc_pos = exec.to_device(&pos_host)?;

        let mut enc_layers = Vec::with_capacity(n_enc_layer);
        for i in 0..n_enc_layer {
            let l = format!("encoder.layers.{i}");
            enc_layers.push(EncLayer {
                // merged q|k|v, same as the decoder: three
                // separate 1280-out GEMMs at 1500 rows leave half the tc5p
                // clusters idle - fused they run 2x faster than the sum
                attn: self_attn(
                    &exec,
                    map,
                    &format!("{l}.self_attn"),
                    &format!("{l}.self_attn_layer_norm"),
                )?,
                mlp: mlp(&exec, map, &l)?,
            });
        }
        let enc_ln = ln(&exec, map, "encoder.layer_norm")?;

        // ---- decoder ----
        let tok_embd = exec.upload(map, "decoder.embed_tokens.weight")?;
        expect_dims(
            "decoder.embed_tokens.weight",
            &tok_embd.dims,
            &[d_model, n_vocab],
        )?;
        // learned positions: upload only the rows actually served (prefix of
        // the row-major table when --max-ctx narrowed)
        let (dpos_host, dpos_dims) = host_f32(map, "decoder.embed_positions.weight")?;
        expect_dims(
            "decoder.embed_positions.weight",
            &dpos_dims,
            &[d_model, n_text_ctx_trained],
        )?;
        let dec_pos = exec.to_device(&dpos_host[..n_text_ctx * d_model])?;

        let mut dec_layers = Vec::with_capacity(n_dec_layer);
        for i in 0..n_dec_layer {
            let l = format!("decoder.layers.{i}");
            dec_layers.push(DecLayer {
                self_attn: self_attn(
                    &exec,
                    map,
                    &format!("{l}.self_attn"),
                    &format!("{l}.self_attn_layer_norm"),
                )?,
                cross_attn: attn(
                    &exec,
                    map,
                    &format!("{l}.encoder_attn"),
                    &format!("{l}.encoder_attn_layer_norm"),
                )?,
                mlp: mlp(&exec, map, &l)?,
            });
        }
        let (cross_wk_all, cross_wv_all, cross_bv_all) =
            cross_kv_all(&exec, map, n_dec_layer, d_model)?;
        let dec_ln = ln(&exec, map, "decoder.layer_norm")?;
        // whisper ties proj_out to the token embedding - always, in every
        // checkpoint. KB-Whisper materializes the duplicate plane (verified
        // bit-identical to embed_tokens); NB-Whisper and Røst just omit it.
        // Reading the embedding plane a second time is the tie, so the head
        // costs f16 bytes either way and no file needs a synthetic copy.
        let head_name = if map.tensor_info("proj_out.weight").is_some() {
            "proj_out.weight"
        } else {
            "decoder.embed_tokens.weight"
        };
        let head = dt(&exec, map, head_name)?;
        expect_dims(head_name, &head.dims, &[d_model, n_vocab])?;

        // OpenAI's `max_initial_timestamp`, 1.0 s, in this checkpoint's own
        // timestamp steps (one step = two mel hops, the conv stem's 2x time
        // downsample). Derived rather than the usual hardcoded 50, because a
        // fine-tune that moved the hop would move the step with it.
        let ts_step = 2.0 * mel.hop as f32 / mel.sr as f32;
        let max_initial_ts = if ts_step > 0.0 {
            (1.0 / ts_step).round() as u32
        } else {
            u32::MAX
        };

        let mut me = Self {
            exec,
            hp: Hparams {
                d_model,
                n_enc_heads,
                enc_ffn,
                n_dec_heads,
                dec_ffn,
                head_dim,
                n_vocab,
                n_audio_ctx,
                n_text_ctx,
                // torch nn.LayerNorm default - the value the checkpoints
                // trained with (HF WhisperConfig declares no eps field)
                eps: 1e-5,
            },
            mel,
            tokens,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            enc_pos,
            enc_layers,
            enc_ln,
            tok_embd,
            dec_pos,
            dec_layers,
            cross_wk_all,
            cross_wv_all,
            cross_bv_all,
            dec_ln,
            head,
            langs,
            // fp8-e4m3 is the DEFAULT. Whisper's cross planes
            // are a full 1500-frame window per slot per layer, static and
            // never shorter, so they dominate both the wall (that one kernel
            // was 27% of all GPU time at c32) and the pool (304 -> 152 MiB
            // per slot). Measured: c32 throughput up, VRAM 13.8 -> 8.9 GiB
            // at 32 slots, and transcripts BIT-IDENTICAL to f16 across the
            // whole Nordic battery (WER 2.09/7.33/7.02% either way).
            // `--kv-cache-dtype f16` restores the exact class.
            kv_dtype: crate::gpu::KvDtype::Fp8E4m3,
            max_initial_ts,
            enc: None,
            decode: None,
            weights_bytes: 0,
        };
        me.weights_bytes = me.resident_bytes();
        tracing::info!(
            enc_layers = n_enc_layer,
            dec_layers = n_dec_layer,
            vocab = n_vocab,
            audio_ctx = n_audio_ctx,
            text_ctx = n_text_ctx,
            weight_mib = me.weights_bytes / (1 << 20),
            "whisper resident at f16 (f32 accumulate; encoder graph, decoder+serving)"
        );
        Ok(me)
    }
}
