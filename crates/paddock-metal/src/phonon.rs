//! Native FlowLM + Mimi speech synthesis. No host neural-network execution,
//! decoder subprocess, imported inference engine or per-frame GPU allocation.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use paddock_engine::phonon::{Backend, Noise, Options};
use paddock_models::phonon::{
    CODEC_CACHE, Checkpoint, FLOW_CACHE, FRAME_SAMPLES, MAX_FRAMES, MAX_TEXT_TOKENS,
};
use paddock_tokenizer::GgufTokenizer;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

const CODEC_BATCH: usize = 8;

fn err(e: impl ToString) -> MetalError {
    MetalError::Model(format!("Phonon: {}", e.to_string()))
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn write(b: &Buffer, v: &[f32]) {
    unsafe { b.write_u32(&v.iter().map(|v| v.to_bits()).collect::<Vec<_>>()) }
}
struct Cache {
    k: Buffer,
    v: Buffer,
    capacity: usize,
}
struct Conv {
    weight: Buffer,
    bias: Buffer,
    history: Buffer,
    input: usize,
    output: usize,
    width: usize,
    stride: usize,
}
struct Workspace {
    x: Buffer,
    n: Buffer,
    tmp: Buffer,
    wide: Buffer,
    q: Buffer,
    att: Buffer,
    att_partials: Buffer,
    staged: Buffer,
    skip: Buffer,
    flow: Buffer,
    h: Buffer,
    cond: Buffer,
    modulation: Buffer,
    latent: Buffer,
    noise: Buffer,
    eos: Buffer,
    projected: Buffer,
    noises: Buffer,
    latents: Buffer,
    endings: Buffer,
}
impl Workspace {
    fn new(d: &MetalDevice) -> Result<Self> {
        let a = || d.alloc(FRAME_SAMPLES * CODEC_BATCH * 64 * 4);
        let small = |n: usize| d.alloc(n * 4);
        Ok(Self {
            x: a()?,
            n: a()?,
            tmp: a()?,
            wide: a()?,
            q: small(CODEC_BATCH * 16 * 512)?,
            att: small(CODEC_BATCH * 16 * 512)?,
            att_partials: small(8 * FLOW_CACHE.div_ceil(64) * 66)?,
            staged: small(FRAME_SAMPLES * CODEC_BATCH * 64 * 3)?,
            skip: a()?,
            flow: small(512)?,
            h: small(512)?,
            cond: small(512)?,
            modulation: small(6 * 1536 + 1024)?,
            latent: small(32)?,
            noise: small(32)?,
            eos: small(1)?,
            projected: small(CODEC_BATCH * 512)?,
            noises: small(CODEC_BATCH * 32)?,
            latents: small(CODEC_BATCH * 32)?,
            endings: small(CODEC_BATCH)?,
        })
    }
}
pub struct Phonon {
    device: MetalDevice,
    weights: BTreeMap<String, Buffer>,
    bf16_weights: BTreeSet<String>,
    ws: Workspace,
    flow_cache: Vec<Cache>,
    voice_cache: Vec<Cache>,
    codec_cache: Vec<Cache>,
    convs: Vec<Conv>,
    up_history: Buffer,
    condition: Buffer,
    time: Buffer,
    voice: Vec<f32>,
    bos: Vec<f32>,
    embeddings: Vec<f32>,
    tokenizer: GgufTokenizer,
    flow_position: usize,
    codec_position: usize,
}
impl Phonon {
    pub fn memory_bytes(&self) -> (u64, u64) {
        let weights = self.weights.values().map(|b| b.len() as u64).sum::<u64>()
            + self
                .convs
                .iter()
                .map(|c| (c.weight.len() + c.bias.len()) as u64)
                .sum::<u64>();
        (
            weights,
            self.device.allocated_bytes().saturating_sub(weights),
        )
    }
    pub fn load(path: &Path, budget: Option<u64>) -> Result<Self> {
        let mut source = Checkpoint::open(path).map_err(err)?;
        let tokenizer = GgufTokenizer::from_hf_dir(path).map_err(err)?;
        // The tokenizer has 4000 entries; row 4000 of the embedding matrix
        // is conditioner padding, not an additional text token.
        if tokenizer.vocab_size != 4000 {
            return Err(err(
                "tokenizer must have the checkpoint's 4000-token vocabulary",
            ));
        }
        let expanded: usize = source.tensors.values().map(|x| x.len() * 4).sum();
        let d = MetalDevice::new_pack(
            budget,
            expanded as u64 + 100 * 1024 * 1024,
            include_str!("../../../packs/metal/phonon.metal"),
            &[
                "phonon_mv",
                "phonon_norm_mv",
                "phonon_mm",
                "phonon_mm32",
                "phonon_mm_tensor",
                "phonon_copy",
                "phonon_denormalize",
                "phonon_point",
                "phonon_norm",
                "phonon_rope",
                "phonon_attention",
                "phonon_attention_split",
                "phonon_attention_join",
                "phonon_conv_input",
                "phonon_history",
                "phonon_transpose_input",
                "phonon_upsample",
            ],
        )?;
        let embeddings = source
            .tensors
            .remove("flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight")
            .expect("tensor checked by the Phonon checkpoint contract");
        let mut convs = Vec::new();
        for (p, input, output, width, stride) in [
            ("mimi.decoder.model.0.conv", 512, 512, 7, 1),
            ("mimi.decoder.model.2.convtr", 512, 256, 12, 6),
            ("mimi.decoder.model.3.block.1.conv", 256, 128, 3, 1),
            ("mimi.decoder.model.3.block.3.conv", 128, 256, 1, 1),
            ("mimi.decoder.model.5.convtr", 256, 128, 10, 5),
            ("mimi.decoder.model.6.block.1.conv", 128, 64, 3, 1),
            ("mimi.decoder.model.6.block.3.conv", 64, 128, 1, 1),
            ("mimi.decoder.model.8.convtr", 128, 64, 8, 4),
            ("mimi.decoder.model.9.block.1.conv", 64, 32, 3, 1),
            ("mimi.decoder.model.9.block.3.conv", 32, 64, 1, 1),
            ("mimi.decoder.model.11.conv", 64, 1, 3, 1),
        ] {
            let w = source
                .tensors
                .remove(&format!("{p}.weight"))
                .expect("tensor checked by the Phonon checkpoint contract");
            let b = source
                .tensors
                .remove(&format!("{p}.bias"))
                .expect("tensor checked by the Phonon checkpoint contract");
            let (w, b) = if stride > 1 {
                let mut packed = vec![0.; w.len()];
                let mut bias = Vec::with_capacity(output * stride);
                for phase in 0..stride {
                    bias.extend_from_slice(&b);
                    for o in 0..output {
                        for i in 0..input {
                            packed[(phase * output + o) * input * 2 + i] =
                                w[(i * output + o) * width + phase + stride];
                            packed[(phase * output + o) * input * 2 + input + i] =
                                w[(i * output + o) * width + phase];
                        }
                    }
                }
                (packed, bias)
            } else {
                (w, b)
            };
            let history = d.alloc(input * if stride > 1 { 1 } else { width - 1 }.max(1) * 4)?;
            convs.push(Conv {
                weight: d.upload(&bytes(&w))?,
                bias: d.upload(&bytes(&b))?,
                history,
                input,
                output,
                width,
                stride,
            });
        }
        // Every residual block consumes the same conditioning vector. Pack
        // those independent rows without changing their products/reductions.
        let mut modulation_weights = Vec::new();
        let mut modulation_bias = Vec::new();
        for i in 0..7 {
            let name = if i < 6 {
                format!("flow_lm.flow_net.res_blocks.{i}.adaLN_modulation.1")
            } else {
                "flow_lm.flow_net.final_layer.adaLN_modulation.1".into()
            };
            modulation_weights.extend(
                source
                    .tensors
                    .remove(&format!("{name}.weight"))
                    .expect("validated modulation"),
            );
            modulation_bias.extend(
                source
                    .tensors
                    .remove(&format!("{name}.bias"))
                    .expect("validated modulation"),
            );
        }
        source.tensors.insert(
            "flow_lm.flow_net.modulation.weight".into(),
            modulation_weights,
        );
        source
            .tensors
            .insert("flow_lm.flow_net.modulation.bias".into(), modulation_bias);
        let mut weights = BTreeMap::new();
        let mut bf16_weights = BTreeSet::new();
        for (n, v) in source.tensors {
            // Preserve exact original BF16 matrix coefficients in half the
            // bandwidth. Activations, products, reductions and Mimi stay F32.
            // F32-only tensors are never rounded to make them fit this route.
            if n.starts_with("flow_lm.")
                && n.ends_with(".weight")
                && v.len() >= 8192
                && v.iter().all(|x| x.to_bits() & 0xffff == 0)
            {
                let packed: Vec<u8> = v
                    .iter()
                    .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                    .collect();
                bf16_weights.insert(n.clone());
                weights.insert(n, d.upload(&packed)?);
            } else {
                weights.insert(n, d.upload(&bytes(&v))?);
            }
        }
        let cache = |capacity| -> Result<Cache> {
            Ok(Cache {
                k: d.alloc(capacity * 512 * 4)?,
                v: d.alloc(capacity * 512 * 4)?,
                capacity,
            })
        };
        let flow_cache = (0..6).map(|_| cache(FLOW_CACHE)).collect::<Result<_>>()?;
        let voice_cache = (0..6)
            .map(|_| cache(source.voice_frames))
            .collect::<Result<_>>()?;
        let codec_cache = (0..2).map(|_| cache(CODEC_CACHE)).collect::<Result<_>>()?;
        let ws = Workspace::new(&d)?;
        let up_history = d.alloc(512 * 4)?;
        let condition = d.alloc(512 * 4)?;
        let time = d.alloc(512 * 4)?;
        let mut m = Self {
            device: d,
            weights,
            bf16_weights,
            ws,
            flow_cache,
            voice_cache,
            codec_cache,
            convs,
            up_history,
            condition,
            time,
            voice: Vec::new(),
            bos: Vec::new(),
            embeddings,
            tokenizer,
            flow_position: 0,
            codec_position: 0,
        };
        m.prepare_constants(&source.voice, source.voice_frames)?;
        m.cache_voice()?;
        tracing::info!(
            allocated_bytes = m.device.allocated_bytes(),
            "native Phonon FlowLM and Mimi ready on Metal"
        );
        Ok(m)
    }
    fn w(&self, n: &str) -> &Buffer {
        &self.weights[n]
    }
    fn mat(
        &self,
        c: &Commands<'_>,
        w: &Buffer,
        b: Option<&Buffer>,
        x: &Buffer,
        y: &Buffer,
        k: usize,
        n: usize,
        rows: usize,
        epilogue: u32,
        packed: bool,
    ) {
        // Keep the 32-wide latent projection on the original SIMD reduction
        // regardless of frame count, so batching cannot change this contract.
        let tiled = rows >= 8 && k > 32;
        let large = tiled && n >= 32 && (rows >= 16 || self.device.tensor_accelerated());
        let kernel = if large && self.device.tensor_accelerated() {
            "phonon_mm_tensor"
        } else if large {
            "phonon_mm32"
        } else if tiled {
            "phonon_mm"
        } else {
            "phonon_mv"
        };
        c.dispatch(
            kernel,
            &[w, x, b.unwrap_or(w), y],
            &[
                k as u32,
                n as u32,
                rows as u32,
                u32::from(b.is_some()),
                epilogue,
                u32::from(packed),
            ],
            if large {
                [n.div_ceil(64), rows.div_ceil(32), 1]
            } else if tiled {
                [n.div_ceil(32), rows.div_ceil(8), 1]
            } else {
                [n.div_ceil(4), rows, 1]
            },
            128,
        );
    }
    fn copy(
        &self,
        c: &Commands<'_>,
        src: &Buffer,
        src_at: usize,
        dst: &Buffer,
        dst_at: usize,
        len: usize,
    ) {
        c.dispatch_at(
            "phonon_copy",
            &[src, dst],
            &[src_at * 4, dst_at * 4],
            &[len as u32],
            [len.div_ceil(256), 1, 1],
            256,
        );
    }
    fn cache_voice(&mut self) -> Result<()> {
        let mut pos = 0;
        for chunk in self.voice.chunks(64 * 512) {
            write(&self.ws.x, chunk);
            let c = self.device.begin()?;
            self.transformer(&c, false, chunk.len() / 512, pos);
            c.finish()?;
            pos += chunk.len() / 512;
        }
        let c = self.device.begin()?;
        for (src, dst) in self.flow_cache.iter().zip(&self.voice_cache) {
            self.copy(&c, &src.k, 0, &dst.k, 0, pos * 512);
            self.copy(&c, &src.v, 0, &dst.v, 0, pos * 512);
        }
        c.finish()?;
        Ok(())
    }
    fn linear(
        &self,
        c: &Commands<'_>,
        p: &str,
        x: &Buffer,
        y: &Buffer,
        k: usize,
        n: usize,
        rows: usize,
    ) {
        self.linear_epilogue(c, p, x, y, k, n, rows, 0);
    }
    fn linear_epilogue(
        &self,
        c: &Commands<'_>,
        p: &str,
        x: &Buffer,
        y: &Buffer,
        k: usize,
        n: usize,
        rows: usize,
        epilogue: u32,
    ) {
        self.mat(
            c,
            self.w(&format!("{p}.weight")),
            self.weights.get(&format!("{p}.bias")),
            x,
            y,
            k,
            n,
            rows,
            epilogue,
            self.bf16_weights.contains(&format!("{p}.weight")),
        );
    }
    fn norm_linear(
        &self,
        c: &Commands<'_>,
        norm: &str,
        matrix: &str,
        x: &Buffer,
        y: &Buffer,
        n: usize,
        eps: f32,
        modulation: Option<usize>,
        epilogue: u32,
    ) {
        let weight = format!("{matrix}.weight");
        let bias = self.weights.get(&format!("{matrix}.bias"));
        let affine = !norm.is_empty();
        c.dispatch(
            "phonon_norm_mv",
            &[
                self.w(&weight),
                x,
                if affine {
                    self.w(&format!("{norm}.weight"))
                } else {
                    x
                },
                if affine {
                    self.w(&format!("{norm}.bias"))
                } else {
                    x
                },
                &self.ws.modulation,
                bias.unwrap_or(x),
                y,
            ],
            &[
                n as u32,
                u32::from(self.bf16_weights.contains(&weight)),
                u32::from(bias.is_some()),
                eps.to_bits(),
                modulation.unwrap_or(0) as u32,
                u32::from(modulation.is_some()),
                u32::from(affine),
                epilogue,
            ],
            [n.div_ceil(4), 1, 1],
            128,
        );
    }
    fn point(
        &self,
        c: &Commands<'_>,
        x: &Buffer,
        z: &Buffer,
        y: &Buffer,
        n: usize,
        mode: u32,
        width: usize,
        scale: f32,
    ) {
        c.dispatch(
            "phonon_point",
            &[x, z, y],
            &[n as u32, mode, width as u32, scale.to_bits()],
            [n.div_ceil(256), 1, 1],
            256,
        );
    }
    fn norm(
        &self,
        c: &Commands<'_>,
        p: &str,
        x: &Buffer,
        y: &Buffer,
        rows: usize,
        mode: u32,
        eps: f32,
    ) {
        let (w, b) = if mode == 2 {
            (x, x)
        } else if mode == 1 {
            (self.w(p), x)
        } else {
            (self.w(&format!("{p}.weight")), self.w(&format!("{p}.bias")))
        };
        c.dispatch(
            "phonon_norm",
            &[x, w, b, y],
            &[512, eps.to_bits(), mode],
            [rows, 1, 1],
            128,
        );
    }
    fn prepare_constants(&mut self, voice: &[f32], frames: usize) -> Result<()> {
        // These conditions do not depend on the utterance. Compute once on GPU.
        let d = &self.device;
        let speakers = d.alloc(16 * 4)?;
        let duration = d.upload(&bytes(&[vec![1.; 64], vec![0.; 64]].concat()))?;
        let values = unsafe {
            self.w("flow_lm.condition_provider.conditioners.num_speakers.embed.weight")
                .read_f32(16, 16)
        };
        write(&speakers, &values);
        let voice_in = d.upload(&bytes(voice))?;
        let voice_out = d.alloc(voice.len() * 4)?;
        let bos = d.alloc(512 * 4)?;
        let mut times = Vec::new();
        for i in 0..2 {
            let f = unsafe {
                self.w(&format!("flow_lm.flow_net.time_embed.{i}.freqs"))
                    .read_f32(0, 128)
            };
            let v: Vec<f32> = f
                .iter()
                .map(|f| (*f * i as f32).cos())
                .chain(f.iter().map(|f| (*f * i as f32).sin()))
                .collect();
            times.push(d.upload(&bytes(&v))?);
        }
        let c = d.begin()?;
        self.linear(
            &c,
            "flow_lm.condition_provider.conditioners.num_speakers.output_proj",
            &speakers,
            &self.condition,
            16,
            512,
            1,
        );
        self.linear(
            &c,
            "flow_lm.condition_provider.conditioners.duration_delta.output_proj",
            &duration,
            &self.ws.tmp,
            128,
            512,
            1,
        );
        self.point(
            &c,
            &self.ws.tmp,
            &self.condition,
            &self.condition,
            512,
            1,
            512,
            1.,
        );
        self.linear(
            &c,
            "flow_lm.input_linear",
            self.w("flow_lm.bos_emb"),
            &bos,
            32,
            512,
            1,
        );
        self.point(&c, &bos, &self.condition, &bos, 512, 1, 512, 1.);
        self.linear(
            &c,
            "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj",
            &voice_in,
            &voice_out,
            512,
            512,
            frames,
        );
        for (i, t) in times.iter().enumerate() {
            let p = format!("flow_lm.flow_net.time_embed.{i}");
            self.linear_epilogue(&c, &format!("{p}.mlp.0"), t, &self.ws.tmp, 256, 512, 1, 3);
            self.linear(
                &c,
                &format!("{p}.mlp.2"),
                &self.ws.tmp,
                &self.ws.n,
                512,
                512,
                1,
            );
            self.norm(
                &c,
                &format!("{p}.mlp.3.alpha"),
                &self.ws.n,
                if i == 0 { &self.time } else { &self.ws.tmp },
                1,
                1,
                1e-5,
            );
        }
        self.point(&c, &self.time, &self.ws.tmp, &self.time, 512, 1, 512, 1.);
        // half the sum; point mode 6 uses zero as the additive input.
        let zero = d.upload(&bytes(&[0.; 512]))?;
        self.point(&c, &self.time, &zero, &self.time, 512, 6, 512, 0.5);
        c.finish()?;
        self.voice = unsafe { voice_out.read_f32(0, voice.len()) };
        self.bos = unsafe { bos.read_f32(0, 512) };
        Ok(())
    }
    fn transformer(&self, c: &Commands<'_>, codec: bool, rows: usize, pos: usize) {
        let layers = if codec {
            &self.codec_cache
        } else {
            &self.flow_cache
        };
        let prefix = if codec {
            "mimi.decoder_transformer.transformer"
        } else {
            "flow_lm.transformer"
        };
        let w = &self.ws;
        for (i, cache) in layers.iter().enumerate() {
            let p = format!("{prefix}.layers.{i}");
            if rows == 1 {
                self.norm_linear(
                    c,
                    &format!("{p}.norm1"),
                    &format!("{p}.self_attn.in_proj"),
                    &w.x,
                    &w.wide,
                    1536,
                    1e-5,
                    None,
                    0,
                );
            } else {
                self.norm(c, &format!("{p}.norm1"), &w.x, &w.n, rows, 0, 1e-5);
                self.linear(
                    c,
                    &format!("{p}.self_attn.in_proj"),
                    &w.n,
                    &w.wide,
                    512,
                    1536,
                    rows,
                );
            }
            c.dispatch(
                "phonon_rope",
                &[&w.wide, &w.q, &cache.k, &cache.v],
                &[rows as u32, pos as u32, cache.capacity as u32],
                [rows, 1, 1],
                256,
            );
            if !codec && rows == 1 {
                let parts = (pos + 1).div_ceil(64);
                c.dispatch(
                    "phonon_attention_split",
                    &[&w.q, &cache.k, &cache.v, &w.att_partials],
                    &[pos as u32, parts as u32],
                    [8, parts, 1],
                    128,
                );
                c.dispatch(
                    "phonon_attention_join",
                    &[&w.att_partials, &w.att],
                    &[parts as u32],
                    [8, 1, 1],
                    64,
                );
            } else {
                c.dispatch(
                    "phonon_attention",
                    &[&w.q, &cache.k, &cache.v, &w.att],
                    &[
                        pos as u32,
                        cache.capacity as u32,
                        if codec { 251 } else { FLOW_CACHE as u32 },
                    ],
                    [8, rows, 1],
                    128,
                );
            }
            self.linear_epilogue(
                c,
                &format!("{p}.self_attn.out_proj"),
                &w.att,
                if codec { &w.tmp } else { &w.x },
                512,
                512,
                rows,
                if codec { 0 } else { 1 },
            );
            if codec {
                self.point(
                    c,
                    &w.tmp,
                    self.w(&format!("{p}.layer_scale_1.scale")),
                    &w.x,
                    rows * 512,
                    5,
                    512,
                    1.,
                );
            }
            if rows == 1 {
                self.norm_linear(
                    c,
                    &format!("{p}.norm2"),
                    &format!("{p}.linear1"),
                    &w.x,
                    &w.wide,
                    2048,
                    1e-5,
                    None,
                    2,
                );
            } else {
                self.norm(c, &format!("{p}.norm2"), &w.x, &w.n, rows, 0, 1e-5);
                self.linear_epilogue(
                    c,
                    &format!("{p}.linear1"),
                    &w.n,
                    &w.wide,
                    512,
                    2048,
                    rows,
                    2,
                );
            }
            self.linear_epilogue(
                c,
                &format!("{p}.linear2"),
                &w.wide,
                if codec { &w.tmp } else { &w.x },
                2048,
                512,
                rows,
                if codec { 0 } else { 1 },
            );
            if codec {
                self.point(
                    c,
                    &w.tmp,
                    self.w(&format!("{p}.layer_scale_2.scale")),
                    &w.x,
                    rows * 512,
                    5,
                    512,
                    1.,
                );
            }
        }
        if !codec {
            self.norm(c, "flow_lm.out_norm", &w.x, &w.n, rows, 0, 1e-5);
            c.dispatch_at(
                "phonon_point",
                &[&w.n, &w.n, &w.h],
                &[(rows - 1) * 512 * 4, 0, 0],
                &[512, 0, 512, 0],
                [2, 1, 1],
                256,
            );
        }
    }
    fn reset(&mut self, text: &str) -> Result<()> {
        let ids = self.tokenizer.encode(text.trim()).map_err(err)?;
        if ids.is_empty() || ids.len() > MAX_TEXT_TOKENS || ids.iter().any(|&id| id >= 4001) {
            return Err(err(format!(
                "input must tokenize to 1-{MAX_TEXT_TOKENS} text tokens"
            )));
        }
        let mut prefix = Vec::with_capacity((ids.len() + 1) * 512);
        for id in ids {
            prefix.extend_from_slice(&self.embeddings[id as usize * 512..(id as usize + 1) * 512]);
        }
        prefix.extend_from_slice(&self.bos);
        self.flow_position = self.voice.len() / 512;
        self.codec_position = 0;
        write(&self.up_history, &[0.; 512]);
        for conv in &self.convs {
            write(&conv.history, &vec![0.; conv.history.len() / 4]);
        }
        for (i, chunk) in prefix.chunks(CODEC_BATCH * 16 * 512).enumerate() {
            write(&self.ws.x, chunk);
            let rows = chunk.len() / 512;
            let c = self.device.begin()?;
            if i == 0 {
                for (src, dst) in self.voice_cache.iter().zip(&self.flow_cache) {
                    self.copy(&c, &src.k, 0, &dst.k, 0, self.voice.len());
                    self.copy(&c, &src.v, 0, &dst.v, 0, self.voice.len());
                }
            }
            self.transformer(&c, false, rows, self.flow_position);
            c.finish()?;
            self.flow_position += rows;
        }
        Ok(())
    }
    fn flow(&self, c: &Commands<'_>) {
        let w = &self.ws;
        let p = "flow_lm.flow_net";
        self.linear(c, "flow_lm.out_eos", &w.h, &w.eos, 512, 1, 1);
        self.linear(c, &format!("{p}.cond_embed"), &w.h, &w.cond, 512, 512, 1);
        self.point(c, &w.cond, &self.time, &w.cond, 512, 9, 512, 1.);
        self.linear(
            c,
            &format!("{p}.modulation"),
            &w.cond,
            &w.modulation,
            512,
            6 * 1536 + 1024,
            1,
        );
        self.linear(c, &format!("{p}.input_proj"), &w.noise, &w.flow, 32, 512, 1);
        for i in 0..6 {
            let p = format!("{p}.res_blocks.{i}");
            self.norm_linear(
                c,
                &format!("{p}.in_ln"),
                &format!("{p}.mlp.0"),
                &w.flow,
                &w.tmp,
                512,
                1e-6,
                Some(i * 1536),
                3,
            );
            self.linear(c, &format!("{p}.mlp.2"), &w.tmp, &w.n, 512, 512, 1);
            c.dispatch_at(
                "phonon_point",
                &[&w.n, &w.modulation, &w.flow],
                &[0, (i * 1536 + 1024) * 4, 0],
                &[512, 5, 512, 0],
                [2, 1, 1],
                256,
            );
        }
        self.norm_linear(
            c,
            "",
            &format!("{p}.final_layer.linear"),
            &w.flow,
            &w.latent,
            32,
            1e-6,
            Some(6 * 1536),
            0,
        );
        self.point(c, &w.latent, &w.noise, &w.latent, 32, 1, 32, 1.);
    }
    fn conv(&self, c: &Commands<'_>, i: usize, rows: usize) {
        let v = &self.convs[i];
        let w = &self.ws;
        let (k, n) = if v.stride > 1 {
            c.dispatch(
                "phonon_transpose_input",
                &[&w.x, &v.history, &w.staged],
                &[v.input as u32, rows as u32],
                [(rows * v.input * 2).div_ceil(256), 1, 1],
                256,
            );
            (v.input * 2, v.output * v.stride)
        } else {
            c.dispatch(
                "phonon_conv_input",
                &[&w.x, &v.history, &w.staged],
                &[v.input as u32, v.width as u32, rows as u32],
                [(rows * v.input * v.width).div_ceil(256), 1, 1],
                256,
            );
            (v.input * v.width, v.output)
        };
        let history = if v.stride > 1 { 1 } else { v.width - 1 };
        if history > 0 {
            c.dispatch(
                "phonon_history",
                &[&w.x, &v.history],
                &[v.input as u32, history as u32, rows as u32],
                [(v.input * history).div_ceil(256), 1, 1],
                256,
            );
        }
        self.mat(
            c,
            &v.weight,
            Some(&v.bias),
            &w.staged,
            &w.x,
            k,
            n,
            rows,
            0,
            false,
        );
    }
    fn codec(&self, c: &Commands<'_>, frames: usize, latents: &Buffer) {
        let w = &self.ws;
        c.dispatch(
            "phonon_denormalize",
            &[
                latents,
                self.w("flow_lm.emb_std"),
                self.w("flow_lm.emb_mean"),
                &w.n,
            ],
            &[(frames * 32) as u32],
            [(frames * 32).div_ceil(256), 1, 1],
            256,
        );
        self.mat(
            c,
            self.w("mimi.quantizer.output_proj.weight"),
            None,
            &w.n,
            &w.projected,
            32,
            512,
            frames,
            0,
            false,
        );
        c.dispatch(
            "phonon_upsample",
            &[
                &w.projected,
                &self.up_history,
                self.w("mimi.upsample.convtr.convtr.weight"),
                &w.x,
            ],
            &[frames as u32],
            [32 * frames, 1, 1],
            256,
        );
        self.copy(
            c,
            &w.projected,
            (frames - 1) * 512,
            &self.up_history,
            0,
            512,
        );
        self.transformer(c, true, 16 * frames, self.codec_position);
        self.conv(c, 0, 16 * frames);
        let mut rows = 16 * frames;
        for (up, first, second, channels, stride) in
            [(1, 2, 3, 256, 6), (4, 5, 6, 128, 5), (7, 8, 9, 64, 4)]
        {
            self.point(
                c,
                &w.x,
                &w.x,
                &w.x,
                rows * channels * 2,
                4,
                rows * channels * 2,
                1.,
            );
            self.conv(c, up, rows);
            rows *= stride;
            self.point(
                c,
                &w.x,
                &w.x,
                &w.skip,
                rows * channels,
                0,
                rows * channels,
                1.,
            );
            self.point(c, &w.x, &w.x, &w.x, rows * channels, 4, rows * channels, 1.);
            self.conv(c, first, rows);
            self.point(
                c,
                &w.x,
                &w.x,
                &w.x,
                rows * channels / 2,
                4,
                rows * channels / 2,
                1.,
            );
            self.conv(c, second, rows);
            self.point(
                c,
                &w.x,
                &w.skip,
                &w.x,
                rows * channels,
                1,
                rows * channels,
                1.,
            );
        }
        self.point(c, &w.x, &w.x, &w.x, rows * 64, 4, rows * 64, 1.);
        self.conv(c, 10, rows);
    }
    fn step(&mut self, noise: &[f32; 32], first: bool) -> Result<(Vec<f32>, f32, f64)> {
        write(&self.ws.noise, noise);
        let c = self.device.begin()?;
        if !first {
            self.linear(
                &c,
                "flow_lm.input_linear",
                &self.ws.latent,
                &self.ws.x,
                32,
                512,
                1,
            );
            self.point(&c, &self.ws.x, &self.condition, &self.ws.x, 512, 1, 512, 1.);
            self.transformer(&c, false, 1, self.flow_position);
        }
        self.flow(&c);
        self.codec(&c, 1, &self.ws.latent);
        let seconds = c.finish()?;
        if !first {
            self.flow_position += 1;
        }
        self.codec_position += 16;
        let pcm = unsafe { self.ws.x.read_f32(0, FRAME_SAMPLES) };
        let eos = unsafe { self.ws.eos.read_f32(0, 1) }[0];
        if !eos.is_finite() || pcm.iter().any(|v| !v.is_finite()) {
            return Err(err("non-finite output"));
        }
        Ok((pcm, eos, seconds))
    }
    fn sample_batch(&mut self, noises: &[[f32; 32]]) -> Result<Vec<f32>> {
        assert!(!noises.is_empty() && noises.len() <= CODEC_BATCH);
        write(&self.ws.noises, noises.as_flattened());
        let c = self.device.begin()?;
        for i in 0..noises.len() {
            self.copy(&c, &self.ws.noises, i * 32, &self.ws.noise, 0, 32);
            self.linear(
                &c,
                "flow_lm.input_linear",
                &self.ws.latent,
                &self.ws.x,
                32,
                512,
                1,
            );
            self.point(&c, &self.ws.x, &self.condition, &self.ws.x, 512, 1, 512, 1.);
            self.transformer(&c, false, 1, self.flow_position + i);
            self.flow(&c);
            self.copy(&c, &self.ws.latent, 0, &self.ws.latents, i * 32, 32);
            self.copy(&c, &self.ws.eos, 0, &self.ws.endings, i, 1);
        }
        c.finish()?;
        self.flow_position += noises.len();
        let eos = unsafe { self.ws.endings.read_f32(0, noises.len()) };
        if eos.iter().any(|v| !v.is_finite()) {
            return Err(err("non-finite EOS"));
        }
        Ok(eos)
    }
    fn decode_batch(&mut self, frames: usize) -> Result<Vec<f32>> {
        assert!((1..=CODEC_BATCH).contains(&frames));
        let c = self.device.begin()?;
        self.codec(&c, frames, &self.ws.latents);
        c.finish()?;
        self.codec_position += frames * 16;
        let pcm = unsafe { self.ws.x.read_f32(0, frames * FRAME_SAMPLES) };
        if pcm.iter().any(|v| !v.is_finite()) {
            return Err(err("non-finite PCM"));
        }
        Ok(pcm)
    }
}
impl Backend for Phonon {
    fn generate(
        &mut self,
        text: &str,
        options: Options,
        emit: &mut dyn FnMut(Vec<f32>) -> bool,
    ) -> std::result::Result<(), String> {
        paddock_engine::phonon::validate_text(text)?;
        options.validate()?;
        self.reset(text).map_err(|e| e.to_string())?;
        let mut noise = Noise::new(options);
        let tail = if text.split_whitespace().count() <= 4 {
            3
        } else {
            1
        };
        // Keep the first frame on the latency path; later frames amortize the
        // decoder while preserving the exact stop boundary and causal history.
        let (pcm, first_eos, _) = self.step(&noise.frame(), true).map_err(|e| e.to_string())?;
        if !emit(pcm) {
            return Ok(());
        }
        let mut generated = 1;
        let mut stop = if first_eos > 4. {
            1 + tail
        } else {
            MAX_FRAMES + 1
        };
        while generated < MAX_FRAMES && generated < stop {
            let count = CODEC_BATCH
                .min(MAX_FRAMES - generated)
                .min(stop - generated);
            let noises: Vec<_> = (0..count).map(|_| noise.frame()).collect();
            let eos = self.sample_batch(&noises).map_err(|e| e.to_string())?;
            if let Some(i) = eos.iter().position(|&v| v > 4.) {
                stop = stop.min(generated + i + 1 + tail);
            }
            let frames = count.min(stop - generated);
            let pcm = self.decode_batch(frames).map_err(|e| e.to_string())?;
            for chunk in pcm.chunks_exact(FRAME_SAMPLES) {
                if !emit(chunk.to_vec()) {
                    return Ok(());
                }
            }
            generated += frames;
        }
        if generated == stop {
            return Ok(());
        }
        Err("Phonon reached its 60-second utterance limit before end of speech; split the input into shorter sentences".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires local Phonon checkpoint and Apple GPU"]
    fn native_phonon_batch_boundaries_and_cancellation() {
        let path = std::env::var("PADDOCK_PHONON_MODEL").expect("PADDOCK_PHONON_MODEL");
        let mut m = Phonon::load(Path::new(&path), None).unwrap();
        let resident = m.memory_bytes();
        let text = "Batch boundaries must not change the sound of this voice.";
        let options = Options {
            seed: 42,
            temperature: 0.3,
        };
        let mut noise = Noise::new(options);
        // Three complete circular KV wraps, including partial decoder batches.
        let noises: Vec<_> = (0..97).map(|_| noise.frame()).collect();
        m.reset(text).unwrap();
        let mut expected = Vec::new();
        let mut endings = Vec::new();
        for (i, noise) in noises.iter().enumerate() {
            let (pcm, eos, _) = m.step(noise, i == 0).unwrap();
            expected.extend(pcm);
            endings.push(eos);
        }
        for batch in [1, 2, 3, 4, 7, CODEC_BATCH] {
            m.reset(text).unwrap();
            let (mut pcm, eos, _) = m.step(&noises[0], true).unwrap();
            let mut actual_endings = vec![eos];
            for chunk in noises[1..].chunks(batch) {
                actual_endings.extend(m.sample_batch(chunk).unwrap());
                pcm.extend(m.decode_batch(chunk.len()).unwrap());
            }
            assert_eq!(actual_endings, endings, "EOS, batch {batch}");
            assert!(
                pcm == expected,
                "decoder samples changed with batch {batch}"
            );
            assert_eq!(m.memory_bytes(), resident, "request allocated GPU memory");
        }
        for (text, options) in [
            (
                "Hello!",
                Options {
                    seed: 0,
                    temperature: 0.,
                },
            ),
            ("A short sentence with a different ending.", options),
            (
                "When a local assistant reads a longer answer aloud, every word should survive the final partial batch, and a cancelled request must not leak its voice or its convolution history into the next answer.",
                Options {
                    seed: 99,
                    temperature: 0.3,
                },
            ),
        ] {
            m.reset(text).unwrap();
            let mut noise = Noise::new(options);
            let mut expected = Vec::new();
            let tail = if text.split_whitespace().count() <= 4 {
                3
            } else {
                1
            };
            let mut stop = MAX_FRAMES + 1;
            for frame in 0..MAX_FRAMES {
                let (pcm, eos, _) = m.step(&noise.frame(), frame == 0).unwrap();
                expected.extend(pcm);
                if eos > 4. {
                    stop = stop.min(frame + 1 + tail);
                }
                if frame + 1 == stop {
                    break;
                }
            }
            assert!(stop <= MAX_FRAMES, "fixture did not finish");
            for cancel_at in [1, 5, 10] {
                let mut emitted = 0;
                m.generate(
                    "Cancel this different utterance before completion.",
                    options,
                    &mut |_| {
                        emitted += 1;
                        emitted < cancel_at
                    },
                )
                .unwrap();
                assert_eq!(emitted, cancel_at);
                let mut actual = Vec::new();
                m.generate(text, options, &mut |pcm| {
                    actual.extend(pcm);
                    true
                })
                .unwrap();
                assert!(
                    actual == expected,
                    "EOS trimming or cancellation changed PCM: {text}"
                );
                assert_eq!(m.memory_bytes(), resident);
            }
        }
    }

    #[test]
    #[ignore = "diagnostic: requires Phonon weights and an otherwise idle Apple GPU"]
    fn native_phonon_stage_costs() {
        let path = std::env::var("PADDOCK_PHONON_MODEL").expect("PADDOCK_PHONON_MODEL");
        let mut m = Phonon::load(Path::new(&path), None).unwrap();
        let text = "Hello there, this is a local speech test.";
        let mut records = Vec::new();
        // The split-command pass attributes GPU execution, not serving latency.
        // Keep the ordinary, unsplit step as the performance control.
        for split in [false, true] {
            m.reset(text).unwrap();
            for frame in 0..96 {
                let start = std::time::Instant::now();
                let mut gpu = [0.; 3];
                if split {
                    write(&m.ws.noise, &[0.; 32]);
                    if frame != 0 {
                        let c = m.device.begin().unwrap();
                        m.linear(
                            &c,
                            "flow_lm.input_linear",
                            &m.ws.latent,
                            &m.ws.x,
                            32,
                            512,
                            1,
                        );
                        m.point(&c, &m.ws.x, &m.condition, &m.ws.x, 512, 1, 512, 1.);
                        m.transformer(&c, false, 1, m.flow_position);
                        gpu[0] = c.finish().unwrap();
                        m.flow_position += 1;
                    }
                    let c = m.device.begin().unwrap();
                    m.flow(&c);
                    gpu[1] = c.finish().unwrap();
                    let c = m.device.begin().unwrap();
                    m.codec(&c, 1, &m.ws.latent);
                    gpu[2] = c.finish().unwrap();
                    m.codec_position += 16;
                    let pcm = unsafe { m.ws.x.read_f32(0, FRAME_SAMPLES) };
                    assert!(pcm.iter().all(|x| x.is_finite()));
                } else {
                    gpu[0] = m.step(&[0.; 32], frame == 0).unwrap().2;
                }
                records.push(serde_json::json!({
                    "split": split, "frame": frame,
                    "wall_s": start.elapsed().as_secs_f64(), "gpu_s": gpu,
                }));
            }
        }
        let output = std::env::var("PADDOCK_PHONON_PROFILE").expect("PADDOCK_PHONON_PROFILE");
        std::fs::write(output, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
    }

    #[test]
    #[ignore = "requires independently generated reference tensors and Apple GPU"]
    fn native_phonon_reference() {
        let path = std::env::var("PADDOCK_PHONON_MODEL").expect("PADDOCK_PHONON_MODEL");
        let mut m = Phonon::load(Path::new(&path), None).unwrap();
        eprintln!("resident weight/workspace bytes: {:?}", m.memory_bytes());
        let dir = std::env::var("PADDOCK_PHONON_REFERENCE").expect("PADDOCK_PHONON_REFERENCE");
        let check = |name: &str, actual: &[f32], tolerance: f64| {
            let b = std::fs::read(Path::new(&dir).join(format!("{name}.f32"))).unwrap();
            let expected: Vec<f32> = b
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(actual.len(), expected.len());
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let magnitude = expected
                .iter()
                .map(|&v| (v as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-8);
            let relative = error / magnitude;
            eprintln!("{name}: relative L2 {relative:.8}");
            assert!(relative < tolerance, "{name}: {relative} >= {tolerance}");
        };
        check("condition", &unsafe { m.condition.read_f32(0, 512) }, 1e-5);
        check("time", &unsafe { m.time.read_f32(0, 512) }, 1e-5);
        check("voice", &m.voice, 1e-5);
        check("bos", &m.bos, 1e-5);
        m.reset("Hello. This is Paddock speaking with its own native speech engine.")
            .unwrap();
        check("hidden", &unsafe { m.ws.h.read_f32(0, 512) }, 1e-4);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(Path::new(&dir).join("manifest.json")).unwrap())
                .unwrap();
        let frames = manifest["frames"].as_u64().unwrap() as usize;
        assert!(
            (33..=MAX_FRAMES).contains(&frames),
            "reference must cross the codec cache wrap"
        );
        let mut pcm = Vec::new();
        for i in 0..frames {
            let (p, eos, _) = m.step(&[0.; 32], i == 0).unwrap();
            pcm.extend(p);
            check(
                &format!("latent-{i}"),
                &unsafe { m.ws.latent.read_f32(0, 32) },
                1e-4,
            );
            check(&format!("eos-{i}"), &[eos], 1e-4);
        }
        check("pcm", &pcm, 0.002);
        // Whole-utterance EOS/tail contract, independent of fixed frame probes.
        let first_eos = (0..frames)
            .find(|i| {
                let b = std::fs::read(Path::new(&dir).join(format!("eos-{i}.f32"))).unwrap();
                f32::from_le_bytes(b.try_into().unwrap()) > 4.
            })
            .expect("reference must include end of speech");
        let mut generated = Vec::new();
        m.generate(
            "Hello. This is Paddock speaking with its own native speech engine.",
            Options {
                temperature: 0.,
                seed: 0,
            },
            &mut |x| {
                generated.extend(x);
                true
            },
        )
        .unwrap();
        assert_eq!(generated.len(), (first_eos + 2) * FRAME_SAMPLES);
        assert!(
            generated == pcm[..generated.len()],
            "whole generation must preserve the native decoder samples"
        );
    }
    #[test]
    #[ignore = "requires local Phonon checkpoint and Apple GPU"]
    fn native_phonon_audio() {
        let path = std::env::var("PADDOCK_PHONON_MODEL").expect("PADDOCK_PHONON_MODEL");
        let mut m = Phonon::load(Path::new(&path), None).unwrap();
        let text = "Hello. This is Paddock speaking with its own native speech engine.";
        let mut pcm = Vec::new();
        let start = std::time::Instant::now();
        m.generate(text, Options::default(), &mut |x| {
            pcm.extend(x);
            true
        })
        .unwrap();
        assert!(pcm.len() > 24000);
        assert!(pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32 > 1e-6);
        let audio =
            paddock_engine::phonon::wav(&paddock_engine::phonon::pcm16(&pcm).unwrap()).unwrap();
        let out = std::env::var("PADDOCK_PHONON_WAV")
            .unwrap_or_else(|_| "/tmp/paddock-native-phonon.wav".into());
        std::fs::write(&out, audio).unwrap();
        eprintln!(
            "{} samples, {:.3}s wall, {:.3}s audio, {}",
            pcm.len(),
            start.elapsed().as_secs_f64(),
            pcm.len() as f64 / 24000.,
            out
        );
        let mut second = Vec::new();
        m.generate(text, Options::default(), &mut |x| {
            second.extend(x);
            true
        })
        .unwrap();
        assert_eq!(pcm.len(), second.len());
        let max = pcm
            .iter()
            .zip(&second)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        assert!(
            pcm == second,
            "request state must reset exactly; maximum sample difference {max}"
        );
    }
}
