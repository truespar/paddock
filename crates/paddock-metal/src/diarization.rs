//! Native Nemotron 3 Diarization. BF16 MLX and packed Q8_0 GGUF weights;
//! reusable, duration-independent scratch and no host neural-network fallback.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use half::f16;
use paddock_models::diarization::checkpoint::{Checkpoint, Format, LAYERS};
use std::path::Path;

const MAX_ROWS: usize = 684;
fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Nemotron diarization: {}", s.into()))
}
fn bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Linear {
    w: Weight,
    b: Buffer,
    bias: bool,
}
struct Norm {
    w: Buffer,
    b: Buffer,
}
struct Layer {
    n1: Norm,
    qkv: Linear,
    out: Linear,
    n2: Norm,
    up: Linear,
    down: Linear,
}
struct Workspace {
    pcm: Buffer,
    input: Buffer,
    x: Buffer,
    n: Buffer,
    wide: Buffer,
    tmp: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    att: Buffer,
    head: Buffer,
    conv: Buffer,
    up: Buffer,
    hidden: Buffer,
    probs: Buffer,
    gemm: Buffer,
}
impl Workspace {
    fn new(d: &MetalDevice) -> Result<Self> {
        let a = |n| d.alloc(MAX_ROWS * n * 4);
        Ok(Self {
            pcm: d.alloc(16000 * 64 * 4)?,
            input: a(1024)?,
            x: a(512)?,
            n: a(512)?,
            wide: a(2048)?,
            tmp: a(512)?,
            q: a(512)?,
            k: a(512)?,
            v: a(512)?,
            att: a(512)?,
            head: a(192)?,
            conv: a(576)?,
            up: a(1536)?,
            hidden: a(1536)?,
            probs: a(64)?,
            gemm: d.alloc(MAX_ROWS.div_ceil(128) * 128 * 2048 * 2)?,
        })
    }
}
pub struct Diarization {
    // Diagnostic A/B control only, absent from production builds. Alternates
    // the original conversions and fused producers on the same loaded model.
    #[cfg(test)]
    prepare_q8_consumers: bool,
    window_gpu: Buffer,
    fb_gpu: Buffer,
    spans: Buffer,
    twiddle: Buffer,
    device: MetalDevice,
    bf: bool,
    weights: u64,
    pre_seconds: f64,
    ws: Workspace,
    pre: Linear,
    embed: Norm,
    layers: Vec<Layer>,
    last: Norm,
    proj: Linear,
    conv: Linear,
    hidden: Linear,
    score: Linear,
    pub window: Vec<f32>,
    pub filterbank: Vec<f32>,
    pub silence: Vec<f32>,
}
impl paddock_engine::diarization::Backend for Diarization {
    fn encode_audio(
        &mut self,
        _fe: &paddock_engine::diarization::Frontend,
        w: paddock_engine::diarization::AudioWindow<'_>,
    ) -> std::result::Result<Vec<f32>, String> {
        self.encode_audio(w).map_err(|e| e.to_string())
    }
    fn pre_encode(&mut self, features: &[f32]) -> std::result::Result<Vec<f32>, String> {
        Self::pre_encode(self, features).map_err(|e| e.to_string())
    }
    fn predict(
        &mut self,
        x: &[f32],
        valid: usize,
    ) -> std::result::Result<(Vec<[f32; 8]>, f64), String> {
        Self::predict(self, x, valid).map_err(|e| e.to_string())
    }
    fn frontend(&self) -> std::result::Result<paddock_engine::diarization::Frontend, String> {
        paddock_engine::diarization::Frontend::new(&self.window, &self.filterbank)
    }
    fn silence(&self) -> &[f32] {
        &self.silence
    }
    fn weight_bytes(&self) -> u64 {
        self.weights
    }
    fn workspace_bytes(&self) -> u64 {
        Self::workspace_bytes(self)
    }
}
impl Diarization {
    pub fn load(path: &Path, budget: Option<u64>) -> Result<Self> {
        let source = Checkpoint::open(path).map_err(error)?;
        let bf = source.format() == Format::MlxBf16;
        // One architecture, shape and stored-type contract for CUDA and Metal.
        // Validate before creating a GPU device or allocating its weight grant.
        let weight_bytes = source.validate().map_err(error)?;
        let window = source.window().map_err(error)?;
        let filterbank = source.filterbank().map_err(error)?;
        let silence = source.silence().map_err(error)?;
        let device = MetalDevice::new_planned(budget, weight_bytes + 40 * 1024 * 1024)?;
        let scalar = |n: &str, shape: &[usize]| -> Result<Buffer> {
            device.upload(&bytes(&source.values(n, shape).map_err(error)?))
        };
        let norm = |n: &str| -> Result<Norm> {
            Ok(Norm {
                w: scalar(&format!("{n}.weight"), &[512])?,
                b: scalar(&format!("{n}.bias"), &[512])?,
            })
        };
        let linear = |n: &str, k: usize, out: usize, bias: bool, conv: bool| -> Result<Linear> {
            let shape = if conv {
                vec![out, 3, 192]
            } else {
                vec![out, k]
            };
            let raw = source.raw(&format!("{n}.weight"), &shape).map_err(error)?;
            let buffer = if conv && !bf {
                // The shared reader already transposes GGUF [out][c][t]
                // to canonical [out][t][c]. Do not transpose a second time.
                let reordered = source
                    .values(&format!("{n}.weight"), &shape)
                    .map_err(error)?;
                device.upload(
                    &reordered
                        .iter()
                        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                        .collect::<Vec<_>>(),
                )?
            } else {
                device.upload(raw.bytes)?
            };
            Ok(Linear {
                w: Weight {
                    buffer,
                    ty: if conv && !bf { 1 } else { raw.ty },
                    k,
                    n: out,
                },
                b: if bias {
                    scalar(&format!("{n}.bias"), &[out])?
                } else {
                    device.upload(&vec![0u8; out * 4])?
                },
                bias,
            })
        };
        let pre = linear("encoder.pre_encode.proj", 1024, 512, false, false)?;
        let embed = norm("encoder.embed_norm")?;
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            let p = format!("encoder.layers.{i}");
            layers.push(Layer {
                n1: norm(&format!("{p}.norm1"))?,
                qkv: linear(&format!("{p}.attn.w_qkv"), 512, 1536, false, false)?,
                out: linear(&format!("{p}.attn.out_proj"), 512, 512, true, false)?,
                n2: norm(&format!("{p}.norm2"))?,
                up: linear(&format!("{p}.ffn.linear1"), 512, 2048, true, false)?,
                down: linear(&format!("{p}.ffn.linear2"), 2048, 512, true, false)?,
            });
        }
        let last = norm("encoder.final_norm")?;
        let proj = linear("sortformer_modules.encoder_proj", 512, 192, true, false)?;
        let conv = linear(
            "sortformer_modules.subpixel_upsample",
            576,
            1536,
            true,
            true,
        )?;
        let hidden = linear(
            "sortformer_modules.first_hidden_to_hidden",
            192,
            192,
            true,
            false,
        )?;
        let score = linear(
            "sortformer_modules.single_hidden_to_spks",
            192,
            8,
            true,
            false,
        )?;
        let mut padded = vec![0.; 512];
        padded[56..456].copy_from_slice(&window);
        let window_gpu = device.upload(&bytes(&padded))?;
        let fb_gpu = device.upload(&bytes(&filterbank))?;
        let span: Vec<u8> = filterbank
            .chunks_exact(257)
            .flat_map(|row| {
                let first = row.iter().position(|&v| v != 0.).unwrap_or(0) as u32;
                let end = row.iter().rposition(|&v| v != 0.).map_or(0, |i| i + 1) as u32;
                first.to_le_bytes().into_iter().chain(end.to_le_bytes())
            })
            .collect();
        let spans = device.upload(&span)?;
        let angles: Vec<f32> = (0..256)
            .flat_map(|i| {
                let (s, c) = (-2.0 * std::f64::consts::PI * i as f64 / 512.).sin_cos();
                [c as f32, s as f32]
            })
            .collect();
        let twiddle = device.upload(&bytes(&angles))?;
        let weights = device.allocated_bytes();
        let ws = Workspace::new(&device)?;
        Ok(Self {
            #[cfg(test)]
            prepare_q8_consumers: true,
            window_gpu,
            fb_gpu,
            spans,
            twiddle,
            device,
            bf,
            weights,
            pre_seconds: 0.,
            ws,
            pre,
            embed,
            layers,
            last,
            proj,
            conv,
            hidden,
            score,
            window,
            filterbank,
            silence,
        })
    }
    fn linear(&self, c: &Commands<'_>, l: &Linear, input: &Buffer, out: &Buffer, rows: usize) {
        if self.bf {
            let (kernel, tile) = self.projection_tile();
            c.dispatch(
                kernel,
                &[&l.w.buffer, input, out, &l.b],
                &[
                    l.w.k as u32,
                    l.w.n as u32,
                    rows as u32,
                    u32::from(l.bias),
                    0,
                ],
                [l.w.n.div_ceil(64), rows.div_ceil(tile), 1],
                128,
            );
        } else if l.w.ty == 0 {
            c.dispatch(
                "kumo_mm",
                &[&l.w.buffer, input, out, &l.b],
                &[l.w.k as u32, l.w.n as u32, rows as u32, 0, 0],
                [l.w.n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
        } else if self.q8_tensor(l, rows) {
            // Small-row vector projections have a different arithmetic
            // contract; keep their original path below. Tensor rows fuse the
            // original bias/pointwise boundaries while retaining packed Q8.
            self.linear_post(c, l, input, out, rows, 0);
        } else {
            l.w.linear(c, input, out, rows, 1.0, &self.ws.gemm);
            if l.bias {
                self.post(c, out, &l.b, out, rows * l.w.n, l.w.n, 1);
            }
        }
    }
    fn projection_tile(&self) -> (&'static str, usize) {
        if self.device.tensor_accelerated() {
            ("diar_project32", 32)
        } else {
            // Preserve the existing tile on shader-core-only Apple GPUs;
            // the M5 timing evidence does not elect an M1–M4 geometry.
            ("diar_project64", 64)
        }
    }
    fn q8_tensor(&self, l: &Linear, rows: usize) -> bool {
        !self.bf && l.w.ty == 8 && rows >= 16 && self.device.tensor_accelerated()
    }
    fn prepare_q8_consumers(&self) -> bool {
        #[cfg(test)]
        {
            self.prepare_q8_consumers
        }
        #[cfg(not(test))]
        {
            true
        }
    }
    /// Consumes a producer's padded F16 operand. Private: the producer must
    /// preserve conversion boundaries and zero all padded rows/columns.
    fn q8_prepared(
        &self,
        c: &Commands<'_>,
        l: &Linear,
        out: &Buffer,
        rows: usize,
        op: u32,
        prepared: &Buffer,
    ) {
        debug_assert!(self.q8_tensor(l, rows));
        c.dispatch(
            "diar_q8_project32",
            &[&l.w.buffer, prepared, out, &l.b],
            &[
                l.w.k as u32,
                l.w.n as u32,
                rows as u32,
                u32::from(l.bias),
                op,
            ],
            [l.w.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
    fn norm_linear(
        &self,
        c: &Commands<'_>,
        n: &Norm,
        x: &Buffer,
        l: &Linear,
        out: &Buffer,
        rows: usize,
    ) {
        if self.q8_tensor(l, rows) && l.w.k == 512 {
            c.dispatch(
                "diar_norm_q8_input",
                &[x, &n.w, &n.b, &self.ws.n, &self.ws.gemm],
                &[rows as u32],
                [rows.div_ceil(128) * 128, 1, 1],
                256,
            );
            self.q8_prepared(c, l, out, rows, 0, &self.ws.gemm);
        } else {
            self.norm(c, n, x, &self.ws.n, rows);
            self.linear(c, l, &self.ws.n, out, rows);
        }
    }
    fn norm(&self, c: &Commands<'_>, n: &Norm, x: &Buffer, y: &Buffer, rows: usize) {
        if self.bf {
            c.dispatch(
                "gmlx_layer_norm",
                &[x, &n.w, &n.b, y],
                &[512, 1e-5f32.to_bits()],
                [rows, 1, 1],
                128,
            );
            return;
        }
        c.dispatch(
            "diar_norm",
            &[x, &n.w, &n.b, y],
            &[u32::from(self.bf)],
            [rows, 1, 1],
            256,
        );
    }
    /// Producer-side pointwise work. Residual writes the existing output in
    /// place, never aliases the projection input. GGUF keeps its F32 contract.
    fn linear_post(
        &self,
        c: &Commands<'_>,
        l: &Linear,
        input: &Buffer,
        out: &Buffer,
        rows: usize,
        op: u32,
    ) {
        if self.bf {
            let (kernel, tile) = self.projection_tile();
            c.dispatch(
                kernel,
                &[&l.w.buffer, input, out, &l.b],
                &[
                    l.w.k as u32,
                    l.w.n as u32,
                    rows as u32,
                    u32::from(l.bias),
                    op,
                ],
                [l.w.n.div_ceil(64), rows.div_ceil(tile), 1],
                128,
            );
        } else if self.q8_tensor(l, rows) {
            c.dispatch(
                "linear_input_padded",
                &[input, &self.ws.gemm],
                &[l.w.k as u32, l.w.n as u32, rows as u32],
                [
                    (l.w.k.div_ceil(128) * 128 * rows.div_ceil(128) * 128).div_ceil(256),
                    1,
                    1,
                ],
                256,
            );
            self.q8_prepared(c, l, out, rows, op, &self.ws.gemm);
        } else if op == 2 {
            self.linear(c, l, input, &self.ws.tmp, rows);
            self.post(c, &self.ws.tmp, &l.b, out, rows * l.w.n, l.w.n, op);
        } else {
            self.linear(c, l, input, out, rows);
            self.post(c, out, &l.b, out, rows * l.w.n, l.w.n, op);
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn post(
        &self,
        c: &Commands<'_>,
        x: &Buffer,
        b: &Buffer,
        y: &Buffer,
        len: usize,
        width: usize,
        op: u32,
    ) {
        c.dispatch(
            "diar_post",
            &[x, b, y],
            &[len as u32, width as u32, u32::from(self.bf), op],
            [len.div_ceil(256), 1, 1],
            256,
        );
    }
    /// Frame-major log-mels, padded by the caller to a multiple of eight.
    pub fn encode_audio(
        &mut self,
        w: paddock_engine::diarization::AudioWindow<'_>,
    ) -> Result<Vec<f32>> {
        w.validate().map_err(error)?;
        let frames = w.count.div_ceil(8) * 8;
        // SAFETY: exclusive queue owner, previous submission has completed.
        unsafe {
            self.ws
                .pcm
                .write_u32(&w.audio.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        let c = self.device.begin()?;
        c.dispatch(
            "diar_frontend",
            &[
                &self.ws.pcm,
                &self.window_gpu,
                &self.fb_gpu,
                &self.spans,
                &self.twiddle,
                &self.ws.input,
            ],
            &[
                w.offset as u32,
                w.total as u32,
                w.start as u32,
                u32::from(self.bf),
                w.count as u32,
            ],
            [frames, 1, 1],
            256,
        );
        self.linear(&c, &self.pre, &self.ws.input, &self.ws.x, frames / 8);
        self.pre_seconds = c.finish()?;
        Ok(unsafe { self.ws.x.read_f32(0, frames / 8 * 512) })
    }
    pub fn pre_encode(&mut self, features: &[f32]) -> Result<Vec<f32>> {
        if features.is_empty()
            || !features.len().is_multiple_of(1024)
            || features.len() > MAX_ROWS * 1024
            || features.iter().any(|v| !v.is_finite())
        {
            return Err(error("invalid feature window"));
        }
        let rows = features.len() / 1024;
        // SAFETY: this owner completed the previous synchronous submission.
        unsafe {
            self.ws
                .input
                .write_u32(&features.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        let c = self.device.begin()?;
        if self.bf {
            c.dispatch(
                "diar_round",
                &[&self.ws.input],
                &[features.len() as u32],
                [features.len().div_ceil(256), 1, 1],
                256,
            );
        }
        self.linear(&c, &self.pre, &self.ws.input, &self.ws.x, rows);
        self.pre_seconds = c.finish()?;
        Ok(unsafe { self.ws.x.read_f32(0, rows * 512) })
    }
    /// Embeddings include bounded speaker-cache and FIFO rows. `valid` masks
    /// padded keys only; right-context rows are real attention inputs.
    pub fn predict(&mut self, embeddings: &[f32], valid: usize) -> Result<(Vec<[f32; 8]>, f64)> {
        self.predict_impl(embeddings, valid, None)
    }
    /// Qualification only: stage snapshots from the actual serving path. Extra
    /// buffers/dispatches exist only when explicitly called by the diagnostic.
    pub fn trace(&mut self, embeddings: &[f32], valid: usize) -> Result<Vec<(String, Vec<f32>)>> {
        let mut captures = Vec::new();
        self.predict_impl(embeddings, valid, Some(&mut captures))?;
        Ok(captures)
    }
    fn predict_impl(
        &mut self,
        embeddings: &[f32],
        valid: usize,
        trace: Option<&mut Vec<(String, Vec<f32>)>>,
    ) -> Result<(Vec<[f32; 8]>, f64)> {
        let rows = embeddings.len() / 512;
        if rows == 0
            || rows > MAX_ROWS
            || !embeddings.len().is_multiple_of(512)
            || valid == 0
            || valid > rows
            || embeddings.iter().any(|v| !v.is_finite())
        {
            return Err(error("invalid encoder window"));
        }
        unsafe {
            self.ws
                .input
                .write_u32(&embeddings.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        let c = self.device.begin()?;
        let w = &self.ws;
        let mut snapshots = Vec::new();
        let mut capture = |name: &str, buffer: &Buffer, count: usize| -> Result<()> {
            if trace.is_some() {
                let copy = self.device.alloc(count * 4)?;
                let heads = ["q", "k"].contains(&name);
                let params = if heads {
                    [rows as u32, u32::from(self.bf), 0]
                } else {
                    [0, 0, count as u32]
                };
                c.dispatch(
                    if heads {
                        "diar_heads_trace"
                    } else {
                        "spec_copy"
                    },
                    &[buffer, &copy],
                    &params,
                    [count.div_ceil(256), 1, 1],
                    256,
                );
                snapshots.push((name.to_owned(), copy, count));
            }
            Ok(())
        };
        self.norm(&c, &self.embed, &w.input, &w.x, rows);
        capture("embed_norm", &w.x, rows * 512)?;
        for (index, l) in self.layers.iter().enumerate() {
            self.norm_linear(&c, &l.n1, &w.x, &l.qkv, &w.wide, rows);
            if index == 0 {
                capture("norm1", &w.n, rows * 512)?;
            }
            if index == 0 {
                capture("qkv", &w.wide, rows * 1536)?;
            }
            c.dispatch(
                "diar_heads",
                &[&w.wide, &w.q, &w.k, &w.v],
                &[rows as u32, u32::from(self.bf)],
                [(rows * 512).div_ceil(256), 1, 1],
                256,
            );
            if index == 0 {
                capture("q", &w.q, rows * 512)?;
                capture("k", &w.k, rows * 512)?;
            }
            let prepared_attention =
                self.prepare_q8_consumers() && self.q8_tensor(&l.out, rows) && l.out.w.k == 512;
            let attention_buffers = [&w.q, &w.k, &w.v, &w.att, &w.gemm];
            c.dispatch(
                if prepared_attention {
                    "diar_attention_q8_input"
                } else if self.bf {
                    "diar_attention_bf16"
                } else {
                    "diar_attention_f32"
                },
                &attention_buffers[..if prepared_attention { 5 } else { 4 }],
                &[
                    8,
                    64,
                    rows as u32,
                    valid as u32,
                    0,
                    0,
                    8,
                    0,
                    rows as u32,
                    rows as u32,
                ],
                [8, rows.div_ceil(32), 1],
                128,
            );
            if prepared_attention {
                self.q8_prepared(&c, &l.out, &w.tmp, rows, 0, &w.gemm);
            } else {
                self.linear(&c, &l.out, &w.att, &w.tmp, rows);
            }
            if index == 0 {
                capture("attention", &w.att, rows * 512)?;
                capture("out_proj", &w.tmp, rows * 512)?;
            }
            let prepared_up = self.q8_tensor(&l.up, rows) && l.up.w.k == 512;
            if prepared_up {
                c.dispatch(
                    "diar_residual_norm_q8_input",
                    &[&w.tmp, &w.x, &l.n2.w, &l.n2.b, &w.n, &w.gemm],
                    &[rows as u32],
                    [rows.div_ceil(128) * 128, 1, 1],
                    256,
                );
            } else {
                c.dispatch(
                    if self.bf {
                        "diar_residual_norm_bf16"
                    } else {
                        "diar_residual_norm_f32"
                    },
                    &[&w.tmp, &w.x, &l.n2.w, &l.n2.b, &w.n],
                    &[],
                    [rows, 1, 1],
                    if self.bf { 128 } else { 256 },
                );
            }
            if index == 0 {
                capture("residual1", &w.x, rows * 512)?;
            }
            let prepared_down = self.prepare_q8_consumers()
                && prepared_up
                && self.q8_tensor(&l.down, rows)
                && l.up.w.n == 2048
                && l.down.w.k == 2048;
            if prepared_down {
                // The up projection still reads gemm in other workgroups.
                // Reuse head scratch (dead until the encoder completes), not
                // its live input. F32 wide remains available for trace parity.
                debug_assert!(w.hidden.len() >= rows.div_ceil(128) * 128 * 2048 * 2);
                c.dispatch(
                    "diar_q8_project32_prepare",
                    &[&l.up.w.buffer, &w.gemm, &w.wide, &l.up.b, &w.hidden],
                    &[512, 2048, rows as u32, u32::from(l.up.bias), 3],
                    [32, rows.div_ceil(32), 1],
                    128,
                );
            } else if prepared_up {
                self.q8_prepared(&c, &l.up, &w.wide, rows, 3, &w.gemm);
            } else {
                self.linear_post(&c, &l.up, &w.n, &w.wide, rows, 3);
            }
            if index == 0 {
                capture("gelu", &w.wide, rows * 2048)?;
            }
            if prepared_down {
                self.q8_prepared(&c, &l.down, &w.x, rows, 2, &w.hidden);
            } else {
                self.linear_post(&c, &l.down, &w.wide, &w.x, rows, 2);
            }
            if trace.is_some() {
                capture(&format!("layer{index}"), &w.x, rows * 512)?;
            }
        }
        self.norm_linear(&c, &self.last, &w.x, &self.proj, &w.head, rows);
        c.dispatch(
            "diar_conv_rows",
            &[&w.head, &w.conv],
            &[rows as u32],
            [(rows * 576).div_ceil(256), 1, 1],
            256,
        );
        self.linear_post(&c, &self.conv, &w.conv, &w.up, rows, 4);
        self.linear_post(&c, &self.hidden, &w.up, &w.hidden, rows * 8, 4);
        self.linear_post(&c, &self.score, &w.hidden, &w.probs, rows * 8, 5);
        let seconds = c.finish()? + std::mem::take(&mut self.pre_seconds);
        if let Some(trace) = trace {
            for (name, buffer, count) in snapshots {
                trace.push((name, unsafe { buffer.read_f32(0, count) }));
            }
        }
        let flat = unsafe { w.probs.read_f32(0, rows * 64) };
        let mut probs: Vec<[f32; 8]> = flat
            .chunks_exact(8)
            .map(|c| c.try_into().expect("eight-speaker chunk"))
            .collect();
        probs[valid * 8..].fill([0.; 8]);
        if probs.iter().flatten().any(|v| !v.is_finite()) {
            return Err(error("nonfinite model output"));
        }
        Ok((probs, seconds))
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weights
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.device.allocated_bytes() - self.weights
    }
}

#[cfg(test)]
mod tests;
