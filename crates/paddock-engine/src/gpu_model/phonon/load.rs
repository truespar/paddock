//! Loading: the checkpoint's planes onto the device (Mimi's convolutions laid
//! out as GEMM rows), the request-independent constants (speaker/duration
//! condition, BOS, time embedding) computed on the GPU and folded into the
//! products that consume them, the voice prompt's K/V cached once, and every
//! batch shape's graph warmed and captured.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::phonon::{CODEC_CACHE, Checkpoint, FLOW_CACHE, FRAME_SAMPLES};
use paddock_tokenizer::GgufTokenizer;

use super::{
    Block, CODEC_BATCH, CODEC_RING, CODEC_ROWS, CODEC_SPLITS, COND_STRIDE, Cache, CodecWork, Conv,
    ConvPath, D, FF, FLOW_SPLITS, FlowLm, GpuPhonon, LN_EPS, Layer, Lin, MAX_ROWS, MODULATION,
    Mimi, Pass, Tx, Work,
};
use crate::gpu::{DiarEpi, DiarWeights, GpuError, GpuExecutor, PhEpi, PhExtra, PhPro, PhWeight};
use crate::gpu_model::gpt_oss::GpuModelError;

/// The Mimi convolutions in pass order: (name, cin, cout, width, stride,
/// path). The paths are the GB10 election (8- and 1-frame batches, every
/// arm against an F64 evaluation): the in-place rows where the im2col plane
/// was the cost (the k3 and the last transposed convolutions: 1.1-2.1x, the
/// same bits), the direct kernel for the mono output (3.5x at 8 frames),
/// the plane where it is small beside the GEMM's work (conv0, the wide
/// transposed convolutions, every k1 - whose plane is the ELU'd copy either
/// way).
const CONVS: [(&str, usize, usize, usize, usize, ConvPath); 11] = [
    ("model.0.conv", 512, 512, 7, 1, ConvPath::Im2col),
    ("model.2.convtr", 512, 256, 12, 6, ConvPath::Im2col),
    ("model.3.block.1.conv", 256, 128, 3, 1, ConvPath::Rows),
    ("model.3.block.3.conv", 128, 256, 1, 1, ConvPath::Im2col),
    ("model.5.convtr", 256, 128, 10, 5, ConvPath::Im2col),
    ("model.6.block.1.conv", 128, 64, 3, 1, ConvPath::Rows),
    ("model.6.block.3.conv", 64, 128, 1, 1, ConvPath::Im2col),
    ("model.8.convtr", 128, 64, 8, 4, ConvPath::Rows),
    ("model.9.block.1.conv", 64, 32, 3, 1, ConvPath::Rows),
    ("model.9.block.3.conv", 32, 64, 1, 1, ConvPath::Im2col),
    ("model.11.conv", 64, 1, 3, 1, ConvPath::Direct),
];

/// The widest Mimi plane: the last stage's 1920 rows a frame at 64 channels;
/// its im2col staging is three of those (k3 over 64 channels).
const CODEC_PLANE: usize = CODEC_BATCH * FRAME_SAMPLES * 64;

/// A plain conv `[cout][cin][k]` as GEMM rows `[cout][k * cin]` (tap-major,
/// the im2col column order). A permutation: no value changes.
fn conv_rows(w: &[f32], cin: usize, cout: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for o in 0..cout {
        for i in 0..cin {
            for j in 0..k {
                out[o * k * cin + j * cin + i] = w[(o * cin + i) * k + j];
            }
        }
    }
    out
}

/// A transposed conv `[cin][cout][2s]` (stride s, trimmed) as GEMM rows
/// `[s * cout][2 cin]`: output row t s + j of channel o is row (j, o) against
/// [x(t - 1) ; x(t)] - taps j + s and j. A permutation: no value changes.
fn convtr_rows(w: &[f32], cin: usize, cout: usize, s: usize) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for j in 0..s {
        for o in 0..cout {
            let row = (j * cout + o) * 2 * cin;
            for i in 0..cin {
                let at = (i * cout + o) * 2 * s;
                out[row + i] = w[at + j + s];
                out[row + cin + i] = w[at + j];
            }
        }
    }
    out
}

struct Loader<'a> {
    exec: &'a GpuExecutor,
    ck: Checkpoint,
    bytes: u64,
}

impl Loader<'_> {
    fn take(&mut self, name: &str) -> Result<Vec<f32>, GpuModelError> {
        self.ck
            .tensors
            .remove(name)
            .ok_or_else(|| GpuModelError::Unsupported(format!("Phonon: missing tensor {name}")))
    }

    fn up(&mut self, v: &[f32]) -> Result<CudaSlice<f32>, GpuModelError> {
        self.bytes += 4 * v.len() as u64;
        Ok(self.exec.to_device(v)?)
    }

    fn vec(&mut self, name: &str) -> Result<CudaSlice<f32>, GpuModelError> {
        let v = self.take(name)?;
        self.up(&v)
    }

    /// A matrix plane: BF16 coefficients (FlowLM's) stay BF16 - the
    /// checkpoint's own bits as raw bytes - everything else F32.
    fn plane(&mut self, v: &[f32]) -> Result<PhWeight, GpuModelError> {
        if v.iter().all(|x| x.to_bits() & 0xffff == 0) {
            let h: Vec<u8> = v
                .iter()
                .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            self.bytes += h.len() as u64;
            Ok(PhWeight::Bf16(self.exec.to_device_u8(&h)?))
        } else {
            Ok(PhWeight::F32(self.up(v)?))
        }
    }

    fn mat(&mut self, name: &str) -> Result<PhWeight, GpuModelError> {
        let v = self.take(&format!("{name}.weight"))?;
        self.plane(&v)
    }

    /// A linear with the bias the checkpoint gives it.
    fn lin(&mut self, name: &str) -> Result<Lin, GpuModelError> {
        let w = self.mat(name)?;
        let bias = format!("{name}.bias");
        let b = match self.ck.tensors.contains_key(&bias) {
            true => Some(self.vec(&bias)?),
            false => None,
        };
        Ok(Lin { w, b })
    }

    fn norm(&mut self, name: &str) -> Result<(CudaSlice<f32>, CudaSlice<f32>), GpuModelError> {
        Ok((
            self.vec(&format!("{name}.weight"))?,
            self.vec(&format!("{name}.bias"))?,
        ))
    }

    fn layers(
        &mut self,
        prefix: &str,
        count: usize,
        scale: bool,
    ) -> Result<Vec<Layer>, GpuModelError> {
        (0..count)
            .map(|i| {
                let p = format!("{prefix}.layers.{i}");
                Ok(Layer {
                    norm1: self.norm(&format!("{p}.norm1"))?,
                    norm2: self.norm(&format!("{p}.norm2"))?,
                    in_proj: self.mat(&format!("{p}.self_attn.in_proj"))?,
                    out_proj: self.mat(&format!("{p}.self_attn.out_proj"))?,
                    linear1: self.mat(&format!("{p}.linear1"))?,
                    linear2: self.mat(&format!("{p}.linear2"))?,
                    scale: match scale {
                        true => Some((
                            self.vec(&format!("{p}.layer_scale_1.scale"))?,
                            self.vec(&format!("{p}.layer_scale_2.scale"))?,
                        )),
                        false => None,
                    },
                })
            })
            .collect()
    }
}

fn caches(exec: &GpuExecutor, n: usize, cap: usize) -> Result<Vec<Cache>, GpuError> {
    (0..n)
        .map(|_| {
            Ok(Cache {
                k: exec.alloc(cap * D)?,
                v: exec.alloc(cap * D)?,
            })
        })
        .collect()
}

impl Tx {
    fn f32s(rows: usize, splits: usize) -> usize {
        rows * D * 4 + rows * 3 * D + rows * 8 * splits * 66 + rows * FF
    }

    fn bytes(rows: usize, splits: usize) -> u64 {
        4 * (Self::f32s(rows, splits) + rows * 8 + 1) as u64
    }

    fn new(exec: &GpuExecutor, rows: usize, splits: usize) -> Result<Self, GpuError> {
        Ok(Self {
            x: exec.alloc(rows * D)?,
            xn: exec.alloc(rows * D)?,
            qkv: exec.alloc(rows * 3 * D)?,
            q: exec.alloc(rows * D)?,
            att: exec.alloc(rows * D)?,
            part: exec.alloc(rows * 8 * splits * 66)?,
            ctr: exec.alloc_u32(rows * 8)?,
            wide: exec.alloc(rows * FF)?,
            pos: exec.alloc_u32(1)?,
        })
    }
}

impl Work {
    fn bytes() -> u64 {
        Tx::bytes(MAX_ROWS, FLOW_SPLITS)
            + 4 * (CODEC_BATCH * (COND_STRIDE + D + 32 * 3) + MODULATION + D + 32 + MAX_ROWS) as u64
    }

    fn new(exec: &GpuExecutor) -> Result<Self, GpuError> {
        Ok(Self {
            tx: Tx::new(exec, MAX_ROWS, FLOW_SPLITS)?,
            cond: exec.alloc(CODEC_BATCH * COND_STRIDE)?,
            modulation: exec.alloc(MODULATION)?,
            flow: exec.alloc(CODEC_BATCH * D)?,
            tmp: exec.alloc(D)?,
            noise: exec.alloc(CODEC_BATCH * 32)?,
            latents: exec.alloc(CODEC_BATCH * 32)?,
            prev: exec.alloc(32)?,
            ids: exec.alloc_u32(MAX_ROWS)?,
            handoff: exec.alloc(CODEC_BATCH * 32)?,
        })
    }
}

/// A decoder batch's rows at Mimi's transformer.
const CODEC_TX_ROWS: usize = CODEC_BATCH * CODEC_ROWS;

impl CodecWork {
    fn bytes() -> u64 {
        Tx::bytes(CODEC_TX_ROWS, CODEC_SPLITS)
            + 4 * (CODEC_BATCH * (32 * 2 + D + 2 * FRAME_SAMPLES) + 6 * CODEC_PLANE) as u64
    }

    fn new(exec: &GpuExecutor) -> Result<Self, GpuError> {
        Ok(Self {
            tx: Tx::new(exec, CODEC_TX_ROWS, CODEC_SPLITS)?,
            lat: exec.alloc(CODEC_BATCH * 32)?,
            denorm: exec.alloc(CODEC_BATCH * 32)?,
            proj: exec.alloc(CODEC_BATCH * D)?,
            p1: exec.alloc(CODEC_PLANE)?,
            p2: exec.alloc(CODEC_PLANE)?,
            p3: exec.alloc(CODEC_PLANE)?,
            stage: exec.alloc(3 * CODEC_PLANE)?,
            pcm: [
                exec.alloc(CODEC_BATCH * FRAME_SAMPLES)?,
                exec.alloc(CODEC_BATCH * FRAME_SAMPLES)?,
            ],
        })
    }
}

impl GpuPhonon {
    /// Load the checkpoint directory (`model.safetensors`, the default voice,
    /// `tokenizer.json`). Everything a request touches is allocated here.
    pub fn load(exec: Arc<GpuExecutor>, path: &Path) -> Result<Self, GpuModelError> {
        if !exec.has_phonon() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates the Phonon lane (slots 908-915) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        let bad = |e: String| GpuModelError::Unsupported(format!("Phonon: {e}"));
        let ck = Checkpoint::open(path).map_err(bad)?;
        let tokenizer = GgufTokenizer::from_hf_dir(path).map_err(|e| bad(e.to_string()))?;
        if tokenizer.vocab_size != 4000 {
            return Err(bad(
                "tokenizer must have the checkpoint's 4000-token vocabulary".into(),
            ));
        }
        // the expanded F32 inventory bounds the resident planes from above
        let expanded: u64 = ck.tensors.values().map(|v| 4 * v.len() as u64).sum();
        let caches_bytes = 4 * 2 * D as u64 * (6 * FLOW_CACHE + 2 * CODEC_CACHE) as u64;
        exec.vram_load_gate(
            expanded + caches_bytes + Work::bytes() + CodecWork::bytes() + (16 << 20),
            "Phonon",
        )
        .map_err(GpuModelError::WontFit)?;
        // one stream, one owning thread, captured graphs - must precede
        // every alloc
        exec.disable_event_tracking();
        let before = exec.synced_mem_used();
        let voice_rows = ck.voice_frames;
        let voice_in = ck.voice.clone();
        let mut l = Loader {
            exec: &exec,
            ck,
            bytes: 0,
        };
        let c = "flow_lm.condition_provider.conditioners";
        let speakers = l.take(&format!("{c}.num_speakers.embed.weight"))?;
        let speaker_proj = l.mat(&format!("{c}.num_speakers.output_proj"))?;
        let duration_proj = l.mat(&format!("{c}.duration_delta.output_proj"))?;
        let voice_proj = l.mat(&format!("{c}.speaker_wavs.output_proj"))?;
        let embed = l.vec(&format!("{c}.transcript_in_segment.embed.weight"))?;
        let bos_emb = l.take("flow_lm.bos_emb")?;
        let input_linear = l.mat("flow_lm.input_linear")?;
        let p = "flow_lm.flow_net";
        let mut times = Vec::new();
        for i in 0..2 {
            let t = format!("{p}.time_embed.{i}");
            times.push((
                l.take(&format!("{t}.freqs"))?,
                l.lin(&format!("{t}.mlp.0"))?,
                l.lin(&format!("{t}.mlp.2"))?,
                l.vec(&format!("{t}.mlp.3.alpha"))?,
            ));
        }
        // every residual block's adaLN reads the same conditioning vector:
        // one stacked product (independent rows, unchanged reductions)
        let (mut mw, mut mb) = (Vec::new(), Vec::new());
        for i in 0..7 {
            let n = match i {
                6 => format!("{p}.final_layer.adaLN_modulation.1"),
                _ => format!("{p}.res_blocks.{i}.adaLN_modulation.1"),
            };
            mw.extend(l.take(&format!("{n}.weight"))?);
            mb.extend(l.take(&format!("{n}.bias"))?);
        }
        debug_assert_eq!(mb.len(), MODULATION);
        let modulation = Lin {
            w: l.plane(&mw)?,
            b: Some(l.up(&mb)?),
        };
        let blocks = (0..6)
            .map(|i| {
                let b = format!("{p}.res_blocks.{i}");
                Ok(Block {
                    ln: l.norm(&format!("{b}.in_ln"))?,
                    mlp0: l.lin(&format!("{b}.mlp.0"))?,
                    mlp2: l.lin(&format!("{b}.mlp.2"))?,
                })
            })
            .collect::<Result<Vec<_>, GpuModelError>>()?;
        let input_proj = l.lin(&format!("{p}.input_proj"))?;
        // cond_embed's rows then out_eos's row: one [513][512] plane
        let mut ce = l.take(&format!("{p}.cond_embed.weight"))?;
        ce.extend(l.take("flow_lm.out_eos.weight")?);
        let cond_eos_w = l.plane(&ce)?;
        let cond_bias = l.take(&format!("{p}.cond_embed.bias"))?;
        let eos_bias = l.take("flow_lm.out_eos.bias")?;
        let final_linear = l.lin(&format!("{p}.final_layer.linear"))?;
        let flow_layers = l.layers("flow_lm.transformer", 6, false)?;
        let out_norm = l.norm("flow_lm.out_norm")?;
        let mut stats = l.take("flow_lm.emb_std")?;
        stats.extend(l.take("flow_lm.emb_mean")?);
        let stats = l.up(&stats)?;
        let quant_proj = l.vec("mimi.quantizer.output_proj.weight")?;
        let upsample = l.vec("mimi.upsample.convtr.convtr.weight")?;
        let codec_layers = l.layers("mimi.decoder_transformer.transformer", 2, true)?;
        let mut convs = Vec::new();
        for (name, cin, cout, width, stride, path) in CONVS {
            let n = format!("mimi.decoder.{name}");
            let w = l.take(&format!("{n}.weight"))?;
            let b = l.take(&format!("{n}.bias"))?;
            let (w, b, k) = if stride > 1 {
                let bias: Vec<f32> = (0..stride).flat_map(|_| b.iter().copied()).collect();
                (convtr_rows(&w, cin, cout, stride), bias, 2)
            } else {
                (conv_rows(&w, cin, cout, width), b, width)
            };
            convs.push(Conv {
                path,
                w: l.up(&w)?,
                b: l.up(&b)?,
                hist: exec.alloc((k - 1).max(1) * cin)?,
                cin,
                cout,
                k,
                stride,
            });
        }
        let weight_bytes = l.bytes;
        let workspace_bytes = Work::bytes() + CodecWork::bytes() + caches_bytes;
        let mut ws = Work::new(&exec)?;
        let cw = CodecWork::new(&exec)?;
        // the decoder lane: the same context on a least-priority stream
        let codec = Arc::new(exec.fork_stream()?);
        let flow_caches = caches(&exec, 6, FLOW_CACHE)?;
        let codec_caches = caches(&exec, 2, CODEC_RING)?;

        // --- constants: condition = speakers(embed[1]) + duration([1; 0]),
        // BOS = input_linear(bos_emb) + condition
        let none = PhExtra::default;
        let e = |v: &[f32]| exec.to_device(v);
        let mut condition = exec.alloc(D)?;
        let spk = e(&speakers[16..32])?;
        let tmp_c = &mut ws.tmp;
        exec.ph_linear(
            &spk,
            0,
            PhPro::None,
            &speaker_proj,
            None,
            PhEpi::None,
            tmp_c,
            0,
            (1, 16, D),
            LN_EPS,
            none(),
        )?;
        let dur = e(&[vec![1.0; 64], vec![0.0; 64]].concat())?;
        exec.ph_linear(
            &dur,
            0,
            PhPro::None,
            &duration_proj,
            None,
            PhEpi::Res(Some(&ws.tmp)),
            &mut condition,
            0,
            (1, 128, D),
            LN_EPS,
            none(),
        )?;
        let mut bos = exec.alloc(D)?;
        let bos_in = e(&bos_emb)?;
        exec.ph_linear(
            &bos_in,
            0,
            PhPro::None,
            &input_linear,
            None,
            PhEpi::Res(Some(&condition)),
            &mut bos,
            0,
            (1, 32, D),
            LN_EPS,
            none(),
        )?;
        // time = mean over i in {0, 1} of RMSNorm(mlp.2(silu(mlp.0(
        // [cos(i f) ; sin(i f)])))) * alpha
        let mut time = exec.alloc(D)?;
        let mut parts = Vec::new();
        for (i, (freqs, mlp0, mlp2, alpha)) in times.iter().enumerate() {
            let f = e(freqs)?;
            let mut arg = exec.alloc(256)?;
            exec.ph_point(Some(&f), 0, None, &mut arg, 0, 256, 128, 4, i as f32)?;
            let mut h = exec.alloc(D)?;
            exec.ph_linear(
                &arg,
                0,
                PhPro::None,
                &mlp0.w,
                mlp0.b.as_ref(),
                PhEpi::Silu,
                &mut h,
                0,
                (1, 256, D),
                LN_EPS,
                none(),
            )?;
            let mut g = exec.alloc(D)?;
            exec.ph_linear(
                &h,
                0,
                PhPro::None,
                &mlp2.w,
                mlp2.b.as_ref(),
                PhEpi::None,
                &mut g,
                0,
                (1, D, D),
                LN_EPS,
                none(),
            )?;
            let mut t = exec.alloc(D)?;
            exec.rmsnorm_batch(&g, alpha, &mut t, D, 1e-5, 1)?;
            parts.push(t);
        }
        let mut sum = exec.alloc(D)?;
        exec.ph_point(
            Some(&parts[0]),
            0,
            Some(&parts[1]),
            &mut sum,
            0,
            D,
            D,
            0,
            1.0,
        )?;
        let half = e(&[vec![0.5; D], vec![0.0; D]].concat())?;
        exec.ph_point(Some(&sum), 0, Some(&half), &mut time, 0, D, D, 3, 1.0)?;
        // fold the per-model constants into the products that consume them:
        // cond_embed's bias + time (silu follows) beside the EOS bias, and
        // input_linear's condition
        let mut ce_bias = exec.alloc(D + 1)?;
        let cb = e(&cond_bias)?;
        exec.ph_point(Some(&cb), 0, Some(&time), &mut ce_bias, 0, D, D, 0, 1.0)?;
        exec.copy_region(&e(&eos_bias)?, 0, &mut ce_bias, D, 1)?;
        let mut lin_bias = exec.alloc(D)?;
        exec.copy_region(&condition, 0, &mut lin_bias, 0, D)?;

        // --- the voice prompt: projected, then its K/V cached at positions
        // 0..voice_rows (the rows no request writes)
        let mut voice = exec.alloc(voice_rows.max(1) * D)?;
        if voice_rows > 0 {
            let vin = e(&voice_in)?;
            let PhWeight::Bf16(vp) = &voice_proj else {
                return Err(bad("speaker_wavs.output_proj must be BF16".into()));
            };
            exec.diar_gemm(
                (&vin, 0),
                DiarWeights::Bf16(vp),
                None,
                &mut voice,
                (D, D, voice_rows),
                DiarEpi::Store,
                None,
            )?;
        }
        let flow = FlowLm {
            layers: flow_layers,
            caches: flow_caches,
            out_norm,
            input_linear: Lin {
                w: input_linear,
                b: Some(lin_bias),
            },
            input_proj,
            cond_eos: Lin {
                w: cond_eos_w,
                b: Some(ce_bias),
            },
            modulation,
            blocks,
            final_linear,
            embed,
            bos,
            voice,
            voice_rows,
            condition,
            time,
        };
        let mimi = Mimi {
            stats,
            quant_proj,
            upsample,
            up_hist: exec.alloc(D)?,
            layers: codec_layers,
            caches: codec_caches,
            convs,
        };
        // device planes outside weights and the workspace proper: the conv
        // histories, the projected voice, the folded constants
        let extra: usize = mimi.convs.iter().map(|c| c.hist.len()).sum::<usize>()
            + mimi.up_hist.len()
            + flow.voice.len()
            + flow.condition.len()
            + flow.time.len()
            + flow.bos.len()
            + flow.cond_eos.b.as_ref().map_or(0, |b| b.len())
            + flow.input_linear.b.as_ref().map_or(0, |b| b.len());
        let workspace_bytes = workspace_bytes + 4 * extra as u64;
        let mut m = Self {
            exec: exec.clone(),
            codec,
            tokenizer,
            flow,
            mimi,
            ws,
            cw,
            sample_graphs: HashMap::new(),
            decode_graphs: HashMap::new(),
            handed: None,
            taken: None,
            pcm_next: 0,
            flow_pos: 0,
            codec_pos: 0,
            weight_bytes,
            workspace_bytes,
        };
        if voice_rows > 0 {
            exec.copy_region(&m.flow.voice, 0, &mut m.ws.tx.x, 0, voice_rows * D)?;
            exec.upload_u32(&[0], &mut m.ws.tx.pos)?;
            m.wide_flow(voice_rows)?;
        }
        // every batch shape once plainly (first launches do their one-time
        // setup outside any capture), then captured: the sampling graphs on
        // FlowLM's lane, the decode graphs on the decoder's. The rows they
        // touch - FlowLM's past the voice prompt, Mimi's ring and histories -
        // are a request's own scratch, rewritten before it reads them.
        let passes: Vec<Pass> = std::iter::once(Pass {
            first: true,
            frames: 1,
        })
        .chain((1..=CODEC_BATCH).map(|frames| Pass {
            first: false,
            frames,
        }))
        .collect();
        m.flow_pos = voice_rows + 1;
        for graph in [false, true] {
            for &pass in &passes {
                let noises = vec![[0f32; 32]; pass.frames];
                m.launch_sample(&noises, pass.first, graph)?;
                let d = m.launch_decode(pass.frames, graph)?;
                m.read_pcm(&d)?;
            }
        }
        exec.synchronize()?;
        m.codec.synchronize()?;
        m.flow_pos = voice_rows;
        m.codec_pos = 0;
        // what the load actually holds on the device (weights, caches,
        // workspace, the graphs' own allocations)
        let resident = before
            .zip(exec.synced_mem_used())
            .map(|(b, a)| a.saturating_sub(b));
        if let Some(r) = resident {
            m.workspace_bytes = r.saturating_sub(weight_bytes).max(m.workspace_bytes);
        }
        tracing::info!(
            weight_bytes,
            workspace_bytes = m.workspace_bytes,
            resident_bytes = resident,
            graphs = m.sample_graphs.len() + m.decode_graphs.len(),
            "Phonon FlowLM + Mimi ready on CUDA"
        );
        Ok(m)
    }
}
