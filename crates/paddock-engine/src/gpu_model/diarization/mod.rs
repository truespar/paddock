//! Nemotron 3 Diarization on CUDA - NVIDIA's streaming Sortformer v3: who
//! spoke when, eight overlapping speakers at 10-ms frames, no transcription.
//! The pass is in `packs/cuda/src/diarization.cuh`'s header; the streaming
//! windows, the arrival-order speaker cache and the service around them are
//! the shared `crate::diarization` (the Metal lane drives the same ones), so
//! this module only holds weights, the resident planes and one window's pass.
//!
//! Precision class: F32 activations against the checkpoint's own weights.
//! The encoder's projections (and the stacked-frame and encoder-to-head
//! ones) stay as stored - BF16 rows, or Q8_0 repacked at load to int8 rows
//! and F32 block scales - and run on a GEMM that splits the activation three
//! ways in bf16 against the exact bf16 weight (diarization.cuh 722): the
//! BF16x6 sum with its three zero products left out, F32 class at half the
//! mma of 3xTF32 and the stored bytes resident. The head's convolution and
//! two small layers (the GGUF keeps them F16/F32) run widened to F32 on the
//! house 3xTF32 GEMM (kumo.cuh 698). Attention is 3xTF32 flash attention on
//! the tensor cores (diarization.cuh 723); LayerNorm, the rope (in the qkv
//! projection's epilogue), softmax, GELU and the frontend's FFT are F32
//! CUDA-core arithmetic in the Transformers reference's operation order.
//! Every pass of a window launches as a programmatic dependent. Nothing is
//! narrowed: the Metal lane's BF16
//! arithmetic contract (MLX Audio's) is a Metal-side choice, and this lane's
//! reference is the model's F64 evaluation instead.

mod forward;

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::diarization::checkpoint::{self, Checkpoint, Format, LAYERS};

use crate::gpu::GpuExecutor;

pub use crate::gpu_model::gpt_oss::GpuModelError;

/// Encoder rows of the largest window: speaker cache, FIFO, chunk and right
/// context of the widest preset (paddock_models::diarization's bound).
pub const MAX_ROWS: usize = 684;
/// Samples the frontend's PCM plane holds (`AudioWindow::validate`'s bound).
const MAX_PCM: usize = 16_000 * 64;

/// A projection's weights on the device.
enum Weights {
    /// widened to F32 (the head), the house 3xTF32 GEMM's operand
    F32(CudaSlice<f32>),
    /// BF16 rows as stored
    Bf16(CudaSlice<u8>),
    /// Q8_0 repacked: int8 rows and F32 block scales `[K/32][N]`
    Q8(CudaSlice<u8>, CudaSlice<f32>),
}

struct Linear {
    w: Weights,
    b: Option<CudaSlice<f32>>,
    k: usize,
    n: usize,
}

/// Q8_0 blocks (an f16 scale and 32 int8 a block, rows of K/32 blocks) as
/// int8 rows `[N][K]` and F32 scales `[K/32][N]` - a k tile's scales one
/// contiguous row. The values are the file's exactly.
fn repack_q8(blocks: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let (mut q, mut scale) = (vec![0u8; n * k], vec![0f32; k / 32 * n]);
    for (i, blk) in blocks.as_chunks::<34>().0.iter().enumerate() {
        let (row, kb) = (i / (k / 32), i % (k / 32));
        scale[kb * n + row] = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        q[row * k + kb * 32..row * k + kb * 32 + 32].copy_from_slice(&blk[2..]);
    }
    (q, scale)
}

struct Norm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

struct Layer {
    n1: Norm,
    qkv: Linear,
    out: Linear,
    n2: Norm,
    up: Linear,
    down: Linear,
}

/// The resident pass planes, sized once for [`MAX_ROWS`]: a window never
/// allocates, and the ledger the catalog carries is this sum.
struct Workspace {
    pcm: CudaSlice<f32>,
    /// features `[8R][128]` = stacked rows `[R][1024]`; also the uploaded
    /// encoder input `[R][512]`
    input: CudaSlice<f32>,
    x: CudaSlice<f32>,
    n: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    att: CudaSlice<f32>,
    wide: CudaSlice<f32>,
    head: CudaSlice<f32>,
    conv: CudaSlice<f32>,
    up: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    probs: CudaSlice<f32>,
}

impl Workspace {
    fn new(e: &GpuExecutor) -> Result<Self, GpuModelError> {
        let p = |n: usize| e.kumo_plane(MAX_ROWS * n);
        Ok(Self {
            pcm: e.kumo_plane(MAX_PCM)?,
            input: p(1024)?,
            x: p(512)?,
            n: p(512)?,
            qkv: p(1536)?,
            q: p(512)?,
            att: p(512)?,
            wide: p(2048)?,
            head: p(192)?,
            conv: p(576)?,
            up: p(1536)?,
            hidden: p(1536)?,
            probs: p(64)?,
        })
    }

    fn bytes() -> u64 {
        4 * (MAX_PCM + MAX_ROWS * (1024 + 512 * 5 + 1536 * 3 + 2048 + 192 + 576 + 64)) as u64
    }
}

/// One loaded checkpoint and its resident planes.
pub struct GpuDiarization {
    exec: Arc<GpuExecutor>,
    format: Format,
    weight_bytes: u64,
    pre: Linear,
    embed: Norm,
    layers: Vec<Layer>,
    last: Norm,
    proj: Linear,
    conv: Linear,
    hidden: Linear,
    score: Linear,
    /// frontend constants (see `DiarFrontendPlanes`) and the rope's (cos,
    /// sin) for every row position of a window
    window_gpu: CudaSlice<f32>,
    fb_gpu: CudaSlice<f32>,
    spans: CudaSlice<u32>,
    twiddle: CudaSlice<f32>,
    rope: CudaSlice<f32>,
    ws: Workspace,
    /// the frontend+pre-encode time of the window `predict` completes
    pre_seconds: f64,
    window: Vec<f32>,
    filterbank: Vec<f32>,
    silence: Vec<f32>,
}

/// The rope's (cos, sin) at every position of a window, `[MAX_ROWS][32]`
/// pairs: inv_freq = 1 / 10000^(2i / 64) and angle = pos * inv_freq rounded
/// to F32 as the reference forms them, then cos and sin of that F32 angle
/// correctly rounded (evaluated in F64).
fn rope_table() -> Vec<f32> {
    let inv: Vec<f32> = (0..32)
        .map(|i| (1.0 / 10000f64.powf(2.0 * i as f64 / 64.0)) as f32)
        .collect();
    let mut t = Vec::with_capacity(MAX_ROWS * 64);
    for pos in 0..MAX_ROWS {
        for &f in &inv {
            let angle = f64::from(pos as f32 * f);
            t.push(angle.cos() as f32);
            t.push(angle.sin() as f32);
        }
    }
    t
}

impl GpuDiarization {
    /// Load the MLX BF16 directory or the GGUF Q8_0 file. The whole
    /// inventory is checked before the first allocation.
    pub fn load(exec: Arc<GpuExecutor>, path: &Path) -> Result<Self, GpuModelError> {
        if !exec.has_diarization() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates the Nemotron 3 Diarization lane (slots 718-723) - \
                 rebuild or update the pack"
                    .into(),
            ));
        }
        let bad = |e: String| GpuModelError::Unsupported(format!("Nemotron 3 Diarization: {e}"));
        let ck = Checkpoint::open(path).map_err(bad)?;
        // the stored bytes plus the head's F32 widening (the gate's figure;
        // the ledger below is what was actually allocated)
        let stored = ck.validate().map_err(bad)?;
        let window = ck.window().map_err(bad)?;
        let filterbank = ck.filterbank().map_err(bad)?;
        let silence = ck.silence().map_err(bad)?;
        exec.vram_load_gate(
            stored + (8 << 20) + Workspace::bytes(),
            "Nemotron 3 Diarization",
        )
        .map_err(GpuModelError::WontFit)?;
        // one stream, one owning thread - must precede every alloc
        exec.disable_event_tracking();
        let weight_bytes = std::cell::Cell::new(0u64);
        let up = |v: &[f32]| {
            weight_bytes.set(weight_bytes.get() + 4 * v.len() as u64);
            exec.to_device(v)
        };
        let up8 = |v: &[u8]| {
            weight_bytes.set(weight_bytes.get() + v.len() as u64);
            exec.to_device_u8(v)
        };
        let vals = |n: &str, shape: &[usize]| ck.values(n, shape).map_err(bad);
        let norm = |n: &str| -> Result<Norm, GpuModelError> {
            Ok(Norm {
                w: up(&vals(&format!("{n}.weight"), &[512])?)?,
                b: up(&vals(&format!("{n}.bias"), &[512])?)?,
            })
        };
        // `stored`: the encoder projections, kept BF16 / Q8_0 as the
        // checkpoint has them (both published exports store them so); the
        // head's three small layers widen to F32 for the Kumo GEMM
        let linear = |n: &str, k: usize, out: usize, bias: bool, stored: bool| {
            let shape = if k == 576 {
                vec![out, 3, 192]
            } else {
                vec![out, k]
            };
            let name = format!("{n}.weight");
            let raw = ck.raw(&name, &shape).map_err(bad)?;
            let w = match raw.ty {
                checkpoint::BF16 if stored => Weights::Bf16(up8(raw.bytes)?),
                checkpoint::Q8_0 if stored => {
                    let (q, scale) = repack_q8(raw.bytes, out, k);
                    let scale = up(&scale)?;
                    Weights::Q8(up8(&q)?, scale)
                }
                ty if stored => {
                    return Err(bad(format!(
                        "{name}: stored as type {ty}, expected BF16 or Q8_0"
                    )));
                }
                _ => Weights::F32(up(&vals(&name, &shape)?)?),
            };
            Ok::<_, GpuModelError>(Linear {
                w,
                b: if bias {
                    Some(up(&vals(&format!("{n}.bias"), &[out])?)?)
                } else {
                    None
                },
                k,
                n: out,
            })
        };
        let pre = linear("encoder.pre_encode.proj", 1024, 512, false, true)?;
        let embed = norm("encoder.embed_norm")?;
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("encoder.layers.{i}");
            layers.push(Layer {
                n1: norm(&format!("{p}.norm1"))?,
                qkv: linear(&format!("{p}.attn.w_qkv"), 512, 1536, false, true)?,
                out: linear(&format!("{p}.attn.out_proj"), 512, 512, true, true)?,
                n2: norm(&format!("{p}.norm2"))?,
                up: linear(&format!("{p}.ffn.linear1"), 512, 2048, true, true)?,
                down: linear(&format!("{p}.ffn.linear2"), 2048, 512, true, true)?,
            });
        }
        let last = norm("encoder.final_norm")?;
        let s = "sortformer_modules";
        let proj = linear(&format!("{s}.encoder_proj"), 512, 192, true, true)?;
        let conv = linear(&format!("{s}.subpixel_upsample"), 576, 1536, true, false)?;
        let hidden = linear(
            &format!("{s}.first_hidden_to_hidden"),
            192,
            192,
            true,
            false,
        )?;
        let score = linear(&format!("{s}.single_hidden_to_spks"), 192, 8, true, false)?;
        let weight_bytes = weight_bytes.get();
        let mut padded = vec![0f32; 512];
        padded[56..456].copy_from_slice(&window);
        let spans: Vec<u32> = filterbank
            .as_chunks::<257>()
            .0
            .iter()
            .flat_map(|row| {
                let first = row.iter().position(|&v| v != 0.).unwrap_or(0) as u32;
                let end = row.iter().rposition(|&v| v != 0.).map_or(0, |i| i + 1) as u32;
                [first, end.max(first)]
            })
            .collect();
        let twiddle: Vec<f32> = (0..256)
            .flat_map(|i| {
                let (s, c) = (-2.0 * std::f64::consts::PI * i as f64 / 512.).sin_cos();
                [c as f32, s as f32]
            })
            .collect();
        let window_gpu = up(&padded)?;
        let fb_gpu = up(&filterbank)?;
        let spans = exec.to_device_u32(&spans)?;
        let twiddle = up(&twiddle)?;
        let rope = up(&rope_table())?;
        let ws = Workspace::new(&exec)?;
        exec.synchronize()?;
        Ok(Self {
            format: ck.format(),
            exec,
            weight_bytes,
            pre,
            embed,
            layers,
            last,
            proj,
            conv,
            hidden,
            score,
            window_gpu,
            fb_gpu,
            spans,
            twiddle,
            rope,
            ws,
            pre_seconds: 0.,
            window,
            filterbank,
            silence,
        })
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// The resident planes and frontend constants: allocated once at load.
    pub fn workspace_bytes(&self) -> u64 {
        Workspace::bytes() + 4 * (512 + 128 * 257 + 256 + 512 + MAX_ROWS * 64) as u64
    }

    pub fn executor(&self) -> &Arc<GpuExecutor> {
        &self.exec
    }
}

impl crate::diarization::Backend for GpuDiarization {
    fn encode_audio(
        &mut self,
        _fe: &crate::diarization::Frontend,
        w: crate::diarization::AudioWindow<'_>,
    ) -> Result<Vec<f32>, String> {
        Self::encode_audio(self, w).map_err(|e| e.to_string())
    }
    fn pre_encode(&mut self, features: &[f32]) -> Result<Vec<f32>, String> {
        Self::pre_encode(self, features).map_err(|e| e.to_string())
    }
    fn predict(&mut self, x: &[f32], valid: usize) -> Result<(Vec<[f32; 8]>, f64), String> {
        Self::predict(self, x, valid).map_err(|e| e.to_string())
    }
    fn frontend(&self) -> Result<crate::diarization::Frontend, String> {
        crate::diarization::Frontend::new(&self.window, &self.filterbank)
    }
    fn silence(&self) -> &[f32] {
        &self.silence
    }
    fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    fn workspace_bytes(&self) -> u64 {
        Self::workspace_bytes(self)
    }
}
