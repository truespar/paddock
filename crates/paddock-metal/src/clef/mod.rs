//! Native Clef/Clef Flash: one packed Qwen backbone pass and its joint decision
//! head, not token generation. The shared ClefDecider owns batching/cancellation.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use paddock_engine::clef_decision::{ClefBackend, ClefInfo, ClefLogits, ClefRequest};
use paddock_models::clef::{ClefBlock, ClefConfig};
use std::path::Path;

mod forward;
mod gguf;
mod head;
mod load;
mod plan;
#[cfg(test)]
mod projection_tests;
mod quant;
#[cfg(test)]
mod tests;
mod vision;
pub(crate) use vision::resize::Cache as ImageResizeCache;
#[cfg(test)]
mod vision_tests;
mod workspace;

const MAX_ROWS: usize = 16384;
const MAX_QUESTIONS: usize = 1024;
const MAX_OPTIONS: usize = 4096;
const MAX_REQUESTS: usize = 256;

fn error(message: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Clef: {}", message.into()))
}
fn upload(d: &MetalDevice, values: &[f32]) -> Result<Buffer> {
    d.upload(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn point(c: &Commands<'_>, name: &str, buffers: &[&Buffer], params: &[u32], n: usize) {
    c.dispatch(name, buffers, params, [n.div_ceil(256), 1, 1], 256);
}
struct Linear {
    weight: quant::Plane,
    bias: Option<Buffer>,
    k: usize,
    n: usize,
}
impl Linear {
    fn parts(&self, c: &Commands<'_>, x: &Buffer, y: &Buffer, rows: usize, epi: u32) {
        debug_assert!(self.bias.is_none());
        if self.weight.kind != 0 {
            self.quantized(c, x, y, rows, epi, true);
            return;
        }
        // M5 election from alternating projection and full-pass measurements.
        // Shape, never neighboring requests, selects the weight stripe width.
        // Older GPUs keep their established route until separately measured.
        let kernel = if c.tensor_accelerated() && self.n >= 1024 {
            if self.k >= 8192 {
                "clef_mm_parts_k256"
            } else {
                "clef_mm_parts_k1024"
            }
        } else {
            "clef_mm_parts"
        };
        #[cfg(test)]
        let kernel = if forward::UNBLOCKED_FOR_TEST.with(|v| v.get()) {
            "clef_mm_parts"
        } else {
            kernel
        };
        #[cfg(test)]
        let kernel = if forward::LINEAR_WALK_FOR_TEST.with(|v| v.get()) {
            "clef_mm_parts_linear"
        } else {
            kernel
        };
        c.dispatch(
            kernel,
            &[&self.weight.data, x, y],
            &[self.k as u32, self.n as u32, rows as u32, epi],
            [self.n.div_ceil(64), rows.div_ceil(64), 1],
            128,
        );
    }
    // epi: 0 store, 1 residual, 2 exact-erf GELU, 3 interleaved SwiGLU.
    fn run(&self, c: &Commands<'_>, x: &Buffer, y: &Buffer, rows: usize, epi: u32) {
        if self.weight.kind != 0 {
            self.quantized(c, x, y, rows, epi, false);
            return;
        }
        c.dispatch(
            "clef_mm",
            &[&self.weight.data, x, y, self.bias.as_ref().unwrap_or(x)],
            &[
                self.k as u32,
                self.n as u32,
                rows as u32,
                u32::from(self.bias.is_some()),
                epi,
            ],
            [self.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
}
struct Norm {
    weight: Buffer,
    bias: Option<Buffer>,
    width: usize,
    eps: f32,
}
impl Norm {
    fn run(&self, c: &Commands<'_>, x: &Buffer, y: &Buffer, rows: usize) {
        c.dispatch(
            "clef_norm",
            &[x, &self.weight, self.bias.as_ref().unwrap_or(x), y],
            &[
                self.width as u32,
                self.eps.to_bits(),
                u32::from(self.bias.is_some()),
            ],
            [rows, 1, 1],
            256,
        );
    }
}
struct Delta {
    qkv: Linear,
    z: Linear,
    ab: Linear,
    out: Linear,
    conv: Buffer,
    a: Buffer,
    dt: Buffer,
    norm: Buffer,
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    qnorm: Buffer,
    knorm: Buffer,
}
enum Mixer {
    Delta(Delta),
    Attention(Attention),
}
struct Layer {
    norm: Norm,
    post: Norm,
    mixer: Mixer,
    gate_up: Linear,
    down: Linear,
}

pub struct Clef {
    device: MetalDevice,
    config: ClefConfig,
    embed: quant::Plane,
    lexical: quant::Plane,
    tiled_heads: bool,
    // MLX uses RMS epsilon for recurrent Q/K; GGUF/HF uses L2 epsilon.
    rms_qk: bool,
    rope: Buffer,
    layers: Vec<Layer>,
    norm: Norm,
    head: head::Head,
    ws: workspace::Workspace,
    weight_bytes: u64,
    workspace_bytes: u64,
    vision: Option<vision::Vision>,
    #[cfg(test)]
    trace: Option<Buffer>,
}
impl Clef {
    pub fn load(dir: &Path, budget: Option<u64>) -> Result<Self> {
        if dir.extension().is_some_and(|x| x == "gguf") {
            return gguf::load(dir, budget);
        }
        load::load(dir, budget)
    }
    pub fn load_with_companion(
        path: &Path,
        companion: Option<&Path>,
        budget: Option<u64>,
    ) -> Result<Self> {
        if path.extension().is_some_and(|x| x == "gguf") {
            return gguf::load_with_companion(path, companion, budget);
        }
        Self::load(path, budget)
    }
    pub fn vision_config(
        &self,
    ) -> Option<&(
        paddock_models::clef::ClefVisionConfig,
        paddock_models::clef::ClefImageConfig,
    )> {
        self.config
            .vision
            .as_ref()
            .filter(|_| self.vision.is_some())
    }
    pub fn forward(&mut self, requests: &[ClefRequest<'_>]) -> Result<ClefLogits> {
        objc2::rc::autoreleasepool(|_| {
            let plan = plan::Plan::new(requests, self.config.vocab)?;
            let has_images = requests.iter().any(|r| !r.images.is_empty());
            if has_images && self.vision.is_none() {
                return Err(error("load the vision companion to read images"));
            }
            if let Some((cfg, _)) = self.vision_config() {
                for r in requests {
                    for im in r.images {
                        if r.ids[im.row..im.row + im.tokens()]
                            .iter()
                            .any(|&id| id != cfg.image_token)
                        {
                            return Err(error("image rows must contain image-pad tokens"));
                        }
                    }
                }
            }
            self.ws.metadata.write(&plan);
            // Tower scratch is bounded by this pass, freed before the backbone
            // runs. Images never retain a second copy of the language weights.
            let mut encoded = Vec::new();
            if let Some(vision) = &self.vision {
                let mut offset = 0;
                for r in requests {
                    for im in r.images {
                        encoded.push((
                            offset + im.row,
                            im.tokens(),
                            vision.encode(&self.device, im)?,
                        ));
                    }
                    offset += r.ids.len();
                }
            }
            let c = self.device.begin()?;
            self.backbone(&c, &plan, &encoded);
            self.head_forward(&c, &plan);
            c.finish()?;
            // SAFETY: the sole execution thread has fenced all GPU writes.
            let values = unsafe { self.ws.head.logits.read_f32(0, plan.options.len() / 2) };
            if values.iter().any(|v| !v.is_finite()) {
                return Err(error("non-finite decision logits"));
            }
            let mut next = 0;
            Ok(requests
                .iter()
                .map(|r| {
                    r.questions
                        .iter()
                        .map(|q| {
                            let start = next;
                            next += q.options.len();
                            values[start..next].to_vec()
                        })
                        .collect()
                })
                .collect())
        })
    }
}
impl ClefBackend for Clef {
    fn info(&self) -> ClefInfo {
        ClefInfo {
            max_rows: MAX_ROWS,
            max_questions: MAX_QUESTIONS,
            max_options: MAX_OPTIONS,
            max_requests: MAX_REQUESTS,
            vocab: self.config.vocab,
            weight_bytes: self.weight_bytes,
            workspace_bytes: self.workspace_bytes,
            images: self.vision.is_some(),
        }
    }
    fn forward(&mut self, reqs: &[ClefRequest<'_>]) -> std::result::Result<ClefLogits, String> {
        Clef::forward(self, reqs).map_err(|e| e.to_string())
    }
}
