//! PP-DocLayoutV3 - PaddlePaddle's layout detector, the first stage of the
//! PaddleOCR-VL page pipeline (`PaddlePaddle/PP-DocLayoutV3_safetensors`).
//!
//! An RT-DETR shape at a fixed 800 x 800 input: an HGNetV2-L backbone, a
//! hybrid encoder (one AIFI transformer layer on the stride-32 map, CCFF
//! fusion of three levels, a mask branch), query selection over the
//! encoder's anchors, a six-layer deformable-attention decoder and its heads
//! (classes, boxes, masks, reading order).
//!
//! The parity reference is the authors' own inference (PaddleX on
//! paddlepaddle), which the Transformers port reproduces to float noise when
//! fed the pipeline's pixels: an OpenCV `INTER_CUBIC` resize on uint8
//! (`crate::cv_resize`), `/ 255`, RGB. Golden taps for the stage-by-stage
//! gate come from that port.
//!
//! The checkpoint is F32; the engine runs it in the vision towers' class:
//! BatchNorm folded into each conv at load, f16 weight planes and f16 NHWC
//! activations between GEMMs, f32 accumulate, norms and residuals. Measured
//! on the reference over the battery pages, activations peak near 124 and
//! the folded weights at 4.1, so f16's range is never in question.

mod backbone;
mod decoder;
mod encoder;
pub mod geom;
mod load;
pub mod polygon;
mod post;

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;

use crate::gpu::{GpuExecutor, HalfTensor};

pub use crate::gpu_model::gpt_oss::GpuModelError;
pub use decoder::{DecoderOut, QUERIES};
pub use encoder::EncoderOut;
pub use post::{LABELS, LayoutBox, order_seq, postprocess};

/// The network's input side: every page is resized to this square.
pub const INPUT: usize = 800;

/// The mask prototypes' side (stride 4 over the input).
const MASK_SIDE: usize = INPUT / 4;

/// A conv with its BatchNorm folded in: the GEMM plane over the im2row's
/// `[ky][kx][cin]` taps (padded to `kpad`, a multiple of 8) and the bias.
struct ConvBn {
    w: HalfTensor,
    b: CudaSlice<f32>,
    k: usize,
    stride: usize,
    cin: usize,
    cout: usize,
    kpad: usize,
}

/// A depthwise conv with its BatchNorm folded in, f32 `[c][k * k]`.
struct DwConv {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    k: usize,
    stride: usize,
    c: usize,
}

/// The activation after a conv's BatchNorm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Act {
    None,
    Relu,
    Silu,
}

/// An NHWC half plane with its spatial shape.
pub struct Plane {
    pub data: CudaSlice<f16>,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

/// The four backbone stages a forward exposes (strides 4 / 8 / 16 / 32).
pub struct BackboneOut {
    pub stages: [Plane; 4],
}

/// PP-DocLayoutV3 on the device.
pub struct GpuDocLayout {
    exec: Arc<GpuExecutor>,
    backbone: backbone::Backbone,
    encoder: encoder::Encoder,
    decoder: decoder::Decoder,
    /// device bytes the weights hold
    pub weight_bytes: u64,
}

impl GpuDocLayout {
    /// Load `dir/model.safetensors` (the checkpoint's own layout).
    pub fn load(exec: Arc<GpuExecutor>, dir: &Path) -> Result<Self, GpuModelError> {
        if !exec.has_doclayout() {
            return Err(GpuModelError::Unsupported(
                "the kernel pack has no PP-DocLayoutV3 ops (slots 836-849)".into(),
            ));
        }
        let st = paddock_models::safetensors::ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("pp-doclayout-v3 weights: {e}")))?;
        let mut rd = load::Reader {
            st: &st,
            exec: &exec,
            bytes: 0,
        };
        let backbone = backbone::Backbone::load(&mut rd)?;
        let encoder = encoder::Encoder::load(&mut rd)?;
        let decoder = decoder::Decoder::load(&mut rd)?;
        let weight_bytes = rd.bytes;
        Ok(Self {
            exec,
            backbone,
            encoder,
            decoder,
            weight_bytes,
        })
    }

    /// The backbone over one page already resized to `INPUT` x `INPUT`
    /// (u8 RGB, HWC).
    pub fn backbone(&self, rgb800: &[u8]) -> Result<BackboneOut, GpuModelError> {
        if rgb800.len() != 3 * INPUT * INPUT {
            return Err(GpuModelError::Unsupported(format!(
                "pp-doclayout-v3 takes {INPUT} x {INPUT} RGB, got {} bytes",
                rgb800.len()
            )));
        }
        self.backbone.forward(&self.exec, rgb800)
    }

    /// The backbone, then the hybrid encoder and the mask prototypes.
    pub fn encode(&self, rgb800: &[u8]) -> Result<(BackboneOut, EncoderOut), GpuModelError> {
        let bb = self.backbone(rgb800)?;
        let enc = self.encoder.forward(&self.exec, &bb)?;
        Ok((bb, enc))
    }

    /// One page, any size (u8 RGB, HWC): the pipeline's own 800 x 800
    /// resize (OpenCV's cubic, byte for byte), the network, the postprocess -
    /// the regions in reading order, in the page's pixels.
    pub fn layout(&self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<LayoutBox>, GpuModelError> {
        if w == 0 || h == 0 || rgb.len() != 3 * w * h {
            return Err(GpuModelError::Unsupported(format!(
                "pp-doclayout-v3: a {w} x {h} page needs {} RGB bytes, got {}",
                3 * w * h,
                rgb.len()
            )));
        }
        let small = crate::cv_resize::resize_cubic_rgb8(rgb, w, h, INPUT, INPUT);
        let (_, _, dec) = self.detect(&small)?;
        let cands = post::select(&dec, (w, h));
        let masks = self.masks(&dec, &cands)?;
        let corners: Vec<[f32; 4]> = cands.iter().map(post::Cand::corners).collect();
        let scale = (INPUT as f64 / w as f64, INPUT as f64 / h as f64);
        let outlines = polygon::outlines(&corners, &masks, MASK_SIDE, scale);
        Ok(post::finish(cands, Some(outlines), (w, h)))
    }

    /// The kept candidates' masks as the reference reads them: each query's
    /// row of the last layer's mask logits, `sigmoid > 0.5` (float32), as
    /// 0 / 1 bytes. Only those rows leave the device.
    fn masks(&self, dec: &DecoderOut, cands: &[post::Cand]) -> Result<Vec<Vec<u8>>, GpuModelError> {
        let px = MASK_SIDE * MASK_SIDE;
        if cands.is_empty() {
            return Ok(Vec::new());
        }
        let rows: Vec<u32> = cands.iter().map(|c| c.query() as u32).collect();
        let idx = self.exec.to_device_u32(&rows)?;
        let mut picked = self.exec.alloc(rows.len() * px)?;
        self.exec
            .dl_gather_rows_f32(&dec.masks, &mut picked, &idx, rows.len(), px)?;
        let host = self.exec.to_host(&picked)?;
        Ok(host
            .chunks_exact(px)
            .map(|r| {
                r.iter()
                    .map(|&v| u8::from(1.0f32 / (1.0 + (-v).exp()) > 0.5))
                    .collect()
            })
            .collect())
    }

    pub fn exec(&self) -> &Arc<GpuExecutor> {
        &self.exec
    }

    /// The whole network: encoder, query selection, decoder, heads - what
    /// the postprocess reads, plus the stage taps.
    pub fn detect(
        &self,
        rgb800: &[u8],
    ) -> Result<(BackboneOut, EncoderOut, DecoderOut), GpuModelError> {
        let (bb, enc) = self.encode(rgb800)?;
        let dec = self.decoder.forward(&self.exec, &enc)?;
        Ok((bb, enc, dec))
    }
}
