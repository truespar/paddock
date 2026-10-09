//! SAM 3 image encoder bring-up. Shared backbone and both feature pyramids,
//! not yet a complete mask-serving backend. Do not advertise catalog/HTTP
//! support before the detector, click decoder and video graph are qualified.
//! CUDA's existing family is the arithmetic/layout contract, Meta's saved
//! FP32 and shipped BF16 outputs the independent numerical qualification.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use paddock_models::sam3::Sam3VisionConfig;
use std::path::Path;

#[cfg(test)]
mod golden_tests;
mod image;
#[cfg(test)]
mod image_tests;
#[cfg(test)]
mod input_golden_tests;
mod load;
mod memory;
pub use image::Sam3InputKind;
#[cfg(test)]
mod tests;
mod vision;

fn error(message: impl Into<String>) -> MetalError {
    MetalError::Model(message.into())
}
fn upload(d: &MetalDevice, values: &[f32]) -> Result<Buffer> {
    d.upload(
        &values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn point(c: &Commands<'_>, name: &str, buffers: &[&Buffer], params: &[u32], n: usize) {
    c.dispatch(name, buffers, params, [n.div_ceil(256), 1, 1], 256);
}
struct Norm {
    w: Buffer,
    b: Buffer,
}
struct Matrix {
    w: Buffer,
    k: usize,
    n: usize,
}
struct Conv {
    w: Matrix,
    b: Buffer,
}
struct Block {
    n1: Norm,
    n2: Norm,
    qkv: Conv,
    out: Conv,
    up: Conv,
    down: Conv,
}
struct Neck {
    up4: [Conv; 2],
    up2: Conv,
    proj1: [Conv; 3],
    proj2: [Conv; 3],
}

struct Workspace {
    pixels: Buffer,
    patches: Buffer,
    x: Buffer,
    norm: Buffer,
    projection: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention: Buffer,
    wide: Buffer,
    raster: Buffer,
    temp: Buffer,
    half_a: Buffer,
    half_b: Buffer,
    det: [Buffer; 3],
    trk: [Buffer; 3],
    s0: Buffer,
    s1: Buffer,
}

/// A component, not a serving model. It accepts already-resized RGB input
/// exactly like CUDA's GpuSam3Vision; image/video resizing is a separate seam.
pub struct Sam3Vision {
    device: MetalDevice,
    cfg: Sam3VisionConfig,
    patch: Matrix,
    pos: Buffer,
    norm: Norm,
    blocks: Vec<Block>,
    det: Neck,
    trk: Neck,
    s0: Conv,
    s1: Conv,
    rope: [Buffer; 2],
    tiles: [Buffer; 2],
    ws: Workspace,
    weight_bytes: u64,
    workspace_bytes: u64,
    encoded: bool,
    tracker: bool,
    input: image::Input,
}

#[derive(Clone, Copy, Debug)]
pub enum Sam3VisionPlane {
    Trunk,
    Detector(usize),
    Tracker(usize),
    TrackerSkip0,
    TrackerSkip1,
}

impl Sam3Vision {
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes + self.input.bytes()
    }
    pub fn config(&self) -> &Sam3VisionConfig {
        &self.cfg
    }
}
