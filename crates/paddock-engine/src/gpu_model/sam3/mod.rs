//! SAM 3 - Meta's "Segment Anything with Concepts" (`facebook/sam3`).
//!
//! This is the image encoder, the part every SAM 3 task runs once per picture
//! or video frame and shares: the ViT backbone and its two FPN necks, the
//! detector's (concept prompts: find every instance) and the tracker's (clicks,
//! boxes and video memory). The text tower, the detector heads and the tracker
//! land on top of it.
//!
//! The parity reference is Meta's own code (facebookresearch/sam3), run
//! outside Paddock to produce golden tensors; transformers' port differs from
//! it in at least eight places that change outputs, two of them here: the
//! ViT's LayerNorm eps (1e-5, not the config's 1e-6) and the MLP's GELU
//! (Meta's inference fuses fc1 + tanh-approximate GELU into one cuBLASLt
//! call; transformers runs the exact erf form). Every number here comes from
//! Meta's model builder and the checkpoint's own configs.
//!
//! The checkpoint is F32; the engine serves it in the vision towers' class:
//! f16 weight planes (F32 -> f16 rounds to nearest; the loader refuses a plane
//! f16 cannot hold), f16 GEMM operands and an f16 activation interface between
//! GEMMs, f32 residual, norms and accumulate. Measured on Meta's fp32 run over
//! the golden pictures: the largest GEMM output in the tower is 65 and the
//! residual peaks at 317, so f16's range is never in question, and f16 keeps 11
//! significant bits where Meta's own bf16 runtime keeps 8.
//!
//! The tower's token rows live in WINDOW-MAJOR order (`packs/cuda/src/sam3/
//! vit.cuh` has the why): window attention is grouped attention over 576-row
//! groups, global attention one group of 5184, and the tiled position table
//! one broadcast. The order is undone once, at the exit to the necks.

mod bank;
mod checkpoint;
mod decoder;
mod detector;
mod fusion;
mod load;
mod memattn;
mod memory;
mod neck;
mod pipeline;
mod pvs;
mod seg;
mod text;
mod video;
mod video_frames;
mod video_masks;
mod video_plan;
mod vit;

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::sam3::Sam3VisionConfig;

use crate::gpu::{GpuExecutor, HalfTensor};

pub use crate::gpu_model::gpt_oss::GpuModelError;
pub use bank::{BANK_MAX_POINTERS, GpuSam3BankKv, Sam3Bank, Sam3BankPlan};
pub use checkpoint::MetaCheckpoint;
pub use detector::{DetGeom, GpuSam3Detector, Sam3Detections};
pub use fusion::Sam3Box;
pub use memattn::{GpuSam3MemAttn, MEMATTN_MAX_KEYS};
pub use memory::{GpuSam3MemEnc, MEM_DIM};
pub use pipeline::{
    GpuSam3, Sam3Click, Sam3Fail, Sam3Instance, Sam3Output, Sam3PointRequest, Sam3Request,
    Sam3Timings,
};
pub use pvs::{GpuSam3Pvs, PVS_MAX_POINTS, PvsFeatures, PvsMask, PvsPoint, PvsPrompt, PvsResult};
pub use text::GpuSam3Text;
pub use video::{MAX_CONCEPTS, Sam3VideoDets, Sam3VideoFrame, Sam3VideoSession, Sam3VideoStep};
pub use video_frames::GpuSam3FrameIn;
pub use video_masks::{GpuSam3VideoMasks, LOW_PX, LOW_RES, Sam3Ious, Sam3VideoObject};
pub use video_plan::{
    DET_NMS_THRESH, HOTSTART_DELAY, SCORE_THRESHOLD_DETECTION, Sam3FrameRecord, Sam3MaskSource,
    Sam3Plan, Sam3PlanIn, Sam3RecordObject, Sam3VideoPlanner,
};

/// LayerNorm weight + bias.
struct Norm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

/// One pre-LN block. No LayerScale in SAM 3: the residual seam is the dense
/// lane's LayerScale seam handed a ones vector, which multiplies exactly.
struct Block {
    ln1: Norm,
    /// q|k|v stacked row-wise, q and k rows permuted inside each head for the
    /// rotate-half rope (see [`RopeTable`])
    wqkv: HalfTensor,
    bq: CudaSlice<f32>,
    bk: CudaSlice<f32>,
    bv: CudaSlice<f32>,
    /// the three biases end to end, for the fused q|k|v landing (slot 776)
    bqkv: CudaSlice<f32>,
    wo: HalfTensor,
    bo: CudaSlice<f32>,
    ln2: Norm,
    fc1: HalfTensor,
    fc1_b: CudaSlice<f32>,
    fc2: HalfTensor,
    fc2_b: CudaSlice<f32>,
    /// attends over the whole grid instead of inside its window
    global: bool,
}

/// A 1x1 conv (a GEMM) or the 3x3's GEMM over a tap-major im2row.
struct Conv {
    w: HalfTensor,
    b: CudaSlice<f32>,
}

/// A 2x2 / stride-2 transposed conv as its GEMM (C_in -> 4 * C_out, rows tap
/// major) plus the depth-to-space seam's bias.
struct ConvT {
    w: HalfTensor,
    b: CudaSlice<f32>,
    cout: usize,
}

/// One neck: three levels at 4x, 2x and 1x the patch grid (288, 144, 72 at
/// 1008 px), each ending in a 1x1 then a 3x3 conv to `fpn_dim` channels.
struct Neck {
    /// convT -> GELU -> convT
    x4_up: [ConvT; 2],
    x2_up: ConvT,
    /// [x4, x2, x1]
    proj1: [Conv; 3],
    proj2: [Conv; 3],
}

/// Every buffer a pass touches, allocated once for `cap` pictures.
struct Workspace {
    cap: usize,
    px: CudaSlice<u8>,
    /// the patch GEMM's input rows, K padded to `kp`
    rows16: CudaSlice<f16>,
    /// the residual stream - the one tower plane that stays f32
    x: CudaSlice<f32>,
    /// pre-norm landing; after the last block, the raster the necks read
    n16: CudaSlice<f16>,
    qkv: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    /// attention landing (the o projection's input)
    att: CudaSlice<f16>,
    proj: CudaSlice<f16>,
    /// fc1's landing after its fused bias + GELU, fc2's input
    ff: CudaSlice<f16>,
    /// only where the f16-landing GEMM is not the device's elected route: the
    /// f32 plane those GEMMs land on before converting
    land32: Option<CudaSlice<f32>>,
    /// f32 GEMM landings: the stem, every convT, every 1x1
    g32: CudaSlice<f32>,
    y32: CudaSlice<f32>,
    /// convT outputs at f16 (two: the x4 chain holds one while making the next)
    h16a: CudaSlice<f16>,
    h16b: CudaSlice<f16>,
    /// the 3x3 im2row, sized for the x4 level
    col16: CudaSlice<f16>,
    /// FPN levels, [pics][side^2][fpn_dim] f32, raster
    det: [CudaSlice<f32>; 3],
    trk: [CudaSlice<f32>; 3],
    /// the tracker's conv_s0 / conv_s1 outputs off its x4 / x2 levels
    trk_s0: CudaSlice<f32>,
    trk_s1: CudaSlice<f32>,
    bytes: u64,
}

/// SAM 3's image encoder, resident.
pub struct GpuSam3Vision {
    exec: Arc<GpuExecutor>,
    cfg: Sam3VisionConfig,
    /// padded patch-GEMM width (588 -> 592)
    kp: usize,
    patch_w: HalfTensor,
    /// the learned absolute position table [window^2][hidden]; one copy per
    /// window, which is the tiling, in window-major rows
    pos: CudaSlice<f32>,
    ln_pre: Norm,
    rope_win: (CudaSlice<f32>, CudaSlice<f32>),
    rope_glob: (CudaSlice<f32>, CudaSlice<f32>),
    blocks: Vec<Block>,
    /// the identity LayerScale the shared residual seam multiplies by
    ones: CudaSlice<f32>,
    det_neck: Neck,
    trk_neck: Neck,
    /// the tracker's mask decoder reads its high-resolution levels through
    /// these two 1x1s (256 -> 32 at 4x, 256 -> 64 at 2x); applied at encode
    /// time, as Meta's processor does
    conv_s0: Conv,
    conv_s1: Conv,
    ws: Workspace,
    weight_bytes: u64,
    /// the input is a video frame as Meta's frame loader leaves it (fp16
    /// storage, fp16 normalize), not a picture as its processor does
    video_frames: bool,
}

/// What [`GpuSam3Vision::encode`] left on the device for each picture. The
/// planes live in the encoder's workspace until the next pass; the readers
/// below copy them out (the gates' view - the heads read them in place).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sam3Plane {
    /// the trunk output, f32, window-major rows [5184][1024]
    Trunk,
    /// detector neck level 0..3 (x4, x2, x1), f32 raster [side^2][256]
    Det(usize),
    /// tracker neck level 0..3, same shape
    Trk(usize),
    /// conv_s0 over the tracker's x4 level, [288^2][32]
    TrkS0,
    /// conv_s1 over the tracker's x2 level, [144^2][64]
    TrkS1,
}

impl GpuSam3Vision {
    pub fn config(&self) -> &Sam3VisionConfig {
        &self.cfg
    }
    /// Normalize the input as a video frame from here on (`true`, slot 801)
    /// or as a picture (`false`, the default). Meta's two loaders round
    /// differently: 140 of the 256 levels land a half step apart.
    pub fn set_video_frames(&mut self, on: bool) {
        self.video_frames = on;
    }
    /// Most pictures one [`Self::encode`] call takes.
    pub fn max_batch(&self) -> usize {
        self.ws.cap
    }
    /// Bytes one input picture must be: side x side x 3, u8 RGB HWC.
    pub fn picture_bytes(&self) -> usize {
        self.cfg.image_size * self.cfg.image_size * self.cfg.channels
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.ws.bytes
    }
    /// Side of FPN level `l` (0: 4x, 1: 2x, 2: 1x the patch grid).
    pub fn level_side(&self, l: usize) -> usize {
        (self.cfg.grid() * 4) >> l
    }
}

/// Meta's axial rope (`vitdet.py::compute_axial_cis`) as cos/sin tables
/// `[rows][head_dim / 2]`, rows in window-major order.
///
/// Meta's form is complex: head dims (2i, 2i+1) are one number, rotated by
/// angle i; angles 0..hd/4 read the column coordinate and hd/4..hd/2 the row,
/// both with frequency `1 / theta^(4k / hd)`. The engine's split kernel
/// rotates the pair (j, j + hd/2) instead (rotate-half addressing), so the
/// loader permutes q and k rows inside each head - Meta dim 2j to j, 2j + 1 to
/// j + hd/2 - and entry j of a row here is the angle of Meta's pair j. The
/// permutation is applied to q and k alike, so every q.k product is unchanged.
///
/// Positions, in the f32 arithmetic Meta's buffers were built with:
/// - window blocks: local coordinates 0..window inside each window (Meta
///   builds the table for a window-sized input, scale 1) - the same 576 rows
///   for every window;
/// - global blocks: grid coordinates times `window / grid` ("interpolated"
///   rope: the table the backbone was pre-trained at, stretched), here
///   `c * f32(24/72)`, which is how torch multiplies a float tensor by a
///   Python scalar.
pub(crate) struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl RopeTable {
    fn freqs(head_dim: usize, theta: f32) -> Vec<f32> {
        // theta ** (arange(0, dim, 4)[: dim // 4].float() / dim), then 1 / that
        (0..head_dim / 4)
            .map(|k| 1.0f32 / theta.powf((4 * k) as f32 / head_dim as f32))
            .collect()
    }

    fn build(coords: impl Iterator<Item = (f32, f32)>, head_dim: usize, theta: f32) -> Self {
        let f = Self::freqs(head_dim, theta);
        let (q, half) = (head_dim / 4, head_dim / 2);
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for (tx, ty) in coords {
            let base = cos.len();
            cos.resize(base + half, 0.0);
            sin.resize(base + half, 0.0);
            for j in 0..q {
                let ax = tx * f[j];
                let ay = ty * f[j];
                cos[base + j] = ax.cos();
                sin[base + j] = ax.sin();
                cos[base + q + j] = ay.cos();
                sin[base + q + j] = ay.sin();
            }
        }
        Self { cos, sin }
    }

    /// The window blocks' table: `win^2` rows, row `ly * win + lx`.
    pub(crate) fn window(win: usize, head_dim: usize, theta: f32) -> Self {
        let it = (0..win * win).map(move |l| ((l % win) as f32, (l / win) as f32));
        Self::build(it, head_dim, theta)
    }

    /// The global blocks' table over the window-major grid: `grid^2` rows.
    pub(crate) fn global(grid: usize, win: usize, head_dim: usize, theta: f32) -> Self {
        let scale = (win as f64 / grid as f64) as f32;
        let it = (0..grid * grid).map(move |r| {
            let (gy, gx) = window_major_cell(r, grid, win);
            (gx as f32 * scale, gy as f32 * scale)
        });
        Self::build(it, head_dim, theta)
    }
}

/// The two tables the loader builds - (cos, sin) for the window blocks, then
/// for the global blocks in window-major rows - for the gate that holds them
/// to the ones Meta saved in its checkpoint.
#[doc(hidden)]
pub fn rope_tables_for_gate(
    grid: usize,
    win: usize,
    head_dim: usize,
    theta: f32,
) -> ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>)) {
    let w = RopeTable::window(win, head_dim, theta);
    let g = RopeTable::global(grid, win, head_dim, theta);
    ((w.cos, w.sin), (g.cos, g.sin))
}

/// Row `r` of the window-major order -> its (row, column) grid cell.
pub(crate) fn window_major_cell(r: usize, grid: usize, win: usize) -> (usize, usize) {
    let ww = win * win;
    let nwx = grid / win;
    let (wi, l) = (r / ww, r % ww);
    ((wi / nwx) * win + l / win, (wi % nwx) * win + l % win)
}

/// Meta's head-dim order -> the rotate-half order the split kernel rotates:
/// new dim `j` is Meta's `2j`, new `j + hd/2` is Meta's `2j + 1`. Returns, for
/// each new dim of a head, the Meta dim it reads.
pub(crate) fn rope_head_perm(head_dim: usize) -> Vec<usize> {
    let half = head_dim / 2;
    (0..head_dim)
        .map(|n| if n < half { 2 * n } else { 2 * (n - half) + 1 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_major_order_tiles_the_grid_once() {
        let (g, w) = (72usize, 24usize);
        let mut seen = vec![false; g * g];
        for r in 0..g * g {
            let (y, x) = window_major_cell(r, g, w);
            assert!(!seen[y * g + x]);
            seen[y * g + x] = true;
        }
        // the first window is the top-left 24x24, row-major inside
        assert_eq!(window_major_cell(25, g, w), (1, 1));
        // window 1 is the next one to the right
        assert_eq!(window_major_cell(576, g, w), (0, 24));
        // window 3 starts the second band
        assert_eq!(window_major_cell(3 * 576, g, w), (24, 0));
    }

    #[test]
    fn rope_permutation_pairs_meta_dims() {
        let p = rope_head_perm(64);
        // rotate-half pair (j, j + 32) must be Meta's complex pair (2j, 2j + 1)
        for j in 0..32 {
            assert_eq!((p[j], p[j + 32]), (2 * j, 2 * j + 1));
        }
    }

    #[test]
    fn global_rope_reads_interpolated_columns_then_rows() {
        let t = RopeTable::global(72, 24, 64, 10000.0);
        // row 1 of window-major order is grid cell (0, 1): column 1/3, row 0
        let half = 32;
        let f0 = 1.0f32;
        let ax = 1.0f32 * (24.0f64 / 72.0f64) as f32 * f0;
        assert!((t.cos[half] - ax.cos()).abs() < 1e-7);
        assert!((t.sin[half] - ax.sin()).abs() < 1e-7);
        // its row angles are all zero
        assert_eq!(t.sin[half + 16], 0.0);
    }
}
