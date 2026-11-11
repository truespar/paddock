//! The DeepEncoder forward graph - pixels of one GEOMETRY (a batch of
//! same-sized views) through SAM, the 16x squeeze, CLIP, and the projector.
//!
//! Dataflow (per batch of B views at `px` ∈ {1024, 640}, g = px/16, t = g/4):
//!
//!   host im2row [B·g², 3·256] -> conv GEMM + bias -> +abs pos (resampled)
//!   -> 12 × SAM block:
//!       LN(1e-6) -> [windowed: partition gather, zero-padded] -> one QKV GEMM
//!       + fused bias -> row_slice q|k|v -> `sam_attn` (decomposed rel-pos bias)
//!       -> [windowed: unpartition gather] -> out GEMM + bias -> +res
//!       -> LN -> lin1 -> exact-erf GELU -> lin2 -> +res
//!   -> neck: 1x1 GEMM -> LN2d -> 3x3 gather+GEMM -> LN2d
//!   -> net_2 (3x3 s2) -> net_3 (3x3 s2)                      [B·t², 1024]
//!   -> CLIP: +CLS row, +pos (resampled, CLS held out), pre-LN(1e-5)
//!   -> 24 × block: LN -> QKV GEMM -> row_slice -> `vision_attn_x` -> out + res
//!                 -> LN -> up -> QUICK-gelu -> down + res
//!   -> projector: clip[1:]·fc_clip + sam·fc_sam + bias      [B·t², 1280]
//!
//! Faithfulness notes, each one a silent-wrong if missed:
//!  * the two halves run different activations (SAM erf-GELU, CLIP quick-GELU)
//!    and different LayerNorm eps (1e-6 / 1e-5) - module doc in `vision.rs`.
//!  * window padding is zero-padding of the NORMED stream, and the pad rows go
//!    through the QKV GEMM (their k/v become pure bias) exactly as the
//!    reference's `F.pad` -> `qkv` does. They are dropped at unpartition; out
//!    proj runs on the unpartitioned (real) rows only, which is legal because
//!    it is row-wise.
//!  * `LayerNorm2d` normalizes over CHANNELS at each pixel - in the NHWC row
//!    layout this whole file keeps, that is the ordinary row layernorm.
//!  * CLIP has no post-layernorm: the reference's `VitModel.forward` returns
//!    the transformer output directly.
//!  * the reference runs `vision_model(x, sam_out)` - CLIP's own conv stem is
//!    dead; its "patch embeds" are net_3's NHWC rows.
//!
//! Everything is one geometry per call: the global 1024 view and the 640 crops
//! have different grids, tables and window counts, so the caller runs two
//! encodes for a gundam-mode image. Crops batch together (B = tiles).
//!
//! ## Workspace ([`TowerWs`])
//!
//! This graph runs MID-SERVE, where one cudaMalloc is a device-wide sync
//! under CUDA's default allocator - and the transient-alloc version of it
//! made ~220 of them per gundam request: ~25
//! activation slabs plus gather/pos/rel-pos tables per encode call, two or
//! more calls per image. The slabs are now allocated once at load for the
//! largest geometry the tower can be asked to run - a closed set: crop
//! chunks of at most [`MAX_VIEWS`] 640² tiles, or one 1024² global view -
//! and every table derived from geometry alone (window partition, conv
//! taps, resampled pos/rel-pos) is cached keyed by that geometry, host-side
//! f64 resample math included. A steady-state request allocates nothing
//! here but its projector output.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use cudarc::driver::CudaSlice;
use half::f16;

use super::resample;
use super::vision::{BASE_PX, DeepEncoder, SamBlock, TILE_PX, VisionHparams};
use crate::gpu::{GpuError, GpuExecutor};

/// Cap on views per encode call. The transient QKV landing is
/// `B·P²·3·embd` f32 (P = padded grid); 8 crops keep it near 130 MB where 32
/// would be half a gigabyte. Callers chunk; the graph does not care.
pub const MAX_VIEWS: usize = 8;

/// Encoded views of one geometry: `[B·t², llm_embd]` rows, row-major t×t per
/// view, in view order. Newline/separator splicing is the caller's job - it
/// is token-layout work (`tiling::Block`), not tower work.
pub struct EncodedViews {
    pub embd: CudaSlice<f32>,
    pub views: usize,
    /// Token side per view (16 at 1024px, 10 at 640px).
    pub t: usize,
}

/// `PADDOCK_OCR_VIS_DUMP=1` prints a per-stage f64 sum of the tower's
/// activations, to be diffed against the same quantities dumped from the
/// checkpoint's own deepencoder.py. Synchronizes per stage - that is the
/// point, and why it is off by default. (Muse precedent.)
fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_OCR_VIS_DUMP").is_some())
}

/// See [`dump_on`]. f64 accumulation so the sums are comparable against a
/// torch dump computed the same way.
fn stage(exec: &GpuExecutor, tag: &str, buf: &CudaSlice<f32>, n: usize) {
    if !dump_on() {
        return;
    }
    match exec.to_host_len(buf, n) {
        Ok(h) => {
            let s: f64 = h.iter().map(|&v| v as f64).sum();
            eprintln!("ocr-vis {tag:>14}: n={n:<9} sum={s:.4}");
        }
        Err(e) => eprintln!("ocr-vis {tag:>14}: readback failed: {e}"),
    }
}

/// Row geometry of one encode call. Every buffer size and gather table is a
/// function of (views, px) through these numbers - computed in exactly one
/// place so the load-time workspace sizing and the per-call graph can never
/// disagree about a size.
pub(super) struct Geom {
    pub(super) views: usize,
    pub(super) px: usize,
    /// SAM patch grid side (64 at 1024px, 40 at 640px).
    g: usize,
    /// g², patches per view.
    n: usize,
    /// views·n - real SAM rows across the batch.
    rows: usize,
    /// Windows per padded-grid side (the pad makes every window complete).
    wins_side: usize,
    /// wins_side², windows per view.
    n_win: usize,
    /// views·n_win·win² - rows after the window partition, pad included.
    part_rows: usize,
    g2: usize,
    /// Rows after net_2 (stride 2).
    rows2: usize,
    /// Token side per view (g/4).
    t: usize,
    /// views·t² - final token rows.
    rows_t: usize,
    /// CLS + grid, per view.
    nc: usize,
    /// views·nc - CLIP rows.
    rc: usize,
}

pub(super) fn geom(hp: &VisionHparams, views: usize, px: usize) -> Geom {
    let g = hp.sam_grid(px);
    let n = g * g;
    let rows = views * n;
    let win = hp.window;
    let p_grid = g.div_ceil(win) * win;
    let wins_side = p_grid / win;
    let n_win = wins_side * wins_side;
    let part_rows = views * n_win * win * win;
    let g2 = g / 2;
    let rows2 = views * g2 * g2;
    let t = g2 / 2;
    let rows_t = views * t * t;
    let nc = t * t + 1;
    let rc = views * nc;
    Geom {
        views,
        px,
        g,
        n,
        rows,
        wins_side,
        n_win,
        part_rows,
        g2,
        rows2,
        t,
        rows_t,
        nc,
        rc,
    }
}

/// Channel widths the workspace sizes against - hp plus the four widths only
/// the tensors themselves state (asserted at load, not assumed).
pub(super) struct TowerDims {
    /// sam_embd.
    pub e: usize,
    pub sam_ff: usize,
    /// SAM conv im2row width (patch²·3) - `sam_patch_w.dims[0]`.
    pub patch_in: usize,
    /// `neck0_w.dims[1]`.
    pub neck_ch: usize,
    /// `net_2.dims[1]`.
    pub mid_ch: usize,
    /// `net_3.dims[1]` - the projector's SAM half.
    pub sam_ch: usize,
    /// clip_embd.
    pub ce: usize,
    pub cff: usize,
    /// llm_embd.
    pub out_dim: usize,
}

const N_SLABS: usize = 25;
const SLAB_NAMES: [&str; N_SLABS] = [
    "patches",
    "s16",
    "x",
    "nrm",
    "part",
    "qkv",
    "q",
    "k",
    "v",
    "attn",
    "proj",
    "ff",
    "nk",
    "g9",
    "nk2",
    "n2",
    "sam_out",
    "cx",
    "cnrm",
    "cqkv",
    "cff",
    "h16",
    "cattn",
    "clip_flat",
    "tmp",
];

/// Element count each slab needs for one geometry - The size-formula site:
/// [`TowerWs::new`] allocates the elementwise max over the closed geometry
/// set, [`TowerWs::check`] asserts a call fits. Order matches [`SLAB_NAMES`]
/// and the field order of [`TowerWs`].
fn need(d: &TowerDims, gm: &Geom) -> [usize; N_SLABS] {
    [
        gm.rows * d.patch_in,
        // f16 staging: widest activation plane any GEMM consumes
        (gm.part_rows * 3 * d.e)
            .max(gm.rows * d.sam_ff)
            .max(gm.rows * 9 * 512),
        gm.rows * d.e,
        // +1: the window partition's always-zero pad row
        (gm.rows + 1) * d.e,
        gm.part_rows * d.e,
        gm.part_rows * 3 * d.e,
        // q/k/v/attn - CLIP reuses them at rc·ce, strictly smaller
        gm.part_rows * d.e,
        gm.part_rows * d.e,
        gm.part_rows * d.e,
        gm.part_rows * d.e,
        gm.rows * d.e,
        // SAM's up plane; also the composed quick-GELU's gate in the CLIP
        // stage (idle there)
        (gm.rows * d.sam_ff).max(gm.rc * d.cff),
        // +1 on nk/nk2/n2: the 3x3 conv gathers' zero tap row
        (gm.rows + 1) * d.neck_ch,
        gm.rows * 9 * d.neck_ch,
        (gm.rows + 1) * d.neck_ch,
        (gm.rows2 + 1) * d.mid_ch,
        // +1: the CLS staging row the CLIP gather reads
        (gm.rows_t + 1) * d.sam_ch,
        gm.rc * d.ce,
        gm.rc * d.ce,
        gm.rc * 3 * d.ce,
        gm.rc * d.cff,
        // f16: the FFN up's fused landing, SAM's or CLIP's
        (gm.rows * d.sam_ff).max(gm.rc * d.cff),
        gm.rc * d.ce,
        gm.rows_t * d.ce,
        gm.rows_t * d.out_dim,
    ]
}

/// Window partition/unpartition, conv-tap and CLIP gather tables for one
/// (px, views) geometry - they bake per-view batch offsets in, so the key
/// carries both.
struct GeoTables {
    part: CudaSlice<u32>,
    unpart: CudaSlice<u32>,
    t1: CudaSlice<u32>,
    t2: CudaSlice<u32>,
    t3: CudaSlice<u32>,
    clip: CudaSlice<u32>,
    flat: CudaSlice<u32>,
}

fn geo_tables(exec: &GpuExecutor, gm: &Geom, win: usize) -> Result<GeoTables, GpuError> {
    let (views, g, n) = (gm.views, gm.g, gm.n);
    let zero_row = gm.rows as u32; // index of the always-zero pad row
    let mut part_idx = Vec::with_capacity(gm.part_rows);
    for b in 0..views {
        for wy in 0..gm.wins_side {
            for wx in 0..gm.wins_side {
                for py in 0..win {
                    for qx in 0..win {
                        let (gy, gx) = (wy * win + py, wx * win + qx);
                        part_idx.push(if gy < g && gx < g {
                            (b * n + gy * g + gx) as u32
                        } else {
                            zero_row
                        });
                    }
                }
            }
        }
    }
    let mut unpart_idx = Vec::with_capacity(gm.rows);
    for b in 0..views {
        for gy in 0..g {
            for gx in 0..g {
                let w_of = (gy / win) * gm.wins_side + gx / win;
                unpart_idx
                    .push(((b * gm.n_win + w_of) * win * win + (gy % win) * win + gx % win) as u32);
            }
        }
    }
    // 3x3 conv taps in NHWC rows; each gathers 9 taps with the zero row
    // (views·src_g²) standing in for the pad. Stride 1 for the neck conv,
    // 2 for net_2/net_3.
    let taps3 = |src_g: usize, stride: usize| -> Vec<u32> {
        let dst_g = src_g / stride;
        let zero = (views * src_g * src_g) as u32;
        let mut idx = Vec::with_capacity(views * dst_g * dst_g * 9);
        for b in 0..views {
            for oy in 0..dst_g {
                for ox in 0..dst_g {
                    for ky in 0..3i64 {
                        for kx in 0..3i64 {
                            let sy = (oy * stride) as i64 + ky - 1;
                            let sx = (ox * stride) as i64 + kx - 1;
                            idx.push(
                                if sy >= 0 && sy < src_g as i64 && sx >= 0 && sx < src_g as i64 {
                                    (b * src_g * src_g + sy as usize * src_g + sx as usize) as u32
                                } else {
                                    zero
                                },
                            );
                        }
                    }
                }
            }
        }
        idx
    };
    // CLS + grid per view, CLS staged at sam_out's spare row (rows_t)
    let mut clip_idx = Vec::with_capacity(gm.rc);
    for b in 0..views {
        clip_idx.push(gm.rows_t as u32);
        clip_idx.extend((0..gm.t * gm.t).map(|i| (b * gm.t * gm.t + i) as u32));
    }
    // clip rows minus each view's CLS, for the projector
    let mut flat_idx = Vec::with_capacity(gm.rows_t);
    for b in 0..views {
        flat_idx.extend((1..gm.nc).map(|i| (b * gm.nc + i) as u32));
    }
    Ok(GeoTables {
        part: exec.to_device_u32(&part_idx)?,
        unpart: exec.to_device_u32(&unpart_idx)?,
        t1: exec.to_device_u32(&taps3(g, 1))?,
        t2: exec.to_device_u32(&taps3(g, 2))?,
        t3: exec.to_device_u32(&taps3(gm.g2, 2))?,
        clip: exec.to_device_u32(&clip_idx)?,
        flat: exec.to_device_u32(&flat_idx)?,
    })
}

fn geo_cached<'a>(
    map: &'a mut HashMap<(usize, usize), GeoTables>,
    exec: &GpuExecutor,
    gm: &Geom,
    win: usize,
) -> Result<&'a GeoTables, GpuError> {
    match map.entry((gm.px, gm.views)) {
        Entry::Occupied(e) => Ok(e.into_mut()),
        Entry::Vacant(v) => Ok(v.insert(geo_tables(exec, gm, win)?)),
    }
}

/// SAM abs pos resampled to the px grid (bicubic + antialias, f64 host math -
/// `get_abs_pos_sam`). Keyed px: the broadcast add covers any view count.
fn pos_cached<'a>(
    map: &'a mut HashMap<usize, CudaSlice<f32>>,
    exec: &GpuExecutor,
    hp: &VisionHparams,
    sam_pos: &[f32],
    gm: &Geom,
) -> Result<&'a CudaSlice<f32>, GpuError> {
    match map.entry(gm.px) {
        Entry::Occupied(e) => Ok(e.into_mut()),
        Entry::Vacant(v) => {
            let src: Vec<f64> = sam_pos.iter().map(|&x| x as f64).collect();
            let r = resample::bicubic_aa_grid(&src, hp.sam_grid_train, gm.g, hp.sam_embd);
            let f: Vec<f32> = r.iter().map(|&x| x as f32).collect();
            Ok(v.insert(exec.to_device(&f)?))
        }
    }
}

/// CLIP pos, CLS held out of the resample exactly like `get_abs_pos`.
fn cpos_cached<'a>(
    map: &'a mut HashMap<usize, CudaSlice<f32>>,
    exec: &GpuExecutor,
    hp: &VisionHparams,
    clip_pos: &[f32],
    gm: &Geom,
) -> Result<&'a CudaSlice<f32>, GpuError> {
    match map.entry(gm.px) {
        Entry::Occupied(e) => Ok(e.into_mut()),
        Entry::Vacant(v) => {
            let ce = hp.clip_embd;
            let train_side = ((hp.clip_positions - 1) as f64).sqrt() as usize;
            let src: Vec<f64> = clip_pos.iter().map(|&x| x as f64).collect();
            let mut out = Vec::with_capacity(gm.nc * ce);
            out.extend_from_slice(&src[..ce]); // CLS row passes through
            out.extend(resample::bicubic_aa_grid(&src[ce..], train_side, gm.t, ce));
            let f: Vec<f32> = out.iter().map(|&x| x as f32).collect();
            Ok(v.insert(exec.to_device(&f)?))
        }
    }
}

/// Decomposed rel-pos device tables, `[side, side, hd]` each. Keyed
/// (block, side): windowed blocks share side = window across both pxs;
/// global blocks get one entry per grid they see.
fn rel_cached<'a>(
    map: &'a mut HashMap<(usize, usize), (CudaSlice<f32>, CudaSlice<f32>)>,
    exec: &GpuExecutor,
    il: usize,
    blk: &SamBlock,
    side: usize,
    hd: usize,
) -> Result<&'a (CudaSlice<f32>, CudaSlice<f32>), GpuError> {
    match map.entry((il, side)) {
        Entry::Occupied(e) => Ok(e.into_mut()),
        Entry::Vacant(v) => {
            let to64 = |x: &[f32]| x.iter().map(|&y| y as f64).collect::<Vec<f64>>();
            let h = resample::rel_pos_table(&to64(&blk.rel_pos_h), blk.rel_rows, hd, side);
            let w = resample::rel_pos_table(&to64(&blk.rel_pos_w), blk.rel_rows, hd, side);
            let hf: Vec<f32> = h.iter().map(|&y| y as f32).collect();
            let wf: Vec<f32> = w.iter().map(|&y| y as f32).collect();
            Ok(v.insert((exec.to_device(&hf)?, exec.to_device(&wf)?)))
        }
    }
}

/// Persistent tower workspace + geometry caches - see the module doc. Held by
/// [`DeepEncoder`] for its whole life; [`TowerWs::device_bytes`] is what the
/// fit estimate prices for it.
pub(super) struct TowerWs {
    dims: TowerDims,
    patches: CudaSlice<f32>,
    /// The u8 stem's pixel landing: interleaved-RGB views, same
    /// ELEMENT count as `patches` at a quarter of the bytes. The serve path
    /// uploads here and `pd_ocr_patches_u8` writes `s16` directly; `patches`
    /// stays as the f32 reference arm (the oracle gate feeds float pixels).
    patches8: CudaSlice<u8>,
    s16: CudaSlice<f16>,
    x: CudaSlice<f32>,
    nrm: CudaSlice<f32>,
    part: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    ff: CudaSlice<f32>,
    nk: CudaSlice<f32>,
    g9: CudaSlice<f32>,
    nk2: CudaSlice<f32>,
    n2: CudaSlice<f32>,
    sam_out: CudaSlice<f32>,
    cx: CudaSlice<f32>,
    cnrm: CudaSlice<f32>,
    cqkv: CudaSlice<f32>,
    cff: CudaSlice<f32>,
    h16: CudaSlice<f16>,
    cattn: CudaSlice<f32>,
    clip_flat: CudaSlice<f32>,
    tmp: CudaSlice<f32>,
    geo: HashMap<(usize, usize), GeoTables>,
    pos: HashMap<usize, CudaSlice<f32>>,
    cpos: HashMap<usize, CudaSlice<f32>>,
    rel: HashMap<(usize, usize), (CudaSlice<f32>, CudaSlice<f32>)>,
    /// Run the seams as their own passes even where the pack lands them in
    /// the GEMMs - the reference arm the fused tower is gated bit-for-bit
    /// against ([`super::vision::DeepEncoder::set_composed_seams`]).
    pub(super) composed: bool,
}

impl TowerWs {
    /// Allocate every slab at the elementwise max over the closed geometry
    /// set and warm every cache both canonical geometries will hit, so the
    /// serve path's first request pays no better or worse than its thousandth.
    /// The host resample sources are handed in by the loader before they move
    /// into the encoder.
    pub(super) fn new(
        exec: &GpuExecutor,
        hp: &VisionHparams,
        dims: TowerDims,
        sam_pos: &[f32],
        clip_pos: &[f32],
        blocks: &[SamBlock],
    ) -> Result<Self, GpuError> {
        let a = geom(hp, MAX_VIEWS, TILE_PX);
        let b = geom(hp, 1, BASE_PX);
        let (na, nb) = (need(&dims, &a), need(&dims, &b));
        let mut mx = [0usize; N_SLABS];
        for i in 0..N_SLABS {
            mx[i] = na[i].max(nb[i]);
        }
        let mut ws = Self {
            patches: exec.alloc(mx[0])?,
            patches8: exec.alloc_u8(mx[0])?,
            s16: exec.alloc_f16(mx[1])?,
            x: exec.alloc(mx[2])?,
            nrm: exec.alloc(mx[3])?,
            part: exec.alloc(mx[4])?,
            qkv: exec.alloc(mx[5])?,
            q: exec.alloc(mx[6])?,
            k: exec.alloc(mx[7])?,
            v: exec.alloc(mx[8])?,
            attn: exec.alloc(mx[9])?,
            proj: exec.alloc(mx[10])?,
            ff: exec.alloc(mx[11])?,
            nk: exec.alloc(mx[12])?,
            g9: exec.alloc(mx[13])?,
            nk2: exec.alloc(mx[14])?,
            n2: exec.alloc(mx[15])?,
            sam_out: exec.alloc(mx[16])?,
            cx: exec.alloc(mx[17])?,
            cnrm: exec.alloc(mx[18])?,
            cqkv: exec.alloc(mx[19])?,
            cff: exec.alloc(mx[20])?,
            h16: exec.alloc_f16(mx[21])?,
            cattn: exec.alloc(mx[22])?,
            clip_flat: exec.alloc(mx[23])?,
            tmp: exec.alloc(mx[24])?,
            dims,
            geo: HashMap::new(),
            pos: HashMap::new(),
            cpos: HashMap::new(),
            rel: HashMap::new(),
            composed: false,
        };
        for gm in [&a, &b] {
            geo_cached(&mut ws.geo, exec, gm, hp.window)?;
            pos_cached(&mut ws.pos, exec, hp, sam_pos, gm)?;
            cpos_cached(&mut ws.cpos, exec, hp, clip_pos, gm)?;
            for (il, blk) in blocks.iter().enumerate() {
                let side = if hp.is_global(il) { gm.g } else { hp.window };
                rel_cached(&mut ws.rel, exec, il, blk, side, hp.sam_head_dim)?;
            }
        }
        Ok(ws)
    }

    /// Device bytes the slabs pin (the geometry caches add a few tens of MB
    /// on top; the slabs are the ~1 GiB that matters to a fit estimate).
    pub(super) fn device_bytes(&self) -> usize {
        (self.patches.len()
            + self.x.len()
            + self.nrm.len()
            + self.part.len()
            + self.qkv.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.attn.len()
            + self.proj.len()
            + self.ff.len()
            + self.nk.len()
            + self.g9.len()
            + self.nk2.len()
            + self.n2.len()
            + self.sam_out.len()
            + self.cx.len()
            + self.cnrm.len()
            + self.cqkv.len()
            + self.cff.len()
            + self.cattn.len()
            + self.clip_flat.len()
            + self.tmp.len())
            * 4
            + (self.s16.len() + self.h16.len()) * 2
            + self.patches8.len()
    }

    /// Every slab use in `encode` must fit. The geometry set is closed - a
    /// miss here means a caller invented a new geometry, which is a design
    /// change (resize the envelope), never a runtime condition to grow past.
    fn check(&self, gm: &Geom) {
        let needs = need(&self.dims, gm);
        let haves = [
            self.patches.len(),
            self.s16.len(),
            self.x.len(),
            self.nrm.len(),
            self.part.len(),
            self.qkv.len(),
            self.q.len(),
            self.k.len(),
            self.v.len(),
            self.attn.len(),
            self.proj.len(),
            self.ff.len(),
            self.nk.len(),
            self.g9.len(),
            self.nk2.len(),
            self.n2.len(),
            self.sam_out.len(),
            self.cx.len(),
            self.cnrm.len(),
            self.cqkv.len(),
            self.cff.len(),
            self.h16.len(),
            self.cattn.len(),
            self.clip_flat.len(),
            self.tmp.len(),
        ];
        for i in 0..N_SLABS {
            assert!(
                needs[i] <= haves[i],
                "tower ws slab {}: need {} have {} - {} views at {}px is outside the \
                 load-time envelope",
                SLAB_NAMES[i],
                needs[i],
                haves[i],
                gm.views,
                gm.px,
            );
        }
    }
}

/// Host im2row for the SAM conv stem: `rgb` is B views of `px`² pixels,
/// values already normalized ((x/255 - mean)/std - the caller owns
/// preprocessing). Emits `[B·g², patch²·3]` rows in the weight's im2row
/// order `c·p² + ky·p + kx`. A free function deliberately: the pipelined
/// admission wave runs it on worker threads that must not touch the tower.
pub(super) fn im2row(rgb: &[f32], views: usize, px: usize, p: usize) -> Vec<f32> {
    let g = px / p;
    assert_eq!(
        rgb.len(),
        views * px * px * 3,
        "rgb is not {views} views of {px}px RGB"
    );
    let k = p * p * 3;
    let mut out = vec![0f32; views * g * g * k];
    for b in 0..views {
        let img = &rgb[b * px * px * 3..][..px * px * 3];
        for gy in 0..g {
            for gx in 0..g {
                let row = &mut out[((b * g + gy) * g + gx) * k..][..k];
                for c in 0..3 {
                    for ky in 0..p {
                        for kx in 0..p {
                            // interleaved RGB pixels in, channel-planar row out
                            row[c * p * p + ky * p + kx] =
                                img[((gy * p + ky) * px + gx * p + kx) * 3 + c];
                        }
                    }
                }
            }
        }
    }
    out
}

/// What [`DeepEncoder::encode`] feeds the SAM conv stem. `U8` is the serve
/// path: raw interleaved-RGB views, normalized + im2row'd + f16-converted on
/// device (`pd_ocr_patches_u8`) - a quarter of the f32 plane's upload bytes
/// and none of its host preprocessing. `F32` is the reference arm: prebuilt
/// [`im2row`] rows, kept for the oracle gate and the stem bit-cmp test.
#[derive(Clone, Copy)]
pub enum PatchSrc<'a> {
    F32(&'a [f32]),
    U8(&'a [u8]),
}

impl DeepEncoder {
    /// [`im2row`] at this tower's patch size.
    pub fn patch_rows(&self, rgb: &[f32], views: usize, px: usize) -> Vec<f32> {
        im2row(rgb, views, px, self.hp.sam_patch)
    }

    /// Encode `views` same-sized views. `&mut self` for the workspace only -
    /// weights and hyperparameters are never touched.
    ///
    /// The two stems land the same `s16` bits (the u8 kernel's math is
    /// bit-identical to normalize_rgb8 + im2row + the f32->f16 convert -
    /// gated by `u8_stem_matches_the_f32_stem`), so everything downstream is
    /// stem-blind. Serving feeds `U8` (a quarter of the upload bytes, no
    /// host normalize/im2row); the oracle gate feeds `F32`.
    pub fn encode(
        &mut self,
        src: PatchSrc<'_>,
        views: usize,
        px: usize,
    ) -> Result<EncodedViews, GpuError> {
        assert!(
            views > 0 && views <= MAX_VIEWS,
            "chunk views to <= {MAX_VIEWS}"
        );
        let gm = geom(&self.hp, views, px);
        self.ws.check(&gm);
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (e, heads, hd, win) = (hp.sam_embd, hp.sam_heads, hp.sam_head_dim, hp.window);
        let (g, n, rows, part_rows) = (gm.g, gm.n, gm.rows, gm.part_rows);
        match src {
            PatchSrc::F32(r) => debug_assert_eq!(r.len(), rows * self.sam_patch_w.dims[0]),
            PatchSrc::U8(p) => debug_assert_eq!(p.len(), views * px * px * 3),
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let neck_ch = self.neck0_w.dims[1];
        let mid_ch = self.net_2.dims[1];

        // Disjoint &mut over the workspace fields; every other read below is
        // a different field of self, so the borrows never collide.
        let TowerWs {
            dims: _,
            patches,
            patches8,
            s16,
            x,
            nrm,
            part,
            qkv,
            q,
            k,
            v,
            attn,
            proj,
            ff,
            nk,
            g9,
            nk2,
            n2,
            sam_out,
            cx,
            cnrm,
            cqkv,
            cff,
            h16,
            cattn,
            clip_flat,
            tmp,
            geo,
            pos,
            cpos,
            rel,
            composed,
        } = &mut self.ws;
        let gt = geo_cached(geo, &exec, &gm, win)?;
        // the tower seams fold into their GEMMs where the pack has the slots
        let fused = exec.has_deepencoder_fused() && !*composed;
        let pos = pos_cached(pos, &exec, hp, &self.sam_pos, &gm)?;

        // Re-arm the always-zero rows the reused slabs carry: nrm's window-
        // partition pad source and the three conv gathers' zero taps. These
        // were free when the buffers were alloc_zeros'd per call.
        exec.zero_region(nrm, rows * e, e)?;
        exec.zero_region(nk, rows * neck_ch, neck_ch)?;
        exec.zero_region(nk2, rows * neck_ch, neck_ch)?;
        exec.zero_region(n2, gm.rows2 * mid_ch, mid_ch)?;

        // -- SAM stage --------------------------------------------------------
        match src {
            PatchSrc::F32(patch_rows) => {
                exec.upload_f32(patch_rows, patches)?;
                exec.convert_f32_f16(patches, s16, patch_rows.len())?;
            }
            PatchSrc::U8(pixels) => {
                exec.upload_u8(pixels, patches8)?;
                exec.ocr_patches_u8(
                    patches8,
                    s16,
                    hp.image_mean,
                    hp.image_std,
                    views,
                    px,
                    hp.sam_patch,
                )?;
            }
        }
        exec.matvec_batch_f16(&self.sam_patch_w, s16, x, rows)?;
        exec.bias_add(x, &self.sam_patch_b, rows, e)?;
        exec.add_rows_bcast(x, pos, rows, n, e)?;
        stage(&exec, "sam_embed", x, rows * e);

        for (il, blk) in self.sam_blocks.iter().enumerate() {
            let global = hp.is_global(il);
            let side = if global { g } else { win };
            let (rh, rw) = {
                let p = rel_cached(rel, &exec, il, blk, side, hd)?;
                (&p.0, &p.1)
            };
            let (arows, batch) = if global {
                (rows, views)
            } else {
                (part_rows, views * gm.n_win)
            };
            if fused && global {
                exec.layernorm_f16(x, &blk.pre_ln_w, &blk.pre_ln_b, s16, rows, e, hp.sam_eps)?;
            } else {
                // a windowed layer partitions the normed rows (f32 gather)
                // before they stage
                exec.layernorm(x, &blk.pre_ln_w, &blk.pre_ln_b, nrm, rows, e, hp.sam_eps)?;
                if !global {
                    exec.pixel_shuffle_rows(nrm, &gt.part, part, part_rows, 1, e)?;
                }
                let src: &CudaSlice<f32> = if global { nrm } else { part };
                exec.convert_f32_f16(src, s16, arows * e)?;
            }
            exec.matvec_batch_f16(&blk.qkv_w, s16, qkv, arows)?;
            if fused {
                exec.qkv_bias_split(qkv, &blk.qkv_b, q, k, v, arows, e)?;
            } else {
                exec.bias_add(qkv, &blk.qkv_b, arows, 3 * e)?;
                exec.row_slice(qkv, q, 3 * e, 0, e, arows)?;
                exec.row_slice(qkv, k, 3 * e, e, e, arows)?;
                exec.row_slice(qkv, v, 3 * e, 2 * e, e, arows)?;
            }
            exec.sam_attn(q, k, v, rh, rw, attn, batch, side, heads, hd, scale)?;

            // out proj reads real rows only: unpartition first when windowed.
            // `part` doubles as the landing - attn is done with q/k/v.
            let attn_rows: &CudaSlice<f32> = if global {
                attn
            } else {
                exec.pixel_shuffle_rows(attn, &gt.unpart, part, rows, 1, e)?;
                part
            };
            exec.convert_f32_f16(attn_rows, s16, rows * e)?;
            if fused {
                // the seams in the GEMMs: bias + residual in the projection
                // landings, the up plane landed f16 with its bias and GELU -
                // bit-for-bit the composed chain below
                exec.matvec_batch_f16_bias_res(&blk.out_w, s16, x, &blk.out_b, proj, rows)?;
                exec.layernorm_f16(x, &blk.post_ln_w, &blk.post_ln_b, s16, rows, e, hp.sam_eps)?;
                exec.matvec_batch_f16_act(&blk.lin1_w, s16, h16, &blk.lin1_b, ff, rows, false)?;
                exec.matvec_batch_f16_bias_res(&blk.lin2_w, h16, x, &blk.lin2_b, proj, rows)?;
            } else {
                exec.matvec_batch_f16(&blk.out_w, s16, proj, rows)?;
                exec.bias_add(proj, &blk.out_b, rows, e)?;
                exec.add(x, proj, rows * e)?;

                exec.layernorm(x, &blk.post_ln_w, &blk.post_ln_b, nrm, rows, e, hp.sam_eps)?;
                exec.convert_f32_f16(nrm, s16, rows * e)?;
                exec.matvec_batch_f16(&blk.lin1_w, s16, ff, rows)?;
                exec.bias_add(ff, &blk.lin1_b, rows, hp.sam_ff)?;
                exec.gelu_erf(ff, rows * hp.sam_ff)?;
                exec.convert_f32_f16(ff, s16, rows * hp.sam_ff)?;
                exec.matvec_batch_f16(&blk.lin2_w, s16, proj, rows)?;
                exec.bias_add(proj, &blk.lin2_b, rows, e)?;
                exec.add(x, proj, rows * e)?;
            }
            if dump_on() {
                stage(&exec, &format!("sam_block-{il}"), x, rows * e);
            }
        }

        // -- neck + the 16x squeeze. NHWC rows throughout; each conv gathers
        // 9 taps (zero pad row) and GEMMs the tap-major plane. Stride 1 for
        // the neck conv, 2 for net_2/net_3. Tap tables come from `gt`.
        exec.convert_f32_f16(x, s16, rows * e)?;
        exec.matvec_batch_f16(&self.neck0_w, s16, nk, rows)?;
        // LayerNorm2d - channel norm per pixel, i.e. row LN in NHWC. In place
        // via the front rows of `nrm` (wide enough: neck_ch < e).
        exec.layernorm(nk, &self.neck1_w, &self.neck1_b, nrm, rows, neck_ch, 1e-6)?;
        exec.copy_region(nrm, 0, nk, 0, rows * neck_ch)?;

        exec.pixel_shuffle_rows(nk, &gt.t1, g9, rows, 9, neck_ch)?;
        exec.convert_f32_f16(g9, s16, rows * 9 * neck_ch)?;
        exec.matvec_batch_f16(&self.neck2_w, s16, nk2, rows)?;
        exec.layernorm(nk2, &self.neck3_w, &self.neck3_b, nrm, rows, neck_ch, 1e-6)?;
        exec.copy_region(nrm, 0, nk2, 0, rows * neck_ch)?;
        stage(&exec, "neck", nk2, rows * neck_ch);

        let rows2 = gm.rows2;
        exec.pixel_shuffle_rows(nk2, &gt.t2, g9, rows2, 9, neck_ch)?;
        exec.convert_f32_f16(g9, s16, rows2 * 9 * neck_ch)?;
        exec.matvec_batch_f16(&self.net_2, s16, n2, rows2)?;

        let (t, rows_t) = (gm.t, gm.rows_t);
        let sam_ch = self.net_3.dims[1];
        exec.pixel_shuffle_rows(n2, &gt.t3, g9, rows_t, 9, mid_ch)?;
        exec.convert_f32_f16(g9, s16, rows_t * 9 * mid_ch)?;
        exec.matvec_batch_f16(&self.net_3, s16, sam_out, rows_t)?;
        stage(&exec, "sam_out", sam_out, rows_t * sam_ch);

        // -- CLIP stage -------------------------------------------------------
        let ce = hp.clip_embd;
        debug_assert_eq!(ce, sam_ch);
        let (nc, rc) = (gm.nc, gm.rc);

        // CLS into sam_out's spare row - a plain write, so the reused slab
        // needs no zeroing here.
        exec.copy_region(&self.class_embd, 0, sam_out, rows_t * ce, ce)?;
        exec.pixel_shuffle_rows(sam_out, &gt.clip, cx, rc, 1, ce)?;

        let cpos = cpos_cached(cpos, &exec, hp, &self.clip_pos, &gm)?;
        exec.add_rows_bcast(cx, cpos, rc, nc, ce)?;

        exec.layernorm(
            cx,
            &self.pre_ln_w,
            &self.pre_ln_b,
            cnrm,
            rc,
            ce,
            super::vision::CLIP_EPS,
        )?;
        std::mem::swap(cx, cnrm);
        stage(&exec, "clip_pre_ln", cx, rc * ce);

        let cffw = hp.clip_ff;
        let cheads = hp.clip_heads;
        let chd = hp.clip_head_dim;
        let cscale = 1.0 / (chd as f32).sqrt();
        for (il, blk) in self.clip_blocks.iter().enumerate() {
            if fused {
                exec.layernorm_f16(
                    cx,
                    &blk.ln1_w,
                    &blk.ln1_b,
                    s16,
                    rc,
                    ce,
                    super::vision::CLIP_EPS,
                )?;
            } else {
                exec.layernorm(
                    cx,
                    &blk.ln1_w,
                    &blk.ln1_b,
                    cnrm,
                    rc,
                    ce,
                    super::vision::CLIP_EPS,
                )?;
                exec.convert_f32_f16(cnrm, s16, rc * ce)?;
            }
            exec.matvec_batch_f16(&blk.qkv_w, s16, cqkv, rc)?;
            if fused {
                exec.qkv_bias_split(cqkv, &blk.qkv_b, q, k, v, rc, ce)?;
            } else {
                exec.bias_add(cqkv, &blk.qkv_b, rc, 3 * ce)?;
                exec.row_slice(cqkv, q, 3 * ce, 0, ce, rc)?;
                exec.row_slice(cqkv, k, 3 * ce, ce, ce, rc)?;
                exec.row_slice(cqkv, v, 3 * ce, 2 * ce, ce, rc)?;
            }
            exec.vision_attn_x(q, k, v, cattn, nc, nc, cheads, chd, views, cscale)?;
            exec.convert_f32_f16(cattn, s16, rc * ce)?;
            if fused {
                exec.matvec_batch_f16_bias_res(&blk.out_w, s16, cx, &blk.out_b, cnrm, rc)?;
                exec.layernorm_f16(
                    cx,
                    &blk.ln2_w,
                    &blk.ln2_b,
                    s16,
                    rc,
                    ce,
                    super::vision::CLIP_EPS,
                )?;
                // QUICK-gelu - x·σ(1.702x) - in the up landing
                exec.matvec_batch_f16_act(&blk.up_w, s16, h16, &blk.up_b, cff, rc, true)?;
                exec.matvec_batch_f16_bias_res(&blk.down_w, h16, cx, &blk.down_b, cnrm, rc)?;
            } else {
                exec.matvec_batch_f16(&blk.out_w, s16, cnrm, rc)?;
                exec.bias_add(cnrm, &blk.out_b, rc, ce)?;
                exec.add(cx, cnrm, rc * ce)?;

                exec.layernorm(
                    cx,
                    &blk.ln2_w,
                    &blk.ln2_b,
                    cnrm,
                    rc,
                    ce,
                    super::vision::CLIP_EPS,
                )?;
                exec.convert_f32_f16(cnrm, s16, rc * ce)?;
                exec.matvec_batch_f16(&blk.up_w, s16, cff, rc)?;
                exec.bias_add(cff, &blk.up_b, rc, cffw)?;
                // QUICK-gelu - x·σ(1.702x), composed: a copy of the up plane
                // scaled into the gate (the fused landing's reference chain;
                // SAM's `ff` plane is idle here and wider than rc·cff)
                exec.copy_region(cff, 0, ff, 0, rc * cffw)?;
                exec.scale(ff, 1.702, rc * cffw)?;
                exec.mul_sigmoid(cff, ff, rc * cffw)?;
                exec.convert_f32_f16(cff, s16, rc * cffw)?;
                exec.matvec_batch_f16(&blk.down_w, s16, cnrm, rc)?;
                exec.bias_add(cnrm, &blk.down_b, rc, ce)?;
                exec.add(cx, cnrm, rc * ce)?;
            }
            if dump_on() {
                stage(&exec, &format!("clip_block-{il}"), cx, rc * ce);
            }
        }

        // -- projector: clip rows minus CLS, plus the sam rows, two GEMMs.
        // `out` is the one per-call allocation left: it escapes as the return
        // value and lives across the caller's chunked prefill.
        exec.pixel_shuffle_rows(cx, &gt.flat, clip_flat, rows_t, 1, ce)?;

        let out_dim = hp.llm_embd;
        let mut out = exec.alloc(rows_t * out_dim)?;
        exec.convert_f32_f16(clip_flat, s16, rows_t * ce)?;
        exec.matvec_batch_f16(&self.fc_clip, s16, &mut out, rows_t)?;
        exec.convert_f32_f16(sam_out, s16, rows_t * sam_ch)?;
        exec.matvec_batch_f16(&self.fc_sam, s16, tmp, rows_t)?;
        exec.add(&mut out, tmp, rows_t * out_dim)?;
        exec.bias_add(&mut out, &self.fc_b, rows_t, out_dim)?;
        stage(&exec, "projected", &out, rows_t * out_dim);

        Ok(EncodedViews {
            embd: out,
            views,
            t,
        })
    }
}

#[cfg(test)]
mod tests {
    /// Window partition + unpartition tables must be exact inverses over the
    /// real rows, and every pad index must point at the zero row. Pure host
    /// logic, no GPU.
    #[test]
    fn partition_tables_invert() {
        for (g, win, views) in [(64usize, 14usize, 1usize), (40, 14, 3), (64, 14, 2)] {
            let n = g * g;
            let p_grid = g.div_ceil(win) * win;
            let ws = p_grid / win;
            let n_win = ws * ws;
            let zero = (views * n) as u32;
            let mut part = Vec::new();
            for b in 0..views {
                for wy in 0..ws {
                    for wx in 0..ws {
                        for py in 0..win {
                            for qx in 0..win {
                                let (gy, gx) = (wy * win + py, wx * win + qx);
                                part.push(if gy < g && gx < g {
                                    (b * n + gy * g + gx) as u32
                                } else {
                                    zero
                                });
                            }
                        }
                    }
                }
            }
            let mut unpart = Vec::new();
            for b in 0..views {
                for gy in 0..g {
                    for gx in 0..g {
                        let w_of = (gy / win) * ws + gx / win;
                        unpart.push(
                            ((b * n_win + w_of) * win * win + (gy % win) * win + gx % win) as u32,
                        );
                    }
                }
            }
            for (r, &pi) in unpart.iter().enumerate() {
                assert_eq!(part[pi as usize] as usize, r, "g={g} views={views} row {r}");
            }
            let pads = part.iter().filter(|&&i| i == zero).count();
            assert_eq!(pads, views * (p_grid * p_grid - n), "pad count g={g}");
        }
    }
}
