//! Inkling 2 Mini's picture tower: pictures in, the language model's rows
//! for their `<|unused_200054|>` placeholders out (2304 wide, f32, one per
//! 40 x 40 cell).
//!
//! **The processor** never resizes: the picture is padded to whole cells
//! with -1 on the 0..255 scale, rescaled by 1/255 and normalized with mean =
//! std = 0.5, and cut into native 20 x 20 patches (slot 1011). A cell is one
//! language-model token and four patches.
//!
//! **The rows** of a picture are its patches in token-major order (the
//! reference's own patch order - its position ids are block-major over the
//! 2 x 2 merge blocks) followed by its class row; the reference puts the
//! class token first, but attention without a mask is permutation
//! equivariant, and with it last the adapter's merge is the patch rows read
//! as [T][4 x 1024]. A pass packs pictures back to back; attention and the
//! three passes of slots 1011-1013 run a picture at a time at its row
//! offset, everything else over the whole pass.
//!
//! **A block** (pre-LN, eps 1e-5, LayerScale on both branches, exact GELU)
//! rides the half tower interface: LN1 into f16 -> q | k | v GEMM -> slot
//! 1013 (every bias - with the rope the k bias does not cancel - the axial
//! interleaved rope, 1/8 into q) -> slot 620 over the picture's rows -> o ->
//! slot 622 (residual + LayerScale + LN2) -> fc1 + bias + GELU in its
//! landing -> fc2 -> slot 622 again with the next block's LN1. The residual
//! stays f32 throughout; the reference keeps it in bf16.
//!
//! **The adapter** reads the last block's residual (no final norm): the
//! patch rows of each picture, merged 2 x 2 by layout, 4096 -> GELU (in the
//! landing) -> 2304 on the same half rings. Every GEMM here routes on the
//! mma rings alone (no GEMV band, no K-split), so a picture's rows do not
//! move with what shares its pass. The rows stay f32 (the reference casts
//! them to bf16 as it scatters them).

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::inkling2::vision as iv;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use crate::gpu::{GpuExecutor, HalfTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::f32_tensor;

const PRE: &str = "model.visual";

/// The rows one tower pass holds: the largest picture this lane takes (its
/// patches + class row). Smaller pictures share a pass up to this.
pub const PASS_ROWS: usize = 4 * iv::MAX_TOKENS + 1;

/// One picture's language-model rows, `[n][2304]` f32.
pub struct PictureRows {
    pub rows: CudaSlice<f32>,
    pub n: usize,
}

struct Norm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

struct Block {
    ln1: Norm,
    wqkv: HalfTensor,
    bq: CudaSlice<f32>,
    bk: CudaSlice<f32>,
    bv: CudaSlice<f32>,
    wo: HalfTensor,
    bo: CudaSlice<f32>,
    ls1: CudaSlice<f32>,
    ln2: Norm,
    fc1: HalfTensor,
    fc1_b: CudaSlice<f32>,
    fc2: HalfTensor,
    fc2_b: CudaSlice<f32>,
    ls2: CudaSlice<f32>,
}

/// A pass's planes, sized for [`PASS_ROWS`] at load (the KV plan then
/// reads what is left).
struct Workspace {
    x: CudaSlice<f32>,
    n16: CudaSlice<f16>,
    qkv: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    a16: CudaSlice<f16>,
    proj: CudaSlice<f16>,
    /// fc1's landing - and the processor's patches before the blocks run
    ff: CudaSlice<f16>,
    /// the merged patch rows [T][4096] (f32, then f16) and the adapter's
    /// hidden [T][4096]
    merged: CudaSlice<f32>,
    merged16: CudaSlice<f16>,
    hid16: CudaSlice<f16>,
}

pub struct Tower {
    exec: Arc<GpuExecutor>,
    patch_w: HalfTensor,
    grid: CudaSlice<f32>,
    cls: CudaSlice<f32>,
    ln_pre: Norm,
    blocks: Vec<Block>,
    proj_in: HalfTensor,
    proj_out: HalfTensor,
    /// proj_in has no bias; its GELU landing takes one
    zero_bias: CudaSlice<f32>,
    ws: Workspace,
    /// Device bytes of the planes (the workspace is `ws_bytes`).
    pub weight_bytes: u64,
    pub ws_bytes: u64,
}

/// A picture a pass encodes: its decoded pixels and token grid.
pub struct Picture<'a> {
    pub rgb: &'a [u8],
    pub w: usize,
    pub h: usize,
}

impl Picture<'_> {
    pub fn grid(&self) -> (usize, usize) {
        iv::grid(self.w, self.h)
    }

    pub fn tokens(&self) -> usize {
        let (r, c) = self.grid();
        r * c
    }

    /// Rows the picture takes in a pass: its patches and its class row.
    fn rows(&self) -> usize {
        4 * self.tokens() + 1
    }
}

impl Tower {
    /// The release's tower, or None when it carries none.
    pub fn load(
        exec: Arc<GpuExecutor>,
        st: &ShardedSafetensors,
    ) -> Result<Option<Self>, GpuModelError> {
        let m = |s: &str| format!("{PRE}.{s}");
        if st
            .bytes(&m("backbone.embeddings.patch_embedding.weight"))
            .is_none()
        {
            return Ok(None);
        }
        if !exec.has_inkling2_vision() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates Inkling 2's picture tower (slots 1011-1013) - \
                 rebuild or update the pack"
                    .into(),
            ));
        }
        if !exec.has_dense_pred_h() || !exec.has_f16_gemm_h_gelu() {
            return Err(GpuModelError::Unsupported(
                "Inkling 2's tower needs the half tower interface (slots 618-624) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        let (d, ff, p) = (iv::WIDTH, iv::FF, iv::PATCH);
        let e = &*exec;
        let weight_bytes = std::cell::Cell::new(0u64);
        let plane = |name: &str, n_out: usize, k_in: usize| -> Result<HalfTensor, GpuModelError> {
            let w = f32_tensor(st, name, n_out * k_in)?;
            weight_bytes.set(weight_bytes.get() + (w.len() * 2) as u64);
            Ok(HalfTensor {
                buf: e.to_device_f16(&w, name)?,
                dims: vec![k_in, n_out],
            })
        };
        let dev = |name: &str, n: usize| -> Result<CudaSlice<f32>, GpuModelError> {
            Ok(e.to_device(&f32_tensor(st, name, n)?)?)
        };
        let norm = |prefix: &str| -> Result<Norm, GpuModelError> {
            Ok(Norm {
                w: dev(&format!("{prefix}.weight"), d)?,
                b: dev(&format!("{prefix}.bias"), d)?,
            })
        };
        let patch_w = plane(
            &m("backbone.embeddings.patch_embedding.weight"),
            d,
            3 * p * p,
        )?;
        let grid = dev(
            &m("backbone.embeddings.position_embedding"),
            iv::POS_GRID * iv::POS_GRID * d,
        )?;
        // the class row's embedding: class + its position embedding, in f32
        let cls = {
            let c = f32_tensor(st, &m("backbone.embeddings.class_embedding"), d)?;
            let pc = f32_tensor(st, &m("backbone.embeddings.cls_position_embedding"), d)?;
            let sum: Vec<f32> = c.iter().zip(&pc).map(|(a, b)| a + b).collect();
            e.to_device(&sum)?
        };
        let ln_pre = norm(&m("backbone.ln_pre"))?;
        let mut blocks = Vec::with_capacity(iv::LAYERS);
        for i in 0..iv::LAYERS {
            let l = |s: &str| m(&format!("backbone.layers.{i}.{s}"));
            // q | k | v stacked into one landing, rows in that order
            let wqkv = {
                let mut w = f32_tensor(st, &l("self_attn.q_proj.weight"), d * d)?;
                w.extend(f32_tensor(st, &l("self_attn.k_proj.weight"), d * d)?);
                w.extend(f32_tensor(st, &l("self_attn.v_proj.weight"), d * d)?);
                weight_bytes.set(weight_bytes.get() + (w.len() * 2) as u64);
                HalfTensor {
                    buf: e.to_device_f16(&w, "Inkling 2 tower qkv")?,
                    dims: vec![d, 3 * d],
                }
            };
            blocks.push(Block {
                ln1: norm(&l("ln_1"))?,
                wqkv,
                bq: dev(&l("self_attn.q_proj.bias"), d)?,
                bk: dev(&l("self_attn.k_proj.bias"), d)?,
                bv: dev(&l("self_attn.v_proj.bias"), d)?,
                wo: plane(&l("self_attn.out_proj.weight"), d, d)?,
                bo: dev(&l("self_attn.out_proj.bias"), d)?,
                ls1: dev(&l("ls_1.gamma"), d)?,
                ln2: norm(&l("ln_2"))?,
                fc1: plane(&l("mlp.c_fc.weight"), ff, d)?,
                fc1_b: dev(&l("mlp.c_fc.bias"), ff)?,
                fc2: plane(&l("mlp.c_proj.weight"), d, ff)?,
                fc2_b: dev(&l("mlp.c_proj.bias"), d)?,
                ls2: dev(&l("ls_2.gamma"), d)?,
            });
        }
        let proj_in = plane(&m("proj_in.weight"), iv::PROJ, 4 * d)?;
        let proj_out = plane(
            &m("proj_out.weight"),
            paddock_models::inkling2::WIDTH,
            iv::PROJ,
        )?;
        let weight_bytes = weight_bytes.get();
        // every bf16 plane is checked above; the small f32 vectors ride free
        for name in [
            m("backbone.embeddings.patch_embedding.weight"),
            m("proj_in.weight"),
        ] {
            if st
                .bytes(&name)
                .is_some_and(|(t, _)| t.dtype != StDtype::Bf16)
            {
                return Err(GpuModelError::Unsupported(format!(
                    "Inkling 2 tower: {name} is not bf16"
                )));
            }
        }
        let rows = PASS_ROWS;
        let toks = iv::MAX_TOKENS;
        let ws = Workspace {
            x: e.alloc(rows * d)?,
            n16: e.alloc_f16(rows * d)?,
            qkv: e.alloc_f16(rows * 3 * d)?,
            q: e.alloc_f16(rows * d)?,
            k: e.alloc_f16(rows * d)?,
            v: e.alloc_f16(rows * d)?,
            a16: e.alloc_f16(rows * d)?,
            proj: e.alloc_f16(rows * d)?,
            ff: e.alloc_f16(rows * ff.max(3 * p * p))?,
            merged: e.alloc(toks * 4 * d)?,
            merged16: e.alloc_f16(toks * 4 * d)?,
            hid16: e.alloc_f16(toks * iv::PROJ)?,
        };
        let ws_bytes = (rows * d * 4
            + rows * d * 2 * 6
            + rows * 3 * d * 2
            + rows * ff * 2
            + toks * 4 * d * 6
            + toks * iv::PROJ * 2) as u64;
        let zero_bias = e.alloc(iv::PROJ)?;
        Ok(Some(Self {
            exec,
            patch_w,
            grid,
            cls,
            ln_pre,
            blocks,
            proj_in,
            proj_out,
            zero_bias,
            ws,
            weight_bytes,
            ws_bytes,
        }))
    }

    /// Encode `pics`, packed into passes of up to [`PASS_ROWS`] rows; each
    /// picture's rows, in order.
    pub fn encode(&mut self, pics: &[Picture<'_>]) -> Result<Vec<PictureRows>, GpuModelError> {
        let mut out = Vec::with_capacity(pics.len());
        let mut i = 0;
        while i < pics.len() {
            let mut j = i;
            let mut rows = 0;
            while j < pics.len() && rows + pics[j].rows() <= PASS_ROWS {
                rows += pics[j].rows();
                j += 1;
            }
            if j == i {
                return Err(GpuModelError::Unsupported(format!(
                    "a picture of {} x {} is {} tokens; this lane takes up to {} a picture",
                    pics[i].w,
                    pics[i].h,
                    pics[i].tokens(),
                    iv::MAX_TOKENS
                )));
            }
            out.extend(self.pass(&pics[i..j])?);
            i = j;
        }
        Ok(out)
    }

    /// One pass: `pics` back to back in the workspace.
    fn pass(&mut self, pics: &[Picture<'_>]) -> Result<Vec<PictureRows>, GpuModelError> {
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let d = iv::WIDTH;
        for p in pics {
            if p.rgb.len() != p.w * p.h * 3 || p.w == 0 || p.h == 0 {
                return Err(GpuModelError::Unsupported(format!(
                    "picture {} x {} carries {} bytes",
                    p.w,
                    p.h,
                    p.rgb.len()
                )));
            }
        }
        // (first row, token grid) of each picture in the pass
        let mut at = Vec::with_capacity(pics.len());
        let mut rows = 0;
        for p in pics {
            at.push((rows, p.grid()));
            rows += p.rows();
        }
        let ws = &mut self.ws;

        // ---- the processor + embedding: patches (in fc1's plane), the patch
        // GEMM over every row of the pass into the residual (class rows get
        // overwritten), then the field, class rows and ln_pre in place
        for (p, &(r0, (tr, tc))) in pics.iter().zip(&at) {
            let src = e.to_device_u8(p.rgb)?;
            e.inkling2_vis_patches(&src, p.w, p.h, tr, tc, &mut ws.ff, r0)?;
        }
        gemm_f32(e, &self.patch_w, &ws.ff, &mut ws.x, rows)?;
        for &(r0, (tr, tc)) in &at {
            e.inkling2_vis_embed(
                (&mut ws.x, r0),
                &self.grid,
                &self.cls,
                (&self.ln_pre.w, &self.ln_pre.b),
                tr,
                tc,
                iv::POS_GRID,
                iv::EPS,
            )?;
        }

        // ---- the blocks
        let first = &self.blocks[0].ln1;
        e.layernorm_f16(&ws.x, &first.w, &first.b, &mut ws.n16, rows, d, iv::EPS)?;
        let qscale = 1.0 / (iv::HEAD_DIM as f32).sqrt();
        let n_layer = self.blocks.len();
        for li in 0..n_layer {
            let blk = &self.blocks[li];
            e.matvec_batch_f16_h(&blk.wqkv, &ws.n16, &mut ws.qkv, rows)?;
            for &(r0, (tr, tc)) in &at {
                e.inkling2_vis_qkv_rope(
                    &ws.qkv,
                    r0,
                    (&blk.bq, &blk.bk, &blk.bv),
                    (&mut ws.q, &mut ws.k, &mut ws.v),
                    tr,
                    tc,
                    iv::HEAD_DIM,
                    iv::ROPE_THETA,
                    qscale,
                )?;
                e.vision_attn_h_rows(
                    &ws.q,
                    &ws.k,
                    &ws.v,
                    &mut ws.a16,
                    r0,
                    4 * tr * tc + 1,
                    iv::HEADS,
                    iv::HEAD_DIM,
                )?;
            }
            e.matvec_batch_f16_h(&blk.wo, &ws.a16, &mut ws.proj, rows)?;
            e.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.bo,
                &blk.ls1,
                &blk.ln2.w,
                &blk.ln2.b,
                &mut ws.n16,
                rows,
                d,
                iv::EPS,
            )?;
            e.matvec_batch_f16_h_gelu(&blk.fc1, &ws.n16, &mut ws.ff, &blk.fc1_b, rows)?;
            e.matvec_batch_f16_h(&blk.fc2, &ws.ff, &mut ws.proj, rows)?;
            // the next block's LN1 rides this seam; after the last one the
            // norm it writes is unused (the adapter reads the residual)
            let next = if li + 1 < n_layer {
                &self.blocks[li + 1].ln1
            } else {
                &blk.ln2
            };
            e.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.fc2_b,
                &blk.ls2,
                &next.w,
                &next.b,
                &mut ws.n16,
                rows,
                d,
                iv::EPS,
            )?;
        }

        // ---- the adapter: each picture's patch rows side by side, merged
        // 2 x 2 by layout, through 4096 -> GELU -> 2304
        let mut t0 = 0;
        for &(r0, (tr, tc)) in &at {
            let n = 4 * tr * tc * d;
            e.copy_region(&ws.x, r0 * d, &mut ws.merged, t0 * 4 * d, n)?;
            t0 += tr * tc;
        }
        let toks = t0;
        // the half rings, whose route never depends on the row count: a
        // picture's rows do not move with what shares its pass
        e.convert_f32_f16(&ws.merged, &mut ws.merged16, toks * 4 * d)?;
        e.matvec_batch_f16_h_gelu(
            &self.proj_in,
            &ws.merged16,
            &mut ws.hid16,
            &self.zero_bias,
            toks,
        )?;
        // the last GEMM lands each picture's rows in their own plane (the
        // ring's bits do not depend on the rows it is handed)
        let width = paddock_models::inkling2::WIDTH;
        let mut out = Vec::with_capacity(pics.len());
        let mut t0 = 0;
        for &(_, (tr, tc)) in &at {
            let n = tr * tc;
            let mut rows = e.alloc(n * width)?;
            if e.has_f16_gemm_ring() {
                e.f16_gemm_ring_rows(&self.proj_out, &ws.hid16, t0, &mut rows, 0, n, None)?;
            } else {
                let mut all = e.alloc(toks * width)?;
                e.matvec_batch_f16(&self.proj_out, &ws.hid16, &mut all, toks)?;
                e.copy_region(&all, t0 * width, &mut rows, 0, n * width)?;
            }
            out.push(PictureRows { rows, n });
            t0 += n;
        }
        Ok(out)
    }
}

/// `y = W x` (f32 landing) on the f16 rings where the pack has them.
fn gemm_f32(
    exec: &GpuExecutor,
    w: &HalfTensor,
    x16: &CudaSlice<f16>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuModelError> {
    if exec.has_f16_gemm_ring() {
        exec.f16_gemm_ring_rows(w, x16, 0, y, 0, rows, None)?;
    } else {
        exec.matvec_batch_f16(w, x16, y, rows)?;
    }
    Ok(())
}
