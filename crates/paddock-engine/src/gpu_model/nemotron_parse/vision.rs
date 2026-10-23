//! The C-RADIOv4-SO400M tower and the neck: page canvases in, each page's
//! decoder prefix out (`(H/16)(W/64)` merged patch rows + one summary row,
//! d_model wide, f32).
//!
//! **A pass packs pages of any size.** Every op in a block is row-wise except
//! attention, so a pass lays its pages back to back - each page's rows as
//! `[patches | 10 prefix tokens]` - and runs the row-wise ops over the whole
//! plane once; attention runs per page (`vision_attn_h_rows`, slot 620 at the
//! page's row offset). A full A4 page alone is ~185 query tiles x 16 heads, so
//! the per-page launches already fill the die; packing buys the GEMMs their
//! weight reuse across small pages. The reference embeds each page separately
//! too (its packed NaViT path splits per page before attention), so a page's
//! rows do not depend on what shares its pass beyond GEMM tile election (the
//! dinov3 "pin the width" lesson - measured in the gate, not assumed).
//!
//! **The prefix tokens are parked AFTER the patches** (the reference puts
//! them first). Unmasked attention is permutation-equivariant, so the bits a
//! patch row computes are the same either way, and the neck then reads a
//! page's patch rows as one contiguous run.
//!
//! **Block** (timm `Block`, pre-LN, eps 1e-6, no LayerScale, exact GELU):
//! LN1 lands f16 -> qkv on the half interface -> the DINOv3 split (q bias +
//! 1/sqrt(72), v bias; the k bias is dropped: adding the same vector to every
//! key moves each query's scores by one constant, which softmax ignores) ->
//! attention -> wo -> residual + LN2 in one seam (622, LayerScale = ones) ->
//! fc1 with bias + exact GELU in its epilogue (624) -> fc2 -> residual + the
//! NEXT block's LN1 in one seam. No final norm: the neck reads the residual.
//!
//! **Neck**: patches -> 1x1 conv (+bias, lands F32 - its output reaches
//! ~38K on real pages, inside f16 but with no margin) -> LN -> the (1,4)/4 conv
//! as ONE GEMM over the patch plane viewed `[n/4][4 * 1024]` (a canvas is a
//! whole number of 4-patch groups across, so four horizontal neighbours are
//! four consecutive rows; the weight is re-laid to (kw, c_in) at load) -> LN.
//! Summary: prefix tokens 0 and 1 are adjacent rows, so their concatenation
//! is 2304 contiguous floats -> linear -> LN -> the page's last prefix row.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::nemotron_parse::{NECK_LN_EPS, NemotronParseConfig, TOWER_LN_EPS};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::preprocess::{PagePrep, position_field};
use crate::gpu::{GpuExecutor, HalfTensor};
use crate::gpu_model::gpt_oss::GpuModelError;

/// Rows one tower pass is sized for: two full 2048 x 1664 pages
/// (2 x (13312 + 10)). The workspace is resident at this size (~1 GB).
pub const TOWER_PASS_ROWS: usize = 2 * (128 * 104 + 10);

/// Position fields kept on the device between passes, keyed by page shape (a
/// full page's is 61 MB; one document is usually one shape). A pass of more
/// shapes than this holds all of its own while it runs.
const PE_CACHE_MAX: usize = 4;

/// `y = W x` (f32 landing) on the unsplit mma rings where the pack has them: a
/// pass packs pages, so its row counts follow what else is in the pass - the
/// summary projection runs over one row a page - and the elected GEMM takes
/// its GEMV band or K-split at small counts, which moved a page's bits with
/// its pass-mates. The rings walk one output's K in one order at any count.
fn ring(
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

pub(crate) struct Norm {
    pub w: CudaSlice<f32>,
    pub b: CudaSlice<f32>,
}

struct Block {
    ln1: Norm,
    wqkv: HalfTensor,
    bq: CudaSlice<f32>,
    bv: CudaSlice<f32>,
    wo: HalfTensor,
    bo: CudaSlice<f32>,
    ln2: Norm,
    fc1: HalfTensor,
    fc1_b: CudaSlice<f32>,
    fc2: HalfTensor,
    fc2_b: CudaSlice<f32>,
}

struct Workspace {
    rows: usize,
    canvas: CudaSlice<u8>,
    patches: CudaSlice<f16>,
    x: CudaSlice<f32>,
    n16: CudaSlice<f16>,
    qkv: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    a16: CudaSlice<f16>,
    proj: CudaSlice<f16>,
    ff: CudaSlice<f16>,
    /// f32 scratch [rows][1152]: the patch embedding, then the neck's planes
    e32: CudaSlice<f32>,
    /// the neck's packed patch plane [rows][1024] f16
    np16: CudaSlice<f16>,
    sum32: CudaSlice<f32>,
    sum16: CudaSlice<f16>,
    s32: CudaSlice<f32>,
}

/// One page's decoder prefix: `rows` x d_model f32.
pub struct PrefixPlane {
    pub plane: CudaSlice<f32>,
    pub rows: usize,
}

pub struct RadioTower {
    exec: Arc<GpuExecutor>,
    dim: usize,
    heads: usize,
    hd: usize,
    patch: usize,
    n_prefix: usize,
    d_model: usize,
    pos_src: usize,
    mean: [f32; 3],
    std: [f32; 3],
    patch_w: HalfTensor,
    cls: CudaSlice<f32>,
    /// the learned grid on the host, `[src * src][dim]` - fields are built from it per shape
    pos_grid: Vec<f32>,
    blocks: Vec<Block>,
    ones: CudaSlice<f32>,
    /// rope table stand-ins for the split (n_rope = 0 never reads them)
    no_rope: CudaSlice<f32>,
    conv1: HalfTensor,
    conv1_b: CudaSlice<f32>,
    ln_a: Norm,
    conv2: HalfTensor,
    ln_b: Norm,
    sum_proj: HalfTensor,
    sum_b: CudaSlice<f32>,
    ln_c: Norm,
    ws: Workspace,
    pe: HashMap<((usize, usize), (usize, usize)), CudaSlice<f32>>,
    pub weight_bytes: u64,
    /// keep the shared vision attention (slot 620) - the A/B and the gate
    /// against the exact-hd72 kernel (`PADDOCK_NP_ATTN620=1`)
    pub attn620: bool,
}

/// Read an F32 tensor whose shape must match exactly.
pub(crate) fn f32_exact(
    st: &ShardedSafetensors,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, GpuModelError> {
    let (t, b) = st
        .bytes(name)
        .ok_or_else(|| GpuModelError::MissingMeta(format!("nemotron-parse tensor {name}")))?;
    if t.dtype != StDtype::F32 || t.shape != shape {
        return Err(GpuModelError::Unsupported(format!(
            "nemotron-parse {name}: {:?} {:?} (want F32 {shape:?})",
            t.dtype, t.shape
        )));
    }
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

pub(crate) struct Up<'a> {
    pub exec: &'a GpuExecutor,
    pub bytes: u64,
}

impl Up<'_> {
    pub fn dev(&mut self, host: &[f32]) -> Result<CudaSlice<f32>, GpuModelError> {
        self.bytes += (host.len() * 4) as u64;
        Ok(self.exec.to_device(host)?)
    }
    /// `[out][in]` row-major f32 (the torch Linear layout) -> an f16 GEMM plane.
    /// Round-to-nearest; `to_device_f16` refuses a plane that overflows.
    pub fn plane(
        &mut self,
        w: &[f32],
        in_dim: usize,
        out_dim: usize,
        what: &str,
    ) -> Result<HalfTensor, GpuModelError> {
        assert_eq!(w.len(), in_dim * out_dim);
        if !in_dim.is_multiple_of(8) {
            return Err(GpuModelError::Unsupported(format!(
                "nemotron-parse {what}: GEMM input width {in_dim} is not a multiple of 8"
            )));
        }
        self.bytes += (w.len() * 2) as u64;
        Ok(HalfTensor {
            buf: self.exec.to_device_f16(w, what)?,
            dims: vec![in_dim, out_dim],
        })
    }
    pub fn norm(
        &mut self,
        st: &ShardedSafetensors,
        prefix: &str,
        n: usize,
    ) -> Result<Norm, GpuModelError> {
        Ok(Norm {
            w: self.dev(&f32_exact(st, &format!("{prefix}.weight"), &[n])?)?,
            b: self.dev(&f32_exact(st, &format!("{prefix}.bias"), &[n])?)?,
        })
    }
    pub fn vec(
        &mut self,
        st: &ShardedSafetensors,
        name: &str,
        n: usize,
    ) -> Result<CudaSlice<f32>, GpuModelError> {
        let v = f32_exact(st, name, &[n])?;
        self.dev(&v)
    }
}

impl RadioTower {
    pub fn load(
        exec: Arc<GpuExecutor>,
        st: &ShardedSafetensors,
        cfg: &NemotronParseConfig,
        pass_rows: usize,
    ) -> Result<Self, GpuModelError> {
        if !exec.has_nemotron_parse() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates Nemotron 3.5 Parse (slots 890-892) - rebuild or update \
                 the pack"
                    .into(),
            ));
        }
        if !exec.has_dense_pred_h() || !exec.has_f16_gemm_h_gelu() || !exec.has_vision_qkv_h() {
            return Err(GpuModelError::Unsupported(
                "nemotron-parse's tower needs the half tower interface (slots 618-624) - rebuild \
                 or update the pack"
                    .into(),
            ));
        }
        let (d, p, np) = (cfg.tower_dim, cfg.patch, cfg.n_prefix_tokens);
        let heads = cfg.tower_heads();
        let hd = cfg.tower_head_dim();
        let pre = "encoder.model_encoder.radio_model";
        // the tower's MLP width is the tensor's, not a config field
        let ffn = st
            .bytes(&format!("{pre}.model.blocks.0.mlp.fc1.bias"))
            .map(|(t, _)| t.shape.first().copied().unwrap_or(0))
            .ok_or_else(|| GpuModelError::MissingMeta("nemotron-parse tower MLP".into()))?;
        // summary_idxs is stored: hold the release to the two leading tokens
        // the neck's concatenation (and this layout) assumes
        let (t, b) = st
            .bytes(&format!("{pre}.summary_idxs"))
            .ok_or_else(|| GpuModelError::MissingMeta("nemotron-parse summary_idxs".into()))?;
        let idx: Vec<i64> = b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect();
        if t.dtype != StDtype::I64 || idx != [0, 1] {
            return Err(GpuModelError::Unsupported(format!(
                "nemotron-parse summary_idxs {idx:?} - the neck reads tokens [0, 1]"
            )));
        }
        let (pt, _) = st
            .bytes(&format!("{pre}.model.patch_generator.pos_embed"))
            .ok_or_else(|| GpuModelError::MissingMeta("nemotron-parse pos_embed".into()))?;
        let n_pos = pt.shape.get(1).copied().unwrap_or(0);
        let pos_src = (n_pos as f64).sqrt() as usize;
        if pos_src * pos_src != n_pos || pt.shape != [1, n_pos, d] {
            return Err(GpuModelError::Unsupported(format!(
                "nemotron-parse pos_embed {:?} - want a square [1, s*s, {d}] grid",
                pt.shape
            )));
        }

        let mut up = Up {
            exec: &exec,
            bytes: 0,
        };
        let m = |s: &str| format!("{pre}.model.{s}");
        let patch_w = {
            let w = f32_exact(st, &m("patch_generator.embedder.weight"), &[d, 3 * p * p])?;
            up.plane(&w, 3 * p * p, d, "patch embedder")?
        };
        let cls = up.dev(&f32_exact(
            st,
            &m("patch_generator.cls_token.token"),
            &[np, d],
        )?)?;
        let pos_grid = f32_exact(st, &m("patch_generator.pos_embed"), &[1, n_pos, d])?;

        let n_layer = (0..)
            .take_while(|i| st.bytes(&m(&format!("blocks.{i}.norm1.weight"))).is_some())
            .count();
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let l = |s: &str| m(&format!("blocks.{i}.{s}"));
            let qkv_b = f32_exact(st, &l("attn.qkv.bias"), &[3 * d])?;
            blocks.push(Block {
                ln1: up.norm(st, &l("norm1"), d)?,
                wqkv: up.plane(
                    &f32_exact(st, &l("attn.qkv.weight"), &[3 * d, d])?,
                    d,
                    3 * d,
                    &l("qkv"),
                )?,
                bq: up.dev(&qkv_b[..d])?,
                bv: up.dev(&qkv_b[2 * d..])?,
                wo: up.plane(
                    &f32_exact(st, &l("attn.proj.weight"), &[d, d])?,
                    d,
                    d,
                    &l("proj"),
                )?,
                bo: up.vec(st, &l("attn.proj.bias"), d)?,
                ln2: up.norm(st, &l("norm2"), d)?,
                fc1: up.plane(
                    &f32_exact(st, &l("mlp.fc1.weight"), &[ffn, d])?,
                    d,
                    ffn,
                    &l("fc1"),
                )?,
                fc1_b: up.vec(st, &l("mlp.fc1.bias"), ffn)?,
                fc2: up.plane(
                    &f32_exact(st, &l("mlp.fc2.weight"), &[d, ffn])?,
                    ffn,
                    d,
                    &l("fc2"),
                )?,
                fc2_b: up.vec(st, &l("mlp.fc2.bias"), d)?,
            });
        }
        if n_layer == 0 {
            return Err(GpuModelError::MissingMeta(
                "nemotron-parse tower blocks".into(),
            ));
        }

        // ---- neck ----
        let dm = cfg.d_model;
        let comp = cfg.neck_compression;
        let conv1 = {
            let w = f32_exact(st, "encoder.conv1.weight", &[dm, d, 1])?;
            up.plane(&w, d, dm, "neck conv1")?
        };
        let conv1_b = up.vec(st, "encoder.conv1.bias", dm)?;
        let ln_a = up.norm(st, "encoder.layer_norm1", dm)?;
        let conv2 = {
            // torch [out][in][1][kw] -> GEMM rows [out][kw * dm + in]
            let w = f32_exact(st, "encoder.conv2.weight", &[dm, dm, 1, comp])?;
            let mut r = vec![0f32; w.len()];
            for o in 0..dm {
                for i in 0..dm {
                    for kw in 0..comp {
                        r[o * comp * dm + kw * dm + i] = w[(o * dm + i) * comp + kw];
                    }
                }
            }
            up.plane(&r, comp * dm, dm, "neck conv2")?
        };
        let ln_b = up.norm(st, "encoder.layer_norm2", dm)?;
        let sum_proj = {
            let w = f32_exact(st, "encoder.sum_proj.weight", &[dm, cfg.summary_dim])?;
            up.plane(&w, cfg.summary_dim, dm, "neck sum_proj")?
        };
        let sum_b = up.vec(st, "encoder.sum_proj.bias", dm)?;
        let ln_c = up.norm(st, "encoder.layer_norm3", dm)?;
        let ones = up.dev(&vec![1.0f32; d])?;
        let no_rope = up.dev(&[0.0f32])?;
        let weight_bytes = up.bytes;

        let rows = pass_rows.max(1);
        let e = &exec;
        let ws = Workspace {
            rows,
            canvas: e.alloc_u8(rows * p * p * 3)?,
            patches: e.alloc_f16(rows * 3 * p * p)?,
            x: e.alloc(rows * d)?,
            n16: e.alloc_f16(rows * d)?,
            qkv: e.alloc_f16(rows * 3 * d)?,
            q: e.alloc_f16(rows * d)?,
            k: e.alloc_f16(rows * d)?,
            v: e.alloc_f16(rows * d)?,
            a16: e.alloc_f16(rows * d)?,
            proj: e.alloc_f16(rows * d)?,
            ff: e.alloc_f16(rows * ffn)?,
            e32: e.alloc(rows * d.max(dm))?,
            np16: e.alloc_f16(rows * dm)?,
            sum32: e.alloc(rows.div_ceil(np) * cfg.summary_dim)?,
            sum16: e.alloc_f16(rows.div_ceil(np) * cfg.summary_dim)?,
            s32: e.alloc(rows.div_ceil(np) * dm)?,
        };
        Ok(Self {
            exec,
            dim: d,
            heads,
            hd,
            patch: p,
            n_prefix: np,
            d_model: dm,
            pos_src,
            mean: cfg.mean,
            std: cfg.std,
            patch_w,
            cls,
            pos_grid,
            blocks,
            ones,
            no_rope,
            conv1,
            conv1_b,
            ln_a,
            conv2,
            ln_b,
            sum_proj,
            sum_b,
            ln_c,
            ws,
            pe: HashMap::new(),
            weight_bytes,
            attn620: paddock_models::dev_var_os!("PADDOCK_NP_ATTN620").is_some(),
        })
    }

    /// Tower rows one page takes in a pass.
    pub fn page_rows(&self, page: &PagePrep) -> usize {
        page.patch_rows(self.patch) + self.n_prefix
    }

    /// Rows a pass may hold (the workspace's sizing).
    pub fn pass_rows(&self) -> usize {
        self.ws.rows
    }

    fn field(&mut self, page: &PagePrep) -> Result<(), GpuModelError> {
        let key = (page.content_hw, page.canvas_hw);
        if self.pe.contains_key(&key) {
            return Ok(());
        }
        let host = position_field(
            &self.pos_grid,
            self.pos_src,
            self.dim,
            page.content_hw,
            page.canvas_hw,
            self.patch,
        );
        let dev = self.exec.to_device(&host)?;
        self.pe.insert(key, dev);
        Ok(())
    }

    /// Make room for the shapes `pages` need: past the cap, every field the
    /// pass does not use goes. Never one it does - evicting mid-pass dropped a
    /// field an earlier page of the same pass still had to read (a pass of
    /// five page shapes panicked on the lookup).
    fn fields_for(&mut self, pages: &[&PagePrep]) -> Result<(), GpuModelError> {
        let need: Vec<_> = pages.iter().map(|p| (p.content_hw, p.canvas_hw)).collect();
        let mut missing = need.clone();
        missing.sort_unstable();
        missing.dedup();
        missing.retain(|k| !self.pe.contains_key(k));
        if self.pe.len() + missing.len() > PE_CACHE_MAX {
            self.pe.retain(|k, _| need.contains(k));
        }
        for p in pages {
            self.field(p)?;
        }
        Ok(())
    }

    /// The position field one shape adds, read back (the oracle gate).
    pub fn field_host(&mut self, page: &PagePrep) -> Result<Vec<f32>, GpuModelError> {
        self.fields_for(&[page])?;
        let n = page.patch_rows(self.patch) * self.dim;
        Ok(self
            .exec
            .to_host_len(&self.pe[&(page.content_hw, page.canvas_hw)], n)?)
    }

    /// The normalized f16 patch rows of the pass's FIRST page (the gate).
    pub fn patches_host(&self, page: &PagePrep) -> Result<Vec<f16>, GpuModelError> {
        let n = page.patch_rows(self.patch) * 3 * self.patch * self.patch;
        Ok(self.exec.to_host_f16_len(&self.ws.patches, n)?)
    }

    /// Mark the end of the work queued on the tower's lane so far.
    pub fn record(&self) -> Result<cudarc::driver::CudaEvent, GpuModelError> {
        Ok(self.exec.record_event()?)
    }

    /// Block the host until the tower's lane is idle (the blocking prefill
    /// path and the gates read planes straight after `encode`).
    pub fn sync(&self) -> Result<(), GpuModelError> {
        self.exec
            .stream
            .synchronize()
            .map_err(crate::gpu::from_driver)?;
        Ok(())
    }

    /// A prefix plane read back (the oracle gate).
    pub fn prefix_host(&self, p: &PrefixPlane) -> Result<Vec<f32>, GpuModelError> {
        self.sync()?;
        Ok(self.exec.to_host_len(&p.plane, p.rows * self.d_model)?)
    }

    /// Encode one pass. The pages' rows must fit [`Self::pass_rows`].
    pub fn encode(&mut self, pages: &[&PagePrep]) -> Result<Vec<PrefixPlane>, GpuModelError> {
        let rows: usize = pages.iter().map(|p| self.page_rows(p)).sum();
        if pages.is_empty() {
            return Ok(Vec::new());
        }
        if rows > self.ws.rows {
            return Err(GpuModelError::BatchTooLarge {
                got: rows,
                max: self.ws.rows,
            });
        }
        self.fields_for(pages)?;
        let exec = self.exec.clone();
        let (d, p, np, dm) = (self.dim, self.patch, self.n_prefix, self.d_model);
        // slot 893 (head_dim 72 exactly) wants the score in log2 units:
        // log2(e) rides q's scale into its one f16 round
        let exact72 = self.hd == 72 && exec.has_np_vit_attn() && !self.attn620;
        let scale = if exact72 {
            std::f32::consts::LOG2_E / (self.hd as f32).sqrt()
        } else {
            1.0 / (self.hd as f32).sqrt()
        };

        // ---- pixels: canvases up, patch rows packed (no prefix gaps) ----
        let mut bytes = Vec::with_capacity(pages.iter().map(|p| p.canvas.len()).sum());
        for pg in pages {
            bytes.extend_from_slice(&pg.canvas);
        }
        exec.upload_u8(&bytes, &mut self.ws.canvas)?;
        let mut coff = 0usize;
        let mut prow = 0usize;
        for pg in pages {
            let (h, w) = pg.canvas_hw;
            exec.np_patch_u8(
                &self.ws.canvas,
                coff,
                &mut self.ws.patches,
                prow,
                h,
                w,
                self.mean,
                self.std,
            )?;
            coff += pg.canvas.len();
            prow += pg.patch_rows(p);
        }
        let n_patch = prow;

        // ---- embed: patch GEMM, then per page [patches + field | prefix tokens]
        let ws = &mut self.ws;
        ring(&exec, &self.patch_w, &ws.patches, &mut ws.e32, n_patch)?;
        let mut row = 0usize;
        let mut src = 0usize;
        for pg in pages {
            let n = pg.patch_rows(p);
            exec.copy_region(&ws.e32, src * d, &mut ws.x, row * d, n * d)?;
            let pe = &self.pe[&(pg.content_hw, pg.canvas_hw)];
            exec.add_at(&mut ws.x, row * d, pe, 0, n * d)?;
            exec.copy_region(&self.cls, 0, &mut ws.x, (row + n) * d, np * d)?;
            row += n + np;
            src += n;
        }

        // ---- 27 blocks ----
        let first = &self.blocks[0].ln1;
        exec.layernorm_f16(
            &ws.x,
            &first.w,
            &first.b,
            &mut ws.n16,
            rows,
            d,
            TOWER_LN_EPS,
        )?;
        let n_layer = self.blocks.len();
        for li in 0..n_layer {
            let blk = &self.blocks[li];
            exec.matvec_batch_f16_h(&blk.wqkv, &ws.n16, &mut ws.qkv, rows)?;
            if exact72 {
                // slot 893 reads the qkv landing itself - no split pass
                let mut r0 = 0usize;
                for pg in pages {
                    let n = pg.patch_rows(p) + np;
                    exec.np_vit_attn_qkv(
                        &ws.qkv,
                        &blk.bq,
                        &blk.bv,
                        &mut ws.a16,
                        r0,
                        n,
                        self.heads,
                        scale,
                    )?;
                    r0 += n;
                }
            } else {
                exec.dp_qkv_split_rope_h(
                    &ws.qkv,
                    &blk.bq,
                    &blk.bv,
                    &self.no_rope,
                    &self.no_rope,
                    &mut ws.q,
                    &mut ws.k,
                    &mut ws.v,
                    d,
                    self.hd,
                    rows,
                    rows,
                    0,
                    scale,
                )?;
                let mut r0 = 0usize;
                for pg in pages {
                    let n = pg.patch_rows(p) + np;
                    exec.vision_attn_h_rows(
                        &ws.q,
                        &ws.k,
                        &ws.v,
                        &mut ws.a16,
                        r0,
                        n,
                        self.heads,
                        self.hd,
                    )?;
                    r0 += n;
                }
            }
            exec.matvec_batch_f16_h(&blk.wo, &ws.a16, &mut ws.proj, rows)?;
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.bo,
                &self.ones,
                &blk.ln2.w,
                &blk.ln2.b,
                &mut ws.n16,
                rows,
                d,
                TOWER_LN_EPS,
            )?;
            exec.matvec_batch_f16_h_gelu(&blk.fc1, &ws.n16, &mut ws.ff, &blk.fc1_b, rows)?;
            exec.matvec_batch_f16_h(&blk.fc2, &ws.ff, &mut ws.proj, rows)?;
            // the next block's LN1 rides this seam; after the last block the
            // norm it writes is unused (the neck reads the raw residual)
            let next = if li + 1 < n_layer {
                &self.blocks[li + 1].ln1
            } else {
                &blk.ln2
            };
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.fc2_b,
                &self.ones,
                &next.w,
                &next.b,
                &mut ws.n16,
                rows,
                d,
                TOWER_LN_EPS,
            )?;
        }

        // ---- neck ----
        // every row through conv1 (the prefix rows' 10 are a rounding error
        // of the pass), then each page's patch rows packed for conv2
        exec.convert_f32_f16(&ws.x, &mut ws.n16, rows * d)?;
        ring(&exec, &self.conv1, &ws.n16, &mut ws.e32, rows)?;
        exec.bias_add(&mut ws.e32, &self.conv1_b, rows, dm)?;
        exec.layernorm_f16(
            &ws.e32,
            &self.ln_a.w,
            &self.ln_a.b,
            &mut ws.qkv,
            rows,
            dm,
            NECK_LN_EPS,
        )?;
        let mut r0 = 0usize;
        let mut dst = 0usize;
        for (pi, pg) in pages.iter().enumerate() {
            let n = pg.patch_rows(p);
            exec.copy_region(&ws.qkv, r0 * dm, &mut ws.np16, dst * dm, n * dm)?;
            // prefix tokens 0 and 1: adjacent rows -> one 2304-float summary row
            exec.copy_region(&ws.x, (r0 + n) * d, &mut ws.sum32, pi * 2 * d, 2 * d)?;
            r0 += n + np;
            dst += n;
        }
        // conv2 reads the packed plane as [merged][comp * dm] - the same bytes
        let comp = self.conv2.dims[0] / dm;
        let merged = n_patch / comp;
        ring(&exec, &self.conv2, &ws.np16, &mut ws.e32, merged)?;
        exec.layernorm(
            &ws.e32,
            &self.ln_b.w,
            &self.ln_b.b,
            &mut ws.x,
            merged,
            dm,
            NECK_LN_EPS,
        )?;
        let n_pages = pages.len();
        exec.convert_f32_f16(&ws.sum32, &mut ws.sum16, n_pages * 2 * d)?;
        ring(&exec, &self.sum_proj, &ws.sum16, &mut ws.s32, n_pages)?;
        exec.bias_add(&mut ws.s32, &self.sum_b, n_pages, dm)?;
        exec.layernorm(
            &ws.s32,
            &self.ln_c.w,
            &self.ln_c.b,
            &mut ws.sum32,
            n_pages,
            dm,
            NECK_LN_EPS,
        )?;

        // ---- per-page prefix planes [merged rows | summary] ----
        let mut out = Vec::with_capacity(n_pages);
        let mut m0 = 0usize;
        for (pi, pg) in pages.iter().enumerate() {
            let m = pg.patch_rows(p) / comp;
            let mut plane = exec.alloc((m + 1) * dm)?;
            exec.copy_region(&ws.x, m0 * dm, &mut plane, 0, m * dm)?;
            exec.copy_region(&ws.sum32, pi * dm, &mut plane, m * dm, dm)?;
            out.push(PrefixPlane { plane, rows: m + 1 });
            m0 += m;
        }
        Ok(out)
    }
}
