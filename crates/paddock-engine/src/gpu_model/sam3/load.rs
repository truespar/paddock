//! Checkpoint -> device planes. `model.safetensors` (1797 tensors, all F32,
//! the transformers `Sam3VideoModel` layout) plus `config.json`. Only the
//! image encoder's tensors are read here: `detector_model.vision_encoder.*`
//! (the ViT and the detector's neck), `tracker_neck.*` and the tracker mask
//! decoder's `conv_s0` / `conv_s1`.
//!
//! The safetensors file holds the same weights as Meta's `sam3.pt`, byte for
//! byte (checked tensor by tensor against the pickle), so goldens made by
//! Meta's code gate an engine that never opens the pickle. Three planes are
//! laid out differently there and arrive here already in transformers' form:
//! the position table without its class row, the four point embeddings
//! stacked, and the (unused) text projection transposed.
//!
//! Kinds of plane coming out:
//!   - GEMM weights, F32 -> f16 rounded to nearest, refused if a value would
//!     overflow, some re-laid first (q/k head permutation, conv tap order) -
//!     always on the f32 values, so a re-lay is never a second rounding;
//!   - everything a fused kernel reads per element (norms, biases, the
//!     position table) kept f32;
//!   - the rope tables, which are not in the file at all.
//!
//! Dead weights in the file, deliberately not loaded: the 0.5x neck level of
//! both necks (`fpn_layers.3`: Meta's backbone drops that level).

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::safetensors::{StDtype, TensorSource};
use paddock_models::sam3::{META_LN_EPS, Sam3VisionConfig};

use super::{
    Block, Conv, ConvT, GpuModelError, GpuSam3Vision, Neck, Norm, RopeTable, Workspace,
    rope_head_perm,
};
use crate::gpu::{GpuExecutor, HalfTensor};

/// The ViT's LayerNorm eps: Meta's, not the config's (see `META_LN_EPS`).
pub(super) const LN_EPS: f32 = META_LN_EPS;

const VIT: &str = "detector_model.vision_encoder.backbone";
const DET_NECK: &str = "detector_model.vision_encoder.neck";
const TRK_NECK: &str = "tracker_neck";
const TRK_DEC: &str = "tracker_model.mask_decoder";

/// The checkpoint reader every SAM 3 part loads through (the text tower and
/// the detector heads too), counting the device bytes it makes resident.
pub(super) struct Reader<'a> {
    pub(super) st: &'a dyn TensorSource,
    pub(super) exec: &'a GpuExecutor,
    pub(super) bytes: u64,
}

impl Reader<'_> {
    /// A tensor's f32 values, its shape checked exactly.
    pub(super) fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, GpuModelError> {
        let (t, b) = self
            .st
            .tensor(name)
            .ok_or_else(|| GpuModelError::MissingMeta(format!("sam3 tensor {name}")))?;
        if t.dtype != StDtype::F32 || t.shape != shape {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 {name}: {:?} {:?} (want F32 {shape:?})",
                t.dtype, t.shape
            )));
        }
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    pub(super) fn dev(&mut self, host: &[f32]) -> Result<CudaSlice<f32>, GpuModelError> {
        self.bytes += (host.len() * 4) as u64;
        Ok(self.exec.to_device(host)?)
    }

    pub(super) fn vec(&mut self, name: &str, n: usize) -> Result<CudaSlice<f32>, GpuModelError> {
        let v = self.f32s(name, &[n])?;
        self.dev(&v)
    }

    pub(super) fn norm(&mut self, prefix: &str, n: usize) -> Result<Norm, GpuModelError> {
        Ok(Norm {
            w: self.vec(&format!("{prefix}.weight"), n)?,
            b: self.vec(&format!("{prefix}.bias"), n)?,
        })
    }

    /// f32 values laid `[out][in]` row-major -> an f16 GEMM plane, refusing a
    /// value f16 cannot hold (the upload names the plane in that error).
    pub(super) fn plane(
        &mut self,
        w: &[f32],
        in_dim: usize,
        out_dim: usize,
        what: &str,
    ) -> Result<HalfTensor, GpuModelError> {
        debug_assert_eq!(w.len(), in_dim * out_dim);
        // the mma GEMM stages 16-byte f16 rows; a ragged in_dim would drop to
        // the portable fallback silently
        if !in_dim.is_multiple_of(8) {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 {what}: GEMM input width {in_dim} is not a multiple of 8"
            )));
        }
        let buf = self.exec.to_device_f16(w, what)?;
        self.bytes += (w.len() * 2) as u64;
        Ok(HalfTensor {
            buf,
            dims: vec![in_dim, out_dim],
        })
    }

    /// `nn.Linear` / 1x1 conv: `[out, in(,1,1)]` is already the GEMM layout.
    pub(super) fn linear(
        &mut self,
        name: &str,
        shape: &[usize],
    ) -> Result<HalfTensor, GpuModelError> {
        let w = self.f32s(name, shape)?;
        let (out_dim, in_dim) = (shape[0], w.len() / shape[0]);
        self.plane(&w, in_dim, out_dim, name)
    }

    pub(super) fn conv1(
        &mut self,
        prefix: &str,
        cin: usize,
        cout: usize,
    ) -> Result<Conv, GpuModelError> {
        Ok(Conv {
            w: self.linear(&format!("{prefix}.weight"), &[cout, cin, 1, 1])?,
            b: self.vec(&format!("{prefix}.bias"), cout)?,
        })
    }

    /// 3x3 conv `[out][in][ky][kx]` -> `[out][ky][kx][in]`: the im2row is
    /// TAP-outer (col = (ky*3+kx)*C + c) so its loads run contiguous.
    pub(super) fn conv3(&mut self, prefix: &str, c: usize) -> Result<Conv, GpuModelError> {
        let name = format!("{prefix}.weight");
        let w = self.f32s(&name, &[c, c, 3, 3])?;
        let mut out = vec![0f32; w.len()];
        for o in 0..c {
            for i in 0..c {
                for t in 0..9 {
                    out[(o * 9 + t) * c + i] = w[(o * c + i) * 9 + t];
                }
            }
        }
        Ok(Conv {
            w: self.plane(&out, 9 * c, c, &name)?,
            b: self.vec(&format!("{prefix}.bias"), c)?,
        })
    }

    /// 2x2 / stride-2 transposed conv, torch layout `[in][out][ky][kx]`, as the
    /// GEMM it is: output row `(ky*2+kx)*C_out + co`, column `ci`.
    pub(super) fn convt2(
        &mut self,
        prefix: &str,
        cin: usize,
        cout: usize,
    ) -> Result<ConvT, GpuModelError> {
        let name = format!("{prefix}.weight");
        let w = self.f32s(&name, &[cin, cout, 2, 2])?;
        let mut out = vec![0f32; w.len()];
        for ci in 0..cin {
            for co in 0..cout {
                for t in 0..4 {
                    out[(t * cout + co) * cin + ci] = w[(ci * cout + co) * 4 + t];
                }
            }
        }
        Ok(ConvT {
            w: self.plane(&out, cin, 4 * cout, &name)?,
            b: self.vec(&format!("{prefix}.bias"), cout)?,
            cout,
        })
    }

    /// One neck: `fpn_layers.{0: x4, 1: x2, 2: x1}`; `scale_layers.{0, 2}` are
    /// the x4 level's two convTs (index 1 is its GELU), `proj1` the 1x1 and
    /// `proj2` the 3x3.
    fn neck(&mut self, root: &str, d: usize, f: usize) -> Result<Neck, GpuModelError> {
        let l = |i: usize, s: &str| format!("{root}.fpn_layers.{i}.{s}");
        Ok(Neck {
            x4_up: [
                self.convt2(&l(0, "scale_layers.0"), d, d / 2)?,
                self.convt2(&l(0, "scale_layers.2"), d / 2, d / 4)?,
            ],
            x2_up: self.convt2(&l(1, "scale_layers.0"), d, d / 2)?,
            proj1: [
                self.conv1(&l(0, "proj1"), d / 4, f)?,
                self.conv1(&l(1, "proj1"), d / 2, f)?,
                self.conv1(&l(2, "proj1"), d, f)?,
            ],
            proj2: [
                self.conv3(&l(0, "proj2"), f)?,
                self.conv3(&l(1, "proj2"), f)?,
                self.conv3(&l(2, "proj2"), f)?,
            ],
        })
    }
}

impl GpuSam3Vision {
    /// Load from a checkpoint directory (`facebook/sam3` as downloaded).
    /// `max_batch` sizes the resident workspace - the most pictures one pass
    /// will ever be handed.
    pub fn load_dir(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_batch: usize,
    ) -> Result<Self, GpuModelError> {
        let cfg = Sam3VisionConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("sam3 config: {e}")))?;
        let st = super::checkpoint::open(dir)?;
        if !exec.has_dense_pred() || !exec.has_dense_pred_h() || !exec.has_f16_gemm() {
            return Err(GpuModelError::Unsupported(
                "sam3 needs the dense-prediction lane (slots 610-623) and the f16 \
                 tensor-core GEMM, which this kernel pack lacks - rebuild or update the pack"
                    .into(),
            ));
        }
        if !exec.has_sam3_vision() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates SAM 3's image encoder (slots 751-755) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        let h_landing = exec.f16_gemm_h_elected();
        let max_batch = max_batch.max(1);

        // Admission: the f16 planes (half the F32 file's tensors we read; the
        // text tower and the heads are not loaded yet) AND the workspace, which
        // at this resolution is the bigger half - the x4 level's im2row alone
        // is 382 MB a picture.
        let ws_bytes = Workspace::bytes_for(&cfg, max_batch, h_landing);
        let weight_estimate = Self::weight_bytes_for(&cfg);
        exec.vram_load_gate(weight_estimate + ws_bytes, "sam3 image encoder")
            .map_err(GpuModelError::WontFit)?;
        // single-stream engine - must precede every alloc
        exec.disable_event_tracking();

        let (d, ffn, p, ch) = (cfg.hidden, cfg.intermediate, cfg.patch, cfg.channels);
        let (hd, win) = (cfg.head_dim(), cfg.window);
        let mut r = Reader {
            st: &*st,
            exec: &exec,
            bytes: 0,
        };

        // ---- stem ----
        // [d, ch, p, p] flattens to the patch rows' column order; the K pad
        // (588 -> 592) is zero columns, which the zero pad of the rows meets
        let k = ch * p * p;
        let kp = k.next_multiple_of(8);
        let pw = r.f32s(
            &format!("{VIT}.embeddings.patch_embeddings.projection.weight"),
            &[d, ch, p, p],
        )?;
        if st
            .tensor(&format!(
                "{VIT}.embeddings.patch_embeddings.projection.bias"
            ))
            .is_some()
        {
            return Err(GpuModelError::Unsupported(
                "sam3: the patch embedding has a bias, which this graph does not apply".into(),
            ));
        }
        let mut padded = vec![0f32; d * kp];
        for o in 0..d {
            padded[o * kp..o * kp + k].copy_from_slice(&pw[o * k..(o + 1) * k]);
        }
        let patch_w = r.plane(&padded, kp, d, "sam3 patch embedding")?;
        let pos = r.f32s(
            &format!("{VIT}.embeddings.position_embeddings"),
            &[1, cfg.pos_side * cfg.pos_side, d],
        )?;
        let pos = r.dev(&pos)?;
        let ln_pre = r.norm(&format!("{VIT}.layer_norm"), d)?;

        let rw = RopeTable::window(win, hd, cfg.rope_theta);
        let rg = RopeTable::global(cfg.grid(), win, hd, cfg.rope_theta);
        let rope_win = (r.dev(&rw.cos)?, r.dev(&rw.sin)?);
        let rope_glob = (r.dev(&rg.cos)?, r.dev(&rg.sin)?);

        // ---- blocks ----
        let perm = rope_head_perm(hd);
        let heads = cfg.n_heads;
        // rows of a [d, d] projection (or entries of its bias) in rotate-half
        // order: new row h*hd + n reads Meta's row h*hd + perm[n]
        let permute = |w: &[f32], width: usize| -> Vec<f32> {
            let mut out = vec![0f32; w.len()];
            for h in 0..heads {
                for (n, &m) in perm.iter().enumerate() {
                    let (dst, src) = ((h * hd + n) * width, (h * hd + m) * width);
                    out[dst..dst + width].copy_from_slice(&w[src..src + width]);
                }
            }
            out
        };
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let l = |s: &str| format!("{VIT}.layers.{i}.{s}");
            let mut qkv = permute(&r.f32s(&l("attention.q_proj.weight"), &[d, d])?, d);
            qkv.extend(permute(&r.f32s(&l("attention.k_proj.weight"), &[d, d])?, d));
            qkv.extend(r.f32s(&l("attention.v_proj.weight"), &[d, d])?);
            let bq = permute(&r.f32s(&l("attention.q_proj.bias"), &[d])?, 1);
            let bk = permute(&r.f32s(&l("attention.k_proj.bias"), &[d])?, 1);
            let bv = r.f32s(&l("attention.v_proj.bias"), &[d])?;
            blocks.push(Block {
                ln1: r.norm(&l("layer_norm1"), d)?,
                wqkv: r.plane(&qkv, d, 3 * d, &l("attention.qkv"))?,
                bq: r.dev(&bq)?,
                bk: r.dev(&bk)?,
                bv: r.dev(&bv)?,
                bqkv: r.dev(&[bq.as_slice(), bk.as_slice(), bv.as_slice()].concat())?,
                wo: r.linear(&l("attention.o_proj.weight"), &[d, d])?,
                bo: r.vec(&l("attention.o_proj.bias"), d)?,
                ln2: r.norm(&l("layer_norm2"), d)?,
                fc1: r.linear(&l("mlp.fc1.weight"), &[ffn, d])?,
                fc1_b: r.vec(&l("mlp.fc1.bias"), ffn)?,
                fc2: r.linear(&l("mlp.fc2.weight"), &[d, ffn])?,
                fc2_b: r.vec(&l("mlp.fc2.bias"), d)?,
                global: cfg.is_global(i),
            });
        }
        let ones = r.dev(&vec![1.0f32; d])?;

        // ---- necks ----
        let f = cfg.fpn_dim;
        let det_neck = r.neck(DET_NECK, d, f)?;
        let trk_neck = r.neck(TRK_NECK, d, f)?;
        let conv_s0 = r.conv1(&format!("{TRK_DEC}.conv_s0"), f, f / 8)?;
        let conv_s1 = r.conv1(&format!("{TRK_DEC}.conv_s1"), f, f / 4)?;

        let weight_bytes = r.bytes;
        let ws = Workspace::new(&exec, &cfg, kp, max_batch, h_landing)?;
        tracing::info!(
            layers = cfg.n_layer,
            hidden = d,
            tokens = cfg.tokens(),
            max_batch,
            weights_mib = weight_bytes >> 20,
            workspace_mib = ws.bytes >> 20,
            "sam3 image encoder resident"
        );
        Ok(Self {
            exec,
            cfg,
            kp,
            patch_w,
            pos,
            ln_pre,
            rope_win,
            rope_glob,
            blocks,
            ones,
            det_neck,
            trk_neck,
            conv_s0,
            conv_s1,
            ws,
            weight_bytes,
            video_frames: false,
        })
    }

    /// What [`Self::load_dir`] will make resident, before it reads a byte -
    /// the admission gate's weight half. f16 GEMM planes, f32 vectors.
    fn weight_bytes_for(cfg: &Sam3VisionConfig) -> u64 {
        let (d, ffn, f) = (cfg.hidden, cfg.intermediate, cfg.fpn_dim);
        let kp = (cfg.channels * cfg.patch * cfg.patch).next_multiple_of(8);
        let block = 2 * (3 * d * d + d * d + 2 * d * ffn) + 4 * (8 * d + ffn);
        let neck = 2 * (d * 2 * d + (d / 2) * d + d * 2 * d)
            + 2 * ((d / 4) * f + (d / 2) * f + d * f)
            + 2 * 3 * 9 * f * f
            + 4 * (3 * d + 6 * f);
        let fixed = 2 * kp * d
            + 4 * (cfg.window_tokens() * d + 2 * d)
            + 4 * (cfg.window_tokens() + cfg.tokens()) * cfg.head_dim();
        (cfg.n_layer * block + 2 * neck + fixed + 2 * f * (f / 8 + f / 4)) as u64
    }
}

impl Workspace {
    /// Element counts per picture, in one place so the admission estimate and
    /// the allocation cannot drift apart.
    fn plan(cfg: &Sam3VisionConfig, kp: usize) -> Plan {
        let (t, d, f) = (cfg.tokens(), cfg.hidden, cfg.fpn_dim);
        let px4 = 16 * t; // 288^2
        let px2 = 4 * t; // 144^2
        Plan {
            px: cfg.image_size * cfg.image_size * cfg.channels,
            rows16: t * kp,
            row: t * d,
            ff: t * cfg.intermediate,
            // widest f32 GEMM landing: a convT into 4 * C_out, or a 1x1 at x4
            g32: (t * 2 * d).max(px2 * d).max(t * d),
            y32: px4 * f,
            // x4's first convT out [144^2][d/2] / x2's [144^2][d/2], then x4's
            // second [288^2][d/4]
            h16a: px2 * (d / 2),
            h16b: px4 * (d / 4).max(f),
            col16: px4 * 9 * f,
            levels: [px4 * f, px2 * f, t * f],
            s0: px4 * (f / 8),
            s1: px2 * (f / 4),
        }
    }

    pub(super) fn bytes_for(cfg: &Sam3VisionConfig, pics: usize, h_landing: bool) -> u64 {
        let kp = (cfg.channels * cfg.patch * cfg.patch).next_multiple_of(8);
        let p = Self::plan(cfg, kp);
        let lv: usize = p.levels.iter().sum();
        let f32s = p.row + if h_landing { 0 } else { p.ff } + p.g32 + p.y32 + 2 * lv + p.s0 + p.s1;
        // n16, q, k, v, att, proj: six row planes; qkv three
        let f16s = p.rows16 + 9 * p.row + p.ff + p.h16a + p.h16b + p.col16;
        (pics * (f32s * 4 + f16s * 2 + p.px)) as u64
    }

    fn new(
        exec: &GpuExecutor,
        cfg: &Sam3VisionConfig,
        kp: usize,
        cap: usize,
        h_landing: bool,
    ) -> Result<Self, GpuModelError> {
        let p = Self::plan(cfg, kp);
        let f = |n: usize| exec.alloc(cap * n);
        let h = |n: usize| exec.alloc_f16(cap * n);
        Ok(Self {
            cap,
            px: exec.alloc_u8(cap * p.px)?,
            rows16: h(p.rows16)?,
            x: f(p.row)?,
            n16: h(p.row)?,
            qkv: h(3 * p.row)?,
            q: h(p.row)?,
            k: h(p.row)?,
            v: h(p.row)?,
            att: h(p.row)?,
            proj: h(p.row)?,
            ff: h(p.ff)?,
            land32: if h_landing { None } else { Some(f(p.ff)?) },
            g32: f(p.g32)?,
            y32: f(p.y32)?,
            h16a: h(p.h16a)?,
            h16b: h(p.h16b)?,
            col16: h(p.col16)?,
            det: [f(p.levels[0])?, f(p.levels[1])?, f(p.levels[2])?],
            trk: [f(p.levels[0])?, f(p.levels[1])?, f(p.levels[2])?],
            trk_s0: f(p.s0)?,
            trk_s1: f(p.s1)?,
            bytes: Self::bytes_for(cfg, cap, h_landing),
        })
    }
}

struct Plan {
    px: usize,
    rows16: usize,
    row: usize,
    ff: usize,
    g32: usize,
    y32: usize,
    h16a: usize,
    h16b: usize,
    col16: usize,
    levels: [usize; 3],
    s0: usize,
    s1: usize,
}
