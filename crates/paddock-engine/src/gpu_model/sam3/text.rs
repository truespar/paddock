//! SAM 3's text tower: a concept prompt -> 32 x 256 prompt tokens.
//!
//!   ids [P, 32] (paddock_tokenizer::sam3)
//!   -> token embedding (f32 table, gathered) + learned positions        [P*32, 1024]
//!   -> 24 x pre-LN block: LN -> fused qkv GEMM -> split + biases (no rope)
//!      -> causal attention -> o GEMM -> x += o + b; LN -> fc1 GEMM (bias +
//!      exact GELU in its landing) -> fc2 GEMM -> x += fc2 + b; next LN
//!   -> ln_final (the last seam's norm) -> resizer GEMM 1024 -> 256 + bias
//!
//! Meta's `VETextEncoder` with transformers' names: `text_model.*` is the
//! CLIP tower, `detector_model.text_projection` is Meta's `resizer`. The
//! tower's own `text_projection` (the EOT-pooled CLIP projection) ships in the
//! checkpoint and SAM 3 never uses it - not loaded.
//!
//! Precision is the image encoder's f16 class. CLIP's massive activation shows
//! up here as it always does - block 0's MLP output reaches 1062 and the
//! residual 1221 on Meta's fp32 run - which f16 holds as a GEMM landing and the
//! f32 residual never has to. The features of a prompt depend on nothing but
//! the prompt, so a server caches them per string.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::sam3::{META_LN_EPS, Sam3TextConfig};

use super::load::Reader;
use super::vit::gemm_h;
use super::{Conv, GpuModelError, Norm};
use crate::gpu::{GpuExecutor, HalfTensor};

const TEXT: &str = "detector_model.text_encoder.text_model";
const RESIZER: &str = "detector_model.text_projection";

struct TextBlock {
    ln1: Norm,
    /// q|k|v stacked row-wise, Meta's order (no rope, nothing to permute)
    wqkv: HalfTensor,
    bq: CudaSlice<f32>,
    bk: CudaSlice<f32>,
    bv: CudaSlice<f32>,
    wo: HalfTensor,
    bo: CudaSlice<f32>,
    ln2: Norm,
    fc1: HalfTensor,
    fc1_b: CudaSlice<f32>,
    fc2: HalfTensor,
    fc2_b: CudaSlice<f32>,
}

struct TextWorkspace {
    cap: usize,
    ids: CudaSlice<u32>,
    x: CudaSlice<f32>,
    n16: CudaSlice<f16>,
    qkv: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    att: CudaSlice<f16>,
    proj: CudaSlice<f16>,
    ff: CudaSlice<f16>,
    land32: Option<CudaSlice<f32>>,
    /// q|k|v (3072 columns, 48 blocks) fills the die unsplit, so it takes the
    /// half entry wherever that is elected; o and fc2 always land f32
    qkv_h: bool,
    /// the prompt tokens, [prompts][32][d_model] f32
    feats: CudaSlice<f32>,
}

/// SAM 3's text tower + resizer, resident.
pub struct GpuSam3Text {
    exec: Arc<GpuExecutor>,
    cfg: Sam3TextConfig,
    /// the token embedding table, f32 [vocab][hidden] - gathered exactly, the
    /// residual starts in f32 as Meta's does
    tok_emb: CudaSlice<f32>,
    pos: CudaSlice<f32>,
    blocks: Vec<TextBlock>,
    ln_final: Norm,
    resizer: Conv,
    ones: CudaSlice<f32>,
    ws: TextWorkspace,
    weight_bytes: u64,
}

impl GpuSam3Text {
    pub fn config(&self) -> &Sam3TextConfig {
        &self.cfg
    }
    pub fn max_batch(&self) -> usize {
        self.ws.cap
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Load from a checkpoint directory; `max_prompts` sizes the workspace.
    pub fn load_dir(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_prompts: usize,
    ) -> Result<Self, GpuModelError> {
        let cfg = Sam3TextConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("sam3 config: {e}")))?;
        let st = super::checkpoint::open(dir)?;
        if !exec.has_sam3_text() || !exec.has_f16_gemm_h_gelu() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates SAM 3's text tower (slot 759) - rebuild or update \
                 the pack"
                    .into(),
            ));
        }
        let qkv_h = exec.f16_gemm_h_elected();
        let cap = max_prompts.max(1);
        let (d, ffn, t, vocab) = (cfg.hidden, cfg.intermediate, cfg.context, cfg.vocab);
        // the token table dominates: 49408 x 1024 f32 = 193 MiB
        let est = (vocab * d * 4
            + cfg.n_layer * (2 * (4 * d * d + 2 * d * ffn) + 4 * (9 * d + ffn))
            + cap * t * (4 * d * 2 + 2 * (9 * d + ffn) + 4 * ffn + 4 * cfg.d_model))
            as u64;
        exec.vram_load_gate(est, "sam3 text tower")
            .map_err(GpuModelError::WontFit)?;

        let mut r = Reader {
            st: &*st,
            exec: &exec,
            bytes: 0,
        };
        let te = r.f32s(
            &format!("{TEXT}.embeddings.token_embedding.weight"),
            &[vocab, d],
        )?;
        let tok_emb = r.dev(&te)?;
        drop(te);
        let pe = r.f32s(
            &format!("{TEXT}.embeddings.position_embedding.weight"),
            &[t, d],
        )?;
        let pos = r.dev(&pe)?;

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let l = |s: &str| format!("{TEXT}.encoder.layers.{i}.{s}");
            let mut qkv = r.f32s(&l("self_attn.q_proj.weight"), &[d, d])?;
            qkv.extend(r.f32s(&l("self_attn.k_proj.weight"), &[d, d])?);
            qkv.extend(r.f32s(&l("self_attn.v_proj.weight"), &[d, d])?);
            blocks.push(TextBlock {
                ln1: r.norm(&l("layer_norm1"), d)?,
                wqkv: r.plane(&qkv, d, 3 * d, &l("self_attn.qkv"))?,
                bq: r.vec(&l("self_attn.q_proj.bias"), d)?,
                bk: r.vec(&l("self_attn.k_proj.bias"), d)?,
                bv: r.vec(&l("self_attn.v_proj.bias"), d)?,
                wo: r.linear(&l("self_attn.out_proj.weight"), &[d, d])?,
                bo: r.vec(&l("self_attn.out_proj.bias"), d)?,
                ln2: r.norm(&l("layer_norm2"), d)?,
                fc1: r.linear(&l("mlp.fc1.weight"), &[ffn, d])?,
                fc1_b: r.vec(&l("mlp.fc1.bias"), ffn)?,
                fc2: r.linear(&l("mlp.fc2.weight"), &[d, ffn])?,
                fc2_b: r.vec(&l("mlp.fc2.bias"), d)?,
            });
        }
        let ln_final = r.norm(&format!("{TEXT}.final_layer_norm"), d)?;
        let resizer = Conv {
            w: r.linear(&format!("{RESIZER}.weight"), &[cfg.d_model, d])?,
            b: r.vec(&format!("{RESIZER}.bias"), cfg.d_model)?,
        };
        let ones = r.dev(&vec![1.0f32; d])?;
        let weight_bytes = r.bytes;

        let rows = cap * t;
        let f = |n: usize| exec.alloc(rows * n);
        let h = |n: usize| exec.alloc_f16(rows * n);
        let ws = TextWorkspace {
            cap,
            ids: exec.alloc_u32(rows)?,
            x: f(d)?,
            n16: h(d)?,
            qkv: h(3 * d)?,
            q: h(d)?,
            k: h(d)?,
            v: h(d)?,
            att: h(d)?,
            proj: h(d)?,
            ff: h(ffn)?,
            // always allocated here: a prompt is 32 rows, and at 32 rows the
            // half entry (which never K-splits) puts o's and fc2's 1024
            // columns on 16 blocks - fc2 71 us of a weight stream worth ~11.
            // The f32 entry K-splits to fill the die (fc2 17 us), and the
            // convert is 32 rows.
            land32: Some(f(ffn)?),
            qkv_h,
            feats: f(cfg.d_model)?,
        };
        tracing::info!(
            layers = cfg.n_layer,
            max_prompts = cap,
            weights_mib = weight_bytes >> 20,
            "sam3 text tower resident"
        );
        Ok(Self {
            exec,
            cfg,
            tok_emb,
            pos,
            blocks,
            ln_final,
            resizer,
            ones,
            ws,
            weight_bytes,
        })
    }

    /// Encode `prompts` tokenized prompts (`ids` is `prompts * 32` ids, the
    /// tokenizer's layout). The tokens stay on the device ([`Self::features`]).
    pub fn encode(&mut self, ids: &[u32], prompts: usize) -> Result<(), GpuModelError> {
        let cfg = &self.cfg;
        let (d, t, hd, heads) = (cfg.hidden, cfg.context, cfg.head_dim(), cfg.n_heads);
        if prompts == 0 {
            return Ok(());
        }
        if prompts > self.ws.cap {
            return Err(GpuModelError::BatchTooLarge {
                got: prompts,
                max: self.ws.cap,
            });
        }
        if ids.len() != prompts * t {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 text: {} ids for {prompts} prompt(s), want {}",
                ids.len(),
                prompts * t
            )));
        }
        if let Some(bad) = ids.iter().find(|&&i| i as usize >= cfg.vocab) {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 text: token id {bad} past the {} vocabulary",
                cfg.vocab
            )));
        }
        let exec = self.exec.clone();
        let rows = prompts * t;
        let scale = 1.0 / (hd as f32).sqrt();
        let ws = &mut self.ws;
        exec.upload_u32(ids, &mut ws.ids)?;
        exec.embed_gather_batch(&self.tok_emb, &ws.ids, &mut ws.x, d, rows)?;
        exec.add_rows_bcast(&mut ws.x, &self.pos, rows, t, d)?;
        let first = &self.blocks[0].ln1;
        exec.whisper_ln_f16(&ws.x, &first.w, &first.b, &mut ws.n16, rows, d, META_LN_EPS)?;

        let n_layer = self.blocks.len();
        for li in 0..n_layer {
            let blk = &self.blocks[li];
            gemm_h(
                &exec,
                &blk.wqkv,
                &ws.n16,
                &mut ws.qkv,
                rows,
                if ws.qkv_h { None } else { ws.land32.as_mut() },
            )?;
            exec.sam3_qkv_split_rope_h(
                &ws.qkv, &blk.bq, &blk.bk, &blk.bv, None, &mut ws.q, &mut ws.k, &mut ws.v, d, hd,
                rows, t, scale,
            )?;
            exec.sam3_text_attn_h(&ws.q, &ws.k, &ws.v, &mut ws.att, prompts, t, heads, hd)?;
            gemm_h(
                &exec,
                &blk.wo,
                &ws.att,
                &mut ws.proj,
                rows,
                ws.land32.as_mut(),
            )?;
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
                META_LN_EPS,
            )?;
            exec.matvec_batch_f16_h_gelu(&blk.fc1, &ws.n16, &mut ws.ff, &blk.fc1_b, rows)?;
            gemm_h(
                &exec,
                &blk.fc2,
                &ws.ff,
                &mut ws.proj,
                rows,
                ws.land32.as_mut(),
            )?;
            // the last seam lands ln_final instead of a next block's ln1
            let next = if li + 1 < n_layer {
                &self.blocks[li + 1].ln1
            } else {
                &self.ln_final
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
                META_LN_EPS,
            )?;
        }
        exec.matvec_batch_f16(&self.resizer.w, &ws.n16, &mut ws.feats, rows)?;
        exec.bias_add(&mut ws.feats, &self.resizer.b, rows, cfg.d_model)?;
        Ok(())
    }

    /// The prompt tokens of the last [`Self::encode`], on the device:
    /// `[prompts][32][d_model]` f32.
    pub fn features(&self) -> &CudaSlice<f32> {
        &self.ws.feats
    }

    /// The same, copied out - the gate's view.
    pub fn read_features(&self, prompts: usize) -> Result<Vec<f32>, GpuModelError> {
        let n = prompts.min(self.ws.cap) * self.cfg.context * self.cfg.d_model;
        Ok(self.exec.to_host_len(&self.ws.feats, n)?)
    }
}
