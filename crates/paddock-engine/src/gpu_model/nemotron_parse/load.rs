//! Checkpoint -> device: the tower ([`RadioTower::load`]), the decoder and
//! the MTP head, all from the release's own F32 `model.safetensors`.
//!
//! - GEMM planes round F32 -> f16 to nearest (every tensor is far inside f16
//!   range; `to_device_f16` refuses one that is not);
//! - norms and biases stay f32 (the fused seams read them per element);
//! - the embedding is kept TWICE: as an f32 gather table pre-multiplied by
//!   sqrt(d_model) = 32 (a power of two - the multiply is exact, and the
//!   reference scales the gathered row the same way), and as the f16 plane the
//!   tied LM head multiplies by. 297 MB + 148 MB; the gather then reads the
//!   checkpoint's own F32 rows, which is what the prompt and the MTP draft's
//!   input embed in the reference.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::nemotron_parse::NemotronParseConfig;
use paddock_models::safetensors::ShardedSafetensors;

use super::batch::BatchState;
use super::chunked::{ChunkedPrefill, PageEnc};
use super::vision::{Norm, RadioTower, TOWER_PASS_ROWS, Up, f32_exact};
use crate::gpu::{GpuExecutor, HalfTensor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;

pub(crate) struct DecLayer {
    pub ln1: Norm,
    pub wq: HalfTensor,
    pub bq: CudaSlice<f32>,
    pub wk: HalfTensor,
    pub bk: CudaSlice<f32>,
    pub wv: HalfTensor,
    pub bv: CudaSlice<f32>,
    pub wo: HalfTensor,
    pub bo: CudaSlice<f32>,
    pub ln2: Norm,
    pub fc1: HalfTensor,
    pub fc1_b: CudaSlice<f32>,
    pub fc2: HalfTensor,
    pub fc2_b: CudaSlice<f32>,
}

/// The trained one-step MTP head: draft = head(extra_head(h + extra_proj(32 E[t]))).
pub(crate) struct Mtp {
    pub proj: HalfTensor,
    pub proj_b: CudaSlice<f32>,
    pub head: HalfTensor,
    pub head_b: CudaSlice<f32>,
    /// the tied LM head again in e4m3 with one f32 scale a row, the DRAFT's
    /// head only: a draft steers acceptance and never output (verify scores
    /// every emitted token on the f16 head), so its half-width stream is
    /// free. Measured on the F64 fixtures' 13.5K drafts: acceptance 96.18%
    /// against the f16 head's 96.28% (argmax equal on 99.4%), for 74 MB a
    /// round instead of 148. None = the pack lacks the row quantizer or the
    /// e4m3 W16 GEMM.
    pub head8: Option<crate::gpu::F8RowPlane>,
}

pub struct GpuNemotronParse {
    pub(crate) exec: Arc<GpuExecutor>,
    pub cfg: NemotronParseConfig,
    pub(crate) tower: RadioTower,
    /// `[vocab][d_model]` f32, x 32 - the input gather table
    pub(crate) embed32: CudaSlice<f32>,
    /// the tied LM head, `[d_model -> vocab]` f16
    pub(crate) head: HalfTensor,
    pub(crate) ln_emb: Norm,
    pub(crate) layers: Vec<DecLayer>,
    pub(crate) ln_f: Norm,
    pub(crate) mtp: Option<Mtp>,
    pub(crate) max_ctx: usize,
    pub(crate) kv_dtype: KvDtype,
    pub(crate) batch: Option<BatchState>,
    pub(crate) chunked: Vec<ChunkedPrefill>,
    pub(crate) enc: VecDeque<PageEnc>,
    /// the tower pass in flight on its lane, if any
    pub(crate) flight: Option<super::chunked::Flight>,
    pub(crate) weights_bytes: u64,
    /// decode-class rows (decode ticks, verify rounds, the draft's MTP
    /// planes) run their linears and the head on `dense_w16` - one
    /// batch-invariant f16 class at every row count, streaming at the die's
    /// read rate where the GEMV / tiled arms elected by row count stream
    /// below it. Prefill and mixed-tick rows keep the tiled GEMMs.
    pub(crate) w16: bool,
    /// the batch-invariant walk (`walk.rs`): decode-class rows on dense_w16
    /// + the pow2 rows partial, prefill rows on the unsplit rings
    pub(crate) inv: bool,
}

impl GpuNemotronParse {
    /// Load a release directory. `max_ctx` caps prefix + prompt + output
    /// (at most the checkpoint's 16384); `kv_dtype` is the paged pool's class.
    pub fn load_dir(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_ctx: usize,
        kv_dtype: KvDtype,
    ) -> Result<Self, GpuModelError> {
        let cfg = NemotronParseConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("nemotron-parse config: {e}")))?;
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("nemotron-parse safetensors: {e}")))?;
        if !exec.has_tower_ffn_fused() {
            return Err(GpuModelError::Unsupported(
                "nemotron-parse's decoder needs the fused FFN landings (slots 834/835) - rebuild \
                 or update the pack"
                    .into(),
            ));
        }
        let file_bytes: u64 = std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|e| e.metadata().ok().map(|m| m.len()))
            .sum();
        // f16 planes are half the file; the gather table adds the embedding
        // back at f32 and the tower workspace is ~1 GB at its pass size
        let tower_ws = TOWER_PASS_ROWS as u64 * 36_000;
        let gather = (cfg.vocab * cfg.d_model * 4) as u64;
        let head8 = (cfg.vocab * cfg.d_model) as u64;
        exec.vram_load_gate(file_bytes / 2 + gather + head8 + tower_ws, "nemotron-parse")
            .map_err(GpuModelError::WontFit)?;
        exec.disable_event_tracking();

        // The tower runs on its own lane (a forked stream on the same
        // context, the lower dispatch priority): a page's ~0.5 s pass is
        // compute-bound and the decode ticks are bandwidth-bound, so they
        // overlap instead of taking turns (see chunked.rs, the encoder budget)
        let tower_exec = Arc::new(exec.fork_stream()?);
        let tower = RadioTower::load(tower_exec, &st, &cfg, TOWER_PASS_ROWS)?;
        let (dm, ffn, v) = (cfg.d_model, cfg.ffn, cfg.vocab);
        let kv = cfg.n_kv_heads * cfg.head_dim();
        let mut up = Up {
            exec: &exec,
            bytes: 0,
        };

        let emb = f32_exact(&st, "decoder.embed_tokens.weight", &[v, dm])?;
        let head = up.plane(&emb, dm, v, "tied LM head")?;
        let scaled: Vec<f32> = emb.iter().map(|&x| x * cfg.embed_scale).collect();
        drop(emb);
        let embed32 = up.dev(&scaled)?;
        drop(scaled);
        let ln_emb = up.norm(&st, "decoder.layernorm_embedding", dm)?;
        let ln_f = up.norm(&st, "decoder.layer_norm", dm)?;

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let l = |s: &str| format!("decoder.layers.{i}.{s}");
            let lin = |up: &mut Up,
                       name: &str,
                       out: usize,
                       inp: usize|
             -> Result<HalfTensor, GpuModelError> {
                let w = f32_exact(&st, &l(&format!("{name}.weight")), &[out, inp])?;
                up.plane(&w, inp, out, &l(name))
            };
            layers.push(DecLayer {
                ln1: up.norm(&st, &l("self_attn_layer_norm"), dm)?,
                wq: lin(&mut up, "self_attn.q_proj", dm, dm)?,
                bq: up.vec(&st, &l("self_attn.q_proj.bias"), dm)?,
                wk: lin(&mut up, "self_attn.k_proj", kv, dm)?,
                bk: up.vec(&st, &l("self_attn.k_proj.bias"), kv)?,
                wv: lin(&mut up, "self_attn.v_proj", kv, dm)?,
                bv: up.vec(&st, &l("self_attn.v_proj.bias"), kv)?,
                wo: lin(&mut up, "self_attn.out_proj", dm, dm)?,
                bo: up.vec(&st, &l("self_attn.out_proj.bias"), dm)?,
                ln2: up.norm(&st, &l("final_layer_norm"), dm)?,
                fc1: lin(&mut up, "fc1", ffn, dm)?,
                fc1_b: up.vec(&st, &l("fc1.bias"), ffn)?,
                fc2: lin(&mut up, "fc2", dm, ffn)?,
                fc2_b: up.vec(&st, &l("fc2.bias"), dm)?,
            });
        }
        // positions do not exist in this decoder; a checkpoint that ships
        // them is a different graph (the reference's own loader drops
        // embed_positions - so would we, silently, without this check)
        if st.bytes("decoder.embed_positions.weight").is_some() {
            return Err(GpuModelError::Unsupported(
                "nemotron-parse: decoder.embed_positions present - this graph has no positions"
                    .into(),
            ));
        }
        let mut mtp = if cfg.n_mtp_heads == 1 {
            let w = |n: &str| f32_exact(&st, n, &[dm, dm]);
            Some(Mtp {
                proj: up.plane(&w("decoder.extra_proj.0.weight")?, dm, dm, "mtp extra_proj")?,
                proj_b: up.vec(&st, "decoder.extra_proj.0.bias", dm)?,
                head: up.plane(
                    &w("decoder.extra_heads.0.weight")?,
                    dm,
                    dm,
                    "mtp extra_head",
                )?,
                head_b: up.vec(&st, "decoder.extra_heads.0.bias", dm)?,
                head8: None,
            })
        } else {
            None
        };
        let mut head8_bytes = 0u64;
        if let Some(m) = mtp.as_mut().filter(|_| {
            exec.has_np_head8() && paddock_models::dev_var_os!("PADDOCK_NP_NO_HEAD8").is_none()
        }) {
            // quantized from the f32 gather table: its x32 is a power of two,
            // so the e4m3 bytes are the embedding's own and only the row
            // scales carry the factor back out
            let h8 = exec.np_head8(&embed32, v, dm, 1.0 / cfg.embed_scale)?;
            head8_bytes = (h8.data.len() + 4 * h8.scale.len()) as u64;
            m.head8 = Some(h8);
        }
        let weights_bytes = up.bytes + tower.weight_bytes + head8_bytes;
        // a pack whose slot 682 predates the f16 class refuses dtype 2: ask it
        // once on the head's first rows rather than at the first decode tick
        let w16 =
            exec.has_dense_w16() && paddock_models::dev_var_os!("PADDOCK_NP_NO_W16").is_none() && {
                let x = exec.alloc_f16(dm)?;
                let mut y = exec.alloc(v)?;
                exec.dense_w16_f16(&head, &x, &mut y, v, 1).is_ok()
            };
        let inv = w16
            && exec.has_f16_gemm_ring()
            && exec.has_attn_rows_partial_pow2()
            && paddock_models::dev_var_os!("PADDOCK_NP_NO_INV").is_none();
        tracing::info!(
            "nemotron-parse: {}decode-class linears on {}, MTP draft head {}",
            if inv { "batch-invariant walk, " } else { "" },
            if w16 {
                "dense_w16 f16"
            } else {
                "the row-count GEMMs"
            },
            if mtp.as_ref().is_some_and(|m| m.head8.is_some()) {
                "e4m3"
            } else {
                "f16"
            },
        );
        Ok(Self {
            max_ctx: max_ctx.clamp(1, cfg.max_ctx),
            cfg,
            exec,
            tower,
            embed32,
            head,
            ln_emb,
            layers,
            ln_f,
            mtp,
            kv_dtype,
            batch: None,
            chunked: Vec::new(),
            enc: VecDeque::new(),
            flight: None,
            weights_bytes,
            w16,
            inv,
        })
    }
}
