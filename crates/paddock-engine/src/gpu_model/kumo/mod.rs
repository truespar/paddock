//! Kumo-Tabular on CUDA - NVIDIA's tabular foundation model, the prepared-
//! table graph at F32 (the checkpoints' own precision; nothing is narrowed).
//! The op train is in `packs/cuda/src/kumo.cuh`'s header; this module holds
//! the weights, the fitted contexts and the `TabularBackend` the tabular
//! service drives.
//!
//! The same graph runs on Metal (`paddock-metal`'s kumo.rs), and the two
//! lanes agree on what a request means: context-only column statistics, query
//! rows that never become keys, medium/large Test-GQA (query rows read the
//! first two KV heads, context rows every head), and a fitted context that
//! keeps the column-attention and ICL keys/values a later query replays.
//!
//! Replay is exact by construction: every GEMM walks the same k sequence
//! whatever tile or row count it runs at, and attention walks keys in fixed
//! chunks whatever query tile holds the row - so the context rows of a direct
//! pass and of a fit produce the same keys, and a replayed query the same
//! bits as the direct pass that carried it.
//!
//! Precision class: F32 activations and weights; GEMMs on the tensor cores as
//! 3xTF32 with a round-nearest F32 accumulator per k tile (the house F32 GEMM
//! class, see kumo.cuh 698); norms, rope, softmax and GELU in F32 CUDA-core
//! arithmetic with PyTorch's operation order.

mod block;
mod forward;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::kumo::{KumoConfig, Table};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use crate::gpu::GpuExecutor;
use crate::tabular::{Info, Output, TabularBackend};

pub use crate::gpu_model::gpt_oss::GpuModelError;

/// The most a fitted-context cache may hold, whatever the budget leaves.
const CACHE_CAP: u64 = 1 << 30;

/// One attention's replayable keys and values: `[rows][heads][hd]`, heads
/// being the first `heads` of the block's (the Test-GQA heads for ICL).
struct Kv {
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    heads: usize,
}

/// A fitted neural context, owned by the tabular thread: actual projected
/// keys/values of the column-attention output blocks and the ICL layers, not
/// a memoized input or prediction. One context holds the ensemble members
/// one pass fitted together, member-major like the pass itself.
pub struct KumoContext {
    columns: Vec<Kv>,
    layers: Vec<Kv>,
    /// each member's column means and categorical flags, `[members][cols]`
    means: Vec<f32>,
    categorical: Vec<bool>,
    members: usize,
    rows: usize,
    bytes: u64,
}

impl KumoContext {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The ensemble members this context holds.
    pub fn members(&self) -> usize {
        self.members
    }
}

/// The single-table request limits (`Table::validate`): a member pass never
/// carries more rows, cells or columns than one table may, so it stays inside
/// the arena ceiling one table is priced at.
const MAX_ROWS: usize = 4096 + 1024;
const MAX_CELLS: usize = 131_072;
const MAX_COLUMNS: usize = 500;

/// One loaded checkpoint.
pub struct GpuKumo {
    exec: Arc<GpuExecutor>,
    pub config: KumoConfig,
    weights: HashMap<String, CudaSlice<f32>>,
    weight_bytes: u64,
    cache_budget: u64,
    /// the resident pass planes (see `forward::Workspace`)
    ws: Option<forward::Workspace>,
    /// run the op-by-op reference blocks instead of the fused ones (see
    /// `block`); forced on by a pack without the fused slots
    reference: bool,
    /// The row blocks' rope inv_freq vectors, one per distinct value (they
    /// are usually one and the same), and the rope table each `..._transform
    /// .0.inv_freq` weight reads: the fused path's rope comes from these
    /// tables (kumo.cuh 713), filled per row length.
    rope_src: Vec<String>,
    rope: HashMap<String, usize>,
}

impl GpuKumo {
    /// Load an exported checkpoint directory (`config.json` +
    /// `model.safetensors`, the lossless F32 export). The inventory is checked
    /// in full before the first allocation.
    pub fn load(exec: Arc<GpuExecutor>, dir: &Path) -> Result<Self, GpuModelError> {
        if !exec.has_kumo() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates the Kumo-Tabular lane (slots 698-708) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        let bad = |e: String| GpuModelError::Unsupported(format!("Kumo-Tabular: {e}"));
        let config = KumoConfig::read(dir).map_err(|e| bad(e.to_string()))?;
        let st = ShardedSafetensors::open_dir(dir).map_err(|e| bad(e.to_string()))?;
        let weight_bytes = config
            .validate_weights(&st)
            .map_err(|e| bad(e.to_string()))?;
        exec.vram_load_gate(weight_bytes, "Kumo-Tabular")
            .map_err(GpuModelError::WontFit)?;
        // one stream, one owning thread - must precede every alloc
        exec.disable_event_tracking();
        let mut weights = HashMap::new();
        let (mut rope_src, mut rope) = (Vec::<(String, Vec<u32>)>::new(), HashMap::new());
        for name in st.names() {
            let (info, bytes) = st
                .bytes(name)
                .ok_or_else(|| bad(format!("missing {name}")))?;
            if info.dtype != StDtype::F32 {
                return Err(bad(format!("{name}: not F32")));
            }
            let host: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if name.ends_with("_transform.0.inv_freq") {
                // same bits, same table
                let bits: Vec<u32> = host.iter().map(|v| v.to_bits()).collect();
                let t = match rope_src.iter().position(|(_, b)| *b == bits) {
                    Some(t) => t,
                    None => {
                        rope_src.push((name.to_owned(), bits));
                        rope_src.len() - 1
                    }
                };
                rope.insert(name.to_owned(), t);
            }
            weights.insert(name.to_owned(), exec.to_device(&host)?);
        }
        exec.synchronize()?;
        // Fitted contexts share what the budget leaves after the weights:
        // a quarter of it, never more than 1 GiB (the Metal lane's rule).
        let cache_budget = exec.vram_headroom().map_or(0, |h| (h / 4).min(CACHE_CAP));
        let reference = !exec.has_kumo_fused();
        Ok(Self {
            exec,
            config,
            weights,
            weight_bytes,
            cache_budget,
            ws: None,
            reference,
            rope_src: rope_src.into_iter().map(|(n, _)| n).collect(),
            rope,
        })
    }

    /// Run the op-by-op reference blocks (on) or the fused ones (off, the
    /// default). The two agree to the bit; the switch is the instrument the
    /// fused path is gated and measured against. A pack without the fused
    /// slots (709-713) stays on the reference. Returns the path in effect.
    pub fn set_reference_path(&mut self, on: bool) -> bool {
        self.reference = on || !self.exec.has_kumo_fused();
        self.reference
    }

    /// Bytes the resident pass planes hold right now.
    pub fn workspace_bytes(&self) -> u64 {
        self.ws.as_ref().map_or(0, forward::Workspace::bytes)
    }

    /// Hand the pass planes back to the driver (a quiet endpoint should not
    /// sit on the largest table it ever saw); the next request re-grows them.
    pub fn release_workspace(&mut self) {
        if self.ws.take().is_some() {
            self.exec.trim_mem_pool();
        }
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    pub fn executor(&self) -> &Arc<GpuExecutor> {
        &self.exec
    }

    fn w(&self, name: &str) -> &CudaSlice<f32> {
        &self.weights[name]
    }

    /// Device bytes a fitted context of `rows` context rows and `cols`
    /// columns holds: per embedding layer the output block's keys and values
    /// over every column's inducing points, per ICL layer the Test-GQA heads'.
    pub fn context_bytes(&self, rows: usize, cols: usize) -> u64 {
        let c = &self.config;
        (8 * (c.embedding_layers * cols * c.inducing * c.cell
            + c.layers * rows * c.hidden * c.query_kv_heads / c.heads)) as u64
    }

    /// Members one pass may carry at `rows` rows and `cols` columns a member:
    /// together they stay within one table's limits (see [`MAX_ROWS`]). A pack
    /// without the member slots runs them one at a time.
    fn group(&self, rows: usize, cols: usize) -> usize {
        if !self.exec.has_kumo_members() {
            return 1;
        }
        (MAX_ROWS / rows.max(1))
            .min(MAX_CELLS / (rows * cols).max(1))
            .min(MAX_COLUMNS / cols.max(1))
            .max(1)
    }

    /// Per-member outputs of one pass: its time split evenly between them.
    fn outputs(out: forward::PassOut) -> Vec<Output> {
        let n = out.values.len().max(1) as f64;
        out.values
            .into_iter()
            .map(|values| Output {
                values,
                gpu_seconds: out.gpu_seconds / n,
                workspace_bytes: out.workspace_bytes,
            })
            .collect()
    }

    fn same_shape(tables: &[Table<'_>]) -> Result<(), GpuModelError> {
        let t = tables
            .first()
            .ok_or_else(|| GpuModelError::Unsupported("Kumo-Tabular: no table".into()))?;
        if tables.iter().any(|m| {
            m.categorical.len() != t.categorical.len()
                || m.y.len() != t.y.len()
                || m.query_rows != t.query_rows
        }) {
            return Err(GpuModelError::Unsupported(
                "Kumo-Tabular: ensemble members must share rows and columns".into(),
            ));
        }
        Ok(())
    }

    pub fn predict(&mut self, t: &Table<'_>) -> Result<Output, GpuModelError> {
        Ok(self.predict_many(std::slice::from_ref(t))?.remove(0))
    }

    /// An ensemble's members - same rows, same columns, their own values,
    /// labels and column kinds - in as few passes as the limits allow. Each
    /// member's outputs are the bits its own pass gives.
    pub fn predict_many(&mut self, tables: &[Table<'_>]) -> Result<Vec<Output>, GpuModelError> {
        for t in tables {
            t.validate(&self.config.task)
                .map_err(GpuModelError::Unsupported)?;
        }
        Self::same_shape(tables)?;
        let t = &tables[0];
        let per = self.group(t.y.len() + t.query_rows, t.categorical.len());
        let mut out = Vec::with_capacity(tables.len());
        for group in tables.chunks(per) {
            out.extend(Self::outputs(self.execute(group, forward::Mode::Plain)?));
        }
        Ok(out)
    }

    /// Encode a context once: its column and ICL keys/values stay resident
    /// for [`Self::query`]. The output carries no values.
    pub fn fit(&mut self, t: &Table<'_>) -> Result<(KumoContext, Output), GpuModelError> {
        let (mut ctx, mut out) = self.fit_many(std::slice::from_ref(t))?;
        Ok((ctx.remove(0), out.remove(0)))
    }

    /// [`Self::fit`] for an ensemble's members: one context a pass, each
    /// holding the members that pass fitted (in order).
    pub fn fit_many(
        &mut self,
        tables: &[Table<'_>],
    ) -> Result<(Vec<KumoContext>, Vec<Output>), GpuModelError> {
        let mut context = Vec::with_capacity(tables.len());
        for t in tables {
            let n = t.y.len().checked_mul(t.categorical.len()).ok_or_else(|| {
                GpuModelError::Unsupported("Kumo-Tabular: context shape overflow".into())
            })?;
            let t = Table {
                x: t.x.get(..n).ok_or_else(|| {
                    GpuModelError::Unsupported("Kumo-Tabular: context shape mismatch".into())
                })?,
                y: t.y,
                categorical: t.categorical,
                query_rows: 0,
            };
            t.validate_context(&self.config.task)
                .map_err(GpuModelError::Unsupported)?;
            context.push(t);
        }
        Self::same_shape(&context)?;
        let (cols, rows) = (context[0].categorical.len(), context[0].y.len());
        let bytes = self.context_bytes(rows, cols) * context.len() as u64;
        if self.exec.vram_headroom().is_some_and(|h| bytes > h) {
            return Err(GpuModelError::WontFit(format!(
                "fitted Kumo contexts need {bytes} bytes beyond what the budget leaves"
            )));
        }
        let per = self.group(rows, cols);
        let (mut contexts, mut outs) = (Vec::new(), Vec::with_capacity(context.len()));
        for group in context.chunks(per) {
            let members = group.len();
            let c = &self.config;
            // every element is written by the fit's record pass before any
            // replay reads it
            let kv = |n: usize, heads: usize| -> Result<Kv, GpuModelError> {
                Ok(Kv {
                    key: self.exec.kumo_plane(n)?,
                    value: self.exec.kumo_plane(n)?,
                    heads,
                })
            };
            let mut ctx = KumoContext {
                columns: (0..c.embedding_layers)
                    .map(|_| kv(members * cols * c.inducing * c.cell, 4))
                    .collect::<Result<_, _>>()?,
                layers: (0..c.layers)
                    .map(|_| {
                        kv(
                            members * rows * c.hidden * c.query_kv_heads / c.heads,
                            c.query_kv_heads,
                        )
                    })
                    .collect::<Result<_, _>>()?,
                means: group.iter().flat_map(forward::means).collect(),
                categorical: group
                    .iter()
                    .flat_map(|t| t.categorical.iter().copied())
                    .collect(),
                members,
                rows,
                bytes: self.context_bytes(rows, cols) * members as u64,
            };
            outs.extend(Self::outputs(
                self.execute(group, forward::Mode::Record(&mut ctx))?,
            ));
            contexts.push(ctx);
        }
        Ok((contexts, outs))
    }

    /// Predict `rows` query rows (`x` row-major over the fitted columns)
    /// against a fitted context of one member.
    pub fn query(
        &mut self,
        ctx: &KumoContext,
        x: &[f32],
        rows: usize,
    ) -> Result<Output, GpuModelError> {
        Ok(self.query_many(ctx, &[x], rows)?.remove(0))
    }

    /// [`Self::query`] for every member a context holds: `xs[m]` is member
    /// m's `rows` query rows. Rows past what one pass may carry for all the
    /// members go in slices (each row's outputs are its own, so the slices
    /// change no bit).
    pub fn query_many(
        &mut self,
        ctx: &KumoContext,
        xs: &[&[f32]],
        rows: usize,
    ) -> Result<Vec<Output>, GpuModelError> {
        let members = ctx.members;
        let cols = ctx.categorical.len() / members.max(1);
        if xs.len() != members
            || !(1..=1024).contains(&rows)
            || (rows + ctx.rows) * cols > MAX_CELLS
            || xs
                .iter()
                .any(|x| x.len() != rows * cols || x.iter().any(|v| v.is_infinite()))
        {
            return Err(GpuModelError::Unsupported(
                "Kumo-Tabular: invalid fitted-context query".into(),
            ));
        }
        let slice = (MAX_ROWS / members)
            .min(MAX_CELLS / (members * cols))
            .max(1);
        let mut out = (0..members)
            .map(|_| Output {
                values: Vec::with_capacity(rows * self.config.outputs()),
                gpu_seconds: 0.,
                workspace_bytes: 0,
            })
            .collect::<Vec<_>>();
        for a in (0..rows).step_by(slice) {
            let b = (a + slice).min(rows);
            let tables = xs
                .iter()
                .enumerate()
                .map(|(m, x)| Table {
                    x: &x[a * cols..b * cols],
                    y: &[],
                    categorical: &ctx.categorical[m * cols..(m + 1) * cols],
                    query_rows: b - a,
                })
                .collect::<Vec<_>>();
            for (o, part) in out.iter_mut().zip(Self::outputs(
                self.execute(&tables, forward::Mode::Replay(ctx))?,
            )) {
                o.values.extend(part.values);
                o.gpu_seconds += part.gpu_seconds;
                o.workspace_bytes = o.workspace_bytes.max(part.workspace_bytes);
            }
        }
        Ok(out)
    }
}

impl TabularBackend for GpuKumo {
    type Context = KumoContext;

    fn info(&self) -> Info {
        Info {
            config: self.config.clone(),
            weight_bytes: self.weight_bytes,
        }
    }

    fn predict(&mut self, input: &Table<'_>) -> Result<Output, String> {
        GpuKumo::predict(self, input).map_err(|e| e.to_string())
    }

    fn cache_budget(&self) -> u64 {
        self.cache_budget
    }

    fn context_bytes(&self, rows: usize, columns: usize) -> u64 {
        GpuKumo::context_bytes(self, rows, columns)
    }

    fn fit(&mut self, input: &Table<'_>) -> Result<(KumoContext, Output), String> {
        GpuKumo::fit(self, input).map_err(|e| e.to_string())
    }

    fn query(&mut self, ctx: &KumoContext, x: &[f32], rows: usize) -> Result<Output, String> {
        GpuKumo::query(self, ctx, x, rows).map_err(|e| e.to_string())
    }

    fn predict_many(&mut self, inputs: &[Table<'_>]) -> Result<Vec<Output>, String> {
        GpuKumo::predict_many(self, inputs).map_err(|e| e.to_string())
    }

    fn fit_many(
        &mut self,
        inputs: &[Table<'_>],
    ) -> Result<(Vec<KumoContext>, Vec<Output>), String> {
        GpuKumo::fit_many(self, inputs).map_err(|e| e.to_string())
    }

    fn query_many(
        &mut self,
        contexts: &[KumoContext],
        xs: &[&[f32]],
        rows: usize,
    ) -> Result<Vec<Output>, String> {
        let mut out = Vec::with_capacity(xs.len());
        let mut at = 0;
        for ctx in contexts {
            let part = xs
                .get(at..at + ctx.members)
                .ok_or("Kumo-Tabular: fewer members than the contexts hold")?;
            out.extend(GpuKumo::query_many(self, ctx, part, rows).map_err(|e| e.to_string())?);
            at += ctx.members;
        }
        if at != xs.len() {
            return Err("Kumo-Tabular: more members than the contexts hold".into());
        }
        Ok(out)
    }

    fn workspace_bytes(&self) -> u64 {
        GpuKumo::workspace_bytes(self)
    }

    fn idle(&mut self) {
        self.release_workspace();
    }
}
