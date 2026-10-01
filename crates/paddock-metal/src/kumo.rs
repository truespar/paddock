//! Native Kumo-Tabular prepared-table inference. One GPU submission, bounded
//! scratch, context-only column statistics and query-isolated ICL attention.
//! F32 throughout; no Python runtime or host neural-network fallback.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use paddock_models::kumo::{KumoConfig, Table, Task};
use paddock_models::safetensors::ShardedSafetensors;
use std::{collections::HashMap, path::Path};

fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Kumo: {}", s.into()))
}
fn upload(device: &MetalDevice, values: &[f32]) -> Result<Buffer> {
    device.upload(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

pub struct Kumo {
    device: MetalDevice,
    pub config: KumoConfig,
    weights: HashMap<String, Buffer>,
    weight_bytes: u64,
    #[cfg(test)]
    unfused: bool,
}

pub struct KumoOutput {
    /// Classification: ten raw logits per query. Regression: 999 raw
    /// quantiles per query. Calibration/inverse transforms belong to recipe.
    pub values: Vec<f32>,
    pub gpu_seconds: f64,
    pub workspace_bytes: u64,
}

struct Kv {
    key: Buffer,
    value: Buffer,
    heads: usize,
}
/// A fitted neural context, owned by the inference thread. These are actual
/// projected attention keys/values, not a memoized input or prediction.
pub struct KumoContext {
    columns: Vec<Kv>,
    layers: Vec<Kv>,
    means: Vec<f32>,
    categorical: Vec<bool>,
    rows: usize,
    bytes: u64,
}
impl KumoContext {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

#[derive(Clone, Copy)]
enum Mode<'a> {
    Plain,
    Record(&'a KumoContext),
    Replay(&'a KumoContext),
}

impl paddock_engine::tabular::TabularBackend for Kumo {
    type Context = KumoContext;
    fn cache_budget(&self) -> u64 {
        (self.device.budget_bytes().saturating_sub(self.weight_bytes) / 4).min(1 << 30)
    }
    fn context_bytes(&self, rows: usize, columns: usize) -> u64 {
        Kumo::context_bytes(self, rows, columns)
    }
    fn fit(
        &mut self,
        input: &Table<'_>,
    ) -> std::result::Result<(KumoContext, paddock_engine::tabular::Output), String> {
        Kumo::fit(self, input)
            .map(|(ctx, out)| {
                (
                    ctx,
                    paddock_engine::tabular::Output {
                        values: out.values,
                        gpu_seconds: out.gpu_seconds,
                        workspace_bytes: out.workspace_bytes,
                    },
                )
            })
            .map_err(|e| e.to_string())
    }
    fn query(
        &mut self,
        ctx: &KumoContext,
        x: &[f32],
        rows: usize,
    ) -> std::result::Result<paddock_engine::tabular::Output, String> {
        Kumo::query(self, ctx, x, rows)
            .map(|out| paddock_engine::tabular::Output {
                values: out.values,
                gpu_seconds: out.gpu_seconds,
                workspace_bytes: out.workspace_bytes,
            })
            .map_err(|e| e.to_string())
    }
    fn info(&self) -> paddock_engine::tabular::Info {
        paddock_engine::tabular::Info {
            config: self.config.clone(),
            weight_bytes: self.weight_bytes,
        }
    }
    fn predict(
        &mut self,
        input: &Table<'_>,
    ) -> std::result::Result<paddock_engine::tabular::Output, String> {
        Kumo::predict(self, input)
            .map(|out| paddock_engine::tabular::Output {
                values: out.values,
                gpu_seconds: out.gpu_seconds,
                workspace_bytes: out.workspace_bytes,
            })
            .map_err(|e| e.to_string())
    }
}

struct Scratch {
    nq: Buffer,
    nk: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    qh: Buffer,
    kh: Buffer,
    att: Buffer,
    tmp: Buffer,
    wide: Buffer,
    gate: Buffer,
}
impl Scratch {
    fn new(d: &MetalDevice, n: usize) -> Result<Self> {
        let a = || d.alloc(n * 4);
        Ok(Self {
            nq: a()?,
            nk: a()?,
            q: a()?,
            k: a()?,
            v: a()?,
            qh: a()?,
            kh: a()?,
            att: a()?,
            tmp: a()?,
            wide: d.alloc(n * 8)?,
            gate: d.alloc(n * 8)?,
        })
    }
}

impl Kumo {
    pub fn load(dir: &Path, budget: Option<u64>) -> Result<Self> {
        let config = KumoConfig::read(dir).map_err(|e| error(e.to_string()))?;
        let st = ShardedSafetensors::open_dir(dir).map_err(|e| error(e.to_string()))?;
        let weight_bytes = config
            .validate_weights(&st)
            .map_err(|e| error(e.to_string()))?;
        let device = MetalDevice::new_planned(budget, weight_bytes)?;
        let mut weights = HashMap::new();
        for name in st.names() {
            let (_, data) = st.bytes(name).ok_or_else(|| error("missing tensor"))?;
            weights.insert(name.to_owned(), device.upload(data)?);
        }
        Ok(Self {
            device,
            config,
            weights,
            weight_bytes,
            #[cfg(test)]
            unfused: false,
        })
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    fn w(&self, name: &str) -> &Buffer {
        &self.weights[name]
    }
    fn mm(
        &self,
        c: &Commands<'_>,
        name: &str,
        x: &Buffer,
        out: &Buffer,
        k: usize,
        n: usize,
        rows: usize,
        offset: usize,
        gelu: bool,
    ) {
        c.dispatch(
            "kumo_mm",
            &[
                self.w(&format!("{name}.weight")),
                x,
                out,
                self.w(&format!("{name}.bias")),
            ],
            &[
                k as u32,
                n as u32,
                rows as u32,
                offset as u32,
                u32::from(gelu),
            ],
            [n.div_ceil(32), rows.div_ceil(32), 1],
            128,
        );
    }
    fn norm(
        &self,
        c: &Commands<'_>,
        x: &Buffer,
        out: &Buffer,
        name: &str,
        d: usize,
        batch: usize,
        len: usize,
        stride: usize,
    ) {
        c.dispatch(
            "kumo_norm",
            &[x, self.w(name), out],
            &[d as u32, len as u32, stride as u32],
            [batch * len, 1, 1],
            256,
        );
    }
    fn copy(c: &Commands<'_>, x: &Buffer, out: &Buffer, period: usize, count: usize) {
        c.dispatch(
            "kumo_copy_repeat",
            &[x, out],
            &[period as u32, count as u32],
            [count.div_ceil(256), 1, 1],
            256,
        );
    }
    fn add(c: &Commands<'_>, a: &Buffer, b: &Buffer, out: &Buffer, count: usize) {
        c.dispatch(
            "kumo_add",
            &[a, b, out],
            &[count as u32],
            [count.div_ceil(256), 1, 1],
            256,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        c: &Commands<'_>,
        s: &Scratch,
        p: &str,
        query: &Buffer,
        kv: &Buffer,
        out: &Buffer,
        batch: usize,
        qlen: usize,
        klen: usize,
        kvstride: usize,
        d: usize,
        heads: usize,
        scaling: u32,
        query_kv_heads: usize,
        cache: Option<(&Kv, bool)>,
    ) {
        let qrows = batch * qlen;
        let krows = batch * klen;
        let hd = d / heads;
        let replay = cache.is_some_and(|(_, replay)| replay);
        let fused = true;
        #[cfg(test)]
        let fused = fused && !self.unfused;
        if klen > 1 {
            self.norm(
                c,
                query,
                &s.nq,
                &format!("{p}.query_norm.weight"),
                d,
                batch,
                qlen,
                qlen,
            );
        }
        if !replay {
            self.norm(
                c,
                kv,
                &s.nk,
                &format!("{p}.key_value_norm.weight"),
                d,
                batch,
                klen,
                kvstride,
            );
        }
        let qkv = format!("{p}.attn.qkv_lin");
        if klen > 1 {
            self.mm(c, &qkv, &s.nq, &s.q, d, d, qrows, 0, false);
        }
        if !replay && fused {
            c.dispatch(
                "kumo_kv",
                &[
                    self.w(&format!("{qkv}.weight")),
                    &s.nk,
                    &s.k,
                    &s.v,
                    self.w(&format!("{qkv}.bias")),
                ],
                &[d as u32, krows as u32],
                [(2 * d).div_ceil(32), krows.div_ceil(32), 1],
                128,
            );
        } else if !replay {
            self.mm(c, &qkv, &s.nk, &s.k, d, d, krows, d, false);
            self.mm(c, &qkv, &s.nk, &s.v, d, d, krows, 2 * d, false);
        }
        for (input, output, rows, len, which) in [
            (&s.q, &s.qh, qrows, qlen, "query"),
            (&s.k, &s.kh, krows, klen, "key"),
        ] {
            if klen == 1 && which == "query" {
                continue;
            }
            if replay && which == "key" {
                continue;
            }
            let freq = if scaling == 2 {
                self.w(&format!("{p}.attn.{which}_transform.0.inv_freq"))
            } else {
                self.w(&format!("{p}.query_norm.weight"))
            };
            c.dispatch(
                "kumo_heads",
                &[input, freq, output],
                &[heads as u32, hd as u32, len as u32, u32::from(scaling == 2)],
                [rows * heads, 1, 1],
                32,
            );
        }
        if let Some((kv, false)) = cache {
            for (src, dst) in [(&s.kh, &kv.key), (&s.v, &kv.value)] {
                c.dispatch(
                    "kumo_cache_heads",
                    &[src, dst],
                    &[heads as u32, kv.heads as u32, hd as u32, krows as u32],
                    [(krows * kv.heads * hd).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
        if scaling > 0 && klen > 1 {
            if scaling == 2 {
                self.mm(
                    c,
                    &format!("{p}.attn.sdpa.query_scaling.gate.0"),
                    &s.qh,
                    &s.gate,
                    hd,
                    64,
                    qrows * heads,
                    0,
                    true,
                );
                self.mm(
                    c,
                    &format!("{p}.attn.sdpa.query_scaling.gate.2"),
                    &s.gate,
                    &s.q,
                    64,
                    hd,
                    qrows * heads,
                    0,
                    false,
                );
            }
            c.dispatch(
                "kumo_scale",
                &[
                    &s.qh,
                    self.w(&format!("{p}.attn.sdpa.query_scaling.head_scale")),
                    &s.q,
                ],
                &[
                    (qrows * d) as u32,
                    hd as u32,
                    heads as u32,
                    klen as u32,
                    u32::from(scaling == 2),
                ],
                [(qrows * d).div_ceil(256), 1, 1],
                256,
            );
        }
        let attention_buffers = [
            &s.qh,
            cache
                .filter(|(_, replay)| *replay)
                .map_or(&s.kh, |(kv, _)| &kv.key),
            cache
                .filter(|(_, replay)| *replay)
                .map_or(&s.v, |(kv, _)| &kv.value),
            &s.att,
        ];
        let mut attention_params = [
            heads as u32,
            hd as u32,
            qlen as u32,
            klen as u32,
            query_kv_heads as u32,
            if replay { 0 } else { klen as u32 },
            cache.filter(|(_, r)| *r).map_or(heads, |(kv, _)| kv.heads) as u32,
            0,
            qlen as u32,
            qrows as u32,
        ];
        // Long KV axes use tiled arithmetic even for singleton queries.
        // Serial attention is only competitive for genuinely short axes;
        // selecting it solely by query count hurts fitted-context replay.
        let tiled = klen >= 32;
        if klen == 1 {
            c.dispatch(
                "kumo_single_value",
                &[attention_buffers[2], &s.att],
                &attention_params,
                [(qrows * d).div_ceil(256), 1, 1],
                256,
            );
        } else if tiled {
            let split = if !replay && query_kv_heads > 0 && query_kv_heads < heads {
                klen.min(qlen)
            } else {
                qlen
            };
            for (offset, count) in [(0, split), (split, qlen - split)] {
                if count == 0 {
                    continue;
                }
                attention_params[7] = offset as u32;
                attention_params[8] = count as u32;
                c.dispatch(
                    if hd == 32 {
                        "kumo_attention_tile32"
                    } else {
                        "kumo_attention_tile64"
                    },
                    &attention_buffers,
                    &attention_params,
                    [heads, count.div_ceil(32), batch],
                    128,
                );
            }
        } else {
            c.dispatch(
                "kumo_attention",
                &attention_buffers,
                &attention_params,
                [qrows * heads, 1, 1],
                32,
            );
        }
        self.mm(
            c,
            &format!("{p}.attn.out_lin"),
            &s.att,
            &s.tmp,
            d,
            d,
            qrows,
            0,
            false,
        );
        if fused {
            c.dispatch(
                "kumo_add_norm",
                &[
                    query,
                    &s.tmp,
                    self.w(&format!("{p}.mlp.0.weight")),
                    out,
                    &s.nq,
                ],
                &[d as u32],
                [qrows, 1, 1],
                256,
            );
        } else {
            Self::add(c, query, &s.tmp, out, qrows * d);
            self.norm(
                c,
                out,
                &s.nq,
                &format!("{p}.mlp.0.weight"),
                d,
                batch,
                qlen,
                qlen,
            );
        }
        self.mm(
            c,
            &format!("{p}.mlp.1"),
            &s.nq,
            &s.wide,
            d,
            2 * d,
            qrows,
            0,
            true,
        );
        if fused {
            c.dispatch(
                "kumo_mlp_out",
                &[
                    self.w(&format!("{p}.mlp.3.weight")),
                    &s.wide,
                    out,
                    self.w(&format!("{p}.mlp.3.bias")),
                ],
                &[d as u32, qrows as u32],
                [d.div_ceil(32), qrows.div_ceil(32), 1],
                128,
            );
        } else {
            self.mm(
                c,
                &format!("{p}.mlp.3"),
                &s.wide,
                &s.tmp,
                2 * d,
                d,
                qrows,
                0,
                false,
            );
            Self::add(c, out, &s.tmp, out, qrows * d);
        }
    }

    pub fn predict(&mut self, t: &Table<'_>) -> Result<KumoOutput> {
        t.validate(&self.config.task).map_err(error)?;
        self.execute(t, Mode::Plain)
    }

    pub fn context_bytes(&self, rows: usize, cols: usize) -> u64 {
        (8 * (self.config.embedding_layers * cols * self.config.inducing * self.config.cell
            + self.config.layers * rows * self.config.hidden * self.config.query_kv_heads
                / self.config.heads)) as u64
    }
    pub fn fit(&mut self, t: &Table<'_>) -> Result<(KumoContext, KumoOutput)> {
        let n =
            t.y.len()
                .checked_mul(t.categorical.len())
                .ok_or_else(|| error("context shape overflow"))?;
        let t = &Table {
            x: t.x
                .get(..n)
                .ok_or_else(|| error("context shape mismatch"))?,
            y: t.y,
            categorical: t.categorical,
            query_rows: 0,
        };
        t.validate_context(&self.config.task).map_err(error)?;
        let cols = t.categorical.len();
        let rows = t.y.len();
        let bytes = self.context_bytes(rows, cols);
        if self.device.allocated_bytes() + bytes > self.device.budget_bytes() {
            return Err(error("fitted context exceeds Metal memory budget"));
        }
        let kv = |n, heads| -> Result<Kv> {
            Ok(Kv {
                key: self.device.alloc(n * 4)?,
                value: self.device.alloc(n * 4)?,
                heads,
            })
        };
        let ctx = KumoContext {
            columns: (0..self.config.embedding_layers)
                .map(|_| kv(cols * self.config.inducing * self.config.cell, 4))
                .collect::<Result<Vec<_>>>()?,
            layers: (0..self.config.layers)
                .map(|_| {
                    kv(
                        rows * self.config.hidden * self.config.query_kv_heads / self.config.heads,
                        self.config.query_kv_heads,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            means: Self::means(t),
            categorical: t.categorical.to_vec(),
            rows,
            bytes,
        };
        let output = self.execute(t, Mode::Record(&ctx))?;
        Ok((ctx, output))
    }
    pub fn query(&mut self, ctx: &KumoContext, x: &[f32], rows: usize) -> Result<KumoOutput> {
        if !(1..=1024).contains(&rows)
            || x.len() != rows * ctx.categorical.len()
            || (rows + ctx.rows) * ctx.categorical.len() > 131_072
            || x.iter().any(|v| v.is_infinite())
        {
            return Err(error("invalid fitted-context query"));
        }
        self.execute(
            &Table {
                x,
                y: &[],
                categorical: &ctx.categorical,
                query_rows: rows,
            },
            Mode::Replay(ctx),
        )
    }
    fn means(t: &Table<'_>) -> Vec<f32> {
        let cols = t.categorical.len();
        (0..cols)
            .map(|j| {
                let (sum, count) = (0..t.y.len())
                    .map(|i| t.x[i * cols + j])
                    .filter(|v| !v.is_nan())
                    .fold((0f64, 0usize), |(s, n), v| (s + f64::from(v), n + 1));
                if count == 0 {
                    0.
                } else {
                    (sum / count as f64) as f32
                }
            })
            .collect()
    }
    fn execute(&self, t: &Table<'_>, mode: Mode<'_>) -> Result<KumoOutput> {
        let cfg = &self.config;
        let nc = t.y.len();
        let r = nc
            + if matches!(mode, Mode::Record(_)) {
                0
            } else {
                t.query_rows
            };
        let key_rows = if let Mode::Replay(ctx) = mode {
            ctx.rows
        } else {
            nc
        };
        let cols = t.categorical.len();
        let d = cfg.cell;
        let h = cfg.hidden;
        let inducing = cfg.inducing;
        let cap = (r * (cols + 4) * d)
            .max(if matches!(mode, Mode::Replay(_)) {
                0
            } else {
                cols * inducing * d
            })
            .max(r * h)
            .max(r * cfg.outputs());
        // Scratch (13 planes), working states (5), and cell preprocessing.
        // Check the entire request before its first allocation/submission.
        let elems = 18 * cap
            + r * 4 * d
            + cols * r * 192
            + cols * d * 192
            + r * cols
            + nc.max(1)
            + cols * 2;
        let workspace_bytes = (elems * 4) as u64;
        let retained = self.device.allocated_bytes();
        if retained + workspace_bytes > self.device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "Kumo request needs {workspace_bytes} workspace bytes in addition to {} weights",
                self.weight_bytes
            )));
        }
        let s = Scratch::new(&self.device, cap)?;
        let a = || self.device.alloc(cap * 4);
        let cells = a()?;
        let cellout = a()?;
        let ind = a()?;
        let indout = a()?;
        let rows = a()?;
        let cls = self.device.alloc(r * 4 * d * 4)?;
        let fourier = self.device.alloc(cols * r * 192 * 4)?;
        let cellw = self.device.alloc(cols * d * 192 * 4)?;
        let x = upload(&self.device, &t.x[..r * cols])?;
        let y = upload(&self.device, if t.y.is_empty() { &[0.] } else { t.y })?;
        let cat = self.device.upload(
            &t.categorical
                .iter()
                .flat_map(|&v| u32::from(v).to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        // Native statistics use only context rows, including all-missing
        // columns. F64 summation avoids overflow in otherwise finite inputs.
        let means = if let Mode::Replay(ctx) = mode {
            ctx.means.clone()
        } else {
            Self::means(t)
        };
        let means = upload(&self.device, &means)?;
        debug_assert_eq!(self.device.allocated_bytes(), retained + workspace_bytes);
        let c = self.device.begin()?;
        #[cfg(test)]
        let mut trace = Vec::new();
        macro_rules! capture {
            ($name:expr, $buffer:expr, $count:expr) => {
                #[cfg(test)]
                if std::env::var_os("PADDOCK_KUMO_TRACE").is_some() {
                    let count = $count;
                    let copy = self.device.alloc(count * 4)?;
                    Self::copy(&c, $buffer, &copy, count, count);
                    trace.push(($name.to_string(), copy, count));
                }
            };
        }
        let ce = "row_embedding.cell_embedding";
        c.dispatch(
            "kumo_fourier",
            &[
                &x,
                &means,
                &cat,
                self.w(&format!("{ce}.num_freq")),
                self.w(&format!("{ce}.cat_freq")),
                &fourier,
            ],
            &[r as u32, cols as u32],
            [(r * cols * 192).div_ceil(256), 1, 1],
            256,
        );
        c.dispatch(
            "kumo_cell_weights",
            &[
                &cat,
                self.w(&format!("{ce}.num_lin.weight")),
                self.w(&format!("{ce}.cat_lin.weight")),
                &cellw,
            ],
            &[cols as u32, d as u32],
            [(cols * d * 192).div_ceil(256), 1, 1],
            256,
        );
        c.dispatch(
            "kumo_cell_mm",
            &[&cellw, &fourier, &cells],
            &[r as u32, d as u32],
            [d.div_ceil(32), r.div_ceil(32), cols],
            128,
        );
        let classification = u32::from(cfg.task == Task::Classification);
        let target = if classification == 1 {
            "y_emb.weight"
        } else {
            "y_lin.weight"
        };
        c.dispatch(
            "kumo_cell_bias",
            &[
                &cells,
                &x,
                &cat,
                self.w(&format!("{ce}.num_lin.bias")),
                self.w(&format!("{ce}.cat_lin.bias")),
                self.w(&format!("{ce}.nan_lin.weight")),
                self.w(&format!("row_embedding.{target}")),
                &y,
            ],
            &[r as u32, cols as u32, d as u32, nc as u32, classification],
            [(cols * r * d).div_ceil(256), 1, 1],
            256,
        );
        Self::copy(
            &c,
            self.w("row_embedding.readout_token"),
            &cls,
            4 * d,
            r * 4 * d,
        );
        for i in 0..cfg.embedding_layers {
            let p = format!("row_embedding.col_blocks.{i}");
            if !matches!(mode, Mode::Replay(_)) {
                Self::copy(
                    &c,
                    self.w(&format!("{p}.inducing_points")),
                    &ind,
                    inducing * d,
                    cols * inducing * d,
                );
                self.block(
                    &c,
                    &s,
                    &format!("{p}.inducing_block"),
                    &ind,
                    &cells,
                    &indout,
                    cols,
                    inducing,
                    nc,
                    r,
                    d,
                    4,
                    1,
                    0,
                    None,
                );
            }
            self.block(
                &c,
                &s,
                &format!("{p}.output_block"),
                &cells,
                &indout,
                &cellout,
                cols,
                r,
                inducing,
                inducing,
                d,
                4,
                0,
                0,
                match mode {
                    Mode::Plain => None,
                    Mode::Record(ctx) => Some((&ctx.columns[i], false)),
                    Mode::Replay(ctx) => Some((&ctx.columns[i], true)),
                },
            );
            capture!(format!("column{i}"), &cellout, cols * r * d);
            c.dispatch(
                "kumo_rows",
                &[&cellout, &cls, &rows],
                &[r as u32, cols as u32, d as u32],
                [(r * (cols + 4) * d).div_ceil(256), 1, 1],
                256,
            );
            let readout_only = i + 1 == cfg.embedding_layers;
            self.block(
                &c,
                &s,
                &format!("row_embedding.row_blocks.{i}"),
                if readout_only { &cls } else { &rows },
                &rows,
                &cellout,
                r,
                if readout_only { 4 } else { cols + 4 },
                cols + 4,
                cols + 4,
                d,
                4,
                2,
                0,
                None,
            );
            capture!(
                format!("row{i}"),
                &cellout,
                r * if readout_only { 4 } else { cols + 4 } * d
            );
            if readout_only {
                Self::copy(&c, &cellout, &cls, r * 4 * d, r * 4 * d);
            } else {
                c.dispatch(
                    "kumo_unrows",
                    &[&cellout, &cells, &cls],
                    &[r as u32, cols as u32, d as u32],
                    [(r * (cols + 4) * d).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
        self.norm(
            &c,
            &cls,
            &s.nq,
            "row_embedding.norm.weight",
            d,
            1,
            r * 4,
            r * 4,
        );
        capture!("embedded", &s.nq, r * 4 * d);
        if d * 4 != h {
            self.mm(&c, "row_project", &s.nq, &rows, d * 4, h, r, 0, false);
        } else {
            Self::copy(&c, &s.nq, &rows, r * h, r * h);
        }
        capture!("projected", &rows, r * h);
        if nc > 0 {
            c.dispatch(
                "kumo_labels",
                &[&rows, &y, self.w(&format!("icl_block.{target}"))],
                &[h as u32, nc as u32, classification],
                [(nc * h).div_ceil(256), 1, 1],
                256,
            );
        }
        for i in 0..cfg.layers {
            self.block(
                &c,
                &s,
                &format!("icl_block.layers.{i}"),
                &rows,
                &rows,
                &cellout,
                1,
                r,
                key_rows,
                r,
                h,
                cfg.heads,
                1,
                cfg.query_kv_heads,
                match mode {
                    Mode::Plain => None,
                    Mode::Record(ctx) => Some((&ctx.layers[i], false)),
                    Mode::Replay(ctx) => Some((&ctx.layers[i], true)),
                },
            );
            capture!(format!("icl{i}"), &cellout, r * h);
            Self::copy(&c, &cellout, &rows, r * h, r * h);
        }
        self.norm(&c, &rows, &s.nq, "icl_block.norm.weight", h, 1, r, r);
        self.mm(&c, "icl_block.head.0", &s.nq, &s.wide, h, 2 * h, r, 0, true);
        self.mm(
            &c,
            "icl_block.head.2",
            &s.wide,
            &cellout,
            2 * h,
            cfg.outputs(),
            r,
            0,
            false,
        );
        capture!("output", &cellout, r * cfg.outputs());
        let gpu_seconds = c.finish()?;
        #[cfg(test)]
        if let Some(path) = std::env::var_os("PADDOCK_KUMO_TRACE") {
            let values: std::collections::BTreeMap<_, _> = trace
                .iter()
                .map(|(name, buffer, count)| (name, unsafe { buffer.read_f32(0, *count) }))
                .collect();
            std::fs::write(path, serde_json::to_vec(&values).unwrap()).unwrap();
        }
        // Only the completed final result crosses to the host.
        let values = if matches!(mode, Mode::Record(_)) {
            Vec::new()
        } else {
            unsafe { cellout.read_f32(nc * cfg.outputs(), t.query_rows * cfg.outputs()) }
        };
        if values.iter().any(|x| !x.is_finite()) {
            return Err(error("nonfinite model output"));
        }
        Ok(KumoOutput {
            values,
            gpu_seconds,
            workspace_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_projection_and_residual_norm_preserve_arithmetic() {
        let device = MetalDevice::new(Some(128 << 20)).unwrap();
        let values = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 193 + salt) % 8191) as f32 / 2013.7 - 2.)
                .collect()
        };
        for d in [128usize, 256, 512, 1024] {
            let w = upload(&device, &values(3 * d * d, 11)).unwrap();
            let bias = upload(&device, &values(3 * d, 19)).unwrap();
            let norm_w = upload(&device, &values(d, 47)).unwrap();
            for rows in [1usize, 17, 33] {
                let n = rows * d;
                let x = upload(&device, &values(n, 29)).unwrap();
                let y = upload(&device, &values(n, 37)).unwrap();
                // A nonzero guard detects writes past a partial matrix tile.
                let plane = || upload(&device, &vec![12345.; n + 16]).unwrap();
                let (k, v, fk, fv, residual, normalized, fr, fnorm) = (
                    plane(),
                    plane(),
                    plane(),
                    plane(),
                    plane(),
                    plane(),
                    plane(),
                    plane(),
                );
                let c = device.begin().unwrap();
                for (offset, output) in [(d, &k), (2 * d, &v)] {
                    c.dispatch(
                        "kumo_mm",
                        &[&w, &x, output, &bias],
                        &[d as u32, d as u32, rows as u32, offset as u32, 0],
                        [d.div_ceil(32), rows.div_ceil(32), 1],
                        128,
                    );
                }
                c.dispatch(
                    "kumo_kv",
                    &[&w, &x, &fk, &fv, &bias],
                    &[d as u32, rows as u32],
                    [(2 * d).div_ceil(32), rows.div_ceil(32), 1],
                    128,
                );
                Kumo::add(&c, &x, &y, &residual, n);
                c.dispatch(
                    "kumo_norm",
                    &[&residual, &norm_w, &normalized],
                    &[d as u32, rows as u32, rows as u32],
                    [rows, 1, 1],
                    256,
                );
                c.dispatch(
                    "kumo_add_norm",
                    &[&x, &y, &norm_w, &fr, &fnorm],
                    &[d as u32],
                    [rows, 1, 1],
                    256,
                );
                let mx = upload(&device, &values(2 * n, 71)).unwrap();
                let (projected, mr, fmr) = (plane(), plane(), plane());
                c.dispatch(
                    "kumo_mm",
                    &[&w, &mx, &projected, &bias],
                    &[(2 * d) as u32, d as u32, rows as u32, 0, 0],
                    [d.div_ceil(32), rows.div_ceil(32), 1],
                    128,
                );
                Kumo::add(&c, &residual, &projected, &mr, n);
                Kumo::copy(&c, &residual, &fmr, n, n);
                c.dispatch(
                    "kumo_mlp_out",
                    &[&w, &mx, &fmr, &bias],
                    &[d as u32, rows as u32],
                    [d.div_ceil(32), rows.div_ceil(32), 1],
                    128,
                );
                c.submit().unwrap().wait().unwrap();
                for (expected, actual) in [
                    (&k, &fk),
                    (&v, &fv),
                    (&residual, &fr),
                    (&normalized, &fnorm),
                    (&mr, &fmr),
                ] {
                    let a = unsafe { expected.read_f32(0, n + 16) };
                    let b = unsafe { actual.read_f32(0, n + 16) };
                    assert_eq!(a, b, "D={d} rows={rows}");
                    assert!(b[n..].iter().all(|&v| v == 12345.));
                }
            }
        }
    }
    #[test]
    #[ignore = "requires pinned SDM recipe fixture; writes optional per-stage diagnostics"]
    fn kumo_recipe_stage_trace() {
        let root = std::env::var("PADDOCK_KUMO_REFERENCE").unwrap();
        let root = Path::new(&root);
        let oracle = std::env::var("PADDOCK_KUMO_RECIPE_ORACLE")
            .unwrap_or_else(|_| "recipe-oracle.json".into());
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join(oracle)).unwrap()).unwrap();
        let m = &v["cases"][0]["members"][2];
        let x: Vec<_> = m["context"]
            .as_array()
            .unwrap()
            .iter()
            .chain(m["query"].as_array().unwrap())
            .map(|v| v.as_f64().map_or(f32::NAN, |v| v as f32))
            .collect();
        let y: Vec<_> = m["y"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let cat: Vec<_> = m["categorical"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let mut model = Kumo::load(root, Some(4 << 30)).unwrap();
        let output = model
            .predict(&Table {
                x: &x,
                y: &y,
                categorical: &cat,
                query_rows: 5,
            })
            .unwrap();
        let expected: Vec<_> = m["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        eprintln!(
            "KUMO_TRACE_MAX_ABS {}",
            output
                .values
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max)
        );
    }
    #[test]
    #[ignore = "requires pinned SDM full-recipe oracle and original weights"]
    fn kumo_recipe_reference_parity() {
        use paddock_models::kumo::recipe::{Cell, Fitted, RawTable};
        let root = std::env::var("PADDOCK_KUMO_REFERENCE").unwrap();
        let root = Path::new(&root);
        let oracle = std::env::var("PADDOCK_KUMO_RECIPE_ORACLE")
            .unwrap_or_else(|_| "recipe-oracle.json".into());
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join(oracle)).unwrap()).unwrap();
        let mut model = Kumo::load(root, Some(4 << 30)).unwrap();
        let mut failures = 0;
        let mut records = Vec::new();
        for case in v["cases"].as_array().unwrap() {
            let req = &case["request"];
            let raw = RawTable {
                context: serde_json::from_value(req["context"].clone()).unwrap(),
                targets: serde_json::from_value(req["targets"].clone()).unwrap(),
                categorical: serde_json::from_value(req["categorical"].clone()).unwrap(),
            };
            let query: Vec<Vec<Cell>> = serde_json::from_value(req["query"].clone()).unwrap();
            let f = Fitted::fit(
                &raw,
                &model.config.task,
                req["num_estimators"].as_u64().unwrap() as usize,
                req["seed"].as_u64().unwrap(),
            )
            .unwrap();
            let mut outputs = Vec::new();
            for (i, (m, expected)) in f
                .members
                .iter()
                .zip(case["members"].as_array().unwrap())
                .enumerate()
            {
                let q = m.transform(&query);
                let mut x = m.context.clone();
                x.extend_from_slice(&q);
                let input = Table {
                    x: &x,
                    y: &m.y,
                    categorical: &m.categorical,
                    query_rows: query.len(),
                };
                let out = model.predict(&input).unwrap();
                let (ctx, _) = model.fit(&input).unwrap();
                let cached = model.query(&ctx, &q, query.len()).unwrap();
                let oracle_x = expected["context"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .chain(expected["query"].as_array().unwrap())
                    .map(|v| v.as_f64().map_or(f32::NAN, |v| v as f32))
                    .collect::<Vec<_>>();
                let oracle_y = expected["y"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>();
                let oracle_input = Table {
                    x: &oracle_x,
                    y: &oracle_y,
                    categorical: &m.categorical,
                    query_rows: query.len(),
                };
                let oracle_native = model.predict(&oracle_input).unwrap();
                let expected = expected["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>();
                let mut max = 0f32;
                let mut outside = 0;
                let max_prepared = oracle_native
                    .values
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "KUMO_RECIPE_DIAG member={i} neural_with_upstream_inputs_max_abs={max_prepared} x_max_abs={} y_max_abs={}",
                    x.iter()
                        .zip(&oracle_x)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max),
                    m.y.iter()
                        .zip(&oracle_y)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max)
                );
                for values in [&out.values, &cached.values] {
                    for (j, (a, b)) in values.iter().zip(&expected).enumerate() {
                        max = max.max((a - b).abs());
                        if (a - b).abs() > 0.0001 + 0.00001 * b.abs() {
                            if outside == 0 {
                                eprintln!("KUMO_RECIPE_MISS member={i} output={j}: {a} != {b}");
                            }
                            outside += 1;
                        }
                    }
                }
                eprintln!(
                    "KUMO_RECIPE size={} task={:?} context={} member={i} max_abs={max} outside={outside}",
                    model.config.size, model.config.task, f.context_rows
                );
                failures += outside;
                assert_eq!(out.values, cached.values, "direct/cache member {i}");
                records.push(serde_json::json!({"context_rows":f.context_rows,"member":i,
                    "native":out.values,"upstream_inputs":oracle_native.values,"reference":expected}));
                outputs.push(out.values);
            }
            let reduced = f.reduce(&outputs, &model.config.task, query.len()).unwrap();
            if model.config.task == Task::Regression {
                let mut max = 0f64;
                for (a, b) in reduced.iter().zip(case["predictions"].as_array().unwrap()) {
                    let b = b.as_f64().unwrap();
                    let diff = (f64::from(*a) - b).abs() / f.target_scale;
                    max = max.max(diff);
                    if diff > 0.0001 + 0.00001 * ((b - f.target_mean) / f.target_scale).abs() {
                        failures += 1;
                    }
                }
                eprintln!(
                    "KUMO_RECIPE_REDUCED scale={} mean={} standardized_max_abs={max}",
                    f.target_scale, f.target_mean
                );
            }
            assert_eq!(model.device.allocated_bytes(), model.weight_bytes);
        }
        if let Ok(path) = std::env::var("PADDOCK_KUMO_MEMBER_OUTPUTS") {
            std::fs::write(path, serde_json::to_vec(&records).unwrap()).unwrap();
        }
        assert_eq!(failures, 0, "full recipe/reference numerical misses");
    }
    #[test]
    #[ignore = "requires original Kumo weights and pinned CPU oracle"]
    fn kumo_reference_parity() {
        let root = std::env::var("PADDOCK_KUMO_REFERENCE").expect("PADDOCK_KUMO_REFERENCE");
        let root = Path::new(&root);
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("oracle.json")).unwrap()).unwrap();
        let mut model = Kumo::load(root, Some(4 << 30)).unwrap();
        for case in v["cases"].as_array().unwrap() {
            let x = case["x"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().map_or(f32::NAN, |n| n as f32))
                .collect::<Vec<_>>();
            let y = case["y"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            let cat = case["categorical"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect::<Vec<_>>();
            let query = case["query_rows"].as_u64().unwrap() as usize;
            let input = Table {
                x: &x,
                y: &y,
                categorical: &cat,
                query_rows: query,
            };
            let out = model.predict(&input).unwrap();
            let expected = case["output"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            assert_eq!(out.values.len(), expected.len());
            let max = out
                .values
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "KUMO_PARITY size={} task={:?} context={} query={query} cols={} max_abs={max} gpu_seconds={} workspace_bytes={}",
                model.config.size,
                model.config.task,
                y.len(),
                cat.len(),
                out.gpu_seconds,
                out.workspace_bytes
            );
            for (i, (&a, &b)) in out.values.iter().zip(&expected).enumerate() {
                assert!(
                    (a - b).abs() <= 0.0001 + 0.00001 * b.abs(),
                    "output {i}: {a} != {b}"
                );
            }
            let single = Table {
                x: &x[..(y.len() + 1) * cat.len()],
                y: &y,
                categorical: &cat,
                query_rows: 1,
            };
            let one = model.predict(&single).unwrap();
            for (&a, &b) in one.values.iter().zip(&out.values) {
                assert!(
                    (a - b).abs() < 0.0001 + 0.00001 * b.abs(),
                    "query isolation {a} != {b}"
                );
            }
            let again = model.predict(&input).unwrap();
            assert_eq!(
                out.values, again.values,
                "repeat prediction after a different shape"
            );
            assert_eq!(
                model.device.allocated_bytes(),
                model.weight_bytes,
                "request scratch must be reclaimed"
            );
            let (ctx, fit) = model.fit(&input).unwrap();
            assert_eq!(
                model.device.allocated_bytes(),
                model.weight_bytes + ctx.bytes()
            );
            let cached = model.query(&ctx, &x[y.len() * cat.len()..], query).unwrap();
            let cached_one = model
                .query(&ctx, &x[y.len() * cat.len()..(y.len() + 1) * cat.len()], 1)
                .unwrap();
            for (a, b) in cached
                .values
                .iter()
                .zip(&expected)
                .chain(cached_one.values.iter().zip(&expected))
            {
                assert!(
                    (a - b).abs() <= 0.0001 + 0.00001 * b.abs(),
                    "cached prediction {a} != {b}"
                );
            }
            let again = model.query(&ctx, &x[y.len() * cat.len()..], query).unwrap();
            assert_eq!(
                cached.values, again.values,
                "cached repeat must not mutate context"
            );
            eprintln!(
                "KUMO_CACHE fit_ms={} query_ms={} direct_ms={} bytes={} query_workspace={}",
                fit.gpu_seconds * 1000.,
                cached.gpu_seconds * 1000.,
                out.gpu_seconds * 1000.,
                ctx.bytes(),
                cached.workspace_bytes
            );
            if std::env::var_os("PADDOCK_KUMO_BENCH").is_some() {
                let mut direct = Vec::new();
                let mut replay = Vec::new();
                for _ in 0..7 {
                    let start = std::time::Instant::now();
                    model.predict(&input).unwrap();
                    direct.push(start.elapsed().as_secs_f64() * 1000.);
                    let start = std::time::Instant::now();
                    model.query(&ctx, &x[y.len() * cat.len()..], query).unwrap();
                    replay.push(start.elapsed().as_secs_f64() * 1000.);
                }
                direct.drain(..2);
                replay.drain(..2);
                eprintln!(
                    "KUMO_BENCH {}",
                    serde_json::json!({"context":y.len(),"query":query,"columns":cat.len(),"direct_ms":direct,"cached_ms":replay})
                );
            }
            if std::env::var_os("PADDOCK_KUMO_FUSION_BENCH").is_some() {
                let mut timings = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for iteration in 0..9 {
                    // Alternate order within each pair to avoid favoring the
                    // second implementation through temperature/cache drift.
                    for unfused in [iteration % 2 == 0, iteration % 2 != 0] {
                        model.unfused = unfused;
                        let start = std::time::Instant::now();
                        let direct = model.predict(&input).unwrap();
                        let direct_ms = start.elapsed().as_secs_f64() * 1000.;
                        let start = std::time::Instant::now();
                        let replay = model.query(&ctx, &x[y.len() * cat.len()..], query).unwrap();
                        let cached_ms = start.elapsed().as_secs_f64() * 1000.;
                        assert_eq!(
                            direct.values, out.values,
                            "fusion must preserve direct arithmetic"
                        );
                        assert_eq!(
                            replay.values, cached.values,
                            "fusion must preserve replay arithmetic"
                        );
                        assert_eq!(direct.workspace_bytes, out.workspace_bytes);
                        if iteration >= 2 {
                            timings[usize::from(unfused) * 2].push(direct_ms);
                            timings[usize::from(unfused) * 2 + 1].push(cached_ms);
                        }
                    }
                }
                model.unfused = false;
                eprintln!(
                    "KUMO_FUSION_BENCH {}",
                    serde_json::json!({
                    "context":y.len(),"query":query,"columns":cat.len(),
                    "fused_direct_ms":timings[0],"fused_cached_ms":timings[1],
                    "unfused_direct_ms":timings[2],"unfused_cached_ms":timings[3]})
                );
            }
            drop(ctx);
            assert_eq!(
                model.device.allocated_bytes(),
                model.weight_bytes,
                "release must reclaim all context and scratch"
            );
        }
        let weight_bytes = model.weight_bytes;
        drop(model);
        let mut limited = Kumo::load(root, Some(weight_bytes + 4096)).unwrap();
        let t = Table {
            x: &[0., 1.],
            y: &[0.],
            categorical: &[false],
            query_rows: 1,
        };
        assert!(matches!(limited.predict(&t), Err(MetalError::Memory(_))));
        assert_eq!(
            limited.device.allocated_bytes(),
            weight_bytes,
            "over-budget requests allocate no scratch"
        );
    }
}
