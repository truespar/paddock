//! One pass: a prepared table in, raw outputs out (ten logits or 999
//! quantiles a query row; calibration and inverse transforms belong to the
//! recipe). The op train is kumo.cuh's header; this file is the launch order
//! and the pass's planes, sized and admitted before the first allocation.

use std::time::Instant;

use cudarc::driver::CudaSlice;
use paddock_models::kumo::{Table, Task};

use super::block::{Cache, Geo, NO_STRIDES, RefPlanes, Rows, Scratch};
use super::{GpuKumo, GpuModelError, KumoContext};
use crate::gpu::{KumoEpi, KumoFuse};

/// A pass's raw outputs: every member's query rows, in member order (none
/// for a fit), and what the pass cost.
pub(super) struct PassOut {
    pub values: Vec<Vec<f32>>,
    pub gpu_seconds: f64,
    pub workspace_bytes: u64,
}

/// What a pass does with the fitted-context state.
pub(super) enum Mode<'a> {
    /// a direct prediction: context and query rows together
    Plain,
    /// a fit: context rows only, keys/values kept in the context
    Record(&'a mut KumoContext),
    /// query rows only, keys/values read from the context
    Replay(&'a KumoContext),
}

/// The per-row RMSNorm statistics the fused blocks read, one plane per
/// producer: the cells (the embedding, then each unpack), the inducing
/// outputs, the packed rows, the inducing points and the ICL stream.
struct InvPlanes {
    cells: CudaSlice<f32>,
    ind: CudaSlice<f32>,
    rows: CudaSlice<f32>,
    pts: CudaSlice<f32>,
    icl: CudaSlice<f32>,
}

/// The pass planes, kept between requests and grown to the largest table
/// seen - a caching arena. Handing hundreds of MB back to the driver after
/// every pass and faulting fresh pages in on the next cost more than the
/// pass itself on GB10 (a warm 33-row replay measured 10.4 ms alone and
/// 19.5 ms interleaved with direct passes). The tabular service drops it
/// after a quiet spell ([`GpuKumo::release_workspace`]).
pub(super) struct Workspace {
    /// capacities in floats: each block plane, readout tokens, Fourier
    /// features, per-column cell weights, each statistics plane (rows), the
    /// rope tables
    dims: [usize; 6],
    s: Scratch,
    inv: InvPlanes,
    cells: CudaSlice<f32>,
    cellout: CudaSlice<f32>,
    indout: CudaSlice<f32>,
    rows: CudaSlice<f32>,
    cls: CudaSlice<f32>,
    fourier: CudaSlice<f32>,
    cellw: CudaSlice<f32>,
}

impl Workspace {
    /// 7 block planes (qh, kh, v, att, tmp, wide x 2) + cells, cellout,
    /// indout, rows; the reference path's 6 more (nq, nk, q, k, gate x 2);
    /// the embedding planes; six statistics planes; the rope tables.
    fn bytes_for(dims: [usize; 6], reference: bool) -> u64 {
        let planes = 11 + if reference { 6 } else { 0 };
        ((planes * dims[0] + dims[1] + dims[2] + dims[3] + 6 * dims[4] + dims[5]) * 4) as u64
    }

    pub(super) fn bytes(&self) -> u64 {
        Self::bytes_for(self.dims, self.s.r.is_some())
    }

    fn new(
        e: &crate::gpu::GpuExecutor,
        dims: [usize; 6],
        reference: bool,
    ) -> Result<Self, GpuModelError> {
        let plane = |n: usize| e.kumo_plane(n);
        let cap = dims[0];
        let r = if reference {
            Some(RefPlanes {
                nq: plane(cap)?,
                nk: plane(cap)?,
                q: plane(cap)?,
                k: plane(cap)?,
                gate: plane(2 * cap)?,
            })
        } else {
            None
        };
        Ok(Self {
            dims,
            s: Scratch {
                qh: plane(cap)?,
                kh: plane(cap)?,
                v: plane(cap)?,
                att: plane(cap)?,
                tmp: plane(cap)?,
                wide: plane(2 * cap)?,
                inv: plane(dims[4])?,
                rope: plane(dims[5])?,
                rope_seq: 0,
                r,
            },
            inv: InvPlanes {
                cells: plane(dims[4])?,
                ind: plane(dims[4])?,
                rows: plane(dims[4])?,
                pts: plane(dims[4])?,
                icl: plane(dims[4])?,
            },
            cells: plane(cap)?,
            cellout: plane(cap)?,
            indout: plane(cap)?,
            rows: plane(cap)?,
            cls: plane(dims[1])?,
            fourier: plane(dims[2])?,
            cellw: plane(dims[3])?,
        })
    }
}

/// Column means over the context rows only (missing cells skipped, F64
/// sums so finite inputs cannot overflow); an all-missing column imputes 0.
pub(super) fn means(t: &Table<'_>) -> Vec<f32> {
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

impl GpuKumo {
    /// Run one pass on the resident workspace, growing it first when this
    /// request needs more than it holds. The request's planes are admitted
    /// whole against the budget before anything is allocated.
    ///
    /// `tables` are an ensemble's members - same rows, same columns - run as
    /// one pass (one table is the plain case): their rows stack member-major,
    /// so the column stage sees a (column, member) column per pair and the
    /// row stage and every projection see all members' rows, and the ICL
    /// attends within each member (a batch of them). Every kernel treats a
    /// row, a column or a batch on its own, so each member's outputs are the
    /// bits its own pass would give.
    pub(super) fn execute(
        &mut self,
        tables: &[Table<'_>],
        mode: Mode<'_>,
    ) -> Result<PassOut, GpuModelError> {
        let cfg = &self.config;
        let e = self.exec.clone();
        let reference = self.reference;
        let record = matches!(mode, Mode::Record(_));
        let replay = matches!(mode, Mode::Replay(_));
        let t = tables
            .first()
            .ok_or_else(|| GpuModelError::Unsupported("Kumo-Tabular: no table".into()))?;
        let members = tables.len();
        let cols = t.categorical.len();
        if tables.iter().any(|m| {
            m.categorical.len() != cols || m.y.len() != t.y.len() || m.query_rows != t.query_rows
        }) || (members > 1 && !e.has_kumo_members())
            || match &mode {
                Mode::Plain => false,
                Mode::Record(ctx) => ctx.members != members,
                Mode::Replay(ctx) => ctx.members != members,
            }
        {
            return Err(GpuModelError::Unsupported(
                "Kumo-Tabular: ensemble members differ in shape or from their context, or \
                 this pack predates member passes"
                    .into(),
            ));
        }
        // per member, then every member's rows and (column, member) columns
        let r = t.y.len() + if record { 0 } else { t.query_rows };
        let (rr, vc) = (members * r, members * cols);
        let (d, h, inducing, outs) = (cfg.cell, cfg.hidden, cfg.inducing, cfg.outputs());
        let cap = (rr * (cols + 4) * d)
            .max(if replay { 0 } else { vc * inducing * d })
            .max(rr * h)
            .max(rr * outs);
        let need = [
            cap,
            rr * 4 * d,
            cols * rr * 192,
            vc * d * 192,
            (rr * (cols + 4)).max(vc * inducing).max(inducing),
            // a table per distinct rope vector, [cols + 4][hd / 2] pairs
            if reference {
                0
            } else {
                self.rope_src.len() * (cols + 4) * (d / 4)
            },
        ];
        let workspace_bytes = Workspace::bytes_for(need, reference);
        let fits = |w: &Workspace| {
            w.dims.iter().zip(need).all(|(&have, n)| have >= n) && (!reference || w.s.r.is_some())
        };
        let ws = match self.ws.take() {
            Some(w) if fits(&w) => w,
            old => {
                let mut dims = need;
                if let Some(w) = &old {
                    for (d, &have) in dims.iter_mut().zip(&w.dims) {
                        *d = (*d).max(have);
                    }
                }
                drop(old);
                // the frees are stream-ordered: let them land before asking
                // what the budget has left
                e.synchronize()?;
                let grow = Workspace::bytes_for(dims, reference);
                if e.vram_headroom().is_some_and(|room| grow > room) {
                    e.trim_mem_pool();
                    return Err(GpuModelError::WontFit(format!(
                        "Kumo request needs {workspace_bytes} workspace bytes (a {grow}-byte \
                         arena) beyond the {} of weights and the fitted contexts the budget \
                         already holds",
                        self.weight_bytes
                    )));
                }
                Workspace::new(&e, dims, reference)?
            }
        };
        let mut ws = ws;
        let out = self.pass(&mut ws, tables, mode, workspace_bytes);
        self.ws = Some(ws);
        out
    }

    fn pass(
        &self,
        ws: &mut Workspace,
        tables: &[Table<'_>],
        mut mode: Mode<'_>,
        workspace_bytes: u64,
    ) -> Result<PassOut, GpuModelError> {
        let cfg = &self.config;
        let e = &*self.exec;
        let fused = !self.reference;
        let record = matches!(mode, Mode::Record(_));
        let replay = matches!(mode, Mode::Replay(_));
        let t = &tables[0];
        let members = tables.len();
        let nc = t.y.len();
        let r = nc + if record { 0 } else { t.query_rows };
        let key_rows = match &mode {
            Mode::Replay(ctx) => ctx.rows,
            _ => nc,
        };
        let cols = t.categorical.len();
        // every member's rows, and the column stage's (column, member) columns
        let (rr, vc) = (members * r, members * cols);
        let (d, h, inducing, outs) = (cfg.cell, cfg.hidden, cfg.inducing, cfg.outputs());
        // disjoint borrows of the arena's planes; `rows`/`cellout` swap
        // roles every ICL layer, so they are rebindable
        let (s, inv, cells, mut cellout, indout) = (
            &mut ws.s,
            &mut ws.inv,
            &mut ws.cells,
            &mut ws.cellout,
            &mut ws.indout,
        );
        let (mut rows, cls, fourier, cellw) =
            (&mut ws.rows, &mut ws.cls, &mut ws.fourier, &mut ws.cellw);
        // the members' rows, labels, categorical flags and means, in order
        let x = e.to_device(
            &tables
                .iter()
                .flat_map(|m| m.x[..r * cols].iter().copied())
                .collect::<Vec<_>>(),
        )?;
        let ys = tables
            .iter()
            .flat_map(|m| m.y.iter().copied())
            .collect::<Vec<_>>();
        let y = e.to_device(if ys.is_empty() { &[0.] } else { &ys })?;
        let cat = e.to_device_u32(
            &tables
                .iter()
                .flat_map(|m| m.categorical.iter().map(|&c| u32::from(c)))
                .collect::<Vec<_>>(),
        )?;
        let means = match &mode {
            Mode::Replay(ctx) => ctx.means.clone(),
            _ => tables.iter().flat_map(means).collect(),
        };
        let means = e.to_device(&means)?;
        let start = Instant::now();
        let ce = "row_embedding.cell_embedding";
        let w = |n: &str| self.w(n);
        e.kumo_fourier(
            &x,
            &means,
            &cat,
            w(&format!("{ce}.num_freq")),
            w(&format!("{ce}.cat_freq")),
            fourier,
            (rr, cols, members),
        )?;
        e.kumo_cell_weights(
            &cat,
            w(&format!("{ce}.num_lin.weight")),
            w(&format!("{ce}.cat_lin.weight")),
            cellw,
            (cols, d, members),
        )?;
        // one problem a (column, member): fourier [cols][members][r][192]
        e.kumo_gemm(
            (fourier, 0),
            (cellw, 0),
            None,
            cells,
            None,
            (192, d, r),
            KumoEpi::Store,
            vc,
            (r * 192, d * 192, r * d),
        )?;
        let classification = cfg.task == Task::Classification;
        let target = if classification {
            "y_emb.weight"
        } else {
            "y_lin.weight"
        };
        e.kumo_cell_bias(
            cells,
            &x,
            &cat,
            (
                w(&format!("{ce}.num_lin.bias")),
                w(&format!("{ce}.cat_lin.bias")),
                w(&format!("{ce}.nan_lin.weight")),
            ),
            (w(&format!("row_embedding.{target}")), &y),
            (rr, cols, d, nc, members),
            classification,
        )?;
        if fused {
            e.kumo_stats(cells, None, &mut inv.cells, d, cols * rr)?;
        }
        e.kumo_copy(
            w("row_embedding.readout_token"),
            cls,
            rr,
            4 * d,
            4 * d,
            4 * d,
            1,
        )?;
        if fused && s.rope_seq != cols + 4 {
            // the row blocks' rope pairs for rows of cols + 4 tokens (the
            // four readout tokens lead): one table per distinct vector, kept
            // until the row length changes
            let (seq, hd) = (cols + 4, d / 4);
            for (t, name) in self.rope_src.iter().enumerate() {
                e.kumo_rope_table(w(name), (&mut s.rope, t * seq * hd), hd / 2, seq)?;
            }
            s.rope_seq = seq;
        }
        for i in 0..cfg.embedding_layers {
            let p = format!("row_embedding.col_blocks.{i}");
            if !replay {
                // the inducing points attend to every column's context cells;
                // their query side is the same for every column, computed once
                let pts = w(&format!("{p}.inducing_points"));
                let g = Geo {
                    batch: vc,
                    qlen: inducing,
                    klen: nc,
                    kvstride: r,
                    d,
                    heads: 4,
                    scaling: 1,
                    qkvh: 0,
                    shared: true,
                };
                let name = format!("{p}.inducing_block");
                if fused {
                    e.kumo_stats(pts, None, &mut inv.pts, d, inducing)?;
                    self.block_fused(
                        s,
                        &name,
                        Rows {
                            x: pts,
                            inv: &inv.pts,
                            gather: None,
                        },
                        Rows {
                            x: cells,
                            inv: &inv.cells,
                            gather: Some((nc, r)),
                        },
                        pts,
                        indout,
                        g,
                        None,
                    )?;
                    e.kumo_stats(indout, None, &mut inv.ind, d, vc * inducing)?;
                } else {
                    self.block_reference(s, &name, pts, cells, indout, g, None)?;
                }
            }
            let cache = match &mut mode {
                Mode::Plain => None,
                Mode::Record(ctx) => Some(Cache::Write(&mut ctx.columns[i])),
                Mode::Replay(ctx) => Some(Cache::Read(&ctx.columns[i])),
            };
            let g = Geo {
                batch: vc,
                qlen: r,
                klen: inducing,
                kvstride: inducing,
                d,
                heads: 4,
                scaling: 0,
                qkvh: 0,
                shared: false,
            };
            let name = format!("{p}.output_block");
            if fused {
                self.block_fused(
                    s,
                    &name,
                    Rows {
                        x: cells,
                        inv: &inv.cells,
                        gather: None,
                    },
                    Rows {
                        x: indout,
                        inv: &inv.ind,
                        gather: None,
                    },
                    cells,
                    cellout,
                    g,
                    cache,
                )?;
                e.kumo_rows_pack_stats(cellout, cls, rows, &mut inv.rows, (rr, cols, d))?;
            } else {
                self.block_reference(s, &name, cells, indout, cellout, g, cache)?;
                e.kumo_rows_pack(cellout, cls, rows, (rr, cols, d))?;
            }
            // the last row block computes only the readout tokens, as
            // upstream does: nothing reads the feature tokens after it
            let readout_only = i + 1 == cfg.embedding_layers;
            let g = Geo {
                batch: rr,
                qlen: if readout_only { 4 } else { cols + 4 },
                klen: cols + 4,
                kvstride: cols + 4,
                d,
                heads: 4,
                scaling: 2,
                qkvh: 0,
                shared: false,
            };
            let name = format!("row_embedding.row_blocks.{i}");
            if fused {
                // the readout tokens are the first four of every packed row
                let q = Rows {
                    x: rows,
                    inv: &inv.rows,
                    gather: readout_only.then_some((4, cols + 4)),
                };
                let kv = Rows {
                    x: rows,
                    inv: &inv.rows,
                    gather: None,
                };
                self.block_fused(
                    s,
                    &name,
                    q,
                    kv,
                    if readout_only { cls } else { rows },
                    cellout,
                    g,
                    None,
                )?;
            } else {
                self.block_reference(
                    s,
                    &name,
                    if readout_only { cls } else { rows },
                    rows,
                    cellout,
                    g,
                    None,
                )?;
            }
            if readout_only {
                e.copy_slice(cellout, 0, rr * 4 * d, cls)?;
            } else if fused {
                e.kumo_rows_unpack_stats(cellout, cells, cls, &mut inv.cells, (rr, cols, d))?;
            } else {
                e.kumo_rows_unpack(cellout, cells, cls, (rr, cols, d))?;
            }
        }
        let norm = w("row_embedding.norm.weight");
        if fused && d * 4 == h {
            // the four normed readout tokens ARE the ICL row
            e.kumo_norm(cls, (norm, 0), rows, d, rr * 4, rr * 4, rr * 4)?;
        } else {
            // the reference writes the normed tokens to their own plane; the
            // fused path borrows a block plane free here
            let normed = match s.r.as_mut() {
                Some(rp) if !fused => &mut rp.nq,
                _ => &mut s.tmp,
            };
            e.kumo_norm(cls, (norm, 0), normed, d, rr * 4, rr * 4, rr * 4)?;
            if d * 4 != h {
                e.kumo_gemm(
                    (normed, 0),
                    (w("row_project.weight"), 0),
                    Some((w("row_project.bias"), 0)),
                    rows,
                    None,
                    (d * 4, h, rr),
                    KumoEpi::Store,
                    1,
                    NO_STRIDES,
                )?;
            } else {
                e.copy_slice(normed, 0, rr * h, rows)?;
            }
        }
        if nc > 0 {
            e.kumo_labels(
                rows,
                &y,
                w(&format!("icl_block.{target}")),
                (h, nc, r, members),
                classification,
            )?;
        }
        if fused {
            e.kumo_stats(rows, None, &mut inv.icl, h, rr)?;
        }
        for i in 0..cfg.layers {
            let cache = match &mut mode {
                Mode::Plain => None,
                Mode::Record(ctx) => Some(Cache::Write(&mut ctx.layers[i])),
                Mode::Replay(ctx) => Some(Cache::Read(&ctx.layers[i])),
            };
            // one attention batch a member: its rows query its context rows
            let g = Geo {
                batch: members,
                qlen: r,
                klen: key_rows,
                kvstride: r,
                d: h,
                heads: cfg.heads,
                scaling: 1,
                qkvh: cfg.query_kv_heads,
                shared: false,
            };
            let name = format!("icl_block.layers.{i}");
            if fused {
                let own = Rows {
                    x: rows,
                    inv: &inv.icl,
                    gather: None,
                };
                // the keys: each member's first key_rows rows
                let keys = Rows {
                    gather: (members > 1).then_some((key_rows, r)),
                    ..own
                };
                self.block_fused(s, &name, own, keys, rows, cellout, g, cache)?;
                e.kumo_stats(cellout, None, &mut inv.icl, h, rr)?;
            } else {
                self.block_reference(s, &name, rows, rows, cellout, g, cache)?;
            }
            std::mem::swap(&mut rows, &mut cellout);
        }
        let (head0_w, head0_b) = (w("icl_block.head.0.weight"), w("icl_block.head.0.bias"));
        if fused {
            // the final norm rides the head's operand staging
            e.kumo_gemm_fused(
                rows,
                (head0_w, 0),
                Some((head0_b, 0)),
                &mut s.wide,
                None,
                (h, 2 * h, rr),
                KumoEpi::Gelu,
                KumoFuse {
                    norm: Some((&inv.icl, w("icl_block.norm.weight"))),
                    ..KumoFuse::default()
                },
            )?;
        } else {
            let rp = s.r.as_mut().ok_or_else(|| {
                GpuModelError::Unsupported("Kumo-Tabular: reference planes not allocated".into())
            })?;
            e.kumo_norm(
                rows,
                (w("icl_block.norm.weight"), 0),
                &mut rp.nq,
                h,
                rr,
                rr,
                rr,
            )?;
            e.kumo_gemm(
                (&rp.nq, 0),
                (head0_w, 0),
                Some((head0_b, 0)),
                &mut s.wide,
                None,
                (h, 2 * h, rr),
                KumoEpi::Gelu,
                1,
                NO_STRIDES,
            )?;
        }
        e.kumo_gemm(
            (&s.wide, 0),
            (w("icl_block.head.2.weight"), 0),
            Some((w("icl_block.head.2.bias"), 0)),
            cellout,
            None,
            (2 * h, outs, rr),
            KumoEpi::Store,
            1,
            NO_STRIDES,
        )?;
        e.synchronize()?;
        let gpu_seconds = start.elapsed().as_secs_f64();
        // Only the final result crosses to the host: each member's query
        // rows, gathered next to each other first when there are several
        // (their context rows sit between them).
        let q = t.query_rows;
        let span = q * outs;
        let bad = |what: &str| GpuModelError::Unsupported(format!("Kumo-Tabular: {what}"));
        let host = if record || q == 0 {
            Vec::new()
        } else if members == 1 {
            let view = cellout
                .try_slice(nc * outs..(nc + q) * outs)
                .ok_or_else(|| bad("output range"))?;
            e.stream
                .clone_dtoh(&view)
                .map_err(|err| bad(&format!("readback: {err}")))?
        } else {
            for m in 0..members {
                let from = cellout
                    .try_slice((m * r + nc) * outs..(m * r + nc) * outs + span)
                    .ok_or_else(|| bad("output range"))?;
                let mut to = s
                    .tmp
                    .try_slice_mut(m * span..(m + 1) * span)
                    .ok_or_else(|| bad("output gather"))?;
                e.stream
                    .memcpy_dtod(&from, &mut to)
                    .map_err(|err| bad(&format!("output gather: {err}")))?;
            }
            let view = s
                .tmp
                .try_slice(0..members * span)
                .ok_or_else(|| bad("output range"))?;
            e.stream
                .clone_dtoh(&view)
                .map_err(|err| bad(&format!("readback: {err}")))?
        };
        if host.iter().any(|v| !v.is_finite()) {
            return Err(bad("nonfinite model output"));
        }
        let values = if host.is_empty() {
            vec![Vec::new(); members]
        } else {
            host.chunks(span).map(<[f32]>::to_vec).collect()
        };
        Ok(PassOut {
            values,
            gpu_seconds,
            workspace_bytes,
        })
    }
}
