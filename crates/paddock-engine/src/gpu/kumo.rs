//! Kumo-Tabular ops - the tabular foundation model's prepared-table graph at
//! F32. Kernel side: `packs/cuda/src/kumo.cuh`, slots 698-708; the graph that
//! strings them together is `gpu_model::kumo`.
//!
//! Every plane is a flat f32 buffer; an operand is `(buffer, element
//! offset)`. Buffers are checked against the geometry before any launch - a
//! short slice is a logic error here, not something to hand the driver.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::*;
use super::*;

/// What a Kumo GEMM does with `acc + bias` (slot 698).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KumoEpi {
    Store = 0,
    Gelu = 1,
    /// accumulate into y - the MLP-down residual
    Resid = 2,
    /// split the n axis at N/2: the first half to y, the second to y2
    Split = 3,
}

/// A read operand: a buffer and the element its plane starts at.
pub type KumoIn<'a> = (&'a CudaSlice<f32>, usize);

/// What a fused projection (slot 710) carries on either side of its product;
/// each part replays the pass it removes, so the output is the unfused
/// sequence's to the bit.
#[derive(Clone, Copy, Default)]
pub struct KumoFuse<'a> {
    /// `(inv, weight)`: the input rows are the RMSNorm `(x * inv[row]) *
    /// weight[k]` (`inv` from [`GpuExecutor::kumo_stats`] or the row-axis
    /// copy with statistics)
    pub norm: Option<(&'a CudaSlice<f32>, &'a CudaSlice<f32>)>,
    /// `(len, stride)`: input row m is x row `m / len * stride + m % len`
    pub gather: Option<(usize, usize)>,
    /// the per-head transform on the output's head columns
    pub heads: Option<KumoHeads<'a>>,
}

/// The per-head output transform of a fused projection: rope (when `rope`),
/// the weightless RMSNorm, then the log query scaling (when `scale`).
#[derive(Clone, Copy)]
pub struct KumoHeads<'a> {
    pub hd: usize,
    /// rope position = output row % seq
    pub seq: usize,
    /// the rope's (cos, sin) pairs from [`GpuExecutor::kumo_rope_table`]:
    /// `[seq][hd / 2]` at a float offset
    pub rope: Option<KumoIn<'a>>,
    /// `(head_scale, klen)`
    pub scale: Option<(&'a CudaSlice<f32>, usize)>,
}

fn span(off: usize, batch: usize, stride: usize, one: usize) -> usize {
    off + (batch - 1) * stride + one
}

impl GpuExecutor {
    /// True when the loaded pack carries the whole Kumo lane (the slots
    /// landed together, so one missing means a pack older than it).
    pub fn has_kumo(&self) -> bool {
        let k = &self.kernels;
        k.kumo_gemm.is_some()
            && k.kumo_norm.is_some()
            && k.kumo_heads.is_some()
            && k.kumo_scale.is_some()
            && k.kumo_attention.is_some()
            && k.kumo_fourier.is_some()
            && k.kumo_cell_weights.is_some()
            && k.kumo_cell_bias.is_some()
            && k.kumo_rows.is_some()
            && k.kumo_labels.is_some()
            && k.kumo_copy.is_some()
    }

    /// True when the pack carries the fused passes (slots 709-713).
    pub fn has_kumo_fused(&self) -> bool {
        let k = &self.kernels;
        k.kumo_stats.is_some()
            && k.kumo_gemm_fused.is_some()
            && k.kumo_qgate.is_some()
            && k.kumo_rows_stats.is_some()
            && k.kumo_rope_table.is_some()
    }

    /// True when the pack runs an ensemble's members in one pass (slots
    /// 714-717).
    pub fn has_kumo_members(&self) -> bool {
        let k = &self.kernels;
        k.kumo_fourier_m.is_some()
            && k.kumo_cell_weights_m.is_some()
            && k.kumo_cell_bias_m.is_some()
            && k.kumo_labels_m.is_some()
    }

    /// The rope's (cos, sin) pairs of `seq` positions for one `half`-entry
    /// inv_freq vector into `table` at a float offset, `[seq][half]` pairs
    /// (slot 713) - what a fused projection's rope reads.
    pub fn kumo_rope_table(
        &self,
        freq: &CudaSlice<f32>,
        (table, off): (&mut CudaSlice<f32>, usize),
        half: usize,
        seq: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_rope_table
            .ok_or(GpuError::MissingOp("kumo_rope_table"))?;
        if freq.len() < half || off % 2 != 0 || table.len() < off + 2 * half * seq {
            return Err(oob("kumo_rope_table: buffers under the geometry"));
        }
        let (fp, _g1) = freq.device_ptr(&self.stream);
        let (tp, _g2) = table.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 713); bounds checked above
        check(unsafe {
            f(
                fp as *const _,
                (tp + 4 * off as u64) as *mut _,
                half as u32,
                seq as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The RMSNorm statistic of `rows` rows of `d` channels, `inv[row]`
    /// (slot 709). With `add = Some((b, period, res))` the rows are first
    /// `x[row % period] + b[row]`, kept in `res`.
    #[allow(clippy::type_complexity)]
    pub fn kumo_stats(
        &self,
        x: &CudaSlice<f32>,
        add: Option<(&CudaSlice<f32>, usize, &mut CudaSlice<f32>)>,
        inv: &mut CudaSlice<f32>,
        d: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_stats
            .ok_or(GpuError::MissingOp("kumo_stats"))?;
        if rows == 0 {
            return Ok(());
        }
        let x_rows = add
            .as_ref()
            .map_or(rows, |(_, period, _)| (*period).min(rows));
        if x.len() < x_rows * d
            || inv.len() < rows
            || add.as_ref().is_some_and(|(b, period, res)| {
                *period == 0 || b.len() < rows * d || res.len() < rows * d
            })
        {
            return Err(oob("kumo_stats: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (ip, _g2) = inv.device_ptr_mut(&self.stream);
        let period = add.as_ref().map_or(0, |(_, p, _)| *p);
        let ap =
            add.map(|(b, _, res)| (b.device_ptr(&self.stream), res.device_ptr_mut(&self.stream)));
        // SAFETY: ABI contract (slot 709); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                ap.as_ref().map_or(0, |((b, _), _)| *b) as *const _,
                ap.as_ref().map_or(0, |(_, (r, _))| *r) as *mut _,
                ip as *mut _,
                d as u32,
                rows as u32,
                period as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::kumo_gemm`] (one problem) with the [`KumoFuse`] fusions
    /// (slot 710). `epi` is Store, Gelu or Split; heads go with Store/Split.
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_gemm_fused(
        &self,
        x: &CudaSlice<f32>,
        w: KumoIn<'_>,
        bias: Option<KumoIn<'_>>,
        y: &mut CudaSlice<f32>,
        y2: Option<&mut CudaSlice<f32>>,
        (k, n, m): (usize, usize, usize),
        epi: KumoEpi,
        fuse: KumoFuse<'_>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_gemm_fused
            .ok_or(GpuError::MissingOp("kumo_gemm_fused"))?;
        if m == 0 {
            return Ok(());
        }
        let cols = if epi == KumoEpi::Split { n / 2 } else { n };
        // the x rows the product reads (the gather's last source row)
        let x_rows = match fuse.gather {
            Some((len, stride)) if len > 0 && stride >= len => {
                (m - 1) / len * stride + (m - 1) % len + 1
            }
            Some(_) => return Err(oob("kumo_gemm_fused: gather needs 0 < len <= stride")),
            None => m,
        };
        if (epi == KumoEpi::Split) != y2.is_some()
            || epi == KumoEpi::Resid
            || x.len() < x_rows * k
            || w.0.len() < w.1 + n * k
            || bias.is_some_and(|(b, o)| b.len() < o + n)
            || y.len() < m * cols
            || y2.as_ref().is_some_and(|y2| y2.len() < m * cols)
            || fuse
                .norm
                .is_some_and(|(inv, nw)| inv.len() < x_rows || nw.len() < k)
            || fuse.heads.is_some_and(|h| {
                epi == KumoEpi::Gelu
                    || !matches!(h.hd, 32 | 64)
                    || h.seq == 0
                    || h.rope
                        .is_some_and(|(t, o)| o % 2 != 0 || t.len() < o + h.seq * h.hd)
                    || h.scale.is_some_and(|(hs, _)| hs.len() < cols / h.hd)
            })
        {
            return Err(oob("kumo_gemm_fused: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.0.device_ptr(&self.stream);
        let bp = bias.map(|(b, o)| (b.device_ptr(&self.stream), o));
        let np = fuse
            .norm
            .map(|(inv, nw)| (inv.device_ptr(&self.stream), nw.device_ptr(&self.stream)));
        let fp = fuse
            .heads
            .and_then(|h| h.rope)
            .map(|(t, o)| (t.device_ptr(&self.stream), o));
        let hp = fuse
            .heads
            .and_then(|h| h.scale)
            .map(|(hs, klen)| (hs.device_ptr(&self.stream), klen));
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let y2p = y2.map(|y2| y2.device_ptr_mut(&self.stream));
        let (alen, astride) = fuse.gather.unwrap_or((0, 0));
        let (hd, seq) = fuse.heads.map_or((0, 0), |h| (h.hd, h.seq));
        // SAFETY: ABI contract (slot 710); bounds checked above, null = none
        check(unsafe {
            f(
                xp as *const _,
                (wp + 4 * w.1 as u64) as *const _,
                bp.as_ref().map_or(0, |((p, _), o)| *p + 4 * *o as u64) as *const _,
                yp as *mut _,
                y2p.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                k as u32,
                n as u32,
                m as u32,
                epi as u32,
                np.as_ref().map_or(0, |((i, _), _)| *i) as *const _,
                np.as_ref().map_or(0, |(_, (w, _))| *w) as *const _,
                alen as u32,
                astride as u32,
                hd as u32,
                seq as u32,
                fp.as_ref().map_or(0, |((p, _), o)| *p + 4 * *o as u64) as *const _,
                hp.as_ref().map_or(0, |((p, _), _)| *p) as *const _,
                hp.as_ref().map_or(0, |(_, klen)| *klen) as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The gated log query scaling in one pass, in place over `vecs` head
    /// vectors of `hd` (slot 711): `gate` = the gate MLP's `(w0, b0, w2, b2)`.
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_qgate(
        &self,
        q: &mut CudaSlice<f32>,
        (w0, b0, w2, b2): (
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            &CudaSlice<f32>,
        ),
        head_scale: &CudaSlice<f32>,
        vecs: usize,
        hd: usize,
        heads: usize,
        klen: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_qgate
            .ok_or(GpuError::MissingOp("kumo_qgate"))?;
        if q.len() < vecs * hd
            || w0.len() < 64 * hd
            || b0.len() < 64
            || w2.len() < hd * 64
            || b2.len() < hd
            || head_scale.len() < heads
        {
            return Err(oob("kumo_qgate: buffers under the geometry"));
        }
        let (w0p, _g1) = w0.device_ptr(&self.stream);
        let (b0p, _g2) = b0.device_ptr(&self.stream);
        let (w2p, _g3) = w2.device_ptr(&self.stream);
        let (b2p, _g4) = b2.device_ptr(&self.stream);
        let (hp, _g5) = head_scale.device_ptr(&self.stream);
        let (qp, _g6) = q.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 711); bounds checked above
        check(unsafe {
            f(
                qp as *mut _,
                w0p as *const _,
                b0p as *const _,
                w2p as *const _,
                b2p as *const _,
                hp as *const _,
                vecs as u32,
                hd as u32,
                heads as u32,
                klen as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::kumo_rows_pack`] writing each rows-layout row's statistic to
    /// `inv` (slot 712).
    pub fn kumo_rows_pack_stats(
        &self,
        cells: &CudaSlice<f32>,
        cls: &CudaSlice<f32>,
        rows: &mut CudaSlice<f32>,
        inv: &mut CudaSlice<f32>,
        (r, cols, d): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_rows_stats
            .ok_or(GpuError::MissingOp("kumo_rows_stats"))?;
        if cells.len() < cols * r * d
            || cls.len() < r * 4 * d
            || rows.len() < r * (cols + 4) * d
            || inv.len() < r * (cols + 4)
        {
            return Err(oob("kumo_rows_pack_stats: buffers under the geometry"));
        }
        let (cp, _g1) = cells.device_ptr(&self.stream);
        let (tp, _g2) = cls.device_ptr(&self.stream);
        let (rp, _g3) = rows.device_ptr_mut(&self.stream);
        let (ip, _g4) = inv.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 712, dir 0 reads cells/cls, writes rows/inv)
        check(unsafe {
            f(
                cp as *mut _,
                tp as *mut _,
                rp as *mut _,
                ip as *mut _,
                r as u32,
                cols as u32,
                d as u32,
                0,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::kumo_rows_unpack`] writing each cell row's statistic to `inv`
    /// (`[cols][r]`, slot 712).
    pub fn kumo_rows_unpack_stats(
        &self,
        rows: &CudaSlice<f32>,
        cells: &mut CudaSlice<f32>,
        cls: &mut CudaSlice<f32>,
        inv: &mut CudaSlice<f32>,
        (r, cols, d): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_rows_stats
            .ok_or(GpuError::MissingOp("kumo_rows_stats"))?;
        if cells.len() < cols * r * d
            || cls.len() < r * 4 * d
            || rows.len() < r * (cols + 4) * d
            || inv.len() < cols * r
        {
            return Err(oob("kumo_rows_unpack_stats: buffers under the geometry"));
        }
        let (rp, _g1) = rows.device_ptr(&self.stream);
        let (cp, _g2) = cells.device_ptr_mut(&self.stream);
        let (tp, _g3) = cls.device_ptr_mut(&self.stream);
        let (ip, _g4) = inv.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 712, dir 1 reads rows, writes cells/cls/inv)
        check(unsafe {
            f(
                cp as *mut _,
                tp as *mut _,
                rp as *mut _,
                ip as *mut _,
                r as u32,
                cols as u32,
                d as u32,
                1,
                self.stream_ptr(),
            )
        })
    }

    /// A pass plane of `n` floats, NOT zeroed: every Kumo plane is written
    /// in full before anything reads it, and zeroing a request's gigabytes of
    /// scratch would be a pass over them for nothing. Device memory, so an
    /// unwritten read is a wrong number, never host UB.
    pub fn kumo_plane(&self, n: usize) -> Result<CudaSlice<f32>, GpuError> {
        // SAFETY: device allocation on our stream; contents are defined by
        // the writers above (see the doc comment)
        unsafe { self.stream.alloc(n.max(1)) }.map_err(drv)
    }

    /// `y[b][m][n] = epi(sum_k x[b][m][k] * w[b][n][k] + bias[n])` for
    /// `batch` problems whose planes sit `strides = (x, w, y)` elements apart.
    /// `w` is an `nn.Linear` weight `[out, in]` (its rows are the n axis).
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_gemm(
        &self,
        x: KumoIn<'_>,
        w: KumoIn<'_>,
        bias: Option<KumoIn<'_>>,
        y: &mut CudaSlice<f32>,
        y2: Option<&mut CudaSlice<f32>>,
        (k, n, m): (usize, usize, usize),
        epi: KumoEpi,
        batch: usize,
        strides: (usize, usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_gemm
            .ok_or(GpuError::MissingOp("kumo_gemm"))?;
        if batch == 0 || m == 0 {
            return Ok(());
        }
        let cols = if epi == KumoEpi::Split { n / 2 } else { n };
        if (epi == KumoEpi::Split) != y2.is_some()
            || x.0.len() < span(x.1, batch, strides.0, m * k)
            || w.0.len() < span(w.1, batch, strides.1, n * k)
            || bias.is_some_and(|(b, o)| b.len() < o + n)
            || y.len() < span(0, batch, strides.2, m * cols)
            || y2
                .as_ref()
                .is_some_and(|y2| y2.len() < span(0, batch, strides.2, m * cols))
        {
            return Err(oob("kumo_gemm: buffers under the GEMM geometry"));
        }
        let (xp, _g1) = x.0.device_ptr(&self.stream);
        let (wp, _g2) = w.0.device_ptr(&self.stream);
        let bp = bias.map(|(b, o)| (b.device_ptr(&self.stream), o));
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let y2p = y2.map(|y2| y2.device_ptr_mut(&self.stream));
        // SAFETY: ABI contract (slot 698); bounds checked above, null bias = none
        check(unsafe {
            f(
                (xp + 4 * x.1 as u64) as *const _,
                (wp + 4 * w.1 as u64) as *const _,
                bp.as_ref().map_or(0, |((p, _), o)| *p + 4 * *o as u64) as *const _,
                yp as *mut _,
                y2p.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                k as u32,
                n as u32,
                m as u32,
                epi as u32,
                batch as u32,
                strides.0 as u64,
                strides.1 as u64,
                strides.2 as u64,
                self.stream_ptr(),
            )
        })
    }

    /// RMSNorm (F32 epsilon) of `rows` rows of `d` channels into `y`, each
    /// read from row `row / len * stride + row % len` of `x` (the context rows
    /// of every column, `stride >= len` rows apart).
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_norm(
        &self,
        x: &CudaSlice<f32>,
        w: KumoIn<'_>,
        y: &mut CudaSlice<f32>,
        d: usize,
        rows: usize,
        len: usize,
        stride: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_norm
            .ok_or(GpuError::MissingOp("kumo_norm"))?;
        if rows == 0 {
            return Ok(());
        }
        if len == 0 || stride < len {
            return Err(oob("kumo_norm: row gather needs 0 < len <= stride"));
        }
        let last = (rows - 1) / len * stride + (rows - 1) % len;
        if x.len() < (last + 1) * d || w.0.len() < w.1 + d || y.len() < rows * d {
            return Err(oob("kumo_norm: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.0.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 699, gather form); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                std::ptr::null(),
                (wp + 4 * w.1 as u64) as *const _,
                std::ptr::null_mut(),
                yp as *mut _,
                d as u32,
                rows as u32,
                len as u32,
                stride as u32,
                0,
                self.stream_ptr(),
            )
        })
    }

    /// The residual add fused with the next RMSNorm:
    /// `res[row] = a[row % period] + b[row]`, `y = norm(res)`.
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_add_norm(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        w: KumoIn<'_>,
        res: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        d: usize,
        rows: usize,
        period: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_norm
            .ok_or(GpuError::MissingOp("kumo_norm"))?;
        if rows == 0 {
            return Ok(());
        }
        if period == 0
            || a.len() < period.min(rows) * d
            || b.len() < rows * d
            || w.0.len() < w.1 + d
            || res.len() < rows * d
            || y.len() < rows * d
        {
            return Err(oob("kumo_add_norm: buffers under the geometry"));
        }
        let (ap, _g1) = a.device_ptr(&self.stream);
        let (bp, _g2) = b.device_ptr(&self.stream);
        let (wp, _g3) = w.0.device_ptr(&self.stream);
        let (rp, _g4) = res.device_ptr_mut(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 699, fused form); bounds checked above
        check(unsafe {
            f(
                ap as *const _,
                bp as *const _,
                (wp + 4 * w.1 as u64) as *const _,
                rp as *mut _,
                yp as *mut _,
                d as u32,
                rows as u32,
                1,
                1,
                period as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Per (token, head) vector of `hd` channels: split-half rope at position
    /// `token % seq` when `freq` is given, then a weightless RMSNorm.
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_heads(
        &self,
        x: &CudaSlice<f32>,
        freq: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        heads: usize,
        hd: usize,
        seq: usize,
        vecs: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_heads
            .ok_or(GpuError::MissingOp("kumo_heads"))?;
        if x.len() < vecs * hd || y.len() < vecs * hd || freq.is_some_and(|q| q.len() < hd / 2) {
            return Err(oob("kumo_heads: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let fp = freq.map(|q| q.device_ptr(&self.stream));
        let (yp, _g2) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 700); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                fp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                yp as *mut _,
                heads as u32,
                hd as u32,
                seq as u32,
                vecs as u32,
                u32::from(fp.is_some()),
                self.stream_ptr(),
            )
        })
    }

    /// Query scaling in place over `n` values: `q * log(max(klen, 1)) *
    /// head_scale[head]`, times `1 + tanh(gate)` when `gate` is given.
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_scale(
        &self,
        q: &mut CudaSlice<f32>,
        head_scale: &CudaSlice<f32>,
        gate: Option<&CudaSlice<f32>>,
        n: usize,
        hd: usize,
        heads: usize,
        klen: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_scale
            .ok_or(GpuError::MissingOp("kumo_scale"))?;
        if q.len() < n || head_scale.len() < heads || gate.is_some_and(|g| g.len() < n) {
            return Err(oob("kumo_scale: buffers under the geometry"));
        }
        let (hp, _g1) = head_scale.device_ptr(&self.stream);
        let gp = gate.map(|g| g.device_ptr(&self.stream));
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 701); bounds checked above
        check(unsafe {
            f(
                qp as *mut _,
                hp as *const _,
                gp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                n as u64,
                hd as u32,
                heads as u32,
                klen as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Attention of queries `q0..q0 + qcount` of every batch over `klen`
    /// keys: `q` rows at `b * q_brows + pos` (`q_brows` 0 shares one query
    /// set), k/v `[batch][klen][kvh][hd]`, out `[batch][qlen][heads][hd]`.
    /// `qkvh` set: every query of the launch reads KV head
    /// `head / (heads / qkvh)` (Test-GQA query rows).
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_attention(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        (heads, hd): (usize, usize),
        (batch, qlen, klen, kvh): (usize, usize, usize, usize),
        q_brows: usize,
        qkvh: usize,
        (q0, qcount): (usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_attention
            .ok_or(GpuError::MissingOp("kumo_attention"))?;
        if batch == 0 || qcount == 0 {
            return Ok(());
        }
        let q_rows = if q_brows == 0 {
            q0 + qcount
        } else {
            (batch - 1) * q_brows + q0 + qcount
        };
        if q0 + qcount > qlen
            || (klen > 1 && q.len() < q_rows * heads * hd)
            || k.len() < batch * klen * kvh * hd
            || v.len() < batch * klen * kvh * hd
            || out.len() < batch * qlen * heads * hd
        {
            return Err(oob("kumo_attention: buffers under the geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 702); bounds checked above
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                heads as u32,
                hd as u32,
                batch as u32,
                qlen as u32,
                klen as u32,
                kvh as u32,
                q_brows as u32,
                qkvh as u32,
                q0 as u32,
                qcount as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The cell Fourier features `[cols][rows][192]` (slot 703). With
    /// `members` > 1 the rows are that many same-shape tables stacked
    /// member-major and `means` / `cat` hold one row of `cols` a member (slot
    /// 714).
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_fourier(
        &self,
        x: &CudaSlice<f32>,
        means: &CudaSlice<f32>,
        cat: &CudaSlice<u32>,
        num_freq: &CudaSlice<f32>,
        cat_freq: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        (rows, cols, members): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        if members == 0 || rows % members != 0 {
            return Err(oob("kumo_fourier: rows split over the members"));
        }
        if x.len() < rows * cols
            || means.len() < members * cols
            || cat.len() < members * cols
            || num_freq.len() < 96
            || cat_freq.len() < 96
            || out.len() < cols * rows * 192
        {
            return Err(oob("kumo_fourier: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (mp, _g2) = means.device_ptr(&self.stream);
        let (cp, _g3) = cat.device_ptr(&self.stream);
        let (np, _g4) = num_freq.device_ptr(&self.stream);
        let (fp, _g5) = cat_freq.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        let args = (
            xp as *const _,
            mp as *const _,
            cp as *const _,
            np as *const _,
            fp as *const _,
            op as *mut _,
        );
        if members == 1 {
            let f = self
                .kernels
                .kumo_fourier
                .ok_or(GpuError::MissingOp("kumo_fourier"))?;
            // SAFETY: ABI contract (slot 703); bounds checked above
            check(unsafe {
                f(
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                    args.5,
                    rows as u32,
                    cols as u32,
                    self.stream_ptr(),
                )
            })
        } else {
            let f = self
                .kernels
                .kumo_fourier_m
                .ok_or(GpuError::MissingOp("kumo_fourier_m"))?;
            // SAFETY: ABI contract (slot 714); bounds checked above
            check(unsafe {
                f(
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                    args.5,
                    rows as u32,
                    cols as u32,
                    (rows / members) as u32,
                    self.stream_ptr(),
                )
            })
        }
    }

    /// Each column's `[d][192]` cell projection (slot 704); for `members`
    /// stacked tables one per (column, member), `[cols][members][d][192]`
    /// (slot 715).
    pub fn kumo_cell_weights(
        &self,
        cat: &CudaSlice<u32>,
        num_w: &CudaSlice<f32>,
        cat_w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        (cols, d, members): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        if members == 0
            || cat.len() < members * cols
            || num_w.len() < d * 64
            || cat_w.len() < d * 64
            || out.len() < cols * members * d * 192
        {
            return Err(oob("kumo_cell_weights: buffers under the geometry"));
        }
        let (cp, _g1) = cat.device_ptr(&self.stream);
        let (np, _g2) = num_w.device_ptr(&self.stream);
        let (wp, _g3) = cat_w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        if members == 1 {
            let f = self
                .kernels
                .kumo_cell_weights
                .ok_or(GpuError::MissingOp("kumo_cell_weights"))?;
            // SAFETY: ABI contract (slot 704); bounds checked above
            check(unsafe {
                f(
                    cp as *const _,
                    np as *const _,
                    wp as *const _,
                    op as *mut _,
                    cols as u32,
                    d as u32,
                    self.stream_ptr(),
                )
            })
        } else {
            let f = self
                .kernels
                .kumo_cell_weights_m
                .ok_or(GpuError::MissingOp("kumo_cell_weights_m"))?;
            // SAFETY: ABI contract (slot 715); bounds checked above
            check(unsafe {
                f(
                    cp as *const _,
                    np as *const _,
                    wp as *const _,
                    op as *mut _,
                    cols as u32,
                    d as u32,
                    members as u32,
                    self.stream_ptr(),
                )
            })
        }
    }

    /// Group biases, missing-cell projections and - context rows only - the
    /// label embedding onto the projected cells `[cols][rows][d]` (slot 705;
    /// 716 for `members` stacked tables, each with its `nc` labels in `y`).
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_cell_bias(
        &self,
        cells: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        cat: &CudaSlice<u32>,
        (num_b, cat_b, missing): (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>),
        (target, y): (&CudaSlice<f32>, &CudaSlice<f32>),
        (rows, cols, d, nc, members): (usize, usize, usize, usize, usize),
        classification: bool,
    ) -> Result<(), GpuError> {
        if members == 0 || rows % members != 0 || nc > rows / members {
            return Err(oob("kumo_cell_bias: rows split over the members"));
        }
        if cells.len() < cols * rows * d
            || x.len() < rows * cols
            || cat.len() < members * cols
            || num_b.len() < d
            || cat_b.len() < d
            || missing.len() < d * 3
            || target.len() < if classification { 10 * d } else { d }
            || y.len() < members * nc
        {
            return Err(oob("kumo_cell_bias: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (cp, _g2) = cat.device_ptr(&self.stream);
        let (nbp, _g3) = num_b.device_ptr(&self.stream);
        let (cbp, _g4) = cat_b.device_ptr(&self.stream);
        let (mp, _g5) = missing.device_ptr(&self.stream);
        let (tp, _g6) = target.device_ptr(&self.stream);
        let (yp, _g7) = y.device_ptr(&self.stream);
        let (op, _g8) = cells.device_ptr_mut(&self.stream);
        let args = (
            op as *mut _,
            xp as *const _,
            cp as *const _,
            nbp as *const _,
            cbp as *const _,
            mp as *const _,
            tp as *const _,
            yp as *const _,
        );
        let dims = (rows as u32, cols as u32, d as u32, nc as u32);
        if members == 1 {
            let f = self
                .kernels
                .kumo_cell_bias
                .ok_or(GpuError::MissingOp("kumo_cell_bias"))?;
            // SAFETY: ABI contract (slot 705); bounds checked above, and the
            // runner admits only class codes 0..9
            check(unsafe {
                f(
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                    args.5,
                    args.6,
                    args.7,
                    dims.0,
                    dims.1,
                    dims.2,
                    dims.3,
                    u32::from(classification),
                    self.stream_ptr(),
                )
            })
        } else {
            let f = self
                .kernels
                .kumo_cell_bias_m
                .ok_or(GpuError::MissingOp("kumo_cell_bias_m"))?;
            // SAFETY: ABI contract (slot 716); bounds checked above, and the
            // runner admits only class codes 0..9
            check(unsafe {
                f(
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                    args.5,
                    args.6,
                    args.7,
                    dims.0,
                    dims.1,
                    dims.2,
                    dims.3,
                    u32::from(classification),
                    (rows / members) as u32,
                    self.stream_ptr(),
                )
            })
        }
    }

    /// `rows [R][cols + 4][d] = [readout tokens | cells]` (slot 706, pack).
    pub fn kumo_rows_pack(
        &self,
        cells: &CudaSlice<f32>,
        cls: &CudaSlice<f32>,
        rows: &mut CudaSlice<f32>,
        (r, cols, d): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_rows
            .ok_or(GpuError::MissingOp("kumo_rows"))?;
        if cells.len() < cols * r * d || cls.len() < r * 4 * d || rows.len() < r * (cols + 4) * d {
            return Err(oob("kumo_rows_pack: buffers under the geometry"));
        }
        let (cp, _g1) = cells.device_ptr(&self.stream);
        let (tp, _g2) = cls.device_ptr(&self.stream);
        let (rp, _g3) = rows.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 706, dir 0 reads cells/cls, writes rows)
        check(unsafe {
            f(
                cp as *mut _,
                tp as *mut _,
                rp as *mut _,
                r as u32,
                cols as u32,
                d as u32,
                0,
                self.stream_ptr(),
            )
        })
    }

    /// The row axis back into cells and readout tokens (slot 706, unpack).
    pub fn kumo_rows_unpack(
        &self,
        rows: &CudaSlice<f32>,
        cells: &mut CudaSlice<f32>,
        cls: &mut CudaSlice<f32>,
        (r, cols, d): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_rows
            .ok_or(GpuError::MissingOp("kumo_rows"))?;
        if cells.len() < cols * r * d || cls.len() < r * 4 * d || rows.len() < r * (cols + 4) * d {
            return Err(oob("kumo_rows_unpack: buffers under the geometry"));
        }
        let (rp, _g1) = rows.device_ptr(&self.stream);
        let (cp, _g2) = cells.device_ptr_mut(&self.stream);
        let (tp, _g3) = cls.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 706, dir 1 reads rows, writes cells/cls)
        check(unsafe {
            f(
                cp as *mut _,
                tp as *mut _,
                rp as *mut _,
                r as u32,
                cols as u32,
                d as u32,
                1,
                self.stream_ptr(),
            )
        })
    }

    /// The ICL label embedding onto the first `nc` rows of `x` (slot 707);
    /// for `members` blocks of `rpm` rows, onto each block's first `nc`, the
    /// labels `[members][nc]` (slot 717).
    pub fn kumo_labels(
        &self,
        x: &mut CudaSlice<f32>,
        y: &CudaSlice<f32>,
        target: &CudaSlice<f32>,
        (d, nc, rpm, members): (usize, usize, usize, usize),
        classification: bool,
    ) -> Result<(), GpuError> {
        if members == 0
            || nc > rpm
            || x.len() < ((members - 1) * rpm + nc) * d
            || y.len() < members * nc
            || target.len() < if classification { 10 * d } else { d }
        {
            return Err(oob("kumo_labels: buffers under the geometry"));
        }
        let (yp, _g1) = y.device_ptr(&self.stream);
        let (tp, _g2) = target.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr_mut(&self.stream);
        if members == 1 {
            let f = self
                .kernels
                .kumo_labels
                .ok_or(GpuError::MissingOp("kumo_labels"))?;
            // SAFETY: ABI contract (slot 707); bounds checked above
            check(unsafe {
                f(
                    xp as *mut _,
                    yp as *const _,
                    tp as *const _,
                    d as u32,
                    nc as u32,
                    u32::from(classification),
                    self.stream_ptr(),
                )
            })
        } else {
            let f = self
                .kernels
                .kumo_labels_m
                .ok_or(GpuError::MissingOp("kumo_labels_m"))?;
            // SAFETY: ABI contract (slot 717); bounds checked above
            check(unsafe {
                f(
                    xp as *mut _,
                    yp as *const _,
                    tp as *const _,
                    d as u32,
                    nc as u32,
                    u32::from(classification),
                    rpm as u32,
                    members as u32,
                    self.stream_ptr(),
                )
            })
        }
    }

    /// `dst[r][c] = src[r % period][c]` over `rows x width`, each side at its
    /// own row stride (slot 708).
    #[allow(clippy::too_many_arguments)]
    pub fn kumo_copy(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        src_stride: usize,
        dst_stride: usize,
        period: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kumo_copy
            .ok_or(GpuError::MissingOp("kumo_copy"))?;
        if rows == 0 || width == 0 {
            return Ok(());
        }
        if period == 0
            || src.len() < (period.min(rows) - 1) * src_stride + width
            || dst.len() < (rows - 1) * dst_stride + width
        {
            return Err(oob("kumo_copy: buffers under the geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 708); bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                rows as u32,
                width as u32,
                src_stride as u64,
                dst_stride as u64,
                period as u32,
                self.stream_ptr(),
            )
        })
    }
}
