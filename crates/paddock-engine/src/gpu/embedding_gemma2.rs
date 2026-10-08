//! EmbeddingGemma 2's text-encoder ops - kernel side
//! `packs/cuda/src/embedding_gemma2.cuh`, slots 806-812.
//!
//! Every plane is a PACKED ragged batch, `[rows][channels]` with sequence `s`
//! owning rows `cu[s]..cu[s + 1]`. The GEMMs are the mmq int8 and bf16 tile
//! ones; what lives here is the encoder's own row work, each kernel doing a
//! row in one fixed shape whatever the row count - so a sequence's vector does
//! not depend on what was packed with it.
//!
//! Buffers are checked against the geometry before any launch - a short slice
//! is a logic error here, not something to hand the driver.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

/// The model width every row kernel here is built for (`PD_EG2_W`).
pub const EG2_WIDTH: usize = 512;
/// The PLE planes per token (`PD_EG2_LAYERS`).
pub const EG2_LAYERS: usize = 24;
/// Query rows one attention block owns at hd 256 / hd 512 (`PdEg2Geo::ROWS`).
pub const EG2_ATTN_ROWS: [usize; 2] = [64, 16];
/// How a tile descriptor packs its sequence (`PD_EG2_TILE_SHIFT`).
pub const EG2_TILE_SHIFT: u32 = 12;

/// Bytes of the mmq activation layout for `rows` rows of `in_dim`:
/// `[ceil(in / 128) chunks][rows padded to 128][144 B]`.
pub fn mmq_bytes(in_dim: usize, rows: usize) -> usize {
    in_dim.div_ceil(128) * rows.next_multiple_of(128) * 144
}

impl GpuExecutor {
    /// True when the loaded pack carries the whole EmbeddingGemma 2 lane (the
    /// slots landed together) and the bf16 tile its PLE projection rides.
    pub fn has_embedding_gemma2(&self) -> bool {
        let k = &self.kernels;
        k.bf16_gemm_mma.is_some()
            && k.eg2_rms.is_some()
            && k.eg2_sandwich.is_some()
            && k.eg2_heads.is_some()
            && k.eg2_attn.is_some()
            && k.eg2_pool.is_some()
            && k.eg2_gemm_rows.is_some()
            && k.eg2_geglu_q.is_some()
    }

    /// `rmsnorm(x * scale) * w` over `n_rows` 512-wide rows into the f32
    /// `out` and/or the mmq-quantized `yq` (at least one). With `ple_tokens`
    /// set, `x` is the PLE projection `[tokens][24][512]` and `out` is laid
    /// layer-major `[24][tokens][512]` (f32 only).
    #[allow(clippy::too_many_arguments)]
    pub fn eg2_rms(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: Option<&mut CudaSlice<f32>>,
        yq: Option<&mut CudaSlice<u8>>,
        n_rows: usize,
        ple_tokens: usize,
        scale: f32,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self.kernels.eg2_rms.ok_or(GpuError::MissingOp("eg2_rms"))?;
        if x.len() < n_rows * EG2_WIDTH
            || w.len() < EG2_WIDTH
            || out.as_ref().is_some_and(|o| o.len() < n_rows * EG2_WIDTH)
            || yq
                .as_ref()
                .is_some_and(|q| q.len() < mmq_bytes(EG2_WIDTH, n_rows))
            || (out.is_none() && yq.is_none())
            || (ple_tokens > 0 && (n_rows != ple_tokens * EG2_LAYERS || yq.is_some()))
        {
            return Err(oob("eg2_rms: buffers under the row geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let op = out.map(|o| o.device_ptr_mut(&self.stream));
        let qp = yq.map(|q| q.device_ptr_mut(&self.stream));
        // SAFETY: ABI contract (slot 806); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                op.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                qp.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                n_rows as u32,
                ple_tokens as u32,
                scale,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// The sandwich seam: `x = (x + rmsnorm(proj) * wpost) * s`; with
    /// `next = Some((w, xn))` also `xn = rmsnorm(x) * w` (`xn` optional).
    /// `yq` receives the mmq-quantized next input: xn with `next`, else x.
    #[allow(clippy::too_many_arguments)]
    pub fn eg2_sandwich(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        wpost: &CudaSlice<f32>,
        next: Option<(&CudaSlice<f32>, Option<&mut CudaSlice<f32>>)>,
        yq: Option<&mut CudaSlice<u8>>,
        rows: usize,
        s: f32,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2_sandwich
            .ok_or(GpuError::MissingOp("eg2_sandwich"))?;
        let n = rows * EG2_WIDTH;
        if x.len() < n
            || proj.len() < n
            || wpost.len() < EG2_WIDTH
            || yq
                .as_ref()
                .is_some_and(|q| q.len() < mmq_bytes(EG2_WIDTH, rows))
            || next.as_ref().is_some_and(|(w, xn)| {
                w.len() < EG2_WIDTH || xn.as_ref().is_some_and(|b| b.len() < n)
            })
        {
            return Err(oob("eg2_sandwich: buffers under the row geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = wpost.device_ptr(&self.stream);
        let (nw, nx) = match next {
            Some((w, xn)) => (
                Some(w.device_ptr(&self.stream)),
                xn.map(|b| b.device_ptr_mut(&self.stream)),
            ),
            None => (None, None),
        };
        let qp = yq.map(|q| q.device_ptr_mut(&self.stream));
        // SAFETY: ABI contract (slot 807); bounds checked above; xn only with
        // wnext
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                nw.as_ref().map_or(0, |(p, _)| *p) as *const _,
                nx.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                qp.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                rows as u32,
                s,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// The q/k/v head transform off the fused landing `qkv` (`[rows][stride]`,
    /// q at 0, k at `4 hd`, v at `4 hd + 512`): RMS norm, split-half rope at
    /// `pos` on q and k, f16 out.
    #[allow(clippy::too_many_arguments)]
    pub fn eg2_heads(
        &self,
        qkv: &CudaSlice<f32>,
        pos: &CudaSlice<u32>,
        qw: &CudaSlice<f32>,
        kw: &CudaSlice<f32>,
        q16: &mut CudaSlice<f16>,
        k16: &mut CudaSlice<f16>,
        v16: &mut CudaSlice<f16>,
        rows: usize,
        stride: usize,
        head_dim: usize,
        theta_scale: f32,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2_heads
            .ok_or(GpuError::MissingOp("eg2_heads"))?;
        if !matches!(head_dim, 256 | 512)
            || stride < 4 * head_dim + 1024
            || qkv.len() < rows * stride
            || pos.len() < rows
            || qw.len() < head_dim
            || kw.len() < head_dim
            || q16.len() < rows * 4 * head_dim
            || k16.len() < rows * 512
            || v16.len() < rows * 512
        {
            return Err(oob("eg2_heads: buffers under the head geometry"));
        }
        let (xp, _g1) = qkv.device_ptr(&self.stream);
        let (pp, _g2) = pos.device_ptr(&self.stream);
        let (qwp, _g3) = qw.device_ptr(&self.stream);
        let (kwp, _g4) = kw.device_ptr(&self.stream);
        let (qp, _g5) = q16.device_ptr_mut(&self.stream);
        let (kp, _g6) = k16.device_ptr_mut(&self.stream);
        let (vp, _g7) = v16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 808); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                pp as *const _,
                qwp as *const _,
                kwp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                rows as u32,
                stride as u32,
                head_dim as u32,
                theta_scale,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Bidirectional varlen attention, f32 out `[rows][4][hd]`. `tiles` holds
    /// `(seq << 12) | tile` per [`EG2_ATTN_ROWS`] query rows of each sequence;
    /// `window` 0 = full, else |i - j| <= window. `split` 2 or 4 spreads each
    /// tile's warps over that many blocks (slot 829, bit-identical per row);
    /// 1 is slot 809's grid.
    #[allow(clippy::too_many_arguments)]
    pub fn eg2_attn(
        &self,
        q16: &CudaSlice<f16>,
        k16: &CudaSlice<f16>,
        v16: &CudaSlice<f16>,
        cu: &CudaSlice<u32>,
        tiles: &CudaSlice<u32>,
        n_tiles: usize,
        out: &mut CudaSlice<f32>,
        rows: usize,
        head_dim: usize,
        window: usize,
        split: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2_attn
            .ok_or(GpuError::MissingOp("eg2_attn"))?;
        if !matches!(split, 1 | 2 | 4) || (split > 1 && self.kernels.eg2_attn_s.is_none()) {
            return Err(oob("eg2_attn: split must be 1, or 2 / 4 with slot 829"));
        }
        if !matches!(head_dim, 256 | 512)
            || q16.len() < rows * 4 * head_dim
            || k16.len() < rows * 512
            || v16.len() < rows * 512
            || out.len() < rows * 4 * head_dim
            || tiles.len() < n_tiles
        {
            return Err(oob("eg2_attn: buffers under the attention geometry"));
        }
        let (qp, _g1) = q16.device_ptr(&self.stream);
        let (kp, _g2) = k16.device_ptr(&self.stream);
        let (vp, _g3) = v16.device_ptr(&self.stream);
        let (cp, _g4) = cu.device_ptr(&self.stream);
        let (tp, _g5) = tiles.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        if let (true, Some(fs)) = (split > 1, self.kernels.eg2_attn_s) {
            // SAFETY: ABI contract (slot 829); bounds checked above; the
            // caller builds cu/tiles for these rows
            return check(unsafe {
                fs(
                    qp as *const _,
                    kp as *const _,
                    vp as *const _,
                    cp as *const _,
                    tp as *const _,
                    n_tiles as u32,
                    op as *mut _,
                    head_dim as u32,
                    window as u32,
                    split as u32,
                    self.stream_ptr(),
                )
            });
        }
        // SAFETY: ABI contract (slot 809); bounds checked above; the caller
        // builds cu/tiles for these rows
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                cp as *const _,
                tp as *const _,
                n_tiles as u32,
                op as *mut _,
                head_dim as u32,
                window as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the split attention grid (slot 829).
    pub fn has_eg2_attn_s(&self) -> bool {
        self.kernels.eg2_attn_s.is_some()
    }

    /// The small-pass Q8_0 GEMM (slot 811): `y [batch][out] = W . X` over
    /// the repacked rows and the mmq-quantized `yq`, bit-identical per output
    /// to [`Self::q8_0_gemm_mmq`]'s plain tile - so a caller may pick either
    /// by row count without the outputs depending on the choice.
    pub fn eg2_gemm_rows(
        &self,
        w: &RepackedQ8,
        yq: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2_gemm_rows
            .ok_or(GpuError::MissingOp("eg2_gemm_rows"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if in_dim % 512 != 0
            || in_dim > 2048
            || yq.len() < mmq_bytes(in_dim, batch)
            || y.len() < batch * out_dim
        {
            return Err(oob("eg2_gemm_rows: buffers under the GEMM geometry"));
        }
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (qp, _g3) = yq.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 811); bounds checked above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                qp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `gelu_tanh(gate) * up` over `in_dim`-wide rows at row stride `ld`
    /// straight into the mmq layout (slot 812).
    #[allow(clippy::too_many_arguments)]
    pub fn eg2_geglu_q(
        &self,
        gate: &CudaSlice<f32>,
        gate_off: usize,
        up: &CudaSlice<f32>,
        up_off: usize,
        ld: usize,
        yq: &mut CudaSlice<u8>,
        in_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2_geglu_q
            .ok_or(GpuError::MissingOp("eg2_geglu_q"))?;
        let need = |off: usize| off + (batch.max(1) - 1) * ld + in_dim;
        if gate.len() < need(gate_off)
            || up.len() < need(up_off)
            || yq.len() < mmq_bytes(in_dim, batch)
        {
            return Err(oob("eg2_geglu_q: buffers under the GLU geometry"));
        }
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_p, _g2) = up.device_ptr(&self.stream);
        let (qp, _g3) = yq.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 812); offsets and strides checked above
        check(unsafe {
            f(
                (gp + (gate_off * 4) as u64) as *const _,
                (up_p + (up_off * 4) as u64) as *const _,
                ld as u32,
                qp as *mut _,
                in_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// One layer's slice of the PLE projection: rows `[first_row, first_row +
    /// out_dim)` of the bf16 plane `w` (`[out][in]`) against `x` (`[batch][in]`
    /// f32) on the tensor-core tile (slot 391), always - never the general
    /// dispatcher, whose GEMV bands below nine rows walk K in a different
    /// order, so the same row would come out differently in a small pass and
    /// a large one. At in = 512 the tile's K-split is one slab whatever the
    /// grid, and its configs move tile ownership, not the K sequence. The
    /// tile refuses one row: callers pad a lone row to two.
    #[allow(clippy::too_many_arguments)]
    pub fn eg2_ple_gemm(
        &self,
        w: &QuantTensor,
        first_row: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .bf16_gemm_mma
            .ok_or(GpuError::MissingOp("bf16_gemm_mma"))?;
        let in_dim = w.dims[0];
        if w.ty != paddock_models::ggml_type::GgmlType::Bf16
            || batch < 2
            || first_row + out_dim > w.dims[1]
            || x.len() < batch * in_dim
            || y.len() < batch * out_dim
        {
            return Err(oob("eg2_ple_gemm: buffers under the projection geometry"));
        }
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 391); the segment stays inside the plane
        // and every buffer was checked above
        check(unsafe {
            f(
                (wp + (first_row * in_dim * 2) as u64) as *const _,
                core::ptr::null(),
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Mean pool `[rows][768]` per sequence and L2-normalize the first `dims`
    /// components into `[n_seq][dims]`.
    pub fn eg2_pool(
        &self,
        tok: &CudaSlice<f32>,
        cu: &CudaSlice<u32>,
        n_seq: usize,
        dims: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2_pool
            .ok_or(GpuError::MissingOp("eg2_pool"))?;
        if !matches!(dims, 128 | 256 | 512 | 768)
            || out.len() < n_seq * dims
            || cu.len() < n_seq + 1
        {
            return Err(oob("eg2_pool: buffers under the pooling geometry"));
        }
        let (tp, _g1) = tok.device_ptr(&self.stream);
        let (cp, _g2) = cu.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 810); bounds checked above
        check(unsafe {
            f(
                tp as *const _,
                cp as *const _,
                n_seq as u32,
                dims as u32,
                op as *mut _,
                self.stream_ptr(),
            )
        })
    }
}
