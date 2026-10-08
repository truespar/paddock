//! Gemma 4's vision tower, the fused glue between its f16 GEMMs - kernel side
//! `packs/cuda/src/gemma4v.cuh`, slots 820-825 and 827. Each op is bit-identical to
//! the unfused chain it replaces (see the kernel file); the norm-bearing ones
//! return `Ok(false)` when the pack declines them (a norm accumulate mode
//! other than the default double-float), and the caller keeps that chain.
//! Buffers are checked against the geometry before any launch.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

/// What a declining launcher returns (`-2`): the caller keeps the unfused chain.
const DECLINED: i32 = -2;

impl GpuExecutor {
    /// True when the pack carries the whole fused tower glue and the half
    /// attention entry it feeds.
    pub fn has_g4v(&self) -> bool {
        let k = &self.kernels;
        k.vision_attn_h.is_some()
            && k.g4v_patchify.is_some()
            && k.g4v_pos_norm.is_some()
            && k.g4v_heads.is_some()
            && k.g4v_post.is_some()
            && k.g4v_geglu.is_some()
            && k.g4v_pool.is_some()
            && k.g4v_rope_table.is_some()
    }

    /// The GEGLU epilogue landing (slot 826) is here too.
    pub fn has_g4v_geglu(&self) -> bool {
        self.has_g4v() && self.kernels.f16_gemm_h_geglu_g4.is_some()
    }

    /// A `[th][tw][3]` u8 picture into f16 im2row patches `[n][3 * patch^2]`.
    pub fn g4v_patchify(
        &self,
        rgb: &CudaSlice<u8>,
        out: &mut CudaSlice<f16>,
        tw: usize,
        th: usize,
        patch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .g4v_patchify
            .ok_or(GpuError::MissingOp("g4v_patchify"))?;
        let n = (tw / patch.max(1)) * (th / patch.max(1));
        if patch == 0 || rgb.len() < tw * th * 3 || out.len() < n * 3 * patch * patch {
            return Err(oob("g4v_patchify: buffers under the picture geometry"));
        }
        let (rp, _g1) = rgb.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 820); bounds checked above
        check(unsafe {
            f(
                rp as *const _,
                op as *mut _,
                tw as u32,
                th as u32,
                patch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `x += pos[cx] + pos[gw + cy]` over `rows` `n`-wide rows (`pos`: the
    /// picture's `gw` x-rows then its y-rows), then `out16 = ln1(x)`.
    #[allow(clippy::too_many_arguments)]
    pub fn g4v_pos_norm(
        &self,
        x: &mut CudaSlice<f32>,
        pos: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out16: &mut CudaSlice<f16>,
        gw: usize,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<bool, GpuError> {
        let f = self
            .kernels
            .g4v_pos_norm
            .ok_or(GpuError::MissingOp("g4v_pos_norm"))?;
        let gh = rows / gw.max(1);
        if gw == 0
            || x.len() < rows * n
            || pos.len() < (gw + gh) * n
            || w.len() < n
            || out16.len() < rows * n
        {
            return Err(oob("g4v_pos_norm: buffers under the row geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = pos.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 821); bounds checked above
        let r = unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                op as *mut _,
                gw as u32,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        };
        if r == DECLINED {
            return Ok(false);
        }
        check(r).map(|_| true)
    }

    /// The 2-D rope's `(cos, sin)` per (row, pair) into `tab` (`rows * hd`
    /// floats), from the rows' grid positions - once a picture, for
    /// [`Self::g4v_heads`].
    pub fn g4v_rope_table(
        &self,
        pos: (&CudaSlice<u32>, &CudaSlice<u32>),
        tab: &mut CudaSlice<f32>,
        rows: usize,
        hd: usize,
        theta_scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .g4v_rope_table
            .ok_or(GpuError::MissingOp("g4v_rope_table"))?;
        if pos.0.len() < rows || pos.1.len() < rows || tab.len() < rows * hd {
            return Err(oob("g4v_rope_table: buffers under the row geometry"));
        }
        let (xp, _g1) = pos.0.device_ptr(&self.stream);
        let (yp, _g2) = pos.1.device_ptr(&self.stream);
        let (tp, _g3) = tab.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 827); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                yp as *const _,
                tp as *mut _,
                rows as u32,
                hd as u32,
                theta_scale,
                self.stream_ptr(),
            )
        })
    }

    /// Per (row, head): q / k / v norms, the 2-D rope on q and k (angles from
    /// [`Self::g4v_rope_table`]'s `tab`), f16 into `[rows][heads][hd]`. q, k,
    /// v are row-strided by `ld`.
    #[allow(clippy::too_many_arguments)]
    pub fn g4v_heads(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        ld: usize,
        qw: &CudaSlice<f32>,
        kw: &CudaSlice<f32>,
        tab: &CudaSlice<f32>,
        out: (
            &mut CudaSlice<f16>,
            &mut CudaSlice<f16>,
            &mut CudaSlice<f16>,
        ),
        rows: usize,
        heads: usize,
        hd: usize,
        eps: f32,
    ) -> Result<bool, GpuError> {
        let f = self
            .kernels
            .g4v_heads
            .ok_or(GpuError::MissingOp("g4v_heads"))?;
        let need = rows.saturating_sub(1) * ld + heads * hd;
        let n = rows * heads * hd;
        if heads * hd > ld
            || [q, k, v].iter().any(|b| b.len() < need)
            || qw.len() < hd
            || kw.len() < hd
            || tab.len() < rows * hd
            || out.0.len() < n
            || out.1.len() < n
            || out.2.len() < n
        {
            return Err(oob("g4v_heads: buffers under the head geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (qwp, _g4) = qw.device_ptr(&self.stream);
        let (kwp, _g5) = kw.device_ptr(&self.stream);
        let (tp, _g6) = tab.device_ptr(&self.stream);
        let (q16, _g8) = out.0.device_ptr_mut(&self.stream);
        let (k16, _g9) = out.1.device_ptr_mut(&self.stream);
        let (v16, _g10) = out.2.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 822); bounds checked above
        let r = unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                ld as u32,
                qwp as *const _,
                kwp as *const _,
                tp as *const _,
                q16 as *mut _,
                k16 as *mut _,
                v16 as *mut _,
                rows as u32,
                heads as u32,
                hd as u32,
                eps,
                self.stream_ptr(),
            )
        };
        if r == DECLINED {
            return Ok(false);
        }
        check(r).map(|_| true)
    }

    /// `x += rmsnorm(proj) * wpost`, then with `next = (w, out16)` the next
    /// norm of x as f16.
    #[allow(clippy::too_many_arguments)]
    pub fn g4v_post(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        wpost: &CudaSlice<f32>,
        next: Option<(&CudaSlice<f32>, &mut CudaSlice<f16>)>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<bool, GpuError> {
        let f = self
            .kernels
            .g4v_post
            .ok_or(GpuError::MissingOp("g4v_post"))?;
        if x.len() < rows * n
            || proj.len() < rows * n
            || wpost.len() < n
            || next
                .as_ref()
                .is_some_and(|(w, o)| w.len() < n || o.len() < rows * n)
        {
            return Err(oob("g4v_post: buffers under the row geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = wpost.device_ptr(&self.stream);
        let (nw, no) = match next {
            Some((w, o)) => (
                Some(w.device_ptr(&self.stream)),
                Some(o.device_ptr_mut(&self.stream)),
            ),
            None => (None, None),
        };
        // SAFETY: ABI contract (slot 823); bounds checked above
        let r = unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                nw.as_ref().map_or(0, |(p, _)| *p) as *const _,
                no.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        };
        if r == DECLINED {
            return Ok(false);
        }
        check(r).map(|_| true)
    }

    /// `out = f16(gelu_tanh(gate) * up)` over `rows` x `ffn`, gate / up
    /// row-strided by `ld`; with `relaid`, `gate` is one `[rows][2 * ffn]`
    /// landing of the 16-row re-laid gate|up weight and `up` is ignored.
    #[allow(clippy::too_many_arguments)]
    pub fn g4v_geglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        ld: usize,
        out: &mut CudaSlice<f16>,
        ffn: usize,
        rows: usize,
        relaid: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .g4v_geglu
            .ok_or(GpuError::MissingOp("g4v_geglu"))?;
        let need = rows.saturating_sub(1) * ld + if relaid { 2 * ffn } else { ffn };
        if (if relaid { 2 * ffn } else { ffn }) > ld
            || gate.len() < need
            || (!relaid && up.len() < need)
            || out.len() < rows * ffn
        {
            return Err(oob("g4v_geglu: buffers under the row geometry"));
        }
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_, _g2) = up.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 824); bounds checked above
        check(unsafe {
            f(
                gp as *const _,
                up_ as *const _,
                ld as u32,
                op as *mut _,
                ffn as u32,
                rows as u32,
                relaid as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The f16 landing with Gemma 4's vision GEGLU in the epilogue (slot
    /// 826): `y[batch][out_dim / 2] = f16(gelu_tanh(gate) * up)` off the
    /// 16-row re-laid gate|up plane `w` (`[out_dim][in_dim]`).
    pub fn f16_gemm_h_geglu_g4(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y: &mut CudaSlice<f16>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_geglu_g4
            .ok_or(GpuError::MissingOp("f16_gemm_h_geglu_g4"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if out_dim % 16 != 0
            || w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y.len() < batch * out_dim / 2
        {
            return Err(oob("f16_gemm_h_geglu_g4: buffers under the GEMM geometry"));
        }
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 826); bounds checked above
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The tower's tail: `[gh][gw][embd]` rows -> pooled f16 `[gh/3 * gw/3][embd]`.
    #[allow(clippy::too_many_arguments)]
    pub fn g4v_pool(
        &self,
        x: &CudaSlice<f32>,
        std: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
        out: &mut CudaSlice<f16>,
        gw: usize,
        gh: usize,
        embd: usize,
        inv: f32,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .g4v_pool
            .ok_or(GpuError::MissingOp("g4v_pool"))?;
        if x.len() < gw * gh * embd
            || out.len() < (gw / 3) * (gh / 3) * embd
            || std.is_some_and(|(b, s)| b.len() < embd || s.len() < embd)
        {
            return Err(oob("g4v_pool: buffers under the grid geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (bp, sp) = match std {
            Some((b, s)) => (
                Some(b.device_ptr(&self.stream)),
                Some(s.device_ptr(&self.stream)),
            ),
            None => (None, None),
        };
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 825); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                bp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                sp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                op as *mut _,
                gw as u32,
                gh as u32,
                embd as u32,
                inv,
                eps,
                self.stream_ptr(),
            )
        })
    }
}
