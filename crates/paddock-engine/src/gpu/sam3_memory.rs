//! SAM 3's memory-path ops. The encoder's: the mask downsampler's small
//! stride-2 convs, the wide ones as the f16 ring's implicit conv at stride 2,
//! the ConvNeXt blocks' depthwise 7x7 + LayerNorm2d front and the
//! LayerNorm2d + GELU seam (`packs/cuda/src/sam3/memory.cuh` and slot 785
//! in `gemm/f16_dense.cuh`, slots 784-787; the 1x1 convs and the blocks'
//! MLPs are the dense lane's GEMMs). The memory attention's rope landing and
//! one-head attention (`sam3/memattn.cuh`, 788-789). The bank's bf16 store
//! and its landing into the attention's planes (`sam3/bank.cuh`, 790-791).
//!
//! Buffers are checked against the geometry in Rust before any launch.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::{bf16, f16};

use super::error::*;
use super::*;

/// What [`GpuExecutor::sam3_mem_down3`] reads: a channel-last f32 plane, or
/// the objects' mask logits made the first stage's input on the fly.
pub enum Sam3MemDownIn<'a> {
    /// `[nb][h][w][cin]`
    Plane(&'a CudaSlice<f32>),
    /// `[nb][sh][sw]` logits: sigmoid, or `binarize` (> 0, a point-prompted
    /// frame), then * 20 - 10, bilinear to the stage's h x w when it differs
    Mask {
        logits: &'a CudaSlice<f32>,
        sh: usize,
        sw: usize,
        binarize: bool,
    },
}

impl GpuExecutor {
    /// The memory encoder's own ops (slots 784-787), on top of the tracker
    /// heads' and the dense lane's GEMMs.
    pub fn has_sam3_memory(&self) -> bool {
        let k = &self.kernels;
        self.has_sam3_tracker()
            && k.sam3_mem_down3.is_some()
            && k.f16_conv3s2_gemm.is_some()
            && k.sam3_dwconv7_ln_h.is_some()
            && k.sam3_ln_gelu_h.is_some()
    }

    /// One small stage of the mask downsampler for `nb` objects:
    /// Conv2d(k3, s2, p1) (`w` `[cout][cin][3][3]`) -> LayerNorm2d (`ln`,
    /// `eps`) -> GELU(erf), from an `h x w` input into `[nb][h/2][w/2][cout]`.
    /// `cout` 4 or 16.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_mem_down3(
        &self,
        inp: Sam3MemDownIn<'_>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        ln: (&CudaSlice<f32>, &CudaSlice<f32>),
        out: Sam3MaskDownOut<'_>,
        nb: usize,
        h: usize,
        wd: usize,
        cin: usize,
        cout: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mem_down3
            .ok_or(GpuError::MissingOp("sam3_mem_down3"))?;
        let on = nb * (h / 2) * (wd / 2) * cout;
        let out_len = match &out {
            Sam3MaskDownOut::F32(o) => o.len(),
            Sam3MaskDownOut::F16(o) => o.len(),
        };
        let (src, mode, sh, sw) = match inp {
            Sam3MemDownIn::Plane(p) => {
                if p.len() < nb * h * wd * cin {
                    return Err(oob("sam3_mem_down3: input under the conv geometry"));
                }
                (p, 0u32, 0usize, 0usize)
            }
            Sam3MemDownIn::Mask {
                logits,
                sh,
                sw,
                binarize,
            } => {
                if cin != 1 || logits.len() < nb * sh * sw {
                    return Err(oob("sam3_mem_down3: mask logits under the geometry"));
                }
                (logits, if binarize { 2 } else { 1 }, sh, sw)
            }
        };
        if !matches!(cout, 4 | 16)
            || !h.is_multiple_of(2)
            || !wd.is_multiple_of(2)
            || w.len() < cout * cin * 9
            || b.len() < cout
            || ln.0.len() < cout
            || ln.1.len() < cout
            || out_len < on
        {
            return Err(oob("sam3_mem_down3: buffers under the conv geometry"));
        }
        let (ip, _g1) = src.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (lwp, _g4) = ln.0.device_ptr(&self.stream);
        let (lbp, _g5) = ln.1.device_ptr(&self.stream);
        let (op, half, _g6) = match out {
            Sam3MaskDownOut::F32(o) => {
                let (p, g) = o.device_ptr_mut(&self.stream);
                (p, 0u32, Guard::F32(g))
            }
            Sam3MaskDownOut::F16(o) => {
                let (p, g) = o.device_ptr_mut(&self.stream);
                (p, 1u32, Guard::F16(g))
            }
        };
        // SAFETY: ABI contract (slot 784); bounds checked above
        check(unsafe {
            f(
                ip as *const _,
                wp as *const _,
                bp as *const _,
                lwp as *const _,
                lbp as *const _,
                op as *mut _,
                nb as u32,
                h as u32,
                wd as u32,
                cin as u32,
                cout as u32,
                mode,
                sh as u32,
                sw as u32,
                eps,
                half,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::f16_conv3_gemm`] at stride 2 (slot 785), Conv2d(k3, s2, p1):
    /// `h x w` is the OUTPUT grid of a `2h x 2w` f16 source
    /// `[chips][src_chip_rows][c]` (`src_chip_rows >= 4 h w`), `wt` the
    /// tap-outer `[9 c][out]` weight, `y` f32 `[chips * h * w][out]`, `bias`
    /// added in the landing when given. `c` a multiple of 8.
    #[allow(clippy::too_many_arguments)]
    pub fn f16_conv3s2_gemm(
        &self,
        wt: &HalfTensor,
        src: &CudaSlice<f16>,
        y: &mut CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        chips: usize,
        h: usize,
        w: usize,
        c: usize,
        src_chip_rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_conv3s2_gemm
            .ok_or(GpuError::MissingOp("f16_conv3s2_gemm"))?;
        let out = wt.dims[1];
        if wt.dims[0] != 9 * c
            || src_chip_rows < 4 * h * w
            || src.len() < chips * src_chip_rows * c
            || y.len() < chips * h * w * out
            || bias.is_some_and(|b| b.len() < out)
        {
            return Err(oob("f16_conv3s2_gemm: buffers under the conv geometry"));
        }
        let (wp, _g1) = wt.buf.device_ptr(&self.stream);
        let (sp, _g2) = src.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let bg = bias.map(|b| b.device_ptr(&self.stream));
        let bp = bg.as_ref().map_or(0, |(p, _)| *p);
        // SAFETY: ABI contract (slot 785); bounds checked above, null bias = none
        check(unsafe {
            f(
                wp as *const _,
                sp as *const _,
                yp as *mut _,
                bp as *const _,
                chips as u32,
                h as u32,
                w as u32,
                c as u32,
                out as u32,
                src_chip_rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The ConvNeXt block's front for `nb` objects: depthwise Conv2d(k7, p3)
    /// with its bias (`w` tap-major `[49][c]`), then LayerNorm2d (`ln`,
    /// `eps`); f32 `[nb][h][w][c]` in, f16 out. `c` a multiple of 32, at
    /// most 1024.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_dwconv7_ln_h(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        ln: (&CudaSlice<f32>, &CudaSlice<f32>),
        out: &mut CudaSlice<f16>,
        nb: usize,
        h: usize,
        wd: usize,
        c: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_dwconv7_ln_h
            .ok_or(GpuError::MissingOp("sam3_dwconv7_ln_h"))?;
        let n = nb * h * wd * c;
        if c == 0
            || c > 1024
            || !c.is_multiple_of(32)
            || x.len() < n
            || out.len() < n
            || w.len() < 49 * c
            || b.len() < c
            || ln.0.len() < c
            || ln.1.len() < c
        {
            return Err(oob("sam3_dwconv7_ln_h: buffers under the conv geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (lwp, _g4) = ln.0.device_ptr(&self.stream);
        let (lbp, _g5) = ln.1.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 786); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                bp as *const _,
                lwp as *const _,
                lbp as *const _,
                op as *mut _,
                nb as u32,
                h as u32,
                wd as u32,
                c as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// LayerNorm2d (`ln`, `eps`) -> GELU(erf) over `rows` channel-last rows
    /// of `n`, f32 in, f16 out. `n` a multiple of 32, at most 1024.
    pub fn sam3_ln_gelu_h(
        &self,
        x: &CudaSlice<f32>,
        ln: (&CudaSlice<f32>, &CudaSlice<f32>),
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_ln_gelu_h
            .ok_or(GpuError::MissingOp("sam3_ln_gelu_h"))?;
        if n == 0
            || n > 1024
            || !n.is_multiple_of(32)
            || x.len() < rows * n
            || out.len() < rows * n
            || ln.0.len() < n
            || ln.1.len() < n
        {
            return Err(oob("sam3_ln_gelu_h: buffers under the row geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (lwp, _g2) = ln.0.device_ptr(&self.stream);
        let (lbp, _g3) = ln.1.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 787); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                lwp as *const _,
                lbp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// The memory attention's own ops (slots 788-789), on top of the
    /// encoder's.
    pub fn has_sam3_memory_attention(&self) -> bool {
        let k = &self.kernels;
        self.has_sam3_memory() && k.sam3_rope_rows_h.is_some() && k.sam3_mem_attn_h.is_some()
    }

    /// f32 rows `[rows][in_stride]` from column `in_off`, `d` wide, to f16
    /// `[rows][d]`: rotate-half rope (pairs `j`, `j + d/2`) on the first
    /// `nrope` rows of every `group`, at table row `(r % group) % period` of
    /// the `[period][d/2]` `rope` tables, then `* scale`. No `rope`: a plain
    /// convert (with the scale).
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_rope_rows_h(
        &self,
        inp: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rope: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
        rows: usize,
        in_stride: usize,
        in_off: usize,
        d: usize,
        group: usize,
        nrope: usize,
        period: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_rope_rows_h
            .ok_or(GpuError::MissingOp("sam3_rope_rows_h"))?;
        if d == 0
            || !d.is_multiple_of(2)
            || in_off + d > in_stride
            || group == 0
            || period == 0
            || inp.len() < rows * in_stride
            || out.len() < rows * d
            || rope.is_some_and(|(c, s)| c.len() < period * d / 2 || s.len() < period * d / 2)
        {
            return Err(oob("sam3_rope_rows_h: buffers under the row geometry"));
        }
        let (ip, _g1) = inp.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        let cg = rope.map(|(c, _)| c.device_ptr(&self.stream));
        let sg = rope.map(|(_, s)| s.device_ptr(&self.stream));
        let cp = cg.as_ref().map_or(0, |(p, _)| *p);
        let sp = sg.as_ref().map_or(0, |(p, _)| *p);
        // SAFETY: ABI contract (slot 788); bounds checked above, null tables = no rope
        check(unsafe {
            f(
                ip as *const _,
                op as *mut _,
                cp as *const _,
                sp as *const _,
                rows as u32,
                in_stride as u32,
                in_off as u32,
                d as u32,
                group as u32,
                nrope as u32,
                period as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// One-head attention for `groups` independent groups: `q`
    /// `[g][nq][256]` f16 pre-scaled by 1/16, `k` `[g][nk][256]`, values `dv`
    /// (64 or 128) wide at row stride `ldv` from column `v_off` of `v`
    /// (`[g][nk][ldv]`), out `dv` wide at row stride `ldo` from column
    /// `o_off` of `out` (`[g][nq][ldo]`).
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_mem_attn_h(
        &self,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        nq: usize,
        nk: usize,
        groups: usize,
        dv: usize,
        (ldv, v_off): (usize, usize),
        (ldo, o_off): (usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mem_attn_h
            .ok_or(GpuError::MissingOp("sam3_mem_attn_h"))?;
        if !matches!(dv, 64 | 128)
            || v_off + dv > ldv
            || o_off + dv > ldo
            || !ldv.is_multiple_of(8)
            || !ldo.is_multiple_of(8)
            || !v_off.is_multiple_of(8)
            || !o_off.is_multiple_of(8)
            || q.len() < groups * nq * 256
            || k.len() < groups * nk * 256
            || v.len() < groups * nk * ldv
            || out.len() < groups * nq * ldo
        {
            return Err(oob("sam3_mem_attn_h: buffers under the attention geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // the value half and the output half are column offsets into wider
        // planes, two bytes an element
        let vp = vp + (v_off * 2) as u64;
        let op = op + (o_off * 2) as u64;
        // SAFETY: ABI contract (slot 789); bounds checked above
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                nq as u32,
                nk as u32,
                groups as u32,
                dv as u32,
                ldv as u32,
                ldo as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The memory bank's landing (slots 790-791) and the bf16 store.
    pub fn has_sam3_bank(&self) -> bool {
        let k = &self.kernels;
        self.has_sam3_memory_attention()
            && k.sam3_bank_mem_rows.is_some()
            && k.sam3_bank_ptr_rows.is_some()
            && k.convert_f32_bf16.is_some()
    }

    /// `n` f32 values from `src[src_off..]` stored bf16 into `dst` - an
    /// object's memory the way Meta keeps it (`.to(torch.bfloat16)`, round
    /// to nearest even). Slot 548.
    pub fn sam3_bank_store(
        &self,
        src: &CudaSlice<f32>,
        src_off: usize,
        dst: &mut CudaSlice<bf16>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .convert_f32_bf16
            .ok_or(GpuError::MissingOp("convert_f32_bf16"))?;
        if src.len() < src_off + n || dst.len() < n {
            return Err(oob("sam3_bank_store: buffers under the store"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        let sp = sp + (src_off * 4) as u64;
        // SAFETY: ABI contract (slot 548); bounds checked above
        check(unsafe { f(sp as *const _, dp as *mut _, n as u64, self.stream_ptr()) })
    }

    /// One memory frame into the bank's planes at row `row0`: `mem` bf16
    /// `[rows][64]`, its position `pos` f32 `[rows][64]` plus row `tpos_row`
    /// of the `[7][64]` temporal table `tpos`. `kin` gets
    /// `f16(m + (pos + tpos))`, `v` gets `f16(m)`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_bank_mem_rows(
        &self,
        mem: &CudaSlice<bf16>,
        pos: &CudaSlice<f32>,
        tpos: &CudaSlice<f32>,
        tpos_row: usize,
        kin: &mut CudaSlice<f16>,
        v: &mut CudaSlice<f16>,
        row0: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        const W: usize = 64;
        let f = self
            .kernels
            .sam3_bank_mem_rows
            .ok_or(GpuError::MissingOp("sam3_bank_mem_rows"))?;
        if mem.len() < rows * W
            || pos.len() < rows * W
            || tpos.len() < (tpos_row + 1) * W
            || kin.len() < (row0 + rows) * W
            || v.len() < (row0 + rows) * W
        {
            return Err(oob("sam3_bank_mem_rows: buffers under the bank geometry"));
        }
        let (mp, _g1) = mem.device_ptr(&self.stream);
        let (pp, _g2) = pos.device_ptr(&self.stream);
        let (tp, _g3) = tpos.device_ptr(&self.stream);
        let (kp, _g4) = kin.device_ptr_mut(&self.stream);
        let (vp, _g5) = v.device_ptr_mut(&self.stream);
        // a 64-wide row is 256 bytes of table and 128 of f16, so both offsets
        // keep the kernel's 16-byte accesses aligned
        let tp = tp + (tpos_row * W * 4) as u64;
        let kp = kp + (row0 * W * 2) as u64;
        let vp = vp + (row0 * W * 2) as u64;
        // SAFETY: ABI contract (slot 790); bounds checked above
        check(unsafe {
            f(
                mp as *const _,
                pp as *const _,
                tp as *const _,
                kp as *mut _,
                vp as *mut _,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The object pointers into the bank's planes from row `row0`, 4 rows
    /// each: pointer `i` is row `rows[i]` of `pool` (f32 `[*][256]`) at
    /// distance `dist[i]` (positions `dist / tmax`; `w` `[64][256]` and `b`
    /// the temporal projection). `meta` is the `[2][n]` u32 staging the
    /// table rides in.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_bank_ptr_rows(
        &self,
        pool: &CudaSlice<f32>,
        rows: &[u32],
        dist: &[u32],
        meta: &mut CudaSlice<u32>,
        (w, b): (&CudaSlice<f32>, &CudaSlice<f32>),
        kin: &mut CudaSlice<f16>,
        v: &mut CudaSlice<f16>,
        row0: usize,
        tmax: f32,
    ) -> Result<(), GpuError> {
        const W: usize = 64;
        let f = self
            .kernels
            .sam3_bank_ptr_rows
            .ok_or(GpuError::MissingOp("sam3_bank_ptr_rows"))?;
        let n = rows.len();
        if n == 0 {
            return Ok(());
        }
        let pool_rows = pool.len() / 256;
        if dist.len() != n
            || rows.iter().any(|&r| r as usize >= pool_rows)
            || meta.len() < 2 * n
            || w.len() < W * 256
            || b.len() < W
            || kin.len() < (row0 + 4 * n) * W
            || v.len() < (row0 + 4 * n) * W
            || tmax <= 0.0
        {
            return Err(oob(
                "sam3_bank_ptr_rows: buffers under the pointer geometry",
            ));
        }
        let host: Vec<u32> = rows.iter().chain(dist).copied().collect();
        self.upload_u32(&host, meta)?;
        let (pp, _g1) = pool.device_ptr(&self.stream);
        let (mp, _g2) = meta.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (bp, _g4) = b.device_ptr(&self.stream);
        let (kp, _g5) = kin.device_ptr_mut(&self.stream);
        let (vp, _g6) = v.device_ptr_mut(&self.stream);
        let kp = kp + (row0 * W * 2) as u64;
        let vp = vp + (row0 * W * 2) as u64;
        // SAFETY: ABI contract (slot 791); every pool row the table names was
        // checked against the pool above
        check(unsafe {
            f(
                pp as *const _,
                mp as *const _,
                wp as *const _,
                bp as *const _,
                kp as *mut _,
                vp as *mut _,
                n as u32,
                tmax,
                self.stream_ptr(),
            )
        })
    }
}

/// Holds whichever device-pointer guard the output's type gave.
enum Guard<A, B> {
    F32(A),
    F16(B),
}
