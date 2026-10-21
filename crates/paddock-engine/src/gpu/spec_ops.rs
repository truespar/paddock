//! Speculative-decoding glue on the device: the MTP chain's position advance,
//! draft-token staging and acceptance, and the hidden-row gathers/stitches the
//! drafter reads.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

impl GpuExecutor {
    /// Device-side `buf[0..n] += k` on a u32 buffer. Graph-capturable; used
    /// to advance MTP chain rope positions inside the draft graph.
    pub fn u32_addk(&self, buf: &mut CudaSlice<u32>, n: usize, k: u32) -> Result<(), GpuError> {
        let f = self
            .kernels
            .u32_addk
            .ok_or(GpuError::MissingOp("u32_addk"))?;
        let (bp, _g) = buf.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; buf holds at least n u32s
        check(unsafe { f(bp as *mut _, n as u32, k, self.stream_ptr()) })
    }

    pub fn has_u32_addk(&self) -> bool {
        self.kernels.u32_addk.is_some()
    }

    /// Async spec round token assembly: write the verify tick's
    /// slot-major token rows from the drafter chain's step-major output
    /// plane, entirely on device. `meta` = [pend|srcrow|ndr|clen|base]
    /// (5n u32); `cmax` = max padded chunk len; `rr` = chain row count.
    pub fn spec_toks(
        &self,
        meta: &CudaSlice<u32>,
        drafts: &CudaSlice<u32>,
        dst: &mut CudaSlice<u32>,
        n: usize,
        cmax: usize,
        rr: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .spec_toks
            .ok_or(GpuError::MissingOp("spec_toks"))?;
        let (mp, _g1) = meta.device_ptr(&self.stream);
        let (dp, _g2) = drafts.device_ptr(&self.stream);
        let (op, _g3) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; meta 5n u32, drafts rr*k plane, dst covers
        // every base+clen row
        check(unsafe {
            f(
                mp as *const _,
                dp as *const _,
                op as *mut _,
                n as u32,
                cmax as u32,
                rr as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_spec_toks(&self) -> bool {
        self.kernels.spec_toks.is_some()
    }

    /// Device-side spec accept: run the accept-while-
    /// match walk on device right after the verify tick; `strip` receives
    /// {accepted, p_final, final_row, new_pending, tokens...} per slot at
    /// the given u32 stride.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_accept(
        &self,
        sampled: &CudaSlice<u32>,
        drafts: &CudaSlice<u32>,
        meta: &CudaSlice<u32>,
        pos: &CudaSlice<u32>,
        strip: &mut CudaSlice<u32>,
        n: usize,
        rr: usize,
        stride: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .spec_accept
            .ok_or(GpuError::MissingOp("spec_accept"))?;
        let (sp, _g1) = sampled.device_ptr(&self.stream);
        let (dp, _g2) = drafts.device_ptr(&self.stream);
        let (mp, _g3) = meta.device_ptr(&self.stream);
        let (pp, _g4) = pos.device_ptr(&self.stream);
        let (op, _g5) = strip.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; strip holds n*stride u32
        check(unsafe {
            f(
                sp as *const _,
                dp as *const _,
                mp as *const _,
                pp as *const _,
                op as *mut _,
                n as u32,
                rr as u32,
                stride as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_spec_accept(&self) -> bool {
        self.kernels.spec_accept.is_some()
    }

    /// Rung B2: accept + next-round device prep in one launch. Writes the
    /// strip AND the chain's tok/rope/bound buffers, the meta pend lane,
    /// and the next verify's position rows. `strip_off` selects the
    /// double-buffer half (u32 elements).
    #[allow(clippy::too_many_arguments)]
    pub fn spec_prep(
        &self,
        sampled: &CudaSlice<u32>,
        drafts: &CudaSlice<u32>,
        meta: &mut CudaSlice<u32>,
        pos: &mut CudaSlice<u32>,
        strip: &mut CudaSlice<u32>,
        strip_off: usize,
        m_tok: &mut CudaSlice<u32>,
        m_pos: &mut CudaSlice<u32>,
        m_attn: &mut CudaSlice<u32>,
        n: usize,
        rr: usize,
        stride: usize,
        hold2: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .spec_prep
            .ok_or(GpuError::MissingOp("spec_prep"))?;
        let (sp, _g1) = sampled.device_ptr(&self.stream);
        let (dp, _g2) = drafts.device_ptr(&self.stream);
        let (mp, _g3) = meta.device_ptr_mut(&self.stream);
        let (pp, _g4) = pos.device_ptr_mut(&self.stream);
        let (op, _g5) = strip.device_ptr_mut(&self.stream);
        let (tp, _g6) = m_tok.device_ptr_mut(&self.stream);
        let (rp, _g7) = m_pos.device_ptr_mut(&self.stream);
        let (ap, _g8) = m_attn.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; strip half holds n*stride u32 at the offset
        check(unsafe {
            f(
                sp as *const _,
                dp as *const _,
                mp as *mut _,
                pp as *mut _,
                (op as usize + strip_off * 4) as *mut _,
                tp as *mut _,
                rp as *mut _,
                ap as *mut _,
                n as u32,
                rr as u32,
                stride as u32,
                hold2 as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Rung B2: gather the accepted-final hiddens into the chain's h input.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_hgather(
        &self,
        normed: &CudaSlice<f32>,
        strip: &CudaSlice<u32>,
        strip_off: usize,
        meta: &CudaSlice<u32>,
        h: &mut CudaSlice<f32>,
        n: usize,
        n_main: usize,
        stride: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .spec_hgather
            .ok_or(GpuError::MissingOp("spec_hgather"))?;
        let (np, _g1) = normed.device_ptr(&self.stream);
        let (sp, _g2) = strip.device_ptr(&self.stream);
        let (mp, _g3) = meta.device_ptr(&self.stream);
        let (hp, _g4) = h.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; h holds rr rows of n_main f32
        check(unsafe {
            f(
                np as *const _,
                (sp as usize + strip_off * 4) as *const _,
                mp as *const _,
                hp as *mut _,
                n as u32,
                n_main as u32,
                stride as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_spec_pipe(&self) -> bool {
        self.kernels.spec_prep.is_some() && self.kernels.spec_hgather.is_some()
    }

    /// Drafter xh stitch - `xh[i] = [emb[i] | h[i]]`,
    /// one launch for the whole batch (bit-identical to the per-row
    /// copy_region pair it replaces).
    pub fn spec_xh_stitch(
        &self,
        emb: &CudaSlice<f32>,
        h: &CudaSlice<f32>,
        xh: &mut CudaSlice<f32>,
        r: usize,
        n_main: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .spec_xh_stitch
            .ok_or(GpuError::MissingOp("spec_xh_stitch"))?;
        let (ep, _g1) = emb.device_ptr(&self.stream);
        let (hp, _g2) = h.device_ptr(&self.stream);
        let (xp, _g3) = xh.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; emb/h hold r rows n_main wide, xh r rows 2*n_main
        check(unsafe {
            f(
                ep as *const _,
                hp as *const _,
                xp as *mut _,
                r as u32,
                n_main as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_spec_xh_stitch(&self) -> bool {
        self.kernels.spec_xh_stitch.is_some()
    }

    /// Host-indexed f32 row gather: `dst[i] = src[idx[i]]` - replaces
    /// per-row copy_region loops whose indices the host already knows.
    pub fn hrow_gather(
        &self,
        src: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        dst: &mut CudaSlice<f32>,
        n: usize,
        n_main: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .hrow_gather
            .ok_or(GpuError::MissingOp("hrow_gather"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (ip, _g2) = idx.device_ptr(&self.stream);
        let (dp, _g3) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; dst holds n rows of n_main f32, idx n u32
        check(unsafe {
            f(
                sp as *const _,
                ip as *const _,
                dp as *mut _,
                n as u32,
                n_main as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_hrow_gather(&self) -> bool {
        self.kernels.hrow_gather.is_some()
    }
}
