//! K-quant (Q4K..Q6K/IQ4) GEMM, GEMV and MoE family.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;

/// Exact host transcode of raw Q4_0 blocks (18 B: f16 d + 16 nibble bytes)
/// into raw Q8_0 blocks (34 B: f16 d + 32 int8). Same 32-weight granularity,
/// same scale, int8 quant = nibble - 8 (range -8..7), so the dequant of every
/// weight is bit-identical - Q4_0 is literally a compressed Q8_0. Host-side
/// load-time work on the established `repack_q8_blocks` seam (the DFlash
/// drafter's bf16 planes quantize on host the same way).
pub fn q40_to_q8_blocks(bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len() / 18;
    let mut out = Vec::with_capacity(n * 34);
    for b in 0..n {
        let blk = &bytes[b * 18..b * 18 + 18];
        out.extend_from_slice(&blk[0..2]); // f16 d, unchanged
        for k in 0..16 {
            out.push(((blk[2 + k] & 0xF) as i8 - 8) as u8);
        }
        for k in 0..16 {
            out.push(((blk[2 + k] >> 4) as i8 - 8) as u8);
        }
    }
    out
}

impl GpuExecutor {
    /// True when the pack carries the k-quant family (dequant/repack/gemv/
    /// gather + the repacked-stream dequant the f32 fallback rides + the
    /// stage-2 W4A8 batch pair).
    pub fn has_kquant(&self) -> bool {
        self.kernels.kquant_dequant.is_some()
            && self.kernels.kquant_repack.is_some()
            && self.kernels.kquant_gemv.is_some()
            && self.kernels.kquant_gather.is_some()
            && self.kernels.kquant_dequant_rp.is_some()
            && self.kernels.mmq_sums.is_some()
            && self.kernels.kquant_gemm_w4a8.is_some()
            && self.kernels.q8_sums_strided.is_some()
            && self.kernels.kquant_gemm_dp4a.is_some()
    }

    /// True when the pack's kquant family serves Q4_0 (the appended
    /// capability slot): the dtype rides the EXISTING entry points, so the
    /// presence of those slots alone cannot answer for an older pack.
    pub fn has_kquant_q40(&self) -> bool {
        self.kernels.kquant_q40.is_some()
    }

    /// True when the pack serves the i-quant family (IQ1/IQ2/IQ3, IQ4_NL)
    /// on the k-quant streams - repack, dequant and the token-batched MoE
    /// pair. Capability marker slot 577.
    pub fn has_kquant_iq(&self) -> bool {
        self.kernels.kquant_iq.is_some()
    }

    /// The pack serves the i-quant family on the DENSE lanes too (slot 578).
    pub fn has_kquant_iq_dense(&self) -> bool {
        self.kernels.kquant_iq_dense.is_some()
    }

    /// The >64-row W4A8 tile GEMM (`kquant_gemm_w4a8_pipe2`) serves the
    /// i-quant family + Q2_K / Q3_K / IQ4_NL (slot 585): prefill on a dense
    /// i-quant plane rides the tile instead of re-reading the plane per token.
    pub fn has_kquant_iq_tile(&self) -> bool {
        self.kernels.kquant_iq_tile.is_some()
    }

    /// True when the pack carries the K-split W4A8 mma rung (appended after
    /// the base k-quant family - older packs fall back to the dp4a z-tiling).
    pub fn has_kquant_mma_ks(&self) -> bool {
        self.kernels.kquant_gemm_mma_ks.is_some()
    }

    /// True when the pack carries the pipelined (cp.async-overlapped) W4A8
    /// GEMM - older packs fall back to `kquant_gemm_w4a8`'s synchronous load
    /// (that kernel was 79.6% of granite-30b's prefill GPU time
    /// on sm_120a, register-bound at `__launch_bounds__(256,1)` with no
    /// latency-hiding rung, unlike Q8_0's `_hi`/`_pipe` siblings).
    pub fn has_kquant_gemm_w4a8_pipe(&self) -> bool {
        self.kernels.kquant_gemm_w4a8_pipe.is_some()
    }

    /// True when the pack carries `kquant_gemm_w4a8_pipe`'s genuinely
    /// double-buffered sibling - a real 2-deep raw byte ring (half-width
    /// tile_x to afford the second copy) so the next super-block's load
    /// overlaps the CURRENT one's entire build+compute phase, not just its
    /// compute phase like the single-buffer pipe kernel. Stays
    /// `__launch_bounds__(256,1)`: a 2-blocks/SM tile hit its register
    /// target (REG:128) but sm_120's SM shared-memory budget
    /// (102,400 B - barely above its own 101,376 B single-block opt-in max)
    /// blocked occupancy from actually rising regardless, so it was
    /// reverted (llama.cpp's own k-quant kernel doesn't pipeline at all and
    /// targets the same occupancy=1, confirmed by reading `ggml-cuda/mmq.cuh`
    /// directly - so occupancy=1 is the settled floor on both engines here).
    pub fn has_kquant_gemm_w4a8_pipe2(&self) -> bool {
        self.kernels.kquant_gemm_w4a8_pipe2.is_some()
    }

    /// Load a matmul weight quantized-resident with per-TENSOR dispatch:
    /// Q8_0 -> the repacked-Q8 streams, k-quant family -> the repacked
    /// k-quant streams. Anything else is a load error (dequant-to-f32 for a
    /// big matmul weight would silently blow the VRAM story).
    pub fn load_quantw(&self, map: &MappedGguf, name: &str) -> Result<QuantW, GpuError> {
        let (info, _) = map.tensor_bytes(name)?;
        match info.ggml_type {
            GgmlType::Q8_0 => Ok(QuantW::Q8(self.repack_q8(map, name)?)),
            // Q4_0 (the QAT lineage's native format): kquant-resident at 4.5
            // bpw when the pack serves the dtype and the row length packs
            // into whole 256-weight super-blocks; otherwise the exact Q8_0
            // transcode (same f16 scale per 32-block, int8 = nibble-8, so
            // dequant is bit-identical) - correctness never degrades, only
            // the resident bytes (~2x) on the fallback.
            GgmlType::Q4_0 => {
                let (info, bytes) = map.tensor_bytes(name)?;
                let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
                if self.has_kquant() && self.has_kquant_q40() && dims[0].is_multiple_of(256) {
                    Ok(QuantW::Kq(self.repack_kquant(map, name)?))
                } else {
                    let q8 = q40_to_q8_blocks(bytes);
                    Ok(QuantW::Q8(self.repack_q8_blocks(&q8, dims)?))
                }
            }
            // the i-quant family on the dense lanes needs both markers: the
            // repack/dequant (577) and the dense entry points (578)
            ty if kq_is_iq(ty) => {
                if !self.has_kquant_iq() || !self.has_kquant_iq_dense() {
                    return Err(GpuError::Unsupported(format!(
                        "{name} is {ty:?} but the kernel pack has no dense i-quant lanes \
                         (slots 577/578) - rebuild packs/cuda"
                    )));
                }
                self.note_dense_iq();
                let kq = self.repack_kquant(map, name)?;
                if !kq.dims[0].is_multiple_of(256) {
                    self.note_dense_iq_flat();
                }
                Ok(QuantW::Kq(kq))
            }
            ty if kq_params(ty).is_some() => {
                if !self.has_kquant() {
                    return Err(GpuError::NoKernel {
                        name: name.to_owned(),
                        ty,
                    });
                }
                Ok(QuantW::Kq(self.repack_kquant(map, name)?))
            }
            ty => Err(GpuError::NoKernel {
                name: name.to_owned(),
                ty,
            }),
        }
    }

    /// Repack a tensor k-quant-resident when it is a supported k-quant type;
    /// `None` when it isn't (caller picks its fallback - e.g. Q8 requant for
    /// MoE expert seats). Checks the type before uploading.
    pub fn try_repack_kquant(
        &self,
        map: &MappedGguf,
        name: &str,
    ) -> Result<Option<RepackedKQ>, GpuError> {
        let (info, _) = map.tensor_bytes(name)?;
        if kq_params(info.ggml_type).is_none() {
            return Ok(None);
        }
        if kq_is_iq(info.ggml_type) && !self.has_kquant_iq() {
            return Err(GpuError::NoKernel {
                name: name.to_owned(),
                ty: info.ggml_type,
            });
        }
        Ok(Some(self.repack_kquant(map, name)?))
    }

    /// Upload a Q4_K/Q5_K/Q6_K weight and repack it into the aligned data +
    /// scale-record streams the fused k-quant GEMV reads. The staged upload
    /// is freed on return - the tensor stays 4/5/6-bit resident.
    pub fn repack_kquant(&self, map: &MappedGguf, name: &str) -> Result<RepackedKQ, GpuError> {
        let (info, bytes) = map.tensor_bytes(name)?;
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        self.repack_kquant_raw(bytes, dims, info.ggml_type, name)
    }

    /// The repack above, off RAW block bytes the caller assembled itself -
    /// the seam for weights that are not a whole GGUF tensor. Its only user
    /// today is the DFlash fusion `fc` plane, which is one [n_taps*embd, embd]
    /// tensor the engine wants as n_taps separate [embd, embd] planes so each
    /// target-layer tap can be consumed the moment the walk produces it
    /// (no [rows, n_taps*embd] concat staging). Valid for exactly the reason
    /// `repack_kquant_concat` is valid in the other direction: k-quant
    /// superblocks are position-independent, so any slice of a row's block
    /// stream that starts and ends on a superblock boundary is itself a
    /// well-formed weight. `bytes` must therefore be whole superblocks and
    /// `dims.product() / 256` of them - the caller's gather does the striding.
    pub fn repack_kquant_raw(
        &self,
        bytes: &[u8],
        dims: Vec<usize>,
        ty: GgmlType,
        what: &str,
    ) -> Result<RepackedKQ, GpuError> {
        let (raw_id, raw_b, data_b) = kq_params(ty).ok_or(GpuError::NoKernel {
            name: what.to_owned(),
            ty,
        })?;
        let f = self
            .kernels
            .kquant_repack
            .ok_or(GpuError::MissingOp("kquant_repack"))?;
        // Rows whose in_dim is not a whole number of 256-weight superblocks
        // (Qwen3.8-Flash-Next's expert `down`: 640 = 2.5 superblocks) exist
        // only in the 32-block types - the 256-block families cannot encode
        // such a row, which is why llama.cpp falls back to IQ4_NL there. An
        // IQ4_NL row is its in_dim/32 blocks laid flat (16 payload bytes + one
        // f16 d each), so no padding is needed: the repack runs over the
        // tensor's whole block stream 8 blocks at a time and the kernels
        // stride rows by in_dim/32 blocks (`pd_kq_row_datab`). The stream is
        // required to be a whole number of superblocks per expert so the
        // slot cache's per-expert byte ranges stay exact.
        let in_dim = dims[0];
        let n_super = if in_dim.is_multiple_of(256) {
            dims.iter().product::<usize>() / 256
        } else {
            if ty != GgmlType::Iq4Nl || !in_dim.is_multiple_of(32) {
                return Err(GpuError::Driver(format!(
                    "kquant repack {what}: in_dim {in_dim} is not superblock-aligned and {ty:?} has no flat 32-block row layout (IQ4_NL only)"
                )));
            }
            let per_row = in_dim / 32;
            let per_expert = per_row * dims.get(1).copied().unwrap_or(1);
            if !per_expert.is_multiple_of(8) {
                return Err(GpuError::Driver(format!(
                    "kquant repack {what}: {per_expert} blocks per [{in_dim} x {}] plane is not a whole number of superblocks",
                    dims.get(1).copied().unwrap_or(1)
                )));
            }
            dims.iter().product::<usize>() / 32 / 8
        };
        if bytes.len() != n_super * raw_b {
            return Err(GpuError::Driver(format!(
                "kquant repack {what}: {} raw bytes for {n_super} superblocks (want {})",
                bytes.len(),
                n_super * raw_b
            )));
        }
        let mut data = self.alloc_u8(n_super * data_b)?;
        let mut scales = self.alloc_u8(n_super * kq_scb(ty))?;
        self.with_staged_raw(bytes, |sp| {
            let (dp, _g2) = data.device_ptr_mut(&self.stream);
            let (scp, _g3) = scales.device_ptr_mut(&self.stream);
            // SAFETY: pack ABI v1 contract; pointers + stream live across the call
            check(unsafe {
                f(
                    sp as *const _,
                    dp as *mut _,
                    scp as *mut _,
                    n_super as u64,
                    raw_id,
                    self.stream_ptr(),
                )
            })
        })?;
        Ok(RepackedKQ {
            data,
            scales,
            dims,
            ty,
        })
    }

    /// Split one k-quant tensor's in dim into `n_bands` equal strips and
    /// repack each as its own weight - the inverse of `repack_kquant_concat`,
    /// and the DFlash fusion `fc`'s loader (see `repack_kquant_raw`). Each
    /// band must land on superblock boundaries, which is what makes the
    /// strips well-formed weights rather than byte slices.
    pub fn repack_kquant_bands(
        &self,
        map: &MappedGguf,
        name: &str,
        n_bands: usize,
    ) -> Result<Vec<RepackedKQ>, GpuError> {
        let (info, bytes) = map.tensor_bytes(name)?;
        let (_, raw_b, _) = kq_params(info.ggml_type).ok_or(GpuError::NoKernel {
            name: name.to_owned(),
            ty: info.ggml_type,
        })?;
        let (in_dim, out_dim) = (info.dims[0] as usize, info.dims[1] as usize);
        if n_bands == 0 || in_dim % n_bands != 0 || !(in_dim / n_bands).is_multiple_of(256) {
            return Err(GpuError::Driver(format!(
                "kquant bands {name}: in_dim {in_dim} does not split into {n_bands} \
                 superblock-aligned strips"
            )));
        }
        let band_in = in_dim / n_bands;
        let (row_sb, band_sb) = (in_dim / 256, band_in / 256);
        let mut out = Vec::with_capacity(n_bands);
        let mut raw = vec![0u8; out_dim * band_sb * raw_b];
        for b in 0..n_bands {
            for o in 0..out_dim {
                let src = (o * row_sb + b * band_sb) * raw_b;
                let dst = o * band_sb * raw_b;
                raw[dst..dst + band_sb * raw_b].copy_from_slice(&bytes[src..src + band_sb * raw_b]);
            }
            out.push(self.repack_kquant_raw(&raw, vec![band_in, out_dim], info.ggml_type, name)?);
        }
        Ok(out)
    }

    /// Repack SEVERAL same-type k-quant tensors into one resident weight,
    /// concatenated along the out dimension (dims[1]) - the merged-projection
    /// GEMV lane: one launch reads [t0 | t1 | ...] output rows where separate
    /// small GEMVs each paid their own fixed cost. Valid because k-quant
    /// superblocks are position-independent and rows are whole superblocks
    /// (in_dim % 256 == 0): the fused matrix's block stream is the tensors'
    /// streams back to back, so each source repacks at its running offset.
    pub fn repack_kquant_concat(
        &self,
        map: &MappedGguf,
        names: &[&str],
    ) -> Result<RepackedKQ, GpuError> {
        assert!(!names.is_empty(), "concat of zero tensors");
        let mut ty: Option<GgmlType> = None;
        let mut in_dim = 0usize;
        let mut out_dim = 0usize;
        let mut total_super = 0usize;
        for name in names {
            let (info, _) = map.tensor_bytes(name)?;
            match ty {
                None => {
                    ty = Some(info.ggml_type);
                    in_dim = info.dims[0] as usize;
                }
                Some(t) if t != info.ggml_type || info.dims[0] as usize != in_dim => {
                    return Err(GpuError::Driver(format!(
                        "kquant concat: {name} breaks type/in_dim agreement"
                    )));
                }
                Some(_) => {}
            }
            out_dim += info.dims[1] as usize;
            total_super += info.dims.iter().product::<u64>() as usize / 256;
        }
        let ty = ty.expect("nonempty names");
        let (raw_id, _, data_b) = kq_params(ty).ok_or(GpuError::NoKernel {
            name: names[0].to_owned(),
            ty,
        })?;
        if !in_dim.is_multiple_of(256) {
            return Err(GpuError::Driver(format!(
                "kquant concat: in_dim {in_dim} not superblock-aligned"
            )));
        }
        let f = self
            .kernels
            .kquant_repack
            .ok_or(GpuError::MissingOp("kquant_repack"))?;
        let scb = kq_scb(ty);
        let mut data = self.alloc_u8(total_super * data_b)?;
        let mut scales = self.alloc_u8(total_super * scb)?;
        let mut done = 0usize;
        for name in names {
            let (info, bytes) = map.tensor_bytes(name)?;
            let n_super = info.dims.iter().product::<u64>() as usize / 256;
            self.with_staged_raw(bytes, |sp| {
                let (dp, _g2) = data.device_ptr_mut(&self.stream);
                let (scp, _g3) = scales.device_ptr_mut(&self.stream);
                // SAFETY: pack ABI v1 contract; dst pointers offset to this
                // tensor's superblock range within the fused allocation
                check(unsafe {
                    f(
                        sp as *const _,
                        (dp + (done * data_b) as u64) as *mut _,
                        (scp + (done * scb) as u64) as *mut _,
                        n_super as u64,
                        raw_id,
                        self.stream_ptr(),
                    )
                })
            })?;
            done += n_super;
        }
        Ok(RepackedKQ {
            data,
            scales,
            dims: vec![in_dim, out_dim],
            ty,
        })
    }

    /// Exact fused decode GEMV over a repacked k-quant weight: y[out] = W·x.
    pub fn kquant_gemv(
        &self,
        w: &RepackedKQ,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemv
            .ok_or(GpuError::MissingOp("kquant_gemv"))?;
        let (raw_id, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, out_dim) = (w.dims[0] as u32, w.dims[1] as u32);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scales.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim,
                out_dim,
                raw_id,
                self.stream_ptr(),
            )
        })
    }

    /// Embedding row-gather from a repacked k-quant table (rows = dims[1]).
    pub fn kquant_gather(
        &self,
        w: &RepackedKQ,
        tokens: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n_tokens: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gather
            .ok_or(GpuError::MissingOp("kquant_gather"))?;
        let (raw_id, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scales.device_ptr(&self.stream);
        let (tp, _g3) = tokens.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                tp as *const _,
                op as *mut _,
                embd as u32,
                n_tokens as u32,
                raw_id,
                self.stream_ptr(),
            )
        })
    }

    /// Dequant a whole repacked k-quant weight into an f32 buffer (the
    /// batch/prefill interim: dequant to scratch -> exact f32 GEMM; the
    /// stage-2 W4A8 int8-MMA replaces this per-use round trip). `dst` must
    /// hold at least product(dims) elements.
    pub fn kquant_dequant_rp(
        &self,
        w: &RepackedKQ,
        dst: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_dequant_rp
            .ok_or(GpuError::MissingOp("kquant_dequant_rp"))?;
        let (raw_id, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        // off the stream, not the dims: rows padded to whole superblocks
        // (partial-superblock in_dim, see `repack_kquant_raw`) carry more
        // superblocks than `dims.product() / 256`
        let (_, _, data_b) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let n_super = w.data.len() / data_b;
        debug_assert!(dst.len() >= n_super * 256);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scales.device_ptr(&self.stream);
        let (op, _g3) = dst.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                op as *mut _,
                n_super as u64,
                raw_id,
                self.stream_ptr(),
            )
        })
    }

    /// Per-32-block activation sums off the mmq int8 layout: sums[chunk]
    /// [col_pad][b] = scl_b * sum(block b). The W4A8 min-term operand for
    /// Q4_K/Q5_K; computed from the already-quantized `yq` so every existing
    /// quantize kernel (incl. the fused variants) stays untouched. `sums`
    /// must hold ceil(in_dim/128) * batch_pad * 4 f32.
    pub fn mmq_sums(
        &self,
        yq: &CudaSlice<u8>,
        sums: &mut CudaSlice<f32>,
        in_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mmq_sums
            .ok_or(GpuError::MissingOp("mmq_sums"))?;
        let batch_pad = batch.div_ceil(128) * 128;
        debug_assert!(sums.len() >= in_dim.div_ceil(128) * batch_pad * 4);
        let (yp, _g1) = yq.device_ptr(&self.stream);
        let (sp, _g2) = sums.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                yp as *const _,
                sp as *mut _,
                in_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Stage-2 W4A8 GEMM off the repacked k-quant streams: y[batch, out] =
    /// W · x with x in the mmq int8 layout (`quantize_q8_mmq` class) and the
    /// weights unpacked to s8 in-kernel - 4-6.6 bpw DRAM traffic, int8 tensor
    /// cores. `xsums` (from `mmq_sums`) is required for Q4_K/Q5_K (min term)
    /// and ignored for Q6_K/IQ4_XS.
    pub fn kquant_gemm_w4a8(
        &self,
        w: &RepackedKQ,
        yq: &CudaSlice<u8>,
        xsums: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemm_w4a8
            .ok_or(GpuError::MissingOp("kquant_gemm_w4a8"))?;
        let (raw_id, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        debug_assert!(y.len() >= out_dim * batch);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scales.device_ptr(&self.stream);
        let (yqp, _g3) = yq.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                yqp as *const _,
                sp,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                raw_id,
                self.stream_ptr(),
            )
        })
    }

    /// Pipelined sibling of [`Self::kquant_gemm_w4a8`] - identical signature,
    /// contract and numerics; the raw weight+scale bytes ride cp.async into a
    /// shared buffer (ports `kquant_gemm_mma_ks`'s already-proven technique
    /// onto the >64-batch 128x128 tile), with the next super-block's fetch
    /// overlapping this one's MMA compute, instead of a synchronous global load.
    pub fn kquant_gemm_w4a8_pipe(
        &self,
        w: &RepackedKQ,
        yq: &CudaSlice<u8>,
        xsums: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemm_w4a8_pipe
            .ok_or(GpuError::MissingOp("kquant_gemm_w4a8_pipe"))?;
        let (raw_id, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        debug_assert!(y.len() >= out_dim * batch);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scales.device_ptr(&self.stream);
        let (yqp, _g3) = yq.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                yqp as *const _,
                sp,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                raw_id,
                self.stream_ptr(),
            )
        })
    }

    /// `kquant_gemm_w4a8_pipe`'s genuinely-double-buffered sibling -
    /// identical signature, contract and numerics; half-width tile_x (rebuilt
    /// fresh per half instead of once per full super-block) frees enough
    /// shared memory for a real 2-deep raw byte ring, so the next
    /// super-block's load overlaps this one's entire build+compute phase
    /// instead of just its compute phase. `__launch_bounds__(256,1)` - see
    /// [`Self::has_kquant_gemm_w4a8_pipe2`] for why occupancy stays at 1.
    pub fn kquant_gemm_w4a8_pipe2(
        &self,
        w: &RepackedKQ,
        yq: &CudaSlice<u8>,
        xsums: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        self.kquant_gemm_w4a8_pipe2_rows(w, yq, xsums, y, 0, batch)
    }

    /// [`Self::kquant_gemm_w4a8_pipe2`] writing from row `y_row0` of `y` -
    /// a chunk-sized `yq` landing in its rows of a wider output.
    pub fn kquant_gemm_w4a8_pipe2_rows(
        &self,
        w: &RepackedKQ,
        yq: &CudaSlice<u8>,
        xsums: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        y_row0: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemm_w4a8_pipe2
            .ok_or(GpuError::MissingOp("kquant_gemm_w4a8_pipe2"))?;
        let (raw_id, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        debug_assert!(y.len() >= out_dim * (y_row0 + batch));
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scales.device_ptr(&self.stream);
        let (yqp, _g3) = yq.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                yqp as *const _,
                sp,
                (yp + (y_row0 * out_dim * 4) as u64) as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                raw_id,
                self.stream_ptr(),
            )
        })
    }

    /// Per-16 activation sums off the STRIDED int8 layout (`quantize_q8`):
    /// sums[col][w16] = raw int sum as f32. The dp4a decode ladder's Q4/Q5
    /// min-term operand. `sums` must hold batch * in_dim/16 f32.
    pub fn q8_sums_strided(
        &self,
        xq: &CudaSlice<i8>,
        sums: &mut CudaSlice<f32>,
        in_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_sums_strided
            .ok_or(GpuError::MissingOp("q8_sums_strided"))?;
        debug_assert!(sums.len() >= batch * in_dim / 16);
        let (xp, _g1) = xq.device_ptr(&self.stream);
        let (sp, _g2) = sums.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                xp as *const _,
                sp as *mut _,
                in_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// W4A8 dp4a batch GEMM off the repacked k-quant streams - the decode-batch
    /// shape (few columns, weight-bandwidth-bound). Activations in the STRIDED
    /// int8 layout (`quantize_q8`: xq row-major + xs per-32 f32). `xsums`
    /// (from `q8_sums_strided`) required for Q4_K/Q5_K, ignored otherwise.
    pub fn kquant_gemm_dp4a(
        &self,
        w: &RepackedKQ,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsums: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemm_dp4a
            .ok_or(GpuError::MissingOp("kquant_gemm_dp4a"))?;
        let (raw_id, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        debug_assert!(y.len() >= out_dim * batch);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scales.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                sp,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                raw_id,
                self.stream_ptr(),
            )
        })
    }

    /// K-split W4A8 mma GEMM off the repacked k-quant streams - the 17..64
    /// decode-batch rung (one weight pass for the whole batch; grid.z K-ranges
    /// + fixed-order combine fill the die like `q8_0_gemm_mma_ks`). Same
    ///   strided activation buffers as the dp4a rung; `part` needs >= 8 * out *
    ///   batch f32 (the ks fixup scratch already is).
    pub fn kquant_gemm_mma_ks(
        &self,
        w: &RepackedKQ,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsums: Option<&CudaSlice<f32>>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemm_mma_ks
            .ok_or(GpuError::MissingOp("kquant_gemm_mma_ks"))?;
        let (raw_id, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        debug_assert!(y.len() >= out_dim * batch);
        debug_assert!(part.len() >= 8 * out_dim * batch);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scales.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                sp,
                pp as *mut _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                raw_id,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the k-quant MoE expert pair (token-batched
    /// decode class).
    pub fn has_kquant_moe(&self) -> bool {
        self.kernels.kquant_moe_gate_up.is_some() && self.kernels.kquant_moe_down.is_some()
    }

    /// k-quant routed-expert gate+up+SwiGLU, token-batched. gate/up may be
    /// different k-quant types (per-tensor dispatch); `xsums` (per-16 int8
    /// sums off `xq`, pd_q8_sums_strided) is required when either is
    /// Q4_K/Q5_K (the mu term). Expert row (e, o) sits at e*ff + o in the
    /// repacked streams; dims are the GGUF 3D [in, ff, n_expert].
    #[allow(clippy::too_many_arguments)]
    pub fn kquant_moe_gate_up(
        &self,
        gate: &RepackedKQ,
        up: &RepackedKQ,
        idx: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsums: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_moe_gate_up
            .ok_or(GpuError::MissingOp("kquant_moe_gate_up"))?;
        let (gid, _, _) = kq_params(gate.ty).expect("RepackedKQ holds a k-quant type");
        let (uid, _, _) = kq_params(up.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        debug_assert_eq!(up.dims[0], in_dim);
        debug_assert_eq!(up.dims[1], ff);
        debug_assert!(out.len() >= batch * n_active * ff);
        let (gdp, _g1) = gate.data.device_ptr(&self.stream);
        let (gsp, _g2) = gate.scales.device_ptr(&self.stream);
        let (udp, _g3) = up.data.device_ptr(&self.stream);
        let (usp, _g4) = up.scales.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (xqp, _g6) = xq.device_ptr(&self.stream);
        let (xsp, _g7) = xs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                udp as *const _,
                usp as *const _,
                ip as *const _,
                xqp as *const _,
                xsp as *const _,
                sp,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                batch as u32,
                gid,
                uid,
                self.stream_ptr(),
            )
        })
    }

    /// k-quant routed-expert down + weighted combine (token-batched). `fsums`
    /// (per-16 sums off `fq`, batch*n_active rows of ff) required when down
    /// is Q4_K/Q5_K.
    #[allow(clippy::too_many_arguments)]
    pub fn kquant_moe_down(
        &self,
        down: &RepackedKQ,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        fsums: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_moe_down
            .ok_or(GpuError::MissingOp("kquant_moe_down"))?;
        let (did, _, _) = kq_params(down.ty).expect("RepackedKQ holds a k-quant type");
        let (ff, embd) = (down.dims[0], down.dims[1]);
        debug_assert!(out.len() >= batch * embd);
        let (ddp, _g1) = down.data.device_ptr(&self.stream);
        let (dsp, _g2) = down.scales.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (twp, _g4) = topk_w.device_ptr(&self.stream);
        let (fqp, _g5) = fq.device_ptr(&self.stream);
        let (fsp, _g6) = fs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match fsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                ip as *const _,
                twp as *const _,
                fqp as *const _,
                fsp as *const _,
                sp,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                batch as u32,
                did,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the sorted k-quant MoE mma pair (the
    /// prefill/serving class for k-quant expert seats).
    pub fn has_kquant_moe_mma(&self) -> bool {
        self.kernels.kquant_moe_gate_up_mma.is_some() && self.kernels.kquant_moe_down_mma.is_some()
    }

    /// Sorted k-quant MoE gate+up+SwiGLU mma (BM=32 moe_align layout). The
    /// pair shares one dtype (single-DT kernel template) - the caller keeps
    /// mixed-type pairs on the token-batched class. Writes fq/fs
    /// SORTED-CONTIGUOUS (max_blocks*32 rows of ff); `xsums` (per-16 sums off
    /// `xq`) required when the type is Q4_K/Q5_K.
    #[allow(clippy::too_many_arguments)]
    /// `kquant_moe_gate_up` over the expert-major prefill's compacted pair
    /// list (slot 583): the wave's in-wave pairs and their device count; the
    /// grid strides the list, so the launch costs the wave's pairs only.
    #[allow(clippy::too_many_arguments)]
    pub fn kquant_moe_gate_up_list(
        &self,
        gate: &RepackedKQ,
        up: &RepackedKQ,
        idx: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsums: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
        pairs: &CudaSlice<u32>,
        n_pairs: &CudaSlice<u32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_moe_gate_up_list
            .ok_or(GpuError::MissingOp("kquant_moe_gate_up_list"))?;
        let (gid, _, _) = kq_params(gate.ty).expect("RepackedKQ holds a k-quant type");
        let (uid, _, _) = kq_params(up.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        debug_assert!(out.len() >= batch * n_active * ff);
        let (gdp, _g1) = gate.data.device_ptr(&self.stream);
        let (gsp, _g2) = gate.scales.device_ptr(&self.stream);
        let (udp, _g3) = up.data.device_ptr(&self.stream);
        let (usp, _g4) = up.scales.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (xqp, _g6) = xq.device_ptr(&self.stream);
        let (xsp, _g7) = xs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        let (pp, _g9) = pairs.device_ptr(&self.stream);
        let (npp, _g10) = n_pairs.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract (slot 583); pointers + stream live across the call
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                udp as *const _,
                usp as *const _,
                ip as *const _,
                xqp as *const _,
                xsp as *const _,
                sp,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                batch as u32,
                gid,
                uid,
                pp as *const _,
                npp as *const _,
                self.stream_ptr(),
            )
        })
    }

    /// `kquant_moe_down` over the wave's compacted token list (slot 584),
    /// ACCUMULATING into `out` - zero it before the first wave.
    #[allow(clippy::too_many_arguments)]
    pub fn kquant_moe_down_list(
        &self,
        down: &RepackedKQ,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        fsums: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
        rows: &CudaSlice<u32>,
        n_rows: &CudaSlice<u32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_moe_down_list
            .ok_or(GpuError::MissingOp("kquant_moe_down_list"))?;
        let (did, _, _) = kq_params(down.ty).expect("RepackedKQ holds a k-quant type");
        let (ff, embd) = (down.dims[0], down.dims[1]);
        debug_assert!(out.len() >= batch * embd);
        let (ddp, _g1) = down.data.device_ptr(&self.stream);
        let (dsp, _g2) = down.scales.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (twp, _g4) = topk_w.device_ptr(&self.stream);
        let (fqp, _g5) = fq.device_ptr(&self.stream);
        let (fsp, _g6) = fs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match fsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (rp, _g8) = rows.device_ptr(&self.stream);
        let (nrp, _g9) = n_rows.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract (slot 584); pointers + stream live across the call
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                ip as *const _,
                twp as *const _,
                fqp as *const _,
                fsp as *const _,
                sp,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                batch as u32,
                did,
                rp as *const _,
                nrp as *const _,
                self.stream_ptr(),
            )
        })
    }

    pub fn kquant_moe_gate_up_mma(
        &self,
        gate: &RepackedKQ,
        up: &RepackedKQ,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsums: Option<&CudaSlice<f32>>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_moe_gate_up_mma
            .ok_or(GpuError::MissingOp("kquant_moe_gate_up_mma"))?;
        assert_eq!(gate.ty, up.ty, "sorted kq pair shares one dtype");
        let (did, _, _) = kq_params(gate.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        debug_assert_eq!(up.dims[0], in_dim);
        debug_assert_eq!(up.dims[1], ff);
        debug_assert!(fq.len() >= max_blocks * 32 * ff);
        debug_assert!(fs.len() >= max_blocks * 32 * ff / 32);
        let (gdp, _g1) = gate.data.device_ptr(&self.stream);
        let (gsp, _g2) = gate.scales.device_ptr(&self.stream);
        let (udp, _g3) = up.data.device_ptr(&self.stream);
        let (usp, _g4) = up.scales.device_ptr(&self.stream);
        let (srp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bep, _g6) = block_expert.device_ptr(&self.stream);
        let (xqp, _g7) = xq.device_ptr(&self.stream);
        let (xsp, _g8) = xs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (fqp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fsp, _g10) = fs.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                udp as *const _,
                usp as *const _,
                srp as *const _,
                bep as *const _,
                xqp as *const _,
                xsp as *const _,
                sp,
                fqp as *mut _,
                fsp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                did,
                self.stream_ptr(),
            )
        })
    }

    /// Sorted k-quant MoE down mma: consumes the gate_up pair's
    /// sorted-contiguous fq/fs, writes deterministic (token, slot) weighted
    /// partials for `moe_slot_combine`. `fsums` (per-16 sums off the sorted
    /// fq rows, max_blocks*32 rows of ff) required when down is Q4_K/Q5_K.
    #[allow(clippy::too_many_arguments)]
    pub fn kquant_moe_down_mma(
        &self,
        down: &RepackedKQ,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        fsums: Option<&CudaSlice<f32>>,
        part: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_moe_down_mma
            .ok_or(GpuError::MissingOp("kquant_moe_down_mma"))?;
        let (did, _, _) = kq_params(down.ty).expect("RepackedKQ holds a k-quant type");
        let (ff, embd) = (down.dims[0], down.dims[1]);
        let (ddp, _g1) = down.data.device_ptr(&self.stream);
        let (dsp, _g2) = down.scales.device_ptr(&self.stream);
        let (srp, _g3) = sorted_row.device_ptr(&self.stream);
        let (slp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bep, _g5) = block_expert.device_ptr(&self.stream);
        let (twp, _g6) = topk_w.device_ptr(&self.stream);
        let (fqp, _g7) = fq.device_ptr(&self.stream);
        let (fsp, _g8) = fs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match fsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                srp as *const _,
                slp as *const _,
                bep as *const _,
                twp as *const _,
                fqp as *const _,
                fsp as *const _,
                sp,
                pp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                did,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the W4A8 b=1 decode GEMV (+ the fused
    /// quantize+sums node the b=1 tick pairs it with).
    pub fn has_kquant_gemv_w4a8(&self) -> bool {
        self.kernels.kquant_gemv_w4a8.is_some() && self.kernels.quantize_q8_sums.is_some()
    }

    /// Fused activation quantize + per-16 int8 sums - bit-identical outputs to
    /// `quantize_q8` followed by `q8_sums_strided`, one launch.
    pub fn quantize_q8_sums(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<f32>,
        sums: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_q8_sums
            .ok_or(GpuError::MissingOp("quantize_q8_sums"))?;
        debug_assert!(q.len() >= n && scale.len() >= n / 32 && sums.len() >= n / 16);
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        let (mp, _g4) = sums.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                mp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// W4A8 b=1 decode GEMV (the mmvq-class serving default): int8-quantized
    /// activations + per-32 scales, dp4a dots off the raw k-quant streams.
    /// `xsums` (per-16 sums off `xq`) required for Q4_K/Q5_K. Same numeric
    /// class as the W4A8 batch ladder; the exact-f32 `kquant_gemv` stays the
    /// oracle path.
    pub fn kquant_gemv_w4a8(
        &self,
        w: &RepackedKQ,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsums: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemv_w4a8
            .ok_or(GpuError::MissingOp("kquant_gemv_w4a8"))?;
        let (did, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        debug_assert!(y.len() >= out_dim);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp_, _g2) = w.scales.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                sp_ as *const _,
                xqp as *const _,
                xsp as *const _,
                sp,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                did,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_kquant_gemv_w4a8_multi(&self) -> bool {
        self.kernels.kquant_gemv_w4a8_multi.is_some() && self.kernels.quantize_q8_sums.is_some()
    }

    /// One launch over 2-3 same-in_dim k-quant planes sharing the staged int8
    /// activation (granite's decode QKV / gate|up merge - the
    /// `q8_0_gemv_repacked_multi` economics on the W4A8 family). Segments may
    /// mix k-quant dtypes (Q4_K_M files pair Q4_K q/k with Q6_K v). `ssums`
    /// is forwarded only when a segment is a mu format (Q4_K/Q5_K) - callers
    /// pass the staged plane unconditionally, mirroring `gemv8_any`.
    pub fn kquant_gemv_w4a8_multi(
        &self,
        segs: &mut [(&RepackedKQ, &mut CudaSlice<f32>)],
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        ssums: &CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemv_w4a8_multi
            .ok_or(GpuError::MissingOp("kquant_gemv_w4a8_multi"))?;
        let n_segs = segs.len();
        assert!((2..=3).contains(&n_segs), "2..=3 segments");
        let in_dim = segs[0].0.dims[0];
        let mut dp = [std::ptr::null::<core::ffi::c_void>(); 3];
        let mut sp = [std::ptr::null::<core::ffi::c_void>(); 3];
        let mut yp = [std::ptr::null_mut::<core::ffi::c_void>(); 3];
        let mut rows = [0u32; 3];
        let mut dts = [0u32; 3];
        let mut needs = false;
        let mut guards = Vec::with_capacity(9);
        for (i, (w, y)) in segs.iter_mut().enumerate() {
            assert_eq!(w.dims[0], in_dim, "segments share in_dim");
            let (did, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
            debug_assert!(y.len() >= w.dims[1]);
            let (d, g1) = w.data.device_ptr(&self.stream);
            let (s, g2) = w.scales.device_ptr(&self.stream);
            let (yy, g3) = y.device_ptr_mut(&self.stream);
            dp[i] = d as *const _;
            sp[i] = s as *const _;
            yp[i] = yy as *mut _;
            rows[i] = w.dims[1] as u32;
            dts[i] = did;
            needs = needs || crate::gpu::kq_needs_sums(w.ty);
            guards.push(g1);
            guards.push(g2);
            guards.push(g3);
        }
        let (xqp, _g4) = xq.device_ptr(&self.stream);
        let (xsp, _g5) = xs.device_ptr(&self.stream);
        let (smp, _g6) = ssums.device_ptr(&self.stream);
        let sump: *const core::ffi::c_void = if needs {
            smp as *const _
        } else {
            core::ptr::null()
        };
        // SAFETY: ABI contract; unused trailing segments pass nulls/0
        check(unsafe {
            f(
                dp[0],
                sp[0],
                yp[0],
                rows[0],
                dts[0],
                dp[1],
                sp[1],
                yp[1],
                rows[1],
                dts[1],
                dp[2],
                sp[2],
                yp[2],
                rows[2],
                dts[2],
                xqp as *const _,
                xsp as *const _,
                sump,
                in_dim as u32,
                n_segs as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_kquant_gemv_w4a8_glu(&self) -> bool {
        self.kernels.kquant_gemv_w4a8_glu.is_some() && self.kernels.quantize_q8_sums.is_some()
    }

    /// Fused gate+up+SwiGLU decode GEMV: one launch computes both
    /// dots per output row over one staged activation and writes
    /// `silu(gate)*up` directly - the split multi+swiglu chain's launches,
    /// 2*n_ff intermediate and its round-trip gone. Bit-exact vs that split
    /// path (identical row walks at the multi's <4,128>, identical epilogue
    /// expression); gated by `kq_w4a8_glu_matches_split`.
    pub fn kquant_gemv_w4a8_glu(
        &self,
        gate: &RepackedKQ,
        up: &RepackedKQ,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        ssums: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemv_w4a8_glu
            .ok_or(GpuError::MissingOp("kquant_gemv_w4a8_glu"))?;
        let in_dim = gate.dims[0];
        let out_dim = gate.dims[1];
        assert_eq!(up.dims[0], in_dim, "gate/up share in_dim");
        assert_eq!(up.dims[1], out_dim, "gate/up share out_dim");
        debug_assert!(y.len() >= out_dim);
        let (dtg, _, _) = kq_params(gate.ty).expect("RepackedKQ holds a k-quant type");
        let (dtu, _, _) = kq_params(up.ty).expect("RepackedKQ holds a k-quant type");
        let needs = crate::gpu::kq_needs_sums(gate.ty) || crate::gpu::kq_needs_sums(up.ty);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scales.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scales.device_ptr(&self.stream);
        let (xqp, _g5) = xq.device_ptr(&self.stream);
        let (xsp, _g6) = xs.device_ptr(&self.stream);
        let (smp, _g7) = ssums.device_ptr(&self.stream);
        let (yp, _g8) = y.device_ptr_mut(&self.stream);
        let sump: *const core::ffi::c_void = if needs {
            smp as *const _
        } else {
            core::ptr::null()
        };
        // SAFETY: ABI contract (slot 324)
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                xqp as *const _,
                xsp as *const _,
                sump,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                dtg,
                dtu,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_kquant_gemv_w4a8_nc(&self) -> bool {
        self.kernels.kquant_gemv_w4a8_nc.is_some()
    }

    /// Whether the multi-column W4A8 GEMV wins for this (weight, ncols) pair
    /// (same-run kernel A/B): r=2..3 beats the dp4a/mma_ks ladder
    /// on every format; at r=4..5 only Q6K holds GEMV class (two-run medians
    /// vs mma_ks on the same tensors: Q6K nc4 73-97 us vs ks 90-101 - nc
    /// wins; Q4K nc4 86-89 vs ks 65-73 and Q5K nc4 87-97 vs ks 82-97 - ks
    /// wins; r=5 mu formats lose by 30-40%). The per-row mu fold (NCOLS>=4
    /// in the kernel, tolerance-gated reorder class) lifted Q4K/Q5K nc4
    /// from the inline-fold collapse (230-360) to 337-417 weight-effective
    /// - the mu term is no longer the drag - but mma_ks still holds the
    ///   rung: with the fold the hot loop is IQ4-shaped and IQ4_XS fails
    ///   r>=4 with no mu at all, so the residual bound is per-chunk LDS
    ///   pressure (32-weight chunks pay 2 LDS.128 + 2 LDS.32 per 16-32
    ///   weight B vs Q6K's 4 LDS.128 per 48 B). Any future r>=4 push for
    ///   these formats needs a chunk-granularity/LDS fix on TOP of the fold;
    ///   the fold path stays compiled + parity-tested for that day.
    pub fn kquant_gemv_w4a8_nc_fits(w: &RepackedKQ, ncols: usize) -> bool {
        match ncols {
            2..=3 => true,
            4..=5 => w.ty == GgmlType::Q6K,
            _ => false,
        }
    }

    /// Multi-column W4A8 GEMV: ncols strided activation rows (`quantize_q8`
    /// layout) against one weight read - the spec-verify r-class (per column
    /// the b=1 GEMV's exact math in its exact chunk order).
    pub fn kquant_gemv_w4a8_nc(
        &self,
        w: &RepackedKQ,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        xsums: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        ncols: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kquant_gemv_w4a8_nc
            .ok_or(GpuError::MissingOp("kquant_gemv_w4a8_nc"))?;
        let (did, _, _) = kq_params(w.ty).expect("RepackedKQ holds a k-quant type");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        debug_assert!(y.len() >= out_dim * ncols);
        // launcher contract only - fits() is dispatch POLICY (what wins),
        // not a correctness bound; the parity suite exercises non-elected
        // (format, ncols) pairs deliberately
        debug_assert!((2..=5).contains(&ncols));
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp_, _g2) = w.scales.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (sump, _gs);
        let sp: *const core::ffi::c_void = match xsums {
            Some(s) => {
                (sump, _gs) = s.device_ptr(&self.stream);
                sump as *const _
            }
            None => core::ptr::null(),
        };
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                sp_ as *const _,
                xqp as *const _,
                xsp as *const _,
                sp,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                ncols as u32,
                did,
                self.stream_ptr(),
            )
        })
    }

    /// Dequant a byte range of a quantized-resident tensor into `dst`
    /// (dst.len() elements from `byte_offset`).
    pub fn dequant_slice(
        &self,
        q: &QuantTensor,
        byte_offset: usize,
        dst: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let (dequant, block_elems) = self.dequant_for(q.ty, "<quant slice>")?;
        let n_blocks = (dst.len() / block_elems) as u64;
        let view = q
            .bytes
            .try_slice(byte_offset..)
            .ok_or_else(|| oob("dequant_slice: offset out of range"))?;
        let (in_ptr, _g1) = view.device_ptr(&self.stream);
        let (out_ptr, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1; range validated above
        check(unsafe {
            dequant(
                in_ptr as *const core::ffi::c_void,
                out_ptr as *mut core::ffi::c_void,
                n_blocks,
                self.stream_ptr(),
            )
        })
    }
}
