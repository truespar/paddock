//! One window's passes: the frontend with the stacked-frame projection, and
//! the encoder plus head over a window of cached, FIFO and chunk rows.

use std::time::Instant;

use cudarc::driver::CudaSlice;

use super::{
    B16Planes, GpuDiarization, GpuModelError, Linear, MAX_PCM, Packed, Weights, Workspace,
};
use crate::gpu::{
    DiarAct, DiarB16Out, DiarEpi, DiarFrontendPlanes, DiarKeys, DiarWeights, GpuExecutor, KumoEpi,
};

const NO_STRIDES: (usize, usize, usize) = (0, 0, 0);

fn bad(what: impl std::fmt::Display) -> GpuModelError {
    GpuModelError::Unsupported(format!("Nemotron 3 Diarization: {what}"))
}

/// `y = epi(x . W + b)` over `m` rows of `l.k` inputs; `rope` (the q plane
/// and the (cos, sin) table) with [`DiarEpi::Rope`]. `parts`: the stored-
/// weight GEMM's activation class (the F32 head GEMMs keep their own).
fn linear(
    e: &GpuExecutor,
    parts: u32,
    l: &Linear,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    m: usize,
    epi: DiarEpi,
    rope: Option<(&mut CudaSlice<f32>, &CudaSlice<f32>)>,
) -> Result<(), GpuModelError> {
    let stored = match &l.w {
        Weights::F32(w) => {
            let epi = match epi {
                DiarEpi::Store => KumoEpi::Store,
                DiarEpi::Gelu => KumoEpi::Gelu,
                DiarEpi::Resid => KumoEpi::Resid,
                DiarEpi::Rope | DiarEpi::SwiGlu | DiarEpi::GeluTanh => {
                    return Err(bad(
                        "the rope, SwiGLU and tanh-GELU epilogues need stored weights",
                    ));
                }
            };
            e.kumo_gemm(
                (x, 0),
                (w, 0),
                l.b.as_ref().map(|b| (b, 0)),
                y,
                None,
                (l.k, l.n, m),
                epi,
                1,
                NO_STRIDES,
            )?;
            return Ok(());
        }
        Weights::Bf16(w) => DiarWeights::Bf16(w),
        Weights::Q8(q, scale) => DiarWeights::Q8 { q, scale },
    };
    e.diar_gemm_parts(
        (x, 0),
        stored,
        l.b.as_ref(),
        y,
        (l.k, l.n, m),
        epi,
        rope,
        parts,
    )?;
    Ok(())
}

/// A stored-weight projection in the BF16 class (slot 965): `x16` the bf16
/// operand rows; the weights are the checkpoint's BF16 rows (the class's
/// setter refuses any other plane).
fn lin16(
    e: &GpuExecutor,
    l: &Linear,
    x16: &CudaSlice<half::bf16>,
    y: DiarB16Out<'_>,
    m: usize,
    rope: Option<(&mut CudaSlice<f32>, &CudaSlice<f32>)>,
) -> Result<(), GpuModelError> {
    let Weights::Bf16(w) = &l.w else {
        return Err(bad("the BF16 class takes the BF16 rows as stored"));
    };
    e.diar_b16_gemm(x16, w, l.b.as_ref(), y, (l.k, l.n, m), rope)?;
    Ok(())
}

/// Copy a host slice into the front of a resident plane.
fn upload(e: &GpuExecutor, host: &[f32], to: &mut CudaSlice<f32>) -> Result<(), GpuModelError> {
    let mut view = to
        .try_slice_mut(0..host.len())
        .ok_or_else(|| bad("upload larger than its plane"))?;
    e.stream
        .memcpy_htod(host, &mut view)
        .map_err(|err| bad(format!("upload: {err}")))
}

fn readback(e: &GpuExecutor, from: &CudaSlice<f32>, n: usize) -> Result<Vec<f32>, GpuModelError> {
    let view = from.try_slice(0..n).ok_or_else(|| bad("readback range"))?;
    e.stream
        .clone_dtoh(&view)
        .map_err(|err| bad(format!("readback: {err}")))
}

/// Windows one packed pass takes at most (its tile plane's sizing).
pub const MAX_PACKED: usize = 64;

/// What a pass's rows attend over.
enum Keys {
    /// one window: every row over the first `valid`
    Window(usize),
    /// packed windows: `tiles` uploaded tiles of `warps` x 16 queries, and
    /// the spare rows between windows
    Packed {
        tiles: usize,
        warps: usize,
        pads: Vec<usize>,
    },
}

impl GpuDiarization {
    /// Frame-major log-mels of a PCM window projected to encoder rows: the
    /// frames padded to a multiple of eight, `count / 8` rounded up rows of
    /// 512 (the Metal lane's contract).
    pub fn encode_audio(
        &mut self,
        w: crate::diarization::AudioWindow<'_>,
    ) -> Result<Vec<f32>, GpuModelError> {
        w.validate().map_err(bad)?;
        let frames = w.count.div_ceil(8) * 8;
        let rows = frames / 8;
        if rows > self.max_rows || w.audio.len() > MAX_PCM {
            return Err(bad("audio window larger than the resident planes"));
        }
        let (e, parts) = (&*self.exec, self.parts);
        let ws = &mut self.ws;
        upload(e, w.audio, &mut ws.pcm)?;
        let start = Instant::now();
        e.diar_frontend(
            &ws.pcm,
            &DiarFrontendPlanes {
                window: &self.window_gpu,
                fb: &self.fb_gpu,
                spans: &self.spans,
                twiddle: &self.twiddle,
            },
            &mut ws.input,
            (w.offset, w.total, w.start, w.count),
            frames,
        )?;
        linear(
            e,
            parts,
            &self.pre,
            &ws.input,
            &mut ws.x,
            rows,
            DiarEpi::Store,
            None,
        )?;
        e.synchronize()?;
        self.pre_seconds = start.elapsed().as_secs_f64();
        readback(e, &ws.x, rows * 512)
    }

    /// The stacked-frame projection of host features `[8R][128]` (frames a
    /// multiple of eight): the qualification path, beside the frontend.
    pub fn pre_encode(&mut self, features: &[f32]) -> Result<Vec<f32>, GpuModelError> {
        if features.is_empty()
            || !features.len().is_multiple_of(1024)
            || features.len() > self.max_rows * 1024
            || features.iter().any(|v| !v.is_finite())
        {
            return Err(bad("invalid feature window"));
        }
        let rows = features.len() / 1024;
        let (e, parts) = (&*self.exec, self.parts);
        let ws = &mut self.ws;
        upload(e, features, &mut ws.input)?;
        let start = Instant::now();
        linear(
            e,
            parts,
            &self.pre,
            &ws.input,
            &mut ws.x,
            rows,
            DiarEpi::Store,
            None,
        )?;
        e.synchronize()?;
        self.pre_seconds = start.elapsed().as_secs_f64();
        readback(e, &ws.x, rows * 512)
    }

    /// The log-mel frames of a whole recording, `audio` (mono 16 kHz), into
    /// `out` `[frames][128]` on the device: `audio.len() / 160` valid frames,
    /// then zero frames up to `frames` (the frontend's padding rule). The
    /// frontend runs over spans the PCM plane holds; every frame reads the
    /// same samples it would in one pass, so the features are the one pass's.
    pub fn features_into(
        &mut self,
        audio: &[f32],
        out: &mut CudaSlice<f32>,
        frames: usize,
    ) -> Result<(), GpuModelError> {
        let total = audio.len();
        if out.len() < frames * 128 || total > u32::MAX as usize {
            return Err(bad("feature plane under the recording"));
        }
        // frames a span: the PCM plane holds its samples plus the 257 the
        // first frame's window and preemphasis reach back, the 256 the last
        // reaches forward; the feature plane holds its frames
        let span = ((MAX_PCM - 1024) / 160).min(self.max_rows * 8);
        let e = &*self.exec;
        let mut f0 = 0usize;
        while f0 < frames {
            let n = span.min(frames - f0);
            let offset = (f0 * 160).saturating_sub(257);
            // the span's samples, the recording's end where it falls inside
            let end = ((f0 + n) * 160 + 256).min(total);
            let ws = &mut self.ws;
            if end > offset {
                upload(e, &audio[offset..end], &mut ws.pcm)?;
            }
            e.diar_frontend(
                &ws.pcm,
                &DiarFrontendPlanes {
                    window: &self.window_gpu,
                    fb: &self.fb_gpu,
                    spans: &self.spans,
                    twiddle: &self.twiddle,
                },
                &mut ws.input,
                (offset, end.max(offset), f0, n),
                n,
            )?;
            e.copy_region(&ws.input, 0, out, f0 * 128, n * 128)?;
            f0 += n;
        }
        Ok(())
    }

    /// The stacked-frame projection of `rows` device feature rows (`[rows][1024]`
    /// = 8 frames of 128 each, starting at row `row0` of `feats`), back to the
    /// host as `[rows][512]` - the encoder input the windows are cut from.
    pub fn pre_encode_rows(
        &mut self,
        feats: &CudaSlice<f32>,
        row0: usize,
        rows: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        if feats.len() < (row0 + rows) * 1024 {
            return Err(bad("feature rows out of range"));
        }
        let mut host = Vec::with_capacity(rows * 512);
        let mut r = 0usize;
        while r < rows {
            let n = (rows - r).min(self.max_rows);
            let (e, parts) = (&*self.exec, self.parts);
            let ws = &mut self.ws;
            e.copy_region(feats, (row0 + r) * 1024, &mut ws.input, 0, n * 1024)?;
            linear(
                e,
                parts,
                &self.pre,
                &ws.input,
                &mut ws.x,
                n,
                DiarEpi::Store,
                None,
            )?;
            host.extend_from_slice(&readback(e, &ws.x, n * 512)?);
            r += n;
        }
        Ok(host)
    }

    /// [`Self::pre_encode_rows`] into the device plane `out` `[rows][512]`,
    /// enqueued - no readback (the speech LM's encode stays on its lane).
    pub fn pre_encode_rows_into(
        &mut self,
        feats: &CudaSlice<f32>,
        row0: usize,
        rows: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuModelError> {
        if feats.len() < (row0 + rows) * 1024 || out.len() < rows * 512 {
            return Err(bad("feature rows out of range"));
        }
        let mut r = 0usize;
        while r < rows {
            let n = (rows - r).min(self.max_rows);
            let (e, parts) = (&*self.exec, self.parts);
            let ws = &mut self.ws;
            e.copy_region(feats, (row0 + r) * 1024, &mut ws.input, 0, n * 1024)?;
            linear(
                e,
                parts,
                &self.pre,
                &ws.input,
                &mut ws.x,
                n,
                DiarEpi::Store,
                None,
            )?;
            e.copy_region(&ws.x, 0, out, r * 512, n * 512)?;
            r += n;
        }
        Ok(())
    }

    /// [`Self::predict`] over the device rows `emb` (`rows` of 512, the first
    /// `valid` keys), enqueued: the speaker probabilities `[rows * 8][8]` stay
    /// in the returned plane. Rows from `valid` on are NOT zeroed - the
    /// consumer masks them, as `predict`'s zeroing does.
    pub fn predict_device(
        &mut self,
        emb: &CudaSlice<f32>,
        rows: usize,
        valid: usize,
    ) -> Result<&CudaSlice<f32>, GpuModelError> {
        if rows == 0 || rows > self.max_rows || emb.len() < rows * 512 || valid == 0 || valid > rows
        {
            return Err(bad("invalid encoder window"));
        }
        let e = &*self.exec;
        e.copy_region(emb, 0, &mut self.ws.input, 0, rows * 512)?;
        self.encoder(rows, valid)?;
        Ok(&self.ws.probs)
    }

    /// Speaker probabilities `[8R]` of a window of `R` encoder rows whose
    /// first `valid` are keys (padded rows are queries only and score zero),
    /// with the GPU seconds of this pass and the frontend before it.
    pub fn predict(
        &mut self,
        embeddings: &[f32],
        valid: usize,
    ) -> Result<(Vec<[f32; 8]>, f64), GpuModelError> {
        let rows = embeddings.len() / 512;
        if rows == 0
            || rows > self.max_rows
            || !embeddings.len().is_multiple_of(512)
            || valid == 0
            || valid > rows
            || embeddings.iter().any(|v| !v.is_finite())
        {
            return Err(bad("invalid encoder window"));
        }
        let e = &*self.exec;
        let ws = &mut self.ws;
        upload(e, embeddings, &mut ws.input)?;
        let start = Instant::now();
        self.encoder(rows, valid)?;
        let ws = &mut self.ws;
        let e = &*self.exec;
        e.synchronize()?;
        let seconds = start.elapsed().as_secs_f64() + std::mem::take(&mut self.pre_seconds);
        let flat = readback(e, &ws.probs, rows * 64)?;
        let mut probs: Vec<[f32; 8]> = flat.as_chunks::<8>().0.to_vec();
        probs[valid * 8..].fill([0.; 8]);
        if probs.iter().flatten().any(|v| !v.is_finite()) {
            return Err(bad("nonfinite model output"));
        }
        Ok((probs, seconds))
    }

    /// Several windows in one pass (slot 950), enqueued: window `i` is
    /// `wins[i] = (rows, keys)` device rows `[rows][512]` in `embs[i]`; they
    /// pack with one spare row between neighbours (its projection zeroed
    /// before the conv rows, the zero padding a window's edge reads). Returns
    /// the probabilities plane `[packed rows * 8][8]` and each window's first
    /// packed row. Every row's bits are its window's alone: the GEMMs'
    /// per-element sequence does not depend on the tile shape the row count
    /// elects, the rope table restarts at each window, and the attention
    /// tiles keep to a window and walk its own keys.
    pub fn predict_packed_device(
        &mut self,
        embs: &[&CudaSlice<f32>],
        wins: &[(usize, usize)],
    ) -> Result<(&CudaSlice<f32>, Vec<usize>), GpuModelError> {
        if embs.len() != wins.len() || wins.is_empty() {
            return Err(bad("packed windows"));
        }
        let mut firsts = Vec::with_capacity(wins.len());
        let mut at = 0usize;
        for (i, &(rows, keys)) in wins.iter().enumerate() {
            if rows == 0 || keys == 0 || keys > rows || embs[i].len() < rows * 512 {
                return Err(bad("invalid encoder window"));
            }
            firsts.push(at);
            at += rows + 1;
        }
        let total = at - 1;
        if total > self.max_rows {
            return Err(bad("packed windows past the resident planes"));
        }
        if self.packed.is_none() {
            let e = &*self.exec;
            self.packed = Some(Packed {
                rope: e.kumo_plane(self.max_rows * 64)?,
                tiles: e.alloc_u32(4 * (self.max_rows / 32 + 2 * MAX_PACKED))?,
            });
        }
        if wins.len() > MAX_PACKED {
            return Err(bad("too many packed windows"));
        }
        // the attention's tile height by the same rule as one window's call
        // (the bits do not depend on it)
        let nsm = self.exec.sm_count();
        let warps = if total.div_ceil(64) * 8 >= nsm { 4 } else { 2 };
        let bq = 16 * warps;
        let mut tiles: Vec<u32> = Vec::new();
        {
            let e = &*self.exec;
            let pk = self.packed.as_mut().expect("packed planes");
            for (i, &(rows, keys)) in wins.iter().enumerate() {
                let f = firsts[i];
                e.copy_region(embs[i], 0, &mut self.ws.input, f * 512, rows * 512)?;
                e.copy_region(&self.rope, 0, &mut pk.rope, f * 64, rows * 64)?;
                for q in (0..rows).step_by(bq) {
                    tiles.extend([
                        (f + q) as u32,
                        bq.min(rows - q) as u32,
                        f as u32,
                        keys as u32,
                    ]);
                }
            }
            e.upload_u32(&tiles, &mut pk.tiles)?;
        }
        let pads: Vec<usize> = firsts[1..].iter().map(|&f| f - 1).collect();
        self.encoder_pass(
            total,
            Keys::Packed {
                tiles: tiles.len() / 4,
                warps,
                pads,
            },
        )?;
        Ok((&self.ws.probs, firsts))
    }

    /// The encoder and head over the uploaded window, enqueued (no sync).
    fn encoder(&mut self, rows: usize, valid: usize) -> Result<(), GpuModelError> {
        self.encoder_pass(rows, Keys::Window(valid))
    }

    /// The encoder and head over `rows` uploaded rows, enqueued: one window
    /// over its first `valid` keys, or packed windows.
    fn encoder_pass(&mut self, rows: usize, keys: Keys) -> Result<(), GpuModelError> {
        let (e, parts) = (&*self.exec, self.parts);
        let rope = match &keys {
            Keys::Window(_) => &self.rope,
            Keys::Packed { .. } => &self.packed.as_ref().expect("packed planes").rope,
        };
        let Workspace {
            input,
            x,
            n,
            qkv,
            q,
            att,
            wide,
            head,
            conv,
            up,
            hidden,
            probs,
            b16,
            ..
        } = &mut self.ws;
        fn nw(nm: &super::Norm) -> (&CudaSlice<f32>, &CudaSlice<f32>) {
            (&nm.w, &nm.b)
        }
        e.diar_norm((&*input, 0), nw(&self.embed), x, rows)?;
        // the BF16 class: every projection's operand landed as bf16 by its
        // producer, the residual stream and the head still f32
        let bf16 = self.parts == 1;
        if let Some(B16Planes { n16, att16, wide16 }) = b16.as_mut().filter(|_| bf16) {
            let packed = self.packed.as_ref();
            for l in &self.layers {
                e.diar_norm_b16((&*x, 0), nw(&l.n1), n16, rows)?;
                lin16(
                    e,
                    &l.qkv,
                    n16,
                    DiarB16Out::F32(qkv, DiarEpi::Rope),
                    rows,
                    Some((q, rope)),
                )?;
                let dk = match &keys {
                    Keys::Window(valid) => DiarKeys::Window(*valid),
                    Keys::Packed { tiles, warps, .. } => DiarKeys::Packed {
                        tiles: &packed.expect("packed planes").tiles,
                        n_tiles: *tiles,
                        warps: *warps,
                    },
                };
                e.diar_attention_b16(q, qkv, att16, rows, dk)?;
                lin16(
                    e,
                    &l.out,
                    att16,
                    DiarB16Out::F32(x, DiarEpi::Resid),
                    rows,
                    None,
                )?;
                e.diar_norm_b16((&*x, 0), nw(&l.n2), n16, rows)?;
                lin16(e, &l.up, n16, DiarB16Out::GeluB16(wide16), rows, None)?;
                lin16(
                    e,
                    &l.down,
                    wide16,
                    DiarB16Out::F32(x, DiarEpi::Resid),
                    rows,
                    None,
                )?;
            }
        }
        for l in self.layers.iter().filter(|_| !bf16) {
            e.diar_norm((&*x, 0), nw(&l.n1), n, rows)?;
            // q to its own plane with the rope, k roped in place, v as is
            linear(
                e,
                parts,
                &l.qkv,
                n,
                qkv,
                rows,
                DiarEpi::Rope,
                Some((q, rope)),
            )?;
            // k and v stay in the qkv rows: 24 heads of 64 a row, the k
            // heads at 512 and the v heads at 1024
            match &keys {
                Keys::Window(valid) => e.diar_attention(q, qkv, att, rows, *valid)?,
                Keys::Packed { tiles, warps, .. } => e.diar_attention_varlen(
                    q,
                    qkv,
                    att,
                    rows,
                    &self.packed.as_ref().expect("packed planes").tiles,
                    *tiles,
                    *warps,
                )?,
            }
            linear(e, parts, &l.out, att, x, rows, DiarEpi::Resid, None)?;
            e.diar_norm((&*x, 0), nw(&l.n2), n, rows)?;
            linear(e, parts, &l.up, n, wide, rows, DiarEpi::Gelu, None)?;
            linear(e, parts, &l.down, wide, x, rows, DiarEpi::Resid, None)?;
        }
        e.diar_norm((&*x, 0), nw(&self.last), n, rows)?;
        linear(e, parts, &self.proj, n, head, rows, DiarEpi::Store, None)?;
        if let Keys::Packed { pads, .. } = &keys {
            // the spare rows read as the conv's zero padding
            for &p in pads {
                e.zero_region(head, p * 192, 192)?;
            }
        }
        e.diar_conv_rows(head, conv, rows)?;
        linear(e, parts, &self.conv, conv, up, rows, DiarEpi::Store, None)?;
        e.diar_act(up, rows * 1536, DiarAct::Relu)?;
        // [R][1536] is [8R][192]: one row a 10-ms frame from here
        linear(
            e,
            parts,
            &self.hidden,
            up,
            hidden,
            rows * 8,
            DiarEpi::Store,
            None,
        )?;
        e.diar_act(hidden, rows * 1536, DiarAct::Relu)?;
        linear(
            e,
            parts,
            &self.score,
            hidden,
            probs,
            rows * 8,
            DiarEpi::Store,
            None,
        )?;
        e.diar_act(probs, rows * 64, DiarAct::Sigmoid)?;
        Ok(())
    }
}
