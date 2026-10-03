//! One window's passes: the frontend with the stacked-frame projection, and
//! the encoder plus head over a window of cached, FIFO and chunk rows.

use std::time::Instant;

use cudarc::driver::CudaSlice;

use super::{GpuDiarization, GpuModelError, Linear, MAX_PCM, MAX_ROWS, Weights, Workspace};
use crate::gpu::{DiarAct, DiarEpi, DiarFrontendPlanes, DiarWeights, GpuExecutor, KumoEpi};

const NO_STRIDES: (usize, usize, usize) = (0, 0, 0);

fn bad(what: impl std::fmt::Display) -> GpuModelError {
    GpuModelError::Unsupported(format!("Nemotron 3 Diarization: {what}"))
}

/// `y = epi(x . W + b)` over `m` rows of `l.k` inputs; `rope` (the q plane
/// and the (cos, sin) table) with [`DiarEpi::Rope`].
fn linear(
    e: &GpuExecutor,
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
    e.diar_gemm((x, 0), stored, l.b.as_ref(), y, (l.k, l.n, m), epi, rope)?;
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
        if rows > MAX_ROWS || w.audio.len() > MAX_PCM {
            return Err(bad("audio window larger than the resident planes"));
        }
        let e = &*self.exec;
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
            || features.len() > MAX_ROWS * 1024
            || features.iter().any(|v| !v.is_finite())
        {
            return Err(bad("invalid feature window"));
        }
        let rows = features.len() / 1024;
        let e = &*self.exec;
        let ws = &mut self.ws;
        upload(e, features, &mut ws.input)?;
        let start = Instant::now();
        linear(
            e,
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
            || rows > MAX_ROWS
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

    /// The encoder and head over the uploaded window, enqueued (no sync).
    fn encoder(&mut self, rows: usize, valid: usize) -> Result<(), GpuModelError> {
        let e = &*self.exec;
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
            ..
        } = &mut self.ws;
        fn nw(nm: &super::Norm) -> (&CudaSlice<f32>, &CudaSlice<f32>) {
            (&nm.w, &nm.b)
        }
        e.diar_norm((&*input, 0), nw(&self.embed), x, rows)?;
        for l in &self.layers {
            e.diar_norm((&*x, 0), nw(&l.n1), n, rows)?;
            // q to its own plane with the rope, k roped in place, v as is
            linear(
                e,
                &l.qkv,
                n,
                qkv,
                rows,
                DiarEpi::Rope,
                Some((q, &self.rope)),
            )?;
            // k and v stay in the qkv rows: 24 heads of 64 a row, the k
            // heads at 512 and the v heads at 1024
            e.diar_attention(q, qkv, att, rows, valid)?;
            linear(e, &l.out, att, x, rows, DiarEpi::Resid, None)?;
            e.diar_norm((&*x, 0), nw(&l.n2), n, rows)?;
            linear(e, &l.up, n, wide, rows, DiarEpi::Gelu, None)?;
            linear(e, &l.down, wide, x, rows, DiarEpi::Resid, None)?;
        }
        e.diar_norm((&*x, 0), nw(&self.last), n, rows)?;
        linear(e, &self.proj, n, head, rows, DiarEpi::Store, None)?;
        e.diar_conv_rows(head, conv, rows)?;
        linear(e, &self.conv, conv, up, rows, DiarEpi::Store, None)?;
        e.diar_act(up, rows * 1536, DiarAct::Relu)?;
        // [R][1536] is [8R][192]: one row a 10-ms frame from here
        linear(e, &self.hidden, up, hidden, rows * 8, DiarEpi::Store, None)?;
        e.diar_act(hidden, rows * 1536, DiarAct::Relu)?;
        linear(
            e,
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
