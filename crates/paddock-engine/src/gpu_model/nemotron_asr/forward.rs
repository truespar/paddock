//! A pass and the session scheduling around it: which encoder frames the
//! audio received so far completes, the mel/subsampler row spans a window
//! reads, then - every stream's window stacked by rows - the 24 conformer
//! blocks, the prompt projector and the batched greedy walk.

use cudarc::driver::{CudaSlice, DevicePtr};
use paddock_models::nemotron_asr::{self as arch, D_MODEL, HOP, PRED, PROMPT_HIDDEN, SUB_CHANNELS};

use super::{
    GpuModelError, GpuNemotronAsr, MAX_PASS, MAX_SEGMENTS, MAX_WINDOW, Session, mel_rows,
    stage2_rows, stem_rows,
};
use crate::gpu::{
    DiarFrontendPlanes, KumoEpi, NASR_JOBS, NasrDecoderWeights, NasrEpi, NasrJob, NasrSeg,
    nasr_bytes,
};

/// Encoder frames whose every mel frame the first `received` samples cover:
/// a chunk ending at frame o1 reads mel frame 8 o1 - 8, whose centered
/// window ends at sample (8 o1 - 8) 160 + 255.
fn covered_frames(received: usize) -> usize {
    if received < 256 {
        return 0;
    }
    ((received - 256) / HOP + 8) / 8
}

/// One stream's window in a pass.
struct Window {
    o0: usize,
    o1: usize,
    last: bool,
    row0: usize,
}

impl GpuNemotronAsr {
    /// The window a session can encode now - `(first frame, end frame,
    /// under the recording's end)` - or None.
    pub fn next_window(s: &Session) -> Option<(usize, usize, bool)> {
        if s.finished {
            return None;
        }
        if s.finishing {
            let total = arch::encoder_frames(arch::mel_frames(s.received));
            return (s.done < total).then(|| (s.done, (s.done + MAX_WINDOW).min(total), true));
        }
        let chunk = arch::chunk_frames(s.lookahead);
        let ready = covered_frames(s.received) / chunk * chunk;
        (s.done < ready).then(|| (s.done, (s.done + MAX_WINDOW).min(ready), false))
    }

    /// Append PCM (16 kHz mono) to a session; [`Self::run`] encodes it.
    pub fn feed(&self, s: &mut Session, pcm: &[f32]) -> Result<(), GpuModelError> {
        if s.finishing {
            return Err(GpuModelError::Unsupported("the session is finished".into()));
        }
        s.pcm.extend_from_slice(pcm);
        s.received += pcm.len();
        Ok(())
    }

    /// Mark a session's recording complete; [`Self::run`] encodes the rest
    /// under its end and closes it.
    pub fn close(&self, s: &mut Session) {
        s.finishing = true;
        if Self::next_window(s).is_none() {
            s.finished = true;
        }
    }

    /// One pass over the sessions' next windows, as many as fit in
    /// [`MAX_PASS`] rows (in order; the rest wait for the next pass). New
    /// tokens land in each session's `tokens` / `frames`. Returns how many
    /// sessions it advanced.
    pub fn run(&mut self, sessions: &mut [&mut Session]) -> Result<usize, GpuModelError> {
        let mut wins: Vec<(usize, Window)> = Vec::new();
        let mut rows = 0usize;
        for (i, s) in sessions.iter().enumerate() {
            let Some((o0, o1, last)) = Self::next_window(s) else {
                continue;
            };
            if rows + (o1 - o0) > MAX_PASS || wins.len() == MAX_SEGMENTS {
                break;
            }
            wins.push((
                i,
                Window {
                    o0,
                    o1,
                    last,
                    row0: rows,
                },
            ));
            rows += o1 - o0;
        }
        if wins.is_empty() {
            return Ok(0);
        }
        // the subsampler, one stream at a time, into the pass's rows
        for (i, w) in &wins {
            self.subsample(sessions[*i], w)?;
        }
        let ex = self.exec.clone();
        let ws = &mut self.ws;
        ex.nasr_gemm(
            (&ws.flat, 0),
            &self.sub_out,
            Some(&self.sub_out_b),
            &mut ws.x,
            rows,
            NasrEpi::Store,
        )?;
        // the segments
        let mut segs = Vec::with_capacity(wins.len());
        let mut chunks = 0usize;
        for (i, w) in &wins {
            let s = &*sessions[*i];
            let chunk = arch::chunk_frames(s.lookahead);
            let (kv, _g1) = s.kv.device_ptr(&ex.stream);
            let (conv, _g2) = s.conv.device_ptr(&ex.stream);
            segs.push(NasrSeg {
                p0: w.o0 as i64,
                kv,
                conv,
                row0: w.row0 as u32,
                rows: (w.o1 - w.o0) as u32,
                chunk: chunk as u32,
                chunk0: chunks as u32,
                prompt: s.prompt,
                pad: 0,
            });
            chunks += (w.o1 - w.o0).div_ceil(chunk);
        }
        ex.upload_u8(&nasr_bytes(&segs), &mut ws.segs)?;
        let nseg = segs.len();
        let mut trace = self.trace.take();
        if let Some(t) = trace.as_mut() {
            t.add("sub", ex.to_host_len(&ws.x, rows * D_MODEL)?);
        }
        // the conformer blocks
        for (l, layer) in self.layers.iter().enumerate() {
            let eps = 1e-5;
            ex.clef_norm(
                &ws.x,
                &layer.ff1_norm.w,
                &layer.ff1_norm.b,
                &mut ws.n,
                D_MODEL,
                rows,
                eps,
            )?;
            ex.nasr_gemm(
                (&ws.n, 0),
                &layer.ff1_up,
                None,
                &mut ws.h,
                rows,
                NasrEpi::Silu,
            )?;
            ex.nasr_gemm(
                (&ws.h, 0),
                &layer.ff1_down,
                None,
                &mut ws.x,
                rows,
                NasrEpi::HalfResid,
            )?;
            ex.clef_norm(
                &ws.x,
                &layer.att_norm.w,
                &layer.att_norm.b,
                &mut ws.n,
                D_MODEL,
                rows,
                eps,
            )?;
            ex.nasr_gemm(
                (&ws.n, 0),
                &layer.qkv,
                None,
                &mut ws.qkv,
                rows,
                NasrEpi::Store,
            )?;
            ex.nasr_attention(
                &ws.qkv,
                &ws.segs,
                (nseg, chunks, rows),
                l,
                &layer.rel,
                (&layer.bias_u, &layer.bias_v),
                &mut ws.att,
            )?;
            ex.nasr_gemm(
                (&ws.att, 0),
                &layer.out,
                None,
                &mut ws.x,
                rows,
                NasrEpi::Resid,
            )?;
            ex.nasr_ring_put(&ws.qkv, (3 * D_MODEL, D_MODEL), &ws.segs, nseg, l, (0, 2))?;
            ex.clef_norm(
                &ws.x,
                &layer.conv_norm.w,
                &layer.conv_norm.b,
                &mut ws.n,
                D_MODEL,
                rows,
                eps,
            )?;
            ex.nasr_gemm((&ws.n, 0), &layer.pw1, None, &mut ws.g, rows, NasrEpi::Glu)?;
            ex.nasr_dwconv(
                &ws.g,
                &ws.segs,
                (nseg, rows),
                l,
                &layer.dw,
                (&layer.dw_norm.w, &layer.dw_norm.b),
                &mut ws.d,
            )?;
            ex.nasr_ring_put(&ws.g, (D_MODEL, 0), &ws.segs, nseg, l, (2, 1))?;
            ex.nasr_gemm(
                (&ws.d, 0),
                &layer.pw2,
                None,
                &mut ws.x,
                rows,
                NasrEpi::Resid,
            )?;
            ex.clef_norm(
                &ws.x,
                &layer.ff2_norm.w,
                &layer.ff2_norm.b,
                &mut ws.n,
                D_MODEL,
                rows,
                eps,
            )?;
            ex.nasr_gemm(
                (&ws.n, 0),
                &layer.ff2_up,
                None,
                &mut ws.h,
                rows,
                NasrEpi::Silu,
            )?;
            ex.nasr_gemm(
                (&ws.h, 0),
                &layer.ff2_down,
                None,
                &mut ws.x,
                rows,
                NasrEpi::HalfResid,
            )?;
            ex.clef_norm(
                &ws.x,
                &layer.out_norm.w,
                &layer.out_norm.b,
                &mut ws.n,
                D_MODEL,
                rows,
                eps,
            )?;
            std::mem::swap(&mut ws.x, &mut ws.n);
            if let Some(t) = trace.as_mut().filter(|t| t.layers.contains(&l)) {
                t.add(
                    &format!("layer_{l}"),
                    ex.to_host_len(&ws.x, rows * D_MODEL)?,
                );
            }
        }
        // the prompt projector (each row's prompt bias, then ReLU) and the
        // joint's encoder half
        ex.kumo_gemm(
            (&ws.x, 0),
            (&self.prompt1, 0),
            None,
            &mut ws.f1,
            None,
            (D_MODEL, PROMPT_HIDDEN, rows),
            KumoEpi::Store,
            1,
            (0, 0, 0),
        )?;
        ex.nasr_prompt_relu(
            &mut ws.f1,
            (PROMPT_HIDDEN, rows),
            &ws.segs,
            nseg,
            &self.prompt1_b,
        )?;
        ex.kumo_gemm(
            (&ws.f1, 0),
            (&self.prompt2, 0),
            Some((&self.prompt2_b, 0)),
            &mut ws.fused,
            None,
            (PROMPT_HIDDEN, D_MODEL, rows),
            KumoEpi::Store,
            1,
            (0, 0, 0),
        )?;
        ex.nasr_gemm(
            (&ws.fused, 0),
            &self.enc,
            Some(&self.enc_b),
            &mut ws.proj,
            rows,
            NasrEpi::Store,
        )?;
        if let Some(t) = trace.as_mut() {
            t.add("fused", ex.to_host_len(&ws.fused, rows * D_MODEL)?);
            t.add("proj", ex.to_host_len(&ws.proj, rows * PRED)?);
        }
        self.trace = trace;
        // the greedy walk, NASR_JOBS streams a launch
        let w = NasrDecoderWeights {
            head: (&self.head, &self.head_b),
            ih: [
                (&self.ih[0].0, &self.ih[0].1),
                (&self.ih[1].0, &self.ih[1].1),
            ],
            hh: [
                (&self.hh[0].0, &self.hh[0].1),
                (&self.hh[1].0, &self.hh[1].1),
            ],
            pred: (&self.pred, &self.pred_b),
            embed: &self.embed,
        };
        let job_bytes = std::mem::size_of::<NasrJob>();
        let mut counts: Vec<usize> = Vec::with_capacity(wins.len());
        for group in wins.chunks(NASR_JOBS) {
            let jobs: Vec<NasrJob> = group
                .iter()
                .map(|(i, win)| {
                    let s = &*sessions[*i];
                    let (st, _g) = s.dec.device_ptr(&ex.stream);
                    NasrJob {
                        st,
                        frame0: win.o0 as i64,
                        row0: win.row0 as u32,
                        rows: (win.o1 - win.o0) as u32,
                        out0: (win.row0 * arch::MAX_SYMBOLS) as u32,
                        cap: ((win.o1 - win.o0) * arch::MAX_SYMBOLS) as u32,
                        prime: u32::from(!s.primed),
                        count: 0,
                    }
                })
                .collect();
            ex.upload_u8(&nasr_bytes(&jobs), &mut ws.jobs)?;
            ex.nasr_greedy(
                &w,
                &ws.proj,
                &mut ws.jobs,
                jobs.len(),
                (&mut ws.tokens, &mut ws.frames),
                &mut ws.part,
            )?;
            let back = ex.to_host_range_u8(&ws.jobs, 0, jobs.len() * job_bytes)?;
            for (j, b) in back.chunks_exact(job_bytes).enumerate() {
                let count = u32::from_le_bytes([b[36], b[37], b[38], b[39]]) as usize;
                if count > jobs[j].cap as usize {
                    return Err(GpuModelError::Unsupported(format!(
                        "Nemotron 3.5 ASR: the greedy walk emitted {count} tokens over {} frames",
                        jobs[j].rows
                    )));
                }
                counts.push(count);
            }
        }
        let tokens = ex.to_host_u32_len(&ws.tokens, rows * arch::MAX_SYMBOLS)?;
        let frames = ex.to_host_u32_len(&ws.frames, rows * arch::MAX_SYMBOLS)?;
        for ((i, win), &count) in wins.iter().zip(&counts) {
            let s = &mut *sessions[*i];
            let at = win.row0 * arch::MAX_SYMBOLS;
            s.tokens.extend_from_slice(&tokens[at..at + count]);
            s.frames.extend_from_slice(&frames[at..at + count]);
            s.primed = true;
            s.done = win.o1;
            if win.last && Self::next_window(s).is_none() {
                s.finished = true;
            }
            // keep the PCM the next window's first frames reach back to
            let keep_from = ((8 * s.done).saturating_sub(14) * HOP).saturating_sub(257);
            if keep_from > s.pcm_base {
                let n = (keep_from - s.pcm_base).min(s.pcm.len());
                s.pcm.drain(..n);
                s.pcm_base += n;
            }
        }
        Ok(wins.len())
    }

    /// One stream's window through the subsampler into the pass's flattened
    /// rows `w.row0 ..`.
    fn subsample(&mut self, s: &Session, w: &Window) -> Result<(), GpuModelError> {
        let (o0, o1) = (w.o0, w.o1);
        let r = o1 - o0;
        let mel_valid = if w.last {
            arch::mel_frames(s.received) as i64
        } else {
            i64::MAX
        };
        let s1_valid = if w.last { mel_valid / 2 + 1 } else { i64::MAX };
        let s2_valid = if w.last { s1_valid / 2 + 1 } else { i64::MAX };
        // the row spans this window reads, each stage's 2t - 2 .. 2t taps
        let m_lo = (8 * o0).saturating_sub(14);
        let m_hi = 8 * o1 - 7;
        let a1 = (4 * o0).saturating_sub(6);
        let b1 = (4 * o1 - 3).min(s1_valid as usize);
        let a2 = (2 * o0).saturating_sub(2);
        let b2 = (2 * o1 - 1).min(s2_valid as usize);
        debug_assert!(m_hi - m_lo <= mel_rows(MAX_WINDOW));
        debug_assert!(b1 - a1 <= stem_rows(MAX_WINDOW) && b2 - a2 <= stage2_rows(MAX_WINDOW));
        // PCM: from the preemphasis tap of the first frame's window to the
        // last frame's window end, or the recording's end
        let s_lo = (m_lo * HOP).saturating_sub(257);
        let s_hi = s.received.min((m_hi - 1) * HOP + 256);
        if s_lo < s.pcm_base || s_hi > s.pcm_base + s.pcm.len() {
            return Err(GpuModelError::Unsupported(
                "Nemotron 3.5 ASR: the session's PCM tail is short of the window".into(),
            ));
        }
        let ex = self.exec.clone();
        let ws = &mut self.ws;
        ex.upload_f32(&s.pcm[s_lo - s.pcm_base..s_hi - s.pcm_base], &mut ws.pcm)?;
        let planes = DiarFrontendPlanes {
            window: &self.window,
            fb: &self.fb,
            spans: &self.spans,
            twiddle: &self.twiddle,
        };
        ex.diar_frontend(
            &ws.pcm,
            &planes,
            &mut ws.mel,
            (s_lo, s_hi, m_lo, m_hi - m_lo),
            m_hi - m_lo,
        )?;
        ex.nasr_sub_stem(
            &ws.mel,
            (m_lo, m_hi - m_lo, mel_valid),
            (&self.stem.0, &self.stem.1),
            &mut ws.s1,
            a1,
            b1 - a1,
        )?;
        ex.nasr_sub_dw(
            &ws.s1,
            (a1, b1 - a1, s1_valid),
            65,
            (&self.dw1.0, &self.dw1.1),
            &mut ws.s2a,
            a2,
            b2 - a2,
            false,
        )?;
        let pointwise = |x: &CudaSlice<f32>,
                         w: &(CudaSlice<f32>, CudaSlice<f32>),
                         y: &mut CudaSlice<f32>,
                         rows: usize| {
            ex.kumo_gemm(
                (x, 0),
                (&w.0, 0),
                Some((&w.1, 0)),
                y,
                None,
                (SUB_CHANNELS, SUB_CHANNELS, rows),
                KumoEpi::Store,
                1,
                (0, 0, 0),
            )
        };
        pointwise(&ws.s2a, &self.pw1, &mut ws.s2b, (b2 - a2) * 33)?;
        ex.nasr_sub_dw(
            &ws.s2b,
            (a2, b2 - a2, s2_valid),
            33,
            (&self.dw2.0, &self.dw2.1),
            &mut ws.s3a,
            o0,
            r,
            true,
        )?;
        pointwise(&ws.s3a, &self.pw2, &mut ws.s3b, r * 17)?;
        ex.nasr_sub_flatten(&ws.s3b, &mut ws.flat, w.row0, r)?;
        Ok(())
    }

    /// Append PCM and encode + decode every chunk it completes (this
    /// session alone). New tokens land in `s.tokens` / `s.frames`.
    pub fn push(&mut self, s: &mut Session, pcm: &[f32]) -> Result<(), GpuModelError> {
        self.feed(s, pcm)?;
        while self.run(&mut [&mut *s])? > 0 {}
        Ok(())
    }

    /// The recording ended: encode the rest under its end (frames past the
    /// last full hop zero, the subsampler's right padding, the last chunk
    /// partial) and close the session.
    pub fn finish(&mut self, s: &mut Session) -> Result<(), GpuModelError> {
        self.close(s);
        while self.run(&mut [&mut *s])? > 0 {}
        Ok(())
    }
}
