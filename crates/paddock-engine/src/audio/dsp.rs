//! Host DSP primitives for the audio families: FFT, Hann window, Slaney mel
//! filterbank, and the log-mel frame loop. Device-free and directly testable
//! (the granite `preprocess.rs` convention). All accumulation runs in f64 -
//! at ~100 frames/s of audio this costs nothing and keeps us at least as
//! close to exact as both references (torch.stft is f32; llama.cpp's FFT is
//! f32 with an f64 mel dot), so oracle deltas are dominated by their
//! rounding, not ours.
//!
//! References studied (algorithms, no code copied): llama.cpp b10327
//! tools/mtmd/mtmd-audio.cpp (recursive even-split FFT, Slaney filterbank,
//! Whisper norm) and transformers 5.14.1 audio_utils.mel_filter_bank +
//! models/qwen3_asr/feature_extraction_qwen3_asr.py (the checkpoint's
//! declared extractor - the correctness oracle).

/// Precomputed trig for a fixed-length FFT. The frame loop used to call
/// `sin_cos` ~10.8k times per 400-point frame (625 per 25-point base DFT ×
/// 16, plus every merge-level twiddle) - ~0.25 s of host time on a 14 s clip,
/// the single largest cost of a transcription request. The tables hold the
/// same values the loops computed (same `theta` expression, evaluated once),
/// and the accumulation order is untouched, so planned output is
/// bit-identical to the unplanned code it replaced - the mel oracle gate
/// (tests/asr_mel_oracle.rs) pins that.
pub struct FftPlan {
    /// even-split sizes in recursion order (n, twiddles[k<n/2] = sin_cos of
    /// -2*pi*k/n); looked up by n during the recursive merge
    levels: Vec<(usize, Vec<(f64, f64)>)>,
    /// odd base case: (n, table[k*n+i] = sin_cos of -2*pi*(k*i)/n)
    base: (usize, Vec<(f64, f64)>),
}

impl FftPlan {
    pub fn new(n_fft: usize) -> Self {
        let mut levels = Vec::new();
        let mut n = n_fft;
        while n > 1 && n.is_multiple_of(2) {
            let tw: Vec<(f64, f64)> = (0..n / 2)
                .map(|k| (-2.0 * std::f64::consts::PI * k as f64 / n as f64).sin_cos())
                .collect();
            levels.push((n, tw));
            n /= 2;
        }
        let base_tab = if n > 1 {
            (0..n * n)
                .map(|ki| {
                    let (k, i) = (ki / n, ki % n);
                    (-2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64).sin_cos()
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            levels,
            base: (n, base_tab),
        }
    }

    fn twiddles(&self, n: usize) -> &[(f64, f64)] {
        &self
            .levels
            .iter()
            .find(|(ln, _)| *ln == n)
            .expect("planned level")
            .1
    }
}

/// Complex radix-2 decimation-in-time FFT with a naive-DFT base case for odd
/// lengths (n_fft = 400 factors 2^4 * 25, so recursion bottoms out at a
/// 25-point DFT - the same even-split structure the references use).
/// `x` is interleaved (re, im) pairs; output likewise.
fn fft(x: &[f64], plan: &FftPlan) -> Vec<f64> {
    let n = x.len() / 2;
    if n == 1 {
        return x.to_vec();
    }
    if n % 2 == 1 {
        return dft(x, plan);
    }
    let half = n / 2;
    let mut even = Vec::with_capacity(half * 2);
    let mut odd = Vec::with_capacity(half * 2);
    for i in 0..half {
        even.extend_from_slice(&x[4 * i..4 * i + 2]);
        odd.extend_from_slice(&x[4 * i + 2..4 * i + 4]);
    }
    let e = fft(&even, plan);
    let o = fft(&odd, plan);
    let mut out = vec![0.0f64; n * 2];
    let tw = plan.twiddles(n);
    for k in 0..half {
        let (s, c) = tw[k];
        let re = c * o[2 * k] - s * o[2 * k + 1];
        let im = s * o[2 * k] + c * o[2 * k + 1];
        out[2 * k] = e[2 * k] + re;
        out[2 * k + 1] = e[2 * k + 1] + im;
        out[2 * (k + half)] = e[2 * k] - re;
        out[2 * (k + half) + 1] = e[2 * k + 1] - im;
    }
    out
}

fn dft(x: &[f64], plan: &FftPlan) -> Vec<f64> {
    let n = x.len() / 2;
    debug_assert_eq!(n, plan.base.0, "planned base size");
    let tab = &plan.base.1;
    let mut out = vec![0.0f64; n * 2];
    for k in 0..n {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for i in 0..n {
            let (s, c) = tab[k * n + i];
            re += x[2 * i] * c - x[2 * i + 1] * s;
            im += x[2 * i] * s + x[2 * i + 1] * c;
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
    out
}

/// Periodic Hann window (`0.5 * (1 - cos(2*pi*i/n))` - offset n, not n-1),
/// matching torch.hann_window's default and the reference preprocessors.
pub fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()))
        .collect()
}

/// Which mel warping the filterbank edges are spaced on. Both are standard;
/// which one a checkpoint wants is a property of its extractor, not of the
/// audio: whisper/Qwen3-ASR declare Slaney (librosa's default), Granite
/// Speech declares HTK (torchaudio's default) - pick from the config, never
/// from habit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MelScale {
    Slaney,
    Htk,
}

/// HTK mel scale: one smooth log curve, `2595 * log10(1 + hz/700)`.
fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// Slaney mel scale: linear below 1000 Hz (3/200 per Hz), logarithmic above
/// (27 steps per ln(6.4)).
fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < 1000.0 {
        hz * 3.0 / 200.0
    } else {
        15.0 + (hz / 1000.0).ln() * 27.0 / 6.4f64.ln()
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < 15.0 {
        mel * 200.0 / 3.0
    } else {
        1000.0 * ((mel - 15.0) * 6.4f64.ln() / 27.0).exp()
    }
}

/// Slaney-scale, Slaney-area-normalized triangular mel filterbank,
/// row-major `[n_mel][n_bins]`, computed and stored in f64. Matches
/// transformers `mel_filter_bank(norm="slaney", mel_scale="slaney")` and
/// llama.cpp's `fill_mel_filterbank_matrix` defaults.
pub fn mel_filterbank_slaney(
    n_mel: usize,
    n_fft: usize,
    sample_rate: f64,
    fmin: f64,
    fmax: f64,
) -> Vec<f64> {
    mel_filterbank(
        n_mel,
        n_fft,
        sample_rate,
        fmin,
        fmax,
        MelScale::Slaney,
        true,
    )
}

/// Triangular mel filterbank, row-major `[n_mel][n_bins]`, in f64.
///
/// `area_norm` is librosa/Slaney's `norm="slaney"`: each triangle is scaled by
/// `2/(f_right - f_left)` so it integrates to a constant. Leaving it off gives
/// unit-peak triangles - torchaudio's `melscale_fbanks(norm=None)`, which is
/// what `torchaudio.transforms.MelSpectrogram` defaults to and therefore what
/// the Granite Speech extractor uses. Getting this pair wrong is silent: the
/// spectrogram still looks like speech, it is just tilted across frequency.
pub fn mel_filterbank(
    n_mel: usize,
    n_fft: usize,
    sample_rate: f64,
    fmin: f64,
    fmax: f64,
    scale: MelScale,
    area_norm: bool,
) -> Vec<f64> {
    let n_bins = n_fft / 2 + 1;
    let (to_mel, to_hz): (fn(f64) -> f64, fn(f64) -> f64) = match scale {
        MelScale::Slaney => (hz_to_mel_slaney, mel_to_hz_slaney),
        MelScale::Htk => (hz_to_mel_htk, mel_to_hz_htk),
    };
    // n_mel + 2 edge frequencies, uniform on the mel axis
    let (mel_lo, mel_hi) = (to_mel(fmin), to_mel(fmax));
    let edges: Vec<f64> = (0..n_mel + 2)
        .map(|i| to_hz(mel_lo + (mel_hi - mel_lo) * i as f64 / (n_mel + 1) as f64))
        .collect();
    let mut fb = vec![0.0f64; n_mel * n_bins];
    for m in 0..n_mel {
        let (f_l, f_c, f_r) = (edges[m], edges[m + 1], edges[m + 2]);
        let enorm = if area_norm { 2.0 / (f_r - f_l) } else { 1.0 };
        for k in 0..n_bins {
            let f = k as f64 * sample_rate / n_fft as f64;
            let rise = (f - f_l) / (f_c - f_l);
            let fall = (f_r - f) / (f_r - f_c);
            let w = rise.min(fall).max(0.0);
            fb[m * n_bins + k] = w * enorm;
        }
    }
    fb
}

/// One STFT frame -> n_mel un-normalized log10-mel values.
///
/// `frame` is exactly `n_fft` samples (caller applies padding semantics),
/// `window` the same length, `fb` from [`mel_filterbank_slaney`]. Power
/// spectrum (re^2 + im^2) over the n_fft/2 + 1 bins, mel dot in f64,
/// floor 1e-10, log10.
pub fn log_mel_frame(frame: &[f64], window: &[f64], fb: &[f64], plan: &FftPlan, out: &mut [f64]) {
    mel_power_frame(frame, window, fb, plan, out);
    for o in out {
        *o = o.max(1e-10).log10();
    }
}

/// Linear mel energies; callers select the model's log/normalization contract.
pub fn mel_power_frame(frame: &[f64], window: &[f64], fb: &[f64], plan: &FftPlan, out: &mut [f64]) {
    let n_fft = frame.len();
    let n_bins = n_fft / 2 + 1;
    debug_assert_eq!(window.len(), n_fft);
    debug_assert_eq!(fb.len(), out.len() * n_bins);
    let mut buf = vec![0.0f64; n_fft * 2];
    for i in 0..n_fft {
        buf[2 * i] = frame[i] * window[i];
    }
    let spec = fft(&buf, plan);
    let mut power = vec![0.0f64; n_bins];
    for (k, p) in power.iter_mut().enumerate() {
        *p = spec[2 * k] * spec[2 * k] + spec[2 * k + 1] * spec[2 * k + 1];
    }
    for (m, o) in out.iter_mut().enumerate() {
        let row = &fb[m * n_bins..(m + 1) * n_bins];
        let mut sum = 0.0f64;
        for k in 0..n_bins {
            sum += row[k] * power[k];
        }
        *o = sum;
    }
}

/// Allocation-free radix-2 frontend. `scratch` holds 2*n complex scalars and
/// n/2+1 powers. The butterfly expressions/trig tables match `fft` above;
/// input bit reversal replaces the recursive temporary vectors, not the FFT.
pub fn mel_power_frame_buffered(
    frame: &[f64],
    window: &[f64],
    fb: &[f64],
    plan: &FftPlan,
    scratch: &mut [f64],
    out: &mut [f64],
) {
    let n = frame.len();
    assert!(n.is_power_of_two() && n > 1);
    assert_eq!(window.len(), n);
    assert_eq!(fb.len(), out.len() * (n / 2 + 1));
    let (buf, power) = scratch.split_at_mut(n * 2);
    assert!(power.len() > n / 2);
    let shift = usize::BITS - n.ilog2();
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        buf[2 * j] = frame[i] * window[i];
        buf[2 * j + 1] = 0.;
    }
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let tw = plan.twiddles(size);
        for base in (0..n).step_by(size) {
            for (k, &(s, c)) in tw.iter().enumerate() {
                let a = 2 * (base + k);
                let b = 2 * (base + k + half);
                let re = c * buf[b] - s * buf[b + 1];
                let im = s * buf[b] + c * buf[b + 1];
                let er = buf[a];
                let ei = buf[a + 1];
                buf[a] = er + re;
                buf[a + 1] = ei + im;
                buf[b] = er - re;
                buf[b + 1] = ei - im;
            }
        }
        size *= 2;
    }
    for (k, p) in power[..=n / 2].iter_mut().enumerate() {
        *p = buf[2 * k] * buf[2 * k] + buf[2 * k + 1] * buf[2 * k + 1];
    }
    for (row, o) in fb.chunks_exact(n / 2 + 1).zip(out) {
        *o = row.iter().zip(&power[..=n / 2]).map(|(f, p)| f * p).sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffered_radix_two_preserves_recursive_fft_arithmetic() {
        for n in [32, 512] {
            let frame: Vec<_> = (0..n)
                .map(|i| ((i * 1337 % 337) as f64 - 168.) / 100.)
                .collect();
            let window = hann_periodic(n);
            let fb = mel_filterbank_slaney(16, n, 16000., 0., 8000.);
            let plan = FftPlan::new(n);
            let mut a = [0.; 16];
            let mut b = [0.; 16];
            let mut scratch = vec![0.; n * 2 + n / 2 + 1];
            mel_power_frame(&frame, &window, &fb, &plan, &mut a);
            mel_power_frame_buffered(&frame, &window, &fb, &plan, &mut scratch, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn fft_matches_dft_on_n400() {
        // impulse + tone mix; the planned FFT of the composite must equal a
        // naive full-length DFT (its own 400-point plan base)
        let n = 400;
        let mut x = vec![0.0f64; n * 2];
        for i in 0..n {
            x[2 * i] = (i as f64 * 0.1).sin() + if i == 3 { 1.0 } else { 0.0 };
        }
        let a = fft(&x, &FftPlan::new(n));
        // a plan for an odd "size" is never split, so force the base case by
        // building the table directly at n
        let full = FftPlan {
            levels: Vec::new(),
            base: (
                n,
                (0..n * n)
                    .map(|ki| {
                        let (k, i) = (ki / n, ki % n);
                        (-2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64).sin_cos()
                    })
                    .collect(),
            ),
        };
        let b = dft(&x, &full);
        for k in 0..n * 2 {
            assert!((a[k] - b[k]).abs() < 1e-9, "bin {k}: {} vs {}", a[k], b[k]);
        }
    }

    #[test]
    fn filterbank_rows_are_normalized_triangles() {
        let fb = mel_filterbank_slaney(128, 400, 16000.0, 0.0, 8000.0);
        // every filter must have positive mass; area-normalized peaks shrink
        // with center frequency spacing, so just sanity-bound the sums
        for m in 0..128 {
            let s: f64 = fb[m * 201..(m + 1) * 201].iter().sum();
            assert!(s > 0.0, "empty mel filter {m}");
        }
    }
}
