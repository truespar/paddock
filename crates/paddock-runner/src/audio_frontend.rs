//! The host side of a served audio model's input: which mel contract it
//! consumes, how many prompt rows a clip costs, and the longest clip a
//! window can hold - one value per family, read by every audio route.

/// Which host mel contract a served audio model consumes.
///
/// The frontend runs on the runner's blocking pool rather than engine-side
/// so the runner has to know which one - and they agree on
/// nothing: 400-point transform / 128 Slaney mels / 100 frames per second
/// for Qwen3-ASR, 512-point / 80 HTK mels / pair-stacked 50 frames per
/// second for granite-speech. Feeding one to the other's tower is caught by
/// a width check rather than transcribed as noise, but the point of naming
/// the choice here is that it never gets that far.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AudioFrontend {
    /// this model takes no audio
    #[default]
    None,
    /// transformers `Qwen3ASRFeatureExtractor` (whisper-derived)
    Qwen3Asr,
    /// Same author frontend, bounded by the native Metal tower's admission.
    Qwen3AsrMetal,
    /// transformers `GraniteSpeechFeatureExtractor` (torchaudio-derived)
    GraniteSpeech,
    /// Same extractor, bounded by the experimental native Metal tower.
    GraniteSpeechMetal,
    /// Gemma 4 12B's `Gemma4UnifiedAudioFeatureExtractor`: no spectrogram at
    /// all - the engine takes the raw clip in 640-sample (40 ms) frames, one
    /// prompt row each.
    Gemma4Raw,
    /// Nemotron 3.5 Transcribe: the engine computes its log-mel on the GPU
    /// (the checkpoint's own window and filterbank), one prompt row per
    /// 80 ms.
    Transcribe,
}

impl AudioFrontend {
    /// The clip's features as `MmChunk::Audio` carries them: the mel for the
    /// spectrogram families, None for the raw-frame one (whose engine reads
    /// the samples themselves).
    pub fn mel(
        self,
        samples: &[f32],
    ) -> Result<Option<paddock_engine::audio::MelFeatures>, String> {
        match self {
            AudioFrontend::Gemma4Raw | AudioFrontend::Transcribe => {
                if samples.is_empty() {
                    Err("audio clip holds no samples".into())
                } else {
                    Ok(None)
                }
            }
            _ => self.features(samples).map(Some),
        }
    }

    /// Run this frontend over a 16 kHz mono clip.
    pub fn features(self, samples: &[f32]) -> Result<paddock_engine::audio::MelFeatures, String> {
        match self {
            AudioFrontend::Qwen3AsrMetal if samples.len() > 120 * 16000 => {
                Err("Qwen3-ASR Metal clip exceeds the 120-second implementation limit".into())
            }
            AudioFrontend::Qwen3Asr | AudioFrontend::Qwen3AsrMetal => {
                paddock_engine::audio::qwen3_asr_features(samples)
            }
            AudioFrontend::GraniteSpeechMetal if samples.len() > 120 * 16000 => {
                Err("Granite Speech Metal clip exceeds the 120-second implementation limit".into())
            }
            AudioFrontend::GraniteSpeech | AudioFrontend::GraniteSpeechMetal => {
                paddock_engine::audio::granite::speech_features(samples)
            }
            AudioFrontend::Gemma4Raw => {
                Err("Gemma 4 12B takes raw audio frames, not a spectrogram".into())
            }
            AudioFrontend::Transcribe => {
                Err("Nemotron 3.5 Transcribe computes its log-mel on the GPU".into())
            }
            AudioFrontend::None => Err("this model does not take audio input".into()),
        }
    }

    /// Prompt rows a clip of `len` samples will occupy - the same rule the
    /// engine's admission uses, so the context check the handler runs against
    /// it is the one that will actually bind.
    pub fn prompt_rows(self, len: usize) -> usize {
        match self {
            AudioFrontend::Qwen3Asr | AudioFrontend::Qwen3AsrMetal => {
                paddock_engine::audio::audio_tokens_for_samples(len)
            }
            AudioFrontend::GraniteSpeech | AudioFrontend::GraniteSpeechMetal => {
                paddock_engine::audio::granite::audio_tokens_for_samples(len)
            }
            AudioFrontend::Gemma4Raw => len.div_ceil(640),
            AudioFrontend::Transcribe => paddock_models::transcribe::audio_tokens(len),
            AudioFrontend::None => 0,
        }
    }

    /// The longest clip, in seconds, whose audio rows still fit in `rows` -
    /// `prompt_rows` run backwards.
    ///
    /// A generative ASR lane spends the whole clip as prompt rows, so its
    /// ceiling is the context window and it is far lower than people expect:
    /// Qwen3-ASR bills 13 rows a second, which puts a 32k-token server at
    /// roughly 42 minutes and an 8k one at ten. Whisper has no such ceiling -
    /// it windows - and answers `None` from the caller.
    ///
    /// Found by bisection rather than algebra on PURPOSE. The forward rule is
    /// a chunked convolution stack whose closed form differs per family and
    /// has already changed once; bisecting the real function cannot drift from
    /// it, and this runs once per `/server` call.
    pub fn max_clip_s(self, rows: usize) -> Option<f64> {
        if self == AudioFrontend::None || rows == 0 {
            return None;
        }
        let rate = paddock_engine::audio::SAMPLE_RATE;
        // 24 h is a ceiling for the search, not a promise - anything near it
        // means the context is enormous and the answer stops being the binding
        // constraint anyway.
        let (mut lo, mut hi) = (
            0usize,
            if matches!(
                self,
                AudioFrontend::GraniteSpeechMetal | AudioFrontend::Qwen3AsrMetal
            ) {
                rate * 120
            } else {
                rate * 60 * 60 * 24
            },
        );
        if self.prompt_rows(hi) <= rows {
            return Some(hi as f64 / rate as f64);
        }
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.prompt_rows(mid) <= rows {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Some(lo as f64 / rate as f64)
    }
}
