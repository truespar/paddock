//! Audio content parts of a chat request: the wire shapes the surfaces accept,
//! their decode to 16 kHz mono through the served model's own frontend, and the
//! prefill rows they cost (the usage split between pictures and clips).

use paddock_engine::service::MmChunk;

use crate::serving::ServingModel;

/// Prefill rows a request's clips take, for the usage split: exact on Gemma 4
/// 12B (its frames plus the `<|audio>` / `<audio|>` pair the engine adds
/// around each - it also takes pictures, so the two must be told apart),
/// "all of them" (usize::MAX) on the audio-only families, 0 without audio.
pub(crate) fn audio_rows(model: &ServingModel, mm_chunks: Option<&[MmChunk]>) -> usize {
    match model.audio_frontend {
        crate::serving::AudioFrontend::None => 0,
        crate::serving::AudioFrontend::Gemma4Raw => mm_chunks
            .unwrap_or_default()
            .iter()
            .map(|c| match c {
                MmChunk::Audio { samples, .. } => {
                    model.audio_frontend.prompt_rows(samples.len()) + 2
                }
                _ => 0,
            })
            .sum(),
        _ => usize::MAX,
    }
}

/// One audio content part: the base64 payload and the declared format (used
/// only for error messages - the decoder sniffs the bytes).
#[derive(Debug)]
pub(crate) struct AudioRef {
    pub b64: String,
    pub format: String,
}

/// Find audio content parts, in render order. Accepted shapes:
/// OpenAI chat's `{"type":"input_audio","input_audio":{"data":b64,"format":..}}`
/// and vLLM's `audio_url` extension carrying a
/// `data:audio/...` URI (string or `{url}`). Remote URLs are refused for the
/// same SSRF/availability reason images refuse them. Detection only - the
/// capability check and decode happen at the call site so an audio-less
/// server gives the useful error.
pub(crate) fn find_audio(messages: &[serde_json::Value]) -> Result<Vec<AudioRef>, String> {
    let mut out = Vec::new();
    for msg in messages {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            if let Some(ia) = part.get("input_audio") {
                let data = ia
                    .get("data")
                    .and_then(|d| d.as_str())
                    .ok_or("input_audio part has no data")?;
                let format = ia
                    .get("format")
                    .and_then(|f| f.as_str())
                    .unwrap_or("wav")
                    .to_owned();
                out.push(AudioRef {
                    b64: data.to_owned(),
                    format,
                });
                continue;
            }
            if let Some(au) = part.get("audio_url") {
                let url = au
                    .as_str()
                    .or_else(|| au.get("url").and_then(|u| u.as_str()))
                    .ok_or("audio_url part has no url")?;
                let Some(rest) = url.strip_prefix("data:audio/") else {
                    return Err(
                        "only data:audio/... URIs are supported (the server does not fetch \
                         remote audio); inline the clip as base64"
                            .into(),
                    );
                };
                let (meta, b64) = rest.split_once(',').ok_or("malformed data:audio URI")?;
                let format = meta.split(';').next().unwrap_or("wav").to_owned();
                out.push(AudioRef {
                    b64: b64.to_owned(),
                    format,
                });
                continue;
            }
            // a bare `{"type":"audio"}` marker carries no payload - that shape
            // only exists POST-normalization; a client sending it meant to
            // send one of the two payload-bearing shapes
            if part.get("type").and_then(|t| t.as_str()) == Some("audio") {
                return Err("audio content part has no data (send `input_audio` or a \
                            data:audio `audio_url`)"
                    .into());
            }
        }
    }
    Ok(out)
}

/// Decode audio parts to 16 kHz mono f32 - the exact pipeline
/// `/v1/audio/transcriptions` runs. Every container OpenAI's transcription
/// endpoint accepts works here too, which is a deliberate SUPERSET of this
/// surface's own spec: OpenAI's chat `input_audio` part declares only wav and
/// mp3, and refusing a flac a caller already holds - on the one endpoint that
/// takes the same bytes as its sibling - would be a distinction with no reason
/// behind it. The declared `format` is recorded in the refusal but never
/// trusted: the container comes from the bytes.
pub(crate) fn decode_audio_parts(
    refs: Vec<AudioRef>,
    frontend: crate::serving::AudioFrontend,
) -> Result<Vec<MmChunk>, String> {
    use base64::Engine as _;
    refs.into_iter()
        .map(|r| {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(r.b64.trim())
                .map_err(|e| format!("audio base64: {e}"))?;
            let wav = paddock_engine::audio::decode::decode_audio(&bytes)
                .map_err(|e| format!("audio decode (declared format {:?}): {e}", r.format))?;
            if wav.samples.is_empty() {
                return Err("audio part holds no samples".into());
            }
            let samples =
                paddock_engine::audio::resample::resample(&wav.samples, wav.sample_rate, 16000)?;
            // mel here, not engine-side: the request thread pays
            // the host DSP so concurrent requests' frontends run in parallel
            // instead of serializing on the engine thread. Which contract
            // runs is the served model's, never a default - see
            // `AudioFrontend`.
            let mel = frontend.mel(&samples)?;
            Ok(MmChunk::Audio { samples, mel })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::build_mm_chunks;

    #[test]
    fn find_audio_takes_both_wire_shapes_in_order() {
        let msgs = [serde_json::json!({"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "QUFB", "format": "wav"}},
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,QkJC"}},
            {"type": "audio_url", "audio_url": "data:audio/wav;base64,Q0ND"},
        ]})];
        let refs = find_audio(&msgs).expect("find");
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0].b64, "QUFB");
        assert_eq!(refs[1].b64, "QkJC");
        assert_eq!(refs[2].b64, "Q0ND");
    }

    #[test]
    fn remote_audio_urls_are_refused() {
        let msgs = [serde_json::json!({"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}},
        ]})];
        let err = find_audio(&msgs).unwrap_err();
        assert!(err.contains("does not fetch remote audio"), "{err}");
    }

    #[test]
    fn a_payloadless_audio_marker_is_refused() {
        let msgs = [serde_json::json!({"role": "user", "content": [
            {"type": "audio"},
        ]})];
        let err = find_audio(&msgs).unwrap_err();
        assert!(err.contains("no data"), "{err}");
    }

    #[test]
    fn audio_chunks_split_at_the_pad_token() {
        let clip = MmChunk::Audio {
            samples: vec![0.0; 160],
            mel: None,
        };
        let mm = build_mm_chunks(&[10, 99, 11], 99, vec![clip], None).expect("chunks");
        assert_eq!(mm.text_ids, vec![10, 11]);
        assert!(matches!(&mm.chunks[1], MmChunk::Audio { samples, .. } if samples.len() == 160));
    }

    #[test]
    fn decode_audio_parts_round_trips_wav() {
        // 16 kHz mono PCM16 WAV, 4 samples
        let mut wav: Vec<u8> = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36u32 + 8).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&8u32.to_le_bytes());
        for s in [0i16, 16384, -16384, 0] {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
        let out = decode_audio_parts(
            vec![AudioRef {
                b64,
                format: "wav".into(),
            }],
            crate::serving::AudioFrontend::Qwen3Asr,
        )
        .expect("decode");
        assert_eq!(out.len(), 1);
        let MmChunk::Audio { samples, mel } = &out[0] else {
            panic!("not audio")
        };
        assert_eq!(samples.len(), 4);
        assert!((samples[1] - 0.5).abs() < 1e-3);
        // the runner-side frontend rode along
        assert!(mel.is_some());
    }

    /// The frontend is the SERVED MODEL's, never a default. The two contracts
    /// share no geometry, so the wrong one hands the tower a plausible plane
    /// of the wrong width - which is a width check away from being noise
    /// transcribed with a straight face.
    #[test]
    fn the_audio_frontend_is_the_models_own_contract() {
        use crate::serving::AudioFrontend;
        // 1 s of 16 kHz mono: long enough that both frontends produce real
        // frames (granite's pair-stacking gives 0 for a handful of samples)
        let mut wav = b"RIFF".to_vec();
        let n = 16000usize;
        wav.extend_from_slice(&(36 + 2 * n as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(2 * n as u32).to_le_bytes());
        for i in 0..n {
            let v = ((i as f32 * 0.05).sin() * 12000.0) as i16;
            wav.extend_from_slice(&v.to_le_bytes());
        }
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
        let mel_for = |fe: AudioFrontend| {
            let out = decode_audio_parts(
                vec![AudioRef {
                    b64: b64.clone(),
                    format: "wav".into(),
                }],
                fe,
            )
            .expect("decode");
            let MmChunk::Audio { mel, .. } = &out[0] else {
                panic!("not audio")
            };
            mel.clone().expect("frontend ran on the request thread")
        };

        // Qwen3-ASR: 128 Slaney mels at 100 frames/s, and the emitted plane
        // runs on to the conv stem's 100-frame chunk boundary (so its data is
        // WIDER than n_frames rows - that padding is the stem's, not audio).
        let q = mel_for(AudioFrontend::Qwen3Asr);
        assert_eq!(q.n_frames, paddock_engine::audio::real_frames(n));
        assert_eq!(q.data.len() % paddock_engine::audio::N_MEL, 0);
        assert!(q.data.len() >= q.n_frames * paddock_engine::audio::N_MEL);

        // granite-speech: 80 HTK mels pair-stacked to 160 wide at 50 frames/s,
        // exactly n_frames rows and nothing else.
        let g = mel_for(AudioFrontend::GraniteSpeech);
        assert_eq!(
            g.n_frames,
            paddock_engine::audio::granite::encoder_frames(n)
        );
        assert_eq!(
            g.data.len(),
            g.n_frames * paddock_engine::audio::granite::INPUT_DIM
        );
        assert_eq!(
            g.n_frames * 2,
            q.n_frames,
            "granite runs at half the frame rate"
        );

        // and the row-count rules the context gate quotes agree with them
        assert_eq!(
            AudioFrontend::GraniteSpeech.prompt_rows(n),
            g.n_frames
                .div_ceil(paddock_engine::audio::granite::WINDOW_SIZE)
                * 3
        );
        assert_eq!(AudioFrontend::None.prompt_rows(n), 0);
        assert!(AudioFrontend::None.features(&[0.0; 16]).is_err());
    }

    /// Gemma 4 12B takes the clip itself: one row per 640-sample frame (the
    /// feature extractor's ceil - the last frame zero-padded), no mel, and
    /// the empty clip is refused rather than sent as zero rows.
    #[test]
    fn gemma4_raw_audio_is_frames_not_a_spectrogram() {
        use crate::serving::AudioFrontend;
        let fe = AudioFrontend::Gemma4Raw;
        assert_eq!(fe.prompt_rows(640), 1);
        assert_eq!(fe.prompt_rows(641), 2);
        assert_eq!(fe.prompt_rows(16_000), 25, "25 tokens a second");
        assert!(fe.mel(&[0.25; 1000]).unwrap().is_none());
        assert!(fe.mel(&[]).is_err());
        assert!(
            fe.features(&[0.0; 640]).is_err(),
            "no mel exists for this lane"
        );
        // the spectrogram families still hand the engine their mel
        assert!(
            AudioFrontend::Qwen3Asr
                .mel(&[0.0; 16_000])
                .unwrap()
                .is_some()
        );
    }

    /// The usage split: the 12B's media rows are pictures and clips at once,
    /// so `audio_tokens` is the clips' share and `image_tokens` the rest; an
    /// audio-only family still reports all of its media as audio.
    #[test]
    fn media_usage_splits_pictures_from_clips() {
        use paddock_api::completions::Usage;
        let v = Usage::media_details(441, 0, 414, 152).unwrap();
        assert_eq!(v["audio_tokens"], 152);
        assert_eq!(v["image_tokens"], 262);
        let v = Usage::media_details(290, 0, 266, usize::MAX).unwrap();
        assert_eq!(v["audio_tokens"], 266);
        assert!(v.get("image_tokens").is_none());
        let v = Usage::media_details(288, 0, 268, 0).unwrap();
        assert_eq!(v["image_tokens"], 268);
        assert!(v.get("audio_tokens").is_none());
    }

    /// `max_clip_s` is `prompt_rows` run backwards, and /server publishes it
    /// so a client can split-or-skip before sending an hour of audio it was
    /// always going to refuse. The property that matters is the
    /// round trip: the answer FITS, and one second more does not.
    #[test]
    fn max_clip_s_is_the_exact_inverse_of_prompt_rows() {
        use crate::serving::AudioFrontend;
        let rate = paddock_engine::audio::SAMPLE_RATE;
        for fe in [AudioFrontend::Qwen3Asr, AudioFrontend::GraniteSpeech] {
            for rows in [1_024usize, 8_192, 32_768] {
                let s = fe
                    .max_clip_s(rows)
                    .expect("a generative lane has a ceiling");
                let at = (s * rate as f64) as usize;
                assert!(fe.prompt_rows(at) <= rows, "{fe:?}@{rows}: {s}s fits");
                assert!(
                    fe.prompt_rows(at + rate) > rows,
                    "{fe:?}@{rows}: {s}s is the LAST second that fits",
                );
            }
        }
        // Qwen3-ASR bills 13 rows a second, which is the whole reason this cap
        // is per-model: a 32k server hears ~42 minutes, not "an hour or so".
        let s = AudioFrontend::Qwen3Asr.max_clip_s(32_768).unwrap();
        assert!((2400.0..2600.0).contains(&s), "~42 min at 32k, got {s}s");
        // Whisper takes this path only via `None` - it windows, so it has no
        // context ceiling and publishes nothing rather than a made-up number.
        assert_eq!(AudioFrontend::None.max_clip_s(32_768), None);
        assert_eq!(AudioFrontend::Qwen3Asr.max_clip_s(0), None);
    }

    #[test]
    fn qwen_asr_metal_caps_frontend_without_changing_cuda() {
        use crate::serving::AudioFrontend;
        let metal = AudioFrontend::Qwen3AsrMetal;
        assert_eq!(metal.max_clip_s(8192), Some(120.));
        assert!(AudioFrontend::Qwen3Asr.max_clip_s(8192).unwrap() > 120.);
        let x = vec![0.; 4800];
        let a = metal.features(&x).unwrap();
        let b = AudioFrontend::Qwen3Asr.features(&x).unwrap();
        assert_eq!(a.data, b.data);
        assert_eq!(a.n_frames, b.n_frames);
        assert!(metal.features(&vec![0.; 120 * 16000 + 1]).is_err());
        assert!(metal.max_clip_s(30).unwrap() < 120.);
    }

    #[test]
    fn granite_speech_metal_caps_frontend_without_changing_cuda() {
        use crate::serving::AudioFrontend;
        let metal = AudioFrontend::GraniteSpeechMetal;
        assert_eq!(metal.max_clip_s(4096), Some(120.));
        assert!(AudioFrontend::GraniteSpeech.max_clip_s(4096).unwrap() > 120.);
        let x = vec![0.; 4800];
        let a = metal.features(&x).unwrap();
        let b = AudioFrontend::GraniteSpeech.features(&x).unwrap();
        assert_eq!(a.data, b.data);
        assert_eq!(a.n_frames, b.n_frames);
        assert!(metal.features(&vec![0.; 120 * 16000 + 1]).is_err());
        assert!(metal.max_clip_s(30).unwrap() < 120.);
    }
}
