//! Every container `/v1/audio/transcriptions` accepts decodes to real audio.
//!
//! The fixtures are one 0.5 s 440 Hz tone at 16 kHz mono, encoded eight ways
//! (`tests/data/audio-formats`, made with ffmpeg - see the commit that added
//! them). Half a second because MP3 and AAC prepend encoder priming and pad
//! the tail: a shorter clip and the padding is most of the file.
//!
//! What each case asserts is that we got the TONE back, not merely that the
//! decoder returned `Ok` with some samples in it. A decoder wired to the wrong
//! container, or handed the wrong channel layout, or reading the bytes at the
//! wrong stride, still returns a plausible-looking buffer - it just does not
//! contain a 440 Hz sine. A Goertzel at 440 against a control bin catches
//! every one of those; a length check catches none of them.

use paddock_engine::audio::decode::{decode_audio, decode_audio_bounded};

/// Power at one frequency, by the Goertzel recurrence - a single-bin DFT, and
/// all we need to ask "is this the tone we encoded?".
fn goertzel(x: &[f32], rate: f32, freq: f32) -> f32 {
    let k = 2.0 * std::f32::consts::PI * freq / rate;
    let coeff = 2.0 * k.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &v in x {
        let s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2) / x.len() as f32
}

fn fixture(name: &str) -> Vec<u8> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/audio-formats")
        .join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("fixture {}: {e}", p.display()))
}

#[test]
fn every_accepted_container_decodes_to_the_same_tone() {
    // The two OPUS rows differ from each other, which is worth pinning rather
    // than smoothing over: the same encoder and the same 16 kHz source come
    // back at 48 kHz from Ogg and at 16 kHz from WebM. Ogg-Opus's mapping
    // fixes the output rate at 48 kHz by convention (the OpusHead's rate field
    // describes the INPUT), while Matroska carries the track's own sampling
    // frequency and libopus decodes straight to it - 16 kHz is one of its
    // legal output rates. Both are correct; the transcription path resamples
    // to 16 kHz afterwards either way. Measured, not assumed: if a symphonia
    // or adapter upgrade changes either of these, this test says so.
    let cases: &[(&str, u32)] = &[
        ("tone.wav", 16000),
        ("tone.flac", 16000),
        ("tone.mp3", 16000),
        ("tone.m4a", 16000),
        ("tone.mp4", 16000),
        ("tone.ogg", 16000),
        ("tone.opus.ogg", 48000),
        ("tone.webm", 16000),
    ];

    for (name, want_rate) in cases {
        let a = decode_audio(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(a.sample_rate, *want_rate, "{name}: sample rate");

        let secs = a.samples.len() as f32 / a.sample_rate as f32;
        assert!(
            (0.4..0.9).contains(&secs),
            "{name}: decoded {secs:.3} s from a 0.5 s clip (codec padding is allowed, this is not)"
        );

        let rate = a.sample_rate as f32;
        let tone = goertzel(&a.samples, rate, 440.0);
        // 1 kHz is silent in the source, so it measures whatever the decode
        // invented. A real 440 Hz sine beats it by orders of magnitude; the
        // 20x floor leaves room for the lossy codecs' quantisation noise.
        let control = goertzel(&a.samples, rate, 1000.0);
        assert!(
            tone > control * 20.0,
            "{name}: 440 Hz power {tone:.4e} vs 1 kHz {control:.4e} - decoded, but not the tone"
        );
    }
}

/// The one format whose bytes must not change hands: RIFF is what the parity
/// gates and the benchmarks feed in, and it stays on our own decoder. Proven by
/// value rather than by reading the code - both paths must agree exactly.
#[test]
fn wav_decodes_identically_through_both_doors() {
    let bytes = fixture("tone.wav");
    let via_router = decode_audio(&bytes).expect("router");
    let direct = paddock_engine::audio::wav::decode_wav(&bytes).expect("wav");
    assert_eq!(via_router.sample_rate, direct.sample_rate);
    assert_eq!(via_router.samples, direct.samples);
}

/// A BROWSER's webm, which is not the same file as ffmpeg's webm.
///
/// `tone.webm` was muxed by a tool that knew the whole clip before it wrote a
/// byte, so every element carries its size. A microphone recording cannot: the
/// file streams out as it is spoken, so `Segment` and every `Cluster` carry
/// EBML's "unknown size" instead. That shape is legal, it is what every browser
/// produces, and symphonia 0.6.0 refuses it at the second cluster - which is
/// why a recording longer than about two seconds used to come back as
/// "malformed stream: mkv (ebml): encountered an unexpected element" while
/// other decoders handled the very same bytes.
///
/// The fixture is turned back into the file a browser would have written, and
/// then has to decode to the same SAMPLES as the fixture it came from. That is
/// the assertion with teeth: filling a size back in is only allowed to write
/// down what the muxer already implied, so a cluster that swallowed the
/// trailing `Cues`, or a segment measured to the wrong end, changes the audio
/// that comes out - it does not merely fail to parse.
#[test]
fn a_live_muxed_webm_decodes_to_the_same_tone() {
    // Offsets into the committed fixture. The id assertions below are the
    // tripwire: regenerate tone.webm and this test says so instead of quietly
    // unsealing the wrong bytes.
    const SEGMENT_ID: usize = 36;
    const SEGMENT_SIZE: (usize, usize) = (40, 8);
    const CLUSTER_ID: usize = 497;
    const CLUSTER_SIZE: (usize, usize) = (501, 2);

    let sealed = fixture("tone.webm");
    assert_eq!(
        &sealed[SEGMENT_ID..SEGMENT_ID + 4],
        &[0x18, 0x53, 0x80, 0x67],
        "Segment moved"
    );
    assert_eq!(
        &sealed[CLUSTER_ID..CLUSTER_ID + 4],
        &[0x1F, 0x43, 0xB6, 0x75],
        "Cluster moved"
    );

    // Unwrite both sizes: the marker bit, then every value bit set.
    let mut live = sealed.clone();
    for (at, w) in [SEGMENT_SIZE, CLUSTER_SIZE] {
        live[at..at + w].fill(0xFF);
        // Widened: at w == 8 the first byte is pure marker, and `0xFFu8 >> 8`
        // would overflow rather than clear.
        live[at] = ((0x80u16 >> (w - 1)) | (0xFFu16 >> w)) as u8;
    }
    assert_ne!(live, sealed, "the fixture must actually have been unsealed");

    let a = decode_audio(&live).expect("a browser-shaped webm must decode");
    let b = decode_audio(&sealed).expect("the fixture itself still decodes");
    assert_eq!(a.sample_rate, b.sample_rate);
    assert_eq!(
        a.samples, b.samples,
        "live muxing must not change one sample"
    );

    let tone = goertzel(&a.samples, a.sample_rate as f32, 440.0);
    let control = goertzel(&a.samples, a.sample_rate as f32, 1000.0);
    assert!(
        tone > control * 20.0,
        "440 Hz {tone:.4e} vs 1 kHz {control:.4e} - not the tone"
    );
}

/// Lossless in, lossless out: FLAC is the same PCM as the WAV it came from, so
/// a FLAC upload has to transcribe bit-identically to its WAV twin. This is the
/// case that would catch a resampler or a downmix quietly entering the path.
#[test]
fn flac_is_bit_identical_to_the_wav_it_was_encoded_from() {
    let wav = decode_audio(&fixture("tone.wav")).expect("wav");
    let flac = decode_audio(&fixture("tone.flac")).expect("flac");
    assert_eq!(flac.sample_rate, wav.sample_rate);
    assert_eq!(flac.samples.len(), wav.samples.len(), "flac length");
    for (i, (f, w)) in flac.samples.iter().zip(&wav.samples).enumerate() {
        assert!((f - w).abs() < 1e-6, "flac sample {i}: {f} != {w}");
    }
}

#[test]
fn bounded_decode_matches_the_lenient_door_and_stops_at_the_bound() {
    for name in ["tone.wav", "tone.mp3", "tone.flac"] {
        let bytes = fixture(name);
        let a = decode_audio(&bytes).expect(name);
        let b = decode_audio_bounded(&bytes, 3600).expect(name);
        assert_eq!(a.sample_rate, b.sample_rate, "{name}");
        assert_eq!(a.samples, b.samples, "{name}");
        let Err(err) = decode_audio_bounded(&bytes, 0) else {
            panic!("{name}: decoded past the bound");
        };
        assert!(err.contains("exceeds 0 seconds"), "{name}: {err}");
    }
}
