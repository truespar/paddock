//! Container/codec decode for the transcription surfaces - the OpenAI file
//! list (`flac mp3 mp4 mpeg mpga m4a ogg wav webm`) down to the mono f32 the
//! ASR frontends eat.
//!
//! Why there is A DEPENDENCY here at all. `wav.rs` says it stays in-house
//! because "the parity requirement makes every sample-level transform part of
//! the numeric contract". That argument covers the transforms we own - the
//! resampler, the mel - not the codec: an MP3 frame has one defined meaning
//! and reimplementing five decoders to get it would be a project, not a
//! feature. The reverse argument is stronger. **libopus, and the rest through
//! symphonia, is what every engine we compare against decodes with** (ffmpeg
//! behind vLLM, whisper.cpp, the standard benchmark loaders), so reusing them
//! makes our
//! PCM bit-identical to the arbiter's instead of approximately equal. A
//! from-scratch decoder would put a difference of our own making inside the
//! one comparison the ASR gates exist to run.
//!
//! WAV does not COME through here. RIFF still goes to `wav::decode_wav`: it
//! is the format every parity gate and every board runs on, its numeric
//! contract is pinned, and rerouting the one measured format through a new
//! library to gain nothing is how a board silently moves.
//!
//! The container is decided by the BYTES, never by the declared media type or
//! the file name - the same rule the image path follows, and for the same
//! reason (a browser's `audio/webm` and a user's `.mp3` that is really an m4a
//! are both routine).

use std::io::Cursor;

use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::errors::Error as SymphErr;
use symphonia::core::formats::probe::{Hint, Probe};
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia_adapter_libopus::OpusDecoder;

use super::wav::{WavAudio, decode_wav};

/// What the first bytes say the file is. Only what we can actually serve gets
/// a variant - everything else is `Unknown` and refuses by name, because
/// "unsupported file" with no noun in it is the error that makes a user guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    /// RIFF/WAVE - ours, not symphonia's.
    Wav,
    Flac,
    /// Ogg: Vorbis or Opus, decided by the codec inside.
    Ogg,
    /// Matroska/WebM: Opus or Vorbis in practice (what MediaRecorder writes).
    Webm,
    /// ISO base media: mp4 / m4a / mpga-in-mp4, AAC-LC or ALAC.
    Mp4,
    /// Bare MPEG audio stream: mp3 / mpeg / mpga, with or without an ID3 tag.
    Mp3,
    /// Known formats we deliberately do not enable, kept apart from Unknown
    /// so the refusal can name the thing the user actually handed us.
    Aiff,
    Caf,
    Unknown,
}

impl Container {
    /// The extension symphonia's probe takes as a hint. Only a hint: the probe
    /// re-checks markers itself, so a wrong guess costs time, not correctness.
    fn hint(self) -> Option<&'static str> {
        match self {
            Container::Flac => Some("flac"),
            Container::Ogg => Some("ogg"),
            Container::Webm => Some("webm"),
            Container::Mp4 => Some("mp4"),
            Container::Mp3 => Some("mp3"),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Container::Wav => "wav",
            Container::Flac => "flac",
            Container::Ogg => "ogg",
            Container::Webm => "webm/mkv",
            Container::Mp4 => "mp4/m4a",
            Container::Mp3 => "mp3",
            Container::Aiff => "aiff",
            Container::Caf => "caf",
            Container::Unknown => "unknown",
        }
    }
}

/// The accepted set, in one place, so the refusal text and the doc comment
/// cannot drift apart. This is OpenAI's list verbatim.
pub const ACCEPTED: &str = "flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm";

fn sniff(b: &[u8]) -> Container {
    let at = |i: usize, tag: &[u8]| b.len() >= i + tag.len() && &b[i..i + tag.len()] == tag;
    if at(0, b"RIFF") && at(8, b"WAVE") {
        return Container::Wav;
    }
    if at(0, b"fLaC") {
        return Container::Flac;
    }
    if at(0, b"OggS") {
        return Container::Ogg;
    }
    // EBML header - Matroska and WebM share it; the DocType inside decides,
    // and symphonia's mkv reader handles both.
    if at(0, &[0x1A, 0x45, 0xDF, 0xA3]) {
        return Container::Webm;
    }
    // ISO base media: a `ftyp` box at offset 4 (the box length precedes it).
    if at(4, b"ftyp") {
        return Container::Mp4;
    }
    if at(0, b"FORM") && (at(8, b"AIFF") || at(8, b"AIFC")) {
        return Container::Aiff;
    }
    if at(0, b"caff") {
        return Container::Caf;
    }
    // MPEG audio: an ID3v2 tag ahead of the stream, or a bare frame sync
    // (11 set bits) - `FF Ex/Fx`, which is every MPEG-1/2/2.5 layer.
    if at(0, b"ID3") {
        return Container::Mp3;
    }
    if b.len() >= 2 && b[0] == 0xFF && (b[1] & 0xE0) == 0xE0 {
        return Container::Mp3;
    }
    Container::Unknown
}

/// symphonia's default codecs plus Opus, built once. Opus is the one format on
/// OpenAI's list symphonia cannot decode itself (its `symphonia-codec-opus` is
/// a reserved name with status "-"), and it is also the one a browser records
/// in - so the adapter is not an edge case, it is the microphone path.
fn codecs() -> &'static CodecRegistry {
    static REG: std::sync::OnceLock<CodecRegistry> = std::sync::OnceLock::new();
    REG.get_or_init(|| {
        let mut reg = CodecRegistry::new();
        symphonia::default::register_enabled_codecs(&mut reg);
        reg.register_audio_decoder::<OpusDecoder>();
        reg
    })
}

fn probe() -> &'static Probe {
    static PROBE: std::sync::OnceLock<Probe> = std::sync::OnceLock::new();
    PROBE.get_or_init(|| {
        let mut p = Probe::default();
        symphonia::default::register_enabled_formats(&mut p);
        p
    })
}

/// Decode any accepted audio file to mono f32 at its own sample rate. The
/// caller resamples (`resample::resample`) - this returns what the file holds.
///
/// Errors are user-facing: they surface as 400s on the transcription
/// endpoints, so each one names the format that arrived and what would work.
pub fn decode_audio(bytes: &[u8]) -> Result<WavAudio, String> {
    decode_with_limit(bytes, None)
}

/// Bound decoded duration as packets arrive, before resampling or queueing.
/// Also reject damaged/changing streams: dropping packets would silently
/// shift the speaker timeline. The older transcription policy is unchanged.
pub fn decode_audio_limited(bytes: &[u8], seconds: u32) -> Result<WavAudio, String> {
    decode_with_limit(bytes, Some(seconds))
}

fn decode_with_limit(bytes: &[u8], seconds: Option<u32>) -> Result<WavAudio, String> {
    let kind = sniff(bytes);
    match kind {
        Container::Wav => {
            let audio = decode_wav(bytes)?;
            check_duration(audio.samples.len(), audio.sample_rate, seconds)?;
            return Ok(audio);
        }
        Container::Aiff | Container::Caf => {
            return Err(format!(
                "this is an {} file, which this build does not decode (accepted: {ACCEPTED})",
                kind.name()
            ));
        }
        Container::Unknown => {
            return Err(format!(
                "unrecognised audio container (accepted: {ACCEPTED}); the format is read from \
                 the file's own bytes, so a wrong extension is not the problem"
            ));
        }
        _ => {}
    }

    // A browser's recording arrives with its sizes unwritten, which symphonia
    // 0.6.0 refuses at the second cluster - seal them first. Sealing only
    // rewrites element size fields, never audio, and returns None (no copy) for
    // the ordinary files everything else produces.
    let owned = match kind {
        Container::Webm => {
            super::webm_live::seal_live_sizes(bytes).unwrap_or_else(|| bytes.to_vec())
        }
        _ => bytes.to_vec(),
    };
    let mss = MediaSourceStream::new(Box::new(Cursor::new(owned)), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = kind.hint() {
        hint.with_extension(ext);
    }
    let mut fmt = probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("{} container: {e}", kind.name()))?;

    let track = fmt
        .default_track(TrackType::Audio)
        .ok_or_else(|| format!("{} file holds no audio track", kind.name()))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| format!("{} track declares no codec parameters", kind.name()))?
        .clone();
    let mut dec = codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| {
            format!(
                "{} container carries a codec this build cannot decode ({e}); accepted: {ACCEPTED}",
                kind.name()
            )
        })?;

    let mut samples: Vec<f32> = Vec::new();
    let mut rate = 0u32;
    let mut inter: Vec<f32> = Vec::new();
    loop {
        let packet = match fmt.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            // A track-list change mid-stream (chained OGG). Everything decoded
            // up to here is real audio, so keep it rather than failing the
            // request - and stop, because the decoder past this point is stale.
            Err(SymphErr::ResetRequired) if seconds.is_some() => {
                return Err("audio track changes mid-stream; convert to a single WAV first".into());
            }
            Err(SymphErr::ResetRequired) => break,
            Err(e) => return Err(format!("{} stream: {e}", kind.name())),
        };
        if packet.track_id != track_id {
            continue;
        }
        let buf = match dec.decode(&packet) {
            Ok(b) => b,
            // Both are per-packet faults: a torn packet in the middle of a
            // recording should cost that packet, not the transcript.
            Err(SymphErr::IoError(_)) | Err(SymphErr::DecodeError(_)) if seconds.is_none() => {
                continue;
            }
            Err(e) => return Err(format!("{} decode: {e}", kind.name())),
        };
        if seconds.is_some() && rate != 0 && buf.spec().rate() != rate {
            return Err("audio sample rate changes mid-stream".into());
        }
        push_mono(&buf, &mut inter, &mut samples, &mut rate);
        check_duration(samples.len(), rate, seconds)?;
    }

    if samples.is_empty() {
        return Err(format!("{} file decoded to no audio", kind.name()));
    }
    if rate == 0 {
        return Err(format!("{} file declares no sample rate", kind.name()));
    }
    Ok(WavAudio {
        samples,
        sample_rate: rate,
    })
}

fn check_duration(samples: usize, rate: u32, seconds: Option<u32>) -> Result<(), String> {
    if let Some(seconds) = seconds
        && (rate == 0 || samples as u64 > u64::from(rate) * u64::from(seconds))
    {
        return Err(format!("audio exceeds {seconds} seconds"));
    }
    Ok(())
}

/// Append one decoded buffer to `out`, averaged down to mono - the same
/// downmix `wav.rs` does, so a stereo file transcribes identically whichever
/// container it arrived in.
fn push_mono(
    buf: &GenericAudioBufferRef<'_>,
    inter: &mut Vec<f32>,
    out: &mut Vec<f32>,
    rate: &mut u32,
) {
    let spec = buf.spec();
    *rate = spec.rate();
    let ch = spec.channels().count().max(1);
    inter.clear();
    buf.copy_to_vec_interleaved(inter);
    if ch == 1 {
        out.extend_from_slice(inter);
        return;
    }
    out.reserve(inter.len() / ch);
    for frame in inter.chunks_exact(ch) {
        out.push(frame.iter().sum::<f32>() / ch as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_duration_limit_is_checked_without_rounding() {
        assert!(check_duration(16000 * 600, 16000, Some(600)).is_ok());
        assert!(check_duration(16000 * 600 + 1, 16000, Some(600)).is_err());
        assert!(check_duration(0, 0, Some(600)).is_err());
        assert!(check_duration(48000 * 601, 48000, Some(600)).is_err());
    }

    #[test]
    fn sniffs_every_accepted_container_from_its_magic() {
        assert_eq!(sniff(b"RIFF\0\0\0\0WAVEfmt "), Container::Wav);
        assert_eq!(sniff(b"fLaC\0\0\0\x22"), Container::Flac);
        assert_eq!(sniff(b"OggS\0\x02\0\0"), Container::Ogg);
        assert_eq!(
            sniff(&[0x1A, 0x45, 0xDF, 0xA3, 0x01, 0, 0, 0]),
            Container::Webm
        );
        assert_eq!(sniff(b"\0\0\0\x20ftypM4A "), Container::Mp4);
        assert_eq!(sniff(b"ID3\x04\0\0\0\0"), Container::Mp3);
        assert_eq!(sniff(&[0xFF, 0xFB, 0x90, 0x00]), Container::Mp3);
        assert_eq!(sniff(b"FORM\0\0\0\0AIFF"), Container::Aiff);
        assert_eq!(sniff(b"caff\0\x01\0\0"), Container::Caf);
        assert_eq!(sniff(b"not audio at all"), Container::Unknown);
        // too short to be anything
        assert_eq!(sniff(b""), Container::Unknown);
        assert_eq!(sniff(b"R"), Container::Unknown);
    }

    /// RIFF must not reach symphonia: the WAV path is the one the parity gates
    /// measure and it stays on our own decoder.
    #[test]
    fn wav_still_goes_through_our_own_decoder() {
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36u32 + 8).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&16000u32.to_le_bytes());
        v.extend_from_slice(&32000u32.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&8u32.to_le_bytes());
        for s in [0i16, 16384, -16384, 32767] {
            v.extend_from_slice(&s.to_le_bytes());
        }
        let a = decode_audio(&v).expect("a hand-built RIFF must decode");
        assert_eq!(a.sample_rate, 16000);
        assert_eq!(a.samples.len(), 4);
    }

    #[test]
    fn a_refusal_names_the_format_and_the_alternatives() {
        // `.err().expect()` rather than `unwrap_err()`: the Ok side is a whole
        // decoded clip, and making it Debug just to print it on a test failure
        // would dump a megabyte of samples into the log.
        let e = decode_audio(b"FORM\0\0\0\0AIFFxxxx")
            .err()
            .expect("aiff must refuse");
        assert!(e.contains("aiff"), "{e}");
        assert!(e.contains("flac, mp3"), "{e}");

        let e = decode_audio(b"this is a text file")
            .err()
            .expect("garbage must refuse");
        assert!(e.contains("unrecognised"), "{e}");
        // the point of byte-sniffing is that renaming the file is not the fix
        assert!(e.contains("wrong extension is not the problem"), "{e}");
    }
}
