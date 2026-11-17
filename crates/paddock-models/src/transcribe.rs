//! Nemotron 3.5 Transcribe (NeMo's `nemo_speechlm` wrapper around the
//! Nemotron-H language model): what its perception config fixes for the
//! speech encoder. The language model's own config is
//! [`crate::nemotron::NemotronConfig`]; this checks the audio side so a
//! sibling checkpoint with another geometry fails at load instead of
//! encoding with the wrong window or threshold.

use std::path::Path;

use serde_json::json;

/// The online inference's window core, in encoder rows (80 ms each), and
/// the context rows on each side - NeMo's ParallelExpertEncoder defaults.
pub const WINDOW: usize = 500;
pub const CONTEXT: usize = 50;
/// The speaker branch's FIFO, its cache update period and the cache length
/// at that geometry (NeMo's online defaults again).
pub const DIAR_FIFO: usize = 40;
pub const DIAR_UPDATE: usize = 300;
pub const DIAR_CACHE: usize = 188;

/// The processor's default instruction (`processor_config.json`'s
/// `default_transcription_prompt`): verbatim transcription in the spoken
/// language.
pub const DEFAULT_PROMPT: &str = "Audio transcription. Language: as spoken. Wording: verbatim, \
     including fillers and repetitions. Formatting: punctuation and capitalization. Output: \
     transcript only.";

/// The 40 languages the model card declares (its `language:` metadata, the
/// rows of its support matrix with ASR), as ISO 639-1. A `language` outside
/// them would put a language the model was never trained on into its
/// Language slot - refused rather than decoded into a guess.
pub const LANGUAGES: [&str; 40] = [
    "ar", "bg", "bn", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "he", "hi", "hr", "hu",
    "id", "it", "ja", "ko", "lt", "lv", "mr", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv",
    "ta", "te", "th", "tl", "tr", "uk", "vi", "zh",
];

/// What one recording of `samples` mono 16-kHz samples costs, in the
/// processor's arithmetic: `(valid frames, padded frames, encoder rows,
/// audio tokens)` - a centered STFT gives 1 + L / 160 frames of which L / 160
/// are valid, padded to a multiple of 8; one token per 8 valid frames (80
/// ms), the last group partial.
pub fn geometry(samples: usize) -> (usize, usize, usize, usize) {
    let valid = samples / 160;
    let frames = (1 + samples / 160).div_ceil(8) * 8;
    (valid, frames, frames / 8, valid.div_ceil(8))
}

/// The `<|audio|>` placeholders a recording of `samples` expands to.
pub fn audio_tokens(samples: usize) -> usize {
    geometry(samples).3
}

/// The card's generation-loop guardrail, applied by NVIDIA to every reported
/// transcription, translation and language-identification result: a
/// character repeated 21 or more times collapses to one copy, then an exact
/// block of 1-4096 characters repeated 20 or more times (the shortest such
/// block at the leftmost position, scanning left to right, as Python's lazy
/// `(.{1,4096}?)\1{19,}` substitution does) collapses to one copy.
pub fn loop_cleanup(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    // a character run of >= 21
    let mut once = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let mut j = i + 1;
        while j < chars.len() && chars[j] == chars[i] {
            j += 1;
        }
        if j - i >= 21 {
            once.push(chars[i]);
        } else {
            once.extend_from_slice(&chars[i..j]);
        }
        i = j;
    }
    // a block of 1..=4096 repeated >= 20 times in a row
    let c = once;
    let mut out = Vec::with_capacity(c.len());
    let mut i = 0;
    'scan: while i < c.len() {
        let room = (c.len() - i) / 20;
        for len in 1..=room.min(4096) {
            let block = &c[i..i + len];
            let mut reps = 1;
            while i + (reps + 1) * len <= c.len()
                && &c[i + reps * len..i + (reps + 1) * len] == block
            {
                reps += 1;
            }
            if reps >= 20 {
                out.extend_from_slice(block);
                i += reps * len;
                continue 'scan;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out.into_iter().collect()
}

/// Check `<dir>/config.json`'s perception against the geometry above and the
/// encoder the CUDA lane is built for.
pub fn check_perception(dir: &Path) -> Result<(), String> {
    let raw = std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| format!("config.json: {e}"))?;
    for (pointer, expected) in [
        ("/model_type", json!("nemo_speechlm")),
        ("/audio_locator_tag", json!("<|audio|>")),
        ("/perception/output_dim", json!(2688)),
        ("/perception/preprocessor/features", json!(128)),
        ("/perception/preprocessor/n_fft", json!(512)),
        ("/perception/preprocessor/sample_rate", json!(16000)),
        ("/perception/preprocessor/window_size", json!(0.025)),
        ("/perception/preprocessor/window_stride", json!(0.01)),
        ("/pe_encoder_config/asr_encoder_type", json!("transformer")),
        ("/pe_encoder_config/asr_encoder_cfg/d_model", json!(1280)),
        ("/pe_encoder_config/asr_encoder_cfg/n_heads", json!(10)),
        ("/pe_encoder_config/asr_encoder_cfg/n_layers", json!(32)),
        ("/pe_encoder_config/asr_encoder_cfg/ff_expansion", json!(4)),
        ("/pe_encoder_config/asr_encoder_cfg/qk_norm", json!(true)),
        ("/pe_encoder_config/asr_encoder_cfg/qkv_bias", json!(false)),
        (
            "/pe_encoder_config/asr_encoder_cfg/pre_block_norm",
            json!(true),
        ),
        (
            "/pe_encoder_config/asr_encoder_cfg/subsampling_factor",
            json!(8),
        ),
        (
            "/pe_encoder_config/asr_encoder_cfg/self_attention_model",
            json!("rope"),
        ),
        (
            "/pe_encoder_config/asr_encoder_cfg/rope_base",
            json!(10000.0),
        ),
        (
            "/pe_encoder_config/asr_encoder_cfg/rotary_fraction",
            json!(0.5),
        ),
        (
            "/pe_encoder_config/asr_encoder_cfg/attn_mode",
            json!("full"),
        ),
        (
            "/pe_encoder_config/asr_normalize_type",
            json!("per_feature"),
        ),
        ("/pe_encoder_config/diar_normalize_type", json!("NA")),
        (
            "/pe_encoder_config/speaker_feature_mode",
            json!("thresholded"),
        ),
        ("/pe_encoder_config/speaker_activity_threshold", json!(0.5)),
        ("/pe_encoder_config/spk_kernel_scale", json!(1.0)),
        (
            "/pe_encoder_config/diarization_model_cfg/max_num_of_spks",
            json!(8),
        ),
        (
            "/pe_encoder_config/diarization_model_cfg/high_resolution",
            json!(true),
        ),
        (
            "/pe_encoder_config/diarization_model_cfg/encoder/d_model",
            json!(512),
        ),
        (
            "/pe_encoder_config/diarization_model_cfg/encoder/n_layers",
            json!(31),
        ),
    ] {
        if v.pointer(pointer) != Some(&expected) {
            return Err(format!(
                "config.json {pointer} is {:?}, this lane is built for {expected}",
                v.pointer(pointer)
            ));
        }
    }
    // the online geometry is NeMo's default unless the checkpoint sets it
    for (key, expected) in [
        ("online_inference_length", WINDOW),
        ("chunk_left_context", CONTEXT),
        ("chunk_right_context", CONTEXT),
        ("diar_fifo_len", DIAR_FIFO),
        ("diar_spkcache_update_period", DIAR_UPDATE),
        ("diar_spkcache_len", DIAR_CACHE),
    ] {
        if let Some(x) = v.pointer(&format!("/pe_encoder_config/{key}"))
            && x.as_u64() != Some(expected as u64)
        {
            return Err(format!(
                "config.json pe_encoder_config.{key} is {x}, this lane is built for {expected}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_follows_the_processor() {
        assert_eq!(geometry(16_000), (100, 104, 13, 13));
        assert_eq!(geometry(12_800), (80, 88, 11, 10));
        assert_eq!(audio_tokens(160), 1);
    }

    #[test]
    fn loop_cleanup_matches_the_card_rule() {
        let run = format!("ok {} done", "a".repeat(30));
        assert_eq!(loop_cleanup(&run), "ok a done");
        // a run of 20 escapes the character rule but is a block of one
        // repeated 20 times (the card's Python gives "b" too)
        assert_eq!(loop_cleanup(&"b".repeat(20)), "b");
        assert_eq!(loop_cleanup(&"b".repeat(19)), "b".repeat(19));
        let block = format!("x{}y", " thank you.".repeat(25));
        assert_eq!(loop_cleanup(&block), "x thank you.y");
        // " la la" x 19 is " la" x 38: a loop at the shorter block
        assert_eq!(loop_cleanup(&" la la".repeat(19)), " la");
        let short = " thank you.".repeat(19);
        assert_eq!(loop_cleanup(&short), short);
        // the shortest block wins: "ab" x 30 collapses to "ab", not "abab"
        assert_eq!(loop_cleanup(&"ab".repeat(30)), "ab");
    }
}
