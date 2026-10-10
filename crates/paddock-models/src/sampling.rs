//! Elected sampling defaults - the decoding knobs a checkpoint's own authors
//! publish, applied when a request sends none.
//!
//! Why this file exists. Until now every model
//! was served at the OpenAI wire defaults: temperature 1.0, top-k off, top-p
//! 1.0. That is unmodified full-entropy sampling with no truncation anywhere,
//! and it made paddock the outlier - llama.cpp ships 0.8/40/0.95 of its own
//! invention, and vLLM reads `generation_config.json` out of the checkpoint
//! and applies whatever the lab put there. A model whose authors measured and
//! published decoding parameters had them silently discarded here.
//!
//! The numbers below are not house taste. Every row was read out of the
//! model's own published artifacts, and the `source` field says
//! which ones; a row with no citable source does not exist. Two artifacts
//! count, in this order:
//!
//! 1. `generation_config.json` in the official repo. This is the
//!    machine-readable file the labs ship FOR inference engines, and it is
//!    what vLLM applies. When it carries sampling fields, they win.
//! 2. The model card's own recommendation, when the repo ships no
//!    `generation_config.json` (Qwen3.5-9B) or the file carries none
//!    (Nemotron's is behind a gate; its card and unsloth's mirror agree).
//!
//! Most GGUFs carry no sampling metadata - verified by dumping all 48 KV pairs
//! of the served Qwen3.5-9B Q8_0 file, and it is why this table has to exist at
//! all rather than being read off the weights. The key is
//! `general.architecture`, because that is the one identity field every file
//! actually fills in honestly: laguna's `general.name` is a bare commit hash
//! and Qwen3-ASR's is too, so name matching would have silently missed them.
//!
//! CORRECTION: "no GGUF carries it" was true of every file we had
//! seen, and is no longer. IBM's granite 4.2 conversion writes
//! `general.sampling.temp` and `general.sampling.top_p` - its
//! `generation_config.json` values, in the header. That also breaks the
//! assumption underneath the arch key: granite 4.1 and 4.2 share
//! `general.architecture = granite` but publish different sampling (4.1
//! publishes none), so no single row keyed on `granite` can be right for both.
//! [`published_in_gguf`] reads it off the file for exactly that case; the table
//! still wins wherever it has a row, because a model card can express the
//! thinking/instruct split that these keys cannot.
//!
//! What is deliberately not here: the presence/frequency penalties. Qwen's
//! card recommends `presence_penalty=1.5` for several modes, and that number
//! is defined over the whole generated sequence - which is what OpenAI and
//! vLLM implement. Our sampler applies both OpenAI penalties over the
//! trailing `repeat_last_n` tokens instead (64 by default, a llama.cpp-ism
//! inherited from `repeat_penalty`), so 1.5 here would be a different,
//! sharper operator than the one Qwen measured. Transplanting the number into
//! an operator it was not defined against is exactly the guesswork this file
//! is meant to end, so the penalties stay off until the window semantics are
//! fixed. A later change carries that, and the qwen rows land their penalties in
//! the same commit that closes it. (A penalty would also drop every sequence
//! off the speculative path - `Sampler::is_spec_safe` - which is a second
//! reason to do it deliberately rather than as a side effect.)
//!
//! Greedy lanes are not listed here and are not an omission: the document
//! parsers (paddleocr, deepseek2-ocr) and Qwen3-ASR all publish greedy
//! generation configs (`do_sample: false`), and their handlers already force
//! temperature 0. Repeating them as rows would create a second place to
//! disagree with. LightOnOCR-3 is the document parser that is NOT greedy -
//! its authors sample at 0.2 - so it has a row, under the identity key below.

/// A published knob widened to f64 carrying the number it was written as, not
/// its binary representation.
///
/// These are f32 because that is what the engine samples with, and serde widens
/// an f32 straight to f64 - which publishes the error rather than the value:
/// qwen's published `top_p` of 0.95 reached both the runner's capability
/// surface and the manager's estimate as `0.949999988079071`, and from there
/// the composer's sampling popover and the Advanced tab's placeholder. Rust's
/// `f32` Display writes the shortest decimal that re-reads as the same f32, so
/// parsing that back as f64 recovers `0.95` without inventing precision this
/// table never had.
///
/// Lives here rather than at either call site because both surfaces publish
/// these same numbers and they must agree.
pub fn as_written(v: f32) -> f64 {
    format!("{v}").parse::<f64>().unwrap_or(f64::from(v))
}

/// The four truncation/temperature knobs a checkpoint can publish. Penalties
/// are not here deliberately - see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Knobs {
    pub temperature: f32,
    /// 0 = off (no top-k truncation).
    pub top_k: usize,
    /// 1.0 = off (no nucleus truncation).
    pub top_p: f32,
    /// 0.0 = off.
    pub min_p: f32,
}

/// What a request gets when nothing published anything and nothing was
/// configured: the OpenAI API's own documented defaults. Conformance is the
/// reason this is the floor - a client that sends no dials on a model we have
/// no data for must still get the wire behavior it asked for.
pub const WIRE_DEFAULTS: Knobs = Knobs {
    temperature: 1.0,
    top_k: 0,
    top_p: 1.0,
    min_p: 0.0,
};

/// One checkpoint family's published decoding parameters.
#[derive(Debug, Clone, Copy)]
pub struct Elected {
    /// The knobs for the model's DEFAULT mode - thinking/reasoning on, which
    /// is how every family here ships - and the only knobs for a family with
    /// no such mode.
    pub thinking: Knobs,
    /// Separate knobs for thinking off, when the source publishes a second
    /// set. `None` means the source publishes one set for both, and the
    /// honest thing is to use it either way rather than invent a variant.
    pub instruct: Option<Knobs>,
    /// Where these numbers came from, specific enough to re-check by hand.
    pub source: &'static str,
}

impl Elected {
    /// The knobs for this turn. `thinking` is the effective template toggle;
    /// a family with no separate instruct row ignores it.
    pub fn knobs(&self, thinking: bool) -> Knobs {
        match (thinking, self.instruct) {
            (false, Some(k)) => k,
            _ => self.thinking,
        }
    }
}

/// The elected profile for a served architecture, or `None` when that
/// checkpoint's authors publish no decoding parameters at all.
///
/// The key is `general.architecture` with one kind of exception: a fine-tune
/// that keeps a base model's graph but publishes its own sampling gets an
/// identity key of its own, and the caller passes that instead of the arch.
/// `lightonocr-3` is the one today - a `qwen35` graph whose authors sample
/// nothing like Qwen's card ([`LIGHTONOCR3`]).
///
/// `None` is a real answer and not a hole to fill: gpt-oss ships a
/// `generation_config.json` with `do_sample: true` and nothing else, and
/// granite's is `_from_model_config` with only token ids in it. Inventing
/// knobs for those two would be precisely the house-taste this table exists
/// to remove - they keep the wire defaults, and the runner says so out loud.
pub fn elected(arch: &str) -> Option<Elected> {
    Some(match arch {
        // LightOn's own client is the most specific source: models.py keys
        // sampling by generation and gives the Qwen3.5-based LightOnOCR-3
        // sizes top_p 1.0 at temperature 0.2, keeping 0.9 for generations 1
        // and 2. The model cards recommend temperature 0.2 and their
        // transformers snippet leaves top_p at its 1.0 default; only their
        // vLLM snippet sends 0.9, the LightOnOCR-2 value. generation_config.json
        // carries token ids only.
        LIGHTONOCR3 => Elected {
            thinking: Knobs {
                temperature: 0.2,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
            },
            instruct: None,
            source: "lightonai/LightOnOCR client lightonocr/models.py (LightOnOCR-3                      0.8B/4B: temperature 0.2, top_p 1.0) and the LightOnOCR-3 model                      cards' recommended temperature 0.2",
        },
        "kolibri1" => Elected {
            thinking: Knobs {
                temperature: 1.0,
                top_k: 128,
                top_p: 0.97,
                min_p: 0.0,
            },
            instruct: None,
            source: "Aleph-Alpha/Kolibri-1 generation_config.json; eins78/Kolibri-1-mlx-mixed-4-8-bit preserves the same sampling defaults",
        },
        // Qwen3.5 / Qwen3.6, dense and MoE. Both cards publish the same four
        // rows; the two we can act on are the general-task rows for each mode.
        // Qwen3.6's generation_config.json ships the thinking row verbatim
        // (temperature 1.0, top_k 20, top_p 0.95) which is the tiebreak for
        // Qwen3.5-9B, whose repo has no generation_config.json at all.
        "qwen35" | "qwen35moe" => Elected {
            thinking: Knobs {
                temperature: 1.0,
                top_k: 20,
                top_p: 0.95,
                min_p: 0.0,
            },
            instruct: Some(Knobs {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.8,
                min_p: 0.0,
            }),
            source: "Qwen3.5/Qwen3.6 model cards, Best Practices (thinking + \
                     non-thinking general-task rows); Qwen3.6-27B and \
                     Qwen3.6-35B-A3B generation_config.json carry the thinking \
                     row verbatim",
        },
        // Gemma 4 publishes one set for every use case and says so in those
        // words ("Use the following standardized sampling configuration across
        // all use cases"), so there is no instruct variant to elect.
        "gemma4" => Elected {
            thinking: Knobs {
                temperature: 1.0,
                top_k: 64,
                top_p: 0.95,
                min_p: 0.0,
            },
            instruct: None,
            source: "google/gemma-4-31B-it generation_config.json and model card \
                     §1 Sampling Parameters; gemma-4-26B-A4B-it ships the same",
        },
        // Muse Glimmer's card and generation_config.json agree exactly.
        "muse-glimmer" => Elected {
            thinking: Knobs {
                temperature: 1.0,
                top_k: 64,
                top_p: 0.95,
                min_p: 0.0,
            },
            instruct: None,
            source: "meta-models/Muse-Glimmer-30B generation_config.json and \
                     model card Best Practices",
        },
        // poolside ships min_p and top_p explicitly at their off values and
        // top_k at 20 - read that as deliberate, not absent. Both sizes agree.
        "laguna" => Elected {
            thinking: Knobs {
                temperature: 1.0,
                top_k: 20,
                top_p: 1.0,
                min_p: 0.0,
            },
            instruct: None,
            source: "poolside/Laguna-S-2.1 and Laguna-XS-2.1 generation_config.json",
        },
        // The NVIDIA repo is gated, so generation_config.json is unreadable;
        // the card's own "Recommended Sampling" row is the source and unsloth's
        // GGUF card mirrors it. It names temperature and top_p only, so top_k
        // stays off - an unstated knob is not a recommended one.
        "nemotron" | "nemotron_h_moe" => Elected {
            thinking: Knobs {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.95,
                min_p: 0.0,
            },
            instruct: None,
            source: "NVIDIA Nemotron 3.5 Lightning 30B-A3B card, Recommended \
                     Sampling (temperature 1.0, top_p 0.95), mirrored by \
                     unsloth's GGUF card",
        },
        // gpt-oss: generation_config.json has do_sample and token ids, nothing
        // else, and the card names no sampling parameters.
        // granite 4.1: generation_config.json is `_from_model_config` - token
        // ids only - on 8b, 30b and the vision sibling alike. granite 4.2 does
        // publish (temperature 1.0, top_p 0.95) but cannot be a row here,
        // because both share `general.architecture = granite` - see
        // [`published_in_gguf`].
        _ => return None,
    })
}

/// The identity key for LightOnOCR-3 checkpoints - see [`elected`].
pub const LIGHTONOCR3: &str = "lightonocr-3";

/// What the CHECKPOINT itself publishes, read off its own header.
///
/// [`elected`] is keyed on `general.architecture`, and that key is not always
/// fine-grained enough to name a decoding recommendation. Granite is the case
/// that proves it: 4.1 and 4.2 are both `granite`, but 4.1's
/// `generation_config.json` is `_from_model_config` (token ids only) while
/// 4.2's publishes `temperature 1.0, top_p 0.95`. A table row keyed on
/// `granite` would be wrong for exactly one of them whichever way it was
/// written.
///
/// `general.sampling.*` is not a standard GGUF key set - IBM's own 4.2
/// conversion writes `general.sampling.temp` and `general.sampling.top_p`, and
/// those are its `generation_config.json` values verbatim (verified against
/// both files). Reading them is the same principle the rest of the
/// loader already follows: geometry, scalars and template all come from the
/// file rather than from a table indexed by family.
///
/// This FILLS A HOLE and never overrides a curated row - the caller tries
/// [`elected`] first, so every family with a model-card election keeps the
/// richer two-row (thinking/instruct) answer this cannot express.
pub fn published_in_gguf(g: &crate::gguf::GgufFile) -> Option<Elected> {
    let f = |k: &str| g.metadata.get(k).and_then(|v| v.as_f32());
    let temp = f("general.sampling.temp");
    let top_p = f("general.sampling.top_p");
    let top_k = f("general.sampling.top_k");
    let min_p = f("general.sampling.min_p");
    // An absent knob is an unstated one, so it stays at its off value rather
    // than borrowing a neighbour's number - the same rule the nemotron row
    // documents for its missing top_k.
    if temp.is_none() && top_p.is_none() && top_k.is_none() && min_p.is_none() {
        return None;
    }
    Some(Elected {
        thinking: Knobs {
            temperature: temp.unwrap_or(WIRE_DEFAULTS.temperature),
            top_k: top_k.map_or(WIRE_DEFAULTS.top_k, |v| v as usize),
            top_p: top_p.unwrap_or(WIRE_DEFAULTS.top_p),
            min_p: min_p.unwrap_or(WIRE_DEFAULTS.min_p),
        },
        // One set of keys, so there is no second mode to elect.
        instruct: None,
        source: "this checkpoint's own header (general.sampling.*), which \
                 carries its published generation_config values",
    })
}

/// The same answer for the safetensors-primary lane, read from
/// `generation_config.json` - the file the GGUF's `general.sampling.*` keys are
/// a copy of, and the one vLLM applies.
///
/// Both lanes must agree: granite 4.2 serves from either a GGUF or its NVFP4
/// checkpoint, and the same model answering at different sampling depending on
/// which file it was loaded from would be a silent behaviour change - and an
/// unrecorded axis in any comparison between the two lanes.
///
/// `do_sample: false` means greedy and is not a knob set: those checkpoints
/// (the OCR/ASR parsers) already force temperature 0 in their handlers, so
/// returning None keeps this from becoming a second place to disagree.
pub fn published_in_hf_dir(dir: &std::path::Path) -> Option<Elected> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("generation_config.json")).ok()?).ok()?;
    if v.get("do_sample").and_then(serde_json::Value::as_bool) == Some(false) {
        return None;
    }
    let f = |k: &str| {
        v.get(k)
            .and_then(serde_json::Value::as_f64)
            .map(|x| x as f32)
    };
    let temp = f("temperature");
    let top_p = f("top_p");
    let top_k = v
        .get("top_k")
        .and_then(serde_json::Value::as_u64)
        .map(|x| x as usize);
    let min_p = f("min_p");
    if temp.is_none() && top_p.is_none() && top_k.is_none() && min_p.is_none() {
        return None;
    }
    Some(Elected {
        thinking: Knobs {
            temperature: temp.unwrap_or(WIRE_DEFAULTS.temperature),
            top_k: top_k.unwrap_or(WIRE_DEFAULTS.top_k),
            top_p: top_p.unwrap_or(WIRE_DEFAULTS.top_p),
            min_p: min_p.unwrap_or(WIRE_DEFAULTS.min_p),
        },
        instruct: None,
        source: "this checkpoint's own generation_config.json - the same file \
                 vLLM reads",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_architectures_resolve_to_their_published_numbers() {
        // spot-check one value per row against the fetched artifact, so a
        // typo in the table fails here rather than in someone's generation
        let q = elected("qwen35").expect("qwen3.5/3.6 publish a recommendation");
        assert_eq!(q.thinking.top_k, 20);
        assert_eq!(q.thinking.top_p, 0.95);
        assert_eq!(elected("qwen35moe").unwrap().thinking, q.thinking);
        assert_eq!(elected("gemma4").unwrap().thinking.top_k, 64);
        assert_eq!(elected("muse-glimmer").unwrap().thinking.top_p, 0.95);
        assert_eq!(elected("laguna").unwrap().thinking.top_p, 1.0);
        assert_eq!(elected("nemotron_h_moe").unwrap().thinking.top_k, 0);
        let lo = elected(LIGHTONOCR3).expect("LightOn publishes sampling");
        assert_eq!((lo.thinking.temperature, lo.thinking.top_p), (0.2, 1.0));
        assert_eq!(lo.thinking.top_k, 0, "unstated, so off");
        // the identity key must not inherit its base graph's row
        assert_ne!(lo.thinking, q.thinking);
    }

    fn hdr(kv: &[(&str, f32)]) -> crate::gguf::GgufFile {
        crate::gguf::GgufFile {
            version: 3,
            alignment: 32,
            metadata: kv
                .iter()
                .map(|(k, v)| ((*k).to_owned(), crate::gguf::Value::F32(*v)))
                .collect(),
            tensors: Vec::new(),
            data_offset: 0,
        }
    }

    /// granite 4.2 writes its generation_config into the GGUF header; 4.1 (same
    /// `general.architecture`) writes nothing. The file is the only thing that
    /// can tell them apart.
    #[test]
    fn a_checkpoint_that_publishes_in_its_header_is_read() {
        let g42 = hdr(&[
            ("general.sampling.temp", 1.0),
            ("general.sampling.top_p", 0.95),
        ]);
        let e = published_in_gguf(&g42).expect("granite 4.2 publishes temp + top_p");
        assert_eq!(e.thinking.temperature, 1.0);
        assert_eq!(e.thinking.top_p, 0.95);
        // Unstated knobs stay off rather than borrowing a neighbour's numbers.
        assert_eq!(e.thinking.top_k, WIRE_DEFAULTS.top_k);
        assert_eq!(e.thinking.min_p, WIRE_DEFAULTS.min_p);
        // One set of keys means one mode - nothing to elect for thinking-off.
        assert!(e.instruct.is_none());
        assert_eq!(e.knobs(true), e.knobs(false));

        // granite 4.1's header: architecture only, no sampling block.
        let g41 = hdr(&[]);
        assert!(published_in_gguf(&g41).is_none());
    }

    /// The two lanes must not disagree: granite 4.2 serves from a GGUF or from
    /// its NVFP4 checkpoint directory, and both carry the same published knobs.
    #[test]
    fn the_gguf_and_checkpoint_lanes_read_the_same_numbers() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("generation_config.json"),
            r#"{"_from_model_config": true, "do_sample": true, "temperature": 1.0,
                "top_p": 0.95, "bos_token_id": 100283}"#,
        )
        .unwrap();
        let hf = published_in_hf_dir(d.path()).expect("4.2 publishes temp + top_p");
        let gguf = published_in_gguf(&hdr(&[
            ("general.sampling.temp", 1.0),
            ("general.sampling.top_p", 0.95),
        ]))
        .unwrap();
        assert_eq!(
            hf.thinking, gguf.thinking,
            "the lanes must serve the same sampling"
        );

        // greedy checkpoints are not a knob set - their handlers force temp 0
        std::fs::write(
            d.path().join("generation_config.json"),
            r#"{"do_sample": false, "temperature": 1.0, "top_p": 0.95}"#,
        )
        .unwrap();
        assert!(published_in_hf_dir(d.path()).is_none());

        // token ids only (granite 4.1's shape) publishes nothing
        std::fs::write(
            d.path().join("generation_config.json"),
            r#"{"_from_model_config": true, "bos_token_id": 100257}"#,
        )
        .unwrap();
        assert!(published_in_hf_dir(d.path()).is_none());
    }

    #[test]
    fn a_family_that_publishes_nothing_gets_nothing() {
        // gpt-oss and granite are the two we verified publish no sampling -
        // they must not silently acquire a neighbour's numbers
        assert!(elected("gpt-oss").is_none());
        assert!(elected("granite").is_none());
        assert!(elected("some-arch-we-have-never-served").is_none());
    }

    #[test]
    fn the_thinking_toggle_only_moves_a_family_that_published_two_rows() {
        let q = elected("qwen35").unwrap();
        assert_eq!(q.knobs(true).temperature, 1.0);
        assert_eq!(q.knobs(false).temperature, 0.7);
        assert_eq!(q.knobs(false).top_p, 0.8);
        // gemma publishes one set "across all use cases" - the toggle is not
        // an excuse to invent a second
        let g = elected("gemma4").unwrap();
        assert_eq!(g.knobs(true), g.knobs(false));
    }

    #[test]
    fn every_elected_row_carries_a_citation() {
        for arch in [
            "qwen35",
            "qwen35moe",
            "gemma4",
            "muse-glimmer",
            "laguna",
            "nemotron_h_moe",
            LIGHTONOCR3,
        ] {
            let e = elected(arch).unwrap();
            assert!(
                e.source.len() > 30,
                "{arch} needs a source specific enough to re-check"
            );
        }
    }

    #[test]
    fn the_floor_is_the_openai_wire() {
        // the fallback must stay byte-identical to the documented API
        // defaults - a model we have no data for is not an excuse to sample
        // it however we like
        assert_eq!(WIRE_DEFAULTS.temperature, 1.0);
        assert_eq!(WIRE_DEFAULTS.top_k, 0);
        assert_eq!(WIRE_DEFAULTS.top_p, 1.0);
        assert_eq!(WIRE_DEFAULTS.min_p, 0.0);
    }

    #[test]
    fn published_knobs_serialize_as_the_numbers_they_were_written_as() {
        for (v, want) in [
            (0.95f32, "0.95"),
            (0.8, "0.8"),
            (0.7, "0.7"),
            (0.0, "0.0"),
            (1.0, "1.0"),
        ] {
            assert_eq!(
                serde_json::json!(as_written(v)).to_string(),
                want,
                "f32 {v}"
            );
        }
        // ...and every row of the table, so a new election cannot smuggle in a
        // number no human would write
        for arch in [
            "qwen35",
            "qwen35moe",
            "gemma4",
            "muse-glimmer",
            "laguna",
            "nemotron_h_moe",
            LIGHTONOCR3,
        ] {
            let Some(e) = elected(arch) else { continue };
            for k in [Some(&e.thinking), e.instruct.as_ref()]
                .into_iter()
                .flatten()
            {
                for v in [k.temperature, k.top_p, k.min_p] {
                    let s = serde_json::json!(as_written(v)).to_string();
                    assert!(s.len() <= 6, "{arch}: {v} serializes as {s}");
                }
            }
        }
    }
}
