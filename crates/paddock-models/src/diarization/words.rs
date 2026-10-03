//! Time-overlap attribution, not recognition or voice identification. One
//! arithmetic contract for native Studio, web Studio and API clients.
use super::{SPEAKERS, Segment};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const MAX_WORDS: usize = 20_000;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Word {
    pub word: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
    // Preserve ASR confidence, alternatives and caller metadata, but overwrite
    // stale diarization fields when a different model/preset is run.
    #[serde(flatten)]
    pub metadata: BTreeMap<String, Value>,
}

pub fn validate(words: &[Word], duration: f64) -> Result<(), String> {
    if words.len() > MAX_WORDS || !duration.is_finite() || duration <= 0.0 {
        return Err("invalid word count or recording duration".into());
    }
    for w in words {
        match (w.start, w.end) {
            (None, None) => (),
            (Some(a), Some(b))
                if a.is_finite()
                    && b.is_finite()
                    && a >= 0.0
                    && b >= a
                    && b <= duration + 0.001 => {}
            _ => {
                return Err(
                    "word timestamps must be a complete, finite interval inside this recording"
                        .into(),
                );
            }
        }
    }
    Ok(())
}

/// Merge each speaker's disjoint intervals and index their cumulative duration.
/// Integrating an interval costs O(log segments); neither word order nor
/// duplicates/overlapping same-speaker segments can inflate attribution.
struct Activity {
    spans: Vec<(f64, f64)>,
    prefix: Vec<f64>,
}
impl Activity {
    fn new(mut spans: Vec<(f64, f64)>) -> Self {
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (start, end) in spans {
            if let Some(last) = merged.last_mut().filter(|last| start <= last.1) {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        }
        let mut prefix = vec![0.0];
        for &(start, end) in &merged {
            prefix.push(prefix[prefix.len() - 1] + end - start);
        }
        Self {
            spans: merged,
            prefix,
        }
    }
    fn until(&self, time: f64) -> f64 {
        let i = self.spans.partition_point(|s| s.1 <= time);
        self.prefix[i] + self.spans.get(i).map_or(0.0, |s| (time - s.0).max(0.0))
    }
}

/// At least half of a timed word must intersect a speaker interval. If several
/// qualify, expose ALL candidates and no primary speaker: overlap or a boundary
/// is ambiguous. Coverage is duration fraction, never model confidence. Silence
/// is not assigned to the nearest speaker; untimed/zero-duration words abstain.
pub fn attribute(
    mut words: Vec<Word>,
    segments: &[Segment],
    duration: f64,
) -> Result<Vec<Word>, String> {
    validate(&words, duration)?;
    if segments.iter().any(|s| {
        s.speaker >= SPEAKERS
            || !s.start.is_finite()
            || !s.end.is_finite()
            || s.start < 0.0
            || s.end <= s.start
            || s.end > duration + 0.001
    }) {
        return Err("invalid speaker timeline".into());
    }
    let activity: Vec<_> = (0..SPEAKERS)
        .map(|id| {
            Activity::new(
                segments
                    .iter()
                    .filter(|s| s.speaker == id)
                    .map(|s| (s.start, s.end))
                    .collect(),
            )
        })
        .collect();
    for w in &mut words {
        let timed = w.start.zip(w.end).filter(|(a, b)| b > a);
        let coverage: Vec<f64> = activity
            .iter()
            .map(|s| {
                timed.map_or(0.0, |(a, b)| {
                    ((s.until(b) - s.until(a)) / (b - a)).clamp(0.0, 1.0)
                })
            })
            .collect();
        let speakers: Vec<usize> = coverage
            .iter()
            .enumerate()
            .filter(|(_, v)| **v >= 0.5 - 1e-9)
            .map(|(i, _)| i)
            .collect();
        let status = if timed.is_none() {
            "untimed"
        } else {
            match speakers.len() {
                0 => "unassigned",
                1 => "assigned",
                _ => "ambiguous",
            }
        };
        w.metadata.insert(
            "speaker".into(),
            if speakers.len() == 1 {
                Value::from(speakers[0])
            } else {
                Value::Null
            },
        );
        w.metadata
            .insert("speakers".into(), serde_json::json!(speakers));
        w.metadata
            .insert("speaker_coverage".into(), serde_json::json!(coverage));
        w.metadata
            .insert("speaker_status".into(), Value::from(status));
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn word(start: Option<f64>, end: Option<f64>) -> Word {
        Word {
            word: "Hello!".into(),
            start,
            end,
            metadata: BTreeMap::from([
                ("confidence".into(), serde_json::json!(0.7)),
                ("speaker".into(), serde_json::json!(7)),
            ]),
        }
    }
    #[test]
    fn json_round_trip_preserves_word_clocks_and_metadata_bits() {
        // These are real failures from the HTTP qualification: best-effort
        // JSON float parsing changed timestamps by one ULP and made Studio's
        // lossless transcript validation correctly reject the entire result.
        for time in [1.4000000000000001, 7.3999999999999995, 9.299999999999999] {
            let mut original = word(Some(time), Some(time + 0.1));
            original.metadata.insert("score".into(), Value::from(time));
            let bytes = serde_json::to_vec(&original).unwrap();
            let decoded: Word = serde_json::from_slice(&bytes).unwrap();
            let out = attribute(vec![decoded], &[], 20.0).unwrap();
            let bytes = serde_json::to_vec(&out[0]).unwrap();
            let restored: Word = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(restored.start.unwrap().to_bits(), time.to_bits());
            assert_eq!(restored.end, original.end);
            assert_eq!(restored.metadata["score"], original.metadata["score"]);
        }
    }
    #[test]
    fn overlap_boundaries_silence_and_missing_times_abstain() {
        let spans = vec![
            Segment {
                speaker: 0,
                start: 0.0,
                end: 2.0,
            },
            Segment {
                speaker: 1,
                start: 1.0,
                end: 3.0,
            },
        ];
        let words = attribute(
            vec![
                word(Some(0.0), Some(1.0)),
                word(Some(1.0), Some(2.0)),
                word(Some(3.0), Some(4.0)),
                word(None, None),
                word(Some(0.0), Some(0.0)),
            ],
            &spans,
            4.0,
        )
        .unwrap();
        assert_eq!(words[0].metadata["speaker"], 0);
        assert_eq!(words[0].word, "Hello!");
        assert_eq!(words[0].metadata["confidence"], 0.7);
        assert_eq!(words[1].metadata["speakers"], serde_json::json!([0, 1]));
        for w in &words[1..] {
            assert!(w.metadata["speaker"].is_null());
        }
        assert_eq!(words[2].metadata["speaker_status"], "unassigned");
        assert_eq!(words[3].metadata["speaker_status"], "untimed");
    }
    #[test]
    fn duplicates_order_and_boundary_ties_do_not_invent_ownership() {
        let spans = vec![
            Segment {
                speaker: 1,
                start: 1.0,
                end: 2.0,
            },
            Segment {
                speaker: 0,
                start: 0.0,
                end: 1.0,
            },
            Segment {
                speaker: 0,
                start: 0.0,
                end: 1.0,
            },
        ];
        let words = attribute(
            vec![word(Some(0.0), Some(2.0)), word(Some(0.0), Some(0.5))],
            &spans,
            2.0,
        )
        .unwrap();
        assert_eq!(words[0].metadata["speaker_status"], "ambiguous");
        assert_eq!(words[0].metadata["speaker_coverage"][0], 0.5);
        assert_eq!(words[1].metadata["speaker"], 0);
        assert!(attribute(vec![word(Some(0.0), None)], &spans, 2.0).is_err());
        assert!(attribute(vec![word(Some(0.0), Some(3.0))], &spans, 2.0).is_err());
    }
}
