//! Word timestamps and speaker labels on `/v1/chat/completions` - the
//! contract vLLM serves Nemotron 3.5 Transcribe with (NeMo's CTC timestamp
//! middleware): a request opts in with
//! `mm_processor_kwargs.capture_ctc_timestamps`, its one recording keeps what
//! the alignment needs engine-side while the transcript generates, and the
//! finished transcript comes back aligned as a top-level `ctc_timestamps`
//! object (`words`, `diarization`, `speaker_tag_to_diarization_speaker`,
//! `error`). Streaming carries it on the terminal finish chunk.
//!
//! A model without the head passes the flag through untouched (the
//! reference middleware does the same), so the response has no
//! `ctc_timestamps` key at all there.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use paddock_engine::generator::MediaAligner;
use paddock_engine::service::MmChunk;
use serde_json::{Value, json};

/// The opt-in key inside `mm_processor_kwargs`.
const KEY: &str = "capture_ctc_timestamps";

/// Capture ids, unique within the runner's life.
static NEXT: AtomicU64 = AtomicU64::new(1);

/// Split the opt-in flag off `mm_processor_kwargs`: whether it asked (the
/// reference reads it with Python's `bool`, so any non-empty string or
/// non-zero number counts) and the remaining kwargs, for the families'
/// own parser (which refuses keys it does not know).
pub(crate) fn split(kwargs: Option<&Value>) -> (bool, Option<Value>) {
    let Some(Value::Object(obj)) = kwargs else {
        return (false, kwargs.cloned());
    };
    let Some(flag) = obj.get(KEY) else {
        return (false, kwargs.cloned());
    };
    let on = match flag {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|v| v != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    };
    let mut rest = obj.clone();
    rest.remove(KEY);
    (on, (!rest.is_empty()).then_some(Value::Object(rest)))
}

/// An armed request: where its capture is kept, or the error its object
/// carries without one.
pub(crate) struct CtcPlan {
    aligner: Arc<dyn MediaAligner>,
    /// Some = this request's capture (forgotten if no transcript comes)
    id: Option<u64>,
    /// the reference's error for a request that cannot be captured
    preset: Option<&'static str>,
}

impl CtcPlan {
    /// Arm a chat request that asked (`wanted`): None when the model has no
    /// head. A request with `n > 1` is refused (the reference's 400); one
    /// recording rides a capture directive, none or several answer with the
    /// reference's `no_capture` / `multiple_audio`.
    pub(crate) fn arm(
        aligner: Option<Arc<dyn MediaAligner>>,
        wanted: bool,
        n: usize,
        chunks: Option<&mut Vec<MmChunk>>,
    ) -> Result<Option<CtcPlan>, String> {
        let (true, Some(aligner)) = (wanted, aligner) else {
            return Ok(None);
        };
        if n > 1 {
            return Err("CTC timestamps do not support n > 1.".into());
        }
        let audios = chunks.as_ref().map_or(0, |c| {
            c.iter()
                .filter(|c| matches!(c, MmChunk::Audio { .. }))
                .count()
        });
        let (id, preset) = match (audios, chunks) {
            (1, Some(c)) => {
                let id = NEXT.fetch_add(1, Ordering::Relaxed);
                c.insert(0, MmChunk::CtcTimestamps { id });
                (Some(id), None)
            }
            (0, _) | (_, None) => (None, Some("no_capture")),
            _ => (None, Some("multiple_audio")),
        };
        Ok(Some(CtcPlan {
            aligner,
            id,
            preset,
        }))
    }

    /// The `ctc_timestamps` object for the finished transcript (the content
    /// the response returns). The head and the alignment block, so they run
    /// off the async threads.
    pub(crate) async fn finish(mut self, transcript: String) -> Value {
        if let Some(e) = self.preset {
            return empty(e);
        }
        let Some(id) = self.id.take() else {
            return empty("no_capture");
        };
        let aligner = self.aligner.clone();
        tokio::task::spawn_blocking(move || aligner.align(id, &transcript))
            .await
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "CTC timestamps: the alignment task died");
                empty("alignment_failed")
            })
    }
}

impl Drop for CtcPlan {
    fn drop(&mut self) {
        // a request that ended without a transcript (an engine error, a
        // client gone mid-stream): its capture goes now, not at the TTL
        if let Some(id) = self.id.take() {
            self.aligner.forget(id);
        }
    }
}

/// The reference's object for a request with nothing to align.
fn empty(error: &str) -> Value {
    json!({
        "words": [],
        "diarization": [],
        "speaker_tag_to_diarization_speaker": {},
        "error": error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo(std::sync::Mutex<Vec<u64>>);
    impl MediaAligner for Echo {
        fn align(&self, id: u64, transcript: &str) -> Value {
            json!({"id": id, "text": transcript})
        }
        fn forget(&self, id: u64) {
            self.0.lock().unwrap().push(id);
        }
    }

    fn audio() -> MmChunk {
        MmChunk::Audio {
            samples: vec![0.0; 1600],
            mel: None,
        }
    }

    #[test]
    fn the_flag_reads_as_python_bool_and_leaves_the_other_keys() {
        let (on, rest) = split(Some(&json!({"capture_ctc_timestamps": true})));
        assert!(on);
        assert!(rest.is_none());
        let (on, rest) = split(Some(
            &json!({"capture_ctc_timestamps": "no", "max_pixels": 100}),
        ));
        assert!(on, "a non-empty string is truthy");
        assert_eq!(rest, Some(json!({"max_pixels": 100})));
        for off in [json!(false), json!(0), json!(""), json!(null), json!([])] {
            assert!(!split(Some(&json!({ "capture_ctc_timestamps": off }))).0);
        }
        assert!(!split(None).0);
    }

    #[test]
    fn arming_follows_the_reference() {
        let echo: Arc<dyn MediaAligner> = Arc::new(Echo(Default::default()));
        // no head: passes through
        let mut c = vec![audio()];
        assert!(CtcPlan::arm(None, true, 1, Some(&mut c)).unwrap().is_none());
        assert_eq!(c.len(), 1);
        // not asked
        assert!(
            CtcPlan::arm(Some(echo.clone()), false, 1, Some(&mut c))
                .unwrap()
                .is_none()
        );
        // n > 1 is the reference's 400
        let e = CtcPlan::arm(Some(echo.clone()), true, 2, Some(&mut c))
            .err()
            .unwrap();
        assert_eq!(e, "CTC timestamps do not support n > 1.");
        // one recording: the directive leads the chunks
        let p = CtcPlan::arm(Some(echo.clone()), true, 1, Some(&mut c))
            .unwrap()
            .unwrap();
        assert!(matches!(c[0], MmChunk::CtcTimestamps { .. }));
        assert!(p.id.is_some() && p.preset.is_none());
        // two recordings / none
        let mut two = vec![audio(), audio()];
        let p2 = CtcPlan::arm(Some(echo.clone()), true, 1, Some(&mut two))
            .unwrap()
            .unwrap();
        assert_eq!((p2.id, p2.preset), (None, Some("multiple_audio")));
        assert_eq!(two.len(), 2);
        let p0 = CtcPlan::arm(Some(echo.clone()), true, 1, None)
            .unwrap()
            .unwrap();
        assert_eq!(p0.preset, Some("no_capture"));
        drop(p);
    }

    #[tokio::test]
    async fn a_plan_aligns_once_and_forgets_when_dropped() {
        let echo = Arc::new(Echo(Default::default()));
        let dyn_echo: Arc<dyn MediaAligner> = echo.clone();
        let mut c = vec![audio()];
        let p = CtcPlan::arm(Some(dyn_echo.clone()), true, 1, Some(&mut c))
            .unwrap()
            .unwrap();
        let id = p.id.unwrap();
        let v = p.finish("<spk:0> hi".into()).await;
        assert_eq!(v, json!({"id": id, "text": "<spk:0> hi"}));
        assert!(
            echo.0.lock().unwrap().is_empty(),
            "an aligned capture is not forgotten"
        );
        let mut c = vec![audio()];
        let p = CtcPlan::arm(Some(dyn_echo), true, 1, Some(&mut c))
            .unwrap()
            .unwrap();
        let id = p.id.unwrap();
        drop(p);
        assert_eq!(*echo.0.lock().unwrap(), vec![id]);
        let v = CtcPlan::arm(
            Some(echo.clone()),
            true,
            1,
            Some(&mut vec![audio(), audio()]),
        )
        .unwrap()
        .unwrap()
        .finish(String::new())
        .await;
        assert_eq!(v["error"], "multiple_audio");
    }
}
