//! The record codec: CBOR inside a dictionary-compressed
//! zstd frame, behind a one-byte format version. Used for `activity.record`
//! rows in the manager DB today; the runner-side event journal uses
//! the same encoder when it ships - one codec for both tiers.
//!
//! Why not JSON text: the records deliberately carry long self-documenting
//! semconv keys (`gen_ai.usage.output_tokens`, `paddock.prefix_resume_pos`) -
//! ~370-450 B of pure key syntax on a ~500 B record, re-told on every row.
//! At the hammered rate that is 218 GB per 30 days of mostly key
//! names.
//!
//! Why not a positional format (bincode/postcard): the record schema is the
//! runner's and may grow fields freely, and newer-runner/older-manager is
//! explicitly supported. A positional format does not error on an
//! unknown field, it MISPARSES - silent corruption. CBOR keeps the keys, so
//! fields this build has never heard of survive the round-trip verbatim.
//!
//! Why a dictionary: an activity row compresses alone, and a lone ~450 B
//! record gives zstd no history to find matches in - plain compression only
//! entropy-codes the key text (~35% off). Preloading that history is exactly
//! what zstd's dictionary mode is for (its documented small-data regime).
//! Ours is a raw-content dictionary made of synthetic sample records encoded
//! through the same path real records take, so every key string and the map
//! structure around it match at long offsets and each row stores little more
//! than its values: ~470 B CBOR -> ~100 B frame. A raw dict rather than a
//! zdict-trained blob deliberately - same redundancy captured, but reviewable
//! in source and deterministic (no training corpus to version).

use std::sync::LazyLock;

use serde_json::{Value, json};
use zstd::dict::{DecoderDictionary, EncoderDictionary};

/// Format tag: dictionary-v1 zstd frame of CBOR. The version space is
/// 0x01..=0x08 - control bytes that can never begin a JSON document (JSON
/// insignificant whitespace is 0x09/0x0A/0x0D/0x20, and a value's first byte
/// is one of `{ [ " t f n -` or a digit) - so legacy plain-JSON rows sniff
/// apart from codec blobs on the first byte, exactly.
pub const RECORD_V1: u8 = 0x01;

/// Compression level baked into the v1 CDict. High levels are effectively
/// free here: the expensive dictionary preprocessing happens once (LazyLock),
/// and per-record inputs are a few hundred bytes.
const LEVEL: i32 = 19;

/// Decompression allocation guard. Records are ~0.5 KB; a frame claiming
/// more than this is corruption, not data.
const MAX_RECORD_BYTES: usize = 4 << 20;

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("record cbor: {0}")]
    Cbor(String),
    #[error("record zstd: {0}")]
    Zstd(#[from] std::io::Error),
    #[error("unknown record format version {0:#04x} (written by a newer paddock?)")]
    Version(u8),
    #[error("empty record blob")]
    Empty,
}

/// Encode one record for storage: CBOR -> zstd (dict v1) -> version prefix.
pub fn encode_record(v: &Value) -> Result<Vec<u8>, CodecError> {
    let mut cbor = Vec::with_capacity(512);
    ciborium::into_writer(&sorted(v), &mut cbor).map_err(|e| CodecError::Cbor(e.to_string()))?;
    let frame = zstd::bulk::Compressor::with_prepared_dictionary(&CDICT_V1)?.compress(&cbor)?;
    let mut out = Vec::with_capacity(frame.len() + 1);
    out.push(RECORD_V1);
    out.extend_from_slice(&frame);
    Ok(out)
}

/// Decode a stored record: sniffs the first byte, so codec blobs and legacy
/// plain-JSON rows (everything written before) both come back.
pub fn decode_record(bytes: &[u8]) -> Result<Value, CodecError> {
    match bytes.first() {
        None => Err(CodecError::Empty),
        Some(&RECORD_V1) => {
            let cbor = zstd::bulk::Decompressor::with_prepared_dictionary(&DDICT_V1)?
                .decompress(&bytes[1..], MAX_RECORD_BYTES)?;
            ciborium::from_reader(cbor.as_slice()).map_err(|e| CodecError::Cbor(e.to_string()))
        }
        Some(&v) if (0x02..=0x08).contains(&v) => Err(CodecError::Version(v)),
        Some(_) => serde_json::from_slice(bytes).map_err(|e| CodecError::Cbor(e.to_string())),
    }
}

/// `v` with every object's keys in sorted order, recursively. v1 was minted
/// when serde_json's maps were sorted by construction, and its dictionary
/// bytes and every stored record's layout assume that order; the workspace
/// now keeps insertion order (templates must see a client's key order), so
/// the codec sorts explicitly - same bytes, same frames as before.
fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|k| (k.clone(), sorted(&map[k])))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

static CDICT_V1: LazyLock<EncoderDictionary<'static>> =
    LazyLock::new(|| EncoderDictionary::copy(&DICT_V1, LEVEL));
static DDICT_V1: LazyLock<DecoderDictionary<'static>> =
    LazyLock::new(|| DecoderDictionary::copy(&DICT_V1));

static DICT_V1: LazyLock<Vec<u8>> = LazyLock::new(dict_v1);

/// The v1 dictionary: synthetic records covering every field the runner emits
/// (events.rs `EventRecord`), CBOR-encoded through the same sorted-key path
/// real records take, concatenated. zstd finds matches anywhere in the dict
/// but references near its END are cheapest, so the most common shape - a
/// fully instrumented streaming chat completion - goes last.
///
/// FROZEN FOREVER: v1 blobs live in user databases and decode against these
/// exact bytes. `dict_v1_is_frozen` pins their hash; if it ever trips (a new
/// sample, a ciborium encoding change), the fix is to MINT `RECORD_V2` with
/// its own dictionary and keep this builder byte-identical - never to edit
/// v1. New runner fields need no new dict at all: unknown keys simply don't
/// match and cost their literal bytes, correctness is untouched.
fn dict_v1() -> Vec<u8> {
    let samples = [
        // Encoder lane: minimal record - the high-rate shape (embeddings do
        // hundreds/s where chat does tens), few keys, no phases.
        json!({
            "seq": 123456,
            "ts_ms": 1_770_000_000_000_u64,
            "endpoint": "/v1/embeddings",
            "status": 200,
            "duration_ms": 12,
            "stream": false,
            "request_id": "req_9d2c4b1e8f6a40c1a7e5d3b2c8f19a04",
            "gen_ai.request.model": "embeddinggemma-300m",
            "gen_ai.usage.input_tokens": 128,
        }),
        // Failure + identity: error status, disconnect bit, every grouping
        // key (session/user/traceparent/api-key-hash/origin) present.
        json!({
            "seq": 123457,
            "ts_ms": 1_770_000_000_000_u64,
            "endpoint": "/v1/messages",
            "status": 429,
            "duration_ms": 3450,
            "stream": true,
            "request_id": "req_0f8e7d6c5b4a39281706f5e4d3c2b1a0",
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "session_id": "claude-code-1a2b3c4d",
            // Every byte of these samples is frozen with the dictionary; a
            // sample value is not free to change even when it looks like one.
            "user": "jens",
            "gen_ai.request.model": "qwen3.6-27b",
            "gen_ai.usage.input_tokens": 2048,
            "gen_ai.usage.output_tokens": 64,
            "gen_ai.response.finish_reasons": ["length"],
            "paddock.api_key_hash": "a1b2c3d4e5f60718",
            "paddock.origin": "studio",
            "paddock.client_disconnected": true,
        }),
        // Agent lane: Responses endpoint, tool-call finish, spec decode on,
        // batch origin.
        json!({
            "seq": 123458,
            "ts_ms": 1_770_000_000_000_u64,
            "endpoint": "/v1/responses",
            "status": 200,
            "duration_ms": 8900,
            "stream": true,
            "request_id": "req_5a4b3c2d1e0f98877665544332211000",
            "gen_ai.request.model": "laguna-xs-2.1",
            "gen_ai.usage.input_tokens": 18234,
            "gen_ai.usage.output_tokens": 512,
            "gen_ai.response.finish_reasons": ["tool_calls"],
            "paddock.prefix_resume_pos": 17920,
            "paddock.spec_drafted": 640,
            "paddock.spec_accepted": 480,
            "paddock.origin": "batch",
        }),
        // The common case, last for the cheapest offsets: streaming chat
        // completion with the full phase/perf instrumentation.
        json!({
            "seq": 123459,
            "ts_ms": 1_770_000_000_000_u64,
            "endpoint": "/v1/chat/completions",
            "status": 200,
            "duration_ms": 5678,
            "stream": true,
            "request_id": "req_c1d2e3f4a5b60798a0b1c2d3e4f50617",
            "session_id": "claude-code-9f8e7d6c",
            "gen_ai.request.model": "qwen3.5-9b",
            "gen_ai.usage.input_tokens": 4096,
            "gen_ai.usage.output_tokens": 256,
            "gen_ai.response.finish_reasons": ["stop"],
            "paddock.prefix_resume_pos": 3968,
            "paddock.tokenize_ms": 8,
            "paddock.queue_ms": 14,
            "paddock.prefill_ms": 120,
            "paddock.decode_ms": 5530,
            "paddock.spec_drafted": 320,
            "paddock.spec_accepted": 240,
            "paddock.kv_pages": 34,
            "paddock.ttft_ms": 145,
            "paddock.decode_tok_s": 46.5,
        }),
    ];
    let mut dict = Vec::with_capacity(1024);
    for s in &samples {
        ciborium::into_writer(&sorted(s), &mut dict).expect("dict samples always encode");
    }
    dict
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
        })
    }

    /// The v1 dictionary bytes are load-bearing for every blob ever written
    /// with them. Any drift here - edited samples, a ciborium encoding
    /// change on upgrade - means old rows stop decoding, so it must fail
    /// loudly: mint RECORD_V2 with a new dict, never mutate v1.
    #[test]
    fn dict_v1_is_frozen() {
        assert_eq!(
            fnv1a(&DICT_V1),
            0x68ec_e12f_b84a_ffed,
            "dict v1 drifted ({} bytes, fnv1a {:#018x}) - mint RECORD_V2, do not edit v1",
            DICT_V1.len(),
            fnv1a(&DICT_V1),
        );
    }

    /// Full-shape record survives the round trip exactly - including the
    /// f64, the array, and keys the dictionary has never seen.
    #[test]
    fn round_trip_is_exact() {
        let rec = json!({
            "seq": 42,
            "ts_ms": 1_771_234_567_890_u64,
            "endpoint": "/v1/chat/completions",
            "status": 200,
            "duration_ms": 1234,
            "stream": true,
            "request_id": "req_00112233445566778899aabbccddeeff",
            "session_id": "sess-x",
            "gen_ai.request.model": "qwen3.6-35b-a3b",
            "gen_ai.usage.input_tokens": 9000,
            "gen_ai.usage.output_tokens": 111,
            "gen_ai.response.finish_reasons": ["stop", "length"],
            "paddock.decode_tok_s": 43.7,
            "paddock.kv_pages": 12,
            // A field no build of this codec has ever heard of - the §6
            // newer-runner case. Must come back verbatim.
            "paddock.some_2027_field": {"nested": [1, 2.5, "three", null, false]},
        });
        let blob = encode_record(&rec).unwrap();
        assert_eq!(blob[0], RECORD_V1);
        assert_eq!(decode_record(&blob).unwrap(), rec);
    }

    /// Legacy rows are plain JSON text; the sniff must hand them back, and
    /// unknown future versions must say so instead of misparsing.
    #[test]
    fn legacy_json_and_unknown_versions() {
        let rec = json!({"seq": 7, "endpoint": "/v1/completions", "status": 200});
        let text = serde_json::to_string(&rec).unwrap();
        assert_eq!(decode_record(text.as_bytes()).unwrap(), rec);

        assert!(matches!(
            decode_record(&[0x02, 0xFF]),
            Err(CodecError::Version(0x02))
        ));
        assert!(matches!(decode_record(&[]), Err(CodecError::Empty)));
    }

    /// The reason this codec exists: a fully instrumented record must land
    /// far below its JSON size. The bound is generous deliberately (real
    /// measurements sit well under it) - it guards the mechanism (dictionary
    /// actually wired, keys actually matching), not a benchmark number.
    #[test]
    fn full_record_compresses_hard() {
        let rec = json!({
            "seq": 998877,
            "ts_ms": 1_771_000_111_222_u64,
            "endpoint": "/v1/chat/completions",
            "status": 200,
            "duration_ms": 7712,
            "stream": true,
            "request_id": "req_fedcba98765432100123456789abcdef",
            "session_id": "claude-code-55aa66bb",
            "gen_ai.request.model": "gemma4-27b",
            "gen_ai.usage.input_tokens": 15360,
            "gen_ai.usage.output_tokens": 448,
            "gen_ai.response.finish_reasons": ["stop"],
            "paddock.prefix_resume_pos": 14848,
            "paddock.tokenize_ms": 11,
            "paddock.queue_ms": 3,
            "paddock.prefill_ms": 310,
            "paddock.decode_ms": 7350,
            "paddock.spec_drafted": 512,
            "paddock.spec_accepted": 390,
            "paddock.kv_pages": 121,
            "paddock.ttft_ms": 330,
            "paddock.decode_tok_s": 61.2,
        });
        let json_len = serde_json::to_string(&rec).unwrap().len();
        let blob = encode_record(&rec).unwrap();
        // The dictionary is the mechanism, so measure against its absence:
        // a lone record gives zstd no history, and without the preloaded key
        // set it can only entropy-code the text (~2.5x); with it the row is
        // basically values. If these two converge, the dict wiring broke.
        let mut cbor = Vec::new();
        ciborium::into_writer(&rec, &mut cbor).unwrap();
        let no_dict = zstd::bulk::compress(&cbor, LEVEL).unwrap();
        eprintln!(
            "record codec: {} B json / {} B cbor -> {} B blob ({} B zstd without the dict)",
            json_len,
            cbor.len(),
            blob.len(),
            no_dict.len()
        );
        assert!(json_len > 550, "sample lost realism ({json_len} B)");
        assert!(
            blob.len() < 200,
            "{} B json compressed to only {} B - the dictionary is not doing its job",
            json_len,
            blob.len()
        );
        assert!(
            blob.len() + 64 < no_dict.len(),
            "dict gains collapsed vs plain zstd"
        );
        assert_eq!(decode_record(&blob).unwrap(), rec);
    }
}
