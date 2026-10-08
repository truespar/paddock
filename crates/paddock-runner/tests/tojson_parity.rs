//! Chat-template renders against the reference environment, byte for byte.
//!
//! Every fixture template renders one Claude Code-shaped conversation - tool
//! schemas with nested objects, apostrophes, `<example>` tags, `&`, non-ASCII
//! text, numbers and booleans, and a history of reasoning plus tool calls -
//! and must produce exactly what transformers' chat-template environment
//! produces for the same input (`fixtures/tojson_parity/*.txt`, rendered by
//! `fixtures/tojson_parity/render_reference.py` from `normalized.json`).
//! That environment is what the templates are written against and what vLLM
//! serves through; its `tojson` is Python's json.dumps, which llama.cpp's
//! engine reproduces. Before this gate minijinja's builtin `tojson` (compact,
//! HTML-escaping `<>&'`) and sorted-key maps put every tool definition and
//! history tool call into bytes no model was trained on.
//!
//! `normalized.json` is the conversation after `normalize_messages` - the
//! input both renderers see. Regenerate it after changing the conversation
//! or the normalizer (`cargo test --test tojson_parity -- --ignored`), then
//! the goldens with the reference script.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use paddock_runner::chat_template;
use serde_json::Value;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tojson_parity");

const TEMPLATES: &[(&str, &str)] = &[
    (
        "kolibri",
        include_str!("fixtures/kolibri_chat_template.jinja"),
    ),
    (
        "laguna",
        include_str!("fixtures/laguna_chat_template.jinja"),
    ),
    (
        "qwen35",
        include_str!("fixtures/qwen35_chat_template.jinja"),
    ),
    (
        "qwen36",
        include_str!("fixtures/qwen36_chat_template.jinja"),
    ),
    (
        "qwen38",
        include_str!("fixtures/qwen38_chat_template.jinja"),
    ),
    (
        "gemma4",
        include_str!("fixtures/gemma4_chat_template.jinja"),
    ),
    (
        "gemma4_qat",
        include_str!("fixtures/gemma4_qat_chat_template.jinja"),
    ),
    (
        "granite",
        include_str!("fixtures/granite_chat_template.jinja"),
    ),
    (
        "granite_vision",
        include_str!("fixtures/granite_vision_chat_template.jinja"),
    ),
    (
        "gptoss",
        include_str!("fixtures/gptoss_chat_template.jinja"),
    ),
    ("muse", include_str!("fixtures/muse_chat_template.jinja")),
];

fn conversation() -> (Vec<Value>, Vec<Value>) {
    let v: Value =
        serde_json::from_str(include_str!("fixtures/tojson_parity/conversation.json")).unwrap();
    let messages = chat_template::normalize_messages(v["messages"].as_array().unwrap());
    (messages, v["tools"].as_array().unwrap().clone())
}

/// Writes `normalized.json`, the reference script's input.
#[test]
#[ignore]
fn write_normalized_input() {
    let (messages, tools) = conversation();
    let out = serde_json::json!({"messages": messages, "tools": tools});
    std::fs::write(
        format!("{DIR}/normalized.json"),
        serde_json::to_string_pretty(&out).unwrap() + "\n",
    )
    .unwrap();
}

#[test]
fn the_normalized_input_is_current() {
    let (messages, tools) = conversation();
    let want: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/normalized.json")).unwrap())
            .unwrap();
    assert_eq!(
        want["messages"],
        Value::Array(messages),
        "rerun write_normalized_input"
    );
    assert_eq!(
        want["tools"],
        Value::Array(tools),
        "rerun write_normalized_input"
    );
}

#[test]
fn every_template_renders_what_transformers_renders() {
    let (messages, tools) = conversation();
    let mut parted = Vec::new();
    for (name, template) in TEMPLATES {
        let got = chat_template::render(template, &messages, Some(&tools), None).unwrap();
        // the reference pins `strftime_now` to 2026-10-08 (gpt-oss prints it)
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        let got = got.replace(&today, "2026-10-08");
        let want = std::fs::read_to_string(format!("{DIR}/{name}.txt")).unwrap();
        if got != want {
            let at = got
                .char_indices()
                .zip(want.chars())
                .find(|((_, a), b)| a != b)
                .map_or(got.len().min(want.len()), |((i, _), _)| i);
            let from = at.saturating_sub(60);
            eprintln!(
                "{name}: parts at byte {at}\n  ours:  {:?}\n  theirs: {:?}",
                got.get(from..(at + 80).min(got.len())).unwrap_or(""),
                want.get(from..(at + 80).min(want.len())).unwrap_or("")
            );
            parted.push(*name);
        }
    }
    assert!(
        parted.is_empty(),
        "renders part from the reference: {parted:?}"
    );
}
