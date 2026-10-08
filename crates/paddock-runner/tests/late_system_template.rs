//! Mid-conversation `system` turns against every served chat template we
//! carry a fixture of: which templates render one in place themselves
//! (`chat_template::renders_late_system` - those get the turn as it is, the
//! way llama.cpp and vLLM hand it to them), and
//! what the fold-or-pass-through choice does to a Kolibri tool loop, whose
//! template keeps an assistant turn's reasoning only after the last USER
//! message. Claude Code ends every request with one of these turns.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use paddock_runner::chat_template;
use serde_json::json;

const KOLIBRI: &str = include_str!("fixtures/kolibri_chat_template.jinja");

#[test]
fn which_templates_render_a_late_system_turn_in_place() {
    let fixtures = [
        ("kolibri", KOLIBRI),
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
    let native: Vec<&str> = fixtures
        .iter()
        .filter(|(_, t)| chat_template::renders_late_system(t))
        .map(|(n, _)| *n)
        .collect();
    // every one of these renders the turn in its own syntax where it sits
    // (`<|turn>system`, `<|start_of_role|>system`, `<system>`, ...) - what
    // llama.cpp and vLLM hand them; Qwen 3.5-3.8 raise and Harmony has no
    // slot, so those keep the `<system-reminder>` fold
    assert_eq!(
        native,
        [
            "kolibri",
            "laguna",
            "gemma4",
            "gemma4_qat",
            "granite",
            "granite_vision",
            "muse"
        ]
    );
}

/// A Claude Code tool loop as the runner hands it to the template: the task,
/// an assistant turn that reasoned and called a tool, the result, and the
/// trailing system turn every request carries.
fn tool_loop() -> Vec<serde_json::Value> {
    vec![
        json!({"role": "system", "content": "You are a coding agent."}),
        json!({"role": "user", "content": "Read the ledger."}),
        json!({"role": "system", "content": "# Environment\nWorking directory: /ws"}),
        json!({"role": "assistant", "content": "", "reasoning_content": "I should read ledger_01 first.",
               "tool_calls": [{"id": "t1", "type": "function",
                               "function": {"name": "Read", "arguments": {"file_path": "/ws/ledger_01.txt"}}}]}),
        json!({"role": "tool", "content": "REC 01-0001 | courier: Saga Holm", "tool_call_id": "t1"}),
        json!({"role": "system", "content": "<reminder>keep going</reminder>"}),
    ]
}

fn render(native: bool) -> String {
    let msgs = chat_template::inline_late_system_messages(&tool_loop(), native).unwrap();
    let msgs = chat_template::normalize_messages(&msgs);
    chat_template::render(KOLIBRI, &msgs, None, None).expect("render")
}

#[test]
fn kolibri_keeps_its_reasoning_through_the_trailing_system_turn() {
    let native = render(true);
    // in place, as the template's own system turns, and the tool loop's
    // reasoning survives: the last user message is still the task
    assert!(native.contains("<|im_start|>system\n<reminder>keep going</reminder><|im_end|>\n"));
    assert!(native.contains("<think>\nI should read ledger_01 first.\n</think>"));
    assert!(native.ends_with(
        "<|im_start|>system\n<reminder>keep going</reminder><|im_end|>\n<|im_start|>assistant\n"
    ));
    // folded into a user turn, the reminder becomes the last query and the
    // template drops every earlier reasoning block
    let folded = render(false);
    assert!(!folded.contains("I should read ledger_01 first."));
    assert!(folded.contains("<system-reminder>"));
}
