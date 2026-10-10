use serde_json::json;

use super::*;
use crate::parsers::{parse, tool_hints};

fn hints() -> ToolHints {
    let tools = [
        json!({"type": "function", "function": {"name": "Write", "parameters": {"type": "object",
            "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}}}),
        json!({"type": "function", "function": {"name": "Edit", "parameters": {"type": "object",
            "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"},
                           "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}}}}),
        json!({"type": "function", "function": {"name": "Read", "parameters": {"type": "object",
            "properties": {"file_path": {"type": "string"}, "limit": {"type": "integer"},
                           "offset": {"type": "integer"}}}}}),
        json!({"type": "function", "function": {"name": "Bash", "parameters": {"type": "object",
            "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}}}}),
    ];
    tool_hints(Some(&tools[..])).unwrap()
}

/// Feed `text` one char at a time through a stream, then finish on the
/// whole: the per-call concatenated fragments must equal the final parse's
/// arguments byte for byte, every call must start, grow and stop in order.
/// Returns the events and how many argument fragments went out before the
/// turn ended.
fn stream_whole(dialect: Dialect, text: &str, thinking_open: bool) -> (Vec<ToolEv>, usize) {
    let h = hints();
    let mut ts = ToolStream::default();
    let mut evs = Vec::new();
    for (i, c) in text.char_indices() {
        let prefix = &text[..i + c.len_utf8()];
        let parsed = parse(dialect, prefix, thinking_open, Some(&h));
        evs.extend(ts.step(dialect, prefix, thinking_open, Some(&h), false, &parsed));
    }
    let early = evs
        .iter()
        .filter(|e| matches!(e, ToolEv::Args { .. }))
        .count();
    let fin = parse(dialect, text, thinking_open, Some(&h));
    evs.extend(ts.finish(&fin));
    let mut args: Vec<String> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut open: Option<usize> = None;
    for e in &evs {
        match e {
            ToolEv::Start { k, name } => {
                assert!(open.is_none(), "call {k} started while {open:?} open");
                assert_eq!(*k, args.len(), "calls start in order");
                open = Some(*k);
                names.push(name.clone());
                args.push(String::new());
            }
            ToolEv::Args { k, json } => {
                assert_eq!(open, Some(*k), "fragment for a call that is not open");
                args[*k].push_str(json);
            }
            ToolEv::Stop { k } => {
                assert_eq!(open, Some(*k));
                open = None;
            }
        }
    }
    assert!(open.is_none(), "a call never stopped");
    assert_eq!(names.len(), fin.tool_calls.len(), "{text:?}");
    for (k, tc) in fin.tool_calls.iter().enumerate() {
        assert_eq!(names[k], tc.name);
        assert_eq!(args[k], tc.arguments, "call {k} of {text:?}");
        // and the arguments are JSON a client can read
        assert!(
            serde_json::from_str::<serde_json::Value>(&args[k]).is_ok(),
            "{}",
            args[k]
        );
    }
    (evs, early)
}

const WRITE: &str = "<tool_call>\n<function=Write>\n<parameter=file_path>\n/tmp/a.py\n</parameter>\n\
<parameter=content>\nimport sys\n\ndef main():\n    print(\"hi\\tthere\")  # \u{e9}\u{1f600}\n    return 0\n\n\
</parameter>\n</function>\n</tool_call>";

#[test]
fn a_write_streams_its_content_while_it_is_written() {
    let (evs, early) = stream_whole(Dialect::QwenXml, WRITE, false);
    // the content went out in many pieces, not one blob at the end
    assert!(early > 40, "only {early} fragments before the end");
    let ToolEv::Start { name, .. } = &evs[0] else {
        panic!()
    };
    assert_eq!(name, "Write");
}

#[test]
fn every_prefix_stays_a_prefix_of_the_final_arguments() {
    let cases: &[(&str, bool)] = &[
        (WRITE, false),
        // reasoning, then preamble text, then two parallel calls
        (
            "Let me look.\n</think>\n\nI'll read both.\n<tool_call>\n<function=Read>\n<parameter=file_path>\n\
/a/b.txt\n</parameter>\n<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n\
<function=Bash>\n<parameter=command>\ncat <<'EOF' > x\n</para not a tag\nEOF\n</parameter>\n\
<parameter=timeout>\n120000\n</parameter>\n</function>\n</tool_call>",
            true,
        ),
        // a call inside a think block that never closes
        (
            "Thinking about it.\n<tool_call>\n<function=Edit>\n<parameter=file_path>\n/x.rs\n</parameter>\n\
<parameter=old_string>\n    a\\b \"q\"\n</parameter>\n<parameter=new_string>\n    c\n\n</parameter>\n\
<parameter=replace_all>\nfalse\n</parameter>\n</function>\n</tool_call>",
            true,
        ),
        // `</think>` inside a value cuts the call where the parse cuts it
        (
            "Hmm.\n<tool_call>\n<function=Write>\n<parameter=content>\nbefore</think>after\n</parameter>\n\
</function>\n</tool_call>",
            true,
        ),
        // unterminated at max_tokens, mid value
        (
            "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la /tmp && echo",
            false,
        ),
        // mid key
        ("<tool_call>\n<function=Bash>\n<parameter=comm", false),
        // a value closed by `</function>` alone, an undeclared key, a duplicate
        (
            "<tool_call>\n<function=Bash>\n<parameter=extra>\n{\"a\": [1, 2]}\n</parameter>\n\
<parameter=command>\nfirst\n</parameter>\n<parameter=command>\nsecond\n</function>\n</tool_call>",
            false,
        ),
        // no framing newlines, empty and newline-only values, control chars
        (
            "<tool_call><function=Write><parameter=file_path>p</parameter><parameter=content></parameter>\
</function></tool_call>\n<tool_call><function=Write><parameter=content>\n\n</parameter>\
<parameter=file_path>\u{1}\r\n\u{7f}</parameter></function></tool_call>",
            false,
        ),
        // an unknown tool (no hints for it): values coerce as untyped
        (
            "<tool_call>\n<function=Other>\n<parameter=n>\n7\n</parameter>\n<parameter=s>\nseven\n</parameter>\n\
</function>\n</tool_call>",
            false,
        ),
        // an empty name is not a call; the next block is
        (
            "<tool_call>\n<function=>\n</function>\n</tool_call>\n<tool_call>\n<function=Read>\n</function>\n\
</tool_call>",
            false,
        ),
    ];
    for (text, open) in cases {
        stream_whole(Dialect::QwenXml, text, *open);
    }
}

#[test]
fn no_tools_declared_means_no_calls() {
    let mut ts = ToolStream::default();
    let parsed = parse(Dialect::QwenXml, WRITE, false, None);
    assert!(
        ts.step(Dialect::QwenXml, WRITE, false, None, false, &parsed)
            .is_empty()
    );
}

#[test]
fn a_single_call_request_streams_only_the_first() {
    let h = hints();
    let text = format!("{WRITE}\n{WRITE}");
    let mut ts = ToolStream::default();
    let mut n = 0;
    for (i, c) in text.char_indices() {
        let prefix = &text[..i + c.len_utf8()];
        let parsed = parse(Dialect::QwenXml, prefix, false, Some(&h));
        n += ts
            .step(Dialect::QwenXml, prefix, false, Some(&h), true, &parsed)
            .iter()
            .filter(|e| matches!(e, ToolEv::Start { .. }))
            .count();
    }
    assert_eq!(n, 1);
}

#[test]
fn other_dialects_send_each_call_whole_when_its_block_closes() {
    let text = "<tool_call>\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"/a\"}}\n</tool_call>\n\
<tool_call>\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"/b\"}}\n</tool_call>";
    let (evs, early) = stream_whole(Dialect::JsonToolCall, text, false);
    // both went out mid-stream, one fragment each
    assert_eq!(early, 2);
    assert_eq!(evs.len(), 6);
}
