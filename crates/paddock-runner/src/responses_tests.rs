use super::*;

#[test]
fn streamed_snapshot_preserves_tool_and_prose_identity() {
    use crate::responses_stream::{Calls, Seq};
    for reasoning in [false, true] {
        for message in [false, true] {
            let parsed = Parsed {
                reasoning: reasoning.then(|| "Inspect the file.".into()),
                content: message.then(|| "Applying a fix.".into()),
                ..Parsed::default()
            };
            let base = 3; // compaction plus two enrichment items
            let first = base + usize::from(reasoning) + usize::from(message);
            let mut sq = Seq::default();
            let mut calls = Calls::default();
            let mut events = Vec::new();
            for k in 0..2 {
                events.extend(calls.event(
                    ToolEv::Start {
                        k,
                        name: "write".into(),
                    },
                    first,
                    &mut sq,
                ));
                for chunk in ["{\"text\":\"", "å😀", "\\n", "\"}"] {
                    events.extend(calls.event(
                        ToolEv::Args {
                            k,
                            json: chunk.into(),
                        },
                        first,
                        &mut sq,
                    ));
                }
                events.extend(calls.event(ToolEv::Stop { k }, first, &mut sq));
            }
            let logprobs =
                [json!({"token":"fix","logprob":-0.1,"bytes":[102,105,120],"top_logprobs":[]})];
            let items = streamed_output_items(
                &parsed,
                Some(&logprobs),
                reasoning.then_some("rs_kept"),
                message.then_some("msg_kept"),
                calls,
            );
            for (_, event) in &events {
                let index = event["output_index"].as_u64().unwrap() as usize - base;
                match event["type"].as_str().unwrap() {
                    "response.output_item.added" => {
                        assert_eq!(event["item"]["id"], items[index]["id"]);
                        assert_eq!(event["item"]["call_id"], items[index]["call_id"]);
                    }
                    "response.output_item.done" => assert_eq!(event["item"], items[index]),
                    _ => assert_eq!(event["item_id"], items[index]["id"]),
                }
            }
            assert!(
                events.windows(2).all(
                    |p| p[0].1["sequence_number"].as_u64() < p[1].1["sequence_number"].as_u64()
                )
            );
            if reasoning {
                assert_eq!(items[0]["id"], "rs_kept");
            }
            if message {
                let item = &items[usize::from(reasoning)];
                assert_eq!(item["id"], "msg_kept");
                assert_eq!(item["content"][0]["logprobs"], json!(logprobs));
            }
            assert_ne!(
                items[first - base]["call_id"],
                items[first - base + 1]["call_id"]
            );
        }
    }
}

#[test]
fn streamed_snapshot_does_not_shift_calls_for_empty_or_late_prose() {
    use crate::responses_stream::{Calls, Seq};
    let mut calls = Calls::default();
    let mut sq = Seq::default();
    calls.event(
        ToolEv::Start {
            k: 0,
            name: "read".into(),
        },
        0,
        &mut sq,
    );
    calls.event(
        ToolEv::Args {
            k: 0,
            json: "{}".into(),
        },
        0,
        &mut sq,
    );
    let done = calls.event(ToolEv::Stop { k: 0 }, 0, &mut sq);
    let parsed = Parsed {
        reasoning: Some(String::new()),
        content: Some("After the call.".into()),
        ..Parsed::default()
    };
    let items = streamed_output_items(&parsed, None, None, None, calls);
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], done.last().unwrap().1["item"]);
    assert_eq!(items[1]["content"][0]["text"], "After the call.");
}

#[test]
fn streamed_snapshot_keeps_calls_discovered_at_finish() {
    use crate::responses_stream::{Calls, Seq};
    let parsed = Parsed {
        tool_calls: vec![crate::parsers::ToolCallRaw {
            name: "read".into(),
            arguments: "{\"path\":\"a.rs\"}".into(),
        }],
        ..Parsed::default()
    };
    let mut tools = crate::tool_stream::ToolStream::default();
    let mut calls = Calls::default();
    let mut sq = Seq::default();
    let events: Vec<_> = tools
        .finish(&parsed)
        .into_iter()
        .flat_map(|e| calls.event(e, 0, &mut sq))
        .collect();
    let items = streamed_output_items(&parsed, None, None, None, calls);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0], events.last().unwrap().1["item"]);
    assert_eq!(items[0]["arguments"], parsed.tool_calls[0].arguments);
}

#[tokio::test]
async fn quiet_response_stream_sends_comments_without_fabricating_events() {
    use http_body_util::BodyExt;
    let mut body = streaming_response(futures::stream::pending()).into_body();
    let frame = tokio::time::timeout(Duration::from_secs(30), body.frame())
        .await
        .expect("quiet vision stream must not remain idle")
        .expect("stream stays open")
        .expect("body frame");
    let bytes = frame.into_data().expect("keepalive data");
    let text = std::str::from_utf8(&bytes).expect("UTF-8 comment");
    assert!(text.starts_with(':') && text.ends_with("\n\n"));
    assert!(!text.contains("data:") && !text.contains("event:"));
}

fn forensics_runtime() -> std::sync::Arc<crate::forensics::ForensicRuntime> {
    crate::forensics::ForensicRuntime::build(&crate::config::ForensicsConfig {
        enabled: true,
        auto: crate::config::ForensicsAuto::Off,
        tool: true,
        device: None,
    })
    .expect("runtime builds when enabled")
}

/// A deterministic checkerboard|smooth PNG that fires ELA, as a data-URI
/// chat message (the shape the tool resolves images from).
fn image_message() -> Value {
    use base64::Engine as _;
    use image::{ExtendedColorType, ImageEncoder};
    let (w, h) = (512u32, 512u32);
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let v: u8 = if x < 256 {
                if (x + y) & 1 == 0 { 0 } else { 255 }
            } else {
                (y * 255 / h) as u8
            };
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    // JPEG so ELA applies (ela is JPEG-only, per the reference should_skip).
    let mut jpg = std::io::Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpg, 92)
        .write_image(&rgb, w, h, ExtendedColorType::Rgb8)
        .expect("encode jpeg");
    let b64 = base64::engine::general_purpose::STANDARD.encode(jpg.into_inner());
    json!({"role":"user","content":[
        {"type":"image_url","image_url":{"url":format!("data:image/jpeg;base64,{b64}")}}
    ]})
}

#[test]
fn forensics_tool_type_is_served_and_dropped_from_function_defs() {
    // The gate the live smoke test caught: `{"type":"forensics"}` must pass
    // request validation (served) and be dropped from the function defs
    // (handled by gather, like web_search/mcp), not 400 as "unsupported".
    assert!(served_tool_type(Some("forensics")));
    let out = normalize_tools(&[json!({"type": "forensics"})]).expect("forensics is served");
    assert!(
        out.is_empty(),
        "forensics is handled by gather, not a function def"
    );
    // an unknown type still errors loudly (no silent drop)
    assert!(normalize_tools(&[json!({"type": "file_search"})]).is_err());
}

#[test]
fn clock_tool_type_is_served_and_dropped_from_function_defs() {
    assert!(served_tool_type(Some("current_time")));
    let out = normalize_tools(&[json!({"type": "current_time", "timezone": "Europe/Stockholm"})])
        .expect("current_time is served");
    assert!(
        out.is_empty(),
        "current_time is handled by gather, not a function def"
    );
}

#[test]
fn clock_tool_classifies_only_when_gathered() {
    let mut g = one_tool_gathered();
    g.current_time = Some(
        crate::clock::parse_spec(&json!({"type": "current_time", "timezone": "Europe/Stockholm"}))
            .expect("valid zone"),
    );
    assert!(
        matches!(
            classify_call(crate::clock::TOOL_NAME, "{}", &g),
            Some(CallKind::Clock)
        ),
        "expected a Clock call"
    );
    // Not gathered -> a caller's own function of the same name passes through.
    g.current_time = None;
    assert!(classify_call(crate::clock::TOOL_NAME, "{}", &g).is_none());
}

#[test]
fn forensics_tool_classifies_only_when_gathered() {
    let mut g = one_tool_gathered();
    g.forensics = Some(forensics_runtime());
    match classify_call(crate::forensics::TOOL_NAME, "{}", &g) {
        Some(CallKind::Forensics { image_index }) => assert_eq!(image_index, None),
        _ => panic!("expected a Forensics call"),
    }
    match classify_call(crate::forensics::TOOL_NAME, r#"{"image_index":2}"#, &g) {
        Some(CallKind::Forensics { image_index }) => assert_eq!(image_index, Some(2)),
        _ => panic!("expected a Forensics call carrying the index"),
    }
    // Not gathered -> a caller's own function of the same name passes through.
    g.forensics = None;
    assert!(classify_call(crate::forensics::TOOL_NAME, "{}", &g).is_none());
}

#[tokio::test]
async fn forensics_tool_executes_over_conversation_image() {
    let rt = forensics_runtime();
    let messages = vec![image_message()];
    let (content, output, error, status) = run_forensics_tool(&rt, &messages, None).await;
    assert_eq!(status, "completed");
    assert!(error.is_none());
    assert!(output.is_some());
    assert!(
        content.contains("ela_block_outliers"),
        "tool result should carry the finding: {content}"
    );

    // No image in the conversation -> a clean failed status, never a panic.
    let (_c, _o, err, status) = run_forensics_tool(&rt, &[], None).await;
    assert_eq!(status, "failed");
    assert!(err.is_some());
}

/// A two-tool `Gathered` with no servers to talk to: enough for the
/// classification and ledger decisions, which is all that runs here.
fn one_tool_gathered() -> Gathered {
    let cfg = paddock_mcp::ServerConfig {
        id: "test".into(),
        label: "artifacts".into(),
        transport: paddock_mcp::Transport::Http {
            url: "http://localhost/mcp".into(),
            headers: HashMap::new(),
        },
    };
    let schema = json!({"type":"object","properties":{"artifact_id":{"type":"string"}},
                            "required":["artifact_id"]});
    Gathered {
        tools: vec![],
        routing: HashMap::from([(
            "artifacts__artifact_read".to_string(),
            (cfg, "artifact_read".to_string()),
        )]),
        needs_approval: HashSet::new(),
        listings: vec![],
        catalog: vec![crate::tool_search::CatalogTool {
            name: "artifacts__artifact_read".into(),
            description: "read one".into(),
            input_schema: schema,
        }],
        web_search: None,
        forensics: None,
        current_time: None,
        instructions: vec![],
    }
}

fn parsed_call(name: &str, arguments: &str) -> Parsed {
    Parsed {
        content: None,
        reasoning: None,
        tool_calls: vec![crate::parsers::ToolCallRaw {
            name: name.into(),
            arguments: arguments.into(),
        }],
        complete_calls: 1,
    }
}

/// Budget lever 1 through the Responses classifier: the same call twice in a
/// turn comes back off the ledger, and a third time is refused.
#[test]
fn a_repeated_call_replays_then_is_refused() {
    let g = one_tool_gathered();
    let mut ledger = loop_budget::CallLedger::new();
    let p = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a1"}"#);

    let first = round_calls(&p, &g, &mut ledger);
    let sig = match &first[0].kind {
        CallKind::Invoke { .. } => first[0].sig.clone().expect("a call that runs is filed"),
        other => panic!("expected an invoke: {:?}", std::mem::discriminant(other)),
    };
    ledger.record(&sig, true, "the page");

    let second = round_calls(&p, &g, &mut ledger);
    match &second[0].kind {
        CallKind::Replay { ns_name, output } => {
            assert_eq!(ns_name, "artifacts__artifact_read");
            assert!(output.ends_with("the page"), "{output}");
        }
        _ => panic!("expected a replay"),
    }
    assert!(
        second[0].sig.is_none(),
        "a replay must not be filed as a run"
    );

    let third = round_calls(&p, &g, &mut ledger);
    match &third[0].kind {
        CallKind::Refuse { message, .. } => assert!(message.contains("twice"), "{message}"),
        _ => panic!("expected a refusal"),
    }
}

/// The wrapper and the direct call are one call: the resolver gives us the
/// resolved identity, and the ledger is keyed on it, so a model that switches
/// spelling mid-loop does not get a free round out of it.
#[test]
fn the_wrapper_and_the_direct_call_are_the_same_call() {
    let g = one_tool_gathered();
    let mut ledger = loop_budget::CallLedger::new();
    let direct = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a1"}"#);
    let sig = round_calls(&direct, &g, &mut ledger)[0]
        .sig
        .clone()
        .expect("filed");
    ledger.record(&sig, true, "the page");

    let wrapped = parsed_call(
        crate::tool_search::CALL_TOOL,
        r#"{"name":"artifacts__artifact_read","arguments_json":"{\"artifact_id\":\"a1\"}"}"#,
    );
    match &round_calls(&wrapped, &g, &mut ledger)[0].kind {
        CallKind::Replay { .. } => {}
        _ => panic!("the same call through mcp_call_tool must hit the ledger"),
    }
}

/// A round cut before any tool ran leaves the user's own turn last, and
/// back-to-back user turns is a shape some chat templates refuse. The
/// nudge joins that turn instead of opening another.
#[test]
fn the_answer_nudge_never_opens_a_second_user_turn() {
    let mut msgs = vec![json!({"role":"user","content":"go"})];
    push_answer_nudge(&mut msgs, "answer now");
    assert_eq!(msgs.len(), 1);
    assert!(
        msgs[0]["content"]
            .as_str()
            .expect("text")
            .ends_with("answer now")
    );

    // A multimodal user turn takes it as one more text part.
    let mut msgs = vec![json!({"role":"user","content":[{"type":"text","text":"go"}]})];
    push_answer_nudge(&mut msgs, "answer now");
    assert_eq!(msgs[0]["content"].as_array().expect("parts").len(), 2);

    // After a tool round it is a turn of its own, which is the usual case.
    let mut msgs = vec![json!({"role":"tool","content":"result","tool_call_id":"c1"})];
    push_answer_nudge(&mut msgs, "answer now");
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[1]["role"], "user");
}

/// The caller's own `max_tool_calls`, through the same classifier.
/// The Nth+1 call is not dispatched - it comes back as a refusal naming
/// their knob - and the loop reads `limit_reached` to take the answer round.
#[test]
fn the_callers_max_tool_calls_stops_dispatch_and_then_the_turn() {
    let g = one_tool_gathered();
    let mut ledger = loop_budget::CallLedger::with_limit(Some(1));
    let a = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a1"}"#);
    let sig = round_calls(&a, &g, &mut ledger)[0]
        .sig
        .clone()
        .expect("the first one runs");
    ledger.record(&sig, true, "the page");
    assert_eq!(
        ledger.limit_reached(),
        Some(loop_budget::Stop::ToolCalls(1))
    );

    // Different arguments, so this is a fresh call by every other rule -
    // only the caller's ceiling stops it.
    let b = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a2"}"#);
    let calls = round_calls(&b, &g, &mut ledger);
    match &calls[0].kind {
        CallKind::Refuse { message, .. } => {
            assert!(message.contains("max_tool_calls: 1"), "{message}");
        }
        _ => panic!("past the caller's limit nothing may be dispatched"),
    }
    assert!(calls[0].sig.is_none(), "a refusal is not a run");
}

/// A schema refusal never ran, so it must not consume a ledger slot -
/// otherwise a model that fixes its arguments on the retry would find its
/// corrected call counted as a repeat.
#[test]
fn a_schema_refusal_leaves_the_ledger_alone() {
    let g = one_tool_gathered();
    let mut ledger = loop_budget::CallLedger::new();
    let bad = parsed_call("artifacts__artifact_read", r#"{}"#);
    assert!(matches!(
        round_calls(&bad, &g, &mut ledger)[0].kind,
        CallKind::Refuse { .. }
    ));
    // ...and again, and it is still the SCHEMA refusal, not a repeat one.
    match &round_calls(&bad, &g, &mut ledger)[0].kind {
        CallKind::Refuse { message, .. } => {
            assert!(message.contains("did not match its schema"), "{message}");
        }
        _ => panic!("expected the schema refusal both times"),
    }
    // The corrected call still runs.
    let good = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a1"}"#);
    assert!(matches!(
        round_calls(&good, &g, &mut ledger)[0].kind,
        CallKind::Invoke { .. }
    ));
}

/// The invariant we asked for: the block lives in the SYSTEM slot, so a
/// compaction that rebuilds the messages from `req.instructions` re-renders
/// it. Putting it only in the message list meant the user's prompt survived
/// compaction and ours did not.
#[test]
fn server_instructions_go_into_the_system_slot_so_compaction_keeps_them() {
    let merged = merge_instructions(Some("You are terse.".into()), &["Use artifacts.".into()]);
    let sys = merged.expect("a system prompt");
    // The user's line goes last: a short instruction stops being obeyed
    // when 200 words of tool procedure trail it.
    assert!(
        sys.ends_with("You are terse."),
        "the user's prompt must end it: {sys}"
    );
    assert!(
        sys.starts_with("Use artifacts."),
        "tool guidance leads: {sys}"
    );
    // Whatever a later rebuild does, it starts from this string.
    let rebuilt = messages_from_input(Some(&sys), &json!("hi")).unwrap();
    assert_eq!(rebuilt[0]["role"], "system");
    assert!(
        rebuilt[0]["content"]
            .as_str()
            .unwrap()
            .contains("Use artifacts.")
    );
}

#[test]
fn merge_leaves_the_prompt_alone_when_no_server_speaks() {
    assert_eq!(
        merge_instructions(Some("mine".into()), &[]),
        Some("mine".into())
    );
    assert_eq!(merge_instructions(None, &[]), None);
}

#[test]
fn merge_creates_the_prompt_when_the_user_set_none() {
    assert_eq!(
        merge_instructions(None, &["block".into()]),
        Some("block".into())
    );
    assert_eq!(
        merge_instructions(Some("   ".into()), &["block".into()]),
        Some("block".into())
    );
}

#[test]
fn server_instructions_append_to_the_users_system_prompt() {
    // A server says how to use itself; it does not get to overwrite what
    // the user asked for. Order matters: the user's prompt leads.
    let mut msgs = vec![
        json!({"role": "system", "content": "You are terse."}),
        json!({"role": "user", "content": "hi"}),
    ];
    apply_server_instructions(&mut msgs, &["Put pages in artifacts.".to_string()]);
    assert_eq!(msgs.len(), 2, "no extra system message when one exists");
    let sys = msgs[0]["content"].as_str().unwrap();
    assert!(
        sys.ends_with("You are terse."),
        "the user's prompt must end it: {sys}"
    );
    assert!(sys.starts_with("Put pages in artifacts."), "{sys}");
}

/// The seam that broke twice. On this API the system prompt is
/// `instructions`, not an item - so a capability applied inside attachment
/// expansion, which only ever sees the items, lands as a SECOND system
/// message after conversion. That is what shipped, and it is why a
/// geotagged photo never got its map: the model was told twice-removed or
/// not at all. The assertion that matters is the COUNT.
#[test]
fn the_map_capability_joins_the_instructions_system_turn() {
    let mut msgs = messages_from_input(
        Some("You are terse."),
        &json!([{"role": "user", "content": "hi"}]),
    )
    .expect("converts");
    crate::doc::add_map_capability(&mut msgs, "43.467448, 11.885127, Arezzo");
    assert_eq!(
        msgs.iter().filter(|m| m["role"] == "system").count(),
        1,
        "one system turn, never two: {msgs:?}"
    );
    let sys = msgs[0]["content"].as_str().expect("string content");
    assert!(sys.contains("```map"), "the block must be shown: {sys}");
    assert!(
        sys.contains("43.467448, 11.885127, Arezzo"),
        "ready to copy: {sys}"
    );
    assert!(
        sys.ends_with("You are terse."),
        "the caller keeps the tail: {sys}"
    );
    assert_eq!(msgs[1]["role"], "user", "the user turn stays after it");
}

/// ...and with no instructions at all, it is the system turn.
#[test]
fn the_map_capability_creates_the_system_turn_when_there_is_none() {
    let mut msgs =
        messages_from_input(None, &json!([{"role": "user", "content": "hi"}])).expect("converts");
    crate::doc::add_map_capability(&mut msgs, "43.4, 11.8, Arezzo");
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[1]["role"], "user");
}

/// The third failure of the same seam, and the one `messages` alone cannot
/// survive: every compaction rebuilds the message list from
/// `req.instructions`, so a capability patched into `messages` is gone
/// while the photo line, which lives in the user item, sails through - the
/// model keeps the coordinates and loses the reason to draw them. The
/// handler folds it into the instructions slot; this pins that a rebuild
/// re-renders it.
#[test]
fn the_map_capability_survives_a_rebuild_from_instructions() {
    let text = crate::doc::map_capability_text("43.467157, 11.885395, Arezzo");
    let instructions =
        merge_instructions(Some("You are terse.".into()), &[text]).expect("some instructions");
    let rebuilt = messages_from_input(
        Some(&instructions),
        &json!([{"role": "user", "content": "hi"}]),
    )
    .expect("converts");
    assert_eq!(
        rebuilt.iter().filter(|m| m["role"] == "system").count(),
        1,
        "still one system turn: {rebuilt:?}"
    );
    let sys = rebuilt[0]["content"].as_str().expect("string content");
    assert!(
        sys.contains("```map"),
        "the block survived the rebuild: {sys}"
    );
    assert!(
        sys.ends_with("You are terse."),
        "and the caller still owns the tail: {sys}"
    );
}

#[test]
fn server_instructions_create_a_system_prompt_when_there_is_none() {
    let mut msgs = vec![json!({"role": "user", "content": "hi"})];
    apply_server_instructions(&mut msgs, &["Use artifacts.".to_string()]);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "Use artifacts.");
    assert_eq!(msgs[1]["role"], "user", "the user turn must stay after it");
}

#[test]
fn several_servers_are_joined_and_none_is_dropped() {
    let mut msgs = vec![json!({"role": "user", "content": "hi"})];
    apply_server_instructions(&mut msgs, &["First.".to_string(), "Second.".to_string()]);
    let sys = msgs[0]["content"].as_str().unwrap();
    assert!(sys.contains("First.") && sys.contains("Second."), "{sys}");
}

#[test]
fn no_instructions_changes_nothing() {
    let before = vec![json!({"role": "user", "content": "hi"})];
    let mut msgs = before.clone();
    apply_server_instructions(&mut msgs, &[]);
    assert_eq!(
        msgs, before,
        "a server with no instructions must not add a system turn"
    );
}

#[test]
fn image_bearing_content_arrays_pass_through() {
    let input = json!([{
        "type": "message", "role": "user",
        "content": [
            {"type": "input_text", "text": "What is this?"},
            {"type": "input_image", "image_url": "data:image/bmp;base64,AAAA"}
        ]
    }]);
    let msgs = messages_from_input(None, &input).unwrap();
    // array kept verbatim so the template renders the image_pad slot
    assert!(msgs[0]["content"].is_array(), "{:?}", msgs[0]);
    let urls = crate::chat::find_images(&msgs).unwrap();
    assert_eq!(
        urls.iter().map(|r| r.url.as_ref()).collect::<Vec<_>>(),
        ["data:image/bmp;base64,AAAA"]
    );
}

#[test]
fn text_only_content_arrays_flatten() {
    let input = json!([{
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "hello "},
                    {"type": "input_text", "text": "world"}]
    }]);
    let msgs = messages_from_input(None, &input).unwrap();
    assert_eq!(msgs[0]["content"], "hello world");
}

/// Echoing the whole output array back is the documented multi-turn pattern
/// for Responses, and that array contains `reasoning` items - so refusing
/// them 400s a conforming client. It used to.
#[test]
fn a_reasoning_item_rides_on_the_assistant_message_it_precedes() {
    let input = json!([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "id": "rs_1", "summary": [],
         "content": [{"type": "reasoning_text", "text": "let me think"}]},
        {"type": "message", "role": "assistant", "content": "a"},
        {"type": "message", "role": "user", "content": "q2"},
    ]);
    let msgs = messages_from_input(None, &input).unwrap();
    // three turns, not four: the reasoning is a FIELD, never a turn of its own
    assert_eq!(msgs.len(), 3, "reasoning must not become its own message");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(msgs[1]["reasoning_content"], "let me think");
    // and it attaches to the message that FOLLOWS it, not the one before
    assert!(msgs[0].get("reasoning_content").is_none());
    assert!(msgs[2].get("reasoning_content").is_none());
}

/// A provider that withholds the chain and ships only a summary still has
/// something to say; take it rather than drop the item silently.
#[test]
fn a_summary_only_reasoning_item_still_carries_its_text() {
    let input = json!([
        {"type": "reasoning", "id": "rs_1",
         "summary": [{"type": "summary_text", "text": "considered options"}]},
        {"type": "message", "role": "assistant", "content": "a"},
    ]);
    let msgs = messages_from_input(None, &input).unwrap();
    assert_eq!(msgs[0]["reasoning_content"], "considered options");
}

/// An empty one attaches nothing rather than an empty string, so a template
/// that tests `is string` does not render a hollow think block for it.
#[test]
fn an_empty_reasoning_item_attaches_nothing() {
    let input = json!([
        {"type": "reasoning", "id": "rs_1", "summary": [], "content": []},
        {"type": "message", "role": "assistant", "content": "a"},
    ]);
    let msgs = messages_from_input(None, &input).unwrap();
    assert!(msgs[0].get("reasoning_content").is_none());
}

/// One response's output - [reasoning] [message] [function_call..] - is ONE
/// assistant turn when an agent echoes it back, the documented Responses
/// multi-turn pattern and what the OpenAI Agents SDK sends every step. It
/// used to become one assistant message per call with the reasoning dropped:
/// the model lost its own chain of thought between tool steps, and every
/// step re-prefilled the previous reply instead of resuming from it.
#[test]
fn a_tool_step_echoed_back_is_one_assistant_turn_with_its_reasoning() {
    let input = json!([
        {"type": "message", "role": "user", "content": "fix the tests"},
        {"type": "reasoning", "id": "rs_1", "summary": [],
         "content": [{"type": "reasoning_text", "text": "look around first"}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "list_files",
         "arguments": "{\"pattern\":\"**/*\"}", "status": "completed"},
        {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "read_file",
         "arguments": "{\"path\":\"README.md\"}", "status": "completed"},
        {"type": "function_call_output", "call_id": "call_1", "output": "README.md"},
        {"type": "function_call_output", "call_id": "call_2", "output": "# freightlib"},
        {"type": "reasoning", "id": "rs_2", "summary": [],
         "content": [{"type": "reasoning_text", "text": "run them"}]},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Running.", "annotations": []}]},
        {"type": "function_call", "id": "fc_3", "call_id": "call_3", "name": "run_shell",
         "arguments": "{\"command\":\"make test\"}", "status": "completed"},
        {"type": "function_call_output", "call_id": "call_3", "output": "ok"},
    ]);
    let msgs = messages_from_input(Some("sys"), &input).unwrap();
    let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(
        roles,
        [
            "system",
            "user",
            "assistant",
            "tool",
            "tool",
            "assistant",
            "tool"
        ]
    );
    assert_eq!(msgs[2]["reasoning_content"], "look around first");
    assert_eq!(msgs[2]["tool_calls"].as_array().map(Vec::len), Some(2));
    assert_eq!(msgs[5]["reasoning_content"], "run them");
    assert_eq!(msgs[5]["content"], "Running.");
    assert_eq!(msgs[5]["tool_calls"][0]["function"]["name"], "run_shell");

    // and on Qwen3.8 it renders byte-for-byte as its chat-completions twin
    let qwen = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
    let render = |m: &[Value]| {
        let m = crate::chat_template::normalize_messages(m);
        crate::chat_template::render(qwen, &m, None, None).expect("render")
    };
    let call = |id: &str, name: &str, args: Value| json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}});
    let chat = [
        json!({"role": "system", "content": "sys"}),
        json!({"role": "user", "content": "fix the tests"}),
        json!({"role": "assistant", "reasoning_content": "look around first",
               "tool_calls": [call("call_1", "list_files", json!({"pattern": "**/*"})),
                              call("call_2", "read_file", json!({"path": "README.md"}))]}),
        json!({"role": "tool", "content": "README.md", "tool_call_id": "call_1"}),
        json!({"role": "tool", "content": "# freightlib", "tool_call_id": "call_2"}),
        json!({"role": "assistant", "content": "Running.", "reasoning_content": "run them",
               "tool_calls": [call("call_3", "run_shell", json!({"command": "make test"}))]}),
        json!({"role": "tool", "content": "ok", "tool_call_id": "call_3"}),
    ];
    let out = render(&msgs);
    assert_eq!(out, render(&chat), "surfaces diverged");
    assert!(
        out.contains("<think>\nlook around first\n</think>"),
        "{out}"
    );
}

/// A reasoning item no assistant output followed (a reply that ended inside
/// its thinking) is dropped, never carried onto some later turn.
#[test]
fn an_orphan_reasoning_item_never_rides_a_later_turn() {
    let input = json!([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "id": "rs_1", "summary": [],
         "content": [{"type": "reasoning_text", "text": "stale"}]},
        {"type": "message", "role": "user", "content": "q2"},
        {"type": "message", "role": "assistant", "content": "a"},
    ]);
    let msgs = messages_from_input(None, &input).unwrap();
    assert!(msgs.iter().all(|m| m.get("reasoning_content").is_none()));
}

#[test]
fn truncation_drops_whole_turns_from_the_front() {
    let mut msgs = vec![
        json!({"role": "system", "content": "sys"}),
        json!({"role": "user", "content": "q1"}),
        json!({"role": "assistant", "content": "a1"}),
        json!({"role": "user", "content": "q2"}),
        json!({"role": "assistant", "content": "a2"}),
        json!({"role": "user", "content": "pending"}),
    ];
    assert_eq!(drop_leading_turn(&mut msgs), 2, "q1 + a1 leave together");
    assert_eq!(
        msgs[0]["content"], "sys",
        "the system turn is never dropped"
    );
    assert_eq!(msgs[1]["content"], "q2");
    assert_eq!(drop_leading_turn(&mut msgs), 2);
    // only the pending turn is left: nothing droppable, the caller 400s
    assert_eq!(drop_leading_turn(&mut msgs), 0);
    assert_eq!(msgs.len(), 2);
}

#[test]
fn truncation_never_eats_into_the_pending_turn() {
    // an agent-loop round: the pending question, then tool round-trips.
    // Dropping stops at the pending user message - draining into it would
    // leave the model a dangling tool result with no request.
    let mut msgs = vec![
        json!({"role": "user", "content": "old"}),
        json!({"role": "assistant", "content": "answer"}),
        json!({"role": "user", "content": "pending"}),
        json!({"role": "assistant", "tool_calls": [{"id": "c1"}]}),
        json!({"role": "tool", "content": "result", "tool_call_id": "c1"}),
    ];
    assert_eq!(
        drop_leading_turn(&mut msgs),
        2,
        "the finished exchange only"
    );
    assert_eq!(msgs[0]["content"], "pending");
    assert_eq!(
        drop_leading_turn(&mut msgs),
        0,
        "the turn in flight is untouchable"
    );
    assert_eq!(msgs.len(), 3);
}

/// A compaction's summarization prompt must share the live conversation's
/// prefix up to its last span token - that is what makes its prefill a
/// radix hit. It did not on Qwen3.8: the pass rendered thinking-off, and
/// Qwen3.8 writes its reasoning-effort instruction into the SYSTEM block
/// while thinking is on, so the two parted a few tokens in and an agent
/// loop's compaction re-prefilled its whole 146K-token span. Rendered with
/// the conversation's own kwargs, the span is a strict prefix again.
#[test]
fn a_summary_render_shares_the_live_prefix_on_qwen38() {
    let qwen = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
    let tools = [json!({"type": "function", "function": {"name": "read_file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}})];
    let live = [
        json!({"role": "system", "content": "You are an agent."}),
        json!({"role": "user", "content": "Audit the ledger."}),
        json!({"role": "assistant", "reasoning_content": "read it first",
               "tool_calls": [{"id": "c1", "type": "function", "function": {
                   "name": "read_file", "arguments": {"path": "ledger_01.txt"}}}]}),
        json!({"role": "tool", "content": "REC 01-0001", "tool_call_id": "c1"}),
    ];
    let mut span = live.to_vec();
    span.push(json!({"role": "user",
                     "content": crate::context_management::OA_COMPACT_INSTRUCTIONS}));
    let render = |m: &[Value], kw: Option<&Value>| {
        let m = crate::chat_template::normalize_messages(m);
        crate::chat_template::render(qwen, &m, Some(&tools), kw).expect("render")
    };
    let live_out = render(&live, None);
    let body = live_out
        .strip_suffix("<|im_start|>assistant\n<think>\n")
        .expect("the live prompt opens a think block");
    assert!(
        render(&span, None).starts_with(body),
        "same kwargs: a prefix"
    );
    // the old thinking-off render moved the system block itself
    let off = json!({"enable_thinking": false});
    let off_out = render(&span, Some(&off));
    assert!(!off_out.starts_with(body));
    assert!(!off_out.contains("Reasoning effort is set to"));
}
