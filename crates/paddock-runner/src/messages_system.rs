//! Mid-conversation `role: "system"` messages on `POST /v1/messages`.
//!
//! Anthropic's operator channel inside `messages[]`: a system message appended
//! after the history instead of an edit to the top-level `system` field, so the
//! cached prefix survives. Claude Code sends one on every request (its
//! `# Environment` block, carrying `output_config.effort` too), and this surface
//! used to 400 the whole request with `invalid message role "system"`.
//!
//! One message shape carries four things, and they land in two places:
//!
//! - TEXT is transcript: rendered where it sits (see [`place_system_text`]),
//!   and dropped once cleared when it is turn-scoped (`clear_at`).
//! - `output_config.effort` and `tool_addition` / `tool_removal` blocks are
//!   REQUEST controls: they change the effort rung and the rendered tool set
//!   for the generation, so every render path applies [`system_controls`].
//!
//! Placement follows Anthropic's rule (a system message with text or a tool
//! change follows a user turn and is last or followed by an assistant turn;
//! an effort-only one may sit anywhere) - the user-turn fold relies on the first
//! half, since the text joins the user turn it follows.

use serde_json::{Value, json};

/// One `role: "system"` message, parsed and shape-checked.
pub(crate) struct SystemMessage {
    pub(crate) text: String,
    pub(crate) effort: Option<String>,
    /// (tool name, true = addition / false = removal), in block order
    pub(crate) tool_changes: Vec<(String, bool)>,
    /// `clear_at: "next_user_message"` - renders for its own turn only
    pub(crate) turn_scoped: bool,
}

impl SystemMessage {
    /// Anything the placement rule applies to. An effort-only message carries
    /// no text, so Anthropic lets it sit anywhere, first included.
    fn placed(&self) -> bool {
        !self.text.trim().is_empty() || !self.tool_changes.is_empty()
    }
}

fn role(m: &Value) -> Option<&str> {
    m.get("role").and_then(Value::as_str)
}

pub(crate) fn parse_system_message(m: &Value) -> Result<SystemMessage, String> {
    let mut out = SystemMessage {
        text: String::new(),
        effort: None,
        tool_changes: Vec::new(),
        turn_scoped: false,
    };
    match m.get("content") {
        Some(Value::String(s)) => out.text = s.clone(),
        Some(Value::Array(blocks)) => {
            let mut texts = Vec::new();
            for b in blocks {
                // `cache_control` on a text block is accepted and ignored, the
                // same as everywhere else on this surface (the radix cache
                // needs no breakpoints)
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => texts.push(b.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some(kind @ ("tool_addition" | "tool_removal")) => {
                        let name = b
                            .get("tool")
                            .filter(|t| t.get("type").and_then(Value::as_str) == Some("tool_reference"))
                            .and_then(|t| t.get("name"))
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                format!("{kind} needs `tool: {{\"type\": \"tool_reference\", \"name\": ...}}`")
                            })?;
                        out.tool_changes
                            .push((name.to_owned(), kind == "tool_addition"));
                    }
                    other => {
                        return Err(format!(
                            "a role \"system\" message carries text, tool_addition and tool_removal \
                             blocks only - got {other:?}"
                        ));
                    }
                }
            }
            out.text = texts.join("\n\n");
        }
        None | Some(Value::Null) => {}
        Some(_) => return Err("message content must be a string or an array of blocks".into()),
    }
    if let Some(cfg) = m.get("output_config") {
        let (effort, format) = crate::messages::parse_output_config(Some(cfg))?;
        if format.is_some() {
            return Err("a role \"system\" message's output_config takes `effort` only".into());
        }
        out.effort = effort;
    }
    match m.get("clear_at") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s == "next_user_message" => out.turn_scoped = true,
        Some(other) => {
            return Err(format!(
                "unsupported clear_at {other} (this server serves \"next_user_message\")"
            ));
        }
    }
    if !out.placed() && out.effort.is_none() {
        return Err(
            "a role \"system\" message needs text, a tool change, or output_config.effort".into(),
        );
    }
    Ok(out)
}

/// An assistant turn that paused mid server tool (`pause_turn`) - the one
/// non-user turn Anthropic lets a system message follow.
fn ends_in_server_tool_use(m: &Value) -> bool {
    m.get("content")
        .and_then(Value::as_array)
        .and_then(|blocks| blocks.last())
        .and_then(|b| b.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|t| t == "server_tool_use" || t.ends_with("_tool_result"))
}

/// Anthropic's placement rule for `messages[i]`. Neighbouring system messages
/// are looked past, so a run of them (text + an effort-only one, say) shares
/// one slot.
fn check_placement(messages: &[Value], i: usize) -> Result<(), String> {
    let before = messages[..i]
        .iter()
        .rev()
        .find(|m| role(m) != Some("system"));
    let follows_ok = match before {
        Some(m) if role(m) == Some("user") => true,
        Some(m) if role(m) == Some("assistant") => ends_in_server_tool_use(m),
        _ => false,
    };
    if !follows_ok {
        return Err(
            "a role \"system\" message must follow a user message (the opening instructions \
             belong in the top-level `system` field)"
                .into(),
        );
    }
    let after = messages[i + 1..].iter().find(|m| role(m) != Some("system"));
    if after.is_some_and(|m| role(m) != Some("assistant")) {
        return Err(
            "a role \"system\" message must be the last message or be followed by an \
             assistant message"
                .into(),
        );
    }
    Ok(())
}

/// The transcript half of `messages[i]` (a system message): checks placement
/// and returns the text to render there, or None when there is nothing to
/// render - an effort/tool-only message, or a turn-scoped one that a later
/// user message has cleared. Cleared means gone from the render, which is
/// Anthropic's semantics ("stops rendering once a later user message
/// exists"); the prefix shifts only from the old reminder's slot on, which was
/// the tail of the previous request anyway.
pub(crate) fn render_text(messages: &[Value], i: usize) -> Result<Option<String>, String> {
    let sm = parse_system_message(&messages[i])?;
    if sm.placed() {
        check_placement(messages, i)?;
    }
    let cleared = sm.turn_scoped && messages[i + 1..].iter().any(|m| role(m) == Some("user"));
    if cleared || sm.text.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(sm.text))
}

/// Render a mid-conversation system message where it sits: as the template's
/// own system turn when it renders one there itself (`native`, probed at
/// load; Kolibri's does, and keeps an assistant turn's reasoning only after
/// the last USER message, so a reminder folded into a user turn would drop
/// the model's reasoning mid tool loop), else [`append_system_reminder`].
pub(crate) fn place_system_text(msgs: &mut Vec<Value>, text: &str, native: bool) {
    if native {
        msgs.push(json!({"role": "system", "content": text}));
    } else {
        append_system_reminder(msgs, text);
    }
}

/// The fold for templates that cannot render a system turn mid-conversation:
/// most allow one only first (Qwen's raises "System message must be at the
/// beginning"), and gemma has no system turn at all. Folding the text into the leading system prompt
/// would render everywhere, but it moves the text to the FRONT - rewriting the
/// prefix of every cached turn, which is exactly the cost Anthropic's feature
/// exists to avoid. Claude Code sends one of these on every request, so that
/// would be a full re-prefill of the session each turn.
///
/// So it rides the user turn it follows (a new user turn when that turn
/// rendered as tool results only), in the `<system-reminder>` wrapper
/// Anthropic's docs give as the fallback for models without the role: the
/// position is kept, the prefix is untouched, and every template renders it.
pub(crate) fn append_system_reminder(msgs: &mut Vec<Value>, text: &str) {
    let block = crate::chat_template::system_reminder(text);
    if let Some(last) = msgs.last_mut()
        && role(last) == Some("user")
    {
        match last.get_mut("content") {
            Some(Value::String(s)) => {
                s.push_str("\n\n");
                s.push_str(&block);
                return;
            }
            Some(Value::Array(parts)) => {
                parts.push(json!({"type": "text", "text": block}));
                return;
            }
            _ => {}
        }
    }
    msgs.push(json!({"role": "user", "content": block}));
}

/// The request-control half, as of the end of the conversation.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct SystemControls {
    /// The last per-message effort. It holds "until a later system message
    /// changes it", so the last one is the rung for this generation, and it
    /// overrides the top-level `output_config.effort`.
    pub(crate) effort: Option<String>,
    /// Tools out of the rendered set: removed and not re-added since.
    pub(crate) removed: Vec<String>,
}

/// Fold every system message's effort and tool changes, in order. A change
/// must name a tool declared in `tools` - Anthropic requires an added tool to
/// be declared up front (with `defer_loading`), and a name that matches
/// nothing is a caller bug worth a 400, not a silent no-op.
///
/// `tool_addition` renders nothing new here: this surface already renders
/// every declared tool, deferred or not (`convert_tools` does not read
/// `defer_loading`), so the addition's tool is already in the set - it only
/// undoes an earlier removal.
pub(crate) fn system_controls(
    messages: &[Value],
    tools: Option<&[Value]>,
) -> Result<SystemControls, String> {
    let declared: Vec<&str> = tools
        .unwrap_or_default()
        .iter()
        .filter_map(|t| t.get("name").and_then(Value::as_str))
        .collect();
    let mut out = SystemControls::default();
    for m in messages.iter().filter(|m| role(m) == Some("system")) {
        let sm = parse_system_message(m)?;
        if sm.effort.is_some() {
            out.effort = sm.effort;
        }
        for (name, added) in sm.tool_changes {
            if !declared.contains(&name.as_str()) {
                let kind = if added {
                    "tool_addition"
                } else {
                    "tool_removal"
                };
                return Err(format!(
                    "{kind} names {name:?}, which is not declared in `tools`"
                ));
            }
            out.removed.retain(|n| n != &name);
            if !added {
                out.removed.push(name);
            }
        }
    }
    Ok(out)
}

/// Drop removed tools from a converted (chat-shaped) tool list. Removing the
/// last one leaves no tools at all rather than an empty list, so nothing
/// downstream mistakes "tools: []" for a tool-calling request.
///
/// This does change the rendered tool header, and with it the prefix - the
/// templates render tools up front, so there is no in-place way to take one
/// away. Correct availability is worth one re-prefill at the turn a tool
/// actually leaves; later turns share the new header.
pub(crate) fn drop_removed_tools(tools: &mut Option<Vec<Value>>, removed: &[String]) {
    if removed.is_empty() {
        return;
    }
    if let Some(ts) = tools.as_mut() {
        ts.retain(|t| {
            let name = t["function"]["name"].as_str().unwrap_or("");
            !removed.iter().any(|r| r == name)
        });
        if ts.is_empty() {
            *tools = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request shape Claude Code 2.1.280 sends, trimmed (captured
    /// 2026-09-23): the prompt as a user turn, then the `# Environment` block
    /// as a trailing system message carrying the effort rung and a cache
    /// breakpoint.
    fn claude_code_messages() -> Vec<Value> {
        vec![
            json!({"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>\nctx\n</system-reminder>"},
                {"type": "text", "text": "Say hi"},
            ]}),
            json!({"role": "system", "output_config": {"effort": "medium"}, "content": [
                {"type": "text", "text": "# Environment\ncwd: /home", "cache_control": {"type": "ephemeral"}},
            ]}),
        ]
    }

    #[test]
    fn claude_code_trailing_system_renders_and_sets_effort() {
        let msgs = claude_code_messages();
        assert_eq!(
            render_text(&msgs, 1).expect("valid").as_deref(),
            Some("# Environment\ncwd: /home")
        );
        let c = system_controls(&msgs, None).expect("valid");
        assert_eq!(c.effort.as_deref(), Some("medium"));
        assert!(c.removed.is_empty());
    }

    #[test]
    fn placement_follows_anthropics_rule() {
        let user = json!({"role": "user", "content": "hi"});
        let asst = json!({"role": "assistant", "content": "hello"});
        let sys = json!({"role": "system", "content": "be terse"});
        // first message: the opening prompt belongs in top-level `system`
        assert!(render_text(&[sys.clone(), user.clone()], 0).is_err());
        // after a plain assistant turn
        assert!(render_text(&[user.clone(), asst.clone(), sys.clone()], 2).is_err());
        // followed by a user turn
        assert!(render_text(&[user.clone(), sys.clone(), user.clone()], 1).is_err());
        // last, or followed by an assistant turn
        assert!(render_text(&[user.clone(), sys.clone()], 1).is_ok());
        assert!(render_text(&[user.clone(), sys.clone(), asst.clone()], 1).is_ok());
        // after an assistant turn paused in server-tool use
        let paused = json!({"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
        ]});
        assert!(render_text(&[user.clone(), paused, sys.clone()], 2).is_ok());
        // a run of system messages shares one slot
        let effort = json!({"role": "system", "content": [], "output_config": {"effort": "low"}});
        assert!(render_text(&[user.clone(), effort.clone(), sys.clone()], 2).is_ok());
    }

    #[test]
    fn effort_only_messages_sit_anywhere_and_render_nothing() {
        let effort = json!({"role": "system", "content": [], "output_config": {"effort": "low"}});
        let msgs = [
            json!({"role": "user", "content": "plan"}),
            json!({"role": "assistant", "content": "ok"}),
            effort.clone(),
            json!({"role": "user", "content": "go"}),
        ];
        assert_eq!(render_text(&msgs, 2).expect("exempt"), None);
        assert_eq!(
            render_text(&[effort, json!({"role": "user", "content": "x"})], 0)
                .expect("first is fine"),
            None
        );
        assert_eq!(
            system_controls(&msgs, None).expect("ok").effort.as_deref(),
            Some("low")
        );
    }

    #[test]
    fn the_last_effort_wins() {
        let msgs = [
            json!({"role": "user", "content": "a"}),
            json!({"role": "system", "content": [], "output_config": {"effort": "low"}}),
            json!({"role": "assistant", "content": "b"}),
            json!({"role": "user", "content": "c"}),
            json!({"role": "system", "content": "x", "output_config": {"effort": "xhigh"}}),
        ];
        assert_eq!(
            system_controls(&msgs, None).expect("ok").effort.as_deref(),
            Some("xhigh")
        );
    }

    #[test]
    fn turn_scoped_text_clears_at_the_next_user_message() {
        let reminder =
            json!({"role": "system", "clear_at": "next_user_message", "content": "check inbox"});
        let live = [
            json!({"role": "user", "content": "run it"}),
            reminder.clone(),
        ];
        assert_eq!(
            render_text(&live, 1).expect("ok").as_deref(),
            Some("check inbox")
        );
        let later = [
            json!({"role": "user", "content": "run it"}),
            reminder,
            json!({"role": "assistant", "content": "done"}),
            json!({"role": "user", "content": "next"}),
        ];
        assert_eq!(render_text(&later, 1).expect("ok"), None);
    }

    #[test]
    fn malformed_shapes_name_themselves() {
        let user = json!({"role": "user", "content": "hi"});
        for (sys, want) in [
            (
                json!({"role": "system", "content": [{"type": "image", "source": {}}]}),
                "text, tool_addition",
            ),
            (
                json!({"role": "system", "content": "x", "clear_at": "never"}),
                "unsupported clear_at",
            ),
            (
                json!({"role": "system", "content": "x", "output_config": {"format": {"type": "json_schema", "schema": {}}}}),
                "takes `effort` only",
            ),
            (json!({"role": "system", "content": []}), "needs text"),
            (
                json!({"role": "system", "content": [{"type": "tool_removal", "tool": {"name": "t"}}]}),
                "tool_reference",
            ),
        ] {
            let e = render_text(&[user.clone(), sys], 1).expect_err(want);
            assert!(e.contains(want), "{e:?} lacks {want:?}");
        }
    }

    #[test]
    fn tool_changes_fold_in_order_against_declared_tools() {
        let tools = [
            json!({"name": "get_weather", "input_schema": {"type": "object"}}),
            json!({"name": "get_forecast", "input_schema": {"type": "object"}, "defer_loading": true}),
        ];
        let removal = |n: &str| json!({"type": "tool_removal", "tool": {"type": "tool_reference", "name": n}});
        let addition = |n: &str| json!({"type": "tool_addition", "tool": {"type": "tool_reference", "name": n}});
        let user = json!({"role": "user", "content": "hi"});
        let asst = json!({"role": "assistant", "content": "ok"});
        let msgs = [
            user.clone(),
            json!({"role": "system", "content": [removal("get_weather"), removal("get_forecast")]}),
            asst.clone(),
            user.clone(),
            json!({"role": "system", "content": [addition("get_forecast")]}),
        ];
        let c = system_controls(&msgs, Some(&tools)).expect("ok");
        assert_eq!(c.removed, vec!["get_weather".to_owned()]);

        let mut converted = Some(vec![
            json!({"type": "function", "function": {"name": "get_weather"}}),
            json!({"type": "function", "function": {"name": "get_forecast"}}),
        ]);
        drop_removed_tools(&mut converted, &c.removed);
        assert_eq!(
            converted.expect("one left")[0]["function"]["name"],
            "get_forecast"
        );

        let mut one = Some(vec![
            json!({"type": "function", "function": {"name": "get_weather"}}),
        ]);
        drop_removed_tools(&mut one, &c.removed);
        assert!(one.is_none(), "an emptied tool list is no tools");

        let unknown = [
            user,
            json!({"role": "system", "content": [removal("nope")]}),
        ];
        let e = system_controls(&unknown, Some(&tools)).expect_err("undeclared");
        assert!(e.contains("not declared"), "{e}");
    }

    #[test]
    fn reminders_join_the_user_turn_they_follow() {
        let mut flat = vec![json!({"role": "user", "content": "hi"})];
        append_system_reminder(&mut flat, "be terse");
        assert_eq!(flat.len(), 1);
        assert_eq!(
            flat[0]["content"],
            "hi\n\n<system-reminder>\nbe terse\n</system-reminder>"
        );

        let mut parts = vec![json!({"role": "user", "content": [{"type": "image"}]})];
        append_system_reminder(&mut parts, "x");
        assert_eq!(parts[0]["content"][1]["type"], "text");

        // a user turn of tool results only renders as tool messages; the
        // reminder opens a user turn after them
        let mut tools = vec![json!({"role": "tool", "content": "42", "tool_call_id": "t1"})];
        append_system_reminder(&mut tools, "x");
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[1]["role"], "user");
    }

    #[test]
    fn a_native_template_gets_the_system_turn_itself() {
        // after tool results too: no user turn is invented to carry it
        let mut msgs = vec![json!({"role": "tool", "content": "42", "tool_call_id": "t1"})];
        place_system_text(&mut msgs, "keep going", true);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1], json!({"role": "system", "content": "keep going"}));
        let mut folded = vec![json!({"role": "user", "content": "hi"})];
        place_system_text(&mut folded, "x", false);
        assert_eq!(folded.len(), 1);
    }
}
