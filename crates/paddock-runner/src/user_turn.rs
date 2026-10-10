//! Does a request open a USER turn? True when the conversation's latest
//! user/assistant message is the user's own words, not tool results coming
//! back. The engine holds such a prompt's end checkpoint as an anchor
//! (`GenRequest::user_turn`): chat templates drop the reasoning of every turn
//! before the latest user message, so the next user message re-renders this
//! turn from its first reply on, and a hybrid model resumes only at a
//! checkpoint - this one, if a long tool loop has not pushed it out.

use serde_json::Value;

/// Anthropic Messages: tool results ride `user` messages as `tool_result`
/// blocks; system messages mid-conversation (Claude Code sends them) are
/// skipped.
pub(crate) fn anthropic(messages: &[Value]) -> bool {
    let Some(last) = messages
        .iter()
        .rev()
        .find(|m| matches!(role(m), "user" | "assistant"))
    else {
        return false;
    };
    role(last) == "user"
        && !last["content"]
            .as_array()
            .is_some_and(|c| c.iter().any(|b| b["type"] == "tool_result"))
}

/// OpenAI chat: tool results are their own `tool` (or legacy `function`)
/// messages, so the latest non-system message being `user` is the test.
pub(crate) fn chat(messages: &[Value]) -> bool {
    messages
        .iter()
        .rev()
        .find(|m| !matches!(role(m), "system" | "developer"))
        .is_some_and(|m| role(m) == "user")
}

/// OpenAI Responses: `input` is a string (one user message) or items, where
/// tool results are `function_call_output` items.
pub(crate) fn responses(input: &Value) -> bool {
    match input {
        Value::String(_) => true,
        Value::Array(items) => items
            .iter()
            .rev()
            .find(|it| !matches!(role(it), "system" | "developer"))
            .is_some_and(|it| {
                role(it) == "user" && it["type"].as_str().unwrap_or("message") == "message"
            }),
        _ => false,
    }
}

fn role(m: &Value) -> &str {
    m["role"].as_str().unwrap_or("")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn tool_results_are_not_a_user_turn() {
        let user = json!({"role": "user", "content": "fix the tests"});
        let call = json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]});
        let result = json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"},
                                                         {"type": "text", "text": "<system-reminder>x</system-reminder>"}]});
        let sys = json!({"role": "system", "content": "<total_tokens>1</total_tokens>"});
        assert!(super::anthropic(std::slice::from_ref(&user)));
        assert!(super::anthropic(&[user.clone(), sys.clone()]));
        assert!(!super::anthropic(&[
            user.clone(),
            call.clone(),
            result.clone()
        ]));
        assert!(!super::anthropic(&[
            user.clone(),
            call.clone(),
            result,
            sys
        ]));
        assert!(!super::anthropic(&[user, call]));

        let tool = json!({"role": "tool", "tool_call_id": "t", "content": "ok"});
        assert!(super::chat(&[
            json!({"role": "user", "content": "hi"}),
            json!({"role": "system", "content": "s"})
        ]));
        assert!(!super::chat(&[
            json!({"role": "user", "content": "hi"}),
            tool
        ]));

        assert!(super::responses(&json!("hi")));
        assert!(super::responses(
            &json!([{"role": "user", "content": "hi"}])
        ));
        assert!(!super::responses(
            &json!([{"role": "user", "content": "hi"},
            {"type": "function_call_output", "call_id": "c", "output": "ok"}])
        ));
    }
}
