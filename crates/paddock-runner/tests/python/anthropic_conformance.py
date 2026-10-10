#!/usr/bin/env python
"""Anthropic-SDK conformance gate for the paddock server.

Driven by tests/sdk_conformance.rs against a live loopback server, using the
official `anthropic` package. The SDK's MessageStream accumulator enforces
the event protocol (message_start -> content_block_start/delta/stop ->
message_delta -> message_stop) - a malformed sequence breaks accumulation.
Each section prints `ok - <name>`; any failure exits nonzero.

Usage:
    python anthropic_conformance.py --base-url http://127.0.0.1:PORT \
        --model qwen35-9b --dialect qwen [--vision]
"""

import argparse
import base64
import json
import struct
import sys

import anthropic
from anthropic import Anthropic

SECTIONS = 0


def ok(name):
    global SECTIONS
    SECTIONS += 1
    print(f"ok - {name}", flush=True)


def red_blue_bmp():
    """256x160 24-bit BMP, red left / blue right - same canvas as the OpenAI
    gate and qwen_vision_http.rs."""
    w, h = 256, 160
    img = w * h * 3
    head = b"BM" + struct.pack(
        "<IIIIiiHHIIIIII", 54 + img, 0, 54, 40, w, h, 1, 24, 0, img, 2835, 2835, 0, 0
    )
    row = b"\x00\x00\xff" * (w // 2) + b"\xff\x00\x00" * (w // 2)
    return head + row * h


# The same picture re-encoded once (Pillow 12.2), embedded so the gate needs no
# imaging dependency. webp + gif complete anthropic's media_type enum
# (jpeg/png/gif/webp); the webp is lossy VP8 - the wild form of the format.
RED_BLUE_WEBP = "UklGRsQAAABXRUJQVlA4ILgAAABQDgCdASoAAaAAPm02mUkkIyKhIkgAgA2JZ27hdVERgAfiBsWPAPxA/VUAJXF6ZWtE5e07xWZU5NYzvFZl4txipneKzLxeVlp4rMvF6Z0xg6eZeL0zum/QZOXtO8VhfywCjGd4rL5LaMZ3isy8O18neKzLxelAwdPMqAAA/v7ge/rRMc1d334PlgkC/+PfNg0ld/jzr+sydFcH/k2Xlrx2agwgAzAycI2CvQHyK9AfIr0B8ivQEAAA"
RED_BLUE_GIF = "R0lGODdhAAGgAIEAAP8AAAAA/wAAAAAAACwAAAAAAAGgAEAI/wABCBxIsKDBgwgTKlzIsKHDhwgDSJxIsaLFixgzatzIsaPHjxghihxJsqRJkSBTqlzJsmXKkzBjypwZ0aXNmzhzXqTJs6dPhzqDCh3q8afRo0aJKl2qFKnTpzCZSp1qE6rVq0Cpat3aEavXrwW5ih1bEaxZr2TTij3LFqrat1Tbyj0Kt+7SuXh72t0rNK9fmXwD4/xL2KTgwy0LK0aJuDHIxZAbOp7cNbLlmpQz77zMmaDmz2U7iwZNOoDo0aU/n+6cWvXqy601v4Ydm/Jsy7Vt34ace/Ju3r0b/14cXPjwwsURH0eeXPByws2dP/cbPfB06tX3Xs+bXfv2ud3tfv8HHx7ueLnlzZ9nm/7tevbt074/G1/+fLD1yd7Hn3/tfrT9cfUfgAFqNSBWBW514FUJGrigWw1O9SCEETI14VMVSnWhUxlauCFdHTb1YVIhEjUiiSX2daJPKQ61IostBvWiXjHqNCNPNdp440w55rQjjz3e9CNgQVY1ZFRFunQkkkmytORJTSb2ZElROjklSVWudCWWWb60JWNdfvQlmGFWNmZWZZp5JkNpFrWmZG1y9CaccWo0J5t12nmnQnnquSdmfVr0Z0KBhjToQYVudmhYiYa2qGeNUvQoo5FKNCmklZp2qUCZWropAJ1qummon4LaaamkfprqqKeq2iqrmaKy+uqlq9I666S14nrro7nyuuuivQL766HBEjvsoMUie+yfyTK77J7NQvvsndFSO+2c1WJ77ZvZcrvtmt2C++2Z4ZI77pjlonvul+myu+6W7cL77pXx0jvvlPXie++T+fK775L9AvzvkQETPPCQBSN88I8JM7zwjg1D/PCNEVM88YwVY3zxixlzvPGKHYP88YkhkzzyiCWjfPKHKbO88oYtw/zyhTHTPPOENeN884M589xpQAA7"


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}

CAPITAL_Q = "What is the capital of France? Answer with one word."


def text_of(msg):
    return "".join(b.text for b in msg.content if b.type == "text")


def sec_basic(client, model):
    r = client.messages.create(
        model=model,
        max_tokens=60,
        system="You are terse.",
        messages=[{"role": "user", "content": CAPITAL_Q}],
        temperature=0.0,
    )
    assert r.type == "message" and r.role == "assistant", r
    assert "Paris" in text_of(r), r.content
    assert r.stop_reason == "end_turn", r.stop_reason
    assert r.usage.input_tokens > 0 and r.usage.output_tokens > 0, r.usage
    assert r.id.startswith("msg_"), r.id
    ok("messages.create + system + usage")


def sec_stream(client, model):
    with client.messages.stream(
        model=model,
        max_tokens=60,
        messages=[{"role": "user", "content": CAPITAL_Q}],
        temperature=0.0,
    ) as s:
        parts = list(s.text_stream)
        final = s.get_final_message()
    assert "Paris" in "".join(parts), "".join(parts)
    assert "Paris" in text_of(final), final.content
    assert final.stop_reason == "end_turn", final.stop_reason
    assert final.usage.output_tokens > 0, final.usage
    ok("messages.stream accumulator (event protocol)")


def sec_thinking(client, model, dialect):
    kwargs = {}
    if dialect == "qwen":
        # explicit opt-in; gpt-oss reasons unconditionally
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2048}
    # Anthropic's own rule, enforced here since then: max_tokens is the
    # whole budget and thinking spends out of it, so it has to be the larger
    # of the two. A probe asking for 800 with a 2048 budget is asking for a
    # documented 400.
    r = client.messages.create(
        model=model,
        max_tokens=3072,
        messages=[{"role": "user", "content": "What is 17 + 25? Answer with just the number."}],
        temperature=0.0,
        **kwargs,
    )
    thinks = [b for b in r.content if b.type == "thinking"]
    assert thinks and thinks[0].thinking, r.content
    assert "42" in text_of(r), r.content

    # streamed: thinking_delta events accumulate into a ThinkingBlock
    with client.messages.stream(
        model=model,
        max_tokens=3072,
        messages=[{"role": "user", "content": "What is 17 + 25? Answer with just the number."}],
        temperature=0.0,
        **kwargs,
    ) as s:
        for _ in s.text_stream:
            pass
        final = s.get_final_message()
    thinks = [b for b in final.content if b.type == "thinking"]
    assert thinks and thinks[0].thinking, final.content
    assert "42" in text_of(final), final.content
    ok("thinking blocks, non-stream + streamed deltas")


def sec_tools(client, model, dialect):
    if dialect == "qwen":
        kwargs = {"tool_choice": {"type": "any"}}
        prompt = "What's the weather in Paris?"
    else:
        # forcing on gpt-oss is a documented honest 400
        try:
            client.messages.create(
                model=model,
                max_tokens=50,
                messages=[{"role": "user", "content": "hi"}],
                tools=[WEATHER_TOOL],
                tool_choice={"type": "any"},
            )
            raise AssertionError("expected 400 for forced tool_choice on gpt-oss")
        except anthropic.BadRequestError as e:
            assert e.status_code == 400, e
        ok("forced tool_choice honest 400 (harmony)")
        kwargs = {}
        prompt = "What's the weather in Paris? Use the get_weather tool."

    r = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
        tools=[WEATHER_TOOL],
        temperature=0.0,
        **kwargs,
    )
    assert r.stop_reason == "tool_use", (r.stop_reason, r.content)
    uses = [b for b in r.content if b.type == "tool_use"]
    assert uses and uses[0].name == "get_weather", r.content
    use = uses[0]
    assert use.id.startswith("toolu_"), use.id
    assert isinstance(use.input, dict) and "city" in use.input, use.input

    # agent loop: tool_result back in a user turn
    r2 = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": use.id, "name": use.name, "input": use.input}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": use.id,
                        "content": '{"temp_c": 21, "sky": "clear"}',
                    }
                ],
            },
        ],
        tools=[WEATHER_TOOL],
        temperature=0.0,
    )
    answer = text_of(r2).lower()
    assert r2.stop_reason == "end_turn", (r2.stop_reason, r2.content)
    assert "21" in answer or "clear" in answer or "sunny" in answer, answer
    print(f"  agent loop answer: {answer!r}", flush=True)
    ok("tool_use round trip (agent loop)")

    # streamed tool_use: input_json_delta accumulates into input
    if dialect == "qwen":
        with client.messages.stream(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "tool", "name": "get_weather"},
            temperature=0.0,
        ) as s:
            for _ in s.text_stream:
                pass
            final = s.get_final_message()
        uses = [b for b in final.content if b.type == "tool_use"]
        assert uses and uses[0].name == "get_weather", final.content
        assert isinstance(uses[0].input, dict) and "city" in uses[0].input, uses[0]
        assert final.stop_reason == "tool_use", final.stop_reason
        ok("named tool_choice + streamed input_json_delta")

        # a long string argument streams while it is written: many
        # input_json_delta pieces, and the accumulator's input is exact JSON
        write_tool = {
            "name": "write_file",
            "description": "Write a text file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        }
        pieces, joined = 0, ""
        with client.messages.stream(
            model=model,
            max_tokens=1200,
            messages=[{"role": "user", "content": "Write a Python module of about 30 lines with three "
                       "small functions and docstrings to fib.py."}],
            tools=[write_tool],
            tool_choice={"type": "tool", "name": "write_file"},
            temperature=0.0,
        ) as s:
            for event in s:
                if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                    pieces += 1
                    joined += event.delta.partial_json
            final = s.get_final_message()
        uses = [b for b in final.content if b.type == "tool_use"]
        assert uses and uses[0].name == "write_file", final.content
        assert json.loads(joined) == uses[0].input, (joined[:200], uses[0].input)
        assert len(uses[0].input.get("content", "")) > 200, uses[0].input
        assert pieces > 20, f"tool input arrived in {pieces} pieces"
        print(f"  write_file input in {pieces} input_json_delta pieces", flush=True)
        ok("tool input streams while it is written")


def sec_stops(client, model):
    r = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": "Count from 1 to 9 as plain digits separated by commas, "
                "like 1, 2, 3, ... Start now.",
            }
        ],
        stop_sequences=["5"],
        temperature=0.0,
    )
    assert r.stop_reason == "stop_sequence", (r.stop_reason, text_of(r))
    assert r.stop_sequence == "5", r.stop_sequence
    assert "5" not in text_of(r) and "6" not in text_of(r), text_of(r)

    r = client.messages.create(
        model=model,
        max_tokens=5,
        messages=[{"role": "user", "content": "Tell me a long story."}],
        temperature=0.0,
    )
    assert r.stop_reason == "max_tokens", r.stop_reason
    ok("stop_sequences + max_tokens stop_reasons")


def sec_count_tokens(client, model):
    r = client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": CAPITAL_Q}]
    )
    assert r.input_tokens > 0, r
    ok("messages.count_tokens")


def sec_errors(client, model, dialect):
    # unknown named tool must be a 400, anthropic error shape (on harmony
    # the forced-tool honest 400 fires first - either way a typed 400)
    try:
        client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": "hi"}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "tool", "name": "no_such_tool"},
        )
        raise AssertionError("expected 400 for unknown named tool")
    except anthropic.BadRequestError as e:
        assert e.status_code == 400, e
        # on harmony the forced-tool refusal fires first and names its reason
        # (the message stopped saying "qwen" when granite gained a grammar)
        expect = "no_such_tool" if dialect == "qwen" else "Harmony"
        assert expect in str(e), e
    ok("validation errors -> typed BadRequestError")


def sec_vision(client, model):
    r = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What two colors is this image? Answer briefly."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/bmp",
                            "data": base64.b64encode(red_blue_bmp()).decode(),
                        },
                    },
                ],
            }
        ],
        temperature=0.0,
    )
    answer = text_of(r).lower()
    assert "red" in answer and "blue" in answer, answer
    print(f"  vision answer: {answer!r}", flush=True)
    ok("vision via base64 image source block")

    # webp + gif complete the media_type enum - the SDK types exactly these
    # four formats, so refusing either would break a spec-legal request
    for media_type, data in (("image/webp", RED_BLUE_WEBP), ("image/gif", RED_BLUE_GIF)):
        r = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What two colors is this image? Answer briefly."},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data},
                        },
                    ],
                }
            ],
            temperature=0.0,
        )
        answer = text_of(r).lower()
        assert "red" in answer and "blue" in answer, (media_type, answer)
        ok(f"vision decodes {media_type}")

    # a picture an agent's file-read tool returned rides its tool_result (what
    # Claude Code's Read does with a .png) - the model has to see it there too
    read_tool = {"name": "read_file", "description": "Read a file",
                 "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                                  "required": ["path"]}}
    r = client.messages.create(
        model=model,
        max_tokens=500,
        tools=[read_tool],
        messages=[
            {"role": "user", "content": "Read picture.bmp and tell me its two colors, briefly."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_pic", "name": "read_file",
                                               "input": {"path": "picture.bmp"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_pic", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/bmp",
                                             "data": base64.b64encode(red_blue_bmp()).decode()}}]}]},
        ],
        temperature=0.0,
    )
    answer = text_of(r).lower()
    assert "red" in answer and "blue" in answer, answer
    print(f"  tool_result vision answer: {answer!r}", flush=True)
    ok("vision inside a tool_result (an agent's file read)")


def sec_context_management(client, model):
    """Server-side context management (betas context-management-2025-06-27 +
    compact-2026-01-12): applied_edits for the clear strategies,
    count_tokens original_input_tokens, and the compact_20260112 flow - the
    leading compaction block, usage.iterations, the streamed compaction_delta
    through the SDK accumulator, pause_after_compaction's stop_reason, and
    the compaction-block resend round-trip. Every shape rides the SDK's own
    Beta* types, so a drifted field name fails here, not in the field."""
    filler = "Numbers: " + " ".join(str(i) for i in range(900))
    clear_cfg = {"edits": [{
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 500},
        "keep": {"type": "tool_uses", "value": 1}}]}
    tool_msgs = [
        {"role": "user", "content": "What's the weather in Paris and Nice?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_cm1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_cm1", "content": filler}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_cm2", "name": "get_weather", "input": {"city": "Nice"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_cm2", "content": "22C, sunny"}]},
        {"role": "user", "content": "Answer in one short sentence: how is Nice's weather?"},
    ]
    betas = ["context-management-2025-06-27"]
    r = client.beta.messages.create(
        model=model, max_tokens=100, temperature=0.0, betas=betas,
        messages=tool_msgs, tools=[WEATHER_TOOL], context_management=clear_cfg)
    edits = r.context_management.applied_edits
    assert edits and edits[0].type == "clear_tool_uses_20250919", r.context_management
    assert edits[0].cleared_tool_uses >= 1 and edits[0].cleared_input_tokens > 0, edits
    ok("clear_tool_uses applied_edits (SDK-typed)")

    c = client.beta.messages.count_tokens(
        model=model, messages=tool_msgs, tools=[WEATHER_TOOL], betas=betas,
        context_management=clear_cfg)
    assert c.context_management.original_input_tokens > c.input_tokens, c
    ok("count_tokens original_input_tokens (SDK-typed)")

    # compact_20260112: the fact planted before the filler has to survive the
    # summarization into iteration 2's answer - that is the feature's bar
    convo = [
        {"role": "user", "content": "Remember this: the vault code is 4711. Acknowledge briefly."},
        {"role": "assistant", "content": "Noted: the vault code is 4711."},
        {"role": "user", "content": filler + "\nAcknowledge these numbers briefly."},
        {"role": "assistant", "content": "Noted the number list."},
        {"role": "user", "content": "What is the vault code? Answer with just the code."},
    ]
    compact_cfg = {"edits": [{
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": 800}}]}
    betas_c = ["compact-2026-01-12"]
    r = client.beta.messages.create(
        model=model, max_tokens=300, temperature=0.0, betas=betas_c,
        messages=convo, context_management=compact_cfg)
    assert r.content[0].type == "compaction" and r.content[0].content, r.content
    its = r.usage.iterations
    assert its and its[0].type == "compaction" and its[-1].type == "message", its
    assert its[0].output_tokens > 0 and its[-1].input_tokens < its[0].input_tokens, its
    assert "4711" in text_of(r), (text_of(r), r.content[0].content)
    ok("compact_20260112 block + usage.iterations, fact survives the summary")

    # round-trip: the client appends the response verbatim (compaction block
    # included) and continues; the server must accept and not re-compact the
    # already-small rewritten prompt
    convo2 = convo + [
        {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]},
        {"role": "user", "content": "Repeat the code once more, just the code."},
    ]
    r2 = client.beta.messages.create(
        model=model, max_tokens=300, temperature=0.0, betas=betas_c,
        messages=convo2, context_management=compact_cfg)
    assert "4711" in text_of(r2), text_of(r2)
    assert not [b for b in r2.content if b.type == "compaction"], r2.content
    ok("compaction block resend round-trip")

    with client.beta.messages.stream(
        model=model, max_tokens=300, temperature=0.0, betas=betas_c,
        messages=convo, context_management=compact_cfg) as s:
        for _ in s.text_stream:
            pass
        final = s.get_final_message()
    comps = [b for b in final.content if b.type == "compaction"]
    assert comps and comps[0].content, final.content
    assert final.usage.iterations and final.usage.iterations[0].type == "compaction", final.usage
    ok("streamed compaction_delta through the SDK accumulator")

    r3 = client.beta.messages.create(
        model=model, max_tokens=300, temperature=0.0, betas=betas_c,
        messages=convo, context_management={"edits": [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 800},
            "pause_after_compaction": True}]})
    assert r3.stop_reason == "compaction", r3.stop_reason
    assert len(r3.content) == 1 and r3.content[0].type == "compaction", r3.content
    assert r3.usage.iterations and len(r3.usage.iterations) == 1, r3.usage
    ok("pause_after_compaction -> stop_reason compaction")


def sec_mcp(client, model, mcp_url):
    """The beta MCP connector end to end: the SDK sends mcp_servers=[{type:url}],
    paddock connects over Streamable HTTP, runs the agent loop, and the SDK's own
    validation accepts the mcp_tool_use / mcp_tool_result blocks (BetaMessage) and
    their streaming events (BetaRawMessageStreamEvent). Needs a live HTTP MCP
    server (e.g. `npx @modelcontextprotocol/server-everything streamableHttp`)."""
    servers = [{"type": "url", "url": mcp_url, "name": "everything"}]
    betas = ["mcp-client-2025-04-04"]
    # temperature 0 + a forcing prompt so the tool call is deterministic
    prompt = ("Call the `echo` tool with message set to {t!r}. You MUST use the "
              "tool - do not answer without calling it. Then report what it returned.")
    msg = client.beta.messages.create(
        model=model, max_tokens=512, temperature=0.0, mcp_servers=servers, betas=betas,
        messages=[{"role": "user", "content": prompt.format(t="mcp-anth")}])
    tu = [b for b in msg.content if b.type == "mcp_tool_use"]
    tr = [b for b in msg.content if b.type == "mcp_tool_result"]
    assert tu and tu[0].server_name == "everything", tu
    assert tr and tr[0].is_error is False, tr
    text = "".join(getattr(c, "text", "") for c in tr[0].content) if isinstance(tr[0].content, list) else str(tr[0].content)
    assert "mcp-anth" in text, text

    seen = set()
    with client.beta.messages.stream(
        model=model, max_tokens=512, temperature=0.0, mcp_servers=servers, betas=betas,
        messages=[{"role": "user", "content": prompt.format(t="mcp-anth-s")}]) as st:
        for e in st:
            if e.type == "content_block_start":
                seen.add(e.content_block.type)
        st.get_final_message()
    assert "mcp_tool_use" in seen and "mcp_tool_result" in seen, seen
    ok("mcp connector (mcp_tool_use + mcp_tool_result, non-streaming + streaming)")

    # context management inside the agent loop: compaction
    # runs once, before the first round, so the block leads the content and
    # usage.iterations bills the pass - in the same response that ran the
    # tool. Until the loops learned this, the combination was a loud 400.
    filler = "Numbers: " + " ".join(str(i) for i in range(900))
    cm = client.beta.messages.create(
        model=model, max_tokens=512, temperature=0.0, mcp_servers=servers,
        betas=betas + ["compact-2026-01-12"],
        messages=[
            {"role": "user", "content": "Remember this: the vault code is 4711."},
            {"role": "assistant", "content": "Noted: 4711."},
            {"role": "user", "content": filler + "\nAcknowledge these numbers briefly."},
            {"role": "assistant", "content": "Noted the number list."},
            {"role": "user", "content": prompt.format(t="mcp-anth-cm")},
        ],
        context_management={"edits": [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 800}}]})
    assert cm.content[0].type == "compaction" and cm.content[0].content, cm.content
    assert cm.usage.iterations and cm.usage.iterations[0].type == "compaction", cm.usage
    assert [b for b in cm.content if b.type == "mcp_tool_use"], cm.content

    # ...and streamed, the shape the Studio actually sends: the agent stream
    # hand-builds the compaction block's start/delta/stop at index 0, so the
    # SDK accumulator is the arbiter that it lands as a real content block.
    with client.beta.messages.stream(
        model=model, max_tokens=512, temperature=0.0, mcp_servers=servers,
        betas=betas + ["compact-2026-01-12"],
        messages=[
            {"role": "user", "content": "Remember this: the vault code is 4711."},
            {"role": "assistant", "content": "Noted: 4711."},
            {"role": "user", "content": filler + "\nAcknowledge these numbers briefly."},
            {"role": "assistant", "content": "Noted the number list."},
            {"role": "user", "content": prompt.format(t="mcp-anth-cms")},
        ],
        context_management={"edits": [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 800}}]}) as st:
        for _ in st:
            pass
        fin = st.get_final_message()
    assert fin.content[0].type == "compaction" and fin.content[0].content, fin.content
    assert fin.usage.iterations and fin.usage.iterations[0].type == "compaction", fin.usage
    assert [b for b in fin.content if b.type == "mcp_tool_use"], fin.content
    ok("agent loop + compact_20260112 (block leads, iterations billed, tool still runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dialect", choices=["qwen", "harmony"], default="qwen")
    ap.add_argument("--vision", action="store_true", help="run only the vision section")
    ap.add_argument("--mcp-url", help="live HTTP MCP server URL; runs the MCP connector section")
    args = ap.parse_args()

    client = Anthropic(
        base_url=args.base_url, api_key="paddock-local", max_retries=0, timeout=600.0
    )

    if args.vision:
        sec_vision(client, args.model)
    else:
        sec_basic(client, args.model)
        sec_stream(client, args.model)
        sec_thinking(client, args.model, args.dialect)
        sec_tools(client, args.model, args.dialect)
        sec_stops(client, args.model)
        sec_count_tokens(client, args.model)
        sec_errors(client, args.model, args.dialect)
        sec_context_management(client, args.model)
        if args.mcp_url:
            sec_mcp(client, args.model, args.mcp_url)

    print(f"CONFORMANCE PASS ({SECTIONS} sections, anthropic {anthropic.__version__})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
