#!/usr/bin/env python
"""OpenAI-SDK conformance gate for the paddock server.

Driven by tests/sdk_conformance.rs, which boots the real server on a
loopback port and runs this script with the official `openai` package.
The SDK's strict pydantic response models are the wire-format arbiter
(the way llama.cpp is the numeric arbiter): every response and
every typed streaming event must parse cleanly, on top of the semantic
assertions below. Each section prints `ok - <name>`; any failure exits
nonzero and fails the Rust test.

Usage:
    python openai_conformance.py --base-url http://127.0.0.1:PORT/v1 \
        --model qwen35-9b --dialect qwen [--vision]
"""

import argparse
import base64
import json
import struct
import sys

import openai
from openai import OpenAI
from pydantic import BaseModel

SECTIONS = 0


def ok(name):
    global SECTIONS
    SECTIONS += 1
    print(f"ok - {name}", flush=True)


def red_blue_bmp():
    """256x160 24-bit BMP, solid red left half / blue right half (BGR rows).
    Byte-identical to red_blue_bmp() in qwen_vision_http.rs."""
    w, h = 256, 160
    img = w * h * 3
    head = b"BM" + struct.pack(
        "<IIIIiiHHIIIIII", 54 + img, 0, 54, 40, w, h, 1, 24, 0, img, 2835, 2835, 0, 0
    )
    row = b"\x00\x00\xff" * (w // 2) + b"\xff\x00\x00" * (w // 2)
    return head + row * h


# The same red/blue 256x160 picture re-encoded once (Pillow 12.2) and embedded
# as base64 so the gate gains no imaging dependency (every extra import
# is a silent-skip hazard). Both formats are in the vendor contracts - OpenAI:
# png/jpeg/webp/non-animated gif; anthropic's media_type enum is exactly
# jpeg/png/gif/webp. The webp is LOSSY VP8 deliberately: that is the wild form
# of the format, so the probe exercises the real decoder, not just VP8L.
RED_BLUE_WEBP = "UklGRsQAAABXRUJQVlA4ILgAAABQDgCdASoAAaAAPm02mUkkIyKhIkgAgA2JZ27hdVERgAfiBsWPAPxA/VUAJXF6ZWtE5e07xWZU5NYzvFZl4txipneKzLxeVlp4rMvF6Z0xg6eZeL0zum/QZOXtO8VhfywCjGd4rL5LaMZ3isy8O18neKzLxelAwdPMqAAA/v7ge/rRMc1d334PlgkC/+PfNg0ld/jzr+sydFcH/k2Xlrx2agwgAzAycI2CvQHyK9AfIr0B8ivQEAAA"
RED_BLUE_GIF = "R0lGODdhAAGgAIEAAP8AAAAA/wAAAAAAACwAAAAAAAGgAEAI/wABCBxIsKDBgwgTKlzIsKHDhwgDSJxIsaLFixgzatzIsaPHjxghihxJsqRJkSBTqlzJsmXKkzBjypwZ0aXNmzhzXqTJs6dPhzqDCh3q8afRo0aJKl2qFKnTpzCZSp1qE6rVq0Cpat3aEavXrwW5ih1bEaxZr2TTij3LFqrat1Tbyj0Kt+7SuXh72t0rNK9fmXwD4/xL2KTgwy0LK0aJuDHIxZAbOp7cNbLlmpQz77zMmaDmz2U7iwZNOoDo0aU/n+6cWvXqy601v4Ydm/Jsy7Vt34ace/Ju3r0b/14cXPjwwsURH0eeXPByws2dP/cbPfB06tX3Xs+bXfv2ud3tfv8HHx7ueLnlzZ9nm/7tevbt074/G1/+fLD1yd7Hn3/tfrT9cfUfgAFqNSBWBW514FUJGrigWw1O9SCEETI14VMVSnWhUxlauCFdHTb1YVIhEjUiiSX2daJPKQ61IostBvWiXjHqNCNPNdp440w55rQjjz3e9CNgQVY1ZFRFunQkkkmytORJTSb2ZElROjklSVWudCWWWb60JWNdfvQlmGFWNmZWZZp5JkNpFrWmZG1y9CaccWo0J5t12nmnQnnquSdmfVr0Z0KBhjToQYVudmhYiYa2qGeNUvQoo5FKNCmklZp2qUCZWropAJ1qummon4LaaamkfprqqKeq2iqrmaKy+uqlq9I666S14nrro7nyuuuivQL766HBEjvsoMUie+yfyTK77J7NQvvsndFSO+2c1WJ77ZvZcrvtmt2C++2Z4ZI77pjlonvul+myu+6W7cL77pXx0jvvlPXie++T+fK775L9AvzvkQETPPCQBSN88I8JM7zwjg1D/PCNEVM88YwVY3zxixlzvPGKHYP88YkhkzzyiCWjfPKHKbO88oYtw/zyhTHTPPOENeN884M589xpQAA7"


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

CAPITAL_Q = "What is the capital of France? Answer with one word."


# ---------------------------------------------------------------- sections --


def sec_models(client, model):
    ids = [m.id for m in client.models.list()]
    assert model in ids, f"{model} not in {ids}"
    ok("models.list")


def sec_chat_basic(client, model):
    # A DEFAULT request - no reasoning knob - so this measures what an
    # unmodified client gets. The budget is deliberately generous: every
    # family we serve reasons by default, and 60 tokens is not a fair budget
    # for a thinking model (it answers correctly and reports finish_reason
    # "length" mid-thought, which is honest but tests nothing).
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": CAPITAL_Q}],
        max_tokens=600,
        temperature=0.0,
    )
    c = r.choices[0]
    assert "Paris" in (c.message.content or ""), c.message.content
    assert c.finish_reason == "stop", c.finish_reason
    u = r.usage
    assert u.prompt_tokens > 0 and u.completion_tokens > 0, u
    assert u.total_tokens == u.prompt_tokens + u.completion_tokens, u

    # Required-by-spec response fields that a strict (non-Python) client would
    # KeyError on. `ChatCompletionResponseMessage` lists role/content/refusal
    # required and the choice lists logprobs required, all nullable - the SDK
    # tolerates their absence, the schema does not.
    raw = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{"role": "user", "content": CAPITAL_Q}],
        max_tokens=600,
        temperature=0.0,
    ).http_response.json()
    choice = raw["choices"][0]
    assert "logprobs" in choice, choice
    for f in ("role", "content", "refusal"):
        assert f in choice["message"], (f, choice["message"])
    ok("chat non-stream + usage + required nullable fields")


def sec_reasoning_effort(client, model, dialect):
    """`reasoning_effort` is the spec's own control over reasoning, and it has
    to reach every model that has a reasoning mode - not just gpt-oss.

    On a graded model (gpt-oss) the level passes through. On a toggle model
    (qwen3.5/3.6, gemma4, laguna) `none` means off and any other level means
    on: pretending `xhigh` differs from `low` there would be a plausible lie,
    but silently ignoring `none` - which is what happened before, since only
    /v1/messages could turn thinking off - is a conformance hole.
    """
    def content_and_reasoning(**extra):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": CAPITAL_Q}],
            max_tokens=600,
            temperature=0.0,
            **extra,
        )
        msg = r.choices[0].message
        return (msg.content or ""), (msg.model_extra or {}).get("reasoning_content")

    if dialect == "qwen":
        content, reasoning = content_and_reasoning(reasoning_effort="none")
        assert "Paris" in content, content
        assert not reasoning, f"reasoning_effort=none still reasoned: {reasoning!r}"
        content, reasoning = content_and_reasoning(reasoning_effort="high")
        assert reasoning, "reasoning_effort=high produced no reasoning_content"
        assert "Paris" in content, content
    else:
        # graded: every level is accepted; the model reasons throughout
        for level in ("low", "high"):
            content, _ = content_and_reasoning(reasoning_effort=level)
            assert "Paris" in content, (level, content)

    # the Responses spelling of the same control
    r = client.responses.create(
        model=model,
        input=CAPITAL_Q,
        max_output_tokens=600,
        temperature=0.0,
        reasoning={"effort": "none"},
    )
    assert "Paris" in r.output_text, r.output_text
    ok("reasoning_effort (chat + responses) reaches this model's reasoning mode")


def sec_chat_stream(client, model):
    parts, finishes, usage, saw_role = [], [], None, False
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": CAPITAL_Q}],
        # same budget reasoning as sec_chat_basic: a thinking model needs room
        # to reach its answer, or this asserts on a truncated thought
        max_tokens=600,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if not chunk.choices:
            usage = chunk.usage
            continue
        ch = chunk.choices[0]
        if ch.delta is not None and ch.delta.role:
            saw_role = True
        if ch.delta is not None and ch.delta.content:
            parts.append(ch.delta.content)
        if ch.finish_reason:
            finishes.append(ch.finish_reason)
    assert "Paris" in "".join(parts), "".join(parts)
    assert finishes == ["stop"], finishes
    assert saw_role, "no role in first delta"
    assert usage is not None and usage.completion_tokens > 0, usage
    ok("chat stream + include_usage terminal chunk")


def sec_reasoning(client, model, dialect):
    # Ask for reasoning through the SPEC's own knob rather than the
    # vLLM-style `chat_template_kwargs` extension - the extension still works
    # and still wins when both are set, but a conformance gate should be
    # driving the standard control. ("thinking is opt-in on qwen3.5" was true
    # of an older template; every family we serve now reasons by default.)
    extra = {"reasoning_effort": "high"}
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 17 + 25? Answer with just the number."}],
        max_tokens=600,
        temperature=0.0,
        **extra,
    )
    msg = r.choices[0].message
    reasoning = (msg.model_extra or {}).get("reasoning_content")
    assert reasoning, "no reasoning_content on the message (SDK model_extra)"
    assert "42" in (msg.content or ""), msg.content

    # streamed reasoning deltas arrive as a non-standard delta field
    r_parts, c_parts = [], []
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 17 + 25? Answer with just the number."}],
        max_tokens=600,
        temperature=0.0,
        stream=True,
        **extra,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if d is None:
            continue
        rc = (d.model_extra or {}).get("reasoning_content")
        if rc:
            r_parts.append(rc)
        if d.content:
            c_parts.append(d.content)
    assert r_parts, "no streamed reasoning_content deltas"
    assert "42" in "".join(c_parts), "".join(c_parts)
    ok("reasoning_content, non-stream + streamed deltas")


def first_tool_call(client, model, dialect, stream=False):
    """One assistant turn that must produce a get_weather call."""
    if dialect == "qwen":
        kwargs = {"tool_choice": "required"}
        prompt = "What's the weather in Paris?"
    else:
        kwargs = {}
        prompt = "What's the weather in Paris? Use the get_weather tool."
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=[WEATHER_TOOL],
        max_tokens=500,
        temperature=0.0,
        stream=stream,
        **kwargs,
    ), prompt


def sec_tools_loop(client, model, dialect):
    if dialect == "harmony":
        # forcing a tool on gpt-oss is a documented honest 400
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                tools=[WEATHER_TOOL],
                tool_choice="required",
                max_tokens=50,
            )
            raise AssertionError("expected 400 for forced tool_choice on gpt-oss")
        except openai.BadRequestError as e:
            assert e.status_code == 400, e
        ok("forced tool_choice honest 400 (harmony)")

    first, prompt = first_tool_call(client, model, dialect)
    ch = first.choices[0]
    assert ch.finish_reason == "tool_calls", ch.finish_reason
    call = ch.message.tool_calls[0]
    assert call.type == "function" and call.function.name == "get_weather", call
    args = json.loads(call.function.arguments)
    assert "city" in args, args

    # agent loop: feed the tool result back, expect a grounded answer
    msgs = [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": '{"temp_c": 21, "sky": "clear"}',
        },
    ]
    second = client.chat.completions.create(
        model=model, messages=msgs, tools=[WEATHER_TOOL], max_tokens=500, temperature=0.0
    )
    sm = second.choices[0].message
    assert sm.content and not sm.tool_calls, sm
    low = sm.content.lower()
    assert "21" in low or "clear" in low or "sunny" in low, sm.content
    print(f"  agent loop answer: {sm.content!r}", flush=True)
    ok("tool round trip (agent loop)")


def sec_tools_stream(client, model, dialect):
    stream, _ = first_tool_call(client, model, dialect, stream=True)
    by_index = {}
    finishes = []
    for chunk in stream:
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        if ch.finish_reason:
            finishes.append(ch.finish_reason)
        d = ch.delta
        if d is None or not d.tool_calls:
            continue
        for tc in d.tool_calls:
            slot = by_index.setdefault(tc.index, {"id": None, "name": None, "args": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["args"] += tc.function.arguments
                slot["pieces"] = slot.get("pieces", 0) + 1
    assert finishes == ["tool_calls"], finishes
    assert by_index, "no streamed tool-call deltas"
    call = by_index[0]
    assert call["id"] and call["name"] == "get_weather", call
    assert "city" in json.loads(call["args"]), call
    # the qwen dialect streams arguments while they are written: the string
    # value arrives apart from the key that opens it
    if dialect == "qwen":
        assert call.get("pieces", 0) >= 2, call
    ok("streamed tool-call deltas (SDK accumulation)")


class CityInfo(BaseModel):
    city: str
    country: str
    population_millions: float


def sec_structured(client, model):
    # .parse() sends the SDK's auto-generated strict json_schema
    # (additionalProperties:false, all-required, title annotations).
    #
    # `reasoning_effort="none"` is what a client actually wants here and what
    # makes the budget predictable: the grammar only takes over at the content
    # boundary, so a thinking model reasons first - unbounded - and then emits
    # the JSON. With reasoning on, the token cap is a race against the model's
    # preamble rather than a test of the grammar.
    r = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": "Give facts about Paris, France."}],
        response_format=CityInfo,
        max_tokens=700,
        temperature=0.0,
        reasoning_effort="none",
    )
    p = r.choices[0].message.parsed
    assert p is not None and p.city and p.country, r.choices[0].message
    assert isinstance(p.population_millions, float), p
    print(f"  parsed: {p!r}", flush=True)

    # json_object free-form mode
    r = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with a JSON object with string keys city and country "
                "for the capital of France.",
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0.0,
        reasoning_effort="none",
    )
    c = r.choices[0]
    assert c.finish_reason == "stop", c.finish_reason
    obj = json.loads(c.message.content)
    assert isinstance(obj, dict) and obj, obj
    ok(".parse() structured output + json_object")


def sec_n_and_logprobs(client, model):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Name a color. Answer with one word."}],
        n=2,
        temperature=0.9,
        seed=7,
        max_tokens=8,
    )
    assert sorted(c.index for c in r.choices) == [0, 1], r.choices
    ok("n=2 indexed choices")

    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": CAPITAL_Q}],
        logprobs=True,
        top_logprobs=3,
        max_tokens=5,
        temperature=0.0,
    )
    lp = r.choices[0].logprobs
    assert lp is not None and lp.content, "missing logprobs.content"
    e0 = lp.content[0]
    assert e0.token and e0.logprob <= 0.0, e0
    assert len(e0.top_logprobs) == 3, e0.top_logprobs
    ok("logprobs typed parse (token/logprob/top_logprobs)")


def sec_completions(client, model):
    r = client.completions.create(
        model=model, prompt="The capital of France is", max_tokens=8, temperature=0.0
    )
    assert "Paris" in r.choices[0].text, r.choices[0].text
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens

    parts, finishes = [], []
    stream = client.completions.create(
        model=model,
        prompt="The capital of France is",
        max_tokens=8,
        temperature=0.0,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices:
            parts.append(chunk.choices[0].text or "")
            if chunk.choices[0].finish_reason:
                finishes.append(chunk.choices[0].finish_reason)
    assert "Paris" in "".join(parts), "".join(parts)
    assert finishes, "no finish_reason in completion stream"
    ok("legacy completions, non-stream + stream")

    # legacy logprobs: the parallel-array shape, offsets over prompt+completion
    prompt = "The capital of France is"
    r = client.completions.create(
        model=model, prompt=prompt, max_tokens=3, temperature=0.0, logprobs=2
    )
    lp = r.choices[0].logprobs
    assert lp is not None, "missing legacy logprobs"
    n = len(lp.tokens)
    assert n == len(lp.token_logprobs) == len(lp.top_logprobs) == len(lp.text_offset), lp
    assert n > 0 and lp.token_logprobs[0] <= 0.0, lp
    assert lp.text_offset[0] == len(prompt), lp.text_offset
    assert len(lp.top_logprobs[0]) == 2, lp.top_logprobs[0]
    ok("legacy completions logprobs (tokens/token_logprobs/top_logprobs/text_offset)")


def sec_responses(client, model):
    r = client.responses.create(
        model=model,
        input=CAPITAL_Q,
        instructions="You are terse.",
        max_output_tokens=200,
        temperature=0.0,
    )
    assert "Paris" in r.output_text, r.output_text
    assert r.usage is not None and r.usage.input_tokens > 0, r.usage
    # request echoes on the response object
    assert r.tool_choice == "auto" and r.tools == [], (r.tool_choice, r.tools)
    assert r.instructions == "You are terse.", r.instructions
    assert r.temperature == 0.0 and r.status == "completed", r
    ok("responses non-stream typed parse + output_text + echoes")

    types, text_parts, seqs, final = [], [], [], None
    stream = client.responses.create(
        model=model,
        input=CAPITAL_Q,
        max_output_tokens=200,
        temperature=0.0,
        stream=True,
    )
    for ev in stream:
        types.append(ev.type)
        seqs.append(ev.sequence_number)
        if ev.type == "response.output_text.delta":
            text_parts.append(ev.delta)
        if ev.type == "response.completed":
            final = ev.response
    assert types[0] == "response.created", types[:3]
    assert types[1] == "response.in_progress", types[:3]
    assert types[-1] == "response.completed", types[-3:]
    assert "response.output_item.added" in types, types
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "sequence_number not monotonic"
    assert "Paris" in "".join(text_parts), "".join(text_parts)
    assert final is not None and final.status == "completed", final
    ok("responses typed event stream")


def sec_responses_structured(client, model):
    # .parse() on the Responses API sends text.format json_schema (flat).
    # `reasoning.effort: none` for the same reason as the chat section: the
    # grammar starts at the content boundary, so reasoning runs first and
    # unbounded, and the cap would be racing the preamble.
    r = client.responses.parse(
        model=model,
        input="Give facts about Paris, France.",
        text_format=CityInfo,
        max_output_tokens=700,
        temperature=0.0,
        reasoning={"effort": "none"},
    )
    p = r.output_parsed
    assert p is not None and p.city and p.country, r
    assert isinstance(p.population_millions, float), p

    # json_object mode via text.format
    r = client.responses.create(
        model=model,
        input="Reply with a JSON object with string keys city and country "
        "for the capital of France.",
        text={"format": {"type": "json_object"}},
        max_output_tokens=500,
        temperature=0.0,
    )
    assert r.status == "completed", r.status
    obj = json.loads(r.output_text)
    assert isinstance(obj, dict) and obj, obj
    ok("responses .parse() text.format json_schema + json_object")


def sec_responses_forced_tool(client, model):
    flat_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    # named forcing from a prompt that would never call it on its own
    r = client.responses.create(
        model=model,
        input="Say hello.",
        tools=[flat_tool],
        tool_choice={"type": "function", "name": "get_weather"},
        max_output_tokens=300,
        temperature=0.0,
    )
    calls = [it for it in r.output if it.type == "function_call"]
    assert calls and calls[0].name == "get_weather", [it.type for it in r.output]
    assert "city" in json.loads(calls[0].arguments), calls[0].arguments
    ok("responses named tool_choice forcing")


def sec_responses_incomplete(client, model):
    r = client.responses.create(
        model=model,
        input="Tell me a very long story about horses.",
        max_output_tokens=16,
        temperature=0.0,
    )
    assert r.status == "incomplete", r.status
    assert r.incomplete_details and r.incomplete_details.reason == "max_output_tokens", r

    types = []
    stream = client.responses.create(
        model=model,
        input="Tell me a very long story about horses.",
        max_output_tokens=16,
        temperature=0.0,
        stream=True,
    )
    final = None
    for ev in stream:
        types.append(ev.type)
        if ev.type == "response.incomplete":
            final = ev.response
    assert types[-1] == "response.incomplete", types[-3:]
    assert final is not None and final.status == "incomplete", final
    ok("responses truthful incomplete status at max_output_tokens")


def sec_responses_tools(client, model):
    flat_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    prompt = "What's the weather in Paris? Use the get_weather tool."
    r = client.responses.create(
        model=model,
        input=prompt,
        tools=[flat_tool],
        max_output_tokens=500,
        temperature=0.0,
    )
    calls = [it for it in r.output if it.type == "function_call"]
    assert calls, [it.type for it in r.output]
    call = calls[0]
    assert call.name == "get_weather" and "city" in json.loads(call.arguments), call

    r2 = client.responses.create(
        model=model,
        input=[
            {"role": "user", "content": prompt},
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            },
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": '{"temp_c": 21, "sky": "clear"}',
            },
        ],
        tools=[flat_tool],
        max_output_tokens=500,
        temperature=0.0,
    )
    assert r2.output_text, [it.type for it in r2.output]
    low = r2.output_text.lower()
    assert "21" in low or "clear" in low or "sunny" in low, r2.output_text
    ok("responses tool round trip")

    # streamed: the arguments arrive as deltas while the call is written, the
    # done event and the SDK-accumulated item agree with them
    deltas, done_args = [], None
    with client.responses.stream(
        model=model, input=prompt, tools=[flat_tool], max_output_tokens=500, temperature=0.0
    ) as s:
        for ev in s:
            if ev.type == "response.function_call_arguments.delta":
                deltas.append(ev.delta)
            elif ev.type == "response.function_call_arguments.done":
                done_args = ev.arguments
        final = s.get_final_response()
    fcs = [it for it in final.output if it.type == "function_call"]
    assert fcs and done_args is not None, [it.type for it in final.output]
    assert "".join(deltas) == done_args, (deltas, done_args)
    assert json.loads(done_args) == json.loads(fcs[0].arguments), (done_args, fcs[0].arguments)
    ok("responses streamed function_call arguments")


def sec_errors(client, model, dialect):
    msgs = [{"role": "user", "content": "hi"}]
    for bad in [
        {"n": 99},
        {"top_logprobs": 3},  # without logprobs
        {"stream_options": {"include_usage": True}},  # without stream
    ]:
        try:
            client.chat.completions.create(model=model, messages=msgs, max_tokens=5, **bad)
            raise AssertionError(f"expected 400 for {bad}")
        except openai.BadRequestError as e:
            assert e.status_code == 400 and e.message, e

    # responses honesty: no persistence, no faked reasoning knobs
    try:
        client.responses.create(model=model, input="hi", store=True, max_output_tokens=20)
        raise AssertionError("expected 400 for store: true")
    except openai.BadRequestError as e:
        assert e.status_code == 400, e
    # `reasoning.effort` reaches every family that has a reasoning mode - it
    # used to 400 on anything but gpt-oss, which left an OpenAI client no
    # spec-standard way to control thinking at all (sec_reasoning_effort gates
    # the mapping). What must still be refused is an unsupported KEY inside
    # `reasoning`, since silently ignoring one would be the same hole.
    try:
        client.responses.create(
            model=model, input="hi", max_output_tokens=20,
            reasoning={"effort": "low", "generate_summary": "concise"},
        )
        raise AssertionError("expected 400 for an unsupported reasoning key")
    except openai.BadRequestError as e:
        assert e.status_code == 400 and "generate_summary" in e.message, e
    if dialect != "qwen":
        # gpt-oss: effort is a graded template knob
        r = client.responses.create(
            model=model,
            input=CAPITAL_Q,
            reasoning={"effort": "low"},
            max_output_tokens=400,
            temperature=0.0,
        )
        assert "Paris" in r.output_text, r.output_text
    ok("validation errors -> typed BadRequestError")


def sec_prefix_cache(client, model, dialect):
    """Multi-turn: turn 2 re-sends the history, whose prefix the server's
    prefix cache serves - surfaced as usage.prompt_tokens_details.cached_tokens
    (the SDK types this field).

    The system prompt is padded deliberately. Reuse is PAGE-granular and needs a
    resumable checkpoint under the shared prefix, so a short history has
    nothing to hand back - measured on qwen3.5-9B: a 41-token turn 2
    reuses 0, 65 reuses 32, 113 reuses 80, 282 reuses 256. The old fixture
    landed at 62 tokens, right on that edge, so it was asserting a coin flip
    rather than the cache.
    """
    system = (
        "You are a concise reference assistant for a European travel agency. "
        "Answer factual questions with a single word or a very short phrase. "
        "Never speculate, never add commentary, and never restate the question. "
        "If a question is ambiguous, answer for the most common interpretation. "
        "Prefer the modern name of a place over any historical one."
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    r1 = client.chat.completions.create(model=model, messages=msgs, max_tokens=200, temperature=0.0)
    msgs.append({"role": "assistant", "content": r1.choices[0].message.content})
    msgs.append({"role": "user", "content": "And of Italy?"})
    r2 = client.chat.completions.create(model=model, messages=msgs, max_tokens=300, temperature=0.0)
    assert "Rome" in (r2.choices[0].message.content or ""), r2.choices[0].message.content
    details = r2.usage.prompt_tokens_details
    # harmony included: the old tripwire (gpt-oss pinned to cached == 0) is
    # gone - the KV pool auto-sizer lit up its radix cache too.
    assert details is not None and (details.cached_tokens or 0) >= 32, r2.usage
    print(f"  turn-2 cached_tokens: {details.cached_tokens}/{r2.usage.prompt_tokens}", flush=True)
    ok("prefix cache visible in typed usage (multi-turn)")


def sec_vision(client, model):
    uri = "data:image/bmp;base64," + base64.b64encode(red_blue_bmp()).decode()
    r = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What two colors is this image? Answer briefly."},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        ],
        max_tokens=500,
        temperature=0.0,
    )
    c = (r.choices[0].message.content or "").lower()
    assert "red" in c and "blue" in c, c
    print(f"  vision answer: {c!r}", flush=True)
    ok("vision via SDK image_url data URI")

    # the other two contract formats: same picture, same question, different
    # codec - a decode failure here is a 400, not a wrong answer
    for mime, b64 in (("image/webp", RED_BLUE_WEBP), ("image/gif", RED_BLUE_GIF)):
        r = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What two colors is this image? Answer briefly."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=500,
            temperature=0.0,
        )
        c = (r.choices[0].message.content or "").lower()
        assert "red" in c and "blue" in c, (mime, c)
        ok(f"vision decodes {mime}")

    # same image through the Responses API's input_image item
    r = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "What two colors is this image? Answer briefly.",
                    },
                    {"type": "input_image", "image_url": uri},
                ],
            }
        ],
        max_output_tokens=500,
        temperature=0.0,
    )
    c = r.output_text.lower()
    assert "red" in c and "blue" in c, c
    print(f"  responses vision answer: {c!r}", flush=True)
    ok("vision via responses input_image item")

    # the same picture returned by a function call: input_image inside a
    # function_call_output (the Responses API allows images there)
    tool = {"type": "function", "name": "read_file", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]}}
    r = client.responses.create(
        model=model,
        tools=[tool],
        input=[
            {"role": "user", "content": "Read picture.bmp and tell me its two colors, briefly."},
            {"type": "function_call", "call_id": "call_pic", "name": "read_file",
             "arguments": '{"path": "picture.bmp"}'},
            {"type": "function_call_output", "call_id": "call_pic",
             "output": [{"type": "input_image", "image_url": uri}]},
        ],
        max_output_tokens=500,
        temperature=0.0,
    )
    c = r.output_text.lower()
    assert "red" in c and "blue" in c, c
    print(f"  function_call_output vision answer: {c!r}", flush=True)
    ok("vision inside a function_call_output")


def sec_mcp(client, model, mcp_url):
    """The MCP `mcp` tool end to end: the SDK sends an inline-URL mcp tool,
    paddock connects over Streamable HTTP, runs the agent loop, and the SDK's
    own validation accepts the mcp_list_tools + mcp_call output items
    (non-streaming) and the response.mcp_list_tools.* / response.mcp_call.*
    stream events. Needs a live HTTP MCP server (e.g.
    `npx @modelcontextprotocol/server-everything streamableHttp`), the same
    fixture the Anthropic gate's connector section uses. (It used to register
    a stdio server through `/api/mcp` - a runner endpoint that no longer
    exists since the manager/runner split, so the section could not run at
    all; the inline URL is what real callers send anyway.)"""
    tool = {"type": "mcp", "server_label": "everything", "server_url": mcp_url,
            "require_approval": "never"}
    # temperature 0 + a forcing prompt/instruction so the tool call is deterministic
    instr = "You MUST call the echo tool - do not answer without calling it. Then report its result."
    resp = client.responses.create(
        model=model, tools=[tool], temperature=0.0,
        input="Call the echo tool with the exact text: mcp-oai. Then report the result.",
        instructions=instr,
    )
    assert any(o.type == "mcp_list_tools" for o in resp.output), "no mcp_list_tools"
    calls = [o for o in resp.output if o.type == "mcp_call"]
    assert calls and calls[0].status == "completed" and calls[0].error is None, calls
    assert calls[0].output and "mcp-oai" in calls[0].output, calls[0].output

    seen = set()
    with client.responses.stream(
        model=model, tools=[tool], temperature=0.0,
        input="Call the echo tool with the exact text: mcp-oai-s. Then report the result.",
        instructions=instr,
    ) as st:
        for e in st:
            seen.add(e.type)
        st.get_final_response()
    want = {"response.mcp_list_tools.completed", "response.mcp_call.in_progress",
            "response.mcp_call_arguments.done", "response.mcp_call.completed"}
    assert want <= seen, f"missing mcp stream events: {sorted(want - seen)}"
    ok("mcp tool (list_tools + mcp_call, non-streaming + streaming)")

    # context management inside the agent loop: the
    # compaction runs before the first round, so one response carries the
    # compaction item (leading the output) and the tool call. This exact
    # combination was a loud 400 until the loops learned to orchestrate it.
    # The filler is deliberately smaller than the one in sec_context_management:
    # this server's tools block is ~1.4k tokens on its own, and the
    # summarization pass renders it too - at 900 rows the SPAN itself overflows
    # an 8k window, which is the fail-open case (no item), not this one.
    filler = "Data log: " + " ".join(f"row{i}={i * 7}" for i in range(400))
    convo = [
        {"role": "user", "content": "Remember this: the vault code is 4711."},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Noted: 4711."}]},
        {"role": "user", "content": filler + "\nAcknowledge these rows briefly."},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Noted the data log."}]},
        {"role": "user",
         "content": "Call the echo tool with the exact text: mcp-cm. Then report the result."},
    ]
    rc = client.responses.create(
        model=model, tools=[tool], temperature=0.0, input=convo, instructions=instr,
        context_management=[{"type": "compaction", "compact_threshold": 800}],
        truncation="auto",
    )
    assert rc.output[0].type == "compaction", rc.output[0]
    assert rc.output[0].encrypted_content.strip(), "summary lives in encrypted_content"
    calls = [o for o in rc.output if o.type == "mcp_call"]
    assert calls and calls[0].status == "completed", rc.output
    assert calls[0].output and "mcp-cm" in calls[0].output, calls[0].output

    # ...and streamed, which is the shape the Studio actually sends: the item
    # rides its own added/done pair at output_index 0 and the mcp items shift
    # past it (the agent stream hand-builds these events).
    added = []
    with client.responses.stream(
        model=model, tools=[tool], temperature=0.0, input=convo, instructions=instr,
        context_management=[{"type": "compaction", "compact_threshold": 800}],
        truncation="auto",
    ) as st:
        for e in st:
            if e.type == "response.output_item.added":
                added.append(e)
        final = st.get_final_response()
    assert added[0].item.type == "compaction" and added[0].output_index == 0, added[0]
    assert all(e.output_index >= 1 for e in added[1:]), added
    assert final.output[0].type == "compaction", final.output[0]
    assert [o for o in final.output if o.type == "mcp_call"], final.output
    ok("agent loop + context management (compaction item leads, tool still runs)")


def sec_context_management(client, model):
    """Responses context management (openai 2.53.0 pins):
    `context_management: [{"type": "compaction", ...}]` compacts past the
    threshold and leads output with a typed compaction item (the plaintext
    summary rides `encrypted_content` - the SDK's one required field);
    the item round-trips as an input param on resend; `compaction_trigger`
    compacts-now; `client.responses.compact()` returns the typed
    CompactedResponse; truncation "auto" is served, not 400d."""
    filler = "Data log: " + " ".join(f"row{i}={i * 7}" for i in range(900))
    convo = [
        {"role": "user", "content": "Remember this: the vault code is 4711. Acknowledge briefly."},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Noted: the vault code is 4711."}]},
        {"role": "user", "content": filler + "\nAcknowledge these rows briefly."},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Noted the data log."}]},
        {"role": "user", "content": "What is the vault code? Answer with just the code."},
    ]

    # in-create compaction: typed item first, the fact survives the summary
    r = client.responses.create(
        model=model, input=convo, max_output_tokens=300, temperature=0.0,
        context_management=[{"type": "compaction", "compact_threshold": 800}],
    )
    assert r.output[0].type == "compaction", r.output[0]
    assert r.output[0].encrypted_content.strip(), "summary lives in encrypted_content"
    assert "4711" in r.output_text, r.output_text

    # resend: output items round-trip as input params; no re-compaction, and
    # the rewritten prompt is a radix hit (typed cached_tokens)
    resend = convo + [
        o.model_dump(exclude_none=True) for o in r.output if o.type != "reasoning"
    ] + [{"role": "user", "content": "Repeat the code once more, just the code."}]
    r2 = client.responses.create(
        model=model, input=resend, max_output_tokens=300, temperature=0.0,
        context_management=[{"type": "compaction", "compact_threshold": 800}],
    )
    assert not [o for o in r2.output if o.type == "compaction"], "no re-compaction"
    assert "4711" in r2.output_text, r2.output_text
    assert r2.usage.input_tokens_details.cached_tokens > 0, "post-compaction radix hit"

    # streamed: the compaction item rides typed output_item.added/done at
    # index 0 and every later item shifts past it
    added = []
    with client.responses.stream(
        model=model, input=convo, max_output_tokens=300, temperature=0.0,
        context_management=[{"type": "compaction", "compact_threshold": 800}],
    ) as stream:
        for event in stream:
            if event.type == "response.output_item.added":
                added.append(event)
        final = stream.get_final_response()
    assert added[0].item.type == "compaction" and added[0].output_index == 0, added[0]
    assert all(e.output_index >= 1 for e in added[1:]), added
    assert final.output[0].type == "compaction", final.output[0]

    # compaction_trigger (must be final): compact-now, item-only output
    rt = client.responses.create(
        model=model, max_output_tokens=300, temperature=0.0,
        input=convo + [{"type": "compaction_trigger"}],
    )
    assert rt.status == "completed" and len(rt.output) == 1, rt.output
    assert rt.output[0].type == "compaction", rt.output[0]

    # the standalone executor: typed CompactedResponse, user messages + item
    rc = client.responses.compact(model=model, input=convo)
    assert rc.object == "response.compaction", rc.object
    assert rc.output[-1].type == "compaction" and rc.output[-1].encrypted_content.strip()
    assert rc.usage.input_tokens > 0 and rc.usage.output_tokens > 0
    followup = [o.model_dump(exclude_none=True) for o in rc.output] + [
        {"role": "user", "content": "What is the vault code? Just the code."}]
    r3 = client.responses.create(
        model=model, input=followup, max_output_tokens=300, temperature=0.0)
    assert "4711" in r3.output_text, r3.output_text

    # truncation "auto" round-trips (the fit-fallback itself is exercised by
    # the server-side probe; here the knob must parse, serve and echo)
    r4 = client.responses.create(
        model=model, input="Say OK.", max_output_tokens=20,
        temperature=0.0, truncation="auto")
    assert r4.truncation == "auto", r4.truncation

    # compaction + truncation "auto" together on a span past the window: the
    # summarization pass cannot run, so compaction fails open (no item) and
    # the truncation backstop drops leading turns and serves - this exact
    # combination 400'd before the phase-4 fail-open fix.
    heavy = " ".join(f"row-{i:04d} holds value {i * 7 % 991}." for i in range(900))
    r5 = client.responses.create(
        model=model, max_output_tokens=300, temperature=0.0,
        input=[
            {"role": "user", "content": "Filler: " + heavy},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Noted."}]},
            {"role": "user", "content": "Reply with the single word OK."},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 800}],
        truncation="auto",
    )
    assert not [o for o in r5.output if o.type == "compaction"], "no item on an unsummarizable span"
    assert r5.output_text.strip(), "the backstop must serve, not 400"
    ok("responses context management (compaction + trigger + compact + truncation)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dialect", choices=["qwen", "harmony"], default="qwen")
    ap.add_argument("--vision", action="store_true", help="run only the vision section")
    ap.add_argument("--mcp-url", help="live HTTP MCP server URL; runs the MCP tool section")
    args = ap.parse_args()

    # no retries: a retried generation would double GPU work and mask flakes
    client = OpenAI(base_url=args.base_url, api_key="paddock-local", max_retries=0, timeout=600.0)

    if args.vision:
        sec_models(client, args.model)
        sec_vision(client, args.model)
    else:
        sec_models(client, args.model)
        sec_chat_basic(client, args.model)
        sec_chat_stream(client, args.model)
        sec_reasoning(client, args.model, args.dialect)
        sec_reasoning_effort(client, args.model, args.dialect)
        sec_tools_loop(client, args.model, args.dialect)
        sec_tools_stream(client, args.model, args.dialect)
        sec_structured(client, args.model)
        sec_n_and_logprobs(client, args.model)
        sec_completions(client, args.model)
        sec_responses(client, args.model)
        sec_responses_structured(client, args.model)
        if args.dialect == "qwen":
            sec_responses_forced_tool(client, args.model)
        sec_responses_incomplete(client, args.model)
        sec_responses_tools(client, args.model)
        sec_context_management(client, args.model)
        sec_prefix_cache(client, args.model, args.dialect)
        sec_errors(client, args.model, args.dialect)
        if args.mcp_url:
            sec_mcp(client, args.model, args.mcp_url)

    print(f"CONFORMANCE PASS ({SECTIONS} sections, openai {openai.__version__})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
