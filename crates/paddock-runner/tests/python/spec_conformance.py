#!/usr/bin/env python
"""Spec-driven conformance gate.

Three mechanical checks against the PINNED SDK spec encodings - the installed
pip versions are the spec revision, the way a pinned llama.cpp build pins
numerics:

0. VERSION CURRENCY: the installed SDKs must match `_pinned` in
   tests/spec/coverage.json. Without this the suite happily validates against
   whatever revision the machine happens to have, which is exactly what it
   used to do. A conformance pass against a stale spec is not a pass.

1. REQUEST-SURFACE AUDIT: every parameter in the SDKs' request TypedDicts
   must have a disposition in tests/spec/coverage.json (implemented / partial
   / advisory / rejected). A new SDK param without a disposition FAILS; a
   stale matrix entry FAILS. Every rejected/partial param is probed and must
   400 with the spec error envelope; every advisory param must 200.

2. STRICT RESPONSE VALIDATION: representative responses and every SSE event
   are validated with full pydantic model_validate against the SDK models
   (required fields, types, literal enums, discriminated stream unions) -
   the strictness the lenient SDK clients don't apply.

Usage:
    python spec_conformance.py --base-url http://127.0.0.1:PORT \
        --model qwen35-9b --dialect qwen

    # /v1/audio/transcriptions: its own server, because a whisper checkpoint
    # has no chat surface to run the rest of the gate against
    python spec_conformance.py --base-url http://127.0.0.1:PORT \
        --model kb-whisper-large --audio-only

The audio MATRIX audit is static and runs either way - a new SDK param must
fail the gate on a box with no ASR checkpoint too.
"""

import argparse
import base64
import difflib
import io
import json
import math
import mimetypes
import os
import pathlib
import re
import sys
import tempfile
import threading
import time

import httpx
from pydantic import TypeAdapter

from openai.types.chat import completion_create_params as chat_params
from openai.types import completion_create_params as comp_params
from openai.types.responses import response_create_params as resp_params
from openai.types.audio import transcription_create_params as audio_params
from openai.types.audio import translation_create_params as translation_params
from openai.types import image_generate_params as img_gen_params
from openai.types import image_edit_params as img_edit_params
from anthropic.types import message_create_params as anth_params

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import Completion, ImageGenStreamEvent, ImagesResponse
from openai.types.audio import Transcription, TranscriptionVerbose
from openai.types.responses import Response as OaiResponse
from openai.types.responses import ResponseStreamEvent
from anthropic.types import Message as AnthMessage
from anthropic.types import RawMessageStreamEvent

MATRIX_PATH = pathlib.Path(__file__).parent.parent / "spec" / "coverage.json"
AUDIO_FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "audio" / "ls-long-concat.wav"

SECTIONS = 0


def ok(name):
    global SECTIONS
    SECTIONS += 1
    print(f"ok - {name}", flush=True)


# probe values for rejected/partial params (a value that exercises the
# refused behavior); unknown-field rejection ignores the value anyway
PROBE_400 = {
    "chat": {
        "audio": {"voice": "alloy", "format": "wav"},
        "metadata": {"a": "b"},
        "modalities": ["text", "audio"],
        "moderation": "auto",
        "prediction": {"type": "content", "content": "x"},
        "prompt_cache_options": {},
        "prompt_cache_retention": "24h",
        "store": True,
        "web_search_options": {},
    },
    "completions": {
        "suffix": "!!",
    },
    "responses": {
        "background": True,
        "context_management": {},
        "conversation": "conv_1",
        "include": ["reasoning.encrypted_content"],
        "metadata": {"a": "b"},
        "moderation": "auto",
        "previous_response_id": "resp_1",
        "prompt": {"id": "p1"},
        "prompt_cache_options": {},
        "prompt_cache_retention": "24h",
        "store": True,
        "stream_options": {"include_obfuscation": True},
        "top_logprobs": 3,
    },
    "anthropic_messages": {
        "container": "c1",
    },
    # every one refuses BEFORE a render, which is what keeps the probe cheap
    "images_generations": {
        "quality": "hd",
        "response_format": "url",
        "style": "vivid",
    },
}

PROBE_200 = {
    "chat": {
        "user": "u1",
        "safety_identifier": "s1",
        "prompt_cache_key": "k1",
        "service_tier": "auto",
        "verbosity": "low",
    },
    "completions": {"user": "u1", "logprobs": 3},
    "responses": {
        "user": "u1",
        "safety_identifier": "s1",
        "prompt_cache_key": "k1",
        "service_tier": "auto",
    },
    "anthropic_messages": {
        "metadata": {"user_id": "u1"},
        "service_tier": "auto",
        "cache_control": {"type": "ephemeral"},
        "inference_geo": "eu",
        "user_profile_id": "p1",
    },
    # each of these is a real one-step render at the family's small base size
    "images_generations": {
        "model": "gpt-image-2.5",
        "moderation": "low",
        "user": "u1",
    },
}


# /v1/audio/transcriptions is multipart, and a form field's refusal can depend
# on the REST of the form - a granularity alongside the default json format is
# refused for the FORMAT, not for the granularity. So an audio probe is the
# whole extra form rather than one value.
AUDIO_PROBE_400 = {
    # `auto` and server_vad(+threshold) are SERVED now; the ms dials are not
    # (they move a live utterance boundary, and this endpoint gates whole
    # encoder windows), so the probe uses one of those.
    "chunking_strategy": [
        ("chunking_strategy", '{"type": "server_vad", "prefix_padding_ms": 300}')
    ],
    # served on verbose_json (per-word confidence); the probe is the format
    # OpenAI puts logprobs on, which is the half we do not serve
    "include": [("response_format", "json"), ("include[]", "logprobs")],
    "keywords": [("keywords", "Concord")],
    "known_speaker_names": [("known_speaker_names[]", "alice")],
    "known_speaker_references": [("known_speaker_references[]", "data:audio/wav;base64,AA==")],
    # `languages` is SERVED on the whisper lane as a soft prior over language
    # detection, so the probe cannot be a valid code - that would
    # 200 there and 400 on the generative lanes, which is not a probe. An
    # unknown code refuses everywhere: on whisper because it is not in the
    # checkpoint's own map, on the generative lanes because they have no
    # detection to bias at all.
    "languages": [("languages[]", "xx")],
}
# The refusal has to NAME what was refused, which for a param whose values
# split (partial) means naming the value, not the field.
AUDIO_PROBE_NAMES = {"include": "logprobs", "languages": "xx"}
AUDIO_PROBE_200 = {"model": [("model", "whisper-1")]}
# /v1/audio/translations: `prompt` is refused outright (the instruction is the
# card's translation request) and srt/vtt are the half of response_format no
# lane serves
TRANSLATION_PROBE_400 = {
    "prompt": [("prompt", "Use British spelling.")],
    "response_format": [("response_format", "srt")],
}
TRANSLATION_PROBE_200 = {"model": [("model", "whisper-1")]}
# `timestamp_granularities` needs neither probe now that both of its values are
# served (landed `word`): it is `implemented`, and what an implemented
# param is worth is checked by the functional half posting it for real -
# `sec_audio_strict` asks for segment AND word together and validates both.


# One request body per nested union member, keyed exactly like `_nested` in
# the matrix. `_unknown` is a made-up member: it must be refused by NAME, and
# it is the probe that catches the silent holes this gate was written for.
def nested_probes(model, img_uri):
    chat = lambda parts: {"model": model, "max_tokens": 8, "temperature": 0.0,
                          "messages": [{"role": "user", "content": parts}]}
    chat_msgs = lambda msgs: {"model": model, "max_tokens": 8, "temperature": 0.0,
                              "messages": msgs}
    fn = {"type": "function", "function": {"name": "f", "description": "d",
                                           "parameters": {"type": "object", "properties": {}}}}
    resp = lambda parts: {"model": model, "max_output_tokens": 16, "temperature": 0.0,
                          "input": [{"type": "message", "role": "user", "content": parts}]}
    atool = {"name": "f", "input_schema": {"type": "object", "properties": {}}}
    anth = lambda blocks, **kw: {"model": model, "max_tokens": 8, "temperature": 0.0,
                                 "messages": [{"role": "user", "content": blocks}], **kw}
    pdf = "data:application/pdf;base64,JVBERi0xLjQK"  # header only: parse error, not a type error
    return {
        "chat": ("/v1/chat/completions", {
            "content_part": {
                "text": chat([{"type": "text", "text": "hi"}]),
                "image_url": chat([{"type": "image_url", "image_url": {"url": img_uri}}]),
                "file": chat([{"type": "file", "file": {"filename": "a.pdf", "file_data": pdf}}]),
                "input_audio": chat([{"type": "input_audio",
                                      "input_audio": {"data": "AAAA", "format": "wav"}}]),
                "_unknown": chat([{"type": "definitely_not_a_part", "x": 1}]),
            },
            "role": {
                r: chat_msgs([{"role": r, "content": "x"}, {"role": "user", "content": "hi"}])
                for r in ("system", "developer", "user")
            } | {
                "assistant": chat_msgs([{"role": "user", "content": "hi"},
                                        {"role": "assistant", "content": "ok"},
                                        {"role": "user", "content": "hi"}]),
                "tool": chat_msgs([{"role": "user", "content": "hi"},
                                   {"role": "assistant", "content": None,
                                    "tool_calls": [{"id": "c1", "type": "function",
                                                    "function": {"name": "f", "arguments": "{}"}}]},
                                   {"role": "tool", "tool_call_id": "c1", "content": "42"}]),
                "function": chat_msgs([{"role": "function", "name": "f", "content": "{}"},
                                       {"role": "user", "content": "hi"}]),
                "_unknown": chat_msgs([{"role": "definitely_not_a_role", "content": "x"},
                                       {"role": "user", "content": "hi"}]),
            },
            "tool": {
                "function": chat_msgs([{"role": "user", "content": "hi"}]) | {"tools": [fn]},
                "custom": chat_msgs([{"role": "user", "content": "hi"}])
                | {"tools": [{"type": "custom", "custom": {"name": "c"}}]},
                "_unknown": chat_msgs([{"role": "user", "content": "hi"}])
                | {"tools": [{"type": "definitely_not_a_tool", "name": "z"}]},
            },
            "tool_choice": {
                tc_name: chat_msgs([{"role": "user", "content": "hi"}])
                | {"tools": [fn], "tool_choice": tc, "max_tokens": 64}
                for tc_name, tc in [
                    ("none", "none"), ("auto", "auto"), ("required", "required"),
                    ("function", {"type": "function", "function": {"name": "f"}}),
                    ("allowed_tools", {"type": "allowed_tools",
                                       "allowed_tools": {"mode": "auto",
                                                         "tools": [{"type": "function", "name": "f"}]}}),
                    ("custom", {"type": "custom", "name": "c"}),
                ]
            },
        }),
        "responses": ("/v1/responses", {
            "content_part": {
                "input_text": resp([{"type": "input_text", "text": "hi"}]),
                "input_image": resp([{"type": "input_image", "image_url": img_uri}]),
                "input_file": resp([{"type": "input_file", "filename": "a.pdf",
                                     "file_data": pdf}]),
                "_unknown": resp([{"type": "definitely_not_a_part", "x": 1}]),
            },
            "input_item": {
                "message": resp([{"type": "input_text", "text": "hi"}]),
                "function_call": {"model": model, "max_output_tokens": 16, "temperature": 0.0,
                                  "input": [{"type": "message", "role": "user",
                                             "content": [{"type": "input_text", "text": "hi"}]},
                                            {"type": "function_call", "call_id": "c1",
                                             "name": "f", "arguments": "{}"},
                                            {"type": "function_call_output", "call_id": "c1",
                                             "output": "42"}]},
                "function_call_output": {"model": model, "max_output_tokens": 16,
                                         "temperature": 0.0,
                                         "input": [{"type": "message", "role": "user",
                                                    "content": [{"type": "input_text",
                                                                 "text": "hi"}]},
                                                   {"type": "function_call_output",
                                                    "call_id": "c1", "output": "42"}]},
                # accepted as an item type; it resolves against a
                # previous_response_id that does not exist here, so a 400 from
                # that check is still proof the type itself is known
                "mcp_approval_response": {"model": model, "max_output_tokens": 16,
                                          "temperature": 0.0,
                                          "input": [{"type": "message", "role": "user",
                                                     "content": [{"type": "input_text",
                                                                  "text": "hi"}]},
                                                    {"type": "mcp_approval_response",
                                                     "approval_request_id": "mcpr_1",
                                                     "approve": True}]},
                "item_reference": {"model": model, "max_output_tokens": 16, "temperature": 0.0,
                                   "input": [{"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": "hi"}]},
                                             {"type": "item_reference", "id": "msg_1"}]},
                "_unknown": {"model": model, "max_output_tokens": 16, "temperature": 0.0,
                             "input": [{"type": "message", "role": "user",
                                        "content": [{"type": "input_text", "text": "hi"}]},
                                       {"type": "definitely_not_an_item", "x": 1}]},
            },
            "tool": {
                name: resp([{"type": "input_text", "text": "hi"}]) | {"tools": [tool]}
                for name, tool in [
                    ("function", {"type": "function", "name": "f", "description": "d",
                                  "parameters": {"type": "object", "properties": {}}}),
                    # both are recognized types handled upstream of the
                    # template; on a runner with no provider/server configured
                    # they 400 for that reason, which still proves the type is
                    # known (the audit only rejects an "unsupported" refusal)
                    ("web_search", {"type": "web_search"}),
                    ("mcp", {"type": "mcp", "server_label": "nope",
                             "server_url": "https://example.invalid/mcp"}),
                    ("file_search", {"type": "file_search", "vector_store_ids": ["vs_1"]}),
                    ("code_interpreter", {"type": "code_interpreter",
                                          "container": {"type": "auto"}}),
                    ("image_generation", {"type": "image_generation"}),
                    ("computer_use_preview", {"type": "computer_use_preview",
                                              "display_width": 1, "display_height": 1,
                                              "environment": "browser"}),
                    ("local_shell", {"type": "local_shell"}),
                    ("custom", {"type": "custom", "name": "c"}),
                    ("_unknown", {"type": "definitely_not_a_tool", "name": "z"}),
                ]
            },
        }),
        "anthropic_messages": ("/v1/messages", {
            "content_block": {
                "text": anth([{"type": "text", "text": "hi"}]),
                "image_base64": anth([{"type": "image", "source": {
                    "type": "base64", "media_type": "image/bmp",
                    "data": img_uri.split(",", 1)[1]}}]),
                "image_url": anth([{"type": "image", "source": {
                    "type": "url", "url": "https://example.com/a.png"}}]),
                "document": anth([{"type": "document", "source": {
                    "type": "base64", "media_type": "application/pdf",
                    "data": pdf.split(",", 1)[1]}}]),
                "tool_use": {"model": model, "max_tokens": 8, "temperature": 0.0,
                             "tools": [atool],
                             "messages": [{"role": "user", "content": "hi"},
                                          {"role": "assistant", "content": [
                                              {"type": "tool_use", "id": "t1", "name": "f",
                                               "input": {}}]},
                                          {"role": "user", "content": [
                                              {"type": "tool_result", "tool_use_id": "t1",
                                               "content": "42"}]}]},
                # same payload as tool_use: the two blocks only exist as a pair
                "tool_result": {"model": model, "max_tokens": 8, "temperature": 0.0,
                                "tools": [atool],
                                "messages": [{"role": "user", "content": "hi"},
                                             {"role": "assistant", "content": [
                                                 {"type": "tool_use", "id": "t1", "name": "f",
                                                  "input": {}}]},
                                             {"role": "user", "content": [
                                                 {"type": "tool_result", "tool_use_id": "t1",
                                                  "content": "42"}]}]},
                "thinking": {"model": model, "max_tokens": 8, "temperature": 0.0,
                             "messages": [{"role": "user", "content": "hi"}],
                             "thinking": {"type": "enabled", "budget_tokens": 64}},
                "search_result": anth([{"type": "search_result", "source": "s", "title": "t",
                                        "content": [{"type": "text", "text": "x"}]}]),
                "_unknown": anth([{"type": "definitely_not_a_block", "x": 1}]),
            },
            "tool_choice": {
                name: anth("hi", tools=[atool], tool_choice=tc, max_tokens=64)
                for name, tc in [("auto", {"type": "auto"}), ("none", {"type": "none"}),
                                 ("any", {"type": "any"}),
                                 ("tool", {"type": "tool", "name": "f"})]
            },
            "tool": {
                "input_schema": anth("hi", tools=[atool]),
                "web_search": anth("hi", tools=[{"type": "web_search_20250305",
                                                 "name": "web_search"}]),
                "_unknown": anth("hi", tools=[{"type": "definitely_not_a_tool", "name": "z"}]),
            },
        }),
    }


def sec_nested_shapes(client, model, matrix):
    """Every discriminated-union member the SDKs define has a disposition, and
    the ones we do not serve are refused BY NAME.

    `tool_result` shares the `tool_use` probe (they only make sense together),
    and a couple of `implemented` members are probed with deliberately
    malformed payloads - a PDF header with no body - so a 400 that names the
    PARSE failure still proves the type itself was recognized. That is the
    distinction the whole section exists to make: "I know this type and your
    data is bad" versus "I have never heard of this type".
    """
    img = "data:image/bmp;base64,Qk1aAAAAAAAAADYAAAAoAAAAAQAAAAEAAAABABgAAAAAAAQAAAASCwAAEgsAAAAAAAAAAAAAAAD/AA=="
    probes = nested_probes(model, img)
    total = 0
    for family, (path, kinds) in probes.items():
        listed = matrix["_nested"].get(family, {})
        for kind, members in kinds.items():
            want = listed.get(kind)
            assert want, f"{family}.{kind}: no dispositions in the matrix"
            missing = set(members) - set(want)
            stale = set(want) - set(members)
            assert not missing, f"{family}.{kind}: probed but undeclared: {sorted(missing)}"
            assert not stale, f"{family}.{kind}: declared but unprobed: {sorted(stale)}"
            for member, body in members.items():
                disposition = want[member]
                r = client.post(path, json=body)
                if disposition == "rejected":
                    assert r.status_code == 400, (
                        f"{family}.{kind}.{member}: expected an honest 400, got "
                        f"{r.status_code} - a member we do not serve must never be "
                        f"silently accepted: {r.text[:160]}"
                    )
                    body_json = r.json()
                    msg = (body_json.get("error") or {}).get("message", "")
                    if member == "_unknown":
                        assert "definitely_not_a" in msg, (
                            f"{family}.{kind}: the refusal must NAME the type: {msg!r}"
                        )
                else:
                    assert r.status_code in (200, 400), (family, kind, member, r.status_code)
                    if r.status_code == 400:
                        msg = (r.json().get("error") or {}).get("message", "")
                        assert "unsupported" not in msg.lower(), (
                            f"{family}.{kind}.{member} is marked implemented but was "
                            f"refused as unsupported: {msg!r}"
                        )
                total += 1
        print(f"  {family}: {sum(len(v) for v in kinds.values())} union members probed", flush=True)
    ok(f"nested-shape audit ({total} union members)")


def sec_sampling_ranges(client, model):
    """Sampling knobs are range-checked against the VENDOR's documented bounds,
    which are not the same on the two surfaces: OpenAI documents temperature
    0..2, Anthropic 0..1.

    A negative temperature is the one that matters. The sampler divides logits
    by it, inverting the distribution, so the model emits its least likely
    token - and it does so fluently enough that nothing looks wrong. Every
    value here used to be accepted silently.
    """
    chat = {"model": model, "max_tokens": 4, "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}]}
    resp = {"model": model, "max_output_tokens": 8, "input": "hi"}
    comp = {"model": model, "max_tokens": 4, "prompt": "hi"}
    cases = [
        ("/v1/chat/completions", chat, "temperature", [-1.0, 2.5], [0.0, 2.0]),
        ("/v1/chat/completions", chat, "top_p", [-0.1, 1.5], [0.0, 1.0]),
        ("/v1/chat/completions", chat, "min_p", [-0.1, 1.5], [0.0, 1.0]),
        ("/v1/chat/completions", chat, "frequency_penalty", [-2.5, 2.5], [-2.0, 2.0]),
        ("/v1/chat/completions", chat, "presence_penalty", [-2.5, 2.5], [-2.0, 2.0]),
        ("/v1/completions", comp, "temperature", [-1.0, 2.5], [0.0, 2.0]),
        ("/v1/responses", resp, "temperature", [-1.0, 2.5], [0.0, 2.0]),
        ("/v1/responses", resp, "top_p", [-0.1, 1.5], [0.0, 1.0]),
        # Anthropic's ceiling is 1.0, so 1.5 must be refused here and accepted
        # on the OpenAI surfaces above - the difference is the point
        ("/v1/messages", chat, "temperature", [-1.0, 1.5], [0.0, 1.0]),
        ("/v1/messages", chat, "top_p", [-0.1, 1.5], [0.0, 1.0]),
    ]
    n = 0
    for path, base, param, bad, good in cases:
        for v in bad:
            r = client.post(path, json={**base, param: v})
            assert r.status_code == 400, (
                f"{path} {param}={v}: expected 400, got {r.status_code} "
                f"({r.text[:120]})"
            )
            msg = (r.json().get("error") or {}).get("message", "")
            assert param in msg, f"{path} {param}={v}: error must name the knob: {msg!r}"
            n += 1
        for v in good:
            r = client.post(path, json={**base, param: v})
            assert r.status_code == 200, (
                f"{path} {param}={v}: a boundary value must be accepted, got "
                f"{r.status_code} ({r.text[:120]})"
            )
            n += 1
    ok(f"sampling ranges ({n} boundary probes)")


def sec_version_currency(matrix):
    """The installed SDKs are the spec revision - so they must be the ones the
    matrix was written against.

    This runs first and hard-fails, because every other check in this file is
    only as current as the pip packages behind it. The failure it exists to
    prevent is silent: a box with an older SDK validates fine against an older
    spec and reports a clean pass, which is how this suite spent months
    checking openai 2.44.0 while the matrix claimed otherwise.
    """
    import anthropic
    import openai

    pinned = matrix["_pinned"]
    installed = {"openai": openai.__version__, "anthropic": anthropic.__version__}
    drift = {k: (v, installed[k]) for k, v in pinned.items() if installed[k] != v}
    assert not drift, (
        "SDK drift - the spec revision under test is not the pinned one: "
        + "; ".join(f"{k}: pinned {want}, installed {got}" for k, (want, got) in drift.items())
        + ". Either `pip install -U` to the pin, or bump `_pinned` in "
        "tests/spec/coverage.json and re-run (new spec params then need "
        "dispositions)."
    )
    print(f"  openai {installed['openai']}, anthropic {installed['anthropic']}", flush=True)
    ok("version currency (installed SDKs == the pin)")


def sdk_params(family):
    """The pinned SDK's request-param set for a family (+ `stream`, which the
    TypedDict variants add)."""
    base = {
        "chat": chat_params.CompletionCreateParamsBase,
        "completions": comp_params.CompletionCreateParamsBase,
        "responses": resp_params.ResponseCreateParamsBase,
        "anthropic_messages": anth_params.MessageCreateParamsBase,
        "audio_transcriptions": audio_params.TranscriptionCreateParamsBase,
        # one TypedDict, no streaming variant - `stream` is vLLM's extension
        "audio_translations": translation_params.TranslationCreateParams,
        "images_generations": img_gen_params.ImageGenerateParamsBase,
        "images_edits": img_edit_params.ImageEditParamsBase,
    }[family]
    extra = set() if family == "audio_translations" else {"stream"}
    return set(base.__annotations__.keys()) | extra


class Family:
    def __init__(self, name, path, base_payload, envelope):
        self.name = name
        self.path = path
        self.base = base_payload
        self.envelope = envelope  # "openai" | "anthropic"


def families(model):
    return [
        Family(
            "chat",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_completion_tokens": 4,
                "temperature": 0.0,
            },
            "openai",
        ),
        Family(
            "completions",
            "/v1/completions",
            {"model": model, "prompt": "Hi", "max_tokens": 4, "temperature": 0.0},
            "openai",
        ),
        Family(
            "responses",
            "/v1/responses",
            {"model": model, "input": "Say hi.", "max_output_tokens": 16, "temperature": 0.0},
            "openai",
        ),
        Family(
            "anthropic_messages",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "Say hi."}],
                "temperature": 0.0,
            },
            "anthropic",
        ),
    ]


def images_family(model):
    """`/v1/images/generations` as a Family, so `sec_surface_audit` probes it
    like the JSON lanes. Every accepted probe is a REAL render, so the base is
    one step at a small size with a pinned seed - the audit stays under a
    minute and its bytes are the same on every run."""
    return Family(
        "images_generations",
        "/v1/images/generations",
        {
            "model": model,
            "prompt": "a red circle on a white background",
            "size": "512x512",
            "steps": 1,
            "seed": 7,
        },
        "openai",
    )


def assert_error_shape(fam, body):
    if fam.envelope == "openai":
        assert "error" in body and body["error"].get("message"), body
        assert body["error"].get("type"), body
    else:
        assert body.get("type") == "error", body
        assert body["error"].get("message"), body


def sec_surface_audit(client, fam, matrix):
    """Matrix completeness + probe every non-implemented disposition."""
    spec = sdk_params(fam.name)
    listed = set(matrix[fam.name].keys())
    missing = spec - listed
    stale = listed - spec
    assert not missing, f"{fam.name}: spec params without a disposition: {sorted(missing)}"
    assert not stale, f"{fam.name}: stale matrix entries: {sorted(stale)}"

    counts = {"implemented": 0, "partial": 0, "advisory": 0, "rejected": 0}
    for param, disposition in sorted(matrix[fam.name].items()):
        assert disposition in counts, f"{fam.name}.{param}: bad disposition {disposition}"
        counts[disposition] += 1
        if disposition in ("rejected", "partial"):
            value = PROBE_400[fam.name].get(param)
            assert value is not None, f"{fam.name}.{param}: {disposition} but no 400-probe value"
            r = client.post(fam.path, json={**fam.base, param: value})
            assert r.status_code == 400, (
                f"{fam.name}.{param}={value!r}: expected 400, got {r.status_code}: {r.text[:200]}"
            )
            assert_error_shape(fam, r.json())
        elif disposition == "advisory":
            value = PROBE_200[fam.name].get(param)
            assert value is not None, f"{fam.name}.{param}: advisory but no 200-probe value"
            r = client.post(fam.path, json={**fam.base, param: value})
            assert r.status_code == 200, (
                f"{fam.name}.{param}={value!r}: expected 200, got {r.status_code}: {r.text[:200]}"
            )

    # unknown fields are rejected by NAME with the spec envelope
    r = client.post(fam.path, json={**fam.base, "definitely_not_a_param": 1})
    assert r.status_code == 400, r.text[:200]
    body = r.json()
    assert_error_shape(fam, body)
    msg = body["error"]["message"]
    assert "definitely_not_a_param" in msg, msg

    total = sum(counts.values())
    print(
        f"  {fam.name}: {total} spec params - {counts['implemented']} implemented, "
        f"{counts['partial']} partial, {counts['advisory']} advisory, "
        f"{counts['rejected']} rejected (+{len(matrix['_extensions'][fam.name])} extensions)",
        flush=True,
    )
    ok(f"surface audit: {fam.name}")


def sse_events(client, path, payload):
    """POST a streaming request, yield (event_name, data_dict) per SSE event."""
    with client.stream("POST", path, json=payload) as r:
        assert r.status_code == 200, r.read()[:200]
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                yield event, json.loads(data)
                event = None


def sec_strict_chat(client, model, dialect):
    base = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
        "max_completion_tokens": 60,
        "temperature": 0.0,
        "logit_bias": {"1": -1.0},  # exercises the new knob end to end
    }
    r = client.post("/v1/chat/completions", json=base)
    assert r.status_code == 200, r.text[:200]
    ChatCompletion.model_validate(r.json())

    n = 0
    for _, data in sse_events(
        client,
        "/v1/chat/completions",
        {**base, "stream": True, "stream_options": {"include_usage": True}},
    ):
        ChatCompletionChunk.model_validate(data)
        n += 1
    assert n >= 3, f"only {n} chunks"

    if dialect == "qwen":
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [{"type": "function", "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object",
                                   "properties": {"city": {"type": "string"}},
                                   "required": ["city"]}}}],
                "tool_choice": "required",
                "max_completion_tokens": 300,
                "temperature": 0.0,
            },
        )
        assert r.status_code == 200, r.text[:200]
        parsed = ChatCompletion.model_validate(r.json())
        assert parsed.choices[0].message.tool_calls, "forced call missing"
    ok(f"strict validation: chat ({n} stream chunks)")


def sec_strict_completions(client, model):
    base = {"model": model, "prompt": "The capital of France is", "max_tokens": 8,
            "temperature": 0.0}
    r = client.post("/v1/completions", json=base)
    assert r.status_code == 200, r.text[:200]
    Completion.model_validate(r.json())
    n = 0
    for _, data in sse_events(
        client, "/v1/completions",
        {**base, "stream": True, "stream_options": {"include_usage": True}},
    ):
        # Documented arbiter exception: the real API streams intermediate
        # chunks with finish_reason null, but the SDK's Completion model
        # (written for the non-stream shape) declares the literal
        # non-nullable. Substitute only a null (a wrong string still fails).
        patched = {**data, "choices": [
            {**c, "finish_reason": c.get("finish_reason") or "stop"}
            for c in data.get("choices", [])
        ]}
        Completion.model_validate(patched)
        n += 1
    assert n >= 2, f"only {n} chunks"
    ok(f"strict validation: completions ({n} stream chunks)")


def sec_strict_responses(client, model):
    base = {"model": model, "input": "What is the capital of France? One word.",
            "max_output_tokens": 200, "temperature": 0.0}
    r = client.post("/v1/responses", json=base)
    assert r.status_code == 200, r.text[:200]
    OaiResponse.model_validate(r.json())

    # max_tool_calls - the spec's own name for the agent loop's
    # tool budget, and once a 400. Spending it needs a connector,
    # which this gate has none of; what it can pin is the contract either side
    # of the loop: the param is accepted and the Response object echoes it,
    # including the 0 that means "no built-in tool may run at all".
    for n in (0, 3):
        r = client.post("/v1/responses", json={**base, "max_tool_calls": n})
        assert r.status_code == 200, f"max_tool_calls={n}: {r.status_code} {r.text[:200]}"
        echoed = OaiResponse.model_validate(r.json()).max_tool_calls
        assert echoed == n, f"max_tool_calls must echo {n}, got {echoed!r}"

    adapter = TypeAdapter(ResponseStreamEvent)
    n = 0
    for _, data in sse_events(client, "/v1/responses", {**base, "stream": True}):
        adapter.validate_python(data)
        n += 1
    assert n >= 4, f"only {n} events"
    ok(f"strict validation: responses ({n} stream events)")


def sec_strict_anthropic(client, model):
    base = {"model": model, "max_tokens": 60,
            "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
            "temperature": 0.0}
    r = client.post("/v1/messages", json=base)
    assert r.status_code == 200, r.text[:200]
    AnthMessage.model_validate(r.json())

    adapter = TypeAdapter(RawMessageStreamEvent)
    n = 0
    names = []
    for event, data in sse_events(client, "/v1/messages", {**base, "stream": True}):
        # Documented arbiter exception: `ping` is part of Anthropic's streaming
        # protocol ("we may send ping events at any time") and the SDK's own
        # decoder skips it by name - anthropic/_streaming.py does
        # `if sse.event == "ping": continue` - but it is deliberately absent
        # from the RawMessageStreamEvent union, which models only the six
        # semantic events. So validate everything EXCEPT ping, and still
        # require the envelope to be well-formed.
        if event == "ping":
            assert data == {"type": "ping"}, data
            names.append(event)
            n += 1
            continue
        adapter.validate_python(data)
        assert event == data.get("type"), (event, data.get("type"))
        names.append(event)
        n += 1
    assert names[0] == "message_start" and names[-1] == "message_stop", names
    ok(f"strict validation: anthropic messages ({n} stream events)")


# ---------------------------------------------------------------------------
# /v1/audio/transcriptions
#
# This surface is the odd one twice over: it is multipart/form-data rather than
# JSON, and answering any request needs a loaded ASR model, which no other
# section here requires. So it splits deliberately:
#
#   sec_audio_matrix   static - no server, no model, runs on every invocation.
#                      This is the half that catches a new spec param, and it
#                      would be worthless if it only ran on a box that happened
#                      to have a whisper checkpoint.
#   sec_audio_surface  the probes, and
#   sec_strict_audio   the functional shapes - both need the model, and run
#                      under --audio-only (sdk_conformance.rs::spec_gate_whisper
#                      starts a whisper server for exactly this).
# ---------------------------------------------------------------------------


def sec_translation_matrix(matrix):
    """Completeness of the translation matrix against the pinned SDK, and a
    probe for every disposition that claims a refusal - static, like the
    transcription audit above."""
    spec = sdk_params("audio_translations")
    listed = set(matrix["audio_translations"].keys())
    assert not (spec - listed), f"audio_translations: spec params without a disposition: {sorted(spec - listed)}"
    assert not (listed - spec), f"audio_translations: stale matrix entries: {sorted(listed - spec)}"
    for param, disposition in sorted(matrix["audio_translations"].items()):
        assert disposition in ("implemented", "partial", "advisory", "rejected"), (
            f"audio_translations.{param}: bad disposition {disposition}")
        if disposition in ("rejected", "partial"):
            assert param in TRANSLATION_PROBE_400, f"audio_translations.{param}: no 400-probe"
        elif disposition == "advisory":
            assert param in TRANSLATION_PROBE_200, f"audio_translations.{param}: no 200-probe"
    unused = (set(TRANSLATION_PROBE_400) | set(TRANSLATION_PROBE_200)) - listed
    assert not unused, f"audio_translations: probes for params not in the matrix: {sorted(unused)}"
    print(f"  audio_translations: {len(listed)} spec params", flush=True)
    ok("audio_translations matrix (static)")


def sec_audio_matrix(matrix):
    """Completeness of the audio matrix + a usable probe for every disposition
    that claims a refusal. Static deliberately - see the block comment above."""
    spec = sdk_params("audio_transcriptions")
    listed = set(matrix["audio_transcriptions"].keys())
    missing = spec - listed
    stale = listed - spec
    assert not missing, f"audio_transcriptions: spec params without a disposition: {sorted(missing)}"
    assert not stale, f"audio_transcriptions: stale matrix entries: {sorted(stale)}"

    counts = {"implemented": 0, "partial": 0, "advisory": 0, "rejected": 0}
    for param, disposition in sorted(matrix["audio_transcriptions"].items()):
        assert disposition in counts, f"audio_transcriptions.{param}: bad disposition {disposition}"
        counts[disposition] += 1
        # a disposition whose probe was never written is a disposition nothing
        # checks, so demand the probe here rather than at run time - that way
        # the box with no ASR model still fails for it
        if disposition in ("rejected", "partial"):
            assert param in AUDIO_PROBE_400, f"audio_transcriptions.{param}: no 400-probe"
        elif disposition == "advisory":
            assert param in AUDIO_PROBE_200, f"audio_transcriptions.{param}: no 200-probe"
    unused = (set(AUDIO_PROBE_400) | set(AUDIO_PROBE_200)) - listed
    assert not unused, f"audio_transcriptions: probes for params not in the matrix: {sorted(unused)}"

    declared = set(matrix["_extensions"]["audio_transcriptions"])
    checked = (
        set(AUDIO_EXTENSION_CHECKS)
        | set(AUDIO_STREAM_EXTENSION_CHECKS)
        | set(AUDIO_GUARD_EXTENSION_CHECKS)
    )
    assert declared == checked, (
        "audio_transcriptions extensions and their checks disagree: "
        f"declared {sorted(declared)}, checked {sorted(checked)} - an extension "
        "nobody exercises can be renamed or dropped without a failure"
    )
    total = sum(counts.values())
    print(
        f"  audio_transcriptions: {total} spec params - {counts['implemented']} implemented, "
        f"{counts['partial']} partial, {counts['advisory']} advisory, "
        f"{counts['rejected']} rejected (+{len(declared)} extensions)",
        flush=True,
    )
    ok("audio_transcriptions matrix (static)")


# ---------------------------------------------------------------------------
# /v1/images/generations and /v1/images/edits
#
# Same split as audio, for the same reason: the surface needs a LOADED
# image-generation model. The matrix audit is static and runs everywhere; the
# probes and the functional cases run under --images-only
# (sdk_conformance.rs::spec_gate_qwen_image starts a Qwen-Image server).
# ---------------------------------------------------------------------------


def sec_images_matrix(matrix):
    """Completeness of both image matrices + a usable probe for every
    generations disposition that claims a refusal; and the edits matrix is
    ALL refused, because the endpoint is - a lone `implemented` there would
    be a claim nothing serves."""
    for family in ("images_generations", "images_edits"):
        spec = sdk_params(family)
        listed = set(matrix[family].keys())
        missing = spec - listed
        stale = listed - spec
        assert not missing, f"{family}: spec params without a disposition: {sorted(missing)}"
        assert not stale, f"{family}: stale matrix entries: {sorted(stale)}"
    counts = {"implemented": 0, "partial": 0, "advisory": 0, "rejected": 0}
    for param, disposition in sorted(matrix["images_generations"].items()):
        assert disposition in counts, f"images_generations.{param}: bad disposition {disposition}"
        counts[disposition] += 1
        if disposition in ("rejected", "partial"):
            assert param in PROBE_400["images_generations"], f"images_generations.{param}: no 400-probe"
        elif disposition == "advisory":
            assert param in PROBE_200["images_generations"], f"images_generations.{param}: no 200-probe"
    unused = (set(PROBE_400["images_generations"]) | set(PROBE_200["images_generations"])) - set(
        matrix["images_generations"]
    )
    assert not unused, f"images_generations: probes for params not in the matrix: {sorted(unused)}"
    edit_counts = {"implemented": 0, "partial": 0, "advisory": 0, "rejected": 0}
    for param, disposition in matrix["images_edits"].items():
        assert disposition in edit_counts, f"images_edits.{param}: bad disposition {disposition}"
        edit_counts[disposition] += 1
        # every refusal the edits matrix claims is one sec_images_edits probes
        if disposition == "rejected":
            assert param in ("mask",), f"images_edits.{param}: refused with no probe for it"
    declared = set(matrix["_extensions"]["images_generations"])
    checked = set(IMAGES_EXTENSION_CHECKS)
    assert declared == checked, (
        "images_generations extensions and their checks disagree: "
        f"declared {sorted(declared)}, checked {sorted(checked)}"
    )
    total = sum(counts.values())
    print(
        f"  images_generations: {total} spec params - {counts['implemented']} implemented, "
        f"{counts['partial']} partial, {counts['advisory']} advisory, "
        f"{counts['rejected']} rejected (+{len(declared)} extensions); "
        f"images_edits: {len(matrix['images_edits'])} spec params - "
        f"{edit_counts['implemented']} implemented, {edit_counts['partial']} partial, "
        f"{edit_counts['advisory']} advisory, {edit_counts['rejected']} rejected",
        flush=True,
    )
    ok("images matrices (static)")


def image_post(client, body):
    return client.post("/v1/images/generations", json=body)


def decoded_image(resp_json, index=0):
    """The image bytes out of a response, plus (format, width, height) read
    off the container itself - a response that says png and ships a jpeg
    fails here, not in a viewer."""
    data = resp_json["data"][index]
    assert data.get("b64_json"), f"image {index} carries no b64_json: {list(data)}"
    raw = base64.b64decode(data["b64_json"])
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        assert raw[12:16] == b"IHDR", "PNG without an IHDR first"
        return raw, ("png", int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big"))
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return raw, ("webp", None, None)
    if raw[:2] == b"\xff\xd8":
        return raw, ("jpeg", None, None)
    raise AssertionError(f"image {index} is not a png, webp or jpeg: {raw[:16]!r}")


def check_ext_size_pair(client, base):
    """`width`/`height` - the pair form of `size`; the two together are refused
    by name rather than one silently winning."""
    body = {k: v for k, v in base.items() if k != "size"}
    r = image_post(client, {**body, "width": 512, "height": 384})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["size"] == "512x384", r.json().get("size")
    _, (fmt, w, h) = decoded_image(r.json())
    assert (fmt, w, h) == ("png", 512, 384), (fmt, w, h)
    r = image_post(client, {**base, "width": 512, "height": 384})
    assert r.status_code == 400, r.text[:200]
    assert "size" in r.json()["error"]["message"], r.text[:200]
    r = image_post(client, {**body, "width": 512})
    assert r.status_code == 400, r.text[:200]
    assert "height" in r.json()["error"]["message"], r.text[:200]


def check_ext_seed(client, base):
    """`seed` - same seed, same bytes; a different seed, different bytes; an
    omitted seed is the documented default and reproducible too."""
    a = image_post(client, {**base, "seed": 11})
    b = image_post(client, {**base, "seed": 11})
    c = image_post(client, {**base, "seed": 12})
    for r in (a, b, c):
        assert r.status_code == 200, r.text[:200]
    assert a.json()["data"][0]["b64_json"] == b.json()["data"][0]["b64_json"], (
        "the same seed rendered different bytes"
    )
    assert a.json()["data"][0]["b64_json"] != c.json()["data"][0]["b64_json"], (
        "a different seed rendered the same bytes"
    )
    no_seed = {k: v for k, v in base.items() if k != "seed"}
    d = image_post(client, no_seed)
    e = image_post(client, no_seed)
    assert d.status_code == 200 and e.status_code == 200, (d.text[:200], e.text[:200])
    assert d.json()["data"][0]["b64_json"] == e.json()["data"][0]["b64_json"], (
        "an omitted seed is not reproducible - the default must be a fixed seed"
    )


def check_ext_steps(client, base):
    """`steps` - overrides the quality's count; 0 and over the ceiling are
    refused by name."""
    for bad in (0, 100_000):
        r = image_post(client, {**base, "steps": bad})
        assert r.status_code == 400, f"steps={bad}: {r.status_code} {r.text[:200]}"
        assert "steps" in r.json()["error"]["message"], r.text[:200]
    r = image_post(client, {**base, "steps": 2})
    assert r.status_code == 200, r.text[:200]


def check_ext_guidance(client, base):
    """`guidance` - classifier-free guidance, off by default (the model's own
    true_cfg_scale is 1); negative values are refused by name."""
    r = image_post(client, {**base, "guidance": -1.0})
    assert r.status_code == 400, r.text[:200]
    assert "guidance" in r.json()["error"]["message"], r.text[:200]
    r = image_post(client, {**base, "guidance": 1.0})
    assert r.status_code == 200, r.text[:200]


def check_ext_negative_prompt(client, base):
    """`negative_prompt` - means nothing without guidance above 1, and says
    so rather than being dropped on the floor."""
    r = image_post(client, {**base, "negative_prompt": "blue"})
    assert r.status_code == 400, r.text[:200]
    assert "guidance" in r.json()["error"]["message"], r.text[:200]
    r = image_post(client, {**base, "negative_prompt": "blue", "guidance": 2.0})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["usage"]["input_tokens"] > 0, r.json().get("usage")


IMAGES_EXTENSION_CHECKS = {
    "width": check_ext_size_pair,
    "height": check_ext_size_pair,
    "seed": check_ext_seed,
    "steps": check_ext_steps,
    "guidance": check_ext_guidance,
    "negative_prompt": check_ext_negative_prompt,
}


def sec_images_edits(client, matrix, model, base_url):
    """`/v1/images/edits`: the GPT-image edit form, multipart. A reference
    rendered by the server itself (one step, a pinned seed, so the bytes are
    the same every run) is handed back with an instruction; the response is
    the generations shape (validated against the SDK's `ImagesResponse`), the
    output takes the LAST reference's shape at the output area unless `size`
    says otherwise, and the usage tells picture tokens from text tokens. Two
    references make two slots; the refusals - a mask, no image, a URL
    response, a bad fidelity, an unknown field - answer by name after the
    whole form is read. Without the vision tower the section reports itself
    not run: the endpoint then refuses everything by name, checked here too."""
    edits = matrix["images_edits"]
    ref_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    caps = client.get("/api/server").json().get("image_generation") or {}
    if not caps.get("edit"):
        r = client.post(
            "/v1/images/edits",
            data={"prompt": "make it blue", "model": model},
            files={"image": ("a.png", ref_png, "image/png")},
        )
        assert r.status_code == 400, f"{r.status_code} {r.text[:200]}"
        assert "tower" in r.json()["error"]["message"], r.text[:200]
        print("  images edits NOT RUN: no vision tower wired (mmproj) - the refusal names it", flush=True)
        return
    # the reference: a small render, then its bytes as the upload
    r = image_post(client, {**images_family(model).base, "size": "512x512"})
    assert r.status_code == 200, r.text[:200]
    ref, _ = decoded_image(r.json())
    form = {"prompt": "Make it a pencil sketch on white paper", "model": model, "steps": "1", "seed": "7"}

    def edit(data, files, expect=200):
        # a form with no picture is still a multipart form (httpx would
        # urlencode a bare `data`, which is a different refusal)
        if not files:
            files, data = {k: (None, str(v)) for k, v in data.items()}, None
        r = client.post("/v1/images/edits", data=data, files=files)
        assert r.status_code == expect, f"{data}: {r.status_code} {r.text[:300]}"
        return r

    # one reference, the output at the reference's aspect at the output area
    # (a 512^2 reference is resized to 1024^2 by the pipeline's rule, so the
    # output is the SDK's own square)
    r = edit(form, {"image": ("ref.png", ref, "image/png")})
    body = r.json()
    spec = ImagesResponse.model_validate(body)
    assert spec.size == "1024x1024" and spec.quality == "high", (spec.size, spec.quality)
    assert spec.usage and spec.usage.input_tokens_details.image_tokens == 32 * 32, spec.usage
    assert spec.usage.input_tokens_details.text_tokens > 0, spec.usage
    assert spec.usage.output_tokens == 64 * 64, spec.usage
    once, (fmt, w, h) = decoded_image(body)
    assert (fmt, w, h) == ("png", 1024, 1024), (fmt, w, h)
    # reproducible: the same form, the same bytes
    again, _ = decoded_image(edit(form, {"image": ("ref.png", ref, "image/png")}).json())
    assert again == once, "the same edit rendered different bytes"

    # a chosen size wins over the reference's shape, and it also sets the
    # reference area (the pipeline's one output_resolution sizes both), so
    # a 1024^2 reference fitted at 512^2 is 16x16 tokens on the merged grid;
    # the SDK's `image[]` spelling for two references makes two slots
    r = edit(
        {**form, "size": "512x512"},
        [("image[]", ("a.png", ref, "image/png")), ("image[]", ("b.png", ref, "image/png"))],
    )
    body = r.json()
    assert body["size"] == "512x512", body.get("size")
    assert body["usage"]["input_tokens_details"]["image_tokens"] == 2 * 16 * 16, body["usage"]
    _, (fmt, w, h) = decoded_image(body)
    assert (fmt, w, h) == ("png", 512, 512), (fmt, w, h)

    # the stream form, as generations: a preview then completed
    events = []
    with client.stream(
        "POST",
        "/v1/images/edits",
        data={**form, "stream": "true", "partial_images": "1"},
        files={"image": ("ref.png", ref, "image/png")},
    ) as r:
        assert r.status_code == 200, r.read()[:200]
        for line in r.iter_lines():
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                events.append(payload if payload == "[DONE]" else json.loads(payload))
    assert events and events[-1] == "[DONE]", events[-1:]
    kinds = [e["type"] for e in events[:-1]]
    assert kinds == ["image_generation.partial_image", "image_generation.completed"], kinds
    adapter = TypeAdapter(ImageGenStreamEvent)
    for e in events[:-1]:
        adapter.validate_python(e)

    # the refusals, by name
    for data, files, needle in (
        ({**form}, {"image": ("ref.png", ref, "image/png"), "mask": ("m.png", ref, "image/png")}, "mask"),
        ({**form}, {}, "image"),
        ({**form, "response_format": "url"}, {"image": ("ref.png", ref, "image/png")}, "response_format"),
        ({**form, "input_fidelity": "ultra"}, {"image": ("ref.png", ref, "image/png")}, "input_fidelity"),
        ({**form, "nonsense": "1"}, {"image": ("ref.png", ref, "image/png")}, "nonsense"),
    ):
        r = edit(data, files, 400)
        msg = r.json()["error"]["message"]
        assert needle in msg, f"{needle}: {msg!r}"
    # not a form at all: the OpenAI error shape still, naming the form
    r = client.post("/v1/images/edits", json={"prompt": "x"})
    assert r.status_code == 400, f"{r.status_code} {r.text[:200]}"
    assert "multipart" in r.json()["error"]["message"], r.text[:200]
    assert edits["mask"] == "rejected" and edits["image"] == "implemented", edits

    # the pinned SDK's own call
    import openai  # noqa: PLC0415

    sdk = openai.OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="conformance", timeout=600.0)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(ref)
        path = f.name
    try:
        with open(path, "rb") as fh:
            out = sdk.images.edit(
                model=model,
                image=fh,
                prompt=form["prompt"],
                size="1024x1024",
                extra_body={"steps": 1, "seed": 7},
            )
    finally:
        os.unlink(path)
    assert out.data and out.data[0].b64_json, out
    assert base64.b64decode(out.data[0].b64_json) == once, "the SDK's edit rendered different bytes"
    ok("images edits: one and two references, stream, refusals, the SDK's images.edit")


def sec_strict_images(client, matrix, model, base_url):
    """The functional half: real renders validated against the SDK's
    `ImagesResponse`, the container checked against what the response claims,
    the extensions exercised, and one round trip through the pinned SDK."""
    base = images_family(model).base

    # a spec size, validated STRICTLY: the SDK's literals for size and quality
    # are the three gpt-image sizes and low/medium/high, so this is the render
    # a strict client can read whole
    r = image_post(client, {**base, "size": "1024x1024"})
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    spec = ImagesResponse.model_validate(body)
    assert spec.size == "1024x1024" and spec.quality == "high", (spec.size, spec.quality)
    assert spec.output_format == "png" and spec.background in ("opaque", "transparent"), (
        spec.output_format, spec.background,
    )
    assert spec.usage and spec.usage.output_tokens == 64 * 64, spec.usage
    assert spec.usage.input_tokens > 0 and spec.usage.total_tokens == (
        spec.usage.input_tokens + spec.usage.output_tokens
    ), spec.usage
    raw, (fmt, w, h) = decoded_image(body)
    assert (fmt, w, h) == ("png", 1024, 1024), (fmt, w, h)

    # the family's own base size (custom, on the grid): the response echoes the
    # size generated, checked by hand because the SDK literal cannot hold it
    r = image_post(client, base)
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["size"] == "512x512", body.get("size")
    _, (fmt, w, h) = decoded_image(body)
    assert (fmt, w, h) == ("png", 512, 512), (fmt, w, h)
    assert body["usage"]["output_tokens"] == 32 * 32, body["usage"]

    # n: one request, several images, each its own draw
    r = image_post(client, {**base, "n": 2})
    assert r.status_code == 200, r.text[:200]
    imgs = r.json()["data"]
    assert len(imgs) == 2 and imgs[0]["b64_json"] != imgs[1]["b64_json"], "n=2 must draw twice"

    # off the grid: refused by name, with the grid in the message
    r = image_post(client, {**base, "size": "1000x1000"})
    assert r.status_code == 400, r.text[:200]
    assert "grid" in r.json()["error"]["message"], r.text[:200]

    # quality's echo is what the render used - low is low
    r = image_post(client, {**base, "quality": "low"})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["quality"] == "low", r.json().get("quality")

    # formats: the container matches the claim; compression only where it means
    # something
    r = image_post(client, {**base, "output_format": "webp", "output_compression": 50})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["output_format"] == "webp", r.json().get("output_format")
    _, (fmt, _, _) = decoded_image(r.json())
    assert fmt == "webp", fmt
    r = image_post(client, {**base, "output_format": "jpeg"})
    assert r.status_code == 200, r.text[:200]
    _, (fmt, _, _) = decoded_image(r.json())
    assert fmt == "jpeg", fmt
    r = image_post(client, {**base, "output_format": "png", "output_compression": 50})
    assert r.status_code == 400, r.text[:200]
    assert "output_compression" in r.json()["error"]["message"], r.text[:200]

    # background: transparent needs an alpha channel, and the echo says what
    # was produced, never `auto`
    r = image_post(client, {**base, "background": "transparent", "output_format": "jpeg"})
    assert r.status_code == 400, r.text[:200]
    assert "transparent" in r.json()["error"]["message"], r.text[:200]
    r = image_post(client, {**base, "background": "transparent"})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["background"] in ("transparent", "opaque"), r.json().get("background")

    # the extensions, each exercised so a rename or a quiet removal fails
    for name, check in IMAGES_EXTENSION_CHECKS.items():
        assert name in matrix["_extensions"]["images_generations"], name
    for check in dict.fromkeys(IMAGES_EXTENSION_CHECKS.values()):
        check(client, base)

    # the listing says what this is: text in, an image out, no window
    entry = client.get("/v1/models").json()["data"][0]
    arch = entry.get("architecture", {})
    assert arch.get("output_modalities") == ["image"], arch
    assert entry.get("capabilities", {}).get("image_generation") is True, entry.get("capabilities")

    # and through the PINNED SDK, the arbiter of the wire format: a typed
    # ImagesResponse with a decodable image, our extensions riding extra_body
    import openai  # noqa: PLC0415

    sdk = openai.OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="conformance")
    via_sdk = sdk.images.generate(
        model=model,
        prompt=base["prompt"],
        size="1024x1024",
        output_format="png",
        extra_body={"steps": 1, "seed": 7},
    )
    assert isinstance(via_sdk, ImagesResponse), type(via_sdk)
    assert via_sdk.data and via_sdk.data[0].b64_json, "the SDK saw no image"
    sdk_raw = base64.b64decode(via_sdk.data[0].b64_json)
    assert sdk_raw == raw, "the SDK client and the raw client rendered different bytes for the same seed"
    ok("strict validation: images generations (spec + custom sizes, n, formats, background, extensions, SDK)")


def sec_images_stream(client, matrix, model, base_url):
    """`stream: true` with `partial_images` previews.

    Two invariants. The events are the SDK's - every one validates against
    `ImageGenStreamEvent`, previews are indexed 0.. in order, `completed` comes
    last and alone carries the usage. And streaming must not CHANGE the answer:
    the completed image is byte-identical to the oneshot render of the same
    seed and steps, because a preview is a decode of the render in flight,
    never a different render."""
    assert matrix["images_generations"]["stream"] == "implemented"
    assert matrix["images_generations"]["partial_images"] == "implemented"
    base = {**images_family(model).base, "steps": 4}

    # the refusals, by name: several images per stream, previews without one,
    # more previews than the API allows
    for extra, needle in (
        ({"stream": True, "n": 2}, "n"),
        ({"partial_images": 2}, "stream"),
        ({"stream": True, "partial_images": 4}, "partial_images"),
    ):
        r = image_post(client, {**base, **extra})
        assert r.status_code == 400, f"{extra}: {r.status_code} {r.text[:200]}"
        assert needle in r.json()["error"]["message"], r.text[:200]

    # The streamed leg renders a SPEC size: the SDK's event models carry the
    # same three-value size literal as ImagesResponse, and a custom size would
    # fail their validation for the size alone (the refusal probes above keep
    # the small base - nothing renders there).
    base = {**base, "size": "1024x1024"}
    events, ctype = [], None
    with client.stream(
        "POST", "/v1/images/generations", json={**base, "stream": True, "partial_images": 2}
    ) as r:
        assert r.status_code == 200, r.read()[:200]
        ctype = r.headers.get("content-type", "")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                events.append("[DONE]")
                continue
            events.append(json.loads(payload))
    assert ctype.startswith("text/event-stream"), f"streamed as {ctype!r}"
    assert events and events[-1] == "[DONE]", events[-1:]
    events = events[:-1]
    adapter = TypeAdapter(ImageGenStreamEvent)
    for e in events:
        adapter.validate_python(e)
    partials = [e for e in events if e["type"] == "image_generation.partial_image"]
    assert [p["partial_image_index"] for p in partials] == [0, 1], (
        f"previews out of order or missing: {[p.get('partial_image_index') for p in partials]}"
    )
    assert events[-1]["type"] == "image_generation.completed", events[-1]["type"]
    assert len(events) == 3, f"{len(events)} events for 2 previews + completed"
    done = events[-1]
    assert done["usage"]["output_tokens"] == 64 * 64, done["usage"]
    assert all("usage" not in p or p["usage"] is None for p in partials), "usage on a preview"
    for e in events:
        _, (fmt, w, h) = decoded_image({"data": [{"b64_json": e["b64_json"]}]})
        assert (fmt, w, h) == ("png", 1024, 1024), (e["type"], fmt, w, h)
        assert e["size"] == "1024x1024" and e["quality"] == "high", (e["size"], e["quality"])

    # streaming must not change the answer
    r = image_post(client, base)
    assert r.status_code == 200, r.text[:200]
    assert done["b64_json"] == r.json()["data"][0]["b64_json"], (
        "the streamed final image differs from the oneshot render of the same seed"
    )

    # and through the PINNED SDK, whose stream class decides what an event is
    import openai  # noqa: PLC0415
    from openai.types import ImageGenCompletedEvent, ImageGenPartialImageEvent  # noqa: PLC0415

    sdk = openai.OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="conformance")
    seen = []
    for ev in sdk.images.generate(
        model=model,
        prompt=base["prompt"],
        size="1024x1024",
        stream=True,
        partial_images=1,
        extra_body={"steps": 2, "seed": 7},
    ):
        seen.append(type(ev))
    assert seen == [ImageGenPartialImageEvent, ImageGenCompletedEvent], seen
    ok("images streaming: partial_image x2 + completed, byte-identical to the oneshot render")


def audio_post(client, wav, fields):
    """One multipart transcription request. `wav` is (filename, bytes) - the
    name and type are sent the way a real client sends them, even though the
    server sniffs the container from the bytes. `fields` is a list of pairs,
    not a dict, because the spec's array params legitimately repeat a name -
    httpx spells that as a list value, which is what the fold below builds."""
    name, blob = wav
    data = {}
    for field, value in fields:
        data.setdefault(field, []).append(value)
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": (name, blob, mimetypes.guess_type(name)[0] or "application/octet-stream")},
        data=data,
    )


def sec_audio_surface(client, matrix, wav, model):
    for param, disposition in sorted(matrix["audio_transcriptions"].items()):
        if disposition in ("rejected", "partial"):
            fields = [("model", model), *AUDIO_PROBE_400[param]]
            r = audio_post(client, wav, fields)
            assert r.status_code == 400, (
                f"audio.{param}={fields!r}: expected 400, got {r.status_code}: {r.text[:200]}"
            )
            body = r.json()
            assert body.get("error", {}).get("message"), body
            assert body["error"].get("type"), body
            # "unsupported parameter" tells a caller nothing about whether to
            # wait for it or work around it - the refusal must say what. For a
            # param whose values split that is the VALUE, because the field is
            # served and only this value is not. `languages` splits only on a
            # lane that has detection to bias: one without refuses the whole
            # parameter for every value, never reads the code, and naming a
            # code it did not read would be a lie - so there it must name the
            # field instead.
            named = AUDIO_PROBE_NAMES.get(param, param)
            if param == "languages" and not language_caps(client)["hints"]:
                named = param
            assert named in body["error"]["message"], (
                f"audio.{param}: refusal does not name {named!r}: {body['error']['message']}"
            )
        elif disposition == "advisory":
            fields = [("model", model), *AUDIO_PROBE_200[param]]
            r = audio_post(client, wav, fields)
            assert r.status_code == 200, (
                f"audio.{param}: expected 200, got {r.status_code}: {r.text[:200]}"
            )

    # An explicit false is not a request for the feature, so it is not refused.
    r = audio_post(client, wav, [("model", model), ("stream", "false")])
    assert r.status_code == 200, f"stream=false must pass: {r.status_code} {r.text[:200]}"

    # Unknown form fields are refused by NAME, same as the JSON lanes: a field
    # read by nobody is a caller belief the server never contradicts.
    r = audio_post(client, wav, [("model", model), ("definitely_not_a_param", "1")])
    assert r.status_code == 400, r.text[:200]
    assert "definitely_not_a_param" in r.json()["error"]["message"], r.text[:200]

    # Values outside what the format accepts, named the same way.
    for fields, needle in (
        ([("response_format", "fancy")], "fancy"),
        ([("timestamp_granularities[]", "sentence")], "sentence"),
        ([("timestamp_granularities[]", "segment")], "verbose_json"),
        # a granularity can be refused for where it was asked rather than for
        # what it is: a subtitle cue is a sentence, so one cue per word is not
        # a subtitle file anyone can read
        ([("response_format", "srt"), ("timestamp_granularities[]", "word")], "srt"),
    ):
        r = audio_post(client, wav, [("model", model), *fields])
        assert r.status_code == 400, f"{fields!r}: {r.status_code} {r.text[:200]}"
        assert needle in r.json()["error"]["message"], r.text[:200]
    ok("surface audit: audio_transcriptions")


def parse_cues(doc, sep, numbered):
    """Parse SRT/WebVTT into (index, start_s, end_s, text). Strict about the
    decimal separator: a comma file parsed as VTT fails here the way it fails
    in a player."""
    stamp = rf"(\d{{2}}):(\d{{2}}):(\d{{2}}){re.escape(sep)}(\d{{3}})"
    timing = re.compile(rf"^{stamp} --> {stamp}$")
    cues = []
    for block in re.split(r"\n\s*\n", doc.strip()):
        lines = block.strip().split("\n")
        n = None
        if numbered:
            assert lines[0].strip().isdigit(), f"cue does not start with a number: {lines[0]!r}"
            n = int(lines.pop(0))
        m = timing.match(lines.pop(0).strip())
        assert m, f"bad timing line in {block!r}"
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(lines).strip()
        assert text, f"empty cue: {block!r}"
        cues.append((n, start, end, text))
    return cues


def check_paddock_words(segments):
    """`paddock_words` - per-word confidence, ours, named so it cannot be
    mistaken for spec (OpenAI's word object carries no probability)."""
    for seg in segments:
        words = seg.get("paddock_words")
        assert words, f"segment {seg['id']} has no paddock_words"
        for w in words:
            # Three required, two optional: `paddock_alt`/`paddock_margin` ride
            # only where the lane could answer them, so their absence is honest
            # rather than a zero reading as "no close call". Written as two
            # subset checks and not one equality because the equality version
            #  went red the day the margin shipped - a gate
            # that forbids a field arriving is not the gate anyone wanted.
            assert {"word", "logprob", "confidence"} <= set(w), f"missing keys: {w!r}"
            assert set(w) <= {"word", "logprob", "confidence", "paddock_alt", "paddock_margin"}, (
                f"unknown keys on a word: {sorted(set(w) - {'word', 'logprob', 'confidence', 'paddock_alt', 'paddock_margin'})} in {w!r}"
            )
            assert w["word"].strip() == w["word"] and w["word"], f"untrimmed word: {w!r}"
            assert 0.0 < w["confidence"] <= 1.0, w
            assert abs(w["confidence"] - math.exp(w["logprob"])) < 1e-4, w
        # the words are the segment, not an approximation of it - a grouping
        # bug here renders the wrong words as uncertain, silently
        rejoined = " ".join(w["word"] for w in words)
        assert rejoined == " ".join(seg["text"].split()), (
            f"segment {seg['id']}: words rejoin to {rejoined!r}, text is {seg['text']!r}"
        )


AUDIO_EXTENSION_CHECKS = {"paddock_words": check_paddock_words}


def check_paddock_verbose(done, segments):
    """`paddock_verbose` - the verbose_json body riding the terminal streaming
    event. OpenAI's `transcript.text.done` carries `text` and nothing else about
    the transcript, so without this a caller who streamed AND wanted segments
    would have to send the clip a second time."""
    v = done.get("paddock_verbose")
    assert v, "streamed verbose_json carries no paddock_verbose"
    assert v["task"] == "transcribe", v
    assert v["text"] == done["text"], (v["text"][:120], done["text"][:120])
    assert v["duration"] > 0, v
    assert v["language"], v
    # A lane with no segments sends no `segments` key rather than an empty
    # array - an empty array is a claim that there were none to find.
    if segments is None:
        assert "segments" not in v, f"a segment-less lane sent a segments key: {v.keys()}"
        return
    # the same object the non-streaming lane returns, or it is a second answer
    # to the same question and one of them is wrong
    assert len(v["segments"]) == len(segments), (len(v["segments"]), len(segments))
    for a, b in zip(v["segments"], segments):
        assert a["text"] == b["text"], (a["text"], b["text"])
        assert abs(a["start"] - b["start"]) < 1e-6, (a["start"], b["start"])
        assert abs(a["end"] - b["end"]) < 1e-6, (a["end"], b["end"])
    if "words" in v:
        # The real word array shipped, so the per-segment copy has RETIRED -
        # the same rule the non-streaming body follows, and it has to hold here
        # too or the streamed answer carries a fact the final one dropped.
        for seg in v["segments"]:
            assert "paddock_words" not in seg, (
                f"streamed segment {seg['id']} carries paddock_words beside a real `words` array"
            )
    else:
        check_paddock_words(v["segments"])


AUDIO_STREAM_EXTENSION_CHECKS = {"paddock_verbose": check_paddock_verbose}


def check_paddock_guards(body, duration):
    """`paddock_guards` - the decode-guard notices.

    Returns how many rode along. ABSENT is the normal answer and the one the
    clean-clip leg asserts; this validates the shape wherever one does appear,
    because a notice a client cannot parse is no better than no notice."""
    g = body.get("paddock_guards")
    if g is None:
        return 0
    assert isinstance(g, list) and g, "paddock_guards present but empty - absence means clean"
    for n in g:
        assert n["reason"] in {"no_speech", "no_speech_marker", "vad",
                               "repetition", "length", "context",
                               "language_mismatch", "timestamps"}, n
        assert isinstance(n["text_dropped"], bool), n
        # the note is the difference between a machine-readable reason and a
        # caller having to read our source to know whether to re-record
        assert n["note"].strip(), n
        assert 0.0 <= n["start"] <= n["end"] <= duration + 1e-6, (n, duration)
        if "no_speech_prob" in n:
            assert 0.0 <= n["no_speech_prob"] <= 1.0, n
        if "avg_logprob" in n:
            assert n["avg_logprob"] <= 0.0, n
        if "entropy" in n:
            assert n["entropy"] >= 0.0, n
    return len(g)


def check_paddock_guard_stats(body, duration, lane_has_no_speech):
    """`paddock_guard_stats` - every guard signal, fired or not.

    The one extension asked for by a request field, and the one that has to be
    PRESENT rather than conditionally absent: a tuning surface that vanishes on
    the clips where nothing went wrong is useless for measuring headroom, which
    is the whole reason it exists."""
    stats = body.get("paddock_guard_stats")
    assert stats, f"asked for guard stats and got {stats!r}"
    prev_end = -1e-9
    for i, s in enumerate(stats):
        assert 0.0 <= s["start"] <= s["end"] <= duration + 1e-6, (s, duration)
        assert s["start"] >= prev_end - 1e-6, f"window {i} starts before {prev_end}: {s}"
        prev_end = s["end"]
        assert isinstance(s["tokens"], int) and s["tokens"] >= 0, s
        assert s["stop"] in {"eot", "no_speech", "no_speech_marker", "vad",
                             "repetition", "length", "context"}, s
        # the thresholds ride along so an artifact keeps meaning what it meant
        # when it was recorded, even after we move one
        t = s["thresholds"]
        assert t["entropy"] > 0 and t["logprob"] < 0 and 0 < t["no_speech"] < 1, t
        if "entropy" in s:
            assert s["entropy"] >= 0.0, s
        if "avg_logprob" in s:
            assert s["avg_logprob"] <= 0.0, s
        if "compression_ratio" in s:
            assert s["compression_ratio"] > 0.0, s
        # ABSENT, not zero, where the lane has no such head - a 0.0 there would
        # read as "certainly speech" and is the exact lie this key must not tell
        if lane_has_no_speech:
            assert 0.0 <= s["no_speech_prob"] <= 1.0, s
        else:
            assert "no_speech_prob" not in s, f"a lane with no no-speech head reported one: {s}"
    return len(stats)


def check_paddock_language(body, caps):
    """`paddock_language` - everything the response can say about LANGUAGE

    The shape is what is asserted, not the verdict: which language a given
    checkpoint hears is the model's business, but a client reading this object
    has to be able to tell a MEASUREMENT from an INSTRUCTION reflected back, and
    that is exactly what `source` carries. A lane that detects nothing sends no
    object at all rather than an empty one."""
    lang = body.get("paddock_language")
    if lang is None:
        # only honest where nothing could say anything - a lane that publishes
        # a detection capability must use it
        assert not caps.get("supported"), (
            "this model advertises language detection and reported none"
        )
        return None
    assert lang["source"] in {"asked", "detected", "reported", "unknown"}, lang
    if "code" in lang:
        assert lang["code"].strip(), lang
        assert lang["name"].strip(), lang
    # A probability is a MEASUREMENT and rides only where one was made -
    # a forced code and a model's self-report have no number, and a 1.0 there
    # would read as certainty nobody claimed.
    if lang["source"] != "detected":
        assert "probability" not in lang, f"a number was invented for {lang['source']}: {lang}"
    else:
        assert 0.0 <= lang["probability"] <= 1.0, lang
        cands = lang["candidates"]
        assert cands, "a detected language with no distribution behind it"
        ps = [c["probability"] for c in cands]
        assert ps == sorted(ps, reverse=True), f"candidates are not ranked: {ps}"
        assert all(0.0 <= p <= 1.0 for p in ps), ps
        # the reported code is the top of its own distribution, or the two
        # halves of this object disagree about the same fact
        assert cands[0]["code"] == lang["code"], (cands[0]["code"], lang["code"])
    if "written" in lang:
        w = lang["written"]
        assert w["code"].strip() and 0.0 <= w["confidence"] <= 1.0, w
        if "code" in lang:
            assert isinstance(w["agrees"], bool), w
    return lang


AUDIO_GUARD_EXTENSION_CHECKS = {
    "paddock_guards": check_paddock_guards,
    "paddock_guard_stats": check_paddock_guard_stats,
    "paddock_language": check_paddock_language,
}


def speechless_wav(name, sample, seconds=6.0, rate=16000):
    """A clip with ENERGY and no speech in it. `sample(i) -> float in [-1, 1]`.

    Digital silence is the wrong probe: whisper usually answers it with an
    immediate `<|endoftext|>`, which is correct and exercises nothing. Energy
    without speech is where the hallucination actually lives."""
    import struct  # noqa: PLC0415
    import wave  # noqa: PLC0415

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", max(-32767, min(32767, int(32767 * sample(i)))))
                for i in range(int(rate * seconds))
            )
        )
    return (name, buf.getvalue())


def speechless_clips():
    """The two shapes of "no speech", both deterministic - a gate that reads a
    different clip on every run is a gate nobody can bisect.

    `room` is the case names: a recording whose lead-in is quiet but
    not empty. Broadband and near the noise floor, which is exactly what
    whisper writes "Thank you for watching" over. `tone` is the music-bed case:
    loud, narrowband, obviously not speech."""

    # a 32-bit LCG (glibc's constants) - reproducible noise, no RNG state to
    # depend on across python versions
    def noise(amp):
        state = [12345]

        def sample(_i):
            state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
            return amp * (state[0] / 0x3FFFFFFF - 1.0)

        return sample

    return [
        speechless_wav("room.wav", noise(0.012)),
        speechless_wav("tone.wav", lambda i: 0.2 * math.sin(2 * math.pi * 440.0 * i / 16000)),
    ]


def sec_audio_guards(client, wav, model):
    """Decode guards: the transcript says when it stopped being
    about the audio.

    Two legs, and the first is the one that must never go red. Guards that fire
    on ordinary speech would move every WER number on the board silently, so
    the clean clip is asserted to carry no notice at all - not an empty array,
    no key.

    The speechless legs are deliberately not equalities. Whether a given
    checkpoint hallucinates over room tone is the model's business, not the
    API's, so demanding a notice here would pin a gate to a model's mood. What
    is asserted is the invariant, which holds either way: a span reported as
    dropped leaves no text behind. A checkpoint that stays quiet passes it
    trivially, and that is the right outcome - there was nothing to guard."""
    fields = [("model", model), ("response_format", "verbose_json")]
    r = audio_post(client, wav, [*fields, ("paddock_guard_stats", "true")])
    assert r.status_code == 200, r.text[:200]
    clean = r.json()
    assert "paddock_guards" not in clean, (
        "a guard fired on ordinary speech - every WER number on the board is "
        f"downstream of this: {clean.get('paddock_guards')}"
    )
    # whisper is the only family with a no-speech head; the generative lanes
    # answer three of the four signals and must say so by absence
    whisper_lane = bool(clean.get("segments")) or any(
        "no_speech_prob" in s for s in clean.get("paddock_guard_stats") or []
    )
    n_stats = check_paddock_guard_stats(clean, clean["duration"], whisper_lane)
    # the headroom this whole key exists to make measurable - printed, not
    # asserted, because the number is a per-checkpoint fact and pinning it here
    # would make the gate a calibration it is not
    worst = min(
        (s["entropy"] - s["thresholds"]["entropy"] for s in clean["paddock_guard_stats"]
         if "entropy" in s),
        default=float("inf"),
    )
    print(
        f"  clean clip: {n_stats} window(s), closest entropy headroom "
        f"{'n/a' if worst == float('inf') else f'{worst:+.2f} nats'}",
        flush=True,
    )
    # and the default answer carries none of it
    bare = audio_post(client, wav, fields).json()
    assert "paddock_guard_stats" not in bare, "guard stats rode along unasked"

    for clip in speechless_clips():
        r = audio_post(client, clip, fields)
        assert r.status_code == 200, r.text[:200]
        body = r.json()
        n = check_paddock_guards(body, body["duration"])
        if any(g["text_dropped"] for g in body.get("paddock_guards", [])):
            # each probe is under 30 s, so it is one whisper window and a drop
            # covers all of it
            assert not body["text"].strip(), (
                "a span was reported dropped and its text is still in the "
                f"transcript: {body['text'][:120]!r}"
            )
        print(
            f"  {clip[0]}: {n} notice(s), transcript {len(body['text'].strip())} chars",
            flush=True,
        )
    ok("audio decode guards")


def audio_sse_events(client, wav, fields):
    """Post a streaming transcription and return its parsed data payloads, in
    order. Written by hand rather than through the SDK because the gate below
    needs to see the RAW event sequence - the SDK skips `[DONE]`, and "did the
    terminal event actually come last" is one of the things being checked.

    NAMED APART from `sse_events` (the JSON-lane one) deliberately. It was
    called `sse_events` too, and since a module has one
    namespace the later definition silently won: every `sec_strict_chat`
    streaming assertion had been calling this with a URL path since the audio
    surface landed, and dying on `name, blob = wav`. Two gates were red for it.
    A multipart poster and a JSON poster do not share a name."""
    name, blob = wav
    data = {}
    for field, value in fields:
        data.setdefault(field, []).append(value)
    out, ctype = [], None
    with client.stream(
        "POST",
        "/v1/audio/transcriptions",
        files={"file": (name, blob, mimetypes.guess_type(name)[0] or "application/octet-stream")},
        data=data,
    ) as r:
        assert r.status_code == 200, f"stream: {r.status_code} {r.read()[:200]!r}"
        ctype = r.headers.get("content-type", "")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                out.append("[DONE]")
                continue
            out.append(json.loads(payload))
    assert ctype.startswith("text/event-stream"), f"streamed as {ctype!r}"
    return out


def sec_audio_stream(client, matrix, wav, model, base_url):
    """`stream=true`.

    The invariant under test is the one a client actually depends on: the
    deltas CONCATENATE to the final text, byte for byte. A live transcript that
    quietly disagrees with the final one is worse than no live transcript."""
    assert matrix["audio_transcriptions"]["stream"] == "implemented"

    events = audio_sse_events(client, wav, [("model", model), ("stream", "true")])
    assert events and events[-1] == "[DONE]", events[-1:]
    events = events[:-1]
    for e in events:
        assert "error" not in e, e
    assert events[-1]["type"] == "transcript.text.done", events[-1]["type"]
    deltas = [e for e in events[:-1] if e["type"] == "transcript.text.delta"]
    assert not [e for e in events[:-1] if e["type"] != "transcript.text.delta"], (
        f"unexpected event types: {sorted({e['type'] for e in events[:-1]})}"
    )
    # More than one, or "streaming" is a 200 in a costume. The fixture crosses
    # two 30 s encoder windows and decodes ~100 tokens, so a per-token stream
    # has plenty to say.
    assert len(deltas) > 1, f"only {len(deltas)} delta(s) - nothing was streamed"
    done = events[-1]
    joined = "".join(d["delta"] for d in deltas)
    assert joined == done["text"], (
        "deltas do not concatenate to the final text:\n"
        f"  deltas -> {joined[:200]!r}\n  done   -> {done['text'][:200]!r}"
    )
    assert done["usage"]["type"] == "tokens", done["usage"]
    assert done["usage"]["output_tokens"] > 0, done["usage"]
    assert done["usage"]["input_tokens"] > 0, done["usage"]
    assert done["languages"] and done["languages"][0]["code"], done.get("languages")

    # Streaming must not CHANGE the answer - same request, same decode.
    r = audio_post(client, wav, [("model", model)])
    assert r.status_code == 200, r.text[:200]
    assert done["text"] == r.json()["text"], (done["text"][:160], r.json()["text"][:160])

    # verbose_json + stream: deltas as usual, and the terminal event carries the
    # whole verbose body so nobody has to re-send the clip for its timings.
    # Whatever this lane times - the granularities it advertises, no more.
    ts_fields = [("response_format", "verbose_json")]
    ts_fields += [("timestamp_granularities[]", g) for g in audio_caps(client)]
    events = audio_sse_events(client, wav, [("model", model), ("stream", "true"), *ts_fields])
    assert events[-1] == "[DONE]"
    done = events[-2]
    assert done["type"] == "transcript.text.done", done["type"]
    joined = "".join(e["delta"] for e in events[:-2] if e["type"] == "transcript.text.delta")
    assert joined == done["text"], (joined[:200], done["text"][:200])
    # The granite lane earns this one twice over: its times are TEXT in the
    # model's own output, so a tag half-emitted at a delta boundary would put
    # bytes in the stream that the final text does not contain.
    r = audio_post(client, wav, [("model", model), *ts_fields])
    assert r.status_code == 200, r.text[:200]
    for name, check in AUDIO_STREAM_EXTENSION_CHECKS.items():
        assert name in matrix["_extensions"]["audio_transcriptions"], name
        check(done, r.json().get("segments"))

    # A subtitle DOCUMENT is not a partial answer, and the refusal says so by
    # name rather than dropping `stream` on the floor.
    for fmt in ("srt", "vtt"):
        r = audio_post(
            client, wav, [("model", model), ("stream", "true"), ("response_format", fmt)]
        )
        assert r.status_code == 400, f"stream+{fmt}: {r.status_code} {r.text[:200]}"
        assert fmt in r.json()["error"]["message"], r.text[:200]

    # And through the PINNED SDK, which is the arbiter of this wire format: it
    # decides how the form is posted AND which event classes the stream must
    # parse into. A server whose events the SDK cannot type is not conformant
    # however good the JSON looks by hand.
    import openai  # noqa: PLC0415
    from openai.types.audio import TranscriptionTextDeltaEvent, TranscriptionTextDoneEvent

    sdk = openai.OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="conformance")
    seen, text = [], ""
    for ev in sdk.audio.transcriptions.create(
        file=(wav[0], wav[1]), model=model, stream=True, temperature=0.0
    ):
        seen.append(type(ev))
        if isinstance(ev, TranscriptionTextDeltaEvent):
            text += ev.delta
        elif isinstance(ev, TranscriptionTextDoneEvent):
            assert text == ev.text, (text[:160], ev.text[:160])
    assert seen and seen[-1] is TranscriptionTextDoneEvent, seen[-3:]
    assert seen.count(TranscriptionTextDoneEvent) == 1, seen
    assert set(seen) == {TranscriptionTextDeltaEvent, TranscriptionTextDoneEvent}, set(seen)
    ok("audio streaming: transcript.text.delta/done")


def pcm16_of(wav_bytes, seconds):
    """The first `seconds` of a 16-bit mono WAV, as raw little-endian frames
    plus its rate - what `input_audio_buffer.append` carries (base64'd)."""
    import wave  # noqa: PLC0415

    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "fixture must be mono PCM16"
        rate = w.getframerate()
        return w.readframes(int(rate * seconds)), rate


def ext(model, key):
    """A `paddock_*` field off an SDK model.

    The OpenAI SDK's models keep unknown keys (`extra="allow"`), which is the
    whole reason our extensions are namespaced rather than shipped through a
    forked client - but where they surface has moved between versions, so read
    both places rather than pin a behaviour."""
    v = getattr(model, key, None)
    if v is None:
        v = (getattr(model, "model_extra", None) or {}).get(key)
    return v


def sec_audio_realtime_vad(base_url, model, wav, enriched):
    """`turn_detection: server_vad` - the session ends its own turn,
    and the closed turn comes back ENRICHED.

    The two belong in one leg because they are the same moment. A turn the
    server closed is the boundary where the richer decode is paid: the final
    pass over that utterance already runs, so it runs with timestamps and word
    timing and hands back the object the file endpoint's verbose_json would
    return for the same audio. That is what stops the live transcript being a
    second-class record - before it, a turn committed straight from this socket
    had text and nothing else, and the Studio re-transcribed the whole
    recording through the file endpoint to get times it could have had here.

    `enriched` is whether this lane can serve that at all. The generative
    families have no timestamp vocabulary, so there is no enriched
    pass for them to run - and this leg asserts they say so BY NAME rather than
    accepting the opt-in and answering with a plain item, which is the same
    distinction the `semantic_vad` refusal draws."""
    import openai  # noqa: PLC0415
    from openai.types.realtime import (  # noqa: PLC0415
        ConversationItemInputAudioTranscriptionCompletedEvent as Completed,
        InputAudioBufferCommittedEvent as Committed,
        InputAudioBufferSpeechStartedEvent as SpeechStarted,
        InputAudioBufferSpeechStoppedEvent as SpeechStopped,
        SessionUpdatedEvent as SessionUpdated,
    )

    frames, rate = pcm16_of(wav[1], 5.0)
    # a second of digital silence, which is what has to end the turn
    quiet = b"\x00\x00" * rate
    sdk = openai.OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="conformance")

    with sdk.realtime.connect(model=model, extra_query={"intent": "transcription"}) as conn:
        events, lock = [], threading.Lock()
        stop = threading.Event()

        def reader():
            try:
                for ev in conn:
                    with lock:
                        events.append(ev)
                    if isinstance(ev, Completed):
                        break
            except Exception:
                pass
            stop.set()

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        def wait_for(kind, timeout=60.0):
            end = time.monotonic() + timeout
            while True:
                with lock:
                    hit = [e for e in events if isinstance(e, kind)]
                if hit:
                    return hit
                if stop.is_set() or time.monotonic() > end:
                    with lock:
                        hit = [e for e in events if isinstance(e, kind)]
                    if hit:
                        return hit
                    raise AssertionError(f"no {kind.__name__}: {names(events)}")
                time.sleep(0.05)

        if not enriched:
            # Refused by NAME, and refused before anything is applied - a
            # session that quietly accepted this and then answered with a plain
            # item would be indistinguishable from a build too old to have it.
            conn.session.update(
                session={"type": "transcription",
                         "audio": {"input": {"transcription": {"paddock_verbose": True}}}}
            )
            e = wait_for(openai.types.realtime.RealtimeErrorEvent, 20.0)[0]
            assert "paddock_verbose" in e.error.message, e.error.message

        conn.session.update(
            session={
                "type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": rate},
                    "turn_detection": {"type": "server_vad", "silence_duration_ms": 400},
                    "transcription": {"language": "en", "paddock_verbose": enriched},
                }},
            }
        )
        updated = wait_for(SessionUpdated, 20.0)[0]
        td = updated.session.audio.input.turn_detection
        assert td is not None and td.type == "server_vad", updated.session.audio
        # the knobs come back as SET, not as defaults - a client that cannot
        # read back its own configuration cannot know what it is running
        assert td.silence_duration_ms == 400, td
        assert td.threshold == 0.5 and td.prefix_padding_ms == 300, td
        # same rule for the extension: a caller who asked and reads it back
        # missing cannot tell a refusal from an old build
        assert ext(updated.session.audio.input.transcription, "paddock_verbose") is enriched, (
            updated.session.audio.input.transcription
        )

        step = int(rate * 0.5) * 2
        for chunk in (frames, quiet):
            for i in range(0, len(chunk), step):
                conn.input_audio_buffer.append(
                    audio=base64.b64encode(chunk[i : i + step]).decode()
                )
                time.sleep(0.05)

        started = wait_for(SpeechStarted, 30.0)[0]
        stopped = wait_for(SpeechStopped, 30.0)[0]
        assert stopped.audio_end_ms > started.audio_start_ms, (started, stopped)
        # the turn ends where the SPEECH ended, not where the silence was
        # noticed: the clip is 5 s and the quiet after it is not part of it
        assert stopped.audio_end_ms <= 5500, stopped
        # nobody sent a commit - the server owes the acknowledgement anyway
        wait_for(Committed, 20.0)
        done = wait_for(Completed, 120.0)[0]

    assert done.transcript.strip(), "server VAD closed a turn with no transcript"
    ok(f"realtime server VAD: turn {started.audio_start_ms}-{stopped.audio_end_ms} ms, no client commit")

    # ---- the enriched utterance  ----
    if not enriched:
        assert ext(done, "paddock_verbose") is None, "a lane that refused the opt-in answered it"
        return
    v = ext(done, "paddock_verbose")
    assert v, f"the session asked for paddock_verbose and the closed item has none: {done}"
    # Where the item sits in the SESSION. Its own key rather than inferred from
    # speech_started, because under manual turns that event never fires and an
    # item nobody can place against the recording cannot be a row in a timeline.
    at = ext(done, "paddock_audio_start_ms")
    assert at is not None and at >= 0, done

    dur = v["duration"]
    assert dur > 0, v
    segs = v["segments"]
    assert segs, f"enriched utterance with no segments: {list(v)}"
    # `words` and not `paddock_words` is the whole point: the spec's key is
    # earned by having real times, and a lane that cannot time a word answers
    # under ours instead. Getting the namespaced one here would mean the DTW
    # pass silently did not run.
    words = v.get("words")
    assert words, f"no timed words on the closed utterance (got {list(v)})"

    # Self-consistency, which is what makes this object usable as a seek index:
    # times inside the utterance, in order, and ending where they start or later.
    for i, s in enumerate(segs):
        assert 0 <= s["start"] <= s["end"] <= dur + 0.5, (i, s["start"], s["end"], dur)
        if i:
            assert s["start"] >= segs[i - 1]["start"], (i, segs[i - 1], s)
    for i, w in enumerate(words):
        assert 0 <= w["start"] <= w["end"] <= dur + 0.5, (i, w)
        if i:
            assert w["start"] >= words[i - 1]["start"], (i, words[i - 1], w)
        assert 0.0 <= w["paddock_confidence"] <= 1.0, w
    # Joined with a space and re-split on both sides, because a `word` here is
    # a WORD - the grouping already trimmed the BPE pieces - so the separator
    # is not carried in the payload the way a delta stream carries it.
    assert " ".join(w["word"] for w in words).split() == v["text"].split(), (
        "the timed words do not spell the enriched transcript:\n"
        f"  words -> {' '.join(w['word'] for w in words)[:160]!r}\n"
        f"  text  -> {v['text'][:160]!r}"
    )

    # MEASURED, not asserted equal - and this is the honest cost of never
    # retracting a word. `transcript` is what the deltas showed: a committed
    # prefix decided from SHORTER buffers plus the final pass's tail. The
    # enriched object is the final pass's own reading of the whole utterance.
    # They usually agree and are allowed not to. The floor is here so a
    # regression that made them unrelated fails loudly, while ordinary
    # settling passes; the number is printed so the drift is visible.
    agree = difflib.SequenceMatcher(
        None, done.transcript.split(), v["text"].split()
    ).ratio()
    assert agree >= 0.5, (
        "the live transcript and the settled one barely overlap:\n"
        f"  live    -> {done.transcript[:160]!r}\n"
        f"  settled -> {v['text'][:160]!r}"
    )
    ok(
        f"enriched utterance: {len(segs)} segment(s), {len(words)} timed words, "
        f"item at {at} ms, live/settled agreement {agree:.0%}"
    )


def sec_audio_realtime(base_url, model, wav, clip_s):
    """`/v1/realtime?intent=transcription` - the microphone half.

    Driven through the SDK's own realtime client, for the same reason the file
    lane is: the SDK decides the URL, the event names and the payload shapes, so
    a session it cannot drive is not conformant however good the JSON looks."""
    import openai  # noqa: PLC0415
    from openai.types.realtime import (  # noqa: PLC0415
        ConversationItemInputAudioTranscriptionCompletedEvent as Completed,
        ConversationItemInputAudioTranscriptionDeltaEvent as Delta,
        InputAudioBufferCommittedEvent as Committed,
        SessionCreatedEvent,
        SessionUpdatedEvent as SessionUpdated,
    )

    # `clip_s` is 34 on the whisper lane on PURPOSE: a buffer that
    # outgrows whisper's 30 s window is what makes the session TRIM, retiring
    # the finished window's words and dropping its audio. That path re-bases
    # the committed/pending bookkeeping onto a shorter buffer, and the
    # delta-concatenation assertion at the end of this function is what proves
    # it did so without losing or duplicating a word. The generative lanes take
    # the whole buffer as one prompt and never trim, so they get a short clip.
    frames, rate = pcm16_of(wav[1], clip_s)
    sdk = openai.OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="conformance")

    with sdk.realtime.connect(model=model, extra_query={"intent": "transcription"}) as conn:
        events, lock = [], threading.Lock()
        stop = threading.Event()

        def reader():
            try:
                for ev in conn:
                    with lock:
                        events.append(ev)
                    if isinstance(ev, Completed):
                        break
            except Exception:  # the socket closing under us ends the read
                pass
            stop.set()

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        def wait_for(kind, timeout=120.0):
            end = time.monotonic() + timeout
            while True:
                with lock:
                    hit = [e for e in events if isinstance(e, kind)]
                if hit:
                    return hit
                # The reader sets `stop` after appending, so one re-read under
                # the lock once it is set sees everything it ever will - check
                # before deciding the stream ended without the event.
                if stop.is_set():
                    with lock:
                        hit = [e for e in events if isinstance(e, kind)]
                    if hit:
                        return hit
                    raise AssertionError(
                        f"stream ended before any {kind.__name__}: {names(events)}"
                    )
                if time.monotonic() > end:
                    raise AssertionError(f"timed out waiting for {kind.__name__}: {names(events)}")
                time.sleep(0.05)

        created = wait_for(SessionCreatedEvent, 20.0)[0]
        assert created.session.type == "transcription", created.session
        # `turn_detection: null` is the CONTRACT here, not an omission - a
        # client that reads the created session learns it must commit itself.
        assert created.session.audio.input.turn_detection is None, created.session.audio

        # Semantic turn detection is refused by NAME rather than silently
        # downgraded to the acoustic one, which is the difference between a gap
        # and a lie. (`server_vad` itself is served - its own leg drives it.)
        conn.session.update(
            session={"type": "transcription", "audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}}
        )
        e = wait_for(openai.types.realtime.RealtimeErrorEvent, 20.0)[0]
        assert "semantic_vad" in e.error.message, e.error.message
        assert "server_vad" in e.error.message, e.error.message

        # Force the language, for the comparison at the end. Detection is not
        # comparable between the two lanes and that is not a bug: the session
        # locks the language on its first pass (a second or two of speech) so
        # the transcript cannot change language mid-utterance, while the file
        # lane decides from a whole 30 s window. Measured on this
        # fixture, the two disagree - `en` live, `sv` (translated!) from the
        # file - which is exactly why the session says which one it picked.
        # Declare the fixture's rate AND force the language. Both matter:
        # OpenAI's realtime input is defined at 24 kHz and that is what the
        # session assumes, so a 16 kHz clip sent without saying so is resampled
        # as if it were 24 kHz and the model hears it 1.5x fast (measured here
        # - it still transcribed, which is exactly why an undeclared
        # rate has to be a client mistake the test makes visible rather than a
        # server guess that usually works).
        conn.session.update(
            session={
                "type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": rate},
                    "transcription": {"language": "en"},
                }},
            }
        )
        updated = wait_for(SessionUpdated, 20.0)[0]
        assert updated.session.audio.input.transcription.language == "en", updated.session
        assert updated.session.audio.input.format.rate == rate, updated.session.audio

        # Feed it faster than real time but in chunks, so passes genuinely
        # overlap the upload - which is the thing being tested.
        step = int(rate * 1.0) * 2  # one second of 16-bit frames
        for i in range(0, len(frames), step):
            conn.input_audio_buffer.append(
                audio=base64.b64encode(frames[i : i + step]).decode()
            )
            time.sleep(0.12)
        # The claim is live text, so at least one delta must land before the
        # commit. Without this the whole session could be a slow POST.
        early = wait_for(Delta, 90.0)
        assert early, "no partial transcript arrived before the commit"

        conn.input_audio_buffer.commit()
        wait_for(Committed, 20.0)
        done = wait_for(Completed, 180.0)[0]

    with lock:
        seq = list(events)
    deltas = [e for e in seq if isinstance(e, Delta)]
    assert len(deltas) > 1, f"only {len(deltas)} delta(s): {names(seq)}"
    joined = "".join(d.delta or "" for d in deltas)
    assert joined == done.transcript, (
        "realtime deltas do not concatenate to the completed transcript:\n"
        f"  deltas -> {joined[:200]!r}\n  done   -> {done.transcript[:200]!r}"
    )
    assert done.transcript.strip(), "empty realtime transcript"
    # A committed word is never taken back, so every delta must be a suffix of
    # the final text at the point it was sent - the one property LocalAgreement-2
    # exists to provide.
    assert done.transcript.startswith(deltas[0].delta or ""), (deltas[0].delta, done.transcript[:80])

    # Not asserted: that the completed transcript equals a one-shot
    # transcription of the same audio. It cannot be, and the reason is the
    # policy rather than a defect. `completed` is the committed prefix plus the
    # final pass's tail, and the committed prefix was decided from SHORTER
    # buffers - so when the model's hypothesis changes as the buffer grows, the
    # live answer keeps what it already showed. Never retracting a word is what
    # a delta stream buys, and this is what it costs.
    #
    # Measured, and a good illustration: kb-whisper-large is a
    # Swedish fine-tune, this fixture is English LibriSpeech, and the model
    # transcribes the first seconds as English then flips to translating into
    # Swedish once it has more audio. The one-shot answer is all Swedish; the
    # live one is English then Swedish, faithfully showing what was committed
    # when. Forcing `language=en` does not change it - on this checkpoint the
    # language token does not move the argmax, which is a property of a
    # single-language fine-tune, not of the lane.
    ok(f"realtime transcription session ({len(deltas)} partials, LocalAgreement-2)")


def names(events):
    return [getattr(e, "type", type(e).__name__) for e in events]


def audio_caps(client):
    """The granularities the SERVER says it can answer.

    Every expectation below is derived from this rather than hardcoded to
    whisper's, because the lanes answer different granularities by unrelated
    mechanisms: whisper has a timestamp VOCABULARY (segments) and
    cross-attention alignment (words); granite-speech-plus is INSTRUCTED to
    write word times into its transcript and has no punctuation in that mode,
    so no sentence boundaries to cut segments on; the rest answer neither. A
    gate that can only be pointed at one lane leaves the others uncovered,
    which is exactly where a silent regression lives - and reading the list
    here also pins the list itself, since a wrong advertisement now fails on
    the assertions it selected."""
    entry = client.get("/v1/models").json()["data"][0]
    caps = entry.get("capabilities", {}).get("timestamp_granularities")
    assert isinstance(caps, list), f"/v1/models publishes no granularity list: {entry!r}"
    assert set(caps) <= {"segment", "word"}, f"unknown granularity advertised: {caps!r}"
    return caps


def language_caps(client):
    """What the SERVER says it can do about language, read the same
    way the granularities are and for the same reason: which lane is loaded
    decides what the assertions below may demand, and reading the list pins the
    advertisement itself."""
    entry = client.get("/v1/models").json()["data"][0]
    caps = entry.get("capabilities", {}).get("language_detection")
    assert isinstance(caps, dict), f"/v1/models publishes no language_detection: {entry!r}"
    for k in ("supported", "probabilities", "hints"):
        assert isinstance(caps[k], bool), caps
    assert isinstance(caps["languages"], list), caps
    # a lane that can be HINTED must say which codes it accepts, or the hint is
    # a field a client can only get wrong
    if caps["hints"]:
        assert caps["languages"], "hints advertised with no language list to hint from"
    return caps


def sec_audio_language(client, wav, model):
    """LANGUAGE: the soft candidate set, the checkpoint's own map,
    and the contradiction check.

    Every assertion is derived from `language_detection` rather than from which
    checkpoint happens to be loaded - a lane with no detection to bias must
    REFUSE `languages`, and a lane that has one must serve it."""
    caps = language_caps(client)
    base = [("model", model), ("response_format", "verbose_json")]
    body = audio_post(client, wav, base).json()
    lang = check_paddock_language(body, caps)

    # An unknown code is refused on every lane, and by NAME: on a hinting lane
    # because it is not in the checkpoint's map, elsewhere because there is no
    # detection to bias at all.
    r = audio_post(client, wav, [*base, ("languages[]", "xx")])
    assert r.status_code == 400, r.text[:200]
    # and it names what it refused: the CODE where codes are read at all, the
    # PARAMETER on a lane that refuses every value of it unread
    assert ("xx" if caps["hints"] else "languages") in r.text, r.text[:200]

    if caps["hints"]:
        # the two ways of answering the same question are mutually exclusive
        picks = caps["languages"][:2]
        r = audio_post(client, wav, [*base, ("language", picks[0]), ("languages[]", picks[1])])
        assert r.status_code == 400, r.text[:200]
        assert "languages" in r.text, r.text[:200]
        # ...and the hint alone is SERVED, with the set echoed back so a caller
        # can see the prior that shaped their answer
        r = audio_post(client, wav, [*base, *[("languages[]", p) for p in picks]])
        assert r.status_code == 200, r.text[:200]
        hinted = check_paddock_language(r.json(), caps)
        assert hinted["hints"]["languages"] == picks, hinted
        assert 0.0 < hinted["hints"]["strength"] < 1.0, (
            "a prior at 0 does nothing and at 1 is a filter wearing a hint's name"
        )
        assert hinted["source"] == "detected", hinted
    else:
        r = audio_post(client, wav, [*base, ("languages[]", (caps["languages"] or ["en"])[0])])
        assert r.status_code == 400, r.text[:200]

    # A forced language is reflected back as an INSTRUCTION, never as a
    # measurement - the distinction the whole object exists to make.
    if caps["languages"]:
        forced = caps["languages"][0]
        forced_body = audio_post(client, wav, [*base, ("language", forced)]).json()
        got = check_paddock_language(forced_body, caps)
        assert got["source"] == "asked" and got["code"] == forced, got
    print(
        f"  language: source={lang and lang['source']}, hints={caps['hints']}, "
        f"{len(caps['languages'])} code(s) published",
        flush=True,
    )
    ok("audio language identification")


def sec_strict_audio(client, matrix, wav, model, base_url):
    caps = audio_caps(client)
    r = audio_post(client, wav, [("model", model)])
    assert r.status_code == 200, r.text[:200]
    plain = Transcription.model_validate(r.json())
    assert plain.text.strip(), "empty transcript"

    r = audio_post(client, wav, [("model", model), ("response_format", "text")])
    assert r.status_code == 200, r.text[:200]
    assert r.text.strip() == plain.text.strip(), (r.text[:200], plain.text[:200])

    # A granularity this lane does not serve is refused by NAME, never with an
    # empty array - the whole reason the capability list exists.
    for gran in sorted({"segment", "word"} - set(caps)):
        r = audio_post(client, wav, [
            ("model", model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", gran),
        ])
        assert r.status_code == 400, f"{gran} is not advertised but was served: {r.text[:200]}"
        assert gran in r.json()["error"]["message"], r.text[:300]

    # `prompt` is served on every lane now, and it means different
    # things on each: whisper feeds it behind `<|startofprev|>` as a spelling
    # and vocabulary hint, the generative lanes carry it as instruction
    # context. What is common - and all a spec client can rely on - is that it
    # is accepted and still produces a transcript. Not asserted: that the text
    # matches the unprompted one. Biasing the decode is the entire point of
    # the field, so a difference is the feature working.
    r = audio_post(client, wav, [
        ("model", model),
        ("prompt", "LibriSpeech, Stephane, Cauchy, Poincare"),
    ])
    assert r.status_code == 200, r.text[:200]
    assert Transcription.model_validate(r.json()).text.strip(), "prompt produced an empty transcript"
    # A control token smuggled in as text would rewrite the decode contract
    # from inside a field that is supposed to be a hint. Whisper is the lane
    # whose tokenizer has those tokens, so it is the lane that must refuse.
    if matrix["audio_transcriptions"]["prompt"] == "implemented" and "segment" in caps:
        r = audio_post(client, wav, [("model", model), ("prompt", "<|notimestamps|>")])
        assert r.status_code == 400, f"a control token in `prompt` was served: {r.text[:200]}"
        assert "control token" in r.json()["error"]["message"], r.text[:300]

    segments, words = [], []
    if "segment" in caps:
        segments, verbose = sec_strict_audio_segments(client, matrix, wav, model)
    if "word" in caps:
        words = sec_strict_audio_words(client, wav, model, caps)
    if "segment" in caps:
        sec_strict_audio_subtitles(client, wav, model, segments)
    else:
        # No segments means no subtitle file, and the refusal says which half
        # is missing rather than emitting one cue over the whole clip.
        for fmt in ("srt", "vtt"):
            r = audio_post(client, wav, [("model", model), ("response_format", fmt)])
            assert r.status_code == 400, f"{fmt} without segments: {r.text[:200]}"
            assert "subtitle" in r.json()["error"]["message"], r.text[:300]

    # One round trip through the PINNED SDK's own client, because the SDK is
    # the arbiter of the wire format here as everywhere else in this file: it
    # decides how the form is encoded, and in particular how the repeated
    # `timestamp_granularities[]` part is spelled. Everything above posts the
    # form by hand, so this is the check that our parser agrees with the
    # client callers actually use.
    import openai  # noqa: PLC0415

    sdk = openai.OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="conformance")
    via_sdk = sdk.audio.transcriptions.create(
        file=(wav[0], wav[1]),
        model=model,
        response_format="verbose_json",
        timestamp_granularities=caps,
        temperature=0.0,
    )
    assert isinstance(via_sdk, TranscriptionVerbose), type(via_sdk)
    if "segment" in caps:
        assert len(via_sdk.segments or []) == len(segments), (
            f"SDK client saw {len(via_sdk.segments or [])} segments, raw form saw {len(segments)}"
        )
    if "word" in caps:
        assert len(via_sdk.words or []) == len(words), (
            f"SDK client saw {len(via_sdk.words or [])} words, raw form saw {len(words)}"
        )

    # The capability list is computed from what actually works, and it is what
    # a caller reads before sending a clip, so it has to agree with the matrix.
    assert caps, "this server transcribes but advertises no granularity at all"
    assert matrix["audio_transcriptions"]["timestamp_granularities"] == "implemented", (
        "the matrix and /v1/models disagree about timestamp granularities"
    )
    ok(
        f"strict validation: audio transcriptions ({'+'.join(caps)}; "
        f"{len(segments)} segments, {len(words)} word times)"
    )


def sec_strict_audio_segments(client, matrix, wav, model):
    r = audio_post(client, wav, [
        ("model", model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
    ])
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    verbose = TranscriptionVerbose.model_validate(body)
    # Not asserted: that this text equals the plain one. Asking for times drops
    # `<|notimestamps|>` from the prompt and constrains the logits, so it is a
    # different decode and the words are allowed to differ - measured here as
    # "and mits" vs "amits" on the fixture. What is invariant is
    # that no timestamp token leaks into `text`, which the segment rejoin below
    # pins exactly.
    assert verbose.text.strip(), "empty transcript with timestamps on"
    assert "<|" not in verbose.text, f"timestamp token leaked into text: {verbose.text[:200]}"
    assert verbose.duration > 0, verbose.duration
    assert verbose.language, "no detected language"

    segments = body["segments"]
    assert segments, "verbose_json asked for segments and got none"
    prev_end = 0.0
    for i, seg in enumerate(segments):
        assert seg["id"] == i, f"segment ids must be 0..n: {seg['id']} at {i}"
        assert seg["start"] <= seg["end"], seg
        assert seg["start"] >= prev_end - 1e-6, f"segment {i} starts before {prev_end}: {seg}"
        assert seg["end"] <= verbose.duration + 1e-6, (
            f"segment {i} ends at {seg['end']} past the clip's {verbose.duration}"
        )
        assert seg["text"].strip(), f"segment {i} is blank"
        assert 0.0 <= seg["no_speech_prob"] <= 1.0, seg
        assert seg["avg_logprob"] <= 0.0, seg
        prev_end = seg["end"]
    joined = " ".join(" ".join(s["text"].split()) for s in segments)
    assert joined == " ".join(verbose.text.split()), (joined[:200], verbose.text[:200])

    for name, check in AUDIO_EXTENSION_CHECKS.items():
        assert name in matrix["_extensions"]["audio_transcriptions"], name
        check(segments)
    return segments, verbose


def sec_strict_audio_words(client, wav, model, caps):
    """Word times  - the other granularity, and never the same
    mechanism as the segment times. On whisper they come from cross-attention
    over a second, teacher-forced pass and the transcript is untouched; on
    granite-speech-plus the model is asked for them and writes `[T:N]` tags
    into its answer, which the runner parses back out. What both must satisfy
    is everything below, which is why it is one block."""
    # Both granularities at once where the lane has both: that is the shape
    # OpenAI documents and the one a caller actually sends.
    fields = [("model", model), ("response_format", "verbose_json")]
    fields += [("timestamp_granularities[]", g) for g in caps]
    r = audio_post(client, wav, fields)
    assert r.status_code == 200, r.text[:200]
    wbody = r.json()
    wverbose = TranscriptionVerbose.model_validate(wbody)
    # A word with no time is not a spec word, so the runner ships the array as
    # `paddock_words` instead - which means the interesting failure is not
    # "empty", it is "landed under the other key". Say which, and say what the
    # decode guards thought, or this reads as a bare KeyError.
    words = wbody.get("words")
    assert words, (
        "asked for `word` granularity and got no spec `words` array - "
        f"paddock_words: {len(wbody.get('paddock_words') or [])}, "
        f"guards: {wbody.get('paddock_guards')}, text: {wbody.get('text', '')[-80:]!r}"
    )
    assert wverbose.words and len(wverbose.words) == len(words), (
        "the SDK does not see the words as words"
    )
    prev = 0.0
    for i, w in enumerate(words):
        assert w["word"].strip() == w["word"] and w["word"], f"untrimmed word: {w!r}"
        assert w["start"] <= w["end"], w
        # Both mechanisms are monotone - DTW walks a monotone path, and the
        # tag clock is un-wrapped by monotonicity - so a word starting before
        # the previous one ended means the times were read in the wrong order.
        # That is the exact shape of whisper's row off-by-one and of a missed
        # granite rollover, and neither shows up any other way.
        assert w["start"] >= prev - 1e-6, f"word {i} starts at {w['start']}, before {prev}: {w!r}"
        assert w["end"] <= wverbose.duration + 1e-6, (
            f"word {i} ends at {w['end']} past the clip's {wverbose.duration}"
        )
        prev = w["start"]
        # our figures, under `paddock_` names because this container is SPEC
        assert 0.0 < w["paddock_confidence"] <= 1.0, w
        assert abs(w["paddock_confidence"] - math.exp(w["paddock_logprob"])) < 1e-4, w
    # The last word must land near the END of the clip, not near the start. A
    # granite lane that missed every 10 s rollover still passes everything
    # above - monotone, in range, rejoining - while packing 46 s of speech into
    # the first ten. Nothing else here can see that.
    assert words[-1]["end"] > wverbose.duration * 0.5, (
        f"the last word ends at {words[-1]['end']} of a {wverbose.duration}s clip - "
        "the times cover a fraction of the audio"
    )
    # The words are the transcript, not a summary of it. This is also what pins
    # that the timing pass and the word grouping saw the same token stream: a
    # disagreement there drops or duplicates words rather than shifting times.
    # (On the granite lane it pins one more thing - that no `[T:N]` tag and no
    # `_` silence marker survived into either the words or the text.)
    rejoined = " ".join(w["word"] for w in words)
    assert rejoined == " ".join(wverbose.text.split()), (rejoined[:200], wverbose.text[:200])
    # and the per-segment copy RETIRES when the real words ship - the same
    # confidences in two places is two places to disagree
    for seg in wbody.get("segments", []):
        assert "paddock_words" not in seg, (
            f"segment {seg['id']} still carries paddock_words alongside a real `words` array"
        )
    return words


def sec_strict_audio_subtitles(client, wav, model, segments):
    """Subtitles: a subtitle file is timestamps, so neither asks for the
    granularity. Escaping and the rolling-stamp cases are pinned by
    subtitles.rs's own tests; what only a live clip can show is that the cues
    are the segments."""
    for fmt, sep, numbered in (("srt", ",", True), ("vtt", ".", False)):
        r = audio_post(client, wav, [("model", model), ("response_format", fmt)])
        assert r.status_code == 200, r.text[:200]
        doc = r.text
        if fmt == "vtt":
            assert doc.startswith("WEBVTT\n\n"), repr(doc[:32])
            doc = doc[len("WEBVTT\n\n"):]
        cues = parse_cues(doc, sep, numbered)
        assert len(cues) == len(segments), f"{fmt}: {len(cues)} cues vs {len(segments)} segments"
        for i, ((n, start, end, text), seg) in enumerate(zip(cues, segments), 1):
            if numbered:
                assert n == i, f"{fmt}: cue numbered {n} at position {i}"
            assert abs(start - seg["start"]) < 0.001, (fmt, start, seg["start"])
            assert abs(end - seg["end"]) < 0.001, (fmt, end, seg["end"])
            assert text == " ".join(seg["text"].split()), (fmt, text, seg["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dialect", choices=["qwen", "harmony"], default="qwen")
    ap.add_argument(
        "--audio-only",
        action="store_true",
        help="the server under test serves /v1/audio/transcriptions and nothing "
             "else (a whisper family model has no chat surface at all)",
    )
    ap.add_argument("--audio-file", default=str(AUDIO_FIXTURE))
    ap.add_argument(
        "--images-only",
        action="store_true",
        help="the server under test serves /v1/images/* and nothing else (an "
             "image-generation model has no text surface at all)",
    )
    args = ap.parse_args()

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    sec_version_currency(matrix)
    client = httpx.Client(base_url=args.base_url, timeout=600.0)

    # Static, and therefore unconditional: this is the half that fails when the
    # SDK grows a param, and it must not depend on the box owning a checkpoint.
    sec_audio_matrix(matrix)
    sec_translation_matrix(matrix)
    sec_images_matrix(matrix)

    if args.images_only:
        sec_surface_audit(client, images_family(args.model), matrix)
        sec_strict_images(client, matrix, args.model, args.base_url)
        sec_images_stream(client, matrix, args.model, args.base_url)
        sec_images_edits(client, matrix, args.model, args.base_url)
    elif args.audio_only:
        clip = pathlib.Path(args.audio_file)
        wav = (clip.name, clip.read_bytes())
        sec_audio_surface(client, matrix, wav, args.model)
        sec_strict_audio(client, matrix, wav, args.model, args.base_url)
        sec_audio_stream(client, matrix, wav, args.model, args.base_url)
        sec_audio_guards(client, wav, args.model)
        sec_audio_language(client, wav, args.model)
        entry = client.get("/v1/models").json()["data"][0]
        # Whisper is the lane that WINDOWS, which is what the long clip is for.
        # Everything that transcribes serves the socket now.
        windows = entry.get("architecture", {}).get("input_modalities") == ["audio"]
        sec_audio_realtime(args.base_url, args.model, wav, 34.0 if windows else 12.0)
        # The enriched utterance rides whisper's timestamp vocabulary, so the
        # same flag that says "this lane windows" says whether to expect it.
        sec_audio_realtime_vad(args.base_url, args.model, wav, windows)
    else:
        print(
            "  audio_transcriptions probes NOT RUN here: they need a loaded ASR "
            "model - sdk_conformance.rs::spec_gate_whisper runs them --audio-only",
            flush=True,
        )
        print(
            "  images probes NOT RUN here: they need a loaded image model - "
            "sdk_conformance.rs::spec_gate_qwen_image runs them --images-only",
            flush=True,
        )
        for fam in families(args.model):
            sec_surface_audit(client, fam, matrix)
        sec_nested_shapes(client, args.model, matrix)
        sec_sampling_ranges(client, args.model)
        sec_strict_chat(client, args.model, args.dialect)
        sec_strict_completions(client, args.model)
        sec_strict_responses(client, args.model)
        sec_strict_anthropic(client, args.model)

    import anthropic
    import openai

    print(
        f"SPEC CONFORMANCE PASS ({SECTIONS} sections; openai {openai.__version__}, "
        f"anthropic {anthropic.__version__})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
