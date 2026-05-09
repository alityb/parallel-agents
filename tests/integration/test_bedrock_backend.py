"""Integration test: Bedrock backend adapter with a mocked boto3 client.

Verifies:
1. URL parsing — region, model_id_override, both, neither
2. warm_prefix — returns stable hash, no AWS call (Bedrock is managed)
3. generate (streaming) — tool call parsing from Bedrock event stream format
4. generate (non-streaming converse) — tool call parsing from converse() response
5. cachePoint injection — present for Claude, absent for Llama/Titan
6. Credential chain — adapter uses the injected factory; no real AWS call
7. Malformed tool blocks — logged and returned as error=True, not raised
8. Tool schema conversion — Anthropic format → Bedrock toolSpec format
9. Message format conversion — assistant_raw, tool_result → Bedrock format
10. Fallback to converse() when converse_stream raises streaming-unsupported error
"""
from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

from batch_agent.backends.bedrock import (
    BedrockBackend,
    _convert_tools_to_bedrock,
    _messages_to_bedrock,
    _parse_bedrock_tool_blocks,
    _supports_prompt_caching,
    _supports_streaming,
)
from batch_agent.backends import ParsedToolCall
from batch_agent.spec import AgentJob, Message, SharedContext


# ── helpers ────────────────────────────────────────────────────────────────────

def make_job(prompt: str = "Say hello.", idx: int = 0) -> AgentJob:
    return AgentJob(
        job_id=f"job-{idx}",
        index=idx,
        input_data={},
        prompt=prompt,
        estimated_prompt_tokens=10,
    )


def make_mock_client(stream_events: list[dict] | None = None, converse_response: dict | None = None) -> MagicMock:
    client = MagicMock()

    if stream_events is not None:
        client.converse_stream.return_value = {"stream": iter(stream_events)}

    if converse_response is not None:
        client.converse.return_value = converse_response

    return client


# ── 1. URL parsing ─────────────────────────────────────────────────────────────

def test_url_parsing_region_and_model():
    b = BedrockBackend.from_url("bedrock://us-east-1/anthropic.claude-sonnet-4-5")
    assert b.region == "us-east-1"
    assert b.model_id_override == "anthropic.claude-sonnet-4-5"
    print("[PASS] URL parsing: region + model")


def test_url_parsing_model_only():
    b = BedrockBackend.from_url("bedrock://anthropic.claude-sonnet-4-5")
    assert b.region is None
    assert b.model_id_override == "anthropic.claude-sonnet-4-5"
    print("[PASS] URL parsing: model only (contains dot)")


def test_url_parsing_region_only():
    b = BedrockBackend.from_url("bedrock://us-west-2")
    assert b.region == "us-west-2"
    assert b.model_id_override is None
    print("[PASS] URL parsing: region only")


def test_url_parsing_empty():
    b = BedrockBackend.from_url("bedrock://")
    assert b.region is None
    assert b.model_id_override is None
    print("[PASS] URL parsing: empty (defaults from env/config)")


# ── 2. warm_prefix ─────────────────────────────────────────────────────────────

def test_warm_prefix_returns_hash_no_aws_call():
    client = make_mock_client()
    backend = BedrockBackend(_client_factory=lambda: client)
    shared = SharedContext(prefix="You are a helpful assistant.")

    result = asyncio.run(backend.warm_prefix(shared, model="anthropic.claude-sonnet-4-5"))

    assert result is not None
    assert len(result) == 64  # sha256 hex
    client.converse_stream.assert_not_called()
    client.converse.assert_not_called()
    print("[PASS] warm_prefix returns stable hash, no AWS call")


def test_warm_prefix_empty_returns_none():
    backend = BedrockBackend()
    result = asyncio.run(backend.warm_prefix(SharedContext(prefix=""), model="any"))
    assert result is None
    print("[PASS] warm_prefix with empty prefix returns None")


# ── 3. generate (streaming) with text response ─────────────────────────────────

def test_generate_stream_simple_text():
    stream_events = [
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello "}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "world!"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    client = make_mock_client(stream_events=stream_events)
    backend = BedrockBackend(_client_factory=lambda: client)
    shared = SharedContext(prefix="")
    job = make_job("Say hello.")

    response = asyncio.run(backend.generate(
        shared=shared, job=job, model="anthropic.claude-sonnet-4-5", timeout=10,
    ))

    assert response.content == "Hello world!"
    assert response.tool_calls == []
    assert response.is_final is True
    assert response.stop_reason == "end_turn"
    print("[PASS] generate (stream): simple text response")


# ── 4. generate (streaming) with tool calls ────────────────────────────────────

def test_generate_stream_tool_call():
    stream_events = [
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Fetching..."}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {
            "toolUse": {"toolUseId": "call_abc", "name": "http_get"}
        }}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {
            "toolUse": {"input": '{"url": "http://example.com"}'}
        }}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    client = make_mock_client(stream_events=stream_events)
    backend = BedrockBackend(_client_factory=lambda: client)
    shared = SharedContext(prefix="")
    job = make_job("Fetch example.com")

    response = asyncio.run(backend.generate(
        shared=shared, job=job, model="anthropic.claude-sonnet-4-5", timeout=10,
    ))

    assert response.content == "Fetching..."
    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "http_get"
    assert tc.args == {"url": "http://example.com"}
    assert tc.error is False
    assert response.is_final is False
    assert response.stop_reason == "tool_use"
    print("[PASS] generate (stream): tool call parsed correctly")


# ── 5. generate (converse) with tool calls ─────────────────────────────────────

def test_generate_converse_tool_call():
    converse_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Let me look that up."},
                    {"toolUse": {
                        "toolUseId": "call_xyz",
                        "name": "web_search",
                        "input": {"query": "python asyncio"},
                    }},
                ],
            }
        },
        "stopReason": "tool_use",
    }
    # Force non-streaming path by using a model that doesn't support streaming
    # (we pass a client that has no converse_stream → will call converse)
    client = make_mock_client(converse_response=converse_response)
    # Make converse_stream raise so we fall back
    client.converse_stream.side_effect = Exception("streaming not supported")

    backend = BedrockBackend(_client_factory=lambda: client)
    shared = SharedContext(prefix="")
    job = make_job("Search for asyncio")

    response = asyncio.run(backend.generate(
        shared=shared, job=job, model="anthropic.claude-sonnet-4-5", timeout=10,
    ))

    assert "look that up" in response.content
    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc.id == "call_xyz"
    assert tc.name == "web_search"
    assert tc.args == {"query": "python asyncio"}
    assert tc.error is False
    print("[PASS] generate (converse fallback): tool call parsed correctly")


# ── 6. cachePoint injection ────────────────────────────────────────────────────

def test_cache_point_injected_for_claude():
    stream_events = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ok"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    client = make_mock_client(stream_events=stream_events)
    backend = BedrockBackend(_client_factory=lambda: client)
    shared = SharedContext(prefix="You are an expert.")
    job = make_job("Summarize.")

    asyncio.run(backend.generate(
        shared=shared, job=job, model="anthropic.claude-sonnet-4-5", timeout=10,
    ))

    call_kwargs = client.converse_stream.call_args[1]  # keyword args
    system = call_kwargs["system"]
    assert len(system) == 2
    assert system[0] == {"text": "You are an expert."}
    assert system[1] == {"cachePoint": {"type": "default"}}
    print("[PASS] cachePoint injected for anthropic.* model")


def test_cache_point_not_injected_for_llama():
    stream_events = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ok"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    client = make_mock_client(stream_events=stream_events)
    backend = BedrockBackend(_client_factory=lambda: client)
    shared = SharedContext(prefix="You are an expert.")
    job = make_job("Summarize.")

    asyncio.run(backend.generate(
        shared=shared, job=job, model="meta.llama3-8b-instruct-v1:0", timeout=10,
    ))

    call_kwargs = client.converse_stream.call_args[1]
    system = call_kwargs["system"]
    assert len(system) == 1  # only the text block, no cachePoint
    assert "cachePoint" not in str(system)
    print("[PASS] cachePoint NOT injected for meta.llama* model")


# ── 7. credential chain — injectable factory ───────────────────────────────────

def test_credential_chain_uses_factory():
    factory_calls = []

    def counting_factory():
        client = make_mock_client(stream_events=[
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ])
        factory_calls.append(1)
        return client

    backend = BedrockBackend(_client_factory=counting_factory)
    shared = SharedContext(prefix="")
    job = make_job("Hello")

    asyncio.run(backend.generate(shared=shared, job=job, model="anthropic.claude-sonnet-4-5", timeout=10))
    assert len(factory_calls) == 1
    print("[PASS] Credential chain: factory called once per generate() call")


# ── 8. malformed tool blocks ───────────────────────────────────────────────────

def test_malformed_tool_block_missing_id():
    result = _parse_bedrock_tool_blocks({
        0: {"toolUseId": "", "name": "http_get", "input_json": '{"url": "x"}'}
    })
    assert len(result) == 1
    assert result[0].error is True
    assert "toolUseId" in result[0].error_message
    print("[PASS] Malformed tool block (missing ID): error=True, not skipped")


def test_malformed_tool_block_bad_json():
    result = _parse_bedrock_tool_blocks({
        0: {"toolUseId": "abc", "name": "http_get", "input_json": '{"url": broken'}
    })
    assert len(result) == 1
    assert result[0].error is True
    assert "JSON" in result[0].error_message
    print("[PASS] Malformed tool block (bad JSON): error=True, not skipped")


# ── 9. tool schema conversion ──────────────────────────────────────────────────

def test_tool_schema_conversion():
    anthropic_tools = [{
        "name": "http_get",
        "description": "Fetch a URL",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    }]
    bedrock_tools = _convert_tools_to_bedrock(anthropic_tools)
    assert len(bedrock_tools) == 1
    spec = bedrock_tools[0]["toolSpec"]
    assert spec["name"] == "http_get"
    assert spec["description"] == "Fetch a URL"
    # Bedrock wraps schema under inputSchema.json
    assert spec["inputSchema"]["json"]["type"] == "object"
    assert "url" in spec["inputSchema"]["json"]["properties"]
    print("[PASS] Tool schema conversion: Anthropic → Bedrock toolSpec")


# ── 10. message format conversion ──────────────────────────────────────────────

def test_message_conversion_assistant_raw():
    messages = [
        Message(role="user", content="Fetch example.com"),
        Message(role="assistant_raw", content=json.dumps([
            {"type": "text", "text": "Fetching..."},
            {"type": "tool_use", "id": "call_1", "name": "http_get", "input": {"url": "http://example.com"}},
        ])),
        Message(role="tool_result", content=json.dumps([
            {"type": "tool_result", "tool_use_id": "call_1", "content": "<html>Example</html>"},
        ])),
    ]
    bedrock_msgs = _messages_to_bedrock(messages)

    # user message
    assert bedrock_msgs[0]["role"] == "user"
    assert bedrock_msgs[0]["content"][0]["text"] == "Fetch example.com"

    # assistant with tool_use
    assert bedrock_msgs[1]["role"] == "assistant"
    assert bedrock_msgs[1]["content"][0] == {"text": "Fetching..."}
    tool_use_block = bedrock_msgs[1]["content"][1]
    assert tool_use_block["toolUse"]["toolUseId"] == "call_1"
    assert tool_use_block["toolUse"]["name"] == "http_get"
    assert tool_use_block["toolUse"]["input"] == {"url": "http://example.com"}

    # tool result as user message with toolResult blocks
    assert bedrock_msgs[2]["role"] == "user"
    tr = bedrock_msgs[2]["content"][0]["toolResult"]
    assert tr["toolUseId"] == "call_1"
    assert tr["content"][0]["text"] == "<html>Example</html>"

    print("[PASS] Message conversion: assistant_raw + tool_result → Bedrock format")


def test_message_conversion_skips_blank_assistant_text_blocks():
    messages = [
        Message(role="assistant", content=""),
        Message(role="assistant_raw", content=json.dumps([
            {"type": "text", "text": ""},
            {"type": "tool_use", "id": "call_1", "name": "http_get", "input": {"url": "http://example.com"}},
        ])),
    ]

    bedrock_msgs = _messages_to_bedrock(messages)

    assert len(bedrock_msgs) == 1
    assert bedrock_msgs[0]["role"] == "assistant"
    assert bedrock_msgs[0]["content"] == [{
        "toolUse": {
            "toolUseId": "call_1",
            "name": "http_get",
            "input": {"url": "http://example.com"},
        }
    }]


# ── 11. supports_prompt_caching per model ──────────────────────────────────────

def test_prompt_caching_support_table():
    assert _supports_prompt_caching("anthropic.claude-sonnet-4-5") is True
    assert _supports_prompt_caching("anthropic.claude-3-haiku-20240307-v1:0") is True
    assert _supports_prompt_caching("meta.llama3-8b-instruct-v1:0") is False
    assert _supports_prompt_caching("amazon.titan-text-express-v1") is False
    assert _supports_prompt_caching("mistral.mistral-7b-instruct-v0:2") is False
    print("[PASS] _supports_prompt_caching: correct per-vendor table")


# ── 12. send_prefetch_hints is a no-op ─────────────────────────────────────────

def test_send_prefetch_hints_noop():
    backend = BedrockBackend()
    # Should not raise
    asyncio.run(backend.send_prefetch_hints([
        {"kv_key": "abc", "priority": 1.0, "eta_seconds": 0.5}
    ]))
    print("[PASS] send_prefetch_hints is a no-op (Bedrock managed service)")


# ── run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_url_parsing_region_and_model()
    test_url_parsing_model_only()
    test_url_parsing_region_only()
    test_url_parsing_empty()
    test_warm_prefix_returns_hash_no_aws_call()
    test_warm_prefix_empty_returns_none()
    test_generate_stream_simple_text()
    test_generate_stream_tool_call()
    test_generate_converse_tool_call()
    test_cache_point_injected_for_claude()
    test_cache_point_not_injected_for_llama()
    test_credential_chain_uses_factory()
    test_malformed_tool_block_missing_id()
    test_malformed_tool_block_bad_json()
    test_tool_schema_conversion()
    test_message_conversion_assistant_raw()
    test_prompt_caching_support_table()
    test_send_prefetch_hints_noop()
    print("\n[ALL PASS] Bedrock backend adapter tests")
