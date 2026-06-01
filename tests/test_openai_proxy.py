import json

import pytest

from hermesxgiga import ChatResult, Message, ToolCall
from hermesxgiga.openai_proxy import (
    create_app,
    giga_result_to_openai_completion,
    openai_messages_to_giga,
    openai_tool_choice_to_function_call,
    openai_tools_to_functions,
)

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


# --- pure translation -------------------------------------------------------


def test_messages_plain_roundtrip():
    out = openai_messages_to_giga(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    )
    assert out == [Message("system", "s"), Message("user", "hi")]


def test_messages_tool_call_and_result_mapping():
    out = openai_messages_to_giga(
        [
            {"role": "user", "content": "погода?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Мск"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "7C"},
        ]
    )
    assert out[1] == Message(
        "assistant", function_call={"name": "get_weather", "arguments": {"city": "Мск"}}
    )
    # tool result resolves its function name from the tool_call_id.
    assert out[2] == Message("function", content="7C", name="get_weather")


def test_content_parts_list_is_flattened():
    out = openai_messages_to_giga(
        [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    )
    assert out == [Message("user", "ab")]


def test_tools_translation():
    schemas = openai_tools_to_functions(
        [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            }
        ],
        None,
    )
    assert schemas == [
        {
            "name": "f",
            "description": "d",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
    ]
    assert openai_tools_to_functions(None, None) is None


def test_tool_choice_translation():
    assert openai_tool_choice_to_function_call("auto") == "auto"
    assert openai_tool_choice_to_function_call("none") == "none"
    assert openai_tool_choice_to_function_call("required") == "auto"
    assert openai_tool_choice_to_function_call(
        {"type": "function", "function": {"name": "f"}}
    ) == {"name": "f"}
    assert openai_tool_choice_to_function_call(None) is None


def test_result_to_completion_text():
    comp = giga_result_to_openai_completion(ChatResult(content="hello"), "GigaChat")
    choice = comp["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "hello"}
    assert choice["finish_reason"] == "stop"
    assert comp["object"] == "chat.completion"


def test_result_to_completion_tool_call():
    result = ChatResult(content=None, tool_call=ToolCall("f", {"a": 1}))
    comp = giga_result_to_openai_completion(result, "GigaChat")
    choice = comp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "f"
    assert json.loads(tc["function"]["arguments"]) == {"a": 1}


# --- endpoints via TestClient ----------------------------------------------


class FakeClient:
    def __init__(self, result=None, deltas=None, models=None):
        self._result = result or ChatResult(content="ok")
        self._deltas = deltas or ["He", "llo"]
        self._models = models or ["GigaChat", "GigaChat-3-Ultra"]
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(("complete", kwargs))
        return self._result

    def stream_chat(self, **kwargs):
        self.calls.append(("stream", kwargs))
        for d in self._deltas:
            yield d

    def list_models(self):
        return self._models


def test_models_endpoint():
    app = create_app(FakeClient())
    resp = TestClient(app).get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "GigaChat-3-Ultra" in ids


def test_chat_completion_non_stream():
    fake = FakeClient(ChatResult(content="привет"))
    app = create_app(fake, default_model="GigaChat")
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "GigaChat-3-Ultra", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "привет"
    assert body["model"] == "GigaChat-3-Ultra"


def test_chat_completion_tool_call_non_stream():
    fake = FakeClient(ChatResult(content=None, tool_call=ToolCall("get_weather", {"city": "Мск"})))
    app = create_app(fake)
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "GigaChat",
            "messages": [{"role": "user", "content": "погода?"}],
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
            ],
        },
    )
    body = resp.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    # functions were forwarded to the client.
    assert fake.calls[0][1]["functions"][0]["name"] == "get_weather"


def test_chat_completion_streaming_text():
    fake = FakeClient(deltas=["At", "om"])
    app = create_app(fake)
    with TestClient(app).stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        raw = "".join(resp.iter_text())
    assert "chat.completion.chunk" in raw
    assert '"content": "At"' in raw
    assert raw.strip().endswith("data: [DONE]")


def test_chat_completion_streaming_tool_call_is_synthesized():
    fake = FakeClient(ChatResult(content=None, tool_call=ToolCall("f", {"a": 1})))
    app = create_app(fake)
    with TestClient(app).stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "stream": True,
        },
    ) as resp:
        raw = "".join(resp.iter_text())
    # tool turns go through complete(), not stream_chat().
    assert fake.calls[0][0] == "complete"
    assert '"tool_calls"' in raw
    assert raw.strip().endswith("data: [DONE]")
