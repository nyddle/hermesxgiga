import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hermesxgiga.gigachat import GigaChatError  # noqa: E402
from hermesxgiga.server import create_app  # noqa: E402


class FakeClient:
    """Stands in for GigaChatClient at the server boundary."""

    model = "GigaChat"

    def __init__(self):
        self.complete_calls = []
        self.stream_calls = []
        self.complete_result = {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self.stream_result = [
            {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]

    def complete(self, messages, **kwargs):
        self.complete_calls.append((messages, kwargs))
        return self.complete_result

    def stream(self, messages, **kwargs):
        self.stream_calls.append((messages, kwargs))
        yield from self.stream_result

    def list_models(self):
        return ["GigaChat", "GigaChat-Pro"]


def make_client(api_key=None):
    fake = FakeClient()
    app = create_app(fake, api_key=api_key)
    return fake, TestClient(app)


def test_health():
    _, client = make_client()
    assert client.get("/health").json() == {"status": "ok"}


def test_list_models():
    _, client = make_client()
    data = client.get("/v1/models").json()
    assert [m["id"] for m in data["data"]] == ["GigaChat", "GigaChat-Pro"]
    assert data["object"] == "list"


def test_chat_completion_non_stream():
    fake, client = make_client()
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "GigaChat-Pro", "messages": [{"role": "user", "content": "hi"}]},
    )
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["model"] == "GigaChat-Pro"
    # request was translated and model forwarded
    messages, kwargs = fake.complete_calls[0]
    assert messages == [{"role": "user", "content": "hi"}]
    assert kwargs["model"] == "GigaChat-Pro"


def test_chat_completion_with_tools_forwarded():
    fake, client = make_client()
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}
            ],
            "tool_choice": "auto",
        },
    )
    _, kwargs = fake.complete_calls[0]
    assert kwargs["functions"] == [{"name": "weather", "parameters": {"type": "object"}}]
    assert kwargs["function_call"] == "auto"


def test_chat_completion_stream():
    _, client = make_client()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        chunks = [line for line in resp.iter_lines() if line]

    payloads = [json.loads(c[len("data: ") :]) for c in chunks if not c.endswith("[DONE]")]
    assert chunks[-1].endswith("[DONE]")
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert any(p["choices"][0]["delta"].get("content") == "hi" for p in payloads)
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_missing_messages_is_400():
    _, client = make_client()
    assert client.post("/v1/chat/completions", json={}).status_code == 400


def test_upstream_error_is_502():
    fake, client = make_client()

    def boom(messages, **kwargs):
        raise GigaChatError("down")

    fake.complete = boom
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
    )
    assert resp.status_code == 502


def test_api_key_enforced_when_set():
    _, client = make_client(api_key="secret")
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
