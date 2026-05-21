import asyncio
import types

import pytest

from hermesxgiga import GigaChatClient, GigaChatError, Message


def completion(content):
    """Build a ChatCompletion-shaped object like the real SDK returns."""
    msg = types.SimpleNamespace(message=types.SimpleNamespace(content=content))
    return types.SimpleNamespace(choices=[msg])


class FakeGiga:
    """Duck-typed stand-in for gigachat.GigaChat."""

    def __init__(self, result=None, error=None):
        self._result = result if result is not None else completion("ok")
        self._error = error
        self.payloads = []
        self.closed = False

    def chat(self, payload):
        self.payloads.append(payload)
        if self._error is not None:
            raise self._error
        return self._result

    def close(self):
        self.closed = True


def test_chat_builds_payload_and_returns_content():
    giga = FakeGiga(completion("привет"))
    client = GigaChatClient(client=giga, model="GigaChat-Pro")

    out = client.chat(
        [Message("system", "s"), {"role": "user", "content": "hi"}],
        temperature=0.3,
        max_tokens=64,
    )

    assert out == "привет"
    assert giga.payloads == [
        {
            "model": "GigaChat-Pro",
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "hi"},
            ],
            "temperature": 0.3,
            "max_tokens": 64,
        }
    ]


def test_per_call_model_override():
    giga = FakeGiga()
    client = GigaChatClient(client=giga)
    client.chat([Message("user", "x")], model="GigaChat-Max")
    assert giga.payloads[0]["model"] == "GigaChat-Max"


def test_dict_shaped_response_is_supported():
    giga = FakeGiga({"choices": [{"message": {"content": "d"}}]})
    assert GigaChatClient(client=giga).chat([Message("user", "x")]) == "d"


def test_sdk_exception_wrapped_in_gigachat_error():
    giga = FakeGiga(error=RuntimeError("boom"))
    with pytest.raises(GigaChatError, match="GigaChat request failed: boom"):
        GigaChatClient(client=giga).chat([Message("user", "x")])


def test_gigachat_error_not_double_wrapped():
    giga = FakeGiga(error=GigaChatError("explicit"))
    with pytest.raises(GigaChatError, match="^explicit$"):
        GigaChatClient(client=giga).chat([Message("user", "x")])


def test_unexpected_response_shape_raises():
    giga = FakeGiga({"weird": True})
    with pytest.raises(GigaChatError, match="Unexpected chat response"):
        GigaChatClient(client=giga).chat([Message("user", "x")])


def test_empty_and_bad_messages_rejected():
    client = GigaChatClient(client=FakeGiga())
    with pytest.raises(ValueError, match="must not be empty"):
        client.chat([])
    with pytest.raises(ValueError, match="role/content"):
        client.chat([{"role": "user"}])


def test_missing_auth_key_rejected_without_injected_client():
    with pytest.raises(ValueError, match="auth_key is required"):
        GigaChatClient("")


def test_context_manager_closes_underlying_client():
    giga = FakeGiga()
    with GigaChatClient(client=giga) as client:
        client.chat([Message("user", "x")])
    assert giga.closed is True


class FakeAsyncGiga:
    """Async duck-typed stand-in for gigachat.GigaChat."""

    def __init__(self, result=None, chunks=None, models=None, error=None):
        self._result = result if result is not None else completion("ok")
        self._chunks = chunks or []
        self._models = models
        self._error = error
        self.payloads = []
        self.aclosed = False

    async def achat(self, payload):
        self.payloads.append(payload)
        if self._error is not None:
            raise self._error
        return self._result

    async def astream(self, payload):
        self.payloads.append(payload)
        if self._error is not None:
            raise self._error
        for c in self._chunks:
            yield c

    async def aget_models(self):
        return self._models

    async def aclose(self):
        self.aclosed = True


def test_acomplete_returns_dict_and_builds_payload():
    giga = FakeAsyncGiga({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    client = GigaChatClient(client=giga, model="GigaChat-Pro")

    out = asyncio.run(
        client.acomplete([Message("user", "x")], functions=[{"name": "f"}], function_call="auto")
    )
    assert out["choices"][0]["message"]["content"] == "hi"
    assert giga.payloads[0]["model"] == "GigaChat-Pro"
    assert giga.payloads[0]["functions"] == [{"name": "f"}]
    assert giga.payloads[0]["function_call"] == "auto"


def test_astream_yields_chunk_dicts():
    chunks = [
        {"choices": [{"delta": {"content": "a"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    giga = FakeAsyncGiga(chunks=chunks)
    client = GigaChatClient(client=giga)

    async def collect():
        return [c async for c in client.astream([Message("user", "x")])]

    out = asyncio.run(collect())
    assert out == chunks
    assert giga.payloads[0]["stream"] is True


def test_acomplete_wraps_errors():
    giga = FakeAsyncGiga(error=RuntimeError("boom"))
    client = GigaChatClient(client=giga)
    with pytest.raises(GigaChatError, match="GigaChat request failed: boom"):
        asyncio.run(client.acomplete([Message("user", "x")]))


def test_alist_models_extracts_ids():
    giga = FakeAsyncGiga(models={"data": [{"id_": "GigaChat"}, {"id": "GigaChat-Pro"}]})
    client = GigaChatClient(client=giga)
    assert asyncio.run(client.alist_models()) == ["GigaChat", "GigaChat-Pro"]


def test_async_context_manager_closes_underlying_client():
    giga = FakeAsyncGiga()

    async def scenario():
        async with GigaChatClient(client=giga) as client:
            await client.acomplete([Message("user", "x")])

    asyncio.run(scenario())
    assert giga.aclosed is True
