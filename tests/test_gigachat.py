import types

import pytest

from hermesxgiga import GigaChatClient, GigaChatError, Message


def completion(content, function_call=None, functions_state_id=None):
    """Build a ChatCompletion-shaped object like the real SDK returns."""
    msg = types.SimpleNamespace(
        content=content,
        function_call=function_call,
        functions_state_id=functions_state_id,
    )
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def chunk(content):
    """Build a ChatCompletionChunk-shaped object."""
    delta = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


class FakeGiga:
    """Duck-typed stand-in for gigachat.GigaChat."""

    def __init__(self, result=None, error=None, stream_chunks=None):
        self._result = result if result is not None else completion("ok")
        self._error = error
        self._stream_chunks = stream_chunks
        self.payloads = []
        self.stream_payloads = []
        self.closed = False

    def chat(self, payload):
        self.payloads.append(payload)
        if self._error is not None:
            raise self._error
        return self._result

    def stream(self, payload):
        self.stream_payloads.append(payload)
        if self._error is not None:
            raise self._error
        for c in self._stream_chunks or []:
            yield c

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


def test_missing_auth_rejected_without_injected_client():
    with pytest.raises(ValueError, match="auth_key, user/password"):
        GigaChatClient("")


def test_user_password_auth_constructs_real_sdk():
    # Smoke: no network, just check the SDK is instantiated and the
    # adapter wires user/password (not credentials) into the kwargs.
    client = GigaChatClient(user="u", password="p", verify_ssl=False)
    assert client._giga.__class__.__module__.startswith("gigachat")
    client.close()


def test_context_manager_closes_underlying_client():
    giga = FakeGiga()
    with GigaChatClient(client=giga) as client:
        client.chat([Message("user", "x")])
    assert giga.closed is True


def test_complete_returns_text_result():
    giga = FakeGiga(completion("hi"))
    result = GigaChatClient(client=giga).complete([Message("user", "x")])
    assert result.content == "hi"
    assert result.tool_call is None


def test_complete_surfaces_tool_call():
    fc = types.SimpleNamespace(name="get_weather", arguments={"city": "Москва"})
    giga = FakeGiga(completion(None, function_call=fc, functions_state_id="s1"))
    result = GigaChatClient(client=giga).complete(
        [Message("user", "погода?")],
        functions=[{"name": "get_weather", "description": "", "parameters": {}}],
    )
    assert result.content is None
    assert result.tool_call.name == "get_weather"
    assert result.tool_call.arguments == {"city": "Москва"}
    assert result.functions_state_id == "s1"
    # functions are forwarded into the SDK payload; function_call omitted when unset.
    assert giga.payloads[0]["functions"][0]["name"] == "get_weather"
    assert "function_call" not in giga.payloads[0]


def test_chat_raises_when_only_tool_call_returned():
    fc = types.SimpleNamespace(name="f", arguments={})
    giga = FakeGiga(completion(None, function_call=fc))
    with pytest.raises(GigaChatError, match="no text content"):
        GigaChatClient(client=giga).chat([Message("user", "x")])


def test_stream_chat_yields_deltas():
    giga = FakeGiga(stream_chunks=[chunk("При"), chunk("вет"), chunk(None), chunk("!")])
    out = list(GigaChatClient(client=giga).stream_chat([Message("user", "x")]))
    assert out == ["При", "вет", "!"]
    assert giga.stream_payloads[0]["messages"] == [{"role": "user", "content": "x"}]


def test_stream_chat_wraps_errors():
    giga = FakeGiga(error=RuntimeError("boom"), stream_chunks=[])
    with pytest.raises(GigaChatError, match="GigaChat stream failed: boom"):
        list(GigaChatClient(client=giga).stream_chat([Message("user", "x")]))


def test_function_message_is_normalized_with_name_and_call():
    giga = FakeGiga()
    client = GigaChatClient(client=giga)
    client.complete(
        [
            Message("assistant", function_call={"name": "f", "arguments": {"a": 1}}),
            Message("function", content="42", name="f"),
        ]
    )
    assert giga.payloads[0]["messages"] == [
        {"role": "assistant", "function_call": {"name": "f", "arguments": {"a": 1}}},
        {"role": "function", "content": "42", "name": "f"},
    ]
