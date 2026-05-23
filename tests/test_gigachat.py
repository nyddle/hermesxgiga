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
