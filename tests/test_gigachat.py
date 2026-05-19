import pytest

from hermesxgiga import GigaChatClient, GigaChatError, Message


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses and records every request made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._responses.pop(0)


def make_client(session, **kw):
    return GigaChatClient("dGVzdDp0ZXN0", session=session, **kw)


def test_oauth_then_chat_and_token_is_cached():
    session = FakeSession(
        [
            FakeResponse(200, {"access_token": "tok", "expires_at": 9_999_999_999_000}),
            FakeResponse(200, {"choices": [{"message": {"content": "привет"}}]}),
            FakeResponse(200, {"choices": [{"message": {"content": "ещё"}}]}),
        ]
    )
    client = make_client(session)

    assert client.chat([Message("user", "hi")]) == "привет"
    # Second call reuses the cached token -> no extra OAuth request.
    assert client.chat([{"role": "user", "content": "again"}]) == "ещё"

    urls = [c[0] for c in session.calls]
    assert urls.count(client.oauth_url) == 1
    assert urls.count(client.api_url) == 2

    # Bearer token is attached to chat calls.
    chat_kwargs = session.calls[1][1]
    assert chat_kwargs["headers"]["Authorization"] == "Bearer tok"
    assert chat_kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]


def test_401_triggers_single_token_refresh_and_retry():
    session = FakeSession(
        [
            FakeResponse(200, {"access_token": "old", "expires_at": 9_999_999_999_000}),
            FakeResponse(401, text="expired"),
            FakeResponse(200, {"access_token": "new", "expires_at": 9_999_999_999_000}),
            FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    client = make_client(session)

    assert client.chat([Message("user", "hi")]) == "ok"

    urls = [c[0] for c in session.calls]
    assert urls == [
        client.oauth_url,
        client.api_url,
        client.oauth_url,
        client.api_url,
    ]
    assert session.calls[-1][1]["headers"]["Authorization"] == "Bearer new"


def test_oauth_failure_raises():
    session = FakeSession([FakeResponse(401, text="bad creds")])
    client = make_client(session)
    with pytest.raises(GigaChatError, match="OAuth failed"):
        client.chat([Message("user", "hi")])


def test_unexpected_chat_shape_raises():
    session = FakeSession(
        [
            FakeResponse(200, {"access_token": "t", "expires_at": 9_999_999_999_000}),
            FakeResponse(200, {"unexpected": True}),
        ]
    )
    client = make_client(session)
    with pytest.raises(GigaChatError, match="Unexpected chat response"):
        client.chat([Message("user", "hi")])


def test_empty_messages_rejected():
    client = make_client(FakeSession([]))
    with pytest.raises(ValueError, match="must not be empty"):
        client.chat([])


def test_bad_message_dict_rejected():
    client = make_client(FakeSession([]))
    with pytest.raises(ValueError, match="role/content"):
        client.chat([{"role": "user"}])


def test_missing_auth_key_rejected():
    with pytest.raises(ValueError, match="auth_key is required"):
        GigaChatClient("")
