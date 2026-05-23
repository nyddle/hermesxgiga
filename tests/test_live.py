"""Live integration tests against the real GigaChat API.

Skipped unless credentials are present in the environment. Set either:

* ``GIGACHAT_AUTH_KEY`` (the base64 ``client_id:client_secret``), or
* ``GIGACHAT_USER`` + ``GIGACHAT_PASSWORD``, or
* ``GIGACHAT_ACCESS_TOKEN``.

Optional: ``GIGACHAT_SCOPE``, ``GIGACHAT_MODEL``,
``GIGACHAT_VERIFY_SSL=0`` to disable TLS verification (needed when the
official Russian CA chain is not installed).
"""

from __future__ import annotations

import os

import pytest

from hermesxgiga import GigaChatClient, HermesBot, Message


def _have_creds() -> bool:
    if os.environ.get("GIGACHAT_AUTH_KEY"):
        return True
    if os.environ.get("GIGACHAT_USER") and os.environ.get("GIGACHAT_PASSWORD"):
        return True
    if os.environ.get("GIGACHAT_ACCESS_TOKEN"):
        return True
    return False


pytestmark = pytest.mark.skipif(
    not _have_creds(),
    reason="no GigaChat credentials in env (GIGACHAT_AUTH_KEY or USER/PASSWORD)",
)


def _client() -> GigaChatClient:
    verify = os.environ.get("GIGACHAT_VERIFY_SSL", "1") not in ("0", "false", "no")
    return GigaChatClient(
        auth_key=os.environ.get("GIGACHAT_AUTH_KEY", ""),
        user=os.environ.get("GIGACHAT_USER"),
        password=os.environ.get("GIGACHAT_PASSWORD"),
        access_token=os.environ.get("GIGACHAT_ACCESS_TOKEN"),
        scope=os.environ.get("GIGACHAT_SCOPE") or None,
        model=os.environ.get("GIGACHAT_MODEL", "GigaChat"),
        verify_ssl=verify,
    )


def test_live_single_turn():
    with _client() as client:
        reply = client.chat([Message("user", "Скажи одно слово: пинг.")])
    assert isinstance(reply, str)
    assert reply.strip()


def test_live_bot_two_turns():
    with _client() as client:
        bot = HermesBot(client, system_prompt="Отвечай очень кратко.")
        first = bot.handle("live", "Назови столицу Франции одним словом.")
        second = bot.handle("live", "А Германии?")
    assert first.strip()
    assert second.strip()
