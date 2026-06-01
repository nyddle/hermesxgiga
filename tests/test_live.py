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

from hermesxgiga import GigaChatClient, HermesBot, Message, Tool


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


def test_live_streaming():
    with _client() as client:
        deltas = list(
            client.stream_chat([Message("user", "Назови три цвета через запятую.")])
        )
    assert deltas, "expected at least one streamed delta"
    assert "".join(deltas).strip()


def test_live_function_calling():
    calls = []

    def get_weather(args):
        calls.append(args)
        return {"city": args.get("city"), "temp_c": 7, "condition": "облачно"}

    tool = Tool(
        name="get_weather",
        description="Возвращает текущую погоду в указанном городе.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "Город"}},
            "required": ["city"],
        },
        handler=get_weather,
    )
    with _client() as client:
        bot = HermesBot(
            client,
            system_prompt=(
                "Если спрашивают про погоду — вызови функцию get_weather "
                "и ответь по её данным."
            ),
            tools=[tool],
        )
        answer = bot.handle("live", "Какая сейчас погода в Москве?")
    assert calls, "model did not invoke the get_weather tool"
    assert calls[0].get("city")
    assert answer.strip()
