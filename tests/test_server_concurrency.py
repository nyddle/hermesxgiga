"""Verify the async server stays responsive while a request is in flight.

The endpoints are async-native (SDK ``achat``/``astream``), so a slow upstream
call must yield the event loop and let other requests (here ``/health``) run
concurrently rather than serialize behind it.
"""

import asyncio
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402

from hermesxgiga.server import create_app  # noqa: E402

SLOW_SECONDS = 0.5


class SlowClient:
    model = "GigaChat"

    async def acomplete(self, messages, **kwargs):
        await asyncio.sleep(SLOW_SECONDS)  # slow upstream I/O, but non-blocking
        return {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ]
        }

    async def alist_models(self):
        return ["GigaChat"]


def test_blocking_completion_does_not_stall_health():
    app = create_app(SlowClient())

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            chat = asyncio.create_task(
                ac.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
            )
            await asyncio.sleep(0.05)  # let the chat request start blocking

            start = time.perf_counter()
            health = await ac.get("/health")
            health_latency = time.perf_counter() - start

            chat_resp = await chat
            return health, health_latency, chat_resp

    health, health_latency, chat_resp = asyncio.run(scenario())

    assert health.status_code == 200
    assert chat_resp.status_code == 200
    # /health must return well before the slow upstream call finishes
    assert health_latency < SLOW_SECONDS / 2
