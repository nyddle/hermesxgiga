"""Verify blocking GigaChat calls don't stall the event loop.

A slow (blocking) ``complete`` must run in a worker thread so other requests
(here ``/health``) stay responsive. Without the threadpool offload the async
endpoint would block the loop and ``/health`` would wait for the full sleep.
"""

import asyncio
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402

from hermesxgiga.server import create_app  # noqa: E402

BLOCK_SECONDS = 0.5


class SlowClient:
    model = "GigaChat"

    def complete(self, messages, **kwargs):
        time.sleep(BLOCK_SECONDS)  # blocking, like a real network call
        return {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ]
        }

    def list_models(self):
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
    # /health must return well before the blocking sleep finishes
    assert health_latency < BLOCK_SECONDS / 2
