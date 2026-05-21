"""OpenAI-compatible HTTP server in front of GigaChat.

Hermes (https://github.com/nousresearch/hermes-agent) talks to any
OpenAI-compatible ``/v1`` endpoint. This module exposes exactly that surface
and proxies it to GigaChat, translating tool/function calls and streaming in
both directions (see :mod:`hermesxgiga.openai_compat`).

Point Hermes at it via ``~/.hermes/config.yaml``::

    model:
      provider: custom
      base_url: "http://localhost:8000/v1"
      api_key: "local"          # any value; checked only if you set one
      model: "GigaChat-Pro"

Run it with ``python -m hermesxgiga.server`` (see :func:`main` for flags) or
``hermesxgiga-server``.

The web framework deps are optional: ``pip install "hermesxgiga[server]"``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from .gigachat import GigaChatClient, GigaChatError
from .openai_compat import (
    new_completion_id,
    to_gigachat_function_call,
    to_gigachat_functions,
    to_gigachat_messages,
    to_openai_response,
    to_openai_stream,
)

try:  # optional dependency group: [server]
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - depends on env
    raise ImportError(
        'the OpenAI-compatible server needs FastAPI/uvicorn: '
        'pip install "hermesxgiga[server]"'
    ) from exc


def _require_api_key(authorization: Optional[str], expected: Optional[str]) -> None:
    if not expected:
        return
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


def create_app(
    client: Optional[GigaChatClient] = None,
    *,
    api_key: Optional[str] = None,
) -> "FastAPI":
    """Build the FastAPI app.

    Parameters
    ----------
    client:
        A ready :class:`GigaChatClient`. If omitted, one is built from
        environment variables (see :func:`client_from_env`).
    api_key:
        If set, callers must send ``Authorization: Bearer <api_key>``. Defaults
        to ``HERMESXGIGA_API_KEY`` from the environment (no check if unset).
    """
    if client is None:
        client = client_from_env()
    if api_key is None:
        api_key = os.environ.get("HERMESXGIGA_API_KEY")

    app = FastAPI(title="hermesxgiga OpenAI shim")

    def auth(authorization: Optional[str] = Header(default=None)) -> None:
        _require_api_key(authorization, api_key)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    def list_models() -> dict:
        try:
            ids = client.list_models()
        except GigaChatError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "owned_by": "gigachat"}
                for mid in ids
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be an object")

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        model = body.get("model") or client.model
        giga_messages = to_gigachat_messages(messages)
        functions = to_gigachat_functions(body.get("tools"), body.get("functions"))
        function_call = to_gigachat_function_call(
            body.get("tool_choice", body.get("function_call"))
        )
        kwargs = {
            "model": model,
            "temperature": body.get("temperature"),
            "max_tokens": body.get("max_tokens"),
            "top_p": body.get("top_p"),
            "functions": functions or None,
            "function_call": function_call,
        }

        if body.get("stream"):
            return _stream_response(client, giga_messages, model, kwargs)

        try:
            giga_response = client.complete(giga_messages, **kwargs)
        except GigaChatError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return JSONResponse(to_openai_response(giga_response, model=model))

    return app


def _stream_response(client, giga_messages, model, kwargs):
    cid = new_completion_id()
    created = int(time.time())

    def event_stream():
        try:
            giga_chunks = client.stream(giga_messages, **kwargs)
            for chunk in to_openai_stream(
                giga_chunks, model=model, completion_id=cid, created=created
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except GigaChatError as exc:
            err = {"error": {"message": str(exc), "type": "upstream_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def client_from_env() -> GigaChatClient:
    """Build a :class:`GigaChatClient` from ``GIGACHAT_*`` environment variables.

    Recognises ``GIGACHAT_AUTH_KEY`` (or ``GIGACHAT_CREDENTIALS``),
    ``GIGACHAT_SCOPE``, ``GIGACHAT_MODEL``, ``GIGACHAT_VERIFY_SSL`` and
    ``GIGACHAT_CA_BUNDLE``.
    """
    auth_key = os.environ.get("GIGACHAT_AUTH_KEY") or os.environ.get(
        "GIGACHAT_CREDENTIALS", ""
    )
    if not auth_key:
        raise ValueError(
            "set GIGACHAT_AUTH_KEY (base64 client_id:client_secret) to start the server"
        )

    ca_bundle = os.environ.get("GIGACHAT_CA_BUNDLE")
    verify_env = os.environ.get("GIGACHAT_VERIFY_SSL")
    if ca_bundle:
        verify_ssl: object = ca_bundle
    elif verify_env is not None:
        verify_ssl = verify_env.strip().lower() not in ("0", "false", "no", "")
    else:
        verify_ssl = True

    return GigaChatClient(
        auth_key,
        scope=os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        model=os.environ.get("GIGACHAT_MODEL", "GigaChat"),
        verify_ssl=verify_ssl,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenAI-compatible server proxying to GigaChat (for Hermes)."
    )
    parser.add_argument("--host", default=os.environ.get("HERMESXGIGA_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("HERMESXGIGA_PORT", "8000"))
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            'uvicorn is required to run the server: pip install "hermesxgiga[server]"'
        ) from exc

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
