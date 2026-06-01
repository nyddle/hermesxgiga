"""OpenAI-compatible HTTP proxy in front of GigaChat.

Lets any tool that speaks the OpenAI ``/v1/chat/completions`` wire protocol
(e.g. the Nous Research **Hermes Agent** with ``provider: custom``) talk to
GigaChat. The proxy translates between the two shapes:

* OpenAI ``tools`` / ``tool_calls`` / ``role:"tool"``  <->  GigaChat
  ``functions`` / ``function_call`` / ``role:"function"``,
* streaming Server-Sent Events  <->  GigaChat SDK streaming.

The translation helpers at the top are pure and unit-tested; the FastAPI app
is a thin shell over :class:`hermesxgiga.gigachat.GigaChatClient`.

Run it with ``python -m hermesxgiga.openai_proxy`` (needs the ``proxy`` extra:
``pip install -e ".[proxy]"``).
"""

import json
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .gigachat import ChatResult, GigaChatClient, GigaChatError, Message

# --------------------------------------------------------------------------- #
# Pure translation helpers (no I/O — unit tested directly)
# --------------------------------------------------------------------------- #


def _content_to_text(content: Any) -> Optional[str]:
    """OpenAI content may be a string or a list of parts; reduce to text."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") in (None, "text")
        ]
        return "".join(parts)
    return str(content)


def _tool_call_id_to_name(messages: List[dict]) -> Dict[str, str]:
    """Map each assistant ``tool_call`` id to its function name."""
    mapping: Dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls") or ():
            fn = (tc.get("function") or {}).get("name")
            if tc.get("id") and fn:
                mapping[tc["id"]] = fn
    return mapping


def openai_messages_to_giga(messages: List[dict]) -> List[Message]:
    """Translate OpenAI chat messages into GigaChat ``Message`` objects."""
    id_to_name = _tool_call_id_to_name(messages)
    out: List[Message] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            name = id_to_name.get(m.get("tool_call_id"), m.get("name") or "tool")
            out.append(
                Message(
                    role="function",
                    content=_content_to_text(m.get("content")) or "",
                    name=name,
                )
            )
        elif role == "assistant" and m.get("tool_calls"):
            # GigaChat carries a single function_call; use the first tool call.
            tc = m["tool_calls"][0]
            fn = tc.get("function") or {}
            out.append(
                Message(
                    role="assistant",
                    function_call={
                        "name": fn.get("name"),
                        "arguments": _parse_arguments(fn.get("arguments")),
                    },
                )
            )
        else:
            out.append(
                Message(role=role, content=_content_to_text(m.get("content")) or "")
            )
    return out


def _parse_arguments(arguments: Any) -> dict:
    """OpenAI sends function arguments as a JSON string; GigaChat wants a dict."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return {}


def openai_tools_to_functions(
    tools: Optional[List[dict]], functions: Optional[List[dict]]
) -> Optional[List[dict]]:
    """Translate OpenAI ``tools`` (or legacy ``functions``) into GigaChat ones."""
    schemas: List[dict] = []
    for t in tools or ():
        fn = t.get("function") if t.get("type") == "function" else t
        if not fn:
            continue
        schemas.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    for fn in functions or ():  # legacy OpenAI "functions" field
        schemas.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return schemas or None


def openai_tool_choice_to_function_call(tool_choice: Any) -> Optional[Any]:
    """Translate OpenAI ``tool_choice`` into GigaChat ``function_call``."""
    if tool_choice in (None, "auto", "none", "required"):
        # GigaChat understands "auto"/"none"; map "required" to "auto".
        return "auto" if tool_choice == "required" else tool_choice
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        if fn.get("name"):
            return {"name": fn["name"]}
    return None


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _tool_calls_payload(result: ChatResult) -> List[dict]:
    tc = result.tool_call
    return [
        {
            "id": _new_id("call"),
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
            },
        }
    ]


def giga_result_to_openai_completion(result: ChatResult, model: str) -> dict:
    """Build a non-streaming OpenAI ``chat.completion`` object."""
    if result.tool_call is not None:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": _tool_calls_payload(result),
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": result.content or ""}
        finish_reason = "stop"
    return {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
    }


def _chunk(model: str, completion_id: str, delta: dict, finish_reason: Optional[str]) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------- #
# Streaming generators
# --------------------------------------------------------------------------- #


def _stream_text_deltas(
    client: GigaChatClient, completion_id: str, call: dict
) -> Iterator[str]:
    """Real upstream streaming for plain-text turns (no tools)."""
    model = call.get("model")
    yield _sse(_chunk(model, completion_id, {"role": "assistant"}, None))
    for delta in client.stream_chat(**call):
        yield _sse(_chunk(model, completion_id, {"content": delta}, None))
    yield _sse(_chunk(model, completion_id, {}, "stop"))
    yield "data: [DONE]\n\n"


def _stream_synthesized(
    result: ChatResult, model: str, completion_id: str
) -> Iterator[str]:
    """Synthesize SSE from a non-streaming result (used for tool-call turns)."""
    if result.tool_call is not None:
        delta = {"role": "assistant", "content": None, "tool_calls": _tool_calls_payload(result)}
        yield _sse(_chunk(model, completion_id, delta, None))
        yield _sse(_chunk(model, completion_id, {}, "tool_calls"))
    else:
        yield _sse(_chunk(model, completion_id, {"role": "assistant"}, None))
        if result.content:
            yield _sse(_chunk(model, completion_id, {"content": result.content}, None))
        yield _sse(_chunk(model, completion_id, {}, "stop"))
    yield "data: [DONE]\n\n"


def _prepare_call(body: dict, default_model: str) -> Tuple[dict, str]:
    """Translate an OpenAI request body into GigaChatClient call kwargs."""
    model = body.get("model") or default_model
    call: dict = {
        "messages": openai_messages_to_giga(body.get("messages") or []),
        "model": model,
    }
    functions = openai_tools_to_functions(body.get("tools"), body.get("functions"))
    if functions is not None:
        call["functions"] = functions
        fc = openai_tool_choice_to_function_call(
            body.get("tool_choice", body.get("function_call"))
        )
        if fc is not None:
            call["function_call"] = fc
    if body.get("temperature") is not None:
        call["temperature"] = body["temperature"]
    if body.get("max_tokens") is not None:
        call["max_tokens"] = body["max_tokens"]
    return call, model


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #


def create_app(client: GigaChatClient, *, default_model: str = "GigaChat"):
    """Build the FastAPI proxy app backed by ``client``."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="hermesxgiga OpenAI proxy")

    def _error(message: str, status: int = 502) -> JSONResponse:
        return JSONResponse(
            {"error": {"message": message, "type": "gigachat_error"}},
            status_code=status,
        )

    @app.get("/v1/models")
    async def list_models() -> Any:
        try:
            ids = await run_in_threadpool(client.list_models)
        except GigaChatError as exc:
            return _error(str(exc))
        return {
            "object": "list",
            "data": [
                {"id": i, "object": "model", "owned_by": "gigachat"} for i in ids
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        try:
            call, model = _prepare_call(body, default_model)
        except Exception as exc:  # malformed request
            return _error(f"bad request: {exc}", status=400)

        has_tools = "functions" in call
        completion_id = _new_id("chatcmpl")

        if body.get("stream"):
            if has_tools:
                # Tool-call turns: fetch non-streaming, then synthesize SSE.
                try:
                    result = await run_in_threadpool(client.complete, **call)
                except GigaChatError as exc:
                    return _error(str(exc))
                gen = _stream_synthesized(result, model, completion_id)
            else:
                gen = _stream_text_deltas(client, completion_id, call)
            return StreamingResponse(gen, media_type="text/event-stream")

        try:
            result = await run_in_threadpool(client.complete, **call)
        except GigaChatError as exc:
            return _error(str(exc))
        return giga_result_to_openai_completion(result, model)

    return app


def _client_from_env() -> GigaChatClient:
    import os

    ca_bundle = os.environ.get("GIGACHAT_CA_BUNDLE")
    if ca_bundle:
        verify: Any = ca_bundle
    else:
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


def main() -> None:
    import os

    import uvicorn

    client = _client_from_env()
    app = create_app(client, default_model=os.environ.get("GIGACHAT_MODEL", "GigaChat"))
    uvicorn.run(
        app,
        host=os.environ.get("PROXY_HOST", "127.0.0.1"),
        port=int(os.environ.get("PROXY_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
