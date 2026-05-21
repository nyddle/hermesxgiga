"""Translation between the OpenAI chat API shape and GigaChat.

Hermes (and most agent frameworks) speak the OpenAI ``/v1/chat/completions``
protocol. GigaChat uses a close but different dialect, most notably:

* tools are called *functions* — OpenAI's ``tools`` / ``tool_calls`` map onto
  GigaChat's ``functions`` / ``function_call`` (a single call, not a list);
* ``function_call.arguments`` is a JSON **object** in GigaChat but a JSON
  **string** in OpenAI;
* tool results use ``role: "function"`` (+ ``name``) rather than
  ``role: "tool"`` (+ ``tool_call_id``).

These helpers are pure functions over plain dicts so they are trivial to test
without any network or SDK.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

__all__ = [
    "to_gigachat_messages",
    "to_gigachat_functions",
    "to_gigachat_function_call",
    "to_openai_response",
    "to_openai_stream",
    "new_completion_id",
]


def new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _arguments_to_dict(arguments: Any) -> dict:
    """OpenAI sends tool-call arguments as a JSON string; GigaChat wants a dict."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _arguments_to_str(arguments: Any) -> str:
    """GigaChat returns an object; OpenAI clients expect a JSON string."""
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False)


def to_gigachat_messages(messages: Iterable[dict]) -> List[dict]:
    """Translate OpenAI-style messages into GigaChat-style messages.

    Resolves ``tool`` results (which carry only a ``tool_call_id``) back to the
    function name announced by the preceding assistant ``tool_calls``.
    """
    # First pass: map tool_call_id -> function name from assistant tool_calls.
    id_to_name: Dict[str, str] = {}
    for m in messages:
        for call in m.get("tool_calls") or ():
            cid = call.get("id")
            name = (call.get("function") or {}).get("name")
            if cid and name:
                id_to_name[cid] = name

    out: List[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):  # OpenAI multimodal parts -> flatten text
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )

        if role == "tool":
            name = m.get("name") or id_to_name.get(m.get("tool_call_id", ""), "")
            out.append({"role": "function", "name": name, "content": content or ""})
            continue

        if role == "assistant" and m.get("tool_calls"):
            call = m["tool_calls"][0]  # GigaChat handles one call at a time
            fn = call.get("function") or {}
            out.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "function_call": {
                        "name": fn.get("name", ""),
                        "arguments": _arguments_to_dict(fn.get("arguments")),
                    },
                }
            )
            continue

        # Legacy single function_call on an assistant message.
        if role == "assistant" and m.get("function_call"):
            fc = m["function_call"]
            out.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "function_call": {
                        "name": fc.get("name", ""),
                        "arguments": _arguments_to_dict(fc.get("arguments")),
                    },
                }
            )
            continue

        out.append({"role": role, "content": content or ""})
    return out


def to_gigachat_functions(
    tools: Optional[Iterable[dict]] = None,
    functions: Optional[Iterable[dict]] = None,
) -> List[dict]:
    """Map OpenAI ``tools`` (or legacy ``functions``) to GigaChat ``functions``."""
    out: List[dict] = []
    if tools:
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function") or {}
            out.append(_function_spec(fn))
    elif functions:
        for fn in functions:
            out.append(_function_spec(fn))
    return out


def _function_spec(fn: dict) -> dict:
    spec: Dict[str, Any] = {"name": fn.get("name", "")}
    if fn.get("description"):
        spec["description"] = fn["description"]
    # GigaChat requires a parameters object; default to an empty schema.
    spec["parameters"] = fn.get("parameters") or {"type": "object", "properties": {}}
    return spec


def to_gigachat_function_call(tool_choice: Any) -> Optional[Any]:
    """Translate OpenAI ``tool_choice`` / ``function_call`` to GigaChat."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in ("none", "auto"):
            return tool_choice
        if tool_choice == "required":  # closest GigaChat equivalent
            return "auto"
        return None
    if isinstance(tool_choice, dict):
        # {"type":"function","function":{"name":...}} or {"name":...}
        fn = tool_choice.get("function") or tool_choice
        name = fn.get("name")
        if name:
            return {"name": name}
    return None


_FINISH_REASON = {"function_call": "tool_calls", "blacklist": "content_filter"}


def _finish_reason(reason: Optional[str]) -> Optional[str]:
    if reason is None:
        return None
    return _FINISH_REASON.get(reason, reason)


def _message_to_openai(message: dict) -> Tuple[dict, bool]:
    """Return (openai_message, has_tool_call)."""
    fc = message.get("function_call")
    if fc:
        call_id = "call_" + uuid.uuid4().hex[:24]
        return (
            {
                "role": "assistant",
                "content": message.get("content") or None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": _arguments_to_str(fc.get("arguments")),
                        },
                    }
                ],
            },
            True,
        )
    return ({"role": "assistant", "content": message.get("content") or ""}, False)


def to_openai_response(
    giga_response: dict,
    *,
    model: str,
    completion_id: Optional[str] = None,
    created: Optional[int] = None,
) -> dict:
    """Translate a GigaChat ``ChatCompletion`` dict into an OpenAI response."""
    choices_in = giga_response.get("choices") or []
    choices_out: List[dict] = []
    for choice in choices_in:
        message, has_call = _message_to_openai(choice.get("message") or {})
        reason = _finish_reason(choice.get("finish_reason"))
        if has_call and reason in (None, "stop"):
            reason = "tool_calls"
        choices_out.append(
            {
                "index": choice.get("index", 0),
                "message": message,
                "finish_reason": reason or "stop",
            }
        )

    out: Dict[str, Any] = {
        "id": completion_id or new_completion_id(),
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": choices_out,
    }
    usage = giga_response.get("usage")
    if isinstance(usage, dict):
        out["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    return out


def to_openai_stream(
    chunks: Iterable[dict],
    *,
    model: str,
    completion_id: Optional[str] = None,
    created: Optional[int] = None,
) -> Iterator[dict]:
    """Translate GigaChat stream chunks into OpenAI ``chat.completion.chunk`` dicts.

    Emits an initial role delta, then content/tool-call deltas, and a final
    chunk carrying ``finish_reason``.
    """
    cid = completion_id or new_completion_id()
    ts = created or int(time.time())
    role_sent = False
    final_reason: Optional[str] = None

    def envelope(delta: dict, finish_reason: Optional[str]) -> dict:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": ts,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }

    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        reason = _finish_reason(choice.get("finish_reason"))
        if reason is not None:
            final_reason = reason

        if not role_sent:
            yield envelope({"role": "assistant"}, None)
            role_sent = True

        fc = delta.get("function_call")
        if fc:
            yield envelope(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_" + uuid.uuid4().hex[:24],
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": _arguments_to_str(fc.get("arguments")),
                            },
                        }
                    ]
                },
                None,
            )
            if final_reason is None:
                final_reason = "tool_calls"
        elif delta.get("content"):
            yield envelope({"content": delta["content"]}, None)

    if not role_sent:  # empty stream — still emit a well-formed start
        yield envelope({"role": "assistant"}, None)
    yield envelope({}, final_reason or "stop")
