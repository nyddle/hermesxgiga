"""Hermes bot orchestrator.

``HermesBot`` is the glue between a messenger transport and GigaChat. It keeps
per-chat conversation history, prepends an optional system prompt, asks
GigaChat for a reply and trims history so it stays bounded.

The bot is transport-agnostic: it talks to anything implementing the
:class:`Transport` protocol, so the same bot can run on a console, a
Telegram/Hermes adapter, a test double, etc.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .gigachat import ChatResult, Message, ToolCall


@runtime_checkable
class ChatEngine(Protocol):
    """Anything that turns a list of messages into a reply string.

    :class:`hermesxgiga.gigachat.GigaChatClient` satisfies this protocol;
    tests can pass a lightweight fake.
    """

    def chat(self, messages: List[Message]) -> str:  # pragma: no cover - proto
        ...


@runtime_checkable
class StreamingEngine(Protocol):
    """An engine that can stream a reply as text deltas."""

    def stream_chat(  # pragma: no cover - proto
        self, messages: List[Message]
    ) -> Iterator[str]:
        ...


@runtime_checkable
class ToolCallingEngine(Protocol):
    """An engine that can surface function/tool calls."""

    def complete(  # pragma: no cover - proto
        self, messages: List[Message], *, functions=None, function_call=None
    ) -> ChatResult:
        ...


@dataclass
class Tool:
    """A function the bot can expose to the model and execute locally.

    ``parameters`` is a JSON-Schema object describing the arguments; ``handler``
    receives the parsed arguments dict and returns a JSON-serialisable result
    (or a plain string) that is fed back to the model.
    """

    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], object]

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@runtime_checkable
class Transport(Protocol):
    """A messenger transport the bot can send replies through and listen on."""

    def send(self, chat_id: str, text: str) -> None:  # pragma: no cover - proto
        """Deliver ``text`` to ``chat_id``."""

    def run(self, handler) -> None:  # pragma: no cover - proto
        """Block, feeding each (chat_id, text) incoming message to ``handler``.

        ``handler`` is ``Callable[[str, str], None]``.
        """


class HermesBot:
    """Conversational bot backed by a :class:`ChatEngine`.

    Parameters
    ----------
    engine:
        The chat backend (e.g. ``GigaChatClient``).
    system_prompt:
        Optional system message prepended to every conversation.
    history_limit:
        Maximum number of *user+assistant* turns kept per chat. Older turns
        are dropped so requests stay bounded.
    """

    def __init__(
        self,
        engine: ChatEngine,
        *,
        system_prompt: Optional[str] = None,
        history_limit: int = 20,
        tools: Optional[Sequence[Tool]] = None,
        max_tool_iterations: int = 5,
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be >= 1")
        if max_tool_iterations < 1:
            raise ValueError("max_tool_iterations must be >= 1")
        self.engine = engine
        self.system_prompt = system_prompt
        self.history_limit = history_limit
        self.max_tool_iterations = max_tool_iterations
        self._tools: Dict[str, Tool] = {t.name: t for t in (tools or ())}
        # Each turn (user msg + assistant reply) is two entries.
        self._history: Dict[str, Deque[Message]] = defaultdict(
            lambda: deque(maxlen=history_limit * 2)
        )

    def reset(self, chat_id: str) -> None:
        """Forget the conversation history for ``chat_id``."""
        self._history.pop(chat_id, None)

    def history(self, chat_id: str) -> List[Message]:
        """Return a copy of the stored history for ``chat_id``."""
        return list(self._history.get(chat_id, ()))

    def _build_prompt(self, history: Deque[Message], text: str) -> List[Message]:
        prompt: List[Message] = []
        if self.system_prompt:
            prompt.append(Message(role="system", content=self.system_prompt))
        prompt.extend(history)
        prompt.append(Message(role="user", content=text))
        return prompt

    def _record_turn(self, history: Deque[Message], text: str, reply: str) -> None:
        history.append(Message(role="user", content=text))
        history.append(Message(role="assistant", content=reply))

    def handle(self, chat_id: str, text: str) -> str:
        """Process an incoming user message and return the assistant reply.

        When the bot has registered tools, this runs a function-calling loop:
        the model may ask for a tool, the bot executes it locally and feeds the
        result back, until the model produces a text answer (bounded by
        ``max_tool_iterations``). Intermediate tool messages are not persisted
        to history — only the user message and the final reply are.
        """
        history = self._history[chat_id]

        if not self._tools:
            reply = self.engine.chat(self._build_prompt(history, text))
            self._record_turn(history, text, reply)
            return reply

        working = self._build_prompt(history, text)
        schemas = [t.schema() for t in self._tools.values()]
        reply = ""
        for _ in range(self.max_tool_iterations):
            result = self.engine.complete(
                working, functions=schemas, function_call="auto"
            )
            if result.tool_call is None:
                reply = result.content or ""
                break
            tc = result.tool_call
            working.append(
                Message(
                    role="assistant",
                    function_call={"name": tc.name, "arguments": tc.arguments},
                )
            )
            output = self._run_tool(tc)
            working.append(Message(role="function", content=output, name=tc.name))
        else:
            # Iterations exhausted; fall back to whatever text the model gave.
            reply = (result.content or "") if result else ""

        self._record_turn(history, text, reply)
        return reply

    def _run_tool(self, tool_call: ToolCall) -> str:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return json.dumps(
                {"error": f"unknown function: {tool_call.name}"}, ensure_ascii=False
            )
        try:
            result = tool.handler(tool_call.arguments or {})
        except Exception as exc:  # surface tool failure to the model, not crash
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    def stream(self, chat_id: str, text: str) -> Iterator[str]:
        """Stream the assistant reply as text deltas, then persist the turn.

        Tools are not applied on the streaming path in this version; use
        :meth:`handle` for function calling.
        """
        history = self._history[chat_id]
        prompt = self._build_prompt(history, text)

        parts: List[str] = []
        for delta in self.engine.stream_chat(prompt):
            parts.append(delta)
            yield delta

        self._record_turn(history, text, "".join(parts))

    def run(self, transport: Transport) -> None:
        """Run the bot on ``transport`` until the transport stops."""

        def _handler(chat_id: str, text: str) -> None:
            reply = self.handle(chat_id, text)
            transport.send(chat_id, reply)

        transport.run(_handler)
