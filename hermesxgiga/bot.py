"""Hermes bot orchestrator.

``HermesBot`` is the glue between a messenger transport and GigaChat. It keeps
per-chat conversation history, prepends an optional system prompt, asks
GigaChat for a reply and trims history so it stays bounded.

The bot is transport-agnostic: it talks to anything implementing the
:class:`Transport` protocol, so the same bot can run on a console, a
Telegram/Hermes adapter, a test double, etc.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Protocol, runtime_checkable

from .gigachat import Message


@runtime_checkable
class ChatEngine(Protocol):
    """Anything that turns a list of messages into a reply string.

    :class:`hermesxgiga.gigachat.GigaChatClient` satisfies this protocol;
    tests can pass a lightweight fake.
    """

    def chat(self, messages: List[Message]) -> str:  # pragma: no cover - proto
        ...


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
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be >= 1")
        self.engine = engine
        self.system_prompt = system_prompt
        self.history_limit = history_limit
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

    def handle(self, chat_id: str, text: str) -> str:
        """Process an incoming user message and return the assistant reply."""
        history = self._history[chat_id]

        prompt: List[Message] = []
        if self.system_prompt:
            prompt.append(Message(role="system", content=self.system_prompt))
        prompt.extend(history)
        prompt.append(Message(role="user", content=text))

        reply = self.engine.chat(prompt)

        history.append(Message(role="user", content=text))
        history.append(Message(role="assistant", content=reply))
        return reply

    def run(self, transport: Transport) -> None:
        """Run the bot on ``transport`` until the transport stops."""

        def _handler(chat_id: str, text: str) -> None:
            reply = self.handle(chat_id, text)
            transport.send(chat_id, reply)

        transport.run(_handler)
