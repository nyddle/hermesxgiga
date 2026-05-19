"""Built-in transports.

Only a console transport is shipped here so the library has zero extra
dependencies and an end-to-end demo works out of the box. Real messenger
adapters (Telegram, Hermes, ...) just need to implement the
:class:`hermesxgiga.bot.Transport` protocol the same way.
"""

from __future__ import annotations

import sys
from typing import Callable, TextIO

Handler = Callable[[str, str], None]


class ConsoleTransport:
    """A REPL-style transport: reads lines from stdin, prints replies.

    All input is attributed to a single chat id so the bot keeps one
    conversation. Type ``/exit`` (or send EOF) to stop.
    """

    def __init__(
        self,
        chat_id: str = "console",
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        prompt: str = "you> ",
    ) -> None:
        self.chat_id = chat_id
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._prompt = prompt

    def send(self, chat_id: str, text: str) -> None:
        self._stdout.write(f"bot> {text}\n")
        self._stdout.flush()

    def run(self, handler: Handler) -> None:
        while True:
            self._stdout.write(self._prompt)
            self._stdout.flush()
            line = self._stdin.readline()
            if not line:  # EOF
                break
            text = line.strip()
            if not text:
                continue
            if text in ("/exit", "/quit"):
                break
            handler(self.chat_id, text)
