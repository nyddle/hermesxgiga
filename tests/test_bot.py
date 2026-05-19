import io

from hermesxgiga import HermesBot, Message
from hermesxgiga.transports import ConsoleTransport


class FakeEngine:
    """Records the prompts it receives and echoes a canned reply."""

    def __init__(self, reply="reply"):
        self.reply = reply
        self.prompts = []

    def chat(self, messages):
        self.prompts.append(list(messages))
        return self.reply


def test_handle_accumulates_history_with_system_prompt():
    engine = FakeEngine("R1")
    bot = HermesBot(engine, system_prompt="sys")

    assert bot.handle("c1", "first") == "R1"
    engine.reply = "R2"
    assert bot.handle("c1", "second") == "R2"

    # First prompt: system + user.
    assert engine.prompts[0] == [
        Message("system", "sys"),
        Message("user", "first"),
    ]
    # Second prompt: system + full prior turn + new user message.
    assert engine.prompts[1] == [
        Message("system", "sys"),
        Message("user", "first"),
        Message("assistant", "R1"),
        Message("user", "second"),
    ]
    assert bot.history("c1") == [
        Message("user", "first"),
        Message("assistant", "R1"),
        Message("user", "second"),
        Message("assistant", "R2"),
    ]


def test_history_is_isolated_per_chat():
    bot = HermesBot(FakeEngine())
    bot.handle("a", "hi")
    assert bot.history("a")
    assert bot.history("b") == []


def test_history_limit_trims_old_turns():
    engine = FakeEngine("ok")
    bot = HermesBot(engine, history_limit=2)
    for i in range(5):
        bot.handle("c", f"m{i}")
    hist = bot.history("c")
    # 2 turns -> 4 entries kept.
    assert len(hist) == 4
    assert hist[0] == Message("user", "m3")
    assert hist[-1] == Message("assistant", "ok")


def test_reset_clears_history():
    bot = HermesBot(FakeEngine())
    bot.handle("c", "hi")
    bot.reset("c")
    assert bot.history("c") == []


def test_run_drives_transport_end_to_end():
    engine = FakeEngine("ответ")
    bot = HermesBot(engine)
    stdin = io.StringIO("hello\n/exit\n")
    stdout = io.StringIO()
    transport = ConsoleTransport(stdin=stdin, stdout=stdout)

    bot.run(transport)

    out = stdout.getvalue()
    assert "bot> ответ" in out
    assert engine.prompts[0] == [Message("user", "hello")]
