import io

import pytest

from hermesxgiga import ChatResult, HermesBot, Message, Tool, ToolCall
from hermesxgiga.transports import ConsoleTransport


class FakeEngine:
    """Records the prompts it receives and echoes a canned reply."""

    def __init__(self, reply="reply"):
        self.reply = reply
        self.prompts = []

    def chat(self, messages):
        self.prompts.append(list(messages))
        return self.reply


class ScriptedEngine:
    """Returns a queued sequence of ChatResults from ``complete``."""

    def __init__(self, results):
        self._results = list(results)
        self.prompts = []
        self.functions = []

    def complete(self, messages, *, functions=None, function_call=None):
        self.prompts.append(list(messages))
        self.functions.append(functions)
        return self._results.pop(0)


class StreamEngine:
    """Yields canned text deltas from ``stream_chat``."""

    def __init__(self, deltas):
        self._deltas = list(deltas)
        self.prompts = []

    def stream_chat(self, messages):
        self.prompts.append(list(messages))
        for d in self._deltas:
            yield d


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


def test_tool_loop_executes_tool_then_returns_text():
    calls = []

    def weather(args):
        calls.append(args)
        return {"temp": 7}

    engine = ScriptedEngine(
        [
            ChatResult(content=None, tool_call=ToolCall("get_weather", {"city": "Мск"})),
            ChatResult(content="В Москве +7."),
        ]
    )
    tool = Tool("get_weather", "погода", {"type": "object", "properties": {}}, weather)
    bot = HermesBot(engine, tools=[tool])

    assert bot.handle("c", "погода?") == "В Москве +7."
    # tool handler was invoked with the model-supplied arguments.
    assert calls == [{"city": "Мск"}]
    # functions schema was passed on each model call.
    assert engine.functions[0][0]["name"] == "get_weather"
    # second model call saw the assistant function_call + function result.
    second = engine.prompts[1]
    assert second[-2] == Message(
        "assistant", function_call={"name": "get_weather", "arguments": {"city": "Мск"}}
    )
    assert second[-1] == Message("function", content='{"temp": 7}', name="get_weather")
    # only the user message and final reply are persisted (no tool plumbing).
    assert bot.history("c") == [
        Message("user", "погода?"),
        Message("assistant", "В Москве +7."),
    ]


def test_unknown_tool_is_reported_to_model_not_raised():
    engine = ScriptedEngine(
        [
            ChatResult(content=None, tool_call=ToolCall("nope", {})),
            ChatResult(content="done"),
        ]
    )
    bot = HermesBot(engine, tools=[])  # tools enabled but registry empty
    # empty registry means no tools -> simple path; force tool path via a stub.
    bot._tools = {"x": Tool("x", "", {}, lambda a: "y")}
    assert bot.handle("c", "go") == "done"
    func_msg = engine.prompts[1][-1]
    assert func_msg.role == "function"
    assert "unknown function: nope" in func_msg.content


def test_tool_iterations_are_bounded():
    # Model keeps asking for a tool forever; loop must stop and not hang.
    loop_results = [
        ChatResult(content="partial", tool_call=ToolCall("t", {})) for _ in range(10)
    ]
    engine = ScriptedEngine(loop_results)
    tool = Tool("t", "", {"type": "object", "properties": {}}, lambda a: "ok")
    bot = HermesBot(engine, tools=[tool], max_tool_iterations=3)
    reply = bot.handle("c", "x")
    # fell back to the last result's text after exhausting iterations.
    assert reply == "partial"
    assert len(engine.prompts) == 3


def test_stream_yields_deltas_and_persists_turn():
    engine = StreamEngine(["At", "om", "!"])
    bot = HermesBot(engine, system_prompt="sys")

    out = list(bot.stream("c", "hi"))

    assert out == ["At", "om", "!"]
    assert engine.prompts[0] == [Message("system", "sys"), Message("user", "hi")]
    assert bot.history("c") == [
        Message("user", "hi"),
        Message("assistant", "Atom!"),
    ]


def test_max_tool_iterations_validated():
    with pytest.raises(ValueError, match="max_tool_iterations"):
        HermesBot(FakeEngine(), tools=[], max_tool_iterations=0)
