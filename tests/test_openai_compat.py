import json

from hermesxgiga.openai_compat import (
    to_gigachat_function_call,
    to_gigachat_functions,
    to_gigachat_messages,
    to_openai_response,
    to_openai_stream,
)


def test_messages_basic_roles():
    out = to_gigachat_messages(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
    )
    assert out == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_assistant_tool_call_becomes_function_call_with_dict_args():
    out = to_gigachat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city": "Moscow"}'},
                    }
                ],
            }
        ]
    )
    assert out == [
        {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "weather", "arguments": {"city": "Moscow"}},
        }
    ]


def test_tool_result_resolves_name_from_preceding_call():
    out = to_gigachat_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "+5C"},
        ]
    )
    assert out[-1] == {"role": "function", "name": "weather", "content": "+5C"}


def test_multimodal_content_parts_flattened_to_text():
    out = to_gigachat_messages(
        [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    )
    assert out == [{"role": "user", "content": "ab"}]


def test_tools_mapped_to_functions():
    fns = to_gigachat_functions(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            },
            {"type": "other", "function": {"name": "ignored"}},
        ]
    )
    assert fns == [
        {
            "name": "f",
            "description": "d",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
    ]


def test_legacy_functions_and_default_parameters():
    fns = to_gigachat_functions(functions=[{"name": "f"}])
    assert fns == [{"name": "f", "parameters": {"type": "object", "properties": {}}}]


def test_tool_choice_translation():
    assert to_gigachat_function_call("auto") == "auto"
    assert to_gigachat_function_call("none") == "none"
    assert to_gigachat_function_call("required") == "auto"
    assert to_gigachat_function_call({"type": "function", "function": {"name": "f"}}) == {"name": "f"}
    assert to_gigachat_function_call(None) is None


def test_response_plain_text():
    giga = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    out = to_openai_response(giga, model="GigaChat", completion_id="x", created=10)
    assert out["id"] == "x"
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def test_response_function_call_becomes_tool_calls():
    giga = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "function_call": {"name": "weather", "arguments": {"city": "Moscow"}},
                },
                "finish_reason": "function_call",
            }
        ]
    }
    out = to_openai_response(giga, model="GigaChat")
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Moscow"}
    assert tc["id"].startswith("call_")


def test_stream_content_chunks():
    giga_chunks = [
        {"choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    out = list(to_openai_stream(giga_chunks, model="GigaChat", completion_id="x", created=5))
    assert out[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert out[1]["choices"][0]["delta"] == {"content": "Hel"}
    assert out[2]["choices"][0]["delta"] == {"content": "lo"}
    assert out[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in out)
    assert all(c["id"] == "x" for c in out)


def test_stream_function_call_chunk():
    giga_chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"function_call": {"name": "weather", "arguments": {"city": "Moscow"}}},
                    "finish_reason": "function_call",
                }
            ]
        }
    ]
    out = list(to_openai_stream(giga_chunks, model="GigaChat"))
    tool_delta = out[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["function"]["name"] == "weather"
    assert json.loads(tool_delta["function"]["arguments"]) == {"city": "Moscow"}
    assert out[-1]["choices"][0]["finish_reason"] == "tool_calls"
