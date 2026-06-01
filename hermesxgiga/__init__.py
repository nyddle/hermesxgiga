"""hermesxgiga: connect GigaChat to a Hermes-style messenger bot."""

from .bot import (
    ChatEngine,
    HermesBot,
    StreamingEngine,
    Tool,
    ToolCallingEngine,
    Transport,
)
from .gigachat import (
    DEFAULT_MODEL,
    DEFAULT_SCOPE,
    ChatResult,
    GigaChatClient,
    GigaChatError,
    Message,
    ToolCall,
)
from .transports import ConsoleTransport

__version__ = "0.1.0"

__all__ = [
    "HermesBot",
    "ChatEngine",
    "StreamingEngine",
    "ToolCallingEngine",
    "Tool",
    "Transport",
    "GigaChatClient",
    "GigaChatError",
    "Message",
    "ChatResult",
    "ToolCall",
    "ConsoleTransport",
    "DEFAULT_SCOPE",
    "DEFAULT_MODEL",
    "__version__",
]
