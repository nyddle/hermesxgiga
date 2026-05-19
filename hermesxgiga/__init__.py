"""hermesxgiga: connect GigaChat to a Hermes-style messenger bot."""

from .bot import ChatEngine, HermesBot, Transport
from .gigachat import (
    DEFAULT_API_URL,
    DEFAULT_MODEL,
    DEFAULT_OAUTH_URL,
    DEFAULT_SCOPE,
    GigaChatClient,
    GigaChatError,
    Message,
)
from .transports import ConsoleTransport

__version__ = "0.1.0"

__all__ = [
    "HermesBot",
    "ChatEngine",
    "Transport",
    "GigaChatClient",
    "GigaChatError",
    "Message",
    "ConsoleTransport",
    "DEFAULT_API_URL",
    "DEFAULT_OAUTH_URL",
    "DEFAULT_SCOPE",
    "DEFAULT_MODEL",
    "__version__",
]
