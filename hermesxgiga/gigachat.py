"""GigaChat client backed by the official ai-forever/gigachat SDK.

This is a thin adapter over :class:`gigachat.GigaChat`. The official SDK
already handles the OAuth2 token exchange, token caching/refresh, the
Russian "Trusted" TLS chain and streaming, so this class only:

* maps our public :class:`Message` / dict format onto the SDK payload,
* validates input,
* exposes a stable ``chat(messages) -> str`` surface (the ``ChatEngine``
  protocol :class:`hermesxgiga.bot.HermesBot` depends on),
* normalises SDK exceptions into :class:`GigaChatError`.

The SDK is imported lazily so ``import hermesxgiga`` works without it, and
so tests can inject a fake client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

DEFAULT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_MODEL = "GigaChat"


class GigaChatError(RuntimeError):
    """Raised when the underlying GigaChat call fails."""


@dataclass
class Message:
    """A single chat message in the ``role``/``content`` format."""

    role: str
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


MessageLike = Union[Message, dict]


class GigaChatClient:
    """Adapter over the official ``gigachat.GigaChat`` SDK.

    Parameters
    ----------
    auth_key:
        Base64-encoded ``client_id:client_secret`` ("Authorization Key"
        from the GigaChat portal). Mapped to the SDK's ``credentials``.
    user, password:
        Alternative auth: GigaChat ``user``/``password`` pair. Either
        ``auth_key``, ``user``+``password`` or ``access_token`` must be
        supplied.
    access_token:
        Alternative auth: pre-issued Bearer access token.
    scope:
        ``GIGACHAT_API_PERS`` (personal), ``GIGACHAT_API_B2B`` or
        ``GIGACHAT_API_CORP``. ``None`` lets the SDK decide.
    model:
        Default model name.
    verify_ssl:
        ``bool`` -> SDK ``verify_ssl_certs``. A ``str`` is treated as a
        path to a CA bundle -> SDK ``ca_bundle_file``.
    client:
        Pre-built ``gigachat.GigaChat``-like object. When given, the SDK
        is not imported/instantiated (used by tests and advanced setups).
    **sdk_kwargs:
        Forwarded verbatim to ``gigachat.GigaChat(...)``.
    """

    def __init__(
        self,
        auth_key: str = "",
        *,
        user: Optional[str] = None,
        password: Optional[str] = None,
        access_token: Optional[str] = None,
        scope: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        verify_ssl: Union[bool, str] = True,
        client: Optional[object] = None,
        **sdk_kwargs,
    ) -> None:
        self.scope = scope
        self.model = model

        if client is not None:
            self._giga = client
            return

        has_creds = bool(auth_key)
        has_userpw = bool(user) and bool(password)
        has_token = bool(access_token)
        if not (has_creds or has_userpw or has_token):
            raise ValueError(
                "provide one of: auth_key, user/password, or access_token"
            )

        try:
            from gigachat import GigaChat
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "the official GigaChat SDK is required: pip install gigachat"
            ) from exc

        if isinstance(verify_ssl, str):
            sdk_kwargs.setdefault("ca_bundle_file", verify_ssl)
        else:
            sdk_kwargs.setdefault("verify_ssl_certs", bool(verify_ssl))

        init_kwargs: dict = {"model": model, **sdk_kwargs}
        if has_creds:
            init_kwargs["credentials"] = auth_key
        if has_userpw:
            init_kwargs["user"] = user
            init_kwargs["password"] = password
        if has_token:
            init_kwargs["access_token"] = access_token
        if scope is not None:
            init_kwargs["scope"] = scope

        self._giga = GigaChat(**init_kwargs)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _normalize(messages: Iterable[MessageLike]) -> List[dict]:
        out: List[dict] = []
        for m in messages:
            if isinstance(m, Message):
                out.append(m.as_dict())
            elif isinstance(m, dict):
                if "role" not in m or "content" not in m:
                    raise ValueError(f"message dict needs role/content: {m!r}")
                out.append({"role": m["role"], "content": m["content"]})
            else:  # pragma: no cover - defensive
                raise TypeError(f"unsupported message type: {type(m)!r}")
        if not out:
            raise ValueError("messages must not be empty")
        return out

    @staticmethod
    def _extract(response: object) -> str:
        # The SDK returns a pydantic ChatCompletion; be tolerant of a plain
        # dict too (older versions / fakes).
        try:
            if isinstance(response, dict):
                return response["choices"][0]["message"]["content"]
            return response.choices[0].message.content  # type: ignore[attr-defined]
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise GigaChatError(
                f"Unexpected chat response shape: {response!r}"
            ) from exc

    # -- chat -----------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[MessageLike],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat request and return the assistant's reply text."""
        payload: dict = {
            "model": model or self.model,
            "messages": self._normalize(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = self._giga.chat(payload)
        except GigaChatError:
            raise
        except Exception as exc:  # SDK/transport/auth errors -> stable surface
            raise GigaChatError(f"GigaChat request failed: {exc}") from exc

        return self._extract(response)

    # -- resource management --------------------------------------------------

    def close(self) -> None:
        close = getattr(self._giga, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "GigaChatClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
