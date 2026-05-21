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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

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
        The base64-encoded ``client_id:client_secret`` issued by Sber
        (the "Authorization Key" from the GigaChat portal). Passed to the
        SDK as ``credentials``.
    scope:
        ``GIGACHAT_API_PERS`` (personal), ``GIGACHAT_API_B2B`` or
        ``GIGACHAT_API_CORP``.
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
        scope: str = DEFAULT_SCOPE,
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

        if not auth_key:
            raise ValueError("auth_key is required")

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

        self._giga = GigaChat(
            credentials=auth_key,
            scope=scope,
            model=model,
            **sdk_kwargs,
        )

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
    def _to_dict(obj: object) -> Any:
        """Coerce an SDK pydantic model (or anything) into plain JSON types."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return obj
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):  # pydantic v2: JSON mode coerces enums to str
            return model_dump(mode="json")
        as_dict = getattr(obj, "dict", None)
        if callable(as_dict):  # pydantic v1 fallback
            return as_dict()
        return obj

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

    def _build_payload(
        self,
        messages: Sequence[MessageLike],
        *,
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float] = None,
        functions: Optional[Sequence[dict]] = None,
        function_call: Optional[Union[str, dict]] = None,
        stream: bool = False,
    ) -> dict:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": self._normalize(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if functions:
            payload["functions"] = list(functions)
        if function_call is not None:
            payload["function_call"] = function_call
        if stream:
            payload["stream"] = True
        return payload

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
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._extract(self._call(payload))

    def complete(
        self,
        messages: Sequence[MessageLike],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        functions: Optional[Sequence[dict]] = None,
        function_call: Optional[Union[str, dict]] = None,
    ) -> dict:
        """Send a chat request and return the full response as a plain dict.

        Unlike :meth:`chat`, this exposes ``function_call``, ``finish_reason``
        and ``usage`` so callers (e.g. the OpenAI-compatible server) can map
        tool calls. The shape mirrors the GigaChat SDK ``ChatCompletion``.
        """
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            functions=functions,
            function_call=function_call,
        )
        return self._to_dict(self._call(payload))

    def stream(
        self,
        messages: Sequence[MessageLike],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        functions: Optional[Sequence[dict]] = None,
        function_call: Optional[Union[str, dict]] = None,
    ) -> Iterator[dict]:
        """Stream a chat response, yielding each chunk as a plain dict."""
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            functions=functions,
            function_call=function_call,
            stream=True,
        )
        try:
            for chunk in self._giga.stream(payload):
                yield self._to_dict(chunk)
        except GigaChatError:
            raise
        except Exception as exc:  # SDK/transport/auth errors -> stable surface
            raise GigaChatError(f"GigaChat stream failed: {exc}") from exc

    def list_models(self) -> List[str]:
        """Return the ids of models available to these credentials."""
        try:
            models = self._to_dict(self._giga.get_models())
        except GigaChatError:
            raise
        except Exception as exc:
            raise GigaChatError(f"GigaChat get_models failed: {exc}") from exc
        data = models.get("data", []) if isinstance(models, dict) else []
        ids: List[str] = []
        for item in data:
            item = self._to_dict(item)
            mid = item.get("id_") or item.get("id") if isinstance(item, dict) else None
            if mid:
                ids.append(mid)
        return ids

    def _call(self, payload: dict) -> object:
        try:
            return self._giga.chat(payload)
        except GigaChatError:
            raise
        except Exception as exc:  # SDK/transport/auth errors -> stable surface
            raise GigaChatError(f"GigaChat request failed: {exc}") from exc

    # -- resource management --------------------------------------------------

    def close(self) -> None:
        close = getattr(self._giga, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "GigaChatClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
