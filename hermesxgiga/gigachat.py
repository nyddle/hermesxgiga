"""Minimal GigaChat API client (OAuth2 + chat completions).

GigaChat is Sber's LLM. Authentication is a two-step OAuth flow:

1. Exchange an "Authorization Key" (base64 of ``client_id:client_secret``)
   for a short-lived access token at the NGW OAuth endpoint.
2. Call the chat completions endpoint with ``Authorization: Bearer <token>``.

Sber serves these endpoints behind the "Russian Trusted" root CA, which is
usually not in the default trust store, so TLS verification can be disabled
or pointed at a custom CA bundle.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Union

import requests

DEFAULT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_MODEL = "GigaChat"


class GigaChatError(RuntimeError):
    """Raised when the GigaChat API returns an error or an unexpected payload."""


@dataclass
class Message:
    """A single chat message in the OpenAI-style ``role``/``content`` format."""

    role: str
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# Anything the client accepts as a list of messages.
MessageLike = Union[Message, dict]


@dataclass
class _Token:
    value: str
    expires_at: float  # epoch seconds

    def is_valid(self, *, leeway: float = 30.0) -> bool:
        return bool(self.value) and time.time() < (self.expires_at - leeway)


class GigaChatClient:
    """Synchronous GigaChat client with automatic token caching/refresh.

    Parameters
    ----------
    auth_key:
        The base64-encoded ``client_id:client_secret`` issued by Sber
        (shown in the GigaChat portal as the "Authorization Key").
    scope:
        API scope: ``GIGACHAT_API_PERS`` (personal), ``GIGACHAT_API_B2B``
        or ``GIGACHAT_API_CORP``.
    model:
        Default model name used when a request does not override it.
    verify_ssl:
        Passed to ``requests`` for both calls. ``False`` disables TLS
        verification; a string is treated as a path to a CA bundle.
    """

    def __init__(
        self,
        auth_key: str,
        *,
        scope: str = DEFAULT_SCOPE,
        model: str = DEFAULT_MODEL,
        verify_ssl: Union[bool, str] = True,
        oauth_url: str = DEFAULT_OAUTH_URL,
        api_url: str = DEFAULT_API_URL,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not auth_key:
            raise ValueError("auth_key is required")
        self.auth_key = auth_key
        self.scope = scope
        self.model = model
        self.verify_ssl = verify_ssl
        self.oauth_url = oauth_url
        self.api_url = api_url
        self.timeout = timeout
        self._session = session or requests.Session()
        self._token: Optional[_Token] = None

    # -- auth -----------------------------------------------------------------

    def _fetch_token(self) -> _Token:
        headers = {
            "Authorization": f"Basic {self.auth_key}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        try:
            resp = self._session.post(
                self.oauth_url,
                headers=headers,
                data={"scope": self.scope},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        except requests.RequestException as exc:  # pragma: no cover - network
            raise GigaChatError(f"OAuth request failed: {exc}") from exc

        if resp.status_code != 200:
            raise GigaChatError(
                f"OAuth failed with status {resp.status_code}: {resp.text}"
            )

        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise GigaChatError(f"OAuth response missing access_token: {payload}")

        # ``expires_at`` is a Unix timestamp in milliseconds. Fall back to a
        # conservative 25-minute lifetime when it is absent.
        expires_at_ms = payload.get("expires_at")
        if expires_at_ms:
            expires_at = float(expires_at_ms) / 1000.0
        else:
            expires_at = time.time() + 25 * 60
        return _Token(value=token, expires_at=expires_at)

    def _access_token(self, *, force_refresh: bool = False) -> str:
        if force_refresh or self._token is None or not self._token.is_valid():
            self._token = self._fetch_token()
        return self._token.value

    # -- chat -----------------------------------------------------------------

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

    def chat(
        self,
        messages: Sequence[MessageLike],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat request and return the assistant's reply text.

        On a ``401`` the access token is refreshed once and the request is
        retried, which transparently handles token expiry.
        """

        body: dict = {
            "model": model or self.model,
            "messages": self._normalize(messages),
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        for attempt in range(2):
            token = self._access_token(force_refresh=attempt == 1)
            try:
                resp = self._session.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=body,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
            except requests.RequestException as exc:  # pragma: no cover - network
                raise GigaChatError(f"Chat request failed: {exc}") from exc

            if resp.status_code == 401 and attempt == 0:
                continue  # token likely expired; refresh and retry once
            if resp.status_code != 200:
                raise GigaChatError(
                    f"Chat failed with status {resp.status_code}: {resp.text}"
                )

            payload = resp.json()
            try:
                return payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise GigaChatError(
                    f"Unexpected chat response shape: {payload}"
                ) from exc

        raise GigaChatError("Chat failed: unauthorized after token refresh")
