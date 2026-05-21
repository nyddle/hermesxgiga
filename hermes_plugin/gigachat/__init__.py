"""GigaChat provider profile for the Hermes agent.

GigaChat is not OpenAI-compatible (different request/response shapes, and
OAuth2 token exchange instead of a static Bearer key), so Hermes cannot point
its chat_completions transport straight at Sber's endpoint. Instead, run the
``hermesxgiga`` proxy (``python -m hermesxgiga.server``) — it speaks OpenAI on
the front and uses the official ``ai-forever/gigachat`` SDK on the back — and
this profile points Hermes at that proxy.

Install: copy this directory to
``$HERMES_HOME/plugins/model-providers/gigachat/`` (default
``~/.hermes/plugins/model-providers/gigachat/``), then::

    hermes model gigachat/GigaChat-Pro

Override the proxy URL / key with the env vars below (e.g. if the proxy runs
on another host or you set ``HERMESXGIGA_API_KEY``).
"""

from providers import register_provider
from providers.base import ProviderProfile


gigachat = ProviderProfile(
    name="gigachat",
    aliases=("giga", "sber"),
    display_name="GigaChat (Sber)",
    description="GigaChat via the local hermesxgiga OpenAI-compatible proxy",
    signup_url="https://developers.sber.ru/portal/products/gigachat-api",
    # First env var = API key sent to the proxy (matches HERMESXGIGA_API_KEY,
    # if you set one). Second = base_url override for the proxy.
    env_vars=("GIGACHAT_PROXY_API_KEY", "GIGACHAT_PROXY_BASE_URL"),
    base_url="http://127.0.0.1:8000/v1",
    auth_type="api_key",
    default_aux_model="GigaChat",
    fallback_models=(
        "GigaChat",
        "GigaChat-Pro",
        "GigaChat-Max",
        "GigaChat-2",
        "GigaChat-2-Pro",
        "GigaChat-2-Max",
    ),
)

register_provider(gigachat)
