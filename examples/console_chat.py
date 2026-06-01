"""Run the Hermes bot in the console, backed by GigaChat.

Reads whatever credentials are in the environment, so it works with any of
the three GigaChat auth methods. Set one of:

    export GIGACHAT_AUTH_KEY="<base64 client_id:client_secret>"   # or
    export GIGACHAT_USER=... GIGACHAT_PASSWORD=...                # or
    export GIGACHAT_ACCESS_TOKEN=...

Sber endpoints use the Russian Trusted root CA; disable verification or point
GIGACHAT_CA_BUNDLE at the proper bundle:

    export GIGACHAT_VERIFY_SSL=0

Optional: GIGACHAT_SCOPE, GIGACHAT_MODEL, GIGACHAT_BASE_URL (e.g. the IFT
contour). Then:

    python -m examples.console_chat
"""

import os

from hermesxgiga import ConsoleTransport, GigaChatClient, HermesBot


def _have_creds() -> bool:
    if os.environ.get("GIGACHAT_AUTH_KEY"):
        return True
    if os.environ.get("GIGACHAT_USER") and os.environ.get("GIGACHAT_PASSWORD"):
        return True
    return bool(os.environ.get("GIGACHAT_ACCESS_TOKEN"))


def main() -> None:
    if not _have_creds():
        raise SystemExit(
            "set GIGACHAT_AUTH_KEY, or GIGACHAT_USER + GIGACHAT_PASSWORD, "
            "or GIGACHAT_ACCESS_TOKEN first"
        )

    ca_bundle = os.environ.get("GIGACHAT_CA_BUNDLE")
    if ca_bundle:
        verify: object = ca_bundle
    else:
        verify = os.environ.get("GIGACHAT_VERIFY_SSL", "1") not in ("0", "false", "no")

    client = GigaChatClient(
        auth_key=os.environ.get("GIGACHAT_AUTH_KEY", ""),
        user=os.environ.get("GIGACHAT_USER"),
        password=os.environ.get("GIGACHAT_PASSWORD"),
        access_token=os.environ.get("GIGACHAT_ACCESS_TOKEN"),
        scope=os.environ.get("GIGACHAT_SCOPE") or None,
        model=os.environ.get("GIGACHAT_MODEL", "GigaChat"),
        verify_ssl=verify,
    )
    bot = HermesBot(
        client,
        system_prompt="Ты — дружелюбный ассистент Hermes. Отвечай кратко.",
    )
    print("Hermes x GigaChat. Type /exit to quit.")
    bot.run(ConsoleTransport())


if __name__ == "__main__":
    main()
