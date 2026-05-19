"""Run the Hermes bot in the console, backed by GigaChat.

Usage:

    export GIGACHAT_AUTH_KEY="<base64 client_id:client_secret>"
    # Sber endpoints use the Russian Trusted root CA; disable verification
    # or point GIGACHAT_CA_BUNDLE at the proper bundle.
    export GIGACHAT_VERIFY_SSL=0
    python -m examples.console_chat
"""

import os

from hermesxgiga import ConsoleTransport, GigaChatClient, HermesBot


def main() -> None:
    auth_key = os.environ.get("GIGACHAT_AUTH_KEY")
    if not auth_key:
        raise SystemExit("set GIGACHAT_AUTH_KEY first")

    ca_bundle = os.environ.get("GIGACHAT_CA_BUNDLE")
    if ca_bundle:
        verify: object = ca_bundle
    else:
        verify = os.environ.get("GIGACHAT_VERIFY_SSL", "1") not in ("0", "false", "")

    client = GigaChatClient(
        auth_key,
        scope=os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
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
