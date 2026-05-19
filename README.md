# hermesxgiga

Python-библиотека/клиент, связывающая **GigaChat** (LLM от Сбера) с
**Hermes**-ботом/мессенджером. GigaChat выступает движком ответов, а бот —
транспортно-независимый: его можно подключить к консоли, Telegram, Hermes
или к тестовому двойнику.

## Установка

```bash
pip install -e .
```

Зависимость одна — официальный SDK [`gigachat`](https://github.com/ai-forever/gigachat)
от ai-forever.

## Компоненты

- `GigaChatClient` — тонкий адаптер над официальным `gigachat.GigaChat`.
  Сам SDK решает OAuth2, кэш/ротацию токена, TLS-цепочку Минцифры и
  стриминг; адаптер лишь маппит формат сообщений, валидирует ввод и
  приводит ошибки SDK к `GigaChatError`. SDK импортируется лениво —
  `import hermesxgiga` работает и без него (например, в тестах с
  инъекцией клиента через `GigaChatClient(client=...)`).
- `HermesBot` — оркестратор: хранит историю диалога по `chat_id`,
  подставляет системный промпт, обрезает историю до лимита.
- `Transport` — протокол транспорта (`send` + `run`). В комплекте
  `ConsoleTransport`; свои адаптеры (Telegram/Hermes) реализуют тот же протокол.

## Быстрый старт

```python
from hermesxgiga import GigaChatClient, HermesBot, ConsoleTransport

client = GigaChatClient("<BASE64 client_id:client_secret>", verify_ssl=False)
bot = HermesBot(client, system_prompt="Ты — ассистент Hermes.")
bot.run(ConsoleTransport())
```

Или через переменные окружения:

```bash
export GIGACHAT_AUTH_KEY="<base64 client_id:client_secret>"
export GIGACHAT_VERIFY_SSL=0          # эндпоинты Сбера на Russian Trusted CA
python -m examples.console_chat
```

`GIGACHAT_CA_BUNDLE` можно указать вместо отключения проверки TLS.

## Свой транспорт

```python
class MyTransport:
    def send(self, chat_id: str, text: str) -> None:
        ...  # отправить ответ в мессенджер

    def run(self, handler) -> None:
        for chat_id, text in incoming_messages():
            handler(chat_id, text)   # handler сам вызовет send()
```

## Тесты

```bash
pip install -e ".[dev]"
pytest
```

Тесты не ходят в сеть и не требуют установленного `gigachat` — в
`GigaChatClient` инъектится фейковый SDK-клиент.

## Области (`scope`)

`GIGACHAT_API_PERS` (по умолчанию), `GIGACHAT_API_B2B`, `GIGACHAT_API_CORP`.
