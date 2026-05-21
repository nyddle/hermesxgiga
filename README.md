# hermesxgiga

Python-библиотека/клиент, связывающая **GigaChat** (LLM от Сбера) с
**Hermes**-ботом/мессенджером. GigaChat выступает движком ответов, а бот —
транспортно-независимый: его можно подключить к консоли, Telegram, Hermes
или к тестовому двойнику.

## Установка

```bash
pip install -e .
```

Базовая зависимость одна — официальный SDK
[`gigachat`](https://github.com/ai-forever/gigachat) от ai-forever.
OpenAI-совместимый сервер для агента Hermes ставится отдельным extra:
`pip install -e ".[server]"` (добавляет FastAPI + uvicorn).

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
- OpenAI-совместимый сервер (`hermesxgiga.server`) — поднимает endpoint
  `/v1/chat/completions` и `/v1/models` поверх GigaChat, чтобы внешний
  агент [Hermes](https://github.com/nousresearch/hermes-agent) (и любой
  OpenAI-клиент) ходил в GigaChat без правок кода. Поддерживает
  tool/function calling, выбор модели и стриминг (SSE).

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

## Подключение агента Hermes к GigaChat

[Hermes](https://github.com/nousresearch/hermes-agent) общается с моделью
через OpenAI-совместимый endpoint. GigaChat такого API не отдаёт, поэтому
`hermesxgiga` поднимает тонкий прокси-сервер, который Hermes указывает как
`base_url`. Сервер транслирует протоколы в обе стороны: OpenAI `tools` /
`tool_calls` ↔ GigaChat `functions` / `function_call`, роль `tool` ↔
`function`, JSON-строка аргументов ↔ объект, плюс стриминг (SSE).

```bash
pip install -e ".[server]"

export GIGACHAT_AUTH_KEY="<base64 client_id:client_secret>"
export GIGACHAT_VERIFY_SSL=0            # или GIGACHAT_CA_BUNDLE=/path/to/ca.pem
export GIGACHAT_MODEL="GigaChat-Pro"    # модель по умолчанию
# export HERMESXGIGA_API_KEY=secret     # включить проверку Authorization (опц.)

python -m hermesxgiga.server --host 127.0.0.1 --port 8000
# либо: hermesxgiga-server
```

Затем в `~/.hermes/config.yaml`:

```yaml
model:
  provider: custom
  base_url: "http://127.0.0.1:8000/v1"
  api_key: "local"          # любое значение; проверяется, только если задан HERMESXGIGA_API_KEY
  model: "GigaChat-Pro"     # GigaChat / GigaChat-Pro / GigaChat-Max / GigaChat-2-...
```

Сменить модель на лету — `hermes model` или поле `model` в запросе:
сервер прокидывает его в GigaChat. Список доступных моделей — `GET /v1/models`.

Эндпоинты: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
(с `"stream": true` и без).

## Тесты

```bash
pip install -e ".[dev]"
pytest
```

Тесты не ходят в сеть и не требуют установленного `gigachat` — в
`GigaChatClient` инъектится фейковый SDK-клиент.

## Области (`scope`)

`GIGACHAT_API_PERS` (по умолчанию), `GIGACHAT_API_B2B`, `GIGACHAT_API_CORP`.
