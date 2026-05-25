# Handoff: hermesxgiga × GigaChat

Контекст для новой локальной сессии Claude Code. Не постоянный документ —
удалить после завершения: `git rm HANDOFF.md && git commit && git push`.

## Репозиторий

- Локальный путь: `~/hermesxgiga`.
- Рабочая ветка: `claude/integrate-gigachat-hermes-cSnQX`.

## Что это

Python-библиотека `hermesxgiga`, связывающая GigaChat (LLM от Сбера) с
Hermes-стиля мессенджером/ботом. Состав:

- `hermesxgiga/gigachat.py` — `GigaChatClient`: тонкий адаптер над
  официальным SDK `gigachat` (ai-forever). Ленивый импорт SDK, инъекция
  `client=` для тестов. Поддерживает все три способа авторизации SDK:
  `auth_key` (credentials), `user`/`password`, `access_token`.
- `hermesxgiga/bot.py` — `HermesBot`: история диалога по `chat_id`,
  системный промпт, ограничение длины. Транспортно-независимый через
  протокол `Transport`.
- `hermesxgiga/transports.py` — `ConsoleTransport` (REPL).
- `tests/test_gigachat.py` — юнит-тесты адаптера через фейковый клиент.
- `tests/test_bot.py` — бот + транспорт.
- `tests/test_live.py` — живые e2e против реального GigaChat API,
  скипаются модулем без кредов (`pytest.mark.skipif`).
- `examples/console_chat.py` — рабочее демо.

## Состояние окружения

- venv: `~/hermesxgiga/.venv` (Python 3.12).
- Установлено: `gigachat 0.2.1`, `pytest 9.0.3` + транзитивные
  зависимости. `pip install -e ".[dev]"` уже отработал.

## Что уже подтверждено

- ✅ Реальный контракт `gigachat 0.2.1` соответствует адаптеру:
  `GigaChat.__init__(..., credentials, user, password, access_token,
  scope, model, verify_ssl_certs, ca_bundle_file, ...)`, `chat()`
  принимает `Union[Chat, dict, str]`, возвращает `ChatCompletion` с
  `.choices[0].message.content`.
- ✅ Юнит-тесты проходили через pytest-совместимый шим (в окружении без
  установленного pytest). Реальный pytest ещё не отработал — см. ниже.

## Что нужно довести

### 1. Прогнать реальный pytest из venv

В прошлый раз `pytest -v` зарезолвился на системный Homebrew
Python 3.11 (`/usr/local/opt/python@3.11/bin/python3.11`), где
`hermesxgiga` не установлен → `ModuleNotFoundError`. venv с установкой
— на Python 3.12.

```bash
cd ~/hermesxgiga
source .venv/bin/activate
which python    # должно быть .../hermesxgiga/.venv/bin/python
python -m pytest -v
```

`python -m pytest` гарантированно поднимает pytest из venv. Если
`which python` не указывает на venv — диагностируй активацию (alias
`pytest` в shell, перебитый `PATH` и т.п.).

Ожидаемое: тесты `test_gigachat.py` и `test_bot.py` зелёные;
`test_live.py` — два теста в статусе SKIPPED, если кредов нет.

### 2. Опционально: живые тесты с реальным GigaChat API

`tests/test_live.py` гоняется, если в окружении есть креды. Достаточно
любого из:

```bash
export GIGACHAT_AUTH_KEY="<base64 client_id:client_secret>"
# или:
export GIGACHAT_USER="..."
export GIGACHAT_PASSWORD="..."
# или:
export GIGACHAT_ACCESS_TOKEN="..."

# почти всегда нужно при отсутствии Russian Trusted CA в системе:
export GIGACHAT_VERIFY_SSL=0

python -m pytest tests/test_live.py -v
```

Опционально: `GIGACHAT_SCOPE`, `GIGACHAT_MODEL`.

### 3. Если правишь — закоммить и запушь

Сохрани публичный интерфейс адаптера: `chat(messages) -> str`, протокол
`ChatEngine`, параметр `client=`. Коммит — осмысленный, push —
`git push -u origin claude/integrate-gigachat-hermes-cSnQX`. **PR не
создавай.**

## Границы

- Адаптер уже сверен с реальным SDK и расширен под все способы
  авторизации — без явной необходимости не переписывай.
- Не добавляй фичи / абстракции «на будущее» — задача узкая.
- Минимум зависимостей — обёртки над `httpx`/`requests` не нужны, SDK
  всё несёт.

## Отчёт

В конце скажи коротко: `pytest` зелёный/нет (сколько passed/skipped),
был ли живой вызов и его результат.
