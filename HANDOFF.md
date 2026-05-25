# Handoff: hermesxgiga × GigaChat

Контекст для новой локальной сессии Claude Code. Не постоянный документ —
удалить после завершения (`git rm HANDOFF.md && git commit && git push`).

## Репозиторий и ветка

- Локальный путь: `~/hermesxgiga` (склонирован из `nyddle/hermesxgiga`).
- Рабочая ветка: `claude/integrate-gigachat-hermes-cSnQX`.

## Что это за проект

Python-библиотека `hermesxgiga`, связывающая GigaChat (LLM от Сбера) с
Hermes-стиля мессенджером/ботом. Компоненты:

- `hermesxgiga/gigachat.py` — `GigaChatClient`: тонкий адаптер над
  официальным SDK `gigachat` от ai-forever. Ленивый импорт SDK,
  инъекция `client=` для тестов.
- `hermesxgiga/bot.py` — `HermesBot`: история диалога по `chat_id`,
  системный промпт, ограничение длины; транспортно-независимый через
  протокол `Transport`.
- `hermesxgiga/transports.py` — `ConsoleTransport` (REPL).
- `tests/test_gigachat.py` (9 тестов) — юнит-тесты адаптера через
  фейковый SDK-клиент.
- `tests/test_bot.py` (5 тестов) — бот + транспорт.
- `examples/console_chat.py` — рабочее демо.

## Состояние окружения

- venv: `~/hermesxgiga/.venv` (Python 3.12).
- Установлено: `gigachat 0.2.1`, `pytest 9.0.3` и транзитивные зависимости
  (`pip install -e ".[dev]"` уже отработал).

## Что уже проверено

- ✅ Контракт SDK сверен с фактической установкой `gigachat 0.2.1` и
  соответствует адаптеру:
  - `GigaChat.__init__(... credentials, scope, model, verify_ssl_certs,
    ca_bundle_file, ...)`
  - `GigaChat.chat(payload: Union[Chat, Dict[str, Any], str]) -> ChatCompletion`
- ✅ Все 14 юнит-тестов проходили в предыдущей сессии через
  pytest-совместимый шим (без реального pytest). Реальный `pytest`
  ещё не прогнался успешно из-за venv-проблемы (см. ниже).

## Что нужно довести

### 1. Прогнать реальный pytest

В прошлый запуск `pytest -v` зарезолвился на **системный Homebrew
Python 3.11** (`/usr/local/opt/python@3.11/bin/python3.11`), куда
`hermesxgiga` не установлен → `ModuleNotFoundError`. venv с реальной
установкой — на Python 3.12.

Команды:

```bash
cd ~/hermesxgiga
source .venv/bin/activate
which python   # должно быть .../hermesxgiga/.venv/bin/python
python -m pytest -v
```

`python -m pytest` гарантированно поднимает pytest из venv. Если
`which python` не указывает на venv — диагностируй активацию (alias
`pytest` в shell-конфиге, перебитый `PATH` и т.п.).

### 2. Разобраться с `tests/test_live.py`

Этот файл есть локально, но **не создавался предыдущей сессией** и в
запушенной ветке его не было. Происхождение неизвестно. Действия:

1. Прочитай содержимое.
2. Если это валидный smoke с реальным GigaChat API — оберни вызовы в
   `@pytest.mark.skipif(not os.environ.get("GIGACHAT_AUTH_KEY"),
   reason="no creds")`, чтобы тест корректно скипался без кредов.
3. Если файл сломан или не нужен — удалить.

### 3. Опциональный живой smoke

Если в окружении задана `GIGACHAT_AUTH_KEY`, сделай один реальный
запрос:

```python
import os
from hermesxgiga import GigaChatClient, Message

c = GigaChatClient(os.environ["GIGACHAT_AUTH_KEY"], verify_ssl=False)
print(c.chat([Message("user", "скажи привет")]))
```

Достаточно одного успешного ответа — для финальной проверки end-to-end.

### 4. Закоммитить и запушить

Если вносил правки — закоммить с осмысленным сообщением и:
`git push -u origin claude/integrate-gigachat-hermes-cSnQX`. **PR не
создавай.**

## Границы

- Адаптер сверен с реальным SDK — без явной необходимости его не
  переписывай. Если правишь — сохраняй публичный интерфейс: `chat(
  messages) -> str`, протокол `ChatEngine`, параметр `client=` для
  инъекции.
- Не добавляй фичи / абстракции / «на будущее» — задача узкая.
- Минимум зависимостей — `requests`/`httpx`-обвязки не нужны, SDK уже
  всё несёт.

## Отчёт

В конце скажи коротко: pytest зелёный/нет, что сделал с
`tests/test_live.py`, был ли живой запрос и его результат.
