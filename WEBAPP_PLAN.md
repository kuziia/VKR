# План прототипа веб-приложения для мониторинга научных статей

**Цель:** прототип веб-сайта, который агрегирует научные публикации через OpenAlex,
показывает динамики и популярные статьи в разрезе таксономии domain/field/subfield/topic,
и предоставляет семантический поиск по пайплайну из Главы 3
(RRF + Query2doc + CSQE + DeepSeek reranker).

**Дизайн:** газетный/редакторский (FT-style) — серифные заголовки, кремовый фон
`#fff9f0`, акцент oxblood `#990f3d`, моно `JetBrains Mono` для чисел,
табличная вёрстка (см. референс `dashboard-v7.html`).

---

## 0. Архитектура целиком

```
                    ┌─────────────────────────────┐
                    │   Frontend (Vite + React)   │
                    │  газетный layout, charts    │
                    └──────────────┬──────────────┘
                                   │ REST (JSON)
                                   ▼
                    ┌─────────────────────────────┐
                    │   Backend (FastAPI, async)  │
                    │  ┌──────────────────────┐   │
                    │  │  /api/dashboard/*    │   │
                    │  │  /api/search         │   │
                    │  │  /api/articles/{id}  │   │
                    │  │  /api/agent/*        │   │
                    │  └──────────────────────┘   │
                    └──┬─────────┬─────────┬──────┘
                       │         │         │
                       ▼         ▼         ▼
         ┌──────────────┐  ┌──────────┐  ┌──────────────────┐
         │ OpenAlex API │  │ DeepSeek │  │ Поисковый        │
         │  + taxonomy  │  │   API    │  │ конвейер (Гл.3)  │
         │   CSV        │  │ (LLM,    │  │ pipeline/* +     │
         │              │  │ агент,   │  │ Giga-Embeddings, │
         │              │  │ summary) │  │ BM25, FAISS      │
         └──────────────┘  └──────────┘  └──────────────────┘
                       │
                       ▼
         ┌─────────────────────────────────────┐
         │ Локальный кэш                        │
         │  • SQLite (метаданные, таймсерии)    │
         │  • disk-cache JSON (LLM-расширения,  │
         │    summary, agent-ответы)            │
         │  • FAISS-индекс корпуса (опц.)       │
         └─────────────────────────────────────┘
```

**MVP-границы.** Прототип, не прод. Один пользователь, без авторизации, без воркеров —
все долгие операции через FastAPI `BackgroundTasks` / `asyncio`. Семантический поиск
в MVP работает по фиксированному снапшоту корпуса OpenAlex (см. §2.4), не «вживую».

---

## 1. Источники данных

### 1.1 OpenAlex API (https://api.openalex.org)
Бесплатный, без ключа. Politeness-режим: добавлять `mailto=eterrrii@gmail.com`
к каждому запросу — даёт более высокий рейт-лимит и стабильность.

Используемые эндпоинты:
- `GET /works` — поиск/фильтры публикаций. Ключевые параметры:
  - `filter=topics.id:T10028,from_publication_date:2024-01-01,language:ru`
  - `filter=primary_topic.id:T10028` (точная привязка к topic)
  - `sort=cited_by_count:desc` / `publication_date:desc`
  - `group_by=publication_year` / `group_by=primary_topic.id` для динамик
  - `per-page=200` (макс), курсорная пагинация `cursor=*`
  - `select=id,doi,title,publication_year,publication_date,language,cited_by_count,authorships,primary_topic,topics,best_oa_location,open_access` — **обязательно**, иначе в ответе ~30 KB на работу
- `GET /works/{openalex_id}` — детали для страницы статьи

### 1.2 Таксономия
Файл `data/OpenAlex_topic_mapping_table_final_topic_field_subfield_table.csv`:
4 уровня (domain → field → subfield → topic), ~4500 топиков, у каждого
есть `keywords`, `summary`, `wikipedia_url`. Загружается в память при старте
backend. ID топика в OpenAlex — `T<topic_id>` (например, `T10028` для
«Topic Modeling»).

### 1.3 Полные тексты (для саммаризации)
Открытые источники, к которым мы пытаемся получить PDF:
- **arXiv**: `https://arxiv.org/pdf/{arxiv_id}.pdf` (если у работы есть `ids.arxiv`)
- **CyberLeninka**: scrape `https://cyberleninka.ru/article/n/{slug}` → PDF-ссылка
  (у нас уже есть `cyberleninka_dataset/` с примерами OCR — переиспользуем
  пайплайн `pipeline/00_build.py` / `_build_ocr_notebook.py`)
- **OpenAlex `best_oa_location.pdf_url`** — прямая ссылка, если работа в OA
- **DOI fallback**: Unpaywall API `https://api.unpaywall.org/v2/{doi}?email=...`

OA-флаг для UI берём из `work.open_access.is_oa` (булево) и
`work.open_access.oa_status` (`gold`/`green`/`bronze`/`hybrid`/`closed`).

---

## 2. Backend

### 2.1 Стек
- **Python 3.11**, **FastAPI** (async-эндпоинты), **Uvicorn**.
- **httpx.AsyncClient** для OpenAlex/Unpaywall (с `Limits`, retry-обёрткой).
- **SQLite + SQLAlchemy** для метаданных и таймсерий (lightweight, hand-off
  на Postgres тривиален).
- **disk-cache JSON** в `webapp_cache/` — переиспользуем паттерн из
  `pipeline/_shared.py` (хэш-ключ → файл).
- **pdfplumber / PyMuPDF** для извлечения текста PDF (есть уже в репо для OCR).
- **claude-agent-sdk** (Python) с OAuth-токеном от подписки Claude Pro/Max —
  единственный LLM-провайдер. Используется для всех LLM-стадий: Q2D, CSQE,
  listwise-rerank в поисковом пайплайне; интерпретация трендов; саммари
  статей. Семафор на 8 параллельных запросов (ниже, чем DeepSeek-32 в
  Главе 3, чтобы не выбивать лимиты подписки).
- **rank_bm25**, **FAISS-CPU**, **transformers**, **sentence-transformers**,
  **pymorphy3** — уже стоят, см. `_check_libs.py`.

### 2.2 Модули (новый пакет `webapp/`)

```
webapp/
  __init__.py
  main.py                  # FastAPI app, роутеры, lifespan (загрузка таксономии)
  settings.py              # env: ANTHROPIC_OAUTH_TOKEN, MAILTO, CACHE_DIR
  taxonomy.py              # парсер CSV, in-memory деревья domain→field→…
  openalex/
    client.py              # AsyncClient + retry + rate-limit + select-проекция
    works.py               # высокоуровневые функции: trends_by_topic, top_cited
    schemas.py             # pydantic-модели Work / Topic / Author
  search/
    pipeline.py            # обёртка над пайплайном Главы 3 (см. §2.5)
    preprocess.py          # повторное использование _shared.py preprocess
    expansion.py           # Query2doc + CSQE — через ClaudeClient
    rerank.py              # listwise-rerank через ClaudeClient
    index.py               # FAISS + BM25 поверх корпуса (см. §2.4)
  llm/
    base.py                # абстрактный LLMClient (chat / stream / json)
    claude.py              # claude-agent-sdk обёртка с OAuth, ретраи, кэш
  agents/
    interpret.py           # /api/agent/interpret-trends
    summarize.py           # /api/agent/summarize
    pdf_fetch.py           # arxiv/cyberleninka/unpaywall fetch + extract
  storage/
    db.py                  # SQLAlchemy engine, таблицы кэша
    cache.py               # disk-cache helpers (повторно из _shared.py)
  api/
    dashboard.py           # /api/dashboard/*
    search.py              # /api/search, /api/search/{job}
    articles.py            # /api/articles/{id}, /api/articles/{id}/summary
    agent.py               # /api/agent/*
```

### 2.3 Эндпоинты (минимум для UI)

| Метод | Путь                                | Назначение |
|---|---|---|
| GET  | `/api/taxonomy`                          | Дерево domain/field/subfield/topic для фильтров |
| GET  | `/api/dashboard/trends`                  | Динамика публикаций. Параметры: `level=domain|field|subfield|topic`, `id=...`, `from=YYYY-MM`, `to=YYYY-MM`, `lang=ru,en|all`, `granularity=month|quarter|year` |
| GET  | `/api/dashboard/top-cited`               | Популярные статьи. `level`, `id`, `period=1m|6m|1y|10y|custom`, `from`, `to`, `lang`, `limit=20` |
| POST | `/api/agent/interpret-trends`            | LLM-комментарий по выбранным агрегатам (стримом или одним JSON) |
| POST | `/api/search`                            | Семантический поиск — запускает джоб, возвращает `job_id` |
| GET  | `/api/search/{job_id}`                   | Статус джоба + результаты (top-K + скоры) |
| GET  | `/api/articles/{openalex_id}`            | Полные метаданные работы из OpenAlex |
| POST | `/api/articles/{openalex_id}/summary`    | Саммари через LLM (если PDF доступен — FT extract; иначе по abstract) |
| GET  | `/api/articles/{openalex_id}/oa-status`  | OA-флаг + список доступных PDF-источников |

Все ответы — JSON, поля в snake_case, числа — без форматирования
(форматирует фронт), даты — ISO-8601.

### 2.4 Корпус для семантического поиска

Пайплайн из Главы 3 требует индексированного корпуса. Для прототипа делаем
**фиксированный снапшот** (а не индексацию всего OpenAlex):

1. **Источник корпуса.** 30–100 тыс. русскоязычных научных работ из OpenAlex,
   отобранных по фильтру `language:ru AND has_abstract:true AND publication_year:>=2018`,
   с балансом по доменам по таксономии. Снапшот делается оффлайн скриптом
   `webapp/scripts/build_snapshot.py` → CSV/JSONL.
2. **Тексты.** `title + abstract` (расшифрованный из `abstract_inverted_index`).
   Для статей с открытым PDF — опциональный bg-job, обогащающий запись
   полным текстом (см. `pdf_fetch.py`).
3. **Индексы.**
   - **BM25**: rank_bm25 поверх предобработанных полей (lowercase →
     pymorphy3-лемматизация → удаление стоп-слов, 100-словный список из 3.3.4).
     Кэш препроцессированного корпуса — JSON, как в Главе 3.
   - **Dense**: `ai-sage/Giga-Embeddings-instruct` (bf16, 4096 maxlen,
     flash-attention 2 если есть GPU; CPU-fallback на `intfloat/multilingual-e5-large`
     fp16 — заметим явно в UI). FAISS `IndexFlatIP` на нормированных векторах.
4. **Версионирование.** `snapshot_id = sha1(timestamp + filter_args)`. UI
   показывает дату снапшота, чтобы пользователь не путал свежие OpenAlex-данные
   (для дашборда тянутся вживую) с фиксированным корпусом для поиска.

> **Важно для прототипа.** Если в первой итерации не хватает GPU для Giga,
> используем mE5-large fp16 — пайплайн модульный, переключение через `_config.py`.
> В отчёт по эксперименту это не идёт; для прода оставляем хук под Giga.

### 2.5 Поисковый пайплайн (тонкая обёртка)

> **СТАТУС ПРОТОТИПА (Phase 4 MVP):** в первой итерации поиск — это
> **тонкая обёртка над `GET /works?search=…&sort=relevance_score:desc`
> у OpenAlex** (нативный BM25-подобный ранкер на `title + abstract`).
> Корпус не строится локально, индекс не нужен. Это позволяет сразу
> поднять UI и проверить интеграцию с FE/БД/кэшем.
>
> **TODO для следующей итерации (полный пайплайн из Главы 3):**
> построить локальный снапшот OpenAlex-RU (`webapp/scripts/build_snapshot.py`,
> §2.4), собрать BM25 (rank_bm25) + dense-индекс
> (`ai-sage/Giga-Embeddings-instruct` или `mE5-large` fallback, FAISS),
> подключить `webapp/search/expansion.py` (Q2D + CSQE через Claude),
> `webapp/search/rerank.py` (listwise через Claude), и собрать всё в
> `webapp/search/pipeline.py:run_search()` по схеме ниже. Текущий MVP-эндпоинт
> `POST /api/search` будет за ним же сидеть — поменяется только реализация.

Финальная конфигурация Главы 3 — **RRF(Giga + BM25) + Query2doc + CSQE + listwise-rerank**.
В прототипе вместо DeepSeek listwise-реранкер реализован через Claude — см.
оговорку про методологию в §2.6. Выносим в `webapp/search/pipeline.py`
функцию:

```python
async def run_search(query: str, top_k: int = 20, lang: str = "ru") -> SearchResult:
    """
    1) preprocess(query) — pymorphy3 + кастомные стоп-слова
    2) query2doc_expansion(query)  → q2d         [Claude, кэш]
    3) первичный RRF(Giga(q), BM25(q)) → top-10 для CSQE
    4) csqe_extract(query, top10) → key_sents    [Claude, кэш]
    5) финальный RRF(Giga(q), BM25(preprocess(q + q2d + key_sents)))
       → top-100 (k=60 в RRF)
    6) listwise rerank по top-20 через Claude → итог
    """
```

Стадии **переиспользуют** структуру и промпты из `pipeline/06_qe_12combos.py`,
`pipeline/07_extra_methods.py`, `pipeline/bm25_rerank_eval.py` — копируем
тексты Q2D/CSQE/rerank-промптов один-в-один (они отлаживались под Главу 3),
но вызовы делаем через `ClaudeClient` вместо `AsyncOpenAI(deepseek)`.
Сигнатуры функций одинаковые, поэтому переключение точечное.

**Кэширование.** Для одного и того же `query`:
- расширения (Q2D, CSQE) — disk-cache по хэшу,
- эмбеддинг запроса — LRU в памяти,
- ответ реранкера — disk-cache по хэшу пары (query, top-20 ids).

Существующий кэш расширений из экспериментов Главы 3 (`qe_cache_*`)
**не переиспользуется** — он содержит ответы DeepSeek, а в прототипе
другая модель. При первом запросе на каждую тестовую строку Claude
вызывается заново, дальше работает диск-кэш.

### 2.6 LLM-провайдер: Claude Agent SDK

Прототип использует **только Claude** (через `claude-agent-sdk` с
OAuth-авторизацией от подписки Pro/Max) — для всех LLM-стадий: внутренних
(Q2D, CSQE, listwise-rerank в поисковом пайплайне) и пользовательских
(интерпретация трендов, саммари статей).

#### 2.6.1 Авторизация и развёртывание

```bash
# Один раз на сервере (вне РФ — см. §7), где будет крутиться бэкенд:
claude setup-token
# Полученный долгоживущий OAuth-токен в .env (chmod 600):
ANTHROPIC_OAUTH_TOKEN=sk-ant-oat01-...
```

Бэкенд читает токен из env и передаёт SDK. Биллинг — против лимитов
подписки (5-часовое окно + недельный кап), без отдельных API-credits
Anthropic.

#### 2.6.2 Использование

Все LLM-вызовы идут через единый `ClaudeClient` из `webapp/llm/claude.py`,
реализующий `LLMClient`-протокол (см. §2.6.4).

| Стадия | Где | Модель | Temperature | max_tokens |
|---|---|---|---|---|
| Query2doc-расширение | `search/expansion.py` | `claude-haiku-4-5` | 1.0 | 256 |
| CSQE-извлечение предложений | `search/expansion.py` | `claude-sonnet-4-6` | 0.0 | 1024 |
| Listwise-rerank top-20 | `search/rerank.py` | `claude-sonnet-4-6` | 0.0 | 512 |
| Интерпретация трендов | `agents/interpret.py` | `claude-sonnet-4-6` | 0.7 | 1200 |
| Саммари статьи | `agents/summarize.py` | `claude-sonnet-4-6` | 0.2 | 1500 |

Распределение моделей мотивировано экономией лимитов:
- **Haiku 4.5** для Q2D — это самая частая операция (один вызов на каждый
  поисковый запрос), и качество псевдодокумента не критично — он нужен
  только для обогащения BM25-ветки.
- **Sonnet 4.6** для всего остального — задачи с более тонким суждением
  (релевантность в CSQE, ранжирование в rerank, длинные ответы агентам).

#### 2.6.3 Параллельность и кэш

- **Семафор `asyncio.Semaphore(8)`** на одновременные обращения к Claude.
  Это сильно ниже DeepSeek-32 из Главы 3 — подписка Pro/Max не рассчитана
  на 32-параллельную нагрузку, заработаешь rate-limit за минуты.
- **Экспоненциальный backoff** 2/4/8/16 сек на 4 попытки.
- **Disk-cache JSON** по хэшу промпта (паттерн `pipeline/_shared.py`).
  Каждый ответ Q2D, CSQE, rerank, summary, interpret кэшируется. Повторное
  открытие той же страницы или повторный поиск того же запроса лимиты
  подписки не расходует.

#### 2.6.4 Абстракция (`webapp/llm/base.py`)

```python
class LLMClient(Protocol):
    async def complete(self, prompt: str, *, system: str | None = None,
                       model: str = "claude-sonnet-4-6",
                       temperature: float = 0.7,
                       max_tokens: int = 1024) -> str: ...
    async def stream(self, prompt: str, *, system: str | None = None,
                     model: str = "claude-sonnet-4-6",
                     temperature: float = 0.7,
                     max_tokens: int = 1024) -> AsyncIterator[str]: ...
    async def complete_json(self, prompt: str, *, schema: dict,
                            model: str = "claude-sonnet-4-6") -> dict: ...
```

`complete_json` — отдельный метод для CSQE и реранкера, где модель должна
вернуть структурированный JSON. Внутри Claude вызывается с `tool_use` или
prefill `{` — реализационная деталь `claude.py`.

Протокол позволит позже свапнуть на Anthropic API (метод `complete` →
`anthropic.AsyncAnthropic.messages.create`) или на локальную модель,
если у пользователя появится свой биллинг или GPU. Сигнатуры функций
поискового пайплайна не зависят от провайдера.

#### 2.6.5 Лимиты подписки и публичность

OAuth-токен Pro/Max выдан конкретному пользователю; формально его
нельзя использовать как backend для произвольных пользователей в
интернете. Для **демонстрации защиты диплома** ограничений нет:
один пользователь, низкий QPS.

Оценка нагрузки на подписку при типичном демо:
- Один поиск ≈ 3 LLM-вызова (Q2D + CSQE + rerank). При 5 поисках в час
  это ~15 вызовов/час.
- Одна интерпретация ≈ 1 вызов; одно саммари ≈ 1-3 вызова в зависимости
  от длины PDF (chunked).
- Pro даёт ~45 сообщений / 5 часов; Max — ~225. Для подготовки и защиты
  с большим запасом хватит Pro.

Для **публичного хостинга** план — мигрировать на Anthropic API
(см. §6, п. 6); реализационно — добавить `webapp/llm/anthropic.py`
с тем же протоколом и переключить через env.

#### 2.6.6 Расхождение с методологией Главы 3 — оговорка

В §3.6.2 диплома `deepseek-chat` валидирован как listwise-реранкер;
Q2D и CSQE-расширения тоже замерены на DeepSeek. **В прототипе используется
Claude вместо DeepSeek**, что значит абсолютные значения метрик прототипа
будут отличаться от опубликованных в Главе 3. Это нормально для
демо-прототипа (архитектура та же, изменён только LLM-бэкенд), но при
защите стоит явно сказать:

> «В эксперименте Главы 3 использовался DeepSeek-chat как наиболее
> сбалансированная по цене и качеству модель в том этапе работы.
> В развёрнутом прототипе тот же пайплайн (Q2D + CSQE + listwise-rerank)
> работает на Claude Sonnet 4.6 — архитектурно идентично, абсолютные
> метрики могут отличаться, но качественные выводы о вкладе CSQE и
> листового реранкинга в общий выигрыш сохраняются».

Если у комиссии будет вопрос «почему не DeepSeek в прототипе» — ответ:
геополитика и доступность инфраструктуры (DeepSeek-API имеет другие
ограничения и SLA-риски; Claude через подписку — стабильнее для
демонстрационного прототипа).

### 2.7 Хранилище

SQLite (`webapp.db`) — три таблицы:

```sql
CREATE TABLE trends_cache (
  cache_key TEXT PRIMARY KEY,   -- sha1(level, id, from, to, lang, granularity)
  payload   TEXT NOT NULL,      -- JSON
  fetched_at TIMESTAMP NOT NULL
);
CREATE TABLE works_cache (
  openalex_id TEXT PRIMARY KEY,
  payload     TEXT NOT NULL,
  fetched_at  TIMESTAMP NOT NULL
);
CREATE TABLE search_jobs (
  job_id     TEXT PRIMARY KEY,
  query      TEXT NOT NULL,
  status     TEXT NOT NULL,     -- pending | running | done | error
  result     TEXT,              -- JSON: top-K, scores, expansions
  created_at TIMESTAMP NOT NULL
);
```

TTL для `trends_cache` — 24 часа (дашборд не должен дёргать OpenAlex на каждый
клик), для `works_cache` — 7 дней.

---

## 3. Frontend

### 3.1 Стек
- **Vite + React 18 + TypeScript** (быстрый dev-сервер, типы для DTO).
- **TanStack Query** — кэш и инвалидация запросов.
- **Recharts** или **uPlot** для линейных графиков и гистограмм
  (uPlot быстрее, лучше для длинных таймсерий — 10 лет помесячно ≈ 120 точек).
- **Tailwind CSS** + кастомный config с FT-палитрой и шрифтами
  (`Source Serif 4`, `Inter`, `JetBrains Mono` — как в `dashboard-v7.html`).
- **react-router-dom** для маршрутов.
- **react-markdown** + `rehype-highlight` для рендера ответов LLM-агента.

### 3.2 Маршруты

| Путь                     | Страница |
|---|---|
| `/`                      | Dashboard (главная) |
| `/search?q=...`          | Результаты поиска |
| `/article/:openalex_id`  | Детали статьи |
| `/topic/:level/:id`      | Drill-down по узлу таксономии (опц., MVP-2) |

### 3.3 Главная — `/`

**Шапка (masthead).** Серифный «бренд» («NAUKA-MONITOR» или название по
вкусу), курсивный подзаголовок-дата, справа — таймстамп снапшота корпуса.
Под шапкой — горизонтальный nav-strip.

**Hero-блок.**
- Слева: заголовок «Динамика публикаций», крупный (~48 px серифом) счётчик
  суммы публикаций за выбранный период (моно-цифры, tabular-nums), курсивный
  «deck» с дельтой к предыдущему периоду.
- Справа: панель фильтров (4 dropdown'а — domain → field → subfield → topic
  с каскадом + мульти-чекбокс языков `ru / en / другие`). Селект периода:
  `1M / 6M / 1Y / 10Y / custom`.

**Линейный график «Equity-curve»-стиль.** Линия — кол-во новых работ помесячно
по выбранному узлу таксономии. Поверх — серая линия общего среднего по узлу
(сравнение). Цвет основной линии — `#990f3d` (FT pink). Сетка пунктиром
`rgba(216,205,182,.7)` — копия `dashboard-v7.html`.

**Колонки-метрики (4 шт. в ряд).** «Этот месяц / этот год / 5 лет / всего».
Заголовок 10 px caps, значение 32 px серифом, рядом дельта в процентах
с цветом `--profit / --loss`. Под этим — секция «By field» (горизонтальные
бары как `.sym-list` в референсе): топ-5 field'ов по росту публикаций.

**Секция «Популярные статьи».** Заголовок секции серифом 24 px,
двойная линия снизу, мелкий курсивный sub. Таблица в стиле `table.editorial`:
| Title (серифом) | Authors | Year | Citations (моно) | Lang | OA |
Клик по строке → `/article/:id`. Рядом с заголовком — переключатель периода
(`1M / 6M / 1Y / 10Y / custom`) и тот же фильтр языка из hero (синхронизация
через URL-параметры).

**Блок «Интерпретация ИИ-агента».** Серый pull-quote-блок. Кнопка «Получить
комментарий» отправляет текущий контекст фильтров на `/api/agent/interpret-trends`
и стримит ответ Markdown-текстом. Под блоком — мелкий disclaimer
«Сгенерировано LLM, не редакторский комментарий».

### 3.4 Поиск — `/search`

Окно поиска живёт в шапке главной и на странице результатов. Submit → POST
на `/api/search`, polling `/api/search/{job_id}` каждые 1.5 с до `done`,
показ скелетона.

**Layout страницы результатов.**
- Хлебные крошки + сам запрос крупно («Поиск: <query>», серифом 32 px).
- Левая колонка ⅔ — список статей (та же `table.editorial`), справа ⅓ — блок
  «Что мы сделали»: показывает Q2D-расширение, извлечённые CSQE-предложения,
  rerank-skore (educational, чтобы пользователь видел работу пайплайна).
- Каждая строка результата: заголовок (link на `/article/:id`), authors+venue
  курсивом, score (моно), бейдж OA (зелёный outline `--profit`), бейдж языка.

### 3.5 Карточка статьи — `/article/:id`

Двухколоночный layout (как в hero):
- Левая колонка ⅔: заголовок серифом 56 px, авторы курсивом, метаданные
  (год, journal, DOI), abstract в `Source Serif 4`. Под abstract — кнопка
  **«Открыть источник»** (ведёт на `doi.org/...` или `best_oa_location.landing_page_url`)
  и **«Получить саммари ИИ-агента»** (POST на `/api/articles/:id/summary`).
- Правая колонка ⅓: «Capital allocation»-стиль, но про статью — citations,
  refs, OA-status, primary topic+subfield+field+domain (links), список соавторов.

Саммари рендерится как Markdown, со спиннером во время генерации.
Если PDF не удалось получить — показываем «Саммари по abstract'у»
с пояснением.

### 3.6 Стили (Tailwind config)

```js
// tailwind.config.js — фрагмент
theme: {
  extend: {
    colors: {
      bg: '#fff9f0', 'bg-2': '#f5eee0',
      surface: '#fffcf5', 'surface-2': '#f5eee0', 'surface-3': '#ebe2ce',
      border: '#d8cdb6', 'border-strong': '#a89c82',
      text: '#1a1a1a', 'text-muted': '#5e564a', 'text-dim': '#8a8170',
      accent: '#990f3d',
      profit: '#1a7c4f', loss: '#c3352f', warn: '#a86b00',
    },
    fontFamily: {
      serif: ['"Source Serif 4"', 'Georgia', 'serif'],
      sans:  ['Inter', 'sans-serif'],
      mono:  ['"JetBrains Mono"', 'monospace'],
    },
    fontFeatureSettings: { tnum: '"tnum"' },
  },
}
```

CSS-переменные не нужны — всё через tailwind. `tabular-nums` — на каждом
числовом элементе. Двойную нижнюю линию masthead'а делаем
`border-b-[3px] border-double border-text`.

---

## 4. Поток данных по сценариям

### 4.1 Открытие главной
1. FE → `GET /api/taxonomy` (1 раз, кэш в TanStack Query на сессию).
2. FE → `GET /api/dashboard/trends?level=domain&id=4&from=2025-05&to=2026-05&lang=ru&granularity=month`.
3. BE проверяет `trends_cache`. Если miss — батчит вызовы OpenAlex
   `group_by=publication_month` (через эмуляцию: год + месяц-фильтр), кэширует.
4. FE параллельно → `GET /api/dashboard/top-cited?...&period=1y&limit=20`.
5. UI рендерит график + таблицу + колонки-метрики.

### 4.2 Запрос интерпретации
1. Пользователь жмёт «Получить комментарий».
2. FE → `POST /api/agent/interpret-trends` с `{level, id, period, lang, snapshot_summary}`,
   где `snapshot_summary` — минимум данных: timeseries, top-5 статей, имя узла.
3. BE строит промпт, вызывает Claude через `claude-agent-sdk` (стрим),
   отдаёт SSE.
4. FE рендерит Markdown по мере поступления.

### 4.3 Семантический поиск
1. FE → `POST /api/search {query, top_k: 20}` → `{job_id}`.
2. BE кладёт джоб в `search_jobs(status=pending)`, запускает `BackgroundTask`.
3. Background runs `run_search()` (см. §2.5), пишет результат в БД.
4. FE поллит `GET /api/search/{job_id}` каждые 1.5 с.
5. Когда `status=done`, рендерит результаты + панель «Что мы сделали».

### 4.4 Саммари статьи
1. FE → `POST /api/articles/{id}/summary`.
2. BE проверяет кэш; если miss — `pdf_fetch.fetch(work)`:
   - сначала `arxiv_id` если есть,
   - потом `best_oa_location.pdf_url`,
   - потом cyberleninka (если язык=ru и source matches),
   - потом Unpaywall fallback.
3. Если PDF получен → PyMuPDF извлекает текст → chunked summarization
   через Claude (`claude-agent-sdk`).
4. Если нет → суммаризация abstract'а через Claude с пометкой
   `source: "abstract"`.
5. Возврат: `{summary_md, source, pdf_url|null, oa_status}`.

---

## 5. Поэтапный roadmap

### Фаза 0 — каркас (1 день)
- [ ] `webapp/` package + FastAPI hello-world.
- [ ] Vite + React + Tailwind с FT-палитрой; статичный mock дашборда из
      `dashboard-v7.html` поверх данных-заглушек.
- [ ] `taxonomy.py` — парсинг CSV + `/api/taxonomy` эндпоинт.

### Фаза 1 — дашборд (3 дня)
- [ ] `openalex/client.py` (httpx, retry, rate-limit, `mailto`).
- [ ] `openalex/works.py` — `trends_by_topic`, `top_cited` с агрегацией
      по `publication_year`+`publication_month`.
- [ ] `/api/dashboard/trends`, `/api/dashboard/top-cited` + SQLite-кэш.
- [ ] FE главная: hero, фильтры (каскад domain→…→topic), линейный график
      (uPlot), таблица популярных, колонки-метрики, sym-bars по полям.
- [ ] Языковой фильтр (мульти).
- [ ] Переключатель периода (1M/6M/1Y/10Y/custom).

### Фаза 2 — карточка статьи + PDF (2 дня)
- [ ] `/api/articles/{id}` + `works_cache`.
- [ ] `agents/pdf_fetch.py` (arxiv → OA-location → cyberleninka → unpaywall).
- [ ] `llm/base.py` (Protocol) + `llm/claude.py` (claude-agent-sdk + OAuth).
- [ ] `claude setup-token` на сервере → токен в `.env`.
- [ ] `agents/summarize.py` — chunked summary через Claude.
- [ ] FE: страница статьи, кнопки «Открыть источник» / «Саммари».
- [ ] Бейдж OA, ссылки на topic/subfield/field/domain.

### Фаза 3 — интерпретация-агент (1 день)
- [ ] `agents/interpret.py` + промпт (Claude через тот же `LLMClient`).
- [ ] SSE-стрим `POST /api/agent/interpret-trends`.
- [ ] FE: блок «Интерпретация ИИ-агента» с Markdown-рендером.
- [ ] Обработка ошибок лимита подписки (UI-сообщение, не 500).

### Фаза 4 — семантический поиск (4 дня)
- [ ] `webapp/scripts/build_snapshot.py` → корпус OpenAlex-RU.
- [ ] `search/index.py` — BM25 + Giga (или mE5-fallback) FAISS.
- [ ] `search/expansion.py` (Q2D + CSQE через `ClaudeClient`),
      `search/rerank.py` (listwise-rerank через `ClaudeClient`).
- [ ] `search/pipeline.py` — финальная связка (промпты — копии из
      `pipeline/06_qe_12combos.py` / `bm25_rerank_eval.py`,
      вызовы — через Claude).
- [ ] `/api/search` (BackgroundTasks) + polling `/api/search/{job_id}`.
- [ ] FE: страница `/search`, окно поиска в шапке, панель «Что мы сделали».

### Фаза 5 — финиш (1 день)
- [ ] Footer, error-states, скелетоны во всех таблицах/графиках.
- [ ] README с инструкцией запуска (`uvicorn webapp.main:app` + `npm run dev`).
- [ ] Проверка golden path в браузере (главная → клик по статье → саммари;
      ввод запроса → результаты → клик → саммари).

**Итого MVP:** ~12 рабочих дней.

---

## 6. Открытые вопросы / решения, которые надо подтвердить

1. **GPU для Giga-Embeddings.** Если на машине с фронтом нет GPU — на чём
   крутить плотный энкодер? Варианты: (a) поднять backend на сервере с A100,
   (b) использовать mE5-large fp16 на CPU (теряем ~10 пунктов NDCG, но
   функционально работает), (c) RemoteEmbeddings через сервис.
2. **Размер корпуса для поиска.** 30k достаточно для прототипа на ноутбуке?
   Если да — можно держать всё в памяти. Если 100k+ — нужен FAISS на диске
   с mmap.
3. **Стрим vs polling для поиска.** Проще polling, но WebSocket / SSE дают
   приятнее UX. Для MVP закладываем polling.
4. **Хранение PDF.** Сохранять скачанные PDF на диск под `webapp_cache/pdfs/`
   или каждый раз заново тянуть? Предлагаю кэшировать на 30 дней.
5. **Анонимизация / лимиты на Claude.** Без авторизации любой может
   упереть подписку в лимит через `/api/agent/*` и `/api/search`
   (один поиск ≈ 3 вызова Claude). Минимум — IP-rate-limit (slowapi:
   10 запросов в час на IP) на оба эндпоинта.
6. **Миграция Claude с подписки на API.** OAuth-токен Pro/Max технически
   привязан к одному пользователю и не подходит для публичного multi-tenant
   доступа. План на случай публичного хостинга: завести Anthropic API
   workspace, добавить `webapp/llm/anthropic_api.py` с тем же
   `LLMClient`-протоколом (см. §2.6.4), переключить через env-переменную.
   Промпты, кэш и весь поисковый пайплайн остаются без изменений. Для
   защиты диплома и локальных демо подписки достаточно.
7. **Обработка лимитов Claude.** При превышении 5h-окна Pro/Max SDK
   возвращает ошибку. Бэкенд должен ловить её и в UI показывать понятное
   сообщение «лимит подписки исчерпан, попробуйте через N минут» вместо
   500. Не путать с реальной ошибкой генерации.

---

## 7. Развёртывание

**Юрисдикция сервера — обязательно вне РФ.** Anthropic geo-блокирует
российские IP на уровне API, Claude Code и `claude setup-token` не
работают. Российские провайдеры (Selectel, Yandex Cloud, Timeweb,
Beget, Immers) — не подходят.

### 7.1 Двухсерверная схема: dev на CPU, защита на GPU

Чтобы не платить за GPU всё время разработки, используем переключаемые
профили (детально в обсуждениях). Основа — `EMBEDDING_MODEL` как env-переменная.

#### Профиль B — dev/CPU (для разработки и отладки)

| Ресурс | Минимум | Комфорт |
|---|---|---|
| vCPU | 4 | 8 |
| RAM | 16 ГБ | 32 ГБ |
| Диск (NVMe) | 50 ГБ | 100 ГБ |
| GPU | — | — |
| Энкодер | `intfloat/multilingual-e5-large` (fp16, CPU) | то же |

**Где:** Hetzner CCX13 (€13/мес, 2 dedicated vCPU / 8 ГБ — впритык) или
CCX23 (€27/мес, 4 dedicated vCPU / 16 ГБ — комфорт). Финляндия/Германия,
Claude доступен. Оплата картой через Wise или подобные сервисы.

**Latency:** энкодинг запроса ~1-3 сек на CPU; полный поиск ~5-10 сек
с учётом 3 вызовов Claude (~3-5 сек на стадию).

**Минус:** не соответствует методологии Главы 3 (там Giga-Embeddings).

#### Профиль C — защита/GPU (для финальной демонстрации)

| Ресурс | Минимум | Комфорт |
|---|---|---|
| vCPU | 8 | 16 |
| RAM | 32 ГБ | 64 ГБ |
| Диск (NVMe) | 100 ГБ | 200 ГБ |
| **GPU** | **RTX 3090 / A4000 (16-24 ГБ VRAM)** | **A100 40/80 ГБ** |
| Энкодер | `ai-sage/Giga-Embeddings-instruct` (bf16) | то же |

**Где:** vast.ai / runpod.io / Lambda Labs (вне РФ, есть Claude). 3090
на vast.ai ~$0.3-0.5/час, A100 ~$1.5-2/час. На время защиты (3-7 дней)
суммарно $30-150.

**Latency:** энкодинг ~50-100 мс, полный поиск ~3-6 сек (доминирует
Claude).

### 7.2 Переключение между B и C

Подводные камни (детально см. обсуждение):
- **FAISS-индекс модель-специфичен** — пересобирать на C под Giga (один
  скрипт `build_snapshot.py`, ~30-45 мин на A100 для 50k документов).
- **Код-путь Giga vs mE5 различен** — обязательно прогнать smoke-test
  на GPU **в середине разработки**, не в день защиты.
- **CUDA / Python / glibc differences** — Dockerfile с `ARG BASE`
  (cpu-base vs cuda-runtime) и отдельные `requirements-cpu.txt` /
  `requirements-gpu.txt`.

### 7.3 Конфигурация сервера (одинаковая для B и C)

**Софт-стек:**
- Ubuntu 22.04 LTS
- Python 3.11 + venv (на C — conda для CUDA-стека)
- nginx как reverse-proxy + TLS через Caddy/certbot (Let's Encrypt)
- systemd-юнит для FastAPI (`uvicorn webapp.main:app --workers 2 --host 127.0.0.1 --port 8000`)
- Frontend — `npm run build` → статика в `/var/www/`, отдаётся nginx'ом
- Логи — journalctl + logrotate

**Переменные окружения (`/etc/webapp.env`, `chmod 600`):**
```
ANTHROPIC_OAUTH_TOKEN=sk-ant-oat01-...      # обязательно
MAILTO=eterrrii@gmail.com                    # для politeness OpenAlex
CACHE_DIR=/var/lib/webapp/cache
EMBEDDING_MODEL=intfloat/multilingual-e5-large    # B
# EMBEDDING_MODEL=ai-sage/Giga-Embeddings-instruct  # C
SNAPSHOT_ID=...
```

**Распределение диска (Профиль C):**
- HuggingFace cache (`~/.cache/huggingface`): 10-15 ГБ (Giga)
- pymorphy3 + словари: 500 МБ
- Снапшот корпуса (50k JSONL + processed JSON): ~400 МБ
- FAISS index: 1.6 ГБ (Giga, fp32)
- LLM-кэш Claude (Q2D, CSQE, rerank, summary, interpret): 50-200 МБ
- PDF cache (TTL 30 дней): рассчитывать на 5-10 ГБ
- SQLite + логи: 1-2 ГБ
- **Итого: 100 ГБ — впритык, 200 ГБ — спокойно.**

### 7.4 Защита OAuth-токена

Anthropic OAuth-токен Pro/Max в `.env` (`chmod 600`) на сервере, который
используется только тобой. Если токен компрометирован — `claude setup-token`
заново и обновить `.env`. **Не комитить в git** (`.gitignore` уже включает
`.env` в репо).
