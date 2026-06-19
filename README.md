# Автоматизированный мониторинг и анализ научных публикаций с LLM

Программная реализация коллективной выпускной квалификационной работы на тему
«Автоматизированный мониторинг и анализ научных публикаций с LLM», выполненной
в Финансовом университете при Правительстве Российской Федерации (факультет
информационных технологий и анализа больших данных, кафедра искусственного
интеллекта).

**Авторы:** Мустаева Екатерина Сергеевна, Кузнецова Ксения Алексеевна (группа ПМ22-5).

---

## Состав репозитория

| Модуль | Описание |
|---|---|
| [`webapp/`](#webapp--веб-приложение) | Веб-приложение: дашборд, семантический поиск, LLM-агенты |
| [`search/`](#search--пайплайн-расширения-запросов) | Пайплайн сравнения методов расширения запросов (QE) |
| [`dataset/`](#dataset--датасет-cyberleninka) | Датасет CyberLeninka — 80 статей, 208 запросов, BEIR-формат |

---

## Основные результаты

- Наилучшая конфигурация поискового конвейера — RRF(Giga-Embeddings + BM25 с
  расширением Query2doc) + CSQE + listwise-переранжирование DeepSeek:
  **NDCG@10 = 0,7783** на rus-scifact, **NDCG@10 = 0,4151** на rus-nfcorpus.
- Тематическая классификация (12 000 статей КиберЛенинки, таксономия OECD):
  **F1-macro = 0,6890** — NMF + CatBoost с выборочным LLM-переранжированием.
- Динамический тематический анализ (BERTrend, 34 318 статей за 12 месяцев):
  505 глобальных тем по классам сигналов.

---

## webapp/ — Веб-приложение

Трёхслойная система: React-фронтенд → FastAPI-бэкенд → OpenAlex / OpenAIRE / Claude.

### Возможности

- **Аналитический дашборд.** Временной ряд публикационной активности, таблица
  наиболее цитируемых работ, распределение по таксономии, автоматическая
  интерпретация сводок LLM-агентом.
- **Семантический поиск.** Каскадная фильтрация по таксономии OpenAlex, источнику,
  стране и периоду. Конвейер: RRF + Query2doc + CSQE + listwise-переранжирование.
- **Карточки публикаций и авторов.** Метаданные, PDF-источники, автоматическое
  резюме (Claude), граф цитирований (D3-Force).
- **LLM-агенты:** `summarize` (иерархическое реферирование PDF), `rerank`
  (listwise-переранжирование), `interpret` (потоковые комментарии через SSE).

### Архитектура

```
        ┌─────────────────────────────┐
        │   Frontend (Vite + React)   │
        └──────────────┬──────────────┘
                       │ REST / SSE (JSON)
                       ▼
        ┌─────────────────────────────┐
        │   Backend (FastAPI, async)  │
        │   /api/{dashboard,search,   │
        │     articles,authors,agent} │
        └──┬─────────┬─────────┬──────┘
           │         │         │
           ▼         ▼         ▼
   ┌──────────────┐ ┌──────────┐ ┌──────────────────┐
   │ OpenAlex /   │ │ Claude   │ │ Локальный кэш    │
   │ OpenAIRE API │ │ (агенты) │ │ SQLite + disk    │
   └──────────────┘ └──────────┘ └──────────────────┘
```

| Маршрутизатор | Назначение |
|---|---|
| `dashboard` | Временной ряд, цитируемые работы, агрегации |
| `search` | Поиск с фильтрацией по источнику, стране, периоду, таксономии |
| `articles` | Метаданные, резюме, граф цитирований |
| `authors` | Профили авторов, списки работ |
| `agent` | Потоковое взаимодействие с LLM |

### Структура webapp/

```
webapp/                  # backend (FastAPI)
├── main.py              # точка входа
├── settings.py          # конфигурация (Pydantic settings)
├── taxonomy.py          # таксономия OpenAlex
├── api/                 # REST-эндпоинты
├── agents/              # LLM-агенты (summarize, interpret, rerank)
├── openalex/            # клиент OpenAlex
├── openaire/            # клиент OpenAIRE
├── search/              # поисковый конвейер
├── llm/                 # интеграция Claude
└── storage/             # асинхронный кэш (SQLite)

web/                     # frontend (Vite + React + TypeScript)
└── src/
    ├── pages/           # Dashboard, Search, Article, Author, Graph
    ├── components/      # фильтры, графики, таблицы
    └── lib/             # api-клиент, утилиты
```

### Запуск webapp

```bash
# Backend (порт 8088)
python -m venv .venv && .venv/Scripts/activate
pip install fastapi uvicorn httpx pydantic pydantic-settings SQLAlchemy aiosqlite tenacity
cp .env.example .env  # заполни ANTHROPIC_OAUTH_TOKEN и MAILTO
uvicorn webapp.main:app --reload --port 8088
# Документация: http://127.0.0.1:8088/docs

# Frontend (порт 5173)
cd web && npm install && npm run dev
```

---

## search/ — Пайплайн расширения запросов

7-этапный пайплайн для сравнения методов расширения запросов (QE) на
русскоязычных научных датасетах.

```
search/
├── pipeline/     # скрипты пайплайна
├── notebooks/    # Jupyter-ноутбуки экспериментов
└── results/      # CSV с результатами экспериментов
```

### Датасеты

| Алиас | HuggingFace | Корпус | Запросы |
|---|---|---|---|
| `nfcorpus` | `kaengreg/rus-nfcorpus` | 3 630 | ~323 |
| `scifact` | `kaengreg/rus-scifact` | 1 109 | ~300 |
| `ruSciBench` | `kaengreg/ruSciBench-retrieval` | 201 тыс. | 1 577 |
| `miracl` | `kaengreg/rus-miracl` | 9.54 млн | 1 252 |

### Этапы пайплайна

| # | Скрипт | Что делает |
|---|---|---|
| 01 | `01_compare_embeddings.py` | Сравнивает 5 embedding-моделей по recall/MAP/nDCG |
| 02 | `02_bm25_baseline.py` | BM25 + dense baseline без QE |
| 03 | `03_encode_best.py` | Кэширует эмбеддинги лучшей модели |
| 04 | `04_build_faiss.py` | Строит FAISS HNSW/Flat индексы |
| 06 | `06_qe_12combos.py` | 4 метода × 3 алайнера = 12 комбинаций |
| 07 | `07_extra_methods.py` | ThinkQE + GenCRF |
| 08 | `08_summary.py` | Сводная таблица результатов |

**Методы QE:** Query2doc, PromptPRF, PQR, Word2Passage, ThinkQE, GenCRF  
**Алайнеры:** `none` / `CSQE` / `AQE`  
**Модели:** multilingual-e5-{small,base,large}, distiluse-base-multilingual-cased-v2, BAAI/bge-m3

### Запуск search pipeline

```bash
# все этапы на всех датасетах
python search/pipeline/00_build.py

# выборочно
python search/pipeline/00_build.py --datasets nfcorpus,scifact --stages 1-4

# с кастомной LLM
QE_LLM=Qwen/Qwen2.5-3B-Instruct python search/pipeline/00_build.py --stages 6
```

Подробно: [`search/pipeline/README.md`](search/pipeline/README.md)

---

## dataset/ — Датасет CyberLeninka

```
dataset/
├── beir/                  # BEIR-формат
│   ├── corpus.jsonl       # 80 статей {_id, title, text}
│   ├── queries.jsonl      # 208 запросов {_id, text, topic, kind}
│   ├── metadata.jsonl     # авторы, журнал, год, URL
│   └── qrels/test.tsv     # разметка релевантности
├── pdfs/                  # исходные PDF (80 шт.)
├── build_beir.py          # сборка датасета из OCR
├── queries_manual.json    # вручную составленные запросы
└── notebooks/
    └── ocr_pipeline_ru.ipynb
```

- **80 статей** из [CyberLeninka](https://cyberleninka.ru) — 8 областей: биология,
  информатика, науки о Земле, экономика, история, языкознание, право, медицина
- **208 запросов**: 80 ключевых (`kw`) + 80 вопросов (`nl`) + 48 состязательных
- **Схема релевантности:** 2 — точное соответствие, 1 — тематически близко

Подробнее: [`dataset/beir/README.md`](dataset/beir/README.md)

---

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
cp .env.example .env
# заполни .env (см. раздел конфигурации)
```

### Конфигурация (.env)

| Переменная | Назначение |
|---|---|
| `HF_TOKEN` | Токен HuggingFace — для загрузки датасетов поиска |
| `QE_LLM` | LLM для Stage 06/07 (по умолчанию `Qwen/Qwen2.5-3B-Instruct`) |
| `ANTHROPIC_OAUTH_TOKEN` | Токен Claude — для LLM-агентов веб-приложения |
| `MAILTO` | Email для OpenAlex API («вежливый пул») |
| `CACHE_DIR` | Каталог кэша веб-приложения |
| `DB_URL` | Строка подключения к БД-кэшу (SQLite) |
| `EMBEDDING_MODEL` | Модель эмбеддингов для семантического поиска |

## Структура репозитория

```
.
├── webapp/            # Backend FastAPI (веб-приложение)
├── web/               # Frontend Vite + React (веб-приложение)
├── search/
│   ├── pipeline/      # Скрипты пайплайна QE
│   ├── notebooks/     # Ноутбуки экспериментов
│   └── results/       # CSV с результатами
├── dataset/
│   ├── beir/          # BEIR-датасет
│   ├── pdfs/          # Исходные PDF
│   ├── build_beir.py
│   └── notebooks/     # OCR-пайплайн
├── .env.example
├── requirements.txt
└── README.md
```
