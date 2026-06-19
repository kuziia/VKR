# Pipeline для экспериментов QE на русско-научных датасетах

7-этапный пайплайн с резюмируемым кэшем. Все артефакты пишутся в `qe_cache/`
(можно переопределить через `--cache-dir` или env `QE_CACHE_DIR`).

## Этапы

| # | Файл | Что делает | Зависит от |
|---|---|---|---|
| 01 | `01_compare_embeddings.py` | Кодирует датасеты 4 моделями, считает recall/map/ndcg, выбирает best | — |
| 02 | `02_bm25_baseline.py` | BM25 + dense baseline (без QE) | (01 опционально, для best dense) |
| 03 | `03_encode_best.py` | Сохраняет manifest на эмбеддинги выбранной модели | 01 |
| 04 | `04_build_faiss.py` | Строит HNSW (или Flat) FAISS-индекс | 03 |
| 06 | `06_qe_12combos.py` | 4 методa × {none, CSQE, AQE} = 12 combos через vLLM (dense retriever) | 03, 04 |
| 07 | `07_extra_methods.py` | ThinkQE × 3 aligner + GenCRF (multi-cluster RRF) | 03, 04 |
| 08 | `08_summary.py` | Сводка всех CSV в одну `all_results.csv` + pivot | 02, 06, 07 |

> **Stage 05 (выбор ретривера) удалён.** В пайплайне теперь всегда используется
> dense retriever на лучшей embedding-модели через FAISS-индекс. Если нужен
> ablation с BM25 / hybrid_rrf — это можно сделать через прямой запуск:
> `python search/pipeline/06_qe_12combos.py --retriever bm25` (флаг сохранён).

## Логирование

Каждая стадия при запуске создаёт файл лога в `qe_cache/logs/`:

```
qe_cache/logs/
  pipeline_orchestrator_20260426_235012.log   # общий лог 00_build.py
  stage01_compare_embeddings_20260426_235013.log
  stage02_bm25_baseline_20260426_235214.log
  ...
```

В каждой строке лога — timestamp `[HH:MM:SS]`. Прогресс-бары tqdm
(перерисовка через `\r`) **не засоряют** лог-файл — пишется только финальная
строка. В терминал всё показывается как обычно.

Чтобы посмотреть прогресс долгого прогона из другой сессии:
```bash
tail -f qe_cache/logs/stage06_qe_12combos_*.log
```

## Установка

```bash
pip install -U sentence-transformers datasets transformers accelerate
pip install faiss-cpu rank-bm25 bm25s tqdm pandas scikit-learn
pip install vllm  # для Stage 06
```

## Полный прогон

```bash
# Все этапы на всех датасетах
python search/pipeline/00_build.py

# С указанием путей и LLM
QE_CACHE_DIR=/mnt/nvme/qe_cache HF_HOME=/mnt/nvme/hf \
python search/pipeline/00_build.py \
    --datasets all \
    --stages all \
    --llm-model Qwen/Qwen2.5-3B-Instruct
```

## Запуск отдельных этапов

Каждый скрипт можно запустить независимо:

```bash
# Только сравнение эмбеддингов на 2 датасетах
python search/pipeline/01_compare_embeddings.py --datasets nfcorpus,scifact

# Stages 3-6, форсим конкретную модель эмбеддингов
python search/pipeline/00_build.py --stages 3-6 --override-embedding BAAI/bge-m3

# Только Stage 6 (предполагает 1-5 уже прогонены)
python search/pipeline/00_build.py --stages 6
```

## Параметр --datasets

Алиасы из `_config.py`:

- `nfcorpus`     — `kaengreg/rus-nfcorpus`     (3,630 docs / ~323 q)
- `scifact`      — `kaengreg/rus-scifact`      (1,109 docs / ~300 q)
- `ruSciBench`   — `kaengreg/ruSciBench-retrieval` (201k docs / 1,577 q)
- `miracl`       — `kaengreg/rus-miracl`       (9.54M docs / 1,252 q dev-split)

Передавать через запятую без пробелов:
```bash
--datasets nfcorpus,scifact,ruSciBench
--datasets all
```

## Состояние пайплайна

`qe_cache/state/`:
- `best_embedding.json`           ← Stage 01
- `best_embedding_manifest.json`  ← Stage 03 (пути на .npy эмбеддинги)
- `faiss_manifest.json`           ← Stage 04

Stages 03-08 читают это состояние; CLI-флаг `--override-embedding`
позволяет форсить модель вручную (например, прогнать Stage 06 со слабой
моделью эмбеддингов для ablation).

## Resume после крэша

Все этапы дописывают результаты в CSV-файл после **каждой** пары
(model, dataset) или (combination, dataset). Повторный запуск пропустит
уже посчитанные строки и продолжит с того же места.

Для force-rebuild — удали соответствующий артефакт:
```bash
rm qe_cache/eval/embeddings_compare.csv  # перезапустит Stage 01
rm qe_cache/faiss/{ds}.index             # пересоберёт индекс для датасета
```

## Артефакты

```
qe_cache/
├── embeddings_compare/          # Stage 01: per-model emb для всех датасетов
│   └── {model_slug}/
│       ├── {ds}_emb.npy         # corpus embeddings (fp16)
│       ├── {ds}_meta.pkl        # ids + texts
│       └── {ds}_qemb.npy        # query embeddings
├── embeddings/best/
│   └── manifest.json            # пути на best-model эмбеддинги
├── faiss/
│   ├── {ds}.index               # FAISS HNSW/Flat
│   └── manifest.json
├── llm_outputs/                 # Stage 06: per-method JSON, resume-friendly
│   ├── Query2doc_{ds}.json
│   ├── PromptPRF_{ds}.json
│   ├── PQR_{ds}.json
│   ├── Word2Passage_{ds}.json
│   └── PromptPRF_features_{ds}_{ftype}.json
├── retrievals/                  # Stage 06: top-k doc_ids per (combo, ds)
│   └── {method}_{aligner}_{ds}.json
├── eval/
│   ├── embeddings_compare.csv
│   └── baselines.csv
├── results/
│   ├── qe_12combos.csv          # Stage 06 (12 combos × N датасета)
│   ├── qe_extra_methods.csv     # Stage 07 (ThinkQE × 3 + GenCRF)
│   ├── all_results.csv          # Stage 08 — всё в одной таблице
│   ├── summary_by_dataset.csv   # Stage 08 — pivot per-dataset
│   └── summary_mean.csv         # Stage 08 — mean по датасетам, ranked
├── logs/                        # Логи каждой стадии с timestamp [HH:MM:SS]
│   ├── pipeline_orchestrator_*.log
│   └── stageNN_*_*.log
└── state/
    ├── best_embedding.json
    ├── best_embedding_manifest.json
    └── faiss_manifest.json
```
