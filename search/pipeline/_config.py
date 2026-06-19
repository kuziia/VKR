"""Общая конфигурация для всех этапов пайплайна.

Изменяй здесь:
- DATASETS  — список IR-датасетов с алиасами для CLI (--datasets)
- MODELS    — embedding-модели для сравнения (Stage 01)
- LLM_*     — генеративная модель для Stage 06
- TOP_K     — глубина retrieval для метрик
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Загрузка переменных из .env (если есть; не комитится в git).
# Формат файла .env (в корне репозитория, рядом с requirements.txt):
#   HF_TOKEN=hf_xxx...
#   QE_LLM=Qwen/Qwen2.5-3B-Instruct
# ---------------------------------------------------------------------------
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ---------------------------------------------------------------------------
# HuggingFace Hub authentication.
# Токен НЕ хранится в коде. Положи его в .env (рядом с pipeline/) или
# export HF_TOKEN=hf_xxx в shell. Без токена unauth-ed requests жёстко
# рейт-лимитятся (особенно для miracl 9.54M — даунлоад встаёт колом).
# ---------------------------------------------------------------------------
if "HF_TOKEN" in os.environ:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# hf-xet 1.4.3 ловит deadlock в futex на некоторых сетевых конфигурациях
# (futex_wait_queue_me, 0 B/s). Отключаем Xet — стандартный HTTP всегда работает.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Корпоративный SSL-перехват (194.67.95.7:3127):
#   /etc/ssl/certs/ca-certificates.crt содержит ТОЛЬКО прокси-CA (5 KB),
#   без стандартных корневых сертификатов — для huggingface.co этого мало.
#   Системно глобально выставлены SSL_CERT_FILE / REQUESTS_CA_BUNDLE на этот файл.
# Решение: подмешать прокси-CA в certifi-bundle (через `python _install_ca.py` —
# запускается один раз) и принудительно перенаправить env на этот объединённый файл.
try:
    import certifi as _certifi
    _MERGED_CA = _certifi.where()  # certifi-bundle, в который мы добавили proxy-CA
    # Перетираем глобальные настройки — иначе они укажут на маленький системный файл.
    os.environ["SSL_CERT_FILE"]      = _MERGED_CA
    os.environ["REQUESTS_CA_BUNDLE"] = _MERGED_CA
    os.environ["CURL_CA_BUNDLE"]     = _MERGED_CA
    os.environ["PIP_CERT"]           = _MERGED_CA
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Датасеты. Ключ — короткий alias для CLI (`--datasets nfcorpus,scifact,...`).
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "nfcorpus":   {"corpus": "kaengreg/rus-nfcorpus",
                   "qrels":  "kaengreg/rus-nfcorpus-qrels"},
    "scifact":    {"corpus": "kaengreg/rus-scifact",
                   "qrels":  "kaengreg/rus-scifact-qrels"},
    "ruSciBench": {"corpus": "kaengreg/ruSciBench-retrieval",
                   "qrels":  "kaengreg/ruSciBench-retrieval-qrels"},
    "miracl":     {"corpus": "kaengreg/rus-miracl",
                   "qrels":  "kaengreg/rus-miracl-qrels"},
}

# Какой split qrels брать; пробуем по очереди.
QRELS_SPLIT_PREF = ["test", "dev", "validation"]

# ---------------------------------------------------------------------------
# Embedding-модели для сравнения (Stage 01).
# ---------------------------------------------------------------------------
MODELS: list[dict] = [
    {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_seq_length": 512,
        "trust_remote_code": False,
    },
    {
        "name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_seq_length": 512,
        "trust_remote_code": False,
    },
    {
        "name": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_seq_length": 512,
        "trust_remote_code": False,
    },
    {
        "name": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "max_seq_length": 128,
        "trust_remote_code": False,
    },
    {
        "name": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "max_seq_length": 512,
        "trust_remote_code": False,
    },
]

MODELS_BY_NAME: dict[str, dict] = {m["name"]: m for m in MODELS}

# ---------------------------------------------------------------------------
# LLM для Stage 06 (12 combos).
# ---------------------------------------------------------------------------
LLM_MODEL_NAME = os.environ.get("QE_LLM", "Qwen/Qwen2.5-3B-Instruct")
VLLM_GPU_UTIL  = float(os.environ.get("QE_VLLM_UTIL", "0.70"))
VLLM_MAX_LEN   = int(os.environ.get("QE_VLLM_MAXLEN", "4096"))

# ---------------------------------------------------------------------------
# Глобальные параметры.
# ---------------------------------------------------------------------------
TOP_K          = 10           # k для retrieval/метрик в Stage 01-05
QE_K           = 5            # k для финальной таблицы Stage 06
EMB_BATCH      = 192
EMB_CHUNK      = 50_000
RETRIEVE_CHUNK = 200_000      # docs per chunk в chunked retrieve
PRF_DEPTH      = 5            # для CSQE/PromptPRF
DISTRACTOR_NONE = True        # full-corpus, без дистракторов


def get_cache_dir(override: str | os.PathLike | None = None) -> Path:
    """Возвращает корневую папку кэша. Приоритет: arg > env QE_CACHE_DIR > ./qe_cache."""
    if override is not None:
        return Path(override)
    return Path(os.environ.get("QE_CACHE_DIR", "qe_cache"))


def model_slug(name: str) -> str:
    """Безопасное имя для путей: 'org/model-name' -> 'org__model-name'."""
    return name.replace("/", "__")


def parse_datasets(s: str) -> list[str]:
    """Парсит --datasets nfcorpus,scifact или 'all'."""
    s = s.strip()
    if s.lower() == "all":
        return list(DATASETS.keys())
    items = [x.strip() for x in s.split(",") if x.strip()]
    unknown = [x for x in items if x not in DATASETS]
    if unknown:
        raise SystemExit(
            f"Unknown dataset alias(es): {unknown}. "
            f"Available: {list(DATASETS.keys())}"
        )
    return items
