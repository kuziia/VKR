"""Утилиты общего назначения для всех этапов пайплайна:

- Загрузка датасетов (full, без редукции)
- Encoding через SentenceTransformer (с per-model prefixes/max_seq_length)
- Chunked retrieve (безопасный для 9.5M doc'ов miracl)
- Метрики recall@k, map@k, ndcg@k

Все функции stateless — каждый этап создаёт/освобождает свои ресурсы.
"""
from __future__ import annotations

import json
import hashlib
import math
import pickle
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

# Чтобы скрипты ловили _config независимо от способа запуска
sys.path.insert(0, str(Path(__file__).parent))

from _config import (
    DATASETS, MODELS_BY_NAME, QRELS_SPLIT_PREF,
    EMB_BATCH, EMB_CHUNK, RETRIEVE_CHUNK,
    model_slug,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TEXT_SOURCE_VERSION = "processed_title+processed_text:v1"


# =============================================================================
# Загрузка датасетов
# =============================================================================
def _proc(row: dict, key: str = "processed_text") -> str:
    val = row.get(key)
    return (val or "").strip()


def _id_field(ds) -> str:
    return "_id" if "_id" in ds.column_names else "id"


def _pick_qrels_split(qrels_d, prefer: list[str] | None = None):
    prefer = prefer or QRELS_SPLIT_PREF
    for s in prefer:
        if s in qrels_d:
            return s, qrels_d[s]
    k = next(iter(qrels_d.keys()))
    return k, qrels_d[k]


def load_full_dataset(name: str, info: dict | None = None,
                      qrels_split: str | None = None) -> dict:
    """Грузит (corpus, queries, qrels). Фильтрует queries до тех, что в qrels.
    `qrels_split` — явно указать split (например 'dev' для miracl).
    """
    info = info or DATASETS[name]
    print(f"[load] {name}")
    corpus  = load_dataset(info["corpus"], "corpus")
    corpus  = corpus[next(iter(corpus.keys()))]
    queries = load_dataset(info["corpus"], "queries")
    queries = queries[next(iter(queries.keys()))]
    qrels_d = load_dataset(info["qrels"])
    if qrels_split is not None and qrels_split in qrels_d:
        split, qrels = qrels_split, qrels_d[qrels_split]
    else:
        split, qrels = _pick_qrels_split(qrels_d)

    qrels_rows = [
        {"query-id":  str(r["query-id"]),
         "corpus-id": str(r["corpus-id"]),
         "score":     int(r["score"])}
        for r in qrels
    ]
    relevant_qids = {r["query-id"] for r in qrels_rows if r["score"] > 0}

    qid_field = _id_field(queries)
    queries = queries.filter(
        lambda x: str(x[qid_field]) in relevant_qids,
        num_proc=4, load_from_cache_file=True,
    )
    cid_field = _id_field(corpus)
    print(f"  qrels-split = {split:<10} | rows={len(qrels_rows):>6,} | "
          f"uniq queries={len(relevant_qids):>5,}")
    print(f"  corpus={len(corpus):>10,} | queries (filtered)={len(queries):>5,}")
    return {
        "name":       name,
        "corpus":     corpus,
        "queries":    queries,
        "qrels":      qrels_rows,
        "qid_field":  qid_field,
        "cid_field":  cid_field,
        "split":      split,
    }


def get_corpus_texts_ids(d: dict, text_field: str = "processed") -> tuple[list[str], list[str]]:
    """Возвращает (ids, texts) для корпуса.

    text_field:
      "processed" — title + body из processed_title + processed_text
                    (лемматизированные, lowercased) — для BM25.
      "raw"       — title + body из text + title в исходном виде —
                    для dense-моделей (e5/bge), которые лучше работают на
                    natural language с сохранённой морфологией.
    """
    cid_field = d["cid_field"]
    if text_field not in ("processed", "raw"):
        raise ValueError(f"text_field must be 'processed' or 'raw', got {text_field}")
    ids, texts = [], []
    for r in d["corpus"]:
        if text_field == "processed":
            title = _proc(r, "processed_title")
            body  = _proc(r, "processed_text")
        else:  # raw
            title = (r.get("title") or "").strip()
            body  = (r.get("text") or "").strip()
            # fallback на processed_*, если raw полей нет
            if not body:
                body = _proc(r, "processed_text")
            if not title:
                title = _proc(r, "processed_title")
        full = (title + " " + body).strip() or " "
        ids.append(str(r[cid_field]))
        texts.append(full)
    return ids, texts


def get_query_texts_ids(d: dict, text_field: str = "processed") -> tuple[list[str], list[str]]:
    """Возвращает (ids, texts) для query.

    text_field:
      "processed" — processed_text (для BM25, AQE-фильтра, lemmatized retrieval).
      "raw"       — text (для dense-моделей и cross-encoder rerank'ера).
    """
    qid_field = d["qid_field"]
    ids = [str(r[qid_field]) for r in d["queries"]]
    if text_field == "raw":
        texts = []
        for r in d["queries"]:
            t = (r.get("text") or "").strip()
            if not t:
                t = _proc(r, "processed_text") or " "
            texts.append(t)
    else:
        texts = [_proc(r, "processed_text") or " " for r in d["queries"]]
    return ids, texts


# =============================================================================
# Encoding через SentenceTransformer (универсально для всех моделей в MODELS)
# =============================================================================
class STEncoder:
    """Wrapper над SentenceTransformer, учитывающий model-specific prefix и max_seq_length.

    `text_field` определяет поведение лемматизации в encode_queries:
        "processed" — корпус уже лемматизирован (processed_text); query лемматизируем
                       для матчинга (default lemmatize=True).
        "raw"       — корпус в естественном виде (text); query НЕ лемматизируем,
                       чтобы не ломать natural language для dense-моделей
                       (default lemmatize=False).
    """

    def __init__(self, model_info: dict, device: str = "cuda:0", fp16: bool = True,
                 text_field: str = "processed"):
        from sentence_transformers import SentenceTransformer
        self.info = model_info
        self.text_field = text_field
        self.model = SentenceTransformer(
            model_info["name"], device=device,
            trust_remote_code=model_info.get("trust_remote_code", False),
        )
        self.model.max_seq_length = int(model_info.get("max_seq_length", 512))
        if fp16 and device.startswith("cuda"):
            self.model = self.model.half()
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], prefix: str,
                batch_size: int = EMB_BATCH) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float16)
        out = self.model.encode(
            [prefix + (t or " ") for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return out.astype(np.float16)

    def encode_passages(self, texts: list[str], chunk: int = EMB_CHUNK) -> np.ndarray:
        prefix = self.info.get("passage_prefix", "")
        pieces = []
        for i in tqdm(range(0, len(texts), chunk),
                      desc=f"encode-passages [{self.info['name']}]"):
            pieces.append(self._encode(texts[i:i + chunk], prefix))
        return (np.concatenate(pieces, axis=0) if pieces
                else np.zeros((0, self.dim), dtype=np.float16))

    def encode_queries(self, texts: list[str], lemmatize: bool | None = None) -> np.ndarray:
        """Энкодит query-тексты с опциональной русской лемматизацией.

        `lemmatize=None` (по умолчанию) — выбор зависит от `text_field` энкодера:
            "processed" → lemmatize=True (выравнивает LLM-генерации с
                          лемматизированным корпусом kaengreg/processed_text)
            "raw"       → lemmatize=False (корпус в natural language; lemma
                          ломает natural-text-style для dense-моделей)
        Можно явно передать `lemmatize=True/False`.
        """
        if lemmatize is None:
            lemmatize = (self.text_field == "processed")
        prefix = self.info.get("query_prefix", "")
        if lemmatize:
            texts = [lemmatize_ru(t) for t in texts]
        return self._encode(texts, prefix)

    def free(self):
        import gc
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Кэш эмбеддингов: per (model, dataset) — для Stage 01 (compare)
# =============================================================================
def _tf_suffix(text_field: str) -> str:
    """Suffix для cache-файлов по text_field. processed → '' (back-compat); raw → '_raw'."""
    return "" if text_field == "processed" else f"_{text_field}"


def emb_cache_paths(cache_dir: Path, model_name: str, ds_name: str,
                    subdir: str = "embeddings_compare",
                    text_field: str = "processed") -> tuple[Path, Path]:
    """(emb_path, meta_path) — кэш зависит от text_field, чтобы processed/raw
    не перетирали друг друга."""
    base = cache_dir / subdir / model_slug(model_name)
    base.mkdir(parents=True, exist_ok=True)
    s = _tf_suffix(text_field)
    return base / f"{ds_name}_emb{s}.npy", base / f"{ds_name}_meta{s}.pkl"


def query_emb_cache_path(cache_dir: Path, model_name: str, ds_name: str,
                         subdir: str = "embeddings_compare",
                         text_field: str = "processed") -> Path:
    base = cache_dir / subdir / model_slug(model_name)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{ds_name}_qemb{_tf_suffix(text_field)}.npy"


def query_emb_meta_cache_path(cache_dir: Path, model_name: str, ds_name: str,
                              subdir: str = "embeddings_compare",
                              text_field: str = "processed") -> Path:
    base = cache_dir / subdir / model_slug(model_name)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{ds_name}_qemb_meta{_tf_suffix(text_field)}.json"


def corpus_cache_is_current(cache_dir: Path, model_name: str, ds_name: str,
                            subdir: str = "embeddings_compare",
                            text_field: str = "processed") -> bool:
    emb_path, meta_path = emb_cache_paths(cache_dir, model_name, ds_name, subdir, text_field)
    if not emb_path.exists() or not meta_path.exists():
        return False
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return (meta.get("text_source") == TEXT_SOURCE_VERSION
            and meta.get("text_field", "processed") == text_field)


def query_cache_is_current(cache_dir: Path, model_name: str, ds_name: str,
                           subdir: str = "embeddings_compare",
                           text_field: str = "processed") -> bool:
    qpath = query_emb_cache_path(cache_dir, model_name, ds_name, subdir, text_field)
    qmeta_path = query_emb_meta_cache_path(cache_dir, model_name, ds_name, subdir, text_field)
    if not qpath.exists() or not qmeta_path.exists():
        return False
    qmeta = json.loads(qmeta_path.read_text(encoding="utf-8"))
    return (qmeta.get("text_source") == TEXT_SOURCE_VERSION
            and qmeta.get("text_field", "processed") == text_field)


def build_or_load_corpus_index(cache_dir: Path, model_info: dict,
                               d: dict, encoder: STEncoder | None = None,
                               subdir: str = "embeddings_compare",
                               text_field: str = "processed") -> dict:
    """Кодирует или загружает корпусные эмбеддинги. encoder создаётся снаружи и
    переиспользуется на нескольких датасетах.

    text_field: "processed" (BM25-style) или "raw" (natural text для dense).
    """
    ds_name = d["name"]
    emb_path, meta_path = emb_cache_paths(cache_dir, model_info["name"], ds_name, subdir, text_field)
    if emb_path.exists() and meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if (meta.get("text_source") == TEXT_SOURCE_VERSION
                and meta.get("text_field", "processed") == text_field):
            emb = np.load(emb_path)
            print(f"  [{ds_name}] load cache (text_field={text_field}): "
                  f"{emb.shape} {emb.dtype} ({emb.nbytes / 1e9:.2f} GB)")
            return {"ids": meta["ids"], "texts": meta["texts"], "emb": emb}
        print(f"  [{ds_name}] stale text cache; rebuilding with text_field={text_field}")

    assert encoder is not None, "Encoder required for fresh encoding"
    ids, texts = get_corpus_texts_ids(d, text_field=text_field)
    print(f"  [{ds_name}] encoding {len(texts):,} docs (text_field={text_field}) ...")
    emb = encoder.encode_passages(texts)
    np.save(emb_path, emb)
    with open(meta_path, "wb") as f:
        pickle.dump({
            "ids": ids,
            "texts": texts,
            "text_source": TEXT_SOURCE_VERSION,
            "text_field": text_field,
        }, f, protocol=4)
    print(f"  [{ds_name}] saved {emb.nbytes / 1e9:.2f} GB -> {emb_path.name}")
    return {"ids": ids, "texts": texts, "emb": emb}


def build_or_load_query_emb(cache_dir: Path, model_info: dict, d: dict,
                            encoder: STEncoder | None = None,
                            subdir: str = "embeddings_compare",
                            text_field: str = "processed"
                            ) -> tuple[list[str], list[str], np.ndarray]:
    ds_name = d["name"]
    qids, qtexts = get_query_texts_ids(d, text_field=text_field)
    qpath = query_emb_cache_path(cache_dir, model_info["name"], ds_name, subdir, text_field)
    qmeta_path = query_emb_meta_cache_path(cache_dir, model_info["name"], ds_name, subdir, text_field)
    if qpath.exists() and qmeta_path.exists():
        qmeta = json.loads(qmeta_path.read_text(encoding="utf-8"))
        if (qmeta.get("text_source") == TEXT_SOURCE_VERSION
                and qmeta.get("text_field", "processed") == text_field):
            return qids, qtexts, np.load(qpath)
        print(f"  [{ds_name}] stale query cache; rebuilding (text_field={text_field})")
    elif qpath.exists():
        print(f"  [{ds_name}] query cache has no metadata; rebuilding (text_field={text_field})")
    assert encoder is not None, "Encoder required for fresh query encoding"
    # Если корпус закэширован в raw — НЕ лемматизируем queries (соответствие модели)
    qemb = encoder.encode_queries(qtexts, lemmatize=(text_field == "processed"))
    np.save(qpath, qemb)
    qmeta_path.write_text(
        json.dumps({"text_source": TEXT_SOURCE_VERSION, "text_field": text_field},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return qids, qtexts, qemb


# =============================================================================
# Retrieval: chunked brute-force
# =============================================================================
def chunked_retrieve(query_emb: np.ndarray, corpus_emb: np.ndarray,
                     top_k: int, chunk: int = RETRIEVE_CHUNK):
    """Top-k retrieval по чанкам корпуса. Безопасно для 9.5M doc'ов.
    Возвращает (idx, scores) фигуры (Q, top_k).
    """
    n_q, n_c = query_emb.shape[0], corpus_emb.shape[0]
    top_k = min(top_k, n_c)
    if n_c == 0 or n_q == 0:
        return (np.zeros((n_q, 0), dtype=np.int64),
                np.zeros((n_q, 0), dtype=np.float32))

    qe = query_emb.astype(np.float32, copy=False)
    if n_c <= chunk:
        sims = qe @ corpus_emb.astype(np.float32).T
        part = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
        rows = np.arange(n_q)[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        idx = part[rows, order]
        return idx, sims[rows, idx]

    best_scores = np.full((n_q, top_k), -np.inf, dtype=np.float32)
    best_idx    = np.zeros((n_q, top_k), dtype=np.int64)
    for start in range(0, n_c, chunk):
        end = min(start + chunk, n_c)
        sims_chunk = qe @ corpus_emb[start:end].astype(np.float32).T
        local_idx = np.tile(np.arange(start, end), (n_q, 1))
        merged_s = np.concatenate([best_scores, sims_chunk], axis=1)
        merged_i = np.concatenate([best_idx, local_idx], axis=1)
        part = np.argpartition(-merged_s, kth=top_k - 1, axis=1)[:, :top_k]
        rows = np.arange(n_q)[:, None]
        best_scores = merged_s[rows, part]
        best_idx    = merged_i[rows, part]
    order = np.argsort(-best_scores, axis=1)
    rows = np.arange(n_q)[:, None]
    return best_idx[rows, order], best_scores[rows, order]


# =============================================================================
# Метрики
# =============================================================================
def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def average_precision_at_k(retrieved: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    rel = set(relevant); hits = 0; ap = 0.0
    for i, doc in enumerate(retrieved[:k], 1):
        if doc in rel:
            hits += 1
            ap += hits / i
    return ap / min(k, len(relevant))


def ndcg_at_k(retrieved: list, rel_dict: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        r = rel_dict.get(doc, 0)
        if r > 0:
            dcg += (2 ** r - 1) / math.log2(i + 2)
    ideal = sorted(rel_dict.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(ret_per_q: dict, qrels_rows: list, ks: tuple[int, ...] = (5, 10)) -> dict:
    """Возвращает dict со всеми метриками для каждого k."""
    qid2rel: dict[str, dict[str, int]] = {}
    for r in qrels_rows:
        qid2rel.setdefault(r["query-id"], {})[r["corpus-id"]] = r["score"]
    out: dict[str, float] = {}
    n_eval = 0
    for k in ks:
        Rs, MAPs, NDCGs = [], [], []
        for qid, ret in ret_per_q.items():
            rel_dict = qid2rel.get(qid, {})
            relevant = [d for d, s in rel_dict.items() if s > 0]
            if not relevant:
                continue
            Rs.append(recall_at_k(ret, relevant, k))
            MAPs.append(average_precision_at_k(ret, relevant, k))
            NDCGs.append(ndcg_at_k(ret, rel_dict, k))
        out[f"recall@{k}"] = float(np.mean(Rs)) if Rs else 0.0
        out[f"map@{k}"]    = float(np.mean(MAPs)) if MAPs else 0.0
        out[f"ndcg@{k}"]   = float(np.mean(NDCGs)) if NDCGs else 0.0
        n_eval = max(n_eval, len(Rs))
    out["n_eval"] = n_eval
    return out


# =============================================================================
# State (best_embedding / best_retriever)
# =============================================================================
def state_path(cache_dir: Path, key: str) -> Path:
    p = cache_dir / "state"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{key}.json"


def write_state(cache_dir: Path, key: str, data: dict) -> None:
    state_path(cache_dir, key).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_state(cache_dir: Path, key: str) -> dict | None:
    p = state_path(cache_dir, key)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# =============================================================================
# Простая токенизация (для BM25) + русский stop-list
# =============================================================================
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


# Русские стоп-слова. Стандартный nltk-список (~150) + few extra для научного
# текста («исследование», «результат» — слишком частотные, плохие признаки).
RU_STOPWORDS: set[str] = set()


def _load_stopwords() -> set[str]:
    """Lazy-load: nltk-список (если есть), иначе встроенный fallback."""
    global RU_STOPWORDS
    if RU_STOPWORDS:
        return RU_STOPWORDS
    try:
        from nltk.corpus import stopwords as _nltk_stopwords
        RU_STOPWORDS = set(_nltk_stopwords.words("russian"))
    except (ImportError, LookupError):
        # Fallback: вшитый минимальный список (~80 самых частых)
        RU_STOPWORDS = {
            "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
            "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
            "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
            "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
            "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни",
            "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж",
            "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может",
            "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
            "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз",
            "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот",
            "того", "потому", "этого", "какой", "совсем", "ним", "здесь",
            "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас",
            "были", "куда", "зачем", "всех", "никогда", "можно", "при",
            "наконец", "два", "об", "другой", "хоть", "после", "над", "больше",
            "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
            "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
            "свою", "этой", "перед", "иногда", "лучше", "чуть", "том",
            "нельзя", "такой", "им", "более", "всегда", "конечно", "всю",
            "между",
        }
    # extra для научного текста — наиболее частотные content-words без discriminative power
    RU_STOPWORDS |= {"это", "также", "однако", "являться", "становиться"}
    return RU_STOPWORDS


def tokenize_simple(text: str, drop_stopwords: bool = True) -> list[str]:
    """Простая токенизация для BM25.

    `drop_stopwords=True` (default) — выкидывает русские stop-слова.
    Их низкий IDF в научных корпусах редко даёт сигнал, чаще шум.
    """
    toks = (t.lower() for t in _TOKEN_RE.findall(text or ""))
    if drop_stopwords:
        stops = _load_stopwords()
        return [t for t in toks if t not in stops]
    return list(toks)


# =============================================================================
# Лемматизация русского текста (pymorphy3)
# =============================================================================
# Датасеты `kaengreg/*` хранят корпус и запросы в `processed_text` —
# уже лемматизованные с lowercased. LLM-генерации (Q2D, PromptPRF, PQR, W2P,
# ThinkQE, GenCRF, CSQE) приходят в обычной русской флексии («лечения хено-
# дезоксихолевой кислоты»), что даёт vocabulary mismatch с корпусом
# («лечение хенодезоксихолев кислота»). Чтобы выровнять — лемматизируем
# любой query-текст перед encoder.encode_queries().
_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        from pymorphy3 import MorphAnalyzer
        _morph = MorphAnalyzer()
    return _morph


def lemmatize_ru(text: str) -> str:
    """Lowercased + слова в нормальной форме (приводит к виду processed_text).
    Идемпотентно — лемма уже лемматизированного слова = она сама."""
    if not text:
        return ""
    morph = _get_morph()
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return ""
    return " ".join(morph.parse(t)[0].normal_form for t in tokens)


# =============================================================================
# Логирование: тиим stdout/stderr в qe_cache/logs/{stage}_{ts}.log
# =============================================================================
class _LogTee:
    """Тиит stdout/stderr в живой терминал и в файл одновременно.

    Из лог-файла удаляются \\r-перезаписи (прогресс-бары tqdm) — пишется
    только финальная строка после последнего \\r. В терминале всё работает
    как обычно.
    """
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file
        self._buf = ""

    def write(self, data: str) -> int:
        # терминал — как есть
        try:
            self.terminal.write(data)
            self.terminal.flush()
        except Exception:
            pass
        # файл — без промежуточных \r-обновлений (прогресс-бары tqdm)
        if "\r" in data:
            tail = data.rsplit("\r", 1)[-1]
            if not tail:
                return len(data)
            data = tail
        # буферизуем всё, флашим построчно с timestamp
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            ts = datetime.now().strftime("%H:%M:%S")
            try:
                self.file.write(f"[{ts}] {line}\n")
            except Exception:
                pass
        try:
            self.file.flush()
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            if self._buf:
                ts = datetime.now().strftime("%H:%M:%S")
                self.file.write(f"[{ts}] {self._buf}")
                self._buf = ""
            self.file.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self) -> int:
        # vLLM (и subprocess'ы) делают os.dup2(devnull, sys.stdout.fileno()),
        # т.е. перенаправляют ВЫВОД на FD-уровне. Делегируем на терминал —
        # его FD будет переадресован, в файл лог по-прежнему пишется через
        # self.file (отдельный FD).
        return self.terminal.fileno()

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", "utf-8") or "utf-8"

    @property
    def errors(self):
        return getattr(self.terminal, "errors", "replace")


_active_log_files: list[Any] = []


def setup_log_file(cache_dir: Path, stage_name: str) -> Path:
    """Активирует лог-файл `qe_cache/logs/{stage}_{ts}.log` и тиит туда stdout/stderr.
    Все последующие print() будут уходить и в терминал, и в файл с timestamp.
    Безопасно вызывать повторно — закроет старый tee и создаст новый.
    """
    log_dir = Path(cache_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{stage_name}_{ts}.log"

    # снимаем предыдущий tee, если был
    if isinstance(sys.stdout, _LogTee):
        try:
            sys.stdout.flush()
            sys.stdout.file.close()
        except Exception:
            pass
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    f = open(log_path, "a", encoding="utf-8", buffering=1)
    _active_log_files.append(f)
    f.write("\n" + "=" * 70 + "\n")
    f.write(f"=== {stage_name}\n")
    f.write(f"=== started at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write("=" * 70 + "\n\n")
    f.flush()

    sys.stdout = _LogTee(sys.__stdout__, f)
    sys.stderr = _LogTee(sys.__stderr__, f)

    print(f"[log] {log_path}")
    return log_path


def close_log_file() -> None:
    """Снимает текущий tee и закрывает лог-файл."""
    if isinstance(sys.stdout, _LogTee):
        try:
            sys.stdout.flush()
            sys.stdout.file.write(f"\n=== finished at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            sys.stdout.file.close()
        except Exception:
            pass
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


# =============================================================================
# BM25 retrieval cache
# =============================================================================
def bm25_cache_dir(cache_dir: Path) -> Path:
    p = cache_dir / "retrievals_bm25"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bm25_cache_path(cache_dir: Path, ds_name: str, top_k: int) -> Path:
    return bm25_cache_dir(cache_dir) / f"{ds_name}_top{top_k}.json"


def _texts_fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update((text or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _load_bm25_rankings(cache_dir: Path, ds_name: str, top_k: int,
                        n_queries: int, n_docs: int,
                        query_hash: str) -> dict[int, list[str]] | None:
    paths = sorted(
        bm25_cache_dir(cache_dir).glob(f"{ds_name}_top*.json"),
        key=lambda p: int(re.search(r"_top(\d+)\.json$", p.name).group(1))
        if re.search(r"_top(\d+)\.json$", p.name) else -1,
    )
    for path in paths:
        m = re.search(r"_top(\d+)\.json$", path.name)
        if not m or int(m.group(1)) < top_k:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as ex:
            backup = path.with_suffix(path.suffix + ".corrupt")
            path.replace(backup)
            print(f"  [BM25 cache] corrupt cache moved to {backup}: {ex}")
            continue
        if payload.get("n_queries") != n_queries or payload.get("n_docs") != n_docs:
            continue
        if payload.get("query_hash") != query_hash:
            continue
        ranked = payload.get("ranked")
        if not isinstance(ranked, list) or len(ranked) != n_queries:
            continue
        print(f"  [BM25 cache] load {path.name}")
        return {qi: [str(d) for d in ranked[qi][:top_k]]
                for qi in range(n_queries)}
    return None


def _save_bm25_rankings(cache_dir: Path, ds_name: str, top_k: int,
                        ranked: dict[int, list[str]],
                        n_queries: int, n_docs: int,
                        query_hash: str) -> None:
    path = _bm25_cache_path(cache_dir, ds_name, top_k)
    payload = {
        "dataset": ds_name,
        "top_k": top_k,
        "n_queries": n_queries,
        "n_docs": n_docs,
        "query_hash": query_hash,
        "ranked": [ranked.get(qi, [])[:top_k] for qi in range(n_queries)],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    print(f"  [BM25 cache] saved {path}")


def _bm25_index_and_search(corpus_texts: list[str], queries: list[str],
                           corpus_ids: list[str], top_k: int) -> dict[int, list[str]]:
    """Build a temporary BM25 index and return top-k document ids per query."""
    import gc

    try:
        import bm25s
        print("  [BM25s] tokenize + index ...")
        corpus_tok = bm25s.tokenize(corpus_texts, stopwords=None)
        retriever = bm25s.BM25()
        retriever.index(corpus_tok)
        print("  [BM25s] retrieve ...")
        q_tok = bm25s.tokenize(queries, stopwords=None)
        docs, _ = retriever.retrieve(q_tok, k=top_k)
        return {
            qi: [corpus_ids[int(j)] for j in docs[qi]]
            for qi in range(len(queries))
        }
    except ImportError:
        pass

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise SystemExit(
            "Neither bm25s nor rank_bm25 installed. Run:\n"
            "  pip install bm25s\n"
            "  pip install rank-bm25"
        ) from e

    print("  [BM25Okapi] tokenize + index ...")
    tok_corpus = [tokenize_simple(t) for t in tqdm(corpus_texts, desc="tok")]
    bm25 = BM25Okapi(tok_corpus)
    del tok_corpus
    gc.collect()
    out: dict[int, list[str]] = {}
    for qi, qtxt in enumerate(tqdm(queries, desc="bm25-query")):
        scores = bm25.get_scores(tokenize_simple(qtxt))
        n_ret = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -n_ret)[-n_ret:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        out[qi] = [corpus_ids[j] for j in top_idx]
    del bm25
    gc.collect()
    return out


def build_or_load_bm25_index(cache_dir: Path, ds_name: str,
                             corpus_texts: list[str],
                             cache_version: str = "v2stop"):
    """Кэш BM25Okapi-индекса (rank_bm25). Один индекс на (ds_name, cache_version),
    переиспользуется между Stage 2 и Stage 6.

    cache_version: тег для инвалидации, когда меняется поведение `tokenize_simple`.
        v1     — без стоп-слов (legacy).
        v2stop — со стоп-словами (Phase 1+).
    """
    p = cache_dir / "bm25_indexes" / f"{ds_name}_{cache_version}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            with open(p, "rb") as f:
                bm25 = pickle.load(f)
            print(f"  [BM25 index] loaded cache: {p.name}")
            return bm25
        except Exception as ex:
            print(f"  [BM25 index] cache load failed ({ex}); rebuilding")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise SystemExit("rank_bm25 not installed: pip install rank-bm25") from e
    print(f"  [BM25 index] building for {ds_name} ({len(corpus_texts):,} docs) ...")
    tok = [tokenize_simple(t) for t in tqdm(corpus_texts, desc="tok")]
    bm25 = BM25Okapi(tok)
    with open(p, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [BM25 index] cached -> {p}")
    return bm25


def build_or_load_bm25_rankings(cache_dir: Path, ds_name: str,
                                corpus_texts: list[str], queries: list[str],
                                corpus_ids: list[str], top_k: int,
                                cache_top_k: int | None = None) -> dict[int, list[str]]:
    """Load cached BM25 rankings or build them once for a dataset/query set.

    The cache stores only ranked document ids, not the BM25 index. This is small
    enough for MIRACL top-100 and avoids rebuilding BM25 in later stages.
    """
    cache_top_k = min(max(cache_top_k or top_k, top_k), len(corpus_ids))
    query_hash = _texts_fingerprint(queries)
    cached = _load_bm25_rankings(
        cache_dir, ds_name, top_k, n_queries=len(queries), n_docs=len(corpus_ids),
        query_hash=query_hash,
    )
    if cached is not None:
        return cached

    # Используем общий кэш BM25-индекса (используется Stage 6 retriever'ами).
    # Если индекс уже на диске — миллисекунды; иначе строим и кэшируем.
    bm25 = build_or_load_bm25_index(cache_dir, ds_name, corpus_texts)
    ranked: dict[int, list[str]] = {}
    for qi, qtxt in enumerate(tqdm(queries, desc="bm25-query")):
        scores = bm25.get_scores(tokenize_simple(qtxt))
        n_ret = min(cache_top_k, len(scores))
        top_idx = np.argpartition(scores, -n_ret)[-n_ret:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        ranked[qi] = [corpus_ids[int(j)] for j in top_idx]
    _save_bm25_rankings(
        cache_dir, ds_name, cache_top_k, ranked,
        n_queries=len(queries), n_docs=len(corpus_ids), query_hash=query_hash,
    )
    return {qi: ranked.get(qi, [])[:top_k] for qi in range(len(queries))}
