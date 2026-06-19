"""Read-only access to the BERTrend evaluation database.

The DB is produced by the offline `bertrend_evaluation.py` pipeline; we just
read it. Path is configurable via the BERTREND_DB env var (default points to
the bundled `bertrend_emb_extracted/...`).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..settings import settings

log = logging.getLogger(__name__)


@dataclass
class Window:
    window_id: str
    n_docs: int
    n_topics: int
    n_outliers: int


@dataclass
class TopicSummary:
    topic_id: int
    signal: str
    total_docs: int
    peak_count: int
    first_window: str
    last_window: str
    words: list[str]
    history: list[tuple[str, int]]  # [(window_id, count), ...]
    # OpenAlex labels of the paper closest to the cluster's centroid in
    # embedding space — used as a single "representative" mapping rather
    # than frequency-based aggregation.
    centroid: dict[str, Any] = field(default_factory=dict)


def _path() -> Path:
    return settings.bertrend_db


def is_available() -> bool:
    p = _path()
    return p.exists() and p.is_file()


def _conn() -> sqlite3.Connection:
    p = _path()
    uri = f"file:{p}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


@lru_cache(maxsize=1)
def list_windows() -> list[Window]:
    if not is_available():
        return []
    with _conn() as c:
        rows = c.execute(
            "SELECT window_id, n_docs, n_topics, n_outliers FROM time_windows ORDER BY window_id"
        ).fetchall()
    return [Window(*r) for r in rows]


@lru_cache(maxsize=1)
def _doc_record_map() -> dict[str, dict[str, Any]]:
    """doc_id → {primary_topic, primary_subfield, primary_field, title, doi,
    openalex_id} from JSONL. Loaded once, kept in memory (~15 MB)."""
    jsonl_path = settings.bertrend_db.parent / "dataset_filtered.jsonl"
    if not jsonl_path.exists():
        log.warning("BERTrend JSONL missing at %s; centroid mapping disabled", jsonl_path)
        return {}
    out: dict[str, dict[str, Any]] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = r.get("doc_id")
            if not did:
                continue
            out[did] = {
                "primary_topic": r.get("primary_topic") or None,
                "primary_subfield": r.get("primary_subfield") or None,
                "primary_field": r.get("primary_field") or None,
                "title": r.get("title") or None,
                "doi": r.get("doi") or None,
                "openalex_id": (r.get("openalex_id") or "").rsplit("/", 1)[-1] or None,
            }
    log.info("Loaded BERTrend JSONL: %d records", len(out))
    return out


@lru_cache(maxsize=1)
def _embeddings_and_index() -> tuple[np.ndarray | None, dict[str, int]]:
    """Memory-mapped embeddings matrix (34k × 2048 fp32) + doc_id → row index."""
    base = settings.bertrend_db.parent
    emb_path = base / "embeddings.npy"
    ids_path = base / "doc_ids.json"
    if not emb_path.exists() or not ids_path.exists():
        log.warning("BERTrend embeddings or doc_ids missing; centroid mapping disabled")
        return None, {}
    try:
        arr = np.load(emb_path, mmap_mode="r")
        doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.exception("BERTrend embeddings load failed: %s", e)
        return None, {}
    index = {did: i for i, did in enumerate(doc_ids)}
    log.info("Loaded BERTrend embeddings: %s, doc_id index=%d", arr.shape, len(index))
    return arr, index


def _centroid_paper(topic_id: int) -> dict[str, Any]:
    """Find the paper closest to the cluster centroid in embedding space,
    return its OpenAlex labels + title."""
    if not is_available():
        return {}
    with _conn() as c:
        rows = c.execute(
            "SELECT doc_id FROM papers WHERE global_topic_id = ?",
            (topic_id,),
        ).fetchall()
    doc_ids = [r[0] for r in rows]
    if not doc_ids:
        return {}

    arr, idx = _embeddings_and_index()
    if arr is None:
        return {}

    rows_idx = [idx[d] for d in doc_ids if d in idx]
    if not rows_idx:
        return {}

    vectors = np.asarray(arr[rows_idx], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = vectors / norms

    centroid = normed.mean(axis=0)
    c_norm = np.linalg.norm(centroid)
    if c_norm == 0:
        return {}
    centroid /= c_norm

    sims = normed @ centroid
    best_local = int(np.argmax(sims))
    best_doc_id = [d for d in doc_ids if d in idx][best_local]
    best_sim = float(sims[best_local])

    rec = (_doc_record_map() or {}).get(best_doc_id, {}) or {}
    return {
        "doc_id": best_doc_id,
        "openalex_id": rec.get("openalex_id"),
        "title": rec.get("title"),
        "doi": rec.get("doi"),
        "primary_topic": rec.get("primary_topic"),
        "primary_subfield": rec.get("primary_subfield"),
        "primary_field": rec.get("primary_field"),
        "similarity": round(best_sim, 4),
        "cluster_size": len(rows_idx),
    }


@lru_cache(maxsize=1)
def signal_counts() -> dict[str, int]:
    if not is_available():
        return {}
    with _conn() as c:
        rows = c.execute(
            "SELECT signal, COUNT(*) FROM global_topics GROUP BY signal"
        ).fetchall()
    return {s: int(n) for s, n in rows}


def top_topics(signal: str, limit: int = 10) -> list[TopicSummary]:
    """Top topics by total_docs for a given signal. Includes per-window history."""
    if not is_available():
        return []
    with _conn() as c:
        topic_rows = c.execute(
            """
            SELECT global_topic_id, top_words, signal, total_docs,
                   first_window, last_window, peak_count
            FROM global_topics
            WHERE signal = ?
            ORDER BY total_docs DESC
            LIMIT ?
            """,
            (signal, limit),
        ).fetchall()
        if not topic_rows:
            return []
        ids = tuple(r[0] for r in topic_rows)
        placeholders = ",".join("?" * len(ids))
        history_rows = c.execute(
            f"""
            SELECT global_topic_id, window_id, doc_count
            FROM topic_metrics_history
            WHERE global_topic_id IN ({placeholders})
            ORDER BY global_topic_id, window_id
            """,
            ids,
        ).fetchall()

    history_by_topic: dict[int, list[tuple[str, int]]] = {}
    for gid, wid, cnt in history_rows:
        history_by_topic.setdefault(gid, []).append((wid, int(cnt)))

    out: list[TopicSummary] = []
    for gid, words_json, sig, total, fw, lw, peak in topic_rows:
        try:
            words = json.loads(words_json)
        except (TypeError, json.JSONDecodeError):
            words = []
        out.append(
            TopicSummary(
                topic_id=int(gid),
                signal=sig,
                total_docs=int(total),
                peak_count=int(peak),
                first_window=fw,
                last_window=lw,
                words=words[:10],
                history=history_by_topic.get(gid, []),
                centroid=_centroid_paper(int(gid)),
            )
        )
    return out


def dashboard_payload(top_n_per_signal: int = 8) -> dict[str, Any]:
    """Everything the FE needs in one call."""
    if not is_available():
        return {
            "available": False,
            "reason": f"BERTrend DB not found at {_path()}",
        }
    wins = list_windows()
    sigs = signal_counts()
    emerging = top_topics("emerging", top_n_per_signal)
    strong = top_topics("strong", top_n_per_signal)

    def _t(t: TopicSummary) -> dict[str, Any]:
        return {
            "topic_id": t.topic_id,
            "signal": t.signal,
            "total_docs": t.total_docs,
            "peak_count": t.peak_count,
            "first_window": t.first_window,
            "last_window": t.last_window,
            "words": t.words,
            "history": [{"period": w, "count": c} for w, c in t.history],
            "centroid": t.centroid,
        }

    return {
        "available": True,
        "from_window": wins[0].window_id if wins else None,
        "to_window": wins[-1].window_id if wins else None,
        "windows": [
            {
                "period": w.window_id,
                "n_docs": w.n_docs,
                "n_topics": w.n_topics,
                "n_outliers": w.n_outliers,
            }
            for w in wins
        ],
        "signal_counts": sigs,
        "emerging": [_t(t) for t in emerging],
        "strong": [_t(t) for t in strong],
    }
