"""Stage 08 — Сводка результатов всех этапов в одну таблицу.

Собирает в одну CSV-табличку:
  - baselines (Stage 02)        - BM25, dense
  - retriever_compare (Stage 05) - dense / bm25 / hybrid_rrf [+ reranker]
  - qe_12combos (Stage 06)       - 12 комбинаций (method × aligner)
  - qe_extra_methods (Stage 07)  - ThinkQE × {none,CSQE,AQE} + GenCRF

Артефакты:
  qe_cache/results/all_results.csv          — все строки одной таблицей
  qe_cache/results/summary_by_dataset.csv   — pivot per-dataset (для отчёта)
  qe_cache/results/summary_mean.csv         — усреднения по датасетам
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _config import get_cache_dir
from _shared import read_state, setup_log_file


SOURCES = {
    "baseline":  "eval/baselines.csv",
    "qe_12":     "results/qe_12combos.csv",
    "qe_extra":  "results/qe_extra_methods.csv",
}


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as ex:
        print(f"  [warn] cannot read {path}: {ex}")
        return pd.DataFrame()


def _normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Приводит все источники к единой схеме: {source, group, dataset, reranker, метрики}."""
    if df.empty:
        return df
    df = df.copy()
    df["source"] = source

    # group — отображает «что мы сравниваем»
    if source == "baseline":
        df["group"] = df["retriever"]   # bm25, dense:{model}
    elif source == "retriever":
        df["group"] = df["retriever"]   # dense / bm25 / hybrid_rrf
    elif source in ("qe_12", "qe_extra"):
        df["group"] = df["combination"] # "Q2D + CSQE", "ThinkQE + AQE", "GenCRF (...)"
    else:
        df["group"] = "?"

    # reranker tag: "none" если колонки нет (старые baseline/retriever строки).
    if "reranker" not in df.columns:
        df["reranker"] = "none"
    df["reranker"] = df["reranker"].fillna("none")
    # retriever tag: для baseline источника берём из колонки retriever (там
    # уже "bm25" или "dense:..."), для qe_* — заполняем "unknown" если строка
    # legacy (до колонки retriever).
    if "retriever" not in df.columns:
        df["retriever"] = "unknown"
    df["retriever"] = df["retriever"].fillna("unknown")
    return df


def main(cache_dir: str | Path | None = None) -> dict:
    cache_dir = get_cache_dir(cache_dir)
    setup_log_file(cache_dir, "stage08_summary")

    print(f"[summary] cache_dir = {cache_dir.resolve()}")
    pieces: list[pd.DataFrame] = []
    for src_name, rel_path in SOURCES.items():
        p = cache_dir / rel_path
        df = _safe_read(p)
        if df.empty:
            print(f"  [{src_name}] {p} — empty/missing, skip")
            continue
        norm = _normalize(df, src_name)
        print(f"  [{src_name}] {p} — {len(norm)} rows")
        pieces.append(norm)

    if not pieces:
        print("Нет данных для сводки. Запусти Stages 02/05/06/07.")
        return {"all": None}

    all_df = pd.concat(pieces, ignore_index=True, sort=False)

    # Унифицируем имена метрик: оставляем только те, что есть в большинстве
    metric_cols = [c for c in all_df.columns
                   if c.startswith(("recall@", "map@", "ndcg@"))]
    time_cols = [c for c in (
        "corpus_encode_time_sec", "query_encode_time_sec",
        "retrieve_time_sec",
        "method_time_sec", "align_time_sec",
        "rerank_time_sec",
        "index_time_sec", "query_time_sec",
        "time_sec",
    ) if c in all_df.columns]
    cols_keep = (["source", "group", "retriever", "reranker", "dataset"]
                 + metric_cols + time_cols)
    if "rerank_top_n" in all_df.columns:
        cols_keep.append("rerank_top_n")
    if "n_eval" in all_df.columns:
        cols_keep.append("n_eval")
    all_df = all_df[cols_keep]

    out_dir = cache_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all_results.csv"
    all_df.to_csv(all_path, index=False)
    print(f"\n[saved] {all_path}  ({len(all_df)} rows)")

    # === Pivot per-dataset (метрики + время) ===
    pivot_metric = "ndcg@5" if "ndcg@5" in metric_cols else metric_cols[0]
    pivot_values = metric_cols + (["time_sec"] if "time_sec" in time_cols else [])
    pivot = (
        all_df.pivot_table(
            index=["source", "group", "retriever", "reranker"], columns="dataset",
            values=pivot_values, aggfunc="first",
        )
        .round(4)
    )
    pivot_path = out_dir / "summary_by_dataset.csv"
    pivot.to_csv(pivot_path)
    print(f"[saved] {pivot_path}")

    # === Усреднения по датасетам (метрики + средние времена) ===
    agg_cols = metric_cols + time_cols
    agg = (
        all_df.groupby(["source", "group", "retriever", "reranker"], sort=False)[agg_cols]
        .mean()
        .round(4)
        .sort_values(by=pivot_metric, ascending=False)
    )
    agg_path = out_dir / "summary_mean.csv"
    agg.to_csv(agg_path)
    print(f"[saved] {agg_path}")

    # Печатаем top-15 в компактном виде: метрики + total time
    print_cols = metric_cols + (["time_sec"] if "time_sec" in time_cols else [])
    print(f"\n=== Top-15 по mean {pivot_metric} ===")
    print(agg[print_cols].head(15))

    # === best from each stage ===
    print("\n=== Победители по этапам (mean " + pivot_metric + ") ===")
    for src_name in ["baseline", "qe_12", "qe_extra"]:
        sub = agg[agg.index.get_level_values("source") == src_name]
        if sub.empty:
            continue
        best_idx = sub[pivot_metric].idxmax()
        # best_idx = (source, group, retriever, reranker)
        label = f"{best_idx[1]} [retr={best_idx[2]}, rerank={best_idx[3]}]"
        print(f"  [{src_name:9}] {label:<70} -> "
              f"{pivot_metric}={sub.loc[best_idx, pivot_metric]:.4f}")

    # state-overview
    print("\n=== State (для информации) ===")
    st = read_state(cache_dir, "best_embedding")
    if st is not None:
        print(f"  best_embedding: {json.dumps(st, ensure_ascii=False)[:200]}")

    return {
        "all":     str(all_path),
        "pivot":   str(pivot_path),
        "mean":    str(agg_path),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()
    main(cache_dir=args.cache_dir)
