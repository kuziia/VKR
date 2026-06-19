"""Stage 01 — Сравнение embedding-моделей.

Для каждой модели из MODELS × каждого выбранного датасета:
  1. Кодирует corpus (один раз, кэширует в qe_cache/embeddings_compare/{model}/{ds}_emb.npy).
  2. Кодирует queries.
  3. Делает retrieval top-k через chunked brute-force.
  4. Считает recall@5/10, map@5/10, ndcg@5/10.

Артефакты:
  qe_cache/eval/embeddings_compare.csv  — все строки (model × dataset)
  qe_cache/state/best_embedding.json    — лучшая по avg ndcg@10

Запуск:
  python pipeline/01_compare_embeddings.py --datasets nfcorpus,scifact
  python pipeline/01_compare_embeddings.py --datasets all --top-k 10
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _config import (DATASETS, MODELS, TOP_K, get_cache_dir, parse_datasets,
                     model_slug)
from _shared import (STEncoder, build_or_load_corpus_index,
                     build_or_load_query_emb, chunked_retrieve,
                     emb_cache_paths, evaluate_run, load_full_dataset,
                     query_emb_cache_path, setup_log_file, write_state)


def run_compare(datasets: list[str], cache_dir: Path,
                models: list[dict] | None = None,
                top_k: int = TOP_K, qrels_split: str | None = None,
                text_field: str = "raw") -> pd.DataFrame:
    """Сравнение embedding-моделей.

    text_field: какие текстовые поля использовать для encoding'а
        "raw"       — natural text (рекомендуется для современных dense-моделей).
                      Embedding-модели (e5/bge/distiluse) обучались на natural
                      language; лемматизация ломает им subword-токенизацию.
        "processed" — лемматизированный (legacy-совместимость).
    """
    models = models or MODELS
    eval_dir = cache_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "embeddings_compare.csv"

    # resume: загружаем уже посчитанное
    rows: list[dict] = []
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        print(f"[resume] loaded {len(rows)} cached rows from {csv_path}")
    done = {(r["model"], r["dataset"]) for r in rows}

    # Загружаем датасеты один раз
    DATA = {n: load_full_dataset(n, qrels_split=qrels_split) for n in datasets}

    for m_info in models:
        m_name = m_info["name"]
        # пропускаем, если все датасеты уже посчитаны
        if all((m_name, ds) in done for ds in datasets):
            print(f"\n[skip] {m_name}: all datasets cached")
            continue
        print(f"\n{'=' * 60}\n  MODEL: {m_name}\n{'=' * 60}")

        encoder = STEncoder(m_info, device="cuda:0", fp16=True,
                            text_field=text_field)
        try:
            for ds_name in datasets:
                if (m_name, ds_name) in done:
                    print(f"  [skip] {ds_name}")
                    continue
                d = DATA[ds_name]
                print(f"\n  -- {ds_name}  (text_field={text_field})")

                # Заранее проверяем, есть ли уже эмбеддинги на диске, чтобы
                # отличить "fresh encoding time" от "load time".
                emb_path = emb_cache_paths(cache_dir, m_name, ds_name,
                                           text_field=text_field)[0]
                qemb_path = query_emb_cache_path(cache_dir, m_name, ds_name,
                                                 text_field=text_field)
                corpus_was_fresh = not emb_path.exists()
                queries_were_fresh = not qemb_path.exists()

                t_corpus = time.time()
                ci = build_or_load_corpus_index(cache_dir, m_info, d, encoder=encoder,
                                                text_field=text_field)
                corpus_dt = time.time() - t_corpus

                t_q = time.time()
                qids, _, qemb = build_or_load_query_emb(cache_dir, m_info, d, encoder=encoder,
                                                        text_field=text_field)
                qenc_dt = time.time() - t_q

                t_r = time.time()
                idx, _ = chunked_retrieve(qemb, ci["emb"], top_k)
                retrieve_dt = time.time() - t_r

                ret_per_q = {
                    qids[i]: [ci["ids"][int(j)] for j in idx[i]]
                    for i in range(len(qids))
                }
                metrics = evaluate_run(ret_per_q, d["qrels"], ks=(5, 10))
                row = {
                    "model": m_name, "dataset": ds_name, **metrics,
                    "dim": int(encoder.dim),
                    "text_field": text_field,
                    # NaN, если корпус был уже закэширован (fresh encoding не запускался)
                    "corpus_encode_time_sec": (round(corpus_dt, 2)
                                               if corpus_was_fresh else None),
                    "query_encode_time_sec":  (round(qenc_dt, 2)
                                               if queries_were_fresh else None),
                    "retrieve_time_sec":      round(retrieve_dt, 3),
                    # total — это сколько РЕАЛЬНО потратилось в этом прогоне
                    # (включая time на load из кэша, если был cache hit)
                    "time_sec": round(corpus_dt + qenc_dt + retrieve_dt, 2),
                    "corpus_from_cache":  not corpus_was_fresh,
                    "queries_from_cache": not queries_were_fresh,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                tag_c = f"encode={corpus_dt:.1f}s" if corpus_was_fresh else f"load={corpus_dt:.1f}s"
                tag_q = f"encode={qenc_dt:.2f}s"   if queries_were_fresh else f"load={qenc_dt:.2f}s"
                print(f"     -> {metrics}  | corpus {tag_c}  queries {tag_q}  retrieve={retrieve_dt:.2f}s")
        finally:
            encoder.free()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    return df


def pick_best(df: pd.DataFrame, metric: str = "ndcg@10") -> str:
    agg = df.groupby("model")[metric].mean().sort_values(ascending=False)
    print("\n=== mean " + metric + " по моделям ===")
    print(agg.round(4))
    return agg.index[0]


def main(datasets: list[str] | None = None, cache_dir: str | Path | None = None,
         top_k: int = TOP_K, qrels_split: str | None = None,
         metric_for_best: str = "ndcg@10",
         text_field: str = "raw") -> dict:
    cache_dir = get_cache_dir(cache_dir)
    setup_log_file(cache_dir, "stage01_compare_embeddings")
    if isinstance(datasets, str):
        datasets = parse_datasets(datasets)
    datasets = datasets or list(DATASETS.keys())

    df = run_compare(datasets, cache_dir, top_k=top_k, qrels_split=qrels_split,
                     text_field=text_field)
    if df.empty:
        raise SystemExit("No rows produced; check inputs.")

    best = pick_best(df, metric=metric_for_best)
    print(f"\nBest embedding (by {metric_for_best}): {best}")
    write_state(cache_dir, "best_embedding", {
        "model": best,
        "metric": metric_for_best,
        "score": float(df.groupby("model")[metric_for_best].mean()[best]),
        "datasets_evaluated": datasets,
        "text_field": text_field,
    })
    return {"best_embedding": best, "csv": str(cache_dir / "eval" / "embeddings_compare.csv")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="all",
                   help="comma-separated aliases (nfcorpus,scifact,...) or 'all'")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--qrels-split", default=None,
                   help="override qrels split (e.g. 'dev' for miracl)")
    p.add_argument("--metric", default="ndcg@10",
                   help="metric for picking the best model")
    p.add_argument("--text-field", default="raw", choices=["raw", "processed"],
                   help="raw=natural text (для dense; default), processed=lemmatized")
    args = p.parse_args()
    main(
        datasets=parse_datasets(args.datasets),
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        qrels_split=args.qrels_split,
        metric_for_best=args.metric,
        text_field=args.text_field,
    )
