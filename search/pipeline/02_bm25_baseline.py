"""Stage 02 — BM25 + dense baseline (без QE).

Считает baseline-метрики:
  - BM25 (через rank_bm25.BM25Okapi или bm25s, если установлен — он быстрее)
  - Dense cosine (через лучшую embedding-модель Stage 01;
                  если best ещё не выбран, используем e5-base из MODELS)

Артефакт: qe_cache/eval/baselines.csv
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import (DATASETS, MODELS_BY_NAME, TOP_K, get_cache_dir,
                     parse_datasets)
from _shared import (STEncoder, build_or_load_bm25_rankings,
                     build_or_load_corpus_index,
                     build_or_load_query_emb, chunked_retrieve,
                     emb_cache_paths, evaluate_run, get_corpus_texts_ids,
                     get_query_texts_ids, load_full_dataset,
                     query_emb_cache_path, read_state, setup_log_file,
                     tokenize_simple)


def _bm25_index_and_search(corpus_texts: list[str], queries: list[str],
                           corpus_ids: list[str], top_k: int) -> dict[int, list[str]]:
    """BM25 retrieval. Использует bm25s если установлен (5-50× быстрее), иначе rank_bm25."""
    try:
        import bm25s
        return _bm25s_search(corpus_texts, queries, corpus_ids, top_k)
    except ImportError:
        pass

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise SystemExit(
            "Neither bm25s nor rank_bm25 installed. Run:\n"
            "  pip install bm25s     # быстрее\n"
            "  pip install rank-bm25 # fallback"
        ) from e

    print("  [BM25] tokenize corpus ...")
    tok_corpus = [tokenize_simple(t) for t in tqdm(corpus_texts, desc="tok")]
    print("  [BM25] build BM25Okapi index ...")
    bm25 = BM25Okapi(tok_corpus)
    del tok_corpus; gc.collect()
    print("  [BM25] querying ...")
    out: dict[int, list[str]] = {}
    for qi, qtxt in enumerate(tqdm(queries, desc="bm25-query")):
        scores = bm25.get_scores(tokenize_simple(qtxt))
        n_ret = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -n_ret)[-n_ret:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        out[qi] = [corpus_ids[j] for j in top_idx]
    del bm25; gc.collect()
    return out


def _bm25s_search(corpus_texts, queries, corpus_ids, top_k):
    import bm25s
    print("  [BM25s] tokenize + index ...")
    corpus_tok = bm25s.tokenize(corpus_texts, stopwords=None)
    retriever = bm25s.BM25()
    retriever.index(corpus_tok)
    print("  [BM25s] querying ...")
    q_tok = bm25s.tokenize(queries, stopwords=None)
    docs, scores = retriever.retrieve(q_tok, k=top_k)
    out = {}
    for qi in range(len(queries)):
        out[qi] = [corpus_ids[int(j)] for j in docs[qi]]
    return out


def run_bm25(datasets: list[str], cache_dir: Path, top_k: int = TOP_K,
             qrels_split: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for ds_name in datasets:
        print(f"\n--- BM25: {ds_name} ---")
        d = load_full_dataset(ds_name, qrels_split=qrels_split)
        cids, ctexts = get_corpus_texts_ids(d)
        qids, qtexts = get_query_texts_ids(d)
        t_total = time.time()
        ranked = build_or_load_bm25_rankings(
            cache_dir, ds_name, ctexts, qtexts, cids, top_k,
            cache_top_k=max(top_k, 100),
        )
        total_time = time.time() - t_total
        ret_per_q = {qids[qi]: ranked.get(qi, []) for qi in range(len(qids))}
        metrics = evaluate_run(ret_per_q, d["qrels"], ks=(5, 10))
        rows.append({
            "retriever": "bm25", "dataset": ds_name, **metrics,
            "index_time_sec": None,   # build_or_load_bm25_rankings даёт совокупное время
            "query_time_sec": None,
            "time_sec": round(total_time, 2),
        })
        print(f"   -> {metrics}  | total={total_time:.1f}s")
    return rows


def run_dense(datasets: list[str], cache_dir: Path, model_info: dict,
              top_k: int = TOP_K, qrels_split: str | None = None,
              text_field: str = "raw") -> list[dict]:
    rows: list[dict] = []
    encoder = STEncoder(model_info, device="cuda:0", fp16=True,
                        text_field=text_field)
    m_name = model_info["name"]
    try:
        for ds_name in datasets:
            print(f"\n--- Dense [{m_name}, text_field={text_field}]: {ds_name} ---")
            d = load_full_dataset(ds_name, qrels_split=qrels_split)

            # Различаем fresh-encoding и load-from-cache
            corpus_was_fresh  = not emb_cache_paths(cache_dir, m_name, ds_name,
                                                     text_field=text_field)[0].exists()
            queries_was_fresh = not query_emb_cache_path(cache_dir, m_name, ds_name,
                                                          text_field=text_field).exists()

            t_corpus = time.time()
            ci = build_or_load_corpus_index(cache_dir, model_info, d, encoder=encoder,
                                             text_field=text_field)
            corpus_dt = time.time() - t_corpus

            t_q = time.time()
            qids, _, qemb = build_or_load_query_emb(cache_dir, model_info, d, encoder=encoder,
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
            rows.append({
                "retriever": f"dense:{m_name}",
                "dataset": ds_name, **metrics,
                "corpus_encode_time_sec": (round(corpus_dt, 2)
                                            if corpus_was_fresh else None),
                "query_encode_time_sec":  (round(qenc_dt, 2)
                                            if queries_was_fresh else None),
                "retrieve_time_sec":       round(retrieve_dt, 3),
                "time_sec": round(corpus_dt + qenc_dt + retrieve_dt, 2),
                "corpus_from_cache":  not corpus_was_fresh,
                "queries_from_cache": not queries_was_fresh,
            })
            tag_c = f"encode={corpus_dt:.1f}s" if corpus_was_fresh else f"load={corpus_dt:.1f}s"
            tag_q = f"encode={qenc_dt:.2f}s"   if queries_was_fresh else f"load={qenc_dt:.2f}s"
            print(f"   -> {metrics}  | corpus {tag_c}  queries {tag_q}  retrieve={retrieve_dt:.2f}s")
    finally:
        encoder.free()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main(datasets: list[str] | None = None, cache_dir: str | Path | None = None,
         dense_model: str | None = None, top_k: int = TOP_K,
         qrels_split: str | None = None, skip_bm25: bool = False,
         skip_dense: bool = False, text_field: str = "raw") -> dict:
    cache_dir = get_cache_dir(cache_dir)
    setup_log_file(cache_dir, "stage02_bm25_baseline")
    datasets = datasets or list(DATASETS.keys())

    eval_dir = cache_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "baselines.csv"

    rows: list[dict] = []
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")

    # Какую dense-модель брать?
    if dense_model is None:
        st = read_state(cache_dir, "best_embedding")
        if st is not None:
            dense_model = st["model"]
            print(f"[dense] using best embedding from state: {dense_model}")
        else:
            dense_model = "intfloat/multilingual-e5-base"
            print(f"[dense] best_embedding not in state; falling back to {dense_model}")

    if dense_model not in MODELS_BY_NAME:
        raise SystemExit(f"Model {dense_model} not in MODELS list (_config.py)")
    model_info = MODELS_BY_NAME[dense_model]

    if not skip_bm25:
        # BM25 ВСЕГДА на processed_text (лемматизация помогает sparse retrieval)
        new = run_bm25(datasets, cache_dir, top_k=top_k, qrels_split=qrels_split)
        rows = [r for r in rows if r.get("retriever") != "bm25" or r["dataset"] not in datasets]
        rows.extend(new)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    if not skip_dense:
        new = run_dense(datasets, cache_dir, model_info,
                        top_k=top_k, qrels_split=qrels_split,
                        text_field=text_field)
        tag = f"dense:{dense_model}"
        rows = [r for r in rows if r.get("retriever") != tag or r["dataset"] not in datasets]
        rows.extend(new)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nBaselines saved to: {csv_path}")
    return {"csv": str(csv_path)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="all")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dense-model", default=None,
                   help="HuggingFace model name; default — best from Stage 01 state, иначе e5-base")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--qrels-split", default=None)
    p.add_argument("--skip-bm25", action="store_true")
    p.add_argument("--skip-dense", action="store_true")
    p.add_argument("--text-field", default="raw", choices=["raw", "processed"])
    args = p.parse_args()
    main(
        datasets=parse_datasets(args.datasets),
        cache_dir=args.cache_dir,
        dense_model=args.dense_model,
        top_k=args.top_k,
        qrels_split=args.qrels_split,
        skip_bm25=args.skip_bm25,
        skip_dense=args.skip_dense,
        text_field=args.text_field,
    )
