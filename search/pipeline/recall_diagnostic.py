"""Diagnostic — recall@K (default 100) для BM25 baseline и BM25 + LLM-expansion.

Использует закэшированные:
  - qe_cache/bm25_indexes/{ds}_v2stop.pkl   — BM25Okapi индекс корпуса
  - qe_cache/llm_outputs/{method}_{ds}.json  — LLM expansion'ы

Не нужен GPU, vLLM, encoder, reranker. Считает за минуты, можно гонять на ноуте.

Что показывает: «потолок» retrieval'а — recall@K, который reranker не сможет
превысить. Помогает понять, какой из QE-методов даёт reranker'у лучший pool.

Запуск:
  python pipeline/recall_diagnostic.py --datasets scifact --top-k 100
  python pipeline/recall_diagnostic.py --datasets scifact,nfcorpus --top-k 100,500

Выход:
  qe_cache/results/recall_diagnostic.csv  — все строки method x dataset x K
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import get_cache_dir, parse_datasets
from _shared import (build_or_load_bm25_index, evaluate_run,
                     get_corpus_texts_ids, get_query_texts_ids,
                     lemmatize_ru, load_full_dataset, tokenize_simple)


METHODS = [
    ("Query2doc",    5),   # Wang 2023: qx5 + d' для BM25
    ("PromptPRF",    1),
    ("PQR",          1),
    ("Word2Passage", 1),
]


def bm25_topk(bm25, queries_lemma: list[str], corpus_ids: list[str],
              top_k: int) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for qi, qtxt in enumerate(tqdm(queries_lemma, desc="bm25", leave=False)):
        scores = bm25.get_scores(tokenize_simple(qtxt))
        n_ret = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -n_ret)[-n_ret:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        out[qi] = [corpus_ids[int(j)] for j in top_idx]
    return out


def w2p_weighted_topk(bm25, weights_per_qid: dict[str, dict[str, float]],
                      qids: list[str], corpus_ids: list[str],
                      top_k: int) -> dict[int, list[str]]:
    """Phase 1.5b — paper-faithful Word2Passage BM25 weighted scoring."""
    n_docs = len(corpus_ids)
    out: dict[int, list[str]] = {}
    for qi, qid in enumerate(tqdm(qids, desc="w2p-weighted", leave=False)):
        weights = weights_per_qid.get(qid, {})
        if not weights:
            out[qi] = []
            continue
        total = np.zeros(n_docs, dtype=np.float32)
        for term, w in weights.items():
            term_lem = lemmatize_ru(term)
            if not term_lem:
                continue
            total += w * bm25.get_scores(tokenize_simple(term_lem))
        n_ret = min(top_k, n_docs)
        top_idx = np.argpartition(total, -n_ret)[-n_ret:]
        top_idx = top_idx[np.argsort(total[top_idx])[::-1]]
        out[qi] = [corpus_ids[int(j)] for j in top_idx]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="scifact")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--top-k", default="100",
                    help="Comma-separated K values (e.g. '10,100,500')")
    ap.add_argument("--qrels-split", default=None)
    ap.add_argument("--include-w2p-weighted", action="store_true",
                    help="Доп. строка W2P weighted BM25 (paper-faithful Choi 2025)")
    args = ap.parse_args()

    cache_dir = get_cache_dir(args.cache_dir)
    datasets = parse_datasets(args.datasets)
    ks = tuple(int(x) for x in args.top_k.split(","))
    max_k = max(ks)

    rows: list[dict] = []
    for ds_name in datasets:
        print(f"\n{'=' * 60}\n  DATASET: {ds_name}  (top-K = {ks})\n{'=' * 60}")
        d = load_full_dataset(ds_name, qrels_split=args.qrels_split)

        # BM25 индексирует processed_text — соответствует общему кэшу
        cids, ctexts = get_corpus_texts_ids(d, text_field="processed")
        qids, qtexts_raw = get_query_texts_ids(d, text_field="raw")

        bm25 = build_or_load_bm25_index(cache_dir, ds_name, ctexts)

        # === 1. Чистый BM25 baseline (только raw query, лемматизирован) ===
        print("\n  -- BM25 baseline (no QE)")
        baseline_q = [lemmatize_ru(q) for q in qtexts_raw]
        ranked = bm25_topk(bm25, baseline_q, cids, top_k=max_k)
        ret_per_qid = {qids[qi]: ranked.get(qi, []) for qi in range(len(qids))}
        metrics = evaluate_run(ret_per_qid, d["qrels"], ks=ks)
        rows.append({"method": "BM25", "aligner": "—", "dataset": ds_name, **metrics})
        for k in ks:
            print(f"     recall@{k} = {metrics.get(f'recall@{k}', float('nan')):.4f}")

        # === 2. Каждый LLM-метод x align_none ===
        for method, n_repeat in METHODS:
            exp_path = cache_dir / "llm_outputs" / f"{method}_{ds_name}.json"
            if not exp_path.exists():
                print(f"\n  [skip] {method}: {exp_path.name} не найден")
                continue
            try:
                expansions = json.loads(exp_path.read_text(encoding="utf-8"))
            except Exception as ex:
                print(f"\n  [skip] {method}: read error ({ex})")
                continue
            print(f"\n  -- {method} + none  (qx{n_repeat} repetition)")

            final_q: list[str] = []
            for qid, q_raw in zip(qids, qtexts_raw):
                exp = expansions.get(qid, "")
                combined = (q_raw + " ") * n_repeat + exp
                final_q.append(lemmatize_ru(combined))

            ranked = bm25_topk(bm25, final_q, cids, top_k=max_k)
            ret_per_qid = {qids[qi]: ranked.get(qi, []) for qi in range(len(qids))}
            metrics = evaluate_run(ret_per_qid, d["qrels"], ks=ks)
            rows.append({"method": method, "aligner": "none",
                         "dataset": ds_name, **metrics})
            for k in ks:
                print(f"     recall@{k} = {metrics.get(f'recall@{k}', float('nan')):.4f}")

        # === 3. Опционально: W2P weighted BM25 (paper-faithful) ===
        if args.include_w2p_weighted:
            wpath = cache_dir / "llm_outputs" / f"Word2Passage_weights_{ds_name}.json"
            if wpath.exists():
                try:
                    weights_per_qid = json.loads(wpath.read_text(encoding="utf-8"))
                except Exception as ex:
                    print(f"\n  [skip] W2P weighted: read error ({ex})")
                else:
                    print("\n  -- Word2Passage + weighted-BM25 (paper Choi 2025)")
                    ranked = w2p_weighted_topk(bm25, weights_per_qid, qids, cids,
                                                top_k=max_k)
                    ret_per_qid = {qids[qi]: ranked.get(qi, []) for qi in range(len(qids))}
                    metrics = evaluate_run(ret_per_qid, d["qrels"], ks=ks)
                    rows.append({"method": "Word2Passage", "aligner": "weighted-BM25",
                                 "dataset": ds_name, **metrics})
                    for k in ks:
                        print(f"     recall@{k} = {metrics.get(f'recall@{k}', float('nan')):.4f}")
            else:
                print(f"\n  [skip] W2P weighted: {wpath.name} не найден "
                      f"(запусти Stage 6 сначала)")

    # === Сводка ===
    df = pd.DataFrame(rows)
    if df.empty:
        print("\nНет строк для сводки.")
        return

    out_csv = cache_dir / "results" / "recall_diagnostic.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\n{'=' * 60}\n  SUMMARY  (saved -> {out_csv})\n{'=' * 60}")
    metric_cols = [c for c in df.columns
                   if c.startswith("recall@") or c.startswith("ndcg@")
                   or c.startswith("map@")]
    pivot = df.pivot_table(
        index=["method", "aligner"], columns="dataset",
        values=metric_cols, aggfunc="first",
    ).round(4)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
