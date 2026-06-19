"""Evaluate BM25 baseline (no QE) with multiple rerankers.

Использует закэшированный BM25-индекс + RerankerWrapper из Stage 06.
НЕ нужны: vLLM (LLM не используется), encoder, FAISS, методы QE.

Запуск:
  python pipeline/bm25_rerank_eval.py \\
      --datasets scifact,nfcorpus \\
      --rerank-models "none,BAAI/bge-reranker-v2-m3,./reranker_scifact_finetuned" \\
      --rerank-top-n 100 --top-k 10

Артефакт: qe_cache/results/bm25_rerank_eval.csv с колонками
  dataset, reranker, recall@5, ndcg@5, map@5, recall@10, ndcg@10, map@10,
  bm25_time_sec, rerank_time_sec, time_sec, n_eval
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import get_cache_dir, parse_datasets
from _shared import (build_or_load_bm25_index, evaluate_run,
                     get_corpus_texts_ids, get_query_texts_ids,
                     lemmatize_ru, load_full_dataset, setup_log_file,
                     tokenize_simple)


def _load_stage06():
    """Stage 06 имеет имя '06_qe_12combos.py' — нельзя import_module напрямую."""
    path = Path(__file__).parent / "06_qe_12combos.py"
    spec = importlib.util.spec_from_file_location("stage06", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="scifact",
                    help="comma-separated alias или 'all'")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--rerank-models", default="none",
                    help="Comma-separated rerank passes: "
                         "'none,BAAI/bge-reranker-v2-m3,./reranker_scifact_finetuned'")
    ap.add_argument("--rerank-top-n", type=int, default=100,
                    help="Сколько кандидатов из BM25 подавать на rerank")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Финальный top-K для метрик")
    ap.add_argument("--qrels-split", default=None)
    args = ap.parse_args()

    cache_dir = get_cache_dir(args.cache_dir)
    setup_log_file(cache_dir, "bm25_rerank_eval")
    datasets = parse_datasets(args.datasets)

    stage06 = _load_stage06()
    RerankerWrapper = stage06.RerankerWrapper
    parse_rerank_passes = stage06.parse_rerank_passes
    _rerank_tag = stage06._rerank_tag

    rerank_models = parse_rerank_passes(args.rerank_models)
    print(f"[rerank passes] {[_rerank_tag(rm) for rm in rerank_models]}")
    any_rerank = any(rm is not None for rm in rerank_models)

    # Lazy reranker cache
    rerankers: dict[str, "RerankerWrapper"] = {}
    def _get_reranker(name: str):
        if name not in rerankers:
            rerankers[name] = RerankerWrapper(name)
        return rerankers[name]

    # Resume: подгружаем уже посчитанные строки
    res_dir = cache_dir / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    res_csv = res_dir / "bm25_rerank_eval.csv"
    rows: list[dict] = (pd.read_csv(res_csv).to_dict("records")
                        if res_csv.exists() else [])
    for r in rows:
        r.setdefault("reranker", "none")
        r.setdefault("rerank_top_n", None)
    done = {(r["dataset"], r.get("reranker", "none")) for r in rows}

    eval_top_k = args.top_k
    wide_n = max(args.rerank_top_n, eval_top_k) if any_rerank else eval_top_k

    for ds_name in datasets:
        print(f"\n{'=' * 60}\n  DATASET: {ds_name}\n{'=' * 60}")
        # Что осталось посчитать?
        pending = [rm for rm in rerank_models
                   if (ds_name, _rerank_tag(rm)) not in done]
        if not pending:
            print("  [skip] все rerank-проходы уже в CSV")
            continue

        d = load_full_dataset(ds_name, qrels_split=args.qrels_split)
        cids, ctexts = get_corpus_texts_ids(d, text_field="processed")
        qids, qtexts_raw = get_query_texts_ids(d, text_field="raw")

        # RAW text для reranker'а — независимо от того, что у BM25
        id2text_raw: dict[str, str] = {}
        if any(rm is not None for rm in pending):
            cids_r, ctexts_r = get_corpus_texts_ids(d, text_field="raw")
            id2text_raw = {cid: ct for cid, ct in zip(cids_r, ctexts_r)}

        bm25 = build_or_load_bm25_index(cache_dir, ds_name, ctexts)

        # === BM25 retrieval one-time per dataset ===
        print(f"\n  BM25 retrieve top-{wide_n} ...")
        t_bm = time.time()
        queries_lem = [lemmatize_ru(q) for q in qtexts_raw]
        ranked_wide = bm25_topk(bm25, queries_lem, cids, top_k=wide_n)
        bm25_time = time.time() - t_bm
        print(f"  BM25 done in {bm25_time:.1f}s")

        # === Rerank passes ===
        for rm in pending:
            tag = _rerank_tag(rm)
            print(f"\n  -- rerank: {tag}")
            t_rr = time.time()
            try:
                if rm is None:
                    ranked = {qi: ranked_wide.get(qi, [])[:eval_top_k]
                              for qi in range(len(qids))}
                else:
                    rr = _get_reranker(rm)
                    ranked = rr.rerank(qtexts_raw, ranked_wide, id2text_raw,
                                       eval_top_k)
                ret_per_q = {qids[qi]: ranked.get(qi, [])
                             for qi in range(len(qids))}
                metrics = evaluate_run(ret_per_q, d["qrels"], ks=(5, 10))
                rerank_time = time.time() - t_rr
                print(f"     -> {metrics}")
                print(f"     bm25={bm25_time:.1f}s rerank={rerank_time:.1f}s")
            except Exception as ex:
                print(f"     [ERR] rerank={tag}: {ex}")
                import traceback; traceback.print_exc()
                metrics = {"recall@5": None, "map@5": None, "ndcg@5": None,
                           "recall@10": None, "map@10": None, "ndcg@10": None,
                           "n_eval": 0}
                rerank_time = time.time() - t_rr

            rows.append({
                "dataset": ds_name,
                "reranker": tag,
                "rerank_top_n": args.rerank_top_n if rm is not None else None,
                **metrics,
                "bm25_time_sec": round(bm25_time, 2),
                "rerank_time_sec": round(rerank_time, 2),
                "time_sec": round(bm25_time + rerank_time, 2),
            })
            done.add((ds_name, tag))
            pd.DataFrame(rows).to_csv(res_csv, index=False)

    for rr in rerankers.values():
        rr.free()

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nНет строк для сводки.")
        return

    print(f"\n{'=' * 60}\n  SUMMARY  (saved -> {res_csv})\n{'=' * 60}")
    metric_cols = [c for c in df.columns if c.startswith(("recall@", "map@", "ndcg@"))]
    pivot = (df.pivot_table(index=["reranker"], columns="dataset",
                            values=metric_cols, aggfunc="first")
               .round(4))
    print(pivot.to_string())


if __name__ == "__main__":
    main()
