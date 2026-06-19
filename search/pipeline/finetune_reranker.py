"""Phase 3 — Fine-tune cross-encoder reranker на scifact-train (или другом датасете).

Берёт qrels-train, формирует (query, positive_doc, label=1) пары + BM25 hard
negatives (label=0), обучает CrossEncoder, сохраняет в указанный путь.

Использование:
  python pipeline/finetune_reranker.py \\
      --dataset scifact \\
      --base-model BAAI/bge-reranker-v2-m3 \\
      --output-dir ./reranker_scifact_finetuned \\
      --epochs 3 --batch-size 16 --neg-per-pos 3

После — Stage 6 берёт fine-tuned путь:
  python pipeline/00_build.py --stages 6 --datasets scifact \\
      --rerank --reranker-model ./reranker_scifact_finetuned
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import get_cache_dir
from _shared import (get_corpus_texts_ids, get_query_texts_ids,
                     load_full_dataset, setup_log_file, tokenize_simple)


def _build_bm25(corpus_texts: list[str]):
    from rank_bm25 import BM25Okapi
    print(f"  [BM25] tokenize {len(corpus_texts):,} docs ...")
    tok = [tokenize_simple(t) for t in tqdm(corpus_texts, desc="tok")]
    return BM25Okapi(tok)


def _hard_negatives(bm25, qtext: str, exclude: set[str], corpus_ids: list[str],
                    n_neg: int, top_k: int = 50) -> list[str]:
    scores = bm25.get_scores(tokenize_simple(qtext))
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    out = []
    for j in top_idx:
        cid = corpus_ids[int(j)]
        if cid in exclude:
            continue
        out.append(cid)
        if len(out) >= n_neg:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="scifact",
                    help="alias из DATASETS (_config.py)")
    ap.add_argument("--qrels-split", default="train")
    ap.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--bm25-top-k", type=int, default=50)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache_dir = get_cache_dir(args.cache_dir)
    setup_log_file(cache_dir, "finetune_reranker")
    random.seed(args.seed); np.random.seed(args.seed)

    t_total_start = time.time()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[ft] dataset={args.dataset} split={args.qrels_split} base={args.base_model}")

    d = load_full_dataset(args.dataset, qrels_split=args.qrels_split)
    qids_raw, qtexts_raw = get_query_texts_ids(d, text_field="raw")
    cids_raw, ctexts_raw = get_corpus_texts_ids(d, text_field="raw")
    qid2text = dict(zip(qids_raw, qtexts_raw))
    id2text = dict(zip(cids_raw, ctexts_raw))

    qid2pos: dict[str, list[str]] = {}
    for r in d["qrels"]:
        if r["score"] > 0:
            qid2pos.setdefault(r["query-id"], []).append(r["corpus-id"])
    qid2pos = {q: ps for q, ps in qid2pos.items() if q in qid2text and ps}
    print(f"[ft] queries with positives: {len(qid2pos):,}")
    if not qid2pos:
        raise SystemExit("Нет positive qrels — нечего обучать.")

    # BM25 для hard-neg mining (на processed_text — лучшая нормализация форм)
    cids_proc, ctexts_proc = get_corpus_texts_ids(d, text_field="processed")
    assert cids_proc == cids_raw, "corpus order mismatch raw vs processed"
    bm25 = _build_bm25(ctexts_proc)

    # Train/dev split — на уровне qid'ов (чтобы dev-queries не утекали в train)
    all_qids = list(qid2pos.keys())
    random.shuffle(all_qids)
    n_dev = max(1, int(len(all_qids) * args.dev_frac))
    dev_qids = set(all_qids[:n_dev])
    train_qids = [q for q in all_qids if q not in dev_qids]
    print(f"[ft] train={len(train_qids):,} dev={len(dev_qids):,}")

    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import \
        CERerankingEvaluator
    from torch.utils.data import DataLoader

    print("[ft] mining hard negatives ...")
    t_mine_start = time.time()
    train_examples: list[InputExample] = []
    for qid in tqdm(train_qids, desc="train-mine"):
        qtxt = qid2text[qid]
        pos_ids = qid2pos[qid]
        pos_set = set(pos_ids)
        for pid in pos_ids:
            train_examples.append(InputExample(
                texts=[qtxt, id2text.get(pid, "")[:1500]], label=1.0))
        negs = _hard_negatives(bm25, qtxt, pos_set, cids_raw,
                               n_neg=args.neg_per_pos, top_k=args.bm25_top_k)
        for nid in negs:
            train_examples.append(InputExample(
                texts=[qtxt, id2text.get(nid, "")[:1500]], label=0.0))
    print(f"[ft] train pairs: {len(train_examples):,}")

    mine_train_seconds = time.time() - t_mine_start

    t_dev_start = time.time()
    dev_samples: list[dict] = []
    for qid in tqdm(sorted(dev_qids), desc="dev-mine"):
        qtxt = qid2text[qid]
        pos_ids = qid2pos[qid]
        pos_set = set(pos_ids)
        negs = _hard_negatives(bm25, qtxt, pos_set, cids_raw,
                               n_neg=max(args.neg_per_pos * 3, 9),
                               top_k=args.bm25_top_k)
        dev_samples.append({
            "query": qtxt,
            "positive": [id2text.get(p, "")[:1500] for p in pos_ids],
            "negative": [id2text.get(n, "")[:1500] for n in negs],
        })

    mine_dev_seconds = time.time() - t_dev_start

    print(f"[ft] loading {args.base_model} ...")
    t_load_start = time.time()
    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)
    load_seconds = time.time() - t_load_start

    loader = DataLoader(train_examples, shuffle=True,
                        batch_size=args.batch_size)
    evaluator = CERerankingEvaluator(dev_samples, name="dev")
    warmup = int(len(loader) * args.epochs * 0.1)
    print(f"[ft] epochs={args.epochs} steps/epoch={len(loader)} warmup={warmup} lr={args.lr}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_fit_start = time.time()
    model.fit(
        train_dataloader=loader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=str(out),
        save_best_model=True,
        evaluation_steps=max(50, len(loader) // 4),
        show_progress_bar=True,
    )
    fit_seconds = time.time() - t_fit_start
    total_seconds = time.time() - t_total_start

    meta = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "qrels_split": args.qrels_split,
        "base_model": args.base_model,
        "output_dir": str(out),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "neg_per_pos": args.neg_per_pos,
        "bm25_top_k": args.bm25_top_k,
        "dev_frac": args.dev_frac,
        "seed": args.seed,
        "n_train_queries": len(train_qids),
        "n_dev_queries": len(dev_qids),
        "n_train_pairs": len(train_examples),
        "steps_per_epoch": len(loader),
        "warmup_steps": warmup,
        "mine_train_seconds": round(mine_train_seconds, 2),
        "mine_dev_seconds": round(mine_dev_seconds, 2),
        "load_model_seconds": round(load_seconds, 2),
        "fit_seconds": round(fit_seconds, 2),
        "total_seconds": round(total_seconds, 2),
    }
    meta_path = out / "training_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[ft] saved to {out}")
    print(f"[ft] training_meta.json -> mine={mine_train_seconds:.1f}s "
          f"fit={fit_seconds:.1f}s total={total_seconds:.1f}s")


if __name__ == "__main__":
    main()
