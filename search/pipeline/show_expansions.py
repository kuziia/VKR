"""Дамп примеров расширений запросов для каждого метода (Q2D / PromptPRF / PQR /
Word2Passage / ThinkQE / GenCRF) на одном датасете в текстовый файл.

Использование:
    python pipeline/show_expansions.py --dataset scifact --n 5 --out qe_examples.txt

Читает кэш из qe_cache/llm_outputs/{Method}_{dataset}.json — то есть всё, что
уже сгенерировала LLM в Stages 06/07. После запуска даёт человеко-читаемый
текстовый отчёт.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _config import get_cache_dir
from _shared import load_full_dataset, read_state

METHODS = ["Query2doc", "PromptPRF", "PQR", "Word2Passage", "ThinkQE"]
PROMPTPRF_FEATURE_TYPES = ["keywords", "facts", "entities", "entities-cot"]


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        print(f"  [warn] failed to parse {p.name}: {ex}")
        return None


def _trunc(text: str, n: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + " […]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n", type=int, default=5,
                    help="сколько query-примеров вытащить")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="qe_examples.txt")
    ap.add_argument("--qrels-split", default=None)
    args = ap.parse_args()

    cache_dir = get_cache_dir(args.cache_dir)
    llm_dir = cache_dir / "llm_outputs"
    if not llm_dir.exists():
        sys.exit(f"Нет директории {llm_dir}; сначала запусти Stage 06/07.")

    # === Загружаем датасет (для оригинальных текстов запросов и qrels) ===
    print(f"Loading dataset {args.dataset} ...")
    d = load_full_dataset(args.dataset, qrels_split=args.qrels_split)
    qid_field = d["qid_field"]
    qid2text = {str(r[qid_field]): (r.get("processed_text") or r.get("text") or "").strip()
                for r in d["queries"]}

    # qrels: qid → [doc_id]
    qid2relevant: dict[str, list[str]] = {}
    for r in d["qrels"]:
        if int(r["score"]) > 0:
            qid2relevant.setdefault(r["query-id"], []).append(r["corpus-id"])

    # тексты документов (для показа ground-truth doc'ов)
    manifest = read_state(cache_dir, "best_embedding_manifest")
    cid2text: dict[str, str] = {}
    if manifest and args.dataset in manifest.get("datasets", {}):
        meta_path = manifest["datasets"][args.dataset].get("meta")
        if meta_path and Path(meta_path).exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            cid2text = {cid: txt for cid, txt in zip(meta["ids"], meta["texts"])}

    # === Загружаем все JSON-кэши методов ===
    expansions: dict[str, dict[str, str]] = {}
    for m in METHODS:
        data = _load_json(llm_dir / f"{m}_{args.dataset}.json")
        if data:
            expansions[m] = data

    # GenCRF — list[str] на qid (несколько переформулировок)
    gencrf_path = llm_dir / f"GenCRF_reformulations_{args.dataset}.json"
    gencrf_refs = _load_json(gencrf_path)

    # PromptPRF features (per doc, не per query) — покажем top-3 для каждого типа
    promptprf_features: dict[str, dict[str, str]] = {}
    for ft in PROMPTPRF_FEATURE_TYPES:
        data = _load_json(llm_dir / f"PromptPRF_features_{args.dataset}_{ft}.json")
        if data:
            promptprf_features[ft] = data

    if not expansions and not gencrf_refs:
        sys.exit("В кэше нет ни одного метода. Stage 06/07 ещё не отработали.")

    print(f"  expansions found:  {list(expansions.keys())}")
    if gencrf_refs:
        print(f"  GenCRF reformulations: yes ({len(gencrf_refs)} qids)")
    if promptprf_features:
        print(f"  PromptPRF feature types: {list(promptprf_features)}")

    # === Выбираем N запросов, у которых есть И оригинал И хотя бы один метод ===
    candidate_qids = [qid for qid in qid2text
                      if qid in qid2relevant
                      and any(qid in e for e in expansions.values())]
    random.seed(args.seed)
    sample_qids = random.sample(candidate_qids,
                                min(args.n, len(candidate_qids)))

    # === Пишем отчёт ===
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Примеры расширений запросов: dataset = {args.dataset}\n")
        f.write(f"# Источник: qe_cache/llm_outputs/  (выходы LLM из Stages 06/07)\n")
        f.write(f"# Sample size: {len(sample_qids)} queries (random seed = {args.seed})\n")
        f.write(f"# LLM-модель — та, под которой запускался последний прогон\n")
        f.write("=" * 78 + "\n\n")

        for i, qid in enumerate(sample_qids, 1):
            qtext = qid2text.get(qid, "")
            f.write(f"\n{'#' * 78}\n")
            f.write(f"# QUERY {i}/{len(sample_qids)}  — qid={qid}\n")
            f.write(f"{'#' * 78}\n\n")
            f.write(f"ORIGINAL QUERY:\n  {qtext}\n\n")

            # Ground truth — первые 1-2 релевантных документа
            relevant = qid2relevant.get(qid, [])
            if relevant and cid2text:
                f.write(f"GROUND-TRUTH DOCS ({len(relevant)} relevant; showing 1-2):\n")
                for did in relevant[:2]:
                    f.write(f"  [{did}] {_trunc(cid2text.get(did, ''), 400)}\n")
                f.write("\n")

            # Каждый метод
            for m in METHODS:
                exp = expansions.get(m, {}).get(qid)
                if exp is None:
                    continue
                f.write(f"--- {m} ---\n")
                f.write(f"{_trunc(exp, 1500)}\n\n")

            # GenCRF (несколько переформулировок)
            if gencrf_refs and qid in gencrf_refs:
                refs = gencrf_refs[qid]
                f.write(f"--- GenCRF (reformulations: {len(refs)}) ---\n")
                for j, r in enumerate(refs, 1):
                    f.write(f"  {j}. {_trunc(r, 200)}\n")
                f.write("\n")

            # PromptPRF features — для top-3 релевантных документов этого запроса
            if promptprf_features and relevant:
                f.write(f"--- PromptPRF features (для GT-релевантных документов) ---\n")
                for ft, feats in promptprf_features.items():
                    feats_for_relevant = [
                        (did, feats[did]) for did in relevant[:3] if did in feats
                    ]
                    if not feats_for_relevant:
                        continue
                    f.write(f"  [{ft}]\n")
                    for did, feat in feats_for_relevant:
                        f.write(f"    {did}: {_trunc(feat, 300)}\n")
                f.write("\n")

        f.write("\n" + "=" * 78 + "\n")
        f.write("# END\n")

    size = out_path.stat().st_size
    print(f"\n[saved] {out_path.resolve()}  ({size / 1024:.1f} KB, "
          f"{len(sample_qids)} queries)")


if __name__ == "__main__":
    main()
