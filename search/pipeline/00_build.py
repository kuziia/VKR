"""Pipeline orchestrator — последовательно прогоняет указанные этапы.

Использование:
  # все этапы на всех датасетах
  python search/pipeline/00_build.py

  # этапы 1-4 только на nfcorpus + scifact
  python search/pipeline/00_build.py --datasets nfcorpus,scifact --stages 1-4

  # только Stage 6 (предполагает, что 1-5 уже прогонены)
  python search/pipeline/00_build.py --stages 6

  # с переопределением best_embedding (например, форсим bge-m3)
  python search/pipeline/00_build.py --stages 3-6 --override-embedding BAAI/bge-m3

Каждый этап имеет свой кэш и resume-логику — повторный запуск пропускает
уже посчитанные пары (model, dataset) или (combination, dataset).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))

from _config import get_cache_dir, parse_datasets, LLM_MODEL_NAME
from _shared import write_state, setup_log_file

STAGES = {
    1: ("01_compare_embeddings.py", "Сравнение embedding-моделей"),
    2: ("02_bm25_baseline.py",      "BM25 + dense baseline"),
    3: ("03_encode_best.py",        "Encoding выбранной модели"),
    4: ("04_build_faiss.py",        "Построение FAISS-индексов"),
    6: ("06_qe_12combos.py",        "QE: 12 комбинаций (4 method × 3 aligner)"),
    7: ("07_extra_methods.py",      "QE: ThinkQE + GenCRF (extra)"),
    8: ("08_summary.py",            "Сводка результатов"),
}


def _load_stage(stage_num: int):
    fname, _ = STAGES[stage_num]
    path = PIPELINE_DIR / fname
    spec = importlib.util.spec_from_file_location(f"stage_{stage_num}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_stages(s: str) -> list[int]:
    if s.lower() == "all":
        return list(STAGES.keys())
    out: list[int] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return [s for s in out if s in STAGES]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="all",
                    help="comma-separated aliases (nfcorpus,scifact,...) или 'all'")
    ap.add_argument("--stages", default="all",
                    help="напр. '1,2,3' или '1-4' или '6' или 'all'")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--qrels-split", default=None,
                    help="принудительно brать конкретный split qrels (e.g. 'dev')")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--qe-k", type=int, default=5,
                    help="k для финальных QE метрик")
    ap.add_argument("--llm-model", default=LLM_MODEL_NAME)
    ap.add_argument("--override-embedding", default=None,
                    help="HF model name; перезаписывает best_embedding в state")
    ap.add_argument("--metric-for-best", default="ndcg@10")
    ap.add_argument("--text-field", default="raw", choices=["raw", "processed"],
                    help="raw=natural text (для dense), processed=lemmatized (legacy)")
    ap.add_argument("--rerank-models", default=None,
                    help="Comma-separated rerank passes: 'none,BAAI/bge-reranker-v2-m3,"
                         "./reranker_scifact_finetuned'. Default empty = без rerank'а.")
    ap.add_argument("--rerank-top-n", type=int, default=100)
    # Legacy single-shot:
    ap.add_argument("--rerank", action="store_true",
                    help="[deprecated] alias for --rerank-models <reranker-model>")
    ap.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    args = ap.parse_args()

    cache_dir = get_cache_dir(args.cache_dir)
    datasets = parse_datasets(args.datasets)
    stages = parse_stages(args.stages)

    # Глобальный лог оркестратора — отдельный файл, поверх него каждая стадия
    # будет создавать свой stage*-log (через setup_log_file внутри main()).
    setup_log_file(cache_dir, "pipeline_orchestrator")

    print("=" * 70)
    print(f"  PIPELINE start")
    print(f"  cache:      {cache_dir.resolve()}")
    print(f"  datasets:   {datasets}")
    print(f"  stages:     {stages}")
    print("=" * 70)

    # override записываем в state до запуска
    if args.override_embedding:
        write_state(cache_dir, "best_embedding", {
            "model": args.override_embedding, "metric": "manual_override",
            "score": None, "datasets_evaluated": datasets,
        })
        print(f"[override] best_embedding -> {args.override_embedding}")

    t0 = time.time()
    for s in stages:
        fname, descr = STAGES[s]
        t_stage = time.time()
        print(f"\n{'#' * 70}")
        print(f"#  STAGE {s}: {descr}  ({fname})")
        print(f"{'#' * 70}")
        mod = _load_stage(s)
        kwargs = {"datasets": datasets, "cache_dir": str(cache_dir),
                  "qrels_split": args.qrels_split}
        if s == 1:
            kwargs.update(top_k=args.top_k, metric_for_best=args.metric_for_best,
                          text_field=args.text_field)
        elif s == 2:
            kwargs.update(top_k=args.top_k, text_field=args.text_field)
        elif s == 3:
            kwargs.update(text_field=args.text_field)
        elif s == 6:
            stage6_mod = _load_stage(6)
            rerank_models = (stage6_mod.parse_rerank_passes(args.rerank_models)
                             if args.rerank_models else None)
            kwargs.update(llm_model=args.llm_model, k=args.qe_k,
                          rerank_models=rerank_models,
                          rerank_top_n=args.rerank_top_n,
                          rerank=args.rerank,
                          reranker_model=args.reranker_model)
        elif s == 7:
            stage6_mod = _load_stage(6)
            rerank_models = (stage6_mod.parse_rerank_passes(args.rerank_models)
                             if args.rerank_models else None)
            kwargs.update(llm_model=args.llm_model, k=args.qe_k,
                          rerank_models=rerank_models,
                          rerank_top_n=args.rerank_top_n)
        elif s == 8:
            # 08_summary не нуждается в qrels_split / datasets-фильтре
            kwargs = {"cache_dir": str(cache_dir)}
        try:
            mod.main(**kwargs)
        except Exception as ex:
            print(f"\n[STAGE {s} FAILED] {ex}")
            import traceback; traceback.print_exc()
            print(f"\nостанавливаюсь, чтобы не запускать зависимые этапы.")
            sys.exit(1)
        # после каждой стадии возвращаем лог-файл оркестратора
        setup_log_file(cache_dir, "pipeline_orchestrator")
        print(f"\n[STAGE {s} done in {(time.time() - t_stage)/60:.1f} min]")

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE done in {(time.time() - t0)/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
