"""Stage 04 — Построение FAISS-индексов.

Для каждого датасета строит HNSWFlat (если корпус >= MIN_FOR_HNSW),
иначе FlatIP. Использует эмбеддинги из Stage 03 (best_embedding_manifest).

Артефакты:
  qe_cache/faiss/{ds}.index  — сериализованный FAISS-индекс
  qe_cache/faiss/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _config import DATASETS, get_cache_dir, parse_datasets
from _shared import read_state, setup_log_file, write_state

# Корпуса меньше этого размера индексируются плоско (брутфорс быстрее HNSW build).
MIN_FOR_HNSW = 50_000
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


def build_index(emb: np.ndarray, ds_name: str, n_corpus: int):
    import faiss
    d = emb.shape[1]
    e32 = emb.astype(np.float32)
    # эмбеддинги уже нормированы (sentence-transformers normalize_embeddings=True),
    # но на всякий случай повторим — стоит копейки
    faiss.normalize_L2(e32)
    if n_corpus >= MIN_FOR_HNSW:
        idx = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        idx.hnsw.efSearch = HNSW_EF_SEARCH
        kind = f"HNSW(M={HNSW_M},efC={HNSW_EF_CONSTRUCTION})"
    else:
        idx = faiss.IndexFlatIP(d)
        kind = "Flat"
    print(f"  [{ds_name}] building {kind} on {n_corpus:,} × {d}d ...")
    idx.add(e32)
    return idx, kind


def main(datasets: list[str] | None = None, cache_dir: str | Path | None = None,
         force_rebuild: bool = False,
         qrels_split: str | None = None) -> dict:
    # qrels_split is accepted for compatibility with the orchestrator.
    del qrels_split
    cache_dir = get_cache_dir(cache_dir)
    setup_log_file(cache_dir, "stage04_build_faiss")
    datasets = datasets or list(DATASETS.keys())

    try:
        import faiss  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "faiss-cpu не установлен. Запусти:\n"
            "  pip install faiss-cpu\n"
            "(GPU-faiss конфликтует с vLLM по VRAM, поэтому используем CPU.)"
        ) from e
    import faiss

    manifest = read_state(cache_dir, "best_embedding_manifest")
    if manifest is None:
        raise SystemExit(
            "best_embedding_manifest не найден. Запусти Stage 03 (encode_best.py)."
        )

    faiss_dir = cache_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    out_manifest: dict = {"model": manifest["model"], "datasets": {}}
    for ds_name in datasets:
        if ds_name not in manifest["datasets"]:
            print(f"[skip] {ds_name}: not in encode-best manifest")
            continue
        idx_path = faiss_dir / f"{ds_name}.index"
        if idx_path.exists() and not force_rebuild:
            print(f"[reuse] {ds_name}: {idx_path}")
            out_manifest["datasets"][ds_name] = {
                "index": str(idx_path),
                "kind": "cached",
            }
            continue

        emb_path = manifest["datasets"][ds_name]["emb"]
        emb = np.load(emb_path)
        idx, kind = build_index(emb, ds_name, emb.shape[0])
        faiss.write_index(idx, str(idx_path))
        out_manifest["datasets"][ds_name] = {
            "index": str(idx_path),
            "kind": kind,
            "n_docs": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
        }
        print(f"  [{ds_name}] saved -> {idx_path} ({idx_path.stat().st_size / 1e9:.2f} GB)")

    write_state(cache_dir, "faiss_manifest", out_manifest)
    (faiss_dir / "manifest.json").write_text(
        json.dumps(out_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFAISS manifest saved.")
    return out_manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="all")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--force-rebuild", action="store_true")
    args = p.parse_args()
    main(
        datasets=parse_datasets(args.datasets),
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild,
    )
