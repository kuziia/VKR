"""Stage 03 — Кодирование выбранной модели в финальную папку embeddings/best/.

Если эмбеддинги уже посчитаны на Stage 01 (в embeddings_compare/{model}/),
просто пишем manifest без копирования (экономит ~14.5 GB на miracl).

Артефакты:
  qe_cache/embeddings/best.json                — manifest: {model, paths_per_dataset}
  qe_cache/embeddings/best/{ds}_emb.npy        — копия (если нужна)
  qe_cache/embeddings/best/{ds}_meta.pkl
  qe_cache/embeddings/best/{ds}_qemb.npy
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from _config import (DATASETS, MODELS_BY_NAME, get_cache_dir, model_slug,
                     parse_datasets)
from _shared import (STEncoder, build_or_load_corpus_index,
                     build_or_load_query_emb, corpus_cache_is_current,
                     emb_cache_paths, load_full_dataset,
                     query_cache_is_current, query_emb_cache_path, read_state,
                     setup_log_file, write_state)


def main(datasets: list[str] | None = None, cache_dir: str | Path | None = None,
         model: str | None = None, force_copy: bool = False,
         qrels_split: str | None = None, text_field: str = "raw") -> dict:
    cache_dir = get_cache_dir(cache_dir)
    setup_log_file(cache_dir, "stage03_encode_best")
    datasets = datasets or list(DATASETS.keys())

    if model is None:
        st = read_state(cache_dir, "best_embedding")
        if st is None:
            raise SystemExit(
                "best_embedding не найден в state. Запусти Stage 01 или передай --model"
            )
        model = st["model"]
        # Если в state есть text_field — используем его, иначе argument
        text_field = st.get("text_field", text_field)
    if model not in MODELS_BY_NAME:
        raise SystemExit(f"Model {model} not in MODELS list (_config.py)")
    model_info = MODELS_BY_NAME[model]
    print(f"[encode-best] model = {model}, text_field = {text_field}")

    best_dir = cache_dir / "embeddings" / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {"model": model, "text_field": text_field, "datasets": {}}

    encoder: STEncoder | None = None
    try:
        for ds_name in datasets:
            print(f"\n-- {ds_name}")
            d = load_full_dataset(ds_name, qrels_split=qrels_split)

            # Если уже закэшировано в embeddings_compare с правильным text_field — переиспользуем
            src_emb, src_meta = emb_cache_paths(cache_dir, model, ds_name, text_field=text_field)
            src_qemb = query_emb_cache_path(cache_dir, model, ds_name, text_field=text_field)
            if (corpus_cache_is_current(cache_dir, model, ds_name, text_field=text_field)
                    and query_cache_is_current(cache_dir, model, ds_name, text_field=text_field)):
                print(f"   [reuse] cached in embeddings_compare/{model_slug(model)}/ (text_field={text_field})")
            else:
                # Лениво создаём encoder и считаем недостающее
                if encoder is None:
                    encoder = STEncoder(model_info, device="cuda:0", fp16=True,
                                         text_field=text_field)
                ci = build_or_load_corpus_index(cache_dir, model_info, d, encoder=encoder,
                                                 text_field=text_field)
                _, _, _ = build_or_load_query_emb(cache_dir, model_info, d, encoder=encoder,
                                                    text_field=text_field)

            if force_copy:
                dst_emb  = best_dir / f"{ds_name}_emb.npy"
                dst_meta = best_dir / f"{ds_name}_meta.pkl"
                dst_qemb = best_dir / f"{ds_name}_qemb.npy"
                shutil.copy2(src_emb, dst_emb)
                shutil.copy2(src_meta, dst_meta)
                shutil.copy2(src_qemb, dst_qemb)
                manifest["datasets"][ds_name] = {
                    "emb":  str(dst_emb),
                    "meta": str(dst_meta),
                    "qemb": str(dst_qemb),
                }
            else:
                manifest["datasets"][ds_name] = {
                    "emb":  str(src_emb),
                    "meta": str(src_meta),
                    "qemb": str(src_qemb),
                }
    finally:
        if encoder is not None:
            encoder.free()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest_path = best_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    write_state(cache_dir, "best_embedding_manifest", manifest)
    print(f"\nManifest saved: {manifest_path}")
    print(f"datasets indexed: {list(manifest['datasets'].keys())}")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="all")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--model", default=None,
                   help="HF model name; default — best_embedding из state Stage 01")
    p.add_argument("--force-copy", action="store_true",
                   help="Скопировать в embeddings/best/, иначе только manifest на исходные пути")
    p.add_argument("--qrels-split", default=None)
    p.add_argument("--text-field", default="raw", choices=["raw", "processed"],
                   help="raw=natural text (для dense; default), processed=lemmatized")
    args = p.parse_args()
    main(
        datasets=parse_datasets(args.datasets),
        cache_dir=args.cache_dir,
        model=args.model,
        force_copy=args.force_copy,
        qrels_split=args.qrels_split,
        text_field=args.text_field,
    )
