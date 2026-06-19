"""Stage 07 — Дополнительные методы переформулирования (ThinkQE + GenCRF).

Идея: в Stage 06 структура «1 метод → 1 текст-расширение → 3 aligner-варианта →
1 retrieval» подходит для большинства методов QE, но **GenCRF** ломает этот
шаблон — у него N переформулировок + кластеризация + multi-retrieve + взвешенная
агрегация. ThinkQE формально вмещается в Stage 06 (одно итеративно уточнённое
расширение на запрос), но его удобно держать рядом с GenCRF.

Чтобы не загрязнять «чистую» 12-combo таблицу, вынесено сюда.

Что делает Stage 07:

  ThinkQE:
    - thinking → retrieve top-K (FAISS) → reflect, T=2 итерации (paper § 3)
    - финальное расширение = q + конкатенация всех thinkings
    - проходит через 3 aligner-варианта (none / CSQE / AQE) — итого 3 строки

  GenCRF:
    - 4 prompt-стратегии (contextual, detailed, aspect, disambig) генерируют
      переформулировки → KMeans-кластеризация → multi-retrieve по центроидам →
      взвешенная агрегация (similarity-weighted) → итого 1 строка

Артефакты:
  qe_cache/llm_outputs/ThinkQE_{ds}.json
  qe_cache/llm_outputs/GenCRF_reformulations_{ds}.json
  qe_cache/retrievals/{ThinkQE_*,GenCRF_full}_{ds}.json
  qe_cache/results/qe_extra_methods.csv
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _config import (DATASETS, LLM_MODEL_NAME, MODELS_BY_NAME, QE_K,
                     get_cache_dir, parse_datasets)
from _shared import (STEncoder, evaluate_run, get_query_texts_ids,
                     load_full_dataset, read_state, setup_log_file)


# ---------------------------------------------------------------------------
# Подгружаем Stage 06 как модуль, чтобы переиспользовать vllm-хелперы и
# aligner'ы (имя файла начинается с цифры → import не работает).
# ---------------------------------------------------------------------------
def _load_stage06():
    path = Path(__file__).parent / "06_qe_12combos.py"
    spec = importlib.util.spec_from_file_location("stage06", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage06 = _load_stage06()
# Шорткаты на функции/классы из Stage 06
make_vllm           = stage06.make_vllm
vllm_generate       = stage06.vllm_generate
vllm_sample_n_batch = stage06.vllm_sample_n_batch
DenseRetriever      = stage06.DenseRetriever
make_retriever_callable = stage06.make_retriever_callable
align_none          = stage06.align_none
align_csqe          = stage06.align_csqe
align_aqe           = stage06.align_aqe
load_expansions     = stage06.load_expansions
save_expansions     = stage06.save_expansions
RerankerWrapper     = stage06.RerankerWrapper
parse_rerank_passes = stage06.parse_rerank_passes
_rerank_tag         = stage06._rerank_tag


# ===========================================================================
# Параметры (paper-defaults; на A100 можно поднять без проблем)
# ===========================================================================
# ThinkQE
THINKQE_ITERS = 2            # paper рекомендует 2-3
THINKQE_FEEDBACK_K = 5
THINKQE_MAX_TOK = 200

# GenCRF
GENCRF_PER_PROMPT = 1        # сколько сэмплов на каждую инструкцию (1-2)
GENCRF_N_CLUSTERS = 3
GENCRF_RETRIEVE_PER_CLUSTER = 50
GENCRF_WEIGHT_MODE = "similarity"  # "similarity" | "uniform"
GENCRF_MAX_TOK = 80


# ===========================================================================
# ThinkQE — Lei, Shen, Yates (Findings of EMNLP 2025, p. 17772-17781)
# ===========================================================================
THINKQE_THINK_SYS = (
    "Ты — научный поисковый ассистент. По заданному запросу пиши краткий "
    "абзац-размышление (3-5 предложений) на русском, перечисляя:\n"
    "- ключевые научные термины и их синонимы\n"
    "- смежные подтемы / методы / измерения\n"
    "- типичный контекст релевантных публикаций.\n\n"
    "Пример.\n"
    "Запрос: эффективность вакцинации против гриппа у пожилых\n"
    "Размышление: Запрос про иммуногенность и клиническую эффективность гриппозной "
    "вакцины (трёхвалентной и квадривалентной) у людей старше 65 лет. Ключевые "
    "термины: иммуносенесценция, антигенный дрейф, серопротекция, hospitalization "
    "rate, mortality reduction. Релевантные исследования включают рандомизированные "
    "клинические испытания, когортные наблюдения и метаанализы CDC/ECDC. Близкие "
    "темы: high-dose vaccines, adjuvanted vaccines, age-related immune decline."
)

THINKQE_REFLECT_SYS = (
    "Ты — научный поисковый ассистент, использующий feedback от корпуса. "
    "Тебе даны: исходный запрос, предыдущие размышления, фрагменты найденных "
    "документов. Уточни размышление: ДОБАВЬ полезные термины из документов, "
    "УБЕРИ уходящие в сторону, СОХРАНИ исходное намерение. Один абзац на русском.\n\n"
    "Пример (упрощённый).\n"
    "Запрос: лечение мигрени триптанами\n"
    "Предыдущие размышления: Запрос про обезболивание при мигрени. Термины: "
    "суматриптан, золмитриптан, серотониновые рецепторы.\n"
    "Фрагменты документов:\n"
    "- Гепанты (рапатент, уброгепант) — новый класс препаратов от мигрени, "
    "блокаторы CGRP-рецепторов, без сосудосуживающего эффекта триптанов.\n"
    "- Эффективность суматриптана 100 мг превосходит плацебо в купировании "
    "острого приступа в 70% случаев.\n"
    "Уточнённое размышление: Запрос про острое купирование приступа мигрени "
    "триптанами (суматриптан, золмитриптан, ризатриптан) — агонистами 5-HT1B/1D "
    "рецепторов. Релевантные публикации сравнивают триптаны с новейшими "
    "CGRP-антагонистами (гепантами) по эффективности и безопасности. Ключевые "
    "метрики: купирование боли через 2 часа, частота рецидивов, побочные "
    "сосудистые эффекты."
)


def _thinkqe_user_initial(query: str) -> str:
    return (f"Запрос: {query}\n\n"
            "Сформулируй размышление о намерении пользователя.")


def _thinkqe_user_reflect(query: str, prev_thoughts: list[str],
                          docs: list[str]) -> str:
    prev = "\n".join(f"- {t}" for t in prev_thoughts) if prev_thoughts else "(пусто)"
    ctx = "\n".join(f"- {d[:500]}" for d in docs) if docs else "(не найдено)"
    return (f"Запрос: {query}\n\n"
            f"Предыдущие размышления:\n{prev}\n\n"
            f"Фрагменты найденных документов:\n{ctx}\n\n"
            "Сформулируй уточнённое размышление.")


def method_thinkqe(*, qids, qtexts, ds_name, cache_dir, llm, sp_cls,
                  encoder: STEncoder, dense_retriever: DenseRetriever,
                  iters: int = THINKQE_ITERS,
                  feedback_k: int = THINKQE_FEEDBACK_K,
                  id2text: dict[str, str], **_) -> list[str]:
    """ThinkQE: T итераций (think → retrieve → reflect).

    Returns: для каждого запроса — конкатенация thoughts (без q;
    оригинал q приклеит aligner). Если уже в кэше — возвращаем закэшированное.
    """
    cache = load_expansions(cache_dir, "ThinkQE", ds_name)
    todo_idx = [i for i, qid in enumerate(qids) if qid not in cache]
    if not todo_idx:
        return [cache[qid] for qid in qids]

    todo_qids   = [qids[i] for i in todo_idx]
    todo_qtexts = [qtexts[i] for i in todo_idx]
    n = len(todo_qtexts)
    print(f"  [ThinkQE] {n:,} queries × ({iters} iter + initial)")

    # === iter 0: чистое thinking без обратной связи ===
    init_prompts = [
        [{"role": "system", "content": THINKQE_THINK_SYS},
         {"role": "user",   "content": _thinkqe_user_initial(q)}]
        for q in todo_qtexts
    ]
    init_thoughts = vllm_generate(llm, sp_cls, init_prompts,
                                  max_new_tokens=THINKQE_MAX_TOK,
                                  temperature=0.0)
    all_thoughts: list[list[str]] = [[t] for t in init_thoughts]

    # === iter 1..T: с обратной связью ===
    for it in range(iters):
        expanded = [
            (q + " " + " ".join(all_thoughts[i])).strip()
            for i, q in enumerate(todo_qtexts)
        ]
        exp_emb = encoder.encode_queries(expanded)
        ranked = dense_retriever.search(exp_emb, feedback_k)

        reflect_prompts = []
        for i, q in enumerate(todo_qtexts):
            doc_ids = ranked.get(i, [])
            docs = [id2text.get(did, "") for did in doc_ids]
            reflect_prompts.append([
                {"role": "system", "content": THINKQE_REFLECT_SYS},
                {"role": "user",
                 "content": _thinkqe_user_reflect(q, all_thoughts[i], docs)},
            ])
        new_thoughts = vllm_generate(llm, sp_cls, reflect_prompts,
                                     max_new_tokens=THINKQE_MAX_TOK,
                                     temperature=0.0)
        for i, t in enumerate(new_thoughts):
            all_thoughts[i].append(t)

    # Сохраняем как expansion (без q): aligner добавит оригинал.
    for i, qid in enumerate(todo_qids):
        cache[qid] = " ".join(all_thoughts[i]).strip()
    save_expansions(cache_dir, "ThinkQE", ds_name, cache)
    return [cache[qid] for qid in qids]


# ===========================================================================
# GenCRF — Seo, Zhang, Zhang et al. (arXiv:2409.10909, 2024)
# ===========================================================================
GENCRF_PROMPTS = {
    "contextual": (
        "Перепиши научный поисковый запрос, ДОБАВИВ смежные понятия и "
        "контекст. Одна фраза на русском.\n"
        "Пример: «эффект витамина D на иммунитет» → "
        "«роль витамина D и кальцитриола в регуляции врождённого иммунитета и "
        "противоинфекционной защиты»"),
    "detailed": (
        "Перепиши запрос с УТОЧНЕНИЯМИ: укажи возможные методы, объекты, "
        "условия исследования. Одна фраза на русском.\n"
        "Пример: «эффект витамина D на иммунитет» → "
        "«рандомизированные клинические испытания добавок витамина D3 у "
        "взрослых: маркеры активности Т-клеток и частота респираторных инфекций»"),
    "aspect": (
        "Перепиши запрос, СУЗИВ его до одного конкретного аспекта (механизм, "
        "эффект, приложение). Одна фраза на русском.\n"
        "Пример: «эффект витамина D на иммунитет» → "
        "«молекулярный механизм действия витамина D через VDR-рецепторы в "
        "макрофагах»"),
    "disambig": (
        "Перепиши запрос, УБРАВ неоднозначности: уточни предметную область "
        "и тип искомого документа. Одна фраза на русском.\n"
        "Пример: «эффект витамина D на иммунитет» → "
        "«биомедицинский обзор: иммуномодулирующее действие витамина D у "
        "людей (не модельных животных)»"),
}
GENCRF_SYS = (
    "Ты помощник по информационному поиску. Перепиши пользовательский "
    "запрос согласно инструкции, сохраняя исходное намерение. "
    "Верни только переформулированный запрос, без пояснений."
)


def _gencrf_clean(raw: str, fallback: str) -> str:
    txt = (raw or "").strip().split("\n")[0].strip(" \"'«»—-•")
    return txt or fallback


def _gencrf_load_reformulations(cache_dir: Path, ds_name: str) -> dict[str, list[str]]:
    p = cache_dir / "llm_outputs" / f"GenCRF_reformulations_{ds_name}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _gencrf_save_reformulations(cache_dir: Path, ds_name: str,
                                data: dict[str, list[str]]) -> None:
    p = cache_dir / "llm_outputs" / f"GenCRF_reformulations_{ds_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def gencrf_generate_reformulations(qids, qtexts, ds_name, cache_dir, llm, sp_cls,
                                   per_prompt: int = GENCRF_PER_PROMPT
                                   ) -> dict[str, list[str]]:
    """Возвращает dict: qid → [original, ref_contextual, ref_detailed, ...].
    Кэшируется на диске; при повторе подбираются только новые qids."""
    cache = _gencrf_load_reformulations(cache_dir, ds_name)
    todo = [(qid, q) for qid, q in zip(qids, qtexts) if qid not in cache]
    if not todo:
        return cache

    print(f"  [GenCRF] generating reformulations for {len(todo):,} queries"
          f" × {len(GENCRF_PROMPTS)} стратегий × {per_prompt} sample(s)")

    # для каждого запроса собираем оригинал + переформулировки
    qid2refs: dict[str, list[str]] = {qid: [q] for qid, q in todo}

    for tag, instr in GENCRF_PROMPTS.items():
        prompts = [
            [{"role": "system", "content": GENCRF_SYS},
             {"role": "user",
              "content": f"Инструкция: {instr}\n\nЗапрос: {q}\n\nПереформулировка:"}]
            for _, q in todo
        ]
        # vLLM n=per_prompt — амортизация prefill для одинакового system
        if per_prompt == 1:
            outs = vllm_generate(llm, sp_cls, prompts,
                                 max_new_tokens=GENCRF_MAX_TOK,
                                 temperature=0.7, top_p=0.95)
            for (qid, q_orig), o in zip(todo, outs):
                qid2refs[qid].append(_gencrf_clean(o, q_orig))
        else:
            outs = vllm_sample_n_batch(llm, sp_cls, prompts, n=per_prompt,
                                       max_new_tokens=GENCRF_MAX_TOK,
                                       temperature=0.7, top_p=0.95)
            for (qid, q_orig), samples in zip(todo, outs):
                for s in samples:
                    qid2refs[qid].append(_gencrf_clean(s, q_orig))

    cache.update(qid2refs)
    _gencrf_save_reformulations(cache_dir, ds_name, cache)
    return cache


def gencrf_full_search(qids, qtexts, ds_name, cache_dir, llm, sp_cls,
                       encoder: STEncoder, dense_retriever: DenseRetriever,
                       corpus_ids: list[str],
                       n_clusters: int = GENCRF_N_CLUSTERS,
                       retrieve_per_cluster: int = GENCRF_RETRIEVE_PER_CLUSTER,
                       weight_mode: str = GENCRF_WEIGHT_MODE,
                       top_k: int = QE_K) -> dict[str, list[str]]:
    """Полный GenCRF: переформулировки → KMeans → multi-retrieve → weighted aggregation.
    Возвращает qid → top-K corpus ids (минуя aligner-стадию)."""
    from sklearn.cluster import KMeans

    refs_by_qid = gencrf_generate_reformulations(qids, qtexts, ds_name,
                                                  cache_dir, llm, sp_cls)

    # эмбеддинги всех reformulations + оригинал
    flat_texts: list[str] = []
    owners: list[int] = []
    for qi, qid in enumerate(qids):
        for r in refs_by_qid.get(qid, [qtexts[qi]]):
            flat_texts.append(r); owners.append(qi)
    flat_emb = encoder.encode_queries(flat_texts).astype(np.float32)
    q_orig_emb = encoder.encode_queries(qtexts).astype(np.float32)

    by_q: dict[int, list[int]] = defaultdict(list)
    for j, o in enumerate(owners):
        by_q[o].append(j)

    print(f"  [GenCRF] clustering + multi-retrieve for {len(qids):,} queries ...")
    out_per_qid: dict[str, list[str]] = {}
    for qi, qid in enumerate(qids):
        idxs = by_q[qi]
        embs = flat_emb[idxs]
        n_pts = embs.shape[0]
        k = max(1, min(n_clusters, n_pts))

        if k == 1 or n_pts <= 1:
            labels = np.zeros(n_pts, dtype=np.int64)
            centroids = embs.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k, n_init=4, random_state=42)
            labels = km.fit_predict(embs)
            centroids = km.cluster_centers_.astype(np.float32)
        # нормируем центроиды для cosine
        norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
        centroids = centroids / norms

        # веса кластеров
        weights = np.zeros(k, dtype=np.float32)
        if weight_mode == "similarity":
            sims_to_orig = embs @ q_orig_emb[qi]
            for ci in range(k):
                mask = labels == ci
                if mask.any():
                    weights[ci] = float(np.mean(sims_to_orig[mask]))
            weights = np.clip(weights, 0.0, None)
            if weights.sum() <= 0:
                weights[:] = 1.0 / k
            else:
                weights = weights / weights.sum()
        else:  # uniform
            weights[:] = 1.0 / k

        # multi-retrieve через FAISS + weighted aggregation
        scores: dict[str, float] = defaultdict(float)
        cluster_ranked = dense_retriever.search(centroids, retrieve_per_cluster)
        # cluster_ranked: dict[cluster_idx -> list[doc_id]]
        # FAISS возвращает _, ids; для score нужен сырой similarity, поэтому
        # достанем напрямую через faiss
        import faiss
        cn = centroids.copy()
        faiss.normalize_L2(cn)
        idx = dense_retriever.idx
        if hasattr(idx, "hnsw"):
            idx.hnsw.efSearch = max(64, retrieve_per_cluster * 2)
        c_scores, c_ids = idx.search(cn, retrieve_per_cluster)
        for ci in range(k):
            for j_doc, sc in zip(c_ids[ci], c_scores[ci]):
                if j_doc < 0:
                    continue
                scores[corpus_ids[int(j_doc)]] += weights[ci] * float(sc)

        if not scores:
            out_per_qid[qid] = []
        else:
            ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
            out_per_qid[qid] = [d for d, _ in ranked]
    return out_per_qid


# ===========================================================================
# Main
# ===========================================================================
def main(datasets: list[str] | None = None, cache_dir: str | Path | None = None,
         llm_model: str = LLM_MODEL_NAME, retriever_kind: str | None = None,
         qrels_split: str | None = None, k: int = QE_K,
         skip_thinkqe: bool = False, skip_gencrf: bool = False,
         thinkqe_aligners: list[str] | None = None,
         rerank_models: list[str | None] | None = None,
         rerank_top_n: int = 100) -> dict:
    cache_dir = get_cache_dir(cache_dir)
    setup_log_file(cache_dir, "stage07_extra_methods")
    datasets = datasets or list(DATASETS.keys())
    thinkqe_aligners = thinkqe_aligners or ["none", "CSQE", "AQE"]
    if rerank_models is None or not rerank_models:
        rerank_models = [None]
    any_rerank = any(m is not None for m in rerank_models)
    print(f"[rerank passes] {[_rerank_tag(m) for m in rerank_models]}")

    manifest_emb = read_state(cache_dir, "best_embedding_manifest")
    manifest_faiss = read_state(cache_dir, "faiss_manifest")
    if manifest_emb is None or manifest_faiss is None:
        raise SystemExit("Запусти Stages 03 + 04 сначала.")

    # Ретривер всегда dense (Stage 05 удалён); параметр сохранён для ablation.
    if retriever_kind is None:
        retriever_kind = "dense"
    print(f"[retriever] {retriever_kind}")
    print(f"[embedding] {manifest_emb['model']}")
    print(f"[llm]      {llm_model}")

    llm, sp_cls = make_vllm(llm_model)
    model_info = MODELS_BY_NAME[manifest_emb["model"]]
    encoder = STEncoder(model_info, device="cuda:0", fp16=True)

    # aligners как в Stage 06
    aligner_map = {"none": align_none, "CSQE": align_csqe, "AQE": align_aqe}

    rerankers: dict[str, RerankerWrapper] = {}
    def _get_reranker(name: str) -> RerankerWrapper:
        if name not in rerankers:
            rerankers[name] = RerankerWrapper(name)
        return rerankers[name]

    res_dir = cache_dir / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    res_csv = res_dir / "qe_extra_methods.csv"
    rows = pd.read_csv(res_csv).to_dict("records") if res_csv.exists() else []
    for r in rows:
        r.setdefault("reranker", "none")
        r.setdefault("rerank_top_n", None)
        r.setdefault("retriever", "unknown")
    done = {(r["dataset"], r["combination"],
             r.get("reranker", "none"), r.get("retriever", "unknown"))
            for r in rows}

    ret_dir = cache_dir / "retrievals"; ret_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        if ds_name not in manifest_emb["datasets"]:
            print(f"[skip] {ds_name}: not in best_embedding manifest")
            continue
        print(f"\n{'=' * 60}\n  DATASET: {ds_name}\n{'=' * 60}")
        d = load_full_dataset(ds_name, qrels_split=qrels_split)

        with open(manifest_emb["datasets"][ds_name]["meta"], "rb") as f:
            meta = pickle.load(f)
        corpus_ids, corpus_texts = meta["ids"], meta["texts"]
        corpus_emb = np.load(manifest_emb["datasets"][ds_name]["emb"])
        # qtexts с тем же text_field, что и manifest (см. 06_qe_12combos.py).
        ds_text_field = manifest_emb.get("text_field", "raw")
        qids, qtexts = get_query_texts_ids(d, text_field=ds_text_field)

        id2text = {cid: ctx for cid, ctx in zip(corpus_ids, corpus_texts)}
        dense_retriever = DenseRetriever(
            manifest_faiss["datasets"][ds_name]["index"], corpus_ids,
        )
        retriever_initial = make_retriever_callable(
            retriever_kind, corpus_ids, corpus_texts,
            manifest_faiss["datasets"][ds_name]["index"],
            cache_dir=cache_dir, ds_name=ds_name,
        )

        # RAW text для cross-encoder rerank'а — независимо от --text-field
        # пайплайна. Если хоть один rerank-проход активен — нужно подгрузить.
        id2text_raw: dict[str, str] = {}
        qtexts_raw: list[str] = []
        if any_rerank:
            from _shared import get_corpus_texts_ids
            cids_r, ctexts_r = get_corpus_texts_ids(d, text_field="raw")
            id2text_raw = {cid: ct for cid, ct in zip(cids_r, ctexts_r)}
            _, qtexts_raw = get_query_texts_ids(d, text_field="raw")

        # ============================== ThinkQE ==============================
        if not skip_thinkqe:
            # «Все aligners × все rerank-проходы» закэшированы?
            thinkqe_skip_all = all(
                (ds_name, f"ThinkQE + {a}", _rerank_tag(rm), retriever_kind) in done
                for a in thinkqe_aligners for rm in rerank_models
            )
            if thinkqe_skip_all:
                print("\n  [ThinkQE] all (aligner × rerank) комбинации кэшированы, skip")
            else:
                print(f"\n  >> ThinkQE")
                t_thinkqe = time.time()
                try:
                    thinkqe_exp = method_thinkqe(
                        qids=qids, qtexts=qtexts, ds_name=ds_name,
                        cache_dir=cache_dir, llm=llm, sp_cls=sp_cls,
                        encoder=encoder, dense_retriever=dense_retriever,
                        id2text=id2text,
                    )
                except Exception as ex:
                    print(f"     [ERR] ThinkQE: {ex}")
                    import traceback; traceback.print_exc()
                    thinkqe_exp = list(qtexts)
                thinkqe_method_time = time.time() - t_thinkqe
                print(f"     [ThinkQE] generation: {thinkqe_method_time:.1f}s")

                align_kwargs = dict(
                    encoder=encoder, retriever_initial=retriever_initial,
                    id2text=id2text, corpus_emb=corpus_emb,
                    llm=llm, sp_cls=sp_cls,
                )
                for a_name in thinkqe_aligners:
                    tag = f"ThinkQE + {a_name}"
                    pending_passes = [rm for rm in rerank_models
                                      if (ds_name, tag, _rerank_tag(rm), retriever_kind) not in done]
                    if not pending_passes:
                        print(f"  [skip] {tag}")
                        continue
                    print(f"\n  -- {tag}  rerank: {[_rerank_tag(rm) for rm in pending_passes]}")
                    wide_n = max(rerank_top_n, k) if any(p is not None for p in pending_passes) else k

                    t_a = time.time()
                    base_failed = False
                    ranked_wide: dict[int, list[str]] = {}
                    try:
                        final_q = aligner_map[a_name](
                            qtexts, thinkqe_exp, **align_kwargs,
                        )
                        f_emb = encoder.encode_queries(final_q)
                        ranked_wide = retriever_initial(f_emb, final_q, wide_n)
                    except Exception as ex:
                        print(f"     [ERR base] {tag}: {ex}")
                        import traceback; traceback.print_exc()
                        base_failed = True
                    base_time = time.time() - t_a

                    for rm in pending_passes:
                        tag_rm = _rerank_tag(rm)
                        t_rr = time.time()
                        if base_failed:
                            metrics = {"recall@5": None, "map@5": None,
                                       "ndcg@5": None, "recall@10": None,
                                       "map@10": None, "ndcg@10": None,
                                       "n_eval": 0}
                            rerank_time = 0.0
                        else:
                            try:
                                if rm is None:
                                    ranked = {qi: ranked_wide.get(qi, [])[:k]
                                              for qi in range(len(qids))}
                                else:
                                    rr = _get_reranker(rm)
                                    ranked = rr.rerank(qtexts_raw, ranked_wide,
                                                       id2text_raw, k)
                                ret_per_q = {qids[qi]: ranked.get(qi, [])
                                             for qi in range(len(qids))}
                                slug = ''.join(c if c.isalnum() or c in '-_' else '_' for c in tag_rm)
                                (ret_dir / f"ThinkQE_{a_name}_{ds_name}__{slug}.json").write_text(
                                    json.dumps(ret_per_q, ensure_ascii=False),
                                    encoding="utf-8")
                                metrics = evaluate_run(ret_per_q, d["qrels"], ks=(5, 10))
                                rerank_time = time.time() - t_rr
                                print(f"     -> [rerank={tag_rm}] {metrics}  | base={base_time:.1f}s rerank={rerank_time:.1f}s")
                            except Exception as ex:
                                print(f"     [ERR rerank={tag_rm}] {tag}: {ex}")
                                import traceback; traceback.print_exc()
                                metrics = {"recall@5": None, "map@5": None,
                                           "ndcg@5": None, "recall@10": None,
                                           "map@10": None, "ndcg@10": None,
                                           "n_eval": 0}
                                rerank_time = time.time() - t_rr
                        rows.append({
                            "dataset": ds_name, "method": "ThinkQE",
                            "aligner": a_name, "combination": tag, **metrics,
                            "retriever": retriever_kind,
                            "reranker": tag_rm,
                            "rerank_top_n": rerank_top_n if rm is not None else None,
                            "method_time_sec": round(thinkqe_method_time, 2),
                            "align_time_sec": round(base_time, 2),
                            "rerank_time_sec": round(rerank_time, 2),
                            "time_sec": round(thinkqe_method_time + base_time + rerank_time, 2),
                        })
                        done.add((ds_name, tag, tag_rm, retriever_kind))
                    pd.DataFrame(rows).to_csv(res_csv, index=False)

        # ============================== GenCRF ==============================
        if not skip_gencrf:
            tag = "GenCRF (multi-cluster RRF)"
            pending_passes = [rm for rm in rerank_models
                              if (ds_name, tag, _rerank_tag(rm), retriever_kind) not in done]
            if not pending_passes:
                print(f"\n  [skip] {tag}")
            else:
                print(f"\n  >> {tag}  rerank: {[_rerank_tag(rm) for rm in pending_passes]}")
                wide_n = max(rerank_top_n, k) if any(p is not None for p in pending_passes) else k
                t_gen = time.time()
                base_failed = False
                ret_per_qid_wide: dict[str, list[str]] = {}
                try:
                    ret_per_qid_wide = gencrf_full_search(
                        qids, qtexts, ds_name, cache_dir, llm, sp_cls,
                        encoder=encoder, dense_retriever=dense_retriever,
                        corpus_ids=corpus_ids, top_k=wide_n,
                    )
                except Exception as ex:
                    print(f"     [ERR base] GenCRF: {ex}")
                    import traceback; traceback.print_exc()
                    base_failed = True
                gen_time = time.time() - t_gen

                for rm in pending_passes:
                    tag_rm = _rerank_tag(rm)
                    t_rr = time.time()
                    if base_failed:
                        metrics = {"recall@5": None, "map@5": None,
                                   "ndcg@5": None, "recall@10": None,
                                   "map@10": None, "ndcg@10": None,
                                   "n_eval": 0}
                        rerank_time = 0.0
                    else:
                        try:
                            if rm is None:
                                ret_per_q = {qid: ret_per_qid_wide.get(qid, [])[:k]
                                             for qid in qids}
                            else:
                                rr = _get_reranker(rm)
                                # rerank ожидает dict[qi -> ids]
                                wide_by_qi = {qi: ret_per_qid_wide.get(qid, [])
                                              for qi, qid in enumerate(qids)}
                                ranked = rr.rerank(qtexts_raw, wide_by_qi,
                                                   id2text_raw, k)
                                ret_per_q = {qids[qi]: ranked.get(qi, [])
                                             for qi in range(len(qids))}
                            slug = ''.join(c if c.isalnum() or c in '-_' else '_' for c in tag_rm)
                            (ret_dir / f"GenCRF_full_{ds_name}__{slug}.json").write_text(
                                json.dumps(ret_per_q, ensure_ascii=False),
                                encoding="utf-8")
                            metrics = evaluate_run(ret_per_q, d["qrels"], ks=(5, 10))
                            rerank_time = time.time() - t_rr
                            print(f"     -> [rerank={tag_rm}] {metrics}  | base={gen_time:.1f}s rerank={rerank_time:.1f}s")
                        except Exception as ex:
                            print(f"     [ERR rerank={tag_rm}] GenCRF: {ex}")
                            import traceback; traceback.print_exc()
                            metrics = {"recall@5": None, "map@5": None,
                                       "ndcg@5": None, "recall@10": None,
                                       "map@10": None, "ndcg@10": None,
                                       "n_eval": 0}
                            rerank_time = time.time() - t_rr
                    rows.append({
                        "dataset": ds_name, "method": "GenCRF",
                        "aligner": "—", "combination": tag, **metrics,
                        "retriever": retriever_kind,
                        "reranker": tag_rm,
                        "rerank_top_n": rerank_top_n if rm is not None else None,
                        "method_time_sec": round(gen_time, 2),
                        "align_time_sec": 0.0,
                        "rerank_time_sec": round(rerank_time, 2),
                        "time_sec": round(gen_time + rerank_time, 2),
                    })
                    done.add((ds_name, tag, tag_rm, retriever_kind))
                pd.DataFrame(rows).to_csv(res_csv, index=False)

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    for rr in rerankers.values():
        rr.free()

    df = pd.DataFrame(rows)
    print("\n=== Mean metrics across datasets ===")
    metric_cols = [c for c in df.columns if c.startswith(("recall@", "map@", "ndcg@"))]
    if not df.empty:
        group_cols = ["combination", "reranker"] if "reranker" in df.columns else ["combination"]
        print(df.groupby(group_cols, sort=False)[metric_cols].mean().round(4))
    print(f"\nSaved: {res_csv}")
    return {"csv": str(res_csv)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="all")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--llm-model", default=LLM_MODEL_NAME)
    p.add_argument("--retriever", default=None,
                   help="dense|bm25|hybrid_rrf; default — best_retriever из state")
    p.add_argument("--qrels-split", default=None)
    p.add_argument("--k", type=int, default=QE_K)
    p.add_argument("--skip-thinkqe", action="store_true")
    p.add_argument("--skip-gencrf",  action="store_true")
    p.add_argument("--thinkqe-aligners", default="none,CSQE,AQE",
                   help="comma-separated subset of {none,CSQE,AQE}")
    p.add_argument("--rerank-models", default=None,
                   help="Comma-separated rerank passes (см. 06_qe_12combos.py).")
    p.add_argument("--rerank-top-n", type=int, default=100)
    args = p.parse_args()
    main(
        datasets=parse_datasets(args.datasets),
        cache_dir=args.cache_dir,
        llm_model=args.llm_model,
        retriever_kind=args.retriever,
        qrels_split=args.qrels_split,
        k=args.k,
        skip_thinkqe=args.skip_thinkqe,
        skip_gencrf=args.skip_gencrf,
        thinkqe_aligners=[s.strip() for s in args.thinkqe_aligners.split(",")
                          if s.strip()],
        rerank_models=parse_rerank_passes(args.rerank_models)
                       if args.rerank_models else None,
        rerank_top_n=args.rerank_top_n,
    )
