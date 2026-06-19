"""Stage 06 — Query Expansion: 12 комбинаций.

4 методa  × 3 aligner-варианта = 12 комбинаций:
  Methods : Query2doc, PromptPRF, PQR, Word2Passage
  Aligners: none (no alignment), CSQE, AQE

Использует:
  - best_embedding из state (Stage 01)  →  e5/bge для encoding запросов
  - faiss_manifest из state (Stage 04)  →  быстрый retrieval (HNSW/Flat)
  - best_retriever из state (Stage 05)  →  dense / bm25 / hybrid_rrf

Артефакты:
  qe_cache/llm_outputs/{method}_{ds}.json
  qe_cache/llm_outputs/PromptPRF_features_{ds}_{ftype}.json
  qe_cache/retrievals/{method}_{aligner}_{ds}.json
  qe_cache/results/qe_12combos.csv
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _config import (DATASETS, LLM_MODEL_NAME, MODELS_BY_NAME, PRF_DEPTH, QE_K,
                     VLLM_GPU_UTIL, VLLM_MAX_LEN, get_cache_dir, parse_datasets)
from _shared import (STEncoder, build_or_load_bm25_index, chunked_retrieve,
                     evaluate_run, get_corpus_texts_ids, get_query_texts_ids,
                     lemmatize_ru, load_full_dataset, read_state,
                     setup_log_file, tokenize_simple)


# =============================================================================
# Retrievers
# =============================================================================
class DenseRetriever:
    def __init__(self, faiss_index_path: str, corpus_ids: list[str]):
        import faiss
        self.idx = faiss.read_index(str(faiss_index_path))
        self.corpus_ids = corpus_ids

    def search(self, query_emb: np.ndarray, top_k: int) -> dict[int, list[str]]:
        import faiss
        qe = query_emb.astype(np.float32)
        faiss.normalize_L2(qe)
        if hasattr(self.idx, "hnsw"):
            self.idx.hnsw.efSearch = max(64, top_k * 4)
        _, ids = self.idx.search(qe, top_k)
        return {i: [self.corpus_ids[int(j)] for j in ids[i]] for i in range(qe.shape[0])}


class BM25Retriever:
    """BM25 поверх кэшированного `BM25Okapi` индекса (`build_or_load_bm25_index`).

    Раньше использовал bm25s (быстрее на запросах), но без сериализации индекса.
    Сейчас единый rank_bm25 для consistency между Stage 2 / Stage 6 и для
    переиспользования с `BM25WeightedRetriever`.
    """
    def __init__(self, corpus_texts: list[str], corpus_ids: list[str],
                 cache_dir: Path | None = None, ds_name: str | None = None):
        self.corpus_ids = corpus_ids
        if cache_dir is not None and ds_name is not None:
            self._bm25 = build_or_load_bm25_index(cache_dir, ds_name, corpus_texts)
        else:
            from rank_bm25 import BM25Okapi
            print("  [BM25Okapi] tokenize+index (no cache) ...")
            tok = [tokenize_simple(t) for t in tqdm(corpus_texts, desc="tok")]
            self._bm25 = BM25Okapi(tok)

    def search(self, queries_text: list[str], top_k: int) -> dict[int, list[str]]:
        # Лемматизируем queries — corpus у kaengreg/* в processed_text
        # (lowercased + norm-form), а LLM-генерации приходят в обычной флексии.
        queries_text = [lemmatize_ru(q) for q in queries_text]
        out: dict[int, list[str]] = {}
        for qi, qtxt in enumerate(queries_text):
            scores = self._bm25.get_scores(tokenize_simple(qtxt))
            n_ret = min(top_k, len(scores))
            top_idx = np.argpartition(scores, -n_ret)[-n_ret:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            out[qi] = [self.corpus_ids[int(j)] for j in top_idx]
        return out


class BM25WeightedRetriever:
    """Paper-faithful Word2Passage scoring (Choi et al., ACL Findings 2025, формула 8):
        S(Q̃, doc) = Σ_t I_t · BM25(t, doc)
    Каждый term t имеет свой вес I_t (вычисленный методом W2P
    через multi-level frequencies × query-type-significance).

    Переиспользует общий кэш BM25Okapi с `BM25Retriever` через
    `build_or_load_bm25_index`.
    """
    def __init__(self, corpus_texts: list[str], corpus_ids: list[str],
                 cache_dir: Path | None = None, ds_name: str | None = None):
        self.corpus_ids = corpus_ids
        self.n_docs = len(corpus_ids)
        if cache_dir is not None and ds_name is not None:
            self._bm25 = build_or_load_bm25_index(cache_dir, ds_name, corpus_texts)
        else:
            from rank_bm25 import BM25Okapi
            print("  [BM25Weighted] tokenize+index (no cache) ...")
            tok = [tokenize_simple(t) for t in tqdm(corpus_texts, desc="tok")]
            self._bm25 = BM25Okapi(tok)

    def search_weighted(self, weighted_queries: list[dict[str, float]],
                        top_k: int) -> dict[int, list[str]]:
        """weighted_queries[qi] = {лемматизированный_term: вес}."""
        out: dict[int, list[str]] = {}
        for qi, weights in enumerate(weighted_queries):
            if not weights:
                out[qi] = []
                continue
            total = np.zeros(self.n_docs, dtype=np.float32)
            for term, w in weights.items():
                term_lem = lemmatize_ru(term)
                if not term_lem:
                    continue
                total += w * self._bm25.get_scores(tokenize_simple(term_lem))
            n_ret = min(top_k, self.n_docs)
            top_idx = np.argpartition(total, -n_ret)[-n_ret:]
            top_idx = top_idx[np.argsort(total[top_idx])[::-1]]
            out[qi] = [self.corpus_ids[int(j)] for j in top_idx]
        return out


class HybridRRF:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever,
                 top_retrieve: int = 100, rrf_k: int = 60):
        self.dense = dense; self.bm25 = bm25
        self.top_retrieve = top_retrieve; self.rrf_k = rrf_k

    def search(self, query_emb: np.ndarray, queries_text: list[str],
               top_k: int) -> dict[int, list[str]]:
        d_rank = self.dense.search(query_emb, self.top_retrieve)
        b_rank = self.bm25.search(queries_text, self.top_retrieve)
        out = {}
        for qi in range(len(queries_text)):
            scores: dict[str, float] = {}
            for ranked in (d_rank.get(qi, []), b_rank.get(qi, [])):
                for rk, doc in enumerate(ranked):
                    scores[doc] = scores.get(doc, 0.0) + 1.0 / (self.rrf_k + rk + 1)
            out[qi] = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        return out


class RerankerWrapper:
    """Cross-encoder rerank: top-N от base retriever → top-K по cross-encoder скорам.

    Использует RAW text (не лемматизированный) для query и candidates — cross-encoder
    обучался на natural language, лемматизация ломает его.

    Все пары (query, candidate_doc) для всех queries одной комбинации
    скорятся одним batched `model.predict()` вызовом — это позволяет
    sentence-transformers'у smart-batch'ить по длине, лучше использовать
    GPU и убирает per-query kernel-launch overhead. На A100 даёт 4-8×
    speedup vs per-query цикла.
    """
    def __init__(self, reranker_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cuda:0", max_length: int = 512,
                 batch_size: int = 128, fp16: bool = True):
        from sentence_transformers import CrossEncoder
        print(f"  [Reranker] loading {reranker_name} ...")
        self.model = CrossEncoder(reranker_name, device=device, max_length=max_length)
        # fp16 inference: на A100 даёт 2.5-3× speedup vs fp32 (tensor cores +
        # half memory bandwidth). Качество классификации не страдает —
        # cross-encoder выдаёт скаляр, не текст.
        if fp16:
            try:
                self.model.model.half()
                print("  [Reranker] converted to fp16")
            except Exception as ex:
                print(f"  [Reranker] fp16 conversion failed: {ex}; staying fp32")
        self.batch_size = batch_size
        self.name = reranker_name

    def rerank(self, qtexts_raw: list[str],
               ranked_topn: dict[int, list[str]],
               id2text_raw: dict[str, str],
               top_k: int, doc_truncate: int = 1500) -> dict[int, list[str]]:
        n = len(qtexts_raw)
        # === 1. Flatten все пары в один список ===
        flat_pairs: list[tuple[str, str]] = []
        # offsets[qi] = (start_index_in_flat, count). count=0 для empty cand.
        offsets: list[tuple[int, int]] = []
        cand_per_qi: dict[int, list[str]] = {}
        for qi in range(n):
            cand_ids = ranked_topn.get(qi, [])
            cand_per_qi[qi] = cand_ids
            start = len(flat_pairs)
            for cid in cand_ids:
                flat_pairs.append(
                    (qtexts_raw[qi], id2text_raw.get(cid, "")[:doc_truncate]))
            offsets.append((start, len(cand_ids)))

        if not flat_pairs:
            return {qi: [] for qi in range(n)}

        # === 2. Один predict на все 30K пар (на A100 ~50-100 сек) ===
        # show_progress_bar=True даёт встроенный tqdm с реальным % batches.
        # desc меняется через прямую переменную (sentence-transformers не
        # принимает desc), так что вместо custom-desc пусть будет дефолтный
        # bar — главное, виден прогресс.
        print(f"  [rerank/{self.name.split('/')[-1]}] scoring {len(flat_pairs):,} pairs "
              f"(batch={self.batch_size}, {n} queries) ...")
        flat_scores = self.model.predict(
            flat_pairs, batch_size=self.batch_size,
            show_progress_bar=True,
        )
        flat_scores = np.asarray(flat_scores, dtype=np.float32)

        # === 3. Разложить scores обратно и взять top-K на query ===
        out: dict[int, list[str]] = {}
        for qi in range(n):
            start, cnt = offsets[qi]
            if cnt == 0:
                out[qi] = []
                continue
            scores = flat_scores[start:start + cnt]
            order = np.argsort(-scores)[:top_k]
            out[qi] = [cand_per_qi[qi][int(j)] for j in order]
        return out

    def free(self):
        try:
            del self.model
        except AttributeError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_retriever_callable(kind: str, corpus_ids: list[str], corpus_texts: list[str],
                            faiss_index_path: str,
                            cache_dir: Path | None = None,
                            ds_name: str | None = None):
    """Возвращает callable f(qemb, qtexts, top_k) -> {qi: [doc_id]}.

    `corpus_texts` для BM25-путей должен быть processed (lemmatized) — иначе
    LLM-лемматизация запросов в `BM25Retriever.search` не совпадёт с корпусом.
    Для dense `corpus_texts` не используется (FAISS индексирует embeddings).
    """
    if kind == "dense":
        d = DenseRetriever(faiss_index_path, corpus_ids)
        return lambda qemb, qtxt, k: d.search(qemb, k)
    if kind == "bm25":
        bm = BM25Retriever(corpus_texts, corpus_ids,
                           cache_dir=cache_dir, ds_name=ds_name)
        return lambda qemb, qtxt, k: bm.search(qtxt, k)
    if kind == "hybrid_rrf":
        d = DenseRetriever(faiss_index_path, corpus_ids)
        bm = BM25Retriever(corpus_texts, corpus_ids,
                           cache_dir=cache_dir, ds_name=ds_name)
        h = HybridRRF(d, bm)
        return lambda qemb, qtxt, k: h.search(qemb, qtxt, k)
    raise ValueError(f"Unknown retriever: {kind}")


# =============================================================================
# vLLM
# =============================================================================
def make_vllm(model_name: str = LLM_MODEL_NAME):
    from vllm import LLM, SamplingParams
    print(f"Loading vLLM with {model_name} ...")
    llm = LLM(
        model=model_name, dtype="bfloat16",
        gpu_memory_utilization=VLLM_GPU_UTIL,
        max_model_len=VLLM_MAX_LEN,
        enable_prefix_caching=True,
        trust_remote_code=False, disable_log_stats=True,
    )
    print("vLLM ready.")
    return llm, SamplingParams


def _fallback_chat_render(msgs: list[dict]) -> str:
    """Универсальный fallback, когда у tokenizer'а нет chat_template (base-модели).
    Используем Gemma-style формат — он совместим с большинством instruct-моделей.
    """
    parts: list[str] = []
    sys_buf = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            sys_buf.append(content)
        elif role == "user":
            user_msg = ("\n\n".join(sys_buf) + "\n\n" + content) if sys_buf else content
            sys_buf = []
            parts.append(f"<start_of_turn>user\n{user_msg}<end_of_turn>\n")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")
    if sys_buf:  # одиночный system без user
        parts.append(f"<start_of_turn>user\n{sys_buf[0]}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def render_chat(llm, messages_or_str):
    tok = llm.get_tokenizer()
    msgs = (messages_or_str if isinstance(messages_or_str, list)
            else [{"role": "user", "content": messages_or_str}])
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except (ValueError, AttributeError):
        # tokenizer.chat_template отсутствует (например, base Gemma-4)
        return _fallback_chat_render(msgs)


def vllm_generate(llm, sp_cls, prompts, max_new_tokens=180,
                  temperature=0.0, top_p=1.0):
    if not prompts: return []
    sp = sp_cls(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, n=1)
    rendered = [render_chat(llm, p) for p in prompts]
    outs = llm.generate(rendered, sp, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outs]


def vllm_sample_n_batch(llm, sp_cls, prompts, n, max_new_tokens=64,
                        temperature=1.0, top_p=1.0):
    if not prompts: return []
    sp = sp_cls(temperature=temperature, top_p=top_p,
                max_tokens=max_new_tokens, n=n)
    rendered = [render_chat(llm, p) for p in prompts]
    outs = llm.generate(rendered, sp, use_tqdm=False)
    return [[c.text.strip() for c in r.outputs] for r in outs]


# =============================================================================
# Cache
# =============================================================================
def _exp_path(cache_dir: Path, method: str, ds_name: str) -> Path:
    p = cache_dir / "llm_outputs"; p.mkdir(parents=True, exist_ok=True)
    return p / f"{method}_{ds_name}.json"


def load_expansions(cache_dir, method, ds_name) -> dict[str, str]:
    p = _exp_path(cache_dir, method, ds_name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        backup = p.with_suffix(p.suffix + ".corrupt")
        p.replace(backup)
        print(f"  [warn] corrupt cache moved to {backup}: {ex}")
        return {}


def save_expansions(cache_dir, method, ds_name, qid2exp) -> None:
    p = _exp_path(cache_dir, method, ds_name)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(qid2exp, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


# =============================================================================
# Method 1 — Query2doc (Wang et al., EMNLP 2023)
# =============================================================================
Q2D_INSTRUCTION = "Напиши абзац, отвечающий на заданный запрос."
Q2D_FEWSHOT = [
    ("симптомы дефицита витамина D у взрослых",
     "Дефицит витамина D у взрослых проявляется болями в мышцах и костях, "
     "хронической усталостью, частыми простудами и снижением плотности костной ткани."),
    ("что такое квантовая запутанность",
     "Квантовая запутанность — явление, при котором состояния двух или более частиц "
     "связаны так, что измерение одной мгновенно определяет состояние другой."),
    ("как работает протокол TLS handshake",
     "TLS handshake устанавливает защищённое соединение: стороны согласуют шифры, "
     "обмениваются сертификатами, выводят сеансовый ключ через ECDHE."),
    ("влияние ферментации на пищевую ценность сои",
     "Ферментация сои разрушает антинутриенты, повышает биодоступность белков, "
     "образует биоактивные пептиды и витамины группы B."),
]


def _q2d_user(query: str) -> str:
    parts = [Q2D_INSTRUCTION, ""]
    for q, p in Q2D_FEWSHOT:
        parts += [f"Запрос: {q}", f"Абзац: {p}", ""]
    parts += [f"Запрос: {query}", "Абзац:"]
    return "\n".join(parts)


def method_q2d(*, qids, qtexts, ds_name, cache_dir, llm, sp_cls, **_):
    cache = load_expansions(cache_dir, "Query2doc", ds_name)
    todo = [(qid, q) for qid, q in zip(qids, qtexts) if qid not in cache]
    if todo:
        print(f"  [Q2D] generating for {len(todo):,} queries ...")
        prompts = [_q2d_user(q) for _, q in todo]
        outs = vllm_generate(llm, sp_cls, prompts, max_new_tokens=128, temperature=1.0)
        for (qid, _), text in zip(todo, outs):
            for stop in ("\nЗапрос:", "\nАбзац:"):
                if stop in text:
                    text = text.split(stop)[0]
            cache[qid] = text.strip()
        save_expansions(cache_dir, "Query2doc", ds_name, cache)
    return [cache[qid] for qid in qids]


# =============================================================================
# Method 2 — PromptPRF (Hang Li et al., 2025)
# =============================================================================
PROMPTPRF_TEMPLATES = {
    "keywords": (
        "Из научного текста выдели 5-10 ключевых терминов, которые "
        "используются в названиях/обзорах публикаций по этой теме.\n\n"
        "Пример.\n"
        "Текст: «Митохондриальный мембранный потенциал поддерживается за счёт "
        "цепи электронного транспорта; нарушения приводят к апоптозу клетки.»\n"
        "Ключевые слова:\n"
        "- митохондриальный мембранный потенциал\n"
        "- цепь электронного транспорта\n"
        "- апоптоз\n"
        "- клеточная биология\n\n"
        "Теперь сделай то же для следующего текста.\n"
        "Текст: {p}\n"
        "Ключевые слова:",
        96, "Ключевые слова"),
    "facts": (
        "Из научного текста извлеки 3-5 фактических утверждений (формулировка "
        "результата или причинно-следственной связи).\n\n"
        "Пример.\n"
        "Текст: «Витамин D повышает всасывание кальция в тонком кишечнике, "
        "что снижает риск переломов у пожилых.»\n"
        "Факты:\n"
        "- Витамин D повышает всасывание кальция в тонком кишечнике.\n"
        "- Достаточный уровень витамина D снижает риск переломов у пожилых.\n\n"
        "Теперь сделай то же для следующего текста.\n"
        "Текст: {p}\n"
        "Факты:",
        320, "Факты"),
    "entities": (
        "Из научного текста выдели именованные сущности (белки, гены, "
        "болезни, химические соединения, виды организмов и т.п.).\n\n"
        "Пример.\n"
        "Текст: «Кишечная палочка E. coli секретирует токсин Stx2, "
        "вызывающий гемолитико-уремический синдром.»\n"
        "Сущности:\n"
        "- E. coli\n"
        "- Stx2\n"
        "- гемолитико-уремический синдром\n\n"
        "Теперь сделай то же для следующего текста.\n"
        "Текст: {p}\n"
        "Сущности:",
        96, "Сущности"),
}


def method_prompt_prf(*, qids, qtexts, ds_name, cache_dir, llm, sp_cls,
                     query_emb, id2text, retriever_initial,
                     feature_type="keywords", prf_depth=PRF_DEPTH, **_):
    # Финальный expansion зависит от prf_depth (другая глубина → другой
    # набор top-K документов → другой текст). Кэш сохраняется отдельно
    # под каждое значение depth, чтобы депт=15 не подхватывал депт=5.
    # При prf_depth == PRF_DEPTH (default из _config.py) используем
    # старое имя кэша для совместимости с уже посчитанным.
    cache_method = ("PromptPRF" if prf_depth == PRF_DEPTH
                    else f"PromptPRF_d{prf_depth}")
    cache = load_expansions(cache_dir, cache_method, ds_name)
    if all(qid in cache for qid in qids):
        return [cache[qid] for qid in qids]

    template, mtok, fname = PROMPTPRF_TEMPLATES[feature_type]
    feat_path = cache_dir / "llm_outputs" / f"PromptPRF_features_{ds_name}_{feature_type}.json"
    if feat_path.exists():
        try:
            feat_cache = json.loads(feat_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as ex:
            backup = feat_path.with_suffix(feat_path.suffix + ".corrupt")
            feat_path.replace(backup)
            print(f"     [warn] corrupt feature cache moved to {backup}: {ex}")
            feat_cache = {}
    else:
        feat_cache = {}

    print(f"  [PromptPRF/{feature_type}] retrieving top-{prf_depth} ...")
    top_per_q = retriever_initial(query_emb, qtexts, prf_depth)

    unique = sorted({d for v in top_per_q.values() for d in v})
    todo = [d for d in unique if d not in feat_cache]
    if todo:
        print(f"     extracting features for {len(todo):,} new docs ...")
        prompts = [template.replace("{p}", id2text.get(d, "")[:1500]) for d in todo]
        outs = vllm_generate(llm, sp_cls, prompts, max_new_tokens=mtok, temperature=0.0)
        for d, txt in zip(todo, outs):
            feat_cache[d] = txt.strip()
        tmp = feat_path.with_suffix(feat_path.suffix + ".tmp")
        tmp.write_text(json.dumps(feat_cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(feat_path)

    for qi, qid in enumerate(qids):
        if qid in cache: continue
        parts = []
        for rank, did in enumerate(top_per_q.get(qi, []), start=1):
            f = feat_cache.get(did, "")
            if f:
                parts.append(f"{fname} для топ-{rank} документа: {f}.")
        cache[qid] = " ".join(parts)
    save_expansions(cache_dir, cache_method, ds_name, cache)
    return [cache[qid] for qid in qids]


# =============================================================================
# Method 3 — PQR (Kang et al., ACL 2025) — query-side adaptation
# =============================================================================
PQR_TEMP = 1.2
PQR_MAX_TOK = 28
PQR_N = 16
PQR_K_RANGE = (2, 3, 4, 5, 6)
PQR_ZS_INSTR = (
    "Перефразируй научный поисковый запрос, сохранив смысл. Используй "
    "синонимичные термины и/или иную формулировку. Выдавай ТОЛЬКО переформулировку.\n\n"
    "Примеры.\n"
    "Запрос: эффект витамина D на плотность костей у пожилых\n"
    "Переформулировка: влияние холекальциферола на минеральную плотность скелета у пожилых пациентов\n\n"
    "Запрос: молекулярные механизмы устойчивости к антибиотикам\n"
    "Переформулировка: биохимические основы резистентности микроорганизмов к антимикробным препаратам"
)

PQR_TOPIC_S1 = (
    "Перечисли через запятую 3-5 ключевых аспектов или подтем научного "
    "поискового запроса. Без пояснений и без нумерации.\n\n"
    "Пример.\n"
    "Запрос: лечение диабета 2 типа\n"
    "Аспекты: метформин, инсулинорезистентность, диета, бариатрическая хирургия, GLP-1 агонисты"
)

PQR_TOPIC_S2 = (
    "Примеры.\n"
    "Аспект «инсулинорезистентность», контекст «лечение диабета 2 типа» → "
    "запрос: инсулинорезистентность при диабете 2 типа: механизмы и терапия\n"
    "Аспект «метформин», контекст «лечение диабета 2 типа» → "
    "запрос: эффективность и побочные эффекты метформина при диабете 2 типа\n\n"
    "Сформулируй короткий научный поисковый запрос об аспекте «{TOPIC}» "
    "в контексте: «{QUERY}». Выдавай ТОЛЬКО запрос."
)


def _pqr_fit_gmm(emb: np.ndarray, k_range=PQR_K_RANGE):
    from sklearn.mixture import GaussianMixture
    n = emb.shape[0]
    best, best_bic = None, float("inf")
    for K in k_range:
        if K >= n: break
        try:
            gm = GaussianMixture(n_components=K, covariance_type="diag",
                                 max_iter=100, n_init=2,
                                 random_state=0, reg_covar=1e-4)
            gm.fit(emb); b = gm.bic(emb)
            if b < best_bic:
                best_bic, best = b, gm
        except Exception:
            continue
    if best is None:
        gm = GaussianMixture(n_components=1, covariance_type="diag",
                             max_iter=100, random_state=0, reg_covar=1e-4)
        gm.fit(emb); return gm
    return best


def method_pqr(*, qids, qtexts, ds_name, cache_dir, llm, sp_cls, encoder, **_):
    cache = load_expansions(cache_dir, "PQR", ds_name)
    todo = [(qid, q) for qid, q in zip(qids, qtexts) if qid not in cache]
    if not todo:
        return [cache[qid] for qid in qids]
    print(f"  [PQR] sampling for {len(todo):,} queries ...")

    zs_prompts = [
        [{"role": "system", "content": PQR_ZS_INSTR},
         {"role": "user",   "content": q}] for _, q in todo
    ]
    zs_per_q = vllm_sample_n_batch(llm, sp_cls, zs_prompts, n=PQR_N,
                                   max_new_tokens=PQR_MAX_TOK,
                                   temperature=PQR_TEMP, top_p=1.0)

    topic_prompts = [
        [{"role": "system", "content": PQR_TOPIC_S1},
         {"role": "user",   "content": q}] for _, q in todo
    ]
    topics_raw = vllm_generate(llm, sp_cls, topic_prompts, max_new_tokens=64,
                               temperature=0.0)

    s2_prompts, s2_route = [], []
    for qi, ((_, q), raw) in enumerate(zip(todo, topics_raw)):
        topics = [t.strip(" *•-—\t\"'.") for t in re.split(r"[,;\n]+", raw)]
        topics = [t for t in topics if 1 <= len(t.split()) <= 6][:5]
        for topic in topics:
            prompt = (PQR_TOPIC_S2.replace("{TOPIC}", topic).replace("{QUERY}", q))
            s2_prompts.append([{"role": "user", "content": prompt}])
            s2_route.append(qi)
    s2_n = max(2, PQR_N // 5)
    s2_per_p = (vllm_sample_n_batch(llm, sp_cls, s2_prompts, n=s2_n,
                                    max_new_tokens=PQR_MAX_TOK,
                                    temperature=PQR_TEMP, top_p=1.0)
                if s2_prompts else [])
    ta_per_q: dict[int, list[str]] = {qi: [] for qi in range(len(todo))}
    for qi, samples in zip(s2_route, s2_per_p):
        ta_per_q[qi].extend(s for s in samples if s)

    print("  [PQR] embedding samples + GMM ...")
    for qi, (qid, _) in enumerate(tqdm(todo, desc="  PQR-GMM")):
        all_samp = [s for s in (list(zs_per_q[qi]) + ta_per_q.get(qi, [])) if s]
        if len(all_samp) < 2:
            cache[qid] = qtexts[qids.index(qid)]
            continue
        emb = encoder.encode_queries(all_samp).astype(np.float32)
        gm = _pqr_fit_gmm(emb)
        means = gm.means_.astype(np.float32)
        means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
        sim = emb @ means.T
        rep = sim.argmax(axis=0)
        cache[qid] = " ".join(all_samp[i] for i in rep)
    save_expansions(cache_dir, "PQR", ds_name, cache)
    return [cache[qid] for qid in qids]


# =============================================================================
# Method 4 — Word2Passage (Choi et al., 2025)
# =============================================================================
W2P_QUERY_TYPES = ("description", "person", "entity", "numeric", "location")
W2P_SIGNIFICANCE = {
    "description": (0.72, 0.57, 0.97), "entity": (0.50, 0.73, 0.48),
    "person": (1.10, 1.08, 0.70), "numeric": (0.78, 1.15, 0.83),
    "location": (1.00, 1.00, 0.73),
}
W2P_DEFAULT_SIG = (1.0, 1.0, 1.0)
W2P_GEN_PROMPT = (
    "Сгенерируй абзац, предложение и список слов, отвечающих на научный ЗАПРОС. "
    "Термины, важные для ответа, должны встречаться во ВСЕХ трёх частях.\n"
    "### Определения:\n"
    "**passage**: информативный связный абзац (3-4 предложения).\n"
    "**sentence**: одно знаниебогатое предложение.\n"
    "**word**: список 5-10 ключевых терминов.\n"
    "### Формат ответа: ТОЛЬКО валидный JSON, без пояснений.\n\n"
    "### Пример.\n"
    "ЗАПРОС: влияние омега-3 жирных кислот на сердечно-сосудистые заболевания\n"
    "ОТВЕТ:\n"
    '{"passage": "Омега-3 полиненасыщенные жирные кислоты (ЭПК и ДГК) снижают уровень триглицеридов '
    'в плазме и подавляют системное воспаление. Метаанализы показывают умеренное снижение риска '
    'сердечно-сосудистых событий при дозах 1-2 г/сутки. Эффект сильнее у пациентов с гипертриглицеридемией.", '
    '"sentence": "Длительный приём омега-3 жирных кислот ассоциирован со снижением риска '
    'инфаркта миокарда и общей сердечно-сосудистой смертности.", '
    '"word": ["омега-3", "ЭПК", "ДГК", "триглицериды", "сердечно-сосудистые заболевания", '
    '"инфаркт миокарда", "воспаление", "полиненасыщенные жирные кислоты"]}\n\n'
    "### ЗАПРОС:\n{QUERY}\n"
    "### ОТВЕТ:"
)

W2P_TYPE_PROMPT = (
    "Классифицируй научный поисковый запрос в одну из 5 категорий. Возвращай "
    "только название категории без пояснений.\n\n"
    "Категории и примеры:\n\n"
    "**description** — запрос о механизме / процессе / описании:\n"
    "  - причины воспаления тазовой области\n"
    "  - механизм действия аспирина\n"
    "  - как работает иммунная система\n\n"
    "**numeric** — запрос о численном значении / измерении:\n"
    "  - средняя зарплата консультанта\n"
    "  - дозировка витамина D у взрослых\n"
    "  - период полураспада йода-131\n\n"
    "**location** — запрос о местоположении:\n"
    "  - какой самый большой континент\n"
    "  - где обитают амурские тигры\n"
    "  - географическое распределение малярии\n\n"
    "**entity** — запрос об объекте / классе объектов / списке:\n"
    "  - какие растения растут в Орегоне\n"
    "  - типы лейкоцитов\n"
    "  - белки теплового шока человека\n\n"
    "**person** — запрос о конкретном человеке / коллективе:\n"
    "  - актёрский состав фильма «Интерстеллар»\n"
    "  - кто открыл пенициллин\n"
    "  - первооткрыватель структуры ДНК\n\n"
    "Классифицируй: [description, numeric, location, entity, person].\n"
    "Запрос: {QUERY}\n"
    "Тип запроса:"
)
_W2P_TOK = re.compile(r"[\w]+", re.UNICODE)


def _w2p_tok(t: str) -> list[str]:
    return [s.lower() for s in _W2P_TOK.findall(t or "")]


def _w2p_parse_json(raw: str) -> dict:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m: s = m.group(0)
    try:
        d = json.loads(s)
    except Exception:
        d = {}
    return {
        "passage": str(d.get("passage", "") or ""),
        "sentence": str(d.get("sentence", "") or ""),
        "word": [str(w) for w in (d.get("word", []) or []) if w],
    }


def _w2p_avg_unique(corpus_texts, sample_n=2000):
    txts = corpus_texts[:sample_n]
    return float(np.mean([len(set(_w2p_tok(t))) for t in txts])) if txts else 1.0


def _w2p_suffix(n_refs: int, alpha: float, repeat_scale: int) -> str:
    """Suffix для разделения W2P-кэшей под разные гиперпараметры.

    Возвращает '' для paper-defaults (n_refs=3, alpha=1.0, repeat_scale=5)
    — backward-compat с уже посчитанными кэшами.
    Иначе '_n8' / '_a0.5_r3' и т.п.
    """
    parts = []
    if n_refs != 3: parts.append(f"n{n_refs}")
    if alpha != 1.0: parts.append(f"a{alpha:g}")
    if repeat_scale != 5: parts.append(f"r{repeat_scale}")
    return ("_" + "_".join(parts)) if parts else ""


def _w2p_weights_path(cache_dir: Path, ds_name: str, suffix: str = "") -> Path:
    p = cache_dir / "llm_outputs"; p.mkdir(parents=True, exist_ok=True)
    return p / f"Word2Passage_weights{suffix}_{ds_name}.json"


def load_w2p_weights(cache_dir: Path, ds_name: str,
                     suffix: str = "") -> dict[str, dict[str, float]]:
    """qid -> {term: weight}. Используется BM25WeightedRetriever (paper-faithful Choi 2025).
    `suffix` — разделение под разные (n_refs, alpha, repeat_scale)."""
    p = _w2p_weights_path(cache_dir, ds_name, suffix=suffix)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def method_word2passage(*, qids, qtexts, ds_name, cache_dir, llm, sp_cls,
                       corpus_texts, n_refs=3, alpha=1.0, repeat_scale=5, **_):
    suffix = _w2p_suffix(n_refs, alpha, repeat_scale)
    cache_method = "Word2Passage" + suffix
    cache = load_expansions(cache_dir, cache_method, ds_name)
    weights_cache: dict[str, dict[str, float]] = load_w2p_weights(
        cache_dir, ds_name, suffix=suffix)
    todo = [(qid, q) for qid, q in zip(qids, qtexts) if qid not in cache]
    if not todo:
        return [cache[qid] for qid in qids]
    print(f"  [W2P] generating refs for {len(todo):,} queries ...")
    W = _w2p_avg_unique(corpus_texts)

    prompts = [
        [{"role": "user", "content": W2P_GEN_PROMPT.replace("{QUERY}", q)}]
        for _, q in todo
    ]
    refs_raw = vllm_sample_n_batch(llm, sp_cls, prompts, n=n_refs,
                                   max_new_tokens=512, temperature=0.7, top_p=0.9)
    type_prompts = [
        [{"role": "user", "content": W2P_TYPE_PROMPT.replace("{QUERY}", q)}]
        for _, q in todo
    ]
    type_raw = vllm_generate(llm, sp_cls, type_prompts, max_new_tokens=20, temperature=0.0)

    for (qid, q), raw_refs, raw_type in zip(todo, refs_raw, type_raw):
        refs = [_w2p_parse_json(r) for r in raw_refs]
        s_lower = (raw_type or "").lower()
        q_type = next((c for c in W2P_QUERY_TYPES if c in s_lower), "description")
        I_qw, I_qs, I_qp = W2P_SIGNIFICANCE.get(q_type, W2P_DEFAULT_SIG)

        word_R, word_R_freq = {}, {}
        for r in refs:
            cw = Counter(_w2p_tok(" ".join(r["word"])))
            cs = Counter(_w2p_tok(r["sentence"]))
            cp = Counter(_w2p_tok(r["passage"]))
            for t in set(cw) | set(cs) | set(cp):
                score = (I_qw * cw.get(t, 0) + I_qs * cs.get(t, 0) + I_qp * cp.get(t, 0))
                word_R[t] = word_R.get(t, 0.0) + score
                word_R_freq[t] = word_R_freq.get(t, 0) + cw[t] + cs[t] + cp[t]
        if W > 0:
            scale_R = alpha / math.sqrt(W)
            word_R = {t: s * scale_R for t, s in word_R.items()}

        q_freq = Counter(_w2p_tok(q))
        sumF_R, sumF_Q = sum(word_R_freq.values()), sum(q_freq.values())
        if sumF_Q == 0:
            word_Q = {}
        else:
            norm = math.sqrt(sumF_R) / math.sqrt(sumF_Q) if sumF_R > 0 else 1.0
            word_Q = {t: norm * q_freq[t] for t in q_freq}

        weights = dict(word_R)
        for t, v in word_Q.items():
            weights[t] = weights.get(t, 0.0) + v

        if not weights:
            cache[qid] = ""
            weights_cache[qid] = {}
        else:
            items = sorted(weights.items(), key=lambda kv: -kv[1])[:200]
            max_I = items[0][1] or 1.0
            parts = []
            for t, I in items:
                n = max(1, int(round((I / max_I) * repeat_scale)))
                parts.extend([t] * n)
            cache[qid] = " ".join(parts)
            weights_cache[qid] = {t: float(w) for t, w in items}

    save_expansions(cache_dir, cache_method, ds_name, cache)
    wpath = _w2p_weights_path(cache_dir, ds_name, suffix=suffix)
    tmp = wpath.with_suffix(wpath.suffix + ".tmp")
    tmp.write_text(json.dumps(weights_cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(wpath)
    return [cache[qid] for qid in qids]


# =============================================================================
# Aligners
# =============================================================================
def align_none(queries, expansions, **_):
    return [(q + " " + e).strip() for q, e in zip(queries, expansions)]


CSQE_ONE_SHOT = (
    "Запрос: «как некоторые акулы являются теплокровными»\n"
    "Найденные документы:\n"
    "1. Большинство акул холоднокровные. Некоторые, такие как мако и большая "
    "белая акула, частично теплокровны. Лососёвая акула — теплокровная.\n"
    "2. Холоднокровные ли акулы или теплокровные? Акулы по большей части "
    "эффективные эктотермные хищники.\n"
    "3. Большие белые акулы — одни из немногих теплокровных акул.\n"
    "4. Лососёвые акулы могут поднимать температуру тела на 20 градусов выше "
    "окружающей воды.\n\n"
    "Сначала проанализируй найденные документы и определи те, которые хотя бы "
    "частично релевантны запросу. Затем извлеки из каждого релевантного документа "
    "ключевые предложения.\n\n"
    "На основе запроса «как некоторые акулы являются теплокровными» я "
    "проанализировал документы. Вот релевантные документы и ключевые предложения:\n"
    "Документ 1:\n«Большинство акул холоднокровные.»\n«Лососёвая акула — теплокровная.»\n"
    "Документ 3:\n«Большие белые акулы — одни из немногих теплокровных акул.»\n"
    "Документ 4:\n«Лососёвые акулы могут поднимать температуру тела на 20 градусов "
    "выше окружающей воды.»"
)
CSQE_INSTR = (
    "Сначала проанализируй найденные документы и определи те, которые хотя бы "
    "частично релевантны запросу. Затем извлеки из каждого релевантного документа "
    "ключевые предложения, которые делают его релевантным."
)
_CSQE_QUOTE_RE = re.compile(r"[«\"](.+?)[»\"]", re.DOTALL)


def _csqe_extract(raw: str) -> list[str]:
    if not raw: return []
    quoted = _CSQE_QUOTE_RE.findall(raw)
    out = [s.strip() for s in quoted if len(s.strip()) >= 5]
    if out: return out
    return [line.strip(" \t-—•«»\"")
            for line in raw.split("\n")
            if line.strip()
            and not line.strip().lower().startswith("документ")
            and len(line.strip()) >= 5]


def align_csqe(queries, expansions, *, encoder, retriever_initial, id2text,
              llm, sp_cls, depth=10, n_samples=2, doc_truncate=128 * 6, **_):
    base = [(q + " " + e).strip() for q, e in zip(queries, expansions)]
    base_emb = encoder.encode_queries(base)
    top_per_q = retriever_initial(base_emb, base, depth)

    prompts = []
    for qi in range(len(queries)):
        retrieved = "\n".join(
            f"{j+1}. {id2text.get(did, '')[:doc_truncate]}"
            for j, did in enumerate(top_per_q.get(qi, []))
        )
        user = (f"{CSQE_ONE_SHOT}\n\n"
                f"Запрос: «{queries[qi]}»\n"
                f"Найденные документы:\n{retrieved}\n\n{CSQE_INSTR}")
        prompts.append([{"role": "user", "content": user}])

    print(f"  [CSQE] {n_samples} samples × {len(prompts):,} queries (prefix-cache)")
    raws_per_q = vllm_sample_n_batch(llm, sp_cls, prompts, n=n_samples,
                                     max_new_tokens=384, temperature=1.0, top_p=1.0)
    out = []
    for qi in range(len(queries)):
        sentences = []
        for s in raws_per_q[qi]:
            sentences.extend(_csqe_extract(s))
        seen = set(); uniq = []
        for s in sentences:
            k = s.lower()[:80]
            if k not in seen:
                seen.add(k); uniq.append(s)
        out.append((queries[qi] + " " + " ".join(uniq) + " "
                    + (expansions[qi] or "")).strip())
    return out


def align_aqe(queries, expansions, *, encoder, corpus_emb,
             keep_frac=0.6, min_kept=1, **_):
    all_sents, owners = [], []
    splitter = re.compile(r"(?<=[.!?])\s+|[\n;]+")
    for qi, e in enumerate(expansions):
        sents = splitter.split(e or "")
        sents = [s.strip(" ,.;-—•\t") for s in sents if len(s.strip()) >= 3]
        for s in sents:
            all_sents.append(s); owners.append(qi)
    out = list(queries)
    if not all_sents:
        return out
    sent_emb = encoder.encode_queries(all_sents).astype(np.float32)
    _, top_scores = chunked_retrieve(sent_emb, corpus_emb, top_k=1)
    scores = top_scores[:, 0]
    by_q: dict[int, list[tuple[str, float]]] = {}
    for s, o, sc in zip(all_sents, owners, scores):
        by_q.setdefault(o, []).append((s, float(sc)))
    for qi in range(len(queries)):
        items = by_q.get(qi, [])
        if not items: continue
        items.sort(key=lambda x: -x[1])
        n_keep = max(min_kept, int(round(len(items) * keep_frac)))
        kept = [s for s, _ in items[:n_keep]]
        out[qi] = (queries[qi] + " " + " ".join(kept)).strip()
    return out


METHODS: dict[str, Callable] = {
    "Query2doc":    method_q2d,
    "PromptPRF":    method_prompt_prf,
    "PQR":          method_pqr,
    "Word2Passage": method_word2passage,
}
ALIGNERS: dict[str, Callable] = {
    "none": align_none,
    "CSQE": align_csqe,
    "AQE":  align_aqe,
}


# =============================================================================
# Main
# =============================================================================
def parse_rerank_passes(spec: str | None) -> list[str | None]:
    """'none,bge,./reranker_ft' → [None, 'bge', './reranker_ft'].

    Пустая/None строка → [None] (один прогон без rerank'а).
    'none' (regardless of case) превращается в Python None.
    """
    if not spec:
        return [None]
    out: list[str | None] = []
    seen: set[str] = set()
    for s in spec.split(","):
        s = s.strip()
        if not s:
            continue
        key = s.lower() if s.lower() == "none" else s
        if key in seen:
            continue
        seen.add(key)
        out.append(None if s.lower() == "none" else s)
    return out or [None]


def _rerank_tag(model_path: str | None) -> str:
    """Имя для CSV-тегирования (короткое, читаемое)."""
    return "none" if model_path is None else model_path


def main(datasets: list[str] | None = None, cache_dir: str | Path | None = None,
         llm_model: str = LLM_MODEL_NAME, retriever_kind: str | None = None,
         qrels_split: str | None = None, k: int = QE_K,
         rerank_models: list[str | None] | None = None,
         rerank_top_n: int = 100,
         methods: list[str] | None = None,
         aligners: list[str] | None = None,
         prf_depth: int | None = None,
         w2p_refs: int | None = None,
         w2p_alpha: float | None = None,
         w2p_repeat_scale: int | None = None,
         # Legacy single-shot args (backward-compat):
         rerank: bool = False,
         reranker_model: str = "BAAI/bge-reranker-v2-m3") -> dict:
    cache_dir = get_cache_dir(cache_dir)
    setup_log_file(cache_dir, "stage06_qe_12combos")
    datasets = datasets or list(DATASETS.keys())

    # Resolve rerank passes: новый rerank_models имеет приоритет над legacy.
    if rerank_models is None:
        rerank_models = [reranker_model] if rerank else [None]
    elif not rerank_models:
        rerank_models = [None]
    print(f"[rerank passes] {[_rerank_tag(m) for m in rerank_models]}")

    # Filter methods / aligners (по умолчанию — все)
    methods_to_run = list(METHODS.keys()) if not methods else [
        m for m in methods if m in METHODS
    ]
    aligners_to_run = list(ALIGNERS.keys()) if not aligners else [
        a for a in aligners if a in ALIGNERS
    ]

    def _method_tag(m_name: str) -> str:
        """Имя метода с annotation для non-default гиперпараметров.
        Используется в combination tag и dedup-ключе, чтобы прогоны с разными
        --prf-depth / --w2p-refs не перетирали друг друга в CSV."""
        if m_name == "PromptPRF" and prf_depth is not None and prf_depth != PRF_DEPTH:
            return f"PromptPRF[d={prf_depth}]"
        if m_name == "Word2Passage":
            suf = _w2p_suffix(
                n_refs=w2p_refs if w2p_refs is not None else 3,
                alpha=w2p_alpha if w2p_alpha is not None else 1.0,
                repeat_scale=w2p_repeat_scale if w2p_repeat_scale is not None else 5,
            )
            if suf:
                return f"Word2Passage[{suf.lstrip('_')}]"
        return m_name
    if not methods_to_run:
        raise SystemExit(f"--methods не дал валидных значений; известны: {list(METHODS.keys())}")
    if not aligners_to_run:
        raise SystemExit(f"--aligners не дал валидных значений; известны: {list(ALIGNERS.keys())}")
    print(f"[methods] {methods_to_run}")
    print(f"[aligners] {aligners_to_run}")
    if prf_depth is not None:
        print(f"[prf_depth] {prf_depth} (override)")
    if any(v is not None for v in (w2p_refs, w2p_alpha, w2p_repeat_scale)):
        print(f"[w2p] refs={w2p_refs} alpha={w2p_alpha} repeat_scale={w2p_repeat_scale} (overrides)")

    manifest_emb = read_state(cache_dir, "best_embedding_manifest")
    manifest_faiss = read_state(cache_dir, "faiss_manifest")
    if manifest_emb is None or manifest_faiss is None:
        raise SystemExit("Запусти Stages 03 + 04 сначала (нужны эмбеддинги и FAISS).")

    # Ретривер всегда dense (Stage 05 удалён из пайплайна).
    # Параметр retriever_kind оставлен для совместимости, но игнорируется
    # за исключением явных значений "bm25" / "hybrid_rrf" (для ablation).
    if retriever_kind is None:
        retriever_kind = "dense"
    print(f"[retriever] {retriever_kind}")
    print(f"[embedding] {manifest_emb['model']}")
    print(f"[llm] {llm_model}")

    llm, sp_cls = make_vllm(llm_model)
    model_info = MODELS_BY_NAME[manifest_emb["model"]]
    encoder = STEncoder(model_info, device="cuda:0", fp16=True)

    # Lazy reranker cache: загружаем только когда впервые встречается имя.
    any_rerank = any(m is not None for m in rerank_models)
    rerankers: dict[str, RerankerWrapper] = {}
    def _get_reranker(name: str) -> RerankerWrapper:
        if name not in rerankers:
            rerankers[name] = RerankerWrapper(name)
        return rerankers[name]

    res_dir = cache_dir / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    res_csv = res_dir / "qe_12combos.csv"
    rows = pd.read_csv(res_csv).to_dict("records") if res_csv.exists() else []
    # Backward-compat: старые строки без полей reranker/retriever помечаем
    # дефолтами. retriever="unknown" — чтобы старые строки не блокировали
    # новые прогоны на другом ретривере (dedup ключ их различает).
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
        # qtexts с тем же text_field, что и manifest. Иначе LLM-промпты получают
        # лемматизированный query (default "processed") при том, что corpus_texts
        # и encoder уже на raw — рассинхрон ломал и LLM-генерации, и dense-encode.
        ds_text_field = manifest_emb.get("text_field", "raw")
        qids, qtexts = get_query_texts_ids(d, text_field=ds_text_field)
        qemb = np.load(manifest_emb["datasets"][ds_name]["qemb"])

        # доменные мапы для CSQE/PromptPRF
        id2text = {cid: ctx for cid, ctx in zip(corpus_ids, corpus_texts)}

        # BM25 ВСЕГДА индексирует processed_text — независимо от --text-field
        # пайплайна. corpus_texts из meta может быть raw (если Stage 03 запущен
        # с text_field=raw) — для BM25 это сломает матчинг лемм.
        bm25_corpus_texts = corpus_texts
        if retriever_kind in ("bm25", "hybrid_rrf"):
            cids_proc, ctexts_proc = get_corpus_texts_ids(d, text_field="processed")
            assert cids_proc == corpus_ids, "corpus order mismatch raw vs processed"
            bm25_corpus_texts = ctexts_proc

        retriever_initial = make_retriever_callable(
            retriever_kind, corpus_ids, bm25_corpus_texts,
            manifest_faiss["datasets"][ds_name]["index"],
            cache_dir=cache_dir, ds_name=ds_name,
        )
        retriever_final = retriever_initial  # тот же

        # RAW text для cross-encoder rerank'а — отдельная мапа, т.к. corpus_texts
        # может быть processed (lemmatized) если pipeline запущен с text-field=processed.
        id2text_raw: dict[str, str] = {}
        qtexts_raw: list[str] = []
        if any_rerank:
            cids_r, ctexts_r = get_corpus_texts_ids(d, text_field="raw")
            id2text_raw = {cid: ct for cid, ct in zip(cids_r, ctexts_r)}
            _, qtexts_raw = get_query_texts_ids(d, text_field="raw")

        # === методы ===
        method_kwargs = dict(
            qids=qids, qtexts=qtexts, ds_name=ds_name, cache_dir=cache_dir,
            llm=llm, sp_cls=sp_cls,
            query_emb=qemb, corpus_texts=corpus_texts, corpus_ids=corpus_ids,
            id2text=id2text, retriever_initial=retriever_initial, encoder=encoder,
        )
        method_expansions: dict[str, list[str]] = {}
        method_times: dict[str, float] = {}
        for m_name in methods_to_run:
            m_fn = METHODS[m_name]
            print(f"\n  >> method: {m_name}")
            t_m = time.time()
            try:
                # Method-specific overrides из CLI (--prf-depth / --w2p-*)
                kw = dict(method_kwargs)
                if m_name == "PromptPRF" and prf_depth is not None:
                    kw["prf_depth"] = prf_depth
                if m_name == "Word2Passage":
                    if w2p_refs is not None: kw["n_refs"] = w2p_refs
                    if w2p_alpha is not None: kw["alpha"] = w2p_alpha
                    if w2p_repeat_scale is not None: kw["repeat_scale"] = w2p_repeat_scale
                method_expansions[m_name] = m_fn(**kw)
            except Exception as ex:
                print(f"     [ERR] {m_name}: {ex}")
                import traceback; traceback.print_exc()
                method_expansions[m_name] = list(qtexts)
            method_times[m_name] = time.time() - t_m
            print(f"     [{m_name}] generation: {method_times[m_name]:.1f}s")

        # === aligners + retrieval + scoring ===
        align_kwargs = dict(
            encoder=encoder, retriever_initial=retriever_initial,
            id2text=id2text, corpus_emb=corpus_emb,
            llm=llm, sp_cls=sp_cls,
        )

        # Lazy init для Phase 1.5b (W2P paper-faithful weighted BM25).
        bm25_weighted: BM25WeightedRetriever | None = None

        for m_name, expansions in method_expansions.items():
            for a_name in aligners_to_run:
                a_fn = ALIGNERS[a_name]
                tag = f"{_method_tag(m_name)} + {a_name}"
                # Какие rerank-проходы ещё надо посчитать?
                pending_passes = [
                    rm for rm in rerank_models
                    if (ds_name, tag, _rerank_tag(rm), retriever_kind) not in done
                ]
                if not pending_passes:
                    print(f"  [skip] {tag} (все rerank-проходы кэшированы)")
                    continue
                print(f"\n  -- {tag}  rerank: {[_rerank_tag(rm) for rm in pending_passes]}")
                eval_top_k = max(k, 10)
                # wide_n: достаточно для самого «глубокого» rerank-прохода.
                wide_n = max(rerank_top_n, eval_top_k) if any(p is not None for p in pending_passes) else eval_top_k

                # === Базовый retrieval — считается один раз для всех rerank-проходов ===
                t_a = time.time()
                base_failed = False
                ranked_wide: dict[int, list[str]] = {}
                try:
                    final_q = a_fn(qtexts, expansions, **align_kwargs)
                    use_w2p_weighted = (
                        m_name == "Word2Passage" and a_name == "none"
                        and retriever_kind == "bm25"
                    )
                    if use_w2p_weighted:
                        if bm25_weighted is None:
                            bm_texts = bm25_corpus_texts
                            if retriever_kind not in ("bm25", "hybrid_rrf"):
                                _, bm_texts = get_corpus_texts_ids(d, text_field="processed")
                            bm25_weighted = BM25WeightedRetriever(
                                bm_texts, corpus_ids,
                                cache_dir=cache_dir, ds_name=ds_name,
                            )
                        # suffix должен совпасть с тем, что method_word2passage
                        # сохранила (под текущие refs/alpha/scale).
                        w2p_suf = _w2p_suffix(
                            n_refs=w2p_refs if w2p_refs is not None else 3,
                            alpha=w2p_alpha if w2p_alpha is not None else 1.0,
                            repeat_scale=w2p_repeat_scale if w2p_repeat_scale is not None else 5,
                        )
                        weights_per_qid = load_w2p_weights(cache_dir, ds_name, suffix=w2p_suf)
                        weights_list = [weights_per_qid.get(qid, {}) for qid in qids]
                        ranked_wide = bm25_weighted.search_weighted(weights_list, top_k=wide_n)
                    else:
                        # Phase 1.5a: Q2D q×5 repetition для BM25 (Wang 2023).
                        final_q_use = final_q
                        if m_name == "Query2doc" and retriever_kind == "bm25":
                            final_q_use = [(q + " ") * 4 + fq
                                           for q, fq in zip(qtexts, final_q)]
                        f_emb = encoder.encode_queries(final_q_use)
                        ranked_wide = retriever_final(f_emb, final_q_use, wide_n)
                except Exception as ex:
                    print(f"     [ERR base] {tag}: {ex}")
                    import traceback; traceback.print_exc()
                    base_failed = True
                base_time = time.time() - t_a

                # === Rerank-проходы поверх базового ranked_wide ===
                m_time = method_times.get(m_name, 0.0)
                for rm in pending_passes:
                    tag_rm = _rerank_tag(rm)
                    t_rr = time.time()
                    if base_failed:
                        metrics = {"recall@5": None, "map@5": None, "ndcg@5": None,
                                   "recall@10": None, "map@10": None, "ndcg@10": None,
                                   "n_eval": 0}
                        rerank_time = 0.0
                    else:
                        try:
                            if rm is None:
                                ranked = {qi: ranked_wide.get(qi, [])[:eval_top_k]
                                          for qi in range(len(qids))}
                            else:
                                rr = _get_reranker(rm)
                                ranked = rr.rerank(qtexts_raw, ranked_wide,
                                                   id2text_raw, eval_top_k)
                            ret_per_q = {qids[qi]: ranked.get(qi, [])
                                         for qi in range(len(qids))}
                            ret_path = ret_dir / f"{m_name}_{a_name}_{ds_name}__{re.sub(r'[^A-Za-z0-9_-]', '_', tag_rm)}.json"
                            ret_path.write_text(
                                json.dumps(ret_per_q, ensure_ascii=False), encoding="utf-8")
                            metrics = evaluate_run(ret_per_q, d["qrels"], ks=(5, 10))
                            rerank_time = time.time() - t_rr
                            print(f"     -> [rerank={tag_rm}] {metrics}  | base={base_time:.1f}s rerank={rerank_time:.1f}s")
                        except Exception as ex:
                            print(f"     [ERR rerank={tag_rm}] {tag}: {ex}")
                            import traceback; traceback.print_exc()
                            metrics = {"recall@5": None, "map@5": None, "ndcg@5": None,
                                       "recall@10": None, "map@10": None, "ndcg@10": None,
                                       "n_eval": 0}
                            rerank_time = time.time() - t_rr
                    rows.append({
                        "dataset": ds_name, "method": m_name, "aligner": a_name,
                        "combination": tag, **metrics,
                        "retriever": retriever_kind,
                        "reranker": tag_rm,
                        "rerank_top_n": rerank_top_n if rm is not None else None,
                        "method_time_sec": round(m_time, 2),
                        "align_time_sec": round(base_time, 2),
                        "rerank_time_sec": round(rerank_time, 2),
                        "time_sec": round(m_time + base_time + rerank_time, 2),
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
    p.add_argument("--rerank-models", default=None,
                   help="Comma-separated rerank passes: 'none,BAAI/bge-reranker-v2-m3,"
                        "./reranker_scifact_finetuned'. 'none' = без rerank'а. "
                        "Default empty = один прогон без rerank'а.")
    p.add_argument("--rerank-top-n", type=int, default=100,
                   help="Сколько кандидатов из base retriever подавать на rerank")
    p.add_argument("--methods", default=None,
                   help="Comma-separated subset из {Query2doc,PromptPRF,PQR,Word2Passage}; "
                        "default = все 4")
    p.add_argument("--aligners", default=None,
                   help="Comma-separated subset из {none,CSQE,AQE}; default = все 3")
    p.add_argument("--prf-depth", type=int, default=None,
                   help="PromptPRF: глубина PRF top-K (override _config.PRF_DEPTH). "
                        "Кэш expansion'ов хранится отдельно под каждое значение depth.")
    p.add_argument("--w2p-refs", type=int, default=None,
                   help="Word2Passage: число LLM samples (default 3). "
                        "8 даёт лучше term coverage, незначительно медленнее.")
    p.add_argument("--w2p-alpha", type=float, default=None,
                   help="Word2Passage: scale_R множитель для reference-частей (default 1.0).")
    p.add_argument("--w2p-repeat-scale", type=int, default=None,
                   help="Word2Passage: max повторов терма в text-expansion (default 5).")
    # Legacy single-shot args (backward-compat):
    p.add_argument("--rerank", action="store_true",
                   help="[deprecated] alias for --rerank-models <reranker-model>")
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3",
                   help="[deprecated] used only with --rerank")
    args = p.parse_args()
    main(
        datasets=parse_datasets(args.datasets),
        cache_dir=args.cache_dir,
        llm_model=args.llm_model,
        retriever_kind=args.retriever,
        qrels_split=args.qrels_split,
        k=args.k,
        rerank_models=parse_rerank_passes(args.rerank_models)
                       if args.rerank_models else None,
        rerank_top_n=args.rerank_top_n,
        methods=[s.strip() for s in args.methods.split(",") if s.strip()]
                 if args.methods else None,
        aligners=[s.strip() for s in args.aligners.split(",") if s.strip()]
                  if args.aligners else None,
        prf_depth=args.prf_depth,
        w2p_refs=args.w2p_refs,
        w2p_alpha=args.w2p_alpha,
        w2p_repeat_scale=args.w2p_repeat_scale,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
    )
