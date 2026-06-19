"""Search wrapper over OpenAlex `?search=` with BM25-like relevance ranking."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..openalex.client import OpenAlexClient

log = logging.getLogger(__name__)

SEARCH_SELECT = (
    "id,doi,title,publication_year,publication_date,language,"
    "cited_by_count,authorships,primary_topic,best_oa_location,open_access,"
    "relevance_score,abstract_inverted_index"
)


@dataclass
class SearchResultItem:
    openalex_id: str
    title: str
    abstract_snippet: str | None
    publication_year: int | None
    language: str | None
    cited_by_count: int
    authors: list[str]
    primary_topic: dict[str, Any]
    open_access: dict[str, Any]
    relevance_score: float | None


@dataclass
class SearchResult:
    query: str
    lang: str
    total: int
    items: list[SearchResultItem]
    pipeline: str  # informational, e.g. "openalex_search_bm25"
    notes: str  # human-readable summary for the "What we did" panel


def _abstract_snippet(idx: dict[str, list[int]] | None, max_chars: int = 280) -> str | None:
    if not idx:
        return None
    positions: list[tuple[int, str]] = []
    for word, poses in idx.items():
        for p in poses:
            positions.append((p, word))
    if not positions:
        return None
    positions.sort(key=lambda x: x[0])
    text = " ".join(w for _, w in positions)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last = cut.rfind(" ")
    if last > 100:
        cut = cut[:last]
    return cut + "…"


def _project(w: dict[str, Any]) -> SearchResultItem:
    auths: list[str] = []
    for a in (w.get("authorships") or [])[:6]:
        au = a.get("author") or {}
        if au.get("display_name"):
            auths.append(au["display_name"])
    pt = w.get("primary_topic") or {}
    sub = (pt.get("subfield") or {}) or {}
    fld = (pt.get("field") or {}) or {}
    dom = (pt.get("domain") or {}) or {}
    oa_loc = w.get("best_oa_location") or {}
    oa = w.get("open_access") or {}
    short_id = (w.get("id") or "").rsplit("/", 1)[-1]
    return SearchResultItem(
        openalex_id=short_id,
        title=w.get("title") or "(no title)",
        abstract_snippet=_abstract_snippet(w.get("abstract_inverted_index")),
        publication_year=w.get("publication_year"),
        language=w.get("language"),
        cited_by_count=int(w.get("cited_by_count") or 0),
        authors=auths,
        primary_topic={
            "id": (pt.get("id") or "").rsplit("/", 1)[-1] or None,
            "display_name": pt.get("display_name"),
            "subfield": {"id": (sub.get("id") or "").rsplit("/", 1)[-1] or None, "display_name": sub.get("display_name")},
            "field": {"id": (fld.get("id") or "").rsplit("/", 1)[-1] or None, "display_name": fld.get("display_name")},
            "domain": {"id": (dom.get("id") or "").rsplit("/", 1)[-1] or None, "display_name": dom.get("display_name")},
        },
        open_access={
            "is_oa": bool(oa.get("is_oa")),
            "oa_status": oa.get("oa_status"),
            "landing_page_url": oa_loc.get("landing_page_url") if isinstance(oa_loc, dict) else None,
            "pdf_url": oa_loc.get("pdf_url") if isinstance(oa_loc, dict) else None,
        },
        relevance_score=(float(w["relevance_score"]) if isinstance(w.get("relevance_score"), (int, float)) else None),
    )


def _lang_filter(lang: str) -> str | None:
    if not lang or lang.lower() == "all":
        return None
    codes = [c.strip().lower() for c in lang.split(",") if c.strip()]
    if not codes:
        return None
    return f"language:{'|'.join(codes)}"


async def run_search(
    client: OpenAlexClient,
    *,
    query: str,
    top_k: int = 20,
    lang: str = "ru",
) -> SearchResult:
    q = (query or "").strip()
    if not q:
        return SearchResult(query="", lang=lang, total=0, items=[], pipeline="noop", notes="Пустой запрос.")

    flt = _lang_filter(lang)
    log.info("search: q=%r lang=%s top_k=%d filter=%s", q, lang, top_k, flt)

    raw, total = await client.works_search(
        query=q,
        filters=flt,
        per_page=top_k,
        select=SEARCH_SELECT,
    )
    items = [_project(w) for w in raw]

    notes = (
        f"Поиск через OpenAlex по запросу {q!r}, ранжирование по релевантности "
        f"(BM25-подобный скоринг по title + abstract)."
    )

    return SearchResult(
        query=q,
        lang=lang,
        total=total,
        items=items,
        pipeline="openalex_search_bm25",
        notes=notes,
    )
