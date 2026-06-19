"""Article endpoints: full work, OA status, summary."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from ..agents import citation_graph, pdf_fetch, summarize
from ..openalex.client import get_client
from ..storage.db import (
    cache_key,
    get_trends_cache,
    get_work_cache,
    put_trends_cache,
    put_work_cache,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/articles", tags=["articles"])

ARTICLE_TTL = 7 * 24 * 60 * 60  # 7 days
SUMMARY_TTL = 30 * 24 * 60 * 60  # 30 days

ARTICLE_SELECT = (
    "id,doi,ids,title,abstract_inverted_index,publication_year,publication_date,"
    "language,cited_by_count,referenced_works_count,authorships,primary_topic,"
    "primary_location,best_oa_location,open_access,locations,type,"
    "keywords,concepts,counts_by_year"
)


def _abstract_from_inverted(idx: dict[str, list[int]] | None) -> str | None:
    if not idx:
        return None
    positions: list[tuple[int, str]] = []
    for word, poses in idx.items():
        for p in poses:
            positions.append((p, word))
    if not positions:
        return None
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def _short_id(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None
    return url_or_id.rsplit("/", 1)[-1] or None


def _project_article(w: dict[str, Any]) -> dict[str, Any]:
    """Article-page shape — full abstract, all authors, full venue."""
    auths = []
    for a in w.get("authorships") or []:
        au = a.get("author") or {}
        if not au.get("display_name"):
            continue
        affs = []
        for inst in (a.get("institutions") or [])[:2]:
            if inst.get("display_name"):
                affs.append(inst["display_name"])
        auths.append(
            {
                "name": au["display_name"],
                "openalex_id": _short_id(au.get("id")),
                "affiliations": affs,
            }
        )

    pt = w.get("primary_topic") or {}
    sub = pt.get("subfield") or {}
    fld = pt.get("field") or {}
    dom = pt.get("domain") or {}

    primary_loc = w.get("primary_location") or {}
    venue = (primary_loc.get("source") or {}) if isinstance(primary_loc, dict) else {}

    oa_loc = w.get("best_oa_location") or {}
    oa = w.get("open_access") or {}

    ids = w.get("ids") or {}
    arxiv_id = None
    arxiv_url = ids.get("arxiv") if isinstance(ids, dict) else None
    if arxiv_url:
        arxiv_id = arxiv_url.rsplit("/", 1)[-1]

    keywords = []
    for k in (w.get("keywords") or [])[:10]:
        if isinstance(k, dict) and k.get("display_name"):
            keywords.append(k["display_name"])

    concepts = []
    for c in (w.get("concepts") or [])[:8]:
        if isinstance(c, dict) and c.get("display_name") and (c.get("level") or 0) <= 2:
            concepts.append({"display_name": c["display_name"], "level": c.get("level")})

    return {
        "openalex_id": _short_id(w.get("id")),
        "doi": w.get("doi"),
        "title": w.get("title") or "(no title)",
        "abstract": _abstract_from_inverted(w.get("abstract_inverted_index")),
        "publication_year": w.get("publication_year"),
        "publication_date": w.get("publication_date"),
        "language": w.get("language"),
        "type": w.get("type"),
        "cited_by_count": int(w.get("cited_by_count") or 0),
        "referenced_works_count": int(w.get("referenced_works_count") or 0),
        "authors": auths,
        "venue": {
            "display_name": venue.get("display_name") if isinstance(venue, dict) else None,
            "type": venue.get("type") if isinstance(venue, dict) else None,
        },
        "primary_topic": {
            "id": _short_id(pt.get("id")),
            "display_name": pt.get("display_name"),
            "subfield": {"id": _short_id(sub.get("id")), "display_name": sub.get("display_name")},
            "field": {"id": _short_id(fld.get("id")), "display_name": fld.get("display_name")},
            "domain": {"id": _short_id(dom.get("id")), "display_name": dom.get("display_name")},
        },
        "open_access": {
            "is_oa": bool(oa.get("is_oa")),
            "oa_status": oa.get("oa_status"),
            "landing_page_url": oa_loc.get("landing_page_url") if isinstance(oa_loc, dict) else None,
            "pdf_url": oa_loc.get("pdf_url") if isinstance(oa_loc, dict) else None,
        },
        "ids": {
            "doi": w.get("doi"),
            "arxiv": arxiv_id,
            "openalex": _short_id(w.get("id")),
        },
        "keywords": keywords,
        "concepts": concepts,
        "counts_by_year": w.get("counts_by_year") or [],
    }


@router.get("/{openalex_id}")
async def get_article(openalex_id: str) -> dict[str, Any]:
    cache_id = f"art:{openalex_id}"
    cached = await get_work_cache(cache_id, ttl_sec=ARTICLE_TTL)
    if cached is not None:
        return cached

    client = get_client()
    try:
        w = await client.work(openalex_id, select=ARTICLE_SELECT)
    except Exception as e:
        log.exception("article fetch failed: %s", openalex_id)
        raise HTTPException(502, f"upstream: {e}") from e

    art = _project_article(w)
    await put_work_cache(cache_id, art)
    return art


@router.get("/{openalex_id}/oa-status")
async def get_oa_status(openalex_id: str) -> dict[str, Any]:
    art = await get_article(openalex_id)
    sources = await pdf_fetch.list_sources(art)
    return {
        "openalex_id": openalex_id,
        "is_oa": art["open_access"]["is_oa"],
        "oa_status": art["open_access"]["oa_status"],
        "sources": sources,
    }


@router.post("/{openalex_id}/summary")
async def post_summary(openalex_id: str, request: Request) -> dict[str, Any]:
    art = await get_article(openalex_id)

    key = cache_key("summary", openalex_id)
    cached = await get_trends_cache(key, ttl_sec=SUMMARY_TTL)
    if cached is not None:
        return cached

    try:
        result = await summarize.summarize_article(art)
    except summarize.LLMNotConfigured as e:
        raise HTTPException(503, str(e)) from e
    except summarize.LLMRateLimited as e:
        raise HTTPException(429, str(e)) from e
    except Exception as e:
        log.exception("summarize failed: %s", openalex_id)
        raise HTTPException(502, f"summary failed: {e}") from e

    await put_trends_cache(key, result)
    return result


GRAPH_TTL = 24 * 60 * 60  # 1 day


@router.get("/{openalex_id}/citation-graph")
async def get_citation_graph(
    openalex_id: str,
    depth: int = Query(1, ge=1, le=3),
    fanout: int = Query(8, ge=2, le=20),
) -> dict[str, Any]:
    key = cache_key("graph", openalex_id, depth, fanout)
    cached = await get_trends_cache(key, ttl_sec=GRAPH_TTL)
    if cached is not None:
        return cached

    client = get_client()
    try:
        g = await citation_graph.build_graph(
            client, root_id=openalex_id, depth=depth, fanout=fanout
        )
    except Exception as e:
        log.exception("citation graph failed: %s", openalex_id)
        raise HTTPException(502, f"upstream: {e}") from e

    await put_trends_cache(key, g)
    return g
