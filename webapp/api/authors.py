"""Author endpoints — author profile + their works."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..openalex.client import get_client
from ..storage.db import (
    cache_key,
    get_trends_cache,
    put_trends_cache,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/authors", tags=["authors"])

AUTHOR_TTL = 7 * 24 * 60 * 60  # 7 days
WORKS_TTL = 24 * 60 * 60  # 1 day

AUTHOR_SELECT = (
    "id,orcid,display_name,display_name_alternatives,works_count,"
    "cited_by_count,summary_stats,last_known_institutions,affiliations,"
    "counts_by_year,topics"
)

WORKS_SELECT = (
    "id,doi,title,publication_year,publication_date,language,cited_by_count,"
    "authorships,primary_topic,best_oa_location,open_access"
)


def _short(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None
    return url_or_id.rsplit("/", 1)[-1] or None


def _project_author(a: dict[str, Any]) -> dict[str, Any]:
    stats = a.get("summary_stats") or {}
    last_inst = []
    for inst in a.get("last_known_institutions") or []:
        if isinstance(inst, dict) and inst.get("display_name"):
            last_inst.append({
                "id": _short(inst.get("id")),
                "display_name": inst["display_name"],
                "country_code": (inst.get("country_code") or "").lower() or None,
                "type": inst.get("type"),
            })
    topics = []
    for t in (a.get("topics") or [])[:5]:
        if isinstance(t, dict) and t.get("display_name"):
            topics.append({
                "id": _short(t.get("id")),
                "display_name": t["display_name"],
                "count": t.get("count"),
            })
    return {
        "openalex_id": _short(a.get("id")),
        "orcid": _short(a.get("orcid")),
        "display_name": a.get("display_name"),
        "alternatives": a.get("display_name_alternatives") or [],
        "works_count": int(a.get("works_count") or 0),
        "cited_by_count": int(a.get("cited_by_count") or 0),
        "h_index": stats.get("h_index"),
        "i10_index": stats.get("i10_index"),
        "mean_citedness": stats.get("2yr_mean_citedness"),
        "last_known_institutions": last_inst,
        "counts_by_year": a.get("counts_by_year") or [],
        "topics": topics,
    }


def _project_work(w: dict[str, Any]) -> dict[str, Any]:
    auths: list[str] = []
    for a in (w.get("authorships") or [])[:6]:
        au = a.get("author") or {}
        if au.get("display_name"):
            auths.append(au["display_name"])
    pt = w.get("primary_topic") or {}
    fld = pt.get("field") or {}
    oa_loc = w.get("best_oa_location") or {}
    oa = w.get("open_access") or {}
    return {
        "openalex_id": _short(w.get("id")),
        "doi": w.get("doi"),
        "title": w.get("title") or "(no title)",
        "publication_year": w.get("publication_year"),
        "publication_date": w.get("publication_date"),
        "language": w.get("language"),
        "cited_by_count": int(w.get("cited_by_count") or 0),
        "authors": auths,
        "primary_topic": {
            "id": _short(pt.get("id")),
            "display_name": pt.get("display_name"),
            "field": {"id": _short(fld.get("id")), "display_name": fld.get("display_name")},
        },
        "open_access": {
            "is_oa": bool(oa.get("is_oa")),
            "oa_status": oa.get("oa_status"),
            "landing_page_url": oa_loc.get("landing_page_url") if isinstance(oa_loc, dict) else None,
            "pdf_url": oa_loc.get("pdf_url") if isinstance(oa_loc, dict) else None,
        },
    }


@router.get("/{author_id}")
async def get_author(author_id: str) -> dict[str, Any]:
    key = cache_key("author", author_id)
    cached = await get_trends_cache(key, ttl_sec=AUTHOR_TTL)
    if cached is not None:
        return cached

    client = get_client()
    try:
        a = await client.author(author_id, select=AUTHOR_SELECT)
    except Exception as e:
        log.exception("author fetch failed: %s", author_id)
        raise HTTPException(502, f"upstream: {e}") from e
    payload = _project_author(a)
    await put_trends_cache(key, payload)
    return payload


@router.get("/{author_id}/works")
async def get_author_works(
    author_id: str,
    sort: str = Query("cited_by_count:desc"),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    key = cache_key("author-works", author_id, sort, limit)
    cached = await get_trends_cache(key, ttl_sec=WORKS_TTL)
    if cached is not None:
        return cached

    client = get_client()
    try:
        works = await client.works_top(
            filters=f"author.id:{author_id}",
            sort=sort,
            per_page=limit,
            select=WORKS_SELECT,
        )
    except Exception as e:
        log.exception("author works failed: %s", author_id)
        raise HTTPException(502, f"upstream: {e}") from e

    items = [_project_work(w) for w in works]
    payload = {"author_id": author_id, "sort": sort, "limit": limit, "items": items}
    await put_trends_cache(key, payload)
    return payload
