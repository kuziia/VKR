"""Search API. Two backends:
  • openalex — `?search=...&sort=relevance_score:desc` (BM25-like ranking).
  • openaire — `?keywords=...&sortBy=influence,descending`.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..openaire import adapter as oaire_adapter
from ..openaire.client import get_openaire
from ..openalex.client import get_client
from ..search import simple
from ..storage.db import (
    cache_key,
    get_trends_cache,
    put_trends_cache,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])

SEARCH_TTL = 60 * 60  # 1h

Source = Literal["openalex", "openaire"]


class SearchBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(20, ge=1, le=100)
    country: str | None = None
    source: Source = "openalex"


def _openalex_to_dict(res: simple.SearchResult) -> dict[str, Any]:
    return {
        "source": "openalex",
        "query": res.query,
        "country": None,
        "total": res.total,
        "pipeline": res.pipeline,
        "notes": res.notes,
        "items": [
            {
                **dataclasses.asdict(it),
                "abstract_snippet": it.abstract_snippet,
            }
            for it in res.items
        ],
    }


async def _search(
    query: str,
    top_k: int,
    country: str | None,
    source: Source,
) -> dict[str, Any]:
    if not query.strip():
        raise HTTPException(400, "query is required")

    key = cache_key("search", source, query.strip().lower(), country, top_k)
    cached = await get_trends_cache(key, ttl_sec=SEARCH_TTL)
    if cached is not None:
        return cached

    if source == "openaire":
        try:
            items, total = await oaire_adapter.fetch_search(
                get_openaire(),
                query=query,
                country=country,
                limit=top_k,
            )
        except Exception as e:
            log.exception("openaire search failed")
            raise HTTPException(502, f"openaire: {e}") from e
        # Pad items so they match SearchItem shape used by FE
        for it in items:
            it.setdefault("abstract_snippet", None)
            it.setdefault("relevance_score", None)
        payload = {
            "source": "openaire",
            "query": query.strip(),
            "country": country,
            "total": total,
            "pipeline": "openaire_influence_rank",
            "notes": (
                f"OpenAIRE search/publications, ранжирование по impact-score (influence). "
                f"Всего найдено {total:,} результатов.".replace(",", " ")
            ),
            "items": items,
        }
        await put_trends_cache(key, payload)
        return payload

    client = get_client()
    try:
        res = await simple.run_search(client, query=query, top_k=top_k, lang="all")
    except Exception as e:
        log.exception("search failed")
        raise HTTPException(502, f"upstream: {e}") from e

    payload = _openalex_to_dict(res)
    payload["country"] = country
    await put_trends_cache(key, payload)
    return payload


@router.post("")
async def post_search(body: SearchBody) -> dict[str, Any]:
    return await _search(body.query, body.top_k, body.country, body.source)


@router.get("")
async def get_search(
    q: str = Query(..., min_length=1, max_length=512),
    top_k: int = Query(20, ge=1, le=100),
    country: str | None = Query(None),
    source: Source = Query("openalex"),
) -> dict[str, Any]:
    return await _search(q, top_k, country, source)
