"""Dashboard endpoints: trends, top-cited, by-field aggregation."""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request

from ..bertrend import store as bertrend_store
from ..openaire import adapter as oaire_adapter
from ..openaire.client import get_openaire
from ..openaire.coverage import count_by_year, count_in_range
from ..openalex import works
from ..openalex.client import get_client
from ..storage.db import (
    cache_key,
    get_trends_cache,
    get_work_cache,
    put_trends_cache,
    put_work_cache,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

TRENDS_TTL = 24 * 60 * 60  # 24h
WORKS_TTL = 7 * 24 * 60 * 60  # 7 days


def _label_for(req: Request, level: str, normalized_id: str | None) -> str | None:
    if level == "all" or not normalized_id:
        return None
    tax = req.app.state.taxonomy
    for d in tax.domains:
        if level == "domain" and d.id == normalized_id:
            return d.display_name
        for f in d.fields:
            if level == "field" and f.id == normalized_id:
                return f.display_name
            for s in f.subfields:
                if level == "subfield" and s.id == normalized_id:
                    return s.display_name
                for t in s.topics:
                    if level == "topic" and t.id == normalized_id:
                        return t.display_name
    return None


Source = Literal["openalex", "openaire"]


@router.get("/trends")
async def get_trends(
    request: Request,
    level: Literal["all", "domain", "field", "subfield", "topic"] = Query("all"),
    id: str | None = Query(None),
    period_from: str = Query(..., alias="from"),
    period_to: str = Query(..., alias="to"),
    granularity: Literal["day", "month", "quarter", "year"] = Query("month"),
    country: str | None = Query(None),
    lang: str = Query("all"),
    source: Source = Query("openalex"),
) -> dict[str, Any]:
    if level != "all" and not id:
        raise HTTPException(400, "id is required when level != 'all'")
    if source == "openaire":
        lang = "all"  # OpenAIRE doesn't filter by language
    key = cache_key("trends", source, level, id, period_from, period_to, granularity, country, lang)
    cached = await get_trends_cache(key, ttl_sec=TRENDS_TTL)
    if cached is not None:
        return cached

    if source == "openaire":
        # OpenAIRE: no taxonomy support, no granularity below year. Force
        # year-aggregation; warn the FE via `notes`.
        if level != "all":
            raise HTTPException(400, "Для фильтра по узлу таксономии переключи источник на OpenAlex.")
        try:
            year_points = await count_by_year(
                get_openaire(),
                country=country,
                period_from=period_from,
                period_to=period_to,
            )
            total = await count_in_range(
                get_openaire(),
                country=country,
                period_from=period_from,
                period_to=period_to,
            )
        except Exception as e:
            log.exception("openaire trends failed")
            raise HTTPException(502, f"openaire: {e}") from e

        payload = {
            "source": "openaire",
            "level": "all",
            "id": None,
            "label": None,
            "granularity": "year",
            "country": country,
            "from": period_from,
            "to": period_to,
            "points": [{"period": str(p.year), "count": p.count} for p in year_points],
            "total": total,
            "notes": "OpenAIRE: годовая агрегация, фильтр по стране автора.",
        }
        await put_trends_cache(key, payload)
        return payload

    # source == openalex
    client = get_client()
    try:
        res = await works.trends(
            client,
            level=level,
            id=id,
            period_from=period_from,
            period_to=period_to,
            lang=lang,
            granularity=granularity,
            country=country,
        )
    except Exception as e:
        log.exception("trends failed")
        raise HTTPException(502, f"upstream: {e}") from e

    payload = {
        "source": "openalex",
        "level": res.level,
        "id": res.id,
        "label": _label_for(request, res.level, res.id),
        "granularity": res.granularity,
        "country": country,
        "lang": lang,
        "from": period_from,
        "to": period_to,
        "points": [{"period": p.period, "count": p.count} for p in res.points],
        "total": res.total,
        "notes": None,
    }
    await put_trends_cache(key, payload)
    return payload


def _strip_work(w: dict[str, Any]) -> dict[str, Any]:
    """Project only the fields the FE renders, drop ~30 KB of clutter."""
    auths = []
    for a in (w.get("authorships") or [])[:6]:
        au = a.get("author") or {}
        if au.get("display_name"):
            auths.append(au["display_name"])
    pt = w.get("primary_topic") or {}
    sub = (pt.get("subfield") or {})
    fld = (pt.get("field") or {})
    dom = (pt.get("domain") or {})
    oa_loc = (w.get("best_oa_location") or {}) or {}
    oa = w.get("open_access") or {}
    short_id = (w.get("id") or "").rsplit("/", 1)[-1]
    return {
        "openalex_id": short_id,
        "doi": w.get("doi"),
        "title": w.get("title") or "(no title)",
        "publication_year": w.get("publication_year"),
        "publication_date": w.get("publication_date"),
        "language": w.get("language"),
        "cited_by_count": int(w.get("cited_by_count") or 0),
        "authors": auths,
        "primary_topic": {
            "id": (pt.get("id") or "").rsplit("/", 1)[-1] or None,
            "display_name": pt.get("display_name"),
            "subfield": {"id": (sub.get("id") or "").rsplit("/", 1)[-1] or None, "display_name": sub.get("display_name")},
            "field": {"id": (fld.get("id") or "").rsplit("/", 1)[-1] or None, "display_name": fld.get("display_name")},
            "domain": {"id": (dom.get("id") or "").rsplit("/", 1)[-1] or None, "display_name": dom.get("display_name")},
        },
        "open_access": {
            "is_oa": bool(oa.get("is_oa")),
            "oa_status": oa.get("oa_status"),
            "landing_page_url": oa_loc.get("landing_page_url"),
            "pdf_url": oa_loc.get("pdf_url"),
        },
    }


@router.get("/top-cited")
async def get_top_cited(
    request: Request,
    level: Literal["all", "domain", "field", "subfield", "topic"] = Query("all"),
    id: str | None = Query(None),
    period_from: str | None = Query(None, alias="from"),
    period_to: str | None = Query(None, alias="to"),
    limit: int = Query(20, ge=1, le=100),
    country: str | None = Query(None),
    lang: str = Query("all"),
    source: Source = Query("openalex"),
) -> dict[str, Any]:
    if level != "all" and not id:
        raise HTTPException(400, "id is required when level != 'all'")
    if source == "openaire":
        lang = "all"
    key = cache_key("top", source, level, id, period_from, period_to, limit, country, lang)
    cached = await get_trends_cache(key, ttl_sec=TRENDS_TTL)
    if cached is not None:
        return cached

    if source == "openaire":
        if level != "all":
            raise HTTPException(400, "Для фильтра по узлу таксономии переключи источник на OpenAlex.")
        try:
            items = await oaire_adapter.fetch_top(
                get_openaire(),
                country=country,
                period_from=period_from,
                period_to=period_to,
                limit=limit,
            )
        except Exception as e:
            log.exception("openaire top failed")
            raise HTTPException(502, f"openaire: {e}") from e
        payload = {
            "source": "openaire",
            "level": "all",
            "id": None,
            "label": None,
            "country": country,
            "from": period_from,
            "to": period_to,
            "limit": limit,
            "items": items,
            "notes": "OpenAIRE: ранжирование по impact-score (influence). Колонка Cit. показывает шкалу influence_alt (0-50).",
        }
        await put_trends_cache(key, payload)
        return payload

    client = get_client()
    try:
        results = await works.top_cited(
            client,
            level=level,
            id=id,
            period_from=period_from,
            period_to=period_to,
            lang=lang,
            limit=limit,
            country=country,
        )
    except Exception as e:
        log.exception("top-cited failed")
        raise HTTPException(502, f"upstream: {e}") from e

    items = [_strip_work(w) for w in results]
    # Drop works whose "title" is actually a PDF URL (LJournal et al. junk
    # records that slip past type:article when Crossref-side metadata is bad).
    items = [
        it for it in items
        if not (it["title"] or "").lower().startswith(("http://", "https://"))
    ]
    # Cache stripped works individually so /article/:id is instant.
    for it in items:
        await put_work_cache(it["openalex_id"], it)

    payload = {
        "source": "openalex",
        "level": level,
        "id": id,
        "label": _label_for(request, level, id),
        "country": country,
        "lang": lang,
        "from": period_from,
        "to": period_to,
        "limit": limit,
        "items": items,
        "notes": None,
    }
    await put_trends_cache(key, payload)
    return payload


@router.get("/by-field")
async def get_by_field(
    period_from: str | None = Query(None, alias="from"),
    period_to: str | None = Query(None, alias="to"),
    domain_id: str | None = Query(None, alias="domain_id"),
    limit: int = Query(10, ge=1, le=30),
    country: str | None = Query(None),
    lang: str = Query("all"),
    source: Source = Query("openalex"),
) -> dict[str, Any]:
    if source == "openaire":
        return {
            "source": "openaire",
            "country": country,
            "from": period_from,
            "to": period_to,
            "domain_id": domain_id,
            "items": [],
            "supported": False,
            "notes": "Распределение по полям доступно при выборе источника OpenAlex.",
        }

    key = cache_key("by-field", domain_id, period_from, period_to, limit, country, lang)
    cached = await get_trends_cache(key, ttl_sec=TRENDS_TTL)
    if cached is not None:
        return cached

    client = get_client()
    try:
        rows = await works.works_by_field(
            client,
            period_from=period_from,
            period_to=period_to,
            lang=lang,
            domain_id=domain_id,
            country=country,
        )
    except Exception as e:
        log.exception("by-field failed")
        raise HTTPException(502, f"upstream: {e}") from e

    rows.sort(key=lambda r: r[2], reverse=True)
    items = [{"id": r[0], "display_name": r[1], "count": r[2]} for r in rows[:limit]]
    payload = {
        "source": "openalex",
        "country": country,
        "lang": lang,
        "from": period_from,
        "to": period_to,
        "domain_id": domain_id,
        "items": items,
        "supported": True,
        "notes": None,
    }
    await put_trends_cache(key, payload)
    return payload


@router.get("/coverage")
async def get_coverage(
    period_from: str = Query(..., alias="from"),
    period_to: str = Query(..., alias="to"),
    country: str | None = Query(None),
) -> dict[str, Any]:
    lang = "all"
    """OpenAlex vs OpenAIRE coverage for the current period.

    The lang filter only applies to OpenAlex (OpenAIRE doesn't support
    language filtering — verified). For meaningful comparison, use
    country=ru or similar.
    """
    from ..openaire.client import get_openaire
    from ..openaire.coverage import count_in_range

    key = cache_key("coverage", period_from, period_to, country, lang)
    cached = await get_trends_cache(key, ttl_sec=TRENDS_TTL)
    if cached is not None:
        return cached

    # OpenAlex: count via meta only
    oa_client = get_client()
    parts: list[str | None] = [
        works._lang_filter(lang) if lang and lang.lower() != "all" else None,
        works._country_filter(country),
    ]
    pf = works._parse_period(period_from)
    pt = works._parse_period(period_to)
    if pt < pf:
        pf, pt = pt, pf
    parts.append(f"from_publication_date:{pf.isoformat()}")
    # Expand `to` to period-end if the user gave YYYY or YYYY-MM
    parts_to = period_to.split("-")
    if len(parts_to) == 1:
        pt = pt.replace(month=12, day=31)
    elif len(parts_to) == 2:
        from calendar import monthrange as _mr
        pt = pt.replace(day=_mr(pt.year, pt.month)[1])
    parts.append(f"to_publication_date:{pt.isoformat()}")
    flt = works._join_filters(parts)
    try:
        oa_count = await oa_client.works_meta(filters=flt)
    except Exception as e:
        log.exception("coverage: openalex failed")
        raise HTTPException(502, f"openalex: {e}") from e

    oaire_count = 0
    oaire_error: str | None = None
    if country:
        try:
            oaire_count = await count_in_range(
                get_openaire(),
                country=country,
                period_from=period_from,
                period_to=period_to,
            )
        except Exception as e:
            log.warning("coverage: openaire failed: %s", e)
            oaire_error = str(e)[:120]

    payload = {
        "from": period_from,
        "to": period_to,
        "country": country,
        "lang": lang,
        "openalex_count": oa_count,
        "openaire_count": oaire_count,
        "openaire_supported": bool(country),
        "openaire_error": oaire_error,
    }
    await put_trends_cache(key, payload)
    return payload


@router.get("/bertrend")
async def get_bertrend() -> dict[str, Any]:
    """BERTrend topic-monitor payload — emerging/strong topics for the
    last year (12 monthly windows). Data is read from a local sqlite DB
    produced by the offline `bertrend_evaluation` pipeline."""
    return bertrend_store.dashboard_payload(top_n_per_signal=8)
