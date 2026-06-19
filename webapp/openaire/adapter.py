"""OpenAIRE → unified shape adapter.

OpenAIRE returns deeply nested JSON (a JSON-flavoured XML really). We project
records into the same shape the FE already understands for OpenAlex top-cited
items. Some fields are unavailable (no taxonomy, weaker OA metadata) — they
get null and the UI degrades gracefully.
"""
from __future__ import annotations

import logging
from calendar import monthrange
from datetime import date
from typing import Any

from .client import OpenAireClient

log = logging.getLogger(__name__)

# OpenAIRE sort keys we expose
SORT_INFLUENCE = "influence,descending"
SORT_POPULARITY = "popularity,descending"
SORT_CITATIONS = "citationCount,descending"
SORT_RECENCY = "dateofacceptance,descending"


def _val(v: Any) -> Any:
    """Unwrap OpenAIRE's `{'$': value}` pattern recursively."""
    if isinstance(v, dict) and "$" in v and len(v) <= 2:
        return v["$"]
    return v


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _pick_main_title(title_field: Any) -> str | None:
    """OpenAIRE's `title` is either a dict or a list of dicts. The main one
    has `@classid == 'main title'`; we take the longest such string (titles
    repeat with different normalisations)."""
    items = _as_list(title_field)
    candidates: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("@classid") not in ("main title", "alternative title", None):
            continue
        text = it.get("$")
        if isinstance(text, str) and text.strip():
            candidates.append(text.strip())
    if not candidates:
        for it in items:
            if isinstance(it, dict):
                text = it.get("$")
                if isinstance(text, str) and text.strip():
                    candidates.append(text.strip())
    if not candidates:
        return None
    return max(candidates, key=len)


def _pick_doi(pid_field: Any) -> str | None:
    for p in _as_list(pid_field):
        if not isinstance(p, dict):
            continue
        if p.get("@classid") == "doi":
            v = p.get("$")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _creators(creator_field: Any, max_n: int = 8) -> list[str]:
    out: list[str] = []
    for c in _as_list(creator_field)[:max_n]:
        if isinstance(c, dict):
            v = c.get("$")
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
    return out


def _year(date_field: Any) -> int | None:
    v = _val(date_field)
    if not isinstance(v, str) or len(v) < 4:
        return None
    try:
        return int(v[:4])
    except ValueError:
        return None


def _measure_value(measure_field: Any, name: str) -> float | None:
    """Each `measure` is a dict with @id and @score. Some records have a list."""
    for m in _as_list(measure_field):
        if not isinstance(m, dict):
            continue
        if m.get("@id") == name:
            try:
                return float(m.get("@score") or 0)
            except (TypeError, ValueError):
                return None
    return None


def _bestaccessright(field: Any) -> tuple[bool, str | None]:
    v = field
    if isinstance(v, dict):
        cid = v.get("@classid") or ""
        return cid in ("OPEN", "OPEN SOURCE"), v.get("@classname") or v.get("@classid")
    return False, None


def _project(record: dict[str, Any]) -> dict[str, Any]:
    """`record` is one element of `response.results.result`."""
    md = ((record.get("metadata") or {}).get("oaf:entity") or {}).get("oaf:result") or {}
    if not md:
        return {}

    title = _pick_main_title(md.get("title")) or "(no title)"
    doi = _pick_doi(md.get("pid"))
    authors = _creators(md.get("creator"))
    year = _year(md.get("dateofacceptance"))
    # OpenAIRE doesn't expose raw citation count via search API; only their
    # categorical `*_alt` impact bands (typical 0-50 range) and continuous
    # `influence`/`popularity` scores. We use `influence_alt` as a sortable
    # proxy for "impact" and surface it in the cited_by_count slot. The FE
    # `notes` makes the substitution explicit.
    citations = _measure_value(md.get("measure"), "citationCount")
    influence = _measure_value(md.get("measure"), "influence")
    popularity = _measure_value(md.get("measure"), "popularity")
    if citations is None:
        citations = _measure_value(md.get("measure"), "influence_alt") or 0
    is_oa, oa_label = _bestaccessright(md.get("bestaccessright"))

    # Stable identifier from the response header
    obj_id = ((record.get("header") or {}).get("dri:objIdentifier") or {})
    raw_id = _val(obj_id) or ""
    short_id = f"oaire:{raw_id}" if raw_id else None

    return {
        # Use openalex_id slot to keep the FE schema intact (FE checks if
        # the prefix is `oaire:` to decide whether the row is clickable).
        "openalex_id": short_id,
        "doi": (f"https://doi.org/{doi}" if doi and not doi.startswith("http") else doi),
        "title": title,
        "publication_year": year,
        "publication_date": _val(md.get("dateofacceptance")),
        "language": None,
        "cited_by_count": int(citations) if citations is not None else 0,
        "authors": authors,
        "primary_topic": {
            "id": None,
            "display_name": None,
            "subfield": {"id": None, "display_name": None},
            "field": {"id": None, "display_name": None},
            "domain": {"id": None, "display_name": None},
        },
        "open_access": {
            "is_oa": is_oa,
            "oa_status": (oa_label.lower() if isinstance(oa_label, str) else None),
            "landing_page_url": None,
            "pdf_url": None,
        },
        "_oaire": {
            "influence": influence,
            "popularity": popularity,
        },
    }


def _expand_to(period_to: str) -> str:
    parts = period_to.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-12-31"
    if len(parts) == 2:
        last = monthrange(int(parts[0]), int(parts[1]))[1]
        return f"{parts[0]}-{parts[1]}-{last:02d}"
    return period_to


def _expand_from(period_from: str) -> str:
    parts = period_from.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-01-01"
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1]}-01"
    return period_from


async def fetch_top(
    client: OpenAireClient,
    *,
    country: str | None,
    period_from: str | None,
    period_to: str | None,
    limit: int = 20,
    sort_by: str = SORT_INFLUENCE,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "size": min(max(limit, 1), 50),
        "format": "json",
        "sortBy": sort_by,
    }
    if country:
        params["country"] = country.upper()
    if period_from:
        params["fromDateAccepted"] = _expand_from(period_from)
    if period_to:
        params["toDateAccepted"] = _expand_to(period_to)

    data = await client._get("/search/publications", params)
    raw = ((data.get("response") or {}).get("results") or {}).get("result") or []
    if isinstance(raw, dict):
        raw = [raw]
    out: list[dict[str, Any]] = []
    for r in raw:
        proj = _project(r)
        if proj.get("title"):
            out.append(proj)
    return out


async def fetch_search(
    client: OpenAireClient,
    *,
    query: str,
    country: str | None,
    limit: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    """Free-text search → projected items + total."""
    params: dict[str, Any] = {
        "keywords": query,
        "size": min(max(limit, 1), 50),
        "format": "json",
        "sortBy": SORT_INFLUENCE,
    }
    if country:
        params["country"] = country.upper()
    data = await client._get("/search/publications", params)
    header = (data.get("response") or {}).get("header") or {}
    total = header.get("total")
    if isinstance(total, dict):
        total = total.get("$") or total.get("value") or 0
    try:
        total_n = int(total or 0)
    except (TypeError, ValueError):
        total_n = 0
    raw = ((data.get("response") or {}).get("results") or {}).get("result") or []
    if isinstance(raw, dict):
        raw = [raw]
    items: list[dict[str, Any]] = []
    for r in raw:
        proj = _project(r)
        if proj.get("title"):
            items.append(proj)
    return items, total_n
