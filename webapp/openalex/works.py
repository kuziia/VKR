"""High-level OpenAlex queries: trends + top-cited.

OpenAlex `group_by=publication_year` gives per-year buckets cheaply (one
request). For monthly granularity there is no native `publication_month`
group_by, so we fan out one count-only request per month in parallel and
read `meta.count`. 12 requests/year × 10 years = 120 in parallel ≈ 5–10 s.
"""
from __future__ import annotations

import asyncio
import logging
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal

from .client import OpenAlexClient

log = logging.getLogger(__name__)

Level = Literal["domain", "field", "subfield", "topic", "all"]
Granularity = Literal["day", "month", "quarter", "year"]

# OpenAlex filter prefixes per taxonomy level.
LEVEL_FILTER = {
    "domain": "primary_topic.domain.id",
    "field": "primary_topic.field.id",
    "subfield": "primary_topic.subfield.id",
    "topic": "primary_topic.id",
}

# OpenAlex IDs: topics keep the "T" prefix in filters; domain/field/subfield
# are bare integers. Filter examples that work:
#   primary_topic.domain.id:4
#   primary_topic.field.id:36
#   primary_topic.subfield.id:3605
#   primary_topic.id:T10350
LEVEL_PREFIX = {
    "domain": "",
    "field": "",
    "subfield": "",
    "topic": "T",
}

WORK_SELECT = (
    "id,doi,title,publication_year,publication_date,language,"
    "cited_by_count,authorships,primary_topic,best_oa_location,open_access"
)


@dataclass
class TrendPoint:
    period: str  # "YYYY" or "YYYY-MM"
    count: int


@dataclass
class TrendsResult:
    level: Level
    id: str | None
    label: str | None
    granularity: Granularity
    lang: str  # comma-separated or "all"
    points: list[TrendPoint]
    total: int


def _normalize_id(level: Level, raw_id: str) -> str:
    """Strip URL → ID; for `topic` ensure leading "T"; others stay bare."""
    if level == "all":
        return ""
    raw = raw_id.strip()
    if raw.lower().startswith("https://"):
        raw = raw.rsplit("/", 1)[-1]
    pref = LEVEL_PREFIX[level]
    if pref and not raw.upper().startswith(pref):
        raw = f"{pref}{raw}"
    return raw.upper() if pref else raw


def _node_filter(level: Level, normalized_id: str) -> str | None:
    if level == "all":
        return None
    return f"{LEVEL_FILTER[level]}:{normalized_id}"


def _lang_filter(lang: str) -> str | None:
    """`lang` is "all" or comma-separated codes like "ru" or "ru,en"."""
    if not lang or lang.lower() == "all":
        return None
    codes = [c.strip().lower() for c in lang.split(",") if c.strip()]
    if not codes:
        return None
    return f"language:{'|'.join(codes)}"


def _country_filter(country: str | None) -> str | None:
    """`country` is "all"/empty or ISO-2 code like "ru" or "ru,us"."""
    if not country or country.lower() == "all":
        return None
    codes = [c.strip().lower() for c in country.split(",") if c.strip()]
    if not codes:
        return None
    return f"institutions.country_code:{'|'.join(codes)}"


def _join_filters(parts: Iterable[str | None]) -> str:
    return ",".join(p for p in parts if p)


def _day_range(start: date, end: date) -> list[tuple[str, str, str]]:
    """Yield list of (label "YYYY-MM-DD", from_date, to_date) inclusive."""
    from datetime import timedelta
    out: list[tuple[str, str, str]] = []
    cur = start
    one = timedelta(days=1)
    while cur <= end:
        iso = cur.isoformat()
        out.append((iso, iso, iso))
        cur += one
    return out


MAX_DAYS_FOR_DAY_GRANULARITY = 400


def _month_range(start: date, end: date) -> list[tuple[str, str, str]]:
    """Yield list of (label "YYYY-MM", from_date, to_date) inclusive."""
    out: list[tuple[str, str, str]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        last = monthrange(y, m)[1]
        out.append((f"{y:04d}-{m:02d}", f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"))
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def _parse_period(s: str) -> date:
    """Accept YYYY or YYYY-MM or YYYY-MM-DD."""
    parts = s.split("-")
    if len(parts) == 1:
        return date(int(parts[0]), 1, 1)
    if len(parts) == 2:
        return date(int(parts[0]), int(parts[1]), 1)
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


# ── trends ──
async def trends(
    client: OpenAlexClient,
    *,
    level: Level,
    id: str | None,
    period_from: str,
    period_to: str,
    lang: str = "ru",
    granularity: Granularity = "month",
    country: str | None = None,
) -> TrendsResult:
    norm_id = _normalize_id(level, id) if (level != "all" and id) else None
    base_filters = _join_filters(
        [
            _node_filter(level, norm_id) if norm_id else None,
            _lang_filter(lang),
            _country_filter(country),
        ]
    )

    pf = _parse_period(period_from)
    pt = _parse_period(period_to)
    if pt < pf:
        pf, pt = pt, pf

    points: list[TrendPoint]

    if granularity == "year":
        years_filter = f"publication_year:{pf.year}-{pt.year}"
        full = _join_filters([base_filters, years_filter])
        groups = await client.works_group_by(filters=full, group_by="publication_year")
        # group["key"] is "2024", group["count"] int
        by_year = {str(g.get("key")): int(g.get("count") or 0) for g in groups}
        points = [TrendPoint(period=str(y), count=by_year.get(str(y), 0)) for y in range(pf.year, pt.year + 1)]

    elif granularity == "month":
        months = _month_range(pf, pt)

        async def one(label: str, dfrom: str, dto: str) -> TrendPoint:
            full = _join_filters([base_filters, f"from_publication_date:{dfrom}", f"to_publication_date:{dto}"])
            cnt = await client.works_meta(filters=full)
            return TrendPoint(period=label, count=cnt)

        sem = asyncio.Semaphore(8)

        async def guarded(label: str, dfrom: str, dto: str) -> TrendPoint:
            async with sem:
                return await one(label, dfrom, dto)

        points = list(await asyncio.gather(*(guarded(*m) for m in months)))

    elif granularity == "day":
        # Expand YYYY → first/last of year, YYYY-MM → first/last of month
        if len(period_from.split("-")) == 1:
            pf = date(pf.year, 1, 1)
        elif len(period_from.split("-")) == 2:
            pf = date(pf.year, pf.month, 1)
        if len(period_to.split("-")) == 1:
            pt = date(pt.year, 12, 31)
        elif len(period_to.split("-")) == 2:
            pt = date(pt.year, pt.month, monthrange(pt.year, pt.month)[1])

        n_days = (pt - pf).days + 1
        if n_days > MAX_DAYS_FOR_DAY_GRANULARITY:
            raise ValueError(
                f"day-granularity period too long: {n_days} days "
                f"(limit {MAX_DAYS_FOR_DAY_GRANULARITY}). Switch to month/quarter/year."
            )
        days = _day_range(pf, pt)

        async def one_day(label: str, dfrom: str, dto: str) -> TrendPoint:
            full = _join_filters([base_filters, f"from_publication_date:{dfrom}", f"to_publication_date:{dto}"])
            cnt = await client.works_meta(filters=full)
            return TrendPoint(period=label, count=cnt)

        sem = asyncio.Semaphore(8)

        async def guarded_day(label: str, dfrom: str, dto: str) -> TrendPoint:
            async with sem:
                return await one_day(label, dfrom, dto)

        points = list(await asyncio.gather(*(guarded_day(*d) for d in days)))

    elif granularity == "quarter":
        # Build quarterly buckets [Q1..Q4] for each year in range.
        buckets: list[tuple[str, str, str]] = []
        for y in range(pf.year, pt.year + 1):
            for q, (m_from, m_to) in enumerate([(1, 3), (4, 6), (7, 9), (10, 12)], start=1):
                d_from = date(y, m_from, 1)
                d_to = date(y, m_to, monthrange(y, m_to)[1])
                if d_to < pf or d_from > pt:
                    continue
                d_from = max(d_from, pf)
                d_to = min(d_to, pt)
                buckets.append((f"{y}-Q{q}", d_from.isoformat(), d_to.isoformat()))

        sem = asyncio.Semaphore(8)

        async def one_q(label: str, dfrom: str, dto: str) -> TrendPoint:
            async with sem:
                full = _join_filters([base_filters, f"from_publication_date:{dfrom}", f"to_publication_date:{dto}"])
                return TrendPoint(period=label, count=await client.works_meta(filters=full))

        points = list(await asyncio.gather(*(one_q(*b) for b in buckets)))

    else:  # pragma: no cover
        raise ValueError(f"unknown granularity {granularity}")

    total = sum(p.count for p in points)
    return TrendsResult(
        level=level,
        id=norm_id,
        label=None,  # filled by API layer from taxonomy
        granularity=granularity,
        lang=lang,
        points=points,
        total=total,
    )


# ── top-cited ──
async def top_cited(
    client: OpenAlexClient,
    *,
    level: Level,
    id: str | None,
    period_from: str | None,
    period_to: str | None,
    lang: str = "ru",
    limit: int = 20,
    country: str | None = None,
) -> list[dict]:
    norm_id = _normalize_id(level, id) if (level != "all" and id) else None
    parts: list[str | None] = [
        _node_filter(level, norm_id) if norm_id else None,
        _lang_filter(lang),
        _country_filter(country),
        # Quality filters: skip DOI-farm reports (LJournal et al.) — they
        # often have URL-as-title and inflated cited_by_count from in-farm
        # cross-references that Crossref doesn't confirm.
        "type:article",
        "is_paratext:false",
        "referenced_works_count:>0",
    ]
    if period_from:
        parts.append(f"from_publication_date:{_parse_period(period_from).isoformat()}")
    if period_to:
        # For "to_publication_date", use end-of-month if user gave only YYYY-MM
        pt = _parse_period(period_to)
        if len(period_to.split("-")) == 2:
            pt = date(pt.year, pt.month, monthrange(pt.year, pt.month)[1])
        parts.append(f"to_publication_date:{pt.isoformat()}")

    flt = _join_filters(parts)
    return await client.works_top(filters=flt, sort="cited_by_count:desc", per_page=limit, select=WORK_SELECT)


# ── by-field aggregation (sym-bars) ──
async def works_by_field(
    client: OpenAlexClient,
    *,
    period_from: str | None,
    period_to: str | None,
    lang: str = "ru",
    domain_id: str | None = None,
    country: str | None = None,
) -> list[tuple[str, str, int]]:
    """Returns [(field_id, field_display_name, count)] aggregated via group_by."""
    parts: list[str | None] = [_lang_filter(lang), _country_filter(country)]
    if domain_id:
        parts.append(_node_filter("domain", _normalize_id("domain", domain_id)))
    if period_from:
        parts.append(f"from_publication_date:{_parse_period(period_from).isoformat()}")
    if period_to:
        pt = _parse_period(period_to)
        if len(period_to.split("-")) == 2:
            pt = date(pt.year, pt.month, monthrange(pt.year, pt.month)[1])
        parts.append(f"to_publication_date:{pt.isoformat()}")
    flt = _join_filters(parts)
    groups = await client.works_group_by(filters=flt, group_by="primary_topic.field.id")
    out: list[tuple[str, str, int]] = []
    for g in groups:
        key = g.get("key", "") or ""
        # OpenAlex returns either "F1701" or full URL — normalize.
        fid = key.rsplit("/", 1)[-1]
        out.append((fid, g.get("key_display_name") or fid, int(g.get("count") or 0)))
    return out
