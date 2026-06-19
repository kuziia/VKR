"""High-level OpenAIRE helpers — count by year, range count.

OpenAIRE has no native group_by, so for year-granularity we fan out
one count-only call per year (12 calls for 12 years, parallel-friendly).
"""
from __future__ import annotations

import asyncio
import logging
from calendar import monthrange
from dataclasses import dataclass
from datetime import date

from .client import OpenAireClient

log = logging.getLogger(__name__)


@dataclass
class YearPoint:
    year: int
    count: int


def _parse_period(s: str) -> date:
    parts = s.split("-")
    if len(parts) == 1:
        return date(int(parts[0]), 1, 1)
    if len(parts) == 2:
        return date(int(parts[0]), int(parts[1]), 1)
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


async def count_by_year(
    client: OpenAireClient,
    *,
    country: str | None,
    period_from: str,
    period_to: str,
    concurrency: int = 4,
) -> list[YearPoint]:
    pf = _parse_period(period_from)
    pt = _parse_period(period_to)
    if pt < pf:
        pf, pt = pt, pf

    years = list(range(pf.year, pt.year + 1))
    sem = asyncio.Semaphore(concurrency)

    async def one(y: int) -> YearPoint:
        async with sem:
            cnt = await client.count_publications(
                country=country,
                from_date=f"{y}-01-01",
                to_date=f"{y}-12-{monthrange(y, 12)[1]:02d}",
            )
            return YearPoint(year=y, count=cnt)

    return list(await asyncio.gather(*(one(y) for y in years)))


async def count_in_range(
    client: OpenAireClient,
    *,
    country: str | None,
    period_from: str,
    period_to: str,
) -> int:
    pf = _parse_period(period_from)
    pt = _parse_period(period_to)
    if pt < pf:
        pf, pt = pt, pf
    parts_to = period_to.split("-")
    if len(parts_to) == 1:  # YYYY → expand to year-end
        pt = date(pt.year, 12, 31)
    elif len(parts_to) == 2:  # YYYY-MM → expand to month-end
        pt = date(pt.year, pt.month, monthrange(pt.year, pt.month)[1])
    return await client.count_publications(
        country=country,
        from_date=pf.isoformat(),
        to_date=pt.isoformat(),
    )
