"""Thin async OpenAlex client.

* `mailto` for the politeness pool (better rate limit, stable).
* `tenacity`-backed retry on 5xx / network errors.
* `select=` projection helper to avoid 30 KB rows.
* Cursor-pagination iterator.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..settings import settings

log = logging.getLogger(__name__)

_TRANSIENT = (httpx.TransportError, httpx.RemoteProtocolError, httpx.ReadTimeout)


class OpenAlexError(RuntimeError):
    pass


class OpenAlexClient:
    def __init__(self, base: str | None = None, mailto: str | None = None) -> None:
        self.base = (base or settings.openalex_base).rstrip("/")
        self.mailto = mailto or settings.mailto
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=10.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            headers={"User-Agent": f"nauka-monitor/0.1 ({self.mailto})"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── core ──
    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "mailto": self.mailto}
        url = f"{self.base}{path}"
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(_TRANSIENT),
            reraise=True,
        ):
            with attempt:
                r = await self._client.get(url, params=params)
                if 500 <= r.status_code < 600:
                    raise httpx.TransportError(f"upstream {r.status_code}")
                if r.status_code == 429:
                    raise httpx.TransportError("rate-limited")
                if r.status_code >= 400:
                    raise OpenAlexError(f"{r.status_code}: {r.text[:200]}")
                return r.json()
        raise OpenAlexError("unreachable")  # pragma: no cover

    # ── high-level helpers ──
    async def works_meta(self, *, filters: str) -> int:
        """Just the count, cheap. per-page=1."""
        data = await self._get(
            "/works",
            {"filter": filters, "per-page": 1, "select": "id"},
        )
        return int(data.get("meta", {}).get("count") or 0)

    async def works_top(
        self,
        *,
        filters: str,
        sort: str = "cited_by_count:desc",
        per_page: int = 20,
        select: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"filter": filters, "sort": sort, "per-page": per_page}
        if select:
            params["select"] = select
        data = await self._get("/works", params)
        return list(data.get("results") or [])

    async def works_search(
        self,
        *,
        query: str,
        filters: str | None = None,
        per_page: int = 20,
        select: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Free-text search over title+abstract, sorted by relevance.

        Returns (results, total_count). OpenAlex caps `per_page` at 200.
        """
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(max(per_page, 1), 200),
            "sort": "relevance_score:desc",
        }
        if filters:
            params["filter"] = filters
        if select:
            params["select"] = select
        data = await self._get("/works", params)
        return list(data.get("results") or []), int((data.get("meta") or {}).get("count") or 0)

    async def works_group_by(
        self,
        *,
        filters: str,
        group_by: str,
        per_page: int = 200,
    ) -> list[dict[str, Any]]:
        data = await self._get(
            "/works",
            {"filter": filters, "group_by": group_by, "per-page": per_page},
        )
        return list(data.get("group_by") or [])

    async def works_iter(
        self,
        *,
        filters: str,
        select: str,
        per_page: int = 200,
    ) -> AsyncIterator[dict[str, Any]]:
        params: dict[str, Any] = {
            "filter": filters,
            "per-page": per_page,
            "select": select,
            "cursor": "*",
        }
        while True:
            data = await self._get("/works", params)
            for w in data.get("results") or []:
                yield w
            nxt = data.get("meta", {}).get("next_cursor")
            if not nxt:
                return
            params["cursor"] = nxt

    async def work(self, openalex_id: str, *, select: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if select:
            params["select"] = select
        return await self._get(f"/works/{openalex_id}", params)

    async def author(self, author_id: str, *, select: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if select:
            params["select"] = select
        return await self._get(f"/authors/{author_id}", params)


_singleton: OpenAlexClient | None = None


def get_client() -> OpenAlexClient:
    global _singleton
    if _singleton is None:
        _singleton = OpenAlexClient()
    return _singleton


async def close_client() -> None:
    global _singleton
    if _singleton is not None:
        await _singleton.aclose()
        _singleton = None
