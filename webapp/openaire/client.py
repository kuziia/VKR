"""Thin async OpenAIRE client.

Public API: https://api.openaire.eu/search/publications
- No API key for low-volume use.
- Filter params: country, fromDateAccepted, toDateAccepted, doi, etc.
- Note: `language` is NOT supported as a search filter (verified).
"""
from __future__ import annotations

import logging
from typing import Any

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
BASE = "https://api.openaire.eu"


class OpenAireError(RuntimeError):
    pass


class OpenAireClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            headers={"User-Agent": f"nauka-monitor/0.1 ({settings.mailto})"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{BASE}{path}"
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=6),
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
                    raise OpenAireError(f"{r.status_code}: {r.text[:200]}")
                return r.json()
        raise OpenAireError("unreachable")  # pragma: no cover

    async def count_publications(
        self,
        *,
        country: str | None,
        from_date: str | None,
        to_date: str | None,
    ) -> int:
        """Total count of publications matching filters. Cheap (size=1)."""
        params: dict[str, Any] = {"size": 1, "format": "json"}
        if country:
            params["country"] = country.upper()
        if from_date:
            params["fromDateAccepted"] = from_date
        if to_date:
            params["toDateAccepted"] = to_date
        data = await self._get("/search/publications", params)
        header = (data.get("response") or {}).get("header") or {}
        total = header.get("total")
        if isinstance(total, dict):
            return int(total.get("$") or total.get("value") or 0)
        try:
            return int(total or 0)
        except (TypeError, ValueError):
            return 0


_singleton: OpenAireClient | None = None


def get_openaire() -> OpenAireClient:
    global _singleton
    if _singleton is None:
        _singleton = OpenAireClient()
    return _singleton


async def close_openaire() -> None:
    global _singleton
    if _singleton is not None:
        await _singleton.aclose()
        _singleton = None
