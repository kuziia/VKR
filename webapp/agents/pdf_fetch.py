"""PDF acquisition: arxiv → OpenAlex OA pdf_url → Unpaywall.

Two public functions:
    list_sources(article)  → list of {kind, url, label} (UI display)
    fetch_pdf(article)     → (pdf_bytes, source_kind, source_url) | (None, None, None)

PDF bytes are cached on disk under settings.cache_dir/pdfs/<openalex_id>.pdf
for 30 days; a sentinel `<id>.miss` records that all sources failed.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from ..settings import settings

log = logging.getLogger(__name__)

PDF_TTL_SEC = 30 * 24 * 60 * 60
MAX_PDF_BYTES = 25 * 1024 * 1024  # 25 MB upper bound, prototype safety

_pdf_dir = settings.cache_dir / "pdfs"
_pdf_dir.mkdir(parents=True, exist_ok=True)


def _doi_clean(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.lower().startswith(prefix):
            d = d[len(prefix):]
            break
    return d.strip().strip("/") or None


async def _unpaywall_pdf_url(doi: str) -> str | None:
    url = f"https://api.unpaywall.org/v2/{doi}"
    async with httpx.AsyncClient(timeout=20.0) as c:
        try:
            r = await c.get(url, params={"email": settings.mailto})
        except httpx.HTTPError as e:
            log.warning("unpaywall request failed for %s: %s", doi, e)
            return None
    if r.status_code != 200:
        return None
    data = r.json() or {}
    best = data.get("best_oa_location") or {}
    if isinstance(best, dict) and best.get("url_for_pdf"):
        return best["url_for_pdf"]
    for loc in data.get("oa_locations") or []:
        if isinstance(loc, dict) and loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    return None


async def list_sources(article: dict[str, Any]) -> list[dict[str, str]]:
    """Display-only list of available PDF sources for the UI."""
    out: list[dict[str, str]] = []
    arxiv_id = (article.get("ids") or {}).get("arxiv")
    if arxiv_id:
        out.append({"kind": "arxiv", "url": f"https://arxiv.org/pdf/{arxiv_id}.pdf", "label": f"arXiv {arxiv_id}"})

    oa = article.get("open_access") or {}
    if oa.get("pdf_url"):
        out.append({"kind": "openalex_oa", "url": oa["pdf_url"], "label": "OpenAlex OA"})
    elif oa.get("landing_page_url"):
        out.append({"kind": "landing", "url": oa["landing_page_url"], "label": "Landing page"})

    doi = _doi_clean(article.get("doi"))
    if doi:
        out.append({"kind": "doi", "url": f"https://doi.org/{doi}", "label": f"DOI {doi}"})

    return out


async def _try_download(url: str) -> bytes | None:
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=10.0),
            follow_redirects=True,
            headers={"User-Agent": f"nauka-monitor/0.1 ({settings.mailto})"},
        ) as c:
            r = await c.get(url)
            if r.status_code != 200:
                log.info("PDF fetch %s → %d", url, r.status_code)
                return None
            ctype = (r.headers.get("content-type") or "").lower()
            body = r.content
            if "pdf" not in ctype and not body.startswith(b"%PDF"):
                log.info("PDF fetch %s → not a PDF (ctype=%s)", url, ctype)
                return None
            if len(body) > MAX_PDF_BYTES:
                log.info("PDF fetch %s too large: %d", url, len(body))
                return None
            return body
    except httpx.HTTPError as e:
        log.info("PDF fetch %s failed: %s", url, e)
        return None


def _cache_paths(openalex_id: str) -> tuple[Path, Path]:
    return _pdf_dir / f"{openalex_id}.pdf", _pdf_dir / f"{openalex_id}.miss"


def _read_cached(openalex_id: str) -> bytes | None:
    pdf, miss = _cache_paths(openalex_id)
    if pdf.exists() and (time.time() - pdf.stat().st_mtime) < PDF_TTL_SEC:
        return pdf.read_bytes()
    if miss.exists() and (time.time() - miss.stat().st_mtime) < PDF_TTL_SEC:
        return b""  # sentinel: known miss
    return None


async def fetch_pdf(article: dict[str, Any]) -> tuple[bytes | None, str | None, str | None]:
    openalex_id = article.get("openalex_id") or ""
    if openalex_id:
        cached = _read_cached(openalex_id)
        if cached is not None:
            return (cached or None), "cache", None

    candidates: list[tuple[str, str]] = []  # (kind, url)
    arxiv_id = (article.get("ids") or {}).get("arxiv")
    if arxiv_id:
        candidates.append(("arxiv", f"https://arxiv.org/pdf/{arxiv_id}.pdf"))
    oa = article.get("open_access") or {}
    if oa.get("pdf_url"):
        candidates.append(("openalex_oa", oa["pdf_url"]))
    doi = _doi_clean(article.get("doi"))
    if doi:
        unpay = await _unpaywall_pdf_url(doi)
        if unpay:
            candidates.append(("unpaywall", unpay))

    for kind, url in candidates:
        body = await _try_download(url)
        if body:
            if openalex_id:
                _cache_paths(openalex_id)[0].write_bytes(body)
            return body, kind, url
        await asyncio.sleep(0)

    if openalex_id:
        _cache_paths(openalex_id)[1].write_bytes(b"")
    return None, None, None
