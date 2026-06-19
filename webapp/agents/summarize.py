"""Article summarization via Claude.

Path 1: PDF available  → extract text via pymupdf → chunked summary.
Path 2: PDF unavailable → summarize abstract (with explicit `source` flag).
"""
from __future__ import annotations

import logging
from typing import Any

from ..llm.base import LLMNotConfigured, LLMRateLimited, get_llm
from . import pdf_fetch

log = logging.getLogger(__name__)

# Re-export for the router so it can catch via single import.
__all__ = ["summarize_article", "LLMNotConfigured", "LLMRateLimited"]

CHUNK_CHARS = 12_000  # ~3-4k tokens per chunk, fits sonnet comfortably
MAX_CHUNKS = 6  # cap big PDFs to keep subscription quota in check

SYSTEM = (
    "Ты — научный редактор. Делай саммари статьи на русском языке, "
    "сохраняя терминологию оригинала там, где она устоялась (можешь оставлять "
    "английские термины в скобках). Структурируй ответ в Markdown."
)

PROMPT_FULL = """Ниже фрагмент научной статьи. Заголовок: {title}.

Сделай краткое саммари этого фрагмента на 4–6 пунктах: задача, метод,
данные/эксперименты, ключевые цифры/находки. Не выдумывай — только то,
что прямо в тексте. Markdown, маркированный список.

ФРАГМЕНТ:
{chunk}"""

PROMPT_ABSTRACT = """Заголовок: {title}
Авторы: {authors}
Год: {year}
Журнал/площадка: {venue}
DOI: {doi}

Аннотация:
{abstract}

Сделай структурированное саммари статьи на русском в Markdown:
- **Тема и контекст** (1–2 предложения)
- **Метод** (что предложено / как устроено)
- **Результаты** (что получили; цифры, если есть)
- **Зачем читать** (1 предложение)

Опирайся ТОЛЬКО на аннотацию выше; если чего-то нет — пометь
«не указано в аннотации»."""

PROMPT_REDUCE = """Ниже несколько локальных саммари по фрагментам одной статьи.
Сведи их в одно цельное саммари в Markdown: задача → метод → данные →
результаты → ограничения (если есть). Не повторяй пункты, не выдумывай.

Заголовок статьи: {title}

ЛОКАЛЬНЫЕ САММАРИ:
{partials}"""


def _extract_text_from_pdf(pdf_bytes: bytes) -> str | None:
    try:
        import fitz  # type: ignore  # PyMuPDF
    except ImportError:
        log.warning("PyMuPDF not installed; cannot extract PDF text")
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        log.warning("PyMuPDF open failed: %s", e)
        return None
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "\n".join(parts).strip() or None
    finally:
        doc.close()


def _chunk_text(text: str, size: int = CHUNK_CHARS, max_chunks: int = MAX_CHUNKS) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(start + size, len(text))
        if end < len(text):
            soft = text.rfind("\n", start + size // 2, end)
            if soft > start:
                end = soft
        chunks.append(text[start:end])
        start = end
    return chunks


async def _summarize_pdf(text: str, title: str) -> str:
    llm = get_llm()
    chunks = _chunk_text(text)

    if len(chunks) == 1:
        return await llm.complete(
            PROMPT_FULL.format(title=title, chunk=chunks[0]),
            system=SYSTEM,
            model="claude-haiku-4-5",
            temperature=0.2,
            max_tokens=1500,
        )

    partials: list[str] = []
    for i, ch in enumerate(chunks, 1):
        log.info("summarize chunk %d/%d (%d chars)", i, len(chunks), len(ch))
        partials.append(
            await llm.complete(
                PROMPT_FULL.format(title=title, chunk=ch),
                system=SYSTEM,
                model="claude-haiku-4-5",
                temperature=0.2,
                max_tokens=900,
            )
        )

    joined = "\n\n---\n\n".join(f"### Фрагмент {i+1}\n{p}" for i, p in enumerate(partials))
    return await llm.complete(
        PROMPT_REDUCE.format(title=title, partials=joined),
        system=SYSTEM,
        model="claude-haiku-4-5",
        temperature=0.2,
        max_tokens=1500,
    )


async def _summarize_abstract(article: dict[str, Any]) -> str:
    abstract = article.get("abstract")
    if not abstract:
        return (
            "_У статьи нет ни PDF, ни аннотации в OpenAlex — суммаризировать нечего._"
        )
    llm = get_llm()
    authors = ", ".join(a["name"] for a in (article.get("authors") or [])[:5]) or "—"
    venue = (article.get("venue") or {}).get("display_name") or "—"
    return await llm.complete(
        PROMPT_ABSTRACT.format(
            title=article.get("title") or "—",
            authors=authors,
            year=article.get("publication_year") or "—",
            venue=venue,
            doi=article.get("doi") or "—",
            abstract=abstract,
        ),
        system=SYSTEM,
        model="claude-haiku-4-5",
        temperature=0.2,
        max_tokens=1200,
    )


async def summarize_article(article: dict[str, Any]) -> dict[str, Any]:
    """Returns {summary_md, source: "pdf"|"abstract"|"none", pdf_url, oa_status}."""
    pdf_bytes, kind, src_url = await pdf_fetch.fetch_pdf(article)

    if pdf_bytes:
        text = _extract_text_from_pdf(pdf_bytes)
        if text and len(text) > 300:
            summary = await _summarize_pdf(text, title=article.get("title") or "—")
            return {
                "summary_md": summary,
                "source": "pdf",
                "pdf_kind": kind,
                "pdf_url": src_url,
                "oa_status": (article.get("open_access") or {}).get("oa_status"),
            }
        log.info("PDF fetched but text extraction empty/short → fallback to abstract")

    summary = await _summarize_abstract(article)
    return {
        "summary_md": summary,
        "source": "abstract" if article.get("abstract") else "none",
        "pdf_kind": None,
        "pdf_url": None,
        "oa_status": (article.get("open_access") or {}).get("oa_status"),
    }
