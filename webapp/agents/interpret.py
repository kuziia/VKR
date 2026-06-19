"""Trends interpretation agent — streams a Claude commentary on the
current dashboard view (selected node, period, language, top articles,
field distribution).
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from ..llm.base import LLMNotConfigured, LLMRateLimited, get_llm

log = logging.getLogger(__name__)

__all__ = ["stream_interpretation", "LLMNotConfigured", "LLMRateLimited"]

SYSTEM = (
    "Ты — научный редактор-аналитик. Тебе показывают сводные данные "
    "о публикационной активности по выбранному узлу таксономии OpenAlex "
    "(домен/поле/подполе/тема). Дай аккуратный, ответственный комментарий "
    "на русском в Markdown: что видно по динамике, какие тематики/авторы "
    "выделяются, какие оговорки про данные стоит сделать. Не выдумывай "
    "цифры — используй ТОЛЬКО те, что приведены ниже."
)


def _format_points(points: list[dict[str, Any]], granularity: str) -> str:
    if not points:
        return "(нет точек)"
    lines = []
    for p in points:
        period = p.get("period")
        cnt = p.get("count")
        lines.append(f"  {period}\t{cnt}")
    if len(lines) > 24:
        head = lines[:12]
        tail = lines[-12:]
        return "\n".join(head + ["  …"] + tail)
    return "\n".join(lines)


def _format_top(items: list[dict[str, Any]]) -> str:
    if not items:
        return "(нет данных)"
    out = []
    for i, it in enumerate(items[:8], 1):
        title = (it.get("title") or "—")[:120]
        year = it.get("publication_year") or "—"
        cit = it.get("cited_by_count") or 0
        authors = ", ".join((it.get("authors") or [])[:3]) or "—"
        out.append(f"  {i}. ({year}, cit={cit}) {title} — {authors}")
    return "\n".join(out)


def _format_by_field(items: list[dict[str, Any]]) -> str:
    if not items:
        return "(нет данных по полям)"
    lines = []
    for it in items[:8]:
        lines.append(f"  {it.get('display_name', '?')}\t{it.get('count', 0)}")
    return "\n".join(lines)


def _format_bertrend(bt: dict[str, Any] | None) -> str | None:
    """Compact view of BERTrend signal — top emerging + strong topics with
    last-3-window dynamic. Returns None if BERTrend data is missing."""
    if not bt or not bt.get("available"):
        return None
    parts: list[str] = []
    parts.append(
        f"Окна: {bt.get('from_window')} → {bt.get('to_window')}. "
        f"Сигналы: {bt.get('signal_counts') or {}}."
    )

    def _topic_line(t: dict[str, Any]) -> str:
        words = ", ".join((t.get("words") or [])[:6])
        hist = t.get("history") or []
        tail = " ".join(f"{h['period'].split('-')[1]}={h['count']}" for h in hist[-4:])
        cn = t.get("centroid") or {}
        oa_label = cn.get("primary_topic") or "—"
        sim = cn.get("similarity")
        sim_str = f" sim={sim}" if sim is not None else ""
        return (
            f"  #{t['topic_id']:>3} {t['signal']:<8} total={t['total_docs']:<4} "
            f"peak={t['peak_count']:<4} | [{words}] | "
            f"last4: {tail} | centroid→OA: {oa_label}{sim_str}"
        )

    if bt.get("emerging"):
        parts.append("EMERGING (новые/набирающие силу):")
        for t in bt["emerging"][:6]:
            parts.append(_topic_line(t))
    if bt.get("strong"):
        parts.append("STRONG (устойчивые):")
        for t in bt["strong"][:6]:
            parts.append(_topic_line(t))
    return "\n".join(parts)


def build_prompt(payload: dict[str, Any]) -> str:
    trends = payload.get("trends") or {}
    top = payload.get("top") or {}
    byf = payload.get("by_field") or {}
    bertrend = payload.get("bertrend") or None

    label = trends.get("label") or "— все домены —"
    level = trends.get("level") or "all"
    granularity = trends.get("granularity") or "month"
    pf = trends.get("from") or "?"
    pt = trends.get("to") or "?"
    source = trends.get("source") or "openalex"
    country = trends.get("country") or "—"
    total = trends.get("total")
    n_points = len(trends.get("points") or [])

    points_block = _format_points(trends.get("points") or [], granularity)
    top_block = _format_top(top.get("items") or [])
    byfield_block = _format_by_field(byf.get("items") or [])
    bertrend_block = _format_bertrend(bertrend)

    bertrend_section = ""
    bertrend_task = ""
    if bertrend_block:
        bertrend_section = (
            "\n\nBERTREND (BERTopic over time, 12 окон ~год):\n"
            "ВАЖНО: BERTrend построен на ОТДЕЛЬНОМ русскоязычном корпусе "
            "(КиберЛенинка, ~34k статей разных дисциплин). Он НЕ ОТФИЛЬТРОВАН "
            "по выбранному выше узлу таксономии/языку/стране OpenAlex и НЕ "
            "обязан быть тематически связан с верхним контекстом. Это "
            "самостоятельный срез русскоязычной академической активности, "
            "приводимый рядом для общей картины.\n"
            f"{bertrend_block}\n"
        )
        bertrend_task = (
            "\n5. **BERTrend как независимый срез.** Кратко (1–2 пункта) "
            "отметь только то, что виднó в BERTrend как самостоятельная "
            "картина русскоязычной науки за последний год — какие 2–3 "
            "emerging-темы выглядят правдоподобно по динамике (рост в "
            "последних окнах) и какие, наоборот, скорее шум. НЕ ПЫТАЙСЯ "
            "увязывать темы BERTrend с фильтрами OpenAlex выше: это разные "
            "корпуса, отсутствие пересечения — норма, а не ошибка.\n"
        )

    return f"""КОНТЕКСТ:
- Источник: {source}
- Узел таксономии: {label} (level={level})
- Период: {pf} → {pt}, гранулярность: {granularity}
- Страна: {country}
- Всего публикаций за период: {total}
- Количество точек на графике: {n_points}

ДИНАМИКА ПУБЛИКАЦИЙ ({granularity}):
{points_block}

ТОП-{min(8, len(top.get('items') or []))} ПО ВЛИЯНИЮ ЗА ПЕРИОД:
{top_block}

РАСПРЕДЕЛЕНИЕ ПО ПОЛЯМ ЗА ПЕРИОД:
{byfield_block}{bertrend_section}

ЗАДАЧА:
Напиши краткий аналитический комментарий (Markdown, 5–8 коротких абзацев
или маркированных пунктов):

1. **Что видно по динамике** — рост/падение/сезонность; если данные за молодой
   календарный месяц — обязательно отметь, что это не финальная цифра
   (OpenAlex/OpenAIRE индексируют с задержкой).
2. **Что в топе** — какие темы/авторы выделяются; чем эти работы могут быть
   интересны.
3. **Распределение по полям** — какие подполя доминируют, есть ли что-то
   неожиданное.
4. **Оговорки** — про country-фильтр (если RU — выборка ограничена), про OA,
   про природу OpenAlex/OpenAIRE как агрегаторов метаданных.{bertrend_task}

Не выдумывай цифры, не комментируй то, чего не видишь в данных выше."""


async def stream_interpretation(payload: dict[str, Any]) -> AsyncIterator[str]:
    llm = get_llm()
    prompt = build_prompt(payload)
    log.info(
        "interpret prompt: level=%s, label=%s, points=%d, top=%d",
        (payload.get("trends") or {}).get("level"),
        (payload.get("trends") or {}).get("label"),
        len((payload.get("trends") or {}).get("points") or []),
        len((payload.get("top") or {}).get("items") or []),
    )
    async for chunk in llm.stream(
        prompt,
        system=SYSTEM,
        model="claude-haiku-4-5",
        temperature=0.7,
        max_tokens=1200,
    ):
        yield chunk
