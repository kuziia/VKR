"""LLM agent endpoints. Streamed via Server-Sent Events."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..agents import interpret

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["agent"])


class TrendsLite(BaseModel):
    level: str
    id: str | None = None
    label: str | None = None
    granularity: str | None = None
    lang: str | None = None
    from_: str | None = None
    to: str | None = None
    total: int | None = None
    points: list[dict[str, Any]] = []

    model_config = {"populate_by_name": True}

    def dump(self) -> dict[str, Any]:
        d = self.model_dump()
        d["from"] = d.pop("from_", None)
        return d


class TopLite(BaseModel):
    items: list[dict[str, Any]] = []


class ByFieldLite(BaseModel):
    items: list[dict[str, Any]] = []


class InterpretBody(BaseModel):
    trends: dict[str, Any]
    top: dict[str, Any] | None = None
    by_field: dict[str, Any] | None = None
    bertrend: dict[str, Any] | None = None


def _sse_event(text: str) -> bytes:
    """SSE: 'data: <json-escaped text>\\n\\n'."""
    payload = json.dumps({"chunk": text}, ensure_ascii=False)
    return f"data: {payload}\n\n".encode("utf-8")


def _sse_error(message: str) -> bytes:
    payload = json.dumps({"error": message}, ensure_ascii=False)
    return f"event: error\ndata: {payload}\n\n".encode("utf-8")


def _sse_done() -> bytes:
    return b"event: done\ndata: {}\n\n"


async def _safe_stream(payload: dict[str, Any]) -> AsyncIterator[bytes]:
    try:
        async for chunk in interpret.stream_interpretation(payload):
            if chunk:
                yield _sse_event(chunk)
    except interpret.LLMNotConfigured as e:
        log.warning("interpret: LLM not configured: %s", e)
        yield _sse_error(f"LLM not configured: {e}")
    except interpret.LLMRateLimited as e:
        log.warning("interpret: rate limited: %s", e)
        yield _sse_error(f"Лимит подписки исчерпан, попробуй позже: {e}")
    except Exception as e:
        log.exception("interpret stream failed")
        yield _sse_error(f"Ошибка генерации: {e}")
    finally:
        yield _sse_done()


@router.post("/interpret-trends")
async def interpret_trends(body: InterpretBody) -> StreamingResponse:
    if not body.trends:
        raise HTTPException(400, "trends payload is required")

    payload = {
        "trends": body.trends,
        "top": body.top or {},
        "by_field": body.by_field or {},
        "bertrend": body.bertrend,
    }
    return StreamingResponse(
        _safe_stream(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable proxy buffering
            "Connection": "keep-alive",
        },
    )
