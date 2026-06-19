"""SQLAlchemy async setup + tiny kv-cache helpers.

Schema follows plan §2.7:
    trends_cache(cache_key PK, payload JSON, fetched_at TS)
    works_cache(openalex_id PK, payload JSON, fetched_at TS)
    search_jobs(job_id PK, query, status, result, created_at)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from typing import Any

from sqlalchemy import DateTime, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..settings import settings

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class TrendsCache(Base):
    __tablename__ = "trends_cache"
    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class WorksCache(Base):
    __tablename__ = "works_cache"
    openalex_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class SearchJob(Base):
    __tablename__ = "search_jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


_engine = create_async_engine(settings.db_url, future=True, echo=False)
SessionLocal = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def cache_key(*parts: Any) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


async def get_trends_cache(key: str, *, ttl_sec: int) -> dict[str, Any] | None:
    cutoff = dt.datetime.utcnow() - dt.timedelta(seconds=ttl_sec)
    async with SessionLocal() as s:
        row = (await s.execute(select(TrendsCache).where(TrendsCache.cache_key == key))).scalar_one_or_none()
        if row is None or row.fetched_at < cutoff:
            return None
        return json.loads(row.payload)


async def put_trends_cache(key: str, payload: dict[str, Any]) -> None:
    async with SessionLocal() as s:
        existing = await s.get(TrendsCache, key)
        if existing is None:
            s.add(TrendsCache(cache_key=key, payload=json.dumps(payload, ensure_ascii=False), fetched_at=dt.datetime.utcnow()))
        else:
            existing.payload = json.dumps(payload, ensure_ascii=False)
            existing.fetched_at = dt.datetime.utcnow()
        await s.commit()


async def get_work_cache(openalex_id: str, *, ttl_sec: int) -> dict[str, Any] | None:
    cutoff = dt.datetime.utcnow() - dt.timedelta(seconds=ttl_sec)
    async with SessionLocal() as s:
        row = await s.get(WorksCache, openalex_id)
        if row is None or row.fetched_at < cutoff:
            return None
        return json.loads(row.payload)


async def put_work_cache(openalex_id: str, payload: dict[str, Any]) -> None:
    async with SessionLocal() as s:
        existing = await s.get(WorksCache, openalex_id)
        if existing is None:
            s.add(WorksCache(openalex_id=openalex_id, payload=json.dumps(payload, ensure_ascii=False), fetched_at=dt.datetime.utcnow()))
        else:
            existing.payload = json.dumps(payload, ensure_ascii=False)
            existing.fetched_at = dt.datetime.utcnow()
        await s.commit()
