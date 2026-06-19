"""Taxonomy: domain -> field -> subfield -> topic.

OpenAlex publishes the four-level taxonomy via separate endpoints, but every
`/topics` row already includes its subfield/field/domain refs. We page through
`/topics` once on first startup, build the tree, and cache it on disk
(`webapp_cache/taxonomy.json`). Subsequent loads are instant.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .settings import settings

log = logging.getLogger(__name__)


@dataclass
class TopicNode:
    id: str
    display_name: str
    keywords: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "keywords": self.keywords,
            "description": self.description,
        }


@dataclass
class SubfieldNode:
    id: str
    display_name: str
    topics: list[TopicNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "topics": [t.to_dict() for t in self.topics],
        }


@dataclass
class FieldNode:
    id: str
    display_name: str
    subfields: list[SubfieldNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "subfields": [s.to_dict() for s in self.subfields],
        }


@dataclass
class DomainNode:
    id: str
    display_name: str
    fields: list[FieldNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "fields": [f.to_dict() for f in self.fields],
        }


@dataclass
class Taxonomy:
    domains: list[DomainNode]

    def to_dict(self) -> dict[str, Any]:
        return {"domains": [d.to_dict() for d in self.domains]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Taxonomy":
        domains = []
        for d in data["domains"]:
            fields_ = []
            for f in d["fields"]:
                subfields_ = []
                for s in f["subfields"]:
                    topics_ = [TopicNode(**t) for t in s["topics"]]
                    subfields_.append(
                        SubfieldNode(id=s["id"], display_name=s["display_name"], topics=topics_)
                    )
                fields_.append(
                    FieldNode(id=f["id"], display_name=f["display_name"], subfields=subfields_)
                )
            domains.append(
                DomainNode(id=d["id"], display_name=d["display_name"], fields=fields_)
            )
        return cls(domains=domains)


def _short_id(openalex_url: str) -> str:
    """Extract `T10028` from `https://openalex.org/T10028`."""
    return openalex_url.rsplit("/", 1)[-1] if openalex_url else ""


async def _fetch_all_topics() -> list[dict[str, Any]]:
    """Cursor-paginate through /topics. ~4500 topics @ 200/page => ~23 pages."""
    url = f"{settings.openalex_base}/topics"
    params: dict[str, Any] = {
        "per-page": 200,
        "cursor": "*",
        "select": "id,display_name,description,keywords,subfield,field,domain",
        "mailto": settings.mailto,
    }
    out: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            out.extend(data["results"])
            nxt = data.get("meta", {}).get("next_cursor")
            if not nxt:
                break
            params["cursor"] = nxt
    log.info("Fetched %d topics from OpenAlex", len(out))
    return out


def _build_tree(topics: list[dict[str, Any]]) -> Taxonomy:
    domains: dict[str, DomainNode] = {}
    for t in topics:
        d = t["domain"]
        f = t["field"]
        s = t["subfield"]
        d_id = _short_id(d["id"])
        f_id = _short_id(f["id"])
        s_id = _short_id(s["id"])
        t_id = _short_id(t["id"])

        if d_id not in domains:
            domains[d_id] = DomainNode(id=d_id, display_name=d["display_name"])
        dom = domains[d_id]

        fld = next((x for x in dom.fields if x.id == f_id), None)
        if fld is None:
            fld = FieldNode(id=f_id, display_name=f["display_name"])
            dom.fields.append(fld)

        sub = next((x for x in fld.subfields if x.id == s_id), None)
        if sub is None:
            sub = SubfieldNode(id=s_id, display_name=s["display_name"])
            fld.subfields.append(sub)

        sub.topics.append(
            TopicNode(
                id=t_id,
                display_name=t["display_name"],
                keywords=list(t.get("keywords") or []),
                description=t.get("description") or "",
            )
        )

    for dom in domains.values():
        dom.fields.sort(key=lambda x: x.display_name)
        for fld in dom.fields:
            fld.subfields.sort(key=lambda x: x.display_name)
            for sub in fld.subfields:
                sub.topics.sort(key=lambda x: x.display_name)
    ordered = sorted(domains.values(), key=lambda x: x.display_name)
    return Taxonomy(domains=ordered)


async def load_taxonomy(force_refresh: bool = False) -> Taxonomy:
    cache: Path = settings.taxonomy_cache
    if cache.exists() and not force_refresh:
        log.info("Loading taxonomy from cache %s", cache)
        return Taxonomy.from_dict(json.loads(cache.read_text("utf-8")))

    log.info("Fetching taxonomy from OpenAlex")
    topics = await _fetch_all_topics()
    tax = _build_tree(topics)
    cache.write_text(json.dumps(tax.to_dict(), ensure_ascii=False), encoding="utf-8")
    return tax
