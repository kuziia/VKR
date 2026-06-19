"""Citation graph builder.

For a root work, performs BFS up to `depth` hops in two directions:
  • outgoing — `referenced_works` (this work cites these)
  • incoming — `cites:Wxxx` filter (works that cite this one)

Limits per node to keep payload bounded; nodes get a `depth` field for FE.
Returns: {nodes: [{id, title, year, cited_by_count, depth}], edges: [...]}.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..openalex.client import OpenAlexClient

log = logging.getLogger(__name__)

NODE_SELECT = "id,title,publication_year,cited_by_count,referenced_works"

MAX_TOTAL_NODES = 80
DEFAULT_FANOUT = 8  # max neighbors per node per direction


def _short(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None
    return url_or_id.rsplit("/", 1)[-1] or None


async def _fetch_work(client: OpenAlexClient, wid: str) -> dict[str, Any]:
    return await client.work(wid, select=NODE_SELECT)


async def _fetch_works_batch(
    client: OpenAlexClient, ids: list[str], fanout: int
) -> list[dict[str, Any]]:
    """Get many works by ID in one /works?filter=ids.openalex:... call."""
    if not ids:
        return []
    # OpenAlex supports `openalex:Wxxx|Wyyy|...` in filter; cap the OR list size.
    ids = ids[:fanout]
    flt = f"openalex:{'|'.join(ids)}"
    return await client.works_top(filters=flt, per_page=len(ids), sort="cited_by_count:desc", select=NODE_SELECT)


async def _fetch_citing(
    client: OpenAlexClient, wid: str, fanout: int
) -> list[dict[str, Any]]:
    """Top-N most-cited works that cite `wid`."""
    return await client.works_top(
        filters=f"cites:{wid}",
        sort="cited_by_count:desc",
        per_page=fanout,
        select=NODE_SELECT,
    )


def _project_node(w: dict[str, Any], depth: int) -> dict[str, Any]:
    return {
        "id": _short(w.get("id")) or "?",
        "title": w.get("title") or "(no title)",
        "year": w.get("publication_year"),
        "cited_by_count": int(w.get("cited_by_count") or 0),
        "depth": depth,
    }


async def build_graph(
    client: OpenAlexClient,
    *,
    root_id: str,
    depth: int = 1,
    fanout: int = DEFAULT_FANOUT,
) -> dict[str, Any]:
    """Bidirectional BFS. depth=1 → root + its refs/cites neighborhood."""
    depth = max(1, min(depth, 3))
    fanout = max(2, min(fanout, 20))

    nodes: dict[str, dict[str, Any]] = {}  # id → node payload
    edges: list[dict[str, str]] = []  # {source, target, kind: 'refs'|'cites'}
    seen_edges: set[tuple[str, str, str]] = set()

    root = await _fetch_work(client, root_id)
    root_short = _short(root.get("id")) or root_id
    nodes[root_short] = _project_node(root, depth=0)

    frontier: list[tuple[str, dict[str, Any], int]] = [(root_short, root, 0)]

    while frontier:
        next_frontier: list[tuple[str, dict[str, Any], int]] = []

        # Collect outgoing/incoming work fetches in parallel for the whole frontier
        async def expand(item: tuple[str, dict[str, Any], int]):
            sid, work, sd = item
            if sd >= depth or len(nodes) >= MAX_TOTAL_NODES:
                return [], []
            # Outgoing: referenced_works are full URLs
            ref_urls = work.get("referenced_works") or []
            ref_ids = [_short(u) for u in ref_urls if u]
            ref_ids = [r for r in ref_ids if r]
            outgoing_task = _fetch_works_batch(client, ref_ids, fanout)
            incoming_task = _fetch_citing(client, sid, fanout)
            outs, ins = await asyncio.gather(outgoing_task, incoming_task, return_exceptions=True)
            if isinstance(outs, Exception):
                log.warning("graph: refs fetch failed for %s: %s", sid, outs)
                outs = []
            if isinstance(ins, Exception):
                log.warning("graph: cites fetch failed for %s: %s", sid, ins)
                ins = []
            return outs, ins

        results = await asyncio.gather(*(expand(it) for it in frontier))

        for (sid, _w, sd), (outs, ins) in zip(frontier, results):
            for w in outs:
                if len(nodes) >= MAX_TOTAL_NODES:
                    break
                tid = _short(w.get("id"))
                if not tid:
                    continue
                if tid not in nodes:
                    nodes[tid] = _project_node(w, depth=sd + 1)
                    next_frontier.append((tid, w, sd + 1))
                edge_key = (sid, tid, "refs")
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({"source": sid, "target": tid, "kind": "refs"})
            for w in ins:
                if len(nodes) >= MAX_TOTAL_NODES:
                    break
                tid = _short(w.get("id"))
                if not tid:
                    continue
                if tid not in nodes:
                    nodes[tid] = _project_node(w, depth=sd + 1)
                    next_frontier.append((tid, w, sd + 1))
                # incoming → "tid cites sid"
                edge_key = (tid, sid, "cites")
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({"source": tid, "target": sid, "kind": "cites"})

        frontier = next_frontier

    log.info(
        "graph: root=%s depth=%d fanout=%d → %d nodes, %d edges",
        root_id, depth, fanout, len(nodes), len(edges),
    )
    return {
        "root_id": root_short,
        "depth": depth,
        "fanout": fanout,
        "nodes": list(nodes.values()),
        "edges": edges,
    }
