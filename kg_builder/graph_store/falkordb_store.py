"""FalkorDB-backed graph store (the graph DB recommended by 思路.md).

FalkorDB speaks the Redis protocol; we use the ``redis`` client and the
``GRAPH.QUERY`` command. Enabled via ``KG_GRAPH_BACKEND=falkordb``.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re

from ..config import Config, config
from ..schemas import GraphData
from .base import GraphStore

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_label(value: str) -> str:
    value = (value or "Entity").strip()
    return value if _LABEL_RE.match(value) else "Entity"


def _safe_rel(value: str) -> str:
    value = (value or "RELATED_TO").strip().upper()
    return value if _REL_RE.match(value) else "RELATED_TO"


def _esc(value) -> str:
    """Escape a value for embedding inside a Cypher single-quoted string."""
    s = "" if value is None else str(value)
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _encode_list(lst) -> str:
    """Encode a list as base64(json) so it survives Cypher single-quote escaping
    intact. Storing plain ``json.dumps`` inside a single-quoted Cypher string is
    corrupted by quotes/backslashes in the data (e.g. a source filename like
    ``O'Reilly.pdf``), which previously made reads silently degrade to []."""
    payload = json.dumps(list(lst) if lst is not None else [], ensure_ascii=False)
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")


def _decode_list(raw) -> list:
    """Inverse of :func:`_encode_list`, with fallbacks for legacy plain-JSON."""
    if not raw:
        return []
    s = _to_str(raw)
    try:
        return json.loads(base64.b64decode(s).decode("utf-8"))
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return []


def _event_id(ev) -> str:
    """Deterministic id for an event so identical events dedupe across builds."""
    raw = "||".join([ev.type, ev.subject, ev.object, ev.time, ev.summary])
    return "event::" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


class FalkorStore(GraphStore):
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or config
        import redis

        self.r = redis.Redis(
            host=self.cfg.falkordb_host,
            port=self.cfg.falkordb_port,
            decode_responses=False,
            socket_connect_timeout=10,
        )
        self.graph = self.cfg.falkordb_graph

    def _query(self, cypher: str):
        return self.r.execute_command("GRAPH.QUERY", self.graph, cypher)

    # ---- public API ----------------------------------------------------
    def upsert_graph(self, graph: GraphData) -> None:
        for e in graph.entities:
            label = _safe_label(e.type)
            aliases = _encode_list(e.aliases)
            sources = _encode_list(list(e.sources))
            q = (
                f"MERGE (n:`{label}` {{name:'{_esc(e.name)}'}}) "
                f"SET n.type='{_esc(e.type)}', n.description='{_esc(e.description)}', "
                f"n.aliases='{_esc(aliases)}', n.sources='{_esc(sources)}'"
            )
            self._query(q)
        for rel in graph.relations:
            rtype = _safe_rel(rel.type)
            sources = _encode_list(list(rel.sources))
            # RedisGraph can't reference the MERGE-bound relationship variable in
            # SET, so create the edge first, then set its property in a MATCH.
            self._query(
                f"MATCH (a {{name:'{_esc(rel.source)}'}}), (b {{name:'{_esc(rel.target)}'}}) "
                f"MERGE (a)-[:`{rtype}`]->(b)"
            )
            self._query(
                f"MATCH (a {{name:'{_esc(rel.source)}'}})-[r:`{rtype}`]->"
                f"(b {{name:'{_esc(rel.target)}'}}) SET r.sources='{_esc(sources)}'"
            )
        for ev in graph.events:
            eid = _event_id(ev)
            name = ev.summary or f"{ev.type}: {ev.subject}→{ev.object}"
            sources = _encode_list(list(ev.sources))
            q = (
                f"MERGE (ev:Event {{eid:'{_esc(eid)}'}}) "
                f"SET ev.name='{_esc(name)}', ev.type='Event', ev.event_type='{_esc(ev.type)}', "
                f"ev.subject='{_esc(ev.subject)}', ev.object='{_esc(ev.object)}', "
                f"ev.time='{_esc(ev.time)}', ev.summary='{_esc(ev.summary)}', "
                f"ev.description='{_esc(ev.summary)}', ev.aliases='{_encode_list([])}', ev.sources='{sources}'"
            )
            self._query(q)
            for ep, role in ((ev.subject, "EVENT_SUBJECT"), (ev.object, "EVENT_OBJECT")):
                if not ep:
                    continue
                s, t = (ep, eid) if role == "EVENT_SUBJECT" else (eid, ep)
                q = (
                    f"MATCH (a {{name:'{_esc(s)}'}}), (b:Event {{eid:'{_esc(t)}'}}) "
                    f"MERGE (a)-[:`{role}`]->(b)"
                )
                self._query(q)

    def get_graph(self) -> tuple[list[dict], list[dict]]:
        nodes: list[dict] = []
        edges: list[dict] = []
        try:
            res = self._query(
                "MATCH (n) RETURN n.name, n.type, n.description, n.aliases, "
                "n.sources, n.eid, n.event_type, n.subject, n.object, n.time, n.summary"
            )
            for row in res[1]:
                name = _to_str(row[0])
                if name is None:
                    continue
                ntype = _to_str(row[1]) or "Entity"
                aliases = _decode_list(row[3] if len(row) > 3 else None)
                sources = _decode_list(row[4] if len(row) > 4 else None)
                eid = _to_str(row[5]) if len(row) > 5 else None
                node = {
                    "id": eid if eid else name,
                    "name": name,
                    "type": ntype,
                    "description": _to_str(row[2]) or "",
                    "aliases": aliases,
                    "sources": sources,
                }
                if ntype == "Event":
                    node.update(
                        {
                            "event_type": _to_str(row[6]) or "",
                            "subject": _to_str(row[7]) or "",
                            "object": _to_str(row[8]) or "",
                            "time": _to_str(row[9]) or "",
                            "summary": _to_str(row[10]) or "",
                        }
                    )
                nodes.append(node)
            res = self._query(
                "MATCH (a)-[r]->(b) RETURN coalesce(a.eid, a.name), type(r), "
                "coalesce(b.eid, b.name), r.sources"
            )
            for row in res[1]:
                src, tgt = _to_str(row[0]), _to_str(row[2])
                if src is None or tgt is None:
                    continue
                edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "type": _to_str(row[1]),
                        "sources": _decode_list(row[3] if len(row) > 3 else None),
                    }
                )
        except Exception as exc:  # pragma: no cover - depends on live server
            raise RuntimeError(f"FalkorDB query failed: {exc}")
        return nodes, edges

    def delete_by_source(self, source: str) -> dict:
        from .base import _src_equals, source_in

        nodes, edges = self.get_graph()
        removed = {"nodes": 0, "edges": 0}

        for e in edges:
            srcs = e.get("sources") or []
            if not source_in(source, srcs):
                continue
            remaining = [x for x in srcs if x != source]
            rel = _safe_rel(e["type"])
            if remaining:
                self._query(
                    f"MATCH (a {{name:'{_esc(e['source'])}'}})-[r:`{rel}`]->"
                    f"(b {{name:'{_esc(e['target'])}'}}) "
                    f"SET r.sources='{_esc(_encode_list(remaining))}'"
                )
            else:
                self._query(
                    f"MATCH (a {{name:'{_esc(e['source'])}'}})-[r:`{rel}`]->"
                    f"(b {{name:'{_esc(e['target'])}'}}) DELETE r"
                )
                removed["edges"] += 1

        for n in nodes:
            srcs = n.get("sources") or []
            if not source_in(source, srcs):
                continue
            remaining = [x for x in srcs if not _src_equals(source, x)]
            if remaining:
                self._query(
                    f"MATCH (n {{name:'{_esc(n['id'])}'}}) "
                    f"SET n.sources='{_esc(_encode_list(remaining))}'"
                )
            else:
                self._query(f"MATCH (n {{name:'{_esc(n['id'])}'}}) DETACH DELETE n")
                removed["nodes"] += 1

        return removed

    def clear(self) -> None:
        try:
            self._query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass


def _to_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)
