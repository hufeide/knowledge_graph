"""Neo4j-backed graph store via the HTTP transactional Cypher endpoint.

No official driver is required: we POST Cypher statements to
``/db/<database>/tx/commit``. Enabled by setting ``KG_GRAPH_BACKEND=neo4j``.
"""
from __future__ import annotations

import re
from pathlib import Path

import requests

from ..config import Config, config
from ..schemas import GraphData
from .base import GraphStore

import hashlib

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_label(value: str) -> str:
    value = (value or "Entity").strip()
    return value if _LABEL_RE.match(value) else "Entity"


def _safe_rel(value: str) -> str:
    value = (value or "RELATED_TO").strip().upper()
    return value if _REL_RE.match(value) else "RELATED_TO"


def _event_id(ev) -> str:
    """Deterministic id for an event so identical events dedupe across builds."""
    raw = "||".join([ev.type, ev.subject, ev.object, ev.time, ev.summary])
    return "event::" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


class Neo4jStore(GraphStore):
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or config
        self.url = f"{self.cfg.neo4j_http_url.rstrip('/')}/db/{self.cfg.neo4j_database}/tx/commit"
        self.auth = (self.cfg.neo4j_user, self.cfg.neo4j_password)

    # ---- low level -----------------------------------------------------
    def _query(self, statements: list[dict]) -> list[list[list]]:
        """Run statements and return parsed rows (list of statements -> list of
        rows -> list of column values)."""
        if not statements:
            return []
        resp = requests.post(
            self.url,
            json={"statements": statements},
            auth=self.auth,
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        out: list[list[list]] = []
        for res in body.get("results", []):
            rows = [d.get("row", []) for d in res.get("data", [])]
            out.append(rows)
            if res.get("error"):
                raise RuntimeError(f"Neo4j error: {res['error']}")
        if body.get("errors"):
            raise RuntimeError(f"Neo4j errors: {body['errors']}")
        return out

    def _tx(self, statements: list[dict]) -> None:
        self._query(statements)

    # ---- public API ----------------------------------------------------
    def upsert_graph(self, graph: GraphData) -> None:
        stmts: list[dict] = []
        for e in graph.entities:
            label = _safe_label(e.type)
            stmts.append(
                {
                    "statement": (
                        f"MERGE (n:`{label}` {{name:$name}}) "
                        "SET n.type=$type, n.description=$desc, n.aliases=$aliases "
                        "WITH n, coalesce(n.sources, []) AS old "
                        "SET n.sources = old + [x IN $srcs WHERE NOT x IN old]"
                    ),
                    "parameters": {
                        "name": e.name,
                        "type": e.type,
                        "desc": e.description,
                        "aliases": e.aliases,
                        "srcs": list(e.sources),
                    },
                }
            )
        for r in graph.relations:
            rtype = _safe_rel(r.type)
            stmts.append(
                {
                    "statement": (
                        f"MATCH (a {{name:$s}}), (b {{name:$t}}) "
                        f"MERGE (a)-[rel:`{rtype}`]->(b) "
                        "WITH rel, coalesce(rel.sources, []) AS old "
                        "SET rel.sources = old + [x IN $srcs WHERE NOT x IN old]"
                    ),
                    "parameters": {"s": r.source, "t": r.target, "srcs": list(r.sources)},
                }
            )
        for ev in graph.events:
            eid = _event_id(ev)
            stmts.append(
                {
                    "statement": (
                        "MERGE (ev:Event {eid:$eid}) "
                        "SET ev.name=$name, ev.type='Event', ev.event_type=$etype, "
                        "ev.subject=$subj, ev.object=$obj, ev.time=$time, ev.summary=$sum, "
                        "ev.description=$sum, ev.aliases=[] "
                        "WITH ev, coalesce(ev.sources, []) AS old "
                        "SET ev.sources = old + [x IN $srcs WHERE NOT x IN old]"
                    ),
                    "parameters": {
                        "eid": eid,
                        "name": ev.summary or f"{ev.type}: {ev.subject}→{ev.object}",
                        "etype": ev.type,
                        "subj": ev.subject,
                        "obj": ev.object,
                        "time": ev.time,
                        "sum": ev.summary,
                        "srcs": list(ev.sources),
                    },
                }
            )
            for ep, role in ((ev.subject, "EVENT_SUBJECT"), (ev.object, "EVENT_OBJECT")):
                if not ep:
                    continue
                s, t = (ep, eid) if role == "EVENT_SUBJECT" else (eid, ep)
                stmts.append(
                    {
                        "statement": (
                            f"MATCH (a {{name:$s}}), (b:Event {{eid:$t}}) "
                            f"MERGE (a)-[:`{role}`]->(b)"
                        ),
                        "parameters": {"s": s, "t": t},
                    }
                )
        self._tx(stmts)

    def get_graph(self) -> tuple[list[dict], list[dict]]:
        nodes: list[dict] = []
        edges: list[dict] = []
        try:
            r = requests.post(
                self.url,
                json={
                    "statements": [
                        {
                            "statement": "MATCH (n) RETURN n.name, n.type, n.description, "
                            "n.aliases, n.sources, n.eid, n.event_type, n.subject, "
                            "n.object, n.time, n.summary",
                            "resultDataContents": ["row"],
                        },
                        {
                            "statement": "MATCH (a)-[r]->(b) RETURN coalesce(a.eid, a.name), "
                            "type(r), coalesce(b.eid, b.name), r.sources",
                            "resultDataContents": ["row"],
                        },
                    ]
                },
                auth=self.auth,
                timeout=120,
            )
            r.raise_for_status()
            body = r.json()
            results = body.get("results", [])
            if results:
                for row in results[0].get("data", []):
                    vals = row.get("row", [])
                    if len(vals) < 5:
                        continue
                    name = vals[0]
                    ntype = vals[1] or "Entity"
                    eid = vals[5]
                    node = {
                        "id": eid if eid else name,
                        "name": name,
                        "type": ntype,
                        "description": vals[2] or "",
                        "aliases": vals[3] or [],
                        "sources": vals[4] or [],
                    }
                    if ntype == "Event":
                        node.update(
                            {
                                "event_type": vals[6] or "",
                                "subject": vals[7] or "",
                                "object": vals[8] or "",
                                "time": vals[9] or "",
                                "summary": vals[10] or "",
                            }
                        )
                    nodes.append(node)
            if len(results) > 1:
                for row in results[1].get("data", []):
                    vals = row.get("row", [])
                    if len(vals) >= 3:
                        edges.append(
                            {
                                "source": vals[0],
                                "target": vals[2],
                                "type": vals[1],
                                "sources": vals[3] or [],
                            }
                        )
        except Exception as exc:  # pragma: no cover - depends on live server
            raise RuntimeError(f"Neo4j query failed: {exc}")
        return nodes, edges

    def delete_by_source(self, source: str) -> dict:
        pre = self._query([
            {"statement": "MATCH (n) WHERE $src IN coalesce(n.sources, []) RETURN count(n)",
             "parameters": {"src": source}},
            {"statement": "MATCH (a)-[r]->(b) WHERE $src IN coalesce(r.sources, []) RETURN count(r)",
             "parameters": {"src": source}},
        ])
        nodes_before = int(pre[0][0][0]) if pre and pre[0] else 0
        edges_before = int(pre[1][0][0]) if len(pre) > 1 and pre[1] else 0
        self._tx([
            {"statement": (
                "MATCH (a)-[r]->(b) "
                "WHERE $src IN coalesce(r.sources, []) "
                "   OR ANY(x IN coalesce(r.sources, []) WHERE x STARTS WITH $srcp) "
                "SET r.sources = [x IN r.sources WHERE NOT (x = $src OR x STARTS WITH $srcp)] "
                "WITH r WHERE size(coalesce(r.sources, [])) = 0 DELETE r"),
             "parameters": {"src": source, "srcp": source + "-"}},
            {"statement": (
                "MATCH (n) "
                "WHERE $src IN coalesce(n.sources, []) "
                "   OR ANY(x IN coalesce(n.sources, []) WHERE x STARTS WITH $srcp) "
                "SET n.sources = [x IN n.sources WHERE NOT (x = $src OR x STARTS WITH $srcp)] "
                "WITH n WHERE size(coalesce(n.sources, [])) = 0 DETACH DELETE n"),
             "parameters": {"src": source, "srcp": source + "-"}},
        ])
        return {"nodes": nodes_before, "edges": edges_before}

    def clear(self) -> None:
        self._tx([{"statement": "MATCH (n) DETACH DELETE n"}])
