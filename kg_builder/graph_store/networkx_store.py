"""NetworkX-backed graph store (default, dependency-free, JSON-persisted)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx

from ..schemas import GraphData
from .base import GraphStore


def _event_id(ev) -> str:
    """Deterministic id for an event so identical events dedupe across builds."""
    raw = "||".join([ev.type, ev.subject, ev.object, ev.time, ev.summary])
    return "event::" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


class NetworkXStore(GraphStore):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.G = nx.MultiDiGraph()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            for n in data.get("nodes", []):
                self.G.add_node(n["id"], **n.get("props", {}))
            for e in data.get("edges", []):
                self.G.add_edge(e["source"], e["target"], **e.get("props", {}))

    def _save(self) -> None:
        nodes = [{"id": n, "props": self.G.nodes[n]} for n in self.G.nodes]
        edges = [
            {"source": u, "target": v, "props": d}
            for u, v, d in self.G.edges(data=True)
        ]
        self.path.write_text(
            json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert_graph(self, graph: GraphData) -> None:
        # ---- entities: merge provenance (sources) on update ----
        for ent in graph.entities:
            nid = ent.name
            srcs = set(ent.sources)
            if self.G.has_node(nid):
                attrs = self.G.nodes[nid]
                attrs["name"] = ent.name
                attrs["type"] = ent.type
                attrs["description"] = ent.description
                attrs["aliases"] = ent.aliases
                attrs["sources"] = sorted(set(attrs.get("sources", [])) | srcs)
            else:
                self.G.add_node(
                    nid,
                    name=ent.name,
                    type=ent.type,
                    description=ent.description,
                    aliases=ent.aliases,
                    sources=sorted(srcs),
                )

        # ---- relations: ensure endpoints exist, merge sources, dedup ----
        for rel in graph.relations:
            for ep in (rel.source, rel.target):
                if not self.G.has_node(ep):
                    self.G.add_node(ep, type="Entity", description="", aliases=[], sources=[], name=ep)
            existing = None
            if self.G.has_edge(rel.source, rel.target):
                for _k, d in self.G[rel.source][rel.target].items():
                    if d.get("type") == rel.type:
                        existing = d
                        break
            if existing is not None:
                existing["sources"] = sorted(set(existing.get("sources", [])) | set(rel.sources))
            else:
                self.G.add_edge(
                    rel.source, rel.target, type=rel.type, sources=sorted(set(rel.sources))
                )

        # ---- events: first-class nodes linked to subject/object entities ----
        for ev in graph.events:
            eid = _event_id(ev)
            srcs = set(ev.sources)
            if self.G.has_node(eid):
                self.G.nodes[eid]["sources"] = sorted(set(self.G.nodes[eid].get("sources", [])) | srcs)
            else:
                self.G.add_node(
                    eid,
                    kind="event",
                    name=ev.summary or f"{ev.type}: {ev.subject}→{ev.object}",
                    type="Event",
                    event_type=ev.type,
                    subject=ev.subject,
                    object=ev.object,
                    time=ev.time,
                    summary=ev.summary,
                    description=ev.summary,
                    aliases=[],
                    sources=sorted(srcs),
                )
            # link to subject/object only if those entity nodes already exist
            for ep, role in ((ev.subject, "EVENT_SUBJECT"), (ev.object, "EVENT_OBJECT")):
                if not ep or not self.G.has_node(ep):
                    continue
                s, t = (ep, eid) if role == "EVENT_SUBJECT" else (eid, ep)
                exists = self.G.has_edge(s, t) and any(
                    d.get("type") == role for _k, d in self.G[s][t].items()
                )
                if not exists:
                    self.G.add_edge(s, t, type=role, sources=sorted(srcs))

        self._save()

    def get_graph(self) -> tuple[list[dict], list[dict]]:
        nodes = []
        for n in self.G.nodes:
            attrs = self.G.nodes[n]
            node = {
                "id": n,
                "name": attrs.get("name", n),
                "type": attrs.get("type", "Entity"),
                "description": attrs.get("description", ""),
                "aliases": attrs.get("aliases", []),
                "sources": attrs.get("sources", []),
            }
            if attrs.get("kind") == "event":
                node.update(
                    {
                        "event_type": attrs.get("event_type", ""),
                        "subject": attrs.get("subject", ""),
                        "object": attrs.get("object", ""),
                        "time": attrs.get("time", ""),
                        "summary": attrs.get("summary", ""),
                    }
                )
            nodes.append(node)
        edges = [
            {
                "source": u,
                "target": v,
                "type": d.get("type", "RELATED_TO"),
                "sources": d.get("sources", []),
            }
            for u, v, d in self.G.edges(data=True)
        ]
        return nodes, edges

    def delete_by_source(self, source: str) -> dict:
        from .base import _src_equals, source_in

        removed = {"nodes": 0, "edges": 0}
        # ---- edges: strip `source` from provenance, drop if it was the only one ----
        edges_to_delete = []
        for u, v, k, d in self.G.edges(keys=True, data=True):
            srcs = set(d.get("sources", []))
            if source_in(source, srcs):
                if len(srcs) > 1:
                    d["sources"] = sorted(s for s in srcs if not _src_equals(source, s))
                else:
                    edges_to_delete.append((u, v, k))
        for u, v, k in edges_to_delete:
            self.G.remove_edge(u, v, k)
        removed["edges"] = len(edges_to_delete)

        # ---- nodes: same logic ----
        nodes_to_delete = []
        for n, d in list(self.G.nodes(data=True)):
            srcs = set(d.get("sources", []))
            if source_in(source, srcs):
                if len(srcs) > 1:
                    d["sources"] = sorted(s for s in srcs if not _src_equals(source, s))
                else:
                    nodes_to_delete.append(n)
        for n in nodes_to_delete:
            self.G.remove_node(n)
        removed["nodes"] = len(nodes_to_delete)

        self._save()
        return removed

    def clear(self) -> None:
        self.G.clear()
        if self.path.exists():
            self.path.unlink()
