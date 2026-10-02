"""Abstract graph store + shared helpers (JSON export & KG context)."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..schemas import GraphData


def _src_equals(source: str, s: str) -> bool:
    """Does a single provenance entry ``s`` belong to the file-level ``source``?

    Provenance entries are chunk ids of the form ``{source}-{n}``, while web
    builds and some stores also keep the bare ``source``. Match either.
    """
    return s == source or s.startswith(source + "-")


def source_in(source: str, srcs) -> bool:
    """True if ``source`` (a file-level source) is provenance of ``srcs``."""
    return any(_src_equals(source, s) for s in (srcs or []))


class GraphStore(ABC):
    @abstractmethod
    def upsert_graph(self, graph: GraphData) -> None: ...

    @abstractmethod
    def get_graph(self) -> tuple[list[dict], list[dict]]:
        """Return (nodes, edges).

        Each node carries at least id/name/type/description/aliases/sources.
        Event nodes additionally carry event_type/subject/object/time/summary
        and have type == "Event". Each edge carries source/target/type/sources;
        event edges use types EVENT_SUBJECT / EVENT_OBJECT.
        """

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def delete_by_source(self, source: str) -> dict:
        """Delete every node/edge/event whose provenance includes ``source``.

        Returns a count dict ``{"nodes": int, "edges": int}`` of the entries that
        were fully removed. Entries shared by multiple sources keep the other
        sources and are only stripped of ``source`` (not deleted).
        """

    # ---- shared implementations ----------------------------------------
    def stats(self) -> dict:
        nodes, edges = self.get_graph()
        types = sorted({n.get("type") for n in nodes if n.get("type")})
        rels = sorted({e.get("type") for e in edges if e.get("type")})
        return {
            "backend": self.__class__.__name__,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": types,
            "relation_types": rels,
        }

    def to_json(self) -> dict:
        nodes, edges = self.get_graph()
        out_nodes = [
            {
                "id": n["id"],
                "label": n.get("name", n["id"]),
                "group": n.get("type", "Entity"),
                "type": n.get("type", "Entity"),
                "description": n.get("description", ""),
                "aliases": n.get("aliases", []),
                "sources": n.get("sources", []),
                **({k: n[k] for k in ("event_type", "subject", "object", "time", "summary") if k in n}),
            }
            for n in nodes
        ]
        out_edges = [
            {
                "id": f"{e['source']}->{e['target']}#{i}",
                "from": e["source"],
                "to": e["target"],
                "label": e.get("type", "RELATED_TO"),
                "type": e.get("type", "RELATED_TO"),
                "sources": e.get("sources", []),
            }
            for i, e in enumerate(edges)
        ]
        return {"nodes": out_nodes, "edges": out_edges}

    def find_nodes_by_keyword(self, query: str) -> list[dict]:
        nodes, _ = self.get_graph()
        q = query.lower()
        tokens = {t for t in re.split(r"\W+", q) if len(t) >= 2}
        tokens.add(q)
        matched = []
        for n in nodes:
            name = str(n.get("name", n.get("id", ""))).lower()
            aliases = [str(a).lower() for a in n.get("aliases", [])]
            if any(t and (t in name or t in aliases or name in t) for t in tokens):
                matched.append(n)
        return matched

    def resolve_mentions(self, mentions: list[str]) -> list[dict]:
        """Map free-text entity mentions to concrete graph nodes.

        Matching considers node name, id and aliases (substring / equality),
        which is the entity-linking step that turns query- understanding output
        into seed nodes for subgraph retrieval.
        """
        nodes, _ = self.get_graph()
        seen: set[str] = set()
        out: list[dict] = []
        for m in mentions:
            mlow = str(m).lower().strip()
            if not mlow:
                continue
            for n in nodes:
                name = str(n.get("name", n.get("id", ""))).lower()
                core = str(n.get("id", "")).lower()
                aliases = [str(a).lower() for a in n.get("aliases", [])]
                hit = (
                    mlow == name
                    or mlow == core
                    or mlow in aliases
                    or any(mlow in a for a in aliases)
                    or mlow in name
                )
                if hit:
                    key = n["id"]
                    if key not in seen:
                        seen.add(key)
                        out.append(n)
                    break
        return out

    def retrieve_subgraph(
        self, seed_names: list[str], max_hops: int = 2, max_nodes: int = 40
    ) -> dict:
        """Expand seed entities into a multi-hop subgraph for QA.

        Returns a structured dict (not text) so the caller can render it as
        JSON-style context and keep provenance (``sources``) attached to every
        node/edge. This is the graph-retrieval half of the hybrid pipeline.
        """
        nodes, edges = self.get_graph()
        node_map = {n["id"]: n for n in nodes}
        seeds = self.resolve_mentions(seed_names)
        seed_ids = {s["id"] for s in seeds}
        ids: set[str] = set(seed_ids)
        frontier = set(seed_ids)
        for _ in range(max(0, max_hops)):
            nxt: set[str] = set()
            for e in edges:
                if e["source"] in frontier and e["target"] not in ids:
                    nxt.add(e["target"])
                if e["target"] in frontier and e["source"] not in ids:
                    nxt.add(e["source"])
            if not nxt:
                break
            ids |= nxt
            frontier = nxt
            if len(ids) >= max_nodes:
                break
        sub_nodes = [node_map[i] for i in list(ids)[:max_nodes] if i in node_map]
        sub_edges = [e for e in edges if e["source"] in ids and e["target"] in ids]
        return {
            "seeds": [s.get("name", s["id"]) for s in seeds],
            "unmatched": [
                m for m in seed_names if not any(m.lower() == s.get("name", "").lower() for s in seeds)
            ],
            "nodes": sub_nodes,
            "edges": sub_edges,
        }

    def describe_related(self, query: str, max_nodes: int = 15, hops: int = 1) -> str:
        """Return a graph-grounded context for QA.

        Starts from keyword-matched entities and expands to their neighbors
        (``hops`` hops) so that *associated* entities/relations are included,
        not just the directly matched ones.
        """
        nodes, edges = self.get_graph()
        node_map = {n["id"]: n for n in nodes}
        matched = self.find_nodes_by_keyword(query)
        if not matched:
            return ""

        # BFS expansion over the graph to collect associated nodes
        core = {n["id"] for n in matched}
        ids: set[str] = set(core)
        frontier = set(core)
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for e in edges:
                if e["source"] in frontier and e["target"] not in ids:
                    nxt.add(e["target"])
                if e["target"] in frontier and e["source"] not in ids:
                    nxt.add(e["source"])
            if not nxt:
                break
            ids |= nxt
            frontier = nxt
            if len(ids) >= max_nodes:
                break

        lines = []
        for nid in list(ids)[:max_nodes]:
            n = node_map.get(nid)
            if not n:
                continue
            tag = "●" if nid in core else "○"
            lines.append(f"{tag} {n.get('name', nid)} ({n.get('type', 'Entity')}): {n.get('description', '')}")
        for e in edges:
            if e["source"] in ids and e["target"] in ids:
                lines.append(f"    {e['source']} -[{e.get('type', 'RELATED_TO')}]-> {e['target']}")
        return "\n".join(lines)

    def matched_entities(self, query: str) -> list[str]:
        """Entity names that directly match the query (used to rerank text chunks)."""
        return [n.get("name", n["id"]) for n in self.find_nodes_by_keyword(query)]
