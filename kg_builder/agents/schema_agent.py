"""Agent 5: 图谱 Schema 设计 Agent (Schema Design Agent).

Derives a canonical schema (node types, relation types and alias mapping)
from the extracted graph. This is deterministic and does not require the LLM,
which keeps the pipeline robust; an LLM-based refinement can be layered on top.
"""
from __future__ import annotations

from ..schemas import GraphData, Schema
from .base import Agent


class SchemaAgent(Agent):
    name = "schema"

    def run(self, graph: GraphData) -> Schema:
        node_types = sorted({e.type for e in graph.entities if e.type})
        relation_types = sorted({r.type for r in graph.relations if r.type})
        event_types = sorted({e.type for e in graph.events if e.type})

        aliases: dict[str, list[str]] = {}
        for e in graph.entities:
            if e.aliases:
                aliases[e.name] = sorted(set(e.aliases))

        return Schema(
            node_types=node_types,
            relation_types=relation_types,
            event_types=event_types,
            entity_count=len(graph.entities),
            relation_count=len(graph.relations),
            event_count=len(graph.events),
            type_aliases=aliases,
        )
