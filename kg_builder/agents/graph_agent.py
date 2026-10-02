"""Agent 6: 图谱写入 Agent (Graph Builder / Writer Agent)."""
from __future__ import annotations

from ..graph_store import GraphStore
from ..schemas import GraphData
from .base import Agent


class GraphAgent(Agent):
    name = "graph_writer"

    def __init__(self, llm=None, graph_store: GraphStore | None = None):
        super().__init__(llm)
        self.graph_store = graph_store

    def run(self, graph: GraphData) -> dict:
        if self.graph_store is None:
            raise RuntimeError("GraphAgent 需要注入一个 GraphStore 实例。")
        self.graph_store.upsert_graph(graph)
        return self.graph_store.stats()
