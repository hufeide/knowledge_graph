"""Graph store package: factory + implementations."""
from __future__ import annotations

from ..config import Config, config
from .base import GraphStore
from .falkordb_store import FalkorStore
from .networkx_store import NetworkXStore
from .neo4j_store import Neo4jStore

__all__ = ["GraphStore", "NetworkXStore", "Neo4jStore", "FalkorStore", "create_graph_store"]


def create_graph_store(cfg: Config | None = None) -> GraphStore:
    cfg = cfg or config
    backend = cfg.graph_backend.lower()
    if backend == "neo4j":
        return Neo4jStore(cfg)
    if backend in ("falkordb", "falkor"):
        return FalkorStore(cfg)
    return NetworkXStore(cfg.graph_store_path())
