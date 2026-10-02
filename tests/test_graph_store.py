"""Tests for graph + vector stores."""
from __future__ import annotations

import pytest

from kg_builder.graph_store import NetworkXStore, create_graph_store
from kg_builder.schemas import Entity, GraphData, Relation
from kg_builder.vector_store import VectorStore


def _sample_graph():
    return GraphData(
        entities=[
            Entity(name="NVIDIA", type="Company", description="GPU 厂商"),
            Entity(name="RTX 4090", type="Product", description="显卡"),
        ],
        relations=[Relation(source="NVIDIA", target="RTX 4090", type="PRODUCES")],
    )


def test_networkx_store_upsert_and_read(tmp_path):
    store = NetworkXStore(tmp_path / "g.json")
    store.upsert_graph(_sample_graph())
    nodes, edges = store.get_graph()
    assert len(nodes) == 2
    assert len(edges) == 1
    j = store.to_json()
    assert j["nodes"][0]["id"] == "NVIDIA"
    assert j["edges"][0]["from"] == "NVIDIA"
    assert j["edges"][0]["to"] == "RTX 4090"


def test_networkx_store_persistence(tmp_path):
    p = tmp_path / "g.json"
    s1 = NetworkXStore(p)
    s1.upsert_graph(_sample_graph())
    s2 = NetworkXStore(p)  # reload
    assert len(s2.get_graph()[0]) == 2


def test_networkx_store_describe_related(tmp_path):
    store = NetworkXStore(tmp_path / "g.json")
    store.upsert_graph(_sample_graph())
    desc = store.describe_related("NVIDIA 生产了什么")
    assert "NVIDIA" in desc
    assert "PRODUCES" in desc


def test_networkx_store_clear(tmp_path):
    store = NetworkXStore(tmp_path / "g.json")
    store.upsert_graph(_sample_graph())
    store.clear()
    assert store.get_graph() == ([], [])


def test_create_graph_store_factory(tmp_path):
    from kg_builder import config

    config.graph_backend = "networkx"
    config.storage_dir = str(tmp_path)
    store = create_graph_store()
    assert isinstance(store, NetworkXStore)


def test_vector_store_search(tmp_path):
    vs = VectorStore(tmp_path / "v.json", dim=8)
    import numpy as np

    a = [1.0] + [0.0] * 7
    b = [0.0] * 7 + [1.0]
    vs.add(["关于 NVIDIA 的文档", "关于烹饪的文档"], [{"s": 1}, {"s": 2}], [a, b])
    q = [0.9] + [0.0] * 7
    hits = vs.search(q, top_k=1)
    assert hits[0][0] == "关于 NVIDIA 的文档"
