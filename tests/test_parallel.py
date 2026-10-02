"""Verify that multi-chunk processing can run in parallel (CHUNK_WORKERS)."""
import os
import tempfile

import pytest

from kg_builder.config import config
from kg_builder.supervisor import Supervisor
from kg_builder.graph_store import NetworkXStore
from kg_builder.vector_store import VectorStore
from conftest import FakeLLM, FakeEmbedding


def _long_text():
    base = (
        "NVIDIA 由黄仁勋于1993年创立，总部位于圣克拉拉。"
        "RTX 4090 是其旗舰显卡，采用 Ada Lovelace 架构。"
        "OpenAI 在2023年发布 GPT-4，训练大量使用 NVIDIA GPU。"
    )
    return " ".join([base] * 40)


def _stores():
    d = tempfile.mkdtemp()
    gs = NetworkXStore(os.path.join(d, "g.json"))
    vs = VectorStore(os.path.join(d, "v.json"), config.emb_dim)
    return gs, vs


def test_parallel_build_matches_sequential():
    text = _long_text()

    # Sequential baseline
    config.chunk_workers = 1
    gs1, vs1 = _stores()
    r1 = Supervisor(graph_store=gs1, vector_store=vs1, llm=FakeLLM(), emb=FakeEmbedding()).build(text, source="t")

    # Parallel (3 workers) on a fresh store
    config.chunk_workers = 3
    gs2, vs2 = _stores()
    prog = []
    r2 = Supervisor(graph_store=gs2, vector_store=vs2, llm=FakeLLM(), emb=FakeEmbedding()).build(
        text, source="t", progress=prog.append
    )

    assert r2["chunks"] > 1, "text should split into multiple chunks"
    # parallel progress reports "(done/total)" with the right total
    chunk_msgs = [m for m in prog if "实体/关系抽取" in m]
    assert chunk_msgs, "expected per-chunk progress messages"
    assert any(f"({r2['chunks']}/{r2['chunks']})" in m for m in chunk_msgs), "final progress should reach total"

    # Both strategies must yield the same merged graph
    assert r1["stats"]["node_count"] == r2["stats"]["node_count"]
    assert r1["stats"]["edge_count"] == r2["stats"]["edge_count"]
    assert r2["stats"]["node_count"] > 0 and r2["stats"]["edge_count"] > 0
