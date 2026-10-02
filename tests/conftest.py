"""Shared pytest fixtures: a FakeLLM that avoids network calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg_builder import config
from kg_builder.clients import EmbeddingClient, LLMClient
from kg_builder.graph_store import NetworkXStore
from kg_builder.supervisor import Supervisor
from kg_builder.vector_store import VectorStore

DEFAULT_EXTRACTION = {
    "entities": [
        {"name": "NVIDIA", "type": "Company", "description": "半导体公司"},
        {"name": "nvidia", "type": "Company", "description": "同名归一测试"},
        {"name": "RTX 4090", "type": "Product", "description": "旗舰显卡"},
        {"name": "OpenAI", "type": "Company", "description": "AI 公司"},
        {"name": "OpenAI", "type": "Person", "description": "（错误类型，用于冲突测试）"},
    ],
    "relations": [
        {"source": "NVIDIA", "target": "RTX 4090", "type": "PRODUCES"},
        {"source": "OpenAI", "target": "GPT-4", "type": "CREATES"},
        {"source": "Ghost", "target": "Nowhere", "type": "RELATED_TO"},
    ],
    "events": [
        {"type": "RELEASE", "subject": "OpenAI", "object": "GPT-4", "time": "2023-03", "summary": "OpenAI 发布 GPT-4"}
    ],
}

UNDERSTANDING_EVENTS = {
    "events": [
        {"type": "RELEASE", "subject": "OpenAI", "object": "GPT-4", "time": "2023-03", "summary": "OpenAI 发布 GPT-4"}
    ]
}


class FakeLLM(LLMClient):
    """Deterministic stand-in for the real LLM; routes by system prompt."""

    def __init__(self, *args, **kwargs):
        # skip network session construction
        self.session = None
        self.cfg = config

    def complete(self, prompt, system=None, temperature=0.3, max_tokens=2048, json_mode=False):
        if system and "理解" in system:
            return json.dumps(UNDERSTANDING_EVENTS, ensure_ascii=False)
        return json.dumps(DEFAULT_EXTRACTION, ensure_ascii=False)

    def complete_json(self, prompt, system=None, temperature=0.0):
        return json.loads(self.complete(prompt, system=system, json_mode=True))


class FakeEmbedding(EmbeddingClient):
    """Tiny deterministic embedding: hashed bag-of-chars vector."""

    def __init__(self, *args, **kwargs):
        self.session = None
        self.cfg = config

    def embed(self, texts):
        import math

        out = []
        for t in texts if isinstance(texts, list) else [texts]:
            vec = [0.0] * self.cfg.emb_dim
            for ch in str(t):
                vec[ord(ch) % self.cfg.emb_dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_emb():
    return FakeEmbedding()


@pytest.fixture
def tmp_graph_store(tmp_path):
    return NetworkXStore(tmp_path / "kg.json")


@pytest.fixture
def tmp_vector_store(tmp_path):
    return VectorStore(tmp_path / "vec.json", config.emb_dim)


@pytest.fixture
def supervisor(tmp_path, fake_llm, fake_emb):
    gs = NetworkXStore(tmp_path / "kg.json")
    vs = VectorStore(tmp_path / "vec.json", config.emb_dim)
    return Supervisor(graph_store=gs, vector_store=vs, llm=fake_llm, emb=fake_emb)


@pytest.fixture
def sample_text():
    return (Path(__file__).resolve().parents[1] / "samples" / "nvidia_demo.txt").read_text(encoding="utf-8")
