"""Unit tests for individual agents (no network, use FakeLLM)."""
from __future__ import annotations

from kg_builder.agents.data_agent import DataAgent
from kg_builder.agents.extraction_agent import ExtractionAgent
from kg_builder.agents.understanding_agent import UnderstandingAgent
from kg_builder.agents.schema_agent import SchemaAgent
from kg_builder.agents.validator_agent import ValidatorAgent
from kg_builder.schemas import Entity, Event, GraphData, Relation


def test_data_agent_chunking():
    agent = DataAgent(chunk_size=200, chunk_overlap=20)
    text = "。".join([f"这是第{i}句话用于测试分块逻辑" for i in range(40)])
    chunks = agent.run(text, source="t")
    assert len(chunks) >= 2
    assert all(c.source == "t" for c in chunks)
    # concatenation roughly preserves content
    joined = "".join(c.text for c in chunks)
    assert "第0句话" in joined and "第39句话" in joined


def test_extraction_parsing(fake_llm):
    agent = ExtractionAgent(fake_llm)
    gd = agent.run("NVIDIA PRODUCES RTX 4090")
    names = {e.name for e in gd.entities}
    assert "NVIDIA" in names and "RTX 4090" in names
    # relation types normalized to uppercase
    assert all(r.type.isupper() for r in gd.relations)
    assert any(r.source == "NVIDIA" and r.target == "RTX 4090" for r in gd.relations)


def test_understanding(fake_llm):
    agent = UnderstandingAgent(fake_llm)
    events = agent.run("OpenAI released GPT-4 in March 2023.")
    assert any(e.subject == "OpenAI" and e.object == "GPT-4" for e in events)


def test_schema_derivation():
    agent = SchemaAgent()
    gd = GraphData(
        entities=[Entity(name="A", type="Company"), Entity(name="B", type="Product")],
        relations=[Relation(source="A", target="B", type="PRODUCES")],
    )
    schema = agent.run(gd)
    assert "Company" in schema.node_types
    assert "PRODUCES" in schema.relation_types
    assert schema.entity_count == 2


def test_validator_dedupe_and_conflict():
    agent = ValidatorAgent()
    gd = GraphData(
        entities=[
            Entity(name="NVIDIA", type="Company"),
            Entity(name="nvidia", type="Company"),
            Entity(name="RTX 4090", type="Product"),
            Entity(name="OpenAI", type="Company"),
            Entity(name="OpenAI", type="Person"),  # conflict type
            Entity(name="GPT-4", type="Product"),
        ],
        relations=[
            Relation(source="NVIDIA", target="RTX 4090", type="PRODUCES"),
            Relation(source="OpenAI", target="GPT-4", type="CREATES"),
            Relation(source="Ghost", target="Nowhere", type="RELATED_TO"),  # orphan
        ],
    )
    cleaned, report = agent.run(gd)
    # NVIDIA + nvidia merged, OpenAI x2 merged -> 4 unique entities
    assert len(cleaned.entities) == 4
    assert report.merged_entities == 2
    # only the Ghost relation is orphan
    assert len(cleaned.relations) == 2
    assert len(report.orphan_relations) == 1
    # conflict detected (OpenAI has two types)
    assert any("OpenAI" in str(c) for c in report.conflicts)


def test_validator_event_time_conflict():
    agent = ValidatorAgent()
    gd = GraphData(
        entities=[Entity(name="GPT-5", type="Product")],
        relations=[],
        events=[
            Event(type="RELEASE", subject="OpenAI", object="GPT-5", time="2025", summary="a"),
            Event(type="RELEASE", subject="OpenAI", object="GPT-5", time="2026", summary="b"),
        ],
    )
    _cleaned, report = agent.run(gd)
    assert any("conflicting_times" in c for c in report.conflicts)
