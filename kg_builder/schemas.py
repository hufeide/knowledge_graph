"""Pydantic data models shared by the agents."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source: str = ""


class Entity(BaseModel):
    name: str
    type: str = "Entity"
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    # Provenance: chunk ids (e.g. "report.pdf-3") this entity was extracted from.
    sources: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    source: str
    target: str
    type: str = "RELATED_TO"
    # Provenance: chunk ids this relation was extracted from.
    sources: list[str] = Field(default_factory=list)


class Event(BaseModel):
    type: str = ""
    subject: str = ""
    object: str = ""
    time: str = ""
    summary: str = ""
    # Provenance: chunk ids this event was extracted from.
    sources: list[str] = Field(default_factory=list)


class GraphData(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)


class Schema(BaseModel):
    node_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    entity_count: int = 0
    relation_count: int = 0
    event_count: int = 0
    type_aliases: dict[str, list[str]] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    merged_entities: int = 0
    orphan_relations: list[Relation] = Field(default_factory=list)
    orphan_nodes: list[str] = Field(default_factory=list)
    dropped_events: list[str] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BuildResult(BaseModel):
    source: str = ""
    chunks: int = 0
