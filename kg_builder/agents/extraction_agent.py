"""Agent 3 & 4: 实体抽取 + 关系抽取 Agent (Entity & Relation Extraction)."""
from __future__ import annotations

from ..clients import LLMClient
from ..prompts import (
    ENTITY_TYPES,
    EXTRACTION_ENTITIES_PROMPT,
    EXTRACTION_PROMPT,
    EXTRACTION_RELATIONS_PROMPT,
    EXTRACTION_SYSTEM,
)
from ..schemas import Entity, Event, GraphData, Relation
from ..utils import as_list, extract_json
from .base import Agent


class ExtractionAgent(Agent):
    name = "extraction"

    def __init__(self, llm: LLMClient | None = None, mode: str = "single"):
        super().__init__(llm)
        # "single" -> one-pass; "two_pass" -> entities then relations
        self.mode = mode if mode in ("single", "two_pass") else "single"

    def run(self, text: str, source: str = "", chunk_id: str = "") -> GraphData:
        if not text or not text.strip():
            return GraphData()
        t = text[:6000]
        if self.mode == "two_pass":
            return self._run_two_pass(t, chunk_id)
        return self._run_single(t, chunk_id)

    # One-pass: extract entities + relations + events in a single LLM call.
    def _run_single(self, t: str, chunk_id: str) -> GraphData:
        data = self.llm.complete_json(
            EXTRACTION_PROMPT.replace("{types}", ", ".join(ENTITY_TYPES)).replace("{text}", t),
            system=EXTRACTION_SYSTEM,
        )
        return self._parse(data, sources=[chunk_id] if chunk_id else [])

    # Two-pass: entities first (bounded output), then feed names back for relations.
    def _run_two_pass(self, t: str, chunk_id: str) -> GraphData:
        sources = [chunk_id] if chunk_id else []
        # Pass 1: extract entities (compact, bounded output)
        e_data = self.llm.complete_json(
            EXTRACTION_ENTITIES_PROMPT.replace("{types}", ", ".join(ENTITY_TYPES)).replace("{text}", t),
            system=EXTRACTION_SYSTEM,
        )
        result = self._parse(e_data, sources=sources)

        # Pass 2: extract relations, feeding the discovered entity names back so
        # the model only needs to emit the (small) relation list within budget.
        names = [e.name for e in result.entities]
        if names:
            r_data = self.llm.complete_json(
                EXTRACTION_RELATIONS_PROMPT.replace(
                    "{entities}", "\n".join(f"- {n}" for n in names)
                ).replace("{text}", t),
                system=EXTRACTION_SYSTEM,
            )
            rel_part = self._parse(r_data, sources=sources)
            result.relations.extend(rel_part.relations)
            result.events.extend(rel_part.events)
        return result

    # ---- parsing -------------------------------------------------------
    # 硬性上限：即便模型不遵守 prompt 中的数量约束（如把清单逐条枚举成上百个
    # 实体），也保证单 chunk 不会污染图谱。超出部分直接丢弃。
    MAX_ENTITIES = 30
    MAX_RELATIONS = 40
    MAX_EVENTS = 15

    def _parse(self, data, sources: list[str] | None = None) -> GraphData:
        result = GraphData()
        if not isinstance(data, dict):
            return result
        for raw in as_list(data.get("entities")):
            if len(result.entities) >= self.MAX_ENTITIES:
                break
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            result.entities.append(
                Entity(
                    name=name,
                    type=str(raw.get("type", "Entity")).strip() or "Entity",
                    aliases=[str(a) for a in as_list(raw.get("aliases")) if str(a).strip()],
                    description=str(raw.get("description", "")).strip(),
                    sources=list(sources) if sources else [],
                )
            )
        seen_rels = set()
        for raw in as_list(data.get("relations")):
            if len(result.relations) >= self.MAX_RELATIONS:
                break
            if not isinstance(raw, dict):
                continue
            src = str(raw.get("source", "")).strip()
            tgt = str(raw.get("target", "")).strip()
            rtype = str(raw.get("type", "RELATED_TO")).strip().upper() or "RELATED_TO"
            if not src or not tgt:
                continue
            key = (src.lower(), tgt.lower(), rtype)
            if key in seen_rels:
                continue
            seen_rels.add(key)
            result.relations.append(
                Relation(source=src, target=tgt, type=rtype, sources=list(sources) if sources else [])
            )
        for raw in as_list(data.get("events")):
            if len(result.events) >= self.MAX_EVENTS:
                break
            if not isinstance(raw, dict):
                continue
            result.events.append(
                Event(
                    type=str(raw.get("type", "")),
                    subject=str(raw.get("subject", "")),
                    object=str(raw.get("object", "")),
                    time=str(raw.get("time", "")),
                    summary=str(raw.get("summary", "")),
                    sources=list(sources) if sources else [],
                )
            )
        return result
