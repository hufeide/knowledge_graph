"""Agent 2: 语义理解 Agent (Understanding Agent)."""
from __future__ import annotations

from ..clients import LLMClient
from ..prompts import UNDERSTANDING_PROMPT, UNDERSTANDING_SYSTEM
from ..schemas import Entity, Event, Relation
from ..utils import as_list, normalize_name
from .base import Agent


def _entity_names(entities) -> list[str]:
    out: list[str] = []
    for e in entities or []:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, Entity):
            out.append(e.name)
    return out


def _relation_lines(relations) -> str:
    lines: list[str] = []
    for r in relations or []:
        if isinstance(r, str):
            lines.append(r)
        elif isinstance(r, Relation):
            lines.append(f"- {r.source} --{r.type}--> {r.target}")
    return "\n".join(lines)


class UnderstandingAgent(Agent):
    name = "understanding"

    def __init__(self, llm: LLMClient | None = None):
        super().__init__(llm)

    def run(
        self,
        text: str,
        source: str = "",
        chunk_id: str = "",
        entities: list | None = None,
        relations: list | None = None,
    ) -> list[Event]:
        """抽取关键事件。

        若提供 ``entities`` / ``relations``（来自实体关系抽取阶段），则把它们作为上下文：
        - 事件端点优先复用已知实体名，便于在 Validator 阶段稳定挂接到图谱实体；
        - 也可抽取少量已知实体之外的重要事件（其新端点由调用方补建为实体节点）。
        """
        if not text or not text.strip():
            return []
        sources = [chunk_id] if chunk_id else []
        entity_names = _entity_names(entities)
        prompt = (
            UNDERSTANDING_PROMPT
            .replace("{entities}", "\n".join(f"- {n}" for n in entity_names) or "（无）")
            .replace("{relations}", _relation_lines(relations) or "（无）")
            .replace("{text}", text[:6000])
        )
        data = self.llm.complete_json(prompt, system=UNDERSTANDING_SYSTEM)
        events: list[Event] = []
        if not isinstance(data, dict):
            return events

        # 端点对齐：事件 subject/object 优先复用已知实体名（规范化 + 子串容错），
        # 这样事件能稳定挂接到图谱实体节点上（对应 validator 的端点规范化逻辑）。
        entity_by_norm = {normalize_name(n): n for n in entity_names}

        def resolve(name: str) -> str:
            nm = normalize_name(name)
            if not nm:
                return name
            if nm in entity_by_norm:
                return entity_by_norm[nm]
            for en, ev in entity_by_norm.items():
                if en and (en in nm or nm in en):
                    return ev
            return name

        for raw in as_list(data.get("events")):
            if not isinstance(raw, dict):
                continue
            subject = str(raw.get("subject", "")).strip()
            obj = str(raw.get("object", "")).strip()
            events.append(
                Event(
                    type=str(raw.get("type", "")),
                    subject=resolve(subject) if subject else "",
                    object=resolve(obj) if obj else "",
                    time=str(raw.get("time", "")),
                    summary=str(raw.get("summary", "")),
                    sources=list(sources),
                )
            )
        return events
