"""Agent 2: 语义理解 Agent (Understanding Agent)."""
from __future__ import annotations

from ..clients import LLMClient
from ..prompts import UNDERSTANDING_PROMPT, UNDERSTANDING_SYSTEM
from ..schemas import Event
from ..utils import as_list
from .base import Agent


class UnderstandingAgent(Agent):
    name = "understanding"

    def __init__(self, llm: LLMClient | None = None):
        super().__init__(llm)

    def run(self, text: str, source: str = "", chunk_id: str = "") -> list[Event]:
        if not text or not text.strip():
            return []
        data = self.llm.complete_json(
            UNDERSTANDING_PROMPT.replace("{text}", text[:6000]),
            system=UNDERSTANDING_SYSTEM,
        )
        events: list[Event] = []
        if not isinstance(data, dict):
            return events
        sources = [chunk_id] if chunk_id else []
        for raw in as_list(data.get("events")):
            if not isinstance(raw, dict):
                continue
            events.append(
                Event(
                    type=str(raw.get("type", "")),
                    subject=str(raw.get("subject", "")),
                    object=str(raw.get("object", "")),
                    time=str(raw.get("time", "")),
                    summary=str(raw.get("summary", "")),
                    sources=list(sources),
                )
            )
        return events
