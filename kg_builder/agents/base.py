"""Base class for all agents."""
from __future__ import annotations

from ..clients import LLMClient


class Agent:
    """Base agent: holds an LLM client and an optional name."""

    name = "agent"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def __repr__(self) -> str:
        return f"<Agent {self.name}>"
