"""QA / GraphRAG Agent: hybrid retrieval over a knowledge graph + vector store.

Pipeline (a lightweight GraphRAG):

    Question
       |
    Query Understanding  (LLM entity linking + intent)
       |
   +-------------+--------------+
   |                            |
 Graph Retrieval          Vector Retrieval
 (multi-hop subgraph)      (semantic chunks)
   |                            |
   +-------------+--------------+
                 |
        Evidence Fusion (rerank: vector + graph + provenance)
                 |
                LLM  ->  Answer + Citations
"""
from __future__ import annotations

from ..clients import EmbeddingClient, LLMClient
from ..config import Config, config
from ..graph_store import GraphStore
from ..prompts import (
    QA_ENTITY_LINKING_PROMPT,
    QA_ENTITY_LINKING_SYSTEM,
    QA_PROMPT,
    QA_SYSTEM,
)
from ..vector_store import VectorStore
from .base import Agent

# Source filenames that signal authoritative financial documents get a small
# provenance boost during reranking (年报/研报/财报/公告 ...).
_AUTHORITATIVE_HINTS = ("年报", "报告", "研报", "财报", "公告", "披露")


class QAAgent(Agent):
    name = "qa"

    def __init__(
        self,
        llm: LLMClient | None = None,
        emb: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        cfg: Config | None = None,
    ):
        super().__init__(llm)
        self.llm = llm or LLMClient()
        self.emb = emb or EmbeddingClient()
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.cfg = cfg or config

    # ---- public API ----------------------------------------------------
    def answer(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or self.cfg.qa_top_k

        # 1. Query understanding: entity linking + intent (the "reasoning layer")
        linking = self._link_entities(question)

        # 2. Graph retrieval: expand linked entities into a multi-hop subgraph
        subgraph = None
        graph_context = ""
        entity_names: list[str] = []
        if self.graph_store:
            seed_names = [e["name"] for e in linking["entities"]]
            subgraph = self.graph_store.retrieve_subgraph(
                seed_names,
                max_hops=self.cfg.qa_graph_hops,
                max_nodes=self.cfg.qa_graph_max_nodes,
            )
            graph_context = self._format_subgraph(subgraph)
            entity_names = [n.get("name", n["id"]) for n in subgraph["nodes"]]
        if not entity_names and linking["entities"]:
            entity_names = [e["name"] for e in linking["entities"]]

        # 3. Vector retrieval: semantic passage recall
        hits = self.vector_store.search(self.emb.embed_query(question), top_k=top_k) if self.vector_store else []

        # 4. Evidence fusion: rerank chunks by vector + graph + provenance
        if self.graph_store and entity_names:
            hits = self._rerank(hits, entity_names)

        context = "\n---\n".join(text for text, _meta, _score in hits)

        # 5. LLM synthesis with structured graph context + citations
        prompt = (
            QA_PROMPT.replace("{question}", question)
            .replace("{intent}", linking.get("intent") or "（未识别）")
            .replace("{graph_context}", graph_context or "（知识图谱中暂无相关实体）")
            .replace("{context}", context or "（无检索片段）")
        )
        answer = self.llm.complete(prompt, system=QA_SYSTEM, temperature=0.3, max_tokens=1024)

        return {
            "answer": answer,
            "question": question,
            "intent": linking.get("intent"),
            "entities": linking.get("entities"),
            "graph_context": graph_context,
            "contexts": [t for t, _m, _s in hits],
            "evidence": [{"text": t, "meta": m} for t, m, _s in hits],
            "citations": self._citations(subgraph, hits),
        }

    # ---- query understanding ------------------------------------------
    def _link_entities(self, question: str) -> dict:
        """LLM-based entity linking + intent; falls back to keyword match."""
        try:
            data = self.llm.complete_json(
                QA_ENTITY_LINKING_PROMPT.replace("{question}", question),
                system=QA_ENTITY_LINKING_SYSTEM,
            )
            if isinstance(data, dict):
                ents = [
                    {"name": e["name"], "type": e.get("type", "")}
                    for e in data.get("entities", [])
                    if isinstance(e, dict) and e.get("name")
                ]
                intent = data.get("intent", "")
            else:
                ents, intent = [], ""
        except Exception:
            ents, intent = [], ""

        # Fallback: if the LLM found nothing, degrade to keyword matching so the
        # graph still contributes (e.g. for simple entity-name questions).
        if not ents and self.graph_store:
            ents = [
                {"name": n.get("name", n["id"]), "type": n.get("type", "")}
                for n in self.graph_store.find_nodes_by_keyword(question)
            ]
        return {"entities": ents, "intent": intent}

    # ---- formatting ----------------------------------------------------
    @staticmethod
    def _format_subgraph(sg: dict) -> str:
        if not sg or not sg.get("nodes"):
            return ""
        seeds = set(sg.get("seeds", []))
        lines: list[str] = []
        if sg.get("unmatched"):
            lines.append(f"（未命中图谱实体：{', '.join(sg['unmatched'])}）")
        for n in sg["nodes"]:
            mark = "●" if n.get("name", n["id"]) in seeds else "○"
            srcs = n.get("sources", [])
            src_str = f" [来源: {', '.join(srcs)}]" if srcs else ""
            if n.get("type") == "Event":
                lines.append(
                    f"{mark} 事件: {n.get('summary', n.get('name', ''))} "
                    f"(时间: {n.get('time', '')}){src_str}"
                )
            else:
                lines.append(
                    f"{mark} {n.get('name', n['id'])} ({n.get('type', 'Entity')}): "
                    f"{n.get('description', '')}{src_str}"
                )
        for e in sg["edges"]:
            sp = e.get("sources", [])
            sp_str = f" [来源: {', '.join(sp)}]" if sp else ""
            lines.append(
                f"    {e['source']} -[{e.get('type', 'RELATED_TO')}]-> {e['target']}{sp_str}"
            )
        return "\n".join(lines)

    # ---- evidence fusion (rerank) -------------------------------------
    def _rerank(self, hits: list, entity_names: list[str]) -> list:
        """Weighted fusion: 0.5*vector + 0.3*graph_entity + 0.2*provenance."""
        wv = self.cfg.qa_rerank_vector
        wg = self.cfg.qa_rerank_graph
        wp = self.cfg.qa_rerank_prov
        low = [n.lower() for n in entity_names if len(n) >= 2]
        denom = max(1, len(low))
        reranked = []
        for text, meta, score in hits:
            tlow = text.lower()
            g = sum(1.0 for n in low if n in tlow) / denom
            prov = self._provenance_score(meta)
            final = wv * score + wg * g + wp * prov
            reranked.append((text, meta, final))
        return sorted(reranked, key=lambda x: x[2], reverse=True)

    @staticmethod
    def _provenance_score(meta: dict | None) -> float:
        if not meta:
            return 0.5
        src = str(meta.get("source", ""))
        if any(h in src for h in _AUTHORITATIVE_HINTS):
            return 1.0
        return 0.7

    @staticmethod
    def _citations(subgraph: dict | None, hits: list) -> list[str]:
        cites: set[str] = set()
        if subgraph:
            for n in subgraph.get("nodes", []):
                cites.update(n.get("sources", []))
            for e in subgraph.get("edges", []):
                cites.update(e.get("sources", []))
        for _t, meta, _s in hits:
            if not meta:
                continue
            src = meta.get("source")
            cid = meta.get("id")
            if src:
                cites.add(f"{src} ({cid})" if cid else src)
        return sorted(c for c in cites if c)
