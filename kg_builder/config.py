"""Central configuration loaded from the project .env file.

The .env in this project mirrors a LightRAG-style deployment and exposes the
LLM / embedding bindings plus graph storage credentials we rely on:

- LLM_BINDING_HOST / LLM_MODEL / LLM_BINDING_API_KEY  -> OpenAI-compatible chat endpoint
- EMBEDDING_BINDING_HOST / EMBEDDING_MODEL / EMBEDDING_DIM / EMBEDDING_BINDING_API_KEY -> embeddings
- NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE -> optional Neo4j backend
- KG_GRAPH_BACKEND -> "networkx" (default, dependency free) or "neo4j"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env(path: Path = ENV_PATH) -> dict[str, str]:
    data: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            data[key.strip()] = value
    return data


_raw = _load_env()


def _get(key: str, default: str | None = None) -> str | None:
    if key in os.environ and os.environ[key] != "":
        return os.environ[key]
    return _raw.get(key, default)


def _int(value: str | None, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # LLM (OpenAI-compatible)
    llm_binding: str = field(default_factory=lambda: _get("LLM_BINDING", "openai"))
    llm_host: str = field(default_factory=lambda: _get("LLM_BINDING_HOST", "http://localhost:9000/v1"))
    llm_model: str = field(default_factory=lambda: _get("LLM_MODEL", "qwen"))
    llm_api_key: str = field(default_factory=lambda: _get("LLM_BINDING_API_KEY", "EMPTY"))

    # Embeddings (OpenAI-compatible)
    emb_binding: str = field(default_factory=lambda: _get("EMBEDDING_BINDING", "openai"))
    emb_host: str = field(default_factory=lambda: _get("EMBEDDING_BINDING_HOST", "http://localhost:9001/v1"))
    emb_model: str = field(default_factory=lambda: _get("EMBEDDING_MODEL", "bge-m3"))
    emb_dim: int = field(default_factory=lambda: _int(_get("EMBEDDING_DIM", "1024"), 1024))
    emb_api_key: str = field(default_factory=lambda: _get("EMBEDDING_BINDING_API_KEY", "EMPTY"))

    # Graph backend
    graph_backend: str = field(default_factory=lambda: _get("KG_GRAPH_BACKEND", "networkx"))

    # FalkorDB (optional backend, the DB recommended by 思路.md)
    falkordb_host: str = field(default_factory=lambda: _get("FALKORDB_HOST", "127.0.0.1"))
    falkordb_port: int = field(default_factory=lambda: _int(_get("FALKORDB_PORT", "6379"), 6379))
    falkordb_graph: str = field(default_factory=lambda: _get("FALKORDB_GRAPH", "kg_builder"))

    # Web service port (default differs from LightRAG's 9621 to avoid clashing)
    web_port: int = field(default_factory=lambda: _int(_get("KG_WEB_PORT", "9622"), 9622))

    # Neo4j (optional backend)
    neo4j_uri: str = field(default_factory=lambda: _get("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _get("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _get("NEO4J_PASSWORD", "hufei1989"))
    neo4j_database: str = field(default_factory=lambda: _get("NEO4J_DATABASE", "neo4j"))
    neo4j_http_url: str = field(default_factory=lambda: _get("NEO4J_HTTP_URL", "http://localhost:7474"))

    # Storage
    storage_dir: str = field(default_factory=lambda: _get("KG_STORAGE_DIR", str(PROJECT_ROOT / "storage")))
    chunk_size: int = field(default_factory=lambda: _int(_get("CHUNK_SIZE", "1000"), 1000))
    chunk_overlap: int = field(default_factory=lambda: _int(_get("CHUNK_OVERLAP_SIZE", "100"), 100))

    # Parallelism: how many chunk-level tasks run concurrently during the
    # understanding + extraction phase (and understanding/extraction of a single
    # chunk overlap within the same pool). Set via CHUNK_WORKERS in .env.
    chunk_workers: int = field(default_factory=lambda: _int(_get("CHUNK_WORKERS", "4"), 4))

    # Extraction strategy: "single" (one-pass) or "two_pass" (entities then relations).
    # Default "single" to halve LLM calls; switch to "two_pass" for dense text.
    extraction_mode: str = field(default_factory=lambda: _get("EXTRACTION_MODE", "single") or "single")

    # Merge the understanding (events) + extraction (entities/relations) stages into a
    # single LLM call (run_merged), saving one LLM call per chunk. Set to false to keep
    # the original two-stage pipeline.
    merge_understanding_extraction: bool = field(
        default_factory=lambda: str(_get("MERGE_UNDERSTANDING_EXTRACTION", "true")).lower()
        in ("1", "true", "yes", "y")
    )

    # After extraction, run an LLM "correction" pass over the extracted entities/
    # relations/events (against the original chunk text) to fix name typos, merge
    # duplicates, and repair relation/event endpoints. Adds one LLM call per chunk.
    extraction_correction: bool = field(
        default_factory=lambda: str(_get("EXTRACTION_CORRECTION", "true")).lower()
        in ("1", "true", "yes", "y")
    )

    # ---- QA / GraphRAG retrieval ----------------------------------------
    # How many chunks to retrieve from the vector store per question.
    qa_top_k: int = field(default_factory=lambda: _int(_get("QA_TOP_K", "5"), 5))
    # Graph traversal depth (hops) when expanding seed entities into a subgraph.
    # 1 hop is often too shallow for finance (entity -> attribute -> implication),
    # so default 2.
    qa_graph_hops: int = field(default_factory=lambda: _int(_get("QA_GRAPH_HOPS", "2"), 2))
    # Upper bound on nodes pulled into the subgraph to avoid graph explosion.
    qa_graph_max_nodes: int = field(default_factory=lambda: _int(_get("QA_GRAPH_MAX_NODES", "40"), 40))
    # Evidence-fusion rerank weights. final = w_vector*vec + w_graph*entity + w_prov*provenance
    qa_rerank_vector: float = field(default_factory=lambda: float(_get("QA_RERANK_VECTOR", "0.5") or "0.5"))
    qa_rerank_graph: float = field(default_factory=lambda: float(_get("QA_RERANK_GRAPH", "0.3") or "0.3"))
    qa_rerank_prov: float = field(default_factory=lambda: float(_get("QA_RERANK_PROV", "0.2") or "0.2"))

    # Service
    host: str = field(default_factory=lambda: _get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int(_get("PORT", "9621"), 9621))

    @property
    def storage_path(self) -> Path:
        p = Path(self.storage_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def graph_store_path(self) -> Path:
        return self.storage_path / "kg_graph.json"

    def vector_store_path(self) -> Path:
        return self.storage_path / "kg_vectors.json"


config = Config()
