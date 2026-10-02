"""Entry point: launch the multi-agent KG web service.

Usage:
    python run.py                 # uses HOST/KG_WEB_PORT from .env (default 0.0.0.0:9622)
    KG_GRAPH_BACKEND=neo4j python run.py   # use Neo4j instead of NetworkX
"""
from __future__ import annotations

import uvicorn

from kg_builder.api import app
from kg_builder.config import config


def main() -> None:
    print(f"多 Agent 知识图谱服务启动中 ...")
    print(f"  LLM    : {config.llm_host}  model={config.llm_model}")
    print(f"  EMB    : {config.emb_host}  model={config.emb_model} dim={config.emb_dim}")
    print(f"  Graph  : {config.graph_backend}")
    print(f"  前端   : http://{config.host}:{config.web_port}/")
    uvicorn.run(app, host=config.host, port=config.web_port, log_level="info")


if __name__ == "__main__":
    main()
