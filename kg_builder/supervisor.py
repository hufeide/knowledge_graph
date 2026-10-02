"""Supervisor: orchestrates the multi-agent KG construction pipeline.

Pipeline (per 思路.md):
    Data Agent -> Understanding + Extraction Agents -> Schema Agent
                -> Validator Agent -> Graph Writer Agent -> Vector Index
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from .agents.data_agent import DataAgent
from .agents.extraction_agent import ExtractionAgent
from .agents.graph_agent import GraphAgent
from .agents.qa_agent import QAAgent
from .agents.schema_agent import SchemaAgent
from .agents.understanding_agent import UnderstandingAgent
from .agents.validator_agent import ValidatorAgent
from .clients import EmbeddingClient, LLMClient
from .config import PROJECT_ROOT, Config, config
from .graph_store import GraphStore, create_graph_store
from .schemas import GraphData
from .vector_store import VectorStore


class BuildCancelled(Exception):
    """Raised by :meth:`Supervisor.build` when cancellation is requested."""


class Supervisor:
    def __init__(
        self,
        cfg: Config | None = None,
        graph_store: GraphStore | None = None,
        vector_store: VectorStore | None = None,
        llm: LLMClient | None = None,
        emb: EmbeddingClient | None = None,
    ):
        self.cfg = cfg or config
        self.llm = llm or LLMClient(self.cfg)
        self.emb = emb or EmbeddingClient(self.cfg)
        self.graph_store = graph_store or create_graph_store(self.cfg)
        self.vector_store = vector_store or VectorStore(
            self.cfg.vector_store_path(), self.cfg.emb_dim
        )

        self.data_agent = DataAgent(self.llm, self.cfg.chunk_size, self.cfg.chunk_overlap)
        self.understanding = UnderstandingAgent(self.llm)
        self.extraction = ExtractionAgent(self.llm, mode=self.cfg.extraction_mode)
        self.schema_agent = SchemaAgent(self.llm)
        self.graph_agent = GraphAgent(self.llm, self.graph_store)
        self.validator = ValidatorAgent(self.llm)
        self.qa = QAAgent(self.llm, self.emb, self.vector_store, self.graph_store, cfg=self.cfg)

    # ---- build ---------------------------------------------------------
    def build(
        self,
        text: str,
        source: str = "document",
        progress: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict:
        def emit(msg: str):
            if callable(progress):
                progress(msg)

        def _cancelled() -> bool:
            return callable(should_cancel) and should_cancel()

        # ---- 各 agent 耗时统计（并发安全）----
        _timings = defaultdict(float)
        _tlock = threading.Lock()
        _wall_start = time.perf_counter()

        def timed(name: str, fn, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                with _tlock:
                    _timings[name] += dt

        emit("① 数据采集与分块 ...")
        chunks = timed("DataAgent(分块)", self.data_agent.run, text, source)
        emit(f"① 分块完成：共 {len(chunks)} 块")

        total = len(chunks)
        graph = GraphData()

        # Phase 1 — per-chunk LLM extraction. Chunks run in parallel, and within a
        # single chunk the understanding + extraction calls also overlap inside the
        # same pool. `workers` is the configurable concurrency knob (CHUNK_WORKERS).
        workers = max(1, min(self.cfg.chunk_workers, total * 2)) if total > 1 else 1

        def _collect(events, gd):
            graph.events.extend(events)
            graph.entities.extend(gd.entities)
            graph.relations.extend(gd.relations)

        if _cancelled():
            raise BuildCancelled(f"构建已取消：{source}")

        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_und = {
                    ex.submit(timed, "UnderstandingAgent(语义理解)", self.understanding.run,
                              c.text, c.source, c.id): i
                    for i, c in enumerate(chunks)
                }
                fut_ext = {
                    ex.submit(timed, "ExtractionAgent(实体/关系)", self.extraction.run,
                              c.text, c.source, c.id): i
                    for i, c in enumerate(chunks)
                }
                und, ext = {}, {}
                for f in as_completed(fut_und):
                    und[fut_und[f]] = f.result()
                for f in as_completed(fut_ext):
                    ext[fut_ext[f]] = f.result()
            for i in range(total):
                _collect(und[i], ext[i])
                emit(f"② 语义理解 + ③④ 实体/关系抽取 ({i + 1}/{total}) ...")
        else:
            for i, c in enumerate(chunks):
                if _cancelled():
                    raise BuildCancelled(f"构建已取消：{source}")
                emit(f"② 语义理解 + ③④ 实体/关系抽取 ({i + 1}/{total}) ...")
                events = timed("UnderstandingAgent(语义理解)", self.understanding.run,
                               c.text, source=c.source, chunk_id=c.id)
                gd = timed("ExtractionAgent(实体/关系)", self.extraction.run,
                           c.text, source=c.source, chunk_id=c.id)
                _collect(events, gd)

        # Phase 2 — validation (global merge/dedup, CPU-only) overlaps the
        # independent vector-embedding step (network-bound) so both finish sooner.
        # Validation must stay a single global pass: de-duplication needs the whole
        # entity set, so it cannot be split per-chunk without losing merge semantics.
        emit("⑦ 图谱验证 + 向量索引（并行）...")
        chunk_texts = [c.text for c in chunks]
        metas = [{"source": c.source, "id": c.id} for c in chunks]
        from concurrent.futures import ThreadPoolExecutor as _TPE

        with _TPE(max_workers=self.cfg.chunk_workers) as ex:
            fut_val = ex.submit(timed, "ValidatorAgent(校验)", self.validator.run, graph)
            fut_emb = ex.submit(timed, "EmbeddingClient(向量)", self.emb.embed,
                                chunk_texts) if chunk_texts else None
            graph, report = fut_val.result()
            vecs = fut_emb.result() if fut_emb else None
        if vecs:
            self.vector_store.add(chunk_texts, metas, vecs)

        emit("⑤ Schema 设计（基于校验后的图谱，计数口径与 stats 一致）...")
        schema = timed("SchemaAgent(Schema)", self.schema_agent.run, graph)

        emit("⑥ 写入知识图谱 ...")
        stats = timed("GraphWriterAgent(写入)", self.graph_agent.run, graph)

        _wall = time.perf_counter() - _wall_start
        self._log_timings(source, total, _timings, _wall, schema, report, stats)
        emit(f"✅ 构建完成（总耗时 {_wall:.1f}s，各 agent 工作合计 {sum(_timings.values()):.1f}s）")
        return {
            "source": source,
            "chunks": total,
            "schema": schema.model_dump(),
            "validation": report.model_dump(),
            "stats": stats,
        }

    def _log_timings(self, source, chunks, timings, wall, schema, report, stats) -> None:
        """把各 agent 耗时与构建结果写入 logs/build_timing.log（追加）。"""
        try:
            log_dir = PROJECT_ROOT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            work_total = sum(timings.values()) or 0.0
            lines = [
                "=" * 64,
                f"[{datetime.now().isoformat()}] build timing | source={source} | chunks={chunks}",
            ]
            # 按耗时从大到小输出，便于定位瓶颈
            for name, sec in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
                pct = (sec / work_total * 100) if work_total else 0.0
                lines.append(f"  {name:<26} {sec:8.2f}s  {pct:5.1f}%")
            lines.append(f"  {'AGENT_WORK_TOTAL':<26} {work_total:8.2f}s")
            lines.append(f"  {'WALL_TOTAL':<26} {wall:8.2f}s")
            try:
                lines.append("  schema: " + json.dumps(schema.model_dump(), ensure_ascii=False)[:400])
            except Exception:
                pass
            try:
                lines.append("  stats : " + json.dumps(stats, ensure_ascii=False)[:400])
            except Exception:
                pass
            lines.append("")
            with open(log_dir / "build_timing.log", "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception as e:
            # 计时日志写入失败不应影响构建主流程
            try:
                print(f"[build_timing] 写日志失败: {e}")
            except Exception:
                pass

    def build_file(self, path: str | Path, source: str | None = None) -> dict:
        source = source or Path(path).name
        text = self.data_agent.load_text_file(path)
        return self.build(text, source=source)

    # ---- delete by source ----------------------------------------------
    def delete_source(self, source: str) -> dict:
        """Remove all ingested data (graph + vector index) tagged with ``source``.

        An entity/relation/event or a vector chunk may carry multiple sources
        (e.g. the same entity extracted from several files). In that case only
        the given ``source`` is stripped; the entry is deleted only when it has
        no remaining sources.
        """
        graph = self.graph_store.delete_by_source(source)
        vectors_removed = self.vector_store.delete_by_source(source)
        return {"graph": graph, "vectors_removed": vectors_removed}

    # ---- query ---------------------------------------------------------
    def query(self, question: str) -> dict:
        return self.qa.answer(question)

    def graph_json(self) -> dict:
        return self.graph_store.to_json()

    def stats(self) -> dict:
        return self.graph_store.stats()

    def reset(self) -> None:
        self.graph_store.clear()
        self.vector_store.clear()
