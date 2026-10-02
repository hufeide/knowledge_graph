"""Agent 3 & 4: 实体抽取 + 关系抽取 Agent (Entity & Relation Extraction)."""
from __future__ import annotations

import json

from ..clients import LLMClient
from ..config import config
from ..prompts import (
    ENTITY_TYPES,
    EXTRACTION_CORRECTION_PROMPT,
    EXTRACTION_ENTITIES_PROMPT,
    EXTRACTION_PROMPT,
    EXTRACTION_RELATIONS_PROMPT,
    EXTRACTION_SYSTEM,
    MERGED_REL_EVENTS_PROMPT,
)
from ..schemas import Entity, Event, GraphData, Relation
from ..utils import as_list, normalize_name, extract_json
from .base import Agent


class ExtractionAgent(Agent):
    name = "extraction"

    def __init__(self, llm: LLMClient | None = None, mode: str = "single", correction: bool | None = None):
        super().__init__(llm)
        # "single" -> one-pass; "two_pass" -> entities then relations
        self.mode = mode if mode in ("single", "two_pass") else "single"
        # Whether to run an LLM correction pass after extraction (see `correct`).
        # Defaults to the EXTRACTION_CORRECTION config flag.
        self.correction = config.extraction_correction if correction is None else correction

    def run(self, text: str, source: str = "", chunk_id: str = "") -> GraphData:
        if not text or not text.strip():
            return GraphData()
        t = text[:6000]
        if self.mode == "two_pass":
            return self._run_two_pass(t, chunk_id)
        return self._run_single(t, chunk_id)

    # 合并抽取（MERGE_UNDERSTANDING_EXTRACTION=true）：
    # 把"语义理解(事件)"合并进"实体关系抽取"阶段，分两步完成（保证关系/事件可靠产出，
    # 因为该 LLM 在单次生成里只会填充第一个数组，无法一次性给出 实体+关系+事件）：
    #   第 1 步：抽实体（紧凑）；第 2 步：给定实体名，一次抽取 关系 + 事件。
    # 每 chunk 由原先的「理解 1 次 + 抽取两段」共 3 次调用降为 2 次。
    def run_merged(self, text: str, source: str = "", chunk_id: str = "") -> GraphData:
        if not text or not text.strip():
            return GraphData()
        t = text[:6000]
        sources = [chunk_id] if chunk_id else []
        # 第 1 步：实体
        e_data = self.llm.complete_json(
            EXTRACTION_ENTITIES_PROMPT.replace("{types}", ", ".join(ENTITY_TYPES)).replace("{text}", t),
            system=EXTRACTION_SYSTEM,
        )
        result = self._parse(e_data, sources=sources)
        # 第 2 步：关系 + 事件（给定实体名，避免凭空造实体）
        names = [e.name for e in result.entities]
        if names:
            re_data = self.llm.complete_json(
                MERGED_REL_EVENTS_PROMPT.replace(
                    "{entities}", "\n".join(f"- {n}" for n in names)
                ).replace("{text}", t),
                system=EXTRACTION_SYSTEM,
            )
            merged = self._parse(re_data, sources=sources)
            # 端点对齐：模型在第 2 步常把实体名改写（如"RTX 30 系列" vs "RTX 30 系列显卡"），
            # 直接丢弃会导致关系及其引用的实体一起被校验器判定为孤儿而丢失。这里把
            # 关系/事件的端点对齐到已有实体（规范化 + 子串容错），缺失则补建实体。
            entity_by_norm = {normalize_name(e.name): e.name for e in result.entities}

            def resolve(name: str):
                nm = normalize_name(name)
                if nm in entity_by_norm:
                    return entity_by_norm[nm]
                for en, ev in entity_by_norm.items():  # 子串容错
                    if en and (en in nm or nm in en):
                        return ev
                return None

            def ensure(name: str):
                if not name:
                    return None
                nm = normalize_name(name)
                if nm in entity_by_norm:
                    return entity_by_norm[nm]
                result.entities.append(Entity(name=name, type="Entity", sources=list(sources)))
                entity_by_norm[nm] = name
                return name

            for r in merged.relations:
                s = resolve(r.source) or ensure(r.source)
                tg = resolve(r.target) or ensure(r.target)
                if s and tg:
                    r.source, r.target = s, tg
                    result.relations.append(r)
            for ev in merged.events:
                if ev.subject:
                    s = resolve(ev.subject) or ensure(ev.subject)
                    if s:
                        ev.subject = s
                if ev.object:
                    o = resolve(ev.object) or ensure(ev.object)
                    if o:
                        ev.object = o
                result.events.append(ev)
        return result

    # 抽取后校正：把初步抽取结果连同原文再次交给 LLM，修正实体名错别字 / 合并重复
    # 实体 / 修复关系与事件的端点（使其严格对应实体名），达到"修正结果"的目的。
    # 该校正步会多消耗一次 LLM 调用（由 config.extraction_correction 控制是否启用）。
    def correct(self, gd: GraphData, text: str, sources: list[str] | None = None) -> GraphData:
        if not (gd.entities or gd.relations or gd.events):
            return gd
        t = text[:10000]
        try:
            payload = json.dumps(gd.model_dump(), ensure_ascii=False)
        except Exception:
            return gd
        # 超出上下文预算则跳过校正，避免截断/报错（宁可保留原始结果）。
        if len(payload) > 12000:
            return gd
        try:
            data = self.llm.complete_json(
                EXTRACTION_CORRECTION_PROMPT.replace("{text}", t).replace("{data}", payload),
                system=EXTRACTION_SYSTEM,
            )
            return self._parse(data, sources=sources or [])
        except Exception as e:
            # 校正步 LLM 调用失败时退回原始抽取结果，避免单次 LLM 抖动中断整个构建。
            try:
                print(f"[extraction] 修正步 LLM 调用失败，保留原始结果: {e}")
            except Exception:
                pass
            return gd

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
