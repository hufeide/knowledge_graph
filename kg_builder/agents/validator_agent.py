"""Agent 7: 图谱验证 Agent (Graph Validator Agent).

Handles the three checks described in 思路.md:
  1. 重复实体 (duplicate entities) -> merge by normalized name
  2. 错误关系 (wrong relations)   -> drop relations whose endpoints do not exist
  3. 冲突检测 (conflict detection) -> same entity with conflicting types / times
"""
from __future__ import annotations

from collections import defaultdict

from ..schemas import Entity, Event, GraphData, Relation, ValidationReport
from ..utils import normalize_name
from .base import Agent


class ValidatorAgent(Agent):
    name = "validator"

    def run(self, graph: GraphData) -> tuple[GraphData, ValidationReport]:
        report = ValidationReport()

        # ---- 1. merge duplicate entities --------------------------------
        groups: dict[str, list[Entity]] = defaultdict(list)
        for e in graph.entities:
            groups[normalize_name(e.name)].append(e)

        merged: list[Entity] = []
        conflicts: list[dict] = []
        for key, items in groups.items():
            types = {i.type for i in items if i.type}
            if len(types) > 1:
                conflicts.append({"name": items[0].name, "types": sorted(types)})
            # keep the richest description as canonical
            best = max(items, key=lambda x: len(x.description or ""))
            aliases: set[str] = set()
            sources: set[str] = set()
            for i in items:
                aliases.add(i.name)
                aliases.update(i.aliases)
                sources.update(i.sources)
            aliases.discard(best.name)
            merged.append(
                Entity(
                    name=best.name,
                    type=best.type or "Entity",
                    aliases=sorted(aliases),
                    description=best.description,
                    sources=sorted(sources),
                )
            )
        report.merged_entities = len(graph.entities) - len(merged)

        # ---- 2. drop orphan relations ------------------------------------
        name_index = {normalize_name(e.name): e.name for e in merged}
        clean_rels: list[Relation] = []
        orphans: list[Relation] = []
        seen_rels: dict[tuple[str, str, str], Relation] = {}
        for r in graph.relations:
            if normalize_name(r.source) in name_index and normalize_name(r.target) in name_index:
                src = name_index[normalize_name(r.source)]
                tgt = name_index[normalize_name(r.target)]
                key = (src, tgt, r.type)
                if key in seen_rels:
                    seen_rels[key].sources = sorted(set(seen_rels[key].sources) | set(r.sources))
                else:
                    seen_rels[key] = Relation(source=src, target=tgt, type=r.type, sources=list(r.sources))
            else:
                orphans.append(r)
        clean_rels = list(seen_rels.values())
        report.orphan_relations = orphans

        # ---- 3. event-time conflict detection ----------------------------
        # events about the same (subject, object) but different times
        event_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for ev in graph.events:
            if ev.subject and ev.object and ev.time:
                event_groups[(normalize_name(ev.subject), normalize_name(ev.object))].append(ev.time)
        for (subj, obj), times in event_groups.items():
            uniq = sorted(set(times))
            if len(uniq) > 1:
                conflicts.append(
                    {"event": f"{subj} -> {obj}", "conflicting_times": uniq}
                )

        report.conflicts = conflicts
        if not conflicts:
            report.notes.append("未检测到明显冲突。")

        # ---- 3.5 remap event endpoints to canonical entity names ----
        # Entities above were merged into canonical names, but events were passed
        # through untouched. Pointing each event's subject/object at the same
        # canonical name lets the graph writer link it to the entity; otherwise an
        # event whose endpoint came back as an alias/case variant (e.g. "nvidia"
        # vs canonical "NVIDIA") would be stored as a disconnected node and stay
        # unreachable during QA traversal.
        def _canon_endpoint(x: str) -> str:
            if not x:
                return x
            return name_index.get(normalize_name(x), x)

        clean_events: list[Event] = []
        for ev in graph.events:
            clean_events.append(
                Event(
                    type=ev.type,
                    subject=_canon_endpoint(ev.subject),
                    object=_canon_endpoint(ev.object),
                    time=ev.time,
                    summary=ev.summary,
                    sources=list(ev.sources),
                )
            )

        # ---- 4. drop orphan nodes & disconnected events ------------------
        # An entity with neither a relation nor an event link becomes a 0-degree
        # node ("orphan node"): unreachable in QA traversal and noise in the UI.
        # Likewise an event whose subject/object are not real entities would be
        # stored as a disconnected Event node. Drop both so the final graph has no
        # isolated nodes.
        connected: set[str] = set()
        for r in clean_rels:
            connected.add(normalize_name(r.source))
            connected.add(normalize_name(r.target))

        canon_events: list[Event] = []
        dropped_events: list[str] = []
        for ev in clean_events:
            has_subj = bool(ev.subject) and normalize_name(ev.subject) in name_index
            has_obj = bool(ev.object) and normalize_name(ev.object) in name_index
            if has_subj or has_obj:
                if has_subj:
                    connected.add(normalize_name(ev.subject))
                if has_obj:
                    connected.add(normalize_name(ev.object))
                canon_events.append(ev)
            else:
                dropped_events.append(ev.summary or f"{ev.type}: {ev.subject}->{ev.object}")

        orphan_nodes: list[str] = []
        kept_entities: list[Entity] = []
        for e in merged:
            if normalize_name(e.name) in connected:
                kept_entities.append(e)
            else:
                orphan_nodes.append(e.name)

        report.orphan_nodes = orphan_nodes
        report.dropped_events = dropped_events
        if orphan_nodes or dropped_events:
            report.notes.append(
                f"移除 {len(orphan_nodes)} 个孤儿节点（无关系/事件连接）、"
                f"{len(dropped_events)} 个孤立事件。"
            )

        cleaned = GraphData(
            entities=kept_entities,
            relations=clean_rels,
            events=canon_events,
        )
        return cleaned, report
