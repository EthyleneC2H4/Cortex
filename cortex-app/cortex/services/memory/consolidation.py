"""Consolidator：记忆巩固（working→episodic→semantic + 近重复合并）。

三条规则（全部软处理，绝不物理删除）：
a) working→episodic：WorkingMemory 中 importance≥0.7 的未过期条目转 EPISODIC 入库，
   随后从工作记忆移除（防重复巩固）；
b) episodic→semantic：importance≥0.8 且 access_count≥2 的情景记忆升级为语义记忆
   （importance×1.1，上限 1.0），原条目 archived 且 importance×0.5；
c) 近重复合并：SEMANTIC 两两余弦 >0.92 成组，有 weak_llm 时生成一句合并摘要作为
   新 SEMANTIC 条目、原条目 archived；无 weak_llm 整体跳过（embedder 注入算余弦）。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from cortex.llm.resilience import resilient_invoke

from .scoring import cosine
from .store import MemoryStore
from .types import MemoryItem, MemoryType
from .working import WorkingMemory

# 巩固阈值常量（规划文档 §记忆服务）
WORKING_PROMOTE_IMPORTANCE = 0.7  # working→episodic 重要性门槛
EPISODIC_UPGRADE_IMPORTANCE = 0.8  # episodic→semantic 重要性门槛
EPISODIC_UPGRADE_ACCESSES = 2  # episodic→semantic 访问次数门槛
DUP_COSINE_THRESHOLD = 0.92  # 近重复合并的余弦阈值


class Consolidator:
    """依赖 store + cfg；working/embedder/weak_llm 可选注入（缺则跳过对应规则）。"""

    def __init__(
        self,
        store: MemoryStore,
        cfg,
        weak_llm=None,
        working: Optional[WorkingMemory] = None,
        embedder=None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.weak_llm = weak_llm
        self.working = working
        self.embedder = embedder

    # ---------------------------------------------------------------- #
    def run(self, now: datetime | str | None = None) -> dict:
        """执行一轮巩固，返回报告 dict（各规则的明细列表）。"""
        report: dict = {
            "working_to_episodic": [],
            "episodic_to_semantic": [],
            "merged_duplicates": [],
            "errors": [],
        }
        try:
            self._promote_working(report)
        except Exception as exc:  # noqa: BLE001 —— 单条规则失败不拖垮整轮
            report["errors"].append(f"working_to_episodic: {exc}")
        try:
            self._upgrade_episodic()
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"episodic_to_semantic: {exc}")
        try:
            self._merge_duplicates(report)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"merge_duplicates: {exc}")
        return report

    # ---------------------------------------------------------------- #
    def _promote_working(self, report: dict) -> None:
        """(a) 高重要度工作记忆 → 情景记忆入库。"""
        if self.working is None:
            return
        for session_id, key, entry in list(self.working.items()):
            if entry.importance < WORKING_PROMOTE_IMPORTANCE:
                continue
            content = str(entry.value)
            item = MemoryItem(
                type=MemoryType.EPISODIC.value,
                content=content,
                importance=float(entry.importance),
                source="working_consolidation",
                metadata={"session_id": session_id, "key": key},
            )
            vector = self.embedder.embed_one(content) if self.embedder is not None else None
            self.store.add(item, vector)
            self.working.remove(key, session_id)
            report["working_to_episodic"].append(
                {"id": item.id, "session_id": session_id, "key": key}
            )

    def _upgrade_episodic(self) -> None:
        """(b) 高重要度高频情景记忆 → 语义记忆；原条目降权归档。"""
        for item in self.store.iterate(type=MemoryType.EPISODIC.value):
            if (
                item.importance < EPISODIC_UPGRADE_IMPORTANCE
                or item.access_count < EPISODIC_UPGRADE_ACCESSES
            ):
                continue
            new_importance = min(1.0, float(item.importance) * 1.1)
            semantic = MemoryItem(
                type=MemoryType.SEMANTIC.value,
                content=item.content,
                importance=new_importance,
                source=item.source or "consolidation",
                metadata={"upgraded_from": item.id},
            )
            vector = (
                self.embedder.embed_one(item.content) if self.embedder is not None else None
            )
            self.store.add(semantic, vector)
            # 原情景条目：软归档 + 重要性减半（绝不删除）
            self.store.update_importance(item.id, float(item.importance) * 0.5)
            self.store.set_status(item.id, "archived")

    def _merge_duplicates(self, report: dict) -> None:
        """(c) 语义记忆近重复合并（需 weak_llm + embedder，缺一跳过）。"""
        items = self.store.iterate(type=MemoryType.SEMANTIC.value)
        if len(items) < 2 or self.embedder is None or self.weak_llm is None:
            return
        vectors = self.embedder.embed([it.content for it in items])
        for group in self._similarity_groups(items, vectors):
            if len(group) < 2:
                continue
            merged = self._merge_group(group)
            if merged is None:
                continue
            report["merged_duplicates"].append(merged)

    def _similarity_groups(self, items, vectors) -> list[list]:
        """两两余弦 > 阈值 → 并查集连通分量。"""
        n = len(items)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if cosine(vectors[i], vectors[j]) > DUP_COSINE_THRESHOLD:
                    parent[find(i)] = find(j)

        groups: dict[int, list] = {}
        for idx in range(n):
            groups.setdefault(find(idx), []).append(items[idx])
        return list(groups.values())

    def _merge_group(self, group: list[MemoryItem]) -> Optional[dict]:
        """用弱模型把一组近重复记忆合并为一句摘要；失败返回 None（原样保留）。"""
        numbered = "\n".join(f"{i + 1}. {it.content}" for i, it in enumerate(group))
        messages = [
            {
                "role": "system",
                "content": "你是记忆整理助手：把多条相似记忆合并成一句话摘要。",
            },
            {
                "role": "user",
                "content": f"请把下列相似记忆合并成一句摘要，只输出摘要本身：\n{numbered}",
            },
        ]
        try:
            resp = resilient_invoke(self.weak_llm, messages, retries=1)
            summary_text = re.sub(r"\s+", " ", str(getattr(resp, "content", "") or "")).strip()
        except Exception:  # noqa: BLE001 —— 合并失败保留原条目
            return None
        if not summary_text:
            return None
        merged_item = MemoryItem(
            type=MemoryType.SEMANTIC.value,
            content=summary_text[:2000],
            importance=max(it.importance for it in group),
            source="consolidation_merge",
            metadata={"merged_from": [it.id for it in group]},
        )
        self.store.add(merged_item, self.embedder.embed_one(summary_text[:2000]))
        for it in group:
            self.store.set_status(it.id, "archived")
        return {"into": merged_item.id, "summary": summary_text[:120], "archived": [it.id for it in group]}
