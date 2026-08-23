"""MemoryService 门面：记忆全链路唯一入口（工具层 / ContextEngine 只看它）。

组装 WorkingMemory + MemoryStore + Consolidator + ForgettingPolicy + 写队列：
- remember：importance 缺省由 weak_llm 打分（失败回退 0.5），嵌入后经异步队列或
  直写入库；
- recall：向量相似 + composite 加权重排，命中即 touch() 回写访问统计；
- recall_for_context：给 ContextEngine 的紧凑高价值记忆列表（无需查询词）；
- summary / forget / consolidate：摘要生成与策略委托。
"""

from __future__ import annotations

import re
import time
from typing import Callable, Optional

from cortex.llm.resilience import resilient_invoke

from .consolidation import Consolidator
from .forgetting import ForgettingPolicy
from .queue import MemoryWriteQueue
from .scoring import composite_score
from .store import MemoryStore, QdrantMemoryIndex
from .types import MemoryItem, MemoryType
from .working import WorkingMemory


class MemoryService:
    """聚合 working/store/consolidator/policy/queue；vector_index 缺省用 Qdrant 本地。"""

    def __init__(
        self,
        cfg,
        db,
        embedder,
        vector_index=None,
        weak_llm=None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.embedder = embedder
        self.weak_llm = weak_llm
        if vector_index is None:
            dim = int(getattr(embedder, "dim", 128) or 128)
            vector_index = QdrantMemoryIndex(cfg.data_dir / "qdrant_local", dim=dim)
        self.store = MemoryStore(db, vector_index)
        self.working_memory = WorkingMemory(time_fn=time_fn or time.time)
        self.consolidator = Consolidator(
            store=self.store,
            cfg=cfg,
            weak_llm=weak_llm,
            working=self.working_memory,
            embedder=embedder,
        )
        self.policy = ForgettingPolicy(store=self.store, cfg=cfg)
        self._queue: Optional[MemoryWriteQueue] = None
        if getattr(cfg, "memory_async_writes", False):
            # sink 收 (item, vector) 二元组；队列满丢最旧，flush 可等待排空
            self._queue = MemoryWriteQueue(sink=self._write_through, maxsize=1000)

    # ---------------------------------------------------------------- #
    # 写路径
    # ---------------------------------------------------------------- #
    def remember(
        self,
        content: str,
        type: str = "episodic",
        importance: Optional[float] = None,
        source: str = "",
        metadata: Optional[dict] = None,
        user_id: str = "default",
    ) -> MemoryItem:
        """写入一条持久记忆并返回条目（异步模式下落库由队列异步完成）。

        Args:
            content: 记忆文本（非空）
            type: working/episodic/semantic/procedural（大小写不敏感；
                  procedural 在 cfg.procedural_memory_enabled=False 时拒绝）
            importance: 重要性 0..1；缺省交 weak_llm 打分，失败回退 0.5
            source: 来源标记（如工具名 / 会话 id）
            metadata: 附加结构化信息
            user_id: 用户隔离键
        """
        text = str(content)
        if not text.strip():
            raise ValueError("记忆内容不能为空")
        mt = MemoryType.parse(type)
        if mt is MemoryType.PROCEDURAL and not bool(self.cfg.procedural_memory_enabled):
            raise ValueError(
                "procedural 记忆未启用（cfg.procedural_memory_enabled=False），拒绝写入"
            )
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata 必须是 dict")
        score = float(importance) if importance is not None else self._score_importance(text)
        score = min(1.0, max(0.0, score))
        item = MemoryItem(
            user_id=user_id or "default",
            type=mt.value,
            content=text,
            importance=score,
            source=source,
            metadata=dict(metadata or {}),
        )
        vector = self.embedder.embed_one(text)
        if self._queue is not None:
            self._queue.put((item, list(vector)))
        else:
            self.store.add(item, list(vector))
        return item

    def _write_through(self, payload: tuple[MemoryItem, list[float]]) -> None:
        """写队列 sink：真正的入库动作（元数据 + 向量）。"""
        item, vector = payload
        self.store.add(item, vector)

    def flush_writes(self, timeout: float = 10.0) -> bool:
        """排空异步写队列（同步模式恒 True）；recall 前需要强一致时调用。"""
        if self._queue is None:
            return True
        return self._queue.flush(timeout=timeout)

    # ---------------------------------------------------------------- #
    # 读路径
    # ---------------------------------------------------------------- #
    def recall(
        self, query: str, top_k: int = 5, type: Optional[str] = None
    ) -> list[dict]:
        """向量召回 + composite 重排；命中即 touch()。

        Args:
            query: 查询文本
            top_k: 返回条数上限
            type: 可选类型过滤（大小写不敏感）

        Returns:
            [{id, content, type, importance, created_at, similarity, score, access_count}]
        """
        text = str(query)
        if not text.strip():
            raise ValueError("查询词不能为空")
        type_value = MemoryType.parse(type).value if type is not None else None
        vector = self.embedder.embed_one(text)
        # 超采样后按综合分重排再截断
        candidates = self.store.search_similar(
            vector, top_k=max(int(top_k) * 4, 20), type=type_value
        )
        scored = [
            (composite_score(sim, item, self.cfg), sim, item)
            for item, sim in candidates
        ]
        scored.sort(key=lambda triple: triple[0], reverse=True)
        results: list[dict] = []
        for total, sim, item in scored[: int(top_k)]:
            self.store.touch(item.id)
            results.append(
                {
                    "id": item.id,
                    "content": item.content,
                    "type": item.type,
                    "importance": item.importance,
                    "created_at": item.created_at,
                    "similarity": float(sim),
                    "score": float(total),
                    "access_count": item.access_count + 1,
                }
            )
        return results

    def recall_for_context(self, limit: int = 10, min_importance: float = 0.3) -> list[dict]:
        """给 ContextEngine 的紧凑记忆列表：最近写入的高重要度 active 条目。"""
        rows = self.db.query(
            "SELECT content, type, importance, created_at FROM memories "
            "WHERE status='active' AND importance>=? "
            "ORDER BY created_at DESC LIMIT ?",
            (float(min_importance), int(limit)),
        )
        return [
            {
                "content": r["content"],
                "type": r["type"],
                "importance": float(r["importance"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def summary(self, limit: int = 20) -> str:
        """把 top 记忆交给 weak_llm 生成一段摘要；无弱模型则返回拼接文本。"""
        entries = self.recall_for_context(limit=limit, min_importance=0.0)
        if not entries:
            return ""
        joined = "\n".join(f"- ({e['type']}) {e['content']}" for e in entries)
        if self.weak_llm is None:
            return joined
        messages = [
            {"role": "system", "content": "你是记忆摘要助手：把记忆条目总结成一段连贯的话。"},
            {"role": "user", "content": f"请总结以下记忆：\n{joined}"},
        ]
        try:
            resp = resilient_invoke(self.weak_llm, messages, retries=1)
            text = str(getattr(resp, "content", "") or "").strip()
        except Exception:  # noqa: BLE001 —— 摘要失败降级为拼接文本
            return joined
        return text or joined

    # ---------------------------------------------------------------- #
    # 策略委托与生命周期
    # ---------------------------------------------------------------- #
    def consolidate(self) -> dict:
        """委托 Consolidator：working→episodic→semantic + 近重复合并。"""
        return self.consolidator.run()

    def forget(self) -> dict:
        """委托 ForgettingPolicy：三策略软遗忘 + 容量归档。"""
        return self.policy.run()

    def stats(self) -> dict:
        """各类型 active 计数 + 队列计数快照。"""
        counts = {
            mt.value: self.store.count(type=mt.value) for mt in MemoryType
        }
        return {
            **counts,
            "total_active": self.store.count(),
            "queue": self._queue.stats if self._queue is not None else None,
        }

    def close(self) -> None:
        """停机：排空并关闭写队列、释放 qdrant 本地文件锁、关闭 sqlite 连接。"""
        if self._queue is not None:
            self._queue.close(timeout=10.0)
        try:
            closer = getattr(self.store.vector_index, "close", None)
            if callable(closer):
                closer()
        except Exception:  # noqa: BLE001 —— 关闭路径尽力而为
            pass
        try:
            self.db.close()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- #
    def _score_importance(self, text: str) -> float:
        """weak_llm 打重要性分（0..1）；缺失/失败回退 0.5。"""
        if self.weak_llm is None:
            return 0.5
        messages = [
            {
                "role": "system",
                "content": "你是记忆重要性打分器：只输出一个 0 到 1 之间的小数。",
            },
            {
                "role": "user",
                "content": f"请为这条记忆输出重要性分数（只输出小数）：\n{text}",
            },
        ]
        try:
            resp = resilient_invoke(self.weak_llm, messages, retries=1)
            raw = str(getattr(resp, "content", "") or "")
            match = re.search(r"\d+(?:\.\d+)?", raw)
            value = float(match.group()) if match else 0.5
        except Exception:  # noqa: BLE001 —— 打分失败回退默认值
            return 0.5
        return min(1.0, max(0.0, value))
