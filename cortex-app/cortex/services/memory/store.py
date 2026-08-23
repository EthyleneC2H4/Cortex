"""MemoryStore：memories 表 CRUD + 向量索引适配（Qdrant 本地模式）。

- ``VectorBackend`` Protocol：upsert / search / delete 三个方法的最小契约，
  测试可注入纯 dict 的 Fake 实现；
- ``MemoryStore``：sqlite 元数据读写（复用 Database），查询一律带
  ``status='active'`` 过滤（软删除语义：forgotten/archived 对检索不可见）；
- ``QdrantMemoryIndex``：qdrant-client 本地文件模式 collection
  ``cortex_memory``（cosine）。点 id 只接受 uint/UUID，用 uuid5(记忆 id)
  稳定映射 → 重复 upsert 幂等覆盖。独立实现，不 import knowledge 包。
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

from ..db import Database, now_iso
from .types import MemoryItem

_COLUMNS = (
    "id",
    "user_id",
    "type",
    "content",
    "importance",
    "created_at",
    "last_access",
    "access_count",
    "source",
    "status",
    "metadata",
)

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "cortex-memory")
_UPSERT_LOCK = threading.Lock()


@runtime_checkable
class VectorBackend(Protocol):
    """记忆向量索引最小契约（id ↔ 向量，payload 自定义）。"""

    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[dict]) -> None: ...

    def search(self, vector: List[float], top_k: int = 10) -> list[tuple[str, float]]: ...

    def delete(self, ids: List[str]) -> None: ...


def _point_id(memory_id: str) -> str:
    """记忆 id → qdrant 合法点 id（UUID 字符串，稳定映射幂等覆盖）。"""
    return str(uuid.uuid5(_NAMESPACE, memory_id))


class QdrantMemoryIndex:
    """qdrant 本地文件模式向量索引（与 docker 同 API；close() 释放排他文件锁）。"""

    def __init__(self, path_or_cfg, dim: int, collection: str = "cortex_memory",
                 client=None) -> None:
        from qdrant_client import QdrantClient, models  # 延迟导入

        self._models = models
        if isinstance(path_or_cfg, (str, Path)):
            path = Path(path_or_cfg)
        else:  # CortexConfig 等：取 data_dir 派生路径
            path = Path(path_or_cfg.data_dir) / "qdrant_local"
        path.mkdir(parents=True, exist_ok=True)
        self.collection = collection
        self.dim = int(dim)
        # client= 注入共享实例：本地模式同路径排他锁，知识库/记忆必须共用一个 client；
        # 注入的 client 生命周期归调用方，close() 不重复释放
        self._owns_client = client is None
        self.client = client if client is not None else QdrantClient(path=str(path))
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )

    # ---------------------------------------------------------------- #
    def upsert(self, ids, vectors, payloads) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=_point_id(mid), vector=vec, payload=dict(pl))
            for mid, vec, pl in zip(ids, vectors, payloads)
        ]
        with _UPSERT_LOCK:
            for i in range(0, len(points), 128):
                self.client.upsert(collection_name=self.collection, points=points[i : i + 128])

    def search(self, vector, top_k=10) -> list[tuple[str, float]]:
        hits = self.client.query_points(
            collection_name=self.collection, query=list(vector), limit=int(top_k), with_payload=True
        ).points
        return [(h.payload.get("id", ""), float(h.score)) for h in hits]

    def delete(self, ids) -> None:
        ids = [str(i) for i in ids]
        if not ids:
            return
        flt = self._models.Filter(
            must=[self._models.FieldCondition(key="id", match=self._models.MatchAny(any=ids))]
        )
        with _UPSERT_LOCK:
            self.client.delete(collection_name=self.collection, points_selector=flt)

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection).count)

    def close(self) -> None:
        """释放本地模式排他文件锁（进程重启/再打开前必须调用）。

        注入共享 client 时不动它——锁的释放归注入方（KnowledgeService.close）。
        """
        if not self._owns_client:
            return
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 —— 关闭路径尽力而为
            pass


class MemoryStore:
    """sqlite memories 表 + 向量索引的门面：全部软删除（status 标记，绝不 DELETE）。"""

    def __init__(self, db: Database, vector_index: Optional[VectorBackend] = None) -> None:
        self.db = db
        self.vector_index = vector_index

    # ---------------------------------------------------------------- #
    # 写路径
    # ---------------------------------------------------------------- #
    def add(self, item: MemoryItem, vector: Optional[List[float]] = None) -> MemoryItem:
        """插入或覆盖一条记忆；vector 非空时同步 upsert 向量索引。"""
        row = item.to_row()
        self.db.execute(
            f"INSERT OR REPLACE INTO memories({','.join(_COLUMNS)}) "
            f"VALUES({','.join('?' for _ in _COLUMNS)})",
            tuple(row[c] for c in _COLUMNS),
        )
        if vector and self.vector_index is not None:
            self.vector_index.upsert(
                [item.id],
                [list(vector)],
                [{"id": item.id, "type": item.type, "user_id": item.user_id}],
            )
        return item

    def touch(self, memory_id: str) -> None:
        """命中回写：access_count += 1 且刷新 last_access。"""
        self.db.execute(
            "UPDATE memories SET access_count=access_count+1, last_access=? WHERE id=?",
            (now_iso(), memory_id),
        )

    def set_status(self, memory_id: str, status: str) -> None:
        """软删除/归档/恢复：只改 status，绝不删行。"""
        self.db.execute("UPDATE memories SET status=? WHERE id=?", (status, memory_id))

    def update_importance(self, memory_id: str, importance: float) -> None:
        """直接改写重要性（巩固降权 ×0.5 等场景）。"""
        self.db.execute(
            "UPDATE memories SET importance=? WHERE id=?", (float(importance), memory_id)
        )

    def drop_vectors(self, memory_ids: List[str]) -> None:
        """尽力而为地从向量索引移除（索引缺失/异常均静默）。"""
        if not memory_ids or self.vector_index is None:
            return
        try:
            self.vector_index.delete(list(memory_ids))
        except Exception:  # noqa: BLE001 —— 索引清理失败不影响元数据一致性
            pass

    # ---------------------------------------------------------------- #
    # 读路径（默认只看 active）
    # ---------------------------------------------------------------- #
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """按 id 取（不限 status，供检查 archived/forgotten 条目）。"""
        row = self.db.query_one("SELECT * FROM memories WHERE id=?", (memory_id,))
        return MemoryItem.from_row(row) if row else None

    def iterate(
        self, type: Optional[str] = None, status: str = "active", user_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """列出条目（缺省 active；type/user_id 可选过滤），按 created_at 倒序。"""
        sql = "SELECT * FROM memories WHERE status=?"
        params: list = [status]
        if type is not None:
            sql += " AND type=?"
            params.append(type)
        if user_id is not None:
            sql += " AND user_id=?"
            params.append(user_id)
        sql += " ORDER BY created_at DESC, id DESC"
        return [MemoryItem.from_row(r) for r in self.db.query(sql, tuple(params))]

    def search_similar(
        self, vector: List[float], top_k: int = 10, type: Optional[str] = None
    ) -> list[tuple[MemoryItem, float]]:
        """向量召回 → sqlite 回表过滤（仅 active、可选 type）→ [(item, score)]。

        注意：候选上限由后端一次返回，调用方如需重排应自行放大 top_k 超采样。
        """
        if not vector or self.vector_index is None:
            return []
        try:
            hits = self.vector_index.search(list(vector), top_k=int(top_k))
        except Exception:  # noqa: BLE001 —— 索引故障退化为无结果而非崩溃
            return []
        out: list[tuple[MemoryItem, float]] = []
        seen: set[str] = set()
        for mid, score in hits:
            mid = str(mid)
            if not mid or mid in seen:
                continue
            seen.add(mid)
            item = self.get(mid)
            if item is None or item.status != "active":
                continue
            if type is not None and item.type != type:
                continue
            out.append((item, float(score)))
            if len(out) >= int(top_k):
                break
        return out

    def count(self, type: Optional[str] = None, status: str = "active") -> int:
        """计数（缺省 active；type 可选）。"""
        sql = "SELECT COUNT(*) AS n FROM memories WHERE status=?"
        params: list = [status]
        if type is not None:
            sql += " AND type=?"
            params.append(type)
        row = self.db.query_one(sql, tuple(params))
        return int(row["n"]) if row else 0

    def all_active_importance_below(self, threshold: float) -> List[MemoryItem]:
        """重要性低于阈值的 active 条目（升序）。"""
        rows = self.db.query(
            "SELECT * FROM memories WHERE status='active' AND importance<? "
            "ORDER BY importance ASC",
            (float(threshold),),
        )
        return [MemoryItem.from_row(r) for r in rows]

    def oldest_low_score(self, n: int) -> List[MemoryItem]:
        """分数最低、最旧的 n 条 active 记忆（importance ASC, created_at ASC）。

        容量淘汰的粗筛辅助；精确 composite 排序由遗忘策略在 Python 侧完成。
        """
        rows = self.db.query(
            "SELECT * FROM memories WHERE status='active' "
            "ORDER BY importance ASC, created_at ASC LIMIT ?",
            (int(n),),
        )
        return [MemoryItem.from_row(r) for r in rows]
