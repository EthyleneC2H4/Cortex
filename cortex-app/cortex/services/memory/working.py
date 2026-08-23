"""WorkingMemory：会话级工作记忆（线程安全 dict，按 session_id 分桶，TTL 过期）。

条目带 ``importance``（供巩固器把高重要度条目升级为情景记忆入库）；
``time_fn`` 注入时钟便于测试（默认 time.time）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

DEFAULT_TTL_SECONDS = 1800.0


@dataclass
class WorkingEntry:
    """工作记忆条目：值 + 重要性 + 过期时刻。"""

    value: Any
    importance: float = 0.5
    expires_at: float = float("inf")
    created_at: float = 0.0

    def expired(self, now: float) -> bool:
        return now >= self.expires_at


class WorkingMemory:
    """按 session_id 分桶的 TTL 字典；所有方法线程安全。"""

    def __init__(self, time_fn: Callable[[], float] | None = None) -> None:
        self._time_fn = time_fn or time.time
        self._lock = threading.RLock()
        # session_id -> {key: WorkingEntry}
        self._buckets: dict[str, dict[str, WorkingEntry]] = {}

    # ---------------------------------------------------------------- #
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        importance: float = 0.5,
        session_id: str = "default",
    ) -> None:
        """写入/覆盖一条工作记忆。

        Args:
            key: 条目键（同桶内覆盖）
            value: 任意值（通常为文本）
            ttl_seconds: 存活秒数
            importance: 重要性 0..1（≥0.7 会被巩固器转为情景记忆）
            session_id: 会话桶
        """
        now = self._time_fn()
        with self._lock:
            bucket = self._buckets.setdefault(session_id, {})
            bucket[key] = WorkingEntry(
                value=value,
                importance=float(importance),
                expires_at=now + float(ttl_seconds),
                created_at=now,
            )

    def get(self, key: str, session_id: str = "default", default: Any = None) -> Any:
        """读取；已过期返回 default 并顺手剔除。"""
        now = self._time_fn()
        with self._lock:
            entry = self._buckets.get(session_id, {}).get(key)
            if entry is None:
                return default
            if entry.expired(now):
                del self._buckets[session_id][key]
                return default
            return entry.value

    def peek(self, key: str, session_id: str = "default") -> "WorkingEntry | None":
        """取完整条目（含 importance/expires_at）；过期视为不存在。"""
        now = self._time_fn()
        with self._lock:
            entry = self._buckets.get(session_id, {}).get(key)
            if entry is None or entry.expired(now):
                return None
            return entry

    def remove(self, key: str, session_id: str = "default") -> bool:
        """显式移除（巩固消费后调用），返回是否确有删除。"""
        with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket and key in bucket:
                del bucket[key]
                return True
            return False

    def clear(self, session_id: str) -> int:
        """清空一个会话桶，返回清除条数。"""
        with self._lock:
            bucket = self._buckets.pop(session_id, {})
            return len(bucket)

    def sweep(self) -> int:
        """清扫所有桶中的过期条目，返回清除总数。"""
        now = self._time_fn()
        removed = 0
        with self._lock:
            for session_id in list(self._buckets.keys()):
                bucket = self._buckets[session_id]
                for key in [k for k, e in bucket.items() if e.expired(now)]:
                    del bucket[key]
                    removed += 1
                if not bucket:
                    del self._buckets[session_id]
        return removed

    def items(self) -> Iterator[tuple[str, str, WorkingEntry]]:
        """遍历全部未过期条目 → (session_id, key, entry)。"""
        now = self._time_fn()
        with self._lock:
            snapshot = [
                (sid, k, e)
                for sid, bucket in self._buckets.items()
                for k, e in bucket.items()
                if not e.expired(now)
            ]
        yield from snapshot

    def size(self) -> int:
        """当前未过期条目总数（跨桶）。"""
        return sum(1 for _ in self.items())
