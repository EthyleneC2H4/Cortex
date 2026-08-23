"""记忆类型与记忆条目：MemoryType 枚举 + MemoryItem 数据类。

MemoryItem 与 sqlite ``memories`` 表行互转（``to_row`` / ``from_row``）；
metadata 列以 JSON 字符串存储，解析失败时容错回退为空 dict，绝不抛异常。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..db import now_iso


def new_memory_id() -> str:
    """16 位十六进制短 id（uuid4 截断，碰撞概率可忽略）。"""
    return uuid.uuid4().hex[:16]


class MemoryType(str, Enum):
    """四类记忆：工作 / 情景 / 语义 / 程序性（程序性默认禁用，见 CortexConfig）。"""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

    @classmethod
    def parse(cls, value) -> "MemoryType":
        """大小写不敏感解析：接受枚举实例 / 枚举名 / 枚举值；非法抛 ValueError。

        Args:
            value: MemoryType 实例或任意字符串（如 "EPISODIC"、"episodic"）
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            for member in cls:
                if token in (member.value, member.name.lower()):
                    return member
        raise ValueError(f"未知记忆类型: {value!r}（可选 working/episodic/semantic/procedural）")


@dataclass
class MemoryItem:
    """一条持久化记忆（对应 memories 表一行）。"""

    id: str = field(default_factory=new_memory_id)
    user_id: str = "default"
    type: str = MemoryType.EPISODIC.value
    content: str = ""
    importance: float = 0.5
    created_at: str = field(default_factory=now_iso)
    last_access: str = field(default_factory=now_iso)
    access_count: int = 0
    source: str = ""
    status: str = "active"
    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- #
    def to_row(self) -> dict:
        """转为 sqlite 行 dict（metadata 序列化为 JSON 字符串）。"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type,
            "content": self.content,
            "importance": float(self.importance),
            "created_at": self.created_at,
            "last_access": self.last_access,
            "access_count": int(self.access_count),
            "source": self.source,
            "status": self.status,
            "metadata": json.dumps(self.metadata or {}, ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row: dict) -> "MemoryItem":
        """从 sqlite 行 dict 还原；metadata 非法 JSON / 非 dict 时容错为空 dict。"""
        meta = _coerce_metadata(row.get("metadata"))
        return cls(
            id=row.get("id") or new_memory_id(),
            user_id=row.get("user_id") or "default",
            type=row.get("type") or MemoryType.EPISODIC.value,
            content=row.get("content") or "",
            importance=float(row.get("importance") or 0.0),
            created_at=row.get("created_at") or "",
            last_access=row.get("last_access") or "",
            access_count=int(row.get("access_count") or 0),
            source=row.get("source") or "",
            status=row.get("status") or "active",
            metadata=meta,
        )

    # ---------------------------------------------------------------- #
    def age_days(self, now: datetime | str | None = None) -> float:
        """距创建时刻的天数（委托 scoring.age_days，便于调用方少一次导入）。"""
        from .scoring import age_days as _age_days

        return _age_days(self.created_at, now=now)


def _coerce_metadata(raw) -> dict:
    """metadata 列容错解析：dict 原样 / JSON 字符串解析 / 其余回退空 dict。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)) and raw:
        text = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}
