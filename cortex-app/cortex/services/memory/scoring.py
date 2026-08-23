"""记忆打分纯函数：recency 衰减 / composite 加权分 / 余弦相似度。

composite = w_sim·similarity + w_importance·importance + w_recency·exp(-λ·age_days)
权重与 λ 全部从 CortexConfig 读取（memory_w_* / memory_decay_lambda），
无任何 IO，精确可测。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

# 兼容多种时间戳格式（db.now_iso 为 "%Y-%m-%dT%H:%M:%S%z"）
_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)


def parse_ts(value) -> Optional[datetime]:
    """宽松解析时间戳：datetime 原样返回；字符串逐格式尝试；失败返回 None。"""
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:  # ISO 8601 带 Z 结尾等变体
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _to_utc_naive(dt: datetime) -> datetime:
    """统一折算为 naive UTC，避免 aware/naive 相减抛 TypeError。"""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def age_days(created_at, now=None) -> float:
    """创建时刻距 now 的天数；now 缺省取当前 UTC。解析失败按 0 天（不惩罚）。"""
    dt = parse_ts(created_at)
    if dt is None:
        return 0.0
    ref = parse_ts(now) if not isinstance(now, datetime) else now
    if ref is None:
        ref = datetime.now(timezone.utc).replace(tzinfo=None)
    delta = _to_utc_naive(ref) - _to_utc_naive(dt)
    return max(0.0, delta.total_seconds() / 86400.0)


def recency_score(days: float, lambda_: float = 0.05) -> float:
    """时间衰减分：exp(-λ · age_days)。"""
    return math.exp(-float(lambda_) * float(days))


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度；任一向量零范数时返回 0.0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def composite_score(similarity: float, item, cfg, now=None) -> float:
    """综合分：w_sim·sim + w_imp·importance + w_rec·recency(age)。

    Args:
        similarity: 向量相似度（容量淘汰等场景传 0）
        item: MemoryItem（读 importance 与 created_at）
        cfg: CortexConfig（读 memory_w_similarity / memory_w_importance /
             memory_w_recency / memory_decay_lambda）
        now: 可注入的参考时刻（datetime 或时间戳字符串），缺省当前 UTC
    """
    days = age_days(item.created_at, now=now)
    recency = recency_score(days, lambda_=getattr(cfg, "memory_decay_lambda", 0.05))
    return (
        float(cfg.memory_w_similarity) * float(similarity)
        + float(cfg.memory_w_importance) * float(item.importance)
        + float(cfg.memory_w_recency) * recency
    )
