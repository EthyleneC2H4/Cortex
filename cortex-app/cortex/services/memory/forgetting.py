"""ForgettingPolicy：三策略遗忘（全部软删 status='forgotten'/'archived'，绝不 DELETE）。

1) 重要性策略：importance < cfg.memory_forget_importance → forgotten；
2) 时间策略：age > memory_time_forget_days 且 importance < memory_time_forget_importance
   → forgotten；
3) 容量策略：active 总数超过 memory_capacity 时，按 composite 分数
   （similarity 视为 0）从低到高把超额部分 archived。
"""

from __future__ import annotations

from datetime import datetime

from .scoring import age_days, composite_score
from .store import MemoryStore

STATUS_FORGOTTEN = "forgotten"
STATUS_ARCHIVED = "archived"


class ForgettingPolicy:
    """依赖 store + cfg；run() 幂等可反复执行。"""

    def __init__(self, store: MemoryStore, cfg) -> None:
        self.store = store
        self.cfg = cfg

    # ---------------------------------------------------------------- #
    def run(self, now: datetime | str | None = None) -> dict:
        """顺序执行三策略，返回报告（各策略处理的记忆 id 列表 + 前后 active 数）。"""
        report: dict = {
            "active_before": self.store.count(),
            "forgotten_low_importance": [],
            "forgotten_old": [],
            "archived_for_capacity": [],
        }

        # 策略 1：重要性不足
        for item in self.store.iterate():
            if float(item.importance) < float(self.cfg.memory_forget_importance):
                self._forget(item.id)
                report["forgotten_low_importance"].append(item.id)

        # 策略 2：过旧且不重要
        days_limit = int(self.cfg.memory_time_forget_days)
        imp_limit = float(self.cfg.memory_time_forget_importance)
        for item in self.store.iterate():
            if age_days(item.created_at, now=now) > days_limit and float(
                item.importance
            ) < imp_limit:
                self._forget(item.id)
                report["forgotten_old"].append(item.id)

        # 策略 3：容量超额 → composite 低者归档
        capacity = int(self.cfg.memory_capacity)
        actives = self.store.iterate()
        excess = len(actives) - capacity
        if excess > 0:
            scored = sorted(
                ((composite_score(0.0, it, self.cfg, now=now), it) for it in actives),
                key=lambda pair: pair[0],
            )
            for _, victim in scored[:excess]:
                self.store.set_status(victim.id, STATUS_ARCHIVED)
                self.store.drop_vectors([victim.id])
                report["archived_for_capacity"].append(victim.id)

        report["active_after"] = self.store.count()
        return report

    # ---------------------------------------------------------------- #
    def _forget(self, memory_id: str) -> None:
        """单条软遗忘：标记 forgotten + 尽力清向量。"""
        self.store.set_status(memory_id, STATUS_FORGOTTEN)
        self.store.drop_vectors([memory_id])
