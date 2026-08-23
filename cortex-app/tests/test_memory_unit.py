"""P4 单元测试：记忆 types/working/store/scoring/consolidation/forgetting/queue/service/tool。

全离线：Database(:memory:) + FakeVectorIndex + HashEmbedder + mock LLM，零网络零 docker。
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timedelta, timezone

import pytest
from hello_agents.tools import ToolRegistry

from cortex.config import CortexConfig
from cortex.services.db import Database, now_iso
from cortex.services.knowledge.embeddings import HashEmbedder
from cortex.services.memory import (
    Consolidator,
    ForgettingPolicy,
    MemoryService,
    MemoryStore,
    MemoryType,
    MemoryWriteQueue,
    WorkingMemory,
    composite_score,
    cosine,
    recency_score,
)
from cortex.services.memory.types import MemoryItem

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------- #
# 测试基建：假向量索引 / mock LLM / 服务构造
# --------------------------------------------------------------------- #
class FakeVectorIndex:
    """dict 暴力余弦检索，满足 VectorBackend 契约。"""

    def __init__(self) -> None:
        self.vectors: dict[str, tuple[list[float], dict]] = {}

    def upsert(self, ids, vectors, payloads) -> None:
        for mid, vec, pl in zip(ids, vectors, payloads):
            self.vectors[mid] = (list(vec), dict(pl))

    def search(self, vector, top_k=10) -> list[tuple[str, float]]:
        scored = [(cosine(vector, v), mid) for mid, (v, _) in self.vectors.items()]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(mid, s) for s, mid in scored[: int(top_k)]]

    def delete(self, ids) -> None:
        for mid in ids:
            self.vectors.pop(mid, None)


class FakeLLM:
    """恒定回复的 mock 弱模型。"""

    def __init__(self, content: str = "0.85") -> None:
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return type("R", (), {"content": self.content})()


class BrokenLLM:
    def invoke(self, messages):  # pragma: no cover - 断言不应走到
        raise RuntimeError("boom")


def make_cfg(tmp_path, **overrides) -> CortexConfig:
    defaults = dict(project_root=tmp_path, embed_provider="hash")
    defaults.update(overrides)
    return CortexConfig(**defaults)


def make_service(tmp_path, *, weak_llm=None, async_writes=False, **cfg_overrides) -> MemoryService:
    cfg = make_cfg(tmp_path, memory_async_writes=async_writes, **cfg_overrides)
    db = Database(":memory:")
    return MemoryService(
        cfg, db, HashEmbedder(dim=128), vector_index=FakeVectorIndex(), weak_llm=weak_llm
    )


def old_iso(days: float) -> str:
    """n 天前的时间戳字符串（aware UTC → 与 now_iso 同格式）。"""
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S%z")


# --------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------- #
class TestTypes:
    def test_roundtrip_via_sqlite(self, tmp_path):
        db = Database(":memory:")
        item = MemoryItem(
            user_id="u1",
            type=MemoryType.SEMANTIC.value,
            content="量子纠缠实验记录",
            importance=0.9,
            source="test",
            metadata={"session": "s1", "tags": ["物理"]},
        )
        row = item.to_row()
        cols = ",".join(row.keys())
        marks = ",".join("?" for _ in row)
        db.execute(f"INSERT INTO memories({cols}) VALUES({marks})", tuple(row.values()))
        back = MemoryItem.from_row(db.query_one("SELECT * FROM memories WHERE id=?", (item.id,)))
        assert back.id == item.id
        assert back.type == "semantic"
        assert back.importance == pytest.approx(0.9)
        assert back.metadata == {"session": "s1", "tags": ["物理"]}
        assert back.access_count == 0 and back.status == "active"

    def test_from_row_tolerates_bad_metadata(self):
        item = MemoryItem.from_row(
            {"id": "x", "type": "episodic", "metadata": "{not-json", "importance": None}
        )
        assert item.metadata == {}
        assert item.importance == 0.0
        assert MemoryItem.from_row({"id": "y", "metadata": None}).metadata == {}
        # 非 dict 的合法 JSON 也容错（包一层）
        wrapped = MemoryItem.from_row({"id": "z", "metadata": "[1,2]"})
        assert wrapped.metadata == {"value": [1, 2]}

    def test_type_parse_case_insensitive_and_rejects_unknown(self):
        assert MemoryType.parse("EPISODIC") is MemoryType.EPISODIC
        assert MemoryType.parse("Semantic") is MemoryType.SEMANTIC
        assert MemoryType.parse(MemoryType.WORKING) is MemoryType.WORKING
        with pytest.raises(ValueError):
            MemoryType.parse("galaxy")

    def test_id_is_16_hex(self):
        assert len(MemoryItem().id) == 16
        assert MemoryItem().id != MemoryItem().id


# --------------------------------------------------------------------- #
# WorkingMemory
# --------------------------------------------------------------------- #
class TestWorkingMemory:
    def _clocked(self):
        clock = {"t": 1000.0}
        wm = WorkingMemory(time_fn=lambda: clock["t"])
        return wm, clock

    def test_set_get_and_ttl_expiry_fake_clock(self):
        wm, clock = self._clocked()
        wm.set("plan", "研究草稿", ttl_seconds=10, importance=0.9)
        assert wm.get("plan") == "研究草稿"
        entry = wm.peek("plan")
        assert entry is not None and entry.importance == pytest.approx(0.9)
        clock["t"] = 1005.0  # 未过期
        assert wm.get("plan") == "研究草稿"
        clock["t"] = 1011.0  # 已过期
        assert wm.get("plan") is None
        assert wm.get("plan", default="gone") == "gone"

    def test_sweep_removes_expired_only(self):
        wm, clock = self._clocked()
        wm.set("a", 1, ttl_seconds=5)
        wm.set("b", 2, ttl_seconds=100)
        clock["t"] = 1010.0
        removed = wm.sweep()
        assert removed == 1
        assert wm.get("a") is None
        assert wm.get("b") == 2

    def test_clear_and_bucket_isolation(self):
        wm, _ = self._clocked()
        wm.set("k", "s1值", session_id="s1")
        wm.set("k", "s2值", session_id="s2")
        assert wm.get("k", session_id="s1") == "s1值"
        assert wm.clear("s1") == 1
        assert wm.get("k", session_id="s1") is None
        assert wm.get("k", session_id="s2") == "s2值"

    def test_items_reports_importance_and_skips_expired(self):
        wm, clock = self._clocked()
        wm.set("hot", "高价值", ttl_seconds=50, importance=0.9, session_id="s1")
        wm.set("cold", "低价值", ttl_seconds=50, importance=0.3, session_id="s2")
        wm.set("dead", "过期条目", ttl_seconds=5, session_id="s2")
        clock["t"] += 10
        entries = {(sid, k): e for sid, k, e in wm.items()}
        assert set(entries) == {("s1", "hot"), ("s2", "cold")}
        assert entries[("s1", "hot")].importance == pytest.approx(0.9)
        assert wm.size() == 2


# --------------------------------------------------------------------- #
# MemoryStore
# --------------------------------------------------------------------- #
class TestMemoryStore:
    def _store(self):
        vec = FakeVectorIndex()
        store = MemoryStore(Database(":memory:"), vec)
        return store, vec

    def test_add_get_roundtrip_with_fake_vector(self):
        store, vec = self._store()
        emb = HashEmbedder(dim=64)
        item = MemoryItem(type="semantic", content="RRF 融合算法", importance=0.8)
        store.add(item, emb.embed_one(item.content))
        back = store.get(item.id)
        assert back is not None and back.content == "RRF 融合算法"
        assert item.id in vec.vectors  # 向量同步入库

    def test_touch_increments_and_refreshes(self):
        store, _ = self._store()
        item = store.add(MemoryItem(content="待访问"))
        before = store.get(item.id)
        assert before.access_count == 0
        store.touch(item.id)
        store.touch(item.id)
        after = store.get(item.id)
        assert after.access_count == 2
        assert after.last_access >= before.last_access

    def test_set_status_hides_from_active_queries(self):
        store, vec = self._store()
        a = store.add(MemoryItem(content="会被归档"), None)
        b = store.add(MemoryItem(content="保持活跃"), None)
        store.set_status(a.id, "archived")
        actives = {i.id for i in store.iterate()}
        assert actives == {b.id}
        assert store.count() == 1
        assert store.count(status="archived") == 1
        assert store.get(a.id).status == "archived"  # get 不限 status
        # 向量召回结果中的非 active 条目被过滤
        vec.vectors[a.id] = ([1.0], {"id": a.id})
        hits = store.search_similar([1.0], top_k=5)
        assert [i.id for i, _ in hits] == []

    def test_search_similar_type_filter(self):
        store, _ = self._store()
        emb = HashEmbedder(dim=64)
        epi = MemoryItem(type="episodic", content="量子纠缠实验记录")
        sem = MemoryItem(type="semantic", content="番茄炒蛋做法")
        store.add(epi, emb.embed_one(epi.content))
        store.add(sem, emb.embed_one(sem.content))
        hits_all = store.search_similar(emb.embed_one("量子纠缠"), top_k=5)
        assert len(hits_all) >= 1
        hits_epi = store.search_similar(emb.embed_one("量子"), top_k=5, type="episodic")
        assert all(i.type == "episodic" for i, _ in hits_epi)

    def test_importance_and_low_score_queries(self):
        store, _ = self._store()
        for imp in (0.9, 0.4, 0.1):
            store.add(MemoryItem(importance=imp))
        lows = store.all_active_importance_below(0.5)
        assert [i.importance for i in lows][0] == pytest.approx(0.1)
        oldest = store.oldest_low_score(2)
        assert len(oldest) == 2
        assert oldest[0].importance <= oldest[1].importance


# --------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------- #
class TestScoring:
    def test_recency_exact_value(self):
        assert recency_score(10, lambda_=0.05) == pytest.approx(math.exp(-0.05 * 10))
        assert recency_score(0) == pytest.approx(1.0)
        assert recency_score(100) < recency_score(1)

    def test_composite_new_item_full_recency(self):
        cfg = CortexConfig(project_root="/tmp")
        now = datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)
        item = MemoryItem(created_at="2026-08-22T12:00:00+0000", importance=0.8)
        score = composite_score(0.5, item, cfg, now=now)
        expected = 0.6 * 0.5 + 0.25 * 0.8 + 0.15 * math.exp(0.0)
        assert score == pytest.approx(expected)

    def test_composite_aged_item(self):
        cfg = CortexConfig(project_root="/tmp")
        now = datetime(2026, 9, 1, tzinfo=timezone.utc)
        item = MemoryItem(created_at="2026-08-22T00:00:00+0000", importance=1.0)
        score = composite_score(0.0, item, cfg, now=now)  # 整 10 天
        expected = 0.25 * 1.0 + 0.15 * math.exp(-0.05 * 10)
        assert score == pytest.approx(expected, rel=1e-6)

    def test_cosine_basics(self):
        assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert cosine([], []) == 0.0
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# --------------------------------------------------------------------- #
# Consolidation
# --------------------------------------------------------------------- #
def make_consolidator(tmp_path, *, weak_llm=None, working=None, embedder=None):
    store = MemoryStore(Database(":memory:"))
    cfg = CortexConfig(project_root=tmp_path)
    return Consolidator(store, cfg, weak_llm=weak_llm, working=working, embedder=embedder), store


class TestConsolidation:
    def test_working_to_episodic_promotion(self, tmp_path):
        clock = {"t": 100.0}
        wm = WorkingMemory(time_fn=lambda: clock["t"])
        wm.set("plan", "研究计划草稿", ttl_seconds=100, importance=0.9)
        wm.set("scratch", "随手记号", ttl_seconds=100, importance=0.3)
        cons, store = make_consolidator(tmp_path, working=wm)
        report = cons.run()
        assert len(report["working_to_episodic"]) == 1
        episodic = store.iterate(type="episodic")
        assert len(episodic) == 1
        assert episodic[0].content == "研究计划草稿"
        assert episodic[0].importance == pytest.approx(0.9)
        assert episodic[0].source == "working_consolidation"
        # 已消费条目从工作记忆移除；低重要度保留
        assert wm.get("plan") is None
        assert wm.get("scratch") == "随手记号"

    def test_episodic_upgrade_rules_with_archive_halfing(self, tmp_path):
        cons, store = make_consolidator(tmp_path)
        yes = store.add(MemoryItem(type="episodic", content="达标记忆", importance=0.9, access_count=2))
        low_imp = store.add(
            MemoryItem(type="episodic", content="重要性不足", importance=0.79, access_count=5)
        )
        low_access = store.add(
            MemoryItem(type="episodic", content="访问不足", importance=0.85, access_count=1)
        )
        report = cons.run()
        assert report["errors"] == []
        upgraded = store.iterate(type="semantic")
        assert len(upgraded) == 1
        new = upgraded[0]
        assert new.content == "达标记忆"
        assert new.importance == pytest.approx(min(1.0, 0.9 * 1.1))
        assert new.metadata["upgraded_from"] == yes.id
        old = store.get(yes.id)
        assert old.status == "archived"
        assert old.importance == pytest.approx(0.45)  # ×0.5 软处理
        # 未达标两条原样 active
        assert store.get(low_imp.id).status == "active"
        assert store.get(low_access.id).status == "active"

    def test_duplicate_merge_with_weak_llm(self, tmp_path):
        emb = HashEmbedder(dim=64)
        llm = FakeLLM(content="向量数据库本地模式的一句话摘要")
        cons, store = make_consolidator(tmp_path, weak_llm=llm, embedder=emb)
        dup_text = "Qdrant 向量库本地模式零依赖"
        d1 = store.add(MemoryItem(type="semantic", content=dup_text, importance=0.7), emb.embed_one(dup_text))
        d2 = store.add(MemoryItem(type="semantic", content=dup_text), emb.embed_one(dup_text))
        other = store.add(
            MemoryItem(type="semantic", content="番茄炒蛋家常做法"), emb.embed_one("番茄炒蛋家常做法")
        )
        report = cons.run()
        assert len(report["merged_duplicates"]) == 1
        merged_ids = report["merged_duplicates"][0]["archived"]
        assert set(merged_ids) == {d1.id, d2.id}
        assert store.get(d1.id).status == "archived"
        assert store.get(d2.id).status == "archived"
        assert store.get(other.id).status == "active"
        merged = [i for i in store.iterate(type="semantic") if i.source == "consolidation_merge"]
        assert len(merged) == 1
        assert "摘要" in merged[0].content
        assert set(merged[0].metadata["merged_from"]) == {d1.id, d2.id}

    def test_duplicate_merge_skipped_without_weak_llm(self, tmp_path):
        emb = HashEmbedder(dim=64)
        cons, store = make_consolidator(tmp_path, weak_llm=None, embedder=emb)
        dup_text = "完全相同的重复语义内容"
        d1 = store.add(MemoryItem(type="semantic", content=dup_text), emb.embed_one(dup_text))
        d2 = store.add(MemoryItem(type="semantic", content=dup_text), emb.embed_one(dup_text))
        report = cons.run()
        assert report["merged_duplicates"] == []
        assert store.get(d1.id).status == "active"
        assert store.get(d2.id).status == "active"


# --------------------------------------------------------------------- #
# ForgettingPolicy
# --------------------------------------------------------------------- #
def make_policy(tmp_path, capacity=5000):
    store = MemoryStore(Database(":memory:"))
    cfg = CortexConfig(project_root=tmp_path, memory_capacity=capacity)
    return ForgettingPolicy(store, cfg), store


class TestForgetting:
    def test_importance_strategy(self, tmp_path):
        policy, store = make_policy(tmp_path)
        low = store.add(MemoryItem(importance=0.15))  # < 0.2
        high = store.add(MemoryItem(importance=0.6))
        report = policy.run()
        assert report["forgotten_low_importance"] == [low.id]
        assert store.get(low.id).status == "forgotten"
        assert store.get(high.id).status == "active"

    def test_time_strategy_thresholds(self, tmp_path):
        policy, store = make_policy(tmp_path)
        old_forgotten = store.add(MemoryItem(importance=0.45, created_at=old_iso(40)))  # >30d 且 <0.5
        old_kept = store.add(MemoryItem(importance=0.6, created_at=old_iso(40)))  # 重要所以留
        fresh = store.add(MemoryItem(importance=0.1, created_at=now_iso()))  # 新但低分走策略1
        report = policy.run()
        assert set(report["forgotten_old"]) == {old_forgotten.id}
        assert store.get(old_kept.id).status == "active"
        assert store.get(fresh.id).status == "forgotten"

    def test_capacity_archives_lowest_composite(self, tmp_path):
        policy, store = make_policy(tmp_path, capacity=3)
        imps = (0.9, 0.8, 0.6, 0.35, 0.25)  # 全部新鲜且 ≥0.2：只触发容量策略
        items = [store.add(MemoryItem(importance=i)) for i in imps]
        report = policy.run()
        assert len(report["archived_for_capacity"]) == 2
        archived = {i.id for i in items if store.get(i.id).status == "archived"}
        expect = {items[-1].id, items[-2].id}  # composite 最低的两条
        assert archived == expect
        assert report["active_after"] == 3

    def test_never_hard_delete(self, tmp_path):
        policy, store = make_policy(tmp_path, capacity=1)
        for i in range(5):
            store.add(MemoryItem(importance=0.1 + i * 0.1))
        total_rows = store.db.query_one("SELECT COUNT(*) AS n FROM memories")["n"]
        policy.run()
        after_rows = store.db.query_one("SELECT COUNT(*) AS n FROM memories")["n"]
        assert total_rows == after_rows == 5  # 只改 status，绝不 DELETE


# --------------------------------------------------------------------- #
# MemoryWriteQueue
# --------------------------------------------------------------------- #
class TestQueue:
    def test_flush_preserves_order(self):
        seen: list[str] = []
        q = MemoryWriteQueue(sink=seen.append)
        for name in ("a", "b", "c"):
            q.put(name)
        assert q.flush(timeout=5) is True
        assert seen == ["a", "b", "c"]
        assert q.stats == {"enqueued": 3, "processed": 3, "dropped": 0, "errors": 0}

    def test_passthrough_after_close(self):
        seen: list[str] = []
        q = MemoryWriteQueue(sink=seen.append)
        q.put("a")
        assert q.flush(timeout=5)
        assert q.close(timeout=5) is True
        q.put("b")  # 关闭后降级为同步直写
        assert seen == ["a", "b"]

    def test_drop_oldest_counts_when_full(self):
        started, release = threading.Event(), threading.Event()
        seen: list[str] = []

        def slow_sink(item: str) -> None:
            started.set()
            release.wait(5)
            seen.append(item)

        q = MemoryWriteQueue(sink=slow_sink, maxsize=2, drop_oldest=True)
        try:
            q.put("a")
            assert started.wait(5), "工作线程应已进入 sink"
            q.put("b")
            q.put("c")
            q.put("d")  # 队列满 → 丢最旧的排队者 "b"
            release.set()
            assert q.flush(timeout=5) is True
            stats = q.stats
            assert stats["dropped"] == 1
            assert seen == ["a", "c", "d"]  # b 被丢弃，顺序仍 FIFO
        finally:
            release.set()
            q.close(timeout=5)

    def test_sink_exception_does_not_kill_queue(self):
        seen: list[str] = []

        def flaky(item: str) -> None:
            if item == "bad":
                raise ValueError("sink 炸了")
            seen.append(item)

        q = MemoryWriteQueue(sink=flaky)
        q.put("bad")
        q.put("good")
        assert q.flush(timeout=5) is True
        assert seen == ["good"]
        assert q.stats["errors"] == 1
        q.close(timeout=5)


# --------------------------------------------------------------------- #
# MemoryService 闭环
# --------------------------------------------------------------------- #
class TestServiceLoop:
    def test_remember_recall_closed_loop(self, tmp_path):
        svc = make_service(tmp_path)
        try:
            # HashEmbedder 以空格分词：词级重叠才能产生正相似度
            i1 = svc.remember("量子纠缠 贝尔不等式 实验 记录", importance=0.9, source="unit")
            svc.remember("番茄 炒蛋 家常 做法", importance=0.4)
            hits = svc.recall("贝尔不等式 实验", top_k=1)
            assert len(hits) == 1
            assert "贝尔不等式" in hits[0]["content"]
            assert hits[0]["id"] == i1.id
            assert hits[0]["type"] == "episodic"
            expected = 0.6 * hits[0]["similarity"] + 0.25 * 0.9 + 0.15
            assert hits[0]["score"] == pytest.approx(expected, rel=1e-3)
            # 命中即 touch
            assert svc.store.get(i1.id).access_count == 1
        finally:
            svc.close()

    def test_recall_respects_type_filter(self, tmp_path):
        svc = make_service(tmp_path)
        try:
            svc.remember("知识图谱 图数据库 笔记", type="semantic", importance=0.8)
            svc.remember("临时 情景 记录 内容", type="episodic", importance=0.8)
            hits = svc.recall("知识图谱 数据库", top_k=5, type="semantic")
            assert all(h["type"] == "semantic" for h in hits)
            assert any("知识图谱" in h["content"] for h in hits)
        finally:
            svc.close()

    def test_async_write_then_flush(self, tmp_path):
        svc = make_service(tmp_path, async_writes=True)
        try:
            item = svc.remember("区块链共识机制学习笔记", importance=0.8)
            assert svc.store.get(item.id) is None  # 尚未落库
            assert svc.flush_writes(timeout=5) is True
            assert svc.store.get(item.id) is not None
            assert svc.recall("共识机制", top_k=1)
            assert svc.stats()["queue"]["enqueued"] == 1
        finally:
            svc.close()

    def test_weak_llm_importance_scoring_and_fallback(self, tmp_path):
        good = make_service(tmp_path, weak_llm=FakeLLM("0.85"))
        try:
            item = good.remember("被打了 0.85 分的记忆")
            assert item.importance == pytest.approx(0.85)
        finally:
            good.close()

        bad = make_service(tmp_path, weak_llm=BrokenLLM())
        try:
            item = bad.remember("打分失败应回退 0.5")
            assert item.importance == pytest.approx(0.5)
        finally:
            bad.close()

        none_llm = make_service(tmp_path, weak_llm=None)
        try:
            assert none_llm.remember("无模型默认分").importance == pytest.approx(0.5)
        finally:
            none_llm.close()

    def test_procedural_rejected_when_disabled(self, tmp_path):
        off = make_service(tmp_path, procedural_memory_enabled=False)
        try:
            with pytest.raises(ValueError, match="procedural"):
                off.remember("技能步骤", type="procedural")
        finally:
            off.close()
        on = make_service(tmp_path, procedural_memory_enabled=True)
        try:
            item = on.remember("技能步骤", type="PROCEDURAL")  # 大小写不敏感
            assert item.type == "procedural"
        finally:
            on.close()

    def test_input_validation_raises_value_error(self, tmp_path):
        svc = make_service(tmp_path)
        try:
            with pytest.raises(ValueError):
                svc.remember("   ")  # 空 内容
            with pytest.raises(ValueError):
                svc.remember("x", type="galaxy")  # 未知类型
            with pytest.raises(ValueError):
                svc.remember("x", metadata=["不是dict"])  # metadata 类型错误
            with pytest.raises(ValueError):
                svc.recall("")  # 空查询
        finally:
            svc.close()

    def test_recall_for_context_shape(self, tmp_path):
        svc = make_service(tmp_path)
        try:
            svc.remember("高价值结论 A", importance=0.9)
            svc.remember("中价值观察 B", importance=0.5)
            svc.remember("噪声碎片 C", importance=0.1)
            ctx = svc.recall_for_context(limit=10, min_importance=0.3)
            assert len(ctx) == 2
            assert {"content", "type", "importance", "created_at"} <= set(ctx[0])
            assert ctx[0]["importance"] >= 0.3
            assert all("噪声" not in c["content"] for c in ctx)
        finally:
            svc.close()

    def test_summary_without_llm_joins_text(self, tmp_path):
        svc = make_service(tmp_path)
        try:
            svc.remember("第一条记忆内容甲", importance=0.8)
            svc.remember("第二条记忆内容乙", importance=0.6)
            text = svc.summary()
            assert "(episodic)" in text
            assert "第一条记忆内容甲" in text and "第二条记忆内容乙" in text
        finally:
            svc.close()

    def test_summary_with_weak_llm(self, tmp_path):
        svc = make_service(tmp_path, weak_llm=FakeLLM("这是合并后的记忆摘要。"))
        try:
            svc.remember("需要被总结的记忆", importance=0.8)
            assert svc.summary() == "这是合并后的记忆摘要。"
        finally:
            svc.close()

    def test_forget_and_consolidate_delegate(self, tmp_path):
        svc = make_service(tmp_path)
        try:
            svc.working_memory.set("k", "高价值草稿", ttl_seconds=60, importance=0.9)
            rep_c = svc.consolidate()
            assert len(rep_c["working_to_episodic"]) == 1
            svc.store.update_importance(rep_c["working_to_episodic"][0]["id"], 0.1)
            rep_f = svc.forget()
            assert rep_f["forgotten_low_importance"]
            assert rep_f["active_after"] == 0
        finally:
            svc.close()


# --------------------------------------------------------------------- #
# MemoryTool
# --------------------------------------------------------------------- #
@pytest.fixture
def tool_service(tmp_path):
    return make_service(tmp_path)


def _status(resp):
    return resp.status.value if hasattr(resp.status, "value") else resp.status


def ok(resp):
    assert _status(resp) == "success", f"期望成功，实际 {resp.error_info}"
    return resp


def err_code(resp, code: str):
    assert _status(resp) != "success", f"期望失败，实际 success: {resp.text}"
    info = resp.error_info
    got = info.get("code") if isinstance(info, dict) else getattr(info, "code", None)
    assert got == code, f"期望 {code}，实际 {got}"
    return resp


class TestMemoryTool:
    def test_expanded_names_exact_set(self, tool_service):
        from cortex.tools.memory_tool import MemoryTool

        names = {t.name for t in MemoryTool(tool_service).get_expanded_tools()}
        assert names == {
            "memory_add",
            "memory_search",
            "memory_summary",
            "memory_forget",
            "memory_consolidate",
        }

    def test_register_expands_into_registry(self, tool_service):
        from cortex.logging_setup import quiet
        from cortex.tools.memory_tool import MemoryTool

        reg = ToolRegistry()
        with quiet():
            reg.register_tool(MemoryTool(tool_service))
        names = set(reg.list_tools())
        assert {
            "memory_add",
            "memory_search",
            "memory_summary",
            "memory_forget",
            "memory_consolidate",
        } <= names
        assert "memory" not in names

    def test_add_then_search_format(self, tool_service):
        from cortex.tools.memory_tool import MemoryTool

        tool = MemoryTool(tool_service)
        r = ok(tool.add(content="量子纠缠 贝尔不等式 实验 笔记", type="EPISODIC", importance=0.9))
        assert r.data["type"] == "episodic"
        ok(tool.add(content="番茄 炒蛋 做法 步骤", importance=0.4))
        s = ok(tool.search(query="贝尔不等式 量子", top_k=2))
        assert s.data["hits"], "应有命中"
        first_line = s.text.splitlines()[0]
        assert first_line.startswith("- [")
        assert "(episodic)" in first_line

    def test_search_content_truncated_to_120_chars(self, tool_service):
        from cortex.tools.memory_tool import MemoryTool

        tool = MemoryTool(tool_service)
        long_text = "重复 长文本 " * 80
        ok(tool.add(content=long_text, importance=0.9))
        s = ok(tool.search(query="重复 长文本", top_k=1))
        line = s.text.splitlines()[0]
        body = line.split(") ", 1)[1]
        assert len(body) <= 120

    def test_procedural_rejected_invalid_param(self, tmp_path):
        from cortex.tools.memory_tool import MemoryTool

        svc = make_service(tmp_path, procedural_memory_enabled=False)
        try:
            resp = err_code(MemoryTool(svc).add(content="技能", type="Procedural"), "INVALID_PARAM")
            assert "procedural" in resp.error_info["message"].lower()
        finally:
            svc.close()

    def test_procedural_allowed_when_cfg_enabled(self, tmp_path):
        from cortex.tools.memory_tool import MemoryTool

        svc = make_service(tmp_path, procedural_memory_enabled=True)
        try:
            # 工具开关跟随 cfg（None）→ 放行
            r = ok(MemoryTool(svc).add(content="调试流程技能", type="procedural", importance=0.7))
            assert r.data["type"] == "procedural"
            # 工具显式收紧 → 即便 cfg 开启也在工具层拒绝
            strict = MemoryTool(svc, procedural_enabled=False)
            err_code(strict.add(content="另一技能", type="procedural"), "INVALID_PARAM")
        finally:
            svc.close()

    def test_empty_content_invalid_param(self, tool_service):
        from cortex.tools.memory_tool import MemoryTool

        err_code(MemoryTool(tool_service).add(content="   "), "INVALID_PARAM")

    def test_bad_type_invalid_param(self, tool_service):
        from cortex.tools.memory_tool import MemoryTool

        err_code(MemoryTool(tool_service).add(content="x", type="galaxy"), "INVALID_PARAM")

    def test_empty_query_invalid_param(self, tool_service):
        from cortex.tools.memory_tool import MemoryTool

        err_code(MemoryTool(tool_service).search(query="  "), "INVALID_PARAM")

    def test_summary_forget_consolidate_responses(self, tmp_path):
        from cortex.tools.memory_tool import MemoryTool

        svc = make_service(tmp_path)
        try:
            tool = MemoryTool(svc)
            ok(tool.add(content="需要摘要的记忆条目", importance=0.8))
            s = ok(tool.summary())
            assert "需要摘要的记忆条目" in s.data["summary"]
            f = ok(tool.forget())
            assert "active_after" in f.data
            c = ok(tool.consolidate())
            assert "working_to_episodic" in c.data
        finally:
            svc.close()

    def test_tool_never_raises_on_bad_input(self, tool_service):
        from cortex.tools.memory_tool import MemoryTool

        tool = MemoryTool(tool_service)
        for call in (
            lambda: tool.add(content="", type="nope"),
            lambda: tool.search(query=""),
            lambda: tool.summary(),
            lambda: tool.forget(),
            lambda: tool.consolidate(),
        ):
            resp = call()
            assert hasattr(resp, "status")
