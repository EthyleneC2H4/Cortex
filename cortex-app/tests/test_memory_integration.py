"""P4 集成测试：真实 sqlite 文件 + Qdrant 本地模式全链路 remember/recall/close/重启。

无 LLM、无 docker（HashEmbedder 离线嵌入）。qdrant 本地模式有排他文件锁：
每个用例独立 tmp 路径，结束前必须 close() 释放。
"""

from __future__ import annotations

import pytest

from cortex.config import CortexConfig
from cortex.services.db import Database
from cortex.services.knowledge.embeddings import HashEmbedder
from cortex.services.memory import MemoryService
from cortex.services.memory.store import QdrantMemoryIndex

pytestmark = pytest.mark.integration


def make_service(tmp_path, **cfg_overrides) -> MemoryService:
    """真实存储组合：sqlite 文件库 + qdrant 本地目录 + hash 嵌入。"""
    defaults = dict(embed_provider="hash", memory_async_writes=False)
    defaults.update(cfg_overrides)
    cfg = CortexConfig(project_root=tmp_path, **defaults)
    cfg.ensure_dirs()
    db = Database(cfg.data_dir / "cortex.db")
    index = QdrantMemoryIndex(cfg.data_dir / "qdrant_local", dim=128)
    return MemoryService(cfg, db, HashEmbedder(dim=128), vector_index=index)


def test_full_chain_remember_recall_touch(tmp_path):
    svc = make_service(tmp_path)
    try:
        # HashEmbedder 按空格分词：词级重叠才产生正相似度
        i1 = svc.remember("量子纠缠 贝尔不等式 实验 记录", importance=0.9, source="integration")
        svc.remember("番茄 炒蛋 家常 做法 步骤", importance=0.5)
        svc.remember("每周 跑步 三次 锻炼 计划", importance=0.6)

        assert svc.store.count() == 3

        hits = svc.recall("贝尔不等式 实验", top_k=2)
        assert hits, "向量检索应有命中"
        assert "贝尔不等式" in hits[0]["content"]
        assert hits[0]["id"] == i1.id
        assert hits[0]["similarity"] > 0.0

        # 命中即 touch：access_count 与 last_access 已回写 sqlite
        row = svc.store.get(i1.id)
        assert row.access_count == 1

        # ContextEngine 紧凑接口
        ctx = svc.recall_for_context(limit=5)
        assert len(ctx) == 3
        assert {"content", "type", "importance", "created_at"} <= set(ctx[0])

        # 策略空跑不误伤
        rep_c = svc.consolidate()
        assert rep_c["errors"] == []
        rep_f = svc.forget()
        assert rep_f["active_after"] == 3

        text = svc.summary()
        assert "贝尔不等式" in text
    finally:
        svc.close()  # 释放 qdrant 排他锁与 sqlite 连接


def test_persist_and_restart_after_close(tmp_path):
    svc = make_service(tmp_path)
    try:
        for text in ("量子纠缠 贝尔不等式 实验 甲", "番茄 炒蛋 做法 乙", "每周 跑步 锻炼 丙"):
            svc.remember(text, importance=0.7)
    finally:
        svc.close()

    # 同路径重建服务：数据应完整存活（sqlite 行 + qdrant 向量）
    svc2 = make_service(tmp_path)
    try:
        assert svc2.store.count() == 3
        hits = svc2.recall("量子纠缠 贝尔不等式", top_k=1)
        assert hits and "量子纠缠" in hits[0]["content"]
        ctx = svc2.recall_for_context(limit=10)
        assert len(ctx) == 3
    finally:
        svc2.close()


def test_async_write_queue_drains_on_close(tmp_path):
    svc = make_service(tmp_path, memory_async_writes=True)
    item = svc.remember("区块链 共识 机制 学习 笔记", importance=0.8)
    assert svc.store.get(item.id) is None  # 异步模式：尚未落库
    assert svc.flush_writes(timeout=5) is True
    assert svc.store.get(item.id) is not None
    assert svc.recall("共识 机制", top_k=1)
    stats = svc.stats()["queue"]
    assert stats["enqueued"] == 1 and stats["processed"] >= 1
    svc.close()


def test_qdrant_memory_index_direct(tmp_path):
    """QdrantMemoryIndex 直连契约：upsert 幂等覆盖 / search / delete / close。"""
    path = tmp_path / "qdrant_local"
    idx = QdrantMemoryIndex(path, dim=128)
    vec_a = [1.0] + [0.0] * 127
    vec_b = [0.0, 1.0] + [0.0] * 126
    idx.upsert(["m1"], [vec_a], [{"id": "m1", "type": "episodic"}])
    idx.upsert(["m1"], [vec_a], [{"id": "m1", "type": "episodic"}])  # 幂等覆盖
    idx.upsert(["m2"], [vec_b], [{"id": "m2", "type": "semantic"}])
    assert idx.count() == 2

    hits = idx.search(vec_a, top_k=2)
    assert hits[0][0] == "m1"
    assert hits[0][1] == pytest.approx(1.0)

    idx.delete(["m1"])
    hits = idx.search(vec_a, top_k=5)
    assert [mid for mid, _ in hits if mid == "m1"] == []

    # 同路径二次打开（close 之后）仍可读 —— 持久化验证
    idx.close()
    idx2 = QdrantMemoryIndex(path, dim=128)
    try:
        assert idx2.count() == 1
        assert idx2.search(vec_b, top_k=1)[0][0] == "m2"
    finally:
        idx2.close()
