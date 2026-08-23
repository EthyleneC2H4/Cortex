"""P4 live：跨会话记忆（HAS_LLM_KEY=1 才跑）。

remember 用弱模型判重要性 → 关闭服务（模拟会话结束）→ 重建服务后
recall 仍能召回该偏好，验证 sqlite+qdrant 持久化链路。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e]

requires_key = pytest.mark.skipif(
    not os.getenv("HAS_LLM_KEY"), reason="需要 HAS_LLM_KEY=1 且 .env 配好 LLM_API_KEY"
)


@requires_key
def test_cross_session_preference_recall(tmp_path):
    root = Path(__file__).resolve().parents[2]
    from cortex.config import load_config

    cfg = load_config(env_file=root / ".env")
    cfg.rebase(tmp_path)
    cfg.memory_async_writes = False  # 直写，测试无需 flush
    cfg.ensure_dirs()

    from cortex.llm.factory import build_llms
    from cortex.services.db import Database
    from cortex.services.knowledge.embeddings import make_embedder
    from cortex.services.memory.service import MemoryService

    _, weak = build_llms(cfg)

    def build():
        db = Database(cfg.data_dir / "cortex.db")
        embedder = make_embedder(cfg)
        return MemoryService(cfg, db=db, embedder=embedder, weak_llm=weak)

    svc = build()
    try:
        svc.remember("用户偏好：回答尽量简洁，先给结论再给论据", user_id="u-live")
        svc.remember("项目背景：Cortex 使用 Qdrant 做向量检索", user_id="u-live")
    finally:
        svc.close()  # 模拟会话结束，释放 qdrant 锁

    # ---- 新"会话"：全新实例 ---------------------------------------- #
    svc2 = build()
    try:
        hits = svc2.recall("用户的输出格式偏好是什么？", top_k=3)
        assert hits, "跨会话应召回偏好记忆"
        assert any("简洁" in h["content"] for h in hits)
    finally:
        svc2.close()
