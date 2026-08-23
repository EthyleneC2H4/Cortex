"""P3 live：摄取→带引用问答全链路（HAS_LLM_KEY=1 才跑）。

用独立 tmp 数据目录（rebase），不触碰仓库 data/（qdrant 排他锁单进程）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e]

requires_key = pytest.mark.skipif(
    not os.getenv("HAS_LLM_KEY"), reason="需要 HAS_LLM_KEY=1 且 .env 配好 LLM_API_KEY"

)

SAMPLE = """# 向量数据库选型指南

Qdrant 是一款开源向量数据库，支持本地文件模式与 docker 服务模式。
本地模式零部署，适合个人知识库；服务模式支持并发与副本。

## 过滤检索

Qdrant 支持在向量搜索时附加 payload 过滤条件，例如按文档来源过滤，
这对多用户知识库的权限隔离非常关键。

## 量化压缩

标量量化可将内存占用降至原来的四分之一，同时保持召回率基本不变。
"""


@requires_key
def test_ingest_then_cited_qa(tmp_path):
    root = Path(__file__).resolve().parents[2]
    from cortex.config import load_config

    cfg = load_config(env_file=root / ".env")
    cfg.rebase(tmp_path)
    cfg.retrieval_min_score = 0.3
    cfg.ensure_dirs()

    from cortex.llm.factory import build_llms
    from cortex.services.knowledge.service import KnowledgeService

    strong, weak = build_llms(cfg)
    kb = KnowledgeService(cfg)
    try:
        info = kb.ingest_text(title="向量库选型", text=SAMPLE, weak_llm=weak)
        assert info["n_chunks"] >= 1

        ans = kb.ask("Qdrant 本地模式适合什么场景？", strong, weak_llm=weak)
        assert ans.band in ("answer", "uncertain")  # 空库拒答已在离线测试覆盖
        assert ans.answer.strip()
        # 引用标记或 citations 至少其一存在
        assert ans.citations or "[" in ans.answer
    finally:
        kb.close()
