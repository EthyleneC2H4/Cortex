"""P5 live：完整研究流水线（HAS_LLM_KEY=1 才跑）。

验收（规划文档 P5 门禁）：报告为结构化中文 Markdown，含 ≥2 条
[doc_id/chunk_id] 引用标记。需先向 tmp 知识库摄取样例文档。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e]

requires_key = pytest.mark.skipif(
    not os.getenv("HAS_LLM_KEY"), reason="需要 HAS_LLM_KEY=1 且 .env 配好 LLM_API_KEY"
)


@requires_key
def test_pipeline_produces_cited_report(tmp_path):
    root = Path(__file__).resolve().parents[2]
    from cortex.config import load_config

    cfg = load_config(env_file=root / ".env")
    cfg.rebase(tmp_path)
    cfg.ensure_dirs()

    from cortex.agents.builders import build_agents
    from cortex.agents.dispatch import ResearchPipeline
    from cortex.context.engine import ContextEngine
    from cortex.llm.factory import build_llms
    from cortex.services.knowledge.service import KnowledgeService

    strong, weak = build_llms(cfg)
    kb = KnowledgeService(cfg)
    sample = root / "knowledge_base" / "samples" / "sample-hybrid-retrieval.md"
    kb.ingest_document(sample, weak_llm=weak)

    agents = build_agents(cfg, strong, weak, registry=None)
    engine = ContextEngine(cfg, memory=None, knowledge=kb, weak_llm=weak)
    pipe = ResearchPipeline(agents, knowledge=kb, context_engine=engine,
                            traces_dir=cfg.data_dir / "traces")
    try:
        result = pipe.run("混合检索中 RRF 融合的作用")
        assert result.report_md.strip()
        citations = re.findall(r"\[[0-9a-f]{8,}:\d+\]", result.report_md) or \
            re.findall(r"\[[^\]\s]+/[^\]\s]+\]", result.report_md)
        assert len(citations) >= 2, f"引用标记不足: {result.report_md[:400]}"
        trace_files = list((cfg.data_dir / "traces").glob("*.jsonl"))
        assert trace_files, "流水线 trace 应落盘"
    finally:
        kb.close()
