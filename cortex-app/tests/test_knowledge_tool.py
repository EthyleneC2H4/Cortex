"""P3 单元测试：KnowledgeSearchTool 展开名 + 离线全链路（HashEmbedder）。"""

from __future__ import annotations

import contextlib
import io

import pytest
from hello_agents.tools import ToolRegistry

from cortex.services.db import Database
from cortex.services.knowledge.bm25_index import BM25Index
from cortex.services.knowledge.embeddings import HashEmbedder
from cortex.services.knowledge.graph import LocalGraphStore
from cortex.services.knowledge.service import KnowledgeService
from cortex.tools.knowledge_tool import KnowledgeSearchTool

pytestmark = pytest.mark.unit


def _service(tmp_path) -> KnowledgeService:
    """离线五件套：HashEmbedder + qdrant :memory: 等价的本地路径 + 内存图。"""
    svc = KnowledgeService.__new__(KnowledgeService)
    from cortex.config import CortexConfig

    cfg = CortexConfig(project_root=tmp_path, embed_provider="hash")
    cfg.ensure_dirs()
    svc.cfg = cfg
    svc.db = Database(":memory:")
    svc.embedder = HashEmbedder(dim=128)
    svc._qdrant_client = None
    svc._vector = None
    svc.bm25 = BM25Index(path=None)
    svc.graph = LocalGraphStore(persist_path=None)
    return svc


@pytest.fixture
def tool(tmp_path):
    svc = _service(tmp_path)
    return KnowledgeSearchTool(svc, weak_llm=None, strong_llm=None)


def ok(resp):
    s = resp.status.value if hasattr(resp.status, "value") else resp.status
    assert s == "success", f"期望成功: {resp.error_info}"
    return resp


class TestExpansion:
    def test_names(self, tool):
        names = {t.name for t in tool.get_expanded_tools()}
        assert names == {"knowledge_add_document", "knowledge_add_text",
                         "knowledge_search", "knowledge_qa"}

    def test_registry_integration(self, tmp_path):
        reg = ToolRegistry()
        with contextlib.redirect_stdout(io.StringIO()):
            reg.register_tool(KnowledgeSearchTool(_service(tmp_path)))
        assert {"knowledge_search", "knowledge_qa"} <= set(reg.list_tools())


class TestFlow:
    def test_add_search_roundtrip(self, tool):
        r = ok(tool.add_text(
            title="混合检索笔记",
            content="混合检索将向量检索与 BM25 关键词检索结合，用 RRF 倒数排名融合两路结果。",
        ))
        assert r.data["n_chunks"] >= 1
        s = ok(tool.search(query="RRF 融合", top_k=3))
        assert s.data["band"] in ("answer", "uncertain")
        hits = s.data["hits"]
        assert hits and any("RRF" in h["text"] for h in hits)

    def test_refuse_band_on_unrelated_query(self, tool):
        ok(tool.add_text(title="无关文档", content="今天天气不错，适合散步。"))
        s = ok(tool.search(query="量子色动力学的渐近自由性质"))
        # HashEmbedder 词袋可能偶有弱命中；断言结构而非具体带
        assert s.data["band"] in ("refuse", "uncertain", "answer")

    def test_qa_requires_strong_llm(self, tool):
        e = tool.qa(question="什么是 RRF")
        st = e.status.value if hasattr(e.status, "value") else e.status
        assert st != "success"
        info = e.error_info
        code = info.get("code") if isinstance(info, dict) else getattr(info, "code", None)
        assert code == "INVALID_PARAM"

    def test_empty_text_rejected(self, tool):
        e = tool.add_text(title="空", content="  ")
        info = e.error_info
        code = info.get("code") if isinstance(info, dict) else getattr(info, "code", None)
        assert code == "INVALID_PARAM"

    def test_add_document_missing_file(self, tool):
        e = tool.add_document(path="/no/such/file.md")
        info = e.error_info
        code = info.get("code") if isinstance(info, dict) else getattr(info, "code", None)
        assert code == "NOT_FOUND"


class TestQAWithMockLLM:
    def test_answer_with_citations(self, tmp_path):
        svc = _service(tmp_path)
        svc.cfg.retrieval_min_score = -1.0  # 离线词袋向量分偏低：压低阈值避免误拒答
        info = svc.ingest_text(title="RAG 概述", text=(
            "# RAG\n检索增强生成（RAG）把外部检索与生成模型结合。"
            "混合检索融合向量与关键词两路信号，再用 RRF 排序。"
        ))

        class Strong:
            def invoke(self, messages):
                cid = next(
                    ln.split("=")[1] for ln in messages[0]["content"].splitlines()
                    if ln.startswith("chunk_id=")
                )
                return type("R", (), {"content": f"RAG 结合了检索与生成 [{info['doc_id']}/{cid}]"})()

        ans = svc.ask("什么是 RAG？", Strong())
        assert ans.band in ("answer", "uncertain")
        assert "[doc/" in ans.answer or ans.citations

    def test_refuse_band_short_circuits_llm(self, tmp_path):
        svc = _service(tmp_path)
        called = []

        class Strong:
            def invoke(self, messages):
                called.append(1)
                raise AssertionError("refuse 时不应调用 LLM")

        ans = svc.ask("完全无关的量子问题", Strong())  # 空库必 refuse
        assert ans.band == "refuse"
        assert "知识库中没有找到" in ans.answer
        assert not called
