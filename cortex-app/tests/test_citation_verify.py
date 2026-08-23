"""P5 单元测试：引用标记宽容归一 + citation_verify 容错查找。

背景（run6 live 复盘）：writer 会把 [doc_id/chunk_id] 写成 'doc/3' 甚至 '/3'，
规范库里的 chunk_id 是 'doc:3'，精确匹配全部 NOT_FOUND —— 归一是硬需求。
"""

from __future__ import annotations

import pytest

from cortex.services.db import Database
from cortex.services.knowledge.bm25_index import BM25Index
from cortex.services.knowledge.embeddings import HashEmbedder
from cortex.services.knowledge.graph import LocalGraphStore
from cortex.services.knowledge.qa import normalize_chunk_id, retrieve_citation_ids
from cortex.services.knowledge.service import KnowledgeService
from cortex.tools.citation_tool import CitationTool
from cortex.tools.knowledge_tool import KnowledgeSearchTool

pytestmark = pytest.mark.unit


class TestNormalizeChunkId:
    def test_canonical_passthrough(self):
        assert normalize_chunk_id("abc12345:3") == "abc12345:3"

    def test_qa_marker_form(self):
        # qa.py 标准标记 [doc/doc:idx]
        assert normalize_chunk_id("abc12345/abc12345:3") == "abc12345:3"

    def test_writer_shorthand(self):
        assert normalize_chunk_id("abc12345/3") == "abc12345:3"

    def test_missing_doc_prefix(self):
        assert normalize_chunk_id("/3") == ":3"
        assert normalize_chunk_id(":3") == ":3"

    def test_brackets_and_whitespace(self):
        assert normalize_chunk_id(" [abc12345/3] ") == "abc12345:3"

    def test_unparseable(self):
        assert normalize_chunk_id("") is None
        assert normalize_chunk_id(None) is None


class TestRetrieveCitationIds:
    def test_mixed_forms_all_resolved(self):
        text = ("结论A [abc12345/abc12345:0]；结论B [abc12345/2]；"
                "重复引用不重复收集 [abc12345/abc12345:0]。")
        assert retrieve_citation_ids(text) == ["abc12345:0", "abc12345:2"]


# --------------------------------------------------------------------- #
def _svc_with_doc(tmp_path) -> tuple[KnowledgeService, str]:
    """离线知识库 + 一篇已入库文档；返回 (service, 规范 chunk_id)。"""
    from cortex.config import CortexConfig

    svc = KnowledgeService.__new__(KnowledgeService)
    cfg = CortexConfig(project_root=tmp_path, embed_provider="hash")
    cfg.ensure_dirs()
    svc.cfg = cfg
    svc.db = Database(":memory:")
    svc.embedder = HashEmbedder(dim=128)
    svc._qdrant_client = None
    svc._vector = None
    svc.bm25 = BM25Index(path=None)
    svc.graph = LocalGraphStore(persist_path=None)

    body = ("RRF 倒数排名融合把每路检索结果的排名倒数相加，k 常取 60，"
            "无需调参、对分数分布不敏感，是混合检索的默认融合方案。")
    ok = KnowledgeSearchTool(svc).add_text(title="混合检索", content=body)
    assert ok.data["n_chunks"] >= 1
    return svc, svc.db.query("SELECT chunk_id FROM chunks ORDER BY chunk_id")[0]["chunk_id"]


def _status(resp):
    return resp.status.value if hasattr(resp.status, "value") else resp.status


class TestCitationVerifyTolerance:
    def test_canonical_and_shorthand_resolve(self, tmp_path):
        svc, cid = _svc_with_doc(tmp_path)
        doc, idx = cid.split(":")
        ct = CitationTool(svc, weak_llm=None)
        claim = svc.db.query_one("SELECT text FROM chunks WHERE chunk_id=?", (cid,))["text"][:12]

        for variant in (cid, f"{doc}/{doc}:{idx}", f"{doc}/{idx}", f"[{doc}/{idx}]"):
            resp = ct.verify(claim=claim, chunk_id=variant)
            assert _status(resp) == "success", f"{variant!r} 应可核验: {resp.error_info}"
            assert resp.data["chunk_id"] == cid          # 归一后回填规范 ID

    def test_bare_index_unique_suffix(self, tmp_path):
        svc, cid = _svc_with_doc(tmp_path)
        _, idx = cid.split(":")
        resp = CitationTool(svc, weak_llm=None).verify(claim="RRF 融合", chunk_id=f"/{idx}")
        assert _status(resp) == "success"                 # 单文档库唯一命中

    def test_missing_chunk_reports_not_found(self, tmp_path):
        svc, _ = _svc_with_doc(tmp_path)
        resp = CitationTool(svc, weak_llm=None).verify(claim="x", chunk_id="zzzz9999:42")
        assert _status(resp) != "success"

    def test_ambiguous_bare_index_refused(self, tmp_path):
        svc, _ = _svc_with_doc(tmp_path)
        ok = KnowledgeSearchTool(svc).add_text(title="第二篇", content="另一篇文档的内容，主题不同。")
        assert ok.data["n_chunks"] >= 1
        resp = CitationTool(svc, weak_llm=None).verify(claim="x", chunk_id="/0")
        assert _status(resp) != "success"                 # 两篇都有 :0 → 拒绝猜测
