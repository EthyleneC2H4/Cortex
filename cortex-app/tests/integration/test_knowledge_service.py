"""P3 集成测试：真实本地存储全链路（qdrant 本地模式 + sqlite 文件 + bm25 jsonl）。

无 LLM、无 docker、离线可跑（HashEmbedder）。live 语义验证在 e2e。
"""

from __future__ import annotations

import pytest

from cortex.services.db import Database
from cortex.services.knowledge.bm25_index import BM25Index
from cortex.services.knowledge.embeddings import HashEmbedder
from cortex.services.knowledge.graph import LocalGraphStore
from cortex.services.knowledge.service import KnowledgeService

pytestmark = pytest.mark.integration


@pytest.fixture
def service(tmp_path):
    svc = KnowledgeService.__new__(KnowledgeService)
    from cortex.config import CortexConfig

    cfg = CortexConfig(project_root=tmp_path, embed_provider="hash")
    cfg.ensure_dirs()
    cfg.retrieval_min_score = 0.05  # 词袋哈希向量分标度远低于真实语义嵌入
    svc.cfg = cfg
    svc.db = Database(cfg.data_dir / "cortex.db")          # 真实 sqlite 文件
    svc.embedder = HashEmbedder(dim=128)
    svc._qdrant_client = None                               # 自建 client（惰性）
    svc._vector = None                                      # 惰性 → 真实 QdrantLocalStore
    svc.bm25 = BM25Index(path=cfg.data_dir / "bm25" / "index.jsonl")
    svc.graph = LocalGraphStore(persist_path=cfg.data_dir / "graph.json")
    return svc


DOC_A = (
    "# 混合检索\nRRF 倒数排名融合把向量与关键词两路结果合并，k 常取 60。"
    "融合后按分数排序截取 top-k。\n\n## 三带拒答\n低置信度时拒答以抑制幻觉。"
)
DOC_B = (
    "# 记忆系统\n记忆巩固将工作记忆提升为情景记忆，重要性阈值 0.7。"
    "遗忘采用软删除策略。"
)


def test_full_lifecycle(service, tmp_path):
    a = service.ingest_text(title="检索文档", text=DOC_A, weak_llm=None)
    b = service.ingest_text(title="记忆文档", text=DOC_B, weak_llm=None)
    assert a["doc_id"] != b["doc_id"]
    assert a["n_chunks"] >= 1

    stats = service.stats()
    assert stats["documents"] == 2 and stats["chunks"] == a["n_chunks"] + b["n_chunks"]
    assert stats["vectors"] == stats["chunks"]
    assert stats["bm25"] == stats["chunks"]

    # 检索命中正确文档
    r = service.search("RRF 融合", top_k=3)
    assert r.hits and all(h.doc_id == a["doc_id"] for h in r.hits[:1])

    # ContextEngine 接口：证据包结构
    packets = service.search_for_context("RRF 融合", limit=3)
    assert packets and {"text", "source", "chunk_id", "score"} <= set(packets[0])

    # 幂等重摄取：同标题同 doc_id，块数不翻倍
    a2 = service.ingest_text(title="检索文档", text=DOC_A + "\n补充段落。")
    assert a2["doc_id"] == a["doc_id"]
    assert service.stats()["documents"] == 2

    # 删除回收四路存储
    assert service.delete(a["doc_id"]) is True
    after = service.stats()
    assert after["documents"] == 1 and after["bm25"] == b["n_chunks"]

    # 服务重启后持久化仍在（sqlite + bm25 jsonl）；先释放 qdrant 本地文件锁
    service.close()
    cfg = service.cfg
    svc2 = KnowledgeService.__new__(KnowledgeService)
    svc2.cfg = cfg
    svc2.db = Database(cfg.data_dir / "cortex.db")
    svc2.embedder = HashEmbedder(dim=128)
    svc2._qdrant_client = None
    svc2._vector = None
    svc2.bm25 = BM25Index(path=cfg.data_dir / "bm25" / "index.jsonl")
    svc2.graph = LocalGraphStore(persist_path=cfg.data_dir / "graph.json")
    assert svc2.stats()["documents"] == 1
    assert svc2.search("软删除").hits


def test_qa_refuse_on_empty_kb(service):
    class ShouldNotRun:
        def invoke(self, m):  # pragma: no cover
            raise AssertionError("空库拒答不应调用 LLM")

    ans = service.ask("任何问题", ShouldNotRun())
    assert ans.band == "refuse"
    assert not ans.citations


def test_graph_triples_persisted_via_ingest(tmp_path):
    """ingest 带 weak_llm 时三元组应落图（mock 弱模型）。"""
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
    svc.graph = LocalGraphStore(persist_path=tmp_path / "g.json")

    class Weak:
        def invoke(self, messages):
            return type("R", (), {"content": (
                '[{"subject":"RRF","relation":"属于","object":"融合算法","confidence":0.9}]'
            )})()

    info = svc.ingest_text(title="t", text="# T\n正文内容 RRf 相关。" , weak_llm=Weak())
    assert info["n_chunks"] >= 1
    assert svc.graph.stats()["edges"] == 1
