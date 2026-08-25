"""BGE 重排器离线测试：假 FlagEmbedding 模块注入 + 检索编排桩，零真实模型。"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from cortex.services.knowledge.rerank import BGEReranker, make_reranker
from cortex.services.knowledge.retrieve import RetrievedChunk, retrieve

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- 假依赖 -- #
def _install_fake_flagembedding(monkeypatch, scores):
    """把假 FlagEmbedding 塞进 sys.modules；compute_score 按 precomputed 回分。"""
    mod = ModuleType("FlagEmbedding")

    class FakeFlagReranker:
        def __init__(self, model_name, use_fp16=False, max_length=512):
            self.model_name = model_name
            self.seen_pairs = []

        def compute_score(self, pairs, batch_size=16, normalize=True):
            self.seen_pairs.extend(pairs)
            out = list(scores)
            if len(out) == 1:
                return out[0]  # 单条时真库返回标量——保持同形
            return out[: len(pairs)] if len(pairs) < len(out) else out

    mod.FlagReranker = FakeFlagReranker
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)
    return mod


class TestBGEReranker:
    def test_sorts_desc_and_fills_scores(self, monkeypatch):
        _install_fake_flagembedding(monkeypatch, scores=[0.1, 0.9, 0.5])
        rr = BGEReranker("BAAI/bge-reranker-v2-m3")
        chunks = [RetrievedChunk(chunk_id=f"c{i}", text=f"文本{i}") for i in range(3)]
        out = rr.rerank("查询", chunks)
        assert [c.chunk_id for c in out] == ["c1", "c2", "c0"]
        assert out[0].rerank_score == pytest.approx(0.9)
        # 输入列表不被就地修改语义破坏（返回新序表）
        assert [c.chunk_id for c in chunks] == ["c0", "c1", "c2"]

    def test_single_chunk_scalar_score(self, monkeypatch):
        _install_fake_flagembedding(monkeypatch, scores=[0.7])
        rr = BGEReranker()
        out = rr.rerank("q", [RetrievedChunk(chunk_id="a", text="正文")])
        assert len(out) == 1 and out[0].rerank_score == pytest.approx(0.7)

    def test_empty_chunks_noop(self, monkeypatch):
        _install_fake_flagembedding(monkeypatch, scores=[])
        rr = BGEReranker()
        assert rr.rerank("q", []) == []

    def test_missing_dependency_raises_runtimeerror(self, monkeypatch):
        # sys.modules 放 None 使 import 抛 ImportError（模拟未装 [rerank] extra）
        monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
        with pytest.raises(RuntimeError, match="FlagEmbedding"):
            BGEReranker()


class TestMakeReranker:
    def test_disabled_returns_none_without_import(self, monkeypatch):
        # sys.modules 放 None：若误触发导入会直接 ImportError，测试即红
        monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
        cfg = SimpleNamespace(enable_rerank=False, rerank_model="m")
        assert make_reranker(cfg) is None

    def test_enabled_but_dependency_missing_degrades_to_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "FlagEmbedding", None)
        cfg = SimpleNamespace(enable_rerank=True, rerank_model="BAAI/bge-reranker-v2-m3")
        assert make_reranker(cfg) is None  # 软降级，不抛异常

    def test_enabled_with_dependency_builds_instance(self, monkeypatch):
        _install_fake_flagembedding(monkeypatch, scores=[0.0])
        cfg = SimpleNamespace(enable_rerank=True, rerank_model="custom/rerank")
        rr = make_reranker(cfg)
        assert isinstance(rr, BGEReranker)
        assert rr._model.model_name == "custom/rerank"


# ---------------------------------------------- 检索编排（注入点契约） ---- #
class FakeVector:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, vector, top_k=20, doc_id=None):
        return [
            type("H", (), {"chunk_id": cid, "doc_id": cid.split(":")[0], "score": s,
                           "text": f"文本{cid}", "heading_path": "", "source": ""})()
            for cid, s in self.mapping.get(tuple(vector), [])
        ][:top_k]


class FakeBM25:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=20, min_score=0.01):
        return [(cid, 1.0) for cid in self.results.get(query, [])][:top_k]


class FakeEmbedder:
    dim = 4

    def embed_one(self, text):
        return (text,)


class Deps:
    def __init__(self, vector, bm25):
        self.embedder = FakeEmbedder()
        self.vector = vector
        self.bm25 = bm25
        self.db = None


class ReverseReranker:
    """确定性桩：整串倒序 + 打标分数，验证排序被重排接管。"""

    def rerank(self, query, chunks):
        out = list(reversed(chunks))
        for i, c in enumerate(out):
            c.rerank_score = float(len(out) - i)
        return out


class ExplodingReranker:
    def rerank(self, query, chunks):
        raise RuntimeError("模型推理失败")


class TestRetrieveRerankIntegration:
    def _deps(self):
        return Deps(
            FakeVector({("q",): [("d:1", 0.8), ("d:2", 0.6), ("d:3", 0.4)]}),
            FakeBM25({"q": []}),
        )

    def test_reranker_overrides_rrf_order_and_topk_applies_after(self):
        r = retrieve(self._deps(), "q", mode="simple", top_k=2,
                     enable_rerank=True, reranker=ReverseReranker())
        # RRF 序为 d:1,d:2,d:3；倒序后 d:3,d:2,d:1 → top_k=2 截取 d:3,d:2
        assert [h.chunk_id for h in r.hits] == ["d:3", "d:2"]
        assert r.hits[0].rerank_score == pytest.approx(3.0)

    def test_band_still_decided_by_vector_cosine_not_rerank(self):
        r = retrieve(self._deps(), "q", mode="simple",
                     enable_rerank=True, reranker=ReverseReranker())
        assert r.band == "answer"          # best vec 0.8，不受重排影响
        assert r.best_vector_score == pytest.approx(0.8)

    def test_reranker_failure_falls_back_to_rrf_order(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            r = retrieve(self._deps(), "q", mode="simple",
                         enable_rerank=True, reranker=ExplodingReranker())
        # 软降级：RRF 原序保留、结果可用、告警可观测
        assert [h.chunk_id for h in r.hits] == ["d:1", "d:2", "d:3"]
        assert all(h.rerank_score is None for h in r.hits)
        assert any("rerank" in rec.message for rec in caplog.records)

    def test_reranker_none_keeps_plain_rrf(self):
        r = retrieve(self._deps(), "q", mode="simple",
                     enable_rerank=False, reranker=None)
        assert [h.chunk_id for h in r.hits] == ["d:1", "d:2", "d:3"]
