"""P3 单元测试：RRF 融合数学 / 复杂度路由 / 三带分类 / 检索编排（全离线假件）。"""

from __future__ import annotations

import pytest

from cortex.services.knowledge.retrieve import (
    RRF_K,
    _band,
    _rrf_fuse,
    classify_query,
    retrieve,
)

pytestmark = pytest.mark.unit


class TestClassify:
    @pytest.mark.parametrize("q", ["RAG 是什么", "bm25"], )
    def test_simple(self, q):
        assert classify_query(q) == "simple"

    @pytest.mark.parametrize(
        "q",
        ["对比向量检索与关键词检索的优劣", "为什么混合检索能提升效果", "请综合分析记忆巩固机制并说明其影响"],
    )
    def test_complex(self, q):
        assert classify_query(q) == "complex"

    def test_long_query_is_complex(self):
        assert classify_query("a" * 30) == "complex"


class TestBand:
    def test_three_bands(self):
        assert _band(0.5, 0.3, 0.15) == "answer"
        assert _band(0.2, 0.3, 0.15) == "uncertain"
        assert _band(0.05, 0.3, 0.15) == "refuse"
        assert _band(-1.0, 0.3, 0.15) == "refuse"


class TestRRF:
    def test_fusion_math_and_dedupe(self):
        metas = {
            f"c{i}": {"doc_id": "d", "text": f"text{i}", "heading_path": "", "source": "", "vector_score": -1.0}
            for i in range(1, 5)
        }
        fused = _rrf_fuse([["c1", "c2", "c3"], ["c2", "c4"]], metas)
        by_id = {c.chunk_id: c for c in fused}
        # c2 出现在两路（list1 rank=1、list2 rank=0）→ 总分最高
        assert fused[0].chunk_id == "c2"
        assert by_id["c2"].rrf == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
        assert by_id["c1"].rrf == pytest.approx(1 / (RRF_K + 1))
        assert len(fused) == 4  # 去重后 4 条

    def test_order_stable_desc(self):
        metas = {f"c{i}": {} for i in range(6)}
        fused = _rrf_fuse([[f"c{i}" for i in range(5)], [f"c{i}" for i in range(3, 6)]], metas)
        scores = [c.rrf for c in fused]
        assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------- #
# 编排（假 embedder/vector/bm25）                                          #
# --------------------------------------------------------------------- #
class FakeVector:
    def __init__(self, mapping):
        self.mapping = mapping  # query_key -> [(chunk_id, score)]

    def search(self, vector, top_k=20, doc_id=None):
        return [
            type("H", (), {"chunk_id": cid, "doc_id": cid.split(":")[0], "score": s,
                           "text": f"文本{cid}", "heading_path": "", "source": ""})()
            for cid, s in self.mapping.get(tuple(vector), [])
        ][:top_k]


class FakeBM25:
    def __init__(self, results):
        self.results = results  # query -> [chunk_id]

    def search(self, query, top_k=20, min_score=0.01):
        return [(cid, 1.0) for cid in self.results.get(query, [])][:top_k]


class FakeEmbedder:
    dim = 4

    def embed_one(self, text):
        return (text,)  # 用查询词本身当 key

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


class Deps:
    def __init__(self, vector, bm25, db=None):
        self.embedder = FakeEmbedder()
        self.vector = vector
        self.bm25 = bm25
        self.db = db


class TestOrchestration:
    def test_simple_mode_fuses_vector_and_bm25(self):
        deps = Deps(
            FakeVector({("q",): [("d:1", 0.8), ("d:2", 0.6)]}),
            FakeBM25({"q": ["d:2", "d:3"]}),
        )
        r = retrieve(deps, "q", mode="simple")
        assert r.band == "answer"  # best vec 0.8 ≥ 0.3
        ids = [h.chunk_id for h in r.hits]
        assert set(ids) == {"d:1", "d:2", "d:3"}
        assert r.hits[0].chunk_id == "d:2"  # 双路命中 RRF 最高

    def test_refuse_when_all_low(self):
        deps = Deps(FakeVector({("q",): [("d:9", 0.05)]}), FakeBM25({"q": []}))
        r = retrieve(deps, "q", mode="simple")
        assert r.band == "refuse"

    def test_bm25_only_hit_filled_from_db(self):
        class DB:
            def query(self, sql, params):
                return [{"chunk_id": "d:3", "doc_id": "d", "text": "库中正文", "heading_path": "H"}]

        deps = Deps(FakeVector({("q",): [("d:1", 0.8)]}), FakeBM25({"q": ["d:3"]}), db=DB())
        r = retrieve(deps, "q", mode="simple")
        hit = next(h for h in r.hits if h.chunk_id == "d:3")
        assert hit.text == "库中正文"
        assert hit.heading_path == "H"

    def test_complex_mode_expands_with_llm(self):
        class LLM:
            def __init__(self):
                self.calls = []

            def invoke(self, messages):
                self.calls.append(messages[0]["content"])
                content = "变体一\n变体二\n变体三"
                if "假想" in messages[0]["content"]:
                    content = "假想文档段落：混合检索通过 RRF 融合两路信号。"
                return type("R", (), {"content": content})()

        llm = LLM()
        deps = Deps(
            FakeVector({(k,): [("d:1", 0.9)] for k in ("对比检索", "变体一", "变体二", "变体三",
                                                       "假想文档段落：混合检索通过 RRF 融合两路信号。")}),
            FakeBM25({k: [] for k in ("对比检索", "变体一", "变体二", "变体三",
                                      "假想文档段落：混合检索通过 RRF 融合两路信号。")}),
        )
        r = retrieve(deps, "对比检索策略的差异", weak_llm=llm, mode="auto")
        assert len(llm.calls) == 2  # MQE + HyDE 各一次
        assert len(r.variants) == 5  # 原 + 3 变体 + HyDE

    def test_llm_failure_degrades_to_plain(self):
        class Bad:
            def invoke(self, messages):
                raise RuntimeError("rate limited")

        deps = Deps(FakeVector({("q",): [("d:1", 0.8)]}), FakeBM25({"q": []}))
        r = retrieve(deps, "q", weak_llm=Bad(), mode="simple")
        assert r.band == "answer" and not r.variants
