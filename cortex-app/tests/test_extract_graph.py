"""P3 单元测试：三元组抽取宽容解析 + 图存储。"""

from __future__ import annotations

import pytest

from cortex.services.knowledge.extract import _parse_json_array, extract_triples
from cortex.services.knowledge.graph import LocalGraphStore

pytestmark = pytest.mark.unit


class TestParse:
    def test_plain_array(self):
        assert _parse_json_array('[{"a":1}]') == [{"a": 1}]

    def test_markdown_fenced(self):
        raw = '```json\n[{"subject":"RAG","relation":"包含","object":"检索"}]\n```'
        assert _parse_json_array(raw)[0]["subject"] == "RAG"

    def test_leading_prose(self):
        raw = '好的，以下是结果：\n[{"subject":"A","relation":"r","object":"B"}]'
        assert len(_parse_json_array(raw)) == 1

    def test_garbage_returns_none(self):
        assert _parse_json_array("完全没有 JSON") is None
        assert _parse_json_array('{"not": "array"}') is None


class TestExtract:
    def _llm(self, content):
        return type("L", (), {"invoke": lambda self, m: type("R", (), {"content": content})()})()

    def test_confidence_and_limit_filters(self):
        raw = (
            '[{"subject":"RRF","relation":"融合","object":"多路排名","confidence":0.9},'
            '{"subject":"低","relation":"过滤","object":"掉","confidence":0.5},'
            '{"subject":"","relation":"缺主体","object":"X","confidence":0.99}]'
        )
        out = extract_triples("文本", self._llm(raw), min_conf=0.8)
        assert len(out) == 1
        assert out[0]["subject"] == "RRF" and out[0]["confidence"] == 0.9

    def test_llm_failure_empty_list(self):
        class Bad:
            def invoke(self, m):
                raise RuntimeError("429")

        assert extract_triples("t", Bad()) == []

    def test_source_doc_attached(self):
        raw = '[{"subject":"A","relation":"r","object":"B","confidence":1.0}]'
        out = extract_triples("t", self._llm(raw), source_doc="d1")
        assert out[0]["source_doc"] == "d1"


class TestLocalGraph:
    def test_triples_neighbors_persistence(self, tmp_path):
        p = tmp_path / "graph.json"
        g = LocalGraphStore(persist_path=p)
        g.add_triples([
            {"subject": "混合检索", "relation": "包含", "object": "RRF", "confidence": 0.9},
            {"subject": "混合检索", "relation": "包含", "object": "BM25", "confidence": 0.9},
            {"subject": "RRF", "relation": "属于", "object": "融合算法", "confidence": 0.85},
        ])
        names = {n["entity"] for n in g.neighbors("混合检索")}
        assert {"RRF", "BM25"} <= names  # depth=1 直接邻居（出边）
        deep = {n["entity"] for n in g.neighbors("混合检索", depth=2)}
        assert {"RRF", "BM25", "融合算法"} <= deep  # 经 RRF 二跳到融合算法

        g2 = LocalGraphStore(persist_path=p)  # 重启加载
        assert g2.stats()["nodes"] == 4
        assert g2.stats()["edges"] == 3

    def test_invalid_triple_skipped(self, tmp_path):
        g = LocalGraphStore(persist_path=None)
        g.add_triples([{"subject": "", "relation": "r", "object": "B"}, {"bad": 1}])
        assert g.stats() == {"nodes": 0, "edges": 0}
