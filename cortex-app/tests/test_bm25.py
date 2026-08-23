"""P3 单元测试：BM25 分词与持久化。"""

from __future__ import annotations

import pytest

from cortex.services.knowledge.bm25_index import BM25Index, tokenize

pytestmark = pytest.mark.unit


class TestTokenize:
    def test_latin_words_lowercased(self):
        assert tokenize("Hello World foo_bar") == ["hello", "world", "foo_bar"]

    def test_cjk_bigrams(self):
        toks = tokenize("混合检索")
        assert "混合" in toks and "合检" in toks and "检索" in toks

    def test_mixed(self):
        toks = tokenize("RAG 检索增强")
        assert "rag" in toks and "检索" in toks


class TestIndex:
    def test_add_search_roundtrip(self, tmp_path):
        idx = BM25Index(path=tmp_path / "idx.jsonl")
        idx.add([
            ("c1", "向量检索使用余弦相似度"),
            ("c2", "BM25 是关键词检索算法"),
            ("c3", "RRF 融合多路排名"),
        ])
        hits = idx.search("关键词检索", top_k=2)
        assert hits and hits[0][0] == "c2"

    def test_readd_same_id_replaces(self, tmp_path):
        idx = BM25Index(path=None)
        idx.add([("c1", "旧内容")])
        idx.add([("c1", "全新内容关于图神经网络")])
        assert idx.size == 1
        assert "图神经网络" in idx.text_of("c1")

    def test_save_load(self, tmp_path):
        p = tmp_path / "idx.jsonl"
        idx = BM25Index(path=p)
        idx.add([("c1", "知识图谱三元组抽取"), ("c2", "记忆巩固机制")],
                metas={"c1": {"doc_id": "d1"}})
        idx.save()
        idx2 = BM25Index(path=p)
        assert idx2.size == 2
        assert idx2.meta_of("c1")["doc_id"] == "d1"
        assert idx2.search("三元组")[0][0] == "c1"

    def test_remove_doc(self, tmp_path):
        idx = BM25Index(path=None)
        idx.add([("a:1", "x 检索"), ("a:2", "y 记忆"), ("b:1", "z 图谱")])
        idx.remove_doc(["a:1", "a:2"])
        assert idx.size == 1
        assert idx.search("检索") == []

    def test_search_empty_index(self, tmp_path):
        assert BM25Index(path=None).search("任意") == []
