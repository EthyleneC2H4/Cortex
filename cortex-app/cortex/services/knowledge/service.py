"""KnowledgeService 门面：RAG 全链路的唯一入口（工具层/ContextEngine 只看它）。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..db import Database
from .bm25_index import BM25Index
from .embeddings import EmbeddingProvider, make_embedder
from .graph import make_graph_store
from .ingest import IngestContext, delete_document, ingest_document, ingest_text
from .qa import QAAnswer, answer_question
from .retrieve import RetrievalResult, retrieve


class KnowledgeService:
    """聚合 embedder/vector/bm25/graph/db 五件套；惰性建存储。

    ``db=``/``qdrant_client=`` 供应用层注入共享实例：sqlite WAL 多连接安全，
    但 qdrant 本地模式同路径必须单 client（排他锁）。
    """

    # 类级默认：兼容测试里 ``__new__`` 跳过 __init__ 的桩构造（属性兜底存在）
    _reranker = None
    _reranker_resolved = False

    def __init__(self, cfg, *, db: Optional[Database] = None,
                 qdrant_client=None) -> None:
        self.cfg = cfg
        self.db = db if db is not None else Database(cfg.data_dir / "cortex.db")
        self.embedder: EmbeddingProvider = make_embedder(cfg)
        self._qdrant_client = qdrant_client
        self._vector = None
        self._reranker = None
        self._reranker_resolved = False  # 惰性构建只尝试一次（失败不反复重试拖慢请求）
        self.bm25 = BM25Index(path=cfg.data_dir / "bm25" / "index.jsonl")
        self.graph = make_graph_store(cfg)

    @property
    def reranker(self):
        """惰性重排器：enable_rerank 且依赖/模型可用才有实例，否则 None。"""
        if not self._reranker_resolved:
            self._reranker_resolved = True
            from .rerank import make_reranker

            self._reranker = make_reranker(self.cfg)
        return self._reranker

    # ---------------------------------------------------------------- #
    @property
    def vector(self):
        if self._vector is None:
            from .stores import make_vector_store

            self._vector = make_vector_store(self.cfg, dim=self.embedder.dim,
                                             client=self._qdrant_client)
        return self._vector

    @property
    def deps(self) -> IngestContext:
        return IngestContext(self.db, self.embedder, self.vector, self.bm25, self.graph)

    # ---------------------------------------------------------------- #
    def ingest_document(self, path: Path | str, *, weak_llm=None) -> dict:
        return ingest_document(self.deps, path, weak_llm=weak_llm,
                               chunk_size=self.cfg.chunk_size, chunk_overlap=self.cfg.chunk_overlap)

    def ingest_text(self, *, title: str, text: str, source_path: str = "", weak_llm=None) -> dict:
        return ingest_text(
            self.deps, title=title, text=text, source_path=source_path, weak_llm=weak_llm,
            chunk_size=self.cfg.chunk_size, chunk_overlap=self.cfg.chunk_overlap,
        )

    def delete(self, doc_id: str) -> bool:
        return delete_document(self.deps, doc_id)

    def list_documents(self) -> List[dict]:
        """知识库文档清单（新→旧），供 API 列表页使用；不写裸 SQL 到调用方。"""
        return self.db.query(
            "SELECT id AS doc_id, title, source, n_chunks, created_at "
            "FROM documents ORDER BY created_at DESC"
        )

    def search(self, query: str, *, top_k: int = 5, weak_llm=None, mode: str = "auto") -> RetrievalResult:
        return retrieve(
            self.deps, query, weak_llm=weak_llm, top_k=top_k, mode=mode,
            min_score=self.cfg.retrieval_min_score,
            n_variants=self.cfg.mqe_variants,
            enable_rerank=self.cfg.enable_rerank,
            reranker=self.reranker,
        )

    def ask(self, question: str, strong_llm, *, weak_llm=None, top_k: int = 5) -> QAAnswer:
        return answer_question(self.deps, question, strong_llm, weak_llm=weak_llm,
                               top_k=top_k, min_score=self.cfg.retrieval_min_score)

    def search_for_context(self, query: str, *, limit: int = 5, min_score: float | None = None) -> List[dict]:
        """给 ContextEngine 的紧凑证据包：[{text, source, chunk_id, score}]。"""
        floor = self.cfg.retrieval_min_score if min_score is None else min_score
        result = retrieve(self.deps, query, top_k=limit, min_score=floor,
                          enable_rerank=self.cfg.enable_rerank, reranker=self.reranker)
        if result.band == "refuse":
            return []
        return [
            {"text": h.text, "source": h.source, "chunk_id": h.chunk_id, "score": h.vector_score}
            for h in result.hits
            if h.vector_score >= floor or h.vector_score < 0  # BM25-only 命中保留（有文本即可用）
        ]

    def graph_stats(self) -> dict:
        return self.graph.stats()

    def stats(self) -> dict:
        docs = self.db.query_one("SELECT COUNT(*) AS n FROM documents") or {"n": 0}
        chunks = self.db.query_one("SELECT COUNT(*) AS n FROM chunks") or {"n": 0}
        return {
            "documents": docs["n"],
            "chunks": chunks["n"],
            "bm25": self.bm25.size,
            "vectors": self.vector.count(),
            "graph": self.graph.stats(),
        }

    def close(self) -> None:
        """释放 qdrant 本地文件锁与 sqlite 连接（进程退出/重启前调用）。"""
        try:
            if self._vector is not None:
                self._vector.client.close()
        except Exception:
            pass
        self.db.close()
