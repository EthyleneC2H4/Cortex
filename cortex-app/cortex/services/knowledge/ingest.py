"""文档摄取管线：文件→MarkItDown→切块→向量/BM25/图/关系库 四路落盘。

doc_id 规则：相对路径的稳定哈希 → 重摄取同一路径幂等覆盖。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Protocol

from ..db import Database, now_iso
from .bm25_index import BM25Index
from .chunking import Chunk, chunk_markdown, default_token_counter
from .embeddings import EmbeddingProvider
from .extract import extract_triples
from .graph import GraphStore
from .stores import VectorStore


class IngestDeps(Protocol):
    db: Database
    embedder: EmbeddingProvider
    vector: VectorStore
    bm25: BM25Index
    graph: GraphStore


def make_doc_id(path: str) -> str:
    return hashlib.sha1(f"cortex-doc:{path}".encode("utf-8")).hexdigest()[:16]


def file_to_markdown(path: Path) -> str:
    """md/txt 直读；其余格式经 MarkItDown（失败给出可读错误）。"""
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return path.read_text(encoding="utf-8")
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(path))
    return result.text_content or ""


def ingest_text(
    deps: IngestDeps,
    *,
    title: str,
    text: str,
    source_path: str = "",
    weak_llm=None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict:
    """单文档全量摄取；返回 {doc_id, n_chunks}。"""
    doc_id = make_doc_id(source_path or title)
    counter = default_token_counter()
    chunks: List[Chunk] = chunk_markdown(text, size=chunk_size, overlap=chunk_overlap, counter=counter)
    if not chunks:
        chunks = [Chunk(idx=0, text=text.strip()[:2000] or "(empty)", heading_path="")]

    # 先清旧数据（幂等重摄取）
    old = deps.db.query("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))
    if old:
        deps.vector.delete_doc(doc_id)
        deps.bm25.remove_doc([r["chunk_id"] for r in old])

    rows = []
    for c in chunks:
        cid = f"{doc_id}:{c.idx}"
        rows.append({"chunk_id": cid, "idx": c.idx, "text": c.text, "heading_path": c.heading_path})

    vectors = deps.embedder.embed([r["text"] for r in rows])
    payloads = [
        {
            "doc_id": doc_id,
            "chunk_id": r["chunk_id"],
            "source": source_path or title,
            "text": r["text"],
            "heading_path": r["heading_path"],
            "title": title,
            "created_at": now_iso(),
        }
        for r in rows
    ]
    deps.vector.upsert(ids=[r["chunk_id"] for r in rows], vectors=vectors, payloads=payloads)
    deps.bm25.add(
        [(r["chunk_id"], r["text"]) for r in rows],
        metas={r["chunk_id"]: {"doc_id": doc_id, "source": source_path or title, "heading_path": r["heading_path"]}
               for r in rows},
    )

    if weak_llm is not None:
        triples = extract_triples(text, weak_llm, source_doc=doc_id)
        deps.graph.add_triples(triples)

    deps.db.upsert_document(doc_id, path=source_path, title=title, source=source_path or title, n_chunks=len(rows))
    deps.db.replace_chunks(doc_id, rows)
    deps.bm25.save()
    return {"doc_id": doc_id, "n_chunks": len(rows), "title": title}


def ingest_document(deps: IngestDeps, path: Path | str, *, weak_llm=None, **kw) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"待摄取文件不存在: {p}")
    text = file_to_markdown(p)
    rel = str(p.resolve())
    title = re.sub(r"\.[^.]+$", "", p.name)
    return ingest_text(deps, title=title, text=text, source_path=rel, weak_llm=weak_llm, **kw)


def delete_document(deps: IngestDeps, doc_id: str) -> bool:
    row = deps.db.query_one("SELECT id FROM documents WHERE id=?", (doc_id,))
    if not row:
        return False
    old = deps.db.query("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))
    deps.vector.delete_doc(doc_id)
    deps.bm25.remove_doc([r["chunk_id"] for r in old])
    deps.bm25.save()
    deps.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    deps.db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    return True


class IngestContext:
    """deps 载体：把四路存储打包成 IngestDeps 协议兼容对象。"""

    def __init__(self, db, embedder, vector, bm25, graph) -> None:
        self.db = db
        self.embedder = embedder
        self.vector = vector
        self.bm25 = bm25
        self.graph = graph
