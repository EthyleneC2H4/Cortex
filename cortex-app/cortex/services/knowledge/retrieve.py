"""混合检索：复杂度路由 → MQE/HyDE 扩查 → RRF 融合 → 三带分类。

简单查询：向量 ∪ BM25 各 top20 直接融合。
复杂查询：弱模型生成 3 个改写变体 + 1 个 HyDE 假想文档，各查向量/BM25
top20 后 RRF(k=60) 融合。带判定用**最佳向量余弦**：≥min_score answer /
≥uncertain_floor uncertain / 否则 refuse（BM25-only 命中不参与带判定）。
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Protocol

from ...llm.resilience import resilient_invoke
from .bm25_index import BM25Index
from .embeddings import EmbeddingProvider
from .stores import VectorStore

RRF_K = 60
VECTOR_FANOUT = 20
BM25_FANOUT = 20

_COMPLEX_HINTS = re.compile(
    r"(对比|比较|分析|评估|综述|异同|优劣|为什么|为何|如何|怎样|关系|影响|演进|总结|归纳|综合)",
    re.UNICODE,
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str = ""
    text: str = ""
    heading_path: str = ""
    source: str = ""
    rrf: float = 0.0          # 融合排序分（越大越靠前）
    vector_score: float = -1.0  # 最佳余弦（无向量命中为 -1）
    rerank_score: float | None = None  # 交叉编码重排分（未启用重排为 None）


@dataclass
class RetrievalResult:
    band: str                  # answer / uncertain / refuse
    hits: List[RetrievedChunk] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)
    best_vector_score: float = -1.0


class _Deps(Protocol):
    embedder: EmbeddingProvider
    vector: VectorStore
    bm25: BM25Index


# --------------------------------------------------------------------- #
# 复杂度路由                                                              #
# --------------------------------------------------------------------- #
def classify_query(query: str, *, threshold_chars: int = 24) -> str:
    """启发式路由：关键词命中或长度超限 → complex。"""
    q = query.strip()
    if len(q) >= threshold_chars or _COMPLEX_HINTS.search(q):
        return "complex"
    return "simple"


_MQE_PROMPT = (
    "把下面的检索问题改写成 {n} 个不同角度的独立检索查询（同语言，每行一个，"
    "不要编号不要解释）：\n{query}"
)
_HYDE_PROMPT = (
    "假设下面问题的答案存在于一篇技术文档中，直接写出这段假想的文档段落"
    "（150 字以内，只输出段落本身）：\n{query}"
)


def _expand_queries(query: str, weak_llm, n_variants: int) -> List[str]:
    """MQE 变体 + HyDE 假想文档；LLM 失败降级为仅原始查询。"""
    variants: List[str] = [query]
    if weak_llm is None:
        return variants
    try:
        resp = resilient_invoke(
            weak_llm, [{"role": "user", "content": _MQE_PROMPT.format(n=n_variants, query=query)}]
        )
        lines = [ln.strip(" \t-0123456789.") for ln in str(getattr(resp, "content", "")).splitlines()]
        lines = [ln for ln in lines if ln]
        variants.extend(lines[:n_variants])

        resp2 = resilient_invoke(weak_llm, [{"role": "user", "content": _HYDE_PROMPT.format(query=query)}])
        hyde = str(getattr(resp2, "content", "")).strip()
        if hyde:
            variants.append(hyde[:500])
    except Exception:
        pass
    return variants


# --------------------------------------------------------------------- #
# 融合                                                                    #
# --------------------------------------------------------------------- #
def _rrf_fuse(ranked_lists: List[List[str]], metas: dict) -> List[RetrievedChunk]:
    """多路排名列表 RRF 融合去重；metas 提供 chunk 元数据与向量分。"""
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    out = []
    for cid, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        meta = metas.get(cid, {})
        out.append(
            RetrievedChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", cid.split(":")[0]),
                text=meta.get("text", ""),
                heading_path=meta.get("heading_path", ""),
                source=meta.get("source", ""),
                rrf=s,
                vector_score=meta.get("vector_score", -1.0),
            )
        )
    return out


def _fill_from_db(db, chunks: List[RetrievedChunk]) -> None:
    """BM25 命中缺正文时从 sqlite 补齐（BM25 索引可能未含最新文本）。"""
    if db is None:
        return
    missing = [c.chunk_id for c in chunks if not c.text]
    if not missing:
        return
    placeholders = ",".join("?" * len(missing))
    rows = db.query(f"SELECT chunk_id,doc_id,text,heading_path FROM chunks WHERE chunk_id IN ({placeholders})", missing)
    by_id = {r["chunk_id"]: r for r in rows}
    for c in chunks:
        if not c.text and c.chunk_id in by_id:
            r = by_id[c.chunk_id]
            c.text, c.doc_id, c.heading_path = r["text"], r["doc_id"], r["heading_path"]


# --------------------------------------------------------------------- #
# 主入口                                                                  #
# --------------------------------------------------------------------- #
def retrieve(
    deps: _Deps,
    query: str,
    *,
    weak_llm=None,
    top_k: int = 5,
    mode: str = "auto",
    min_score: float = 0.3,
    uncertain_floor: float = 0.15,
    n_variants: int = 3,
    enable_rerank: bool = False,
    reranker=None,
) -> RetrievalResult:
    q = query.strip()
    if not q:
        return RetrievalResult(band="refuse")

    routed = classify_query(q) if mode == "auto" else mode
    queries = _expand_queries(q, weak_llm, n_variants) if routed == "complex" else [q]

    # 每个查询并行扇出：向量 top20 + BM25 top20
    def one_query(sub_q: str):
        vec_hits = deps.vector.search(deps.embedder.embed_one(sub_q), top_k=VECTOR_FANOUT)
        kw_hits = deps.bm25.search(sub_q, top_k=BM25_FANOUT)
        return vec_hits, kw_hits

    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
        pairs = list(pool.map(one_query, queries))

    metas: dict = {}
    ranked_lists: List[List[str]] = []
    for vec_hits, kw_hits in pairs:
        vlist = []
        for h in vec_hits:
            vlist.append(h.chunk_id)
            m = metas.setdefault(h.chunk_id, {})
            m.update(doc_id=h.doc_id, text=h.text, heading_path=h.heading_path, source=h.source)
            m["vector_score"] = max(m.get("vector_score", -1.0), h.score)
        ranked_lists.append(vlist)
        ranked_lists.append([cid for cid, _ in kw_hits])
        for cid, _score in kw_hits:
            metas.setdefault(cid, {})  # 正文稍后由 _fill_from_db 补

    fused = _rrf_fuse(ranked_lists, metas)
    if enable_rerank and reranker is not None and fused:
        try:
            # 重排只调序不改带判定（band 仍由最佳向量余弦决定）；失败软降级回 RRF 序
            fused = reranker.rerank(q, fused)
        except Exception:  # noqa: BLE001 —— 重排是增强路径，绝不阻断检索
            logging.getLogger(__name__).warning("rerank 执行失败，回退 RRF 排序", exc_info=True)

    best_vec = max((c.vector_score for c in fused), default=-1.0)
    result = RetrievalResult(band=_band(best_vec, min_score, uncertain_floor),
                             hits=fused[:top_k], variants=queries if len(queries) > 1 else [],
                             best_vector_score=best_vec)

    if any(not c.text for c in result.hits):
        _fill_from_db(getattr(deps, "db", None), result.hits)
    return result


def _band(best_vec: float, min_score: float, uncertain_floor: float) -> str:
    if best_vec >= min_score:
        return "answer"
    if best_vec >= uncertain_floor:
        return "uncertain"
    return "refuse"
