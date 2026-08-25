"""BGE 重排器：交叉编码精排（可选增强，默认关闭）。

RRF 融合只看"出现在哪一路、排第几"，不看 query 与正文的实际语义匹配度；
重排器用 cross-encoder 逐对打 (query, chunk) 相关性分，把真正回答问题的
片段顶到前面。代价是每条候选一次前向推理，故仅对 RRF 融合后的头部使用。

依赖与降级策略：
- FlagEmbedding 是可选依赖（``pip install -e ".[rerank]"``），**惰性导入**——
  不装 extra、不开 ``CORTEX_ENABLE_RERANK`` 时零开销；
- :func:`make_reranker` 在未启用 / 缺依赖 / 模型加载失败任一情况返回 None，
  retrieve() 收到 None 即照常走纯 RRF 排序，只记 warning，不阻断检索。
"""

from __future__ import annotations

import logging

from .retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    """bge-reranker 交叉编码重排；接口满足 retrieve(reranker=) 注入约定。"""

    def __init__(self, model_name: str = _DEFAULT_MODEL,
                 max_length: int = 512, batch_size: int = 16) -> None:
        try:
            from FlagEmbedding import FlagReranker  # 可选依赖 [rerank] extra
        except ImportError as exc:
            raise RuntimeError(
                '未安装 FlagEmbedding：先 pip install -e ".[rerank]" 再开启 enable_rerank'
            ) from exc
        # HF_HUB_OFFLINE=1 时直接读本地缓存；模型缺失在此抛出、由工厂降级
        self._model = FlagReranker(model_name, use_fp16=False, max_length=max_length)
        self._batch_size = batch_size

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """按 (query, chunk.text) 相关性分降序返回新列表，并回填 rerank_score。

        空正文块退化为用 chunk_id 占位（cross-encoder 需要非空文本）。
        """
        if not chunks:
            return []
        pairs = [[query, c.text or c.chunk_id] for c in chunks]
        scores = self._model.compute_score(pairs, batch_size=self._batch_size,
                                           normalize=True)  # sigmoid → [0,1]
        if isinstance(scores, (int, float)):  # 单条时 FlagEmbedding 返回标量
            scores = [scores]
        ranked = list(chunks)
        for chunk, s in zip(ranked, scores):
            chunk.rerank_score = float(s)
        return sorted(ranked, key=lambda c: c.rerank_score, reverse=True)


def make_reranker(cfg) -> BGEReranker | None:
    """按配置构建重排器；未启用/缺依赖/模型不可得一律返回 None（软降级）。"""
    if not getattr(cfg, "enable_rerank", False):
        return None
    model_name = getattr(cfg, "rerank_model", None) or _DEFAULT_MODEL
    try:
        return BGEReranker(model_name)
    except Exception:  # noqa: BLE001 —— 缓存缺失/网络失败都不该阻断检索主路径
        logger.warning("BGE 重排器初始化失败（模型 %s），已回退纯 RRF 排序",
                       model_name, exc_info=True)
        return None
