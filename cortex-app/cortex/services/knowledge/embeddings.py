"""EmbeddingProvider：Local（默认）/ OpenAICompat / Hash（离线测试）三实现。

OpenRouter 不提供 /embeddings → 默认本地 sentence-transformers
BAAI/bge-small-zh-v1.5（512 维）。HashEmbedder 为确定性词袋哈希，
零依赖零网络，供单元测试与无模型环境使用。
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashEmbedder:
    """确定性 hash 词袋：token→桶累加后 L2 归一。测试专用，语义为零。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[\w一-鿿]+", text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "big")
            vec[h % self.dim] += 1.0
        return _l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vec(text)


class LocalEmbedder:
    """sentence-transformers 本地编码；首次加载下载模型（prewarm 一次性）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._model = None  # 惰性加载，避免 import 即拉 torch
        self.dim = 512

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer  # 延迟导入

                    model = SentenceTransformer(self.model_name)
                    dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
                    self.dim = int(dim_fn())
                    self._model = model
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        with self._lock:
            arr = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [[float(x) for x in row] for row in arr]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAICompatEmbedder:
    """任意 OpenAI 兼容 /embeddings 端点（自建/其他供应商；OpenRouter 无此端点）。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        import httpx  # 延迟导入便于离线测试

        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.dim = 0  # 首次调用探测

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        vecs = [[float(x) for x in d["embedding"]] for d in data]
        if self.dim == 0 and vecs:
            self.dim = len(vecs[0])
        return vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def make_embedder(cfg) -> EmbeddingProvider:
    """按 cfg.embed_provider 构造；未知 provider 报 ValueError。"""
    if cfg.embed_provider == "local":
        return LocalEmbedder(model_name=cfg.embed_model)
    if cfg.embed_provider == "openai_compat":
        return OpenAICompatEmbedder(
            base_url=cfg.openai_compat_embed_url or cfg.llm_base_url,
            api_key=cfg.openai_compat_embed_key or cfg.llm_api_key,
            model=cfg.embed_model,
        )
    if cfg.embed_provider == "hash":
        return HashEmbedder()
    raise ValueError(f"未知 embed provider: {cfg.embed_provider}")
