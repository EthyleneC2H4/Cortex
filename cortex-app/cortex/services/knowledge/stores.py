"""VectorStore：Qdrant local path 模式（默认，零 docker）/ docker 模式。

collection ``cortex_chunks``（cosine）。qdrant 点 id 只接受 uint/UUID，
用 uuid5(chunk_id) 稳定映射 → 重复摄取幂等覆盖。写操作串行化（本地模式单进程）。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "cortex-chunk")
_UPSERT_LOCK = threading.Lock()


@dataclass
class VectorHit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    heading_path: str
    source: str


class VectorStore(Protocol):
    dim: int

    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[dict]) -> None: ...

    def search(self, vector: List[float], top_k: int = 20, doc_id: Optional[str] = None) -> List[VectorHit]: ...

    def delete_doc(self, doc_id: str) -> None: ...

    def count(self) -> int: ...


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantLocalStore:
    """qdrant-client 本地文件模式：与 docker 同 API，dev/测试零依赖。

    本地模式对存储目录持排他锁 —— 同一进程内多个 collection 必须**共享同一个
    QdrantClient**（经 ``client=`` 注入），否则第二把锁直接报错。
    """

    def __init__(self, path: Path | str, dim: int, collection: str = "cortex_chunks",
                 client=None) -> None:
        from qdrant_client import QdrantClient, models  # 延迟导入

        self._models = models
        self.client = client if client is not None else QdrantClient(path=str(path))
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=_point_id(cid), vector=vec, payload=dict(pl))
            for cid, vec, pl in zip(ids, vectors, payloads)
        ]
        with _UPSERT_LOCK:
            for i in range(0, len(points), 128):
                self.client.upsert(collection_name=self.collection, points=points[i : i + 128])

    def search(self, vector, top_k=20, doc_id=None) -> List[VectorHit]:
        flt = None
        if doc_id:
            flt = self._models.Filter(
                must=[self._models.FieldCondition(key="doc_id", match=self._models.MatchValue(value=doc_id))]
            )
        hits = self.client.query_points(
            collection_name=self.collection, query=vector, limit=top_k, query_filter=flt, with_payload=True
        ).points
        return [
            VectorHit(
                chunk_id=h.payload.get("chunk_id", ""),
                doc_id=h.payload.get("doc_id", ""),
                score=float(h.score),
                text=h.payload.get("text", ""),
                heading_path=h.payload.get("heading_path", ""),
                source=h.payload.get("source", ""),
            )
            for h in hits
        ]

    def delete_doc(self, doc_id: str) -> None:
        flt = self._models.Filter(
            must=[self._models.FieldCondition(key="doc_id", match=self._models.MatchValue(value=doc_id))]
        )
        with _UPSERT_LOCK:
            self.client.delete(collection_name=self.collection, points_selector=flt)

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection).count)


class QdrantDockerStore(QdrantLocalStore):
    """同库 REST/gRPC 模式（CORTEX_STORE_MODE=docker 时经 url 访问）。"""

    def __init__(self, url: str, dim: int, collection: str = "cortex_chunks") -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim
        for _ in range(30):  # compose healthcheck 后仍可能瞬时不就绪
            try:
                if self.client.collection_exists(collection):
                    break
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                )
                break
            except Exception:
                time.sleep(1.0)


def make_vector_store(cfg, dim: int, client=None) -> VectorStore:
    if cfg.store_mode == "docker":
        return QdrantDockerStore(url=cfg.qdrant_url, dim=dim)
    return QdrantLocalStore(path=cfg.qdrant_local_path, dim=dim, client=client)
