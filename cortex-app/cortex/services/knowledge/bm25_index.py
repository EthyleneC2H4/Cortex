"""BM25 索引：rank-bm25 + 拉丁词/CJK bigram 分词，jsonl 持久化。"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]+")


def tokenize(text: str) -> List[str]:
    """拉丁按词、CJK 连续串切 bigram（整串与单字兜底一并保留）。"""
    out: List[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if re.fullmatch(r"[一-鿿]+", raw):
            if len(raw) == 1:
                out.append(raw)
            else:
                out.extend(raw[i : i + 2] for i in range(len(raw) - 1))
                out.append(raw)
        else:
            out.append(raw)
    return out


@dataclass
class BM25Index:
    """内存倒排 + 元数据表；save/load 全量 jsonl 快照。"""

    path: Path | str | None = None
    _corpus: List[Tuple[str, str]] = field(default_factory=list)  # (chunk_id, text)
    _meta: Dict[str, dict] = field(default_factory=dict)
    _bm25 = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        if self.path and Path(self.path).exists():
            self.load()

    # ---------------------------------------------------------------- #
    def add(self, items: List[Tuple[str, str]], metas: Optional[Dict[str, dict]] = None) -> None:
        """追加 (chunk_id, text)；重复 id 覆盖后整体重建（文档级幂等）。"""
        with self._lock:
            seen = {cid for cid, _ in self._corpus}
            for cid, text in items:
                if cid in seen:
                    self._corpus = [(i, t) for i, t in self._corpus if i != cid]
                self._corpus.append((cid, text))
            if metas:
                self._meta.update(metas)
            self._rebuild()

    def remove_doc(self, chunk_ids: List[str]) -> None:
        drop = set(chunk_ids)
        with self._lock:
            self._corpus = [(i, t) for i, t in self._corpus if i not in drop]
            for cid in drop:
                self._meta.pop(cid, None)
            self._rebuild()

    def _rebuild(self) -> None:
        from rank_bm25 import BM25Okapi

        corpus_tokens = [tokenize(t) for _, t in self._corpus]
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def search(self, query: str, top_k: int = 20, min_score: float = 0.01) -> List[Tuple[str, float]]:
        q_tokens = tokenize(query)
        with self._lock:
            if self._bm25 is None or not q_tokens:
                return []
            scores = self._bm25.get_scores(q_tokens)
        ids = [cid for cid, _ in self._corpus]
        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
        hits = [(cid, float(s)) for cid, s in ranked[:top_k] if s > min_score]
        if hits:
            return hits
        # 小语料兜底：rank_bm25 在 N 很小时命中词的 idf 也可能为 0（负值被清零），
        # 全零分数会误判"无命中"。按查询词覆盖数给 epsilon 分。
        fallback = []
        q_set = set(q_tokens)
        for (cid, text), s in zip(self._corpus, scores):
            overlap = len(q_set & set(tokenize(text)))
            if overlap:
                fallback.append((cid, 1e-6 * overlap))
        fallback.sort(key=lambda x: (-x[1], x[0]))
        return fallback[:top_k]

    def text_of(self, chunk_id: str) -> str:
        for cid, text in self._corpus:
            if cid == chunk_id:
                return text
        return ""

    def meta_of(self, chunk_id: str) -> dict:
        return self._meta.get(chunk_id, {})

    @property
    def size(self) -> int:
        return len(self._corpus)

    # ---------------------------------------------------------------- #
    def save(self) -> None:
        if not self.path:
            return
        with self._lock, open(self.path, "w", encoding="utf-8") as f:
            for cid, text in self._corpus:
                f.write(json.dumps({"id": cid, "text": text, "meta": self._meta.get(cid, {})}, ensure_ascii=False) + "\n")

    def load(self) -> None:
        p = Path(self.path)
        if not p.exists():
            return
        corpus, meta = [], {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                corpus.append((row["id"], row["text"]))
                meta[row["id"]] = row.get("meta", {})
        self._corpus, self._meta = corpus, meta
        self._rebuild()
