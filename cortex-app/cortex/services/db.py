"""sqlite3 关系库封装：documents/chunks/memories 元数据表。

单连接 + RLock（框架 sync-first，无高并发需求）；WAL 提升读写并行。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    path       TEXT,
    title      TEXT,
    source     TEXT,
    n_chunks   INTEGER DEFAULT 0,
    created_at TEXT,
    meta       TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT,
    idx          INTEGER,
    text         TEXT,
    heading_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    user_id      TEXT DEFAULT 'default',
    type         TEXT,
    content      TEXT,
    importance   REAL DEFAULT 0.5,
    created_at   TEXT,
    last_access  TEXT,
    access_count INTEGER DEFAULT 0,
    source       TEXT,
    status       TEXT DEFAULT 'active',
    metadata     TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type, status);
"""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Database:
    """极薄 sqlite 封装：所有方法线程安全，行以 dict 返回。"""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---------------------------------------------------------------- #
    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params))
            self._conn.commit()

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, [tuple(r) for r in rows])
            self._conn.commit()

    def query(self, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---------------------------------------------------------------- #
    def upsert_document(
        self,
        doc_id: str,
        *,
        path: str = "",
        title: str = "",
        source: str = "",
        n_chunks: int = 0,
        meta: Optional[dict] = None,
    ) -> None:
        self.execute(
            "INSERT INTO documents(id,path,title,source,n_chunks,created_at,meta) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET path=excluded.path,title=excluded.title,"
            "source=excluded.source,n_chunks=excluded.n_chunks,meta=excluded.meta",
            (doc_id, path, title, source, n_chunks, now_iso(), json.dumps(meta or {}, ensure_ascii=False)),
        )

    def replace_chunks(self, doc_id: str, rows: List[Dict[str, Any]]) -> None:
        """整文档替换 chunks（重摄取幂等）。"""
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.executemany(
                "INSERT INTO chunks(chunk_id,doc_id,idx,text,heading_path) VALUES(?,?,?,?,?)",
                [(r["chunk_id"], doc_id, r["idx"], r["text"], r["heading_path"]) for r in rows],
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
