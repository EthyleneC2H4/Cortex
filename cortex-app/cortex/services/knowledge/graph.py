"""GraphStore：networkx + JSON 持久化（默认）/ Neo4j（docker extra）。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Protocol


class GraphStore(Protocol):
    def add_triples(self, triples: List[dict]) -> None: ...

    def neighbors(self, entity: str, depth: int = 1) -> List[dict]: ...

    def stats(self) -> dict: ...


class LocalGraphStore:
    """networkx MultiDiGraph；节点=实体，边属性含 relation/confidence/source_doc。"""

    def __init__(self, persist_path: Optional[Path | str] = None) -> None:
        import networkx as nx

        self._nx = nx
        self.path = Path(persist_path) if persist_path else None
        self._lock = threading.RLock()
        self.g = nx.MultiDiGraph()
        if self.path and self.path.exists():
            self._load()

    # ---------------------------------------------------------------- #
    def add_triples(self, triples: List[dict]) -> None:
        with self._lock:
            for t in triples or []:
                subj, obj, rel = t.get("subject", "").strip(), t.get("object", "").strip(), t.get("relation", "").strip()
                if not subj or not obj or not rel:
                    continue
                conf = float(t.get("confidence", 0.0))
                self.g.add_node(subj)
                self.g.add_node(obj)
                self.g.add_edge(subj, obj, relation=rel, confidence=conf,
                                source_doc=t.get("source_doc", ""))
            self._save()

    def neighbors(self, entity: str, depth: int = 1) -> List[dict]:
        if entity not in self.g:
            return []
        found: Dict[str, dict] = {}
        frontier = {entity}
        for _ in range(max(1, depth)):
            nxt: set = set()
            for node in frontier:
                for _, o, data in self.g.out_edges(node, data=True):
                    if o not in found:
                        found[o] = {"entity": o, "via": node, **data}
                        nxt.add(o)
                for s, _, data in self.g.in_edges(node, data=True):
                    if s not in found:
                        found[s] = {"entity": s, "via": node, **data}
                        nxt.add(s)
            frontier = nxt
        return list(found.values())

    def stats(self) -> dict:
        with self._lock:
            return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}

    # ---------------------------------------------------------------- #
    def _save(self) -> None:
        if not self.path:
            return
        data = {
            "nodes": [{"id": n, **(d or {})} for n, d in self.g.nodes(data=True)],
            "edges": [
                {"source": u, "target": v, "key": k, **d}
                for u, v, k, d in self.g.edges(data=True, keys=True)
            ],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for n in data.get("nodes", []):
            self.g.add_node(n.pop("id"), **n)
        for e in data.get("edges", []):
            d = {k: v for k, v in e.items() if k not in ("source", "target", "key")}
            self.g.add_edge(e["source"], e["target"], key=e.get("key"), **d)


def make_graph_store(cfg) -> GraphStore:
    """docker 模式且装了 neo4j extra → Neo4jGraphStore；否则/降级 → 本地 JSON 图。"""
    if cfg.store_mode == "docker":
        try:
            from cortex.services.knowledge.neo4j_store import Neo4jGraphStore

            store = Neo4jGraphStore(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
            store.ensure_schema()
            return store
        except ImportError:
            pass  # 未装 docker extra → 降级本地图
    return LocalGraphStore(persist_path=getattr(cfg, "graph_path", None))
