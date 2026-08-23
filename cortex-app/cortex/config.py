"""配置聚合：.env 加载 + hello-agents Config 桥接。

职责（规划文档 任务1.1）：
- 读取 .env，构造 :class:`CortexConfig`（含强/弱双档 LLM 参数、存储、RAG、记忆参数）
- :func:`framework_config` 把 Cortex 配置映射为框架 ``hello_agents.Config``，
  所有持久化目录绝对化到 ``data/`` 下，避免框架相对路径污染 CWD。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from hello_agents import Config as HAConfig


class CortexConfigError(Exception):
    """配置缺失或非法。"""


_TRUE = {"1", "true", "yes", "on"}


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default) or default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUE


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw not in (None, "") else default


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (root / p)


@dataclass
class CortexConfig:
    """Cortex 全量配置（字段一一对应 .env.example）。"""

    # ---- LLM ----
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model_id: str = ""  # 空 → 运行时经 /models 解析
    llm_timeout: int = 120
    strong_model_id: str = ""
    weak_model_id: str = ""
    strong_temperature: float = 0.3
    strong_timeout: int = 120
    weak_temperature: float = 0.0
    weak_timeout: int = 60

    # ---- Embeddings ----
    embed_provider: str = "local"  # local | openai_compat | hash(仅测试)
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    openai_compat_embed_url: str = ""
    openai_compat_embed_key: str = ""

    # ---- 存储 ----
    store_mode: str = "local"  # local | docker
    project_root: Path = field(default_factory=Path.cwd)
    data_dir: Optional[Path] = None
    qdrant_local_path: Optional[Path] = None
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ---- RAG ----
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_min_score: float = 0.3
    mqe_variants: int = 3
    enable_rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ---- 记忆 ----
    memory_w_similarity: float = 0.6
    memory_w_importance: float = 0.25
    memory_w_recency: float = 0.15
    memory_decay_lambda: float = 0.05  # recency = exp(-λ * age_days)
    memory_forget_importance: float = 0.2
    memory_time_forget_days: int = 30
    memory_time_forget_importance: float = 0.5
    memory_capacity: int = 5000
    memory_async_writes: bool = True
    procedural_memory_enabled: bool = False

    # ---- 协议 ----
    mcp_servers_json: str = ""
    a2a_researcher_url: str = ""

    # ---- HTTP API ----
    api_key: str = ""  # X-API-Key；空 = 不鉴权

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.data_dir = (
            _abs(self.data_dir, self.project_root) if self.data_dir else self.project_root / "data"
        )
        if self.qdrant_local_path:
            self.qdrant_local_path = _abs(self.qdrant_local_path, self.project_root)
        else:
            self.qdrant_local_path = self.data_dir / "qdrant_local"

    def rebase(self, project_root: str | Path) -> None:
        """把 data_dir/qdrant 等派生路径整体迁移到新 project_root 下（测试用）。"""
        old_root = self.project_root
        self.project_root = Path(project_root).expanduser().resolve()

        def _move(p: Optional[Path]) -> Optional[Path]:
            if p is None:
                return None
            try:
                return self.project_root / Path(p).relative_to(old_root)
            except ValueError:  # 不在旧 root 下 → 显式路径，保持
                return p

        self.data_dir = _move(self.data_dir)
        self.qdrant_local_path = _move(self.qdrant_local_path)

    @property
    def has_live_key(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def notes_dir(self) -> Path:
        return self.data_dir / "notes"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.qdrant_local_path,
            self.notes_dir,
            self.data_dir / "bm25",
            self.data_dir / "traces",
            self.data_dir / "sessions",
            self.data_dir / "todos",
            self.data_dir / "devlogs",
            self.data_dir / "tool-output",
            self.data_dir / "eval",
            self.data_dir / "tiktoken_cache",
        ):
            d.mkdir(parents=True, exist_ok=True)

    def mcp_servers(self) -> list[dict[str, Any]]:
        import json

        raw = (self.mcp_servers_json or "").strip()
        if not raw:
            return []
        servers = json.loads(raw)
        if not isinstance(servers, list):
            raise CortexConfigError("CORTEX_MCP_SERVERS 必须是 JSON 数组")
        return servers


def find_env_file(explicit: str | Path | None = None) -> Optional[Path]:
    """定位 .env：显式指定 > CWD 向上查找 > 包根。"""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise CortexConfigError(f"指定的 env 文件不存在: {p}")
        return p.resolve()
    cur = Path.cwd()
    for cand in (cur, *cur.parents):
        if (cand / ".env").exists():
            return cand / ".env"
    pkg_root = Path(__file__).resolve().parent.parent
    if (pkg_root / ".env").exists():
        return pkg_root / ".env"
    return None


def load_config(env_file: str | Path | None = None) -> CortexConfig:
    """从 .env + 环境变量构造配置；目录自动创建。"""
    found = find_env_file(env_file)
    if found:
        load_dotenv(found, override=False)
        root = found.parent
    else:
        root = Path.cwd().resolve()

    cfg = CortexConfig(
        llm_base_url=_env("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        llm_api_key=_env("LLM_API_KEY"),
        llm_model_id=_env("LLM_MODEL_ID"),
        llm_timeout=_env_int("LLM_TIMEOUT", 120),
        strong_model_id=_env("STRONG_MODEL_ID"),
        weak_model_id=_env("WEAK_MODEL_ID"),
        strong_temperature=_env_float("STRONG_TEMPERATURE", 0.3),
        strong_timeout=_env_int("STRONG_TIMEOUT", 120),
        weak_temperature=_env_float("WEAK_TEMPERATURE", 0.0),
        weak_timeout=_env_int("WEAK_TIMEOUT", 60),
        embed_provider=_env("CORTEX_EMBED_PROVIDER", "local"),
        embed_model=_env("CORTEX_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
        openai_compat_embed_url=_env("OPENAI_COMPAT_EMBED_URL"),
        openai_compat_embed_key=_env("OPENAI_COMPAT_EMBED_KEY"),
        store_mode=_env("CORTEX_STORE_MODE", "local"),
        project_root=root,
        data_dir=_env("CORTEX_DATA_DIR", "data") or None,
        qdrant_local_path=_env("QDRANT_LOCAL_PATH") or None,
        qdrant_url=_env("QDRANT_URL", "http://localhost:6333"),
        neo4j_uri=_env("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=_env("NEO4J_USER", "neo4j"),
        neo4j_password=_env("NEO4J_PASSWORD"),
        chunk_size=_env_int("CORTEX_CHUNK_SIZE", 512),
        chunk_overlap=_env_int("CORTEX_CHUNK_OVERLAP", 64),
        retrieval_min_score=_env_float("CORTEX_RETRIEVAL_MIN_SCORE", 0.3),
        mqe_variants=_env_int("CORTEX_MQE_VARIANTS", 3),
        enable_rerank=_env_bool("CORTEX_ENABLE_RERANK", False),
        rerank_model=_env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        memory_w_similarity=_env_float("CORTEX_MEMORY_W_SIMILARITY", 0.6),
        memory_w_importance=_env_float("CORTEX_MEMORY_W_IMPORTANCE", 0.25),
        memory_w_recency=_env_float("CORTEX_MEMORY_W_RECENCY", 0.15),
        memory_decay_lambda=_env_float("CORTEX_MEMORY_DECAY_LAMBDA", 0.05),
        memory_forget_importance=_env_float("CORTEX_MEMORY_FORGET_IMPORTANCE", 0.2),
        memory_time_forget_days=_env_int("CORTEX_MEMORY_TIME_FORGET_DAYS", 30),
        memory_time_forget_importance=_env_float("CORTEX_MEMORY_TIME_FORGET_IMPORTANCE", 0.5),
        memory_capacity=_env_int("CORTEX_MEMORY_CAPACITY", 5000),
        memory_async_writes=_env_bool("CORTEX_MEMORY_ASYNC_WRITES", True),
        procedural_memory_enabled=_env_bool("CORTEX_PROCEDURAL_MEMORY_ENABLED", False),
        mcp_servers_json=_env("CORTEX_MCP_SERVERS"),
        a2a_researcher_url=_env("CORTEX_A2A_RESEARCHER_URL"),
        api_key=_env("API_KEY"),
    )
    cfg.ensure_dirs()
    return cfg


def framework_config(cfg: CortexConfig, *, subagent: bool = False) -> HAConfig:
    """映射为框架 Config。

    - 全部持久化目录绝对化到 ``cfg.data_dir`` 下；
    - ``subagent=True`` 关闭 Task/Todo/DevLog/Skill 自动注册，
      防止子代理内部重复注册同名工具（框架在传入 registry 时会自动注册）。
    """
    ha = HAConfig(
        trace_dir=str(cfg.data_dir / "traces"),
        session_dir=str(cfg.data_dir / "sessions"),
        todowrite_persistence_dir=str(cfg.data_dir / "todos"),
        devlog_persistence_dir=str(cfg.data_dir / "devlogs"),
        tool_output_dir=str(cfg.data_dir / "tool-output"),
        skills_dir=str(cfg.project_root / "skills"),
        enable_smart_compression=False,
    )
    if subagent:
        ha.subagent_enabled = False
        ha.todowrite_enabled = False
        ha.devlog_enabled = False
    return ha
