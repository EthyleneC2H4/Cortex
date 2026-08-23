"""应用组装入口与 CLI。

完整装配（P1–P6）：
    load_config → resolve model → build_llms
    → 共享 sqlite/qdrant → KnowledgeService + MemoryService
    → ToolRegistry(builtin + cortex 工具) → ContextEngine
    → 4 子代理先建 → orchestrator 后建 → TaskTool 注册（解环）
    → ResearchPipeline / ExplorationMode

CLI：--chat / --ingest PATH / --research TOPIC [--mode pipeline|explore]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from hello_agents import ToolRegistry

from .config import CortexConfig, load_config
from .llm.factory import build_llms
from .llm.model_resolver import resolve_model_id
from .logging_setup import quiet


def make_qdrant_client(cfg: CortexConfig):
    """本地模式共享单 client（排他锁）；docker 模式返回 None 由各 store 自连。"""
    if cfg.store_mode == "docker":
        return None
    from qdrant_client import QdrantClient

    return QdrantClient(path=str(cfg.qdrant_local_path))


class CortexApp:
    """持有配置、双档 LLM、服务容器、工具注册表与装配好的 Agent 集合。"""

    def __init__(self, cfg, strong_llm, weak_llm, registry, agents, services) -> None:
        self.cfg = cfg
        self.strong_llm = strong_llm
        self.weak_llm = weak_llm
        self.registry = registry
        self.agents = agents          # AgentSet
        self.services = services      # {"knowledge":…, "memory":…, "mcp":…}
        self.agent = getattr(agents, "orchestrator", None)

    # ---------------------------------------------------------------- #
    @classmethod
    def build(cls, cfg: Optional[CortexConfig] = None, *, model_id: Optional[str] = None,
              allow_write: bool = False) -> "CortexApp":
        cfg = cfg or load_config()
        mid = model_id or resolve_model_id(cfg)
        strong, weak = build_llms(cfg, model_id=mid)
        cfg.ensure_dirs()

        services: dict = {"weak_llm": weak, "strong_llm": strong}

        # ---- 服务层（降级优先：单个失败不拖垮启动） ---------------- #
        qdrant_client = None
        try:
            qdrant_client = make_qdrant_client(cfg)
        except Exception as exc:
            print(f"⚠️ qdrant 本地客户端初始化失败，向量检索将不可用: {exc}")

        knowledge = memory = None
        try:
            from .services.knowledge.service import KnowledgeService

            knowledge = KnowledgeService(cfg, qdrant_client=qdrant_client)
            services["knowledge"] = knowledge
        except Exception as exc:
            print(f"⚠️ KnowledgeService 初始化失败: {exc}")

        try:
            from .services.memory.service import MemoryService
            from .services.memory.store import QdrantMemoryIndex

            db = knowledge.db if knowledge else _fresh_db(cfg)
            embedder = knowledge.embedder if knowledge else None
            # 共享同一 qdrant 本地 client（排他锁：cortex_chunks/cortex_memory 同路径）
            vector_index = None
            if qdrant_client is not None and embedder is not None:
                vector_index = QdrantMemoryIndex(
                    cfg.qdrant_local_path, dim=embedder.dim,
                    collection="cortex_memory", client=qdrant_client,
                )
            memory = MemoryService(cfg, db=db, embedder=embedder,
                                   vector_index=vector_index, weak_llm=weak)
            services["memory"] = memory
        except Exception as exc:
            print(f"⚠️ MemoryService 初始化失败: {exc}")

        # ---- 协议桥接（可选：未配置即跳过） ------------------------ #
        try:
            servers = cfg.mcp_servers()
            if servers:
                from .services.protocol.mcp_client import MCPClientManager

                services["mcp"] = MCPClientManager(servers)
        except Exception as exc:
            print(f"⚠️ MCP 管理器初始化失败: {exc}")
        if cfg.a2a_researcher_url:
            try:
                from .services.protocol.a2a import A2AClient

                services["a2a"] = A2AClient(cfg.a2a_researcher_url)
            except Exception as exc:
                print(f"⚠️ A2A 客户端初始化失败: {exc}")

        # ---- GSSC 上下文引擎 --------------------------------------- #
        try:
            from .context.engine import ContextEngine

            services["context"] = ContextEngine(cfg, memory=memory,
                                                knowledge=knowledge, weak_llm=weak)
        except Exception as exc:
            print(f"⚠️ ContextEngine 初始化失败: {exc}")

        # ---- 工具注册表 ------------------------------------------- #
        registry = ToolRegistry()
        from .tools.registry_setup import register_builtin_tools, register_cortex_tools

        with quiet():
            register_builtin_tools(registry, project_root=cfg.project_root, allow_write=allow_write)
            tools = register_cortex_tools(registry, cfg, container=services)

        # ---- 代理装配（子代理先建，orchestrator 后建解环） --------- #
        from hello_agents.tools.builtin.task_tool import TaskTool

        from .agents.builders import build_agents
        from .agents.subagent_factory import make_subagent_factory

        agent_set = build_agents(cfg, strong, weak, registry)
        with quiet():
            registry.register_tool(TaskTool(agent_factory=make_subagent_factory(agent_set),
                                            tool_registry=registry))

        services["tools"] = tools
        return cls(cfg, strong, weak, registry, agent_set, services)

    # ---------------------------------------------------------------- #
    @property
    def pipeline(self):
        from .agents.dispatch import ResearchPipeline

        # 装配完成时的全量工具快照：看门狗放弃的阶段没走 run_as_subagent
        # 的 finally 还原，会把共享注册表留在"被过滤"状态，用快照写回兜底。
        snapshot = {name: self.registry.get_tool(name) for name in self.registry.list_tools()}

        def _repair_registry() -> None:
            with quiet():
                self.registry._tools.update(snapshot)

        return ResearchPipeline(
            self.agents,
            knowledge=self.services.get("knowledge"),
            memory=self.services.get("memory"),
            context_engine=self.services.get("context"),
            traces_dir=self.cfg.data_dir / "traces",
            registry_repair=_repair_registry,
        )

    def research(self, topic: str, mode: str = "pipeline"):
        if mode == "explore":
            from .agents.dispatch import ExplorationMode

            return ExplorationMode(self.agents.orchestrator).run(topic)
        return self.pipeline.run(topic)

    def ingest(self, path: Path | str) -> dict:
        kb = self.services.get("knowledge")
        if kb is None:
            raise RuntimeError("知识库服务未初始化，无法摄取")
        return kb.ingest_document(path, weak_llm=self.weak_llm)

    # ---------------------------------------------------------------- #
    def chat(self) -> None:
        """CLI 多轮对话（P1 验收切片保留）。"""
        print("Cortex CLI — 输入 exit/quit 退出")
        history = []
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                break
            out = self.agents.orchestrator.run(text) if hasattr(self.agents, "orchestrator") \
                else self.agent.run(text)
            print(f"cortex> {out}")
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": str(out)})


def _fresh_db(cfg):
    from .services.db import Database

    return Database(cfg.data_dir / "cortex.db")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cortex.app", description="Cortex 应用入口")
    parser.add_argument("--chat", action="store_true", help="多轮对话")
    parser.add_argument("--ingest", metavar="PATH", help="摄取文件入知识库")
    parser.add_argument("--research", metavar="TOPIC", help="执行研究任务")
    parser.add_argument("--mode", choices=["pipeline", "explore"], default="pipeline",
                        help="研究派发模式（默认 pipeline）")
    args = parser.parse_args(argv)

    if args.ingest:
        app = CortexApp.build()
        t0 = time.time()
        info = app.ingest(args.ingest)
        print(f"✅ 已摄取「{info['title']}」 doc_id={info['doc_id']} "
              f"chunks={info['n_chunks']} ({time.time() - t0:.1f}s)")
    elif args.research:
        app = CortexApp.build()
        result = app.research(args.research, mode=args.mode)
        reports = app.cfg.data_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        out = reports / f"report-{time.strftime('%Y%m%d-%H%M%S')}.md"
        out.write_text(result.report_md, encoding="utf-8")
        print(result.report_md)
        print(f"\n📄 报告已保存: {out} （{result.duration_seconds:.0f}s, revised={result.revised}）")
    elif args.chat:
        CortexApp.build().chat()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
