"""工具注册表装配（P2 起步，P3–P6 在 register_cortex_tools 内追加）。

注册一律包 ``quiet()``：框架 register_tool 会打印展开明细，污染 stdout。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hello_agents.tools import ToolRegistry

from cortex.config import CortexConfig
from cortex.logging_setup import quiet


def register_builtin_tools(
    registry: ToolRegistry,
    project_root: Path,
    *,
    allow_write: bool = False,
) -> None:
    """框架内置基础工具：计算器 + 只读文件；写文件需显式 allow_write。"""
    from hello_agents.tools.builtin.calculator import CalculatorTool
    from hello_agents.tools.builtin.file_tools import EditTool, ReadTool, WriteTool

    with quiet():
        registry.register_tool(CalculatorTool())
        registry.register_tool(ReadTool(project_root=str(project_root), registry=registry))
        if allow_write:
            registry.register_tool(WriteTool(project_root=str(project_root), registry=registry))
            registry.register_tool(EditTool(project_root=str(project_root), registry=registry))


def register_cortex_tools(
    registry: ToolRegistry,
    cfg: CortexConfig,
    container: Optional[dict] = None,
) -> dict:
    """注册 Cortex 自研工具，返回 {逻辑名: 工具实例} 供 bind/测试/关闭使用。

    container 携带已构建的服务（P3 起：knowledge / memory；P6 起：mcp / a2a）。
    缺哪个服务就跳过对应工具 —— 降级优先，注册表永远可用。
    """
    from cortex.tools.note_tool import NoteTool

    container = container or {}
    tools: dict = {}

    note = NoteTool(notes_dir=cfg.notes_dir)
    tools["note"] = note

    knowledge = container.get("knowledge")
    if knowledge is not None:
        from cortex.tools.knowledge_tool import KnowledgeSearchTool

        tools["knowledge"] = KnowledgeSearchTool(
            knowledge, weak_llm=container.get("weak_llm"), strong_llm=container.get("strong_llm")
        )

    memory = container.get("memory")
    if memory is not None:
        from cortex.tools.memory_tool import MemoryTool

        tools["memory"] = MemoryTool(
            memory, procedural_enabled=cfg.procedural_memory_enabled
        )

    citation = None
    if knowledge is not None:
        from cortex.tools.citation_tool import CitationTool

        citation = CitationTool(knowledge, weak_llm=container.get("weak_llm"))
        tools["citation"] = citation

    mcp = container.get("mcp")
    if mcp is not None:
        from cortex.tools.mcp_tool import MCPBridgeTool

        tools["mcp"] = MCPBridgeTool(mcp)

    a2a = container.get("a2a")
    if a2a is not None:
        from cortex.tools.a2a_tool import A2ATool

        tools["a2a"] = A2ATool(a2a)

    with quiet():
        registry.register_tool(note)
        note.bind_registry(registry)
        for name in ("knowledge", "memory", "citation", "mcp", "a2a"):
            if name in tools:
                registry.register_tool(tools[name])

    return tools
