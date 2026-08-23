"""角色 → 工具可见性过滤器（单一事实来源）。

builders 与 TaskTool 派发共用本模块。注意过滤器匹配的是**展开后的**
子工具名（note_read / memory_search / citation_verify …）。
"""

from __future__ import annotations

from hello_agents.tools import CustomFilter, ReadOnlyFilter


def retriever_filter():
    """检索子代理：只读 + 知识库检索 + 记忆召回。"""
    return ReadOnlyFilter(additional_allowed=["knowledge_search", "memory_search"])


def researcher_filter():
    """联网研究子代理：MCP 桥接 + 笔记 + Skills。"""
    return CustomFilter(
        allowed=[
            "mcp_list_tools",
            "mcp_call_tool",
            "note_create",
            "note_read",
            "note_update",
            "note_search",
            "Skill",
        ]
    )


def writer_filter():
    """写作子代理：记忆读写 + 笔记读写 + 引用校验 + 只读文件。"""
    return CustomFilter(
        allowed=[
            "memory_add",
            "memory_search",
            "memory_summary",
            "note_read",
            "note_update",
            "citation_verify",
            "Read",
        ]
    )


def critic_filter():
    """审校子代理：只读 + 引用核验 + 知识库比对。"""
    return ReadOnlyFilter(additional_allowed=["citation_verify", "knowledge_search"])


ROLE_FILTERS = {
    "retriever": retriever_filter,
    "researcher": researcher_filter,
    "writer": writer_filter,
    "critic": critic_filter,
}


def filter_for_role(role: str):
    factory = ROLE_FILTERS.get(role)
    return factory() if factory else None
